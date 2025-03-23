# -*- coding: utf-8 -*-
import copy
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.attention.CA import CoordAtt
from models.attention.CBAM import CBAM
from models.attention.ECA import ECAAttention


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    此函数取自原始 tf 存储库。
    它确保所有层的通道号可被 8 整除
    可以在这里看到：
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    每个样本的下降路径（随机深度）（应用于残差块的主路径时）。
    “具有随机深度的深度网络”，https://arxiv.org/pdf/1603.09382.pdf

    这个函数取自 rwightman。
    可以在这里看到：
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    每个样本的下降路径（随机深度）（应用于残差块的主路径时）。
    "具有随机深度的深度网络", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# MBConv
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False), norm_layer(out_planes), activation_layer())


# Conv模块替换
class GhostModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_planes = out_planes
        init_channels = math.ceil(out_planes / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_planes, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)  # 这里的输入应为x1，而不是x
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_planes, :, :]


class GhostShuffleConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, groups=2, ratio=2):
        super(GhostShuffleConv, self).__init__()
        self.out_planes = out_planes
        self.groups = groups
        init_channels = math.ceil(out_planes / ratio)
        new_channels = init_channels * (ratio - 1)

        # 主路径：使用 ShuffleNet Group Convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_planes, init_channels, kernel_size, stride,
                      padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        # 生成额外特征（Ghost 计算）
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1,
                      padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def channel_shuffle(self, x):
        """ 通道洗牌操作 """
        batch_size, num_channels, height, width = x.shape
        group_size = num_channels // self.groups
        x = x.view(batch_size, self.groups, group_size, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

    def forward(self, x):
        x1 = self.primary_conv(x)
        x1 = self.channel_shuffle(x1)  # 在 Ghost 主路径中添加通道洗牌
        x2 = self.cheap_operation(x1)  # 生成 Ghost 特征
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_planes, :, :]


class DynamicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, reduction=4):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # 中间降维通道数（用于减少参数量）
        reduced_channels = max(in_channels // reduction, 1)

        # 动态生成卷积核
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_channels,
                      out_channels * in_channels // groups * kernel_size * kernel_size,
                      kernel_size=1)
        )

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()

        # 全局特征聚合，用于动态核生成
        kernel_weights = self.fc(x).view(
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size
        )

        # 重塑输入数据尺寸为[1, batch_size * in_channels, H, W]
        x = x.view(1, batch_size * in_channels, height, width)

        # 执行group卷积，每个样本有独立卷积核
        output = F.conv2d(x, kernel_weights, stride=self.stride,
                          padding=self.kernel_size // 2, groups=batch_size * self.groups)

        # 还原输出尺寸为[batch_size, out_channels, H_out, W_out]
        output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        return output


class DynamicConvBNActivation(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 reduction=4, norm_layer=None, activation_layer=None):
        super(DynamicConvBNActivation, self).__init__()

        self.dynamic_conv = DynamicConv(in_planes, out_planes, kernel_size, stride=stride, groups=groups,
                                        reduction=reduction)
        self.bn = norm_layer(out_planes) if norm_layer else nn.BatchNorm2d(out_planes)
        self.act = activation_layer() if activation_layer else nn.SiLU()

    def forward(self, x):
        x = self.dynamic_conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MixDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1):
        super(MixDepthwiseSeparableConv, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5]
        assert out_channels % len(kernel_sizes) == 0, \
            "out_channels必须是kernel_sizes长度的整数倍"

        branch_out_channels = out_channels // len(kernel_sizes)
        self.branches = nn.ModuleList()

        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            branch = nn.Sequential(
                # Depthwise卷积
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),

                # Pointwise卷积
                nn.Conv2d(in_channels, branch_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_out_channels),
                nn.SiLU(inplace=True)
            )
            self.branches.append(branch)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation_rates=None, reduction=4):
        """
        Receptive Field Block (RFB) 实现
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 步长（一般为 1）
        :param dilation_rates: 不同的扩张率，决定不同感受野大小
        :param reduction: 通道缩减比率（用于SE模块）
        """
        super(ReceptiveFieldBlock, self).__init__()

        if dilation_rates is None:
            dilation_rates = [1, 3, 5]
        mid_channels = out_channels // len(dilation_rates)

        # 1x1 基础卷积层 (用于降维)
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.ReLU(inplace=True)

        # 并行扩张卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            ))

        # 1x1 卷积融合特征
        self.conv_fusion = nn.Conv2d(mid_channels * len(dilation_rates), out_channels, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(out_channels)

        # Squeeze-and-Excitation (SE) 机制 (可选)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=False)
        self.relu_se = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1x1(x)))

        # 并行计算多尺度卷积
        out_branches = [branch(x) for branch in self.branches]
        out = torch.cat(out_branches, dim=1)

        # 1x1 卷积融合
        out = self.bn_fusion(self.conv_fusion(out))

        # SE 注意力机制
        se = self.global_avg_pool(out)
        se = self.relu_se(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        out = out * se

        return out


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, expansion=4, kernel_size=7):
        """
        ConvNeXt Block 结构：
        1. 7x7 Depthwise Conv
        2. LayerNorm
        3. 1x1 Pointwise Conv (升维)
        4. GELU 激活
        5. 1x1 Pointwise Conv (降维)
        6. 残差连接
        """
        super(ConvNeXtBlock, self).__init__()
        mid_channels = in_channels * expansion  # 扩展通道数

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                stride=1, padding=kernel_size // 2, groups=in_channels)  # 深度可分离卷积

        self.norm = nn.LayerNorm(in_channels, eps=1e-6)  # LayerNorm 替代 BatchNorm
        self.pwconv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)  # 逐点卷积（升维）
        self.act = nn.GELU()  # GELU 激活函数
        self.pwconv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1)  # 逐点卷积（降维）

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # 转换维度以适配 LayerNorm
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # 还原回原来的维度
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual  # 残差连接


class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 norm_layer=None, activation_layer=None):
        super(FusedMBConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            norm_layer(out_channels),
            activation_layer()
        )

    def forward(self, x):
        return self.block(x)


# se模块
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,  # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.SiLU):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # alias F.adaptive_avg_pool2d(x, output_size=(1, 1))

        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = activation_layer()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.global_avg_pool(x)
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, attention_type, drop_connect_rate
    def __init__(self,
                 kernel: int,  # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,  # 1 or 2
                 attention_type: str,  # se, cbam, ca
                 ghost_conv: bool,  # False
                 gs_conv: bool,  # False
                 use_fused: bool,  # False
                 drop_rate: float,
                 index: str,  # 1a, 2a, 2b, ...
                 width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_ratio = expanded_ratio
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.attention_type = attention_type
        self.ghost_conv = ghost_conv
        self.use_fused = use_fused
        self.gs_conv = gs_conv
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module],
                 squeeze_factor: int = 4,
                 activation_layer: Callable[..., nn.Module] = nn.SiLU):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        # activation_layer = nn.SiLU  # alias Swish

        if cnf.use_fused:
            if cnf.expanded_ratio != 1:
                # 使用FusedMBConvBlock
                layers.update({"expand_conv": FusedMBConvBlock(cnf.input_c,
                                                               cnf.expanded_c,
                                                               kernel_size=cnf.kernel,
                                                               stride=cnf.stride,
                                                               norm_layer=norm_layer,
                                                               activation_layer=activation_layer)})
        else:
            # 扩张卷积
            if cnf.expanded_c != cnf.input_c:
                if cnf.ghost_conv:
                    layers.update({"expand_conv": GhostModule(cnf.input_c,
                                                              cnf.expanded_c,
                                                              kernel_size=1,
                                                              stride=1,
                                                              relu=True)})
                elif cnf.gs_conv:
                    layers.update({"expand_conv": GhostShuffleConv(cnf.input_c,
                                                                   cnf.expanded_c,
                                                                   kernel_size=1,
                                                                   stride=1)})
                else:
                    layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                                   cnf.expanded_c,
                                                                   kernel_size=1,
                                                                   norm_layer=norm_layer,
                                                                   activation_layer=activation_layer)})

            # 深度卷积
            layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                      cnf.expanded_c,
                                                      kernel_size=cnf.kernel,
                                                      stride=cnf.stride,
                                                      groups=cnf.expanded_c,
                                                      norm_layer=norm_layer,
                                                      activation_layer=activation_layer)})
            if cnf.attention_type == "se":
                layers.update({"se": SqueezeExcitation(cnf.input_c, cnf.expanded_c, squeeze_factor=squeeze_factor,
                                                       activation_layer=activation_layer)})
            elif cnf.attention_type == "cbam":
                layers.update({"cbam": CBAM(cnf.expanded_c)})
            elif cnf.attention_type == "ca":
                layers.update({"coord": CoordAtt(cnf.expanded_c)})
            elif cnf.attention_type == "eca":
                layers.update({"eca": ECAAttention(cnf.expanded_c)})
            else:
                raise ValueError("illegal attention type.")

        # 投影卷积
        if cnf.ghost_conv:
            layers.update({"project_conv": GhostModule(cnf.expanded_c,
                                                       cnf.out_c,
                                                       kernel_size=1,
                                                       stride=1,
                                                       relu=False)})
        elif cnf.gs_conv:
            layers.update({"project_conv": GhostShuffleConv(cnf.expanded_c,
                                                            cnf.out_c,
                                                            kernel_size=1,
                                                            stride=1)})
        else:
            layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                            cnf.out_c,
                                                            kernel_size=1,
                                                            norm_layer=norm_layer,
                                                            activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 10,
                 squeeze_factor: int = 4,
                 dropout_rate: float = 0.2,
                 drop_rate: float = 0.2,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.SiLU,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 classifier_modify: bool = False,
                 attention_type: str = "se",
                 ghost_conv: bool = False,
                 use_fused: bool = False,
                 gs_conv: bool = False,
                 ):
        super(EfficientNet, self).__init__()

        exp_ratio = 4 if use_fused else 6
        # kernel_size, in_channel, out_channel, exp_ratio, strides,
        # attention_type, use_ghost_conv, use_gs_conv, use_fused , drop_connect_rate, repeats
        default_cnf = [
            [3, 32, 16, 1, 1, attention_type, ghost_conv, gs_conv, use_fused, drop_rate, 1],  # stage 2
            [3, 16, 24, exp_ratio, 2, attention_type, ghost_conv, gs_conv, use_fused, drop_rate, 2],  # stage 3
            [5, 24, 40, exp_ratio, 2, attention_type, ghost_conv, gs_conv, use_fused, drop_rate, 2],  # stage 4
            [3, 40, 80, 6, 2, attention_type, ghost_conv, gs_conv, False, drop_rate, 3],  # stage 5
            [5, 80, 112, 6, 1, attention_type, ghost_conv, gs_conv, False, drop_rate, 3],  # stage 6
            [5, 112, 192, 6, 2, attention_type, ghost_conv, gs_conv, False, drop_rate, 4],  # stage 7
            [3, 192, 320, 6, 1, attention_type, ghost_conv, gs_conv, False, drop_rate, 1]]  # stage 8

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[4] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        if ghost_conv:
            layers.update({"stem_conv": GhostModule(in_planes=3,
                                                    out_planes=adjust_channels(32),
                                                    kernel_size=3,
                                                    stride=2,
                                                    relu=True)})
        else:
            layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                         out_planes=adjust_channels(32),
                                                         kernel_size=3,
                                                         stride=2,
                                                         norm_layer=norm_layer,
                                                         activation_layer=activation_layer)})

        # building inverted residual blocks
        for cnf in inverted_residual_setting: layers.update(
            {cnf.index: block(cnf, norm_layer, squeeze_factor, activation_layer)})

        # build top
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer,
                                               activation_layer=activation_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))

        if not classifier_modify:
            classifier.append(nn.Linear(last_conv_output_c, num_classes))
        else:
            classifier.append(nn.Linear(last_conv_output_c, 512))
            classifier.append(nn.ReLU(inplace=True))
            classifier.append(nn.Linear(512, num_classes))

        self.classifier = nn.Sequential(*classifier)
        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=10, activation_layer=nn.SiLU, classifier_modify=False, attention_type="se",
                    ghost_conv=False, use_fused=False, gs_conv=False):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes,
                        activation_layer=activation_layer,
                        classifier_modify=classifier_modify,
                        attention_type=attention_type,
                        ghost_conv=ghost_conv,
                        use_fused=use_fused,
                        gs_conv=gs_conv,
                        )


def efficientnet_b1(num_classes=10):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=10):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=10):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=10):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=10):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=10):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=10):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)
