import torch
from torch import nn


class CoordAtt(nn.Module):
    """
    Coordinate Attention (CVPR 2021)
    将通道注意力与坐标信息相结合，适合移动端
    """

    def __init__(self, in_channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)

        # conv1 用于通道压缩 (共享)
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0)

        # conv_h / conv_w 分别映射回原通道
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(self.mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        # 水平池化: [B, C, 1, W]
        x_h = torch.mean(x, dim=2, keepdim=True)
        # 垂直池化: [B, C, H, 1]
        x_w = torch.mean(x, dim=3, keepdim=True).transpose(2, 3)

        # 将 x_h, x_w 合并后用同一个 conv1 做通道降维
        y = torch.cat([x_h, x_w], dim=2)  # 拼在height维
        y = self.conv1(y)
        y = self.bn(y)
        y = self.act(y)
        # 分割成两部分
        x_h, x_w = torch.split(y, [1, y.shape[2] - 1], dim=2)
        # x_h shape: [B, mid_channels, 1, W]
        # x_w shape: [B, mid_channels, (H-1?), W]  -> 需再调 reshape 使之为 [B, mid_channels, W, 1] or similar

        # 假设要让 x_w 变回 [B, mid_channels, 1, H] 再 conv => [B, C, 1, H]
        x_w = x_w.permute(0, 1, 3, 2)  # 做示例变换
        # 分别通过 conv_h, conv_w
        att_h = self.conv_h(x_h)  # [B, C, 1, W]
        att_w = self.conv_w(x_w)  # [B, C, W, 1]
        # 激活
        att_h = self.sigmoid_h(att_h)
        att_w = self.sigmoid_w(att_w)
        # 还原 shape, 并与输入相乘
        out = x * att_h.expand_as(x) * att_w.expand_as(x)
        return out
