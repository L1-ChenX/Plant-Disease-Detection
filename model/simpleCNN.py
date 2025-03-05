from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# 🔹 DropBlock (替代 Dropout)
# ------------------------
class DropBlock2D(nn.Module):
    def __init__(self, block_size=3, drop_prob=0.1):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = torch.bernoulli(torch.full(x.shape, gamma, device=x.device, dtype=x.dtype))
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        return x * mask * mask.numel() / mask.sum()


# ------------------------
# 🔹 SE (Squeeze-Excitation) 模块增强版
# ------------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se_weight = self.global_avg_pool(x)
        se_weight = self.act(self.fc1(se_weight))
        se_weight = self.sigmoid(self.fc2(se_weight))
        return x * se_weight


# ------------------------
# 🔹 改进 Inverted Residual Block
# ------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, stride, use_se=True, drop_prob=0.2):
        super(InvertedResidual, self).__init__()

        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = OrderedDict()
        layers["expand_conv"] = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        layers["bn1"] = nn.BatchNorm2d(hidden_dim)
        layers["act1"] = nn.Mish()

        layers["depthwise_conv"] = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                             padding=kernel_size // 2, groups=hidden_dim, bias=False)
        layers["bn2"] = nn.BatchNorm2d(hidden_dim)
        layers["act2"] = nn.Mish()

        if use_se:
            layers["se"] = SEBlock(hidden_dim)

        layers["project_conv"] = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        layers["bn3"] = nn.BatchNorm2d(out_channels)

        self.conv = nn.Sequential(layers)
        self.dropblock = DropBlock2D(block_size=3, drop_prob=drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.dropblock(out)
        if self.use_res_connect:
            out += x
        return out


# ------------------------
# 🔹 改进 EfficientNet-B0
# ------------------------
class ModifiedEfficientNetB0(nn.Module):
    def __init__(self, num_classes=10, width_coefficient=1.1, depth_coefficient=1.2, dropout_rate=0.2):
        super(ModifiedEfficientNetB0, self).__init__()

        def adjust_channels(channels):
            return max(8, int(channels * width_coefficient) // 8 * 8)

        self.stem = nn.Sequential(
            nn.Conv2d(3, adjust_channels(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(adjust_channels(32)),
            nn.Mish()
        )

        self.blocks = nn.Sequential(
            InvertedResidual(adjust_channels(32), adjust_channels(16), 1, 3, 1, True, 0.2),
            InvertedResidual(adjust_channels(16), adjust_channels(24), 6, 3, 2, True, 0.2),
            InvertedResidual(adjust_channels(24), adjust_channels(40), 6, 5, 2, True, 0.2),
            InvertedResidual(adjust_channels(40), adjust_channels(80), 6, 3, 2, True, 0.2),
            InvertedResidual(adjust_channels(80), adjust_channels(112), 6, 5, 1, True, 0.2),
            InvertedResidual(adjust_channels(112), adjust_channels(192), 6, 5, 2, True, 0.2),
            InvertedResidual(adjust_channels(192), adjust_channels(320), 6, 3, 1, True, 0.2)
        )

        self.head = nn.Sequential(
            nn.Conv2d(adjust_channels(320), adjust_channels(1280), kernel_size=1, bias=False),
            nn.BatchNorm2d(adjust_channels(1280)),
            nn.Mish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(adjust_channels(1280), 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x).flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ------------------------
# ✅ 测试模型
# ------------------------
if __name__ == "__main__":
    model = ModifiedEfficientNetB0(num_classes=10)
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print(model)
    print("Output shape:", output.shape)
