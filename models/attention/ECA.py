from torch import nn


class ECAAttention(nn.Module):
    """
    ECA: Efficient Channel Attention
    来自论文: "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    通过 1D 卷积来实现高效的通道注意力
    """

    def __init__(self, channel, k_size=3):
        super(ECAAttention, self).__init__()
        # 1D卷积的卷积核大小 (k_size) 决定了跨通道的交互范围
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B,C,H,W] -> [B,C,1,1]
        self.conv = nn.Conv1d(in_channels=1, out_channels=1,
                              kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, H, W]
        b, c, h, w = x.size()

        # 全局平均池化
        y = self.avg_pool(x)  # [B, C, 1, 1]

        # 1D卷积需要 [B, 1, C] 形状 -> 先把 C 作为时序维度
        y = y.squeeze(-1).transpose(-1, -2)  # [B, C, 1,1] -> [B, C, 1] -> [B,1,C]

        # 进行 1D 卷积
        y = self.conv(y)  # [B,1,C]

        # reshape 回去
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B,1,C] -> [B,C,1] -> [B,C,1,1]
        y = self.sigmoid(y)  # 通道注意力

        # 与输入 x 做逐通道乘法
        return x * y.expand_as(x)