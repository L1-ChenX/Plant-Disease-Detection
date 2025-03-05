import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (1, 16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (1, 32, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # (1, 64, 28, 28)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 测试模型
if __name__ == "__main__":
    model = SimpleCNN(num_classes=50)  # 5 类分类任务
    print(model)
    test_input = torch.randn(1, 3, 224, 224)  # 1 个 batch 的测试输入
    output = model(test_input)
    print("Output shape:", output.shape)  # 应该是 (1, num_classes)
