# -*- coding: utf-8 -*-

import timm
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.EfficientNet import efficientnet_b0
from models.ResNet import resnet50


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


def create_model(model_name, num_classes=10):
    if model_name == "cnn":
        return SimpleCNN(num_classes)
    elif model_name == "resnet50":
        return resnet50(num_classes)
    elif model_name == "efficientnet_b0":
        return efficientnet_b0(num_classes)
    elif model_name == "cbam":
        return efficientnet_b0(num_classes, attention_type="cbam")
    elif model_name == "coord":
        return efficientnet_b0(num_classes, attention_type="coord")
    elif model_name == "eca":
        return efficientnet_b0(num_classes, attention_type="eca")
    elif model_name == "modify":
        return efficientnet_b0(num_classes, attention_type="eca", ghost=True)
    elif model_name == 'vit':
        return timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("model_name not found.")


if __name__ == '__main__':
    model = create_model("modify")
    X = torch.randn(1, 3, 224, 224)
    print(model)
    print(model(X).shape)
