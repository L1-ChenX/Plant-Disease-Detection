import os

import torch

from models.model import create_model
from utils.utils import load_latest_model

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name = "eca"
    num_classes = 71
    model = create_model(model_name, num_classes).to(device)  # 创建模型

    load_latest_model(model, model_name, device)
    model.eval()

    X = torch.randn(1, 3, 224, 224).to(device)
    print(model(X).shape)

    with torch.no_grad():
        torch.onnx.export(
            model,  # 要转换的模型
            X,  # 模型的任意一组输入
            # './ONNX/resnet50_24.onnx',  # 导出的 ONNX 文件名
            os.path.join(".", "ONNX", f"{model_name}_{num_classes}.onnx"),
            opset_version=11,  # ONNX 算子集版本
            input_names=['input'],  # 输入 Tensor 的名称（自己起名字）
            output_names=['output']  # 输出 Tensor 的名称（自己起名字）
        )
