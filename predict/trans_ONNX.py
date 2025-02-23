import timm
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=38).to(device)
model_weight_path = "../train/save_weight/model-b0-1.pth"  # 训练保存的权重路径
weights_dict = torch.load(model_weight_path, map_location=device)
model_dict = model.state_dict()
model.load_state_dict({k: v for k, v in weights_dict.items() if k in model_dict})
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

output = model(x)
print(output.shape)

x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    torch.onnx.export(
        model,  # 要转换的模型
        x,  # 模型的任意一组输入
        './ONNX/efficientnet_b0_PlantDiseases38_1.onnx',  # 导出的 ONNX 文件名
        opset_version=11,  # ONNX 算子集版本
        input_names=['input'],  # 输入 Tensor 的名称（自己起名字）
        output_names=['output']  # 输出 Tensor 的名称（自己起名字）
    )
