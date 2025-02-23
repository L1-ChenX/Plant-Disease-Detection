import json

import onnxruntime
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ort_session = onnxruntime.InferenceSession('./ONNX/efficientnet_b0_PlantDiseases38_int8.onnx')
x = torch.randn(1, 3, 224, 224).numpy()
print(x.shape)

# onnx runtime 输入
ort_inputs = {'input': x}

# onnx runtime 输出
ort_output = ort_session.run(['output'], ort_inputs)[0]

print(ort_output.shape)

img_size = {"B0": 224,
            "B1": 240,
            "B2": 260,
            "B3": 300,
            "B4": 380,
            "B5": 456,
            "B6": 528,
            "B7": 600}
num_model = "B0"
data_transform = transforms.Compose(
    [transforms.Resize(img_size[num_model]),
     transforms.CenterCrop(img_size[num_model]),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img_path = 'TomatoYellowCurlVirus5.JPG'

# 用 pillow 载入
img_pil = Image.open(img_path)

input_img = data_transform(img_pil)
print(input_img.shape)
input_tensor = input_img.unsqueeze(0).numpy()
print(input_tensor.shape)

# ONNX 运行
# ONNX Runtime 输入
ort_inputs = {'input': input_tensor}
# ONNX Runtime 输出
pred_logits = ort_session.run(['output'], ort_inputs)[0]
pred_logits = torch.tensor(pred_logits)
print(pred_logits.shape)
pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
print(pred_softmax.shape)
# 解析预测结果
# 取置信度最大的 n 个结果
n = 3
top_n = torch.topk(pred_softmax, n)
print(top_n)

# 预测类别
pred_ids = top_n.indices.numpy()[0]
print(pred_ids)
# 预测置信度
confs = top_n.values.numpy()[0]
print(confs)
# 打印预测结果
# 载入类别索引
with open('../train/class_indices.json', 'r') as f:
    idx_to_labels = json.load(f)

# 打印预测结果
for i in range(3):
    class_name = idx_to_labels[str(pred_ids[i])]  # 需要转字符串
    confidence = confs[i] * 100
    print(f"{class_name:<30} {confidence:.2f}%")
