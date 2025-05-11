import json
import time  # 用于计时

import onnxruntime
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':

    ort_session = onnxruntime.InferenceSession('ONNX/eca_71.onnx')
    # ort_session = onnxruntime.InferenceSession('ONNX/eca_71.onnx')
    x = torch.randn(1, 3, 224, 224).numpy()
    print(x.shape)

    # onnx runtime 输入
    ort_inputs = {'input': x}
    # onnx runtime 输出(先测试一个dummy输入)
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
    data_transform = transforms.Compose([
        transforms.Resize(img_size[num_model]),
        transforms.CenterCrop(img_size[num_model]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # img_path = '../data_set/Plant_data/71/filtered/Grape___healthy/104812.jpg'
    img_path = ''

    # 用 pillow 载入
    img_pil = Image.open(img_path)
    input_img = data_transform(img_pil)
    print(input_img.shape)

    input_tensor = input_img.unsqueeze(0).numpy()  # [1,3,224,224]
    print(input_tensor.shape)

    # 计时开始
    start_time = time.time()

    # ONNX 推理
    ort_inputs = {'input': input_tensor}
    pred_logits = ort_session.run(['output'], ort_inputs)[0]

    # 计时结束
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000

    print(f"\nInference time for one image: {inference_time:.2f} ms")

    # 后续解析结果
    pred_logits = torch.tensor(pred_logits)
    pred_softmax = F.softmax(pred_logits, dim=1)
    n = 3
    top_n = torch.topk(pred_softmax, n)

    pred_ids = top_n.indices.numpy()[0]
    confs = top_n.values.numpy()[0]

    with open('../train/class_indices.json', 'r') as f:
        idx_to_labels = json.load(f)

    for i in range(n):
        class_name = idx_to_labels[str(pred_ids[i])]
        # class_idx = int(pred_ids[i])
        confidence = confs[i] * 100
        print(f"{class_name:<30} {confidence:.2f}%")
