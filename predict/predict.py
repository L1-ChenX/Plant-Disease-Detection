# -*- coding: utf-8 -*-
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

from models.model import create_model
from utils.utils import load_latest_model


def predict_images(model, data_transform, device):
    # 读取 class_indict
    json_path = os.path.join("..", "train", "class_indices.json")  # 对应图像标签(json格式)路径
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 预测所有图像
    test_img_dir = os.path.join(".", "test_img")
    result_dir = os.path.join(".", "result_img")

    os.makedirs(result_dir, exist_ok=True)
    # rm result_dir
    if os.path.exists(result_dir):
        for file in os.listdir(result_dir):
            file_path = os.path.join(result_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                os.rmdir(file_path)

    result_file = os.path.join(result_dir, "predict.txt")
    with open(result_file, "w") as data:
        for img_name in os.listdir(test_img_dir):
            img_path = os.path.join(test_img_dir, img_name)
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue  # 跳过非图像文件

            img = Image.open(img_path).convert("RGB")
            img_tensor = data_transform(img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)
            # print(img_tensor.shape)
            with torch.no_grad():
                output = torch.squeeze(model(img_tensor)).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = f"class: {class_indict[str(predict_cla)]}   prob: {predict[predict_cla]:.3f}"
            print(print_res)

            # 保存预测结果
            data.write(
                f"image: {img_name}   class: {class_indict[str(predict_cla)]}   prob: {predict[predict_cla]:.9f}\n")

            # 保存预测图像
            plt.imshow(img)
            plt.title(print_res)
            plt.axis("off")
            save_path = os.path.join(result_dir, f"{img_name.split('.')[0]}_pred.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

    print(f"All predictions saved to {result_file}")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    # create models
    model_name = "efficientnet_b0"
    model = create_model(model_name=model_name, num_classes=24).to(device)  # 创建模型

    # 加载模型权重
    load_latest_model(model, model_name, device)
    model.eval()

    predict_images(model, data_transform, device)
