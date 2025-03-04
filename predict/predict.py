# -*- coding: utf-8 -*-
import json
import os

import matplotlib.pyplot as plt
import timm
import torch
from PIL import Image
from torchvision import transforms


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义预测时的预处理方法,如下所示:------------------------->>>
    # 此部分需要参赛队伍添加，和测试的预处理方法保持一致。
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

    # 读取 class_indict
    json_path = '../train/class_indices.json'  # 对应图像标签(json格式)路径
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=38).to(device)
    # model = create_model(num_classes=27).to(device)  # 创建模型

    # 加载模型权重
    model_weight_path = "../train/save_weight/model-b0-1.pth"  # 训练保存的权重路径
    weights_dict = torch.load(model_weight_path, map_location=device)
    model_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in weights_dict.items() if k in model_dict})
    model.eval()

    # 预测所有图像
    test_img_dir = "./test_img"
    result_dir = "./result_img"
    os.makedirs(result_dir, exist_ok=True)

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
    main()
