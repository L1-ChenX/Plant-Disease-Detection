# -*- coding: utf-8 -*-
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model.model import create_model
from utils.utils import MyDataSet, read_split_data


@torch.no_grad()
def test_accuracy(model, data_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def test_acc(model, data_transform, device, batch_size=64):
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./.."))  # 数据集根目录
    train_images_path, train_images_label, \
        test_images_path, test_images_label = read_split_data(data_root)  # 读取数据集，默认使用增强后的数据集

    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=0)
    # loss, accuracy = test_model(model=model, data_loader=test_loader, device=device, epoch=1)
    loss, accuracy = test_accuracy(model, test_loader, device)
    print(f"Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


def predict_images(model, data_transform, device):
    # 读取 class_indict
    json_path = '../train/class_indices.json'  # 对应图像标签(json格式)路径
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

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

    # create model
    # model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=38).to(device)
    model = create_model(model_name="cnn", num_classes=71).to(device)  # 创建模型

    # 加载模型权重
    model_weight_path = "../train/save_weight/cnn/cnn_19.pth"  # 训练保存的权重路径
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    # weights_dict = torch.load(model_weight_path, map_location=device)
    # model_dict = model.state_dict()
    # model.load_state_dict({k: v for k, v in weights_dict.items() if k in model_dict})
    # model.eval()

    test_acc(model, data_transform, device, batch_size=256)
    # predict_images(model, data_transform, device)
