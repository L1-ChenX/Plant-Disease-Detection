# -*- coding: utf-8 -*-
import os

import torch.utils.data
from torchvision import transforms

from models.model import create_model
from utils.utils import MyDataSet, read_split_data, load_latest_model, evaluate_model

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
    # model_weight_path = "../train/save_weights/efficientnet_b0/efficientnet_b0_74.pth"  # 训练保存的权重路径
    # model_weight_path = os.path.join("../train/save_weights", model_name, "efficientnet_b0_74.pth")
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    load_latest_model(model, model_name, device)
    model.eval()
    # weights_dict = torch.load(model_weight_path, map_location=device)
    # model_dict = models.state_dict()
    # models.load_state_dict({k: v for k, v in weights_dict.items() if k in model_dict})
    # models.eval()

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./.."))  # 数据集根目录
    train_images_path, train_images_label, \
        test_images_path, test_images_label = read_split_data(data_root)  # 读取数据集

    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=0)

    # test_acc(model, data_transform, device, batch_size=256)
    evaluate_model(model, test_loader, device)
    # predict_images(models, data_transform, device)
