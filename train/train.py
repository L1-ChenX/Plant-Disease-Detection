# -*- coding: utf-8 -*-
import sys

import timm
from timm.scheduler import CosineLRScheduler

[sys.path.append(i) for i in ['.', '..']]

import argparse
import os

import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.model import create_model
from utils.utils import plot_class_preds, train_one_epoch, test_model, read_split_data, MyDataSet


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=./runs", '
          'view at http://localhost:6006/')
    log_dir = os.path.join("./runs", args.model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_writer = SummaryWriter(log_dir=log_dir)

    # 定义训练以及测试时的预处理方法
    if args.cloud:
        data_root = r'/home/featurize/data'
    else:
        data_root = os.path.abspath(os.path.join(os.getcwd(), "./.."))  # 数据集根目录
    print("data_root=" + data_root)
    train_images_path, train_images_label, \
        test_images_path, test_images_label = read_split_data(data_root)  # 读取数据集，默认使用增强后的数据集
    img_size = {"B0": 224, "B1": 240, "B2": 260, "B3": 300, "B4": 380, "B5": 456, "B6": 528, "B7": 600}
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size[num_model]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([
            transforms.Resize(img_size[num_model]),
            transforms.CenterCrop(img_size[num_model]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform["test"])
    image_path = os.path.join(data_root, "data_set", "Plant_data")  # 数据集目录
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_num = len(train_dataset)
    test_num = len(test_dataset)
    print("using {} images for training, {} images fot test.".format(train_num, test_num))

    batch_size = args.batch_size

    # 使用num_workers的数量
    num_workers = args.num_workers
    print('Using {} dataloader workers every process'.format(num_workers))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers)

    data, target = next(iter(train_loader))
    print(data.shape)

    # 实例化模型
    model_name = args.model_name
    model = create_model(model_name, num_classes=args.num_classes).to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tb_writer.add_graph(model, init_img)  # 添加网络的结构图

    # 如果存在预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError(
                "not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "SGD":
        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)  # 优化器

    elif args.optimizer == "Adam":
        optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=1e-4)
    else:
        raise ValueError("optimizer not support")

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs - args.warmup_epochs,
        lr_min=args.lrf * args.lr,
        warmup_t=args.warmup_epochs,  # warm-up epoch 数
        warmup_lr_init=args.warmup_lr,  # warm-up开始lr
        warmup_prefix=True,
    )

    best_acc = 0.
    i = 0
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # update learning rate
        scheduler.step(epoch)

        # test
        test_loss, test_acc = test_model(model=model,
                                         data_loader=test_loader,
                                         device=device,
                                         epoch=epoch)

        # add loss, acc and lr into tensorboard
        tags = ["train_loss", "train_acc", "test_loss", "test_acc",
                "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], test_loss, epoch)
        tb_writer.add_scalar(tags[3], test_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # add figure into tensorboard
        fig = plot_class_preds(net=model,
                               images_dir="../plot_img",
                               # 用来在tensorboard中预测的图片路径
                               transform=data_transform["test"],
                               num_plot=10,
                               device=device)

        if fig is not None:
            tb_writer.add_figure("predictions vs. actual",
                                 figure=fig,
                                 global_step=epoch)
        # 定义模型保存路径
        save_dir = os.path.join("save_weights", model_name)
        # save weights
        if best_acc < test_acc:
            os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
            save_path = os.path.join(save_dir, f"{model_name}_{i}.pth")
            torch.save(model.state_dict(), save_path)  # 保存模型权重
            best_acc = test_acc
        i += 1


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=71)  # 图像类别
    parser.add_argument('--epochs', type=int, default=100)  # 训练次数
    parser.add_argument('--warmup_epochs', type=int, default=10)  # warmup训练次数
    parser.add_argument('--batch_size', type=int, default=32)  # 批次大小
    parser.add_argument('--num_workers', type=int, default=0)  # 使用线程数目
    parser.add_argument('--lr', type=float, default=0.001)  # 初始学习率
    parser.add_argument('--lrf', type=float, default=0.01)  # 最终学习率比例
    parser.add_argument('--warmup_lr', type=float, default=1e-6)  # warmup初始学习率

    parser.add_argument('--model_name', type=str, default='efficientnet_b0')  # cnn cbam ca resnet50 vit
    parser.add_argument('--weights', type=str, default='', help='initial weights path')  # 预训练权重路径
    parser.add_argument('--freeze-layers', type=bool, default=False)  # 是否冻结权重
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--cloud', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default="Adam")  # 优化器

    opt = parser.parse_args()

    main(opt)
