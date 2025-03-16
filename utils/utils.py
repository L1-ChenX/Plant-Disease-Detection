# -*- coding: utf-8 -*-

import json
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2, augment=True):
    random.seed(4)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    dataset_path = os.path.join(root, "data_set", "Plant_data")

    if augment:
        train_path = os.path.join(dataset_path, "train-augmented")
    else:
        train_path = os.path.join(dataset_path, "train")

    test_path = os.path.join(dataset_path, "test")
    # 遍历文件夹，一个文件夹对应一个类别
    train_class = [cla for cla in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, cla))]
    test_class = [cla for cla in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, cla))]
    # 排序，保证各平台顺序一致
    train_class.sort()
    test_class.sort()
    # 生成类别名称以及对应的数字索引
    train_class_indices = dict((k, v) for v, k in enumerate(train_class))
    test_class_indices = dict((k, v) for v, k in enumerate(test_class))
    json_str = json.dumps(dict((val, key) for key, val in train_class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in train_class:
        train_cla_path = os.path.join(train_path, cla)
        test_cla_path = os.path.join(test_path, cla)
        # 遍历获取supported支持的所有文件路径
        train_images = [os.path.join(train_path, cla, i) for i in os.listdir(train_cla_path)
                        if os.path.splitext(i)[-1] in supported]
        test_images = [os.path.join(test_path, cla, i) for i in os.listdir(test_cla_path)
                       if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        train_images.sort()
        test_images.sort()
        # 获取该类别对应的索引
        train_image_class = train_class_indices[cla]
        test_image_class = test_class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(train_images) + len(test_images))
        # # 按比例随机采样验证样本
        # test_path = random.sample(
        # train_images, k=int(len(train_images) * val_rate))

        for img_path in train_images:
            train_images_path.append(img_path)
            train_images_label.append(train_image_class)
        for img_path in test_images:
            test_images_path.append(img_path)
            test_images_label.append(test_image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for testing.".format(len(test_images_path)))
    assert len(train_images_path) > 0, "number of training images must " \
                                       "greater than 0."
    assert len(test_images_path) > 0, "number of validation images must " \
                                      "greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(train_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(train_class)), train_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, \
        test_images_path, test_images_label


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb', encoding='utf-8') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb', encoding='utf-8') as f:
        info_list = pickle.load(f)
        return info_list


def plot_class_preds(net,  # 实例化的模型
                     images_dir: str,
                     transform,  # 验证集使用的图像预处理
                     num_plot: int = 10,  # 总共需要展示图片数量
                     device="cpu"):
    if not os.path.exists(images_dir):
        print("not found {} path, ignore add figure.".format(images_dir))
        return None

    label_path = os.path.join(images_dir, "label.txt")
    if not os.path.exists(label_path):
        print("not found {} file, ignore add figure".format(label_path))
        return None

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path
                          ), "not found {}".format(json_label_path)
    json_file = open(json_label_path, 'r', encoding='utf-8')
    # {"0": "daisy"}
    flower_class = json.load(json_file)
    # {"daisy": "0"}
    class_indices = dict((v, k) for k, v in flower_class.items())

    # reading label.txt file
    label_info = []
    with open(label_path, "r", encoding='utf-8') as rd:
        for line in rd.readlines():
            line = line.strip()
            if len(line) > 0:
                split_info = [i for i in line.split(" ") if len(i) > 0]
                assert len(split_info) == 2, \
                    "label format error, expect file_name and class_name"
                image_name, class_name = split_info
                image_path = os.path.join(images_dir, image_name)
                # 如果文件不存在，则跳过
                if not os.path.exists(image_path):
                    print("not found {}, skip.".format(image_path))
                    continue
                # 如果读取的类别不在给定的类别内，则跳过
                if class_name not in class_indices.keys():
                    print("unrecognized category {}, skip".format(class_name))
                    continue
                label_info.append([image_path, class_name])

    if len(label_info) == 0:
        return None

    # get first num_plot info
    if len(label_info) > num_plot:
        label_info = label_info[:num_plot]

    num_imgs = len(label_info)
    images = []
    labels = []
    for img_path, class_name in label_info:
        # read img
        img = Image.open(img_path).convert("RGB")
        label_index = int(class_indices[class_name])

        # preprocessing
        img = transform(img)
        images.append(img)
        labels.append(label_index)

    # batching images
    images = torch.stack(images, dim=0).to(device)

    # inference
    with torch.no_grad():
        output = net(images)
        probs, preds = torch.max(torch.softmax(output, dim=1), dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    # width, height
    # fig = plt.figure(figsize=(num_imgs * 2.5, 3), dpi=300)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(18, 8), dpi=200)
    fig.subplots_adjust(wspace=0.5, hspace=0.4)  # 调整水平和垂直间距

    for i, ax in enumerate(axes.flat):
        if i >= num_imgs:
            ax.axis("off")  # 隐藏多余的子图
            continue

        # CHW -> HWC
        npimg = images[i].cpu().numpy().transpose(1, 2, 0)

        # 还原标准化
        npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        ax.imshow(npimg.astype('uint8'))
        ax.axis("off")  # 关闭坐标轴

        title = "{}, {:.2f}%\n(label: {})".format(
            flower_class[str(preds[i])],
            probs[i] * 100,
            flower_class[str(labels[i])]
        )
        ax.set_title(title, fontsize=12,
                     color=("green" if preds[i] == labels[i] else "red"))

    return fig


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()  # 标签平滑 label_smoothing=0.1
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()  # 梯度清零

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()  # 反向传播
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()  # 更新参数
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


class KDLoss(nn.Module):
    """
    Knowledge Distillation Loss
    alpha * CE(student_logits, true_labels) + (1-alpha) * KL(student_logits, teacher_logits, T)
    """

    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # CE Loss
        loss_ce = self.ce(student_logits, labels)
        # KL Divergence with temperature
        teacher_probs = F.softmax(teacher_logits / self.T, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        loss_kd = self.kl(student_log_probs, teacher_probs) * (self.T * self.T)

        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd
        return loss


def distill_one_epoch(model, teacher_model, optimizer, data_loader, device, epoch):
    model.train()
    teacher_model.eval()
    loss_function = KDLoss(alpha=0.5, temperature=4.0)  # 你可以调参数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()  # 梯度清零

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        with torch.no_grad():
            teacher_pred = teacher_model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, teacher_pred, labels.to(device))
        loss.backward()  # 反向传播
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()  # 更新参数
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def My_train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step +
                     loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(
            epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


class MyDataSet(Dataset):

    def __init__(self, images_path: list,
                 images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


@torch.no_grad()
def test_model(model, data_loader, device, epoch):
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

        data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def load_latest_model(model, model_name, device):
    # 获取目录下所有.pth文件，并按修改时间排序
    # weights_dir = os.path.join("../train/save_weights", model_name)
    weights_dir = os.path.join("..", "train", "save_weights", model_name)
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
    assert len(weight_files) > 0, f"No .pth files found in {weights_dir}"

    latest_weight_file = max(
        weight_files,
        key=lambda x: os.path.getmtime(os.path.join(weights_dir, x))
    )

    latest_model_path = os.path.join(weights_dir, latest_weight_file)
    print(f"加载最新的模型权重：{latest_model_path}")

    model.load_state_dict(torch.load(latest_model_path, map_location=device))


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in tqdm(data_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        # 保存所有真实标签和预测标签
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    # 计算所有指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("Accuracy: {:.4f}".format(accuracy))  # 整体准确率
    print("Precision:", precision)  # 计算每个样本的精确率然后求平均值
    print("Recall:", recall)  # 计算每个样本的召回率然后求平均值
    print("F1-score:", f1)  # 计算每个样本的F1-score然后求平均值（P、R调和平均数）

    return accuracy, precision, recall, f1
