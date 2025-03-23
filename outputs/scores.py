# -*- coding: utf-8 -*-
import csv
import os

import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

from models.model import create_model
from utils.utils import MyDataSet, read_split_data, load_latest_model, evaluate_model, get_model_complexity

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

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./.."))  # 数据集根目录
    train_images_path, train_images_label, \
        test_images_path, test_images_label = read_split_data(data_root)  # 读取数据集

    data_transform = transforms.Compose([transforms.Resize(img_size[num_model]),
                                         transforms.CenterCrop(img_size[num_model]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_dataset = MyDataSet(images_path=test_images_path, images_class=test_images_label, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=0)

    results = []
    models = ["efficientnet_b0", "eca", "cbam", "ca"]
    # models = ["eca", "ghost", "gs", "fused"]
    # 测试模型并记录结果
    for model_name in models:
        model = create_model(model_name=model_name, num_classes=71).to(device)
        load_latest_model(model, model_name, device)
        model.eval()

        accuracy, precision, recall, f1, fps, avg_time_ms = evaluate_model(model, test_loader, device)
        flops, params = get_model_complexity(model, device=device)
        results.append([model_name, accuracy, precision, recall, f1, fps, avg_time_ms, flops, params])

    # 打印并输出CSV
    csv_file_path = "model_evaluation_results_1.csv"
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'FPS', 'Avg Time (ms)', 'MFLOPs', 'Params (M)']

    with open(csv_file_path, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for res in results:
            writer.writerow([
                res[0],  # model name
                f"{float(res[1]):.4f}",  # accuracy
                f"{float(res[2]):.4f}",  # precision
                f"{float(res[3]):.4f}",  # recall
                f"{float(res[4]):.4f}",  # f1
                f"{float(res[5]):.2f}",  # fps
                f"{float(res[6]):.2f}",  # avg_time_ms
                f"{float(res[7]):.2f}",  # flops
                f"{float(res[8]):.2f}"  # params
            ])

    print(f"测试完成，结果已保存到 {csv_file_path}.csv")
