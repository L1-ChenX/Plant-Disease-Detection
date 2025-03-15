import hashlib
import os
import random
import shutil

import imagehash
import torchvision.transforms as transforms
from PIL import Image


def get_md5_hash(image_path):
    """ 计算图片的 MD5 哈希值（用于检测完全相同的图片） """
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_phash(file_path):
    """ 计算图片的感知哈希（pHash） """
    with Image.open(file_path) as img:
        return imagehash.phash(img)


def ensure_empty_dir(directory):
    """ 确保目录存在且为空 """
    if os.path.exists(directory):
        shutil.rmtree(directory)  # 删除整个目录
    os.makedirs(directory)  # 重新创建


def deduplicate_and_copy_images(base_dir, phash_threshold=5):
    """ 检测重复和相似图片，并复制保留的图片到新文件夹 """
    print("检测重复和相似图片")
    input_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "filtered")
    ensure_empty_dir(output_dir)

    hash_dict = {}  # 记录已保存的图片（MD5 哈希）
    phash_dict = {}  # 记录已保存的图片（pHash）
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        count = 0
        if os.path.isdir(category_path):
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)

                # 计算 MD5 哈希（完全相同）
                md5_hash = get_md5_hash(img_path)
                if md5_hash in hash_dict:
                    count += 1
                    # print(f"发现完全相同图片（已跳过）: {img_path}")
                    continue  # 跳过重复图片

                # 计算 pHash（感知哈希，检测相似图片）
                img_phash = get_phash(img_path)
                found_similar = False
                for existing_phash in phash_dict.keys():
                    if img_phash - existing_phash < phash_threshold:  # 设定相似阈值
                        count += 1
                        # print(f"发现相似图片（已跳过）: {img_path}")
                        found_similar = True
                        break

                if not found_similar:
                    hash_dict[md5_hash] = img_path  # 记录 MD5
                    phash_dict[img_phash] = img_path  # 记录 pHash

                    # 复制图片到新目录
                    dest_path = os.path.join(output_category_path, img_name)
                    shutil.copy2(img_path, dest_path)
                    # print(f"已复制: {img_path} -> {dest_path}")

        print(f"类别 '{category}' 处理完成，保留 {len(os.listdir(output_category_path))} 张图片，剔除 {count} 张图片.")


def undersample_majority_classes(base_dir, target_count=1000):
    """ 通过随机删除图片，减少大类别样本，使其不超过 target_count """
    print("欠采样大类别")
    input_dir = os.path.join(base_dir, "filtered")
    output_dir = os.path.join(base_dir, "under-sampled")
    ensure_empty_dir(output_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if os.path.isdir(category_path):
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            images = os.listdir(category_path)
            current_count = len(images)

            if current_count > target_count:
                print(f"类别 '{category}' 共有 {current_count} 张图片，欠采样至 {target_count} 张")

                # 随机选取 target_count 张图片
                selected_images = random.sample(images, target_count)
            else:
                selected_images = images  # 小类别不删除

            # 复制到新目录
            for img_name in selected_images:
                src_path = os.path.join(category_path, img_name)
                dst_path = os.path.join(output_category_path, img_name)
                shutil.copy2(src_path, dst_path)
            print(f"类别 '{category}' 处理完成，保留 {len(selected_images)} 张图片")


def split_dataset(base_dir, train_ratio):
    """ 将数据集划分为训练集和测试集 """
    print("划分训练集和测试集")
    data_dir = os.path.join(base_dir, 'under-sampled')
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # 创建训练集和测试集目录
    ensure_empty_dir(train_dir)
    ensure_empty_dir(test_dir)

    # 遍历每个类别文件夹
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            random.shuffle(images)
            split_index = int(len(images) * train_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]

            # 创建类别目录
            train_category_dir = os.path.join(train_dir, category)
            test_category_dir = os.path.join(test_dir, category)
            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(test_category_dir, exist_ok=True)

            # 复制训练集图片
            for image in train_images:
                src = os.path.join(category_path, image)
                dst = os.path.join(train_category_dir, image)
                shutil.copy2(src, dst)

            # 复制测试集图片
            for image in test_images:
                src = os.path.join(category_path, image)
                dst = os.path.join(test_category_dir, image)
                shutil.copy2(src, dst)

            print(f"类别 '{category}'：训练集 {len(train_images)} 张，测试集 {len(test_images)} 张")


def augment_images(base_dir, target_count):
    """ 对输入目录的所有类别文件夹中的图片进行数据增强，并保存到新目录 """
    print("数据增强")

    input_dir = os.path.join(base_dir, "train")
    output_dir = os.path.join(base_dir, "train-augmented")
    ensure_empty_dir(output_dir)

    # 遍历所有类别文件夹
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)

        if os.path.isdir(category_path):
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            images = os.listdir(category_path)
            current_count = len(images)
            # print(f"对类别 '{category}' 进行数据增强, 处理 {current_count} 张图片.")
            if current_count < target_count:
                for i in range(target_count - current_count):
                    img_name = random.choice(images)
                    img_path = os.path.join(category_path, img_name)
                    try:
                        with Image.open(img_path) as img:
                            # 确保图片为RGB格式，避免错误
                            img = img.convert("RGB")

                            # 定义数据增强变换
                            transform = transforms.Compose([
                                transforms.RandomRotation(30),  # 随机旋转
                                transforms.ColorJitter(
                                    brightness=(0.9, 1.1),  # 限制亮度变化范围
                                    contrast=(0.9, 1.1),  # 限制对比度变化范围
                                    saturation=(0.9, 1.1),  # 限制饱和度变化范围
                                    hue=(-0.02, 0.02)  # 适当减少色调变化
                                ),
                            ])

                            # 进行数据增强
                            augmented_img = transform(img)

                            # 生成新的文件名
                            new_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                            new_img_path = os.path.join(output_category_path, new_img_name)

                            # 保存增强后的图片
                            augmented_img.save(new_img_path)
                            # print(f"已生成: {new_img_path}")
                    except Exception as e:
                        print(f"处理图片 {img_name} 时出错: {e}")

            for img_name in images:
                src_path = os.path.join(category_path, img_name)
                dst_path = os.path.join(output_category_path, img_name)
                shutil.copy2(src_path, dst_path)

            print(
                f"类别 '{category}': {current_count} 张图片，共生成 {target_count - current_count} 张图片，合计 {len(os.listdir(output_category_path))} 张图片.")


if __name__ == "__main__":
    base_directory = r"../data_set/Plant_data"

    # deduplicate_and_copy_images(base_directory, phash_threshold=5)

    undersample_majority_classes(base_directory, target_count=2000)

    split_dataset(base_directory, train_ratio=0.8)

    augment_images(base_directory, target_count=1600)
