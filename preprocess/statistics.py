import os


def count_images_in_categories(base_dir):
    """ 统计目录中每个类别的图片数量，并返回 {类别: 数量} """
    if not os.path.exists(base_dir):
        return {}

    category_counts = {}
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):  # 确保是目录
            image_count = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
            category_counts[category] = image_count

    return category_counts


def print_category_counts(base_dir):
    """ 统计所有处理阶段的数据量，并格式化对齐输出 """
    stages = {
        "data": os.path.join(base_dir, "data"),
        "filtered": os.path.join(base_dir, "filtered"),
        "under-sampled": os.path.join(base_dir, "under-sampled"),
        "train": os.path.join(base_dir, "train"),
        "test": os.path.join(base_dir, "test"),
        "train-augmented": os.path.join(base_dir, "train-augmented"),
    }

    # 统计所有阶段的数据
    stage_counts = {stage_name: count_images_in_categories(stage_path) for stage_name, stage_path in stages.items()}

    # 获取所有类别的最大名称长度，确保对齐
    all_categories = set()
    for counts in stage_counts.values():
        all_categories.update(counts.keys())
    max_category_length = max(len(cat) for cat in all_categories) if all_categories else 10

    # 计算各阶段的总数
    total_counts = {stage: sum(counts.values()) for stage, counts in stage_counts.items()}

    # 表头
    print("各类数据在不同处理阶段的数量")
    header = f"{'Categories'.ljust(max_category_length)} | " + " | ".join(
        f"{stage.center(15)}" for stage in stages.keys())
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # 按类别输出
    for category in sorted(all_categories):
        row = f"{category.ljust(max_category_length)} | "
        row += " | ".join(f"{str(stage_counts[stage].get(category, 0)).rjust(15)}" for stage in stages.keys())
        print(row)

    # 输出 `Total`（各阶段总数）
    print("=" * len(header))
    total_row = f"{'Total'.ljust(max_category_length)} | "
    total_row += " | ".join(f"{str(total_counts[stage]).rjust(15)}" for stage in stages.keys())
    print(total_row)
    print("=" * len(header))
    print("\n数据统计完成")


if __name__ == "__main__":
    base_directory = r"../data_set/Plant_data"
    print_category_counts(base_directory)
