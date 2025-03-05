import os
import shutil


def merge_folders(source_dirs, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"源目录 '{source_dir}' 不存在，跳过。")
            continue

        for root, dirs, files in os.walk(source_dir):
            for dir_name in dirs:
                src_dir_path = os.path.join(root, dir_name)
                relative_path = os.path.relpath(src_dir_path, source_dir)
                dest_dir_path = os.path.join(destination_dir, relative_path)

                if not os.path.exists(dest_dir_path):
                    os.makedirs(dest_dir_path)

                for file_name in os.listdir(src_dir_path):
                    src_file_path = os.path.join(src_dir_path, file_name)
                    dest_file_path = os.path.join(dest_dir_path, file_name)

                    if os.path.exists(dest_file_path):
                        pass
                    else:
                        shutil.copy2(src_file_path, dest_file_path)


if __name__ == "__main__":
    base_path = r"/data_set/New Plant Diseases Dataset"
    test_dir = os.path.join(base_path, 'test')
    train_dir = os.path.join(base_path, 'train')
    data_dir = os.path.join(base_path, 'data')

    merge_folders([test_dir, train_dir], data_dir)
