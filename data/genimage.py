"""
GenImage 数据集读取工具
合并自 SD14.py 和 TestGenImageALL.py
"""
import os
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path


def SD14_read_split_data(root: str):
    random.seed(0)  # 保证随机结果可复现

    assert os.path.exists(root), f"dataset root: {root} does not exist."

    # 定义训练集和验证集的路径
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')

    assert os.path.exists(train_dir), f"train directory: {train_dir} does not exist."
    assert os.path.exists(val_dir), f"validation directory: {val_dir} does not exist."

    # 获取类别列表（从训练集目录获取）
    flower_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
    flower_class.sort()

    # 生成类别与索引的映射关系
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('SD14_class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 初始化数据列表
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    # 处理训练集
    for cla in flower_class:
        cla_train_path = os.path.join(train_dir, cla)
        images = [os.path.join(cla_train_path, img) for img in os.listdir(cla_train_path)
                  if os.path.splitext(img)[-1] in supported]
        images.sort()

        class_idx = class_indices[cla]
        train_images_path.extend(images)
        train_images_label.extend([class_idx] * len(images))

    # 处理验证集
    for cla in flower_class:
        cla_val_path = os.path.join(val_dir, cla)
        images = [os.path.join(cla_val_path, img) for img in os.listdir(cla_val_path)
                  if os.path.splitext(img)[-1] in supported]
        images.sort()

        class_idx = class_indices[cla]
        val_images_path.extend(images)
        val_images_label.extend([class_idx] * len(images))

    # 输出统计信息
    print(f"Training images: {len(train_images_path)}")
    print(f"Validation images: {len(val_images_path)}")

    # 可选：绘制类别分布图
    plot_image = False
    if plot_image:
        plt.figure(figsize=(10, 6))
        plt.bar(flower_class, [len(os.listdir(os.path.join(train_dir, cla))) for cla in flower_class])
        plt.title('Training Data Class Distribution')
        plt.xlabel('Class Name')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_all_train_data(root: str):
    random.seed(0)

    assert os.path.exists(root), f"dataset root: {root} does not exist."

    train_images_path = []
    train_images_label = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    # 固定类别名：ai 和 nature
    class_name_list = ["ai", "nature"]
    class_indices = {name: idx for idx, name in enumerate(class_name_list)}

    # 保存类别索引
    with open("GenImage_class_indices.json", "w", encoding="utf-8") as f:
        json.dump({v: k for k, v in class_indices.items()}, f, indent=4, ensure_ascii=False)

    # 递归查找所有 train 文件夹
    train_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "train":
            train_dirs.append(dirpath)

    print(f"🔍 共找到 {len(train_dirs)} 个 train 文件夹")

    for train_dir in train_dirs:
        for class_name in class_name_list:
            class_path = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"⚠️ 跳过缺失类别：{class_path}")
                continue

            image_files = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if os.path.splitext(f)[-1] in supported
            ]
            label = class_indices[class_name]
            train_images_path.extend(image_files)
            train_images_label.extend([label] * len(image_files))

    print(f"✅ 共收集训练图像数量: {len(train_images_path)}")
    print(f"✅ 类别数量: {len(class_indices)}")

    return train_images_path, train_images_label


def read_val_dataset_GenImage(val_dir):
    """
    统一读取两种结构的验证数据集，返回 ai=0, nature=1 的标签。

    :param val_dir: 子验证集根目录，如 '/path/to/test/ADM' 或 '/path/to/test/progan'
    :return: (image_paths, labels)
    """
    image_paths = []
    labels = []

    val_dir = Path(val_dir)

    # 判断是否为嵌套结构（即存在二级文件夹）
    sub_dirs = [d for d in os.listdir(val_dir) if (val_dir / d).is_dir()]
    is_nested = any([(val_dir / d / 'nature').exists() or (val_dir / d / 'ai').exists() for d in sub_dirs])

    if not is_nested:
        # 简单结构
        ai_dir = val_dir / 'ai'
        nature_dir = val_dir / 'nature'

        if ai_dir.exists():
            ai_images = list(ai_dir.glob('*.*'))
            image_paths += ai_images
            labels += [0] * len(ai_images)

        if nature_dir.exists():
            nature_images = list(nature_dir.glob('*.*'))
            image_paths += nature_images
            labels += [1] * len(nature_images)

    else:
        # 嵌套结构
        for cls_dir in sub_dirs:
            cls_path = val_dir / cls_dir
            if not cls_path.is_dir():
                continue

            ai_dir = cls_path / 'ai'
            nature_dir = cls_path / 'nature'

            if ai_dir.exists():
                ai_images = list(ai_dir.glob('*.*'))
                image_paths += ai_images
                labels += [0] * len(ai_images)

            if nature_dir.exists():
                nature_images = list(nature_dir.glob('*.*'))
                image_paths += nature_images
                labels += [1] * len(nature_images)

    return image_paths, labels
