"""数据集工具：格式转换、统计分析、划分等"""

from collections import Counter
from pathlib import Path


def count_classes_in_dataset(labels_dir):
    """统计 YOLO 标注数据集中各类别的出现次数.

    Args:
        labels_dir: 标注文件目录路径

    Returns:
        Counter: {class_id: count}
    """
    labels_dir = Path(labels_dir)
    class_counts = Counter()

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    return class_counts


def count_boxes_per_image(labels_dir):
    """统计每张图片的标注框数量.

    Returns:
        list of int: 每张图片的框数
    """
    labels_dir = Path(labels_dir)
    counts = []
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file) as f:
            lines = [l for l in f if l.strip()]
            counts.append(len(lines))
    return counts


def read_data_yaml(yaml_path):
    """读取 data.yaml 配置文件.

    Args:
        yaml_path: yaml 文件路径

    Returns:
        dict: 包含 path, train, val, names 等字段
    """
    import yaml

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def print_dataset_summary(yaml_path):
    """打印数据集概要信息."""
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    print("=" * 50)
    print("数据集概要")
    print("=" * 50)
    print(f"路径: {data.get('path', 'N/A')}")
    print(f"训练集: {data.get('train', 'N/A')}")
    print(f"验证集: {data.get('val', 'N/A')}")
    print(f"测试集: {data.get('test', 'N/A')}")
    print(f"类别数: {data.get('nc', 'N/A')}")

    names = data.get("names", [])
    if isinstance(names, dict):
        print(f"类别名: {names}")
    elif isinstance(names, list):
        for i, name in enumerate(names):
            print(f"  {i}: {name}")
    print("=" * 50)
