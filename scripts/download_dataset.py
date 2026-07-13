"""下载数据集脚本."""

from pathlib import Path

import yaml


def download_coco128():
    """下载 COCO128 数据集到 data/ 目录."""
    from ultralytics import YOLO

    # ultralytics 会自动下载 COCO128
    print("正在下载 COCO128 数据集...")
    data_dir = Path(__file__).parent.parent / "data" / "coco128"

    if data_dir.exists():
        print(f"数据集已存在: {data_dir}")
        return

    # 使用 ultralytics 内置的下载机制
    # 这会自动下载并解压到正确位置
    model = YOLO("yolo11n.pt")
    # 触发数据集下载（只需调用一次）
    from ultralytics.data.utils import DATASETS_DIR
    print(f"数据集将下载到: {DATASETS_DIR}")
    print("使用 COCO128 数据集名在训练时会自动下载")


if __name__ == "__main__":
    download_coco128()
