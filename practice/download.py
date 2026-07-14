"""
下载安全帽检测数据集（Roboflow）.

使用方法:
    1. 注册 https://roboflow.com/ 并获取 API Key
    2. 运行此脚本:
       uv run python practice/download.py
       或设置环境变量: export ROBOFLOW_API_KEY="your_key"
"""

import os
import sys
from pathlib import Path

from yolo_learn.data.download import download_roboflow_dataset


def download_dataset():
    """从 Roboflow 下载安全帽检测数据集."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        api_key = input("请输入 Roboflow API Key (从 https://app.roboflow.com/settings 获取): ").strip()
        if not api_key:
            print("❌ 需要 API Key 才能下载数据集")
            print("   注册: https://roboflow.com/")
            print("   设置: export ROBOFLOW_API_KEY='your_key'")
            sys.exit(1)

    print("正在下载数据集...")

    data_dir = download_roboflow_dataset(
        api_key=api_key,
        workspace="safety-helmet-detection-g9amj",
        project="ai-workshop-ft5dq",
        version=1,
        target_dir=Path(__file__).parent / "data",
    )

    print(f"\n✅ 数据集已下载到: {data_dir}")
    train_imgs = list((data_dir / "train" / "images").glob("*"))
    print(f"   包含 {len(train_imgs)} 张训练图片")

    # 生成 data.yaml（如果不存在）
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        import yaml

        yaml_content = {
            "path": str(data_dir.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 3,
            "names": {0: "helmet", 1: "person", 2: "vest"},
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print(f"   data.yaml 已生成: {yaml_path}")

    print("\n开始挑战吧！参考 README.md 了解任务要求。")


if __name__ == "__main__":
    download_dataset()
