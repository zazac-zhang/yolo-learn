"""
下载安全帽检测数据集（Roboflow）.

使用方法:
    1. 注册 https://roboflow.com/ 并获取 API Key
    2. 运行此脚本:
       python download.py
       或设置环境变量: export ROBOFLOW_API_KEY="your_key"
"""

import os
import sys
from pathlib import Path


def download_dataset():
    """从 Roboflow 下载安全帽检测数据集."""

    try:
        from roboflow import Roboflow
    except ImportError:
        print("正在安装 roboflow SDK...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    # 获取 API Key
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        api_key = input("请输入 Roboflow API Key (从 https://app.roboflow.com/settings 获取): ").strip()
        if not api_key:
            print("❌ 需要 API Key 才能下载数据集")
            print("   注册: https://roboflow.com/")
            print("   设置: export ROBOFLOW_API_KEY='your_key'")
            sys.exit(1)

    print("正在下载数据集...")

    rf = Roboflow(api_key=api_key)

    # Safety Helmet Detection 数据集
    # 来源: https://universe.roboflow.com/
    project = rf.workspace("safety-helmet-detection-g9amj").project("ai-workshop-ft5dq")
    version = project.version(1)

    # 下载 YOLOv8 格式
    data_dir = Path(__file__).parent / "data" / "safety_helmet"
    version.download(model_format="yolov8", location=str(data_dir))

    print(f"\n✅ 数据集已下载到: {data_dir}")
    print(f"   包含 {len(list((data_dir / 'train' / 'images').glob('*')))} 张训练图片")

    # 生成 data.yaml
    yaml_content = f"""# 安全帽检测数据集
path: {data_dir.resolve()}
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: helmet
  1: person
  2: vest
"""
    yaml_path = data_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"   data.yaml 已生成: {yaml_path}")
    print("\n开始挑战吧！参考 README.md 了解任务要求。")


if __name__ == "__main__":
    download_dataset()
