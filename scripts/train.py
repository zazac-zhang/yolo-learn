"""训练脚本 - 独立的训练入口，可从命令行运行."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(model_name="yolo11n.pt", data="coco128.yaml", epochs=50, **kwargs):
    """训练 YOLO 模型.

    Args:
        model_name: 预训练模型名或路径
        data: 数据集配置文件路径或名称
        epochs: 训练轮数
        **kwargs: 其他训练参数
    """
    print("=" * 50)
    print(f"模型: {model_name}")
    print(f"数据集: {data}")
    print(f"训练轮数: {epochs}")
    print("=" * 50)

    # 加载模型
    model = YOLO(model_name)

    # 开始训练
    results = model.train(
        data=data,
        epochs=epochs,
        **kwargs,
    )

    print(f"\n训练完成! 结果保存在: {results.save_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="训练 YOLO 模型")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练模型")
    parser.add_argument("--data", default="coco128", help="数据集名或配置文件")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸")
    parser.add_argument("--device", default="mps", help="设备 (cpu/mps/0)")
    parser.add_argument("--name", default="train", help="实验名称")

    args = parser.parse_args()

    train(
        model_name=args.model,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
    )


if __name__ == "__main__":
    main()
