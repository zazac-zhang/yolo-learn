"""训练脚本 - 薄 CLI 壳，核心逻辑在 yolo_learn.models.train."""

import argparse

from yolo_learn.models.train import train


def main():
    parser = argparse.ArgumentParser(description="训练 YOLO 模型")
    parser.add_argument("--model", default="yolo11n.pt", help="预训练模型")
    parser.add_argument("--data", default="configs/data/coco128.yaml", help="数据集配置文件")
    parser.add_argument("--config", default=None, help="训练配置 YAML 文件")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸")
    parser.add_argument("--device", default="auto", help="设备 (auto/cpu/mps/0)")
    parser.add_argument("--name", default="train", help="实验名称")
    parser.add_argument("--output-dir", default="outputs", help="输出根目录")

    args = parser.parse_args()

    train(
        model=args.model,
        data=args.data,
        config=args.config,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
