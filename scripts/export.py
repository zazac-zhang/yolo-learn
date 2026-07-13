"""模型导出脚本 - 将训练好的模型导出为各种格式."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def export_model(model_path, format="onnx", **kwargs):
    """导出 YOLO 模型.

    Args:
        model_path: .pt 模型文件路径
        format: 导出格式 (onnx, coreml, torchscript 等)
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"导出格式: {format}")
    export_path = model.export(format=format, **kwargs)

    print(f"导出完成: {export_path}")
    return export_path


def main():
    parser = argparse.ArgumentParser(description="导出 YOLO 模型")
    parser.add_argument("model", help="模型文件路径 (.pt)")
    parser.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "coreml", "torchscript", "tflite", "paddle", "ncnn"],
        help="导出格式",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")

    args = parser.parse_args()

    export_model(args.model, format=args.format, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
