"""模型导出脚本 - 薄 CLI 壳，核心逻辑在 yolo_learn.models.export."""

import argparse

from yolo_learn.models.export import export_model


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
    parser.add_argument("--half", action="store_true", help="半精度 (FP16)")

    args = parser.parse_args()

    result = export_model(
        args.model,
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
    )
    print(f"导出完成: {result}")


if __name__ == "__main__":
    main()
