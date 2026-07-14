"""导出模块：将训练好的模型导出为 ONNX、CoreML 等格式."""

from pathlib import Path

from ultralytics import YOLO


def export_model(
    model_path: str | Path,
    format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
    **kwargs,
) -> Path:
    """导出 YOLO 模型.

    Args:
        model_path: 模型权重路径 (.pt)
        format: 导出格式 ("onnx", "coreml", "torchscript", "tflite", "paddle", "ncnn")
        imgsz: 输入图片尺寸
        half: 是否使用半精度 (FP16)
        simplify: 是否简化 ONNX 模型
        **kwargs: 其他导出参数

    Returns:
        Path: 导出文件路径
    """
    model = YOLO(str(model_path))
    result = model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        **kwargs,
    )
    return Path(result)


SUPPORTED_FORMATS = {
    "onnx": {"ext": ".onnx", "desc": "ONNX (通用格式)"},
    "coreml": {"ext": ".mlpackage", "desc": "CoreML (Apple 设备)"},
    "torchscript": {"ext": ".torchscript", "desc": "TorchScript"},
    "tflite": {"ext": ".tflite", "desc": "TensorFlow Lite (移动端)"},
    "paddle": {"ext": "_paddle_model", "desc": "PaddlePaddle"},
    "ncnn": {"ext": "_ncnn_model", "desc": "NCNN (移动端)"},
}


def list_formats() -> None:
    """打印支持的导出格式."""
    print("支持的导出格式:")
    print("-" * 40)
    for fmt, info in SUPPORTED_FORMATS.items():
        print(f"  {fmt:<15} {info['desc']}")
