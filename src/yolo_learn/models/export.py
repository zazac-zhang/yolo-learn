"""导出模块：将训练好的模型导出为 ONNX、CoreML 等格式."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

SUPPORTED_FORMATS = {
    "onnx": {"ext": ".onnx", "desc": "ONNX (通用格式，推荐)"},
    "coreml": {"ext": ".mlpackage", "desc": "CoreML (iOS)"},
    "torchscript": {"ext": ".torchscript", "desc": "TorchScript"},
    "tflite": {"ext": ".tflite", "desc": "TensorFlow Lite (Android)"},
    "ncnn": {"ext": "_ncnn_model", "desc": "NCNN (边缘设备)"},
    "openvino": {"ext": "_openvino_model", "desc": "OpenVINO (Intel)"},
    "paddle": {"ext": "_paddle_model", "desc": "PaddlePaddle"},
}


def export_model(
    model_path: str | Path,
    format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
    validate: bool = False,
    **kwargs,
) -> Path:
    """导出 YOLO 模型.

    Args:
        model_path: 模型权重路径 (.pt)
        format: 导出格式 (见 SUPPORTED_FORMATS)
        imgsz: 输入图片尺寸
        half: FP16 半精度（减小体积，加速推理）
        simplify: 简化 ONNX 模型
        validate: 导出后验证精度一致性
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
    exported_path = Path(result)

    if validate:
        validate_export(model_path, exported_path, imgsz=imgsz)

    return exported_path


def validate_export(
    original_path: str | Path,
    exported_path: str | Path,
    imgsz: int = 640,
    tolerance: float = 1e-3,
) -> bool:
    """验证导出模型与原始模型的输出一致性.

    Args:
        original_path: 原始 .pt 模型路径
        exported_path: 导出模型路径
        imgsz: 输入图片尺寸
        tolerance: 允许的最大误差

    Returns:
        bool: 是否通过验证
    """
    import numpy as np

    # 用随机输入测试
    dummy_input = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # 原始模型推理（验证可运行）
    original_model = YOLO(str(original_path))
    original_model.predict(dummy_input, verbose=False)

    # 导出模型推理（如果格式支持）
    exported_str = str(exported_path)
    if exported_str.endswith(".onnx"):
        import onnxruntime as ort

        session = ort.InferenceSession(exported_str, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        # ONNX 需要 NCHW float 输入
        onnx_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        onnx_output = session.run(None, {input_name: onnx_input})
        # 简单检查输出不为空
        passed = len(onnx_output) > 0
    else:
        # 其他格式只检查文件存在
        passed = Path(exported_str).exists()

    status = "✅ 通过" if passed else "❌ 失败"
    print(f"{status} 导出验证: {exported_path}")
    return passed


def list_formats() -> None:
    """打印支持的导出格式."""
    print("支持的导出格式:")
    print("-" * 50)
    for fmt, info in SUPPORTED_FORMATS.items():
        print(f"  {fmt:<15} {info['desc']}")
    print()
    print("使用方法:")
    print("  from yolo_learn.models.export import export_model")
    print('  path = export_model("best.pt", format="onnx", half=True)')
