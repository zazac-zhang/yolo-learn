"""推理模块：封装 ultralytics 推理，统一接口."""

from pathlib import Path

from ultralytics import YOLO


def predict(
    model_path: str | Path,
    source: str | Path | list,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str = "auto",
    classes: list[int] | None = None,
    **kwargs,
) -> list:
    """运行 YOLO 推理.

    Args:
        model_path: 模型权重路径
        source: 输入源（图片路径、图片列表、目录、视频）
        conf: 置信度阈值
        iou: NMS IoU 阈值
        imgsz: 输入图片尺寸
        device: 设备
        classes: 只检测指定类别（可选）
        **kwargs: 其他 ultralytics 参数

    Returns:
        list: 检测结果列表
    """
    model = YOLO(str(model_path))
    results = model(
        source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        classes=classes,
        **kwargs,
    )
    return results


def predict_and_extract(
    model_path: str | Path,
    source: str | Path,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
) -> list[dict]:
    """推理并提取结构化结果.

    Args:
        model_path: 模型权重路径
        source: 输入图片路径
        conf: 置信度阈值
        iou: NMS IoU 阈值

    Returns:
        list[dict]: 每个检测结果包含 class_id, class_name, confidence, box (xyxy)
    """
    results = predict(model_path, source, conf=conf, iou=iou)
    result = results[0]  # 单张图片

    detections = []
    for box in result.boxes:
        detections.append(
            {
                "class_id": int(box.cls),
                "class_name": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "box": box.xyxy[0].tolist(),  # xyxy 格式
            }
        )
    return detections
