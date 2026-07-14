"""YOLO Learn: 系统学习 YOLO 目标检测 —— 从原理到实践."""

# 核心领域类型
from yolo_learn.data import (
    count_boxes_per_image,
    count_classes_in_dataset,
    print_dataset_summary,
    read_data_yaml,
)
from yolo_learn.data.augment import get_augment_params
from yolo_learn.data.download import download_coco128
from yolo_learn.eval import (
    compute_ap,
    compute_iou,
    compute_iou_batch,
    non_max_suppression,
)
from yolo_learn.eval.evaluate import (
    Annotation,
    EvaluationResult,
    Prediction,
    evaluate_predictions,
    load_class_names,
)
from yolo_learn.models.export import export_model, list_formats
from yolo_learn.models.predict import predict, predict_and_extract
from yolo_learn.models.train import train
from yolo_learn.viz import draw_boxes_pil, read_yolo_label, yolo_to_xyxy

__all__ = [
    # Data
    "count_classes_in_dataset",
    "count_boxes_per_image",
    "read_data_yaml",
    "print_dataset_summary",
    "get_augment_params",
    "download_coco128",
    # Eval
    "compute_iou",
    "compute_iou_batch",
    "non_max_suppression",
    "compute_ap",
    "Annotation",
    "Prediction",
    "EvaluationResult",
    "evaluate_predictions",
    "load_class_names",
    # Models
    "train",
    "export_model",
    "list_formats",
    "predict",
    "predict_and_extract",
    # Viz
    "draw_boxes_pil",
    "yolo_to_xyxy",
    "read_yolo_label",
]
