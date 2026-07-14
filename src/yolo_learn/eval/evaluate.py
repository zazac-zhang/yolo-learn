"""评估模块：完整的评估流程，基于 metrics.py 的基础指标."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from yolo_learn.eval.metrics import compute_ap, compute_iou


@dataclass
class Annotation:
    """Ground-truth 标注."""

    class_id: int
    box: list[float]  # [x1, y1, x2, y2] xyxy 像素坐标


@dataclass
class Prediction:
    """模型预测结果."""

    class_id: int
    box: list[float]  # [x1, y1, x2, y2] xyxy 像素坐标
    confidence: float


@dataclass
class EvaluationResult:
    """一次评估的完整结果."""

    mAP: float
    per_class_ap: dict[int, float] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)
    num_images: int = 0
    num_predictions: int = 0
    num_annotations: int = 0

    def summary(self) -> str:
        """打印评估摘要."""
        lines = [
            "=" * 50,
            "评估结果",
            "=" * 50,
            f"mAP@0.5: {self.mAP:.4f}",
            f"图片数: {self.num_images}",
            f"预测数: {self.num_predictions}",
            f"标注数: {self.num_annotations}",
        ]
        if self.per_class_ap:
            lines.append("-" * 50)
            lines.append("各类别 AP:")
            for cls_id, ap in self.per_class_ap.items():
                name = self.class_names.get(cls_id, f"class_{cls_id}")
                lines.append(f"  {name}: {ap:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


def evaluate_predictions(
    predictions: list[dict],
    annotations: list[dict],
    iou_threshold: float = 0.5,
    class_names: dict[int, str] | None = None,
) -> EvaluationResult:
    """评估预测结果.

    Args:
        predictions: 预测列表，每个包含 class_id, box (xyxy), confidence
        annotations: 标注列表，每个包含 class_id, box (xyxy)
        iou_threshold: IoU 阈值
        class_names: 类别 ID → 名称映射

    Returns:
        EvaluationResult: 评估结果
    """
    # 按类别分组
    pred_by_class: dict[int, list] = {}
    for p in predictions:
        cls = p["class_id"]
        pred_by_class.setdefault(cls, []).append(p)

    ann_by_class: dict[int, list] = {}
    for a in annotations:
        cls = a["class_id"]
        ann_by_class.setdefault(cls, []).append(a)

    # 计算每个类别的 AP
    all_classes = set(pred_by_class.keys()) | set(ann_by_class.keys())
    per_class_ap = {}

    for cls_id in all_classes:
        preds = sorted(
            pred_by_class.get(cls_id, []),
            key=lambda x: x["confidence"],
            reverse=True,
        )
        anns = ann_by_class.get(cls_id, [])

        if not anns:
            per_class_ap[cls_id] = 0.0
            continue

        # 计算 precision/recall
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = set()

        for i, pred in enumerate(preds):
            best_iou = 0.0
            best_idx = -1
            for j, ann in enumerate(anns):
                if j in matched:
                    continue
                iou = compute_iou(pred["box"], ann["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold and best_idx >= 0:
                tp[i] = 1
                matched.add(best_idx)
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / len(anns) if anns else np.zeros(len(preds))
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        per_class_ap[cls_id] = compute_ap(recall, precision)

    # 计算 mAP
    mAP = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0

    return EvaluationResult(
        mAP=mAP,
        per_class_ap=per_class_ap,
        class_names=class_names or {},
        num_images=0,
        num_predictions=len(predictions),
        num_annotations=len(annotations),
    )


def load_class_names(data_yaml: str | Path) -> dict[int, str]:
    """从 data.yaml 加载类别名称.

    Args:
        data_yaml: 数据集配置文件路径

    Returns:
        dict: {class_id: class_name}
    """
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return names
