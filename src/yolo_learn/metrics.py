"""指标计算工具：IoU、mAP 等核心指标的手动实现，帮助理解原理."""

import numpy as np


def compute_iou(box1, box2):
    """计算两个 bounding box 的 IoU (Intersection over Union).

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        float: IoU 值 (0-1)

    IoU = 交集面积 / 并集面积

    原理:
    ┌─────────┐
    │  box1   │
    │    ┌────┼────┐
    │    │交集│    │
    └────┼────┘    │
         │   box2  │
         └─────────┘
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 并集面积 = 两个面积之和 - 交集
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_iou_batch(boxes1, boxes2):
    """批量计算 IoU 矩阵.

    Args:
        boxes1: shape (N, 4) - N 个 [x1, y1, x2, y2]
        boxes2: shape (M, 4) - M 个 [x1, y1, x2, y2]

    Returns:
        shape (N, M) 的 IoU 矩阵
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    N = boxes1.shape[0]
    M = boxes2.shape[0]

    # 扩展维度进行广播计算
    boxes1 = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1, M, 4)

    # 交集坐标
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - intersection

    return np.where(union == 0, 0, intersection / union)


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """非极大值抑制 (NMS) - YOLO 后处理的核心步骤.

    原理：
    1. 按置信度排序所有预测框
    2. 选择置信度最高的框
    3. 删除与它 IoU > 阈值的其他框
    4. 重复直到没有框剩余

    Args:
        boxes: shape (N, 4) - [x1, y1, x2, y2]
        scores: shape (N,) - 置信度分数
        iou_threshold: IoU 阈值

    Returns:
        list: 保留的框的索引
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)

    # 按分数降序排列
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        # 选择当前最高分的框
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # 计算当前框与剩余所有框的 IoU
        remaining = order[1:]
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in remaining])

        # 保留 IoU 小于阈值的框
        mask = ious < iou_threshold
        order = remaining[mask]

    return keep


def compute_ap(recall, precision):
    """计算 Average Precision (AP).

    使用 11 点插值法（VOC 2007）或所有点插值法（VOC 2012/COCO）.

    AP = PR 曲线下面积

    Args:
        recall: recall 值数组 (单调递增)
        precision: precision 值数组

    Returns:
        float: AP 值
    """
    # 在首尾添加哨兵值
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # 确保 precision 单调递减（取右端最大值）
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 找到 recall 变化的点
    indices = np.where(recall[1:] != recall[:-1])[0]

    # 计算面积
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap
