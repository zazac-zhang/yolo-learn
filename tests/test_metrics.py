"""Tests for yolo_learn.eval.metrics — IoU, NMS, AP."""

import numpy as np
import pytest

from yolo_learn.eval.metrics import (
    compute_ap,
    compute_iou,
    compute_iou_batch,
    non_max_suppression,
)


class TestComputeIoU:
    """IoU 计算测试."""

    def test_identical_boxes(self):
        """完全重叠的框，IoU = 1.0."""
        box = [0, 0, 10, 10]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        """无重叠的框，IoU = 0.0."""
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """部分重叠."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        # 交集 = 5*5 = 25, 并集 = 100 + 100 - 25 = 175
        expected = 25 / 175
        assert compute_iou(box1, box2) == pytest.approx(expected, abs=1e-4)

    def test_contained_box(self):
        """一个框完全包含另一个."""
        outer = [0, 0, 20, 20]
        inner = [5, 5, 15, 15]
        # 交集 = 10*10 = 100, 并集 = 400 + 100 - 100 = 400
        expected = 100 / 400
        assert compute_iou(outer, inner) == pytest.approx(expected)

    def test_zero_area_box(self):
        """零面积框，IoU = 0."""
        box1 = [5, 5, 5, 5]
        box2 = [0, 0, 10, 10]
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_symmetric(self):
        """IoU 应该是对称的."""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        assert compute_iou(box1, box2) == pytest.approx(compute_iou(box2, box1))


class TestComputeIoUBatch:
    """批量 IoU 计算测试."""

    def test_single_pair(self):
        """单对框."""
        boxes1 = [[0, 0, 10, 10]]
        boxes2 = [[5, 5, 15, 15]]
        result = compute_iou_batch(boxes1, boxes2)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(compute_iou([0, 0, 10, 10], [5, 5, 15, 15]))

    def test_multiple_pairs(self):
        """多对框."""
        boxes1 = [[0, 0, 10, 10], [20, 20, 30, 30]]
        boxes2 = [[5, 5, 15, 15]]
        result = compute_iou_batch(boxes1, boxes2)
        assert result.shape == (2, 1)
        assert result[0, 0] == pytest.approx(compute_iou([0, 0, 10, 10], [5, 5, 15, 15]))
        assert result[1, 0] == pytest.approx(0.0)

    def test_shape(self):
        """输出形状正确."""
        boxes1 = [[0, 0, 10, 10]] * 3
        boxes2 = [[5, 5, 15, 15]] * 4
        result = compute_iou_batch(boxes1, boxes2)
        assert result.shape == (3, 4)


class TestNonMaxSuppression:
    """NMS 测试."""

    def test_no_overlap(self):
        """无重叠的框全部保留."""
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        scores = [0.9, 0.8]
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert sorted(keep) == [0, 1]

    def test_full_overlap(self):
        """完全重叠的框只保留最高分."""
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        scores = [0.9, 0.8]
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert keep == [0]

    def test_partial_overlap(self):
        """部分重叠，根据阈值决定保留."""
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
        scores = [0.9, 0.8, 0.7]
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        # 前两个重叠度高，应该只保留第一个；第三个独立保留
        assert sorted(keep) == [0, 2]

    def test_empty_input(self):
        """空输入返回空列表."""
        assert non_max_suppression([], []) == []

    def test_single_box(self):
        """单个框返回自身."""
        keep = non_max_suppression([[0, 0, 10, 10]], [0.9])
        assert keep == [0]


class TestComputeAP:
    """AP 计算测试."""

    def test_perfect_precision(self):
        """完美精度，AP = 1.0."""
        recall = np.array([0.0, 1.0])
        precision = np.array([1.0, 1.0])
        ap = compute_ap(recall, precision)
        assert ap == pytest.approx(1.0)

    def test_zero_precision(self):
        """零精度，AP = 0.0."""
        recall = np.array([0.0, 1.0])
        precision = np.array([0.0, 0.0])
        ap = compute_ap(recall, precision)
        assert ap == pytest.approx(0.0)

    def test_monotonic_decrease(self):
        """Precision 应该单调递减（包络线）."""
        recall = np.array([0.0, 0.5, 1.0])
        precision = np.array([1.0, 0.2, 0.8])  # 非单调
        ap = compute_ap(recall, precision)
        # 包络后 precision = [1.0, 0.8, 0.8]
        # AP = 0.5 * 0.8 + 0.5 * 0.8 = 0.8
        assert ap == pytest.approx(0.8)
