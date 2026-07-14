"""Tests for yolo_learn.eval.evaluate — Annotation, Prediction, EvaluationResult."""

import pytest

from yolo_learn.eval.evaluate import (
    Annotation,
    EvaluationResult,
    Prediction,
    evaluate_predictions,
    load_class_names,
)


class TestAnnotation:
    """Annotation dataclass 测试."""

    def test_create(self):
        ann = Annotation(class_id=0, box=[0, 0, 10, 10])
        assert ann.class_id == 0
        assert ann.box == [0, 0, 10, 10]


class TestPrediction:
    """Prediction dataclass 测试."""

    def test_create(self):
        pred = Prediction(class_id=1, box=[5, 5, 15, 15], confidence=0.95)
        assert pred.class_id == 1
        assert pred.confidence == 0.95


class TestEvaluatePredictions:
    """评估流程测试."""

    def test_perfect_match(self):
        """预测和标注完全匹配，mAP = 1.0."""
        preds = [{"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9}]
        anns = [{"class_id": 0, "box": [0, 0, 10, 10]}]
        result = evaluate_predictions(preds, anns, iou_threshold=0.5, class_names={0: "cat"})
        assert result.mAP == pytest.approx(1.0)
        assert result.per_class_ap[0] == pytest.approx(1.0)
        assert result.num_predictions == 1
        assert result.num_annotations == 1

    def test_no_match(self):
        """预测和标注不匹配，mAP = 0.0."""
        preds = [{"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9}]
        anns = [{"class_id": 0, "box": [100, 100, 110, 110]}]
        result = evaluate_predictions(preds, anns, iou_threshold=0.5)
        assert result.mAP == pytest.approx(0.0)

    def test_multiple_classes(self):
        """多类别评估."""
        preds = [
            {"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9},
            {"class_id": 1, "box": [50, 50, 60, 60], "confidence": 0.8},
        ]
        anns = [
            {"class_id": 0, "box": [0, 0, 10, 10]},
            {"class_id": 1, "box": [50, 50, 60, 60]},
        ]
        result = evaluate_predictions(
            preds, anns, iou_threshold=0.5, class_names={0: "cat", 1: "dog"}
        )
        assert result.mAP == pytest.approx(1.0)
        assert result.per_class_ap[0] == pytest.approx(1.0)
        assert result.per_class_ap[1] == pytest.approx(1.0)

    def test_empty_predictions(self):
        """无预测时 mAP = 0."""
        preds = []
        anns = [{"class_id": 0, "box": [0, 0, 10, 10]}]
        result = evaluate_predictions(preds, anns)
        assert result.mAP == pytest.approx(0.0)

    def test_empty_annotations(self):
        """无标注时 mAP = 0."""
        preds = [{"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9}]
        anns = []
        result = evaluate_predictions(preds, anns)
        assert result.mAP == pytest.approx(0.0)

    def test_summary_output(self):
        """summary() 返回可读字符串."""
        result = EvaluationResult(mAP=0.85, per_class_ap={0: 0.9, 1: 0.8}, class_names={0: "cat", 1: "dog"})
        summary = result.summary()
        assert "mAP@0.5: 0.8500" in summary
        assert "cat" in summary
        assert "dog" in summary


class TestLoadClassNames:
    """类别名称加载测试."""

    def test_load_from_yaml(self, tmp_path):
        """从 YAML 文件加载类别名."""
        yaml_content = "nc: 2\nnames:\n  0: cat\n  1: dog\n"
        yaml_file = tmp_path / "data.yaml"
        yaml_file.write_text(yaml_content)
        names = load_class_names(yaml_file)
        assert names == {0: "cat", 1: "dog"}

    def test_load_list_format(self, tmp_path):
        """列表格式的类别名."""
        yaml_content = "nc: 2\nnames: [cat, dog]\n"
        yaml_file = tmp_path / "data.yaml"
        yaml_file.write_text(yaml_content)
        names = load_class_names(yaml_file)
        assert names == {0: "cat", 1: "dog"}
