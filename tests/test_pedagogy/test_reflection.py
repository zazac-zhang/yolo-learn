"""Tests for yolo_learn.pedagogy.reflection."""

import yaml

from yolo_learn.pedagogy.reflection import (
    CommonMistake,
    MistakeLibrary,
    ReflectionSet,
    SelfAssessment,
)


class TestSelfAssessment:
    def test_show_scale(self):
        sa = SelfAssessment(skill="IoU 计算")
        text = sa.show_scale()
        assert "IoU 计算" in text
        assert "1." in text
        assert "5." in text

    def test_set_level(self):
        sa = SelfAssessment(skill="test")
        sa.set_level(4, "能独立完成")
        assert sa.level == 4
        assert sa.evidence == "能独立完成"

    def test_set_level_clamp(self):
        sa = SelfAssessment(skill="test")
        sa.set_level(10)
        assert sa.level == 5
        sa.set_level(0)
        assert sa.level == 1

    def test_show_with_level(self):
        sa = SelfAssessment(skill="test", level=3)
        text = sa.show_scale()
        assert "→" in text


class TestCommonMistake:
    def test_create(self):
        m = CommonMistake(
            id="m1", description="wrong box format",
            wrong_code="xywh", correct_code="xyxy", explanation="must be xyxy"
        )
        assert m.topic == ""


class TestMistakeLibrary:
    def test_from_yaml(self, tmp_path):
        data = [
            {"id": "m1", "description": "err", "topic": "iou", "explanation": "fix"},
        ]
        (tmp_path / "common_mistakes.yaml").write_text(yaml.dump(data))
        lib = MistakeLibrary.from_yaml(base_dir=tmp_path)
        assert len(lib.mistakes) == 1

    def test_from_yaml_missing(self, tmp_path):
        lib = MistakeLibrary.from_yaml(base_dir=tmp_path)
        assert len(lib.mistakes) == 0

    def test_get_by_topic(self, tmp_path):
        data = [
            {"id": "m1", "description": "err1", "topic": "iou"},
            {"id": "m2", "description": "err2", "topic": "nms"},
            {"id": "m3", "description": "err3", "topic": "iou"},
        ]
        (tmp_path / "common_mistakes.yaml").write_text(yaml.dump(data))
        lib = MistakeLibrary.from_yaml(base_dir=tmp_path)
        assert len(lib.get_by_topic("iou")) == 2
        assert len(lib.get_by_topic("nms")) == 1

    def test_show_for_topic(self, tmp_path):
        data = [
            {"id": "m1", "description": "err", "topic": "iou", "wrong_code": "bad", "correct_code": "good"},
        ]
        (tmp_path / "common_mistakes.yaml").write_text(yaml.dump(data))
        lib = MistakeLibrary.from_yaml(base_dir=tmp_path)
        text = lib.show_for_topic("iou")
        assert "err" in text
        assert "bad" in text
        assert "good" in text

    def test_show_for_topic_empty(self):
        lib = MistakeLibrary()
        assert "暂无" in lib.show_for_topic("anything")


class TestReflectionSet:
    def test_from_yaml(self, tmp_path):
        data = {
            "reflection": [
                {"id": "r1", "question": "What?", "type": "understanding"},
            ],
            "self_assessment": [
                {"skill": "IoU"},
            ],
        }
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        rs = ReflectionSet.from_yaml("t", base_dir=tmp_path)
        assert len(rs.prompts) == 1
        assert len(rs.assessments) == 1

    def test_from_yaml_missing(self, tmp_path):
        rs = ReflectionSet.from_yaml("nonexistent", base_dir=tmp_path)
        assert len(rs.prompts) == 0

    def test_show_reflections(self, tmp_path):
        data = {"reflection": [{"id": "r1", "question": "Why?"}]}
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        rs = ReflectionSet.from_yaml("t", base_dir=tmp_path)
        assert "Why?" in rs.show_reflections()

    def test_show_reflections_empty(self):
        rs = ReflectionSet(topic="empty")
        assert "暂无" in rs.show_reflections()

    def test_show_assessments(self, tmp_path):
        data = {"self_assessment": [{"skill": "NMS"}]}
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        rs = ReflectionSet.from_yaml("t", base_dir=tmp_path)
        assert "NMS" in rs.show_assessments()
