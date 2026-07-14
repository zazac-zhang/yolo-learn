"""Tests for yolo_learn.pedagogy.checkpoint."""

import yaml

from yolo_learn.pedagogy.checkpoint import Checkpoint, PredictBeforeRun, Quiz


class TestCheckpoint:
    def test_check_correct(self):
        cp = Checkpoint(id="q1", question="?", answer="A", explanation="")
        assert cp.check("A") is True

    def test_check_case_insensitive(self):
        cp = Checkpoint(id="q1", question="?", answer="correct", explanation="")
        assert cp.check("Correct") is True

    def test_check_wrong(self):
        cp = Checkpoint(id="q1", question="?", answer="A", explanation="")
        assert cp.check("B") is False

    def test_show_choice(self):
        cp = Checkpoint(
            id="q1", question="1+1=?", answer="B", explanation="",
            type="choice", options=["1", "2", "3"]
        )
        text = cp.show()
        assert "A. 1" in text
        assert "B. 2" in text

    def test_show_true_false(self):
        cp = Checkpoint(id="q1", question="太阳是恒星?", answer="A", explanation="", type="true_false")
        text = cp.show()
        assert "正确" in text
        assert "错误" in text


class TestPredictBeforeRun:
    def test_check_correct(self):
        p = PredictBeforeRun(id="p1", code_snippet="print(1+1)", expected_output="2", explanation="")
        assert p.check("2") is True

    def test_check_whitespace_tolerance(self):
        p = PredictBeforeRun(id="p1", code_snippet="print('hi')", expected_output="hi", explanation="")
        assert p.check("  hi  ") is True

    def test_check_wrong(self):
        p = PredictBeforeRun(id="p1", code_snippet="print(1+1)", expected_output="2", explanation="")
        assert p.check("3") is False


class TestQuiz:
    def test_from_yaml(self, tmp_path):
        data = {
            "checkpoints": [
                {"id": "q1", "question": "1+1=?", "answer": "2", "explanation": "数学", "type": "fill"},
            ],
            "predict_before_run": [
                {"id": "p1", "code": "print(1)", "expected_output": "1", "explanation": "print"},
            ],
        }
        (tmp_path / "test_topic.yaml").write_text(yaml.dump(data))
        quiz = Quiz.from_yaml("test_topic", base_dir=tmp_path)
        assert len(quiz.checkpoints) == 1
        assert len(quiz.predict_before_run) == 1
        assert quiz.total == 2

    def test_from_yaml_missing_file(self, tmp_path):
        quiz = Quiz.from_yaml("nonexistent", base_dir=tmp_path)
        assert quiz.total == 0

    def test_ask(self, tmp_path):
        data = {"checkpoints": [{"id": "q1", "question": "?", "answer": "A", "explanation": ""}]}
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        quiz = Quiz.from_yaml("t", base_dir=tmp_path)
        assert "❓" in quiz.ask(0)

    def test_check_correct(self, tmp_path):
        data = {"checkpoints": [{"id": "q1", "question": "?", "answer": "A", "explanation": "ok"}]}
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        quiz = Quiz.from_yaml("t", base_dir=tmp_path)
        correct, msg = quiz.check(0, "A")
        assert correct is True
        assert "✅" in msg

    def test_check_wrong(self, tmp_path):
        data = {"checkpoints": [{"id": "q1", "question": "?", "answer": "A", "explanation": "ok", "hint": "想想"}]}
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        quiz = Quiz.from_yaml("t", base_dir=tmp_path)
        correct, msg = quiz.check(0, "B")
        assert correct is False
        assert "💡" in msg

    def test_run_predict(self, tmp_path):
        data = {"predict_before_run": [{"id": "p1", "code": "print(1)", "expected_output": "1", "explanation": ""}]}
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        quiz = Quiz.from_yaml("t", base_dir=tmp_path)
        p = quiz.run_predict(0)
        assert p is not None
        assert p.expected_output == "1"

    def test_run_predict_empty(self):
        quiz = Quiz(topic="empty")
        assert quiz.run_predict(0) is None
