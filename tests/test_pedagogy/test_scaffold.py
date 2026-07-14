"""Tests for yolo_learn.pedagogy.scaffold."""

import yaml

from yolo_learn.pedagogy.scaffold import Exercise, ExerciseSet, HintSystem


class TestExercise:
    def test_show(self):
        ex = Exercise(id="e1", title="Test", description="Do this", difficulty="basic")
        text = ex.show()
        assert "🟢 基础" in text
        assert "Test" in text

    def test_show_advanced(self):
        ex = Exercise(id="e2", title="Adv", description="...", difficulty="advanced")
        assert "🟡 进阶" in ex.show()

    def test_show_challenge(self):
        ex = Exercise(id="e3", title="Hard", description="...", difficulty="challenge")
        assert "🔴 挑战" in ex.show()


class TestHintSystem:
    def test_next_hint(self):
        hs = HintSystem(hints=["hint1", "hint2", "hint3"])
        assert hs.remaining == 3
        h1 = hs.next_hint()
        assert "1/3" in h1
        assert hs.remaining == 2

    def test_exhausted(self):
        hs = HintSystem(hints=["only one"])
        hs.next_hint()
        assert hs.exhausted is True
        assert hs.next_hint() is None

    def test_reset(self):
        hs = HintSystem(hints=["a", "b"])
        hs.next_hint()
        hs.reset()
        assert hs.remaining == 2

    def test_empty(self):
        hs = HintSystem()
        assert hs.exhausted is True
        assert hs.next_hint() is None


class TestExerciseSet:
    def test_from_yaml(self, tmp_path):
        data = {
            "exercises": [
                {
                    "id": "e1",
                    "title": "Test",
                    "description": "Do it",
                    "difficulty": "basic",
                    "hints": ["hint1", "hint2"],
                    "solution": "Done",
                },
            ],
        }
        (tmp_path / "test_topic.yaml").write_text(yaml.dump(data))
        es = ExerciseSet.from_yaml("test_topic", base_dir=tmp_path)
        assert es.total == 1
        assert es.exercises[0].id == "e1"

    def test_from_yaml_missing(self, tmp_path):
        es = ExerciseSet.from_yaml("nonexistent", base_dir=tmp_path)
        assert es.total == 0

    def test_get_by_difficulty(self, tmp_path):
        data = {
            "exercises": [
                {"id": "e1", "title": "A", "description": "", "difficulty": "basic"},
                {"id": "e2", "title": "B", "description": "", "difficulty": "advanced"},
                {"id": "e3", "title": "C", "description": "", "difficulty": "basic"},
            ],
        }
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        es = ExerciseSet.from_yaml("t", base_dir=tmp_path)
        assert len(es.get_by_difficulty("basic")) == 2
        assert len(es.get_by_difficulty("advanced")) == 1

    def test_get_hint_system(self, tmp_path):
        data = {
            "exercises": [
                {"id": "e1", "title": "A", "description": "", "hints": ["h1", "h2"]},
            ],
        }
        (tmp_path / "t.yaml").write_text(yaml.dump(data))
        es = ExerciseSet.from_yaml("t", base_dir=tmp_path)
        hs = es.get_hint_system("e1")
        assert hs.remaining == 2

    def test_get_hint_system_missing(self, tmp_path):
        es = ExerciseSet.from_yaml("nonexistent", base_dir=tmp_path)
        hs = es.get_hint_system("x")
        assert hs.exhausted is True
