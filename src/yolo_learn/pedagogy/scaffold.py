"""脚手架引擎 — 分层练习、提示系统."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class Exercise:
    """单个练习题."""

    id: str
    title: str
    description: str
    difficulty: Literal["basic", "advanced", "challenge"] = "basic"
    hints: list[str] = field(default_factory=list)
    solution: str = ""
    solution_code: str = ""
    validation_hint: str = ""

    def show(self) -> str:
        """展示练习."""
        level_map = {"basic": "🟢 基础", "advanced": "🟡 进阶", "challenge": "🔴 挑战"}
        lines = [
            f"✍️ {self.title}",
            f"   难度: {level_map.get(self.difficulty, self.difficulty)}",
            f"   {self.description}",
        ]
        if self.validation_hint:
            lines.append(f"   验证: {self.validation_hint}")
        return "\n".join(lines)


@dataclass
class HintSystem:
    """分层提示系统."""

    hints: list[str] = field(default_factory=list)
    _current_level: int = 0

    def next_hint(self) -> str | None:
        """获取下一层提示."""
        if self._current_level >= len(self.hints):
            return None
        hint = self.hints[self._current_level]
        self._current_level += 1
        return f"💡 提示 {self._current_level}/{len(self.hints)}: {hint}"

    def reset(self):
        """重置提示层级."""
        self._current_level = 0

    @property
    def remaining(self) -> int:
        return len(self.hints) - self._current_level

    @property
    def exhausted(self) -> bool:
        return self._current_level >= len(self.hints)


@dataclass
class ExerciseSet:
    """一组练习题."""

    topic: str
    exercises: list[Exercise] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, topic: str, base_dir: Path | str | None = None) -> ExerciseSet:
        """从 YAML 文件加载练习库."""
        if base_dir is None:
            base_dir = Path(__file__).parents[3] / "configs" / "pedagogy" / "exercises"
        path = Path(base_dir) / f"{topic}.yaml"
        if not path.exists():
            return cls(topic=topic)

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        exercises = [
            Exercise(
                id=e["id"],
                title=e["title"],
                description=e["description"],
                difficulty=e.get("difficulty", "basic"),
                hints=e.get("hints", []),
                solution=e.get("solution", ""),
                solution_code=e.get("solution_code", ""),
                validation_hint=e.get("validation_hint", ""),
            )
            for e in data.get("exercises", [])
        ]

        return cls(topic=topic, exercises=exercises)

    def get_by_difficulty(self, difficulty: str) -> list[Exercise]:
        """按难度筛选练习."""
        return [e for e in self.exercises if e.difficulty == difficulty]

    def get_hint_system(self, exercise_id: str) -> HintSystem:
        """获取某个练习的提示系统."""
        for ex in self.exercises:
            if ex.id == exercise_id:
                return HintSystem(hints=ex.hints)
        return HintSystem()

    @property
    def total(self) -> int:
        return len(self.exercises)
