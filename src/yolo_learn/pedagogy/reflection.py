"""反思引擎 — 自评量表、常见错误库."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ReflectionPrompt:
    """反思问题."""

    id: str
    question: str
    type: str = "understanding"  # understanding | difficulty | mistake


@dataclass
class SelfAssessment:
    """自评量表."""

    skill: str
    level: int = 0  # 1-5
    evidence: str = ""

    LEVELS = {
        1: "😕 完全不懂",
        2: "🤔 知道概念，不会用",
        3: "😐 能跟着做，不独立",
        4: "😊 能独立完成",
        5: "😎 能教别人",
    }

    def show_scale(self) -> str:
        """展示评分量表."""
        lines = [f"📝 自评: {self.skill}"]
        for level, desc in self.LEVELS.items():
            marker = "→" if level == self.level else " "
            lines.append(f"  {marker} {level}. {desc}")
        return "\n".join(lines)

    def set_level(self, level: int, evidence: str = ""):
        """设置评分."""
        self.level = max(1, min(5, level))
        self.evidence = evidence


@dataclass
class CommonMistake:
    """常见错误条目."""

    id: str
    description: str
    wrong_code: str
    correct_code: str
    explanation: str
    topic: str = ""


@dataclass
class MistakeLibrary:
    """常见错误库."""

    mistakes: list[CommonMistake] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, base_dir: Path | str | None = None) -> MistakeLibrary:
        """从 YAML 文件加载错误库."""
        if base_dir is None:
            base_dir = Path(__file__).parents[3] / "configs" / "pedagogy"
        path = Path(base_dir) / "common_mistakes.yaml"
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or []

        mistakes = [
            CommonMistake(
                id=m["id"],
                description=m["description"],
                wrong_code=m.get("wrong_code", ""),
                correct_code=m.get("correct_code", ""),
                explanation=m.get("explanation", ""),
                topic=m.get("topic", ""),
            )
            for m in data
        ]
        return cls(mistakes=mistakes)

    def get_by_topic(self, topic: str) -> list[CommonMistake]:
        """按主题筛选错误."""
        return [m for m in self.mistakes if topic.lower() in m.topic.lower()]

    def show_for_topic(self, topic: str) -> str:
        """展示某主题的常见错误."""
        mistakes = self.get_by_topic(topic)
        if not mistakes:
            return f"（暂无 {topic} 的常见错误记录）"
        lines = [f"❌ {topic} 常见错误:"]
        for m in mistakes:
            lines.append(f"\n  {m.description}")
            if m.wrong_code:
                lines.append(f"  ❌ {m.wrong_code}")
            if m.correct_code:
                lines.append(f"  ✅ {m.correct_code}")
            if m.explanation:
                lines.append(f"  💡 {m.explanation}")
        return "\n".join(lines)


@dataclass
class ReflectionSet:
    """一组反思问题 + 自评量表."""

    topic: str
    prompts: list[ReflectionPrompt] = field(default_factory=list)
    assessments: list[SelfAssessment] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, topic: str, base_dir: Path | str | None = None) -> ReflectionSet:
        """从 YAML 文件加载反思内容."""
        if base_dir is None:
            base_dir = Path(__file__).parents[3] / "configs" / "pedagogy" / "checkpoints"
        path = Path(base_dir) / f"{topic}.yaml"
        if not path.exists():
            return cls(topic=topic)

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        prompts = [
            ReflectionPrompt(
                id=p["id"],
                question=p["question"],
                type=p.get("type", "understanding"),
            )
            for p in data.get("reflection", [])
        ]

        assessments = [
            SelfAssessment(skill=a["skill"])
            for a in data.get("self_assessment", [])
        ]

        return cls(topic=topic, prompts=prompts, assessments=assessments)

    def show_reflections(self) -> str:
        """展示反思问题."""
        if not self.prompts:
            return "（暂无反思问题）"
        lines = ["🤔 反思问题:"]
        for p in self.prompts:
            lines.append(f"  - {p.question}")
        return "\n".join(lines)

    def show_assessments(self) -> str:
        """展示自评量表."""
        if not self.assessments:
            return "（暂无自评量表）"
        return "\n\n".join(a.show_scale() for a in self.assessments)
