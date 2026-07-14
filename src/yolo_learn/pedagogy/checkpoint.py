"""检查点引擎 — 形成性评估：预测-验证、自测题."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class Checkpoint:
    """单个检查点（选择题/判断题/填空题）."""

    id: str
    question: str
    answer: str
    explanation: str
    type: Literal["choice", "true_false", "fill"] = "choice"
    options: list[str] = field(default_factory=list)
    hint: str = ""

    def check(self, user_answer: str) -> bool:
        """验证用户答案."""
        return user_answer.strip().lower() == self.answer.strip().lower()

    def show(self) -> str:
        """返回题目展示文本."""
        lines = [f"❓ {self.question}"]
        if self.type == "choice" and self.options:
            for i, opt in enumerate(self.options):
                lines.append(f"  {chr(65 + i)}. {opt}")
        elif self.type == "true_false":
            lines.append("  A. 正确  B. 错误")
        return "\n".join(lines)


@dataclass
class PredictBeforeRun:
    """预测-验证练习：先猜输出，再跑代码."""

    id: str
    code_snippet: str
    expected_output: str
    explanation: str
    hint: str = ""

    def check(self, user_prediction: str) -> bool:
        """粗略匹配用户预测（忽略空白差异）."""

        def normalize(s: str) -> str:
            return s.strip().lower().replace(" ", "")

        return normalize(user_prediction) == normalize(self.expected_output)


@dataclass
class Quiz:
    """一组检查点."""

    topic: str
    checkpoints: list[Checkpoint] = field(default_factory=list)
    predict_before_run: list[PredictBeforeRun] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, topic: str, base_dir: Path | str | None = None) -> Quiz:
        """从 YAML 文件加载题库.

        Args:
            topic: 主题名（对应文件名，如 "01_what_is_object_detection"）
            base_dir: 题库目录，默认 configs/pedagogy/checkpoints/
        """
        if base_dir is None:
            base_dir = Path(__file__).parents[3] / "configs" / "pedagogy" / "checkpoints"
        path = Path(base_dir) / f"{topic}.yaml"
        if not path.exists():
            return cls(topic=topic)

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        checkpoints = [
            Checkpoint(
                id=c["id"],
                question=c["question"],
                answer=c["answer"],
                explanation=c.get("explanation", ""),
                type=c.get("type", "choice"),
                options=c.get("options", []),
                hint=c.get("hint", ""),
            )
            for c in data.get("checkpoints", [])
        ]

        predicts = [
            PredictBeforeRun(
                id=p["id"],
                code_snippet=p["code"],
                expected_output=p["expected_output"],
                explanation=p.get("explanation", ""),
                hint=p.get("hint", ""),
            )
            for p in data.get("predict_before_run", [])
        ]

        return cls(topic=topic, checkpoints=checkpoints, predict_before_run=predicts)

    def ask(self, index: int | None = None) -> str:
        """出题.

        Args:
            index: 指定题目索引，None 则返回第一题
        Returns:
            题目展示文本
        """
        if not self.checkpoints:
            return "（暂无检查点）"
        idx = index if index is not None else 0
        return self.checkpoints[idx].show()

    def check(self, index: int, user_answer: str) -> tuple[bool, str]:
        """验证答案.

        Returns:
            (是否正确, 解释)
        """
        cp = self.checkpoints[index]
        correct = cp.check(user_answer)
        if correct:
            return True, f"✅ 正确！{cp.explanation}"
        else:
            hint = f"\n💡 提示：{cp.hint}" if cp.hint else ""
            return False, f"❌ 错误。正确答案：{cp.answer}\n{cp.explanation}{hint}"

    def run_predict(self, index: int = 0) -> PredictBeforeRun | None:
        """获取预测-验证练习."""
        if not self.predict_before_run:
            return None
        return self.predict_before_run[index]

    @property
    def total(self) -> int:
        return len(self.checkpoints) + len(self.predict_before_run)
