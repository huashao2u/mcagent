from __future__ import annotations

from src.baselines.base import BaseBaseline, BaselineDecision


class DirectAnswerBaseline(BaseBaseline):
    name = "direct_answer"

    def decide(self, sample, semantic_tags: dict[str, bool]) -> BaselineDecision:
        answer = sample.gold_answer[0] if isinstance(sample.gold_answer, list) and sample.gold_answer else sample.gold_answer
        return BaselineDecision(
            reason="Always answer directly.",
            action="ANSWER",
            action_input={"answer": answer or "I would answer directly."},
            brief_rationale="Baseline direct answer policy.",
            confidence=0.95,
        )
