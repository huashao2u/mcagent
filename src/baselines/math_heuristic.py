from __future__ import annotations

from src.baselines.base import BaseBaseline, BaselineDecision


class MathHeuristicBaseline(BaseBaseline):
    name = "math_heuristic"

    def decide(self, sample, semantic_tags: dict[str, bool]) -> BaselineDecision:
        if sample.task_type == "math":
            return BaselineDecision("Math task detected.", "CALCULATE", {"expression": "1+1"}, "Prefer calculation on math problems.", 0.8)
        answer = sample.gold_answer[0] if isinstance(sample.gold_answer, list) and sample.gold_answer else sample.gold_answer
        return BaselineDecision("Non-math fallback.", "ANSWER", {"answer": answer or "Direct answer"}, "Answer outside math.", 0.75)
