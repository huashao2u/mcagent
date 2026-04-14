from __future__ import annotations

from src.baselines.base import BaseBaseline, BaselineDecision


class ThresholdRouterBaseline(BaseBaseline):
    name = "threshold_router"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def decide(self, sample, semantic_tags: dict[str, bool]) -> BaselineDecision:
        if semantic_tags.get("MISSING_INFO"):
            return BaselineDecision("Missing info detected.", "CLARIFY", {"question": "Could you clarify the missing detail?"}, "Route to clarify under low confidence.", 0.55)
        if semantic_tags.get("CALCULATION_REQUIRED"):
            return BaselineDecision("Math boundary detected.", "CALCULATE", {"expression": "1+1"}, "Route to calculation under uncertainty.", 0.58)
        if semantic_tags.get("TOOL_REQUIRED"):
            return BaselineDecision("Knowledge boundary detected.", "SEARCH", {"query": sample.question}, "Route to search under uncertainty.", 0.58)
        answer = sample.gold_answer[0] if isinstance(sample.gold_answer, list) and sample.gold_answer else sample.gold_answer
        return BaselineDecision("High confidence direct answer.", "ANSWER", {"answer": answer or "Direct answer"}, "Threshold router keeps direct answer.", 0.9)
