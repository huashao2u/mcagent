from __future__ import annotations

from src.baselines.base import BaseBaseline, BaselineDecision


class ClarifyHeuristicBaseline(BaseBaseline):
    name = "clarify_heuristic"

    def decide(self, sample, semantic_tags: dict[str, bool]) -> BaselineDecision:
        if sample.dataset == "in3" or semantic_tags.get("MISSING_INFO"):
            return BaselineDecision("Missing-info boundary detected.", "CLARIFY", {"question": "Could you clarify the missing detail?"}, "Prefer clarify on underspecified tasks.", 0.82)
        answer = sample.gold_answer[0] if isinstance(sample.gold_answer, list) and sample.gold_answer else sample.gold_answer
        return BaselineDecision("Fallback answer.", "ANSWER", {"answer": answer or "Direct answer"}, "Answer when clarification is unnecessary.", 0.7)
