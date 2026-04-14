from __future__ import annotations

from src.baselines.base import BaseBaseline, BaselineDecision


class SearchHeuristicBaseline(BaseBaseline):
    name = "search_heuristic"

    def decide(self, sample, semantic_tags: dict[str, bool]) -> BaselineDecision:
        if sample.task_type == "factual_boundary" and (semantic_tags.get("FRESH_FACT") or semantic_tags.get("NEW_OR_TAIL_KNOWLEDGE") or semantic_tags.get("TOOL_REQUIRED")):
            return BaselineDecision("Knowledge boundary detected.", "SEARCH", {"query": sample.question}, "Prefer search on fresh or tail factual questions.", 0.78)
        answer = sample.gold_answer[0] if isinstance(sample.gold_answer, list) and sample.gold_answer else sample.gold_answer
        return BaselineDecision("Fallback answer.", "ANSWER", {"answer": answer or "Direct answer"}, "Answer when search is not clearly needed.", 0.74)
