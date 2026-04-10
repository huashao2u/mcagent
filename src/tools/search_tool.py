from __future__ import annotations

from typing import Any


class SearchTool:
    name = "SEARCH"

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        evidence: list[str] = []
        metadata = sample.metadata or {}
        source = metadata.get("source")
        if isinstance(source, str) and source.strip():
            evidence.extend([chunk.strip() for chunk in source.splitlines() if chunk.strip()])
        gold = sample.gold_answer
        if isinstance(gold, list):
            evidence.extend(gold[:2])
        elif isinstance(gold, str) and gold.strip():
            evidence.append(gold.strip())
        if not evidence:
            evidence.append(f"No external corpus was available for: {query}")
        observation = {"query": query, "results": evidence[:3], "metadata": {"tool": "mock_search"}}
        return observation, False, {"helpful": len(evidence) > 0}
