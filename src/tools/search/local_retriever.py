from __future__ import annotations

from typing import Any


class LocalRetriever:
    name = "local_retriever"

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        metadata = sample.metadata or {}
        evidence: list[str] = []
        for key in ("source", "fact_type", "effective_year", "graph_size"):
            value = metadata.get(key)
            if value not in (None, "", []):
                evidence.append(f"{key}: {value}")
        if not evidence:
            evidence.append(f"No indexed local evidence found for query: {query}")
        observation = {
            "query": query,
            "results": evidence[: self.top_k],
            "metadata": {"tool": self.name, "mode": "local_cache"},
        }
        return observation, False, {"helpful": any("No indexed local evidence" not in item for item in evidence)}
