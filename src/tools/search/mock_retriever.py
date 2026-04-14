from __future__ import annotations

from typing import Any


class MockRetriever:
    name = "mock_retriever"

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        metadata = sample.metadata or {}
        evidence: list[str] = []
        source = metadata.get("source")
        if isinstance(source, str) and source.strip():
            evidence.extend(chunk.strip() for chunk in source.splitlines() if chunk.strip())
        graph_preview = metadata.get("graph_preview") or []
        if isinstance(graph_preview, list):
            evidence.extend(str(item) for item in graph_preview[: self.top_k])
        if not evidence:
            evidence.append(f"Cached retrieval placeholder for query: {query}")
        observation = {
            "query": query,
            "results": evidence[: self.top_k],
            "metadata": {"tool": self.name, "mode": "train_proxy"},
        }
        return observation, False, {"helpful": bool(evidence)}
