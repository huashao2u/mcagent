from __future__ import annotations

from src.features.semantic_tags import build_semantic_tags


def choose_oracle_action(
    sample,
    semantic_tags: dict[str, bool] | None = None,
    clarify_allowed: bool = True,
    retrieval_allowed: bool = True,
) -> str:
    tags = build_semantic_tags(sample) if semantic_tags is None else semantic_tags
    if tags["MISSING_INFO"] and clarify_allowed:
        return "CLARIFY"
    if tags["TOOL_REQUIRED"] and retrieval_allowed:
        return "SEARCH"
    if tags["FALSE_PREMISE"] or tags["JUSTIFIED_REFUSE"]:
        return "REFUSE"
    return "ANSWER"
