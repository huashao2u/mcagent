from __future__ import annotations

import re
from typing import Any


TIME_WORDS = ("latest", "current", "today", "recent", "now", "next month", "this year")


def build_semantic_tags(sample) -> dict[str, bool]:
    question = sample.question.lower()
    metadata = sample.metadata or {}
    false_premise = bool(metadata.get("false_premise")) or "false premise" in question
    missing_info = bool(metadata.get("vague")) or bool(metadata.get("missing_details"))
    time_sensitive = any(word in question for word in TIME_WORDS) or bool(metadata.get("effective_year"))
    tool_required = sample.task_type == "factual_boundary" and (time_sensitive or bool(metadata.get("source")))
    justified_refuse = false_premise or ("not possible" in question) or ("does not exist" in question)
    return {
        "TIME_SENSITIVE": time_sensitive,
        "FALSE_PREMISE": false_premise,
        "MISSING_INFO": missing_info,
        "TOOL_REQUIRED": tool_required,
        "JUSTIFIED_REFUSE": justified_refuse,
    }


def active_semantic_tags(sample) -> list[str]:
    tags = build_semantic_tags(sample)
    return [name for name, enabled in tags.items() if enabled]
