from __future__ import annotations

import re
from typing import Any


FRESH_FACT_WORDS = ("latest", "current", "today", "recent", "now", "this year", "updated", "as of")
MISCONCEPTION_WORDS = ("always", "never", "prove that", "is it true", "does it exist")
MATH_PATTERN = re.compile(r"[\d\)\(]+\s*[\+\-\*\/=]|\bsolve\b|\bcompute\b|\bcalculate\b|\bfind the value\b", re.IGNORECASE)


def build_semantic_tags(sample) -> dict[str, bool]:
    question = sample.question.lower()
    metadata = sample.metadata or {}
    false_premise = bool(metadata.get("false_premise")) or "false premise" in question
    missing_info = bool(metadata.get("vague")) or bool(metadata.get("missing_details"))
    fresh_fact = any(word in question for word in FRESH_FACT_WORDS) or bool(metadata.get("effective_year"))
    new_or_tail_knowledge = fresh_fact or bool(metadata.get("source")) or bool(metadata.get("graph_preview"))
    misconception_risk = any(word in question for word in MISCONCEPTION_WORDS) or false_premise
    calculation_required = sample.task_type == "math" and (
        bool(MATH_PATTERN.search(sample.question)) or sample.dataset in {"gsm8k", "competition_math"}
    )
    tool_required = sample.task_type == "factual_boundary" and (fresh_fact or new_or_tail_knowledge)
    justified_refuse = false_premise or ("not possible" in question) or ("does not exist" in question)
    return {
        "FRESH_FACT": fresh_fact,
        "FALSE_PREMISE": false_premise,
        "MISCONCEPTION_RISK": misconception_risk,
        "NEW_OR_TAIL_KNOWLEDGE": new_or_tail_knowledge,
        "MISSING_INFO": missing_info,
        "TOOL_REQUIRED": tool_required,
        "JUSTIFIED_REFUSE": justified_refuse,
        "CALCULATION_REQUIRED": calculation_required,
    }


def active_semantic_tags(sample) -> list[str]:
    tags = build_semantic_tags(sample)
    return [name for name, enabled in tags.items() if enabled]
