from __future__ import annotations

from typing import Any

from src.scoring.action_oracle import choose_oracle_action


def load_utility_config(config: dict[str, Any]) -> dict[str, float]:
    scoring = config["scoring"]
    return {
        "lambda_tool": float(scoring["lambda_tool"]),
        "lambda_clar": float(scoring["lambda_clar"]),
        "answer_correct": float(scoring["answer_correct"]),
        "answer_wrong": float(scoring["answer_wrong"]),
        "refuse_justified": float(scoring["refuse_justified"]),
        "refuse_unjustified": float(scoring["refuse_unjustified"]),
        "search_helpful": float(scoring["search_helpful"]),
        "search_unhelpful": float(scoring["search_unhelpful"]),
        "clarify_helpful": float(scoring["clarify_helpful"]),
        "clarify_unhelpful": float(scoring["clarify_unhelpful"]),
    }


def score_action(
    action: str,
    sample,
    semantic_tags: dict[str, bool],
    utility_config: dict[str, float],
    correctness: bool | None = None,
) -> float:
    normalized = action.upper()
    oracle_action = choose_oracle_action(sample, semantic_tags=semantic_tags)
    if normalized == "ANSWER":
        if correctness is None:
            correctness = oracle_action == "ANSWER"
        return utility_config["answer_correct"] if correctness else utility_config["answer_wrong"]
    if normalized in {"SEARCH", "CALCULATE"}:
        helpful = oracle_action in {"SEARCH", "CALCULATE"} or semantic_tags["TOOL_REQUIRED"]
        base = utility_config["search_helpful"] if helpful else utility_config["search_unhelpful"]
        return base - utility_config["lambda_tool"]
    if normalized == "CLARIFY":
        helpful = oracle_action == "CLARIFY" or semantic_tags["MISSING_INFO"]
        base = utility_config["clarify_helpful"] if helpful else utility_config["clarify_unhelpful"]
        return base - utility_config["lambda_clar"]
    justified = oracle_action == "REFUSE" or semantic_tags["JUSTIFIED_REFUSE"]
    return utility_config["refuse_justified"] if justified else utility_config["refuse_unjustified"]


def estimate_candidate_utilities(sample, semantic_tags: dict[str, bool], utility_config: dict[str, float], answer_correctness: bool | None) -> dict[str, float]:
    return {
        "ANSWER": score_action("ANSWER", sample, semantic_tags, utility_config, correctness=answer_correctness),
        "SEARCH": score_action("SEARCH", sample, semantic_tags, utility_config),
        "CALCULATE": score_action("CALCULATE", sample, semantic_tags, utility_config),
        "CLARIFY": score_action("CLARIFY", sample, semantic_tags, utility_config),
        "REFUSE": score_action("REFUSE", sample, semantic_tags, utility_config),
    }
