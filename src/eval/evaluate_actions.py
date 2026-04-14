from __future__ import annotations

import argparse
import json
from typing import Any

from src.utils.io import read_jsonl, write_json


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate_actions(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rollouts)
    action_counts: dict[str, int] = {}
    oracle_counts: dict[str, int] = {}
    per_action_correct: dict[str, int] = {}
    action_correct = 0
    search_total = 0
    unnecessary_searches = 0
    calculate_total = 0
    helpful_calculates = 0
    refuse_total = 0
    justified_refuses = 0
    clarify_total = 0
    helpful_clarifies = 0
    answer_total = 0
    over_answers = 0
    for record in rollouts:
        action = record["decision"]["action"]
        oracle = record["oracle_action"]
        action_counts[action] = action_counts.get(action, 0) + 1
        oracle_counts[oracle] = oracle_counts.get(oracle, 0) + 1
        action_correct += int(action == oracle)
        per_action_correct[action] = per_action_correct.get(action, 0) + int(action == oracle)
        if action == "SEARCH":
            search_total += 1
            unnecessary_searches += int(oracle != "SEARCH")
        if action == "CALCULATE":
            calculate_total += 1
            helpful_calculates += int(oracle == "CALCULATE")
        if action == "REFUSE":
            refuse_total += 1
            justified_refuses += int(oracle == "REFUSE")
        if action == "CLARIFY":
            clarify_total += 1
            helpful_clarifies += int(oracle == "CLARIFY")
        if action == "ANSWER":
            answer_total += 1
            over_answers += int(oracle != "ANSWER")
    non_answer_rate = 1 - (action_counts.get("ANSWER", 0) / total) if total else None
    per_action_accuracy = {
        action: _safe_rate(per_action_correct.get(action, 0), count) for action, count in sorted(action_counts.items())
    }
    return {
        "action_accuracy": _safe_rate(action_correct, total),
        "unnecessary_search_rate": _safe_rate(unnecessary_searches, search_total),
        "calculate_helpfulness": _safe_rate(helpful_calculates, calculate_total),
        "unnecessary_calculate_rate": _safe_rate(calculate_total - helpful_calculates, calculate_total),
        "justified_refuse_rate": _safe_rate(justified_refuses, refuse_total),
        "over_refusal_rate": _safe_rate(refuse_total - justified_refuses, refuse_total),
        "clarify_helpfulness": _safe_rate(helpful_clarifies, clarify_total),
        "over_answer_rate": _safe_rate(over_answers, answer_total),
        "non_answer_rate": non_answer_rate,
        "action_counts": action_counts,
        "oracle_action_counts": oracle_counts,
        "per_action_accuracy": per_action_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate action-level metrics from rollout logs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = evaluate_actions(read_jsonl(args.input))
    write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
