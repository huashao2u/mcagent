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
    action_correct = 0
    search_total = 0
    unnecessary_searches = 0
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
        action_correct += int(action == oracle)
        if action == "SEARCH":
            search_total += 1
            unnecessary_searches += int(oracle != "SEARCH")
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
    return {
        "action_accuracy": _safe_rate(action_correct, total),
        "unnecessary_search_rate": _safe_rate(unnecessary_searches, search_total),
        "justified_refuse_rate": _safe_rate(justified_refuses, refuse_total),
        "over_refusal_rate": _safe_rate(refuse_total - justified_refuses, refuse_total),
        "clarify_helpfulness": _safe_rate(helpful_clarifies, clarify_total),
        "over_answer_rate": _safe_rate(over_answers, answer_total),
        "non_answer_rate": non_answer_rate,
        "action_counts": action_counts,
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
