from __future__ import annotations

import argparse
import json

from src.eval.calibration_utils import (
    binary_auroc,
    brier_score,
    expected_calibration_error,
    extract_binary_action_targets,
    extract_binary_answer_targets,
    summarize_risk_coverage,
)
from src.utils.io import read_jsonl, write_json


def _summarize_block(records: list[dict], extractor) -> dict:
    labels, scores = extractor(records)
    return {
        "overall_auroc": binary_auroc(labels, scores),
        "overall_ece": expected_calibration_error(labels, scores),
        "overall_brier": brier_score(labels, scores),
        "num_samples": len(labels),
        **summarize_risk_coverage(labels, scores),
    }


def evaluate_calibration(rollouts: list[dict]) -> dict:
    per_dataset: dict[str, dict] = {}
    per_action: dict[str, dict] = {}
    for key in sorted({record.get("dataset") for record in rollouts if record.get("dataset")}):
        subset = [record for record in rollouts if record.get("dataset") == key]
        block = _summarize_block(subset, extract_binary_action_targets)
        per_dataset[key] = {
            "auroc": block["overall_auroc"],
            "ece": block["overall_ece"],
            "brier": block["overall_brier"],
            "best_coverage_at_90_precision": block["best_coverage_at_90_precision"],
            "utility_coverage": block["utility_coverage"],
            "num_samples": block["num_samples"],
        }
    for key in sorted({(record.get("decision") or {}).get("action") for record in rollouts if record.get("decision")}):
        subset = [record for record in rollouts if (record.get("decision") or {}).get("action") == key]
        block = _summarize_block(subset, extract_binary_action_targets)
        per_action[key] = {
            "auroc": block["overall_auroc"],
            "ece": block["overall_ece"],
            "brier": block["overall_brier"],
            "best_coverage_at_90_precision": block["best_coverage_at_90_precision"],
            "utility_coverage": block["utility_coverage"],
            "num_samples": block["num_samples"],
        }
    decision_block = _summarize_block(rollouts, extract_binary_action_targets)
    answer_block = _summarize_block(rollouts, extract_binary_answer_targets)
    answer_per_dataset = {
        key: _summarize_block([record for record in rollouts if record.get("dataset") == key], extract_binary_answer_targets)
        for key in sorted({record.get("dataset") for record in rollouts if record.get("dataset")})
    }
    return {
        "headline_metric": "decision_auroc",
        "overall_auroc": decision_block["overall_auroc"],
        "overall_ece": decision_block["overall_ece"],
        "overall_brier": decision_block["overall_brier"],
        "num_samples": decision_block["num_samples"],
        "per_dataset": per_dataset,
        "per_action": per_action,
        "decision_calibration": {
            **decision_block,
            "per_dataset": per_dataset,
            "per_action": per_action,
        },
        "answer_calibration": {
            **answer_block,
            "per_dataset": answer_per_dataset,
        },
        "best_coverage_at_90_precision": decision_block["best_coverage_at_90_precision"],
        "utility_coverage": decision_block["utility_coverage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate placeholder calibration metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = evaluate_calibration(read_jsonl(args.input))
    write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
