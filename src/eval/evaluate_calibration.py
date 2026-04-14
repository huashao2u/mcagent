from __future__ import annotations

import argparse
import json

from src.eval.calibration_utils import binary_auroc, brier_score, expected_calibration_error, extract_binary_action_targets, summarize_risk_coverage
from src.utils.io import read_jsonl, write_json


def evaluate_calibration(rollouts: list[dict]) -> dict:
    labels, scores = extract_binary_action_targets(rollouts)
    per_dataset: dict[str, dict] = {}
    per_action: dict[str, dict] = {}
    for key in sorted({record.get("dataset") for record in rollouts if record.get("dataset")}):
        subset = [record for record in rollouts if record.get("dataset") == key]
        ds_labels, ds_scores = extract_binary_action_targets(subset)
        per_dataset[key] = {
            "auroc": binary_auroc(ds_labels, ds_scores),
            "ece": expected_calibration_error(ds_labels, ds_scores),
            "brier": brier_score(ds_labels, ds_scores),
            **summarize_risk_coverage(ds_labels, ds_scores),
            "num_samples": len(ds_labels),
        }
    for key in sorted({(record.get("decision") or {}).get("action") for record in rollouts if record.get("decision")}):
        subset = [record for record in rollouts if (record.get("decision") or {}).get("action") == key]
        action_labels, action_scores = extract_binary_action_targets(subset)
        per_action[key] = {
            "auroc": binary_auroc(action_labels, action_scores),
            "ece": expected_calibration_error(action_labels, action_scores),
            "brier": brier_score(action_labels, action_scores),
            **summarize_risk_coverage(action_labels, action_scores),
            "num_samples": len(action_labels),
        }
    return {
        "overall_auroc": binary_auroc(labels, scores),
        "overall_ece": expected_calibration_error(labels, scores),
        "overall_brier": brier_score(labels, scores),
        "num_samples": len(labels),
        "per_dataset": per_dataset,
        "per_action": per_action,
        **summarize_risk_coverage(labels, scores),
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
