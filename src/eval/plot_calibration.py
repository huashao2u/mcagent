from __future__ import annotations

import argparse
import json

from src.utils.io import read_jsonl, write_json


def build_plot_payload(metrics: dict) -> dict:
    return {
        "title": "Calibration Summary",
        "overall_auroc": metrics.get("overall_auroc"),
        "overall_ece": metrics.get("overall_ece"),
        "overall_brier": metrics.get("overall_brier"),
        "per_dataset": metrics.get("per_dataset", {}),
        "per_action": metrics.get("per_action", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight calibration plot payload.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.metrics, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    payload = build_plot_payload(metrics)
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
