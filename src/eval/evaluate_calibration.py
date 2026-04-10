from __future__ import annotations

import argparse
import json

from src.utils.io import read_jsonl, write_json


def evaluate_calibration(rollouts: list[dict]) -> dict:
    return {
        "auroc": None,
        "ece": None,
        "brier": None,
        "note": "Phase-one placeholder. Confidence-aware calibration metrics can be added once confidence scores are logged.",
        "num_samples": len(rollouts),
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
