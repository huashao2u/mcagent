from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.evaluate_actions import evaluate_actions
from src.eval.evaluate_answers import evaluate_answers
from src.eval.evaluate_calibration import evaluate_calibration
from src.utils.config import load_config
from src.utils.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MCAgent rollout metrics.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    input_path = Path(args.input or config["paths"]["rollout_output"])
    rollouts = read_jsonl(input_path)
    aggregate = {
        "actions": evaluate_actions(rollouts),
        "answers": evaluate_answers(rollouts),
        "calibration": evaluate_calibration(rollouts),
    }
    output_path = Path(config["paths"]["aggregate_output"])
    write_json(output_path, aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
