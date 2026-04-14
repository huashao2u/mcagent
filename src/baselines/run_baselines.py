from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baselines.clarify_heuristic import ClarifyHeuristicBaseline
from src.baselines.direct_answer import DirectAnswerBaseline
from src.baselines.math_heuristic import MathHeuristicBaseline
from src.baselines.search_heuristic import SearchHeuristicBaseline
from src.baselines.threshold_router import ThresholdRouterBaseline
from src.data.loaders import load_mixed_datasets
from src.eval.evaluate_actions import evaluate_actions
from src.eval.evaluate_answers import evaluate_answers, is_answer_correct
from src.eval.evaluate_calibration import evaluate_calibration
from src.features.semantic_tags import active_semantic_tags, build_semantic_tags
from src.scoring.action_oracle import choose_oracle_action
from src.utils.config import load_config
from src.utils.io import write_json, write_jsonl


BASELINES = {
    "direct_answer": DirectAnswerBaseline,
    "threshold_router": ThresholdRouterBaseline,
    "math_heuristic": MathHeuristicBaseline,
    "search_heuristic": SearchHeuristicBaseline,
    "clarify_heuristic": ClarifyHeuristicBaseline,
}


def run_baseline(name: str, config: dict) -> dict:
    baseline = BASELINES[name]()
    eval_mix = config["data"]["eval_mix"]
    samples = load_mixed_datasets(
        dataset_limits=eval_mix["per_dataset"],
        split_map=config["data"]["default_split"],
        dataset_root=Path(config["paths"]["dataset_root"]).resolve(),
        seed=int(eval_mix["seed"]),
    )[: int(eval_mix["total_limit"])]
    records = []
    for sample in samples:
        tags = build_semantic_tags(sample)
        decision = baseline.decide(sample, tags)
        final_answer = decision.action_input.get("answer")
        correctness = is_answer_correct(sample.to_dict(), final_answer)
        records.append(
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "task_type": sample.task_type,
                "question": sample.question,
                "metadata": sample.metadata,
                "reason": decision.reason,
                "decision": {
                    "action": decision.action,
                    "action_input": decision.action_input,
                    "brief_rationale": decision.brief_rationale,
                },
                "oracle_action": choose_oracle_action(sample, semantic_tags=tags),
                "semantic_tags": active_semantic_tags(sample),
                "state_tags": active_semantic_tags(sample),
                "final_answer": final_answer,
                "correctness": correctness,
                "action_confidence": decision.confidence,
                "verbal_confidence": decision.confidence,
            }
        )
    return {
        "records": records,
        "action_metrics": evaluate_actions(records),
        "answer_metrics": evaluate_answers(records),
        "calibration_metrics": evaluate_calibration(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCAgent baselines.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--baseline", choices=sorted(BASELINES), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    payload = run_baseline(args.baseline, config)
    output_path = Path(args.output)
    write_jsonl(output_path.with_suffix(".jsonl"), payload["records"])
    write_json(output_path, {key: value for key, value in payload.items() if key != "records"})
    print(json.dumps({"baseline": args.baseline, "output": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
