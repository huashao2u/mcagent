from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.loaders import load_mixed_datasets
from src.features.semantic_tags import build_semantic_tags
from src.prompting.build_prompts import build_prompt_text
from src.scoring.action_oracle import choose_oracle_action
from src.scoring.local_utility import estimate_candidate_utilities, load_utility_config
from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl


def _build_shared_prefix_prompt(prompt: str, reason_prefix: str) -> str:
    return (
        prompt
        + "\nResponse JSON prefix:\n"
        + json.dumps({"reason": reason_prefix}, ensure_ascii=False)[:-1]
        + ', "decision": {"action": "'
    )


def _build_action_completion(action: str, utility: float) -> str:
    payload = {
        "action": action,
        "action_input": {},
        "brief_rationale": f"Utility={utility:.2f}",
    }
    json_text = json.dumps(payload, ensure_ascii=False)
    return json_text[len('{"action": "') :] + "}"


def build_pairs(rollouts: list[dict[str, Any]], min_gap: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for record in rollouts:
        if not record.get("candidate_utilities"):
            diagnostics.append({"id": record.get("id"), "reason": "missing_candidate_utilities"})
            continue
        ranked = sorted(record["candidate_utilities"].items(), key=lambda item: item[1], reverse=True)
        chosen_action, chosen_utility = ranked[0]
        rejected_action, rejected_utility = ranked[-1]
        if chosen_action == rejected_action or chosen_utility - rejected_utility < min_gap:
            diagnostics.append(
                {
                    "id": record["id"],
                    "reason": "utility_gap_too_small",
                    "ranked_actions": ranked,
                }
            )
            continue
        reason_prefix = record.get("reason_prefix") or "Assess the current state and choose the next action."
        prompt = record.get("state_prompt") or _build_shared_prefix_prompt(record["prompt"], reason_prefix)
        pairs.append(
            {
                "id": record["id"],
                "dataset": record["dataset"],
                "prompt": prompt,
                "chosen": _build_action_completion(chosen_action, chosen_utility),
                "rejected": _build_action_completion(rejected_action, rejected_utility),
                "chosen_action": chosen_action,
                "rejected_action": rejected_action,
                "utility_gap": chosen_utility - rejected_utility,
                "state_tags": record.get("state_tags", []),
                "reason_prefix": reason_prefix,
                "pair_source": "rollout_pairs",
            }
        )
    return pairs, diagnostics


def build_oracle_pairs(samples, utility_config: dict[str, float], min_gap: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for sample in samples:
        semantic_tags = build_semantic_tags(sample)
        oracle_action = choose_oracle_action(sample, semantic_tags=semantic_tags)
        candidate_utilities = estimate_candidate_utilities(
            sample,
            semantic_tags,
            utility_config,
            answer_correctness=(oracle_action == "ANSWER"),
        )
        ranked = sorted(candidate_utilities.items(), key=lambda item: item[1], reverse=True)
        chosen_action, chosen_utility = ranked[0]
        rejected_action, rejected_utility = ranked[-1]
        if chosen_action == rejected_action or chosen_utility - rejected_utility < min_gap:
            diagnostics.append({"id": sample.id, "reason": "utility_gap_too_small", "ranked_actions": ranked})
            continue
        state_tags = [name for name, enabled in semantic_tags.items() if enabled]
        reason_prefix = "Assess the current state and choose the next action."
        prompt = _build_shared_prefix_prompt(
            build_prompt_text(sample, enable_tool_schema=True, state_tags=state_tags, reason_prefix=reason_prefix),
            reason_prefix,
        )
        pairs.append(
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "task_type": sample.task_type,
                "oracle_action": oracle_action,
                "prompt": prompt,
                "chosen": _build_action_completion(chosen_action, chosen_utility),
                "rejected": _build_action_completion(rejected_action, rejected_utility),
                "chosen_action": chosen_action,
                "rejected_action": rejected_action,
                "utility_gap": chosen_utility - rejected_utility,
                "state_tags": state_tags,
                "reason_prefix": reason_prefix,
                "pair_source": "oracle_pairs",
            }
        )
    return pairs, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs from rollout logs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--mode", choices=("rollouts", "oracle_pairs"), default="rollouts")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    args = parser.parse_args()

    config = load_config(args.config)
    min_gap = float(config["pairs"]["min_utility_gap"])
    if args.mode == "rollouts":
        rollouts = read_jsonl(args.input or config["paths"]["rollout_output"])
        pairs, diagnostics = build_pairs(rollouts, min_gap=min_gap)
    else:
        mix_cfg = config["data"][f"{args.split}_mix"]
        samples = load_mixed_datasets(
            dataset_limits=mix_cfg["per_dataset"],
            split_map=config["data"]["default_split"],
            dataset_root=Path(config["paths"]["dataset_root"]).resolve(),
            seed=int(mix_cfg["seed"]),
        )
        total_limit = int(mix_cfg["total_limit"])
        if len(samples) > total_limit:
            samples = samples[:total_limit]
        utility_config = load_utility_config(config)
        pairs, diagnostics = build_oracle_pairs(samples, utility_config=utility_config, min_gap=min_gap)
    write_jsonl(args.output or config["paths"]["pair_output"], pairs)
    write_jsonl(args.diagnostics_output or config["paths"]["diagnostics_output"], diagnostics)
    print(
        json.dumps(
            {
                "num_pairs": len(pairs),
                "num_diagnostics": len(diagnostics),
                "pair_output": args.output or config["paths"]["pair_output"],
                "mode": args.mode,
                "split": args.split,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
