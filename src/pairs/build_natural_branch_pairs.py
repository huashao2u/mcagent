from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.loaders import load_mixed_datasets
from src.rollout.generate_rollouts import run_rollout
from src.rollout.policy import build_policy
from src.scoring.local_utility import load_utility_config
from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl


def _build_action_completion(action: str, utility: float) -> str:
    payload = {
        "action": action,
        "action_input": {},
        "brief_rationale": f"Utility={utility:.2f}",
    }
    json_text = json.dumps(payload, ensure_ascii=False)
    return json_text[len('{"action": "') :] + "}"


def build_natural_branch_pairs(rollouts: list[dict[str, Any]], min_gap: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in rollouts:
        state_prompt = record.get("state_prompt")
        if not state_prompt or record.get("dataset") == "system":
            continue
        # Multiple rollout records can correspond to the same pre-action state; we group by the
        # shared prompt prefix so DPO compares actions rather than unrelated questions.
        grouped.setdefault(state_prompt, []).append(record)

    pairs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for state_prompt, records in grouped.items():
        seed = records[0]
        branch_utilities: dict[str, float] = {}
        for record in records:
            action = (record.get("decision") or {}).get("action")
            if not action:
                continue
            utility = (record.get("candidate_utilities") or {}).get(action, record.get("actual_utility"))
            if utility is None:
                continue
            # Keep the best observed utility for each action branch before ranking branches.
            branch_utilities[action] = max(float(utility), branch_utilities.get(action, float("-inf")))
        if len(branch_utilities) < 2:
            # If exploration did not materialize enough distinct branches, fall back to the
            # per-action utility estimates stored on the seed rollout.
            for action, utility in (seed.get("candidate_utilities") or {}).items():
                branch_utilities.setdefault(action, float(utility))
        ranked = sorted(branch_utilities.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) < 2:
            diagnostics.append({"id": seed.get("id"), "reason": "not_enough_branches", "state_prompt": state_prompt[:120]})
            continue
        chosen_action, chosen_utility = ranked[0]
        rejected_action, rejected_utility = ranked[-1]
        if chosen_action == rejected_action or chosen_utility - rejected_utility < min_gap:
            diagnostics.append({"id": seed.get("id"), "reason": "utility_gap_too_small", "ranked_actions": ranked})
            continue
        if not seed.get("state_tags") and not seed.get("semantic_tags"):
            diagnostics.append({"id": seed.get("id"), "reason": "semantic_state_unclear"})
            continue
        pairs.append(
            {
                "id": seed["id"],
                "dataset": seed["dataset"],
                "task_type": seed["task_type"],
                "oracle_action": seed.get("oracle_action"),
                "prompt": state_prompt,
                "chosen": _build_action_completion(chosen_action, chosen_utility),
                "rejected": _build_action_completion(rejected_action, rejected_utility),
                "chosen_action": chosen_action,
                "rejected_action": rejected_action,
                "utility_gap": chosen_utility - rejected_utility,
                "state_tags": seed.get("state_tags", []),
                "reason_prefix": seed.get("reason_prefix", ""),
                "pair_source": "natural_branch_pairs",
            }
        )
    return pairs, diagnostics


def build_natural_branch_pairs_from_samples(samples, config: dict[str, Any], min_gap: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    utility_config = load_utility_config(config)
    policy = build_policy(
        backend=config["rollout"]["backend"],
        model_path=str(Path(config["paths"]["model_root"]).resolve()),
        exploration_rate=float(config["rollout"]["exploration_rate"]),
        max_new_tokens=int(config["rollout"]["max_new_tokens"]),
    )
    # Re-run the rollout stage here so pair construction reflects the same policy/config used
    # during training curriculum generation.
    rollouts = [
        run_rollout(
            sample,
            policy=policy,
            utility_config=utility_config,
            token_threshold=int(config["features"]["long_reason_token_threshold"]),
            config=config,
            search_phase=str(config["rollout"].get("search_mode", "train")),
        )
        for sample in samples
    ]
    pairs, diagnostics = build_natural_branch_pairs(rollouts, min_gap=min_gap)
    return pairs, diagnostics, rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build natural-branch shared-prefix DPO pairs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics-output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rollouts = read_jsonl(args.input or config["paths"]["rollout_output"])
    pairs, diagnostics = build_natural_branch_pairs(rollouts, min_gap=float(config["pairs"]["min_utility_gap"]))
    write_jsonl(args.output, pairs)
    write_jsonl(args.diagnostics_output, diagnostics)
    print(json.dumps({"output": args.output, "num_pairs": len(pairs), "num_diagnostics": len(diagnostics)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
