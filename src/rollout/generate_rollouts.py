from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.loaders import DatasetArtifactMissingError, UnifiedSample, load_dataset
from src.envs.sandbox import SandboxEnv
from src.eval.evaluate_answers import is_answer_correct
from src.features.extract_process_features import extract_process_features
from src.features.semantic_tags import active_semantic_tags, build_semantic_tags
from src.prompting.build_prompts import build_prompt_text
from src.rollout.policy import HFLocalPolicy, build_policy
from src.scoring.action_oracle import choose_oracle_action
from src.scoring.local_utility import estimate_candidate_utilities, load_utility_config, score_action
from src.utils.config import load_config
from src.utils.io import write_jsonl


def _build_state_prompt(prompt_text: str, reason_prefix: str) -> str:
    return (
        prompt_text
        + "\nResponse JSON prefix:\n"
        + json.dumps({"reason": reason_prefix}, ensure_ascii=False)[:-1]
        + ', "decision": {"action": "'
    )


def _resolve_datasets(config: dict[str, Any], requested: str | None) -> list[str]:
    if requested:
        return [name.strip().lower() for name in requested.split(",") if name.strip()]
    return [name.lower() for name in config["data"]["phase_a_datasets"]]


def _load_samples(config: dict[str, Any], datasets: list[str], limit_per_dataset: int | None) -> tuple[list[UnifiedSample], list[dict[str, str]]]:
    split_map = config["data"]["default_split"]
    samples: list[UnifiedSample] = []
    skipped: list[dict[str, str]] = []
    for dataset_name in datasets:
        split = split_map[dataset_name]
        try:
            loaded = load_dataset(dataset_name, split=split, limit=limit_per_dataset)
            samples.extend(loaded)
        except (DatasetArtifactMissingError, FileNotFoundError, RuntimeError) as exc:
            if config["data"].get("skip_unavailable", True):
                skipped.append({"dataset": dataset_name, "reason": str(exc)})
                continue
            raise
    return samples, skipped


def run_rollout(
    sample: UnifiedSample,
    policy,
    utility_config: dict[str, float],
    token_threshold: int,
    config: dict[str, Any],
    search_phase: str,
) -> dict[str, Any]:
    semantic_tags = build_semantic_tags(sample)
    state_tags = active_semantic_tags(sample)
    prompt_text = build_prompt_text(sample, enable_tool_schema=True, state_tags=state_tags)
    policy_output = policy.generate_decision(sample, prompt_text)
    decision = dict(policy_output.decision)
    reason_prefix = policy_output.reason.strip()
    history_prefix: list[dict[str, Any]] = []
    # `state_prompt` freezes the shared prefix up to the action token so later pair builders
    # can compare alternative actions under the exact same state.
    state_prompt = _build_state_prompt(prompt_text, reason_prefix)
    state_snapshot = {
        "dataset": sample.dataset,
        "task_type": sample.task_type,
        "state_tags": state_tags,
        "history": history_prefix,
    }
    env = SandboxEnv(sample, config=config, search_phase=search_phase)
    tool_observation = None
    final_answer = None
    final_status = "pending"
    if decision["action"] == "ANSWER":
        _, _, _ = env.step("ANSWER", decision["action_input"])
        final_answer = decision["action_input"].get("answer")
        final_status = "answered"
    else:
        # Tools either terminate immediately (e.g. REFUSE/CLARIFY in heuristic mode) or yield
        # an observation that the policy converts into a final answer.
        tool_observation, done, info = env.step(decision["action"], decision["action_input"])
        if done:
            final_answer = tool_observation.get("reason")
            final_status = tool_observation.get("status", "completed")
        else:
            followup = policy.finalize_after_tool(sample, decision, tool_observation)
            final_answer = followup["final_answer"]
            final_status = followup["final_status"]
    correctness = is_answer_correct(sample.to_dict(), final_answer)
    oracle_action = choose_oracle_action(sample, semantic_tags=semantic_tags)
    candidate_utilities = estimate_candidate_utilities(sample, semantic_tags, utility_config, answer_correctness=correctness)
    actual_utility = score_action(decision["action"], sample, semantic_tags, utility_config, correctness=correctness)
    confidence_value = decision.get("confidence")
    try:
        action_confidence = None if confidence_value is None else max(0.0, min(1.0, float(confidence_value)))
    except (TypeError, ValueError):
        action_confidence = None
    confidence_source = policy_output.confidence_source or "unknown"
    # Heuristic policies and malformed model outputs may omit confidence, so we backfill from
    # action probabilities first and only then fall back to a neutral default.
    if action_confidence is None and policy_output.action_probabilities:
        action_confidence = float(policy_output.action_probabilities.get(decision["action"], 0.5))
        confidence_source = "action_probability_fallback"
    if action_confidence is None:
        action_confidence = 0.5
        confidence_source = "neutral_fallback"
    verbal_confidence = round(action_confidence, 4)
    answer_confidence = verbal_confidence if final_answer not in (None, "") else None
    process_features = extract_process_features(
        reason=policy_output.reason,
        raw_text=policy_output.raw_text,
        tool_observation=tool_observation,
        token_threshold=token_threshold,
    )
    return {
        "id": sample.id,
        "dataset": sample.dataset,
        "task_type": sample.task_type,
        "question": sample.question,
        "gold_answer": sample.gold_answer,
        "metadata": sample.metadata,
        "prompt": prompt_text,
        "state_prompt": state_prompt,
        "state_snapshot": state_snapshot,
        "reason_prefix": reason_prefix,
        "history_prefix": history_prefix,
        "reason": policy_output.reason,
        "decision": decision,
        "tool_observation": tool_observation,
        "final_answer": final_answer,
        "final_status": final_status,
        "correctness": correctness,
        "oracle_action": oracle_action,
        "semantic_tags": state_tags,
        "state_tags": state_tags,
        "process_features": process_features,
        "candidate_utilities": candidate_utilities,
        "actual_utility": actual_utility,
        "verbal_confidence": verbal_confidence,
        "action_confidence": action_confidence,
        "answer_confidence": answer_confidence,
        "action_confidence_source": confidence_source,
        "action_scores": policy_output.action_scores or {},
        "action_probabilities": policy_output.action_probabilities or {},
        "score_answer": candidate_utilities.get("ANSWER"),
        "score_search": candidate_utilities.get("SEARCH"),
        "score_calculate": candidate_utilities.get("CALCULATE"),
        "score_clarify": candidate_utilities.get("CLARIFY"),
        "score_refuse": candidate_utilities.get("REFUSE"),
        "raw_text": policy_output.raw_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCAgent rollout logs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--backend", default=None, choices=("heuristic", "hf"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--verify-hf-assets", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    backend = args.backend or config["rollout"]["backend"]
    model_path = str(Path(config["paths"]["model_root"]).resolve())
    if args.verify_hf_assets:
        print(json.dumps(HFLocalPolicy.verify_assets(model_path), ensure_ascii=False, indent=2))
        return

    datasets = _resolve_datasets(config, args.datasets)
    limit_per_dataset = args.limit_per_dataset or config["data"]["limit_per_dataset"]
    output_path = Path(args.output or config["paths"]["rollout_output"])
    utility_config = load_utility_config(config)
    samples, skipped = _load_samples(config, datasets, limit_per_dataset)
    policy = build_policy(
        backend=backend,
        model_path=model_path,
        exploration_rate=float(config["rollout"]["exploration_rate"]),
        max_new_tokens=int(config["rollout"]["max_new_tokens"]),
    )

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
    if skipped:
        rollouts.append(
            {
                "id": "resource-report",
                "dataset": "system",
                "task_type": "meta",
                "question": "resource availability report",
                "gold_answer": None,
                "metadata": {"skipped_datasets": skipped},
                "prompt": "",
                "state_prompt": "",
                "state_snapshot": {},
                "reason_prefix": "",
                "history_prefix": [],
                "reason": "Datasets with missing real artifacts were skipped.",
                "decision": {"action": "REFUSE", "confidence": 1.0, "action_input": {}, "brief_rationale": "Meta report"},
                "tool_observation": None,
                "final_answer": None,
                "final_status": "resource_report",
                "correctness": None,
                "oracle_action": "REFUSE",
                "semantic_tags": [],
                "state_tags": [],
                "process_features": {},
                "candidate_utilities": {},
                "actual_utility": 0.0,
                "verbal_confidence": 1.0,
                "action_confidence": 1.0,
                "answer_confidence": None,
                "action_confidence_source": "meta_report",
                "action_scores": {},
                "action_probabilities": {},
                "raw_text": "",
            }
        )
    write_jsonl(output_path, rollouts)
    print(json.dumps({"output": str(output_path), "num_rollouts": len(rollouts), "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
