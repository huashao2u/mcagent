from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.loaders import load_dataset, load_mixed_datasets
from src.teacher.client import TeacherClient
from src.teacher.label_semantic_tags import label_semantic_tags
from src.teacher.label_standard_action import label_standard_action
from src.utils.config import load_config
from src.utils.io import write_jsonl


def _load_samples(config: dict, split_name: str, datasets: str | None, limit: int | None):
    if datasets:
        dataset_root = Path(config["paths"]["dataset_root"]).resolve()
        split_map = config["data"]["default_split"]
        loaded = []
        for dataset_name in [item.strip().lower() for item in datasets.split(",") if item.strip()]:
            loaded.extend(load_dataset(dataset_name, split=split_map[dataset_name], limit=limit, dataset_root=dataset_root))
        return loaded
    mix_cfg = config["data"][f"{split_name}_mix"]
    samples = load_mixed_datasets(
        dataset_limits=mix_cfg["per_dataset"],
        split_map=config["data"]["default_split"],
        dataset_root=Path(config["paths"]["dataset_root"]).resolve(),
        seed=int(mix_cfg["seed"]),
    )
    if limit is not None:
        return samples[:limit]
    return samples[: int(mix_cfg["total_limit"])]


def run_teacher_labeling(config: dict, split_name: str, datasets: str | None, limit: int | None) -> list[dict]:
    teacher_client = TeacherClient(config)
    records = []
    for sample in _load_samples(config, split_name, datasets, limit):
        semantic_payload = label_semantic_tags(sample, teacher_client)
        action_payload = label_standard_action(sample, teacher_client, semantic_payload["final_tags"])
        teacher_note = " | ".join(
            note for note in [semantic_payload.get("teacher_note"), action_payload.get("teacher_note")] if note
        )
        records.append(
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "task_type": sample.task_type,
                "question": sample.question,
                "rule_tags": semantic_payload["rule_tags"],
                "teacher_tags": semantic_payload["teacher_tags"],
                "final_tags": semantic_payload["final_tags"],
                "rule_action": action_payload["rule_action"],
                "teacher_action": action_payload["teacher_action"],
                "final_action": action_payload["final_action"],
                "teacher_note": teacher_note,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCAgent teacher labeling.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    records = run_teacher_labeling(config, args.split, args.datasets, args.limit)
    output_path = Path(args.output or config["paths"]["teacher_label_output"])
    write_jsonl(output_path, records)
    print(json.dumps({"output": str(output_path), "num_records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
