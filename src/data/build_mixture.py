from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.loaders import load_mixed_datasets
from src.utils.config import load_config
from src.utils.io import write_json


def build_mixture_summary(config: dict, split_name: str) -> dict:
    mix_cfg = config["data"][f"{split_name}_mix"]
    split_map = config["data"]["default_split"]
    samples = load_mixed_datasets(
        dataset_limits=mix_cfg["per_dataset"],
        split_map=split_map,
        dataset_root=Path(config["paths"]["dataset_root"]).resolve(),
        seed=int(mix_cfg["seed"]),
    )
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.dataset] = counts.get(sample.dataset, 0) + 1
    total_limit = int(mix_cfg["total_limit"])
    if len(samples) > total_limit:
        samples = samples[:total_limit]
        counts = {}
        for sample in samples:
            counts[sample.dataset] = counts.get(sample.dataset, 0) + 1
    return {
        "split": split_name,
        "total": len(samples),
        "counts": counts,
        "requested_limits": mix_cfg["per_dataset"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and summarize MCAgent mixed datasets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    summary = build_mixture_summary(config, args.split)
    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
