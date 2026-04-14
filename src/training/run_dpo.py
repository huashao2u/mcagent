from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from src.data.loaders import load_mixed_datasets
from src.eval.decision_accuracy import evaluate_decision_accuracy
from src.pairs.build_natural_branch_pairs import build_natural_branch_pairs_from_samples
from src.pairs.build_pairs import build_oracle_pairs
from src.scoring.local_utility import load_utility_config
from src.utils.config import load_config
from src.utils.io import read_jsonl, write_json, write_jsonl


def _wait_for_file(path: Path, timeout_seconds: int = 300) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(1)
    raise FileNotFoundError(f"Timed out waiting for {path}")


class DecisionEvalCallback(TrainerCallback):
    def __init__(self, eval_samples, tokenizer, output_dir: Path, max_eval_samples: int = 200):
        self.eval_samples = eval_samples[:max_eval_samples]
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.best_metric = None
        self.latest_payload: dict[str, Any] | None = None

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return control
        payload = evaluate_decision_accuracy(model, self.tokenizer, self.eval_samples)
        self.latest_payload = payload
        if metrics is not None:
            metrics.update(payload)
        if int(os.environ.get("RANK", "0")) != 0:
            return control
        self.best_metric = payload["decision_action_accuracy"]
        report_path = self.output_dir / f"decision_eval_step_{state.global_step}.json"
        write_json(report_path, payload)
        try:
            import wandb

            if wandb.run is not None:
                flat = {"eval/" + k: v for k, v in payload.items() if isinstance(v, (int, float)) or v is None}
                flat["train/global_step"] = state.global_step
                wandb.log(flat, step=state.global_step)
        except Exception:
            pass
        return control


def _setup_wandb(config: dict[str, Any]) -> None:
    wandb_cfg = config["training"]["wandb"]
    if not wandb_cfg.get("enabled", False):
        os.environ["WANDB_DISABLED"] = "true"
        return
    os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
    os.environ["WANDB_MODE"] = str(wandb_cfg.get("mode", "online"))
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
    if wandb_cfg.get("run_name"):
        os.environ["WANDB_NAME"] = str(wandb_cfg["run_name"])
    if wandb_cfg.get("log_model"):
        os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _build_pair_dataset(config: dict[str, Any], split_name: str, output_path: Path) -> list[dict[str, Any]]:
    mix_cfg = config["data"][f"{split_name}_mix"]
    curriculum_cfg = config["training"].get("curriculum", {})
    builder_name = str(curriculum_cfg.get(f"{split_name}_builder", curriculum_cfg.get("train_builder", "oracle_pairs")))
    print(f"[mcagent] building {split_name} mixed samples...")
    samples = load_mixed_datasets(
        dataset_limits=mix_cfg["per_dataset"],
        split_map=config["data"]["default_split"],
        dataset_root=Path(config["paths"]["dataset_root"]).resolve(),
        seed=int(mix_cfg["seed"]),
    )
    total_limit = int(mix_cfg["total_limit"])
    if len(samples) > total_limit:
        samples = samples[:total_limit]
    print(f"[mcagent] {split_name} samples loaded: {len(samples)}")
    if builder_name == "natural_branch_pairs":
        # The default curriculum derives preferences from actual rollout branches instead of
        # static oracle labels, which keeps the DPO prompt closer to inference-time states.
        print(f"[mcagent] building {split_name} natural-branch pairs...")
        pairs, diagnostics, rollouts = build_natural_branch_pairs_from_samples(
            samples,
            config=config,
            min_gap=float(config["pairs"]["min_utility_gap"]),
        )
        write_jsonl(output_path.with_name(output_path.stem + "_rollouts.jsonl"), rollouts)
    else:
        utility_config = load_utility_config(config)
        print(f"[mcagent] building {split_name} oracle pairs...")
        pairs, diagnostics = build_oracle_pairs(samples, utility_config=utility_config, min_gap=float(config["pairs"]["min_utility_gap"]))
    write_jsonl(output_path, pairs)
    if diagnostics:
        diag_path = output_path.with_name(output_path.stem + "_diagnostics.jsonl")
        write_jsonl(diag_path, diagnostics)
    print(f"[mcagent] {split_name} pairs built with `{builder_name}`: {len(pairs)}")
    return pairs


def _load_model_and_tokenizer(config: dict[str, Any]):
    model_path = str(Path(config["paths"]["model_root"]).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bool(config["training"].get("bf16", False)) else (torch.float16 if bool(config["training"].get("fp16", True)) else "auto"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if config["training"]["lora"]["enabled"]:
        from peft import LoraConfig, get_peft_model

        lora_cfg = config["training"]["lora"]
        peft_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    return model, tokenizer


def run_formal_dpo(config: dict[str, Any], output_dir: Path, force_smoke: bool = False) -> dict[str, Any]:
    from trl import DPOConfig, DPOTrainer

    _setup_wandb(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = json.loads(json.dumps(config))
    if force_smoke:
        # Smoke mode shrinks both the mixed dataset and evaluation budget so the full distributed
        # training stack can be validated quickly without changing the main config on disk.
        runtime_config["data"]["train_mix"]["total_limit"] = min(int(runtime_config["data"]["train_mix"]["total_limit"]), 256)
        runtime_config["data"]["eval_mix"]["total_limit"] = min(int(runtime_config["data"]["eval_mix"]["total_limit"]), 64)
        runtime_config["data"]["train_mix"]["per_dataset"] = {
            key: min(int(value), 64) for key, value in runtime_config["data"]["train_mix"]["per_dataset"].items()
        }
        runtime_config["data"]["eval_mix"]["per_dataset"] = {
            key: min(int(value), 16) for key, value in runtime_config["data"]["eval_mix"]["per_dataset"].items()
        }
        runtime_config["training"]["max_eval_samples"] = min(int(runtime_config["training"]["max_eval_samples"]), 32)
        print("[mcagent] smoke mode active: reduced train/eval mixture sizes for fast GPU validation.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    fsdp_enabled = world_size > 1 and bool(runtime_config["training"].get("fsdp"))
    if fsdp_enabled:
        runtime_config["training"]["precompute_ref_log_probs"] = False
        runtime_config["training"]["gradient_checkpointing"] = False
        runtime_config["training"].setdefault("fsdp_config", {}).pop("activation_checkpointing", None)

    train_pairs_path = output_dir / "train_pairs.jsonl"
    eval_pairs_path = output_dir / "eval_pairs.jsonl"
    if local_rank == 0:
        # Rank 0 materializes the pair cache once; other workers block until the files exist so
        # all processes train on the exact same preference data.
        train_pairs = _build_pair_dataset(runtime_config, "train", train_pairs_path)
        eval_pairs = _build_pair_dataset(runtime_config, "eval", eval_pairs_path)
    else:
        train_pairs = []
        eval_pairs = []

    if local_rank != 0:
        _wait_for_file(train_pairs_path)
        _wait_for_file(eval_pairs_path)
        train_pairs = read_jsonl(train_pairs_path)
        eval_pairs = read_jsonl(eval_pairs_path)
    train_dataset = Dataset.from_list(train_pairs)
    eval_dataset = Dataset.from_list(eval_pairs)

    eval_samples = load_mixed_datasets(
        dataset_limits=runtime_config["data"]["eval_mix"]["per_dataset"],
        split_map=runtime_config["data"]["default_split"],
        dataset_root=Path(runtime_config["paths"]["dataset_root"]).resolve(),
        seed=int(runtime_config["data"]["eval_mix"]["seed"]),
    )
    eval_samples = eval_samples[: int(runtime_config["training"]["max_eval_samples"])]

    print("[mcagent] loading model/tokenizer...")
    model, tokenizer = _load_model_and_tokenizer(runtime_config)
    training_cfg = runtime_config["training"]
    max_steps = 2 if force_smoke else int(training_cfg["max_steps"])
    print(f"[mcagent] starting DPO training with {len(train_pairs)} train pairs and {len(eval_pairs)} eval pairs...")
    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        max_steps=max_steps,
        warmup_steps=int(training_cfg["warmup_steps"]),
        beta=float(training_cfg["beta"]),
        max_length=int(training_cfg["max_length"]),
        logging_steps=int(training_cfg["logging_steps"]),
        eval_steps=int(training_cfg["eval_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        bf16=bool(training_cfg.get("bf16", False)),
        fp16=bool(training_cfg.get("fp16", False)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        report_to=list(training_cfg.get("report_to", [])),
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=bool(training_cfg.get("ddp_find_unused_parameters", False)),
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        precompute_ref_log_probs=bool(training_cfg.get("precompute_ref_log_probs", False)),
        fsdp=training_cfg.get("fsdp") if fsdp_enabled else "",
        fsdp_transformer_layer_cls_to_wrap=training_cfg.get("fsdp_transformer_layer_cls_to_wrap") if fsdp_enabled else None,
        fsdp_config=training_cfg.get("fsdp_config") if fsdp_enabled else None,
    )
    decision_callback = DecisionEvalCallback(
        eval_samples=eval_samples,
        tokenizer=tokenizer,
        output_dir=output_dir,
        max_eval_samples=int(training_cfg["max_eval_samples"]),
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.add_callback(decision_callback)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    summary = {
        "trainer": "trl_dpo",
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "output_dir": str(output_dir),
        "smoke": force_smoke,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        decision_metrics = decision_callback.latest_payload
        if decision_metrics is None:
            if fsdp_enabled:
                decision_metrics = {
                    "decision_action_accuracy": None,
                    "num_samples": 0,
                    "per_dataset": {},
                    "records": [],
                    "note": "Decision evaluation payload was unavailable after distributed training.",
                }
            else:
                fallback_eval_samples = eval_samples[: min(len(eval_samples), 8 if force_smoke else len(eval_samples))]
                decision_metrics = evaluate_decision_accuracy(trainer.model, tokenizer, fallback_eval_samples)
        train_history = trainer.state.log_history
        curve_path = output_dir / "training_curve.json"
        write_json(curve_path, {"history": train_history})
        summary["decision_metrics"] = decision_metrics
        summary["curve_path"] = str(curve_path)
        if fsdp_enabled:
            # Full final-model export is skipped under FSDP because checkpoint shards/adapters are
            # the stable artifact in this training mode.
            summary["final_model_dir"] = None
            summary["final_model_note"] = "Skipped final_model export under FSDP; use checkpoint artifacts for adapter weights."
        else:
            final_model_dir = output_dir / "final_model"
            trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))
            summary["final_model_dir"] = str(final_model_dir)
        write_json(output_dir / "metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal LoRA + DPO training for MCAgent.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["paths"]["dpo_output_dir"])
    try:
        metrics = run_formal_dpo(config, output_dir, force_smoke=args.smoke or bool(config["training"].get("smoke", False)))
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
