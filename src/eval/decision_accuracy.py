from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from src.data.loaders import UnifiedSample
from src.features.semantic_tags import build_semantic_tags
from src.prompting.build_prompts import build_prompt_text
from src.scoring.action_oracle import choose_oracle_action


ACTION_TOKENS = ["ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE"]


@dataclass
class DecisionEvalRecord:
    sample_id: str
    dataset: str
    oracle_action: str
    predicted_action: str
    correct: bool


def _action_token_id_map(tokenizer) -> dict[str, int | None]:
    token_map: dict[str, int | None] = {}
    for action in ACTION_TOKENS:
        token_ids = tokenizer.encode(" " + action, add_special_tokens=False)
        token_map[action] = token_ids[0] if token_ids else None
    return token_map


def _batched_prompt_last_token_distributions(model, tokenizer, prompts: list[str], batch_size: int = 8) -> list[dict[str, float]]:
    action_token_ids = _action_token_id_map(tokenizer)
    all_scores: list[dict[str, float]] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        attention_mask = inputs["attention_mask"]
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        last_token_logits = logits[batch_indices, last_token_indices]
        for row in last_token_logits:
            scores: dict[str, float] = {}
            for action, token_id in action_token_ids.items():
                if token_id is None:
                    scores[action] = float("-inf")
                    continue
                scores[action] = float(row[token_id].detach().cpu().item())
            all_scores.append(scores)
        del inputs, outputs, logits, last_token_logits
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
    return all_scores


def evaluate_decision_accuracy(model, tokenizer, samples: list[UnifiedSample]) -> dict[str, Any]:
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    local_samples = samples[rank::world_size]
    records: list[DecisionEvalRecord] = []
    dataset_names = sorted({sample.dataset for sample in samples})
    dataset_to_index = {name: index for index, name in enumerate(dataset_names)}
    local_correct = 0
    local_total = 0
    local_per_dataset_correct = [0 for _ in dataset_names]
    local_per_dataset_total = [0 for _ in dataset_names]
    prompts = [build_prompt_text(sample, enable_tool_schema=True) for sample in local_samples]
    all_scores = _batched_prompt_last_token_distributions(model, tokenizer, prompts)
    for sample, scores in zip(local_samples, all_scores):
        oracle_action = choose_oracle_action(sample, semantic_tags=build_semantic_tags(sample))
        predicted_action = max(scores.items(), key=lambda item: item[1])[0]
        correct = predicted_action == oracle_action
        dataset_index = dataset_to_index[sample.dataset]
        local_correct += int(correct)
        local_total += 1
        local_per_dataset_correct[dataset_index] += int(correct)
        local_per_dataset_total[dataset_index] += 1
        records.append(
            DecisionEvalRecord(
                sample_id=sample.id,
                dataset=sample.dataset,
                oracle_action=oracle_action,
                predicted_action=predicted_action,
                correct=correct,
            )
        )
    device = model.device if hasattr(model, "device") else torch.device("cpu")
    summary_tensor = torch.tensor(
        [local_correct, local_total, *local_per_dataset_correct, *local_per_dataset_total],
        device=device,
        dtype=torch.long,
    )
    if world_size > 1:
        dist.all_reduce(summary_tensor, op=dist.ReduceOp.SUM)
    reduced = summary_tensor.detach().cpu().tolist()
    total_correct = int(reduced[0])
    total = int(reduced[1])
    split = 2 + len(dataset_names)
    reduced_correct = reduced[2:split]
    reduced_total = reduced[split:]
    overall = total_correct / total if total else None
    return {
        "decision_action_accuracy": overall,
        "num_samples": total,
        "per_dataset": {
            dataset_name: {
                "decision_action_accuracy": reduced_correct[index] / reduced_total[index] if reduced_total[index] else None,
                "num_samples": reduced_total[index],
            }
            for index, dataset_name in enumerate(dataset_names)
        },
        "records": [record.__dict__ for record in records[:20]],
    }
