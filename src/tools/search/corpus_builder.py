from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.loaders import UnifiedSample, load_dataset


SAFE_METADATA_KEYS = (
    "category",
    "fact_type",
    "effective_year",
    "next_review",
    "num_hops",
    "source",
    "graph_preview",
    "graph_size",
    "thought",
)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def build_corpus_document(sample: UnifiedSample) -> dict[str, Any]:
    metadata = sample.metadata or {}
    metadata_lines: list[str] = []
    for key in SAFE_METADATA_KEYS:
        text = _stringify(metadata.get(key))
        if text:
            metadata_lines.append(f"{key}: {text}")
    text_parts = [sample.question.strip()]
    if metadata_lines:
        text_parts.append("\n".join(metadata_lines))
    text = "\n".join(part for part in text_parts if part)
    snippet = text.replace("\n", " ")[:240]
    return {
        "id": sample.id,
        "dataset": sample.dataset,
        "task_type": sample.task_type,
        "title": sample.question.strip()[:120],
        "text": text,
        "snippet": snippet,
        "metadata": {
            "dataset": sample.dataset,
            "task_type": sample.task_type,
            "source_sample_id": sample.id,
        },
    }


def build_search_corpus(config: dict[str, Any], split_name: str | None = None) -> list[dict[str, Any]]:
    dataset_root = Path(config["paths"]["dataset_root"]).resolve()
    default_split = config["data"]["default_split"]
    dataset_names = list(config["data"]["phase_a_datasets"])
    max_docs = int(config.get("tools", {}).get("search", {}).get("max_docs_per_dataset", 400))
    corpus: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for dataset_name in dataset_names:
        split = default_split[dataset_name]
        samples = load_dataset(dataset_name, split=split, limit=max_docs, dataset_root=dataset_root)
        for sample in samples:
            if sample.id in seen_ids:
                continue
            corpus.append(build_corpus_document(sample))
            seen_ids.add(sample.id)
    return corpus
