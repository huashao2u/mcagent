from __future__ import annotations

from typing import Any

from src.features.semantic_tags import active_semantic_tags
from src.teacher.prompts import build_teacher_tag_prompt


def label_semantic_tags(sample, teacher_client) -> dict[str, Any]:
    rule_tags = active_semantic_tags(sample)
    prompt = build_teacher_tag_prompt(sample.question, sample.metadata or {}, rule_tags)
    teacher_payload = teacher_client.complete_json(prompt)
    teacher_tags = teacher_payload.get("teacher_tags", rule_tags)
    return {
        "rule_tags": rule_tags,
        "teacher_tags": teacher_tags,
        "final_tags": teacher_tags if isinstance(teacher_tags, list) and teacher_tags else rule_tags,
        "teacher_note": teacher_payload.get("teacher_note") or teacher_payload.get("note", ""),
    }
