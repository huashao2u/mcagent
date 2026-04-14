from __future__ import annotations

from typing import Any

from src.scoring.action_oracle import choose_oracle_action
from src.teacher.prompts import build_teacher_action_prompt


def label_standard_action(sample, teacher_client, final_tags: list[str]) -> dict[str, Any]:
    rule_action = choose_oracle_action(sample)
    prompt = build_teacher_action_prompt(sample.question, sample.metadata or {}, rule_action, final_tags)
    teacher_payload = teacher_client.complete_json(prompt)
    teacher_action = teacher_payload.get("teacher_action", rule_action)
    return {
        "rule_action": rule_action,
        "teacher_action": teacher_action,
        "final_action": teacher_action if isinstance(teacher_action, str) and teacher_action else rule_action,
        "teacher_note": teacher_payload.get("teacher_note") or teacher_payload.get("note", ""),
    }
