from __future__ import annotations

from typing import Any

from src.teacher.prompts import build_teacher_brief_rationale_prompt


def label_brief_rationale(sample, teacher_client, final_action: str, final_tags: list[str]) -> dict[str, Any]:
    prompt = build_teacher_brief_rationale_prompt(sample.question, final_action, final_tags)
    teacher_payload = teacher_client.complete_json(prompt)
    rationale = teacher_payload.get("teacher_brief_rationale") or f"{final_action} is the most suitable next action under the current boundary."
    return {
        "teacher_brief_rationale": rationale,
        "teacher_note": teacher_payload.get("teacher_note") or teacher_payload.get("note", ""),
    }
