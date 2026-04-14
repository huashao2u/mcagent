from __future__ import annotations

import json


def build_teacher_tag_prompt(question: str, metadata: dict, rule_tags: list[str]) -> str:
    return (
        "You are labeling MCAgent boundary tags.\n"
        "Return JSON only with keys: teacher_tags, teacher_note.\n"
        f"Question: {question}\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
        f"Rule tags: {json.dumps(rule_tags, ensure_ascii=False)}"
    )


def build_teacher_action_prompt(question: str, metadata: dict, rule_action: str, final_tags: list[str]) -> str:
    return (
        "You are adjudicating the best MCAgent action.\n"
        "Return JSON only with keys: teacher_action, teacher_note.\n"
        f"Question: {question}\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
        f"Rule action: {rule_action}\n"
        f"Tags: {json.dumps(final_tags, ensure_ascii=False)}"
    )


def build_teacher_brief_rationale_prompt(question: str, final_action: str, final_tags: list[str]) -> str:
    return (
        "Write one short action rationale for MCAgent.\n"
        "Return JSON only with key: teacher_brief_rationale.\n"
        f"Question: {question}\n"
        f"Final action: {final_action}\n"
        f"Tags: {json.dumps(final_tags, ensure_ascii=False)}"
    )
