from __future__ import annotations

from typing import Any


def to_tool_call_payload(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": (decision.get("action") or "").lower(),
        "arguments": decision.get("action_input", {}) or {},
    }
