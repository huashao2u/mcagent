from __future__ import annotations

from typing import Any


class SerperBackend:
    name = "serper"

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        raise RuntimeError("Serper backend is not enabled in this environment.")
