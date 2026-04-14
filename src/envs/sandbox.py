from __future__ import annotations

from typing import Any

from src.tools.calculator_tool import CalculatorTool
from src.tools.clarify_tool import ClarifyTool
from src.tools.refuse_tool import RefuseTool
from src.tools.search_tool import SearchTool


class SandboxEnv:
    def __init__(self, sample, config: dict[str, Any] | None = None):
        self.sample = sample
        self.tools = {
            "SEARCH": SearchTool(config=config),
            "CALCULATE": CalculatorTool(),
            "CLARIFY": ClarifyTool(),
            "REFUSE": RefuseTool(),
        }
        self.history: list[dict[str, Any]] = []

    def step(self, action_name: str, action_input: dict[str, Any]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        normalized = action_name.upper()
        if normalized == "ANSWER":
            observation = {"status": "answered"}
            info = {"terminal": True}
            self.history.append({"action": normalized, "action_input": action_input, "observation": observation})
            return observation, True, info
        if normalized not in self.tools:
            raise KeyError(f"Unsupported action for sandbox: {action_name}")
        observation, done, info = self.tools[normalized].run(action_input, self.sample, self.history)
        self.history.append({"action": normalized, "action_input": action_input, "observation": observation})
        return observation, done, info
