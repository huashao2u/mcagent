from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaselineDecision:
    reason: str
    action: str
    action_input: dict[str, Any]
    brief_rationale: str
    confidence: float


class BaseBaseline:
    name = "base"

    def decide(self, sample, semantic_tags: dict[str, bool]) -> BaselineDecision:
        raise NotImplementedError
