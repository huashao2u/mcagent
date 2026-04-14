from __future__ import annotations

import json
import os
from typing import Any


class TeacherClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config.get("teacher", {})
        self.enabled = bool(self.config.get("enabled", False))
        self.model = self.config.get("model", "gpt-4o")
        self.api_key = os.environ.get(str(self.config.get("api_key_env", "OPENAI_API_KEY")), "")
        self.base_url = os.environ.get(str(self.config.get("base_url_env", "OPENAI_BASE_URL")), "")

    def is_ready(self) -> bool:
        return self.enabled and bool(self.api_key)

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if not self.is_ready():
            return {"disabled": True, "prompt": prompt, "note": "Teacher labeling is disabled or credentials are missing."}
        # Future online provider integration point.
        return {"disabled": False, "model": self.model, "prompt": prompt, "note": "Online teacher client is not implemented in this environment."}
