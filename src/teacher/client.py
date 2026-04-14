from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class TeacherClient:
    def __init__(self, config: dict[str, Any]):
        self.config = config.get("teacher", {})
        self.enabled = bool(self.config.get("enabled", False))
        self.model = self.config.get("model", "gpt-4o")
        self.api_key = os.environ.get(str(self.config.get("api_key_env", "OPENAI_API_KEY")), "")
        self.base_url = os.environ.get(str(self.config.get("base_url_env", "OPENAI_BASE_URL")), "")
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_retries = int(self.config.get("max_retries", 3))

    def is_ready(self) -> bool:
        return self.enabled and bool(self.api_key)

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if not self.is_ready():
            return {"disabled": True, "prompt": prompt, "note": "Teacher labeling is disabled or credentials are missing."}
        endpoint = self._build_endpoint()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise JSON labeling assistant. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        last_error = None
        for _ in range(self.max_retries):
            try:
                response = self._post_json(endpoint, payload)
                content = (((response.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
                parsed = self._parse_json_content(content)
                parsed["disabled"] = False
                parsed["model"] = self.model
                return parsed
            except Exception as exc:  # pragma: no cover - exercised via fallback in offline envs
                last_error = str(exc)
        return {
            "disabled": False,
            "model": self.model,
            "prompt": prompt,
            "note": f"Teacher request failed; falling back to rule-only labeling. error={last_error}",
        }

    def _build_endpoint(self) -> str:
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return base_url + "/chat/completions"

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        try:
            from json_repair import repair_json

            candidate = repair_json(content, return_objects=True)
            if isinstance(candidate, dict):
                return candidate
        except Exception:
            pass
        return json.loads(content)
