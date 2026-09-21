from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

from .io_utils import append_jsonl, iter_jsonl, sha256_json


class LLMError(RuntimeError):
    pass


T = TypeVar("T")


def read_api_key(settings: dict[str, Any]) -> str:
    key_name = str(settings.get("api_key_env", "DEEPSEEK_API_KEY"))
    api_key = os.environ.get(key_name)
    if not api_key:
        raise LLMError(f"Environment variable {key_name} is not set")
    if not api_key.isascii():
        raise LLMError(
            f"Environment variable {key_name} contains non-ASCII characters. "
            "Set it to the actual API key, not the example placeholder."
        )
    if any(character.isspace() for character in api_key):
        raise LLMError(
            f"Environment variable {key_name} contains whitespace. "
            "Re-enter the API key without spaces or line breaks."
        )
    return api_key


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Response does not contain a JSON object")


class LLMClient:
    def __init__(self, settings: dict[str, Any], cache_path: str | Path):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("The openai package is required for DeepSeek calls; install requirements.txt") from exc
        self.settings = dict(settings)
        api_key = read_api_key(settings)
        self.client = OpenAI(
            api_key=api_key,
            base_url=str(settings["base_url"]),
            timeout=float(settings.get("timeout_seconds", 120)),
            max_retries=2,
        )
        self.cache_path = Path(cache_path)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for record in iter_jsonl(self.cache_path):
                key = str(record["key"])
                if record.get("status") == "ok":
                    self.cache[key] = str(record["response"])
                elif record.get("status") == "invalid":
                    self.cache.pop(key, None)

    def _key(self, prompt: str, purpose: str, attempt: int) -> str:
        return sha256_json(
            {
                "model": self.settings["model"],
                "temperature": self.settings.get("temperature", 0),
                "max_tokens": self.settings.get("max_tokens", 800),
                "thinking": self.settings.get("thinking"),
                "prompt": prompt,
                "purpose": purpose,
                "attempt": attempt,
            }
        )

    def complete(self, prompt: str, purpose: str, attempt: int = 0) -> str:
        key = self._key(prompt, purpose, attempt)
        if key in self.cache:
            return self.cache[key]
        request: dict[str, Any] = {
            "model": str(self.settings["model"]),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.settings.get("temperature", 0)),
            "max_tokens": int(self.settings.get("max_tokens", 800)),
        }
        thinking = self.settings.get("thinking")
        if thinking is not None:
            request["extra_body"] = {
                "thinking": {"type": "enabled" if bool(thinking) else "disabled"}
            }
        response = self.client.chat.completions.create(**request)
        if not response.choices or response.choices[0].message.content is None:
            raise LLMError(f"Empty response for {purpose}")
        text = response.choices[0].message.content.strip()
        append_jsonl(
            self.cache_path,
            {
                "key": key,
                "model": self.settings["model"],
                "purpose": purpose,
                "attempt": attempt,
                "status": "ok",
                "response": text,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                },
            },
        )
        self.cache[key] = text
        return text

    def _invalidate(self, prompt: str, purpose: str, attempt: int, error: Exception) -> None:
        key = self._key(prompt, purpose, attempt)
        self.cache.pop(key, None)
        append_jsonl(
            self.cache_path,
            {
                "key": key,
                "model": self.settings["model"],
                "purpose": purpose,
                "attempt": attempt,
                "status": "invalid",
                "error": str(error),
            },
        )

    def call_json(
        self,
        prompt: str,
        purpose: str,
        validator: Callable[[dict[str, Any]], T],
    ) -> T:
        max_attempts = int(self.settings.get("max_retries", 3))
        errors: list[str] = []
        for attempt in range(max_attempts):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\n\nYour previous response did not match the required JSON schema. "
                    "Return exactly one valid JSON object with every required field and no markdown. "
                    "Keep every string within the prompt's word limit, summarize long inputs, "
                    "and never copy descriptions or other long passages verbatim."
                )
            request_prompt = prompt + retry_note
            cache_key = self._key(request_prompt, purpose, attempt)
            was_cached = cache_key in self.cache
            response = self.complete(request_prompt, purpose, attempt)
            while True:
                try:
                    return validator(extract_json_object(response))
                except (KeyError, TypeError, ValueError) as exc:
                    source = "cached" if was_cached else "fresh"
                    errors.append(f"attempt {attempt + 1} ({source}): {exc}")
                    self._invalidate(request_prompt, purpose, attempt, exc)
                    if not was_cached:
                        break
                    # A raw response can be cached before schema validation. If an
                    # interrupted run resumes with such an invalid entry, evict it
                    # and make one fresh request for the same logical attempt.
                    was_cached = False
                    response = self.complete(request_prompt, purpose, attempt)
        raise LLMError(f"Invalid {purpose} response after {max_attempts} attempts: {'; '.join(errors)}")


class ScriptedLLM:
    """Deterministic test double implementing the LLMClient call_json interface."""

    def __init__(self, responder: Callable[[str, str], dict[str, Any]]):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def call_json(
        self,
        prompt: str,
        purpose: str,
        validator: Callable[[dict[str, Any]], T],
    ) -> T:
        self.calls.append((purpose, prompt))
        return validator(self.responder(purpose, prompt))
