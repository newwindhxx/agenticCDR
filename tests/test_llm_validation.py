from __future__ import annotations

import pytest

from agentic_cdr.config import public_config
from agentic_cdr.evaluation import _ranking_validator
from agentic_cdr.llm import LLMClient, LLMError, extract_json_object, read_api_key


def test_extract_json_accepts_wrapping_text():
    assert extract_json_object('prefix ```json\n{"ranking":["C00"]}\n``` suffix') == {"ranking": ["C00"]}


@pytest.mark.parametrize("ranking", [["C00"], ["C00", "C00"], ["C00", "C02"]])
def test_ranking_validator_rejects_missing_duplicate_or_unknown(ranking):
    with pytest.raises(ValueError):
        _ranking_validator(["C00", "C01"])({"ranking": ranking})


def test_call_json_retries_invalid_format(monkeypatch):
    client = object.__new__(LLMClient)
    client.settings = {"model": "deepseek-flash", "max_retries": 3}
    client.cache = {}
    responses = iter(["not json", '{"value":"ok"} trailing text'])
    monkeypatch.setattr(client, "complete", lambda prompt, purpose, attempt=0: next(responses))
    monkeypatch.setattr(client, "_invalidate", lambda prompt, purpose, attempt, error: None)
    assert client.call_json("prompt", "test", lambda value: value["value"]) == "ok"


def test_call_json_evicts_invalid_cached_response(monkeypatch):
    client = object.__new__(LLMClient)
    client.settings = {
        "model": "deepseek-flash",
        "max_retries": 3,
        "max_tokens": 800,
        "temperature": 0,
        "thinking": False,
    }
    key = client._key("prompt", "test", 0)
    client.cache = {key: "truncated response"}
    fresh_calls = []

    def complete(prompt, purpose, attempt=0):
        response_key = client._key(prompt, purpose, attempt)
        if response_key in client.cache:
            return client.cache[response_key]
        fresh_calls.append((prompt, purpose, attempt))
        return '{"value":"ok"}'

    def invalidate(prompt, purpose, attempt, error):
        client.cache.pop(client._key(prompt, purpose, attempt), None)

    monkeypatch.setattr(client, "complete", complete)
    monkeypatch.setattr(client, "_invalidate", invalidate)

    assert client.call_json("prompt", "test", lambda value: value["value"]) == "ok"
    assert fresh_calls == [("prompt", "test", 0)]



def test_api_key_rejects_non_ascii_placeholder(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "example-non-ascii-密钥")
    with pytest.raises(LLMError, match="non-ASCII"):
        read_api_key({"api_key_env": "DEEPSEEK_API_KEY"})


def test_api_key_rejects_whitespace(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-invalid key")
    with pytest.raises(LLMError, match="whitespace"):
        read_api_key({"api_key_env": "DEEPSEEK_API_KEY"})


def test_api_key_accepts_ascii_value(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-value")
    assert read_api_key({"api_key_env": "DEEPSEEK_API_KEY"}) == "sk-test-value"


def test_complete_disables_thinking_when_configured(tmp_path):
    class Completions:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            message = type("Message", (), {"content": '{"value":"ok"}'})()
            choice = type("Choice", (), {"message": message})()
            usage = type(
                "Usage",
                (),
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )()
            return type("Response", (), {"choices": [choice], "usage": usage})()

    client = object.__new__(LLMClient)
    client.settings = {
        "model": "deepseek-flash",
        "temperature": 0,
        "max_tokens": 800,
        "thinking": False,
    }
    completions = Completions()
    chat = type("Chat", (), {"completions": completions})()
    client.client = type("Client", (), {"chat": chat})()
    client.cache_path = tmp_path / "cache.jsonl"
    client.cache = {}

    assert client.complete("prompt", "test") == '{"value":"ok"}'
    assert completions.request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_public_config_removes_inline_api_key():
    config = {
        "llm": {
            "model": "deepseek-flash",
            "api_key": "secret-value",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
        "_project_root": "/tmp/project",
    }
    result = public_config(config)
    assert "api_key" not in result["llm"]
    assert result["llm"]["api_key_env"] == "DEEPSEEK_API_KEY"
