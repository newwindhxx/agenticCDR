from __future__ import annotations

import pytest

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
    client.settings = {"max_retries": 3}
    responses = iter(["not json", '{"value":"ok"} trailing text'])
    monkeypatch.setattr(client, "complete", lambda prompt, purpose, attempt=0: next(responses))
    assert client.call_json("prompt", "test", lambda value: value["value"]) == "ok"



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
