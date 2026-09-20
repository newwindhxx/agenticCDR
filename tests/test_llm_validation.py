from __future__ import annotations

import pytest

from agentic_cdr.evaluation import _ranking_validator
from agentic_cdr.llm import LLMClient, extract_json_object


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
