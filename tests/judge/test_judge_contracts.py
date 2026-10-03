"""Contracts: requests, results, statuses, reasoning/decoding specs, schema parsing and validation."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from judgerl.judge import (BackendResponse, Decoding, JudgeRequest, JudgeResult, JudgeStatus, ReasoningSpec,
                           Usage)
from judgerl.judge.schema import (extract_json, mini_validate, parse_strict, schema_to_dict, split_reasoning,
                                  validate, with_schema_instruction)
from judgerl.judge.types import ALL_STATUSES


def test_status_enum_has_every_typed_failure():
    expected = {"ok", "timeout", "rate_limited", "refusal", "empty_content", "truncated", "malformed",
                "semantic_reject", "auth", "quota", "backend_error", "cancelled", "budget_exceeded", "circuit_open"}
    assert {s.value for s in ALL_STATUSES} == expected
    assert JudgeStatus("truncated") is JudgeStatus.TRUNCATED


def test_request_validation_and_defaults():
    r = JudgeRequest(messages=[{"role": "user", "content": "x"}])
    assert len(r.request_id) == 32 and r.priority == 0 and r.reasoning.mode == "default"
    assert r.versions == {"prompt_version": "0", "schema_version": "0", "parser_version": "0", "policy_version": None}
    with pytest.raises(ValueError):
        JudgeRequest(messages=[])
    with pytest.raises(ValueError):
        JudgeRequest(messages=[{"content": "no role"}])
    with pytest.raises(ValueError):
        JudgeRequest(messages=[{"role": "user", "content": "x"}], timeout_s=0)
    r2 = JudgeRequest(messages=[{"role": "user", "content": "x"}], decoding={"max_tokens": 5, "stop": "END"},
                      reasoning=2048, policy_version=7, tags={"channel": "process"})
    assert r2.decoding.max_tokens == 5 and r2.decoding.stop == ("END",)
    assert r2.reasoning == ReasoningSpec("budget", 2048)
    d = r2.to_dict()
    assert d["policy_version"] == 7 and d["tags"] == {"channel": "process"}
    json.dumps(d)  # serializable


def test_reasoning_spec_parsing():
    assert ReasoningSpec.parse(None).mode == "default"
    assert ReasoningSpec.parse("off").mode == "off" and not ReasoningSpec.parse("off").enabled
    assert ReasoningSpec.parse(True).mode == "on" and ReasoningSpec.parse(True).enabled
    assert ReasoningSpec.parse(0).mode == "off"
    assert ReasoningSpec.parse("512") == ReasoningSpec("budget", 512)
    assert ReasoningSpec.parse({"mode": "budget", "budget_tokens": 3}).budget_tokens == 3
    with pytest.raises(ValueError):
        ReasoningSpec.parse("sometimes")
    with pytest.raises(ValueError):
        ReasoningSpec("budget")


def test_decoding_merge():
    d = Decoding(max_tokens=10).merged(Decoding(max_tokens=99, temperature=0.5))
    assert d.max_tokens == 10 and d.temperature == 0.5 and d.top_p is None


def test_result_serialization_with_pydantic_parsed():
    class M(BaseModel):
        score: int

    r = JudgeResult(request_id="a", status=JudgeStatus.OK, parsed=M(score=3), usage=Usage(1, 2, 0), raw={"x": 1})
    d = r.to_dict()
    assert d["status"] == "ok" and d["parsed"] == {"score": 3} and d["usage"]["completion_tokens"] == 2
    assert "raw" not in r.to_dict(include_raw=False)
    json.dumps(d)
    assert r.ok


def test_backend_response_from_mapping():
    r = BackendResponse.from_mapping({"text": "hi", "prompt_tokens": 3, "completion_tokens": 4})
    assert r.text == "hi" and r.usage.total_tokens == 7


# ----------------------------------------------------------------------------------------- schema
def test_parse_strict_and_lenient():
    assert parse_strict('{"a": 1}') == (True, {"a": 1})
    assert parse_strict('```json\n{"a": 1}\n```') == (True, {"a": 1})
    assert parse_strict('Sure! {"a": 1}')[0] is False
    assert extract_json('Sure! here: {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert extract_json('text ```json\n{"a": 2}\n``` more') == {"a": 2}
    assert extract_json("{broken {\"ok\": 1}") == {"ok": 1}
    assert extract_json("nothing") is None


def test_split_reasoning():
    assert split_reasoning("<think>hmm</think>\n{\"a\":1}") == ('{"a":1}', "hmm")
    assert split_reasoning("<think>never closed") == ("", "never closed")
    assert split_reasoning("plain") == ("plain", None)


def test_validate_json_schema_and_mini_validator():
    schema = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 0}, "tag": {"enum": ["a", "b"]}},
              "required": ["score"], "additionalProperties": False}
    assert validate(schema, {"score": 1})[1] == []
    assert validate(schema, {"score": -1})[1]
    assert validate(schema, {"tag": "a"})[1]
    for bad in ({"score": "1"}, {"score": 1, "tag": "c"}, {"score": 1, "x": 0}, [], {"score": True}):
        assert mini_validate(schema, bad), bad
    assert mini_validate(schema, {"score": 2, "tag": "b"}) == []
    assert mini_validate({"type": "array", "items": {"type": "number"}, "minItems": 1}, []) != []
    assert mini_validate({"anyOf": [{"type": "string"}, {"type": "null"}]}, None) == []


def test_validate_pydantic_model():
    class Verdict(BaseModel):
        score: int = Field(ge=0, le=10)
        reason: str = ""

    parsed, errs = validate(Verdict, {"score": 4})
    assert errs == [] and isinstance(parsed, Verdict) and parsed.score == 4
    parsed, errs = validate(Verdict, {"score": 42})
    assert parsed is None and errs and "score" in errs[0]
    assert schema_to_dict(Verdict)["properties"]["score"]["type"] == "integer"


def test_schema_instruction_appends_to_last_user_message():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out = with_schema_instruction(msgs, {"type": "object"})
    assert msgs[1]["content"] == "u"  # not mutated
    assert out[1]["content"].startswith("u") and "JSON Schema" in out[1]["content"]
    parts = with_schema_instruction([{"role": "user", "content": [{"type": "text", "text": "u"}]}], {"type": "object"})
    assert parts[0]["content"][-1]["type"] == "text"
