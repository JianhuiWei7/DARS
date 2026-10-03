"""Helpers shared by the judge tests (imported by module name; tests/judge is on sys.path)."""
from __future__ import annotations

from judgerl.judge import JudgeClient, JudgeRequest

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}},
    "required": ["score"],
}

QUIET = {"enabled": False}


def req(content: str = "judge this", schema=SCORE_SCHEMA, **kw) -> JudgeRequest:
    return JudgeRequest(messages=[{"role": "user", "content": content}], output_schema=schema, **kw)


def make_client(*backends, backend_configs=None, **config) -> JudgeClient:
    """Client with breakers off and near-zero backoff unless the test overrides them."""
    config.setdefault("breaker", dict(QUIET))
    retry = {"base_delay_s": 0.001, "max_delay_s": 0.01, "jitter": 0.0}
    retry.update(config.pop("retry", {}))
    config["retry"] = retry
    bc = {b.name: {"breaker": dict(QUIET)} for b in backends}
    for name, spec in (backend_configs or {}).items():
        bc.setdefault(name, {}).update(spec)
    return JudgeClient.from_backends(*backends, backend_configs=bc, **config)
