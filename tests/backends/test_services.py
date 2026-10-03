import pytest

from judgerl.backends.verl import services


def test_resolve_replaces_placeholder_and_model(monkeypatch):
    monkeypatch.setattr(services, "lookup", lambda name: {"url": "http://10.0.0.1:9000", "model": "Qwen/Qwen3-8B"})
    cfg = {"backend": {"type": "openai_compat", "base_url": "verl://reward_model"},
           "backends": [{"type": "litellm", "model": "openai/gpt-4.1-mini"}], "budget_usd": 5}
    out = services.resolve_judge_config(cfg)
    assert out["backend"] == {"type": "openai_compat", "base_url": "http://10.0.0.1:9000", "model": "Qwen/Qwen3-8B"}
    assert out["backends"][0]["model"] == "openai/gpt-4.1-mini"          # other backends untouched
    assert cfg["backend"]["base_url"] == "verl://reward_model"           # input not mutated


def test_resolve_keeps_explicit_model_and_passes_through(monkeypatch):
    monkeypatch.setattr(services, "lookup", lambda name: {"url": "http://h:1", "model": "m"})
    out = services.resolve_judge_config({"backend": {"type": "openai_compat", "base_urls": ["verl://reward_model"],
                                                     "model": "served-name"}})
    assert out["backend"]["base_urls"] == ["http://h:1"] and out["backend"]["model"] == "served-name"
    plain = {"backend": {"type": "litellm", "model": "x"}}
    assert services.resolve_judge_config(plain) is plain


def test_resolve_without_reward_model_explains(monkeypatch):
    monkeypatch.setattr(services, "lookup", lambda name: None)
    with pytest.raises(RuntimeError, match="reward.reward_model.enable"):
        services.resolve_judge_config({"backend": {"type": "openai_compat", "base_url": "verl://reward_model"}})
