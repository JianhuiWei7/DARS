"""String-named backend registry. Built-ins are registered lazily by import path so that selecting
``litellm`` never imports transformers, and ``hf_local`` never imports LiteLLM."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Union

Factory = Callable[..., Any]

_REGISTRY: Dict[str, Union[str, Factory]] = {
    "litellm": "judgerl.judge.backends.litellm_backend:LiteLLMBackend",
    "openai_compat": "judgerl.judge.backends.openai_compat:OpenAICompatBackend",
    "hf_local": "judgerl.judge.backends.hf_local:HFLocalBackend",
    "hf_server": "judgerl.judge.backends.hf_server:HFServerBackend",
    "colocated": "judgerl.judge.backends.colocated:ColocatedBackend",
    "scripted": "judgerl.judge.backends.scripted:ScriptedBackend",
}


def register_backend(name: str, factory: Union[str, Factory, None] = None):
    """Register a backend class/factory (or ``"module:attr"`` path). Usable as a decorator."""
    if factory is not None:
        _REGISTRY[name] = factory
        return factory

    def deco(f: Factory) -> Factory:
        _REGISTRY[name] = f
        return f

    return deco


def import_object(path: str) -> Any:
    """Import ``"package.module:attr"`` (or ``"package.module.attr"``)."""
    if ":" in path:
        mod, _, attr = path.partition(":")
    else:
        mod, _, attr = path.rpartition(".")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def get_backend_factory(name: str) -> Factory:
    if name not in _REGISTRY:
        if ":" in name:  # a custom backend given by import path
            return import_object(name)
        raise KeyError(f"unknown judge backend type {name!r}; known: {sorted(_REGISTRY)}")
    f = _REGISTRY[name]
    if isinstance(f, str):
        f = import_object(f)
        _REGISTRY[name] = f
    return f


def available_backends() -> list:
    return sorted(_REGISTRY)
