"""Adapter for a judge engine that lives inside the trainer process (e.g. on the trainer's GPUs).

The trainer integration supplies the engine; this module has no trainer-specific code. An engine is
either an async callable ``generate(call: BackendCall) -> BackendResponse | Mapping`` or an object
with such an async ``generate`` method and optional async ``wake_up()`` / ``sleep()`` methods (for
engines that release GPU memory between phases).

While the backend is asleep, calls wait (the client's deadlines still apply); the trainer wakes
the engine when it wants labels, awaits the session barrier, and puts it back to sleep. A mapping
response may use the keys ``text``, ``finish_reason``, ``prompt_tokens``, ``completion_tokens``,
``reasoning_tokens``, ``reasoning_text``, ``model``, ``refusal`` and ``raw``.

Engines can be registered by name so configs can refer to them before they exist::

    register_colocated_engine("rm", engine)      # at trainer setup
    {type: colocated, engine: rm}                # judge config
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol, Union, runtime_checkable

from judgerl.judge.backends.base import BaseBackend
from judgerl.judge.types import BackendCall, BackendError, BackendResponse, JudgeStatus


@runtime_checkable
class ColocatedEngine(Protocol):
    async def generate(self, call: BackendCall) -> Union[BackendResponse, Mapping[str, Any]]: ...


GenerateFn = Callable[[BackendCall], Awaitable[Union[BackendResponse, Mapping[str, Any]]]]

_ENGINES: Dict[str, Any] = {}


def register_colocated_engine(name: str, engine: Union[ColocatedEngine, GenerateFn]) -> None:
    _ENGINES[name] = engine


def unregister_colocated_engine(name: str) -> None:
    _ENGINES.pop(name, None)


class ColocatedBackend(BaseBackend):
    def __init__(self, name: str = "colocated", model: str = "colocated-judge", revision: Optional[str] = None,
                 structured: str = "none", reasoning_style: str = "chat_template",
                 engine: Union[str, ColocatedEngine, GenerateFn, None] = None, start_awake: Optional[bool] = None):
        super().__init__(name=name, model=model, revision=revision,
                         structured="none" if structured == "auto" else structured, reasoning_style=reasoning_style)
        self._engine_ref = engine
        self._engine: Any = None
        self._start_awake = start_awake
        self._attached: Optional[asyncio.Event] = None
        self._awake: Optional[asyncio.Event] = None

    def _events(self) -> None:
        if self._attached is None:
            self._attached = asyncio.Event()
            self._awake = asyncio.Event()

    def attach(self, engine: Union[ColocatedEngine, GenerateFn], awake: Optional[bool] = None) -> None:
        """Bind (or replace) the engine at runtime. Must be called from the client's event loop."""
        self._events()
        self._engine = engine
        has_sleep = hasattr(engine, "sleep") or hasattr(engine, "wake_up")
        start_awake = awake if awake is not None else (self._start_awake if self._start_awake is not None
                                                       else not has_sleep)
        if start_awake:
            self._awake.set()
        else:
            self._awake.clear()
        self._attached.set()

    def _resolve(self) -> None:
        if self._engine is not None or self._engine_ref is None:
            return
        ref = self._engine_ref
        if isinstance(ref, str):
            if ref in _ENGINES:
                ref = _ENGINES[ref]
            elif ":" in ref:
                from judgerl.judge.backends.registry import import_object

                ref = import_object(ref)
            else:
                return  # not registered yet; generate() waits for attach()
        self.attach(ref)

    async def start(self) -> None:
        self._events()
        self._resolve()

    @property
    def is_awake(self) -> bool:
        return self._awake is not None and self._awake.is_set()

    async def wake(self) -> None:
        self._events()
        self._resolve()
        f = getattr(self._engine, "wake_up", None)
        if f is not None:
            r = f()
            if inspect.isawaitable(r):
                await r
        self._awake.set()

    async def sleep(self) -> None:
        self._events()
        self._awake.clear()
        f = getattr(self._engine, "sleep", None)
        if f is not None:
            r = f()
            if inspect.isawaitable(r):
                await r

    async def generate(self, call: BackendCall) -> BackendResponse:
        self._events()
        self._resolve()
        if not self._attached.is_set():
            await self._attached.wait()
        if not self._awake.is_set():
            await self._awake.wait()
        eng = self._engine
        fn = getattr(eng, "generate", None) or eng
        if not callable(fn):
            raise BackendError(JudgeStatus.BACKEND_ERROR, "colocated engine is not callable", retryable=False)
        out = fn(call)
        if inspect.isawaitable(out):
            out = await out
        if isinstance(out, BackendResponse):
            return out
        if isinstance(out, Mapping):
            return BackendResponse.from_mapping(out)
        if isinstance(out, str):
            return BackendResponse(text=out, finish_reason="stop", model=self.model_id)
        raise BackendError(JudgeStatus.BACKEND_ERROR, f"colocated engine returned {type(out).__name__}",
                           retryable=False)
