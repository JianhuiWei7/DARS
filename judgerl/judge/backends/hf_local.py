"""In-process Hugging Face ``transformers`` judge (small models, no server).

``torch`` and ``transformers`` are imported lazily on first use, so this module imports without
them. Concurrent requests are micro-batched (up to ``max_batch_size``, waiting ``batch_wait_s`` for
more) and grouped by decoding parameters; one batch runs at a time in a worker thread. Prompts are
rendered with the tokenizer's chat template; ``enable_thinking`` follows the request's reasoning
setting unless fixed in the config. There is no guided decoding: the client appends the schema
instruction and parses the answer.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from judgerl.judge.backends.base import BaseBackend
from judgerl.judge.schema import split_reasoning
from judgerl.judge.types import BackendCall, BackendError, BackendResponse, JudgeStatus, Usage


@dataclass
class _Item:
    call: BackendCall
    future: asyncio.Future


class HFLocalBackend(BaseBackend):
    def __init__(self, model: str, name: Optional[str] = None, revision: Optional[str] = None,
                 structured: str = "none", reasoning_style: str = "chat_template", device: str = "auto",
                 dtype: str = "auto", max_batch_size: int = 8, batch_wait_s: float = 0.01,
                 trust_remote_code: bool = False, enable_thinking: Optional[bool] = None,
                 chat_template_kwargs: Optional[Dict[str, Any]] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None):
        if structured not in ("none", "auto"):
            raise ValueError("hf_local has no guided decoding; use structured: none (instruct + parse)")
        super().__init__(name=name or model, model=model, revision=revision, structured="none",
                         reasoning_style=reasoning_style)
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_wait_s = batch_wait_s
        self.trust_remote_code = trust_remote_code
        self.enable_thinking = enable_thinking
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.model_kwargs = dict(model_kwargs or {})
        self.model = None
        self.tokenizer = None
        self._torch = None
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None
        self.batches_run = 0
        self.batch_sizes: List[int] = []

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """Load tokenizer and model (blocking; called in a worker thread)."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision,
                                            trust_remote_code=self.trust_remote_code)
        tok.padding_side = "left"
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        kw: Dict[str, Any] = dict(self.model_kwargs)
        if self.dtype == "auto":
            kw.setdefault("torch_dtype", "auto")
        else:
            kw.setdefault("torch_dtype", getattr(torch, self.dtype))
        if self.device == "auto":
            kw.setdefault("device_map", "auto")
        model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision,
                                                     trust_remote_code=self.trust_remote_code, **kw)
        if self.device != "auto":
            model = model.to(self.device)
        model.eval()
        self._torch, self.tokenizer, self.model = torch, tok, model

    async def start(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._worker = asyncio.create_task(self._batch_loop(), name=f"hf_local[{self.name}]")

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._worker = None
        if self._queue is not None:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(BackendError(JudgeStatus.BACKEND_ERROR, "backend closed"))
            self._queue = None

    # ------------------------------------------------------------------ rendering / generation
    def _thinking_flag(self, call: BackendCall) -> Optional[bool]:
        if self.enable_thinking is not None:
            return self.enable_thinking
        if self.capabilities.reasoning_style != "chat_template" or call.reasoning.mode == "default":
            return None
        return call.reasoning.enabled

    def render(self, messages: List[Dict[str, Any]], enable_thinking: Optional[bool]) -> str:
        kw: Dict[str, Any] = dict(tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **kw)
        except TypeError:  # template/tokenizer without the flag
            kw.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kw)

    def _eos_ids(self) -> set:
        ids = set()
        for src in (getattr(self.tokenizer, "eos_token_id", None),
                    getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)):
            if isinstance(src, int):
                ids.add(src)
            elif isinstance(src, (list, tuple)):
                ids.update(int(x) for x in src)
        return ids

    def generate_batch(self, calls: List[BackendCall]) -> List[BackendResponse]:
        """Run one batch (all calls share decoding parameters). Blocking."""
        self.load()
        torch, tok, model = self._torch, self.tokenizer, self.model
        c0 = calls[0]
        prompts = [self.render(c.messages, self._thinking_flag(c)) for c in calls]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        device = getattr(model, "device", None)
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        gen: Dict[str, Any] = {"max_new_tokens": c0.max_tokens, "pad_token_id": tok.pad_token_id}
        temp = c0.temperature if c0.temperature is not None else 0.0
        if temp > 0:
            gen.update(do_sample=True, temperature=temp)
            if c0.top_p is not None:
                gen["top_p"] = c0.top_p
        else:
            gen["do_sample"] = False
        if c0.seed is not None:
            torch.manual_seed(c0.seed)
        with torch.inference_mode():
            out = model.generate(**enc, **gen)
        plen = enc["input_ids"].shape[1]
        eos = self._eos_ids()
        pad = tok.pad_token_id
        results = []
        for i, call in enumerate(calls):
            ids = list(out[i].tolist())[plen:]
            finished = False
            for j, t in enumerate(ids):
                if t in eos:
                    ids, finished = ids[:j], True
                    break
            n_gen = len(ids) + (1 if finished else 0)
            while ids and pad is not None and ids[-1] == pad and pad not in eos:
                ids.pop()
            text = tok.decode(ids, skip_special_tokens=True)
            finish = "stop" if finished or n_gen < call.max_tokens else "length"
            for s in call.stop or ():
                k = text.find(s)
                if k >= 0:
                    text, finish = text[:k], "stop"
            answer, reasoning = split_reasoning(text)
            prompt_tokens = int(sum(enc["attention_mask"][i].tolist())) if "attention_mask" in enc else plen
            results.append(BackendResponse(text=answer, reasoning_text=reasoning, finish_reason=finish,
                                           usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=n_gen),
                                           model=self.model_id, cost_usd=0.0))
        return results

    # ------------------------------------------------------------------ batching loop
    @staticmethod
    def _group_key(call: BackendCall) -> Tuple:
        return (call.max_tokens, call.temperature, call.top_p, call.seed, tuple(call.stop or ()))

    async def _batch_loop(self) -> None:
        assert self._queue is not None
        while True:
            first: _Item = await self._queue.get()
            items = [first]
            loop = asyncio.get_running_loop()
            end = loop.time() + self.batch_wait_s
            while len(items) < self.max_batch_size:
                timeout = end - loop.time()
                if timeout <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self._queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            items = [it for it in items if not it.future.done()]
            groups: Dict[Tuple, List[_Item]] = {}
            for it in items:
                groups.setdefault(self._group_key(it.call), []).append(it)
            for group in groups.values():
                try:
                    outs = await asyncio.to_thread(self.generate_batch, [it.call for it in group])
                except Exception as e:  # noqa: BLE001 - OOM, bad model id, ...
                    err = e if isinstance(e, BackendError) else BackendError(
                        JudgeStatus.BACKEND_ERROR, f"{type(e).__name__}: {e}"[:500], retryable=True)
                    for it in group:
                        if not it.future.done():
                            it.future.set_exception(err)
                    continue
                self.batches_run += 1
                self.batch_sizes.append(len(group))
                for it, out in zip(group, outs):
                    if not it.future.done():
                        it.future.set_result(out)

    async def generate(self, call: BackendCall) -> BackendResponse:
        await self.start()
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put(_Item(call, fut))
        return await fut
