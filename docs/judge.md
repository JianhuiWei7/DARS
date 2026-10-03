# Judge control plane (`judgerl.judge`)

One async client for LLM judges, whether the judge is an API model, a self-hosted server, any
Hugging Face model id, a small in-process model or an engine that lives inside the trainer. The
client queues requests, applies rate limits, retries and fails over, parses and validates structured
output, caches results durably, and enforces budgets and circuit breakers. Every request ends with a
typed status, and every status is counted.

Importing `judgerl.judge` does not import torch, transformers, a trainer, LiteLLM or the OpenAI
SDK; each backend imports what it needs when it is first used.

## Quick start

```python
from judgerl.judge import JudgeClient, JudgeRequest

client = JudgeClient({
    "backend": {"type": "litellm", "model": "deepseek/deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
    "budget_usd": 20,
    "cache": {"path": "outputs/judge_cache.sqlite"},
})

schema = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}},
          "required": ["score"]}

async def main():
    async with client:
        print(await client.preflight(strict=True))      # one real request per backend
        r = await client.judge(JudgeRequest(messages=[{"role": "user", "content": "Rate: ..."}],
                                            output_schema=schema, tags={"channel": "outcome"}))
        if r.ok:
            print(r.parsed["score"])
        else:
            print(r.status, r.error)                     # e.g. truncated, malformed, rate_limited
        print(client.metrics())                          # {"judge/ok": 1, "judge/cost_usd": ..., ...}
```

From synchronous code, use `SyncJudgeClient(config)`; it runs the same client on a private event
loop thread and exposes `judge`, `judge_many`, `submit` (a `concurrent.futures.Future`),
`session()`, `preflight`, `metrics` and `close`.

## Requests and results

`JudgeRequest` fields:

| field | meaning |
|---|---|
| `messages` | OpenAI-style chat messages |
| `output_schema` | JSON Schema dict, a pydantic model class, or `None` for a free-text answer |
| `validator` | optional `parsed -> str or None`; a string rejects the answer and is sent back as the corrective message |
| `decoding` | `max_tokens` (the **answer** budget), `temperature`, `top_p`, `stop`, `seed` |
| `reasoning` | `"default"`, `"off"`, `"on"`, an int budget, or `ReasoningSpec` |
| `tags` | free-form labels for cost/metric attribution (`channel`, `domain`, `step`, ...) |
| `priority` | higher is dispatched first |
| `timeout_s` / `deadline_s` | per attempt / whole request from submission (queueing, retries and backoff) |
| `pool` | named failover pool (default: all backends in config order) |
| `prompt_version`, `schema_version`, `parser_version`, `policy_version` | provenance; the first three are part of the cache key |
| `sample_index` | distinguishes deliberate repeated draws of one prompt (cache and single-flight) |
| `use_cache` | `False` bypasses the cache and single-flight |
| `extra_params` | extra provider parameters (request invariants still win) |

`JudgeResult` has `status`, `parsed` (dict, pydantic instance or text), `text`, `reasoning_text`,
`finish_reason`, `usage` (prompt, completion and reasoning tokens summed over attempts), `cost_usd`,
`latency_s`, `queue_s`, `attempts`, `attempt_log`, `backend`, `model`, `error`, `cache_hit`,
`coalesced`, `lenient_parse`, `reasks`, `failovers`, `cache_key`, `tags`, `versions` and `raw` (the
provider response, for audits).

Statuses: `ok`, `timeout`, `rate_limited`, `refusal`, `empty_content`, `truncated`, `malformed`,
`semantic_reject`, `auth`, `quota`, `backend_error`, `cancelled`, `budget_exceeded`,
`circuit_open`. A result is never silently replaced by a default: callers decide what a missing
label means (and the count of each status is in the metrics).

## Backends

Each config entry is one backend. Common keys: `type`, `model`, `name` (defaults to the model),
`revision`, `rate_limit`, `breaker`, `prices`, `structured`, `reasoning_style`, `reasoning_budget`,
`timeout_s`. Any other key goes to the backend constructor, which rejects keys it does not know.

### Any API model (`litellm`)

```yaml
backends:
  - {type: litellm, model: deepseek/deepseek-chat, api_key_env: DEEPSEEK_API_KEY}
  - {type: litellm, model: anthropic/claude-sonnet-4-5, api_key_env: ANTHROPIC_API_KEY}
  - {type: litellm, model: openai/gpt-4.1-mini}          # key from LiteLLM's usual variables
  - {type: litellm, model: gemini/gemini-2.5-flash, rate_limit: {rpm: 1000, tpm: 2000000}}
```

Options: `api_base`, `api_key`, `api_key_env`, `extra_kwargs`, `extra_body`, `strict_schema`,
`drop_params` (default true: LiteLLM drops parameters a provider does not accept; the preflight
shows what actually happens). If `api_key_env` names an empty variable, requests fail with `auth`
without being sent, and the preflight says so.

### A self-hosted server (`openai_compat`)

```yaml
backends:
  - type: openai_compat
    model: Qwen/Qwen3-8B                      # the served model name
    base_urls: ["http://10.0.0.5:8000", "http://10.0.0.6:8000"]
    structured: json_schema                   # or guided_json (vLLM extra_body), or none
```

Requests go to the replica with the fewest outstanding requests; a replica that fails at the
connection level is skipped for `unhealthy_cooldown_s`. The OpenAI SDK client is built with
`max_retries=0`.

### Any Hugging Face chat model (`hf_server`)

```yaml
backend: {type: hf_server, model: Qwen/Qwen3-8B, engine: vllm, gpus: "0"}
```

The client launches the server (vLLM or SGLang) as a subprocess when it starts, picks free ports,
sets `CUDA_VISIBLE_DEVICES`, waits until `/v1/models` lists the model (or fails with the log tail
if the process exits), routes through `openai_compat`, and stops the process group on `close()`.
Options: `engine` (`vllm` | `sglang`), `gpus` (`"0,1,2,3"` split by `tensor_parallel_size` into
replicas, or an explicit list of groups such as `["0", "1"]`), `tensor_parallel_size`, `port`,
`host`, `served_model_name`, `dtype`, `max_model_len`, `gpu_memory_utilization`,
`reasoning_parser` (e.g. `qwen3`), `trust_remote_code`, `extra_args`, `python` (an interpreter
with the engine installed, if it is not this one), `env`, `log_dir`, `ready_timeout_s`.

The same machinery is available directly:

```python
from judgerl.judge.backends.hf_server import launch_hf_server
server, backend = launch_hf_server("Qwen/Qwen3-8B", engine="sglang", gpus="0,1", tensor_parallel_size=1)
client = JudgeClient.from_backends(backend)
...
server.stop()
```

### Small in-process judges (`hf_local`)

```yaml
backend: {type: hf_local, model: Qwen/Qwen3-0.6B, device: "cuda:0", max_batch_size: 16}
```

Loads with `transformers` on first use, micro-batches concurrent requests (`max_batch_size`,
`batch_wait_s`), renders prompts with the chat template and passes `enable_thinking` from the
request's reasoning setting (or a fixed `enable_thinking` option). There is no guided decoding: the
schema is appended to the prompt and the answer is parsed and validated.

### Colocated with the trainer (`colocated`)

```yaml
backend: {type: colocated, name: rm, engine: rm}
```

The trainer supplies the engine: an async callable `generate(call: BackendCall)` returning a
`BackendResponse` (or a dict with `text`, `finish_reason`, `prompt_tokens`, `completion_tokens`,
...) or an object with that `generate` method and optional async `wake_up()` / `sleep()`. Bind it
with `register_colocated_engine("rm", engine)` before the client starts, or later with
`client.attach_colocated("rm", engine)`. Requests submitted while the engine is asleep (or not yet
attached) wait in the queue; `await client.backend("rm").wake()` releases them and
`await client.backend("rm").sleep()` puts the engine back to sleep.

### A judge served by verl (`verl://reward_model`)

With the verl backend, verl itself can serve a Hugging Face model as the judge
(`reward.reward_model.enable=True reward.reward_model.model_path=<HF id>`). Point an `openai_compat`
backend at it with the placeholder URL; the model name defaults to the served model:

```yaml
backend: {type: openai_compat, base_url: verl://reward_model, structured: json_object}
```

* Colocated (default, `enable_resource_pool=False`): the judge shares the trainer's GPUs and sleeps
  during rollout and training. Set `judgerl.reward_stage=trainer`: the trainer wakes the judge after
  each rollout, runs the per-episode and group reward programs, and puts it back to sleep.
* Own GPUs (`enable_resource_pool=True`, `n_gpus_per_node`, `nnodes`): the judge runs alongside
  rollout, and episodes can be judged as they finish (`judgerl.reward_stage=rollout`).

The recipes set all of this with `COLOCATED_JUDGE=<HF id>`.

### Tests and dry runs (`scripted`)

Deterministic outputs (`responder`, `responses`, `default_output`) and fault injection:
`faults: {timeout: 0.05, rate_limited: 0.1, malformed: 0.05, hang: 0.01, ...}` (random per request
and attempt, reproducible with `seed`) or `sequence: [rate_limited, ok]` (status of attempt k).

### Offline batches

```python
from judgerl.judge.backends.batch import BatchJob
job = BatchJob(client, "outputs/labels.jsonl", concurrency=32)   # or BatchJob(backend, ...)
job.extend(requests)                                           # give them stable request_id values
summary = await job.run()                                      # rerun later: finished ids are skipped
```

### Custom backends

Implement `start()`, `generate(call) -> BackendResponse` (one attempt; raise `BackendError(status,
message, retry_after=...)` for typed failures) and `close()`, set `name`, `model_id`, `revision`
and `capabilities`, then `register_backend("mine", MyBackend)` or use `type: "package.module:Class"`.

## Structured output and reasoning

- **Schema first.** If the backend enforces schemas (`structured: json_schema` as OpenAI-style
  `response_format`, or `guided_json` for vLLM), the schema is sent natively. Otherwise the schema
  is appended to the last user message (`structured.schema_instruction: auto | always | never`).
- **Parsing.** The whole answer must be one JSON value (a `<think>` block and a single code fence
  are stripped). If that fails, a lenient extractor looks for an embedded object; results recovered
  that way have `lenient_parse=True` and are counted in `judge/lenient_parse`. Disable with
  `structured.lenient_fallback: false`.
- **Validation.** Parsed output is validated against the schema (`jsonschema`, or the pydantic
  model), then by the request's `validator`. A rejection is `semantic_reject`; with
  `structured.reask: true` the retry appends the previous answer and a corrective message.
- **Reasoning.** `reasoning: off | on | <budget> | default` is translated per backend
  (`reasoning_style`: `anthropic` thinking budget, `openai_effort` reasoning effort, `gemini`
  thinking budget, `chat_template` `enable_thinking`, or `none`; `auto` infers it from the model
  string). The reasoning budget is **added** to `max_tokens`, so reasoning cannot consume the answer
  budget. For models that always reason, set `reasoning_budget` on the backend. An answer cut off
  inside its reasoning is `truncated`, not `empty_content`.

## Scheduling, retries and failover

```yaml
queue: {max_queue: 10000, max_inflight: 64}
retry:
  max_attempts: 6               # over all backends
  attempts_per_backend: 2       # then fail over to the next backend of the pool
  base_delay_s: 1.0             # exponential backoff with jitter
  max_retry_after_s: 120        # provider retry-after is honored up to this cap
  retry_on: [timeout, rate_limited, backend_error, empty_content, malformed, truncated, semantic_reject]
  truncation_growth: 1.0        # e.g. 2.0 doubles the answer budget after a truncation
pools: {train: [deepseek/deepseek-chat, Qwen/Qwen3-8B], eval: [anthropic/claude-sonnet-4-5]}
defaults: {max_tokens: 1024, temperature: 0.0, reasoning: "off", timeout_s: 120}
```

`submit()` waits while the queue is full; if the client closes or halts meanwhile, the waiting
producer is woken at once (its request resolves as `cancelled`, or `submit()` raises `JudgeHalted`). Higher `priority` goes first. `client.cancel(request_id)`
(or cancelling the returned future) resolves the request as `cancelled`. `auth` and `quota`
disable a backend for the rest of the request and fail over immediately. The SDKs never retry on
their own: LiteLLM gets `num_retries=0, max_retries=0`, the OpenAI client `max_retries=0`, and
`stream=False`, `n=1`, `model` and `messages` always come from the client, whatever `extra_kwargs`
say.

Per-backend admission: `rate_limit: {rpm, tpm, max_concurrency, burst_s}`. Token limits reserve
an estimate (prompt + completion cap) and reconcile with the reported usage; a `retry-after` pauses
the whole backend.

## Circuit breakers and budgets

```yaml
breaker:                          # global, over final results
  window_s: 120
  min_calls: 50
  failure_rate: 0.5
  consecutive_failures: 25
  cooldown_s: 30
  policy: pause                   # pause: hold the queue, probe after cooldown | stop: raise JudgeHalted
budget_usd: 50                    # hard cap for the run
budget_policy: stop               # stop: raise JudgeHalted | pause: remaining requests get budget_exceeded
backends:
  - type: litellm
    model: deepseek/deepseek-chat
    prices: {input_per_mtok: 0.27, output_per_mtok: 1.10}
    breaker: {consecutive_failures: 10, cooldown_s: 30, cost_budget_usd: 30}
```

Each backend has its own breaker (per attempt; `auth`/`quota` trip it immediately). A backend's
`cost_budget_usd` is a hard cap enforced like the run budget: each call atomically reserves its
worst-case cost before dispatch (the priced estimate, else the largest cost observed for one call;
while neither is known, calls to that backend go one at a time), and once the cap cannot cover the
next call the backend's breaker trips for good and the pool fails over. an open backend
breaker makes the pool fail over, and when no backend can serve a request it waits for a probe
(`pause`) or ends as `circuit_open` (`stop`). With `policy: stop` on the global breaker, a trip
raises `JudgeHalted` from every unresolved future, from `submit()` and from `session.barrier()`.

Budgets: before each attempt the client reserves its worst-case cost (estimated prompt tokens times
a per-backend calibration factor learned from reported usage, plus the full completion cap) and
refuses the attempt if it could cross `budget_usd`; in-flight reservations are settled with actual
cost. With known prices the budget is not overrun; when a price is unknown the cap is enforced
after the fact (the preflight warns). Cost comes from configured `prices`, else the backend's
reported cost (LiteLLM cost tables), else LiteLLM's table for the model id, else it is counted as
unknown (`judge/cost_unknown`).

`client.estimate_cost(requests)` prices a planned workload before launch. Use it for prefix-style
judges (one call per step over growing prefixes), whose prompt volume grows quadratically with
trajectory length.

## Cache

```yaml
cache: {path: outputs/judge_cache.sqlite, ttl_s: null, cross_process: true, lease_ttl_s: 300}
redact: mypkg.judging:strip_hidden_fields   # JudgeRequest -> JudgeRequest, applied before hashing and sending
```

- Key: SHA-256 of canonical JSON over messages (after redaction), output schema, prompt / schema /
  parser versions, backend model id and revision, decoding, reasoning, structured-output mode,
  `sample_index` and `extra_params`. The rollout `policy_version`, tags and request id are not in
  the key.
- Only `ok` results are stored, with the raw provider response. The cache is one SQLite file (WAL)
  and survives restarts.
- Single-flight: identical requests in one process share one call (`coalesced=True`; a failure is
  shared too, it is not retried by each follower); across processes a lease row makes one process
  call the model while the others wait for its entry (or take over if the lease is released or
  expires). Each lease carries a fencing token from a monotonic counter; renewal (every
  `lease_ttl_s / 3`), the result write and the release all require the matching owner and token, so
  a process paused past its TTL that lost the lease cannot overwrite the newer result. It still
  returns its own result, without caching it, and `judge/event/cache_lease_lost` counts this.
- Sampling judges (temperature > 0) return the cached sample for an identical request; use
  `sample_index` to request distinct draws, or `use_cache=False`.

## Sessions and reward watermarks

```python
session = client.session(default_tags={"step": step})
for traj in rollouts_as_they_finish():
    session.submit(build_request(traj))          # returns a future at once
report = await session.barrier(watermark=1.0, timeout=600)
rewards = {rid: r.parsed for rid, r in report.results.items() if r.ok}
logger.log({**client.metrics(window=True, reset_window=True), **report.metrics()})
```

`barrier` returns when `ceil(watermark * n)` of the requests submitted since the previous barrier
have resolved (success or typed failure), or at `timeout`. Unresolved requests are then cancelled
and listed in `report.cancelled_ids` (`cancel_unresolved=False` keeps them running). A request
counts toward the barrier from the moment `submit` / `asubmit` is called, including while it is
still waiting for queue space. Use
`await session.asubmit(request)` to apply queue backpressure to the producer.

## Telemetry

`client.metrics(window=False, reset_window=False)` returns flat keys for trainer loggers:
`judge/total`, one key per status (`judge/ok`, `judge/truncated`, ...), `judge/ok_rate`,
`judge/cost_usd`, `judge/cost_per_label_usd`, `judge/{prompt,completion,reasoning}_tokens`,
`judge/cache_hit`, `judge/coalesced`, `judge/lenient_parse`, `judge/reask`, `judge/retries`,
`judge/failover`, `judge/p50_latency_s` / `p90` / `p99`, `judge/attempt/<status>`, gauges
(`judge/queue_depth`, `judge/inflight`, `judge/spent_usd`, `judge/budget_remaining_usd`,
`judge/breaker_open`, `judge/halted`), and per-backend (`judge/backend/<name>/...`) and per-tag
(`judge/tag/<key>=<value>/{total,ok,cost_usd,...}`) breakdowns. `window=True` covers only results
since the last `reset_window=True`, which gives per-step numbers.

`telemetry: {jsonl_path: outputs/judge_calls.jsonl, log_raw: false}` appends every request and its
result (including the attempt log) as one JSON line.

## Preflight

`await client.preflight(strict=True)` sends one tiny structured request to every backend, bypassing
the cache and breakers, and reports per backend: status, latency, cost, whether the cost is known,
and problems such as a missing key (`auth`), an unknown model, quota, or an answer that fails parsing
(usually a `structured` / `reasoning` / `max_tokens` setting). With `strict=True` it raises
`PreflightError` listing every problem. Call it at launch, before the first rollout.

## Trainer integration notes

- Create the client on the trainer's event loop (or use `SyncJudgeClient`) and `await
  client.start()` once; `close()` at shutdown stops managed servers.
- Submit a trajectory's requests as soon as it finishes, and call `session.barrier(...)` before the
  optimizer step, so judging overlaps rollout without training on partial rewards.
- For a judge on the trainer's GPUs, register the engine for a `colocated` backend; wake it before
  the barrier and put it to sleep afterwards.
- Log `client.metrics(window=True, reset_window=True)` and `report.metrics()` every step. A step
  whose rewards are missing labels shows it in `judge/<status>` counts rather than as silently
  substituted values.
