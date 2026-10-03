# Environments (`judgerl.envs`) and the code sandbox (`judgerl.sandbox`)

An environment turns a task into a sequence of per-step prompts, executes the policy's actions, and
reports rewards and flags. Judge RL trains one row per step, so what the environment puts into each
step's prompt and anchor directly shapes the training data.

Built-in environments:

| name | module | reward | needs |
|---|---|---|---|
| `numberline` | `judgerl.envs.toy` | 10 on reaching the target | nothing (smoke tests, CI) |
| `alfworld` | `judgerl.envs.alfworld` | 10 when the game is won | `alfworld` package and data |
| `python_tool` | `judgerl.envs.python_tool` | 10 for a verified answer | a sandbox backend |
| `open_ended` | `judgerl.envs.open_ended` | none: judge only | a judge reward program |

WebShop and Search are documented in [environments_webshop_search.md](environments_webshop_search.md).

## Writing an environment

### The contract

Subclass `judgerl.envs.base.Env`. **One instance plays one episode at a time**; the rollout loop
creates an instance per episode, calls `reset` once, then `step` until `done` or `max_steps`, then
`close`. All methods are synchronous: the rollout loop runs them off its event loop, in a thread
(`thread_safe = True`) or in a worker process of `judgerl.envs.pool` (the default).

```python
import re

from judgerl.envs.base import Env, Observation, StepResult, register_env


def parse_guess(action: str):
    m = re.search(r"<guess>\s*(\d+)\s*</guess>", action or "")
    return int(m.group(1)) if m else None


@register_env("guess")
class GuessEnv(Env):
    max_steps = 8
    thread_safe = True            # no module-level mutable state: episodes may share a process

    def reset(self, task, seed) -> Observation:
        self.secret, self.tries = int(task["secret"]), []
        return self._observe("Guess a number between 1 and 100.")

    def step(self, action) -> StepResult:
        guess = parse_guess(action)                      # None when the format is wrong
        self.tries.append(guess)
        won = guess == self.secret
        text = "correct" if won else ("higher" if guess and guess < self.secret else "lower")
        return StepResult(self._observe(text), 10.0 if won else 0.0, won,
                          {"won": won, "is_action_valid": guess is not None})

    def _observe(self, text) -> Observation:
        prompt = [{"role": "user", "content": f"{text}\nEarlier guesses: {self.tries}\nAnswer <guess>N</guess>."}]
        return Observation(prompt=prompt, anchor=f"tries={self.tries}", text=text)

    def task_text(self):
        return "guess the secret number"

    def metadata(self):
        return {"secret": self.secret}
```

| method / field | meaning |
|---|---|
| `reset(task, seed) -> Observation` | `task` is the dataset row's task dict (`extra_info.task`), `seed` its seed |
| `step(action) -> StepResult` | `action` is the decoded policy response for the current prompt |
| `Observation.prompt` | the chat messages the policy sees for the next step (built by the environment) |
| `Observation.anchor` | identity of the state the next action is taken from (see below) |
| `Observation.text` | the raw observation; judges see it as the step's state / result |
| `StepResult.reward`, `done` | reward of the action just taken; whether the episode ended |
| `StepResult.info` | `won` (success), `is_action_valid` (format), anything else for logging |
| `task_text()` | plain task description shown to judges |
| `metadata()` | hidden task data for judges and reward programs (gold answers, rubrics); JSON-serializable |
| `max_steps` | the rollout stops after this many actions |
| `thread_safe` | `True` only if instances keep no shared mutable global state |

The episode's success is the last step's `won`, its score the sum of the step rewards (a reward
program may replace either). `is_action_valid=False` feeds the optional invalid-action penalty.

### Anchors

GiGPO groups the steps of one task's rollouts by equal anchors and compares their returns, so the
anchor must identify **the state the policy acts from, and nothing else**:

* equal states must give equal anchors across rollouts: normalize away whitespace, reasoning text,
  step-local ids and timestamps. The tool environments hash the whitespace-normalized prompt
  messages (`judgerl.envs.python_tool.state_hash`), so any two rollouts that show the policy the same
  prompt share a step group;
* different states must give different anchors; if the prompt includes history, the anchor must
  cover that history too (otherwise the group mixes states the policy sees differently);
* anchors are strings; keep them short (hash long states).

### Metadata versus prompt

Anything the policy must not see (gold answers, rubrics, reference solutions, hidden goals) goes into
`metadata()` and never into `Observation.prompt`. Reward programs receive `task_text()` and the
`metadata()` captured at the end of the episode (`ascore(task=..., metadata=..., rows=...)`), so
put there everything a judge or verifier needs, including end-of-episode values (e.g. the final
response of `open_ended`). Metadata crosses process boundaries: keep it JSON-serializable.

### Task providers

A task is a plain dict; the dataset row carries it in `extra_info.task` and all rollouts of that row
form one group. An environment with a fixed task set provides a function

```python
def tasks(split: str, limit: int = 0, seed: int = 0) -> list[dict]: ...
```

that returns deterministic task dicts for a split (shuffle only with the given seed). The dataset
builder turns task dicts into verl rows:

```bash
python -m judgerl.backends.verl.data --env python_tool --tasks math_train.jsonl --train 5000 --val 500 --out data/pt
```

(`--tasks` takes a JSONL of task dicts; environments listed in the builder's `TASK_PROVIDERS` can
use their provider instead.)

### Registering

* In-tree or in your package: decorate the class with `@register_env("name")` and make sure the
  module is imported (built-ins are also listed in `judgerl.envs.base.BUILTIN_ENVS` and imported
  lazily on first use).
* From another package without importing it yourself: declare an entry point

  ```toml
  [project.entry-points."judgerl.envs"]
  guess = "my_pkg.envs:GuessEnv"
  ```

* Or refer to it by path anywhere an environment name is accepted: `env: my_pkg.envs:GuessEnv`.

`make_env(name, **kwargs)` resolves the name in that order (registry, entry points, import path).

## Built-in environments

### `numberline`

A tiny dependency-free environment for smoke tests and CI: stand at a position on a number line,
reach the target with `<action>left</action>` / `<action>right</action>`. States repeat across
rollouts, so GiGPO step groups are non-trivial; the anchor is `pos=..;target=..`.
Kwargs: `size`, `max_steps`, `history`. Tasks: `{"seed": i}` (or explicit `start` / `target`).

### `alfworld`

Text ALFWorld with verl-agent's prompts, history memory, action projection and rewards, so results
are comparable with published GiGPO numbers. Action = lower-cased text inside `<action>...</action>`,
valid only with a `<think>...</think>` block and no CJK characters; reward 10 when the game is won; the
anchor is the raw game observation. Tasks are game files (`judgerl.envs.alfworld.tasks(split)` with
splits `train`, `valid_seen`, `valid_unseen`, from `$ALFWORLD_DATA`). Needs `pip install alfworld`
and `alfworld-download`. Kwargs: `history_length` (2), `max_steps` (50). TextWorld keeps global state,
so ALFWorld episodes run in worker processes.

### `python_tool`

Math / QA with a Python interpreter, ReAct-style. Every turn the policy writes either

* `<python>code</python>`: the code runs in the sandbox (a fresh interpreter per call; variables do
  not persist, so the policy prints what it needs); the output (stdout, then stderr, then notes such
  as `[execution stopped: time limit exceeded]`, middle-clipped to `max_result_chars`) comes back as a
  `<result>...</result>` user message; or
* `<answer>...</answer>`: ends the episode; reward 10 if the answer verifies, else 0.

The first complete tag outside `<think>...</think>` counts. A response with neither tag (or empty
code) is an invalid action: `is_action_valid=False`, a format reminder is returned and the episode
continues. The episode ends with reward 0 after `max_steps` actions (the prompt announces the turns
left and asks for an answer on the last turn).

* **Verifier**: `verify_answer(pred, gold)` accepts the answer if either a normalized comparison
  (`\boxed{}`, `$`, spacing, `\dfrac`, `\text{}`, thousands separators, "the answer is", and numeric
  equality of integers, decimals and fractions) or the `math_verify` package
  (used when installed, with its signal-based timeouts disabled because environments run in threads)
  says they are equivalent.
* **Prompt**: a system message explaining the tags, the question, and the last `history` tool calls
  with their results as assistant/user turns (older calls are summarized as a count). Invalid
  responses appear in the history as a fixed placeholder, not verbatim.
* **Anchor**: hash of the whitespace-normalized prompt (question + visible history + turns left).
* **Metadata**: `{"id", "question", "answer", "reference"}`; the gold answer is never in the prompt.
* **Info**: `won`, `is_action_valid`, `action_type` (`python` / `answer` / `invalid`), and for tool
  calls `exit_code`, `timed_out`, `oom`, `truncated`, `exec_s`.
* **Tasks**: JSONL rows `{"question": ..., "answer": ...}` (`problem` / `gold` also accepted) via
  `tasks(split, limit, path=...)` or `$JUDGERL_PYTHON_TOOL_TASKS` (`{split}` in the path is replaced
  by the split); `tasks("sample")` returns a four-question built-in sample.

```yaml
env: python_tool
env_kwargs:
  max_steps: 6
  history: 4
  timeout_s: 10                     # per tool call
  max_result_chars: 2000
  sandbox: {backend: docker, image: python:3.11-slim, memory_mb: 1024, max_concurrency: 32}
```

The sandbox configuration is shared by all episodes of a process: `judgerl.sandbox.shared_sandbox`
keeps one pool per configuration, so `max_concurrency` bounds the programs running at once in that
process (default: the CPU count).

### `open_ended`

Judge-only instruction following, single- or multi-turn. A task is an instruction with a rubric of
weighted checklist items:

```json
{"id": "t1", "instruction": "Write a haiku about autumn rain.",
 "rubric": [{"item": "Exactly three lines", "weight": 1}, {"item": "Mentions rain", "weight": 3}],
 "followups": ["Make it sadder."], "max_turns": 2}
```

(`rubric` may also be a list of strings, weight 1 each.) The environment gives **reward 0** at every
step. The policy puts its final response in `<final>...</final>`, which ends the episode; otherwise
the environment answers with the next scripted follow-up, or asks it to continue. After `max_steps`
turns (a task's `max_turns` overrides it; `max_steps=1` is single-turn) the last response is taken as
final and that step is flagged invalid. The rubric and, at the end, the final response are in
`metadata()` (`{"id", "instruction", "rubric", "final_response"}`), never in the prompt. The anchor is
the hash of the normalized visible conversation (`<think>` blocks are dropped from the history).
Tasks: JSONL via `tasks(split, path=...)`, or `tasks("sample")`.

**It must be trained with a judge reward program**; with none, every episode scores 0. The rubric
lives in each task, so use `RubricOutcome` (in `judgerl.envs.open_ended`, built on the `outcome`
program), which reads `metadata["rubric"]` for every episode. Reward program types are resolved by
import path, so no registration beyond the config is needed:

```yaml
reward_program:
  type: judgerl.envs.open_ended:RubricOutcome
  mode: checklist                 # or "score": one holistic score with the rubric as guidance
  scale: 10.0
  judge: {backend: {type: litellm, model: <provider/model>}, cache: {path: outputs/judge_cache.sqlite}}
```

Validation episodes are only judged with `judgerl.judge_val=true` (the `open_ended` recipe sets it);
otherwise validation of this environment reports 0.

In `checklist` mode the judge returns one boolean per rubric item (the schema pins the length) and
the score is the weighted fraction of satisfied items, times `scale`. The score becomes the episode
score and is added to the final step's reward. The judge sees the task, the conversation and the
final response as quoted data, but not the policy's `<think>` blocks (`show_reasoning: true` changes
that) or an environment outcome (`show_outcome`). A judge failure leaves the episode unlabelled
(`status="failed"`) and is counted in the judge metrics; it never becomes a silent 0.

## The code sandbox (`judgerl.sandbox`)

```python
from judgerl.sandbox import make_sandbox

sandbox = make_sandbox({"backend": "process", "timeout_s": 5, "memory_mb": 512, "max_concurrency": 8})
r = sandbox.run("import sys; print(sum(map(int, sys.stdin.read().split())))", stdin="1 2 3",
                files={"data/input.csv": "a,b\n1,2\n"})
r.stdout, r.stderr, r.exit_code, r.timed_out, r.oom, r.truncated, r.duration_s
```

`run(code, stdin="", timeout_s=None, files=None)` writes `code` as `main.py` and the `files` (relative
paths only; absolute paths and `..` are rejected) into a fresh work directory, runs it, and returns an
`ExecResult`. It raises only on invalid arguments; crashes, timeouts, memory errors and output floods
are results. `make_sandbox` returns a `SandboxPool` that blocks callers beyond `max_concurrency`
(`submit` returns a `Future`); `backend: auto` picks a container runtime when one is available with
the image present and falls back to the process backend with a warning. `capabilities()` reports what
is actually enforced on the current host.

### `process` backend: best-effort isolation

For development, CI, and trusted or low-risk code. What it does:

* a fresh private work directory (mode 0700) as working directory, `HOME` and `TMPDIR`, removed after
  every run;
* a new session and process group; on the wall-clock timeout, on output overflow, and right after the
  program exits (to reap background children), the whole group is killed with SIGKILL;
* resource limits applied by a launcher before `exec` (thread-safe, no `preexec_fn`): `RLIMIT_CPU`,
  `RLIMIT_AS` (memory), `RLIMIT_NPROC`, `RLIMIT_FSIZE`, `RLIMIT_NOFILE`, `RLIMIT_CORE=0`;
* stdout and stderr capped per stream (`max_output_bytes`); a program that exceeds the cap is stopped
  and the result is marked `truncated`;
* a scrubbed environment: only `PATH`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TZ` and names you add with
  `env_allowlist` are passed; API keys, tokens and `PYTHON*` variables are not. Numeric libraries are
  pinned to one thread. The interpreter runs with `-I` (isolated mode);
* no inherited file descriptors; stdin comes from a file;
* network isolation where the host allows it without privileges: an unprivileged user + network
  namespace with `unshare` (plus a PID namespace, so no process survives the run, when supported) on
  Linux, or a deny-network `sandbox-exec` profile on macOS. `network: require` refuses to start when
  neither works; the default `deny` logs a warning and continues **without** network isolation.

What it does **not** guarantee (it is not a security boundary):

* the program runs as the trainer's user and can **read any file that user can read** (datasets,
  checkpoints, credentials files in the home directory), and write anywhere that user can write;
* without a network namespace it can reach the network, including local services;
* without a PID namespace, a process that calls `setsid()` leaves the process group and survives the
  group kill (it is still bound by its CPU-time limit);
* `RLIMIT_NPROC` counts all processes (on Linux: threads) of the user, so the sandbox sets it to the
  user's current count plus `max_processes`. The budget is shared with everything else the user runs
  at that moment: a fork bomb is contained, but can briefly starve the user's other processes of new
  processes until it is killed;
* `RLIMIT_AS` is not enforced on macOS (`capabilities()["memory_limit"]` is then `False`), and on Linux
  it limits address space, not resident memory; out-of-memory detection is a best-effort heuristic
  (`MemoryError` in stderr);
* kernel attack surface is fully exposed.

### `docker` / `podman` backend: container isolation

For untrusted code, i.e. RL policies. Each run starts a throw-away container:

```
docker run --rm -i --init --pull never --network none --read-only
  --tmpfs /tmp:rw,nosuid,nodev,size=<tmpfs_mb>m --pids-limit <pids_limit>
  --memory <memory_mb>m --memory-swap <memory_mb>m --cpus <cpus> --user 65534:65534
  --cap-drop ALL --security-opt no-new-privileges --ulimit nofile/fsize/cpu/core
  -v <run dir>:/work:ro <image> python -I -c <bootstrap>
```

Guarantees (as enforced by the container runtime): no network interfaces except loopback; a read-only
root filesystem and no host filesystem access beyond the run directory, mounted read-only (the code
runs from a copy in the size-bounded `/tmp`); an unprivileged user without capabilities or privilege
escalation; a PID limit (fork bombs are contained); a hard memory limit without swap (the kill is
reported as `oom`); a CPU quota and CPU-time limit; a wall-clock timeout after which the container is
killed and removed; only the variables set by the sandbox in the container environment.

Limits: containers share the host kernel (for a stronger boundary use a sandboxed runtime such as
gVisor with `runtime: runsc`); each run pays the container start (typically 0.3–1 s, covered by
`start_overhead_s`, which is added to the wall-clock deadline); the image must be present locally
(it is never pulled implicitly); the run directory is made world-readable (0755/0644) so the
unprivileged container user can read it; and the directory must be visible to the daemon (set
`tmp_root` to a shared path when the daemon runs in a VM). `container_available(binary, image)` checks
the binary, the daemon and the image.

### Choosing a backend

Use `docker`/`podman` (ideally with a sandboxed runtime) whenever a policy under training writes the
code: RL optimizes against whatever the environment lets it do, including reading files or reaching
services it should not. Use `process` for development and CI, on hosts where the trainer's user has
nothing to protect, and check `capabilities()` at startup.
