## WebShop and Search (ports of verl-agent)

Both environments reproduce verl-agent's (GiGPO repository, commit `20bd331`) environment managers
one episode per instance: prompt templates (verbatim), history memory, observation formatting, action
projection and validity rule, and rewards. `tests/envs/test_webshop_port.py` and
`tests/envs/test_search_port.py` run the upstream manager code against the same mocked simulator and
retrieval server and require byte-identical prompts, anchors, rewards and flags; point
`$VERL_AGENT_ROOT` at a verl-agent checkout to enable those checks (they are skipped otherwise).

### `webshop` (`judgerl.envs.webshop`)

```yaml
env: {name: webshop, kwargs: {history_length: 2, max_steps: 15}}
```

* Needs the WebShop simulator (`web_agent_site`, its data and search index). Set `$WEBSHOP_ROOT` (or
  `webshop_root:`) to the directory that contains `web_agent_site`; verl-agent vendors it under
  `agent_system/environments/env_package/webshop/webshop`. Defaults match verl-agent's `env.webshop`:
  small product set (`items_shuffle_1000.json`, `items_ins_v2_1000.json`), synthetic goals.
* Action = lower-cased text inside `<action>...</action>` (else the last 20 characters); valid only with
  `<think>...</think>` and no CJK characters. Reward 10 when a purchase scores 1.0, else 0;
  `info["task_score"]` keeps WebShop's score, `info["won"]` the success flag.
* Tasks are `{"goal_idx", "goal_seed"}`. The simulator shuffles and prices its goals with its seed, so
  an index is only meaningful with that seed. `tasks("test")` = goals 0-499, `tasks("train")` = goals
  500+ (verl-agent's split), all with one seed (default 42, WebShop's own default).
  `verl_agent_draws(split, env_seed, batch_size, group_n, n_resets)` reproduces the exact goals
  verl-agent serves: per-worker simulator seeds (`env.seed + i // group_n` for training,
  `env.seed + 1000 + i` for validation) and `RandomState(seed).choice(...)` draws per reset.
* `metadata()` (hidden from the policy): `goal`, `attributes`, `goal_options`, `price_upper`, `asin`,
  `product_category`, and, with the DARS package installed, `goal_schema`. With DARS, `info["state_key"]`
  carries the page state key (the purchase step is keyed on the page the purchase was made from).
* Simulator servers are shared per seed within a process (each instance has its own session prefix);
  `share_server: false` gives every instance its own simulator, as verl-agent's workers do.

### `search` (`judgerl.envs.search`)

```yaml
env: {name: search, kwargs: {history_length: 4, max_steps: 4, search_url: http://127.0.0.1:8000/retrieve}}
```

* The policy issues `<search> query </search>` or `<answer> ... </answer>`. Retrieval uses Search-R1's
  `/retrieve` endpoint (`{"query", "topk": 3, "return_scores": true}`), with verl-agent's retry policy
  (connection errors, timeouts and HTTP 5xx retried up to 10 times with linear back-off). `search_url`
  can be a list (or comma-separated); `$JUDGERL_SEARCH_URL` sets the default. Pass `retriever=` for any
  object with `search(query) -> str`.
* The episode ends on an answer or after `max_steps` actions. Score = Search-R1 exact match of the last
  `<answer>` against any target. **Reward scale:** verl-agent's Search environment returns the raw score
  (1.0 on success); Judge RL defaults to `success_reward: 10.0` like its other environments. Set
  `success_reward: 1.0` for verl-agent's numbers. `info["em_score"]` always holds the raw score.
* Tasks: `tasks(split, path=... | data_dir=...)` reads verl-agent's preprocessed parquet
  (`examples/data_preprocess/preprocess_search_r1_dataset.py`, default `~/data/searchR1_processed_direct`,
  column `env_kwargs`) into `{"question", "ground_truth": {"target": [...]}, "data_source", "index"}`.
  `metadata()` = `{"reference", "answers", "data_source"}`.
