"""Search-R1 style question answering with a retrieval server, as a per-episode Judge RL environment.

A port of verl-agent's ``SearchEnvironmentManager`` + ``SearchMultiProcessEnv`` + the vendored SkyRL
``SearchEnv`` / ``SearchToolGroup`` (GiGPO repository), one episode per instance. Prompts, memory,
action parsing, retrieval formatting and scoring are byte-identical to verl-agent:

* the policy answers with ``<search> query </search>`` (the environment queries the retrieval server) or
  ``<answer> ... </answer>`` (ends the episode). The projection keeps the first complete ``<search>``
  block (else the first ``<answer>`` block, else the empty action) after cutting the response at the
  first ``</search>`` / ``</answer>``; the step is *invalid* when no block is found, when both tag kinds
  appear, or when either tag appears twice (invalid actions still reach the environment);
* the observation after a search is ``<information>{json}</information>`` where ``{json}`` is
  ``json.dumps({"result": "Doc 1: ...\\nDoc 2: ...\\n"})`` of the top-k passages (an error or "no results"
  message in the same JSON shape when retrieval fails); an action without a search block observes "";
* first step: the question-only template; later steps: the question plus the last ``history_length``
  steps as ``Step k:<action> <observation>`` (verl-agent's ``SearchMemory``);
* the episode ends on an answer or after ``max_steps`` actions (4 in verl-agent's Search scripts); the
  final score is exact match (Search-R1 normalization: lower-case, strip punctuation and articles,
  collapse whitespace) of the last ``<answer>`` in the concatenated interaction against any target.
  ``won`` = score 1; the reward is ``success_reward * score`` (see the note on ``success_reward``).
  The anchor is the raw observation (the question at reset, the information text after a step).

Tasks are ``{"question", "ground_truth", "data_source"}`` rows of verl-agent's preprocessed Search-R1
parquet (``examples/data_preprocess/preprocess_search_r1_dataset.py``, column ``env_kwargs``); see
:func:`tasks`. The retrieval server is Search-R1's ``/retrieve`` endpoint: POST
``{"query", "topk", "return_scores": true}`` -> ``{"result": [[{"document": {"contents": ...}}, ...]]}``.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import re
import string
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from judgerl.envs.base import Env, Observation, StepResult, register_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- verl-agent prompts
# Verbatim from verl-agent ``agent_system/environments/prompts/search.py`` (trailing spaces included).
SEARCH_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

DEFAULT_SEARCH_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_DATA_DIR = "~/data/searchR1_processed_direct"

_RE_SEARCH_BLOCK = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
_RE_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_RE_SEARCH_TAG = re.compile(r"<search>", re.IGNORECASE)
_RE_ANSWER_TAG = re.compile(r"<answer>", re.IGNORECASE)


# ---------------------------------------------------------------------------- projection
def _trim_action(action: str) -> str:
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def project_action(response: str) -> Tuple[str, bool]:
    """verl-agent's ``search_projection`` for one response: (action sent to the env, valid flag)."""
    trimmed = _trim_action(response)
    valid = True
    m = _RE_SEARCH_BLOCK.search(trimmed)
    if m:
        action = f"<search>{m.group(1).strip()}</search>"
    else:
        m = _RE_ANSWER_BLOCK.search(trimmed)
        if m:
            action = f"<answer>{m.group(1).strip()}</answer>"
        else:
            action = ""
            valid = False
    n_search = len(_RE_SEARCH_TAG.findall(response))
    n_answer = len(_RE_ANSWER_TAG.findall(response))
    if (n_search and n_answer) or n_search > 1 or n_answer > 1:
        valid = False
    return action, valid


# ---------------------------------------------------------------------------- scoring (Search-R1 qa_em)
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction: str, golden_answers) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def extract_solution(solution_str: str) -> Optional[str]:
    """The last ``<answer>...</answer>`` (case-sensitive, as upstream), stripped."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def targets(ground_truth) -> List[str]:
    """Target answers of a ground truth: ``{"target": [...]}`` (Search-R1), a list, or a string."""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth["target"]
    if isinstance(ground_truth, str):
        return [ground_truth]
    return [str(t) for t in list(ground_truth)]


def compute_score(solution_str: str, ground_truth, format_score: float = 0.0, score: float = 1.0) -> float:
    """Search-R1 exact-match score of the last answer in ``solution_str``."""
    answer = extract_solution(solution_str)
    if answer is None:
        return 0
    return score if em_check(answer, targets(ground_truth)) else format_score


# ---------------------------------------------------------------------------- retrieval client
def passages_to_string(retrieval_result) -> str:
    out = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"].strip()
        out += f"Doc {idx + 1}: {content}\n"
    return out


class RetrievalClient:
    """HTTP client for a Search-R1 retrieval server, same request, retry policy and result text as
    verl-agent's ``SearchToolGroup.search``.

    ``search(query)`` returns the JSON result text (``{"result": ...}``); failures (after retries on
    connection errors, timeouts and HTTP 500/502/503/504) come back in the same shape as an error
    message, never as an exception. ``session`` is any object with a ``requests``-style
    ``post(url, headers=, json=, timeout=)``; the default is a pooled ``requests.Session``.
    """

    def __init__(self, url: str = DEFAULT_SEARCH_URL, topk: int = 3, timeout: float = 60,
                 max_retries: int = 10, retry_delay: float = 1.0, session=None,
                 sleep: Callable[[float], None] = time.sleep):
        self.url = url
        self.topk = topk
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = retry_delay
        self._session = session
        self._sleep = sleep
        self.last_status: Optional[str] = None

    @property
    def session(self):
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter
            s = requests.Session()
            adapter = HTTPAdapter(pool_connections=512, pool_maxsize=512, max_retries=0, pool_block=False)
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            self._session = s
        return self._session

    def call(self, query: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """verl-agent ``call_search_api``: (response json, error message)."""
        try:
            import requests
            exc = requests.exceptions
            conn_err, timeout_err, req_err = exc.ConnectionError, exc.Timeout, exc.RequestException
        except ImportError:                       # an injected session without requests installed
            conn_err, timeout_err, req_err = ConnectionError, TimeoutError, OSError
        log_prefix = f"[Search Request ID: {uuid.uuid4()}] "
        payload = {"query": query, "topk": self.topk, "return_scores": True}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        n = self.max_retries
        last_error = None
        for attempt in range(n):
            response = None
            try:
                response = self.session.post(self.url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code in (500, 502, 503, 504):
                    last_error = (f"{log_prefix}API Request Error: Server Error ({response.status_code}) "
                                  f"on attempt {attempt + 1}/{n}")
                    logger.warning(last_error)
                    if attempt < n - 1:
                        self._sleep(self.retry_delay * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json(), None
            except conn_err as e:
                last_error = f"{log_prefix}Connection Error: {e}"
            except timeout_err as e:
                last_error = f"{log_prefix}Timeout Error: {e}"
            except req_err as e:             # before JSONDecodeError, as upstream (requests' JSON
                last_error = f"{log_prefix}API Request Error: {e}"   # error subclasses both)
                break
            except json.JSONDecodeError as e:
                raw = getattr(response, "text", "N/A") if response is not None else "N/A"
                last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw[:200]}"
                break
            except Exception as e:
                last_error = f"{log_prefix}Unexpected Error: {e}"
                break
            logger.warning(last_error)
            if attempt < n - 1:
                self._sleep(self.retry_delay * (attempt + 1))
        logger.error(f"{log_prefix}API Request Failed after {n} attempts: {last_error}")
        return None, last_error

    def search(self, query: Optional[str]) -> str:
        """verl-agent ``SearchToolGroup.search``: the JSON result text ("" for a missing query)."""
        if query is None:
            return ""
        query = query.strip()
        api_response, error_msg = self.call(query)
        if error_msg:
            self.last_status = "api_error"
            return json.dumps({"result": f"Search error: {error_msg}"})
        if api_response:
            try:
                raw_results = api_response.get("result", [])
                if raw_results:
                    self.last_status = "success"
                    return json.dumps({"result": "\n---\n".join(passages_to_string(r) for r in raw_results)})
                self.last_status = "no_results"
                return json.dumps({"result": "No search results found."})
            except Exception as e:
                self.last_status = "processing_error"
                return json.dumps({"result": f"Error processing search results: {e}"})
        self.last_status = "unknown_api_state"
        return json.dumps({"result": "Unknown API state (no response and no error message)."})


_URL_COUNTER = itertools.count()


# ---------------------------------------------------------------------------- environment
def format_memory(history: List[Dict[str, str]], history_length: int) -> str:
    """verl-agent ``SearchMemory.fetch`` for one environment."""
    recent = history[-history_length:]
    start = len(history) - len(recent)
    return "\n".join(f"Step {start + j + 1}:{r['search']} {r['information']}\n" for j, r in enumerate(recent))


def build_prompt(question: str, history: List[Dict[str, str]], history_length: int, init: bool) -> str:
    """verl-agent ``SearchEnvironmentManager.build_text_obs`` for one environment."""
    if init or history_length <= 0:
        return SEARCH_TEMPLATE_NO_HIS.format(task_description=question)
    return SEARCH_TEMPLATE.format(task_description=question, memory_context=format_memory(history, history_length),
                                  step_count=len(history))


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "tolist"):
        return _plain(x.tolist())
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            pass
    return x


@register_env("search")
class SearchEnv(Env):
    """One Search-R1 question per instance.

    Args:
        history_length: steps of memory in the prompt (verl-agent's Search script uses 4).
        max_steps: actions per episode; the last one ends the episode (verl-agent: 4).
        search_url: retrieval endpoint, or a list of endpoints (instances pick one round-robin).
        topk / timeout / max_retries: retrieval parameters (verl-agent: 3 / 60 s / 10).
        success_reward: reward of a correct answer (default 1.0, verl-agent's raw exact-match score, which
            its invalid-action penalty of 0.01 is calibrated against).
        retriever: a ``RetrievalClient``-like object with ``search(query) -> str`` (tests, custom backends).
    """

    def __init__(self, history_length: int = 4, max_steps: int = 4,
                 search_url: Union[str, Sequence[str], None] = None, topk: int = 3, timeout: float = 60,
                 max_retries: int = 10, success_reward: float = 1.0, retriever=None):
        self.history_length = history_length
        self.max_steps = max_steps
        self.success_reward = float(success_reward)
        if retriever is None:
            urls = search_url or os.environ.get("JUDGERL_SEARCH_URL") or DEFAULT_SEARCH_URL
            if isinstance(urls, str):
                urls = [u.strip() for u in urls.split(",") if u.strip()]
            urls = list(urls)
            retriever = RetrievalClient(urls[next(_URL_COUNTER) % len(urls)], topk=topk, timeout=timeout,
                                        max_retries=max_retries)
        self.retriever = retriever
        self._question = ""
        self._ground_truth: Any = None
        self._data_source = "unknown"
        self._history: List[Dict[str, str]] = []
        self._chat: List[str] = []
        self._turns = 0
        self._done = False

    def reset(self, task: Dict[str, Any], seed: int = 0) -> Observation:
        if "ground_truth" not in task:
            raise KeyError("ground_truth is required in the task")
        self._question = task["question"]
        self._ground_truth = _plain(task["ground_truth"])
        self._data_source = task.get("data_source", "unknown")
        self._history = []
        self._chat = []
        self._turns = 0
        self._done = False
        prompt = build_prompt(self._question, self._history, self.history_length, init=True)
        return Observation(prompt=[{"role": "user", "content": prompt}], anchor=self._question,
                           text=self._question, info={"data_source": self._data_source})

    def step(self, response: str) -> StepResult:
        action, valid = project_action(response)
        # SkyRL SearchEnv.step on the projected action
        self._turns += 1
        self._chat.append(action)
        if not self._done:
            self._done = self._turns >= self.max_steps or ("<answer>" in action and "</answer>" in action)
        done = self._done
        info: Dict[str, Any] = {"data_source": self._data_source, "is_action_valid": valid, "action": action}
        score = 0.0
        if done:
            score = float(compute_score("".join(self._chat), self._ground_truth))
            obs = ""
            info["tool_calling"] = False
        else:
            query = None
            error = None
            try:
                if "<search>" in action and "</search>" in action:
                    m = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
                    query = m.group(1) if m else None
                result = self.retriever.search(query)
                content = "\n<information>" + result + "</information>\n" if len(result) > 0 else None
            except Exception as e:
                error = str(e)
                content = None
                logger.warning("search tool error: %s", error)
            if content:
                self._chat.append(content)
                obs = content.strip()
            elif error:
                self._chat.append(error)
                obs = error.strip()
            else:
                obs = ""
            info.update({"tool_calling": True, "tool_input": [query],
                         "retrieval_status": getattr(self.retriever, "last_status", None)})
        won = bool(done and score >= 1.0)
        info.update({"won": won, "em_score": score})
        self._history.append({"search": action, "information": obs})
        prompt = build_prompt(self._question, self._history, self.history_length, init=False)
        observation = Observation(prompt=[{"role": "user", "content": prompt}], anchor=obs, text=obs,
                                  info={"data_source": self._data_source})
        return StepResult(observation, self.success_reward * score, done, info)

    def task_text(self) -> str:
        return self._question

    def metadata(self) -> Dict[str, Any]:
        return {"reference": self._ground_truth, "answers": targets(self._ground_truth) if self._ground_truth
                is not None else [], "data_source": self._data_source}


# ---------------------------------------------------------------------------- task provider
def row_to_task(row: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """A task from a row of verl-agent's preprocessed Search-R1 data (``env_kwargs`` column), falling
    back to raw Search-R1 columns (``question``, ``reward_model.ground_truth`` / ``golden_answers``)."""
    kw = row.get("env_kwargs")
    if isinstance(kw, dict) and "question" in kw:
        question, gt, ds = kw["question"], kw.get("ground_truth"), kw.get("data_source")
    else:
        question = row.get("question") or (row.get("extra_info") or {}).get("question", "")
        rm = row.get("reward_model")
        gt = rm.get("ground_truth") if isinstance(rm, dict) and "ground_truth" in rm else row.get("golden_answers", [])
        ds = row.get("data_source")
    gt = _plain(gt)
    if not isinstance(gt, dict):
        gt = {"target": gt if isinstance(gt, list) else [gt]}
    return {"question": question, "ground_truth": gt,
            "data_source": str(ds) if ds is not None else "unknown", "index": index}


def _read_rows(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else list(data.values())
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except ImportError:
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")


def tasks(split: str = "train", limit: int = 0, path: Optional[str] = None, data_dir: Optional[str] = None,
          data_sources: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Tasks from verl-agent's preprocessed parquet (``path``, else ``<data_dir>/<split>.parquet`` with
    verl-agent's default ``~/data/searchR1_processed_direct``), in file order; optionally only some
    ``data_sources`` (e.g. ``["nq", "hotpotqa"]``). ``.jsonl`` / ``.json`` files with the same columns work
    too."""
    if path is None:
        path = os.path.join(os.path.expanduser(data_dir or DEFAULT_DATA_DIR), f"{split}.parquet")
    out = []
    for i, row in enumerate(_read_rows(os.path.expanduser(path))):
        task = row_to_task(row, i)
        if data_sources and task["data_source"] not in data_sources:
            continue
        out.append(task)
        if limit > 0 and len(out) >= limit:
            break
    return out
