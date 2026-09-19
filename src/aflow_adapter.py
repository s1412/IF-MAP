"""AFlow bridge: prompts, evaluation, traces, and cache."""

import os
import re
import sys
import asyncio
import contextlib
import hashlib
import importlib
import json
import multiprocessing
import sqlite3
import threading
import time
import traceback
import contextvars
from typing import Dict, List, Optional

from src.errors import fail
from src.paths import project_path

# ------------------------------------------------------------------

# ------------------------------------------------------------------
_AFLOW_ROOT = str(project_path(os.environ.get("IFIP_AFLOW_ROOT", "aflow")).resolve())
if _AFLOW_ROOT not in sys.path:
    sys.path.insert(0, _AFLOW_ROOT)

_DATASETS_DIR = os.path.join(_AFLOW_ROOT, "data", "datasets")
_EVAL_LOG_DIR = os.path.join(_AFLOW_ROOT, ".eval_logs")

SUPPORTED_DATASETS = ["GSM8K", "MATH", "HumanEval", "HotpotQA", "DROP", "MBPP", "NQ"]


EVAL_TIMEOUT = int(os.environ.get("IFIP_EVAL_TIMEOUT", "1800"))


class _HardEvaluationTimeout(TimeoutError):
    """Raised when an isolated AFlow evaluation exceeds its wall-clock limit."""


def _subprocess_worker_entry(conn, worker):
    """Execute ``worker`` in a child and return its value or traceback."""
    try:
        conn.send(("ok", worker()))
    except BaseException as exc:  # noqa: BLE001 - transport child failures
        try:
            conn.send((
                "err", type(exc).__name__, str(exc), traceback.format_exc()
            ))
        except Exception:  # noqa: BLE001 - parent will report child exit
            pass
    finally:
        conn.close()


def _run_with_hard_timeout(worker, timeout_s: int):
    """Run a complete evaluation attempt with a real wall-clock timeout.

    ``asyncio.wait_for`` waits for cancellation to finish, so an HTTP request
    or generated program that ignores cancellation can block forever. Process
    isolation lets the parent terminate the entire attempt at the deadline.
    The child is non-daemonic because code benchmarks spawn sandbox children.
    """
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError as exc:  # pragma: no cover - IFIP production is Linux
        raise RuntimeError("hard evaluation timeout requires fork support") from exc

    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_subprocess_worker_entry,
        args=(child_conn, worker),
        daemon=False,
    )
    process.start()
    child_conn.close()
    deadline = time.monotonic() + max(1, int(timeout_s))
    message = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if parent_conn.poll(min(0.25, remaining)):
                message = parent_conn.recv()
                break
            if not process.is_alive():
                if parent_conn.poll(0.1):
                    message = parent_conn.recv()
                break
    finally:
        parent_conn.close()

    if message is None:
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            raise _HardEvaluationTimeout(
                f"evaluation exceeded hard timeout of {int(timeout_s)}s"
            )
        process.join(1)
        raise RuntimeError(
            f"evaluation subprocess exited without a result (exitcode={process.exitcode})"
        )

    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)

    if message[0] == "ok":
        return message[1]
    _, error_type, error_text, child_traceback = message
    raise RuntimeError(
        f"evaluation subprocess raised {error_type}: {error_text}\n{child_traceback}"
    )


LAST_EVAL_COST: Dict[str, object] = {}

COST_ACCUM = {"usd": 0.0, "in": 0, "out": 0,
              "cache_hits": 0, "cache_misses": 0,
              "eval_cache_hits": 0, "eval_cache_misses": 0}
_EVAL_CACHE_LOCK = threading.Lock()
_EVAL_CACHE_INIT = set()
_EVAL_CACHE_MEM: Dict[str, dict] = {}


def reset_cost():
    COST_ACCUM["usd"] = 0.0
    COST_ACCUM["in"] = 0
    COST_ACCUM["out"] = 0
    COST_ACCUM["cache_hits"] = 0
    COST_ACCUM["cache_misses"] = 0
    COST_ACCUM["eval_cache_hits"] = 0
    COST_ACCUM["eval_cache_misses"] = 0


def get_cost() -> Dict[str, float]:
    return {"usd": round(COST_ACCUM["usd"], 6),
            "input_tokens": COST_ACCUM["in"], "output_tokens": COST_ACCUM["out"],
            "cache_hits": COST_ACCUM["cache_hits"],
            "cache_misses": COST_ACCUM["cache_misses"],
            "eval_cache_hits": COST_ACCUM["eval_cache_hits"],
            "eval_cache_misses": COST_ACCUM["eval_cache_misses"]}


def _silence_aflow_logger():
    """Disable verbose AFlow logs."""
    try:
        from scripts.logs import logger, LogLevel
        logger.log_level = LogLevel.ERROR.value[0]
    except Exception:  # noqa: BLE001
        pass


_silence_aflow_logger()


def _env_true(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _eval_cache_enabled() -> bool:
    return _env_true("IFIP_EVAL_CACHE", default=False)


def _eval_cache_db_path() -> str:
    path = os.environ.get("IFIP_EVAL_CACHE_DB", "").strip()
    return os.path.abspath(path) if path else ""


def _eval_cache_key(dataset: str, agent_prompts: Dict[str, str],
                    query_indices: List[int], model: str, n_repeats: int,
                    split: str, agent_model_overrides: Dict[str, str] = None) -> str:
    payload = {
        "schema": 1,
        "run_id": os.environ.get("IFIP_RUN_ID", ""),
        "eval_cache_version": os.environ.get("IFIP_EVAL_CACHE_VERSION", "rq2-eval-v1"),
        "aflow_cache_version": os.environ.get("AFLOW_LLM_CACHE_VERSION", "rq2-v2"),
        "dataset": dataset,
        "split": split,
        "model": model,
        "agent_model_overrides": dict(sorted((agent_model_overrides or {}).items())),
        "n_repeats": int(n_repeats),
        "query_indices": list(query_indices),
        "agent_prompts": {k: agent_prompts[k] for k in sorted(agent_prompts)},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _eval_cache_get(key: str, need_failures: bool):
    if not _eval_cache_enabled():
        return None
    path = _eval_cache_db_path()
    if not path:
        item = _EVAL_CACHE_MEM.get(key)
        if item and (not need_failures or item.get("failures") is not None):
            COST_ACCUM["eval_cache_hits"] += 1
            return item
        COST_ACCUM["eval_cache_misses"] += 1
        return None

    _ensure_eval_cache_schema(path)
    with _EVAL_CACHE_LOCK, sqlite3.connect(path, timeout=30.0) as conn:
        row = conn.execute(
            "SELECT loss, failures_json FROM eval_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    if row is None:
        COST_ACCUM["eval_cache_misses"] += 1
        return None
    failures = json.loads(row[1]) if row[1] else None
    if need_failures and failures is None:
        COST_ACCUM["eval_cache_misses"] += 1
        return None
    COST_ACCUM["eval_cache_hits"] += 1
    return {"loss": float(row[0]), "failures": failures}


def _eval_cache_put(key: str, loss: float, failures=None) -> None:
    if not _eval_cache_enabled():
        return
    item = {"loss": float(loss), "failures": failures}
    path = _eval_cache_db_path()
    if not path:
        _EVAL_CACHE_MEM[key] = item
        return

    _ensure_eval_cache_schema(path)
    failures_json = json.dumps(failures, ensure_ascii=False) if failures is not None else None
    with _EVAL_CACHE_LOCK, sqlite3.connect(path, timeout=30.0) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_cache
              (cache_key, loss, failures_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, float(loss), failures_json, time.time()),
        )


def _ensure_eval_cache_schema(path: str) -> None:
    if path in _EVAL_CACHE_INIT:
        return
    with _EVAL_CACHE_LOCK:
        if path in _EVAL_CACHE_INIT:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with sqlite3.connect(path, timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_cache (
                    cache_key TEXT PRIMARY KEY,
                    loss REAL NOT NULL,
                    failures_json TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_cache_created "
                "ON eval_cache(created_at)"
            )
        _EVAL_CACHE_INIT.add(path)


@contextlib.contextmanager
def _in_aflow():
    """Run with the AFlow working directory."""
    old = os.getcwd()
    os.chdir(_AFLOW_ROOT)
    try:
        yield
    finally:
        os.chdir(old)


def get_topology(dataset: str) -> List[dict]:
    """Return dataset workflow topology."""
    _check_dataset(dataset)
    from config.workflow_topology import WORKFLOW_TOPOLOGIES_ROUND_1
    if dataset not in WORKFLOW_TOPOLOGIES_ROUND_1:
        fail("aflow_adapter.get_topology", "dataset not in WORKFLOW_TOPOLOGIES_ROUND_1",
             dataset=dataset, available=list(WORKFLOW_TOPOLOGIES_ROUND_1))
    return list(WORKFLOW_TOPOLOGIES_ROUND_1[dataset])


def get_agent_names(dataset: str) -> List[str]:
    return [t["name"] for t in get_topology(dataset)]


def get_original_prompts(dataset: str) -> Dict[str, str]:
    """Return original operator prompts."""
    from config.prompt_config import PROMPT_CONFIGS
    out = {}
    for t in get_topology(dataset):
        pd = t["prompt_domain"]
        if pd not in PROMPT_CONFIGS:
            fail("aflow_adapter.get_original_prompts", "prompt_domain missing in PROMPT_CONFIGS",
                 dataset=dataset, agent=t["name"], prompt_domain=pd)
        out[t["name"]] = PROMPT_CONFIGS[pd]["template"]
    return out


def get_prompt_goals(dataset: str) -> Dict[str, str]:
    """Return operator goals."""
    from config.prompt_config import PROMPT_CONFIGS
    return {t["name"]: PROMPT_CONFIGS[t["prompt_domain"]].get("goal", "")
            for t in get_topology(dataset)}


def get_required_placeholders(dataset: str) -> Dict[str, set]:
    """Return required template placeholders."""
    return {name: extract_placeholders(p)
            for name, p in get_original_prompts(dataset).items()}


def extract_placeholders(text: str) -> set:
    return set(re.findall(r"\{(\w+)\}", text or ""))


def _load_workflow_class(dataset: str):
    module = importlib.import_module(f"workspace.{dataset}.workflows.round_1.graph")
    return module.Workflow


def _cache_namespace_extra(extra: str = "") -> str:
    extra = (extra or "").strip().strip("|")
    return f"|{extra}" if extra else ""


def _build_workflow(dataset: str, model: str, split: str,
                    cache_namespace_extra: str = ""):
    from scripts.async_llm import LLMsConfig
    WorkflowClass = _load_workflow_class(dataset)
    llm_config = LLMsConfig.default().get(model)
    flow = WorkflowClass(name=f"{dataset}_wf", llm_config=llm_config, dataset=dataset)
    cache_version = os.environ.get("AFLOW_LLM_CACHE_VERSION", "rq2-v2")
    cache_run_id = os.environ.get("AFLOW_LLM_CACHE_RUN_ID") or os.environ.get("IFIP_RUN_ID", "")
    # Hard isolation prevents cross-dataset and DEV/TEST leakage. Bump
    # AFLOW_LLM_CACHE_VERSION whenever workflow semantics change materially.
    # run_id prevents a fresh experiment run from silently reusing old run outputs.
    flow.llm.cache_namespace = (
        f"dataset={dataset}|split={split}|workflow=round_1|"
        f"version={cache_version}|run={cache_run_id}"
        f"{_cache_namespace_extra(cache_namespace_extra)}"
    )
    return flow


def _agent_operator_from_prompt_attr(flow, attr_path: str):
    """Given topology attr_path ending in '.prompt', return the operator object."""
    parts = attr_path.split(".")
    if len(parts) < 2:
        return None
    obj = flow
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj


def _inject_agent_model_overrides(flow, topo, agent_model_overrides: Dict[str, str],
                                  dataset: str, split: str,
                                  cache_namespace_extra: str = ""):
    """Replace selected agents' operator.llm while all other operators keep flow.llm.

    This implements RQ3 model intervention: only the chosen operator runs with a
    stronger model; prompts and all other operators remain unchanged.
    """
    if not agent_model_overrides:
        return
    from scripts.async_llm import LLMsConfig, create_llm_instance

    cache_version = os.environ.get("AFLOW_LLM_CACHE_VERSION", "rq2-v2")
    cache_run_id = os.environ.get("AFLOW_LLM_CACHE_RUN_ID") or os.environ.get("IFIP_RUN_ID", "")
    configs = LLMsConfig.default()
    topo_by_name = {t["name"]: t for t in topo}
    for agent, model_name in agent_model_overrides.items():
        if agent not in topo_by_name:
            fail("aflow_adapter._inject_agent_model_overrides",
                 "agent not found in topology", dataset=dataset, agent=agent,
                 available=list(topo_by_name))
        op = _agent_operator_from_prompt_attr(flow, topo_by_name[agent]["attr_path"])
        if op is None or not hasattr(op, "llm"):
            fail("aflow_adapter._inject_agent_model_overrides",
                 "operator has no llm field", dataset=dataset, agent=agent,
                 attr_path=topo_by_name[agent]["attr_path"])
        llm = create_llm_instance(configs.get(model_name))
        llm.cache_namespace = (
            f"dataset={dataset}|split={split}|workflow=round_1|"
            f"version={cache_version}|run={cache_run_id}|"
            f"model_swap_agent={agent}|model={model_name}"
            f"{_cache_namespace_extra(cache_namespace_extra)}"
        )
        op.llm = llm


def _unique_flow_llms(flow, topo) -> list:
    """Return each workflow-owned LLM instance exactly once."""
    llms = []
    if hasattr(flow, "llm"):
        llms.append(flow.llm)
    for t in topo:
        op = _agent_operator_from_prompt_attr(flow, t["attr_path"])
        if op is not None and hasattr(op, "llm"):
            llms.append(op.llm)
    seen = set()
    unique = []
    for llm in llms:
        if id(llm) in seen:
            continue
        seen.add(id(llm))
        unique.append(llm)
    return unique


async def _close_flow_llms(flow, topo) -> None:
    """Close HTTP pools before the short-lived asyncio.run() loop exits."""
    for llm in _unique_flow_llms(flow, topo):
        close = getattr(llm, "close", None)
        if close is None:
            continue
        try:
            await asyncio.wait_for(close(), timeout=10.0)
        except Exception:
            pass


def _summarize_flow_usage(flow, topo) -> dict:
    """Sum token/cost over all unique LLM instances in the workflow."""
    out = {"total_cost": 0.0, "total_input_tokens": 0, "total_output_tokens": 0,
           "cache_hits": 0, "cache_misses": 0, "response_count": 0,
           "output_cap_hits": 0, "finish_reason_counts": {},
           "max_tokens_counts": {}}
    for llm in _unique_flow_llms(flow, topo):
        try:
            summ = llm.get_usage_summary()
        except Exception:  # noqa: BLE001
            continue
        out["total_cost"] += float(summ.get("total_cost", 0.0))
        out["total_input_tokens"] += int(summ.get("total_input_tokens", 0) or 0)
        out["total_output_tokens"] += int(summ.get("total_output_tokens", 0) or 0)
        out["cache_hits"] += int(summ.get("cache_hits", 0) or 0)
        out["cache_misses"] += int(summ.get("cache_misses", 0) or 0)
        out["response_count"] += int(summ.get("response_count", 0) or 0)
        out["output_cap_hits"] += int(summ.get("output_cap_hits", 0) or 0)
        for reason, count in (summ.get("finish_reason_counts") or {}).items():
            out["finish_reason_counts"][str(reason)] = (
                int(out["finish_reason_counts"].get(str(reason), 0))
                + int(count or 0)
            )
        for limit, count in (summ.get("max_tokens_counts") or {}).items():
            out["max_tokens_counts"][str(limit)] = (
                int(out["max_tokens_counts"].get(str(limit), 0))
                + int(count or 0)
            )
    out["output_cap_rate"] = (
        out["output_cap_hits"] / out["response_count"]
        if out["response_count"] else 0.0
    )
    return out


def _inject_prompts(flow, topo, agent_prompts, original_prompts):
    """Apply operator prompt overrides."""
    for t in topo:
        name = t["name"]
        text = agent_prompts[name]
        orig = original_prompts[name]
        if extract_placeholders(text) != extract_placeholders(orig):
            text = orig  # fallback
        parts = t["attr_path"].split(".")
        obj = flow
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], text)


_TARGET_TRACE_CONTEXT = contextvars.ContextVar("ifip_target_trace", default=None)


def _trace_clip(value, limit: int):
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        text = str(value)
    return _clip_text(text, limit)


class _TargetTraceProxy:
    """Capture target and downstream calls."""

    def __init__(self, wrapped, agent_name: str, max_chars: int):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_agent_name", agent_name)
        object.__setattr__(self, "_max_chars", max_chars)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)

    async def __call__(self, *args, **kwargs):
        ctx = _TARGET_TRACE_CONTEXT.get()
        call = None
        if ctx is not None:
            call = {
                "agent": object.__getattribute__(self, "_agent_name"),
                "input": _trace_clip(
                    {"args": args, "kwargs": kwargs},
                    object.__getattribute__(self, "_max_chars"),
                ),
            }
        try:
            result = await object.__getattribute__(self, "_wrapped")(*args, **kwargs)
            if call is not None:
                call["output"] = _trace_clip(
                    result, object.__getattribute__(self, "_max_chars"))
                ctx["calls"].append(call)
            return result
        except Exception as exc:
            if call is not None:
                call["error"] = _clip_text(
                    f"{type(exc).__name__}: {exc}",
                    object.__getattribute__(self, "_max_chars"),
                )
                ctx["calls"].append(call)
            raise


def _install_target_trace(flow, topo, target_agent: str, max_chars: int) -> None:
    if not target_agent:
        return
    topo_by_name = {t["name"]: t for t in topo}
    if target_agent not in topo_by_name:
        fail("aflow_adapter._install_target_trace", "agent not found in topology",
             target_agent=target_agent, available=list(topo_by_name))


    traced = {target_agent}
    changed = True
    while changed:
        changed = False
        for item in topo:
            if item["name"] not in traced and any(dep in traced for dep in item.get("dependencies", [])):
                traced.add(item["name"])
                changed = True
    for agent_name in traced:
        parts = topo_by_name[agent_name]["attr_path"].split(".")
        if len(parts) < 2:
            fail("aflow_adapter._install_target_trace", "invalid attr_path",
                 target_agent=agent_name, attr_path=topo_by_name[agent_name]["attr_path"])
        parent = flow
        for part in parts[:-2]:
            parent = getattr(parent, part)
        operator_attr = parts[-2]
        wrapped = getattr(parent, operator_attr)
        setattr(parent, operator_attr, _TargetTraceProxy(wrapped, agent_name, max_chars))


def _wrap_benchmark_trace(bench, data: List[dict], trace_by_row: dict,
                          target_agent: str) -> None:
    original = bench.evaluate_problem
    row_by_problem_id = {id(problem): i for i, problem in enumerate(data)}

    async def evaluate_problem_with_trace(problem, agent):
        row_index = row_by_problem_id.get(id(problem), -1)
        state = {"row_index": row_index, "agent": target_agent, "calls": []}
        token = _TARGET_TRACE_CONTEXT.set(state)
        try:
            return await original(problem, agent)
        finally:
            _TARGET_TRACE_CONTEXT.reset(token)
            trace_by_row[row_index] = state

    bench.evaluate_problem = evaluate_problem_with_trace


def _benchmark_class(dataset: str):
    from benchmarks.gsm8k import GSM8KBenchmark
    from benchmarks.math import MATHBenchmark
    from benchmarks.humaneval import HumanEvalBenchmark
    from benchmarks.hotpotqa import HotpotQABenchmark
    from benchmarks.drop import DROPBenchmark
    from benchmarks.mbpp import MBPPBenchmark
    from benchmarks.nq import NQBenchmark
    return {
        "GSM8K": GSM8KBenchmark, "MATH": MATHBenchmark,
        "HumanEval": HumanEvalBenchmark, "HotpotQA": HotpotQABenchmark,
        "DROP": DROPBenchmark, "MBPP": MBPPBenchmark, "NQ": NQBenchmark,
    }[dataset]


def _data_path(dataset: str, split: str = "validate") -> str:
    """Resolve the dataset split path."""
    if split not in ("validate", "test"):
        fail("aflow_adapter._data_path", "split must be 'validate' or 'test'", split=split)
    p = os.path.join(_DATASETS_DIR, f"{dataset.lower()}_{split}.jsonl")
    if not os.path.exists(p):
        fail("aflow_adapter._data_path", f"{split} jsonl not found (porting incomplete?)",
             dataset=dataset, path=p)
    return p


def get_split_size(dataset: str, split: str = "validate") -> int:
    """Count rows in a dataset split."""
    _check_dataset(dataset)
    p = _data_path(dataset, split)
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def evaluate_with_prompts(
    dataset: str,
    agent_prompts: Dict[str, str],
    query_indices: List[int],
    model: str = "gpt-4o-mini-exec",
    n_repeats: int = 1,
    concurrent_batch: int = 50,
    dry_run: bool = False,
    split: str = "validate",
    collect_failures: bool = False,
    max_failure_examples: int = 0,
    failure_example_chars: int = 500,
    agent_model_overrides: Dict[str, str] = None,
    return_details: bool = False,
    collect_sample_details: bool = False,
    trace_agent: str = None,
    trace_chars: int = 800,
) -> float:
    """Evaluate prompt overrides and optionally collect traces."""
    # AsyncLLM reads this marker to apply dataset-specific request safeguards.
    # RQ2 evaluates datasets sequentially, and every call refreshes the value.
    os.environ["IFIP_CURRENT_DATASET"] = str(dataset)

    if dry_run:
        if return_details:
            out = {"loss": 0.5, "loss_repeats": [0.5] * max(1, int(n_repeats or 1))}
            out.update({
                "finish_reason_counts": {}, "max_tokens_counts": {},
                "response_count": 0, "output_cap_hits": 0,
                "output_cap_rate": 0.0,
            })
            if collect_failures:
                out["failures"] = []
            if collect_sample_details or trace_agent:
                out["sample_details"] = []
            return out
        if collect_failures:
            return {"loss": 0.5, "failures": []}
        return 0.5

    _check_dataset(dataset)
    agent_model_overrides = dict(agent_model_overrides or {})
    cache_key = None
    if not return_details:
        cache_key = _eval_cache_key(
            dataset, agent_prompts, query_indices, model, n_repeats, split,
            agent_model_overrides=agent_model_overrides)
        cached = _eval_cache_get(
            cache_key, need_failures=bool(collect_failures and max_failure_examples > 0))
        if cached is not None:
            if collect_failures:
                return {"loss": cached["loss"], "failures": cached.get("failures") or []}
            return cached["loss"]

    topo = get_topology(dataset)
    original_prompts = get_original_prompts(dataset)
    BenchClass = _benchmark_class(dataset)
    data_path = _data_path(dataset, split)
    log_path = os.path.join(_EVAL_LOG_DIR, dataset)
    os.makedirs(log_path, exist_ok=True)

    scores: List[float] = []
    loss_repeats: List[float] = []
    real_cost = 0.0
    in_tok = out_tok = 0
    response_count = output_cap_hits = 0
    finish_reason_counts = {}
    max_tokens_counts = {}
    failures = []
    failure_examples_by_repeat: List[List[dict]] = []
    sample_details_by_repeat: List[List[dict]] = []
    n_repeat_total = max(1, int(n_repeats or 1))
    with _in_aflow():
        for repeat_i in range(n_repeat_total):
            # When evaluate_with_prompts itself repeats an evaluation, each
            # repeat must have an isolated AFlow LLM cache namespace.  This
            # preserves normal cache reuse *within* one repeat while preventing
            # repeat #2/#3 from replaying repeat #1's deterministic outputs.
            repeat_cache_extra = (
                f"eval_repeat={repeat_i + 1}" if n_repeat_total > 1 else ""
            )


            avg_score = None
            repeat_usage = {}
            for attempt in range(2):
                try:
                    def _attempt_worker():
                        flow = _build_workflow(
                            dataset, model, split,
                            cache_namespace_extra=repeat_cache_extra,
                        )
                        _inject_prompts(
                            flow, topo, agent_prompts, original_prompts
                        )
                        _inject_agent_model_overrides(
                            flow, topo, agent_model_overrides,
                            dataset=dataset, split=split,
                            cache_namespace_extra=repeat_cache_extra,
                        )
                        if trace_agent:
                            _install_target_trace(
                                flow, topo, trace_agent, trace_chars
                            )
                        bench = BenchClass(
                            name=dataset, file_path=data_path, log_path=log_path
                        )
                        if not _env_true(
                            "IFIP_AFLOW_SAVE_MISMATCH_LOGS", default=False
                        ):
                            bench.log_mismatch = lambda *args, **kwargs: None

                        async def _run():
                            try:
                                data = await bench.load_data(list(query_indices))
                                trace_by_row = {}
                                if trace_agent:
                                    _wrap_benchmark_trace(
                                        bench, data, trace_by_row, trace_agent
                                    )
                                results = await bench.evaluate_all_problems(
                                    data, flow,
                                    max_concurrent_tasks=concurrent_batch,
                                    show_progress=False,
                                )
                                columns = bench.get_result_columns()
                                attempt_score, total_cost = _summarize_results(
                                    results, columns
                                )
                                need_details = bool(
                                    collect_failures
                                    or collect_sample_details
                                    or trace_agent
                                )
                                details = _collect_sample_details(
                                    results, columns, trace_by_row,
                                    failure_example_chars,
                                ) if need_details else []
                                usage = _summarize_flow_usage(flow, topo)
                                return attempt_score, total_cost, details, usage
                            finally:
                                await _close_flow_llms(flow, topo)

                        return asyncio.run(_run())

                    (avg_score, _, repeat_sample_details,
                     repeat_usage) = _run_with_hard_timeout(
                        _attempt_worker, EVAL_TIMEOUT
                    )
                    failures = list(repeat_sample_details)
                    break
                except _HardEvaluationTimeout:
                    print(f"    [WARN] {dataset} eval timeout {EVAL_TIMEOUT}s "
                          f"(attempt {attempt+1}/2), retrying...", flush=True)
                    continue
                except Exception as e:
                    fail("aflow_adapter.evaluate_with_prompts",
                         f"AFlow run_evaluation raised: {type(e).__name__}: {str(e)[:200]}",
                         dataset=dataset, n_query=len(query_indices), model=model)
            if avg_score is None:
                fail("aflow_adapter.evaluate_with_prompts",
                     f"eval timed out {EVAL_TIMEOUT}s x2 (evaluation stalled or workload too large)",
                     dataset=dataset, n_query=len(query_indices), model=model)
            scores.append(float(avg_score))
            loss_repeats.append(1.0 - float(avg_score))
            if collect_failures:
                failure_examples_by_repeat.append(list(failures or []))
            if collect_sample_details or trace_agent:
                sample_details_by_repeat.append(list(failures or []))


            try:
                summ = dict(repeat_usage or {})
                real_cost += float(summ.get("total_cost", 0.0))
                in_tok += int(summ.get("total_input_tokens", 0) or 0)
                out_tok += int(summ.get("total_output_tokens", 0) or 0)
                COST_ACCUM["cache_hits"] += int(summ.get("cache_hits", 0) or 0)
                COST_ACCUM["cache_misses"] += int(summ.get("cache_misses", 0) or 0)
                response_count += int(summ.get("response_count", 0) or 0)
                output_cap_hits += int(summ.get("output_cap_hits", 0) or 0)
                for reason, count in (summ.get("finish_reason_counts") or {}).items():
                    finish_reason_counts[str(reason)] = (
                        int(finish_reason_counts.get(str(reason), 0))
                        + int(count or 0)
                    )
                for limit, count in (summ.get("max_tokens_counts") or {}).items():
                    max_tokens_counts[str(limit)] = (
                        int(max_tokens_counts.get(str(limit), 0))
                        + int(count or 0)
                    )
            except Exception:
                pass

    merged_sample_details = []
    if collect_failures or collect_sample_details or trace_agent:
        source = sample_details_by_repeat or failure_examples_by_repeat
        merged_sample_details = _merge_sample_details(source)
    if collect_failures:
        failures = sorted(
            merged_sample_details,
            key=lambda ex: (float(ex.get("score", 1.0)), int(ex.get("row_index", 0))),
        )[:max(0, int(max_failure_examples or 0))]

    mean_score = sum(scores) / len(scores)
    loss = 1.0 - mean_score
    global LAST_EVAL_COST
    LAST_EVAL_COST = {"dataset": dataset, "real_cost_usd": round(real_cost, 6),
                      "input_tokens": in_tok, "output_tokens": out_tok,
                      "response_count": response_count,
                      "output_cap_hits": output_cap_hits,
                      "finish_reason_counts": dict(finish_reason_counts)}

    COST_ACCUM["usd"] += real_cost
    COST_ACCUM["in"] += in_tok
    COST_ACCUM["out"] += out_tok
    if not return_details:
        _eval_cache_put(cache_key, loss, failures if collect_failures else None)
    if return_details:
        out = {
            "loss": loss,
            "loss_repeats": loss_repeats,
            "finish_reason_counts": dict(finish_reason_counts),
            "max_tokens_counts": dict(max_tokens_counts),
            "response_count": response_count,
            "output_cap_hits": output_cap_hits,
            "output_cap_rate": (
                output_cap_hits / response_count if response_count else 0.0
            ),
        }
        if collect_failures:
            out["failures"] = failures
        if collect_sample_details or trace_agent:
            out["sample_details"] = merged_sample_details
        return out
    if collect_failures:
        return {"loss": loss, "failures": failures}
    return loss


def _collect_sample_details(results, columns, trace_by_row: dict, max_chars: int):
    """Collect per-row scores and traces."""
    try:
        score_i = columns.index("score")
    except ValueError:
        return []

    def idx(*names):
        for n in names:
            if n in columns:
                return columns.index(n)
        return None

    input_i = idx("inputs", "input", "question")
    pred_i = idx("prediction", "model_output", "output")
    exp_i = idx("expected_output", "right_answer", "answer")

    out = []
    for row_index, row in enumerate(results):
        if row is None:
            continue
        try:
            score = float(row[score_i])
        except Exception:  # noqa: BLE001
            continue
        example = {
            "row_index": row_index,
            "score": score,
            "score_repeats": [score],
            "input": _clip_text(row[input_i] if input_i is not None else "", max_chars),
            "prediction": _clip_text(row[pred_i] if pred_i is not None else "", max_chars),
            "expected": _clip_text(row[exp_i] if exp_i is not None else "", max_chars),
            # Benchmarks expose a final task-level reference only.  This is
            # metadata for downstream prompt optimization; it must not be
            # interpreted as a target for an intermediate workflow operator.
            "expected_scope": "final_ground_truth",
            "target_trace": list((trace_by_row.get(row_index) or {}).get("calls") or []),
        }
        out.append(example)
    return out


def _merge_sample_details(examples_by_repeat: List[List[dict]]) -> List[dict]:
    """Aggregate repeats by stable row identity."""
    grouped = {}
    for repeat_examples in examples_by_repeat:
        for ex in repeat_examples or []:
            row_index = int(ex.get("row_index", 0))
            stable_input = str(ex.get("input", ""))
            key = (row_index, stable_input)
            score = float(ex.get("score", 1.0))
            if key not in grouped:
                grouped[key] = {
                    "scores": [], "best_score": score,
                    "representative": dict(ex),
                }
            g = grouped[key]
            g["scores"].append(score)
            if score < g["best_score"]:
                g["best_score"] = score
                g["representative"] = dict(ex)

    merged = []
    for g in grouped.values():
        representative = dict(g["representative"])
        representative["score_repeats"] = list(g["scores"])
        representative["score"] = sum(g["scores"]) / max(1, len(g["scores"]))
        merged.append(representative)
    merged.sort(key=lambda ex: int(ex.get("row_index", 0)))
    return merged


def _collect_failure_examples(results, columns, max_examples: Optional[int], max_chars: int):
    """Collect low-scoring examples."""
    if max_examples is not None and max_examples <= 0:
        return []
    details = _collect_sample_details(results, columns, {}, max_chars)
    details.sort(key=lambda ex: (float(ex.get("score", 1.0)), int(ex.get("row_index", 0))))
    return details if max_examples is None else details[:max_examples]


def _merge_lowest_score_examples(
    examples_by_repeat: List[List[dict]], max_examples: int
) -> List[dict]:
    """Select the lowest aggregate scores across repeats."""
    if not max_examples or max_examples <= 0:
        return []
    merged = _merge_sample_details(examples_by_repeat)
    merged.sort(key=lambda ex: (float(ex.get("score", 1.0)), int(ex.get("row_index", 0))))
    return merged[:max_examples]


def _summarize_results(results, columns):
    """Compute score/cost in IFIP without asking AFlow benchmarks to write CSV."""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return 0.0, 0.0
    try:
        score_i = columns.index("score")
        cost_i = columns.index("cost")
    except ValueError:
        return 0.0, 0.0
    avg_score = sum(float(r[score_i]) for r in valid_results) / len(valid_results)
    total_cost = sum(float(r[cost_i]) for r in valid_results)
    return avg_score, total_cost


def _clip_text(value, limit: int) -> str:
    text = str(value or "").replace("\r", " ").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _check_dataset(dataset: str):
    if dataset not in SUPPORTED_DATASETS:
        fail("aflow_adapter._check_dataset", "unsupported dataset for RQ2",
             dataset=dataset, supported=SUPPORTED_DATASETS)
