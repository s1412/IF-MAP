"""SQLite request cache, provider dispatch, and cost accounting."""

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import List, Dict, Optional
from src.paths import project_path

_DB_PATH: Optional[str] = None
_LOCK = threading.Lock()

_LOCAL = threading.local()


_PRICES = {

    "qwen/qwen3-8b":                    (0.117, 0.455),
    "qwen/qwen3-14b":                   (0.10, 0.24),
    "qwen/qwen3-32b":                   (0.08, 0.28),
    "qwen/qwen3-30b-a3b-instruct-2507": (0.048, 0.193),
    "qwen/qwen3.7-plus":                (0.32, 1.28),
    "qwen/qwen3-235b-a22b-2507":        (0.09, 0.10),
    "qwen/qwen-2.5-72b-instruct":       (0.36, 0.40),
    "qwen/qwen3-embedding-8b":          (0.01, 0.0),
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "deepseek/deepseek-v3.2":           (0.2288, 0.3432),
    "gpt-4o-mini":            (0.15, 0.60),
    "gpt-4o":                 (2.50, 10.00),
    "gpt-4.1-mini":           (0.40, 1.60),
    "gpt-4.1":                (2.00, 8.00),
    "o3-mini":                (1.10, 4.40),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "claude-3-5-haiku":       (0.80, 4.00),
    "claude-3-5-sonnet":      (3.00, 15.00),
    "claude-haiku-4-5":       (1.00, 5.00),
    "claude-sonnet-4-5":      (3.00, 15.00),
}


def init_cache_db(db_path: str) -> None:
    """Initialize the request cache."""
    global _DB_PATH
    _DB_PATH = str(project_path(db_path).resolve())
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = _get_conn()
    with _LOCK:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key            TEXT PRIMARY KEY,
                provider       TEXT,
                model          TEXT,
                request_json   TEXT,
                response_text  TEXT,
                input_tokens   INTEGER,
                output_tokens  INTEGER,
                cost_usd       REAL,
                created_at     REAL
            );
            CREATE INDEX IF NOT EXISTS idx_model   ON llm_cache(model);
            CREATE INDEX IF NOT EXISTS idx_created ON llm_cache(created_at);
            """
        )
        conn.commit()


def _get_conn() -> sqlite3.Connection:
    """Return the thread-local SQLite connection."""
    if _DB_PATH is None:
        raise RuntimeError("Cache DB not initialised — call init_cache_db() first.")
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_DB_PATH, timeout=30.0, check_same_thread=False)
        _LOCAL.conn = conn
    return conn


def _compute_key(provider, model, messages, temperature, max_tokens, kwargs) -> str:
    """Hash the normalized request and namespace."""
    namespace = os.environ.get("IFIP_LLM_CACHE_NAMESPACE", "")
    canonical = json.dumps(
        {
            "namespace":   namespace,
            "provider":    provider,
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "kwargs":      {k: kwargs[k] for k in sorted(kwargs)},
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def call_llm_cached(
    provider: str,
    model: str,
    messages: List[Dict],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    force_refresh: bool = False,
    **kwargs,
) -> Dict:
    """Reuse cached requests unless force_refresh is set."""
    key = _compute_key(provider, model, messages, temperature, max_tokens, kwargs)


    if not force_refresh:
        conn = _get_conn()
        with _LOCK:
            row = conn.execute(
                "SELECT response_text, input_tokens, output_tokens, cost_usd "
                "FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is not None:
            return {
                "content":       row[0],
                "cached":        True,
                "input_tokens":  row[1],
                "output_tokens": row[2],
                "cost_usd":      row[3],
            }


    content, in_tok, out_tok = _call_provider(
        provider, model, messages, temperature, max_tokens, **kwargs
    )
    cost = _compute_cost(provider, model, in_tok, out_tok)


    conn = _get_conn()
    with _LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache "
            "(key, provider, model, request_json, response_text, "
            " input_tokens, output_tokens, cost_usd, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                key, provider, model,
                json.dumps({"messages": messages, "temperature": temperature,
                            "max_tokens": max_tokens, **kwargs}, ensure_ascii=False),
                content, in_tok, out_tok, cost, time.time(),
            ),
        )
        conn.commit()

    return {
        "content":       content,
        "cached":        False,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "cost_usd":      cost,
    }


def _call_provider(provider, model, messages, temperature, max_tokens, **kwargs):
    """Dispatch a request with bounded retries."""
    provider = provider.lower()
    max_retries = kwargs.pop("max_retries", 5)
    retry_delay = kwargs.pop("retry_delay", 5.0)

    last_err = None
    for attempt in range(max_retries):
        try:
            if provider in ("openai", "vllm"):
                return _call_openai(model, messages, temperature, max_tokens, **kwargs)
            elif provider == "anthropic":
                return _call_anthropic(model, messages, temperature, max_tokens, **kwargs)
            else:
                raise NotImplementedError(f"Unknown provider: {provider}")
        except NotImplementedError:
            raise
        except Exception as e:  # noqa: BLE001 — network / rate-limit, retry
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


def _call_openai(model, messages, temperature, max_tokens, **kwargs):
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    if "qwen3" in model.lower() and "extra_body" not in kwargs:
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
    return content, in_tok, out_tok


def _call_anthropic(model, messages, temperature, max_tokens, **kwargs):
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    chat_msgs = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model,
        system="\n\n".join(system_parts) if system_parts else None,
        messages=chat_msgs,
        temperature=temperature,
        max_tokens=max_tokens or 4096,
        **kwargs,
    )
    content = "".join(block.text for block in resp.content if block.type == "text")
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    return content, in_tok, out_tok


def _compute_cost(provider, model, input_tokens, output_tokens) -> float:
    """Estimate token cost from the price table."""

    key = None
    for name in _PRICES:
        if model.startswith(name) and (key is None or len(name) > len(key)):
            key = name
    if key is None:
        from src.errors import report_invalid
        report_invalid("llm_cache._compute_cost",
                       "model not in price table — cost recorded as 0",
                       model=model, provider=provider)
        return 0.0
    in_price, out_price = _PRICES[key]
    return input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price


def cache_stats() -> Dict:
    """Summarize cache entries and costs."""
    conn = _get_conn()
    with _LOCK:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), "
            "COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0) FROM llm_cache"
        ).fetchone()
    return {
        "entries":       row[0],
        "input_tokens":  row[1],
        "output_tokens": row[2],
        "total_cost":    row[3],
    }


def get_cost_since(start_time: float) -> Dict:
    """Summarize request costs since a timestamp."""
    conn = _get_conn()
    with _LOCK:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), "
            "COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0) "
            "FROM llm_cache WHERE created_at >= ?",
            (start_time,),
        ).fetchone()
    return {
        "new_calls":     row[0],
        "input_tokens":  row[1],
        "output_tokens": row[2],
        "cost_usd":      row[3],
    }


def clear_cache_for_model(model: str) -> int:
    """Delete cache entries for one model."""
    conn = _get_conn()
    with _LOCK:
        cur = conn.execute("DELETE FROM llm_cache WHERE model = ?", (model,))
        conn.commit()
        return cur.rowcount
