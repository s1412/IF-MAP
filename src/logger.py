"""Run artifacts, metadata, and cost summaries."""

import json
import os
import platform
import random
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import List

import numpy as np

from src.evaluator import aggregate_metrics
from src import llm_cache
from src.paths import project_path, display_path
from src.privacy import public_metadata


_RUN_START_TIME = None


def init_run(experiment_tag: str, results_root: str, seed: int, all_configs: dict):
    """Create a run directory and save config metadata."""
    global _RUN_START_TIME
    _RUN_START_TIME = time.time()

    ts = datetime.now().strftime("%Y_%m%d_%H%M%S")
    run_id = f"{ts}__{uuid.uuid4().hex[:8]}__seed{seed}__{experiment_tag}"
    run_dir = str(project_path(results_root) / run_id)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)


    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


    _dump_yaml(os.path.join(run_dir, "config.yaml"), public_metadata(all_configs))

    # meta.json
    meta = {
        "run_id": run_id,
        "timestamp": ts,
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "packages": _get_package_versions(),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Logger] run_id = {run_id}")
    print(f"[Logger] run_dir = {display_path(run_dir)}")
    return run_id, run_dir


def save_results(run_dir: str, all_predictions: List, configs_snapshot: dict):
    """Save predictions, metrics, and costs."""
    # predictions.jsonl
    pred_path = os.path.join(run_dir, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(json.dumps(_serialize(p), ensure_ascii=False) + "\n")

    # metrics.json
    metrics = aggregate_metrics(all_predictions)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # cost_summary.json
    cost = get_run_cost_summary(_RUN_START_TIME)
    with open(os.path.join(run_dir, "cost_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cost, f, indent=2, ensure_ascii=False)

    print_summary_table(metrics, cost)
    return metrics


def get_run_cost_summary(run_start_time) -> dict:
    """Summarize costs for this run."""
    if run_start_time is None:
        return {}
    try:
        return llm_cache.get_cost_since(run_start_time)
    except Exception:
        return {}


def print_summary_table(metrics: dict, cost: dict = None):
    """Print a compact metric table."""
    if metrics.get("n_traces", 0) == 0:
        print("[Logger] No predictions to summarize.")
        return

    rows = [
        ("Top-1 Accuracy", f"{metrics['top1_accuracy']:.5f} "
                           f"[{metrics['top1_ci'][0]:.5f}, {metrics['top1_ci'][1]:.5f}]"),
        ("Top-3 Accuracy", f"{metrics['top3_accuracy']:.5f}"),
        ("MRR", f"{metrics['mrr']:.5f}"),
        ("Traces evaluated", str(metrics["n_traces"])),
    ]
    if cost:
        rows.append(("New LLM calls", str(cost.get("new_calls", 0))))
        rows.append(("Total cost (USD)", f"${cost.get('cost_usd', 0.0):.5f}"))

    try:
        from rich.console import Console
        from rich.table import Table
        table = Table(title="IF-MAP Run Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for k, v in rows:
            table.add_row(k, v)
        Console().print(table)
    except ImportError:
        print("\n" + "=" * 45)
        print("  IF-MAP Run Summary")
        print("=" * 45)
        for k, v in rows:
            print(f"  {k:<20} {v}")
        print("=" * 45)


def _get_package_versions() -> dict:
    import importlib.metadata

    pkgs = {}
    for name in ["openai", "anthropic", "sentence_transformers", "numpy",
                 "scikit-learn", "pyyaml"]:
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pkgs[name] = "not-installed"
    return pkgs


def _dump_yaml(path: str, obj: dict) -> None:
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
    except Exception:
        with open(path.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)


def _serialize(obj):
    """Convert dataclasses and NumPy values to JSON types."""
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
