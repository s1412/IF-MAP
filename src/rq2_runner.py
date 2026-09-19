"""RQ2 optimization, optional RQ3 model swaps, and reporting."""

import os
import csv
import json
import pickle
import time
import gc
import hashlib
import contextlib
import re
import subprocess
import urllib.error
import urllib.request
import shutil
import copy
import random
import math
from types import SimpleNamespace
from typing import Dict, List, Tuple, Callable, Optional

from src.score import compute_scores, normalize_main_metric, POTENTIAL_METRICS
from src.rq2_perturbation import run_aflow_perturbation, progress_bar
from src.textgrad_opt import textgrad_optimize_agent
from src import aflow_adapter, llm_cache
from src.paths import project_path, display_path


_TEXTGRAD_OPTIMIZER_VERSION = "reversal-examples-meanabs005-devbc0025-v35"


def _test_entry_point(sample_input) -> str:
    """Extract only a code task's public function name; never persist prompts."""
    text = str(sample_input or "")
    patterns = (
        r'["\']entry_point["\']\s*:\s*["\']([A-Za-z_]\w*)["\']',
        r'\bdef\s+([A-Za-z_]\w*)\s*\(',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def _compact_test_samples(details: List[dict], query_indices: List[int]) -> List[dict]:
    """Persist anonymous TEST identities and scores without inputs or outputs."""
    compact = []
    for row in details or []:
        local_index = int(row.get("row_index", -1))
        if local_index < 0:
            continue
        task_index = (
            int(query_indices[local_index])
            if local_index < len(query_indices) else local_index
        )
        compact.append({
            "task_index": task_index,
            "entry_point": _test_entry_point(row.get("input", "")),
            "score": float(row.get("score", 0.0)),
        })
    compact.sort(key=lambda item: item["task_index"])
    return compact


def _paired_test_samples(baseline: List[dict], optimized: List[dict]) -> List[dict]:
    """Build per-task paired score deltas (positive means improvement)."""
    left = {int(row["task_index"]): row for row in baseline or []}
    right = {int(row["task_index"]): row for row in optimized or []}
    paired = []
    for task_index in sorted(set(left) & set(right)):
        base = left[task_index]
        trial = right[task_index]
        baseline_score = float(base.get("score", 0.0))
        optimized_score = float(trial.get("score", 0.0))
        paired.append({
            "task_index": task_index,
            "entry_point": trial.get("entry_point") or base.get("entry_point") or "",
            "baseline_score": baseline_score,
            "optimized_score": optimized_score,
            "delta_score": optimized_score - baseline_score,
        })
    return paired


def _dataset_config(dataset: str, cfg: dict) -> dict:
    """Resolve dataset overrides without mutating the shared configuration."""
    overrides = (cfg.get("per_dataset_overrides") or {}).get(dataset)
    if not overrides:
        return cfg
    resolved = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(resolved.get(key), dict):
            resolved[key].update(copy.deepcopy(value))
        else:
            resolved[key] = copy.deepcopy(value)
    return resolved


def run_rq2(run_dir: str, encoder, cfg: dict, dry_run: bool) -> dict:
    """Run the configured dataset sequence."""
    rq2_dir = os.path.join(run_dir, "rq2")
    os.makedirs(rq2_dir, exist_ok=True)

    import shutil


    seed = cfg.get("seed_results_dir") if cfg.get("allow_cross_run_seed_results") else None
    seeded_datasets = cfg.get("seed_completed_datasets")
    seeded_datasets = (
        None if seeded_datasets is None
        else {str(item) for item in seeded_datasets}
    )
    per_dataset = []
    preserved = {str(item) for item in (cfg.get("preserve_completed_datasets") or [])}
    shared_cfg = cfg
    for dataset in shared_cfg["datasets"]:
        cfg = _dataset_config(dataset, shared_cfg)
        jp = os.path.join(rq2_dir, f"{dataset}.json")
        if dataset in preserved:
            # Explicit user-authorized legacy results stay byte-for-byte intact.
            # Refresh rank metrics only in memory; no file backfill or GPUs.
            rec = _read_preserved_completed_dataset(rq2_dir, dataset, cfg)
            _refresh_record_metric_evaluations(rec, cfg)
            print(f"\n[RQ2 FROZEN] {dataset}: verified complete; keeping original "
                  "protocol/results read-only, no recomputation or backfill", flush=True)
            per_dataset.append(rec)
            continue
        dataset_seed = (
            seed if seed and (
                seeded_datasets is None or dataset in seeded_datasets)
            else None
        )


        if not os.path.exists(jp) and dataset_seed:
            sp = os.path.join(dataset_seed, f"{dataset}.json")
            if os.path.exists(sp):
                with open(sp, encoding="utf-8") as f:
                    seed_record = json.load(f)
                _require_compatible_legacy_selector(seed_record, cfg)
                seed_checkpoint_path = os.path.join(
                    dataset_seed, f"{dataset}_stage_checkpoint.json")
                seed_checkpoint = {}
                if os.path.exists(seed_checkpoint_path):
                    with open(seed_checkpoint_path, encoding="utf-8") as f:
                        seed_checkpoint = json.load(f)
                seed_is_complete = bool(
                    seed_record.get("dataset") == dataset
                    and seed_checkpoint.get("dataset") == dataset
                    and seed_checkpoint.get("complete") is True
                    and seed_checkpoint.get("stage") == "complete"
                    and _record_matches_requested_dev_splits(seed_record, cfg)
                    and _record_matches_requested_stages(seed_record, cfg)
                )
                if seed_is_complete:
                    copied = []
                    for suffix in (
                        ".json", ".csv", "_perturb.pkl",
                        "_stage_checkpoint.json",
                    ):
                        source = os.path.join(dataset_seed, f"{dataset}{suffix}")
                        target = os.path.join(rq2_dir, f"{dataset}{suffix}")
                        if os.path.exists(source) and not os.path.exists(target):
                            shutil.copy2(source, target)
                            copied.append(os.path.basename(target))
                    print(
                        f"\n[RQ2 MIGRATE] {dataset}: verified complete; "
                        f"copied from {dataset_seed}: {', '.join(copied)}",
                        flush=True,
                    )
                else:
                    print(
                        f"\n[RQ2 MIGRATE WARN] {dataset}: source bundle is not "
                        "verified complete/currently authorized; not copied",
                        flush=True,
                    )

        if os.path.exists(jp):
            with open(jp, encoding="utf-8") as f:
                rec = json.load(f)
            # Unknown legacy selections are history, not an implicit match to
            # today's selector. Fail before backfill, file writes or a rerun.
            _require_compatible_legacy_selector(rec, cfg)
            # Check before normalization: never relabel an old reduced DEV run
            # (e.g. MATH 80/50/40) as a full-DEV result and skip its evaluation.
            if _record_matches_requested_dev_splits(rec, cfg):
                if _normalize_record_sample_counts(rec, cfg):
                    with open(jp, "w", encoding="utf-8") as f:
                        json.dump(rec, f, ensure_ascii=False, indent=2)
                if _record_ensure_derived_metrics(rec, rq2_dir, dataset, cfg):
                    with open(jp, "w", encoding="utf-8") as f:
                        json.dump(rec, f, ensure_ascii=False, indent=2)
                    _write_dataset_table(rq2_dir, rec)
                if _record_matches_requested_stages(rec, cfg):
                    print(f"\n[RQ2 RESUME] {dataset}: matching completed result; skipped ({jp})", flush=True)
                    per_dataset.append(rec)
                    _print_dataset_report(rec)
                    continue
            print(f"\n[RQ2 RESUME] {dataset}: split or stage config changed; recomputing ({jp})", flush=True)
        _prepare_dataset_runtime(dataset, cfg)
        try:
            rec = run_rq2_for_dataset(dataset, rq2_dir, encoder, cfg, dry_run)
            per_dataset.append(rec)
            _write_dataset_table(rq2_dir, rec)
            _print_dataset_report(rec)
        finally:
            _cleanup_after_dataset(dataset, cfg)

    summary = _aggregate(per_dataset)
    _write_summary(rq2_dir, per_dataset, summary)
    ib = (summary.get("by_metric") or {}).get("g_norm") or {}
    print(f"\n✓ RQ2 done. Hit rate (selected==oracle) = "
          f"{_fmt_pct(summary['hit_rate'])} ; G-norm mean Spearman = {summary['mean_spearman']:.5f} ; "
          f"mean Kendall τ-b = {_fmt_kendall(ib.get('mean_kendall_tau_b'))}")
    print(f"  Tables in: {rq2_dir}")
    return summary


def _read_preserved_completed_dataset(rq2_dir: str, dataset: str, cfg: dict) -> dict:
    """Fail closed instead of rerunning or modifying an explicitly frozen dataset."""
    record_path = os.path.join(rq2_dir, f"{dataset}.json")
    checkpoint_path = _dataset_checkpoint_path(rq2_dir, dataset)
    if not os.path.isfile(record_path) or not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"Frozen dataset bundle is missing: {dataset}")
    with open(record_path, encoding="utf-8") as stream:
        record = json.load(stream)
    with open(checkpoint_path, encoding="utf-8") as stream:
        checkpoint = json.load(stream)
    if (record.get("dataset") != dataset or checkpoint.get("dataset") != dataset
            or checkpoint.get("complete") is not True
            or checkpoint.get("stage") != "complete"
            or not record.get("rows")):
        raise RuntimeError(f"Frozen dataset is not verified complete: {dataset}")
    if not _record_matches_requested_stages(
            record, cfg, allow_historical_selector=True):
        raise RuntimeError(f"Frozen dataset does not satisfy requested stages/TEST repeats: {dataset}")
    return record


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)


def _se(xs: List[float]) -> float:
    """Standard error of the mean."""
    return _std(xs) / (len(xs) ** 0.5) if xs else 0.0


def _textgrad_result_status(result: Optional[dict]) -> Tuple[bool, Optional[str], int, int]:
    """Return compatibility counters; TextGrad no longer has an invalid state."""
    result = result or {}
    history = result.get("history") or []
    candidates = [
        candidate
        for step in history
        for candidate in (step.get("candidates") or [])
    ]
    generated = int(result.get("generated_candidate_count", len(candidates)) or 0)
    evaluated = int(result.get(
        "evaluated_candidate_count",
        sum(candidate.get("L") is not None for candidate in candidates),
    ) or 0)
    return True, None, generated, evaluated


def _requested_stage_flags(cfg: dict) -> Tuple[bool, bool]:
    run_rq2_textgrad = bool(cfg.get("run_rq2_textgrad", True))
    run_rq3_model_swap = (
        bool(cfg.get("run_rq3_model_swap", True))
        and bool((cfg.get("rq3_model_swap") or {}).get("enabled", True))
    )
    return run_rq2_textgrad, run_rq3_model_swap


def _recorded_primary_metric(rec: dict) -> Optional[str]:
    """Recover a selector only from explicit metadata or an unambiguous rank.

    A legacy ``i_best`` declaration is not normalized to today's G-norm alias:
    this function describes the recorded decision, not a new configuration.
    Equal rankings under multiple known criteria are deliberately ambiguous.
    """
    known = (*POTENTIAL_METRICS, "i_best", "i_self", "pert_top3",
             "pert_abs_top3", "pert_abs_margin_0p01", "pert_abs_margin_0p005")
    for field in ("primary_metric", "main_metric"):
        if rec.get(field) is not None:
            return rec[field] if rec[field] in known else None
    rows = rec.get("rows") or []
    if len(rows) < 2 or not rec.get("selected"):
        return None
    try:
        ordered = sorted(rows, key=lambda r: int(r["potential_rank"]))
        if [int(r["potential_rank"]) for r in ordered] != list(range(1, len(rows) + 1)):
            return None
        agents = [r["agent"] for r in ordered]
        if len(set(agents)) != len(rows) or agents[0] != rec["selected"]:
            return None
        if any(r.get(metric) is None for r in ordered
               for metric in (*POTENTIAL_METRICS, "i_best")):
            return None
        matches = []
        for metric in known:
            if any(r.get(metric) is None for r in ordered):
                continue
            values = [float(r[metric]) for r in ordered]
            # Tied scores cannot establish which old tie-break was used.
            if (all(math.isfinite(v) for v in values)
                    and all(a > b for a, b in zip(values, values[1:]))):
                matches.append(metric)
        return matches[0] if len(matches) == 1 else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _require_compatible_legacy_selector(rec: dict, cfg: dict) -> None:
    """Do not automatically rerun or relabel ambiguous historical results."""
    if rec.get("primary_metric") is not None:
        return
    recorded = _recorded_primary_metric(rec)
    requested = normalize_main_metric(cfg.get("main_metric", "g_norm"))
    if recorded != requested:
        raise RuntimeError(
            f"Legacy dataset {rec.get('dataset', '?')} has recorded selector "
            f"{recorded or 'unknown/ambiguous'}, not verified {requested}. "
            "Use preserve_completed_datasets for explicit read-only history, "
            "or supply verified selector provenance before normal resume. "
            "No backfill or automatic rerun was performed.")


def _record_matches_requested_stages(
        rec: dict, cfg: dict, *, allow_historical_selector: bool = False) -> bool:
    req_rq2, req_rq3 = _requested_stage_flags(cfg)
    rec_rq2 = bool(rec.get("rq2_textgrad_enabled", True))
    rec_rq3 = bool(rec.get("rq3_model_swap_enabled", bool(rec.get("rq3_model_swap"))))
    accepted_migrations = {
        str(item) for item in
        (cfg.get("accept_migrated_completed_datasets") or [])
    }
    accept_legacy_version = str(rec.get("dataset")) in accepted_migrations
    if rec_rq2 != req_rq2 or rec_rq3 != req_rq3:
        return False
    # Only the explicitly frozen read path may accept an unrelated/unknown
    # historical selector. Normal resume must establish the recorded criterion.
    if (not allow_historical_selector
            and _recorded_primary_metric(rec)
            != normalize_main_metric(cfg.get("main_metric", "g_norm"))):
        return False
    if req_rq2:
        if (rec.get("textgrad_optimizer_version") != _TEXTGRAD_OPTIMIZER_VERSION
                and not accept_legacy_version):
            return False
        if bool(rec.get("textgrad_simple_control_enabled", False)) != bool(
                cfg.get("textgrad_simple_control_enabled", False)):
            return False
        if cfg.get("textgrad_simple_control_enabled", False):
            if "textgrad_simple_control" not in rec:
                return False
            if "simple_control_final_test_repeats" not in rec:
                return False
        expected_rq2_repeats = max(
            1,
            int(cfg.get("final_test_repeats_total",
                        cfg.get("final_test_repeats", 1)) or 1),
        )
        rec_rq2_repeats = int((rec.get("final_test_repeats") or {}).get("n_repeats") or 1)


        if rec_rq2_repeats < expected_rq2_repeats:
            return False
    if req_rq3 and not rec.get("rq3_model_swap"):
        return False
    if req_rq3:
        expected_rq3_repeats = max(
            1,
            int(((cfg.get("rq3_model_swap") or {}).get("final_test_repeats_total", 1)) or 1),
        )
        for rr in (rec.get("rq3_model_swap") or {}).values():
            if int(rr.get("n_repeats") or 1) < expected_rq3_repeats:
                return False
    return True


@contextlib.contextmanager
def _temporary_final_repeat_cache(disable: bool):
    """Temporarily disable caches for extra final TEST repeats.

    The first TEST run keeps the normal run cache semantics and is counted as
    repeat #1.  Extra repeats default to cache-off so they measure independent
    stochastic workflow executions instead of replaying the same deterministic
    cached responses.
    """
    if not disable:
        yield
        return
    keys = ("IFIP_EVAL_CACHE", "AFLOW_LLM_CACHE", "AFLOW_LLM_CACHE_NONDETERMINISTIC")
    old = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["IFIP_EVAL_CACHE"] = "0"
        os.environ["AFLOW_LLM_CACHE"] = "0"
        os.environ.pop("AFLOW_LLM_CACHE_NONDETERMINISTIC", None)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_final_test_once(
    dataset: str,
    original_prompts: Dict[str, str],
    opt: Dict[str, dict],
    agent_names: List[str],
    query_indices: List[int],
    eval_model: str,
    n_repeats: int,
    concurrent_batch: int,
    dry_run: bool,
    repeat_idx: int,
    show_progress: bool = True,
    initial_repeat: Optional[dict] = None,
    on_update: Optional[Callable[[dict], None]] = None,
    intervention_targets: Optional[Dict[str, str]] = None,
) -> dict:
    _nt = len(agent_names) + 1
    result = dict(initial_repeat or {})
    result["repeat"] = repeat_idx
    agents = dict(result.get("agents") or {})
    result["agents"] = agents
    completed = (1 if result.get("baseline_L") is not None else 0) + sum(
        1 for a in agent_names if a in agents and agents[a].get("optimized_L") is not None
    )

    if result.get("baseline_L") is None or not result.get("baseline_samples"):
        baseline_detail = aflow_adapter.evaluate_with_prompts(
            dataset=dataset, agent_prompts=original_prompts, query_indices=query_indices,
            model=eval_model, n_repeats=n_repeats,
            concurrent_batch=concurrent_batch, dry_run=dry_run, split="test",
            return_details=True, collect_sample_details=True,
            failure_example_chars=2000,
        )
        baseline_L = float(baseline_detail["loss"])
        result["baseline_L"] = baseline_L
        result["baseline_samples"] = _compact_test_samples(
            baseline_detail.get("sample_details") or [], query_indices)
        if on_update:
            on_update(result)
        completed = 1
    else:
        baseline_L = float(result["baseline_L"])

    if show_progress:
        progress_bar(f"{dataset} eval(TEST r{repeat_idx})", completed, _nt)
    for _i, agent in enumerate(agent_names, start=1):
        if (agent in agents
                and agents[agent].get("optimized_L") is not None
                and agents[agent].get("paired_sample_deltas") is not None):
            continue
        target_agent = (intervention_targets or {}).get(agent, agent)
        best_prompt = opt[agent]["best_prompt"]
        if best_prompt == original_prompts[target_agent]:
            # The intervention is exactly identical to baseline.  Re-running a
            # stochastic workflow here only manufactures noise, so every repeat
            # inherits its own baseline loss and has exactly zero improvement.
            agents[agent] = {
                "optimized_L": float(baseline_L),
                "delta_L": 0.0,
                "reused_baseline_identical_prompt": True,
                "paired_sample_deltas": _paired_test_samples(
                    result.get("baseline_samples") or [],
                    result.get("baseline_samples") or [],
                ),
            }
        else:
            trial = dict(original_prompts)
            trial[target_agent] = best_prompt
            trial_detail = aflow_adapter.evaluate_with_prompts(
                dataset=dataset, agent_prompts=trial, query_indices=query_indices,
                model=eval_model, n_repeats=n_repeats,
                concurrent_batch=concurrent_batch, dry_run=dry_run, split="test",
                return_details=True, collect_sample_details=True,
                failure_example_chars=2000,
            )
            L_test = float(trial_detail["loss"])
            trial_samples = _compact_test_samples(
                trial_detail.get("sample_details") or [], query_indices)
            agents[agent] = {
                "optimized_L": L_test,
                "delta_L": float(baseline_L) - float(L_test),
                "reused_baseline_identical_prompt": False,
                "paired_sample_deltas": _paired_test_samples(
                    result.get("baseline_samples") or [], trial_samples),
            }
        if on_update:
            on_update(result)
        if show_progress:
            done_now = (1 if result.get("baseline_L") is not None else 0) + sum(
                1 for a in agent_names if a in agents and agents[a].get("optimized_L") is not None
            )
            progress_bar(f"{dataset} eval(TEST r{repeat_idx})", done_now, _nt)
    return result


def _summarize_final_test_repeats(
    repeats: List[dict],
    selected: Optional[str],
    agent_names: List[str],
) -> dict:
    baseline_vals = [float(r["baseline_L"]) for r in repeats]
    agents = {}
    for agent in agent_names:
        opt_vals = [float(r["agents"][agent]["optimized_L"]) for r in repeats]
        delta_vals = [float(r["agents"][agent]["delta_L"]) for r in repeats]
        agents[agent] = {
            "optimized_L_mean": _mean(opt_vals),
            "optimized_L_std": _std(opt_vals),
            "optimized_L_se": _se(opt_vals),
            "delta_L_mean": _mean(delta_vals),
            "delta_L_std": _std(delta_vals),
            "delta_L_se": _se(delta_vals),
            "per_repeat_optimized_L": opt_vals,
            "per_repeat_delta_L": delta_vals,
            "per_repeat_paired_sample_deltas": [
                list((r["agents"][agent].get("paired_sample_deltas") or []))
                for r in repeats
            ],
        }

    repeat_rows = []
    for r in repeats:
        deltas = {a: float(r["agents"][a]["delta_L"]) for a in agent_names}
        oracle_agent = max(deltas, key=lambda a: deltas[a])
        oracle_delta = deltas[oracle_agent]
        repeat_rows.append({
            "repeat": int(r["repeat"]),
            "baseline_L": float(r["baseline_L"]),
            "selected_delta_L": deltas.get(selected) if selected else None,
            "random_baseline_delta_L": _mean(list(deltas.values())),
            "oracle_agent": oracle_agent,
            "oracle_delta_L": oracle_delta,
            "selected_is_oracle": bool(
                selected in deltas and abs(deltas[selected] - oracle_delta) < 1e-9
            ),
            "agents": {
                a: {
                    "optimized_L": float(r["agents"][a]["optimized_L"]),
                    "delta_L": float(r["agents"][a]["delta_L"]),
                    "paired_sample_deltas": list(
                        r["agents"][a].get("paired_sample_deltas") or []),
                }
                for a in agent_names
            },
        })

    mean_deltas = {a: agents[a]["delta_L_mean"] for a in agent_names}
    oracle_agent_by_mean = max(mean_deltas, key=lambda a: mean_deltas[a])
    oracle_delta_by_mean = mean_deltas[oracle_agent_by_mean]
    selected_delta_mean = mean_deltas.get(selected) if selected else None
    return {
        "n_repeats": len(repeats),
        "include_original_first_repeat": True,
        "baseline_L_mean": _mean(baseline_vals),
        "baseline_L_std": _std(baseline_vals),
        "baseline_L_se": _se(baseline_vals),
        "selected": selected,
        "selected_delta_mean": selected_delta_mean,
        "selected_delta_std": agents[selected]["delta_L_std"] if selected in agents else None,
        "selected_delta_se": agents[selected]["delta_L_se"] if selected in agents else None,
        "oracle_agent_by_mean": oracle_agent_by_mean,
        "oracle_delta_by_mean": oracle_delta_by_mean,
        "selected_is_oracle_by_mean": bool(
            selected in mean_deltas and abs(selected_delta_mean - oracle_delta_by_mean) < 1e-9
        ),
        "selected_hit_rate_over_repeats": _mean([
            1.0 if r["selected_is_oracle"] else 0.0 for r in repeat_rows
        ]),
        "agents": agents,
        "repeats": repeat_rows,
    }


def _sanitize_model_name(model: str) -> str:
    return (model or "model").replace("/", "__").replace(":", "_").replace(".", "_")


def _service_cfg(cfg: dict) -> dict:
    return cfg.get("local_qwen3_service") or {}


def _dataset_runtime_mode(dataset: str, cfg: dict) -> str:
    service = _service_cfg(cfg)
    use_16k_for = set(service.get("use_16k_for") or [])
    return "16k" if dataset in use_16k_for else "8k"


def _run_service_script(script: Optional[str], label: str, timeout_s: int = 120) -> None:
    if not script:
        return
    script = str(project_path(script).resolve())
    if not os.path.exists(script):
        print(f"  [runtime WARN] {label}: script not found: {display_path(script)}", flush=True)
        return
    try:
        subprocess.run(
            ["bash", script],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"  [runtime WARN] {label}: timeout after {timeout_s}s ({display_path(script)})", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  [runtime WARN] {label}: exit={e.returncode} ({display_path(script)})", flush=True)


def _http_ok(url: str, timeout_s: float = 5.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _http_json(url: str, timeout_s: float = 5.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if not (200 <= int(resp.status) < 300):
                return None
            value = json.loads(resp.read().decode("utf-8"))
            return value if isinstance(value, dict) else None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _active_local_qwen_replicas(cfg: dict) -> int:
    """Return router-advertised healthy replicas, capped by desired count."""
    service = _service_cfg(cfg)
    desired = max(1, int(service.get("desired_replicas", 1) or 1))
    health_url = service.get("health_url") or "http://127.0.0.1:8000/health"
    status = _http_json(health_url)
    if not status:
        return 0
    backends = list(status.get("backends") or [])
    healthy = sum(
        _http_ok(str(url).rstrip("/") + "/models") for url in backends
    )
    return min(desired, healthy)


def _scaled_local_concurrency(configured: int, cfg: dict) -> Tuple[int, int, int]:
    """Scale concurrency by both manual scale and live replica availability."""
    service = _service_cfg(cfg)
    desired = max(1, int(service.get("desired_replicas", 1) or 1))
    active = (_active_local_qwen_replicas(cfg)
              if service.get("enabled", False) else desired)
    if service.get("enabled", False) and active <= 0:
        raise RuntimeError("local qwen3-8b router has no healthy backend replicas")
    if not service.get("enabled", False):
        active = desired
    manual_scale = float(cfg.get("local_concurrent_scale", 1.0) or 1.0)
    if manual_scale <= 0:
        raise ValueError("local_concurrent_scale must be > 0")
    effective = max(1, int(configured * manual_scale * active / desired))
    return effective, active, desired


def _wait_local_qwen_service(service: dict, mode: str) -> bool:
    timeout_s = int(service.get("wait_timeout_s", 420) or 420)
    grace_s = max(0, int(service.get("health_grace_s", 30) or 0))
    deadline = time.time() + timeout_s
    health_url = service.get("health_url") or "http://127.0.0.1:8000/health"
    last_status = None

    def healthy_backend_urls(status: Optional[dict]) -> List[str]:
        dynamic_backends = list((status or {}).get("backends") or [])
        configured_dynamic = list(
            (status or {}).get("configured_backends") or dynamic_backends
        )
        configured_backends = list(service.get("backend_model_urls") or [])
        backend_urls = dynamic_backends or configured_backends
        all_healthy = (
            bool(status)
            and bool(backend_urls)
            and len(dynamic_backends) >= len(configured_dynamic)
            and all(
                _http_ok(str(url).rstrip("/") + (
                    "" if str(url).rstrip("/").endswith("/models") else "/models"
                ))
                for url in backend_urls
            )
        )
        return backend_urls if all_healthy else []

    def wait_until(end_time: float) -> Tuple[Optional[dict], List[str]]:
        nonlocal last_status
        while time.time() < end_time:
            status = _http_json(health_url)
            last_status = status
            backend_urls = healthy_backend_urls(status)
            if backend_urls:
                return status, backend_urls
            time.sleep(5)
        return last_status, []

    _, backend_urls = wait_until(deadline)
    if backend_urls:
        print(
            f"  [runtime] local qwen3-8b {mode} healthy replicas="
            f"{len(backend_urls)}/{int(service.get('desired_replicas', len(backend_urls)) or len(backend_urls))}",
            flush=True,
        )
        return True

    if grace_s:
        print(
            f"  [runtime] local qwen3-8b {mode} startup reached {timeout_s}s; "
            f"performing final {grace_s}s health grace check",
            flush=True,
        )
        _, backend_urls = wait_until(time.time() + grace_s)
        if backend_urls:
            print(
                f"  [runtime] local qwen3-8b {mode} became healthy during grace: "
                f"replicas={len(backend_urls)}/"
                f"{int(service.get('desired_replicas', len(backend_urls)) or len(backend_urls))}",
                flush=True,
            )
            return True

    healthy = list((last_status or {}).get("backends") or [])
    if healthy:
        desired = int(service.get("desired_replicas", len(healthy)) or len(healthy))
        print(
            f"  [runtime WARN] local qwen3-8b {mode} degraded startup: "
            f"healthy replicas={len(healthy)}/{desired}; concurrency will be scaled",
            flush=True,
        )
        return True
    print(
        f"  [runtime WARN] local qwen3-8b {mode} service not healthy after "
        f"{timeout_s + grace_s}s",
        flush=True,
    )
    return False


def _prepare_dataset_runtime(dataset: str, cfg: dict) -> None:
    """Switch local qwen3-8b context profile before a dataset.

    The qwen3-8b executor itself is configured in aflow/config/config2.yaml as
    qwen3-8b-local-exec -> http://127.0.0.1:8000/v1.  This function only
    chooses which router/backend profile is alive: 8k for normal datasets,
    16k for MATH.
    """
    service = _service_cfg(cfg)
    if not service.get("enabled", False) or not service.get("auto_switch", False):
        return
    mode = _dataset_runtime_mode(dataset, cfg)
    print(f"  [runtime] preparing local qwen3-8b service for {dataset}: context={mode}", flush=True)
    if mode == "16k":
        _run_service_script(service.get("stop_8k"), "stop 8k")
        _run_service_script(service.get("start_16k"), "start 16k", timeout_s=180)
    else:
        _run_service_script(service.get("stop_16k"), "stop 16k")
        _run_service_script(service.get("start_8k"), "start 8k", timeout_s=180)
    _wait_local_qwen_service(service, mode)


def _cleanup_after_dataset(dataset: str, cfg: dict) -> None:
    service = _service_cfg(cfg)
    if not service.get("cleanup_between_datasets", False):
        return
    # AFlow may create per-dataset eval log directories even when detailed CSV
    # logging is disabled.  They are diagnostic scratch state, not caches/checkpoints.
    eval_log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "aflow",
        ".eval_logs",
        dataset,
    )
    if os.path.isdir(eval_log_dir):
        shutil.rmtree(eval_log_dir, ignore_errors=True)
    gc.collect()
    print(f"  [cleanup] {dataset}: cleared transient eval logs and Python GC state", flush=True)


def _dataset_checkpoint_path(rq2_dir: str, dataset: str) -> str:
    return os.path.join(rq2_dir, f"{dataset}_stage_checkpoint.json")


def _checkpoint_signature(
    dataset: str,
    agent_names: List[str],
    dev_indices: List[int],
    textgrad_indices: List[int],
    textgrad_select_indices: List[int],
    test_indices: List[int],
    concurrent_batch: int,
    cfg: dict,
) -> str:
    rq3 = cfg.get("rq3_model_swap") or {}
    payload = {
        "dataset": dataset,
        "agents": agent_names,
        "dev_indices": dev_indices,
        "textgrad_indices": textgrad_indices,
        "textgrad_select_indices": textgrad_select_indices,
        "test_indices": test_indices,
        "eval_model": cfg.get("eval_model"),
        "primary_metric": normalize_main_metric(cfg.get("main_metric", "g_norm")),
        "potential_metrics": list(POTENTIAL_METRICS),
        "n_repeats": cfg.get("n_repeats"),
        "perturb_eval_repeats": cfg.get("perturb_eval_repeats"),
        "n_perturb": cfg.get("n_perturb"),
        "perturb_mix": cfg.get("perturb_mix"),
        "textgrad_mode": cfg.get("textgrad_mode"),
        "textgrad_n_iterations": cfg.get("textgrad_n_iterations"),
        "textgrad_eval_repeats": cfg.get("textgrad_eval_repeats"),
        "textgrad_beam_size": cfg.get("textgrad_beam_size"),
        "textgrad_accept_margin": cfg.get("textgrad_accept_margin"),
        "textgrad_confidence_z": cfg.get("textgrad_confidence_z"),
        "textgrad_min_positive_repeats": cfg.get("textgrad_min_positive_repeats"),
        "textgrad_min_nonworse_repeats": cfg.get("textgrad_min_nonworse_repeats"),
        "textgrad_evidence_confidence_z": cfg.get("textgrad_evidence_confidence_z"),
        "textgrad_stage1_abs_margin": cfg.get("textgrad_stage1_abs_margin"),
        "textgrad_evidence_reuse": cfg.get("textgrad_evidence_reuse"),
        "textgrad_simple_control_enabled": cfg.get(
            "textgrad_simple_control_enabled"),
        "textgrad_simple_control_samples": cfg.get(
            "textgrad_simple_control_samples"),
        "textgrad_generation_attempts": cfg.get("textgrad_generation_attempts"),
        "textgrad_candidate_min_length_ratio": cfg.get(
            "textgrad_candidate_min_length_ratio"),
        "textgrad_candidate_max_length_ratio": cfg.get(
            "textgrad_candidate_max_length_ratio"),
        "textgrad_devc_max_mean_regression": cfg.get(
            "textgrad_devc_max_mean_regression"),
        "textgrad_devc_regression_z": cfg.get("textgrad_devc_regression_z"),
        "textgrad_devc_net_gain_margin": cfg.get(
            "textgrad_devc_net_gain_margin"),
        "textgrad_devb_rescue_max_regression": cfg.get(
            "textgrad_devb_rescue_max_regression"),
        "textgrad_devbc_rescue_mean_margin": cfg.get(
            "textgrad_devbc_rescue_mean_margin"),
        "textgrad_candidate_pool_extra": cfg.get("textgrad_candidate_pool_extra"),
        "textgrad_semantic_similarity_threshold": cfg.get(
            "textgrad_semantic_similarity_threshold"),
        "textgrad_tokenizer_path": cfg.get("textgrad_tokenizer_path"),
        "textgrad_meta_max_tokens": cfg.get("textgrad_meta_max_tokens"),
        "textgrad_reversal_example_max_tokens": cfg.get(
            "textgrad_reversal_example_max_tokens"),
        "textgrad_early_stop_patience": cfg.get("textgrad_early_stop_patience"),
        "textgrad_optimizer_version": _TEXTGRAD_OPTIMIZER_VERSION,
        "final_test_repeats_total": cfg.get("final_test_repeats_total"),
        "run_rq2_textgrad": cfg.get("run_rq2_textgrad", True),
        "run_rq3_model_swap": cfg.get("run_rq3_model_swap", True),
        "rq3_models": rq3.get("models"),
        "rq3_repeats": rq3.get("final_test_repeats_total"),
        "concurrent_batch": concurrent_batch,
    }
    if cfg.get("textgrad_max_directions") is not None:
        payload["textgrad_max_directions"] = cfg["textgrad_max_directions"]
    if cfg.get("dev_subsample") is not None:
        payload["dev_subsample"] = cfg["dev_subsample"]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_dataset_checkpoint(path: str, signature: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  [checkpoint WARN] failed to read {path}: {e}", flush=True)
        return {}
    if d.get("signature") != signature:
        print(f"  [checkpoint] ignore stale dataset checkpoint (signature changed): {path}", flush=True)
        return {}
    return d


def _save_dataset_checkpoint(path: str, signature: str, **updates) -> None:
    cur = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                cur = json.load(f)
        except Exception:  # noqa: BLE001
            cur = {}
    if cur.get("signature") not in (None, signature):
        cur = {}
    cur.update(updates)
    cur["signature"] = signature
    cur["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def _find_repeat(repeats: List[dict], repeat_idx: int) -> Optional[dict]:
    for r in repeats or []:
        if int(r.get("repeat", -1)) == int(repeat_idx):
            return r
    return None


def _repeat_complete(rep: Optional[dict], agent_names: List[str]) -> bool:
    if (not rep or rep.get("baseline_L") is None
            or not rep.get("baseline_samples")):
        return False
    agents = rep.get("agents") or {}
    return all(
        a in agents
        and agents[a].get("optimized_L") is not None
        and agents[a].get("paired_sample_deltas") is not None
        for a in agent_names
    )


def _upsert_repeat(repeats: List[dict], rep: dict) -> List[dict]:
    out = [r for r in (repeats or []) if int(r.get("repeat", -1)) != int(rep.get("repeat", -2))]
    out.append(rep)
    return sorted(out, key=lambda r: int(r.get("repeat", 0)))


def _resolve_openrouter_concurrent_batch(dataset: str, local_concurrent_batch: int, cfg: dict) -> int:
    """Resolve concurrency for calls that include OpenRouter/API models.

    Local qwen3-8b calls keep ``local_concurrent_batch``.  RQ3 model-swap
    calls contain one OpenRouter override agent, so they can use a separate
    API-side throttle without changing local vLLM pressure.
    """
    per_dataset = cfg.get("per_dataset_openrouter_concurrent_batch") or {}
    if dataset in per_dataset:
        return int(per_dataset[dataset])
    mapping = cfg.get("openrouter_concurrent_batch_map") or {}
    # JSON configs may deserialize integer keys as strings; support both.
    return int(mapping.get(local_concurrent_batch,
                          mapping.get(str(local_concurrent_batch),
                                      local_concurrent_batch)))


def run_rq2_for_dataset(dataset, rq2_dir, encoder, cfg, dry_run) -> dict:
    cfg = _dataset_config(dataset, cfg)
    original_prompts = aflow_adapter.get_original_prompts(dataset)
    goals = aflow_adapter.get_prompt_goals(dataset)
    placeholders = aflow_adapter.get_required_placeholders(dataset)
    agent_names = list(original_prompts.keys())


    #   cfg["per_dataset_samples"] = {"MBPP": [100, 86], ...}
    #   cfg["per_dataset_concurrent_batch"] = {"MBPP": 12, ...}
    seed_dir = cfg.get("seed_results_dir") if cfg.get("allow_cross_run_seed_results") else None
    seeded_datasets = cfg.get("seed_completed_datasets")
    if (seeded_datasets is not None
            and dataset not in {str(item) for item in seeded_datasets}):
        seed_dir = None
    (dev_indices, textgrad_indices, textgrad_select_indices,
     dev_note, textgrad_note, textgrad_select_note,
     requested_dev, requested_textgrad, requested_textgrad_select,
     available_dev) = _resolve_dev_splits(dataset, cfg)
    n_dev = len(dev_indices)
    n_textgrad = len(textgrad_indices)
    n_textgrad_select = len(textgrad_select_indices)
    n_test, requested_test, available_test = _resolve_test_count(dataset, cfg)
    per_dataset_concurrent = cfg.get("per_dataset_concurrent_batch") or {}
    if dataset in per_dataset_concurrent:
        configured_concurrent_batch = int(per_dataset_concurrent[dataset])
    else:
        configured_concurrent_batch = int(cfg["concurrent_batch"])
    # Concurrency is an operational resource control, not an experiment
    # semantic.  Keep the configured value in checkpoint signatures so an
    # interrupted run remains resumable when the number of local replicas is
    # reduced, while every local qwen3-8b evaluation uses the scaled value.
    concurrent_batch, active_replicas, desired_replicas = (
        _scaled_local_concurrency(configured_concurrent_batch, cfg)
    )
    local_concurrent_scale = float(cfg.get("local_concurrent_scale", 1.0) or 1.0)
    openrouter_concurrent_batch = _resolve_openrouter_concurrent_batch(
        dataset, configured_concurrent_batch, cfg)
    skip_failed_candidates = bool((cfg.get("skip_failed_candidates") or {}).get(dataset, False))
    perturb_eval_max_attempts = int(cfg.get("perturb_eval_max_attempts", 3) or 1)
    perturb_eval_repeats = int(cfg.get("perturb_eval_repeats", cfg.get("n_repeats", 1)) or 1)
    keep_failed_perturbations = bool(cfg.get("keep_failed_perturbations", True))
    test_indices = list(range(n_test))
    print(f"\n{'='*60}\n[RQ2 START] {dataset}  DEV(validate)={n_dev}  "
          f"TEST(test)={n_test}  concurrent(local={concurrent_batch} "
          f"[configured={configured_concurrent_batch} x {local_concurrent_scale:g} "
          f"x replicas={active_replicas}/{desired_replicas}], "
          f"openrouter={openrouter_concurrent_batch})\n{'='*60}", flush=True)
    print("  [samples] requested "
          f"DEV-A/TextGrad-B/TextGrad-C/TEST={requested_dev}/{requested_textgrad}/"
          f"{requested_textgrad_select}/{requested_test}; "
          f"available validate/test={available_dev}/{available_test}; "
          f"using DEV-A/TextGrad-B/TextGrad-C/TEST="
          f"{n_dev}/{n_textgrad}/{n_textgrad_select}/{n_test}", flush=True)
    print(f"  [DEV-A] {dev_note}", flush=True)
    print(f"  [TextGrad DEV-B] {textgrad_note}", flush=True)
    print(f"  [TextGrad DEV-C] {textgrad_select_note}", flush=True)
    stage_ckpt_path = _dataset_checkpoint_path(rq2_dir, dataset)
    stage_sig = _checkpoint_signature(
        dataset=dataset,
        agent_names=agent_names,
        dev_indices=dev_indices,
        textgrad_indices=textgrad_indices,
        textgrad_select_indices=textgrad_select_indices,
        test_indices=test_indices,
        concurrent_batch=configured_concurrent_batch,
        cfg=cfg,
    )
    dataset_ckpt = _load_dataset_checkpoint(stage_ckpt_path, stage_sig)

    def _save_stage(**updates):
        _save_dataset_checkpoint(stage_ckpt_path, stage_sig, **updates)

    _save_stage(
        complete=False,
        stage=dataset_ckpt.get("stage", "initialized") if dataset_ckpt else "initialized",
        dataset=dataset,
        agent_names=agent_names,
        n_dev=n_dev,
        n_textgrad=n_textgrad,
        n_textgrad_select=n_textgrad_select,
        n_test=n_test,
        runtime_context=_dataset_runtime_mode(dataset, cfg),
    )
    aflow_adapter.reset_cost()
    _ifip_t0 = time.time()


    import pickle
    pcache = os.path.join(rq2_dir, f"{dataset}_perturb.pkl")
    expected_perturb_config = {
        "evidence_version": "confidence-counterfactual-v7-reversals3-final-gt",
        "n_perturb": int(cfg["n_perturb"]),
        "perturb_mix": dict(cfg["perturb_mix"]),
        "n_repeats": int(perturb_eval_repeats),
        "eval_model": cfg["eval_model"],
        "n_dev": int(n_dev),
    }
    exact_pool = cfg.get("perturb_mix", {}).get("construction_protocol") == "perturbation_pool_exact_v1"
    expected_execution_fingerprint = None
    if exact_pool:
        from src.rq2_perturbation import _pool_execution_fingerprint

        fingerprint_args = dict(
            dataset=dataset, original_prompts=original_prompts, query_indices=dev_indices,
            generator_llm=cfg["generator_llm"], eval_model=cfg["eval_model"], dry_run=dry_run,
        )
        # The outer complete-cache shortcut must bind the same actual inputs
        # as the inner partial-checkpoint path, not only count/model aliases.
        expected_perturb_config["pool_execution_context"] = _pool_execution_fingerprint(
            **fingerprint_args, config=cfg["perturb_mix"],
        )
        expected_execution_fingerprint = _pool_execution_fingerprint(
            **fingerprint_args,
            config={"perturb_mix": cfg["perturb_mix"], "perturb_config": expected_perturb_config,
                    "n_repeats": perturb_eval_repeats,
                    "reference_root": cfg["perturb_mix"].get("reference_root")},
        )
    if not os.path.exists(pcache) and seed_dir:
        sp = os.path.join(seed_dir, f"{dataset}_perturb.pkl")
        if os.path.exists(sp):
            import shutil as _sh; _sh.copy(sp, pcache)
    reuse_complete_perturb = False
    if os.path.exists(pcache):
        with open(pcache, "rb") as f:
            d = pickle.load(f)
        cached_perturb_config = d.get("perturb_config")
        execution_matches = not exact_pool or d.get("execution_fingerprint") == expected_execution_fingerprint
        if cached_perturb_config != expected_perturb_config or not execution_matches:
            old_without_version = dict(cached_perturb_config or {})
            old_without_version.pop("evidence_version", None)
            new_without_version = dict(expected_perturb_config)
            new_without_version.pop("evidence_version", None)
            if not exact_pool and old_without_version == new_without_version and d.get("variants"):
                # Evidence schema changed, but perturbation texts are still the
                # exact historical interventions.  Preserve those variants for
                # a controlled before/after comparison and recompute only the
                # repeat/sample-level evidence required by the new optimizer.
                d = {
                    "complete": False,
                    "dataset": dataset,
                    "variants": d["variants"],
                    "perturb_config": expected_perturb_config,
                    "results": {}, "evidence": {}, "failed_candidates": [],
                }
                with open(pcache, "wb") as f:
                    pickle.dump(d, f)
                print(
                    "  [stage 1/4] reuse variant text; "
                    "rebuild repeated per-row evidence",
                    flush=True,
                )
            else:
                stale = f"{pcache}.stale_{int(time.time())}"
                shutil.move(pcache, stale)
                print(f"  [stage 1/4] incompatible perturbation cache moved to {stale}; recomputing", flush=True)
                d = {"complete": False}

        reuse_complete_perturb = bool(d.get("complete", True))
        if reuse_complete_perturb:
            print(f"  [stage 1/4] reuse perturbation checkpoint ({pcache}), skip recomputation", flush=True)
            results, variants, baseline_L_dev, evidence = (
                d["results"], d["variants"], d["baseline_L_dev"], d["evidence"])
        else:
            done_n = sum(max(0, len(v) - 1) for v in d.get("results", {}).values())
            print(f"  [stage 1/4] resume partial perturbation checkpoint ({done_n} candidates complete); resuming ...",
                  flush=True)
    if not reuse_complete_perturb:
        pool_count_label = (
            f"up to {cfg['n_perturb']} variants (redundant adaptive slots omitted)"
            if cfg.get("perturb_mix", {}).get("construction_protocol")
            == "perturbation_pool_exact_v1"
            else f"{cfg['n_perturb']} variants"
        )
        print(f"  [stage 1/4] perturbations per agent: {pool_count_label}, "
              f"DEV({n_dev}) potential; repeats={perturb_eval_repeats} ...", flush=True)
        results, variants, baseline_L_dev, evidence = run_aflow_perturbation(
            dataset=dataset, original_prompts=original_prompts, encoder=encoder,
            query_indices=dev_indices, n_perturb=cfg["n_perturb"],
            perturb_mix=cfg["perturb_mix"], generator_llm=cfg["generator_llm"],
            eval_model=cfg["eval_model"], n_repeats=perturb_eval_repeats,
            concurrent_batch=concurrent_batch, dry_run=dry_run,
            checkpoint_path=pcache,
            skip_failed_candidates=skip_failed_candidates,
            max_eval_attempts=perturb_eval_max_attempts,
            keep_failed_candidates=keep_failed_perturbations,
            perturb_config=expected_perturb_config,
        )
        failed_candidates = []
        pool_build = {}
        saved_perturb = {}
        if os.path.exists(pcache):
            try:
                with open(pcache, "rb") as f:
                    saved_perturb = pickle.load(f) or {}
                    failed_candidates = saved_perturb.get("failed_candidates", []) or []
                    # Preserve the exact pool's generation/Pilot/adaptive
                    # audit trail when Stage 1 is marked complete.
                    pool_build = saved_perturb.get("pool_build", {}) or {}
            except Exception:  # noqa: BLE001
                failed_candidates = []
                pool_build = {}
                saved_perturb = {}
        # Preserve complete inner checkpoint metadata, including its execution
        # fingerprint and per-sample/repeat baseline, rather than reconstructing
        # a lossy result-only pickle that cannot be safely resumed later.
        saved_perturb.update({
            "results": results, "variants": variants,
            "perturb_config": expected_perturb_config,
            "baseline_L_dev": baseline_L_dev, "evidence": evidence,
            "complete": True, "pool_build": pool_build,
            "failed_candidates": failed_candidates,
        })
        with open(pcache, "wb") as f:
            pickle.dump(saved_perturb, f)
    primary_metric = normalize_main_metric(cfg.get("main_metric", "g_norm"))
    scores = compute_scores(
        perturbation_results=results, agent_variants=variants,
        main_metric=primary_metric,
        regression_method="ridge_with_pca", ridge_lambda=1.0, pca_components=None,
    )
    ranking = sorted(scores.values(), key=lambda s: s.rank)
    selected = ranking[0].agent_name
    _save_stage(
        stage="prompt_probe_done",
        selected=selected,
        primary_metric=primary_metric,
        baseline_L_dev=baseline_L_dev,
        ranking=[
            {
                "agent": s.agent_name,
                "rank": s.rank,
                "i_abs": s.i_abs,
                "mean_abs_delta": s.mean_abs_delta,
                "g_norm": s.g_norm,
            }
            for s in ranking
        ],
    )
    run_rq2_textgrad = bool(cfg.get("run_rq2_textgrad", True))
    run_rq3_model_swap = (
        bool(cfg.get("run_rq3_model_swap", True))
        and bool((cfg.get("rq3_model_swap") or {}).get("enabled", True))
    )
    print(f"  [stages] shared prompt-probe selected={selected}; "
          f"run_rq2_textgrad={run_rq2_textgrad}; "
          f"run_rq3_model_swap={run_rq3_model_swap}", flush=True)
    if run_rq3_model_swap:
        rq3_specs = (cfg.get("rq3_model_swap") or {}).get("models") or []
        rq3_labels = []
        for spec in rq3_specs:
            if isinstance(spec, str):
                rq3_labels.append(spec)
            else:
                rq3_labels.append(f"{spec.get('name') or spec.get('eval_model')}={spec.get('eval_model')}")
        print(f"  [models] base_eval_model={cfg['eval_model']}; "
              f"rq3_swap_models=[{', '.join(rq3_labels)}]; "
              f"rq3_openrouter_concurrent={openrouter_concurrent_batch}", flush=True)

    opt: Dict[str, dict] = dict(dataset_ckpt.get("textgrad") or {})
    simple_control_enabled = bool(
        cfg.get("textgrad_simple_control_enabled", False))
    simple_control_opt: Dict[str, dict] = dict(
        dataset_ckpt.get("textgrad_simple_control") or {})
    # TextGrad v35 obtains direction-consistent reversal examples only from
    # each selected Stage-1 DEV-A evidence item.  Do not admit unrelated
    # baseline-example payloads or any DEV-B/DEV-C/TEST sample content.
    failure_examples = []
    baseline_L_textgrad = dataset_ckpt.get("baseline_L_textgrad")
    baseline_loss_repeats_textgrad = list(
        dataset_ckpt.get("baseline_loss_repeats_textgrad") or [])
    baseline_sample_details_textgrad = []
    baseline_L_textgrad_select = dataset_ckpt.get(
        "baseline_L_textgrad_select")
    baseline_loss_repeats_textgrad_select = list(
        dataset_ckpt.get("baseline_loss_repeats_textgrad_select") or [])
    textgrad_eval_repeats = max(1, int(
        cfg.get("textgrad_eval_repeats", cfg.get("n_repeats", 1)) or 1))
    test_repeat_summary = None
    baseline_L_test = None
    test_optimized_L: Dict[str, float] = {}
    test_delta: Dict[str, float] = {}

    if run_rq2_textgrad:

        print(f"  [stage 2/4] TextGrad: DEV-B({len(textgrad_indices)}) optimize {len(agent_names)} agents "
              f"(selected by {primary_metric} = {selected}) ...", flush=True)
        if baseline_L_textgrad is not None:
            pass
        elif dry_run:
            baseline_L_textgrad = 0.5
            baseline_loss_repeats_textgrad = [0.5] * textgrad_eval_repeats
            baseline_sample_details_textgrad = []
            failure_examples = []
        else:
            detail = aflow_adapter.evaluate_with_prompts(
                dataset=dataset, agent_prompts=original_prompts,
                query_indices=textgrad_indices, model=cfg["eval_model"],
                n_repeats=textgrad_eval_repeats, concurrent_batch=concurrent_batch,
                dry_run=dry_run, split="validate",
                return_details=True,
            )
            baseline_L_textgrad = detail["loss"]
            baseline_loss_repeats_textgrad = list(detail.get("loss_repeats") or [baseline_L_textgrad])
            # Only aggregate loss/repeat statistics are needed.  Per-example
            # outputs, failures, traces, and summaries are intentionally absent.
            baseline_sample_details_textgrad = []
            failure_examples = []
        _save_stage(
            complete=False,
            stage="textgrad_baseline_done",
            baseline_L_textgrad=baseline_L_textgrad,
            baseline_loss_repeats_textgrad=baseline_loss_repeats_textgrad,
            baseline_sample_details_textgrad=baseline_sample_details_textgrad,
            failure_examples=failure_examples,
            textgrad=opt,
            textgrad_simple_control=simple_control_opt,
        )
        if textgrad_select_indices:
            if baseline_L_textgrad_select is not None:
                pass
            elif dry_run:
                baseline_L_textgrad_select = 0.5
                baseline_loss_repeats_textgrad_select = [
                    0.5] * textgrad_eval_repeats
            else:
                selection_detail = aflow_adapter.evaluate_with_prompts(
                    dataset=dataset,
                    agent_prompts=original_prompts,
                    query_indices=textgrad_select_indices,
                    model=cfg["eval_model"],
                    n_repeats=textgrad_eval_repeats,
                    concurrent_batch=concurrent_batch,
                    dry_run=False,
                    split="validate",
                    return_details=True,
                )
                baseline_L_textgrad_select = float(selection_detail["loss"])
                baseline_loss_repeats_textgrad_select = list(
                    selection_detail.get("loss_repeats")
                    or [baseline_L_textgrad_select]
                )
            _save_stage(
                complete=False,
                stage="textgrad_devc_baseline_done",
                baseline_L_textgrad=baseline_L_textgrad,
                baseline_loss_repeats_textgrad=baseline_loss_repeats_textgrad,
                baseline_sample_details_textgrad=baseline_sample_details_textgrad,
                baseline_L_textgrad_select=baseline_L_textgrad_select,
                baseline_loss_repeats_textgrad_select=(
                    baseline_loss_repeats_textgrad_select),
                failure_examples=failure_examples,
                textgrad=opt,
                textgrad_simple_control=simple_control_opt,
            )
        textgrad_ckpt_dir = os.path.join(rq2_dir, f"{dataset}_textgrad_ckpt")
        for _i, agent in enumerate(agent_names, start=1):
            if agent in opt and opt[agent].get("best_prompt") is not None:
                print(f"    [TextGrad RESUME] {dataset}/{agent}: reuse completed optimization", flush=True)
                progress_bar(f"{dataset} optimize(DEV)", _i, len(agent_names))
                continue
            opt[agent] = textgrad_optimize_agent(
                dataset=dataset, target_agent=agent, base_prompts=original_prompts,
                goal=goals[agent], perturbation_evidence=evidence[agent],
                required_placeholders=placeholders[agent], query_indices=textgrad_indices,
                engine_llm=cfg["engine_llm"], eval_model=cfg["eval_model"],
                n_iterations=cfg["textgrad_n_iterations"], n_repeats=textgrad_eval_repeats,
                concurrent_batch=concurrent_batch, dry_run=dry_run, encoder=encoder,
                baseline_L=baseline_L_textgrad,
                baseline_loss_repeats=baseline_loss_repeats_textgrad,
                selection_query_indices=textgrad_select_indices,
                selection_baseline_L=baseline_L_textgrad_select,
                selection_baseline_loss_repeats=(
                    baseline_loss_repeats_textgrad_select),
                mode=cfg.get("textgrad_mode", "single"),
                min_signal=cfg.get("textgrad_stage1_abs_margin", 0.005),
                accept_margin=cfg.get("textgrad_accept_margin", 0.01),
                confidence_z=cfg.get("textgrad_confidence_z", 1.0),
                min_positive_repeats=cfg.get(
                    "textgrad_min_nonworse_repeats",
                    cfg.get("textgrad_min_positive_repeats", 2)),
                evidence_confidence_z=cfg.get("textgrad_evidence_confidence_z", 1.0),
                beam_size=cfg.get("textgrad_beam_size", 1),
                max_directions=cfg.get("textgrad_max_directions"),
                generation_attempts=cfg.get("textgrad_generation_attempts", 3),
                tokenizer_path=cfg.get("textgrad_tokenizer_path", ""),
                meta_max_tokens=cfg.get("textgrad_meta_max_tokens", 8192),
                reversal_example_max_tokens=cfg.get(
                    "textgrad_reversal_example_max_tokens", 2048),
                early_stop_patience=cfg.get("textgrad_early_stop_patience", 4),
                candidate_pool_extra=cfg.get("textgrad_candidate_pool_extra", 2),
                semantic_similarity_threshold=cfg.get(
                    "textgrad_semantic_similarity_threshold", 0.995),
                evidence_reuse=cfg.get("textgrad_evidence_reuse", True),
                devc_max_mean_regression=cfg.get(
                    "textgrad_devc_max_mean_regression", 0.01),
                devc_regression_z=cfg.get(
                    "textgrad_devc_regression_z", 2.0),
                devc_net_gain_margin=cfg.get(
                    "textgrad_devc_net_gain_margin", 0.005),
                devb_rescue_max_regression=cfg.get(
                    "textgrad_devb_rescue_max_regression", 0.005),
                devbc_rescue_mean_margin=cfg.get(
                    "textgrad_devbc_rescue_mean_margin", 0.0025),
                allow_zero_stage1_evidence=cfg.get(
                    "textgrad_allow_zero_stage1_evidence", False),
                candidate_min_length_ratio=cfg.get(
                    "textgrad_candidate_min_length_ratio", 0.20),
                candidate_max_length_ratio=cfg.get(
                    "textgrad_candidate_max_length_ratio", 3.00),
                candidate_max_token_floor=cfg.get(
                    "textgrad_candidate_max_token_floor", 512),
                checkpoint_path=os.path.join(textgrad_ckpt_dir, f"{agent}.json"),
            )
            _save_stage(
                complete=False,
                stage="textgrad_agents",
                textgrad=opt,
                baseline_L_textgrad=baseline_L_textgrad,
                baseline_loss_repeats_textgrad=baseline_loss_repeats_textgrad,
                baseline_sample_details_textgrad=baseline_sample_details_textgrad,
                failure_examples=failure_examples,
                textgrad_simple_control=simple_control_opt,
            )
            progress_bar(f"{dataset} optimize(DEV)", _i, len(agent_names))

        if simple_control_enabled:
            control_beam = max(
                1, int(cfg.get("textgrad_simple_control_samples", 4) or 4))
            print(
                f"  [stage 2C/4] Simple-4 control: independent DeepSeek calls per round="
                f"{control_beam} (temperature=0.7); DEV-B "
                "lowest mean loss selection ...",
                flush=True,
            )
            simple_ckpt_dir = os.path.join(
                rq2_dir, f"{dataset}_textgrad_simple4_ckpt")
            for _i, agent in enumerate(agent_names, start=1):
                if (agent in simple_control_opt
                        and simple_control_opt[agent].get("best_prompt") is not None):
                    print(
                        f"    [Simple-4 RESUME] {dataset}/{agent}: "
                        "reuse completed control result",
                        flush=True,
                    )
                    progress_bar(
                        f"{dataset} Simple-4(DEV)", _i, len(agent_names))
                    continue
                simple_control_opt[agent] = textgrad_optimize_agent(
                    dataset=dataset, target_agent=agent,
                    base_prompts=original_prompts, goal=goals[agent],
                    perturbation_evidence=evidence[agent],
                    required_placeholders=placeholders[agent],
                    query_indices=textgrad_indices,
                    engine_llm=cfg["engine_llm"],
                    eval_model=cfg["eval_model"],
                    n_iterations=cfg["textgrad_n_iterations"],
                    n_repeats=textgrad_eval_repeats,
                    concurrent_batch=concurrent_batch, dry_run=dry_run,
                    encoder=encoder, baseline_L=baseline_L_textgrad,
                    baseline_loss_repeats=baseline_loss_repeats_textgrad,
                    mode=cfg.get("textgrad_mode", "single"),
                    min_signal=cfg.get("textgrad_stage1_abs_margin", 0.005),
                    accept_margin=cfg.get("textgrad_accept_margin", 0.0),
                    confidence_z=cfg.get("textgrad_confidence_z", 1.0),
                    min_positive_repeats=cfg.get(
                        "textgrad_min_nonworse_repeats",
                        cfg.get("textgrad_min_positive_repeats", 2)),
                    evidence_confidence_z=cfg.get(
                        "textgrad_evidence_confidence_z", 1.0),
                    beam_size=control_beam,
                    generation_attempts=cfg.get(
                        "textgrad_generation_attempts", 5),
                    tokenizer_path=cfg.get("textgrad_tokenizer_path", ""),
                    meta_max_tokens=cfg.get("textgrad_meta_max_tokens", 8192),
                    reversal_example_max_tokens=cfg.get(
                        "textgrad_reversal_example_max_tokens", 2048),
                    early_stop_patience=cfg.get(
                        "textgrad_early_stop_patience", 4),
                    candidate_pool_extra=0,
                    semantic_similarity_threshold=1.0,
                    evidence_reuse=cfg.get("textgrad_evidence_reuse", True),
                    candidate_generation_mode="simple4_control",
                    # Keep the evidence pool identical to the guided arm.
                    # The only experimental variable is candidate generation.
                    allow_zero_stage1_evidence=cfg.get(
                        "textgrad_allow_zero_stage1_evidence", False),
                    candidate_min_length_ratio=cfg.get(
                        "textgrad_candidate_min_length_ratio", 0.20),
                    candidate_max_length_ratio=cfg.get(
                        "textgrad_candidate_max_length_ratio", 3.00),
                    candidate_max_token_floor=cfg.get(
                        "textgrad_candidate_max_token_floor", 512),
                    checkpoint_path=os.path.join(
                        simple_ckpt_dir, f"{agent}.json"),
                )
                _save_stage(
                    complete=False, stage="textgrad_simple_control_agents",
                    textgrad=opt,
                    textgrad_simple_control=simple_control_opt,
                    baseline_L_textgrad=baseline_L_textgrad,
                    baseline_loss_repeats_textgrad=(
                        baseline_loss_repeats_textgrad),
                    baseline_sample_details_textgrad=(
                        baseline_sample_details_textgrad),
                    failure_examples=failure_examples,
                )
                progress_bar(
                    f"{dataset} Simple-4(DEV)", _i, len(agent_names))

        # Every agent proceeds to TEST.  If TextGrad produced no usable
        # candidate, its best_prompt is the current/original prompt.
        textgrad_valid_agents = list(agent_names)
        textgrad_invalid_agents = {}

        simple_control_valid_agents = []
        simple_control_invalid_agents = {}
        if simple_control_enabled:
            simple_control_valid_agents = list(agent_names)


        final_test_repeats_total = max(
            1,
            int(cfg.get("final_test_repeats_total",
                        cfg.get("final_test_repeats", 1)) or 1),
        )
        repeat_extra_cache = bool(cfg.get("final_test_repeats_disable_cache", True))
        control_labels = {
            f"{agent}::simple4": agent
            for agent in simple_control_valid_agents
        }
        test_opt = dict(opt)
        test_opt.update({
            label: simple_control_opt[target]
            for label, target in control_labels.items()
        })
        test_interventions = {
            agent: agent for agent in textgrad_valid_agents
        }
        test_interventions.update(control_labels)
        test_labels = list(textgrad_valid_agents) + list(control_labels)
        print(f"  [stage 3/4] RQ2 eval: TEST({n_test}) shared baseline; "
              f"guided={len(textgrad_valid_agents)} + "
              f"simple4={len(simple_control_valid_agents)} interventions "
              f"(target repeats={final_test_repeats_total}; checkpoint after each repeat/agent) ...", flush=True)

        test_repeat_results = list(dataset_ckpt.get("rq2_test_repeats") or [])

        def _save_rq2_repeat(partial_repeat: dict):
            nonlocal test_repeat_results
            test_repeat_results = _upsert_repeat(test_repeat_results, partial_repeat)
            _save_stage(
                complete=False,
                stage="rq2_final_test",
                textgrad=opt,
                textgrad_simple_control=simple_control_opt,
                rq2_test_repeats=test_repeat_results,
            )

        if test_labels:
            for rep_idx in range(1, final_test_repeats_total + 1):
                existing = _find_repeat(test_repeat_results, rep_idx)
                if _repeat_complete(existing, test_labels):
                    print(f"  [stage 3/4] RQ2 eval: TEST repeat {rep_idx}/{final_test_repeats_total} "
                          "complete in checkpoint; skipped", flush=True)
                    continue
                cache_off = (rep_idx > 1 and repeat_extra_cache and not dry_run)
                print(f"  [stage 3/4] RQ2 eval: TEST repeat {rep_idx}/{final_test_repeats_total} "
                      f"({'cache OFF' if cache_off else 'cache ON'}) ...", flush=True)
                with _temporary_final_repeat_cache(disable=cache_off):
                    rep = _run_final_test_once(
                        dataset=dataset, original_prompts=original_prompts,
                        opt=test_opt, agent_names=test_labels,
                        query_indices=test_indices,
                        eval_model=cfg["eval_model"], n_repeats=cfg["n_repeats"],
                        concurrent_batch=concurrent_batch, dry_run=dry_run,
                        repeat_idx=rep_idx, show_progress=True,
                        initial_repeat=existing,
                        on_update=_save_rq2_repeat,
                        intervention_targets=test_interventions,
                    )
                _save_rq2_repeat(rep)

            test_repeat_summary = (
                _summarize_final_test_repeats(
                    test_repeat_results,
                    selected=(
                        selected if selected in textgrad_valid_agents else None),
                    agent_names=textgrad_valid_agents,
                )
                if textgrad_valid_agents else None
            )
            simple_control_repeat_summary = (
                _summarize_final_test_repeats(
                    test_repeat_results,
                    selected=(
                        f"{selected}::simple4"
                        if selected in simple_control_valid_agents else None),
                    agent_names=list(control_labels),
                )
                if simple_control_valid_agents else None
            )
        else:
            test_repeat_results = []
            test_repeat_summary = None
            simple_control_repeat_summary = None
        _save_stage(
            complete=False,
            stage="rq2_final_test_done",
            textgrad=opt,
            textgrad_simple_control=simple_control_opt,
            rq2_test_repeats=test_repeat_results,
            final_test_repeats=test_repeat_summary,
            simple_control_final_test_repeats=(
                simple_control_repeat_summary),
        )
        if test_repeat_summary:
            baseline_L_test = test_repeat_summary["baseline_L_mean"]
            test_optimized_L = {
                a: test_repeat_summary["agents"][a]["optimized_L_mean"]
                for a in textgrad_valid_agents
            }
            test_delta = {
                a: test_repeat_summary["agents"][a]["delta_L_mean"]
                for a in textgrad_valid_agents
            }
        simple_control_test_optimized_L = {}
        simple_control_test_delta = {}
        if simple_control_repeat_summary:
            simple_control_test_optimized_L = {
                agent: simple_control_repeat_summary["agents"][
                    f"{agent}::simple4"]["optimized_L_mean"]
                for agent in simple_control_valid_agents
            }
            simple_control_test_delta = {
                agent: simple_control_repeat_summary["agents"][
                    f"{agent}::simple4"]["delta_L_mean"]
                for agent in simple_control_valid_agents
            }
    else:
        textgrad_valid_agents = []
        textgrad_invalid_agents = {}
        simple_control_valid_agents = []
        simple_control_invalid_agents = {}
        simple_control_repeat_summary = None
        simple_control_test_optimized_L = {}
        simple_control_test_delta = {}
        print("  [stage 2/4] RQ2 TextGrad disabled: skip prompt optimization and RQ2 TEST gains.", flush=True)
        if run_rq3_model_swap:
            print(f"  [stage 3/4] RQ3 baseline: TEST({n_test}) qwen3-8b baseline "
                  "for model-swap gains ...", flush=True)
            baseline_L_test = aflow_adapter.evaluate_with_prompts(
                dataset=dataset, agent_prompts=original_prompts,
                query_indices=test_indices, model=cfg["eval_model"],
                n_repeats=cfg["n_repeats"], concurrent_batch=concurrent_batch,
                dry_run=dry_run, split="test",
            )
            baseline_L_test = float(baseline_L_test)


    rows = []
    for s in ranking:
        a = s.agent_name
        best_perturb = _best_perturbation_info(results.get(a, []), variants.get(a, []))
        rows.append({
            "agent": a,
            "potential_rank": s.rank,
            "i_abs": s.i_abs if s.i_abs is not None else 0.0,
            "mean_abs_delta": s.mean_abs_delta if s.mean_abs_delta is not None else 0.0,
            "g_norm": s.g_norm,
            "best_prompt_source": best_perturb["source"],
            "best_prompt_edit_type": best_perturb["edit_type"],
            "best_prompt_variant_idx": best_perturb["variant_idx"],
            "best_prompt_delta_L_dev": best_perturb["delta_L_dev"],
            "best_prompt_delta_L_dev_se": best_perturb.get("delta_L_dev_se", 0.0),
            "baseline_L": baseline_L_test,
            "baseline_L_std": (test_repeat_summary or {}).get("baseline_L_std"),
            "baseline_L_se": (test_repeat_summary or {}).get("baseline_L_se"),
            "optimized_L": test_optimized_L.get(a),
            "optimized_L_std": (
                (test_repeat_summary or {}).get("agents", {}).get(a, {}).get("optimized_L_std")
            ),
            "optimized_L_se": (
                (test_repeat_summary or {}).get("agents", {}).get(a, {}).get("optimized_L_se")
            ),
            "delta_L": test_delta.get(a),
            "delta_L_std": (
                (test_repeat_summary or {}).get("agents", {}).get(a, {}).get("delta_L_std")
            ),
            "delta_L_se": (
                (test_repeat_summary or {}).get("agents", {}).get(a, {}).get("delta_L_se")
            ),
            "delta_L_repeats": (
                (test_repeat_summary or {}).get("agents", {}).get(a, {}).get("per_repeat_delta_L", [])
            ),
            "dev_delta_L": opt.get(a, {}).get("improvement"),
            "textgrad_generated_candidates": (
                (opt.get(a) or {}).get("generated_candidate_count")
                if run_rq2_textgrad else None
            ),
            "textgrad_evaluated_candidates": (
                (opt.get(a) or {}).get("evaluated_candidate_count")
                if run_rq2_textgrad else None
            ),
            "simple_control_generated_candidates": (
                (simple_control_opt.get(a) or {}).get(
                    "generated_candidate_count")
                if run_rq2_textgrad and simple_control_enabled else None
            ),
            "simple_control_evaluated_candidates": (
                (simple_control_opt.get(a) or {}).get(
                    "evaluated_candidate_count")
                if run_rq2_textgrad and simple_control_enabled else None
            ),
            "simple_control_optimized_L": (
                simple_control_test_optimized_L.get(a)
                if run_rq2_textgrad and simple_control_enabled else None
            ),
            "simple_control_optimized_L_std": (
                (simple_control_repeat_summary or {}).get(
                    "agents", {}).get(
                        f"{a}::simple4", {}).get("optimized_L_std")
            ),
            "simple_control_optimized_L_se": (
                (simple_control_repeat_summary or {}).get(
                    "agents", {}).get(
                        f"{a}::simple4", {}).get("optimized_L_se")
            ),
            "simple_control_delta_L": (
                simple_control_test_delta.get(a)
                if run_rq2_textgrad and simple_control_enabled else None
            ),
            "simple_control_delta_L_std": (
                (simple_control_repeat_summary or {}).get(
                    "agents", {}).get(
                        f"{a}::simple4", {}).get("delta_L_std")
            ),
            "simple_control_delta_L_se": (
                (simple_control_repeat_summary or {}).get(
                    "agents", {}).get(
                        f"{a}::simple4", {}).get("delta_L_se")
            ),
            "simple_control_delta_L_repeats": (
                (simple_control_repeat_summary or {}).get(
                    "agents", {}).get(
                        f"{a}::simple4", {}).get(
                            "per_repeat_delta_L", [])
            ),
            "simple_control_dev_delta_L": (
                (simple_control_opt.get(a) or {}).get("improvement")
                if run_rq2_textgrad and simple_control_enabled else None
            ),
            "is_selected": (a == selected),
        })

    rq2_result_valid = bool(run_rq2_textgrad and selected in test_delta)

    if run_rq2_textgrad and test_delta:
        deltas = test_delta
        random_baseline = sum(deltas.values()) / len(deltas)
        oracle_dL = max(deltas.values())

        # ---- Three retained potential metrics; evaluation statistics stay. ----
        metric_eval = _evaluate_metric_predictions(
            ranking=ranking,
            deltas=deltas,
            topk_values=cfg.get("topk_values", [1, 2, 3]),
            extra_metric_values=_perturb_metric_values_from_rows(rows),
        )


        selected_dL = deltas.get(selected)
        selected_is_max = bool(
            selected_dL is not None and abs(selected_dL - oracle_dL) < 1e-9
        )
        spearman = (metric_eval.get(primary_metric) or {}).get("spearman", 0.0)
        valid_rows = [r for r in rows if r.get("delta_L") is not None]
        pearson = _pearson([r["g_norm"] for r in valid_rows], [r["delta_L"] for r in valid_rows])
        pearson_iabs = _pearson([r["i_abs"] for r in valid_rows], [r["delta_L"] for r in valid_rows])
        pearson_mean_abs = _pearson(
            [r["mean_abs_delta"] for r in valid_rows], [r["delta_L"] for r in valid_rows])
    else:
        deltas = {}
        random_baseline = None
        oracle_dL = None
        metric_eval = {}
        selected_dL = None
        selected_is_max = False
        spearman = None
        pearson = None
        pearson_iabs = None
        pearson_mean_abs = None

    simple_control_deltas = (
        dict(simple_control_test_delta)
        if run_rq2_textgrad and simple_control_enabled else {}
    )
    if simple_control_deltas:
        simple_control_metric_eval = _evaluate_metric_predictions(
            ranking=ranking,
            deltas=simple_control_deltas,
            topk_values=cfg.get("topk_values", [1, 2, 3]),
            extra_metric_values=_perturb_metric_values_from_rows(rows),
        )
        simple_control_selected_delta = simple_control_deltas.get(selected)
        simple_control_oracle_delta = max(simple_control_deltas.values())
        simple_control_oracle_agent = max(
            simple_control_deltas, key=simple_control_deltas.get)
    else:
        simple_control_metric_eval = {}
        simple_control_selected_delta = None
        simple_control_oracle_delta = None
        simple_control_oracle_agent = None
    paired_generation_comparison = {
        agent: {
            "guided_delta_L": test_delta.get(agent),
            "simple4_delta_L": simple_control_deltas.get(agent),
            "guided_minus_simple4": (
                test_delta[agent] - simple_control_deltas[agent]
                if agent in test_delta and agent in simple_control_deltas
                else None
            ),
        }
        for agent in agent_names
    }

    rq3_model_swap = {}
    if run_rq3_model_swap:
        if baseline_L_test is None:
            print(f"  [stage 3/4] RQ3 baseline: TEST({n_test}) qwen3-8b baseline "
                  "for model-swap gains ...", flush=True)
            baseline_L_test = float(aflow_adapter.evaluate_with_prompts(
                dataset=dataset, agent_prompts=original_prompts,
                query_indices=test_indices, model=cfg["eval_model"],
                n_repeats=cfg["n_repeats"], concurrent_batch=concurrent_batch,
                dry_run=dry_run, split="test",
            ))
            for r in rows:
                r["baseline_L"] = baseline_L_test
        rq3_model_swap = _run_rq3_model_swap(
            dataset=dataset,
            original_prompts=original_prompts,
            agent_names=agent_names,
            ranking=ranking,
            prompt_probe_extra_metric_values=_perturb_metric_values_from_rows(rows),
            baseline_L_test=baseline_L_test,
            test_indices=test_indices,
            cfg=cfg,
            concurrent_batch=openrouter_concurrent_batch,
            dry_run=dry_run,
            checkpoint_path=stage_ckpt_path,
            checkpoint_signature=stage_sig,
        )
    else:
        print("  [stage 4/4] RQ3 model-swap disabled: skip stronger-model TEST interventions.", flush=True)


    wf = aflow_adapter.get_cost()
    po = llm_cache.get_cost_since(_ifip_t0)
    cost = {"workflow_usd": wf["usd"], "promptops_usd": round(po["cost_usd"], 6),
            "total_usd": round(wf["usd"] + po["cost_usd"], 6),
            "workflow_cache_hits": wf.get("cache_hits", 0),
            "workflow_cache_misses": wf.get("cache_misses", 0),
            "eval_cache_hits": wf.get("eval_cache_hits", 0),
            "eval_cache_misses": wf.get("eval_cache_misses", 0),
            "tokens": {"workflow_in": wf["input_tokens"], "workflow_out": wf["output_tokens"],
                       "promptops_in": po["input_tokens"], "promptops_out": po["output_tokens"]}}

    record = {
        "dataset": dataset, "agents": agent_names, "selected": selected,
        "primary_metric": primary_metric,
        "potential_metrics": list(POTENTIAL_METRICS),
        "rq2_textgrad_enabled": run_rq2_textgrad,
        "textgrad_optimizer_version": _TEXTGRAD_OPTIMIZER_VERSION,
        "textgrad_simple_control_enabled": simple_control_enabled,
        "textgrad_simple_control": simple_control_opt,
        "rq2_result_valid": rq2_result_valid,
        "rq3_model_swap_enabled": run_rq3_model_swap,
        "baseline_L": baseline_L_test, "baseline_L_dev": baseline_L_dev,
        "baseline_L_textgrad": baseline_L_textgrad,
        "n_dev": len(dev_indices), "n_textgrad": len(textgrad_indices),
        "n_textgrad_select": len(textgrad_select_indices),
        "dev_indices_note": dev_note,
        "textgrad_indices_note": textgrad_note,
        "textgrad_select_indices_note": textgrad_select_note,
        "baseline_L_textgrad_select": baseline_L_textgrad_select,
        "n_test": len(test_indices), "rows": rows,
        "selected_delta_L": selected_dL, "random_baseline_delta_L": random_baseline,
        "oracle_delta_L": oracle_dL, "selected_is_max": selected_is_max,
        "spearman_potential_vs_delta": spearman, "pearson_gnorm_vs_delta": pearson,
        "pearson_iabs_vs_delta": pearson_iabs,
        "pearson_mean_abs_delta_vs_delta": pearson_mean_abs,
        "metric_eval": metric_eval, "textgrad": opt, "failure_examples": failure_examples,
        "simple_control_final_test_repeats": simple_control_repeat_summary,
        "simple_control_metric_eval": simple_control_metric_eval,
        "simple_control_selected_delta_L": simple_control_selected_delta,
        "simple_control_oracle_delta_L": simple_control_oracle_delta,
        "simple_control_oracle_agent": simple_control_oracle_agent,
        "paired_generation_comparison": paired_generation_comparison,
        "rq3_model_swap": rq3_model_swap,
        "final_test_repeats": test_repeat_summary,
        "cost": cost,
    }
    _save_stage(complete=True, stage="complete", final_record=record)
    return record


def _run_rq3_model_swap(
    dataset: str,
    original_prompts: Dict[str, str],
    agent_names: List[str],
    ranking: list,
    prompt_probe_extra_metric_values: Dict[str, Dict[str, float]],
    baseline_L_test: float,
    test_indices: List[int],
    cfg: dict,
    concurrent_batch: int,
    dry_run: bool,
    checkpoint_path: Optional[str] = None,
    checkpoint_signature: Optional[str] = None,
) -> dict:
    """Evaluate one-agent model upgrades on TEST."""
    probe_cfg = cfg.get("rq3_model_swap") or {}
    if not probe_cfg.get("enabled"):
        return {}
    models = probe_cfg.get("models") or []
    if not models:
        return {}
    if not test_indices:
        print(f"  [RQ3 model-swap] skip {dataset}: empty TEST split", flush=True)
        return {}

    ckpt = (
        _load_dataset_checkpoint(checkpoint_path, checkpoint_signature)
        if checkpoint_path and checkpoint_signature else {}
    )
    rq3_progress = dict(ckpt.get("rq3_model_swap_progress") or {})
    rq3_partial = dict(ckpt.get("rq3_model_swap_partial") or {})

    def _save_rq3(stage: str):
        if checkpoint_path and checkpoint_signature:
            _save_dataset_checkpoint(
                checkpoint_path,
                checkpoint_signature,
                complete=False,
                stage=stage,
                rq3_model_swap_progress=rq3_progress,
                rq3_model_swap_partial=rq3_partial,
            )

    out = {}
    for spec in models:
        if isinstance(spec, str):
            label = spec
            eval_model = spec
        else:
            label = spec.get("name") or spec.get("eval_model")
            eval_model = spec.get("eval_model") or spec.get("model") or label
        if not eval_model:
            continue

        print(f"  [RQ3 model-swap] {dataset}: TEST-only intervention model={eval_model}; "
              f"replace one agent at a time, others={cfg['eval_model']} ...", flush=True)
        repeats_total = max(1, int(probe_cfg.get("final_test_repeats_total", 1) or 1))
        disable_cache = bool(probe_cfg.get("disable_cache", True))
        label_progress = rq3_progress.setdefault(label, {"eval_model": eval_model, "per_agent": {}})
        label_progress["eval_model"] = eval_model
        per_agent = {}
        for a in agent_names:
            saved = (label_progress.get("per_agent") or {}).get(a) or {}
            per_agent[a] = {
                "optimized_L_values": list(saved.get("optimized_L_values") or []),
                "delta_L_values": list(saved.get("delta_L_values") or []),
            }
        label_progress["per_agent"] = per_agent
        total = repeats_total * len(agent_names)
        done = 0
        for rep_idx in range(1, repeats_total + 1):
            with _temporary_final_repeat_cache(disable=disable_cache and not dry_run):
                for agent in agent_names:
                    if len(per_agent[agent]["delta_L_values"]) >= rep_idx:
                        done += 1
                        progress_bar(f"{dataset} RQ3 model-swap({label})", done, total)
                        continue
                    L_swap = aflow_adapter.evaluate_with_prompts(
                        dataset=dataset, agent_prompts=original_prompts,
                        query_indices=test_indices, model=cfg["eval_model"],
                        n_repeats=cfg["n_repeats"],
                        concurrent_batch=concurrent_batch, dry_run=dry_run,
                        split="test",
                        agent_model_overrides={agent: eval_model},
                    )
                    delta = float(baseline_L_test) - float(L_swap)
                    per_agent[agent]["optimized_L_values"].append(float(L_swap))
                    per_agent[agent]["delta_L_values"].append(delta)
                    done += 1
                    _save_rq3(f"rq3_model_swap:{label}:repeat{rep_idx}:{agent}")
                    progress_bar(f"{dataset} RQ3 model-swap({label})", done, total)

        selected = ranking[0].agent_name
        agents_summary = {
            a: {
                "optimized_L_mean": _mean(v["optimized_L_values"]),
                "optimized_L_std": _std(v["optimized_L_values"]),
                "optimized_L_se": _se(v["optimized_L_values"]),
                "delta_L_mean": _mean(v["delta_L_values"]),
                "delta_L_std": _std(v["delta_L_values"]),
                "delta_L_se": _se(v["delta_L_values"]),
                "per_repeat_optimized_L": v["optimized_L_values"],
                "per_repeat_delta_L": v["delta_L_values"],
            }
            for a, v in per_agent.items()
        }
        deltas = {a: agents_summary[a]["delta_L_mean"] for a in agent_names}
        metric_eval = _evaluate_metric_predictions(
            ranking=ranking,
            deltas=deltas,
            topk_values=cfg.get("topk_values", [1, 2, 3]),
            extra_metric_values=prompt_probe_extra_metric_values,
        )
        selected_delta = deltas[selected]
        oracle_agent = max(deltas, key=lambda a: deltas[a])
        oracle_delta = deltas[oracle_agent]
        out[label] = {
            "label": label,
            "eval_model": eval_model,
            "base_eval_model": cfg["eval_model"],
            "baseline_L": baseline_L_test,
            "n_test": len(test_indices),
            "n_repeats": repeats_total,
            "agents": agents_summary,
            "deltas": deltas,
            "prompt_probe_selected": selected,
            "selected_delta_L": selected_delta,
            "oracle_agent": oracle_agent,
            "oracle_delta_L": oracle_delta,
            "selected_is_oracle": abs(selected_delta - oracle_delta) < 1e-9,
            "metric_eval": metric_eval,
        }
        rq3_partial[label] = out[label]
        _save_rq3(f"rq3_model_swap:{label}:done")
        ib = metric_eval.get(normalize_main_metric(cfg.get("main_metric", "g_norm"))) or {}
        selected_se = agents_summary.get(selected, {}).get("delta_L_se", 0.0)
        oracle_se = agents_summary.get(oracle_agent, {}).get("delta_L_se", 0.0)
        print(f"    [RQ3 model-swap RESULT] {dataset}/{label}: "
              f"replace selected={selected} ΔL={_fmt_pct(selected_delta, signed=True)}"
              f"±SE={_fmt_pct(selected_se)}; "
              f"oracle={oracle_agent} ΔL={_fmt_pct(oracle_delta, signed=True)}"
              f"±SE={_fmt_pct(oracle_se)}; "
              f"Spearman={ib.get('spearman', 0.0):+.5f}; "
              f"Kendall τ-b={_fmt_kendall(ib.get('kendall_tau_b'))}", flush=True)

    return out


def _resolve_sample_counts(dataset: str, cfg: dict):
    """Resolve split sizes, caps, and held-out partitions."""
    (dev_indices, _, _, _, _, _, requested_dev, _, _,
     available_dev) = _resolve_dev_splits(dataset, cfg)
    n_test, requested_test, available_test = _resolve_test_count(dataset, cfg)
    return len(dev_indices), n_test, requested_dev, requested_test, available_dev, available_test


def _perturb_delta_repeats(baseline_result, variant_result) -> List[float]:
    base_losses = list(getattr(baseline_result, "task_level_losses", []) or [])
    var_losses = list(getattr(variant_result, "task_level_losses", []) or [])
    if base_losses and var_losses and len(base_losses) == len(var_losses):
        return [float(b) - float(v) for b, v in zip(base_losses, var_losses)]
    return [float(baseline_result.L_observed) - float(variant_result.L_observed)]


def _best_perturbation_info(agent_results: list, agent_variants: list) -> dict:
    """Return the best measured perturbation."""
    empty = {
        "source": "none",
        "edit_type": "",
        "variant_idx": -1,
        "delta_L_dev": 0.0,
        "delta_L_dev_se": 0.0,
    }
    if not agent_results or len(agent_results) < 2:
        return empty
    baseline_L = agent_results[0].L_observed
    best = None
    for r in agent_results[1:]:
        delta = baseline_L - r.L_observed
        if best is None or delta > best["delta_L_dev"]:
            vi = r.variant_idx - 1
            variant = agent_variants[vi] if 0 <= vi < len(agent_variants) else None
            best = {
                "source": getattr(variant, "source", "unknown") if variant else "unknown",
                "edit_type": getattr(variant, "edit_type", "") if variant else "",
                "variant_idx": r.variant_idx,
                "delta_L_dev": float(delta),
                "delta_L_dev_se": _se(_perturb_delta_repeats(agent_results[0], r)),
            }
    return best or empty


def _perturb_metric_values_from_rows(rows: List[dict]) -> Dict[str, Dict[str, float]]:
    return {
        key: {r["agent"]: float(r[key]) for r in rows if r.get(key) is not None}
        for key in POTENTIAL_METRICS
    }


def _record_ranking_from_rows(rows: List[dict]) -> List[SimpleNamespace]:
    ordered = sorted(rows, key=lambda r: int(r.get("potential_rank") or 10**9))
    return [
        SimpleNamespace(
            agent_name=r.get("agent"),
            i_abs=float(r.get("i_abs", 0.0) or 0.0),
            mean_abs_delta=float(r.get("mean_abs_delta", 0.0) or 0.0),
            g_norm=(None if r.get("g_norm") is None else float(r.get("g_norm") or 0.0)),
        )
        for r in ordered
        if r.get("agent")
    ]


def _record_ensure_derived_metrics(rec: dict, rq2_dir: str, dataset: str, cfg: dict) -> bool:
    """Backfill retained perturbation metrics without deleting legacy fields.

    This is intentionally read-only w.r.t. model calls: it only reads the saved
    stage-1 perturbation pickle and recomputes table metrics / metric_eval.
    """
    rows = rec.get("rows") or []
    if not rows:
        return False

    changed = False
    results = {}
    pkl = os.path.join(rq2_dir, f"{dataset}_perturb.pkl")
    if os.path.exists(pkl):
        try:
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            results = obj.get("results") or {}
        except Exception as e:
            print(f"  [resume WARN] {dataset}: cannot backfill perturb metrics from {pkl}: {e}", flush=True)
            results = {}

    for r in rows:
        agent = r.get("agent")
        if agent not in results:
            # Migrating RankAcc -> Kendall must not zero saved potential scores
            # simply because the optional historical perturb pickle is absent.
            continue
        agent_results = results.get(agent, [])
        best_info = _best_perturbation_info(agent_results, [])
        new_vals = {"best_prompt_delta_L_dev_se": best_info.get("delta_L_dev_se", 0.0)}
        for key, val in new_vals.items():
            if abs(float(r.get(key, 0.0) or 0.0) - float(val)) > 1e-12:
                r[key] = float(val)
                changed = True

    return _refresh_record_metric_evaluations(rec, cfg) or changed


def _refresh_record_metric_evaluations(rec: dict, cfg: dict) -> bool:
    """Recompute RQ2/control/RQ3 rank metrics from saved means, with no I/O."""
    rows = rec.get("rows") or []
    changed = False
    ranking = _record_ranking_from_rows(rows)
    topk_values = cfg.get("topk_values", [1, 2, 3])
    extra_metric_values = _perturb_metric_values_from_rows(rows)

    rq2_deltas = {
        r["agent"]: float(r.get("delta_L"))
        for r in rows
        if r.get("agent") and r.get("delta_L") is not None
    }
    if ranking and rq2_deltas:
        new_eval = _evaluate_metric_predictions(
            ranking=ranking,
            deltas=rq2_deltas,
            topk_values=topk_values,
            extra_metric_values=extra_metric_values,
        )
        merged_eval = {**(rec.get("metric_eval") or {}), **new_eval}
        if rec.get("metric_eval") != merged_eval:
            rec["metric_eval"] = merged_eval
            changed = True

    control_deltas = {
        r["agent"]: float(r["simple_control_delta_L"])
        for r in rows
        if r.get("agent") and r.get("simple_control_delta_L") is not None
    }
    if ranking and control_deltas:
        new_eval = _evaluate_metric_predictions(
            ranking=ranking, deltas=control_deltas, topk_values=topk_values,
            extra_metric_values=extra_metric_values,
        )
        merged_eval = {**(rec.get("simple_control_metric_eval") or {}), **new_eval}
        if rec.get("simple_control_metric_eval") != merged_eval:
            rec["simple_control_metric_eval"] = merged_eval
            changed = True

    for rr in (rec.get("rq3_model_swap") or {}).values():
        rr_deltas = {
            a: float(v)
            for a, v in (rr.get("deltas") or {}).items()
            if v is not None
        }
        if ranking and rr_deltas:
            new_eval = _evaluate_metric_predictions(
                ranking=ranking,
                deltas=rr_deltas,
                topk_values=topk_values,
                extra_metric_values=extra_metric_values,
            )
            merged_eval = {**(rr.get("metric_eval") or {}), **new_eval}
            if rr.get("metric_eval") != merged_eval:
                rr["metric_eval"] = merged_eval
                changed = True

    return changed


def _best_source_label(row: dict) -> str:
    source = row.get("best_prompt_source", "none") or "none"
    edit_type = row.get("best_prompt_edit_type", "") or ""
    if edit_type:
        return f"{source}:{edit_type}"
    return source


def _format_topk_hits(e: dict) -> str:
    hits = e.get("topk_hit") or {}
    if not hits:
        return "-"
    def label(value):
        probability = float(value)
        return ("Y" if probability == 1.0 else "N" if probability == 0.0
                else _fmt_pct(probability, digits=1))
    return ",".join(f"{k}={label(v)}" for k, v in sorted(hits.items()))


def _metric_top1_probability(e: dict) -> float:
    """New expected hits and old boolean hits share an explicit numeric unit."""
    if e.get("top1_hit_probability") is not None:
        return float(e["top1_hit_probability"])
    return float(bool(e.get("is_max", False)))


def _metric_selection_label(e: dict) -> str:
    tied = e.get("selected_agents") or []
    if len(tied) > 1:
        return " = ".join(tied) + " (uniform)"
    return str(e.get("selected") or (tied[0] if tied else "-"))


def _fmt_num(value, fmt: str = ".5f", empty: str = "-") -> str:
    if value is None:
        return empty
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return empty


def _fmt_pct(value, digits: int = 3, signed: bool = False, empty: str = "-") -> str:
    if value is None:
        return empty
    try:
        val = float(value) * 100.0
    except (TypeError, ValueError):
        return empty
    sign = "+" if signed else ""
    return f"{val:{sign}.{digits}f}%"


def _fmt_kendall(value) -> str:
    """A signed correlation, not a percentage accuracy; undefined is N/A."""
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):+.5f}"


_METRIC_LABELS = (
    ("g_norm", "G-norm"),
    ("i_abs", "I_abs"),
    ("mean_abs_delta", "MeanAbs"),
)


def _format_agent_ranking(order: List[str], values: Dict[str, float] = None,
                          max_len: int = None) -> str:
    parts = []
    values = values or {}
    for a in order:
        if a in values and values[a] is not None:
            parts.append(f"{a}({_fmt_pct(values[a], signed=True)})")
        else:
            parts.append(str(a))
    text = " > ".join(parts)
    if max_len and len(text) > max_len:
        return text[:max_len - 1] + "…"
    return text


def _ranking_comparison_rows(metric_eval: dict, deltas: Dict[str, float]) -> List[dict]:
    if not deltas:
        return []
    actual_order = sorted(deltas, key=lambda a: deltas[a], reverse=True)
    rows = [{
        "source": "Actual ΔL",
        "ranking": _format_agent_ranking(actual_order, deltas),
        "kendall_tau_b": "-",
    }]
    for key, label in _METRIC_LABELS:
        e = (metric_eval or {}).get(key)
        if not e:
            continue
        rows.append({
            "source": label,
            "ranking": (
                " > ".join(" = ".join(group) for group in e["predicted_tie_groups"])
                if e.get("predicted_tie_groups")
                else _format_agent_ranking(e.get("predicted_order") or [])),
            "kendall_tau_b": _fmt_kendall(e.get("kendall_tau_b")),
        })
    return rows


def _append_ranking_table_log(lines: List[str], title: str, metric_eval: dict,
                              deltas: Dict[str, float], prefix: str = "│ ") -> None:
    rows = _ranking_comparison_rows(metric_eval, deltas)
    if not rows:
        return
    lines.append(f"{prefix}[{title}]  {'Ranking source':12s} {'Gain ranking (best to worst)':70s} {'Kendall τ-b':>11s}")
    for row in rows:
        lines.append(f"{prefix}         {row['source']:12s} "
                     f"{row['ranking']:70s} {row['kendall_tau_b']:>11s}")


def _ranking_table_md(metric_eval: dict, deltas: Dict[str, float]) -> List[str]:
    rows = _ranking_comparison_rows(metric_eval, deltas)
    if not rows:
        return []
    lines = [
        "| Ranking source | Gain ranking (best to worst) | Kendall τ-b |",
        "|---|---|:--:|",
    ]
    for row in rows:
        lines.append(f"| {row['source']} | {row['ranking']} | {row['kendall_tau_b']} |")
    return lines


def _resolve_test_count(dataset: str, cfg: dict):
    override = (cfg.get("per_dataset_samples") or {}).get(dataset)
    available_test = aflow_adapter.get_split_size(dataset, "test")
    requested_test = override[1] if override else cfg.get("n_test_samples", cfg.get("n_eval_samples"))
    use_all = bool(cfg.get("use_all_test", False)) or requested_test in (None, "all", "ALL", 0)
    if use_all:
        n_test = available_test
        requested_test = "ALL"
    else:
        requested_test = int(requested_test)
        n_test = min(requested_test, available_test)
    return n_test, requested_test, available_test


def _resolve_dev_splits(dataset: str, cfg: dict):
    """Split validate into disjoint DEV-A, DEV-B, and DEV-C partitions.

    DEV-A keeps floor(N/3) plus the remainder and supplies Stage-1 evidence.
    DEV-B and DEV-C come from the second and third thirds respectively and are
    independently capped (normally at 200).  DEV-B selects the best prompt;
    DEV-C sees only the DEV-B winner and vetoes any mean regression against
    the original or previous accepted prompt.  Neither DEV-B nor DEV-C sample
    content enters generation.
    """
    cfg = _dataset_config(dataset, cfg)
    subsample = cfg.get("dev_subsample")
    if subsample is not None:
        available = aflow_adapter.get_split_size(dataset, "validate")
        third, remainder = divmod(available, 3)
        a_end = third + remainder
        original = (list(range(a_end)), list(range(a_end, a_end + third)),
                    list(range(a_end + third, available)))
        counts = subsample.get("counts")
        seed = subsample.get("seed")
        if (not isinstance(counts, (list, tuple)) or len(counts) != 3
                or any(type(n) is not int or n <= 0 for n in counts)
                or type(seed) is not int):
            raise ValueError("dev_subsample requires three positive counts and an integer seed")
        if any(n > len(pool) for n, pool in zip(counts, original)):
            raise ValueError("DEV subsample exceeds its original partition")
        selected = [sorted(random.Random(seed).sample(pool, n))
                    for pool, n in zip(original, counts)]
        notes = [f"seed={seed} random subsample {n}/{len(pool)} from original "
                 f"validate[{pool[0]}:{pool[-1] + 1}]"
                 for pool, n in zip(original, counts)]
        return (*selected, *notes, *counts, available)
    override = (cfg.get("per_dataset_samples") or {}).get(dataset)
    available_dev = aflow_adapter.get_split_size(dataset, "validate")
    b_cap_raw = cfg.get("textgrad_max_samples", 200)
    c_cap_raw = cfg.get("textgrad_select_max_samples", 200)

    def _cap(raw, available):
        if raw in (None, "all", "ALL", 0):
            return available
        return max(0, int(raw))

    b_cap = _cap(b_cap_raw, available_dev)
    c_cap = _cap(c_cap_raw, available_dev)
    if cfg.get("use_all_validate", False):
        third = available_dev // 3
        remainder = available_dev - 3 * third
        a_end = third + remainder
        second_third_end = min(available_dev, a_end + third)
        b_end = min(second_third_end, a_end + b_cap)
        c_start = second_third_end
        c_third_end = min(available_dev, c_start + third)
        c_end = min(c_third_end, c_start + c_cap)
        dev_indices = list(range(0, a_end))
        textgrad_indices = list(range(a_end, b_end))
        textgrad_select_indices = list(range(c_start, c_end))
        return (
            dev_indices, textgrad_indices, textgrad_select_indices,
            f"validate[0:{a_end}] floor(N/3)={third} + remainder={remainder}; DEV-A uncapped",
            f"validate[{a_end}:{b_end}] second third={third}; DEV-B primary-selection cap={b_cap}",
            f"validate[{c_start}:{c_end}] third third={third}; "
            f"DEV-C net-gain guard cap={c_cap}",
            f"floor(validate/3)+remainder({remainder})", b_cap, c_cap, available_dev,
        )

    # Compatibility path for explicit sample-count experiments.
    requested_dev_raw = override[0] if override else cfg.get("n_eval_samples")
    requested_dev = (
        available_dev // 3
        if requested_dev_raw in (None, "all", "ALL", 0)
        else max(0, int(requested_dev_raw))
    )
    a_end = min(available_dev, requested_dev)
    remaining = max(0, available_dev - a_end)
    b_partition = remaining // 2
    b_partition_end = a_end + b_partition
    b_end = min(b_partition_end, a_end + b_cap)
    c_start = b_partition_end
    c_end = min(available_dev, c_start + c_cap)
    return (
        list(range(0, a_end)), list(range(a_end, b_end)), list(range(c_start, c_end)),
        f"validate[0:{a_end}] explicit DEV-A request={requested_dev}",
        f"validate[{a_end}:{b_end}] first half of remaining rows; DEV-B cap={b_cap}",
        f"validate[{c_start}:{c_end}] second half of remaining rows; DEV-C veto cap={c_cap}",
        requested_dev, b_cap, c_cap, available_dev,
    )


def _record_matches_requested_dev_splits(rec: dict, cfg: dict) -> bool:
    """Reject completed results from a different DEV sampling protocol."""
    if not rec.get("dataset"):
        return False
    a, b, c, a_note, b_note, c_note, *_ = _resolve_dev_splits(rec["dataset"], cfg)
    for key, indices in (("n_dev", a), ("n_textgrad", b), ("n_textgrad_select", c)):
        if rec.get(key) != len(indices):
            return False
    # Counts alone cannot distinguish a changed seed or a changed partition.
    for key, note in (("dev_indices_note", a_note),
                      ("textgrad_indices_note", b_note),
                      ("textgrad_select_indices_note", c_note)):
        if rec.get(key) is not None and rec[key] != note:
            return False
    return True


def _normalize_record_sample_counts(rec: dict, cfg: dict) -> bool:
    """Normalize historical sample-count fields."""
    dataset = rec.get("dataset")
    if not dataset:
        return False
    try:
        (dev_indices, textgrad_indices, textgrad_select_indices,
         dev_note, textgrad_note, textgrad_select_note,
         req_dev, req_textgrad, req_textgrad_select, avail_dev) = (
            _resolve_dev_splits(dataset, cfg)
        )
        n_test, req_test, avail_test = _resolve_test_count(dataset, cfg)
        n_dev = len(dev_indices)
        n_textgrad = len(textgrad_indices)
        n_textgrad_select = len(textgrad_select_indices)
    except Exception:
        return False
    changed = False
    for key, value in (
        ("n_dev", n_dev),
        ("n_textgrad", n_textgrad),
        ("n_textgrad_select", n_textgrad_select),
        ("n_test", n_test),
        ("dev_indices_note", dev_note),
        ("textgrad_indices_note", textgrad_note),
        ("textgrad_select_indices_note", textgrad_select_note),
    ):
        if rec.get(key) != value:
            rec[key] = value
            changed = True

    extras = {
        "requested_n_dev": req_dev,
        "requested_n_textgrad": req_textgrad,
        "requested_n_textgrad_select": req_textgrad_select,
        "requested_n_test": req_test,
        "available_n_dev": avail_dev,
        "available_n_test": avail_test,
    }
    for key, value in extras.items():
        if rec.get(key) != value:
            rec[key] = value
            changed = True
    return changed


def _print_dataset_report(rec):
    ds = rec["dataset"]
    rq2_enabled = bool(rec.get("rq2_textgrad_enabled", True))
    simple_control_enabled = bool(
        rec.get("textgrad_simple_control_enabled", False))
    L0d, L0t = rec.get("baseline_L_dev", float("nan")), rec.get("baseline_L")
    me = rec.get("metric_eval") or {}
    fr = rec.get("final_test_repeats") or {}
    n_rep = int(fr.get("n_repeats") or 1)
    metric_selected = {
        mkey: (me.get(mkey) or {}).get("selected")
        for mkey, _label in _METRIC_LABELS
    }
    test_acc = "-" if L0t is None else _fmt_pct(1 - float(L0t))
    test_l0 = _fmt_pct(L0t)
    L = [f"\n┌─ [RQ2/RQ3 RESULT] {ds}  "
         f"(DEV-A={rec['n_dev']} / DEV-B={rec.get('n_textgrad', '-')} / "
         f"DEV-C={rec.get('n_textgrad_select', '-')} / "
         f"TEST={rec['n_test']}) ────────────",
         f"│ Baseline:  DEV acc={_fmt_pct(1-L0d)} (L0={_fmt_pct(L0d)})  |  TEST acc={test_acc} (L0={test_l0})",
         f"│ [Stages] RQ2_TextGrad={'ON' if rq2_enabled else 'OFF'}  "
         f"Simple4_Control={'ON' if simple_control_enabled else 'OFF'}  "
         f"RQ3_ModelSwap={'ON' if rec.get('rq3_model_swap_enabled', bool(rec.get('rq3_model_swap'))) else 'OFF'}",
         f"│ [Probe potential @DEV]   {'agent':18s} {'G-norm':>9s} {'I_abs':>9s} "
         f"{'MeanAbs':>9s} {'BestΔ_SE':>9s} {'BestSrc':>18s}  selected"]
    for r in rec["rows"]:
        gn = "   -   " if r["g_norm"] is None else f"{r['g_norm']:.5f}"
        i_abs = r.get("i_abs", 0.0)
        mean_abs = r.get("mean_abs_delta", 0.0)
        best_src = _best_source_label(r)
        mk = [
            label for mkey, label in _METRIC_LABELS
            if r["agent"] == metric_selected.get(mkey)
        ]
        marker = ("★ " + "+".join(mk)) if mk else ("★ selected" if r["agent"] == rec["selected"] else "")
        L.append(f"│                    {r['agent']:18s} {gn:>9s} "
                 f"{_fmt_pct(i_abs):>9s} {_fmt_pct(mean_abs):>9s} "
                 f"{_fmt_pct(r.get('best_prompt_delta_L_dev_se', 0.0)):>9s} "
                 f"{best_src[:18]:>18s}  {marker}")
    if rq2_enabled:
        L.append(f"│ [Optimized eval @TEST]  {'agent':18s} {'baseL':>9s} {'optL':>9s} {'ΔL':>9s} {'ΔL_SE':>9s}  (positive delta = gain)")
        for r in rec["rows"]:
            L.append(f"│                    {r['agent']:18s} "
                     f"{_fmt_pct(r.get('baseline_L')):>9s} "
                     f"{_fmt_pct(r.get('optimized_L')):>9s} "
                     f"{_fmt_pct(r.get('delta_L'), signed=True):>9s} "
                     f"{_fmt_pct(r.get('delta_L_se')):>9s}")
        if n_rep > 1:
            L.append(f"│ [Repeated TEST] repeats={n_rep} (includes first repeat); "
                     f"selected ΔL={_fmt_pct(fr.get('selected_delta_mean', rec.get('selected_delta_L')), signed=True)}"
                     f"±SE={_fmt_pct(fr.get('selected_delta_se', 0.0))} "
                     f"(sd={_fmt_pct(fr.get('selected_delta_std', 0.0))}); "
                     f"oracle={fr.get('oracle_agent_by_mean', '')} "
                     f"{_fmt_pct(fr.get('oracle_delta_by_mean', rec.get('oracle_delta_L')), signed=True)}; "
                     f"repeat hit rate={_fmt_pct(fr.get('selected_hit_rate_over_repeats', 0.0))}")
        L.append(f"│ [Effect]  random delta={_fmt_pct(rec.get('random_baseline_delta_L'), signed=True)}  "
                 f"oracle delta={_fmt_pct(rec.get('oracle_delta_L'), signed=True)}")
        rq2_deltas = {
            r["agent"]: r.get("delta_L")
            for r in rec["rows"]
            if r.get("delta_L") is not None
        }
        _append_ranking_table_log(L, "RQ2 ranking comparison @TEST", me, rq2_deltas)
        if simple_control_enabled:
            L.append(
                f"│ [Generation comparison @TEST]  {'agent':18s} "
                f"{'Guided ΔL':>11s} {'SE':>9s} "
                f"{'Simple-4 ΔL':>12s} {'SE':>9s} "
                f"{'G-S4':>9s} {'winner':>10s}")
            for r in rec["rows"]:
                guided = r.get("delta_L")
                simple = r.get("simple_control_delta_L")
                diff = (
                    float(guided) - float(simple)
                    if guided is not None and simple is not None else None
                )
                if diff is None:
                    winner = "-"
                elif abs(diff) <= 1e-12:
                    winner = "tie"
                elif diff > 0:
                    winner = "guided"
                else:
                    winner = "simple-4"
                L.append(
                    f"│                          {r['agent']:18s} "
                    f"{_fmt_pct(guided, signed=True):>11s} "
                    f"{_fmt_pct(r.get('delta_L_se')):>9s} "
                    f"{_fmt_pct(simple, signed=True):>12s} "
                    f"{_fmt_pct(r.get('simple_control_delta_L_se')):>9s} "
                    f"{_fmt_pct(diff, signed=True):>9s} "
                    f"{winner:>10s}")
            L.append(
                "│ [Control effect] prompt-probe selected="
                f"{rec.get('selected')}: guided "
                f"{_fmt_pct(rec.get('selected_delta_L'), signed=True)} vs "
                f"simple-4 "
                f"{_fmt_pct(rec.get('simple_control_selected_delta_L'), signed=True)}; "
                f"simple-4 oracle={rec.get('simple_control_oracle_agent') or '-'} "
                f"{_fmt_pct(rec.get('simple_control_oracle_delta_L'), signed=True)}")
    else:
        L.append("│ [Optimized eval @TEST]  RQ2 TextGrad disabled; no optimized prompt or gain.")
    rq3 = rec.get("rq3_model_swap") or {}
    if rq3:
        L.append("│ [RQ3 model-swap TEST] prompt-probe ranking -> model-swap gain")
        for label, rr in rq3.items():
            ib = (rr.get("metric_eval") or {}).get("g_norm") or {}
            L.append(f"│         {label}: eval_model={rr.get('eval_model')}  "
                     f"selected={rr.get('prompt_probe_selected')} "
                     f"ΔL={_fmt_pct(rr.get('selected_delta_L', 0.0), signed=True)}"
                     f"±SE={_fmt_pct((rr.get('agents') or {}).get(rr.get('prompt_probe_selected'), {}).get('delta_L_se', 0.0))}; "
                     f"oracle={rr.get('oracle_agent')} "
                     f"ΔL={_fmt_pct(rr.get('oracle_delta_L', 0.0), signed=True)}"
                     f"±SE={_fmt_pct((rr.get('agents') or {}).get(rr.get('oracle_agent'), {}).get('delta_L_se', 0.0))}; "
                     f"G-norm Spearman={ib.get('spearman', 0.0):+.5f} "
                     f"Kendall τ-b={_fmt_kendall(ib.get('kendall_tau_b'))} "
                     f"{_format_topk_hits(ib)}")
            _append_ranking_table_log(
                L, f"RQ3 ranking comparison {label}",
                rr.get("metric_eval") or {}, rr.get("deltas") or {},
            )
    c = rec.get("cost") or {}
    L.append(f"│ [Cost]  workflow(qwen)=${c.get('workflow_usd',0):.5f}  prompt-ops=${c.get('promptops_usd',0):.5f}  total=${c.get('total_usd',0):.5f}")
    L.append(f"│ [Cache]  workflow hits={c.get('workflow_cache_hits',0)}  "
             f"misses={c.get('workflow_cache_misses',0)}  "
             f"| eval hits={c.get('eval_cache_hits',0)}  "
             f"misses={c.get('eval_cache_misses',0)}")
    L.append("└" + "─" * 62)
    print("\n".join(L), flush=True)


def _write_dataset_table(rq2_dir, rec):
    # CSV
    csv_path = os.path.join(rq2_dir, f"{rec['dataset']}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "potential_rank", "g_norm", "i_abs", "mean_abs_delta",
                    "best_prompt_source", "best_prompt_edit_type",
                    "best_prompt_variant_idx", "best_prompt_delta_L_dev",
                    "best_prompt_delta_L_dev_se",
                    "baseline_L", "baseline_L_std", "baseline_L_se",
                    "optimized_L", "optimized_L_std", "optimized_L_se",
                    "delta_L", "delta_L_std", "delta_L_se", "delta_L_repeats",
                    "textgrad_generated_candidates", "textgrad_evaluated_candidates",
                    "simple_control_generated_candidates",
                    "simple_control_evaluated_candidates",
                    "simple_control_optimized_L",
                    "simple_control_optimized_L_std",
                    "simple_control_optimized_L_se",
                    "simple_control_delta_L",
                    "simple_control_delta_L_std",
                    "simple_control_delta_L_se",
                    "simple_control_delta_L_repeats",
                    "guided_minus_simple4",
                    "is_selected"])
        for r in rec["rows"]:
            w.writerow([r["agent"], r["potential_rank"],
                        ("" if r["g_norm"] is None else f"{r['g_norm']:.5f}"),
                        f"{r.get('i_abs', 0.0):.5f}",
                        f"{r.get('mean_abs_delta', 0.0):.5f}",
                        r.get("best_prompt_source", "none"),
                        r.get("best_prompt_edit_type", ""),
                        r.get("best_prompt_variant_idx", -1),
                        f"{r.get('best_prompt_delta_L_dev', 0.0):+.5f}",
                        f"{r.get('best_prompt_delta_L_dev_se', 0.0):.5f}",
                        _fmt_num(r.get("baseline_L"), ".5f", ""),
                        _fmt_num(r.get("baseline_L_std"), ".5f", ""),
                        _fmt_num(r.get("baseline_L_se"), ".5f", ""),
                        _fmt_num(r.get("optimized_L"), ".5f", ""),
                        _fmt_num(r.get("optimized_L_std"), ".5f", ""),
                        _fmt_num(r.get("optimized_L_se"), ".5f", ""),
                        _fmt_num(r.get("delta_L"), "+.5f", ""),
                        _fmt_num(r.get("delta_L_std"), ".5f", ""),
                        _fmt_num(r.get("delta_L_se"), ".5f", ""),
                        json.dumps(r.get("delta_L_repeats", []), ensure_ascii=False),
                        r.get("textgrad_generated_candidates"),
                        r.get("textgrad_evaluated_candidates"),
                        r.get("simple_control_generated_candidates"),
                        r.get("simple_control_evaluated_candidates"),
                        _fmt_num(r.get("simple_control_optimized_L"), ".5f", ""),
                        _fmt_num(r.get("simple_control_optimized_L_std"), ".5f", ""),
                        _fmt_num(r.get("simple_control_optimized_L_se"), ".5f", ""),
                        _fmt_num(r.get("simple_control_delta_L"), "+.5f", ""),
                        _fmt_num(r.get("simple_control_delta_L_std"), ".5f", ""),
                        _fmt_num(r.get("simple_control_delta_L_se"), ".5f", ""),
                        json.dumps(
                            r.get("simple_control_delta_L_repeats", []),
                            ensure_ascii=False),
                        _fmt_num(
                            (rec.get("paired_generation_comparison") or {})
                            .get(r["agent"], {}).get("guided_minus_simple4"),
                            "+.5f", ""),
                        int(r["is_selected"])])

    with open(os.path.join(rq2_dir, f"{rec['dataset']}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False, default=_json_default)


def _dataset_md(rec) -> str:
    rq2_enabled = bool(rec.get("rq2_textgrad_enabled", True))
    simple_control_enabled = bool(
        rec.get("textgrad_simple_control_enabled", False))
    hit = ("Yes" if rec["selected_is_max"] else "No") if rq2_enabled else "-"
    lines = [
        f"### RQ2/RQ3 — {rec['dataset']}  (selected = {rec['selected']}, RQ2 hit={hit})",
        "",
        f"Potential: DEV-A(N={rec.get('n_dev','?')}); "
        f"TextGrad: DEV-B(N={rec.get('n_textgrad','?')}) candidate selection, "
        f"DEV-C(N={rec.get('n_textgrad_select','?')}) net-gain/rescue guard; "
        f"TEST(N={rec.get('n_test','?')}) final evaluation."
        f" DEV baseline L = {_fmt_pct(rec.get('baseline_L_dev', float('nan')))}",
        f"Stages: RQ2_TextGrad={'ON' if rq2_enabled else 'OFF'}; "
        f"Simple4_Control={'ON' if simple_control_enabled else 'OFF'}; "
        f"RQ3_ModelSwap={'ON' if rec.get('rq3_model_swap_enabled', bool(rec.get('rq3_model_swap'))) else 'OFF'}",
        "",
        "| Agent | Potential rank | G-norm | I_abs(max abs delta) | MeanAbs(mean abs delta) | BestΔ SE | Best variant source | baseline L(test) | optimized L(test) | ΔL(test) | ΔL SE | Selected |",
        "|-------|:------:|:----------:|:----------:|:----------:|:----------:|:---------:|:-----------:|:--------:|:--------:|:--------:|:--------:|",
    ]
    for r in rec["rows"]:
        star = "★" if r["is_selected"] else ""
        lines.append(f"| {r['agent']} | {r['potential_rank']} | {_fmt_num(r.get('g_norm'))} | "
                     f"{_fmt_pct(r.get('i_abs', 0.0))} | {_fmt_pct(r.get('mean_abs_delta', 0.0))} | "
                     f"{_fmt_pct(r.get('best_prompt_delta_L_dev_se', 0.0))} | "
                     f"{_best_source_label(r)} | "
                     f"{_fmt_pct(r.get('baseline_L'))} | "
                     f"{_fmt_pct(r.get('optimized_L'))} | "
                     f"{_fmt_pct(r.get('delta_L'), signed=True)} | "
                     f"{_fmt_pct(r.get('delta_L_se'))} | {star} |")
    if rq2_enabled:
        lines += [
            "",
            f"Random delta (agent mean) = {_fmt_pct(rec.get('random_baseline_delta_L'), signed=True)} ; "
            f"Oracle delta (agent max) = {_fmt_pct(rec.get('oracle_delta_L'), signed=True)}",
            f"Spearman(G-norm, ΔL) = {_fmt_num(((rec.get('metric_eval') or {}).get('g_norm') or {}).get('spearman'), '.5f')} ; "
            f"Pearson(G-norm, ΔL) = {_fmt_num(rec.get('pearson_gnorm_vs_delta'), '.5f')} ; "
            f"Pearson(I_abs, ΔL) = {_fmt_num(rec.get('pearson_iabs_vs_delta'), '.5f')} ; "
            f"Pearson(MeanAbs, ΔL) = {_fmt_num(rec.get('pearson_mean_abs_delta_vs_delta'), '.5f')} ; "
            f"selected_ΔL == oracle? {rec['selected_is_max']} ; "
            f"selected vs random = {_fmt_pct(rec.get('selected_delta_L'), signed=True)} vs "
            f"{_fmt_pct(rec.get('random_baseline_delta_L'), signed=True)}",
            "",
        ]
        me = rec.get("metric_eval") or {}
        rq2_deltas = {
            r["agent"]: r.get("delta_L")
            for r in rec["rows"]
            if r.get("delta_L") is not None
        }
        lines += ["#### RQ2 ranking comparison @TEST", ""]
        lines += _ranking_table_md(me, rq2_deltas)
        lines.append("")
        if simple_control_enabled:
            lines += [
                "#### Guided candidate pool vs Simple-4 comparison",
                "",
                "Both modes share Stage-1 evidence, selection rules, and "
                "TEST baseline; only candidate generation differs.",
                "",
                "| Agent | Guided ΔL±SE | Simple-4 ΔL±SE | Guided−Simple-4 | Winner |",
                "|---|---:|---:|---:|:--:|",
            ]
            for r in rec["rows"]:
                guided = r.get("delta_L")
                simple = r.get("simple_control_delta_L")
                diff = (
                    float(guided) - float(simple)
                    if guided is not None and simple is not None else None
                )
                if diff is None:
                    winner = "-"
                elif abs(diff) <= 1e-12:
                    winner = "tie"
                elif diff > 0:
                    winner = "guided"
                else:
                    winner = "simple-4"
                lines.append(
                    f"| {r['agent']} | {_fmt_pct(guided, signed=True)}±"
                    f"{_fmt_pct(r.get('delta_L_se'))} | "
                    f"{_fmt_pct(simple, signed=True)}±"
                    f"{_fmt_pct(r.get('simple_control_delta_L_se'))} | "
                    f"{_fmt_pct(diff, signed=True)} | {winner} |")
            lines += [
                "",
                f"Prompt-probe selected `{rec.get('selected')}`: guided "
                f"{_fmt_pct(rec.get('selected_delta_L'), signed=True)} vs "
                f"simple-4 "
                f"{_fmt_pct(rec.get('simple_control_selected_delta_L'), signed=True)}. "
                f"Simple-4 oracle: `{rec.get('simple_control_oracle_agent') or '-'}` "
                f"{_fmt_pct(rec.get('simple_control_oracle_delta_L'), signed=True)}.",
                "",
            ]
    else:
        lines += ["", "RQ2 TextGrad disabled; no optimized prompt or gain for this dataset.", ""]

    rq3 = rec.get("rq3_model_swap") or {}
    if rq3:
        lines += [
            "#### RQ3 TEST-only model-swap probe",
            "",
            "| Model | eval_model | prompt-probe selected/ΔL±SE | oracle/ΔL±SE | G-norm Spearman | G-norm Kendall τ-b | G-norm TopK |",
            "|---|---|---|---|:--:|:--:|---|",
        ]
        for label, rr in rq3.items():
            ib = (rr.get("metric_eval") or {}).get("g_norm") or {}
            selected_agent = rr.get("prompt_probe_selected")
            oracle_agent = rr.get("oracle_agent")
            agents = rr.get("agents") or {}
            lines.append(f"| {label} | `{rr.get('eval_model')}` | "
                         f"{selected_agent}/{_fmt_pct(rr.get('selected_delta_L', 0.0), signed=True)}±{_fmt_pct(agents.get(selected_agent, {}).get('delta_L_se', 0.0))} | "
                         f"{oracle_agent}/{_fmt_pct(rr.get('oracle_delta_L', 0.0), signed=True)}±{_fmt_pct(agents.get(oracle_agent, {}).get('delta_L_se', 0.0))} | "
                         f"{ib.get('spearman', 0.0):.5f} | "
                         f"{_fmt_kendall(ib.get('kendall_tau_b'))} | "
                         f"{_format_topk_hits(ib)} |")
        lines.append("")
        for label, rr in rq3.items():
            lines += [f"##### RQ3 ranking comparison — {label}", ""]
            lines += _ranking_table_md(rr.get("metric_eval") or {}, rr.get("deltas") or {})
            lines.append("")
    return "\n".join(lines)


def _write_summary(rq2_dir, per_dataset, summary):
    lines = ["# RQ2/RQ3 Results\n",
             "Kendall tau-b compares agent ranks against mean TEST gains. "
             "Exact equals are ties; this is not mean per-repeat correlation. "
             "Range [-1, 1]; higher is better. Undefined correlations are N/A "
             "and excluded from dataset means; hit rate and TopK are unchanged.\n"]
    for rec in per_dataset:
        lines.append(_dataset_md(rec))
    lines.append("## RQ2 summary (all datasets)\n")
    lines.append("| Dataset | selected agent | selected ΔL | Random delta | oracle ΔL | Selected is best? | G-norm Spearman |")
    lines.append("|---------|----------------|:-----------:|:-----------:|:---------:|:----------------:|:--------:|")
    for rec in per_dataset:
        rq2_enabled = bool(rec.get("rq2_textgrad_enabled", True))
        mark = ("✅" if rec["selected_is_max"] else "❌") if rq2_enabled else "-"
        lines.append(f"| {rec['dataset']} | {rec['selected']} | "
                     f"{_fmt_pct(rec.get('selected_delta_L'), signed=True)} | "
                     f"{_fmt_pct(rec.get('random_baseline_delta_L'), signed=True)} | "
                     f"{_fmt_pct(rec.get('oracle_delta_L'), signed=True)} | "
                     f"{mark} | {_fmt_num(((rec.get('metric_eval') or {}).get('g_norm') or {}).get('spearman'), '.5f')} |")
    lines.append("")
    lines.append(f"RQ2 hit rate (selected is best; enabled datasets only)= "
                 f"{summary['hit_count']}/{summary['n_datasets']} = {_fmt_pct(summary['hit_rate'])} ; "
                 f"G-norm mean Spearman = {summary['mean_spearman']:.5f} ; "
                 f"dataset count = {summary.get('total_datasets', len(per_dataset))}")

    paired = summary.get("paired_generation") or {}
    paired_selected = paired.get("selected_agent") or {}
    if paired_selected.get("n", 0):
        lines += [
            "",
            "## RQ2 candidate-generation ablation: guided pool vs Simple-4\n",
            "Both modes share evidence, selection and global-best rules, "
            "and TEST baseline.",
            "",
            "| Dataset | Prompt-probe selected | Guided ΔL | Simple-4 ΔL | "
            "Guided−Simple-4 | Winner |",
            "|---|---|---:|---:|---:|:--:|",
        ]
        for item in paired_selected.get("per_dataset", []):
            diff = item["guided_minus_simple4"]
            winner = (
                "tie" if abs(diff) <= 1e-12
                else ("guided" if diff > 0 else "simple-4")
            )
            lines.append(
                f"| {item['dataset']} | {item['selected']} | "
                f"{_fmt_pct(item['guided_delta_L'], signed=True)} | "
                f"{_fmt_pct(item['simple4_delta_L'], signed=True)} | "
                f"{_fmt_pct(diff, signed=True)} | {winner} |")
        lines += [
            "",
            f"Selected-agent paired results: N={paired_selected['n']}; "
            f"guided/simple-4/tie="
            f"{paired_selected['guided_wins']}/"
            f"{paired_selected['simple4_wins']}/"
            f"{paired_selected['ties']}; "
            f"mean delta={_fmt_pct(paired_selected.get('guided_mean_delta_L'), signed=True)}"
            f" vs {_fmt_pct(paired_selected.get('simple4_mean_delta_L'), signed=True)}; "
            f"mean difference={_fmt_pct(paired_selected.get('mean_guided_minus_simple4'), signed=True)}.",
        ]
        paired_all = paired.get("all_valid_agent_pairs") or {}
        if paired_all.get("n", 0):
            lines.append(
                f"All valid agent pairs: N={paired_all['n']}; "
                f"guided/simple-4/tie={paired_all['guided_wins']}/"
                f"{paired_all['simple4_wins']}/{paired_all['ties']}; "
                f"mean difference="
                f"{_fmt_pct(paired_all.get('mean_guided_minus_simple4'), signed=True)}.")


    bm = summary.get("by_metric") or {}
    lines.append("")
    metric_title = " vs ".join(label for _mkey, label in _METRIC_LABELS)
    lines.append(f"## Metric comparison: {metric_title} (selection accuracy)\n")
    lines.append("Top ties use uniform expected gain and hit rate; the executed selection is unchanged.\n")
    header = ["Dataset"]
    sep = ["---"]
    for _mkey, label in _METRIC_LABELS:
        header += [f"{label} selected/tied", f"{label}Expected Top1", f"{label} Spearman", f"{label} Kendall τ-b"]
        sep += ["---", ":--:", ":--:", ":--:"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(sep) + "|")
    for rec in per_dataset:
        me = rec.get("metric_eval") or {}
        def _c(e):
            return (
                _metric_selection_label(e), _fmt_pct(_metric_top1_probability(e)),
                f"{e['spearman']:.5f}", _fmt_kendall(e.get('kendall_tau_b')),
            ) if e else ("-", "-", "-", "-")
        row = [rec["dataset"]]
        for mkey, _label in _METRIC_LABELS:
            row.extend(_c(me.get(mkey)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for mkey, label in _METRIC_LABELS:
        m = bm.get(mkey)
        if m:
            lines.append(f"- **{label}**: hit rate {m['hit_count']}/{m['n']} = {_fmt_pct(m['hit_rate'])} ; "
                         f"mean Spearman = {m['mean_spearman']:.5f} ; "
                         f"mean Kendall τ-b = {_fmt_kendall(m.get('mean_kendall_tau_b'))} "
                         f"(valid {m.get('n_kendall_defined', 0)}/{m['n']}) ; "
                         f"TopK = {m.get('topk_hit_rate', {})}")
    rq3_summary = summary.get("rq3_model_swap") or {}
    if rq3_summary:
        lines.append("")
        lines.append("## RQ3 TEST-only model-swap summary\n")
        hit_headers = [f"{label} hit" for _mkey, label in _METRIC_LABELS]
        lines.append("| Model | N | " + " | ".join(hit_headers) + " | G-norm Spearman | G-norm Kendall τ-b (valid N) |")
        lines.append("|---|---:|" + "|".join([":--:"] * len(hit_headers)) + "|:--:|:--:|")
        for label, sm in rq3_summary.items():
            bm2 = sm.get("by_metric") or {}
            def _hm(k):
                m = bm2.get(k) or {}
                return f"{m.get('hit_count', 0)}/{m.get('n', 0)}"
            ib = bm2.get("g_norm") or {}
            hits_row = [_hm(mkey) for mkey, _label in _METRIC_LABELS]
            lines.append(f"| {label} | {sm.get('n_datasets', 0)} | "
                         + " | ".join(hits_row)
                         + f" | {ib.get('mean_spearman', 0.0):.5f} | "
                         f"{_fmt_kendall(ib.get('mean_kendall_tau_b'))} "
                         f"({ib.get('n_kendall_defined', 0)}) |")
    md = "\n".join(lines)
    with open(os.path.join(rq2_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(rq2_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + md)


def _evaluate_metric_predictions(ranking: list, deltas: Dict[str, float],
                                 topk_values: List[int],
                                 extra_metric_values: Dict[str, Dict[str, float]] = None) -> dict:
    """Compute top1/topk, Spearman and Kendall τ-b against mean TEST gains.

    ``deltas`` contains each agent's repeat-mean gain, not individual repeats.
    Ties use the original unrounded numeric values. Comparison utilities and
    hit probabilities integrate uniform tie-breaking; they do not overwrite
    the actual executed selector stored on the enclosing record.
    """
    metric_eval = {}
    extra_metric_values = extra_metric_values or {}
    # TextGrad-invalid agents have no TEST intervention and must not enter any
    # oracle, pairwise-ranking, Spearman, or top-k calculation.
    valid_ranking = [s for s in ranking if s.agent_name in deltas]
    agents_order = [s.agent_name for s in valid_ranking]
    if not agents_order:
        return {mkey: None for mkey, _label in _METRIC_LABELS}
    for mkey, _label in _METRIC_LABELS:
        vals = {}
        for s in valid_ranking:
            if mkey in extra_metric_values:
                v = extra_metric_values.get(mkey, {}).get(s.agent_name, float("-inf"))
            else:
                v = getattr(s, mkey, None)
            vals[s.agent_name] = float(v) if v is not None else float("-inf")
        if all(v == float("-inf") for v in vals.values()):
            metric_eval[mkey] = None
            continue

        # Canonical ordering is serialization only, never an implicit tie-break
        # inherited from whichever primary metric sorted ``ranking``.
        pred_order = sorted(agents_order, key=lambda a: (-vals[a], a))
        tie_groups = []
        for agent in pred_order:
            if not tie_groups or vals[agent] != vals[tie_groups[-1][0]]:
                tie_groups.append([])
            tie_groups[-1].append(agent)
        selected_agents = tie_groups[0]
        sel = selected_agents[0] if len(selected_agents) == 1 else None
        oracle_dL = max(deltas.values())
        oracle_agents = [a for a, d in deltas.items() if abs(d - oracle_dL) < 1e-9]
        oracle_set = set(oracle_agents)
        sel_dL = _mean([deltas[a] for a in selected_agents])
        top1_probability = len(oracle_set.intersection(selected_agents)) / len(selected_agents)
        sp = _spearman([vals[a] for a in agents_order], [deltas[a] for a in agents_order])
        kendall = _kendall_tau_b(
            [vals[a] for a in agents_order],
            [deltas[a] for a in agents_order],
        )
        topk_hit = {}
        for k in sorted({int(k) for k in topk_values if int(k) > 0}):
            kk = min(k, len(pred_order))
            # Preserve the requested metric name even when a workflow has
            # fewer than k agents.  For example, DROP has two agents, so its
            # requested top3 metric is evaluated over both agents but must
            # still be reported as ``top3``.  Renaming it to top2 made DROP
            # disappear from cross-dataset top3 aggregation and understated
            # the resulting hit rate.
            # At a partially included tied boundary, count draws containing at
            # least one empirical-best node (hypergeometric complement).
            slots, miss_probability = kk, 1.0
            for group in tie_groups:
                take = min(slots, len(group))
                if take == 0:
                    break
                non_oracles = len(group) - len(oracle_set.intersection(group))
                miss_probability *= (
                    math.comb(non_oracles, take) / math.comb(len(group), take)
                    if take <= non_oracles else 0.0)
                slots -= take
            topk_hit[f"top{k}"] = 1.0 - miss_probability

        metric_eval[mkey] = {
            "selected": sel,
            "selected_agents": selected_agents,
            "selection_weights": {a: 1.0 / len(selected_agents) for a in selected_agents},
            "selection_policy": "uniform_over_top_ties",
            "selected_delta_L": sel_dL,
            "top1_hit_probability": top1_probability,
            "is_max": abs(sel_dL - oracle_dL) < 1e-9,
            "spearman": sp,
            "kendall_tau_b": kendall,
            "topk_hit": topk_hit,
            "predicted_order": pred_order,
            "predicted_tie_groups": tie_groups,
            "oracle_agents": oracle_agents,
        }
    return metric_eval


def _aggregate(per_dataset) -> dict:
    rq2_records = [
        r for r in per_dataset
        if r.get("rq2_textgrad_enabled", True) and r.get("rq2_result_valid", True)
    ]
    n = len(rq2_records)
    hits = sum(1 for r in rq2_records if r["selected_is_max"])
    sp = [e["spearman"] for r in rq2_records
          if (e := (r.get("metric_eval") or {}).get("g_norm"))
          and e.get("spearman") is not None]
    mean_sp = sum(sp) / len(sp) if sp else 0.0


    by_metric = {}
    for mkey, _label in _METRIC_LABELS:
        recs = [r["metric_eval"][mkey] for r in rq2_records
                if r.get("metric_eval", {}).get(mkey) is not None]
        if not recs:
            by_metric[mkey] = None
            continue
        h = sum(_metric_top1_probability(e) for e in recs)
        msp = sum(e["spearman"] for e in recs) / len(recs)
        kendall_summary = _aggregate_kendall(recs)
        topk_keys = sorted({k for e in recs for k in (e.get("topk_hit") or {})})
        topk_hit_rate = {
            k: sum(float((e.get("topk_hit") or {}).get(k, 0.0)) for e in recs) / len(recs)
            for k in topk_keys
        }
        by_metric[mkey] = {"n": len(recs), "hit_count": h,
                           "hit_rate": h / len(recs), "mean_spearman": msp,
                           **kendall_summary,
                           "topk_hit_rate": topk_hit_rate}

    rq3_summary = {}
    labels = sorted({
        label for r in per_dataset
        for label in ((r.get("rq3_model_swap") or {}).keys())
    })
    for label in labels:
        records = [
            (r.get("rq3_model_swap") or {}).get(label)
            for r in per_dataset
            if (r.get("rq3_model_swap") or {}).get(label)
        ]
        model_by_metric = {}
        for mkey, _label in _METRIC_LABELS:
            recs = [
                rr["metric_eval"][mkey] for rr in records
                if rr.get("metric_eval", {}).get(mkey) is not None
            ]
            if not recs:
                model_by_metric[mkey] = None
                continue
            h = sum(_metric_top1_probability(e) for e in recs)
            msp = sum(e["spearman"] for e in recs) / len(recs)
            kendall_summary = _aggregate_kendall(recs)
            topk_keys = sorted({k for e in recs for k in (e.get("topk_hit") or {})})
            topk_hit_rate = {
                k: sum(float((e.get("topk_hit") or {}).get(k, 0.0)) for e in recs) / len(recs)
                for k in topk_keys
            }
            model_by_metric[mkey] = {
                "n": len(recs), "hit_count": h, "hit_rate": h / len(recs),
                "mean_spearman": msp, **kendall_summary,
                "topk_hit_rate": topk_hit_rate,
            }
        rq3_summary[label] = {
            "n_datasets": len(records),
            "by_metric": model_by_metric,
            "per_dataset": [
                {
                    "dataset": next(r["dataset"] for r in per_dataset
                                    if (r.get("rq3_model_swap") or {}).get(label) is rr),
                    "eval_model": rr.get("eval_model"),
                    "prompt_probe_selected": rr.get("prompt_probe_selected"),
                    "selected_delta_L": rr.get("selected_delta_L"),
                    "oracle_agent": rr.get("oracle_agent"),
                    "oracle_delta_L": rr.get("oracle_delta_L"),
                    "selected_is_oracle": rr.get("selected_is_oracle"),
                    "metric_eval": {k: (rr.get("metric_eval") or {}).get(k)
                                    for k in POTENTIAL_METRICS},
                }
                for rr in records
            ],
        }

    paired_dataset_rows = []
    paired_agent_rows = []
    for rec in per_dataset:
        if not rec.get("textgrad_simple_control_enabled", False):
            continue
        guided_selected = rec.get("selected_delta_L")
        simple_selected = rec.get("simple_control_selected_delta_L")
        if guided_selected is not None and simple_selected is not None:
            paired_dataset_rows.append({
                "dataset": rec["dataset"],
                "selected": rec["selected"],
                "guided_delta_L": float(guided_selected),
                "simple4_delta_L": float(simple_selected),
                "guided_minus_simple4": (
                    float(guided_selected) - float(simple_selected)),
            })
        for agent, item in (
                rec.get("paired_generation_comparison") or {}).items():
            guided = item.get("guided_delta_L")
            simple = item.get("simple4_delta_L")
            if guided is None or simple is None:
                continue
            paired_agent_rows.append({
                "dataset": rec["dataset"],
                "agent": agent,
                "guided_delta_L": float(guided),
                "simple4_delta_L": float(simple),
                "guided_minus_simple4": float(guided) - float(simple),
            })

    def _paired_summary(rows):
        if not rows:
            return {
                "n": 0, "guided_wins": 0, "simple4_wins": 0, "ties": 0,
                "guided_mean_delta_L": None, "simple4_mean_delta_L": None,
                "mean_guided_minus_simple4": None,
            }
        eps = 1e-12
        diffs = [r["guided_minus_simple4"] for r in rows]
        return {
            "n": len(rows),
            "guided_wins": sum(d > eps for d in diffs),
            "simple4_wins": sum(d < -eps for d in diffs),
            "ties": sum(abs(d) <= eps for d in diffs),
            "guided_mean_delta_L": (
                sum(r["guided_delta_L"] for r in rows) / len(rows)),
            "simple4_mean_delta_L": (
                sum(r["simple4_delta_L"] for r in rows) / len(rows)),
            "mean_guided_minus_simple4": sum(diffs) / len(diffs),
        }

    paired_generation = {
        "selected_agent": {
            **_paired_summary(paired_dataset_rows),
            "per_dataset": paired_dataset_rows,
        },
        "all_valid_agent_pairs": {
            **_paired_summary(paired_agent_rows),
            "pairs": paired_agent_rows,
        },
    }

    return {
        "total_datasets": len(per_dataset),
        "n_datasets": n, "hit_count": hits, "hit_rate": (hits / n if n else 0.0),
        "mean_spearman": mean_sp, "by_metric": by_metric,
        "rq3_model_swap": rq3_summary,
        "paired_generation": paired_generation,
        "per_dataset": [{"dataset": r["dataset"], "selected": r["selected"],
                         "selected_delta_L": r["selected_delta_L"],
                         "oracle_delta_L": r["oracle_delta_L"],
                         "random_baseline_delta_L": r["random_baseline_delta_L"],
                         "selected_is_max": r["selected_is_max"],
                         "spearman": ((r.get("metric_eval") or {}).get("g_norm") or {}).get("spearman"),
                         "metric_eval": {k: (r.get("metric_eval") or {}).get(k)
                                         for k in POTENTIAL_METRICS},
                         "simple_control_selected_delta_L": (
                             r.get("simple_control_selected_delta_L")),
                         "paired_generation_comparison": (
                             r.get("paired_generation_comparison"))}
                        for r in per_dataset],
    }


def _rankdata(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va < 1e-12 or vb < 1e-12:
        return 0.0
    return cov / (va * vb)


def _spearman(a, b):
    if len(a) < 2:
        return 0.0
    return _pearson(_rankdata(a), _rankdata(b))


def _kendall_tau_b(pred, truth) -> Optional[float]:
    """Tie-adjusted rank correlation on paired, unrounded agent scores.

    τ-b = (C-D) / sqrt((C+D+T_pred) * (C+D+T_truth)), where T counts
    ties on exactly one side. Joint ties contribute to neither factor.
    A zero denominator (including a constant vector) is undefined, not 0 or 1.
    """
    if len(pred) != len(truth):
        raise ValueError("Kendall τ-b requires paired vectors of equal length")
    if len(pred) < 2 or any(math.isnan(float(v)) for v in (*pred, *truth)):
        return None
    concordant = discordant = tied_pred = tied_truth = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            dp = 0 if pred[i] == pred[j] else (1 if pred[i] > pred[j] else -1)
            dt = 0 if truth[i] == truth[j] else (1 if truth[i] > truth[j] else -1)
            if dp == 0 and dt == 0:
                continue
            if dp == 0:
                tied_pred += 1
            elif dt == 0:
                tied_truth += 1
            elif dp == dt:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    denominator = math.sqrt((comparable + tied_pred) * (comparable + tied_truth))
    return (concordant - discordant) / denominator if denominator else None


def _aggregate_kendall(entries: List[dict]) -> dict:
    """Macro-average defined correlations only; never reuse legacy RankAcc."""
    values = [float(e["kendall_tau_b"]) for e in entries
              if e.get("kendall_tau_b") is not None
              and math.isfinite(float(e["kendall_tau_b"]))]
    return {
        "mean_kendall_tau_b": sum(values) / len(values) if values else None,
        "n_kendall_defined": len(values),
        "n_kendall_undefined": len(entries) - len(values),
    }


def _pairwise_rank_accuracy(pred, truth):
    """Legacy pairwise ranking accuracy."""
    n = len(pred)
    if n < 2:
        return 0.0
    if len(set(truth)) <= 1:
        return 1.0
    total = 0
    correct = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dp = 0 if pred[i] == pred[j] else (1 if pred[i] > pred[j] else -1)
            dt = 0 if truth[i] == truth[j] else (1 if truth[i] > truth[j] else -1)
            total += 1
            if dp == dt:
                correct += 1.0
            elif dp == 0 or dt == 0:
                correct += 0.5
    return correct / total if total else 0.0


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)
