"""IF-MAP entry point and experiment config."""

import os
import sys
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.paths import PROJECT_ROOT, project_path, display_path

def _bootstrap_env():
    """Read local credentials; exported variables take precedence."""
    envf = PROJECT_ROOT / ".env"
    if os.path.exists(envf):
        with open(envf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                value = v.strip().strip('"').strip("'")
                if value and not value.startswith("YOUR_"):
                    os.environ.setdefault(k.strip(), value)

    key_file = os.environ.get("IFIP_OPENROUTER_KEY_FILE", "").strip()
    if key_file and not os.environ.get("OPENROUTER_API_KEY"):
        private_key_file = project_path(key_file)
        try:
            with open(private_key_file, encoding="utf-8") as f:
                private_key = f.read().strip()
        except OSError:
            raise RuntimeError("Cannot read configured OpenRouter credential file") from None
        if not private_key.startswith("sk-or-"):
            raise RuntimeError("Invalid OpenRouter credential file")
        os.environ["OPENROUTER_API_KEY"] = private_key

    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        os.environ.setdefault("OPENAI_API_KEY", or_key)
        os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

_bootstrap_env()

# AFlow workflow/base evaluation model.
# Local Qwen via an OpenAI-compatible router.
BASE_EVAL_MODEL = os.environ.get("IFIP_BASE_EVAL_MODEL", "qwen3-8b-local-exec")

from src.workflow_loader import load_workflow, validate_workflow_for_trace
from src.dataset_loader import load_dataset_subset
from src.encoder import init_encoder, get_embedding_dim
from src.variant_generator import generate_variants
from src.perturbation import run_perturbation_loop
from src.score import compute_scores
from src.evaluator import evaluate_ranking
from src.logger import init_run, save_results
from src.errors import IFIPError
from src import llm_cache, mas_runner, variant_generator, perturbation

# Project-relative paths.
PATHS = {
    "who_when":  "data/Failure_Attribution_Benchmark/Who_and_When",
    "trail":     "data/Failure_Attribution_Benchmark/TRAIL",
    "mp_bench":  "data/Failure_Attribution_Benchmark/MP-Bench",

    "workflows": {
        "who_when_hc": "configs/workflows/who_when_hc.yaml",
        "who_when_ag": "configs/workflows/who_when_ag.yaml",
    },

    "cache_dir":   "cache",
    "results_dir": "results",
}

# Experiment config.
EXPERIMENT = {

    "rq_to_run": ["rq2"],

    "experiment_tag": "rq2_aflow",

    "rq1": {
        "dataset_name":   "who_when",
        "dataset_subset": "hc",        # "hc" / "ag" / "trail" / "mp_bench"
        "n_traces_limit": 3,
    },

    "rq2": {

        "datasets": ["HumanEval", "MBPP", "NQ", "DROP", "HotpotQA", "MATH"],

        "allow_cross_run_seed_results": False,
        "seed_results_dir": None,

        "use_all_validate": True,
        "dev_a_fraction_of_total_validate": 1 / 3,
        "textgrad_max_samples": 200,
        "textgrad_select_max_samples": 200,
        "validate_split_ratio": [1, 1, 1],
        "max_validate_samples": None,
        "use_all_test": True,
        "n_eval_samples": None,
        "n_test_samples": None,

        "per_dataset_samples": {},
        # Full validate thirds; MATH retains its existing search budget.
        "per_dataset_overrides": {
            "MATH": {
                "textgrad_beam_size": 3,
                "textgrad_max_directions": 5,  # attempts, including rejections
                "final_test_repeats_total": 3,
                "perturb_mix": {
                    "reference_resample_pilot": True,
                    "pilot_random_seed": 42,
                },
            },
        },

        "per_dataset_concurrent_batch": {
            "MBPP": 32,
            "NQ": 32,
            "GSM8K": 32,
            "HumanEval": 64,
            "HotpotQA": 48,
            "DROP": 48,
            "MATH": 24,
        },

        "local_concurrent_scale": 1.0,

        "openrouter_concurrent_batch_map": {64: 80, 32: 50, 8: 32},
        "per_dataset_openrouter_concurrent_batch": {},

        "perturb_eval_max_attempts": 3,

        "perturb_eval_repeats": 3,
        "keep_failed_perturbations": True,

        "skip_failed_candidates": {"MBPP": True},
        # 36 Pilot slots -> 6 fixed + rephrase + up to 3 adaptive variants.
        # Pilot supplies adaptive evidence; no formal-pool filtering.
        "n_perturb": 10,
        "perturb_mix": {
            "operator_pool_version": "perturbation-pool-fixed32-v35-generation-v13",
            "construction_protocol": "perturbation_pool_exact_v1",
            "operator_pool": list(variant_generator.RECOMMENDED_STATIC_PERTURBATION_FOCI),
            "generation_attempts": 4,
            # Canonical discovery request per slot; no extra +100-index prompt
            # refills. Historical initial candidates are replayed byte-for-byte.
            "slot_refill_rounds": 1,
            # All six datasets use 32 rows. References bind IDs to every DEV-A
            # input in the saved (possibly truncated) sample_details format.
            "pilot_samples": 32,
            "pilot_fraction": 0.25,
            "pilot_min_samples": 32,
            "pilot_max_samples": 32,
            "reference_root": "results/perturbation_pool_6datasets_devonly_testgeneralization_v35_20260818",
            # Historical Test base prompts changed in HumanEval/MBPP. Keep the
            # current workflow: reject incompatible old candidates explicitly,
            # retain the frozen question IDs, and use the same canonical generator.
            "reference_candidate_mismatch": "regenerate",
            "pilot_initial_candidates": 36,
            "failure_examples": 4,
            "failure_score_threshold": 0.999,
            "temperature": 0.7,
        },

        "run_rq2_textgrad": True,
        "run_rq3_model_swap": False,
        "main_metric": "g_norm",

        "textgrad_mode": "single",

        "textgrad_n_iterations": 6,

        "textgrad_eval_repeats": 3,

        "textgrad_accept_margin": 0.0,
        "textgrad_confidence_z": 1.0,
        "textgrad_min_nonworse_repeats": 2,
        "textgrad_min_positive_repeats": 2,
        "textgrad_evidence_confidence_z": 1.0,

        "textgrad_stage1_abs_margin": 0.005,
        "textgrad_allow_zero_stage1_evidence": False,
        "textgrad_evidence_reuse": False,

        "textgrad_beam_size": 4,
        "textgrad_candidate_pool_extra": 0,
        "textgrad_semantic_similarity_threshold": 1.0,

        "textgrad_generation_attempts": 3,

        "textgrad_candidate_min_length_ratio": 0.20,
        "textgrad_candidate_max_length_ratio": 3.00,
        "textgrad_candidate_max_token_floor": 512,

        "textgrad_simple_control_enabled": False,
        "textgrad_simple_control_samples": 4,

        "textgrad_early_stop_patience": 0,

        "textgrad_devc_max_mean_regression": 0.01,
        "textgrad_devc_regression_z": 2.0,
        "textgrad_devc_max_single_regression": 0.03,

        "textgrad_devc_net_gain_margin": 0.005,

        "textgrad_devb_rescue_max_regression": 0.005,
        "textgrad_devbc_rescue_mean_margin": 0.0025,
        "textgrad_eval_samples": None,

        "textgrad_tokenizer_path": "../qwen3-8b/model",
        "textgrad_meta_max_tokens": 8192,

        "textgrad_reversal_example_max_tokens": 2048,

        "final_test_repeats_total": 5,

        "final_test_repeats_disable_cache": True,

        "topk_values": [1, 2, 3],

        "rq3_model_swap": {
            "enabled": True,
            "models": [

                # {"name": "qwen3-32b", "eval_model": "qwen3-32b-exec"},
                # {"name": "qwen3.7-plus", "eval_model": "qwen3.7-plus-exec"},

                {"name": "qwen3-235b-a22b-2507", "eval_model": "qwen3-235b-a22b-2507-exec"},
            ],
            "final_test_repeats_total": 5,
            "disable_cache": True,
        },

        "engine_llm": "deepseek/deepseek-v3.2",
        "generator_llm": "deepseek/deepseek-v3.2",
        "eval_model": BASE_EVAL_MODEL,
        "n_repeats": 1,
        "concurrent_batch": 48,

        "local_qwen3_service": {
            "enabled": True,
            "auto_switch": True,
            "qwen_root": "../qwen3-8b",
            "use_16k_for": ["MATH"],
            "start_8k": "../qwen3-8b/start_qwen3_8b_dynamic_4gpu_8k.sh",
            # The dynamic start script owns reuse/switch/stop atomically.  Do
            # not pre-stop the opposite profile here: both modes share the
            # same unified stop script, which would otherwise reload weights
            # before every consecutive dataset.
            "stop_8k": None,
            "start_16k": "../qwen3-8b/start_qwen3_8b_dynamic_4gpu_16k.sh",
            "stop_16k": None,
            "health_url": "http://127.0.0.1:8000/health",
            "backend_model_urls": [],
            "desired_replicas": 4,
            "target_gpus": [1, 2, 3, 4],
            # Four 16k replicas can take a little over seven minutes to load
            # concurrently from disk.  Keep the normal wait above that observed
            # startup time and allow a short final grace window before declaring
            # the router unavailable.
            "wait_timeout_s": 600,
            "health_grace_s": 30,
            "cleanup_between_datasets": True,
        },
    },
}

# Attribution config.
METHOD = {

    "main_metric": "g_norm",

    "num_variants_per_agent": 10,
    "variant_mix": {
        # Generic RQ1 traces do not expose an independent DEV-A table, so they
        # cannot honestly run the discovery Pilot/adaptive protocol. The exact
        # shared protocol is enabled only in EXPERIMENT['rq2'] above.
        "operator_pool_version": "legacy-static-non-aflow-v2",
        "operator_pool": list(variant_generator.RECOMMENDED_STATIC_PERTURBATION_FOCI),
        "generation_attempts": 4,
    },
    "variant_generator_llm": "deepseek/deepseek-v3.2",

    "encoder_name": "qwen/qwen3-embedding-8b",

    "n_eval_samples": 1,
    "n_repeats":      3,
    "eval_temperature": 0.0,
    "eval_llm": BASE_EVAL_MODEL,

    "regression_method": "ridge_with_pca",
    "ridge_lambda":    1.0,
    "pca_components":  None,

    # Legacy evaluator arguments retained; retired self-score requests disabled.
    "compute_i_self": False,
    "i_self_judge_llm": "deepseek/deepseek-v3.2",

    "seed": 42,
}

# Runtime and budgets.
RUNTIME = {
    "use_llm_cache":   True,
    "cache_db_path":   "cache/llm_calls.db",

    "use_aflow_llm_cache": True,
    "aflow_cache_db_path": "cache/aflow_llm_calls.db",

    "aflow_cache_version": "rq2-v12-reversal3-devbcmean0025-temp03-test5",

    "save_aflow_eval_logs": False,
    "save_aflow_mismatch_logs": False,

    "use_eval_cache": True,
    "eval_cache_version": "rq2-eval-v12-reversal3-devbcmean0025-temp03-test5",
    # Only identical-input repeated sampling nodes use this temperature. They
    # deliberately omit seed so repeated requests sample independently; all
    # other workflow operators inherit eval model temperature=0.
    "diversity_sampling_temperature": 0.3,
    "eval_timeout_seconds": 1800,

    "max_completion_tokens_by_dataset": {
        "MBPP": 2048,
        "GSM8K": 2048,
        "NQ": 2048,
        "HumanEval": 2048,
        "HotpotQA": 2048,
        "DROP": 2048,
        "MATH": 8192,
    },

    "max_prompt_tokens_by_dataset": {
        "MBPP": 6144,
        "GSM8K": 6144,
        "NQ": 6144,
        "HumanEval": 6144,
        "HotpotQA": 6144,
        "DROP": 6144,
        "MATH": 8192,
    },
    "llm_provider":    "openai",     # "openai" / "anthropic" / "vllm"
    "max_budget_usd":  1000.0,
    "dry_run":         False,
}

def main():
    os.chdir(PROJECT_ROOT)

    resume_dir = os.environ.get("IFIP_RESUME_RUN_DIR", "").strip()
    if resume_dir:
        run_dir = str(project_path(resume_dir).resolve())
        os.makedirs(run_dir, exist_ok=True)
        run_id = os.path.basename(run_dir.rstrip(os.sep))
        print(f"[Logger] resume run_id = {run_id}")
        print(f"[Logger] resume run_dir = {display_path(run_dir)}")
    else:
        run_id, run_dir = init_run(
            experiment_tag=EXPERIMENT["experiment_tag"],
            results_root=PATHS["results_dir"],
            seed=METHOD["seed"],
            all_configs={"PATHS": PATHS, "EXPERIMENT": EXPERIMENT,
                         "METHOD": METHOD, "RUNTIME": RUNTIME},
        )
    os.environ["IFIP_RUN_ID"] = run_id

    cache_namespace_run_id = os.environ.get(
        "IFIP_CACHE_NAMESPACE_RUN_ID", "").strip() or run_id
    os.environ["IFIP_LLM_CACHE_NAMESPACE"] = f"run={cache_namespace_run_id}"
    os.environ["AFLOW_LLM_CACHE_RUN_ID"] = cache_namespace_run_id
    if cache_namespace_run_id != run_id:
        print(f"[Cache] controlled reuse namespace = {cache_namespace_run_id}", flush=True)
    os.environ["IFIP_AFLOW_SAVE_EVAL_LOGS"] = "1" if RUNTIME["save_aflow_eval_logs"] else "0"
    os.environ["IFIP_AFLOW_SAVE_MISMATCH_LOGS"] = (
        "1" if RUNTIME["save_aflow_mismatch_logs"] else "0")
    os.environ["IFIP_EVAL_CACHE"] = "1" if RUNTIME["use_eval_cache"] else "0"
    os.environ["IFIP_EVAL_CACHE_DB"] = os.path.abspath(
        os.path.join(run_dir, "rq2", "eval_cache.sqlite"))
    os.environ["IFIP_EVAL_CACHE_VERSION"] = RUNTIME["eval_cache_version"]

    # A new experiment can request a physically isolated prompt-operations
    # cache instead of sharing the default SQLite file.  Request namespaces
    # already prevent logical cross-run reuse; this override additionally
    # separates the on-disk database for controlled full runs.
    promptops_cache_db = os.environ.get(
        "IFIP_PROMPTOPS_CACHE_DB", RUNTIME["cache_db_path"]
    ).strip()
    llm_cache.init_cache_db(promptops_cache_db)
    print(f"[Cache] prompt-ops db = {display_path(promptops_cache_db)}", flush=True)
    os.environ["AFLOW_LLM_CACHE"] = "1" if RUNTIME["use_aflow_llm_cache"] else "0"
    os.environ["AFLOW_LLM_CACHE_DB"] = os.path.abspath(
        os.path.join(run_dir, "rq2", "aflow_llm_cache.sqlite"))
    os.environ["AFLOW_LLM_CACHE_VERSION"] = RUNTIME["aflow_cache_version"]
    os.environ["AFLOW_DIVERSITY_SAMPLING_TEMPERATURE"] = str(
        RUNTIME["diversity_sampling_temperature"])
    os.environ.pop("AFLOW_DIVERSITY_SAMPLING_SEED", None)
    os.environ["IFIP_EVAL_TIMEOUT"] = str(RUNTIME["eval_timeout_seconds"])
    os.environ["IFIP_MAX_COMPLETION_TOKENS_BY_DATASET"] = json.dumps(
        RUNTIME.get("max_completion_tokens_by_dataset", {}), sort_keys=True
    )
    os.environ["IFIP_MAX_PROMPT_TOKENS_BY_DATASET"] = json.dumps(
        RUNTIME.get("max_prompt_tokens_by_dataset", {}), sort_keys=True
    )
    from src import textgrad_opt
    for mod in (mas_runner, variant_generator, perturbation, textgrad_opt):
        mod.set_default_provider(RUNTIME["llm_provider"])

    encoder_name = "hash_dev" if RUNTIME["dry_run"] else METHOD["encoder_name"]
    encoder = init_encoder(encoder_name)
    print(f"[Run] Encoder dim = {get_embedding_dim(encoder)}")

    if "rq1" in EXPERIMENT["rq_to_run"]:
        run_rq1_attribution_accuracy(run_dir, encoder)
    if "rq2" in EXPERIMENT["rq_to_run"]:
        run_rq2_improvement_prediction(run_dir, encoder)
    if "rq3" in EXPERIMENT["rq_to_run"]:
        run_rq3_cross_method_consistency(run_dir, encoder)

    print(f"\n✓ Run complete: {run_id}")
    print(f"  Results saved to: {display_path(run_dir)}")

def run_rq2_improvement_prediction(run_dir, encoder):
    """Probe agents, optimize prompts, and evaluate on TEST."""
    from src.rq2_runner import run_rq2
    cfg = copy.deepcopy(EXPERIMENT["rq2"])
    rq2_override = os.environ.get("IFIP_RUN_RQ2_TEXTGRAD", "").strip().lower()
    rq3_override = os.environ.get("IFIP_RUN_RQ3_MODEL_SWAP", "").strip().lower()
    if rq2_override:
        cfg["run_rq2_textgrad"] = rq2_override in ("1", "true", "yes", "on")
    if rq3_override:
        cfg["run_rq3_model_swap"] = rq3_override in ("1", "true", "yes", "on")
    datasets_override = os.environ.get("IFIP_DATASETS_OVERRIDE", "").strip()
    if datasets_override:
        cfg["datasets"] = [d.strip() for d in datasets_override.split(",") if d.strip()]
        print(f"[Run] datasets override = {cfg['datasets']}", flush=True)
    seed_results_override = os.environ.get(
        "IFIP_SEED_RESULTS_DIR", "").strip()
    migrated_completed_override = os.environ.get(
        "IFIP_MIGRATED_COMPLETED_DATASETS", "").strip()
    if seed_results_override:
        cfg["allow_cross_run_seed_results"] = True
        cfg["seed_results_dir"] = str(project_path(seed_results_override).resolve())
        print(f"[Run] seed results dir = {display_path(cfg['seed_results_dir'])}", flush=True)
    if migrated_completed_override:
        migrated = [
            item.strip() for item in migrated_completed_override.split(",")
            if item.strip()
        ]
        # The same explicit list both limits which source artifacts may be
        # copied and allows those verified-complete legacy records to remain
        # complete even when the optimizer implementation version changed.
        cfg["seed_completed_datasets"] = migrated
        cfg["accept_migrated_completed_datasets"] = migrated
        print(f"[Run] migrated completed datasets = {migrated}", flush=True)
    run_rq2(run_dir=run_dir, encoder=encoder,
            cfg=cfg, dry_run=RUNTIME["dry_run"])

def run_rq3_cross_method_consistency(run_dir, encoder):
    """Reserved standalone RQ3 entry point."""
    raise NotImplementedError("Use RQ2 model-swap mode for RQ3; standalone entry is reserved")

def run_rq1_attribution_accuracy(run_dir, encoder):
    cfg = EXPERIMENT["rq1"]
    traces = load_dataset_subset(
        dataset_root=PATHS[cfg["dataset_name"]],
        subset_name=cfg["dataset_subset"],
        n_limit=cfg["n_traces_limit"],
    )
    workflow_def = load_workflow(
        PATHS["workflows"][f"{cfg['dataset_name']}_{cfg['dataset_subset']}"]
    )

    all_predictions = []
    failed_traces = []
    for trace in traces:
        print(f"\n{'='*60}\n[RQ1] trace {trace.trace_id}  "
              f"({len(trace.agent_names)} agents)\n{'='*60}")
        try:
            prediction_record, scores = _process_trace(trace, workflow_def, encoder)
        except IFIPError as e:
            print(f"  [SKIP trace {trace.trace_id}] {e}")
            failed_traces.append({"trace_id": trace.trace_id, "error": str(e)})
            continue
        _print_trace_result(prediction_record, scores)
        all_predictions.append(prediction_record)

    if failed_traces:
        print(f"\n[RQ1] {len(failed_traces)} trace(s) skipped due to explicit errors:")
        for ft in failed_traces:
            print(f"  - {ft['trace_id']}: {ft['error']}")

    save_results(
        run_dir=run_dir,
        all_predictions=all_predictions,
        configs_snapshot={"PATHS": PATHS, "EXPERIMENT": EXPERIMENT,
                          "METHOD": METHOD, "RUNTIME": RUNTIME},
    )

def _process_trace(trace, workflow_def, encoder):
    """Run attribution on one trace."""

    _, warns = validate_workflow_for_trace(workflow_def, trace.agent_names)
    for w in warns:
        print(f"  {w}")

    agent_variants = generate_variants(
        original_prompts=trace.agent_prompts,
        num_per_agent=METHOD["num_variants_per_agent"],
        mix=METHOD["variant_mix"],
        generator_llm=METHOD["variant_generator_llm"],
        encoder=encoder,
        dry_run=RUNTIME["dry_run"],
    )

    perturbation_results = run_perturbation_loop(
        trace=trace, workflow_def=workflow_def, agent_variants=agent_variants,
        n_eval_samples=METHOD["n_eval_samples"], n_repeats=METHOD["n_repeats"],
        eval_llm=METHOD["eval_llm"], eval_temperature=METHOD["eval_temperature"],
        compute_i_self=METHOD["compute_i_self"],
        i_self_judge_llm=METHOD["i_self_judge_llm"],
        dry_run=RUNTIME["dry_run"],
    )

    scores = compute_scores(
        perturbation_results=perturbation_results,
        agent_variants=agent_variants,
        main_metric=METHOD["main_metric"],
        regression_method=METHOD["regression_method"],
        ridge_lambda=METHOD["ridge_lambda"],
        pca_components=METHOD["pca_components"],
    )

    prediction_record = evaluate_ranking(trace=trace, predicted_scores=scores)
    return prediction_record, scores

def _print_trace_result(record, scores):
    print(f"\n  GT failure agent: {record.gt_failure_agent}")
    print(f"  Predicted ranking: {record.predicted_ranking}")
    print(f"  Top-1 {'✓' if record.top1_correct else '✗'}  "
          f"Top-3 {'✓' if record.top3_correct else '✗'}  "
          f"rank_of_gt={record.rank_of_gt}")
    print(f"  {'Agent':<16} {'rank':>4} {'g_norm':>8} {'I_abs':>8} {'MeanAbs':>8}")
    for s in sorted(scores.values(), key=lambda x: x.rank):
        gn = f"{s.g_norm:.5f}" if s.g_norm is not None else "  -  "
        ia = f"{s.i_abs:.5f}" if s.i_abs is not None else "  -  "
        ma = f"{s.mean_abs_delta:.5f}" if s.mean_abs_delta is not None else "  -  "
        print(f"  {s.agent_name:<16} {s.rank:>4} {gn:>8} {ia:>8} {ma:>8}")

if __name__ == "__main__":
    main()
