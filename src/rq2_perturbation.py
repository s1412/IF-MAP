"""Stage-1 perturbation pools, repeats, and evidence."""

import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from src.perturbation import PerturbationResult
from src.variant_generator import generate_variants, AgentVariant
from src import aflow_adapter, perturbation_pool_core, variant_generator
from src.errors import IFIPError, report_invalid
from src.pilot_reference import load_reference, PilotReferenceError
from src.paths import project_path


POOL_PROTOCOL_VERSION = "pilot36-fixed32-v35-generation-budget10-v13"
_REFERENCE_GENERATION_FIELDS = (
    "id", "agent", "prompt", "source", "generation_round", "focus",
    "edit_type", "changed_span", "evidence_link", "rationale",
    "expected_mechanism", "regression_risk", "generation_attempt",
    "prompt_sha256", "length_ratio", "diff", "generation_rejections",
    "reference_initial_slot", "reference_candidate_id",
)


def _pool_execution_fingerprint(*, dataset, original_prompts, query_indices,
                                generator_llm, eval_model, config, dry_run):
    """Bind resume shortcuts to the actual current system and reference bytes."""
    reference_hashes = {}
    if config.get("reference_root") and not dry_run:
        for name in ("checkpoint.pkl", "all_candidates_summary.json"):
            path = project_path(config["reference_root"]) / dataset / name
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                reference_hashes[name] = digest.hexdigest()
            else:
                reference_hashes[name] = None  # strict loader diagnoses missing assets
    payload = {
        "protocol": POOL_PROTOCOL_VERSION, "dataset": dataset,
        "original_prompt_hashes": {a: perturbation_pool_core.sha(p) for a, p in original_prompts.items()},
        "agent_order": list(original_prompts), "query_indices": list(query_indices),
        "generator_llm": generator_llm, "eval_model": eval_model,
        "config": config, "reference_hashes": reference_hashes, "dry_run": bool(dry_run),
    }
    return perturbation_pool_core.sha(json.dumps(payload, sort_keys=True, default=str))


class _PerturbationSlotExhausted(RuntimeError):
    """A named slot exhausted all bounded generation/refill attempts."""

    def __init__(self, message: str, rejections: Sequence[str]):
        super().__init__(message)
        self.rejections = list(rejections)


class _DuplicatePerturbationExhausted(_PerturbationSlotExhausted):
    """A named slot produced only prompts already forbidden by the pool."""


def progress_bar(label: str, i: int, total: int, width: int = 24):
    """Print compact stage progress."""
    frac = i / total if total else 1.0
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    print(f"    [{label}] {bar} {i}/{total} ({frac*100:.0f}%)", flush=True)


def _se(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _paired_deltas(base_losses: List[float], variant_losses: List[float], fallback: float):
    if base_losses and variant_losses and len(base_losses) == len(variant_losses):
        return [float(b) - float(v) for b, v in zip(base_losses, variant_losses)]
    return [float(fallback)]


def _stage1_counterfactual(base_details: List[dict], variant_details: List[dict], limit: int = 6):
    """Build paired Stage-1 sample evidence with a fully hydrated lowest set.

    Benchmark datasets expose only a final task-level ground truth.  They do
    not provide a ground-truth target for an intermediate agent step, so the
    ``expected`` value below is always explicitly tagged as
    ``final_ground_truth``.  In particular, it must never be presented to the
    prompt optimizer as the expected intermediate output of ``target_agent``.
    Six complete lowest-scoring cases are retained so negative evidence can
    use a broader sample for risk induction; positive TextGrad rounds still
    consume at most three of them.
    """
    base = {int(x.get("row_index", -1)): x for x in (base_details or [])}
    var = {int(x.get("row_index", -1)): x for x in (variant_details or [])}
    rows = []
    for idx in sorted(set(base) & set(var)):
        b, v = base[idx], var[idx]
        bs, vs = float(b.get("score", 0.0)), float(v.get("score", 0.0))
        input_value = v.get("input") or b.get("input", "")
        expected_value = (
            v.get("expected")
            if v.get("expected") not in (None, "")
            else b.get("expected", "")
        )
        rows.append({
            "row_index": idx, "delta_score": vs - bs,
            "baseline_score": bs, "variant_score": vs,
            "input": input_value,
            "baseline_prediction": b.get("prediction", ""),
            "variant_prediction": v.get("prediction", ""),
            "expected": expected_value,
            "expected_scope": "final_ground_truth",
            "analysis_payload_complete": bool(
                str(input_value).strip() and str(expected_value).strip()
                and "prediction" in b and "prediction" in v
            ),
            "target_trace": v.get("target_trace") or [],
        })
    # Reversal examples are defined by the paired score direction itself.  Do
    # not require an exact 0->1 or 1->0 flip: graded datasets can improve or
    # regress substantially without either endpoint being exactly binary.
    fixed = sorted(
        [x for x in rows if x["delta_score"] > 0],
        key=lambda x: x["delta_score"], reverse=True)[:limit]
    broken = sorted(
        [x for x in rows if x["delta_score"] < 0],
        key=lambda x: x["delta_score"])[:limit]
    hard = sorted(
        [x for x in rows if x["baseline_score"] < 0.999 and x["variant_score"] < 0.999],
        key=lambda x: (x["variant_score"], x["baseline_score"]))[:limit]
    # This is the authoritative sample set consumed by TextGrad.  Keep the
    # complete payload here instead of trying to hydrate a compact score row
    # later from fixed/broken/hard subsets, which may not contain the actual
    # three lowest-scoring cases.
    lowest = sorted(
        [x for x in rows if x["analysis_payload_complete"]],
        key=lambda x: (
            x["variant_score"], x["delta_score"], x["row_index"]),
    )[:limit]
    # Preserve every row's paired scores in compact form.  This lets TextGrad
    # identify examples that remain unresolved across multiple perturbations
    # exactly, without duplicating all prompt text and traces in the checkpoint.
    compact_rows = [
        {
            "row_index": x["row_index"],
            "delta_score": x["delta_score"],
            "baseline_score": x["baseline_score"],
            "variant_score": x["variant_score"],
        }
        for x in rows
    ]
    return {
        "fixed": fixed,
        "broken": broken,
        "hard": hard,
        "lowest": lowest,
        "rows": compact_rows,
        "expected_scope": "final_ground_truth",
    }


_MAIN_INITIAL_FOCI = (
    "downstream_interface",
    "decomposition_control",
    "ambiguity_resolution",
    "semantic_precision",
    "verification",
    "edge_cases",
)
_MAIN_ADAPTIVE_FOCI = (
    "adaptive_score_conditioned",
    "adaptive_counterfactual_repair",
    "adaptive_regression_repair",
)


def _pilot_evaluate_candidate(*, dataset: str, original_prompts: Dict[str, str],
                              agent: str, prompt: str, pilot_indices: Sequence[int],
                              pilot_baseline: dict, eval_model: str,
                              concurrent_batch: int, dry_run: bool,
                              collect_details: bool = False) -> dict:
    """Exact one-repeat DEV-A pilot measurement used by pool discovery."""
    try:
        out = perturbation_pool_core.evaluate(
            dataset=dataset,
            original_prompts=original_prompts,
            target_agent=agent,
            candidate_prompt=prompt,
            indices=list(pilot_indices),
            model=eval_model,
            repeats=1,
            concurrent=concurrent_batch,
            dry_run=dry_run,
            collect_details=collect_details,
        )
        baseline_loss = float(pilot_baseline["loss"])
        measured = {
            "loss": float(out["loss"]),
            "mean_delta": baseline_loss - float(out["loss"]),
            "output_cap_rate": float(out.get("output_cap_rate", 0.0) or 0.0),
            "failed": False,
        }
        if collect_details:
            measured["sample_details"] = list(out.get("sample_details") or [])
        return measured
    except IFIPError as exc:
        return {
            "loss": None,
            "mean_delta": float("-inf"),
            "output_cap_rate": 1.0,
            "failed": True,
            "error": str(exc)[:1000],
        }


def _records_to_agent_variants(original: str, records: Sequence[dict], encoder) -> List[AgentVariant]:
    texts = [original] + [str(item["prompt"]) for item in records]
    embeddings = encoder.encode_batch(texts)
    e_original = embeddings[0]
    variants: List[AgentVariant] = []
    for item, embedding in zip(records, embeddings[1:]):
        diff = embedding - e_original
        norm = float(np.linalg.norm(diff))
        if norm < 1e-6:
            raise RuntimeError(
                f"strict perturbation-pool candidate is embedding-degenerate: "
                f"agent={item.get('agent')} focus={item.get('focus')}"
            )
        variants.append(AgentVariant(
            variant_text=str(item["prompt"]),
            embedding_diff=diff,
            norm=norm,
            unit_direction=diff / norm,
            source=str(item.get("source", "perturbation_pool")),
            edit_type=str(item.get("focus", item.get("edit_type", ""))),
        ))
    return variants


def _build_exact_perturbation_pool(
    *,
    dataset: str,
    original_prompts: Dict[str, str],
    baseline_sample_details: Sequence[dict],
    query_indices: Sequence[int],
    generator_llm: str,
    eval_model: str,
    concurrent_batch: int,
    dry_run: bool,
    encoder,
    config: dict,
    saved_state: dict,
    persist: Callable[[dict], None],
) -> Tuple[Dict[str, List[AgentVariant]], dict]:
    """Request up to 36 unique candidates for a measured prior, then emit ten.

    Candidate construction, diagnostic evidence, pilot sampling, validation,
    and prior ranking are shared verbatim with perturbation-pool discovery.
    Pilot is a prior-search stage only: it does not filter the fixed formal
    Stage-1 pool (six fixed mechanisms, sectionwise rephrase, three adaptive
    mechanisms).  The Pilot winner receives one detail-collecting pass so the
    adaptive mechanisms have genuine fixed/broken/hard sample evidence.
    """
    state = dict(saved_state or {})
    protocol_version = POOL_PROTOCOL_VERSION
    execution_fingerprint = _pool_execution_fingerprint(
        dataset=dataset, original_prompts=original_prompts, query_indices=query_indices,
        generator_llm=generator_llm, eval_model=eval_model, config=config, dry_run=dry_run,
    )
    if state and state.get("protocol_version") != protocol_version:
        # Never reuse measurements/candidates from a different sampling or
        # generator protocol, even when its formal pool also has ten slots.
        state = {}
    if state and state.get("execution_fingerprint") != execution_fingerprint and not (
        dry_run and "execution_fingerprint" not in state  # hand-built offline fixtures only
    ):
        state = {}
    state["protocol_version"] = protocol_version
    failure_limit = int(config.get("failure_examples", 4) or 4)
    pilot_n = perturbation_pool_core.resolve_pilot_sample_count(
        len(query_indices),
        requested=config.get("pilot_samples", 32),
        fraction=float(config.get("pilot_fraction", 0.25) or 0.25),
        min_samples=int(config.get("pilot_min_samples", 32) or 32),
        max_samples=config.get("pilot_max_samples", 32),
    )
    pilot_initial_n = int(config.get("pilot_initial_candidates", 36) or 36)
    if pilot_initial_n != 36:
        raise ValueError(
            f"{POOL_PROTOCOL_VERSION} requires exactly 36 initial "
            f"Pilot candidates; got {pilot_initial_n}"
        )
    temperature = float(config.get("temperature", 0.7))
    generation_attempts = int(config.get("generation_attempts", 4))
    if not dry_run and (
        temperature != perturbation_pool_core.GENERATION_TEMPERATURE
        or generation_attempts != perturbation_pool_core.GENERATION_MAX_ATTEMPTS
    ):
        raise ValueError("v35 generation requires temperature=0.7 and max_attempts=4")
    goals = aflow_adapter.get_prompt_goals(dataset)

    reference = None
    resample_reference_pilot = bool(config.get("reference_resample_pilot", False))
    simulation_without_details = bool(dry_run and not baseline_sample_details)
    if config.get("reference_root") and not simulation_without_details:
        if not baseline_sample_details:
            raise PilotReferenceError("real reference replay requires current full-DEV sample details")
        reference = load_reference(
            dataset, query_indices, baseline_sample_details,
            config["reference_root"], expected_count=pilot_n,
            **({"resample_pilot": True} if resample_reference_pilot else {}),
        )
    reference_provenance = (reference or {}).get("provenance")
    if state.get("reference_provenance") != reference_provenance:
        state = {"protocol_version": protocol_version}
    state["reference_provenance"] = reference_provenance
    state["generation_protocol_version"] = perturbation_pool_core.GENERATION_PROTOCOL_VERSION
    reference_candidates = dict((reference or {}).get("initial_candidates_by_agent") or {})
    mismatch_policy = config.get("reference_candidate_mismatch", "error")
    if mismatch_policy not in ("error", "regenerate"):
        raise ValueError("reference_candidate_mismatch must be 'error' or 'regenerate'")
    incompatible_agents = {}
    # Validate generated text against today's frozen operator contract before
    # spending a Pilot query. Never import historical pilot/full/TEST scores.
    for agent, original in original_prompts.items():
        if agent in reference_candidates:
            slots = [item.get("reference_initial_slot") for item in reference_candidates[agent]]
            if len(slots) != 36 or set(slots) != {f"initial_{i:03d}" for i in range(1, 37)}:
                raise PilotReferenceError(f"reference must supply all 36 initial slots: {dataset}/{agent}")
        reasons = []
        for item in reference_candidates.get(agent, []):
            valid, reason = perturbation_pool_core.candidate_valid(original, item["prompt"])
            if not valid:
                reasons.append({
                    "slot": item["reference_initial_slot"], "reason": reason,
                    "reference_prompt_sha256": perturbation_pool_core.sha(item["prompt"]),
                })
        if reasons:
            if mismatch_policy != "regenerate":
                raise PilotReferenceError(f"reference prompt incompatible: {dataset}/{agent}: {reasons[0]['reason']}")
            # This explicit policy rejects the entire agent's old pool. It
            # never edits old text or weakens validation, and does not change
            # the separately validated reference task indices. Broken hashes,
            # malformed artifacts and incomplete slot sets still fail closed.
            incompatible_agents[agent] = {
                "reason": "reference_candidates_incompatible_with_current_original",
                "action": "regenerate_all_36_with_canonical_v35_request",
                "candidate_rejections": reasons,
                "generation_evidence": "fresh_current_DEV_diagnostics",
            }
            reference_candidates.pop(agent)
            print(
                f"[Pilot reference] {dataset}/{agent}: rejected historical initial pool "
                f"({len(reasons)}/36 incompatible); canonical regeneration; "
                "validated v35 Pilot indices retained", flush=True,
            )
    if (
        state.get("reference_candidate_policy", mismatch_policy) != mismatch_policy
        or state.get("reference_unavailable_agents", incompatible_agents) != incompatible_agents
    ):
        state = {"protocol_version": protocol_version,
                 "reference_provenance": reference_provenance,
                 "generation_protocol_version": perturbation_pool_core.GENERATION_PROTOCOL_VERSION}
    state["reference_candidate_policy"] = mismatch_policy
    state["reference_unavailable_agents"] = incompatible_agents
    state["execution_fingerprint"] = execution_fingerprint
    state["reference_mode"] = (
        "offline_simulation_without_real_DEV_details" if simulation_without_details
        else "validated_initial_prompts_fresh_subset_pilot" if reference and resample_reference_pilot
        else "validated_v35_reference" if reference else "fresh_canonical_generation"
    )

    pilot_indices = list(state.get("pilot_indices") or [])
    if reference and not resample_reference_pilot and pilot_indices and pilot_indices != reference["pilot_indices"]:
        raise PilotReferenceError("resumed Pilot indices differ from validated v35 reference")
    if not pilot_indices:
        pilot_indices = (
            list(reference["pilot_indices"]) if reference and not resample_reference_pilot
            else perturbation_pool_core.pilot_indices(
                query_indices, baseline_sample_details, pilot_n,
                **({"random_seed": config.get("pilot_random_seed", 42),
                    "detail_indices_are_global": True} if resample_reference_pilot else {}),
            )
        )
        state["pilot_indices"] = list(pilot_indices)
        low_available = len(baseline_sample_details or [])
        low_selected = min(pilot_n // 2, low_available)
        state["pilot_sampling"] = {
            "policy": (
                "subset_fixed32_half_lowest_half_seeded_random" if resample_reference_pilot
                else "v35_exact_reference_indices" if reference
                else "v35_fixed32_half_lowest_half_seeded_random"
            ),
            "random_seed": (config.get("pilot_random_seed", 42) if resample_reference_pilot
                            else 42 + len(query_indices) * 1009),
            "dev_a_count": len(query_indices),
            "pilot_count": len(pilot_indices),
            "low_score_selected": low_selected,
            "random_selected": len(pilot_indices) - low_selected,
            "low_score_available": low_available,
        }
        print(
            f"[Pilot sample] {dataset}: DEV-A={len(query_indices)} "
            f"Pilot={len(pilot_indices)} low={low_selected} "
            f"random={len(pilot_indices) - low_selected} no_overlap=True",
            flush=True,
        )
        persist(state)

    pilot_baseline = state.get("pilot_baseline")
    if not pilot_baseline or "sample_details" not in pilot_baseline:
        pilot_baseline = perturbation_pool_core.evaluate(
            dataset=dataset,
            original_prompts=original_prompts,
            target_agent=None,
            candidate_prompt=None,
            indices=pilot_indices,
            model=eval_model,
            repeats=1,
            concurrent=concurrent_batch,
            dry_run=dry_run,
            collect_details=True,
        )
        state["pilot_baseline"] = pilot_baseline
        persist(state)

    shared_failures = list(state.get("shared_failure_examples") or [])
    if not shared_failures:
        shared_failures = perturbation_pool_core.compact_examples(
            baseline_sample_details,
            limit=failure_limit,
            nonperfect_only=True,
            failure_threshold=float(config.get("failure_score_threshold", 0.999)),
        )
        state["shared_failure_examples"] = shared_failures
        persist(state)

    state_agents = state.setdefault("agents", {})
    generated_by_agent: Dict[str, List[AgentVariant]] = {}
    final_order = tuple(variant_generator.RECOMMENDED_STATIC_PERTURBATION_FOCI)

    for agent, original in original_prompts.items():
        agent_state = state_agents.setdefault(agent, {})
        failures = list(agent_state.get("failure_examples") or [])
        reference_failures = (reference or {}).get("dev_diagnostics_by_agent") or {}
        if not failures and agent in reference_failures and agent not in incompatible_agents:
            failures = copy.deepcopy(reference_failures[agent])
            agent_state["failure_examples"] = failures
            agent_state["failure_examples_source"] = "v35_DEV_generation_evidence_only"
        if not failures:
            diagnostic_indices = [int(item["row_index"]) for item in shared_failures]
            if diagnostic_indices and not dry_run:
                diagnostic_eval = perturbation_pool_core.evaluate(
                    dataset=dataset,
                    original_prompts=original_prompts,
                    target_agent=agent,
                    candidate_prompt=original,
                    indices=diagnostic_indices,
                    model=eval_model,
                    repeats=1,
                    concurrent=concurrent_batch,
                    dry_run=False,
                    collect_details=True,
                )
                failures = perturbation_pool_core.attach_agent_trace(
                    shared_failures,
                    diagnostic_eval.get("sample_details") or [],
                )
            else:
                failures = list(shared_failures)
            agent_state["failure_examples"] = failures
            persist(state)

        # ``initial_pilot_candidates`` is an auxiliary search pool.  Up to 36
        # unique prompts are requested, but a slot that keeps reproducing an
        # existing prompt is recorded as mechanism-redundant instead of
        # aborting the formal Stage-1 experiment.  These prompts are never
        # reported as the ten formal Stage-1 perturbations.
        initial_pilot_candidates = agent_state.setdefault(
            "initial_pilot_candidates", {})
        redundant_initial_pilot_slots = agent_state.setdefault(
            "redundant_initial_pilot_slots", {})
        candidates = agent_state.setdefault("candidates", {})

        def stored_candidates(scope: str):
            """Yield prompts from one duplicate namespace.

            The 36-way Pilot pool is auxiliary: it searches for a measured
            prior and is never part of the reported ten-perturbation pool.
            Consequently, an adaptive formal perturbation is allowed to
            reproduce the selected auxiliary prior when that prompt is not
            already one of the other formal slots.  Duplicate prevention is
            still strict *within* each namespace.
            """
            if scope == "pilot":
                for label, stored in initial_pilot_candidates.items():
                    yield f"pilot:{label}", stored
                return
            if scope == "formal":
                for label, stored in candidates.items():
                    yield f"formal:{label}", stored
                return
            raise ValueError(f"unknown duplicate scope: {scope!r}")

        def ensure_unique(item: dict, focus: str,
                          duplicate_scope: str = "formal") -> None:
            candidate_key = perturbation_pool_core.normalise(item.get("prompt", ""))
            duplicate_of = next(
                (
                    stored_focus
                    for stored_focus, stored in stored_candidates(duplicate_scope)
                    if stored.get("id") != item.get("id")
                    and perturbation_pool_core.normalise(stored.get("prompt", ""))
                    == candidate_key
                ),
                None,
            )
            if duplicate_of is not None:
                # Discovery drops duplicates. A fixed ten-slot experiment
                # must not silently replace one with a generic fallback.
                raise RuntimeError(
                    f"strict perturbation-pool duplicate: {dataset}/{agent}/{focus} "
                    f"duplicates {duplicate_of}"
                )

        def generate_unique(focus: str, candidate_index: int,
                            prior: dict = None,
                            duplicate_scope: str = "formal",
                            forbidden_seed: Sequence[dict] = ()) -> dict:
            """One canonical discovery request; duplicates are rejected locally.

            No +100 refill index or forbidden-prompt appendix is permitted.
            Adaptive duplicates advance to the next fresh-DEV prior through
            the existing top-five policy, without changing the focus payload.
            """
            item, rejected = perturbation_pool_core.generate_one(
                dataset=dataset, agent=agent, original=original,
                goal=goals.get(agent, ""), failures=failures, focus=focus,
                candidate_index=candidate_index, model=generator_llm,
                temperature=temperature, dry_run=dry_run, prior=prior,
                max_attempts=generation_attempts,
            )
            if item is None:
                raise _PerturbationSlotExhausted(
                    f"canonical discovery slot exhausted: {dataset}/{agent}/{focus}", rejected)
            candidate_key = perturbation_pool_core.normalise(item.get("prompt", ""))
            forbidden = [(str(x.get("focus", "prior")), x) for x in forbidden_seed]
            forbidden.extend(stored_candidates(duplicate_scope))
            for label, stored in forbidden:
                if perturbation_pool_core.normalise(stored.get("prompt", "")) == candidate_key:
                    raise _DuplicatePerturbationExhausted(
                        f"canonical discovery duplicate: {dataset}/{agent}/{focus}: {label}",
                        list(rejected) + [f"duplicate_of:{label}"],
                    )
            item["generation_rejections"] = list(rejected)
            item["slot_refill_round"] = 1
            return item

        if agent in reference_candidates and not initial_pilot_candidates:
            for raw in sorted(reference_candidates[agent], key=lambda x: x["reference_initial_slot"]):
                item = {key: copy.deepcopy(raw[key]) for key in _REFERENCE_GENERATION_FIELDS if key in raw}
                slot = item["reference_initial_slot"]
                item["pilot_slot"] = slot
                item["generation_replay"] = "v35_initial_prompt_only_fresh_DEV_measurement"
                initial_pilot_candidates[slot] = item
            agent_state["initial_generation_source"] = "v35_reference_original_text"
            persist(state)

        # Match discovery's requested 36-slot initial design: the twelve
        # interpretable initial mechanisms are cycled three times.  This broad
        # pool is measured only to find the adaptive prior.  Repeated samples
        # of one mechanism can legitimately converge to the same prompt; such
        # auxiliary duplicates add no Pilot information and are omitted.
        for position in range(1, pilot_initial_n + 1):
            slot = f"initial_{position:03d}"
            if (
                slot in initial_pilot_candidates
                or slot in redundant_initial_pilot_slots
            ):
                continue
            initial_foci = (
                perturbation_pool_core.INITIAL_FOCI if failures
                else perturbation_pool_core.NO_FAILURE_FOCI
            )
            focus = initial_foci[(position - 1) % len(initial_foci)]
            try:
                item = generate_unique(
                    focus, position, prior=None, duplicate_scope="pilot",
                )
            except _PerturbationSlotExhausted as exc:
                # The 36-way pool is auxiliary: it exists only to measure a
                # broad set of possible priors for adaptive construction.  A
                # mechanism occurrence that repeatedly stays unchanged or
                # violates an immutable signature contributes no Pilot
                # information and should be recorded as redundant rather than
                # aborting the entire dataset.  Provider/network exhaustion is
                # different: do not silently disguise an unavailable service
                # as scientific redundancy.
                if any(":provider_" in reason for reason in exc.rejections):
                    raise
                duplicate_only = isinstance(exc, _DuplicatePerturbationExhausted)
                redundant_initial_pilot_slots[slot] = {
                    "focus": focus,
                    "reason": (
                        "duplicate_exhausted_within_auxiliary_pilot_pool"
                        if duplicate_only
                        else "structural_exhausted_within_auxiliary_pilot_pool"
                    ),
                    "rejections": list(exc.rejections),
                }
                persist(state)
                continue
            item["pilot_slot"] = slot
            initial_pilot_candidates[slot] = item
            persist(state)

        # The formal six non-adaptive mechanisms are fixed in advance.  Take
        # their first generated occurrence deterministically; Pilot scores do
        # not choose or filter these six prompts.
        for formal_position, focus in enumerate(_MAIN_INITIAL_FOCI, start=1):
            if focus in candidates:
                continue
            source_item = next(
                (
                    item for _, item in sorted(initial_pilot_candidates.items())
                    if item.get("focus") == focus
                ),
                None,
            )
            if source_item is None:
                # An auxiliary mechanism may have produced only cross-focus
                # duplicates.  The fixed formal mechanism is still required,
                # so give it an independent formal generation request.  This
                # is not Pilot filtering and remains subject to strict formal
                # uniqueness.
                source_item = generate_unique(
                    focus,
                    perturbation_pool_core.INITIAL_FOCI.index(focus) + 1,
                    prior=None,
                    duplicate_scope="formal",
                )
                source_item["formal_slot_source"] = (
                    "direct_after_auxiliary_redundancy"
                )
            formal_item = copy.deepcopy(source_item)
            formal_item.setdefault(
                "formal_slot_source", source_item.get("pilot_slot")
            )
            formal_item["formal_selection"] = "fixed_first_occurrence_no_pilot_filter"
            ensure_unique(formal_item, focus, duplicate_scope="formal")
            candidates[focus] = formal_item
            persist(state)

        if "sectionwise_rephrase" not in candidates:
            rephrased, source = variant_generator._generate_sectionwise_rephrase(
                prompt=original,
                agent_name=agent,
                dataset=dataset,
                generator_llm=generator_llm,
                dry_run=dry_run,
                max_attempts=generation_attempts,
                # The fallback is local and uses the same protected-marker
                # reconstruction, so it remains structurally valid while
                # preventing a transient provider failure from deleting slot 10.
                allow_fallback=True,
            )
            valid, reason = perturbation_pool_core.candidate_valid(original, rephrased)
            if not valid:
                raise RuntimeError(
                    f"strict sectionwise rephrase failed: {dataset}/{agent}; reason={reason}"
                )
            sectionwise_item = {
                "id": f"{agent}:llm:sectionwise_rephrase:7:"
                      f"{perturbation_pool_core.sha(rephrased)[:12]}",
                "agent": agent,
                "prompt": rephrased,
                "source": source,
                "generation_round": "initial",
                "focus": "sectionwise_rephrase",
                "edit_type": "sectionwise_rephrase",
                "changed_span": "all logical sections, independently rephrased",
                "evidence_link": "frozen original prompt structure",
                "rationale": "Meaning-equivalent section-wise wording control.",
                "expected_mechanism": "Measure wording sensitivity without adding a task rule.",
                "regression_risk": "Equivalent wording may alter model interpretation.",
                "prompt_sha256": perturbation_pool_core.sha(rephrased),
                "length_ratio": len(rephrased) / max(1, len(original)),
                "diff": perturbation_pool_core.text_diff(original, rephrased),
            }
            ensure_unique(sectionwise_item, "sectionwise_rephrase")
            candidates["sectionwise_rephrase"] = sectionwise_item
            persist(state)

        pilot_results = agent_state.setdefault("pilot_results", {})
        for slot, item in initial_pilot_candidates.items():
            if slot in pilot_results:
                item["pilot_dev"] = pilot_results[slot]
                continue
            measured = _pilot_evaluate_candidate(
                dataset=dataset,
                original_prompts=original_prompts,
                agent=agent,
                prompt=item["prompt"],
                pilot_indices=pilot_indices,
                pilot_baseline=pilot_baseline,
                eval_model=eval_model,
                concurrent_batch=concurrent_batch,
                dry_run=dry_run,
            )
            item["pilot_dev"] = measured
            pilot_results[slot] = measured
            persist(state)

        measured_initial = [
            item for item in initial_pilot_candidates.values()
            if not (item.get("pilot_dev") or {}).get("failed")
        ]
        if not measured_initial:
            raise RuntimeError(f"all strict DEV-A Pilot candidates failed: {dataset}/{agent}")
        ranked_initial = sorted(
            measured_initial,
            key=lambda item: (
                -float(item["pilot_dev"].get("mean_delta", float("-inf"))),
                str(item.get("pilot_slot", "")),
            ),
        )
        best = ranked_initial[0]
        # Re-evaluate only the selected prior with sample details.  This adds a
        # single small Pilot pass per agent and equips counterfactual/regression
        # adaptive mechanisms with actual paired evidence instead of an empty
        # summary.  Ranking remains frozen from the original 36-way Pilot.
        prior_detail = agent_state.get("selected_prior_detail") or {}
        if (
            prior_detail.get("candidate_id") != best.get("id")
            or (prior_detail.get("measurement") or {}).get("failed")
        ):
            detail_measurement = _pilot_evaluate_candidate(
                dataset=dataset,
                original_prompts=original_prompts,
                agent=agent,
                prompt=best["prompt"],
                pilot_indices=pilot_indices,
                pilot_baseline=pilot_baseline,
                eval_model=eval_model,
                concurrent_batch=concurrent_batch,
                dry_run=dry_run,
                collect_details=True,
            )
            prior_detail = {
                "candidate_id": best.get("id"),
                "measurement": detail_measurement,
            }
            agent_state["selected_prior_detail"] = prior_detail
            persist(state)

        compact_counterfactual = {}
        detail_measurement = prior_detail.get("measurement") or {}
        baseline_details = list(pilot_baseline.get("sample_details") or [])
        candidate_details = list(detail_measurement.get("sample_details") or [])
        if baseline_details and candidate_details and not detail_measurement.get("failed"):
            counterfactual = _stage1_counterfactual(
                baseline_details, candidate_details, limit=6,
            )
            compact_counterfactual = {
                "DEV-A-Pilot": perturbation_pool_core.compact_counterfactual(
                    counterfactual
                )
            }

        selected_mean_delta = float(best["pilot_dev"]["mean_delta"])
        detailed_mean_delta = (
            float(detail_measurement.get("mean_delta"))
            if not detail_measurement.get("failed")
            and detail_measurement.get("mean_delta") is not None
            else None
        )
        prior = {
            "prompt": best["prompt"],
            "focus": best.get("focus", ""),
            "direction": (
                "positive" if selected_mean_delta > 0
                else "neutral" if selected_mean_delta == 0
                else "negative"
            ),
            "mean_delta": selected_mean_delta,
            "detail_mean_delta": detailed_mean_delta,
            "lcb": selected_mean_delta,
            "positive_repeats": int(selected_mean_delta > 0),
            "counterfactual_compact": compact_counterfactual,
            "expected_mechanism": best.get("expected_mechanism", ""),
            "regression_risk": best.get("regression_risk", ""),
        }
        agent_state["selected_prior_focus"] = best["focus"]
        agent_state["selected_prior_slot"] = best.get("pilot_slot")
        agent_state["selected_prior"] = prior
        persist(state)

        def compact_prior(item: dict, *, rank: int) -> dict:
            mean_delta = float(item["pilot_dev"]["mean_delta"])
            return {
                "prompt": item["prompt"],
                "focus": item.get("focus", ""),
                "direction": (
                    "positive" if mean_delta > 0
                    else "neutral" if mean_delta == 0
                    else "negative"
                ),
                "mean_delta": mean_delta,
                "detail_mean_delta": None,
                "lcb": mean_delta,
                "positive_repeats": int(mean_delta > 0),
                "counterfactual_compact": {},
                "expected_mechanism": item.get("expected_mechanism", ""),
                "regression_risk": item.get("regression_risk", ""),
                "pilot_rank": rank,
                "pilot_slot": item.get("pilot_slot"),
            }

        # Adaptive construction may use a neutral best Pilot candidate.  A
        # single Pilot repeat on a finite 0/1 sample can easily produce an
        # aggregate tie even when the edit mechanism remains informative.
        # Negative Pilot candidates remain ineligible.
        nonnegative_ranked_initial = [
            item for item in ranked_initial
            if float(item["pilot_dev"].get("mean_delta", 0.0)) >= 0.0
        ][:5]
        adaptive_priors = []
        for prior_rank, prior_item in enumerate(nonnegative_ranked_initial, start=1):
            if prior_item.get("id") == best.get("id"):
                ranked_prior = copy.deepcopy(prior)
                ranked_prior["pilot_rank"] = prior_rank
                ranked_prior["pilot_slot"] = prior_item.get("pilot_slot")
            else:
                ranked_prior = compact_prior(prior_item, rank=prior_rank)
            adaptive_priors.append(ranked_prior)

        adaptive_policy_version = "nonnegative-pilot-top5-distinct-v2"
        if agent_state.get("adaptive_prior_policy_version") != adaptive_policy_version:
            # A redundancy decision made under the old rank-1/rank-2 policy
            # must not prevent the new policy from trying ranks 3--5.
            agent_state["redundant_adaptive_slots"] = {}
            agent_state["adaptive_prior_policy_version"] = adaptive_policy_version
            persist(state)
        redundant_slots = agent_state.setdefault("redundant_adaptive_slots", {})

        for adaptive_position, focus in enumerate(_MAIN_ADAPTIVE_FOCI, start=1):
            existing = candidates.get(focus)
            existing_source_slot = (
                existing.get("adaptive_prior_slot") if existing else None
            )
            source_prior = next(
                (
                    row for row in adaptive_priors
                    if row.get("pilot_slot") == existing_source_slot
                ),
                None,
            )
            duplicates_source_prior = bool(
                existing is not None
                and source_prior is not None
                and perturbation_pool_core.normalise(existing.get("prompt", ""))
                == perturbation_pool_core.normalise(source_prior.get("prompt", ""))
            )
            duplicates_other_formal = bool(
                existing is not None
                and any(
                    other_focus != focus
                    and perturbation_pool_core.normalise(other.get("prompt", ""))
                    == perturbation_pool_core.normalise(existing.get("prompt", ""))
                    for other_focus, other in candidates.items()
                )
            )
            # Older checkpoints did not record adaptive_prior_slot. In that
            # case, still invalidate an exact copy of the rank-1 selected prior.
            duplicates_legacy_selected_prior = bool(
                existing is not None
                and existing_source_slot is None
                and perturbation_pool_core.normalise(existing.get("prompt", ""))
                == perturbation_pool_core.normalise(prior.get("prompt", ""))
            )
            if existing is not None and (
                duplicates_source_prior
                or duplicates_other_formal
                or duplicates_legacy_selected_prior
            ):
                # Resume migration: older checkpoints allowed an adaptive slot
                # to copy the selected auxiliary prior verbatim.
                agent_state.setdefault("invalidated_adaptive_slots", {})[focus] = {
                    "reason": (
                        "candidate_duplicate_of_source_prior_or_formal_slot"
                    ),
                    "candidate_id": existing.get("id"),
                }
                candidates.pop(focus, None)
                existing = None
                persist(state)
            if existing is not None or focus in redundant_slots:
                continue

            item = None
            used_prior = None
            used_rank = None
            attempted_priors = []
            forbidden_priors = []
            for candidate_prior in adaptive_priors:
                prior_rank = int(candidate_prior["pilot_rank"])
                prior_slot = candidate_prior.get("pilot_slot")
                forbidden_priors.append({
                    "focus": f"pilot_prior_rank{prior_rank}:{prior_slot}",
                    "prompt": candidate_prior["prompt"],
                })
                try:
                    item = generate_unique(
                        focus,
                        10_001 + perturbation_pool_core.ADAPTIVE_FOCI.index(focus),
                        prior=candidate_prior,
                        duplicate_scope="formal",
                        forbidden_seed=forbidden_priors,
                    )
                    used_prior = candidate_prior
                    used_rank = prior_rank
                    attempted_priors.append({
                        "rank": prior_rank,
                        "slot": prior_slot,
                        "status": "generated-distinct-candidate",
                    })
                    break
                except _DuplicatePerturbationExhausted as exc:
                    attempted_priors.append({
                        "rank": prior_rank,
                        "slot": prior_slot,
                        "status": "duplicate-exhausted",
                        "rejections": exc.rejections,
                    })
                    continue
                except _PerturbationSlotExhausted as exc:
                    # The canonical generator also exhausts a prior when all
                    # bounded attempts fail JSON/placeholder/length guards.
                    # That is not a dataset failure: try the next eligible
                    # prior, just as for duplicates. Provider, billing,
                    # cache and unexpected exceptions are NOT caught here.
                    if any(":provider_" in reason for reason in exc.rejections):
                        raise
                    attempted_priors.append({
                        "rank": prior_rank,
                        "slot": prior_slot,
                        "status": "generation-validation-exhausted",
                        "rejections": exc.rejections,
                    })
                    print(
                        f"[VariantGen adaptive retry] {dataset}/{agent}/{focus}: "
                        f"prior={prior_rank}/{len(adaptive_priors)} "
                        "generation checks exhausted; advancing prior",
                        flush=True,
                    )
                    continue
            if item is None:
                redundant_slots[focus] = {
                    "reason": (
                        "no_valid_distinct_candidate_after_nonnegative_prior_exhaustion"
                        if any(row["status"] == "generation-validation-exhausted"
                               for row in attempted_priors)
                        else "no_distinct_mechanism_after_nonnegative_prior_exhaustion"
                    ),
                    "nonnegative_prior_count": len(nonnegative_ranked_initial),
                    "positive_prior_count": sum(
                        float(item["pilot_dev"].get("mean_delta", 0.0)) > 0.0
                        for item in nonnegative_ranked_initial
                    ),
                    "attempted_prior_count": len(attempted_priors),
                    "attempted_priors": attempted_priors,
                }
                persist(state)
                continue
            item["adaptive_prior_rank"] = used_rank
            item["adaptive_prior_slot"] = used_prior.get("pilot_slot")
            item["adaptive_prior_attempts"] = attempted_priors
            candidates[focus] = item
            redundant_slots.pop(focus, None)
            persist(state)

        required_static = tuple(_MAIN_INITIAL_FOCI) + ("sectionwise_rephrase",)
        missing_required = [
            focus for focus in required_static if focus not in candidates
        ]
        if missing_required:
            raise RuntimeError(
                f"required perturbation slots incomplete: {dataset}/{agent}: "
                f"{missing_required}"
            )
        unresolved_adaptive = [
            focus for focus in _MAIN_ADAPTIVE_FOCI
            if focus not in candidates and focus not in redundant_slots
        ]
        if unresolved_adaptive:
            raise RuntimeError(
                f"adaptive slots neither generated nor marked redundant: "
                f"{dataset}/{agent}: {unresolved_adaptive}"
            )
        ordered_records = [
            candidates[focus] for focus in final_order if focus in candidates
        ]
        # Pilot has exactly one role in the maximum-ten main experiment:
        # choose the measured prior needed to construct the adaptive prompts.
        # It never score-filters a formal prompt. A mechanism-redundant
        # adaptive slot may be omitted; every distinct formal prompt goes
        # directly to the normal full DEV-A x3 Stage-1 evaluation.
        agent_state["selected_finalist_foci"] = [
            item["focus"] for item in ordered_records
        ]
        agent_state["pilot_selection"] = {
            "mode": "full36_prior_only_no_filter_with_sample_evidence",
            "initial_pilot_requested": pilot_initial_n,
            "initial_piloted": len(initial_pilot_candidates),
            "redundant_initial_pilot_slots": sorted(
                redundant_initial_pilot_slots
            ),
            # Six fixed mechanisms plus the sectionwise wording control.
            "formal_initial_candidates": len(required_static),
            "adaptive_piloted": 0,
            "full_dev_candidates": len(ordered_records),
            "formal_slot_budget": len(final_order),
            "nonnegative_adaptive_prior_count": len(nonnegative_ranked_initial),
            "positive_adaptive_prior_count": sum(
                float(item["pilot_dev"].get("mean_delta", 0.0)) > 0.0
                for item in nonnegative_ranked_initial
            ),
            "adaptive_prior_threshold": "mean_delta>=0",
            "adaptive_prior_limit": 5,
            "redundant_adaptive_slots": sorted(redundant_slots),
            "selected_prior_detail_rows": len(candidate_details),
        }
        persist(state)
        records = ordered_records
        generated_by_agent[agent] = _records_to_agent_variants(original, records, encoder)
        print(
            f"[VariantGen exact-pool] {dataset}/{agent}: prior={best['focus']} "
            f"PilotΔL={float(best['pilot_dev']['mean_delta']):+.5f}; "
            f"prior_detail_rows={len(candidate_details)}; "
            f"initial_piloted={len(initial_pilot_candidates)}/{pilot_initial_n} "
            f"initial_redundant={sorted(redundant_initial_pilot_slots)} "
            f"adaptive_generated="
            f"{sum(focus in candidates for focus in _MAIN_ADAPTIVE_FOCI)} "
            f"adaptive_redundant={sorted(redundant_slots)} "
            f"full_DEV-A={len(ordered_records)} "
            "(Pilot repeats=1; filtering=OFF; formal budget<=10)",
            flush=True,
        )

    state["complete"] = True
    persist(state)
    return generated_by_agent, state


def run_aflow_perturbation(
    dataset: str,
    original_prompts: Dict[str, str],
    encoder,
    query_indices: List[int],
    n_perturb: int,
    perturb_mix: dict,
    generator_llm: str,
    eval_model: str,
    n_repeats: int,
    concurrent_batch: int,
    dry_run: bool,
    checkpoint_path: str = None,
    skip_failed_candidates: bool = False,
    max_eval_attempts: int = 3,
    keep_failed_candidates: bool = True,
    perturb_config: dict = None,
) -> Tuple[Dict[str, List[PerturbationResult]],
           Dict[str, List[AgentVariant]],
           float,
           Dict[str, List[dict]]]:
    """Evaluate Stage-1 variants and aggregate repeated evidence."""
    max_eval_attempts = max(1, int(max_eval_attempts or 1))
    keep_failed_candidates = bool(keep_failed_candidates or skip_failed_candidates)
    perturb_config = dict(perturb_config or {})
    execution_fingerprint = _pool_execution_fingerprint(
        dataset=dataset, original_prompts=original_prompts, query_indices=query_indices,
        generator_llm=generator_llm, eval_model=eval_model,
        config={"perturb_mix": perturb_mix, "perturb_config": perturb_config,
                "n_repeats": n_repeats, "reference_root": perturb_mix.get("reference_root")},
        dry_run=dry_run,
    )
    ckpt = _load_checkpoint(checkpoint_path)
    if (
        ckpt
        and perturb_mix.get("construction_protocol") == "perturbation_pool_exact_v1"
        and (
            (ckpt.get("pool_build") or {}).get("protocol_version") != POOL_PROTOCOL_VERSION
            or ckpt.get("execution_fingerprint") != execution_fingerprint
        )
    ):
        # The variants shortcut below must not bypass the new pool protocol.
        # Keep no old candidate scores/results; the old file remains intact
        # until normal checkpoint persistence in the caller's chosen run dir.
        ckpt = None

    def save_checkpoint(data):
        data["execution_fingerprint"] = execution_fingerprint
        _save_checkpoint(checkpoint_path, data)


    # The exact pool needs genuine DEV-A failures. Evaluate the baseline before
    # generation so full-prompt initial candidates and the later measured
    # adaptive candidates use the same DEV-A-only evidence as discovery;
    # sectionwise_rephrase sees only the frozen original prompt.
    if ckpt and "baseline_L_dev" in ckpt:
        baseline_L = float(ckpt["baseline_L_dev"])
        baseline_loss_repeats = list(ckpt.get("baseline_loss_repeats") or [baseline_L])
        baseline_sample_details = list(ckpt.get("baseline_sample_details") or [])
        baseline_completion_stats = dict(
            ckpt.get("baseline_completion_stats") or {})
    else:
        baseline_eval = perturbation_pool_core.evaluate(
            dataset=dataset,
            original_prompts=original_prompts,
            target_agent=None,
            candidate_prompt=None,
            indices=query_indices,
            model=eval_model,
            repeats=n_repeats,
            concurrent=concurrent_batch,
            dry_run=dry_run,
            collect_details=True,
        )
        baseline_L = float(baseline_eval["loss"])
        baseline_loss_repeats = list(baseline_eval.get("loss_repeats") or [baseline_L])
        baseline_sample_details = list(baseline_eval.get("sample_details") or [])
        baseline_completion_stats = {
            "finish_reason_counts": dict(
                baseline_eval.get("finish_reason_counts") or {}),
            "max_tokens_counts": dict(
                baseline_eval.get("max_tokens_counts") or {}),
            "response_count": int(baseline_eval.get("response_count", 0) or 0),
            "output_cap_hits": int(
                baseline_eval.get("output_cap_hits", 0) or 0),
            "output_cap_rate": float(
                baseline_eval.get("output_cap_rate", 0.0) or 0.0),
        }


    pool_build = dict((ckpt or {}).get("pool_build") or {})
    if ckpt and ckpt.get("variants"):
        agent_variants = ckpt["variants"]
    elif perturb_mix.get("construction_protocol") == "perturbation_pool_exact_v1":
        if int(n_perturb) != len(variant_generator.RECOMMENDED_STATIC_PERTURBATION_FOCI):
            raise ValueError(
                "perturbation_pool_exact_v1 requires exactly "
                f"{len(variant_generator.RECOMMENDED_STATIC_PERTURBATION_FOCI)} slots; "
                f"got n_perturb={n_perturb}"
            )

        def persist_pool_build(pool_build_state: dict) -> None:
            save_checkpoint({
                "complete": False,
                "dataset": dataset,
                "variants": {},
                "pool_build": pool_build_state,
                "perturb_config": perturb_config,
                "baseline_L_dev": baseline_L,
                "baseline_loss_repeats": baseline_loss_repeats,
                "baseline_sample_details": baseline_sample_details,
                "baseline_completion_stats": baseline_completion_stats,
                "results": dict((ckpt or {}).get("results") or {}),
                "evidence": dict((ckpt or {}).get("evidence") or {}),
                "failed_candidates": list(
                    (ckpt or {}).get("failed_candidates") or []),
            })

        agent_variants, pool_build = _build_exact_perturbation_pool(
            dataset=dataset,
            original_prompts=original_prompts,
            baseline_sample_details=baseline_sample_details,
            query_indices=query_indices,
            generator_llm=generator_llm,
            eval_model=eval_model,
            concurrent_batch=concurrent_batch,
            dry_run=dry_run,
            encoder=encoder,
            config=perturb_mix,
            saved_state=pool_build,
            persist=persist_pool_build,
        )
    else:
        agent_variants = generate_variants(
            original_prompts=original_prompts,
            num_per_agent=n_perturb,
            mix=perturb_mix,
            generator_llm=generator_llm,
            encoder=encoder,
            dry_run=dry_run,
            dataset=dataset,
            baseline_sample_details=baseline_sample_details,
        )
        # Legacy experiments retain their historical farthest-first selection.
        # The recommended operator pool evaluates every named operator exactly
        # once, so it must not be silently re-ranked by embedding distance.
        if not perturb_mix.get("operator_pool"):
            for a in list(agent_variants):
                agent_variants[a] = _farthest_first(agent_variants[a], n_perturb)

    results: Dict[str, List[PerturbationResult]] = (ckpt or {}).get("results", {}) or {}
    evidence: Dict[str, List[dict]] = (ckpt or {}).get("evidence", {}) or {}
    failed_candidates = list((ckpt or {}).get("failed_candidates", []) or [])

    save_checkpoint({
        "complete": False, "dataset": dataset, "variants": agent_variants,
        "pool_build": pool_build,
        "perturb_config": perturb_config,
        "baseline_L_dev": baseline_L, "baseline_loss_repeats": baseline_loss_repeats,
        "baseline_sample_details": baseline_sample_details,
        "baseline_completion_stats": baseline_completion_stats,
        "results": results, "evidence": evidence,
        "failed_candidates": failed_candidates,
    })

    total = sum(len(v) for v in agent_variants.values())
    done = sum(max(0, len(results.get(a, [])) - 1) for a in agent_variants)


    for agent_name, variants in agent_variants.items():
        if not results.get(agent_name):
            results[agent_name] = [PerturbationResult(
                agent_name=agent_name, variant_idx=0, L_observed=baseline_L,
                variance=float(np.var(baseline_loss_repeats, ddof=1)) if len(baseline_loss_repeats) > 1 else 0.0,
                task_level_losses=list(baseline_loss_repeats),
                sample_details=list(baseline_sample_details),
            )]
        evidence.setdefault(agent_name, [])
        completed_variant_idx = {r.variant_idx for r in results[agent_name]}
        for v_idx, variant in enumerate(variants, start=1):
            if v_idx in completed_variant_idx:
                continue
            L = None
            loss_repeats = None
            sample_details = None
            completion_stats = {}
            last_error = None
            try:
                for attempt in range(1, max_eval_attempts + 1):
                    try:
                        eval_out = perturbation_pool_core.evaluate(
                            dataset=dataset,
                            original_prompts=original_prompts,
                            target_agent=agent_name,
                            candidate_prompt=variant.variant_text,
                            indices=query_indices,
                            model=eval_model,
                            repeats=n_repeats,
                            concurrent=concurrent_batch,
                            dry_run=dry_run,
                            collect_details=True,
                        )
                        L = float(eval_out["loss"])
                        loss_repeats = list(eval_out.get("loss_repeats") or [L])
                        sample_details = list(eval_out.get("sample_details") or [])
                        completion_stats = {
                            "finish_reason_counts": dict(
                                eval_out.get("finish_reason_counts") or {}),
                            "max_tokens_counts": dict(
                                eval_out.get("max_tokens_counts") or {}),
                            "response_count": int(
                                eval_out.get("response_count", 0) or 0),
                            "output_cap_hits": int(
                                eval_out.get("output_cap_hits", 0) or 0),
                            "output_cap_rate": float(
                                eval_out.get("output_cap_rate", 0.0) or 0.0),
                        }
                        break
                    except IFIPError as e:
                        last_error = e
                        if attempt < max_eval_attempts:
                            print(
                                f"    [WARN] {dataset} perturb eval failed "
                                f"({agent_name} #{v_idx}, attempt {attempt}/{max_eval_attempts}); retrying ...",
                                flush=True,
                            )
                        else:
                            raise
            except IFIPError as e:
                if not keep_failed_candidates:
                    raise
                L = baseline_L
                loss_repeats = list(baseline_loss_repeats)
                sample_details = list(baseline_sample_details)
                completion_stats = dict(baseline_completion_stats)
                failed_candidates.append({
                    "agent": agent_name,
                    "variant_idx": v_idx,
                    "attempts": max_eval_attempts,
                    "kept_as_neutral": True,
                    "reason": str(e)[:1000],
                })
                report_invalid(
                    "rq2_perturbation.run_aflow_perturbation",
                    "candidate eval failed after retries — kept with neutral loss",
                    dataset=dataset, agent=agent_name, variant_idx=v_idx,
                    attempts=max_eval_attempts,
                )
            if L is None:
                if last_error and not keep_failed_candidates:
                    raise last_error
                L = baseline_L
            if loss_repeats is None:
                loss_repeats = [float(L)]
            if sample_details is None:
                sample_details = []
            results[agent_name].append(PerturbationResult(
                agent_name=agent_name, variant_idx=v_idx, L_observed=L,
                variance=float(np.var(loss_repeats, ddof=1)) if len(loss_repeats) > 1 else 0.0,
                task_level_losses=list(loss_repeats),
                sample_details=list(sample_details),
            ))
            mean_delta = float(baseline_L - L)
            delta_repeats = _paired_deltas(baseline_loss_repeats, loss_repeats, mean_delta)
            evidence[agent_name].append({
                "prompt": variant.variant_text,
                "delta": mean_delta,
                "se": _se(delta_repeats),
                "delta_repeats": delta_repeats,
                "source": getattr(variant, "source", "unknown"),
                "edit_type": getattr(variant, "edit_type", ""),
                "origin": "stage1",
                "counterfactual": _stage1_counterfactual(
                    baseline_sample_details, sample_details),
                "finish_reason_counts": dict(
                    completion_stats.get("finish_reason_counts") or {}),
                "max_tokens_counts": dict(
                    completion_stats.get("max_tokens_counts") or {}),
                "response_count": int(
                    completion_stats.get("response_count", 0) or 0),
                "output_cap_hits": int(
                    completion_stats.get("output_cap_hits", 0) or 0),
                "output_cap_rate": float(
                    completion_stats.get("output_cap_rate", 0.0) or 0.0),
            })
            if int(completion_stats.get("output_cap_hits", 0) or 0) > 0:
                print(
                    f"    [Stage1 completion-cap] {agent_name} variant={v_idx} "
                    f"source={getattr(variant, 'source', 'unknown')} "
                    f"edit_type={getattr(variant, 'edit_type', '') or '-'} "
                    f"hits={int(completion_stats.get('output_cap_hits', 0))}/"
                    f"{int(completion_stats.get('response_count', 0))} "
                    f"finish={completion_stats.get('finish_reason_counts') or {}}",
                    flush=True,
                )
            done += 1
            completed_variant_idx.add(v_idx)
            save_checkpoint({
                "complete": False, "dataset": dataset, "variants": agent_variants,
                "pool_build": pool_build,
                "perturb_config": perturb_config,
                "baseline_L_dev": baseline_L,
                "baseline_loss_repeats": baseline_loss_repeats,
                "baseline_sample_details": baseline_sample_details,
                "baseline_completion_stats": baseline_completion_stats,
                "results": results, "evidence": evidence,
                "failed_candidates": failed_candidates,
            })
            progress_bar(f"{dataset} probe(DEV)", done, total)

    save_checkpoint({
        "complete": True, "dataset": dataset, "variants": agent_variants,
        "pool_build": pool_build,
        "perturb_config": perturb_config,
        "baseline_L_dev": baseline_L,
        "baseline_loss_repeats": baseline_loss_repeats,
        "baseline_sample_details": baseline_sample_details,
        "baseline_completion_stats": baseline_completion_stats,
        "results": results, "evidence": evidence,
        "failed_candidates": failed_candidates,
    })
    return results, agent_variants, baseline_L, evidence


def _load_checkpoint(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and not data.get("complete", True):
            return data
    except Exception as e:  # noqa: BLE001
        report_invalid("rq2_perturbation._load_checkpoint",
                       f"cannot load perturb checkpoint: {type(e).__name__}",
                       path=path)
    return None


def _save_checkpoint(path: str, data: dict) -> None:
    if not path:
        return
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)


def _farthest_first(variants: List[AgentVariant], k: int) -> List[AgentVariant]:
    """Select directions by farthest-first traversal."""
    if len(variants) <= k:
        return variants
    dirs = [v.unit_direction for v in variants]
    selected = [0]
    while len(selected) < k:
        best_i, best_d = None, -1.0
        for i in range(len(variants)):
            if i in selected:
                continue

            d = min(float(np.linalg.norm(dirs[i] - dirs[j])) for j in selected)
            if d > best_d:
                best_d, best_i = d, i
        selected.append(best_i)
    return [variants[i] for i in selected]
