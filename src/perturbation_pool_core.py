"""Shared candidate-generation core for discovery and the IFIP main Stage 1.

This module is deliberately free of experiment orchestration.  Both the
large perturbation-pool discovery script and the fixed ten-slot main experiment
use these exact functions, so an operator name cannot silently acquire a
different prompt-construction meaning in the two paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src import aflow_adapter, llm_cache, variant_generator


CODE_DATASETS = {"MBPP", "HumanEval"}
QA_DATASETS = {"NQ", "HotpotQA", "DROP"}
MATH_DATASETS = {"MATH"}
EVAL_CHUNK_SIZE_BY_DATASET = {"HotpotQA": 24, "DROP": 64, "MATH": 32}
INITIAL_FOCI = (
    "critique_repair",
    "constraint_recovery",
    "agent_contract",
    "downstream_interface",
    "verification",
    "edge_cases",
    "output_contract",
    "ambiguity_resolution",
    "example_rule_induction",
    "semantic_precision",
    "decomposition_control",
    "residual_preserving_patch",
)
ADAPTIVE_FOCI = (
    "adaptive_score_conditioned",
    "adaptive_counterfactual_repair",
    "adaptive_residual_crossover",
    "adaptive_regression_repair",
)
NO_FAILURE_FOCI = (
    "agent_contract", "downstream_interface", "output_contract",
    "semantic_precision", "residual_preserving_patch",
)
GENERATION_PROTOCOL_VERSION = "v35-canonical-request-v1"
GENERATION_MAX_ATTEMPTS = 4
GENERATION_TEMPERATURE = 0.7
PILOT_SAMPLES = 32

CHINESE_RE = re.compile(r"[\u3400-\u9fff]")
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
SPACE_RE = re.compile(r"\s+")


def sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def normalise(text: str) -> str:
    return SPACE_RE.sub(" ", text or "").strip()


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def se(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values) / len(values) ** 0.5)


def clip(text: Any, limit: int = 900) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    head = max(1, int(limit * 0.68))
    tail = max(1, limit - head - 28)
    return value[:head] + "\n...[middle omitted]...\n" + value[-tail:]


def extract_json_object(content: str) -> Optional[dict]:
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def text_diff(original: str, candidate: str, max_lines: int = 80) -> str:
    import difflib

    lines = list(difflib.unified_diff(
        original.splitlines(), candidate.splitlines(),
        fromfile="original", tofile="candidate", lineterm="", n=2,
    ))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... diff truncated ..."]
    return "\n".join(lines)


def candidate_valid(original: str, candidate: str) -> Tuple[bool, str]:
    candidate = str(candidate or "").strip()
    if not candidate:
        return False, "empty"
    if normalise(candidate) == normalise(original):
        return False, "unchanged"
    if CHINESE_RE.search(candidate):
        return False, "contains_non_English_CJK_text"
    if PLACEHOLDER_RE.findall(candidate) != PLACEHOLDER_RE.findall(original):
        return False, "placeholder_sequence_changed"
    if not variant_generator._preserves_critical(original, candidate):
        return False, "critical_token_or_signature_changed"
    ratio = len(candidate) / max(1, len(original))
    if ratio < 0.50:
        return False, f"too_short:{ratio:.3f}"
    if ratio > 3.00:
        return False, f"too_long:{ratio:.3f}"
    return True, "ok"


def compact_examples(details: Sequence[dict], limit: int = 4,
                     nonperfect_only: bool = True,
                     failure_threshold: float = 0.999) -> List[dict]:
    source_rows = list(details or [])
    if nonperfect_only:
        source_rows = [
            row for row in source_rows
            if float(row.get("score", 1.0)) < failure_threshold
        ]
    rows = sorted(source_rows, key=lambda x: (
        float(x.get("score", 1.0)), int(x.get("row_index", 0)),
    ))
    out = []
    for row in rows[:max(0, int(limit))]:
        out.append({
            "row_index": int(row.get("row_index", 0)),
            "mean_score": round(float(row.get("score", 0.0)), 6),
            "score_repeats": [
                round(float(x), 6) for x in row.get("score_repeats", [])
            ],
            "input": clip(row.get("input", ""), 950),
            "prediction": clip(row.get("prediction", ""), 850),
            "expected_or_error": clip(row.get("expected", ""), 850),
            "target_trace": [
                {
                    "input": clip(call.get("input", ""), 450),
                    "output": clip(call.get("output", ""), 450),
                    "error": clip(call.get("error", ""), 250),
                }
                for call in (row.get("target_trace") or [])[:3]
                if isinstance(call, dict)
            ],
        })
    return out


def attach_agent_trace(baseline_failures: Sequence[dict],
                       diagnostic_details: Sequence[dict]) -> List[dict]:
    """Exact discovery semantics: retain baseline identity, attach trace by order."""
    import copy

    details = list(diagnostic_details or [])
    merged: List[dict] = []
    for pos, original_row in enumerate(baseline_failures):
        row = copy.deepcopy(original_row)
        n_repeats = len(row.get("score_repeats") or [])
        row["evidence_source"] = f"all_DEV-A_baseline_{n_repeats or 'unknown'}repeat"
        row["trace_source"] = "same_failure_row_target_agent_1repeat"
        if pos < len(details):
            diagnostic = details[pos]
            row["diagnostic_repeat_score"] = round(
                float(diagnostic.get("score", 0.0)), 6)
            row["diagnostic_prediction"] = clip(
                diagnostic.get("prediction", ""), 850)
            row["target_trace"] = [
                {
                    "input": clip(call.get("input", ""), 450),
                    "output": clip(call.get("output", ""), 450),
                    "error": clip(call.get("error", ""), 250),
                }
                for call in (diagnostic.get("target_trace") or [])[:3]
                if isinstance(call, dict)
            ]
        merged.append(row)
    return merged


def agent_guidance(dataset: str, agent: str, goal: str) -> str:
    lower = agent.lower()
    if dataset in CODE_DATASETS:
        if "test" in lower:
            return (
                "Focus on executable, discriminative tests, boundary cases, and assertions "
                "that expose incorrect implementations without rejecting valid alternatives."
            )
        if "fix" in lower or "refine" in lower:
            return (
                "Focus on diagnosing the concrete execution failure, making a minimal repair, "
                "and preserving behavior that already passes."
            )
        if "ensemble" in lower:
            return (
                "Focus on comparing candidate implementations by correctness, signature "
                "compliance, edge cases, and executable evidence."
            )
        return (
            "Focus on producing complete executable Python, preserving the exact function "
            "signature, required imports, return type, and difficult edge cases."
        )
    if dataset in QA_DATASETS:
        if "format" in lower or "extract" in lower:
            return (
                "Focus on extracting the shortest answer that satisfies the benchmark's "
                "answer format without losing essential entities, numbers, or spans."
            )
        if "ensemble" in lower:
            return (
                "Focus on evidence-grounded candidate comparison, conflict resolution, and "
                "avoiding unsupported majority decisions."
            )
        return (
            "Focus on answering from the supplied evidence, resolving multi-hop dependencies, "
            "and avoiding unsupported elaboration."
        )
    if dataset in MATH_DATASETS:
        if "program" in lower:
            return (
                "Focus on a correct symbolic or computational plan, variable definitions, "
                "boundary conditions, and verification of the computed result."
            )
        if "refine" in lower or "ensemble" in lower:
            return (
                "Focus on detecting algebraic, logical, and formatting errors while preserving "
                "correct reasoning and the required final answer form."
            )
        return (
            "Focus on explicit mathematical reasoning, assumption tracking, verification, "
            "and a correctly formatted final answer."
        )
    return goal or "Improve the agent's task performance while preserving its contract."


def focus_instruction(focus: str) -> str:
    return {
        "critique_repair": (
            "Following the ProTeGi principle, infer one specific and actionable critique from "
            "the observed failures, then edit the prompt in the opposite semantic direction "
            "of that critique. The repair must generalize beyond the shown examples."
        ),
        "constraint_recovery": (
            "Identify why an explicit task constraint was ignored or misapplied, then add a "
            "general procedure for recovering and enforcing constraints from each supplied "
            "input. Do not copy the particular constraint from the evidence row."
        ),
        "agent_contract": (
            "Use a program-aware proposal: clarify only this agent's responsibility, inputs, "
            "decision boundary, and what it must not do. Do not add generic expertise claims."
        ),
        "downstream_interface": (
            "Use the multi-stage workflow structure: clarify the information or artifact this "
            "agent must hand to downstream agents, while preserving the existing output type."
        ),
        "verification": (
            "Add a concrete, task-relevant verification action that the agent can actually "
            "perform before returning its output. Name what is checked and what to do on failure."
        ),
        "edge_cases": (
            "Improve handling of the most plausible boundary or ambiguity exposed by the "
            "examples, without enumerating irrelevant cases."
        ),
        "output_contract": (
            "Clarify only the output contract that affects downstream parsing or evaluation. "
            "Do not force brevity when intermediate reasoning or code is required."
        ),
        "ambiguity_resolution": (
            "Resolve one ambiguous instruction or precedence conflict evidenced by the failures. "
            "State a deterministic decision rule instead of adding generic detail."
        ),
        "example_rule_induction": (
            "Induce a reusable meta-procedure from the low-scoring examples, but do not copy an "
            "example, its task rule, answer, identifier, literal value, or dataset-specific "
            "surface form."
        ),
        "semantic_precision": (
            "Apply a semantics-preserving precision edit: replace a vague or misleading phrase "
            "with clearer task-equivalent wording. Do not change the intended task."
        ),
        "decomposition_control": (
            "Add only a short task-specific decomposition or decision sequence when the evidence "
            "shows omitted dependencies. Do not add generic chain-of-thought boilerplate."
        ),
        "residual_preserving_patch": (
            "Following residual prompt refinement, retain every useful original instruction and "
            "make the smallest evidence-backed local patch that addresses semantic drift risk."
        ),
        "adaptive_score_conditioned": (
            "Use the measured prior score and repeat stability as optimization history. Retain "
            "the prior's likely beneficial mechanism and produce a distinct local refinement."
        ),
        "adaptive_counterfactual_repair": (
            "Use the measured fixed, broken, and still-hard examples. Preserve behavior on fixed "
            "rows while repairing the most coherent broken or unresolved failure mechanism."
        ),
        "adaptive_residual_crossover": (
            "Combine only the demonstrated strength of the measured prior with the unchanged "
            "original prompt, as a residual crossover; discard unsupported prior edits."
        ),
        "adaptive_regression_repair": (
            "Keep the measured gain but explicitly repair its recorded regression risk using "
            "the smallest possible instruction change."
        ),
    }[focus]


def generation_prompt(*, dataset: str, agent: str, original: str, goal: str,
                      failures: Sequence[dict], focus: str, candidate_index: int,
                      generation_attempt: int, rejection_feedback: Sequence[str],
                      prior: Optional[dict] = None,
                      forbidden_candidates: Sequence[dict] = ()) -> str:
    """The v35 discovery payload, including its original length instruction.

    Duplicate detection is orchestration, not extra generator evidence.  The
    compatibility ``forbidden_candidates`` argument never enters this payload.
    Additional main-run prior audit fields likewise must not alter the text.
    """
    prior_block = ""
    if prior:
        prior_block = (
            "\n\nMEASURED PRIOR CANDIDATE:\n"
            f"mean_delta={prior.get('mean_delta', 0.0):+.6f}; "
            f"lcb={prior.get('lcb', 0.0):+.6f}; "
            f"positive_repeats={prior.get('positive_repeats', 0)}\n"
            f"Prompt:\n{prior.get('prompt', '')}\n\n"
            f"Original-to-prior diff:\n"
            f"{text_diff(original, str(prior.get('prompt', '')))}\n\n"
            f"Counterfactual summary:\n"
            f"{json.dumps(prior.get('counterfactual_compact', {}), ensure_ascii=False)}"
        )
    placeholders = PLACEHOLDER_RE.findall(original)
    rejection_block = ""
    if rejection_feedback:
        rejection_block = (
            "\n\nPREVIOUS ATTEMPTS WERE REJECTED FOR:\n- "
            + "\n- ".join(rejection_feedback[-4:])
            + "\nCorrect these issues; do not repeat the rejected edit."
        )
    return (
        "You are constructing a measured prompt perturbation for a multi-agent workflow.\n"
        "The objective is a likely positive task-level intervention, not stylistic variety.\n"
        "Return exactly one JSON object with keys: prompt, edit_type, changed_span, "
        "evidence_link, rationale, expected_mechanism, regression_risk. Do not return "
        "markdown or commentary outside the JSON object.\n\n"
        "HARD REQUIREMENTS:\n"
        "- Write every instruction in English.\n"
        "- Preserve the full original task and all unaffected instructions.\n"
        "- Preserve every placeholder exactly once, in the same order.\n"
        "- Preserve code signatures, LaTeX commands, protected markers, and required formats.\n"
        "- Do not summarize or globally rewrite the prompt.\n"
        "- Do not add generic role claims or generic 'think step by step' text unless the "
        "measured failures specifically justify it.\n"
        "- The new prompt must be between 50% and 300% of the original character length.\n"
        "- Make one interpretable intervention whose causal effect can be measured.\n\n"
        "QUALITY REQUIREMENTS:\n"
        "- evidence_link must name the supplied DEV-A row index or trace behavior that supports "
        "the edit; do not claim evidence that is not present.\n"
        "- changed_span must identify the original span being changed or the exact insertion "
        "location.\n"
        "- expected_mechanism must predict a concrete behavioral change, not merely 'better "
        "accuracy' or 'more clarity'.\n"
        "- Do not copy sample-specific answers, constants, names, or code into the prompt.\n"
        "- When the original contains a task/input placeholder, do not add any concrete "
        "object-level rule or edge case taken from a DEV-A row. Convert evidence only into a "
        "general procedure that applies to every possible placeholder value.\n"
        "- If the evidence does not support the assigned focus, make the most conservative "
        "evidence-backed version of that focus.\n\n"
        f"DATASET: {dataset}\n"
        f"TARGET AGENT: {agent}\n"
        f"AGENT GOAL: {goal}\n"
        f"AGENT-SPECIFIC GUIDANCE: {agent_guidance(dataset, agent, goal)}\n"
        f"FOCUS FOR THIS CANDIDATE: {focus_instruction(focus)}\n"
        f"CANDIDATE INDEX: {candidate_index}\n"
        f"GENERATION ATTEMPT: {generation_attempt}\n"
        f"REQUIRED PLACEHOLDER SEQUENCE: {placeholders}\n\n"
        f"GENUINE NON-PERFECT DEV-A EXAMPLES "
        f"(empty means no failure evidence is available):\n"
        f"{json.dumps(list(failures), ensure_ascii=False, indent=2)}"
        f"{prior_block}{rejection_block}\n\n"
        f"ORIGINAL PROMPT:\n{original}"
    )


def dry_candidate(original: str, dataset: str, agent: str,
                  focus: str, idx: int) -> dict:
    # Dry runs validate orchestration rather than model quality. Retain a
    # unique focus label so deduplication and ordering are exercised.
    prompt = original + (
        f"\n\n{idx}: {focus.replace('_', ' ')}."
    )
    return {
        "prompt": prompt,
        "rationale": "offline dry-run candidate",
        "expected_mechanism": focus,
        "regression_risk": "dry-run only",
    }


def generate_one(*, dataset: str, agent: str, original: str, goal: str,
                 failures: Sequence[dict], focus: str, candidate_index: int,
                 model: str, temperature: float, dry_run: bool,
                 prior: Optional[dict] = None,
                 forbidden_candidates: Sequence[dict] = (),
                 max_attempts: int = 4) -> Tuple[Optional[dict], List[str]]:
    rejected: List[str] = []
    for attempt in range(1, max_attempts + 1):
        if dry_run:
            parsed = dry_candidate(original, dataset, agent, focus, candidate_index)
        else:
            request = generation_prompt(
                dataset=dataset, agent=agent, original=original, goal=goal,
                failures=failures, focus=focus, candidate_index=candidate_index,
                generation_attempt=attempt, rejection_feedback=rejected, prior=prior,
                forbidden_candidates=forbidden_candidates,
            )
            response = llm_cache.call_llm_cached(
                provider="openai", model=model,
                messages=[{"role": "user", "content": request}],
                temperature=temperature, max_tokens=2048,
            )
            parsed = extract_json_object(response.get("content", ""))
        if not parsed:
            rejected.append(f"attempt{attempt}:invalid_json")
            continue
        required_explanations = (
            "edit_type", "changed_span", "evidence_link",
            "expected_mechanism", "regression_risk",
        )
        missing = [
            key for key in required_explanations
            if not str(parsed.get(key, "")).strip()
        ]
        if missing and not dry_run:
            rejected.append(f"attempt{attempt}:missing_explanations:{','.join(missing)}")
            continue
        candidate = str(parsed.get("prompt", "")).strip()
        valid, reason = candidate_valid(original, candidate)
        if not valid:
            rejected.append(f"attempt{attempt}:{reason}")
            continue
        item = {
            "id": f"{agent}:llm:{focus}:{candidate_index}:{sha(candidate)[:12]}",
            "agent": agent,
            "prompt": candidate,
            "source": (
                "llm_adaptive_positive" if candidate_index >= 10_000
                else "llm_positive_seeking"
            ),
            "generation_round": "adaptive" if candidate_index >= 10_000 else "initial",
            "focus": focus,
            "edit_type": str(parsed.get("edit_type", focus))[:300],
            "changed_span": str(parsed.get("changed_span", ""))[:1200],
            "evidence_link": str(parsed.get("evidence_link", ""))[:1200],
            "rationale": str(parsed.get("rationale", ""))[:1200],
            "expected_mechanism": str(parsed.get("expected_mechanism", ""))[:1200],
            "regression_risk": str(parsed.get("regression_risk", ""))[:1200],
            "generation_attempt": attempt,
            "prompt_sha256": sha(candidate),
            "length_ratio": len(candidate) / max(1, len(original)),
            "diff": text_diff(original, candidate),
        }
        return item, rejected
    return None, rejected


def resolve_pilot_sample_count(dev_a_count: int, requested: Optional[int] = None,
                               fraction: float = 0.25,
                               min_samples: int = 32,
                               max_samples: Optional[int] = 32) -> int:
    """Fixed v35 Pilot budget; explicit small counts support offline fixtures.

    Legacy fraction/min/max arguments remain accepted but do not alter the
    fixed default.  Production explicitly requests 32 and validates references.
    """
    dev_a_count = max(0, int(dev_a_count))
    if dev_a_count == 0:
        return 0
    if requested is not None:
        return min(dev_a_count, max(0, int(requested)))
    return min(dev_a_count, PILOT_SAMPLES)


def pilot_indices(dev_indices: Sequence[int], baseline_details: Sequence[dict],
                  n: int, *, random_seed=None, detail_indices_are_global=False) -> List[int]:
    """v35 fallback: half lowest-score rows plus seeded non-overlapping fill.

    Historical exact indices, when available, are loaded by pilot_reference.
    Like v35 this quota can include perfect rows; filtering them changed the
    seeded remainder and thus the actual tasks even with an identical seed.
    """
    dev_indices = list(dev_indices)
    n = min(max(0, int(n)), len(dev_indices))
    if n <= 0:
        return []
    by_local_row = {
        int(x.get("row_index", -1)): float(x.get("score", 1.0))
        for x in baseline_details or []
    }
    if n == len(dev_indices):
        return dev_indices
    low_target = max(1, n // 2)
    low_local = [
        i
        for i, score in sorted(by_local_row.items(), key=lambda kv: (kv[1], kv[0]))
        if 0 <= i < len(dev_indices)
    ][:low_target]
    if detail_indices_are_global:
        if not set(by_local_row).issubset(dev_indices):
            raise ValueError("Pilot baseline details contain tasks outside current DEV-A")
        selected = {i for i, score in sorted(
            by_local_row.items(), key=lambda kv: (kv[1], kv[0]))[:low_target]}
    else:
        selected = {dev_indices[i] for i in low_local}
    remaining = [x for x in dev_indices if x not in selected]
    rng = random.Random(42 + len(dev_indices) * 1009 if random_seed is None else random_seed)
    rng.shuffle(remaining)
    for idx in remaining:
        selected.add(idx)
        if len(selected) >= n:
            break
    return sorted(selected)


def paired_stats(baseline_losses: Sequence[float], candidate_losses: Sequence[float],
                 z: float) -> dict:
    if len(baseline_losses) == len(candidate_losses) and baseline_losses:
        deltas = [float(b) - float(c) for b, c in zip(baseline_losses, candidate_losses)]
    else:
        deltas = [mean(baseline_losses) - mean(candidate_losses)]
    mean_delta = mean(deltas)
    stderr = se(deltas)
    return {
        "mean_delta": mean_delta,
        "se": stderr,
        "lcb": mean_delta - z * stderr,
        "ucb": mean_delta + z * stderr,
        "delta_repeats": deltas,
        "positive_repeats": sum(x > 0 for x in deltas),
        "nonnegative_repeats": sum(x >= 0 for x in deltas),
        "negative_repeats": sum(x < 0 for x in deltas),
    }


def choose_dev_finalists(
    candidates: Sequence[dict],
    positive_limit: int,
    negative_limit: int,
    max_cap_rate: float,
    pilot_dev_baseline_cap_rate: float = 0.0,
) -> List[dict]:
    """Exact discovery DEV-A Pilot finalist selection; TEST is never read."""
    cap_margin = max(0.0, float(max_cap_rate))
    dev_cap_limit = float(pilot_dev_baseline_cap_rate) + cap_margin

    def cap_safe(item: dict) -> bool:
        return bool(
            float((item.get("pilot_dev") or {}).get(
                "output_cap_rate", 0.0) or 0.0) <= dev_cap_limit
        )

    valid = []
    for item in candidates:
        dev = item.get("pilot_dev") or {}
        try:
            delta = float(dev.get("mean_delta", float("-inf")))
        except (TypeError, ValueError):
            delta = float("-inf")
        if not dev.get("failed") and math.isfinite(delta):
            valid.append(item)
    positive = [
        item for item in valid
        if (
            item.get("source") != "negative_control"
            and float(item["pilot_dev"]["mean_delta"]) > 0
            and cap_safe(item)
        )
    ]
    positive.sort(
        key=lambda item: (
            float(item["pilot_dev"]["mean_delta"]),
            -float(item["pilot_dev"].get("output_cap_rate", 0.0) or 0.0),
        ),
        reverse=True,
    )
    selected = (
        list(positive)
        if positive_limit <= 0
        else list(positive[:positive_limit])
    )
    negative = [
        item for item in valid
        if (
            float(item["pilot_dev"]["mean_delta"]) < 0
            and cap_safe(item)
            and item not in selected
        )
    ]
    negative.sort(
        key=lambda item: (
            float(item["pilot_dev"]["mean_delta"]),
            float(item["pilot_dev"].get("output_cap_rate", 0.0) or 0.0),
        )
    )
    selected.extend(negative[:max(0, negative_limit)])
    return selected


def evaluate(*, dataset: str, original_prompts: Dict[str, str],
             target_agent: Optional[str], candidate_prompt: Optional[str],
             indices: Sequence[int], model: str, repeats: int, concurrent: int,
             dry_run: bool, collect_details: bool,
             split: str = "validate") -> dict:
    """Exact discovery evaluator, including its dataset chunking semantics."""
    prompts = dict(original_prompts)
    if target_agent and candidate_prompt is not None:
        prompts[target_agent] = candidate_prompt
    query_indices = list(indices)

    def evaluate_chunk(chunk_indices: Sequence[int]) -> dict:
        heartbeat_seconds = max(
            1.0, float(os.environ.get("IFIP_POOL_HEARTBEAT_SECONDS", "300")),
        )
        finished = threading.Event()
        started = time.monotonic()
        target_label = target_agent or "baseline"

        def heartbeat() -> None:
            while not finished.wait(heartbeat_seconds):
                elapsed = time.monotonic() - started
                print(
                    f"    [EVAL HEARTBEAT] {dataset}/{target_label} "
                    f"split={split} rows={len(chunk_indices)} repeats={repeats} "
                    f"concurrent={concurrent} elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

        reporter = threading.Thread(
            target=heartbeat,
            name=f"pool-heartbeat-{dataset}-{target_label}",
            daemon=True,
        )
        reporter.start()
        try:
            out = aflow_adapter.evaluate_with_prompts(
                dataset=dataset,
                agent_prompts=prompts,
                query_indices=list(chunk_indices),
                model=model,
                n_repeats=repeats,
                concurrent_batch=concurrent,
                dry_run=dry_run,
                split=split,
                return_details=True,
                collect_sample_details=collect_details,
                trace_agent=(target_agent if collect_details and target_agent else None),
                trace_chars=800,
            )
            if collect_details:
                # The adapter numbers result rows within this exact request,
                # including gaps left by skipped rows. A 32-row MATH Pilot is
                # unchunked but still needs this mapping to dataset indices.
                # Preserve request order for attach_agent_trace diagnostics.
                import copy

                out = dict(out)
                details = []
                for raw in out.get("sample_details") or []:
                    detail = copy.deepcopy(raw)
                    local_index = int(detail.get("row_index", -1))
                    if not 0 <= local_index < len(chunk_indices):
                        raise ValueError(f"adapter sample row outside request: {local_index}")
                    detail["row_index"] = int(chunk_indices[local_index])
                    details.append(detail)
                out["sample_details"] = details
            return out
        finally:
            finished.set()
            reporter.join(timeout=1.0)

    chunk_size = int(EVAL_CHUNK_SIZE_BY_DATASET.get(dataset, 0) or 0)
    if chunk_size <= 0 or len(query_indices) <= chunk_size:
        return evaluate_chunk(query_indices)

    chunks = [
        query_indices[start:start + chunk_size]
        for start in range(0, len(query_indices), chunk_size)
    ]
    outputs = []
    target_label = target_agent or "baseline"
    for chunk_i, chunk in enumerate(chunks, start=1):
        print(
            f"    [EVAL CHUNK] {dataset}/{target_label} "
            f"{chunk_i}/{len(chunks)} rows={len(chunk)}",
            flush=True,
        )
        outputs.append((chunk, evaluate_chunk(chunk)))

    total_weight = sum(len(chunk) for chunk, _ in outputs)
    repeat_count = max(
        (len(out.get("loss_repeats") or []) for _, out in outputs), default=0,
    )
    merged_loss_repeats = []
    for repeat_i in range(repeat_count):
        weighted_loss = 0.0
        weight = 0
        for chunk, out in outputs:
            losses = list(out.get("loss_repeats") or [])
            if repeat_i >= len(losses):
                continue
            weighted_loss += float(losses[repeat_i]) * len(chunk)
            weight += len(chunk)
        if weight:
            merged_loss_repeats.append(weighted_loss / weight)
    if not merged_loss_repeats:
        merged_loss_repeats = [
            sum(float(out["loss"]) * len(chunk) for chunk, out in outputs)
            / max(1, total_weight)
        ]

    finish_reason_counts: Counter[str] = Counter()
    max_tokens_counts: Counter[str] = Counter()
    response_count = 0
    output_cap_hits = 0
    sample_details: List[dict] = []
    for chunk, out in outputs:
        finish_reason_counts.update({
            str(key): int(value or 0)
            for key, value in (out.get("finish_reason_counts") or {}).items()
        })
        max_tokens_counts.update({
            str(key): int(value or 0)
            for key, value in (out.get("max_tokens_counts") or {}).items()
        })
        response_count += int(out.get("response_count", 0) or 0)
        output_cap_hits += int(out.get("output_cap_hits", 0) or 0)
        if collect_details:
            sample_details.extend(out.get("sample_details") or [])
    sample_details.sort(key=lambda row: int(row.get("row_index", 0)))

    result = {
        "loss": mean(merged_loss_repeats),
        "loss_repeats": merged_loss_repeats,
        "finish_reason_counts": dict(finish_reason_counts),
        "max_tokens_counts": dict(max_tokens_counts),
        "response_count": response_count,
        "output_cap_hits": output_cap_hits,
        "output_cap_rate": output_cap_hits / response_count if response_count else 0.0,
        "evaluation_chunks": len(chunks),
    }
    if collect_details:
        result["sample_details"] = sample_details
    return result


def compact_counterfactual(counterfactual: dict) -> dict:
    return {
        key: [
            {
                "row_index": int(x.get("row_index", 0)),
                "delta_score": round(float(x.get("delta_score", 0.0)), 6),
                "baseline_score": round(float(x.get("baseline_score", 0.0)), 6),
                "variant_score": round(float(x.get("variant_score", 0.0)), 6),
                "input": clip(x.get("input", ""), 500),
                "baseline_prediction": clip(x.get("baseline_prediction", ""), 450),
                "variant_prediction": clip(x.get("variant_prediction", ""), 450),
                "target_trace": [
                    {
                        "input": clip(call.get("input", ""), 300),
                        "output": clip(call.get("output", ""), 300),
                        "error": clip(call.get("error", ""), 200),
                    }
                    for call in (x.get("target_trace") or [])[:2]
                    if isinstance(call, dict)
                ],
            }
            for x in counterfactual.get(key, [])
        ]
        for key in ("fixed", "broken", "hard")
    }
