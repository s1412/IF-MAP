"""Prompt-variant helpers and legacy generators.

The primary RQ2 path is orchestrated by :mod:`src.rq2_perturbation` and uses
the same full-prompt generation and DEV-A Pilot implementation as the large
perturbation-pool discovery script, via :mod:`src.perturbation_pool_core`.
This module retains protected section-wise rephrasing plus historical
paraphrase/structural/adversarial paths for older experiments.
"""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from src.encoder import Encoder, encode_prompt
from src.llm_cache import call_llm_cached

_MIN_NORM = 1e-6
_DEFAULT_PROVIDER = "openai"

# Selected from the completed six-dataset DEV-A-selected / TEST-generalisation
# study. Every AFlow agent receives exactly these ten named slots. In the
# primary RQ2 path 36 auxiliary initial prompts are first measured by DEV-A
# Pilot to select one evidence-bearing prior. The six fixed non-adaptive
# mechanisms below use their deterministic first occurrences (not Pilot
# filtering), section-wise rephrase is added, and the three ``adaptive_*``
# prompts are generated from the measured prior. This tuple controls only the
# ten formal Stage-1 reporting/evaluation slots and their order.
RECOMMENDED_STATIC_PERTURBATION_FOCI = (
    "adaptive_counterfactual_repair",
    "adaptive_regression_repair",
    "adaptive_score_conditioned",
    "downstream_interface",
    "decomposition_control",
    "ambiguity_resolution",
    "semantic_precision",
    "verification",
    "edge_cases",
    "sectionwise_rephrase",
)

_OPERATOR_INSTRUCTIONS = {
    "adaptive_regression_repair": (
        "Infer the smallest general repair for a recurring baseline failure while explicitly "
        "guarding every useful behavior and constraint already present in the original prompt."
    ),
    "adaptive_counterfactual_repair": (
        "Use the lowest-scoring DEV-A input, baseline prediction, and final ground truth as "
        "counterfactual evidence. Add a general rule that would prevent the shared failure "
        "mechanism without copying sample-specific content."
    ),
    "adaptive_score_conditioned": (
        "Prioritize the failure mechanism supported by the lowest mean DEV-A scores and make "
        "one compact, score-conditioned repair; do not react to an isolated surface detail."
    ),
    "decomposition_control": (
        "Add a short task-specific decision sequence only when the failures indicate an "
        "omitted dependency; avoid generic chain-of-thought boilerplate."
    ),
    "semantic_precision": (
        "Clarify one vague or misleading instruction without changing the intended task."
    ),
    "verification": (
        "Add one concrete, task-relevant verification action before returning, including what "
        "must be checked and how to correct a detected failure."
    ),
    "edge_cases": (
        "Strengthen handling of plausible boundary conditions exposed by the task while "
        "remaining general and avoiding an enumerated sample-specific rule."
    ),
    "ambiguity_resolution": (
        "Resolve one evidenced ambiguity or precedence conflict with a deterministic, "
        "general decision rule."
    ),
    "downstream_interface": (
        "Clarify the information or artifact this agent must hand to downstream agents while "
        "preserving the existing output type."
    ),
    "constraint_recovery": (
        "Add a general procedure for extracting and enforcing explicit constraints from each "
        "new input; do not copy a constraint from an example."
    ),
    "adaptive_residual_crossover": (
        "Keep the complete original prompt as the residual path and add one compact patch that "
        "combines only complementary, evidence-supported safeguards shared by the worst DEV-A "
        "failures; discard unsupported edits."
    ),
    "sectionwise_rephrase": (
        "Rewrite each logical prompt section independently with equivalent English wording, "
        "without adding a new task rule or changing placeholders, protected tokens, structure, "
        "constraints, or output requirements."
    ),
}

_CHINESE_RE = re.compile(r"[\u3400-\u9fff]")

_PLACEHOLDER_RE = re.compile(r"\{\w+\}")                       # {input} {problem} ...
_LATEX_RE       = re.compile(r"\\[a-zA-Z]+(?:\{[^{}]*\})?")    # \boxed \frac ...
_CODEFENCE_RE   = re.compile(r"```.*?```", re.DOTALL)          # ```python ... ```
_INLINECODE_RE  = re.compile(r"`[^`]+`")                       # `def solve():`
_DEFSIG_RE      = re.compile(r"\bdef\s+\w+\s*\([^)]*\)")       # def solve(): / def f(x)
_EMPTYPAREN_RE  = re.compile(r"\(\s*\)")
_LLM_MARKER_RE  = re.compile(r"\[\[K\d+\]\]")

_MASK_ORDER = [_CODEFENCE_RE, _INLINECODE_RE, _DEFSIG_RE, _LATEX_RE, _PLACEHOLDER_RE, _EMPTYPAREN_RE]


def _mask_critical(text: str, fmt: str = "\x00{}\x00"):
    """Mask protected fragments before paraphrasing."""
    spans: List[str] = []

    def repl(m):
        spans.append(m.group(0))
        return fmt.format(len(spans) - 1)

    for rx in _MASK_ORDER:
        text = rx.sub(repl, text)
    return text, spans


def _unmask(text: str, spans: List[str], fmt: str = "\x00{}\x00") -> str:
    for i, s in enumerate(spans):
        text = text.replace(fmt.format(i), s)
    return text


def _critical_tokens(text: str):
    """Extract protected markers and signatures."""
    return (set(_PLACEHOLDER_RE.findall(text)),
            set(_LATEX_RE.findall(text)),
            set(_DEFSIG_RE.findall(text)))


def _placeholder_sequence(text: str) -> tuple:
    """Return placeholders with multiplicity and order preserved."""
    return tuple(_PLACEHOLDER_RE.findall(text or ""))


def _preserves_critical(original: str, variant: str) -> bool:
    """Check protected fragments and multiplicity."""
    # Membership alone is insufficient: a rewrite that duplicates, removes, or
    # reorders a placeholder changes the workflow interface.  Require the exact
    # sequence before applying the legacy LaTeX/signature guards below.
    if _placeholder_sequence(original) != _placeholder_sequence(variant):
        return False
    ph, latex, defs = _critical_tokens(original)
    for t in ph | latex:
        if t not in variant:
            return False
    var_ns = re.sub(r"\s+", "", variant)
    for t in defs:
        if re.sub(r"\s+", "", t) not in var_ns:
            return False
    return True


def set_default_provider(provider: str) -> None:
    global _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = provider


@dataclass
class AgentVariant:
    variant_text:   str
    embedding_diff: np.ndarray     # E(variant) - E(original)
    norm:           float
    unit_direction: np.ndarray     # embedding_diff / norm
    source:         str            # operator_pool / operator_pool_fallback / legacy sources
    edit_type:      str = ""       # operator name (or legacy structural edit type)


def generate_variants(original_prompts: Dict[str, str], num_per_agent: int,
                      mix: dict, generator_llm: str, encoder: Encoder,
                      dry_run: bool = False,
                      dataset: str = "",
                      baseline_sample_details: Sequence[dict] = (),
                      ) -> Dict[str, List[AgentVariant]]:
    """Generate structurally valid prompt variants."""
    result: Dict[str, List[AgentVariant]] = {}

    for agent_name, prompt in original_prompts.items():
        configured_pool = tuple(mix.get("operator_pool") or ())
        if configured_pool:
            unknown = [x for x in configured_pool if x not in _OPERATOR_INSTRUCTIONS]
            if unknown:
                raise ValueError(f"unknown perturbation operators: {unknown}")
            selected_pool = configured_pool[:max(0, int(num_per_agent))]
            raw_variants = _generate_operator_pool_variants(
                prompt=prompt,
                agent_name=agent_name,
                dataset=dataset,
                sample_details=baseline_sample_details,
                foci=selected_pool,
                generator_llm=generator_llm,
                dry_run=dry_run,
                max_attempts=int(mix.get("generation_attempts", 3) or 3),
            )
        else:
            base_n_para = int(mix.get("paraphrase", 0) or 0)
            requested_n_struct = int(mix.get("structural_edit", 0) or 0)
            n_adv = int(mix.get("adversarial", 0) or 0)

            structural_edits = _generate_structural_edits(prompt, requested_n_struct)
            struct_shortfall = max(0, requested_n_struct - len(structural_edits))
            n_para = base_n_para + struct_shortfall
            if struct_shortfall:
                print(
                    f"[VariantGen] {agent_name}: structural requested {requested_n_struct}, "
                    f"available {len(structural_edits)}; move {struct_shortfall} to paraphrase",
                    flush=True,
                )

            if dry_run:
                paraphrases = [f"{prompt} (paraphrase stub {i})" for i in range(n_para)]
            else:
                paraphrases = _generate_paraphrases(prompt, n_para, generator_llm,
                                                    agent_name=agent_name)

            raw_variants = []   # (text, source, edit_type)
            raw_variants += [(t, "paraphrase", "") for t in paraphrases]
            raw_variants += [(t, "structural", et) for t, et in structural_edits]
            raw_variants += [(t, "adversarial", et) for t, et in
                             _generate_adversarial(prompt, n_adv)]
        raw_variants = _dedupe_raw_variants(raw_variants)
        raw_variants = raw_variants[:num_per_agent]


        variants: List[AgentVariant] = []
        if raw_variants:
            texts = [prompt] + [text for text, _, _ in raw_variants]
            embs = encoder.encode_batch(texts)
            e_orig = embs[0]
        else:
            embs = np.empty((1, encoder.dim), dtype=np.float32)
            e_orig = embs[0]
        for (text, source, edit_type), e_var in zip(raw_variants, embs[1:]):
            diff = e_var - e_orig
            norm = float(np.linalg.norm(diff))
            unit = diff / norm if norm > _MIN_NORM else np.zeros_like(diff)
            if norm < _MIN_NORM:
                continue
            variants.append(AgentVariant(
                variant_text=text, embedding_diff=diff, norm=norm,
                unit_direction=unit, source=source, edit_type=edit_type,
            ))

        result[agent_name] = variants
        print(f"[VariantGen] {agent_name}: {len(variants)} usable/evaluable variants "
              f"(requested {num_per_agent}; fallback may be included after retries)")

    return result


def _generate_paraphrases(prompt: str, n: int, generator_llm: str,
                          agent_name: str = "") -> List[str]:
    if n <= 0:
        return []


    LLM_FMT = "[[K{}]]"
    masked_prompt, spans = _mask_critical(prompt, fmt=LLM_FMT)


    n_request = n + 2
    marker_note = (f" The text contains protected markers like [[K0]], [[K1]]. "
                   f"Copy every marker EXACTLY and keep it in its place; never alter, "
                   f"translate, reorder, or drop a marker." if spans else "")
    max_attempts = 3
    log_subject = f"{agent_name} paraphrase" if agent_name else "paraphrase"

    if spans:
        return _generate_structure_preserving_paraphrases(
            prompt=prompt,
            masked_prompt=masked_prompt,
            spans=spans,
            marker_fmt=LLM_FMT,
            n=n,
            n_request=n_request,
            generator_llm=generator_llm,
            max_attempts=max_attempts,
            log_subject=log_subject,
        )

    out: List[str] = []
    last_usable: List[str] = []
    last_invalid: List[str] = []

    for attempt in range(1, max_attempts + 1):

        instruction = (
            f"Paraphrase the following prompt while keeping its meaning and intent "
            f"fully intact. Generate {n_request} distinct versions with varied wording.{marker_note} "
            f"This is generation attempt {attempt}/{max_attempts}; be especially careful "
            f"to preserve protected markers and code signatures. "
            f"Output ONLY a JSON list of strings, no extra text.\n\n"
            f"Prompt to paraphrase:\n{masked_prompt}"
        )
        resp = call_llm_cached(
            provider=_DEFAULT_PROVIDER,
            model=generator_llm,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.7,
        )
        candidates = _parse_json_list(resp["content"])
        valid: List[str] = []
        invalid: List[str] = []
        usable: List[str] = []
        for c in candidates:
            c = _unmask(str(c).strip(), spans, fmt=LLM_FMT).strip()
            if not (c and c != prompt.strip() and len(c) > 10):
                continue
            usable.append(c)
            if _preserves_critical(prompt, c):
                valid.append(c)
            else:
                invalid.append(c)

        out = _dedupe_texts(out + valid)
        last_usable = usable
        last_invalid = invalid
        if len(out) >= n:
            break
        if attempt < max_attempts:
            print(
                f"[VariantGen] {log_subject}: strict-valid {len(out)}/{n}; "
                f"this round failed critical-token check {len(invalid)} candidates, "
                f"retry {attempt + 1}/{max_attempts}",
                flush=True,
            )

    if len(out) < n:
        before = len(out)

        out = _dedupe_texts(out + last_invalid + last_usable)
        used_last = max(0, len(out) - before)
        if used_last:
            print(
                f"[VariantGen] {log_subject}: strict-valid still insufficient after retries; "
                f"forced-use last-attempt candidates {used_last} "
                f"(may not pass critical-token check)",
                flush=True,
            )

    if len(out) < n:


        fallback = _fallback_paraphrases(prompt, n - len(out))
        out = _dedupe_texts(out + fallback)
        print(
            f"[VariantGen] {log_subject}: last-attempt candidates still insufficient; "
            f"use local safe fallback to fill {min(len(out), n)}/{n}",
            flush=True,
        )

    return _stable_sample_texts(out, n, salt=f"paraphrase:{prompt}")


def _generate_structure_preserving_paraphrases(
    prompt: str,
    masked_prompt: str,
    spans: List[str],
    marker_fmt: str,
    n: int,
    n_request: int,
    generator_llm: str,
    max_attempts: int,
    log_subject: str,
) -> List[str]:
    """Paraphrase editable sections; preserve protected boundaries."""
    parts = re.split(f"({_LLM_MARKER_RE.pattern})", masked_prompt)
    text_indices = [
        i for i, part in enumerate(parts)
        if part and not _LLM_MARKER_RE.fullmatch(part) and part.strip()
    ]
    if not text_indices:
        return _fallback_paraphrases(prompt, n)[:n]

    text_segments = [parts[i] for i in text_indices]
    segment_payload = json.dumps(text_segments, ensure_ascii=False, indent=2)
    out: List[str] = []

    for attempt in range(1, max_attempts + 1):
        instruction = (
            "Rewrite prompt text fragments without changing the prompt structure.\n"
            "Protected markers were removed from TEXT_SEGMENTS and will be reinserted "
            "by code in their original positions. You are NOT writing the full prompt.\n\n"
            f"Generate {n_request} distinct variants. Return ONLY a JSON list of arrays.\n"
            f"Each array MUST contain exactly {len(text_segments)} strings.\n"
            "array[j] must be a paraphrase of TEXT_SEGMENTS[j].\n"
            "Do not summarize. Do not omit constraints, section labels, markdown headings, "
            "or required output-format instructions.\n"
            "Do not include protected markers such as [[K0]] in any returned string; "
            "the program will insert them back automatically.\n"
            "If a segment is mostly formatting or a heading, keep it nearly unchanged.\n"
            f"This is generation attempt {attempt}/{max_attempts}.\n\n"
            f"TEXT_SEGMENTS = {segment_payload}"
        )
        resp = call_llm_cached(
            provider=_DEFAULT_PROVIDER,
            model=generator_llm,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.7,
        )

        parsed = _parse_json_value(resp["content"])
        segment_variants = _coerce_segment_variants(parsed, len(text_segments))
        valid: List[str] = []
        rejected = 0
        for segs in segment_variants:
            candidate_masked = _stitch_masked_prompt(parts, text_indices, segs)
            candidate = _unmask(candidate_masked, spans, fmt=marker_fmt).strip()
            if not (candidate and candidate != prompt.strip() and len(candidate) > 10):
                rejected += 1
                continue
            if _preserves_critical(prompt, candidate):
                valid.append(candidate)
            else:
                rejected += 1

        out = _dedupe_texts(out + valid)
        if len(out) >= n:
            break
        if attempt < max_attempts:
            print(
                f"[VariantGen] {log_subject}: structure-preserving strict-valid {len(out)}/{n}; "
                f"parsed segment variants {len(segment_variants)}, rejected {rejected}; "
                f"retry {attempt + 1}/{max_attempts}",
                flush=True,
            )

    if len(out) < n:
        before = len(out)
        out = _dedupe_texts(out + _fallback_paraphrases(prompt, n - len(out)))
        added = max(0, len(out) - before)
        print(
            f"[VariantGen] {log_subject}: structure-preserving strict-valid still insufficient; "
            f"use local safe fallback {added} to fill {min(len(out), n)}/{n}",
            flush=True,
        )

    return _stable_sample_texts(out, n, salt=f"structure-preserving:{prompt}")


def _coerce_segment_variants(obj, n_segments: int) -> List[List[str]]:
    """Normalize generated segment variants."""
    if not isinstance(obj, list):
        return []
    out: List[List[str]] = []
    for item in obj:
        segs = None
        if isinstance(item, list) and len(item) == n_segments:
            segs = [str(x) for x in item]
        elif isinstance(item, dict):
            values = []
            ok = True
            for i in range(n_segments):
                candidates = (
                    f"SEGMENT_{i}", f"segment_{i}", f"Segment_{i}",
                    f"S{i}", f"s{i}", str(i), i,
                )
                found = None
                for key in candidates:
                    if key in item:
                        found = item[key]
                        break
                if found is None:
                    ok = False
                    break
                values.append(str(found))
            if ok:
                segs = values
        if segs is not None:
            out.append(segs)
    return out


def _stitch_masked_prompt(parts: List[str], text_indices: List[int],
                          segment_rewrites: List[str]) -> str:
    new_parts = list(parts)
    for idx, rewrite in zip(text_indices, segment_rewrites):
        cleaned = _LLM_MARKER_RE.sub("", str(rewrite or ""))
        original = parts[idx]
        core = cleaned.strip()
        if not core:
            new_parts[idx] = original
            continue
        # Protected markers split the prompt into editable text segments.  LLMs
        # commonly strip the segment's boundary newlines, which previously
        # glued a marker directly to the next sentence (for example
        # ``{problem}Provide the function signature``).  Preserve the exact
        # original whitespace on both sides and replace only the textual core.
        leading = re.match(r"^\s*", original).group(0)
        trailing = re.search(r"\s*$", original).group(0)
        new_parts[idx] = f"{leading}{core}{trailing}"
    return "".join(new_parts)


def _dedupe_texts(texts: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for text in texts:
        key = re.sub(r"\s+", " ", (text or "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _stable_sample_texts(texts: List[str], n: int, salt: str = "") -> List[str]:
    """Select text samples reproducibly."""
    texts = list(texts or [])
    if n <= 0:
        return []
    if len(texts) <= n:
        return texts[:n]
    payload = json.dumps({"salt": salt, "texts": texts}, ensure_ascii=False, sort_keys=True)
    seed = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(texts)), n))
    return [texts[i] for i in indices]


def _fallback_paraphrases(prompt: str, n: int) -> List[str]:
    templates = [
        "{prompt}\n\nPlease preserve the exact task requirements and answer in the required format.",
        "Carefully follow this task while preserving every required token and output convention:\n\n{prompt}",
        "{prompt}\n\nRestate the requirements internally, then provide only the requested answer.",
        "Solve the following instruction exactly as written, keeping all signatures and placeholders unchanged:\n\n{prompt}",
        "{prompt}\n\nBe precise, preserve all constraints, and avoid unnecessary commentary.",
        "Using the same requirements, solve this task carefully:\n\n{prompt}",
    ]
    out = []
    i = 0
    while len(out) < n:
        text = templates[i % len(templates)].format(prompt=prompt)
        if i >= len(templates):
            text += f"\n\nVariant note {i // len(templates) + 1}: maintain correctness and format."
        out.append(text)
        i += 1
    return out


# ============================================================
# Recommended static operator pool
# ============================================================
def _clip_operator_field(value: Any, limit: int = 700) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.68))
    tail = max(1, limit - head - 24)
    return text[:head] + "\n...[omitted]...\n" + text[-tail:]


def _compact_failure_examples(sample_details: Sequence[dict], limit: int = 3) -> List[dict]:
    """Return stable worst DEV-A rows, averaging duplicate repeat scores."""
    grouped: Dict[int, List[dict]] = {}
    for pos, row in enumerate(sample_details or ()):
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("row_index", pos))
        except (TypeError, ValueError):
            idx = pos
        grouped.setdefault(idx, []).append(row)
    ranked = []
    for idx, rows in grouped.items():
        scores = []
        for row in rows:
            try:
                scores.append(float(row.get("score", 0.0)))
            except (TypeError, ValueError):
                scores.append(0.0)
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0
        if mean_score >= 0.999999:
            continue
        representative = rows[-1]
        ranked.append((mean_score, idx, {
            "row_index": idx,
            "mean_score": round(mean_score, 6),
            "input": _clip_operator_field(representative.get("input")),
            "prediction": _clip_operator_field(representative.get("prediction")),
            "expected": _clip_operator_field(representative.get("expected")),
        }))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:max(0, int(limit))]]


def _operator_fallback_instruction(focus: str) -> str:
    fallbacks = {
        "adaptive_regression_repair": (
            "Preserve every existing requirement; before returning, identify and correct the "
            "most likely recurring failure without weakening already-correct behavior."
        ),
        "adaptive_counterfactual_repair": (
            "Compare the proposed result with every explicit input constraint and revise any "
            "step that would cause a mismatch, without importing sample-specific rules."
        ),
        "adaptive_score_conditioned": (
            "Prioritize the most consequential unmet requirement, repair it first, and then "
            "verify that the remaining requirements still hold."
        ),
        "decomposition_control": (
            "Identify the required intermediate decisions, resolve them in dependency order, and "
            "then construct the requested output."
        ),
        "semantic_precision": (
            "Interpret every requirement literally and consistently; do not replace a precise "
            "condition with a convenient approximation."
        ),
        "verification": (
            "Before returning, perform one concrete task-relevant check and correct the result "
            "if that check reveals a violated requirement."
        ),
        "edge_cases": (
            "Identify and handle the relevant boundary conditions implied by the current input "
            "without adding sample-specific exceptions."
        ),
        "ambiguity_resolution": (
            "When requirements appear ambiguous, prefer the interpretation that satisfies all "
            "explicit constraints and the stated output contract."
        ),
        "downstream_interface": (
            "Return a complete artifact that preserves every field and fact needed by downstream "
            "workflow components, using the existing output type."
        ),
        "constraint_recovery": (
            "Extract all explicit constraints from each supplied input and confirm that the final "
            "result enforces every one of them."
        ),
        "adaptive_residual_crossover": (
            "Keep every existing behavior and add only the smallest evidence-supported safeguard "
            "needed to prevent the recurring failure."
        ),
    }
    return fallbacks[focus]


def _operator_generation_request(*, prompt: str, agent_name: str, dataset: str,
                                 focus: str, failures: Sequence[dict],
                                 attempt: int) -> str:
    return (
        "You are producing one measured prompt perturbation for a multi-agent workflow.\n"
        "Return exactly one JSON object with string keys instruction, placement, and rationale.\n"
        "placement must be either prefix or suffix. Do not return markdown.\n\n"
        "Write a compact English instruction patch, not a replacement prompt. The program will "
        "insert it before or after the original prompt. The patch must be reusable for every "
        "future input, preserve the task and output type, and implement exactly one operator. "
        "Do not copy any example answer, literal, identifier, function body, or dataset-specific "
        "rule. Do not add generic role-play or generic chain-of-thought text. Do not introduce "
        "a new placeholder or repeat an existing placeholder in the patch.\n\n"
        f"DATASET: {dataset or 'unspecified'}\n"
        f"TARGET AGENT: {agent_name}\n"
        f"OPERATOR: {focus}\n"
        f"OPERATOR GOAL: {_OPERATOR_INSTRUCTIONS[focus]}\n"
        f"GENERATION ATTEMPT: {attempt}\n\n"
        "WORST NON-PERFECT DEV-A EXAMPLES (evidence only; never copy their content):\n"
        f"{json.dumps(list(failures), ensure_ascii=False, indent=2)}\n\n"
        f"ORIGINAL PROMPT:\n{prompt}"
    )


def _contains_verbatim_example(instruction: str, failures: Sequence[dict]) -> bool:
    normalized_instruction = re.sub(r"\s+", " ", instruction).lower()
    for row in failures:
        for field in ("input", "prediction", "expected"):
            value = re.sub(r"\s+", " ", str(row.get(field, ""))).strip().lower()
            if len(value) >= 48 and value[:80] in normalized_instruction:
                return True
    return False


def _compose_operator_candidate(prompt: str, instruction: str, placement: str) -> str:
    instruction = instruction.strip()
    if placement == "prefix":
        return f"{instruction}\n\n{prompt}"
    return f"{prompt}\n\n{instruction}"


def _logical_prompt_sections(masked_prompt: str):
    """Split a masked prompt into non-empty logical paragraphs and separators.

    Blank-line separators remain byte-for-byte unchanged.  This makes the
    section-wise rephrase a single measured variant while preventing the model
    from merging headings, examples, or output-contract paragraphs.
    """
    parts = re.split(r"(\n[ \t]*\n+)", masked_prompt)
    section_indices = [
        i for i, part in enumerate(parts)
        if part and not re.fullmatch(r"\n[ \t]*\n+", part) and part.strip()
    ]
    return parts, section_indices


def _sectionwise_rephrase_request(sections: Sequence[str], *, agent_name: str,
                                  dataset: str, attempt: int) -> str:
    return (
        "Rephrase the logical sections of one system prompt independently.\n"
        "Return ONLY a JSON object with one key, sections, whose value is a list of exactly "
        f"{len(sections)} strings in the original order. Do not return markdown.\n\n"
        "Each output string must be an English, meaning-equivalent rewrite of the matching "
        "input section. Preserve every instruction, constraint, heading, example role, and "
        "output requirement. Do not add a new task rule, merge sections, split sections, "
        "summarize the prompt, or copy text between sections.\n"
        "Tokens such as [[K0]] are protected placeholders, code, signatures, or formatting. "
        "Within each section, copy every protected token exactly once and in the same order. "
        "Never translate, alter, move, duplicate, or delete a protected token.\n\n"
        f"DATASET: {dataset or 'unspecified'}\n"
        f"TARGET AGENT: {agent_name}\n"
        f"GENERATION ATTEMPT: {attempt}\n\n"
        "LOGICAL_SECTIONS:\n"
        f"{json.dumps(list(sections), ensure_ascii=False, indent=2)}"
    )


def _coerce_rephrased_sections(obj: Any, n_sections: int) -> List[str]:
    if isinstance(obj, dict):
        obj = obj.get("sections")
    if not isinstance(obj, list) or len(obj) != n_sections:
        return []
    return [str(item) for item in obj]


def _local_sectionwise_rephrase(parts: Sequence[str], section_indices: Sequence[int]) -> str:
    """Deterministic provider-free fallback used by dry runs and outages."""
    replacements = (
        (r"\bPlease\b", "Kindly"),
        (r"\bReturn\b", "Provide"),
        (r"\bGenerate\b", "Produce"),
        (r"\bSolve\b", "Work out"),
        (r"\bUse\b", "Employ"),
        (r"\bYou are\b", "Act as"),
        (r"\bmust\b", "is required to"),
    )
    rebuilt = list(parts)
    changed = False
    for idx in section_indices:
        original = parts[idx]
        rewritten = original
        for pattern, replacement in replacements:
            rewritten, count = re.subn(pattern, replacement, rewritten, count=1)
            if count:
                changed = True
                break
        rebuilt[idx] = rewritten
    if not changed and section_indices:
        idx = section_indices[0]
        original = parts[idx]
        leading = re.match(r"^\s*", original).group(0)
        trailing = re.search(r"\s*$", original).group(0)
        core = original[len(leading):len(original) - len(trailing) if trailing else None]
        rebuilt[idx] = f"{leading}In equivalent terms: {core}{trailing}"
    return "".join(rebuilt)


def _generate_sectionwise_rephrase(*, prompt: str, agent_name: str, dataset: str,
                                    generator_llm: str, dry_run: bool,
                                    max_attempts: int,
                                    allow_fallback: bool = True):
    """Build one variant by independently rephrasing every logical section."""
    marker_fmt = "[[K{}]]"
    masked_prompt, spans = _mask_critical(prompt, fmt=marker_fmt)
    parts, section_indices = _logical_prompt_sections(masked_prompt)
    if not section_indices:
        return "", "sectionwise_rephrase_invalid"
    sections = [parts[idx].strip() for idx in section_indices]

    if dry_run:
        candidate_masked = _local_sectionwise_rephrase(parts, section_indices)
        candidate = _unmask(candidate_masked, spans, fmt=marker_fmt).strip()
        return candidate, "sectionwise_rephrase_dry"

    rejection_counts: Dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for attempt in range(1, max(1, int(max_attempts or 1)) + 1):
        try:
            response = call_llm_cached(
                provider=_DEFAULT_PROVIDER,
                model=generator_llm,
                messages=[{"role": "user", "content": _sectionwise_rephrase_request(
                    sections, agent_name=agent_name, dataset=dataset, attempt=attempt,
                )}],
                temperature=0.7,
                max_tokens=2048,
            )
        except Exception as exc:
            reject(f"provider_{type(exc).__name__}")
            continue
        rewrites = _coerce_rephrased_sections(
            _parse_json_value(response.get("content", "")), len(sections),
        )
        if not rewrites:
            reject("invalid_section_list")
            continue
        if any(_CHINESE_RE.search(item) for item in rewrites):
            reject("non_english_section")
            continue
        marker_mismatch = any(
            tuple(_LLM_MARKER_RE.findall(original))
            != tuple(_LLM_MARKER_RE.findall(rewritten))
            for original, rewritten in zip(sections, rewrites)
        )
        if marker_mismatch:
            reject("protected_marker_sequence")
            continue

        rebuilt = list(parts)
        for idx, rewritten in zip(section_indices, rewrites):
            original = parts[idx]
            leading = re.match(r"^\s*", original).group(0)
            trailing = re.search(r"\s*$", original).group(0)
            rebuilt[idx] = f"{leading}{rewritten.strip()}{trailing}"
        candidate = _unmask("".join(rebuilt), spans, fmt=marker_fmt).strip()
        if candidate == prompt.strip():
            reject("unchanged")
            continue
        if not _preserves_critical(prompt, candidate):
            reject("critical_guard")
            continue
        return candidate, "sectionwise_rephrase"

    if not allow_fallback:
        print(
            f"[VariantGen] {agent_name}/sectionwise_rephrase: provider candidates rejected; "
            f"reasons={rejection_counts}; strict perturbation-pool mode keeps no fallback",
            flush=True,
        )
        return "", "sectionwise_rephrase_invalid"

    candidate_masked = _local_sectionwise_rephrase(parts, section_indices)
    candidate = _unmask(candidate_masked, spans, fmt=marker_fmt).strip()
    print(
        f"[VariantGen] {agent_name}/sectionwise_rephrase: provider candidates rejected; "
        f"reasons={rejection_counts}; use protected local fallback",
        flush=True,
    )
    return candidate, "sectionwise_rephrase_fallback"


def _generate_operator_pool_variants(*, prompt: str, agent_name: str, dataset: str,
                                     sample_details: Sequence[dict],
                                     foci: Sequence[str], generator_llm: str,
                                     dry_run: bool, max_attempts: int) -> List[tuple]:
    """Generate exactly one local, evidence-aware patch per configured operator."""
    failures = _compact_failure_examples(sample_details, limit=3)
    variants: List[tuple] = []
    max_attempts = max(1, int(max_attempts or 1))
    for focus in foci:
        if focus == "sectionwise_rephrase":
            candidate, source = _generate_sectionwise_rephrase(
                prompt=prompt,
                agent_name=agent_name,
                dataset=dataset,
                generator_llm=generator_llm,
                dry_run=dry_run,
                max_attempts=max_attempts,
            )
            if candidate and _preserves_critical(prompt, candidate):
                variants.append((candidate, source, focus))
            continue

        candidate = ""
        source = "operator_pool"
        rejection_counts: Dict[str, int] = {}

        def reject(reason: str) -> None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        if not dry_run:
            for attempt in range(1, max_attempts + 1):
                try:
                    response = call_llm_cached(
                        provider=_DEFAULT_PROVIDER,
                        model=generator_llm,
                        messages=[{"role": "user", "content": _operator_generation_request(
                            prompt=prompt, agent_name=agent_name, dataset=dataset,
                            focus=focus, failures=failures, attempt=attempt,
                        )}],
                        temperature=0.7,
                        max_tokens=800,
                    )
                except Exception as exc:  # provider/network failures use a visible safe fallback
                    reject(f"provider_{type(exc).__name__}")
                    continue
                parsed = _parse_json_value(response.get("content", ""))
                if not isinstance(parsed, dict):
                    reject("invalid_json_object")
                    continue
                instruction = str(parsed.get("instruction", "")).strip()
                placement = str(parsed.get("placement", "suffix")).strip().lower()
                if placement not in ("prefix", "suffix"):
                    placement = "suffix"
                if not (20 <= len(instruction) <= 1800):
                    reject("instruction_length")
                    continue
                if _CHINESE_RE.search(instruction):
                    reject("non_english_instruction")
                    continue
                if _contains_verbatim_example(instruction, failures):
                    reject("verbatim_example_copy")
                    continue
                proposed = _compose_operator_candidate(prompt, instruction, placement)
                ratio = len(proposed) / max(1, len(prompt))
                if ratio > 3.0 or not _preserves_critical(prompt, proposed):
                    reject("length_or_critical_guard")
                    continue
                candidate = proposed
                break
        if not candidate:
            # Offline tests and exhausted provider attempts use a safe local patch.  The
            # original prompt remains byte-for-byte intact in either placement.
            instruction = _operator_fallback_instruction(focus)
            candidate = _compose_operator_candidate(prompt, instruction, "suffix")
            if len(candidate) / max(1, len(prompt)) > 3.0:
                instruction = f"Apply this check before returning: {focus.replace('_', ' ')}."
                candidate = _compose_operator_candidate(prompt, instruction, "suffix")
            source = "operator_pool_dry" if dry_run else "operator_pool_fallback"
            if not dry_run:
                print(
                    f"[VariantGen] {agent_name}/{focus}: provider candidates rejected; "
                    f"reasons={rejection_counts}; use protected local fallback",
                    flush=True,
                )
        variants.append((candidate, source, focus))
    return variants


_EDIT_OPS = [


    "add_cot", "add_role_definition", "add_constraint", "add_few_shot",
]


def _generate_structural_edits(prompt: str, n: int) -> List[tuple]:
    """Generate task-general structural edits."""
    if n <= 0:
        return []
    out: List[tuple] = []
    seen = set()
    original_key = re.sub(r"\s+", " ", prompt).strip()
    for op in _EDIT_OPS:
        text = _apply_edit_op(prompt, op)
        key = re.sub(r"\s+", " ", (text or "")).strip()
        if not key or key == original_key or key in seen:
            continue
        if not _preserves_critical(prompt, text):
            continue
        seen.add(key)
        out.append((text, op))
        if len(out) >= n:
            break
    return out[:n]


def _apply_edit_op(prompt: str, op: str) -> str:
    if op == "add_cot":
        return prompt + "\n\nLet's think step by step before answering."
    if op == "add_role_definition":
        return ("You are a senior domain expert with extensive experience. "
                "Apply rigorous reasoning.\n\n" + prompt)
    if op == "add_constraint":
        return prompt + "\n\nConstraints: Be precise and concise. " \
                        "Output only the final answer in the required format."
    if op == "add_few_shot":
        return (
            prompt
            + "\n\nExample:\n"
            "Input: <example input>\n"
            "Output: <example output following the required format>"
        )
    if op == "remove_redundancy":

        masked, spans = _mask_critical(prompt)
        masked = re.sub(r"\s*\([^)]*\)", "", masked)
        masked = re.sub(r"\s+", " ", masked).strip()
        return _unmask(masked, spans)
    if op == "change_tone":

        if re.search(r"\bplease\b", prompt, re.IGNORECASE):
            return re.sub(r"\bplease\b\s*", "", prompt, flags=re.IGNORECASE).strip()
        return "Please " + prompt[0].lower() + prompt[1:] if prompt else prompt
    return prompt


def _generate_adversarial(prompt: str, n: int) -> List[tuple]:
    """Generate controlled adverse edits."""
    if n <= 0:
        return []
    ops = [
        ("misleading_instr",  prompt + "\n\nBe brief and don't bother verifying your answer."),
        ("drop_constraint",   _adv_drop_constraint(prompt)),
        ("bad_role",          "You are a careless and unreliable assistant.\n\n" + prompt),
    ]
    out = [(t, name) for name, t in ops if t and t.strip() != prompt.strip()]
    return out[:n]


def _adv_drop_constraint(prompt: str) -> str:
    """Remove one constraint."""
    weakened = re.sub(r"(?i)(constraint|must|require)[^.]*\.", "", prompt).strip()
    if weakened and weakened != prompt.strip():
        return weakened
    return prompt + "\n\nIgnore optional constraints and prioritize a short answer over strict formatting."


def _adv_add_typo(prompt: str) -> str:
    """Insert a controlled typo."""
    masked, spans = _mask_critical(prompt)
    words = masked.split()
    for idx in range(0, len(words), max(1, len(words) // 5 or 1)):
        w = words[idx]

        if len(w) > 3 and "\x00" not in w and w.isalpha():
            j = len(w) // 2
            words[idx] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return _unmask(" ".join(words), spans)


def _compute_embedding_diff(original_text: str, variant_text: str, encoder: Encoder):
    """Compute embedding displacement and unit direction."""
    e_orig = encode_prompt(encoder, original_text)
    e_var = encode_prompt(encoder, variant_text)
    diff = e_var - e_orig
    norm = float(np.linalg.norm(diff))
    unit = diff / norm if norm > _MIN_NORM else np.zeros_like(diff)
    return diff, norm, unit


def _dedupe_raw_variants(raw_variants: List[tuple]) -> List[tuple]:
    """Deduplicate normalized variant text."""
    out = []
    seen = set()
    for text, source, edit_type in raw_variants:
        key = re.sub(r"\s+", " ", (text or "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((text, source, edit_type))
    return out


def _parse_json_value(text: str):
    """Parse a generated JSON value."""
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    block = re.search(r"\[.*\]", text, re.DOTALL)
    if block:
        candidates.append(block.group(0))
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _parse_json_list(text: str) -> List[str]:
    """Parse a generated JSON list."""

    obj = _parse_json_value(text)
    if isinstance(obj, list):
        return [str(x) for x in obj]

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except json.JSONDecodeError:
            pass

    lines = [l.strip(" -*\"'\t") for l in text.splitlines() if l.strip()]
    return [l for l in lines if len(l) > 10]


def _filter_degenerate_variants(variants: List[AgentVariant],
                                min_norm_threshold: float = _MIN_NORM) -> List[AgentVariant]:
    """Remove zero-displacement variants."""
    return [v for v in variants if v.norm >= min_norm_threshold]
