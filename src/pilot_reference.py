"""Read-only, identity-checked v35 Pilot references; never reuse candidate scores.

``load_reference`` is deliberately independent of the model/runtime modules.
Only a missing dataset directory returns None (allowing a caller's fixed-32
fallback). A configured root or existing dataset with missing/corrupt/mismatched
artifacts fails closed. Historical TEST fields are never returned.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import pickle
import pickletools
import re
from pathlib import Path
from src.paths import project_path


class PilotReferenceError(ValueError):
    """A configured historical reference cannot safely identify this DEV split."""


_MAX_BYTES = 64 * 1024 * 1024
_FORBIDDEN_OPCODES = {
    "GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "INST", "OBJ", "NEWOBJ",
    "NEWOBJ_EX", "EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID",
}
_GENERATION_FIELDS = {
    "id", "agent", "prompt", "source", "focus", "edit_type", "changed_span",
    "evidence_link", "rationale", "expected_mechanism", "regression_risk",
    "generation_attempt", "prompt_sha256", "length_ratio", "diff",
    "generation_rejections", "slot_refill_round", "semantic_audit",
}
_DIAGNOSTIC_FIELDS = {
    "mean_score", "score_repeats", "prediction", "expected_or_error",
    "evidence_source", "trace_source", "diagnostic_repeat_score",
    "diagnostic_prediction",
}
_MEASUREMENT_KEY = re.compile(r"(?:^|_)(?:pilot|full|test|joint|score|scores|loss|delta|lcb|ucb|prior|selected|selection|measurement)(?:_|$)")


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise PilotReferenceError("pickle globals are forbidden")

    def persistent_load(self, pid):
        raise PilotReferenceError("pickle persistent IDs are forbidden")


def _plain(value, depth=0, active=None):
    """Reject object graphs, cycles, and non-plain types after opcode screening."""
    if depth > 100:
        raise PilotReferenceError("reference nesting is too deep")
    if type(value) in (str, bytes, int, float, bool, type(None)):
        return
    if type(value) not in (dict, list, tuple, set, frozenset):
        raise PilotReferenceError("reference contains a non-plain type")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise PilotReferenceError("reference contains a cycle")
    active.add(identity)
    try:
        items = (item for pair in value.items() for item in pair) if isinstance(value, dict) else value
        for item in items:
            _plain(item, depth + 1, active)
    finally:
        active.remove(identity)


def _read_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_BYTES + 1)
    except OSError as exc:
        raise PilotReferenceError(f"cannot read reference artifact: {path}") from exc
    if len(raw) > _MAX_BYTES:
        raise PilotReferenceError(f"reference artifact exceeds size limit: {path}")
    return raw


def _read_checkpoint(raw: bytes) -> dict:
    try:
        stop = None
        for op, _, position in pickletools.genops(raw):
            if op.name in _FORBIDDEN_OPCODES:
                raise PilotReferenceError(f"forbidden pickle opcode: {op.name}")
            if op.name == "STOP":
                stop = position + 1
        if stop != len(raw):
            raise PilotReferenceError("pickle has missing STOP or trailing content")
        value = _PlainUnpickler(io.BytesIO(raw)).load()
        _plain(value)
    except PilotReferenceError:
        raise
    except Exception as exc:
        raise PilotReferenceError("invalid plain reference checkpoint") from exc
    if not isinstance(value, dict):
        raise PilotReferenceError("reference checkpoint must be a dictionary")
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _generation_metadata(value):
    if isinstance(value, dict):
        return {key: _generation_metadata(item) for key, item in value.items()
                if not _MEASUREMENT_KEY.search(str(key).lower())}
    if isinstance(value, list):
        return [_generation_metadata(item) for item in value]
    return copy.deepcopy(value)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PilotReferenceError(f"duplicate reference JSON key: {key}")
        result[key] = value
    return result


def _indices(value, label):
    if not isinstance(value, (list, tuple)) or not value:
        raise PilotReferenceError(f"{label} must be a nonempty index sequence")
    result = list(value)
    if any(type(i) is not int or i < 0 for i in result):
        raise PilotReferenceError(f"{label} contains invalid indices")
    if len(set(result)) != len(result):
        raise PilotReferenceError(f"{label} contains duplicate indices")
    return result


def _inputs(details, dev_indices, label):
    """Accept complete global-index or unambiguous local-index DEV rows."""
    if not isinstance(details, (list, tuple)) or len(details) != len(dev_indices):
        raise PilotReferenceError(f"{label} must cover every DEV row exactly once")
    by_index = {}
    for row in details:
        if not isinstance(row, dict):
            raise PilotReferenceError(f"{label} contains a non-dictionary row")
        index = row.get("row_index")
        if type(index) is not int or index < 0 or index in by_index:
            raise PilotReferenceError(f"{label} has invalid/duplicate row_index")
        if not isinstance(row.get("input"), str):
            raise PilotReferenceError(f"{label} input must be exact text")
        by_index[index] = row["input"]
    global_match = set(by_index) == set(dev_indices)
    local_match = set(by_index) == set(range(len(dev_indices)))
    if global_match and local_match and dev_indices != list(range(len(dev_indices))):
        raise PilotReferenceError(f"{label} row_index mapping is ambiguous")
    if global_match:
        return {index: by_index[index] for index in dev_indices}
    if local_match:
        return {index: by_index[local] for local, index in enumerate(dev_indices)}
    raise PilotReferenceError(f"{label} row indices do not match the DEV split")


def _artifact_path(directory, name):
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise PilotReferenceError(f"reference artifact escapes its dataset directory: {name}")
    if not path.is_file():
        raise PilotReferenceError(f"missing reference artifact: {path}")
    return path


def _initial_candidates(summary):
    if not isinstance(summary, dict):
        raise PilotReferenceError("candidate summary must be a dictionary")
    result = {}
    for agent, rows in summary.items():
        if not isinstance(agent, str) or not agent or not isinstance(rows, list):
            raise PilotReferenceError("invalid candidate agent/list")
        candidates, slots, prompts = [], set(), set()
        for row in rows:
            if not isinstance(row, dict):
                raise PilotReferenceError("candidate must be a dictionary")
            if row.get("source") != "llm_positive_seeking":
                continue
            if row.get("agent") != agent:
                raise PilotReferenceError("candidate agent does not match summary key")
            match = re.fullmatch(re.escape(agent) + r":llm:[^:]+:([1-9][0-9]*):[0-9a-f]+", str(row.get("id", "")))
            if not match or not 1 <= int(match[1]) <= 36:
                raise PilotReferenceError("initial candidate has an invalid historical slot")
            slot = f"initial_{int(match[1]):03d}"
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise PilotReferenceError("candidate prompt must be nonempty text")
            if row.get("prompt_sha256") != _sha(prompt.encode("utf-8")):
                raise PilotReferenceError("candidate prompt hash mismatch")
            if slot in slots or prompt in prompts:
                raise PilotReferenceError("duplicate initial candidate slot or prompt")
            slots.add(slot)
            prompts.add(prompt)
            # Deliberate allowlist: no pilot/full/TEST/prior/selection measures.
            clean = {key: _generation_metadata(value) for key, value in row.items()
                     if key in _GENERATION_FIELDS}
            clean["reference_initial_slot"] = slot
            clean["reference_candidate_id"] = row["id"]
            clean["generation_round"] = "initial"
            candidates.append(clean)
        if candidates:
            result[agent] = sorted(candidates, key=lambda row: row["reference_initial_slot"])
    return result


def _diagnostics(checkpoint, inputs):
    groups = checkpoint.get("agent_failure_diagnostics", {})
    if not isinstance(groups, dict):
        raise PilotReferenceError("DEV diagnostics must be grouped by agent")
    result = {}
    for agent, rows in groups.items():
        if not isinstance(agent, str) or not isinstance(rows, list):
            raise PilotReferenceError("invalid DEV diagnostic group")
        seen, clean_rows = set(), []
        for row in rows:
            if not isinstance(row, dict):
                raise PilotReferenceError("invalid DEV diagnostic row")
            index = row.get("row_index")
            if type(index) is not int or index not in inputs or index in seen:
                raise PilotReferenceError("DEV diagnostic row does not identify a unique DEV task")
            if row.get("input") != inputs[index]:
                raise PilotReferenceError("DEV diagnostic task text mismatch")
            seen.add(index)
            # Generation serializes these dictionaries without sort_keys.
            # Filter in source insertion order so an unchanged DEV payload
            # stays byte-identical, independent of PYTHONHASHSEED.
            allowed = _DIAGNOSTIC_FIELDS | {"row_index", "input", "target_trace"}
            clean = {key: copy.deepcopy(value) for key, value in row.items()
                     if key in allowed}
            clean.update(row_index=index, input=inputs[index])
            trace = row.get("target_trace", [])
            if not isinstance(trace, list) or any(not isinstance(call, dict) for call in trace):
                raise PilotReferenceError("invalid DEV diagnostic trace")
            if "target_trace" in row:
                clean["target_trace"] = [
                    {key: copy.deepcopy(value) for key, value in call.items()
                     if key in {"input", "output", "error"}}
                    for call in trace
                ]
            clean_rows.append(clean)
        result[agent] = clean_rows
    return result


def load_reference(dataset, dev_indices, baseline_details, reference_root, *,
                   expected_count=32, resample_pilot=False):
    """Return validated historical indices and score-free initial candidates.

    ``baseline_details`` must describe the complete *current* DEV task texts;
    its scores are intentionally ignored. Candidate prompt bytes (including
    whitespace) are preserved. ``dev_diagnostics_by_agent`` contains only old
    original-system DEV generation evidence, never perturbation gains.
    Missing agents have no entry and may use fresh generation on the SAME
    returned Pilot indices. No historical baseline/candidate scores for
    ranking are returned. Missing dataset directories return None; every other
    missing or mismatched configured reference raises PilotReferenceError.
    With ``resample_pilot=True``, permit a verified DEV subset and return only
    initial Prompt texts/metadata, not historical Pilot indices or diagnostics.
    """
    if not isinstance(dataset, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", dataset):
        raise PilotReferenceError("invalid dataset path component")
    if type(expected_count) is not int or expected_count != 32:
        raise PilotReferenceError("historical Pilot references require exactly 32 tasks")
    current_dev = _indices(dev_indices, "current DEV indices")
    current_inputs = _inputs(baseline_details, current_dev, "current DEV details")
    if not isinstance(reference_root, (str, Path)) or not str(reference_root).strip():
        raise PilotReferenceError("reference root must be an explicit nonempty path")
    root = project_path(reference_root).resolve()
    if not root.is_dir():
        raise PilotReferenceError(f"reference root is missing or not a directory: {root}")
    directory = root / dataset
    if not directory.exists() and not directory.is_symlink():
        return None
    if not directory.is_dir() or not directory.resolve().is_relative_to(root):
        raise PilotReferenceError("reference dataset directory is invalid or escapes root")
    checkpoint_path = _artifact_path(directory, "checkpoint.pkl")
    summary_path = _artifact_path(directory, "all_candidates_summary.json")
    checkpoint_raw, summary_raw = _read_bytes(checkpoint_path), _read_bytes(summary_path)
    checkpoint = _read_checkpoint(checkpoint_raw)
    if checkpoint.get("dataset") != dataset:
        raise PilotReferenceError("reference dataset identity mismatch")
    old_dev = _indices(checkpoint.get("dev_indices"), "reference DEV indices")
    if resample_pilot and not set(current_dev).issubset(old_dev):
        raise PilotReferenceError("current DEV subset contains tasks outside reference DEV")
    if not resample_pilot and old_dev != current_dev:
        raise PilotReferenceError("reference DEV split/order differs from current DEV")
    baseline = checkpoint.get("baseline_dev")
    if not isinstance(baseline, dict):
        raise PilotReferenceError("reference has no original DEV task details")
    historical_inputs = _inputs(baseline.get("sample_details"), old_dev, "reference DEV details")
    for index in current_dev:
        if historical_inputs[index] != current_inputs[index]:
            raise PilotReferenceError(f"reference DEV task text mismatch at index {index}")
    pilot = _indices(checkpoint.get("pilot_dev_indices"), "reference Pilot indices")
    if len(pilot) != expected_count or not set(pilot).issubset(
            old_dev if resample_pilot else current_dev):
        raise PilotReferenceError("reference Pilot must contain exactly 32 unique in-DEV indices")
    try:
        summary = json.loads(summary_raw, object_pairs_hook=_unique_json_object)
    except (ValueError, UnicodeError) as exc:
        raise PilotReferenceError("invalid candidate summary JSON") from exc
    initial = _initial_candidates(summary)
    # Subsampled experiments replay only the fixed initial Prompt library.
    # All Pilot identities, baseline diagnostics and measurements are fresh.
    diagnostics = {} if resample_pilot else _diagnostics(checkpoint, current_inputs)
    if resample_pilot:
        pilot = []
    task_hashes = {str(index): _sha(current_inputs[index].encode("utf-8")) for index in current_dev}
    fingerprint = _sha(json.dumps(
        [[index, task_hashes[str(index)]] for index in current_dev],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8"))
    result = {
        "schema_version": "pilot_reference_v1",
        "dataset": dataset,
        "dev_indices": current_dev,
        "pilot_indices": pilot,
        "random_seed": 42 + len(current_dev) * 1009,
        "initial_candidates_by_agent": initial,
        "dev_diagnostics_by_agent": diagnostics,
        "provenance": {
            "reference_root": str(root),
            "source_paths": {"checkpoint": str(checkpoint_path), "candidates": str(summary_path)},
            "source_sha256": {"checkpoint": _sha(checkpoint_raw), "candidates": _sha(summary_raw)},
            "dev_input_sha256_by_index": task_hashes,
            "dev_input_fingerprint": fingerprint,
            "dev_input_fingerprint_scope": "Exact saved benchmark sample_details.input text; historical whitespace cleanup/truncation and omitted context cannot be recovered from this snapshot.",
            "possible_truncated_input_indices": [index for index in current_dev if current_inputs[index].endswith(" ...")],
            "original_prompt_snapshot_available": False,
            "original_prompt_provenance": "Historical checkpoint has no standalone original-prompt snapshot; candidate diff is retained, not claimed as a verified snapshot.",
            "candidate_measurements_reused": False,
            "historical_baseline_scores_reused": False,
            "diagnostics_scope": "original-system DEV generation evidence only; not current scoring",
        },
    }
    if resample_pilot:
        result["provenance"]["resample_current_dev_pilot"] = True
        result["provenance"]["diagnostics_scope"] = "fresh current DEV only; historical diagnostics excluded"
        result["random_seed"] = None  # The caller supplies the new sampling seed.
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PilotReferenceError("retained reference metadata is not finite JSON data") from exc
    return result
