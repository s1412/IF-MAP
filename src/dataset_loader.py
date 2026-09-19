"""Attribution datasets and trace normalization."""

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.errors import report_invalid
from src.paths import project_path


@dataclass
class TraceData:
    trace_id:                  str
    source_dataset:            str
    task_query:                str
    task_answer:               str
    agent_prompts:             Dict[str, str]        # {agent_name: original_prompt_text}
    agent_history:             List[dict]            # [{"agent", "step", "content"}, ...]
    failure_responsible_agent: str
    decisive_error_step:       Optional[int] = None
    failure_explanation:       str = ""
    extra:                     dict = field(default_factory=dict)

    @property
    def agent_names(self) -> set:
        return set(self.agent_prompts.keys())


_LOADERS = {}


def load_dataset_subset(dataset_root: str, subset_name: str,
                        n_limit: Optional[int] = None) -> List[TraceData]:
    """Load normalized traces from a dataset subset."""
    if subset_name not in _LOADERS:
        raise ValueError(
            f"Unknown subset '{subset_name}'. Available: {list(_LOADERS)}"
        )

    traces = _LOADERS[subset_name](str(project_path(dataset_root)))
    if n_limit is not None:
        traces = traces[:n_limit]

    _print_load_stats(subset_name, traces)
    return traces


def _print_load_stats(subset_name: str, traces: List[TraceData]) -> None:
    if not traces:
        print(f"[DatasetLoader] {subset_name}: 0 traces loaded")
        return
    avg_agents = sum(len(t.agent_names) for t in traces) / len(traces)
    print(f"[DatasetLoader] {subset_name}: {len(traces)} traces, "
          f"avg {avg_agents:.1f} agents/trace")


def _normalize_agent_name(role_or_name: str) -> Optional[str]:
    """Normalize trace agent names."""
    if not role_or_name:
        return None
    name = role_or_name.strip()

    name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
    if name.lower() in ("human", "user", "system", "assistant", ""):
        return None
    return name


def _extract_agents_from_history(history: List[dict]):
    """Extract agents and prompts from trace history."""
    agent_prompts: Dict[str, str] = {}
    norm_history: List[dict] = []
    seen_order: List[str] = []

    for step, item in enumerate(history):
        role = item.get("role") or item.get("name") or ""

        agent = _normalize_agent_name(item.get("name") or role)
        content = item.get("content", "")
        norm_history.append({"agent": agent, "step": step, "content": content,
                             "raw_role": role})
        if agent is None:
            continue
        if agent not in agent_prompts and content.strip():
            agent_prompts[agent] = content
            seen_order.append(agent)

    return seen_order, agent_prompts, norm_history


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ============================================================
# Who&When — Hand-Crafted (Magentic-One)
# ============================================================
def _load_who_when_hc(dataset_root: str) -> List[TraceData]:
    files = sorted(
        glob.glob(os.path.join(dataset_root, "Hand-Crafted", "*.json")),
        key=lambda p: _safe_int(os.path.splitext(os.path.basename(p))[0]) or 0,
    )
    return [t for f in files if (t := _parse_who_when_file(f, "who_when_hc")) is not None]


def _load_who_when_ag(dataset_root: str) -> List[TraceData]:
    files = sorted(
        glob.glob(os.path.join(dataset_root, "Algorithm-Generated", "*.json")),
        key=lambda p: _safe_int(os.path.splitext(os.path.basename(p))[0]) or 0,
    )
    return [t for f in files if (t := _parse_who_when_file(f, "who_when_ag")) is not None]


def _parse_who_when_file(path: str, source: str) -> Optional[TraceData]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    history = d.get("history", [])
    if not history:
        report_invalid("dataset_loader._parse_who_when_file",
                       "trace has empty history — skipped", source_file=os.path.basename(path))
        return None

    _, agent_prompts, norm_history = _extract_agents_from_history(history)
    if not agent_prompts:
        report_invalid("dataset_loader._parse_who_when_file",
                       "no agent prompts extracted from history — skipped",
                       source_file=os.path.basename(path))
        return None

    decisive_step = _safe_int(d.get("mistake_step"))
    if decisive_step is None and d.get("mistake_step") is not None:
        report_invalid("dataset_loader._parse_who_when_file",
                       "mistake_step not parseable as int",
                       source_file=os.path.basename(path),
                       mistake_step=d.get("mistake_step"))


    sys_prompt = d.get("system_prompt")
    if sys_prompt:
        for a in list(agent_prompts.keys()):
            agent_prompts[a] = sys_prompt

    gt_agent = _normalize_agent_name(d.get("mistake_agent", "")) or d.get("mistake_agent", "")
    trace_id = d.get("question_ID") or os.path.splitext(os.path.basename(path))[0]

    return TraceData(
        trace_id=str(trace_id),
        source_dataset=source,
        task_query=d.get("question", ""),
        task_answer=d.get("ground_truth", ""),
        agent_prompts=agent_prompts,
        agent_history=norm_history,
        failure_responsible_agent=gt_agent,
        decisive_error_step=decisive_step,
        failure_explanation=d.get("mistake_reason", ""),
        extra={
            "level": d.get("level"),
            "is_correct": d.get("is_correct"),
            "source_file": os.path.basename(path),
        },
    )


def _load_trail(dataset_root: str) -> List[TraceData]:
    raise NotImplementedError(
        "TRAIL loader is not implemented. Parse OpenInference Parquet "
        "spans into agent_history with pandas/pyarrow."
    )


def _load_mp_bench(dataset_root: str) -> List[TraceData]:
    raise NotImplementedError(
        "MP-Bench loader is not implemented; attribution labels may be lists."
    )


_LOADERS.update({
    "hc":       _load_who_when_hc,
    "ag":       _load_who_when_ag,
    "trail":    _load_trail,
    "mp_bench": _load_mp_bench,
})
