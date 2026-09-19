"""YAML workflow loading and validation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import yaml
from src.paths import project_path


@dataclass
class WorkflowDef:
    topology_type: str                       # "pipeline" / "debate" / "hierarchical" / "graph"
    agent_names:   List[str]
    edges:         List[Tuple[str, str]]
    entry_agent:   str
    exit_agent:    str
    extra:         dict = field(default_factory=dict)


_REQUIRED_FIELDS = ["topology_type", "agent_names", "edges", "entry_agent", "exit_agent"]
_VALID_TOPOLOGIES = {"pipeline", "debate", "hierarchical", "graph"}


def load_workflow(workflow_yaml_path: str) -> WorkflowDef:
    """Load a YAML workflow."""
    with open(project_path(workflow_yaml_path), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)


    missing = [k for k in _REQUIRED_FIELDS if k not in raw]
    if missing:
        raise ValueError(f"Workflow {workflow_yaml_path} missing fields: {missing}")

    topology = raw["topology_type"]
    if topology not in _VALID_TOPOLOGIES:
        raise ValueError(
            f"Unknown topology_type '{topology}'. Valid: {_VALID_TOPOLOGIES}"
        )

    agent_names = list(raw["agent_names"])
    name_set = set(agent_names)


    edges = [tuple(e) for e in raw["edges"]]
    for src, dst in edges:
        if src not in name_set:
            raise ValueError(f"Edge source '{src}' not in agent_names")
        if dst not in name_set:
            raise ValueError(f"Edge target '{dst}' not in agent_names")


    for role, agent in [("entry_agent", raw["entry_agent"]),
                        ("exit_agent", raw["exit_agent"])]:
        if agent not in name_set:
            raise ValueError(f"{role} '{agent}' not in agent_names")

    return WorkflowDef(
        topology_type=topology,
        agent_names=agent_names,
        edges=edges,
        entry_agent=raw["entry_agent"],
        exit_agent=raw["exit_agent"],
        extra=raw.get("extra", {}) or {},
    )


def validate_workflow_for_trace(workflow_def: WorkflowDef, trace_agent_names: set):
    """Check workflow compatibility with a trace."""
    warnings: List[str] = []
    wf_names = set(workflow_def.agent_names)


    undefined = trace_agent_names - wf_names
    if undefined:
        warnings.append(
            f"[ERROR] trace uses agents not in workflow: {undefined}"
        )


    unused = wf_names - trace_agent_names
    if unused:
        warnings.append(
            f"[WARN] workflow defines agents unused by trace: {unused}"
        )

    is_ok = len(undefined) == 0
    return is_ok, warnings
