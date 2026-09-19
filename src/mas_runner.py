"""Pipeline, debate, and hierarchical workflows."""

import re
from typing import List, Dict, Tuple

from src.llm_cache import call_llm_cached
from src.workflow_loader import WorkflowDef
from src.errors import fail


# ============================================================


# ============================================================
_DEFAULT_PROVIDER = "openai"


def set_default_provider(provider: str) -> None:
    global _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = provider


def run_mas(workflow_def: WorkflowDef, agent_prompts: Dict[str, str],
            task_query: str, eval_llm: str, eval_temperature: float,
            dry_run: bool) -> Tuple[str, List[dict]]:
    """Execute a workflow with optional prompt overrides."""
    if dry_run:
        return "[DRY_RUN_OUTPUT]", [{"step": 0, "agent": "stub", "content": "stub"}]

    topo = workflow_def.topology_type
    if topo == "pipeline":
        return _run_pipeline(workflow_def, agent_prompts, task_query, eval_llm, eval_temperature)
    elif topo == "debate":
        return _run_debate(workflow_def, agent_prompts, task_query, eval_llm, eval_temperature)
    elif topo == "hierarchical":
        return _run_hierarchical(workflow_def, agent_prompts, task_query, eval_llm, eval_temperature)
    else:
        raise NotImplementedError(f"Topology {topo} not supported")


def _agent_step(agent_name: str, system_prompt: str, user_message: str,
                eval_llm: str, eval_temperature: float) -> str:
    """Execute one agent call."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    resp = call_llm_cached(
        provider=_DEFAULT_PROVIDER,
        model=eval_llm,
        messages=messages,
        temperature=eval_temperature,
    )
    return resp["content"]


def _agent_order_from_edges(workflow_def: WorkflowDef) -> List[str]:
    """Resolve the workflow execution order."""
    succ: Dict[str, List[str]] = {a: [] for a in workflow_def.agent_names}
    for src, dst in workflow_def.edges:
        if dst not in succ[src]:
            succ[src].append(dst)

    order, visited = [], set()
    cur = workflow_def.entry_agent
    while cur is not None and cur not in visited:
        order.append(cur)
        visited.add(cur)
        nxt = [n for n in succ.get(cur, []) if n not in visited]
        cur = nxt[0] if nxt else None


    for a in workflow_def.agent_names:
        if a not in visited:
            order.append(a)
    return order


def _run_pipeline(workflow_def, agent_prompts, task_query, eval_llm, eval_temperature):
    """Execute a sequential workflow."""
    order = _agent_order_from_edges(workflow_def)
    trace: List[dict] = []
    message = task_query

    for step, agent in enumerate(order):
        sys_prompt = agent_prompts.get(agent, f"You are {agent}.")
        user_msg = (f"Task: {task_query}\n\n"
                    f"Input from previous step:\n{message}")
        output = _agent_step(agent, sys_prompt, user_msg, eval_llm, eval_temperature)
        trace.append({"step": step, "agent": agent, "content": output})
        message = output

    final_output = trace[-1]["content"] if trace else ""
    return final_output, trace


def _run_debate(workflow_def, agent_prompts, task_query, eval_llm, eval_temperature):
    """Execute a multi-round debate."""
    agents = workflow_def.agent_names
    max_rounds = int(workflow_def.extra.get("max_rounds", 2))
    trace: List[dict] = []
    step = 0


    answers: Dict[str, str] = {}
    for agent in agents:
        sys_prompt = agent_prompts.get(agent, f"You are {agent}.")
        out = _agent_step(agent, sys_prompt, f"Task: {task_query}\n\nGive your answer.",
                          eval_llm, eval_temperature)
        answers[agent] = out
        trace.append({"step": step, "agent": agent, "content": out})
        step += 1


    for _ in range(max_rounds - 1):
        new_answers = {}
        for agent in agents:
            sys_prompt = agent_prompts.get(agent, f"You are {agent}.")
            others = "\n\n".join(f"[{a}]: {answers[a]}" for a in agents if a != agent)
            user_msg = (f"Task: {task_query}\n\nOther agents' answers:\n{others}\n\n"
                        f"Critique them and give your refined answer.")
            out = _agent_step(agent, sys_prompt, user_msg, eval_llm, eval_temperature)
            new_answers[agent] = out
            trace.append({"step": step, "agent": agent, "content": out})
            step += 1
        answers = new_answers


    aggregator = workflow_def.exit_agent
    sys_prompt = agent_prompts.get(aggregator, f"You are {aggregator}.")
    summary_input = "\n\n".join(f"[{a}]: {answers[a]}" for a in agents)
    final_output = _agent_step(
        aggregator, sys_prompt,
        f"Task: {task_query}\n\nAll answers:\n{summary_input}\n\n"
        f"Provide the final consensus answer.",
        eval_llm, eval_temperature,
    )
    trace.append({"step": step, "agent": aggregator, "content": final_output})
    return final_output, trace


def _run_hierarchical(workflow_def, agent_prompts, task_query, eval_llm, eval_temperature):
    """Execute a manager-worker workflow."""
    orchestrator = workflow_def.entry_agent
    workers = [a for a in workflow_def.agent_names if a != orchestrator]
    max_rounds = int(workflow_def.extra.get("max_rounds", len(workers) or 1))
    trace: List[dict] = []
    step = 0

    orch_prompt = agent_prompts.get(orchestrator, f"You are {orchestrator}, a manager.")
    context = f"Task: {task_query}"


    worker_outputs: List[str] = []
    for r in range(min(max_rounds, len(workers))):
        worker = workers[r]

        subtask = _agent_step(
            orchestrator, orch_prompt,
            f"{context}\n\nDelegate the next subtask to {worker}. "
            f"State the subtask instruction only.",
            eval_llm, eval_temperature,
        )
        trace.append({"step": step, "agent": orchestrator, "content": subtask})
        step += 1


        w_prompt = agent_prompts.get(worker, f"You are {worker}.")
        w_out = _agent_step(worker, w_prompt, f"Subtask: {subtask}",
                            eval_llm, eval_temperature)
        trace.append({"step": step, "agent": worker, "content": w_out})
        step += 1

        worker_outputs.append(f"[{worker}]: {w_out}")
        context += f"\n\n{worker}'s result: {w_out}"


    final_output = _agent_step(
        orchestrator, orch_prompt,
        f"{context}\n\nBased on all worker results, give the FINAL_ANSWER.",
        eval_llm, eval_temperature,
    )
    trace.append({"step": step, "agent": orchestrator, "content": final_output})
    return final_output, trace


# ============================================================
# Task-level loss
# ============================================================
def compute_task_loss(final_output: str, task_answer: str) -> float:
    """Score a workflow result as loss."""
    if not task_answer or not task_answer.strip():
        fail("mas_runner.compute_task_loss",
             "task_answer is empty — cannot score this task",
             final_output_preview=(final_output or "")[:80])

    pred = _normalize_answer(final_output)
    gold = _normalize_answer(task_answer)
    if not gold:
        fail("mas_runner.compute_task_loss",
             "task_answer normalized to empty string",
             raw_answer=task_answer[:80])


    gold_num = _extract_last_number(task_answer)
    if gold_num is not None:
        pred_num = _extract_last_number(final_output)
        return 0.0 if (pred_num is not None and abs(pred_num - gold_num) < 1e-6) else 1.0


    return 0.0 if gold in pred else 1.0


def _normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_last_number(text: str):
    nums = re.findall(r"-?\d+\.?\d*", (text or "").replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None
