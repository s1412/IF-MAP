"""Prompt perturbation and workflow evaluation."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from tqdm import tqdm

from src.mas_runner import run_mas, compute_task_loss
from src.llm_cache import call_llm_cached
from src.errors import fail, report_invalid

_DEFAULT_PROVIDER = "openai"


def set_default_provider(provider: str) -> None:
    global _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = provider


@dataclass
class PerturbationResult:
    agent_name:        str
    variant_idx:       int                  # 0 = baseline, 1..M = variants
    L_observed:        float
    L_self_observed:   Optional[float] = None
    variance:          float = 0.0
    task_level_losses: List[float] = field(default_factory=list)
    sample_details:    List[dict] = field(default_factory=list)


def run_perturbation_loop(
    trace, workflow_def, agent_variants,
    n_eval_samples, n_repeats,
    eval_llm, eval_temperature,
    compute_i_self, i_self_judge_llm,
    dry_run,
) -> Dict[str, List[PerturbationResult]]:
    """Measure single-agent prompt interventions."""

    eval_task_set = _prepare_eval_task_set(trace, n_eval_samples)


    baseline_L, baseline_self, baseline_losses = _evaluate_mas_on_tasks(
        agent_prompts=trace.agent_prompts, workflow_def=workflow_def,
        tasks=eval_task_set, n_repeats=n_repeats,
        eval_llm=eval_llm, eval_temperature=eval_temperature,
        compute_i_self=compute_i_self, i_self_judge_llm=i_self_judge_llm,
        dry_run=dry_run,
    )

    results: Dict[str, List[PerturbationResult]] = {a: [] for a in agent_variants}


    total_variants = sum(len(v) for v in agent_variants.values())
    pbar = tqdm(total=total_variants, desc=f"[Perturb] {trace.trace_id[:8]}",
                disable=dry_run)

    for agent_name, variants in agent_variants.items():

        results[agent_name].append(PerturbationResult(
            agent_name=agent_name, variant_idx=0,
            L_observed=baseline_L,
            L_self_observed=baseline_self.get(agent_name),
            variance=0.0, task_level_losses=list(baseline_losses),
        ))

        for v_idx, variant in enumerate(variants, start=1):
            perturbed = dict(trace.agent_prompts)
            perturbed[agent_name] = variant.variant_text

            L_obs, self_dict, task_losses = _evaluate_mas_on_tasks(
                agent_prompts=perturbed, workflow_def=workflow_def,
                tasks=eval_task_set, n_repeats=n_repeats,
                eval_llm=eval_llm, eval_temperature=eval_temperature,
                compute_i_self=compute_i_self, i_self_judge_llm=i_self_judge_llm,
                dry_run=dry_run, focus_agent=agent_name,
            )
            results[agent_name].append(PerturbationResult(
                agent_name=agent_name, variant_idx=v_idx,
                L_observed=L_obs,
                L_self_observed=self_dict.get(agent_name),
                variance=_compute_variance(task_losses),
                task_level_losses=task_losses,
            ))
            pbar.update(1)

    pbar.close()
    return results


def _evaluate_mas_on_tasks(
    agent_prompts, workflow_def, tasks,
    n_repeats, eval_llm, eval_temperature,
    compute_i_self, i_self_judge_llm, dry_run,
    focus_agent: Optional[str] = None,
):
    """Average workflow loss over evaluation tasks."""
    if dry_run:
        return 0.5, {a: 0.5 for a in agent_prompts}, [0.5]

    all_losses: List[float] = []
    self_scores: Dict[str, List[float]] = {}

    for task in tasks:
        for _ in range(n_repeats):
            final_output, exec_trace = run_mas(
                workflow_def=workflow_def, agent_prompts=agent_prompts,
                task_query=task["query"], eval_llm=eval_llm,
                eval_temperature=eval_temperature, dry_run=False,
            )

            all_losses.append(compute_task_loss(final_output, task.get("answer", "")))


            if compute_i_self:
                agents_to_judge = [focus_agent] if focus_agent else list(agent_prompts)
                for agent in agents_to_judge:
                    self_scores.setdefault(agent, []).append(
                        _judge_agent_output(agent, exec_trace, task, i_self_judge_llm)
                    )

    if not all_losses:
        fail("perturbation._evaluate_mas_on_tasks",
             "no task losses produced (empty task set or all repeats failed)",
             n_tasks=len(tasks), n_repeats=n_repeats, focus_agent=focus_agent)

    L_avg = sum(all_losses) / len(all_losses)
    self_avg = {a: sum(v) / len(v) for a, v in self_scores.items() if v}
    return L_avg, self_avg, all_losses


def _judge_agent_output(agent_name, exec_trace, task, judge_llm) -> float:
    """Judge an agent response."""
    agent_steps = [s for s in exec_trace if s.get("agent") == agent_name]
    if not agent_steps:
        fail("perturbation._judge_agent_output",
             "focus agent produced no steps in exec_trace — cannot judge self quality",
             agent=agent_name,
             trace_agents=sorted({s.get("agent") for s in exec_trace}))
    output_text = "\n".join(s["content"] for s in agent_steps)[:3000]

    instruction = (
        f"You are evaluating the quality of a single agent's output in a "
        f"multi-agent system solving this task:\n{task['query']}\n\n"
        f"Agent '{agent_name}' produced:\n{output_text}\n\n"
        f"Rate the intrinsic quality of THIS agent's output (ignore downstream) "
        f"on a scale 0-5. Reply with only the number."
    )
    resp = call_llm_cached(
        provider=_DEFAULT_PROVIDER, model=judge_llm,
        messages=[{"role": "user", "content": instruction}],
        temperature=0.0,
    )
    score = _parse_score(resp["content"], agent_name)
    quality = max(0.0, min(1.0, score / 5.0))
    return 1.0 - quality   # self loss


def _parse_score(text: str, agent_name: str) -> float:
    """Parse a numeric judge score."""
    import re
    m = re.search(r"-?\d+\.?\d*", text or "")
    if not m:
        fail("perturbation._parse_score",
             "judge output contains no parseable number",
             agent=agent_name, judge_output=(text or "")[:120])
    try:
        return float(m.group(0))
    except ValueError:
        fail("perturbation._parse_score",
             "matched token is not a valid float",
             agent=agent_name, token=m.group(0))


def _prepare_eval_task_set(trace, n_samples: int) -> List[dict]:
    """Build the evaluation task set."""
    base = {"query": trace.task_query, "answer": trace.task_answer}

    return [base]


def _compute_variance(task_level_losses: List[float]) -> float:
    n = len(task_level_losses)
    if n < 2:
        return 0.0
    mean = sum(task_level_losses) / n
    return sum((x - mean) ** 2 for x in task_level_losses) / (n - 1)
