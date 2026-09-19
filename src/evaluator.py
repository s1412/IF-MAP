"""Attribution ranking metrics."""

from dataclasses import dataclass, field
from typing import Dict, List

from src.score import AgentScore
from src.errors import fail


@dataclass
class PredictionRecord:
    trace_id:          str
    gt_failure_agent:  str
    predicted_ranking: List[str]
    predicted_scores:  Dict[str, AgentScore]
    top1_correct:      bool
    top3_correct:      bool
    rank_of_gt:        int
    metadata:          dict = field(default_factory=dict)


def evaluate_ranking(trace, predicted_scores: Dict[str, AgentScore]) -> PredictionRecord:
    """Compare an agent ranking with attribution labels."""

    ranking = [s.agent_name for s in
               sorted(predicted_scores.values(), key=lambda s: s.rank)]


    gt = trace.failure_responsible_agent
    gt_set = set(gt) if isinstance(gt, (list, set, tuple)) else {gt}

    rank_of_gt = -1
    for i, agent in enumerate(ranking, start=1):
        if agent in gt_set:
            rank_of_gt = i
            break

    top1 = bool(ranking) and ranking[0] in gt_set
    top3 = any(a in gt_set for a in ranking[:3])

    return PredictionRecord(
        trace_id=trace.trace_id,
        gt_failure_agent=gt if isinstance(gt, str) else ",".join(gt_set),
        predicted_ranking=ranking,
        predicted_scores=predicted_scores,
        top1_correct=top1,
        top3_correct=top3,
        rank_of_gt=rank_of_gt,
        metadata={
            "source_dataset": trace.source_dataset,
            "n_agents": len(predicted_scores),
            **trace.extra,
        },
    )


def aggregate_metrics(all_predictions: List[PredictionRecord]) -> dict:
    """Aggregate attribution metrics across traces."""
    n = len(all_predictions)
    if n == 0:
        return {"n_traces": 0}

    top1 = sum(p.top1_correct for p in all_predictions) / n
    top3 = sum(p.top3_correct for p in all_predictions) / n
    mrr = sum(1.0 / p.rank_of_gt for p in all_predictions if p.rank_of_gt > 0) / n


    per_dataset: Dict[str, List[PredictionRecord]] = {}
    for p in all_predictions:
        per_dataset.setdefault(p.metadata.get("source_dataset", "unknown"), []).append(p)

    per_dataset_metrics = {
        ds: {
            "n": len(preds),
            "top1": sum(x.top1_correct for x in preds) / len(preds),
            "top3": sum(x.top3_correct for x in preds) / len(preds),
        }
        for ds, preds in per_dataset.items()
    }


    per_agent_count: Dict[int, List[PredictionRecord]] = {}
    for p in all_predictions:
        per_agent_count.setdefault(p.metadata.get("n_agents", 0), []).append(p)
    per_agent_metrics = {
        k: {"n": len(v), "top1": sum(x.top1_correct for x in v) / len(v)}
        for k, v in sorted(per_agent_count.items())
    }

    return {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "mrr": mrr,
        "n_traces": n,
        "top1_ci": _wilson_ci(sum(p.top1_correct for p in all_predictions), n),
        "per_dataset": per_dataset_metrics,
        "per_agent_count": per_agent_metrics,
    }


def _wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson interval for a binomial proportion."""
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def compute_spearman_with_gt(predicted_ranking, gt_full_ranking):
    """Spearman correlation with ground-truth scores."""
    from scipy.stats import spearmanr
    rank_map = {a: i for i, a in enumerate(predicted_ranking)}
    gt_map = {a: i for i, a in enumerate(gt_full_ranking)}
    common = [a for a in gt_full_ranking if a in rank_map]
    if len(common) < 2:
        fail("evaluator.compute_spearman_with_gt",
             "fewer than 2 agents common to predicted and GT rankings",
             n_common=len(common), predicted=predicted_ranking,
             gt=gt_full_ranking)
    x = [rank_map[a] for a in common]
    y = [gt_map[a] for a in common]
    corr, _ = spearmanr(x, y)
    return corr
