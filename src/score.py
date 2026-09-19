"""G-norm, I_abs, and MeanAbs potential scores."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings

import numpy as np

from src.errors import fail, report_invalid


POTENTIAL_METRICS = ("g_norm", "i_abs", "mean_abs_delta")
DEFAULT_MAIN_METRIC = "g_norm"


def normalize_main_metric(main_metric: str) -> str:
    """Read old configuration names without reactivating retired metrics."""
    if main_metric in (None, "both", "i_best"):
        if main_metric is not None:
            warnings.warn(
                f"Legacy main_metric={main_metric!r} now selects by g_norm; "
                "stored historical selections are not changed.",
                DeprecationWarning, stacklevel=2,
            )
        return DEFAULT_MAIN_METRIC
    if main_metric not in POTENTIAL_METRICS:
        raise ValueError(f"main_metric must be one of {POTENTIAL_METRICS}")
    return main_metric


@dataclass
class AgentScore:
    agent_name: str
    i_abs:      Optional[float] = None
    mean_abs_delta: Optional[float] = None
    g_norm:     Optional[float] = None
    g_vector:   Optional[np.ndarray] = None
    rank:       int = -1
    ci_low:     float = 0.0
    ci_high:    float = 0.0


def compute_scores(perturbation_results, agent_variants, main_metric,
                   regression_method, ridge_lambda, pca_components
                   ) -> Dict[str, AgentScore]:
    """Compute potential scores and rank agents."""
    main_metric = normalize_main_metric(main_metric)

    scores: Dict[str, AgentScore] = {}

    for agent_name, agent_results in perturbation_results.items():
        score = AgentScore(agent_name=agent_name)


        score.i_abs = _compute_i_abs(agent_results)

        score.mean_abs_delta = _compute_mean_abs_delta(agent_results)

        g_norm, g_vec = _compute_g_norm(
            agent_results, agent_variants.get(agent_name, []),
            regression_method, ridge_lambda, pca_components,
        )
        score.g_norm = g_norm
        score.g_vector = g_vec


        metric_fn = {
            "i_abs": _compute_i_abs,
            "mean_abs_delta": _compute_mean_abs_delta,
            "g_norm": lambda r: _compute_g_norm(
                r, agent_variants.get(agent_name, []), regression_method,
                ridge_lambda, pca_components)[0],
        }[main_metric]
        score.ci_low, score.ci_high = _bootstrap_ci(agent_results, metric_fn)

        scores[agent_name] = score

    _rank_agents(scores, main_metric)
    return scores


# ============================================================
# Absolute perturbation effects
# ============================================================
def _compute_i_abs(agent_results: List) -> float:
    """Maximum absolute loss change."""
    if not agent_results:
        return 0.0
    baseline_L = agent_results[0].L_observed
    best = 0.0
    for r in agent_results[1:]:
        change = abs(baseline_L - r.L_observed)
        if change > best:
            best = change
    return best


def _compute_mean_abs_delta(agent_results: List) -> float:
    """Mean absolute loss change."""
    if not agent_results or len(agent_results) < 2:
        return 0.0
    baseline_L = agent_results[0].L_observed
    changes = [abs(baseline_L - r.L_observed) for r in agent_results[1:]]
    return float(np.mean(changes)) if changes else 0.0


def _compute_g_norm(agent_results, agent_variants_list, regression_method,
                    ridge_lambda, pca_components):
    """Fit directional loss derivatives by ridge regression."""
    agent = agent_results[0].agent_name if agent_results else "?"
    if not agent_results or len(agent_results) < 2 or not agent_variants_list:
        fail("score._compute_g_norm",
             "insufficient data to estimate gradient (need baseline + >=1 variant)",
             agent=agent, n_results=len(agent_results),
             n_variants=len(agent_variants_list))

    baseline_L = agent_results[0].L_observed


    directions, derivatives = [], []
    for r in agent_results[1:]:
        vi = r.variant_idx - 1
        if vi < 0 or vi >= len(agent_variants_list):
            report_invalid("score._compute_g_norm",
                           "variant_idx out of range vs agent_variants_list — skipped",
                           agent=agent, variant_idx=r.variant_idx,
                           n_variants=len(agent_variants_list))
            continue
        variant = agent_variants_list[vi]
        if variant.norm < 1e-9:
            report_invalid("score._compute_g_norm",
                           "degenerate variant (||diff||~0) — skipped",
                           agent=agent, variant_idx=r.variant_idx, norm=variant.norm)
            continue
        directions.append(variant.unit_direction)
        derivatives.append((r.L_observed - baseline_L) / variant.norm)

    if len(directions) < 1:
        fail("score._compute_g_norm",
             "no valid direction/derivative pairs after filtering",
             agent=agent)

    V = np.stack(directions, axis=0)          # (M, d)
    g_obs = np.array(derivatives)             # (M,)

    if regression_method == "ridge_with_pca":
        g_hat = _ridge_with_pca(V, g_obs, ridge_lambda, pca_components)
    else:  # ridge_raw
        g_hat = _ridge_raw(V, g_obs, ridge_lambda)

    return float(np.linalg.norm(g_hat)), g_hat


def _ridge_raw(V, g_obs, ridge_lambda):
    """Fit ridge regression in the original embedding space."""
    d = V.shape[1]
    A = V.T @ V + ridge_lambda * np.eye(d)
    return np.linalg.solve(A, V.T @ g_obs)


def _ridge_with_pca(V, g_obs, ridge_lambda, pca_components):
    """Fit ridge regression after PCA projection."""
    M, d = V.shape
    r = pca_components if pca_components is not None else max(1, min(M - 1, M))
    r = max(1, min(r, M))


    mean = V.mean(axis=0, keepdims=True)
    Vc = V - mean


    try:
        U, S, Wt = np.linalg.svd(Vc, full_matrices=False)
    except np.linalg.LinAlgError:
        return _ridge_raw(V, g_obs, ridge_lambda)
    r = min(r, Wt.shape[0])
    W = Wt[:r].T


    Z = V @ W                          # (M, r)
    A = Z.T @ Z + ridge_lambda * np.eye(r)
    g_pca = np.linalg.solve(A, Z.T @ g_obs)   # (r,)


    g_hat = W @ g_pca                  # (d,)
    return g_hat


# ============================================================
# Bootstrap CI
# ============================================================
def _bootstrap_ci(agent_results, metric_fn, n_bootstrap=1000, alpha=0.05):
    """Bootstrap the selected metric."""
    variants = agent_results[1:]
    if len(variants) < 2:
        val = metric_fn(agent_results)
        return float(val), float(val)

    baseline = agent_results[0]
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_bootstrap):
        sampled = list(rng.choice(variants, size=len(variants), replace=True))
        vals.append(metric_fn([baseline] + sampled))
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi


def _rank_agents(scores_dict: Dict[str, AgentScore], main_metric: str) -> None:
    """Rank by the configured retained metric (default: G-norm)."""
    main_metric = normalize_main_metric(main_metric)
    def key_of(s: AgentScore):
        value = getattr(s, main_metric)
        return value if value is not None else -1.0

    ordered = sorted(scores_dict.values(), key=key_of, reverse=True)
    for rank, s in enumerate(ordered, start=1):
        s.rank = rank
