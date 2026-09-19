"""Perturbation geometry diagnostics."""
from typing import Dict, List, Optional
import numpy as np

from src.variant_generator import generate_variants, AgentVariant
from src.encoder import encode_prompt
from src import aflow_adapter


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _effective_rank(unit_dirs: np.ndarray, energy=0.90) -> int:
    """Compute effective rank from singular values."""
    if unit_dirs.shape[0] < 2:
        return unit_dirs.shape[0]
    s = np.linalg.svd(unit_dirs, compute_uv=False)
    e = (s ** 2)
    c = np.cumsum(e) / e.sum()
    return int(np.searchsorted(c, energy) + 1)


def _mean_pairwise_cos(unit_dirs: List[np.ndarray]) -> float:
    n = len(unit_dirs)
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += _cos(unit_dirs[i], unit_dirs[j]); cnt += 1
    return tot / cnt if cnt else 0.0


def diagnose_dataset(dataset: str, encoder, num_per_agent: int, mix: dict,
                     generator_llm: str, dry_run: bool = False,
                     gradient_eval=None) -> dict:
    """Summarize perturbation coverage and geometry."""
    original = aflow_adapter.get_original_prompts(dataset)
    variants = generate_variants(original, num_per_agent, mix, generator_llm,
                                 encoder, dry_run=dry_run)

    out_agents = {}
    for agent, vs in variants.items():
        z_orig = encode_prompt(encoder, original[agent])
        norm_orig = float(np.linalg.norm(z_orig))
        rows = []
        for v in vs:
            z_var = z_orig + v.embedding_diff
            rows.append({
                "source": v.source, "edit_type": v.edit_type,
                "dist": v.norm,
                "rel_dist": (v.norm / norm_orig if norm_orig > 1e-9 else 0.0),
                "cos_orig": _cos(z_orig, z_var),
                "char_delta": len(v.variant_text) - len(original[agent]),
                "preview": v.variant_text[:80].replace("\n", " "),
                "text": v.variant_text,
            })
        dists = [r["dist"] for r in rows]
        dirs = [v.unit_direction for v in vs]
        agent_rec = {
            "n_variants": len(vs),
            "original_prompt": original[agent],
            "norm_orig": norm_orig,
            "dist_mean": float(np.mean(dists)) if dists else 0.0,
            "dist_min": float(np.min(dists)) if dists else 0.0,
            "dist_max": float(np.max(dists)) if dists else 0.0,
            "mean_pairwise_cos_dir": _mean_pairwise_cos(dirs),
            "effective_rank_90": _effective_rank(np.stack(dirs)) if dirs else 0,
            "rows": rows,
        }


        if gradient_eval is not None:
            ge = gradient_eval
            qi = list(range(ge["n_samples"]))
            L0 = aflow_adapter.evaluate_with_prompts(
                dataset, original, qi, model=ge["eval_model"],
                n_repeats=ge.get("n_repeats", 1),
                concurrent_batch=ge.get("concurrent_batch", 50),
                dry_run=dry_run, split="validate")
            for r, v in zip(rows, vs):
                trial = dict(original); trial[agent] = v.variant_text
                Lm = aflow_adapter.evaluate_with_prompts(
                    dataset, trial, qi, model=ge["eval_model"],
                    n_repeats=ge.get("n_repeats", 1),
                    concurrent_batch=ge.get("concurrent_batch", 50),
                    dry_run=dry_run, split="validate")
                r["delta_L"] = L0 - Lm
                r["dir_deriv"] = (Lm - L0) / v.norm if v.norm > 1e-9 else 0.0
            agent_rec["baseline_L"] = L0

        out_agents[agent] = agent_rec

    return {"dataset": dataset, "num_per_agent": num_per_agent,
            "mix": mix, "agents": out_agents}
