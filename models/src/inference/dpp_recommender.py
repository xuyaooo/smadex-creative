"""
DPP / MMR diversification + perf-aware re-ranking for the creative recommender.

Two complementary tweaks on top of vanilla cosine-NN retrieval:

  1. rerank_by_perf — pull a wider candidate set from FAISS, then re-rank by a
     convex combination of cosine similarity and (normalized) perf_score so the
     "what to clone" use case surfaces high-performing similar creatives, not
     similarly-styled low-performers.

  2. mmr_diversify  — greedy submodular Maximal Marginal Relevance
     (Carbonell & Goldstein, SIGIR 1998) to pick a slate of k creatives that is
     simultaneously relevant (high re-rank score) and diverse (low pairwise
     embedding similarity inside the chosen slate). This is the same
     submodular relaxation behind k-DPP greedy MAP inference.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]. Constant arrays collapse to all-zeros."""
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def rerank_by_perf(
    candidates: Sequence[int],
    similarities: Sequence[float],
    perf_scores: Sequence[float],
    alpha: float = 0.7,
) -> List[int]:
    """Re-rank candidate indices by ``alpha * sim + (1 - alpha) * perf``.

    Both signals are min-max-normalized into [0, 1] over the candidate pool
    before mixing, so the alpha knob is meaningful even when perf_score and
    similarity live on different scales.

    Args:
        candidates: candidate indices (positions into the original arrays).
        similarities: cosine similarity of each candidate to the query.
        perf_scores: perf_score of each candidate.
        alpha: weight on similarity (default 0.7). 1-alpha goes to perf.

    Returns:
        ``candidates`` re-ordered descending by combined score.
    """
    if len(candidates) == 0:
        return []
    sims = _normalize(np.asarray(similarities, dtype=np.float64))
    perfs = _normalize(np.asarray(perf_scores, dtype=np.float64))
    combined = alpha * sims + (1.0 - alpha) * perfs
    order = np.argsort(-combined)  # descending
    return [int(candidates[i]) for i in order]


def mmr_diversify(
    candidate_embeddings: np.ndarray,
    scores: Sequence[float],
    k: int = 5,
    lambda_: float = 0.5,
) -> List[int]:
    """Greedy MMR / submodular slate selection.

    Starts with the highest-scoring candidate, then iteratively adds the
    candidate that maximizes::

        lambda_ * relevance(c) - (1 - lambda_) * max_{s in S} sim(c, s)

    where ``S`` is the slate built so far. Embeddings should already be
    L2-normalized so that the dot product is cosine similarity.

    Args:
        candidate_embeddings: ``(N, d)`` matrix of L2-normalized embeddings.
        scores: per-candidate relevance score (e.g. the rerank_by_perf score).
        k: slate size to return.
        lambda_: mixing weight in [0, 1]. 1.0 reduces to pure top-k by score.

    Returns:
        Indices into ``candidate_embeddings`` of the selected slate, in the
        order they were greedily picked.
    """
    embs = np.asarray(candidate_embeddings, dtype=np.float64)
    n = embs.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    rel = _normalize(np.asarray(scores, dtype=np.float64))

    selected: List[int] = []
    remaining = list(range(n))

    # Seed with the highest-relevance candidate.
    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    # Track max similarity from each remaining candidate to the selected slate.
    max_sim = embs[remaining] @ embs[first] if remaining else np.array([])

    while len(selected) < k and remaining:
        rem_arr = np.asarray(remaining)
        mmr_score = lambda_ * rel[rem_arr] - (1.0 - lambda_) * max_sim
        best_local = int(np.argmax(mmr_score))
        chosen = int(rem_arr[best_local])
        selected.append(chosen)

        # Drop ``chosen`` from remaining and update its max-sim cache.
        del remaining[best_local]
        max_sim = np.delete(max_sim, best_local)
        if remaining:
            new_sims = embs[remaining] @ embs[chosen]
            max_sim = np.maximum(max_sim, new_sims)

    return selected
