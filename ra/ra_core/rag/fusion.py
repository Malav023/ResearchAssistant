from __future__ import annotations


def rrf(
    rankings: list[list[int]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion. Each ranking is doc ids, best first.
    score(d) = sum_i w_i / (k + rank_i(d) + 1). Returns [(doc, score)] best first."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
