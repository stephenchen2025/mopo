"""Frontier evaluation metrics.

Reporting mean reward for a multi-objective run defeats the purpose of the run. These are
the four numbers worth looking at, and the fourth one -- controllability -- is the one
that catches the most common silent failure.
"""

from __future__ import annotations

import numpy as np

from .pareto import hypervolume, nondominated_mask

__all__ = [
    "spacing",
    "max_spread",
    "spearman",
    "controllability",
    "steerability",
    "frontier_summary",
    "FrontierArchive",
]


def spacing(F: np.ndarray) -> float:
    """Schott's spacing metric over the non-dominated subset. **Lower is better.**

    Standard deviation of the distance from each front point to its nearest neighbour: 0
    means perfectly even coverage, large means the frontier is a few clumps with gaps.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    F = F[nondominated_mask(F)]
    n = F.shape[0]
    if n < 2:
        return 0.0
    d = np.abs(F[:, None, :] - F[None, :, :]).sum(axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    return float(np.sqrt(np.mean((nearest.mean() - nearest) ** 2)))


def max_spread(F: np.ndarray) -> float:
    """Zitzler's maximum spread over the non-dominated subset. **Higher is better.**

    The diagonal of the bounding box of the front: how much of the trade-off range the
    policy family actually reaches.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    F = F[nondominated_mask(F)]
    if F.shape[0] < 2:
        return 0.0
    return float(np.sqrt(np.sum((F.max(axis=0) - F.min(axis=0)) ** 2)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    uniq, inv = np.unique(x, return_inverse=True)
    if uniq.size < x.size:
        sums = np.bincount(inv, weights=ranks, minlength=uniq.size)
        counts = np.bincount(inv, minlength=uniq.size)
        ranks = (sums / counts)[inv]
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Returns 0.0 when either input is constant."""
    ra, rb = _rankdata(a), _rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def controllability(W: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Per-objective Spearman correlation between *requested* weight and *achieved* reward.

    **Higher is better; near zero means the preference knob does nothing.**

    This is the metric to check first. A policy that completely ignores its conditioning
    can still post a respectable hypervolume by parking on the knee of the frontier, and
    every other metric will look fine while the feature is silently broken.
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    return np.array([spearman(W[:, j], R[:, j]) for j in range(W.shape[1])])


def steerability(W: np.ndarray, R_obs: np.ndarray) -> np.ndarray:
    """Per-objective normalized regression slope of achieved reward on requested weight.

    **A finer replacement for :func:`controllability`, and the reason is measurement noise.**

    Spearman over ``D`` direction-*aggregates* has two problems that together make it unable
    to resolve the differences it is used to compare. It sees only ``D`` points -- 9 in these
    experiments -- so it is coarse and heavily quantized, and it returns exactly 0.0 whenever
    achieved rewards happen to be constant, which happened in 10 of 24 runs and turns a
    continuous quantity into a bimodal one that cannot be averaged meaningfully.

    This uses every ``(direction, problem)`` observation instead of the per-direction means,
    so ``D x P`` points rather than ``D``, and reports a slope rather than a rank statistic:

        slope_j = cov(w_j, r_j) / var(w_j),  divided by the observed range of r_j

    The normalization makes it comparable across objectives on different scales, and a slope
    degrades gracefully instead of saturating. Measured per-seed SD drops accordingly -- see
    ``results/ALIGNMENT_RESULTS.md`` for why that matters more than any single comparison.

    Args:
        W: ``(D, m)`` requested directions.
        R_obs: ``(D, P, m)`` achieved rewards, per direction and per problem.

    **Higher is better; near zero means the preference knob does nothing.**
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    R_obs = np.asarray(R_obs, dtype=float)
    if R_obs.ndim != 3:
        raise ValueError(f"expected (D, P, m) observations, got shape {R_obs.shape}")
    D, P, m = R_obs.shape
    if W.shape[0] != D or W.shape[1] != m:
        raise ValueError(f"W {W.shape} incompatible with observations {R_obs.shape}")

    out = np.zeros(m, dtype=float)
    for j in range(m):
        x = np.repeat(W[:, j], P)  # requested weight, one entry per observation
        y = R_obs[:, :, j].reshape(-1)  # achieved reward for that observation
        var = x.var()
        if var < 1e-12:
            continue
        slope = float(((x - x.mean()) * (y - y.mean())).mean() / var)
        span = float(y.max() - y.min())
        out[j] = slope / span if span > 1e-12 else 0.0
    return out


def frontier_summary(
    R: np.ndarray,
    ref: np.ndarray,
    W: np.ndarray | None = None,
    hv_samples: int = 100_000,
    observations: np.ndarray | None = None,
) -> dict[str, float]:
    """Bundle the frontier metrics for a set of achieved reward vectors."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    out = {
        "hypervolume": hypervolume(R, ref, n_samples=hv_samples),
        "spacing": spacing(R),
        "max_spread": max_spread(R),
        "n_nondominated": float(nondominated_mask(R).sum()),
    }
    if W is not None:
        ctrl = controllability(W, R)
        for j, c in enumerate(ctrl):
            out[f"controllability_obj{j}"] = float(c)
        out["controllability_mean"] = float(ctrl.mean())
        if observations is not None:
            steer = steerability(W, observations)
            for j, s in enumerate(steer):
                out[f"steerability_obj{j}"] = float(s)
            out["steerability_mean"] = float(np.abs(steer).mean())
    return out


class FrontierArchive:
    """Accumulates ``(direction, achieved reward)`` pairs from evaluation rollouts."""

    def __init__(self) -> None:
        self._w: list[np.ndarray] = []
        self._r: list[np.ndarray] = []

    def add(self, w: np.ndarray, r: np.ndarray) -> None:
        self._w.append(np.asarray(w, dtype=float))
        self._r.append(np.asarray(r, dtype=float))

    def __len__(self) -> int:
        return len(self._r)

    @property
    def directions(self) -> np.ndarray:
        return np.atleast_2d(np.asarray(self._w, dtype=float))

    @property
    def rewards(self) -> np.ndarray:
        return np.atleast_2d(np.asarray(self._r, dtype=float))

    def summary(self, ref: np.ndarray, **kwargs) -> dict[str, float]:
        return frontier_summary(self.rewards, ref, self.directions, **kwargs)
