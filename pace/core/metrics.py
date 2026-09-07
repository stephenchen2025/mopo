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


def frontier_summary(
    R: np.ndarray,
    ref: np.ndarray,
    W: np.ndarray | None = None,
    hv_samples: int = 100_000,
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
