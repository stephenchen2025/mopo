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
    """Per-objective Pearson correlation between requested weight and per-direction mean
    reward, computed on low-noise means rather than raw observations.

    **This replaced Spearman-over-aggregates because that saturates and quantizes.** It also
    replaced two of its own predecessors that were tried and discarded here, because both had
    a real pathology, not just a cosmetic one -- worth recording so the mistake is not
    repeated:

    1. *Slope of raw observations, normalized by the raw observation range.* This dilutes
       any high-variance objective. A 0/1 accuracy check has huge per-problem variance
       unrelated to steerability, so normalizing by its raw range understates the effect:
       simulated with an identical injected direction-dependent shift, a Bernoulli-like
       reward reported -0.04 against +0.33 for a continuous one -- an 8x spurious gap from
       noise character alone.
    2. *Slope normalized by the range of per-direction means, or by a variance-decomposition
       debiased estimate of it.* This fixes the dilution but creates the opposite failure:
       under the null (no true effect), the sampled between-direction spread is itself a
       noisy quantity that can be small by chance, and dividing by a small denominator
       explodes. Simulated repeatedly under a genuine null with D=9: this version returned
       values up to 9.28, i.e. not bounded at all.

    Pearson correlation on the means avoids both. It is bounded to ``[-1, 1]`` by
    construction, so it cannot explode under the null the way a slope-over-noisy-spread can.
    It is not diluted by per-observation noise, because the correlation is computed on
    per-direction *means*, each already averaged over every problem at that direction, so
    increasing the problem count per direction directly sharpens the statistic.

    Two things this does **not** fix, stated plainly because an earlier version of this
    docstring overstated the case against Spearman: at realistic noise levels the two
    correlate similarly rather than one saturating well before the other, and Spearman's
    average-rank tie handling means quantized rewards rarely force it to exactly 0.0 purely
    from ties (measured: 4/500 simulated trials, despite 479/500 having at least one tied
    mean) -- so the exactly-0.000 values seen in real runs were most likely genuine policy
    collapse, the correct reading, which Pearson reports identically. Neither statistic is
    to blame for that; a constant array has no correlation with anything, in any flavor.

    **A more fundamental limit this exposed, not fixed by any normalization choice:** at
    ``D = 9`` directions, a correlation coefficient's own sampling distribution has a
    standard error around 0.35-0.40 (verified by simulating a true null 200 times), and this
    holds for *any* correlation-flavored statistic computed over 9 points -- Pearson,
    Spearman, or otherwise. More evaluation *problems* per direction sharpens each of the 9
    means but does not touch this; only more evaluation *directions* does. Where earlier
    documents in this repo attributed measurement noise to problem count and recommended
    more evaluation problems, that recommendation should be read as addressing only the
    per-mean noise, not this coarser limit.

    Args:
        W: ``(D, m)`` requested directions.
        R_obs: achieved rewards, either ``(D, P, m)`` per-problem observations (averaged over
            ``P`` here) or already-aggregated ``(D, m)`` per-direction means.

    **Higher is better; near zero means the preference knob does nothing.**
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    R_obs = np.asarray(R_obs, dtype=float)
    if R_obs.ndim == 3:
        means = R_obs.mean(axis=1)  # (D, m): average out the per-problem noise first
    elif R_obs.ndim == 2:
        means = R_obs
    else:
        raise ValueError(f"expected (D, P, m) or (D, m) observations, got shape {R_obs.shape}")
    D, m = means.shape
    if W.shape[0] != D or W.shape[1] != m:
        raise ValueError(f"W {W.shape} incompatible with observations of shape {R_obs.shape}")

    out = np.zeros(m, dtype=float)
    for j in range(m):
        x, y = W[:, j], means[:, j]
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue  # a constant direction weight or a genuinely flat response: no signal
        out[j] = float(np.corrcoef(x, y)[0, 1])
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
