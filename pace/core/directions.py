"""Preference directions on the simplex, and the coverage bandit that samples them.

A "direction" is a weight vector ``w`` on the probability simplex describing which
trade-off a rollout is being asked to make. PaCE keeps a fixed reference grid of
directions and learns *where to spend rollouts*: frontier regions that are weak, still
improving, or under-explored get more budget than regions that have already converged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

__all__ = ["das_dennis", "das_dennis_at_most", "CoverageBandit", "UniformDirections", "BanditConfig"]


def das_dennis(m: int, n_partitions: int) -> np.ndarray:
    """Das-Dennis uniform reference directions on the ``(m-1)``-simplex.

    Enumerates every vector of non-negative integers summing to ``n_partitions`` and
    divides by ``n_partitions``. For ``m = 2, n = 8`` this gives 9 evenly spaced
    directions; for ``m = 3, n = 6`` it gives 28.
    """
    if m < 2:
        raise ValueError("need at least 2 objectives")
    if n_partitions < 1:
        raise ValueError("n_partitions must be >= 1")
    rows = []
    # Compositions of n_partitions into m non-negative parts, via stars and bars.
    for bars in combinations(range(n_partitions + m - 1), m - 1):
        prev = -1
        parts = []
        for b in bars:
            parts.append(b - prev - 1)
            prev = b
        parts.append(n_partitions + m - 2 - prev)
        rows.append(parts)
    return np.asarray(rows, dtype=float) / float(n_partitions)


def das_dennis_at_most(m: int, max_points: int) -> np.ndarray:
    """Densest Das-Dennis grid on the ``(m-1)``-simplex with at most ``max_points`` points.

    Grid size grows as ``C(n + m - 1, m - 1)``, so a resolution chosen for two objectives
    explodes at three: ``das_dennis(2, 8)`` is 9 directions but ``das_dennis(3, 8)`` is 45
    and ``das_dennis(4, 8)`` is 165. Anything that means "give me about N directions" must
    solve for the resolution rather than pass N straight through, or it silently requests
    an order of magnitude more work as soon as an objective is added.
    """
    if max_points < m:
        raise ValueError(f"need at least {m} points to cover {m} objectives")
    best = das_dennis(m, 1)
    for n in range(1, 200):
        grid = das_dennis(m, n)
        if grid.shape[0] > max_points:
            break
        best = grid
    return best


@dataclass
class BanditConfig:
    """Knobs for :class:`CoverageBandit`."""

    weakness: float = 1.0
    """Weight on ``1 - achievement``: prioritize frontier regions that score poorly."""

    improvement: float = 1.0
    """Weight on recent |change| in achievement: prioritize regions that are still
    moving. Without this the bandit pours budget into regions that are weak because they
    are *infeasible* rather than because they are neglected."""

    ucb_c: float = 0.5
    """UCB exploration coefficient on ``sqrt(log N / n_d)``."""

    temperature: float = 0.5
    """Softmax temperature over scores. Higher is closer to uniform."""

    epsilon: float = 0.15
    """Floor of uniform mass mixed into the sampling distribution. Guarantees no
    direction is ever starved, which is what stops the frontier quietly losing an arm."""

    ema: float = 0.85
    """Decay for the achievement / improvement running estimates."""

    jitter: float | None = 50.0
    """If set, the sampled grid direction is perturbed by a Dirichlet with concentration
    ``jitter * w``, so the policy sees a continuum of directions rather than ``D`` discrete
    ones. ``None`` disables. Higher values jitter less."""

    unique_per_group: bool = True
    """Sample the ``K`` directions of a group without replacement, maximizing within-group
    trade-off diversity (this is what makes the cross-direction matrix informative)."""


@dataclass
class CoverageBandit:
    """Non-stationary bandit over preference directions, driven by frontier coverage.

    The arms are reference directions. The "reward" being tracked is the achievement the
    policy attains when asked to serve that direction -- but unlike a normal bandit we
    sample arms where achievement is *low* or *changing*, because the goal is to cover the
    frontier, not to exploit its best point.
    """

    directions: np.ndarray
    config: BanditConfig = field(default_factory=BanditConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        self.directions = np.atleast_2d(np.asarray(self.directions, dtype=float))
        D = self.directions.shape[0]
        self.counts = np.zeros(D, dtype=float)
        self.achievement = np.zeros(D, dtype=float)
        self.improvement = np.zeros(D, dtype=float)
        self.seen = np.zeros(D, dtype=bool)
        self.total = 0.0
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_directions(self) -> int:
        return self.directions.shape[0]

    @property
    def n_objectives(self) -> int:
        return self.directions.shape[1]

    def _norm01(self, x: np.ndarray) -> np.ndarray:
        if not self.seen.any():
            return np.full_like(x, 0.5)
        vals = x[self.seen]
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.full_like(x, 0.5)
        out = (x - lo) / (hi - lo)
        # Unvisited arms are assumed mid-pack rather than optimistic or pessimistic; the
        # UCB term is what gets them sampled, and it does so on its own terms.
        out[~self.seen] = 0.5
        return np.clip(out, 0.0, 1.0)

    def scores(self) -> np.ndarray:
        cfg = self.config
        ach = self._norm01(self.achievement)
        imp = self._norm01(self.improvement)
        # Unvisited arms get a large but finite exploration bonus (count floored at 0.5)
        # rather than an infinite one, so the softmax stays well defined.
        n_eff = np.maximum(self.counts, 0.5)
        ucb = np.sqrt(np.log(max(self.total, 1.0) + 1.0) / n_eff)
        return cfg.weakness * (1.0 - ach) + cfg.improvement * imp + cfg.ucb_c * ucb

    def probabilities(self) -> np.ndarray:
        cfg = self.config
        s = self.scores() / max(cfg.temperature, 1e-8)
        s = s - s.max()
        p = np.exp(s)
        p /= p.sum()
        D = self.n_directions
        return (1.0 - cfg.epsilon) * p + cfg.epsilon / D

    def sample(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``k`` directions for one group.

        Returns ``(idx, W)`` where ``idx`` are grid indices (for :meth:`update`) and ``W``
        is ``(k, m)``, jittered off the grid if configured.
        """
        p = self.probabilities()
        D = self.n_directions
        replace = not (self.config.unique_per_group and k <= D)
        idx = self._rng.choice(D, size=k, replace=replace, p=p)
        W = self.directions[idx].copy()

        if self.config.jitter is not None:
            alpha = self.config.jitter * W + 1e-3
            W = np.stack([self._rng.dirichlet(a) for a in alpha])
        return idx, W

    def update(self, idx: np.ndarray, achievements: np.ndarray) -> None:
        """Fold the achievement attained at each sampled direction into the estimates."""
        cfg = self.config
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        achievements = np.atleast_1d(np.asarray(achievements, dtype=float))
        for d, a in zip(idx, achievements):
            if self.seen[d]:
                delta = a - self.achievement[d]
                self.achievement[d] = cfg.ema * self.achievement[d] + (1 - cfg.ema) * a
                self.improvement[d] = cfg.ema * self.improvement[d] + (1 - cfg.ema) * abs(delta)
            else:
                self.achievement[d] = a
                self.seen[d] = True
            self.counts[d] += 1.0
            self.total += 1.0


class UniformDirections(CoverageBandit):
    """Ablation: ignore coverage signal, sample the grid uniformly."""

    def probabilities(self) -> np.ndarray:  # noqa: D102
        return np.full(self.n_directions, 1.0 / self.n_directions)
