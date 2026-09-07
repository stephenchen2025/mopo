"""Running per-objective reward normalizer.

The PaCE *advantage* deliberately uses within-group ranks, which are relative by
construction: a group's achievement values say who beat whom, not how good the group was.
That is the right signal for a policy gradient and the wrong signal for the coverage
bandit, which needs to know which frontier regions are *absolutely* weak. Feeding the
bandit group-relative achievements makes its "weakness" term nearly constant across
directions -- it then chases noise, which in practice means over-sampling whichever
directions the noise favours. This normalizer supplies the absolute scale the bandit needs
without contaminating the advantage.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RunningMinMax"]


class RunningMinMax:
    """Tracks a per-objective ``[lo, hi]`` range with fast expansion and slow contraction.

    Expanding immediately when a reward falls outside the known range, but contracting only
    via an EMA, keeps the normalizer from being yanked around by a single outlier while
    still tracking a genuinely shifting reward scale during training.
    """

    def __init__(self, n_objectives: int, momentum: float = 0.99) -> None:
        self.n_objectives = n_objectives
        self.momentum = momentum
        self.lo = np.full(n_objectives, np.inf)
        self.hi = np.full(n_objectives, -np.inf)
        self.initialized = False

    def update(self, R: np.ndarray) -> None:
        R = np.atleast_2d(np.asarray(R, dtype=float))
        batch_lo, batch_hi = R.min(axis=0), R.max(axis=0)
        if not self.initialized:
            self.lo, self.hi = batch_lo.copy(), batch_hi.copy()
            self.initialized = True
            return
        m = self.momentum
        self.lo = np.where(batch_lo < self.lo, batch_lo, m * self.lo + (1 - m) * batch_lo)
        self.hi = np.where(batch_hi > self.hi, batch_hi, m * self.hi + (1 - m) * batch_hi)

    def normalize(self, R: np.ndarray) -> np.ndarray:
        """Map rewards into ``[0, 1]`` using the tracked range."""
        R = np.atleast_2d(np.asarray(R, dtype=float))
        if not self.initialized:
            return np.full_like(R, 0.5)
        span = self.hi - self.lo
        span = np.where(span > 1e-12, span, 1.0)
        return np.clip((R - self.lo) / span, 0.0, 1.0)
