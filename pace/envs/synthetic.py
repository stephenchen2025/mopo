"""A synthetic multi-objective contextual bandit with a Pareto front of tunable curvature.

The point of this environment is to make the *geometry* controllable. Set ``curvature < 1``
and the Pareto front is concave -- the regime where linear scalarization provably cannot
reach the interior of the front no matter how the weights are chosen. Set
``curvature > 1`` and the front is convex, where weighted sums are perfectly adequate.
Being able to flip between the two is what turns "Tchebycheff should help" into a
measurement.

This validates the *optimizer*, not any language model. It is deliberately tiny and
torch-free so the full ablation grid runs on a CPU in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.directions import das_dennis

__all__ = ["SyntheticFrontier", "TRANSFORMS"]


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _exp4(x: np.ndarray) -> np.ndarray:
    return np.exp(4.0 * x)


def _pow5(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, None) ** 5


TRANSFORMS = {"identity": _identity, "exp4": _exp4, "pow5": _pow5}
"""Strictly increasing maps, referenced by name rather than passed as callables.

Named functions rather than lambdas so the environment stays picklable: the experiment
runner parallelizes seeds across processes, and a lambda in a dataclass field breaks that
with an error that points nowhere near the cause."""


@dataclass
class SyntheticFrontier:
    """Contextual bandit whose actions trace out a parametric Pareto front.

    Each action ``a`` maps to a point ``t_a`` on the simplex (a "behaviour mix"), and the
    reward for objective ``j`` is ``1 - (1 - t_j)^curvature``, scaled per context.

    Args:
        n_objectives: number of objectives ``m``.
        curvature: exponent ``p``. ``p < 1`` gives a **concave** (non-convex) front,
            ``p = 1`` a linear one, ``p > 1`` a convex front.
        action_resolution: Das-Dennis resolution of the action grid. For ``m = 2`` this
            gives ``action_resolution + 1`` actions.
        n_contexts: number of distinct prompts; each has its own reward scaling, so the
            policy has to use the context and not just the direction.
        noise: std of the Gaussian noise added to observed rewards. The *true* front is
            noiseless; noise only affects what the learner sees.
        objective_transforms: optional per-objective strictly-increasing maps applied to
            the observed reward. Used to test scale invariance: these do not move the
            Pareto set at all, so an ordering-based method must be unaffected by them
            while a z-score-based one is not.
    """

    n_objectives: int = 2
    curvature: float = 0.5
    action_resolution: int = 20
    n_contexts: int = 4
    noise: float = 0.05
    noise_skew: float = 0.0
    """Makes the frontier unevenly hard to learn. Observation noise for an action is
    scaled by ``1 + noise_skew * t_0``, so behaviours at one end of the trade-off need
    many more rollouts to resolve than the other. A uniform direction sampler spends the
    same budget everywhere and under-serves the noisy end; this is the condition a
    coverage bandit is supposed to exploit, so it is the condition it should be tested
    under."""

    objective_transforms: tuple | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_objectives < 2:
            raise ValueError("need at least 2 objectives")
        self.action_points = das_dennis(self.n_objectives, self.action_resolution)
        rng = np.random.default_rng(self.seed)
        # Per-context scaling in [0.75, 1.0] per objective: contexts differ in which
        # objective is cheap, so the optimal action for a given direction is context
        # dependent and the policy cannot succeed by ignoring the context.
        self.context_scale = rng.uniform(0.75, 1.0, size=(self.n_contexts, self.n_objectives))

    @property
    def n_actions(self) -> int:
        return self.action_points.shape[0]

    def true_rewards(self, context: int) -> np.ndarray:
        """Noiseless expected reward vector for every action. ``(n_actions, m)``."""
        base = 1.0 - np.power(np.clip(1.0 - self.action_points, 0.0, 1.0), self.curvature)
        return base * self.context_scale[context]

    def observe(self, context: int, actions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample noisy observed rewards for the given actions. ``(len(actions), m)``."""
        actions = np.atleast_1d(np.asarray(actions, dtype=int))
        R = self.true_rewards(context)[actions]
        if self.noise > 0:
            scale = self.noise
            if self.noise_skew:
                scale = self.noise * (1.0 + self.noise_skew * self.action_points[actions, 0])
                scale = scale[:, None]
            R = R + rng.normal(0.0, 1.0, size=R.shape) * scale
        if self.objective_transforms is not None:
            R = R.copy()
            for j, name in enumerate(self.objective_transforms):
                if name is not None:
                    R[:, j] = TRANSFORMS[name](R[:, j])
        return R

    def average_action_front(self) -> np.ndarray:
        """Context-averaged reward of each action, if the *same* action is used everywhere.

        A lower bound on what is attainable, not the oracle: a context-conditioned policy
        can pick a different action per context and beat this curve. Use
        :meth:`oracle_frontier` for the real ceiling.
        """
        return np.mean([self.true_rewards(c) for c in range(self.n_contexts)], axis=0)

    def oracle_frontier(self) -> np.ndarray:
        """Exact Pareto front over all deterministic context-to-action assignments.

        A deterministic policy picks one action per context and is scored on the
        context-average, so the achievable set is the Minkowski average of the per-context
        reward sets. Enumerating that is ``n_actions ** n_contexts`` points, but the front
        of a Minkowski sum only ever uses points from the summands' own fronts, so
        accumulating context by context and filtering as we go gives the same answer
        exactly, in milliseconds.

        This is the ceiling for *deterministic* behaviour. A stochastic policy can reach
        the convex hull of these points, which on a concave front means mixing the two
        extremes. That mixture posts a good average and is a bad product: half the
        responses maximally long, half maximally terse, none of them what was asked for.
        This is exactly why the evaluation decodes greedily and why the training objective
        scores individual samples rather than the group mean.
        """
        from ..core.pareto import nondominated_mask

        acc = self.true_rewards(0)
        acc = acc[nondominated_mask(acc)]
        for c in range(1, self.n_contexts):
            other = self.true_rewards(c)
            other = other[nondominated_mask(other)]
            summed = (acc[:, None, :] + other[None, :, :]).reshape(-1, self.n_objectives)
            acc = summed[nondominated_mask(summed)]
        return acc / self.n_contexts

    def reference_point(self, margin: float = 0.05) -> np.ndarray:
        """Fixed hypervolume reference point, shared across every compared run.

        Set slightly *below* the worst attainable reward rather than at the origin. At the
        origin a policy that only ever emits the two corner behaviours scores exactly zero
        hypervolume -- arguably correct, but it collapses every degenerate method to the
        same number and destroys the ability to rank them. The margin keeps the ordering
        while restoring discrimination.
        """
        return np.full(self.n_objectives, -margin)
