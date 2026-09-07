"""The PaCE advantage estimator, plus the baselines it is compared against.

The input to every function here is a group of ``K`` rollouts for one prompt: their reward
*vectors* ``R`` of shape ``(K, m)`` and the preference directions ``W`` of shape ``(K, m)``
they were generated under. The output is one scalar advantage per rollout, ready to
multiply the usual clipped policy-gradient surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PaCEConfig
from .pareto import nondominated_sort, normalized_crowding
from .scalarization import achievement_matrix, rank_normalize

__all__ = [
    "AdvantageOutput",
    "pace_advantages",
    "linear_scalar_advantages",
    "mo_grpo_advantages",
    "dominance_advantages",
]


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std < eps:
        return np.zeros_like(x)
    return (x - x.mean()) / (std + eps)


@dataclass
class AdvantageOutput:
    """Advantages plus the intermediates worth logging.

    Keeping ``S`` and the two components around is not decoration: when a multi-objective
    run misbehaves, the question is almost always "is it the achievement term or the
    spread term", and you cannot answer that from the final scalar.
    """

    advantages: np.ndarray  # (K,)
    achievement: np.ndarray  # (K,) diagonal of S -- own-direction achievement
    achievement_advantage: np.ndarray  # (K,) cross-direction LOO advantage
    front_advantage: np.ndarray  # (K,) dominance + crowding shaping
    fronts: np.ndarray  # (K,) 1-based non-dominated front index
    crowding: np.ndarray  # (K,) normalized crowding distance
    utilities: np.ndarray  # (K, m) rank-normalized rewards
    matrix: np.ndarray  # (K, K) S[k, l] = achievement of rollout k under direction l


def pace_advantages(
    R: np.ndarray,
    W: np.ndarray,
    config: PaCEConfig | None = None,
) -> AdvantageOutput:
    """Compute PaCE advantages for one group.

    Args:
        R: ``(K, m)`` reward vectors (maximization).
        W: ``(K, m)`` preference direction each rollout was generated under. Rows should
            lie on the simplex.

    The four steps, in order:

    1. **Rank-normalize** rewards within the group, per objective (monotone-invariant).
    2. Build the ``(K, K)`` **cross-direction achievement matrix** ``S[k, l]``: what
       rollout ``k`` scores under direction ``l``. Free, because rewards are vectors.
    3. **Leave-one-out advantage**: rollout ``k``'s own achievement ``S[k, k]`` minus the
       mean of the *other* rollouts scored under the *same* direction ``w_k``. The
       baseline never touches ``y_k``, so it is a valid control variate and the estimator
       stays unbiased for ``E[s(u; w_k)]``.
    4. **Frontier shaping**: reward being non-dominated within the group, and being in a
       sparse region of that front.
    """
    config = config or PaCEConfig()
    R = np.atleast_2d(np.asarray(R, dtype=float))
    W = np.atleast_2d(np.asarray(W, dtype=float))
    K, m = R.shape
    if W.shape != R.shape:
        raise ValueError(f"W {W.shape} must match R {R.shape}: one direction per rollout")

    U = rank_normalize(R, blend=config.rank_blend)

    if K == 1:
        # A group of one has no relative signal at all; returning zeros is the honest
        # answer, not a degenerate baseline.
        zero = np.zeros(1)
        return AdvantageOutput(
            advantages=zero,
            achievement=zero.copy(),
            achievement_advantage=zero.copy(),
            front_advantage=zero.copy(),
            fronts=np.ones(1, dtype=int),
            crowding=np.ones(1),
            utilities=U,
            matrix=np.zeros((1, 1)),
        )

    S = achievement_matrix(U, W, kind=config.scalarization, mu=config.mu, rho=config.rho)
    own = np.diag(S).copy()  # S[k, k]

    # Column k holds every rollout's achievement under direction w_k. The leave-one-out
    # mean over l != k is independent of rollout k's own sample.
    loo = (S.sum(axis=0) - own) / (K - 1)
    a_ach = own - loo

    if config.loo_shrinkage > 0.0:
        constant = own - own.mean()
        a_ach = (1.0 - config.loo_shrinkage) * a_ach + config.loo_shrinkage * constant

    fronts = nondominated_sort(U)
    crowd = normalized_crowding(U, fronts)

    # Spread pressure applies *within the first front only*. NSGA-II uses crowding solely
    # to break ties inside a front, and here the terms are additive, so a lone point in
    # front 3 would otherwise collect the singleton's maximal crowding score and compete
    # against genuinely non-dominated points. Dominated rollouts get the front-1 mean:
    # neutral on this term, already penalized by the rank term.
    on_front = fronts == 1
    crowd_eff = crowd.copy()
    if on_front.any() and not on_front.all():
        crowd_eff[~on_front] = crowd[on_front].mean()

    a_front = _zscore(-fronts.astype(float), config.eps) + config.lambda_crowd * _zscore(crowd_eff, config.eps)

    adv = _zscore(a_ach, config.eps) + config.lambda_front * a_front
    if config.advantage_clip is not None:
        adv = np.clip(adv, -config.advantage_clip, config.advantage_clip)

    return AdvantageOutput(
        advantages=adv,
        achievement=own,
        achievement_advantage=a_ach,
        front_advantage=a_front,
        fronts=fronts,
        crowding=crowd,
        utilities=U,
        matrix=S,
    )


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


def linear_scalar_advantages(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Standard group-relative advantage on a weighted sum of raw rewards.

    This is what GRPO does today with a multi-objective reward: scalarize, then z-score
    within the group. It inherits both failure modes -- the highest-variance objective
    dominates the sum, and concave frontier regions are unreachable.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    W = np.atleast_2d(np.asarray(W, dtype=float))
    return _zscore(np.sum(R * W, axis=1))


def mo_grpo_advantages(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """MO-GRPO-style baseline: z-score each objective *first*, then take the weighted sum.

    This fixes the variance-mismatch half of the problem (and is a real improvement over
    :func:`linear_scalar_advantages`), but the aggregation is still linear, so concave
    frontier regions remain unreachable.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    W = np.atleast_2d(np.asarray(W, dtype=float))
    Z = np.stack([_zscore(R[:, j]) for j in range(R.shape[1])], axis=1)
    return _zscore(np.sum(Z * W, axis=1))


def dominance_advantages(R: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    """Pareto-rank-only advantage (PRPO-flavoured ablation).

    Advantage is the z-scored negative non-dominated front index. Scale-free and
    direction-agnostic, but very coarse: at ``K = 8`` most rollouts land in front 1 and
    the signal saturates. Included to show why PaCE uses dominance as *shaping* on top of
    a continuous achievement term rather than as the whole advantage.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    return _zscore(-nondominated_sort(R).astype(float))
