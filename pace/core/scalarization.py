"""Scalarization functions mapping a reward vector plus a preference direction to a scalar.

The choice here is not cosmetic. A weighted sum can only ever select points that lie on a
supporting hyperplane of the attainable set, so it is *provably* unable to reach the
interior of a concave Pareto front no matter how the weights are tuned. Tchebycheff
scalarization has no such restriction: every Pareto-optimal point maximizes it for some
weight. PaCE defaults to the smooth (log-sum-exp) Tchebycheff form so the gradient is
well-behaved.

All functions here follow the package-wide **maximization** convention.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rank_normalize",
    "linear",
    "tchebycheff",
    "smooth_tchebycheff",
    "achievement_matrix",
]


def rank_normalize(R: np.ndarray, blend: float = 0.0) -> np.ndarray:
    """Map each objective column of ``R`` to within-group normalized ranks in ``[0, 1]``.

    Args:
        R: ``(K, m)`` raw reward vectors for one group.
        blend: ``0.0`` is pure rank (invariant to any strictly increasing rescaling of an
            objective); ``1.0`` is pure min-max (keeps magnitude, only affine-invariant).
            Values in between interpolate. Rank normalization is the reason PaCE is
            robust to objectives with wildly different scales or heavy tails -- a
            per-objective z-score only survives *affine* rescaling.

    Ties get the average rank, so a group where an objective is constant maps to all
    ``0.5`` and contributes nothing to the achievement differences. That is the correct
    behaviour: a constant objective carries no signal in this group.
    """
    R = np.atleast_2d(np.asarray(R, dtype=float))
    K, m = R.shape
    if K == 1:
        return np.full((1, m), 0.5)

    out = np.empty((K, m), dtype=float)
    for j in range(m):
        col = R[:, j]
        order = np.argsort(col, kind="stable")
        ranks = np.empty(K, dtype=float)
        ranks[order] = np.arange(K, dtype=float)
        # Average the ranks of tied values so ties are not broken by sample order.
        uniq, inv = np.unique(col, return_inverse=True)
        if uniq.size < K:
            sums = np.bincount(inv, weights=ranks, minlength=uniq.size)
            counts = np.bincount(inv, minlength=uniq.size)
            ranks = (sums / counts)[inv]
        out[:, j] = ranks / (K - 1)

    if blend > 0.0:
        lo, hi = R.min(axis=0), R.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        minmax = np.where(hi > lo, (R - lo) / span, 0.5)
        out = (1.0 - blend) * out + blend * minmax
    return out


def linear(U: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted-sum scalarization. ``(K, m) x (D, m) -> (K, D)``.

    Present as a baseline and an ablation, not as a default: see the module docstring for
    why this cannot reach concave frontier regions.
    """
    U = np.atleast_2d(U)
    W = np.atleast_2d(W)
    return U @ W.T


def tchebycheff(U: np.ndarray, W: np.ndarray, ideal: float | np.ndarray = 1.0) -> np.ndarray:
    """Exact weighted Tchebycheff achievement (to maximize). ``(K, m) x (D, m) -> (K, D)``.

    ``s(u; w) = -max_j w_j (z*_j - u_j)``, with the ideal point ``z*`` defaulting to 1
    (the top of the rank-normalized range).
    """
    U = np.atleast_2d(U)
    W = np.atleast_2d(W)
    gap = np.asarray(ideal, dtype=float) - U  # (K, m)
    weighted = W[None, :, :] * gap[:, None, :]  # (K, D, m)
    return -np.max(weighted, axis=-1)


def smooth_tchebycheff(
    U: np.ndarray,
    W: np.ndarray,
    mu: float = 0.1,
    rho: float = 0.05,
    ideal: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Smooth (log-sum-exp) Tchebycheff achievement with an augmentation term.

    ``s_mu(u; w) = -mu * log sum_j exp( w_j (z*_j - u_j) / mu )  +  rho * sum_j w_j u_j``

    Args:
        mu: smoothing temperature. ``mu -> 0`` recovers the exact ``max``; larger values
            give a softer, better-conditioned surrogate. The smooth form is what makes
            this usable as an RL reward -- the exact ``max`` puts all the gradient on a
            single objective per sample, which is very high variance.
        rho: augmentation weight. A small positive value rules out weakly-Pareto points
            (points that tie on one objective and lose on another).

    Shapes: ``(K, m) x (D, m) -> (K, D)``.
    """
    if mu <= 0:
        return tchebycheff(U, W, ideal=ideal) + rho * linear(U, W)
    U = np.atleast_2d(U)
    W = np.atleast_2d(W)
    gap = np.asarray(ideal, dtype=float) - U  # (K, m)
    a = W[None, :, :] * gap[:, None, :] / mu  # (K, D, m)
    a_max = np.max(a, axis=-1, keepdims=True)
    lse = a_max + np.log(np.sum(np.exp(a - a_max), axis=-1, keepdims=True))
    return -mu * lse.squeeze(-1) + rho * linear(U, W)


def achievement_matrix(
    U: np.ndarray,
    W: np.ndarray,
    kind: str = "smooth_tchebycheff",
    mu: float = 0.1,
    rho: float = 0.05,
) -> np.ndarray:
    """Build the ``(K, D)`` matrix ``S[k, l] = s(u_k; w_l)``.

    This is the object the cross-direction advantage is built from: because scalarizing an
    already-computed reward *vector* is free, every rollout can be scored under every
    direction for ``O(K * D * m)`` flops and no extra sampling.
    """
    if kind == "linear":
        return linear(U, W)
    if kind == "tchebycheff":
        return tchebycheff(U, W)
    if kind == "smooth_tchebycheff":
        return smooth_tchebycheff(U, W, mu=mu, rho=rho)
    raise ValueError(f"unknown scalarization: {kind!r}")


def direction_scale(W: np.ndarray) -> np.ndarray:
    """Per-direction scale of the Tchebycheff achievement: ``max_j w_j``. ``(D,) -> (D,)``.

    Achievement is **not comparable across directions** without this. ``s(u; w)`` ranges
    over ``[-max_j w_j, 0]``, so an extreme direction like ``(1, 0)`` spans ``[-1, 0]``
    while a balanced one like ``(0.5, 0.5)`` spans only ``[-0.5, 0]``. Anything that
    compares achievements *between* directions -- the coverage bandit above all -- will
    otherwise conclude that extreme directions are permanently underperforming, and pour
    budget into them forever. Dividing by this puts every direction on ``[-1, 0]``.
    """
    W = np.atleast_2d(np.asarray(W, dtype=float))
    return np.maximum(W.max(axis=1), 1e-8)
