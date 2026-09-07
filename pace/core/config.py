"""Configuration objects for the PaCE advantage estimator and trainer."""

from __future__ import annotations

from dataclasses import dataclass, field

from .directions import BanditConfig

__all__ = ["PaCEConfig"]


@dataclass
class PaCEConfig:
    """Hyperparameters of the PaCE advantage.

    Defaults are the ones used in the synthetic experiments; the two that actually matter
    in practice are ``lambda_front`` (how hard to push for frontier spread) and ``mu``
    (how sharp the Tchebycheff max is).
    """

    scalarization: str = "smooth_tchebycheff"
    """One of ``smooth_tchebycheff`` (default), ``tchebycheff``, ``linear``. ``linear`` is
    the ablation that reproduces the concave-front failure mode."""

    mu: float = 0.1
    """Tchebycheff smoothing temperature. Small = closer to the exact max = sharper
    trade-off targeting but higher gradient variance."""

    rho: float = 0.05
    """Augmentation weight; rules out weakly-Pareto solutions."""

    rank_blend: float = 0.0
    """0 = pure rank normalization (monotone-invariant), 1 = pure min-max."""

    lambda_front: float = 0.3
    """Mixing weight on the frontier-shaping term (dominance rank + crowding). Set to 0
    to recover pure conditional Tchebycheff RL."""

    lambda_crowd: float = 1.0
    """Weight on crowding relative to dominance rank inside the shaping term."""

    loo_shrinkage: float = 0.0
    """Interpolates the baseline between the cross-direction leave-one-out mean (0) and a
    single group-constant baseline (1). Raise it if conditional policies diverge enough
    that off-direction rollouts stop being informative."""

    advantage_clip: float | None = 5.0
    """Symmetric clip on the final advantage, in units of group std. ``None`` disables."""

    eps: float = 1e-8

    bandit: BanditConfig = field(default_factory=BanditConfig)
    n_partitions: int = 8
    """Das-Dennis resolution for the direction grid."""
