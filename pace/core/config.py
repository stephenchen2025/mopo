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

    lambda_align: float = 0.0
    """Weight on the direction-alignment term: the diagonal of the double-centered
    achievement matrix.

    This targets a failure the other terms do not touch. Measured in
    ``results/HEADROOM_RESULTS.md``, RL more than doubled hypervolume while every method
    became *less* steerable than it started -- the policy converged on one good behaviour and
    emitted it at every direction, and hypervolume rewarded that, because one dominant point
    beats a spread of weaker ones. The crowding term is supposed to prevent this, but it acts
    within a rollout group and in raw reward space, so it does not survive to the deployed
    policy.

    Collapse is a *row* phenomenon in the ``K x K`` matrix: when every rollout behaves the
    same, every row of ``S`` is identical. Double-centering removes the "this rollout is
    generally good" and "this direction is generally easy" main effects, leaving only whether
    the pairing (rollout k, direction k) is *specifically* good. A collapsed group gives
    exactly zero, so collapse earns nothing; a direction-matched group gives a positive
    diagonal, so matching is what gets reinforced.

    Defaults to 0.0 so previously published results reproduce unchanged. See
    ``results/ALIGNMENT_RESULTS.md`` for whether it works.
    """

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
