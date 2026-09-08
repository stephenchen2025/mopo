"""Training loops for the synthetic environment, and the method registry compared in the
experiments.

Every method here sees the **same total rollout budget**, which is the only comparison
that means anything: PaCE's pitch is that one conditioned run covers a frontier that the
status quo needs N runs to sketch, so budget parity is the whole question.

The update is plain on-policy REINFORCE with the method's advantage. No importance ratios
and no clipping, because there is exactly one gradient step per batch of fresh samples --
adding PPO machinery would only obscure the thing being measured, which is the advantage
estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.advantages import (
    AdvantageOutput,
    dominance_advantages,
    linear_scalar_advantages,
    mo_grpo_advantages,
    pace_advantages,
)
from ..core.config import PaCEConfig
from ..core.directions import CoverageBandit, UniformDirections, das_dennis, das_dennis_at_most
from ..core.metrics import frontier_summary
from ..core.normalizer import RunningMinMax
from ..core.scalarization import achievement_matrix, direction_scale
from .policy import LinearSoftmaxPolicy
from .synthetic import SyntheticFrontier

__all__ = ["TrainConfig", "METHODS", "run_method", "evaluate", "total_rollouts"]


@dataclass
class TrainConfig:
    steps: int = 300
    prompts_per_step: int = 4
    group_size: int = 8
    lr: float = 1.0
    entropy_coef: float = 0.01
    n_partitions: int = 8
    """Das-Dennis resolution of the *training* direction grid."""
    eval_partitions: int = 20
    """Resolution of the evaluation grid (finer: conditioned policies can be queried at
    any direction, which is part of what they buy you)."""
    n_experts: int = 9
    """Number of independent policies for the ``fixed_scalar`` baseline."""
    seed: int = 0
    pace: PaCEConfig = field(default_factory=PaCEConfig)


def total_rollouts(cfg: TrainConfig) -> int:
    return cfg.steps * cfg.prompts_per_step * cfg.group_size


def evaluate(
    env: SyntheticFrontier,
    policy: LinearSoftmaxPolicy,
    dirs: np.ndarray,
    mode: str = "greedy",
) -> np.ndarray:
    """Noiseless reward vector per evaluation direction. ``(D, m)``.

    ``mode="greedy"`` (default) takes the policy's most likely action per context -- the
    analogue of greedy decoding, and the honest measure of *what the model actually does*
    when asked for trade-off ``w``.

    ``mode="expected"`` takes the expectation over the action distribution. It is reported
    for reference but must not be used as the headline metric: expectation over a
    stochastic policy convexifies the attainable set, so a policy that flips between the
    two extremes scores as though it had learned the interior of a concave front. It has
    not. Averaging away that distinction is exactly the mistake this environment exists to
    expose.
    """
    out = np.zeros((dirs.shape[0], env.n_objectives))
    for c in range(env.n_contexts):
        p = policy.probs(c, dirs)  # (D, A)
        if mode == "greedy":
            out += env.true_rewards(c)[p.argmax(axis=1)]
        elif mode == "expected":
            out += p @ env.true_rewards(c)
        else:
            raise ValueError(f"unknown eval mode: {mode!r}")
    return out / env.n_contexts


def _train_conditioned(
    env: SyntheticFrontier,
    cfg: TrainConfig,
    advantage_fn,
    sampler,
    heterogeneous: bool = True,
) -> tuple[LinearSoftmaxPolicy, list[dict]]:
    """Train one conditioned policy.

    Args:
        heterogeneous: if True, each of the ``K`` rollouts in a group gets its own
            direction (the PaCE setting). If False, one direction is drawn per group and
            shared by all ``K`` rollouts -- which is what conditioned GRPO does in
            practice, and the only setting in which a plain within-group z-score of a
            scalarized reward is a coherent advantage at all. Comparing a heterogeneous
            PaCE run against a heterogeneous z-score baseline would be beating up a
            strawman: that baseline compares each rollout's score under *its own*
            direction to other rollouts' scores under *theirs*, which is not a baseline.
    """
    policy = LinearSoftmaxPolicy(env.n_actions, env.n_contexts, env.n_objectives, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    history: list[dict] = []
    # Absolute reward scale for the coverage bandit only -- see core/normalizer.py for why
    # the group-relative achievement used by the advantage is the wrong signal here.
    norm = RunningMinMax(env.n_objectives)

    for step in range(cfg.steps):
        grad = np.zeros_like(policy.theta)
        ent_grad = np.zeros_like(policy.theta)
        for _ in range(cfg.prompts_per_step):
            ctx = int(rng.integers(env.n_contexts))
            if heterogeneous:
                idx, W = sampler.sample(cfg.group_size)
            else:
                idx1, w1 = sampler.sample(1)
                idx = np.repeat(idx1, cfg.group_size)
                W = np.repeat(w1, cfg.group_size, axis=0)
            actions, p, phi = policy.sample(ctx, W, rng)
            R = env.observe(ctx, actions, rng)

            result = advantage_fn(R, W)
            adv = result.advantages if isinstance(result, AdvantageOutput) else result

            norm.update(R)
            abs_achievement = np.diagonal(
                achievement_matrix(
                    norm.normalize(R),
                    W,
                    kind=cfg.pace.scalarization,
                    mu=cfg.pace.mu,
                    rho=cfg.pace.rho,
                )
            ) / direction_scale(W)
            sampler.update(idx, abs_achievement)

            grad += policy.policy_gradient(actions, p, phi, adv)
            ent_grad += policy.entropy_gradient(p, phi)

        policy.theta += cfg.lr * (grad + cfg.entropy_coef * ent_grad) / cfg.prompts_per_step

        if step % max(1, cfg.steps // 20) == 0 or step == cfg.steps - 1:
            history.append({"step": step, "direction_probs": sampler.probabilities().copy()})

    return policy, history


def _train_fixed_scalar(env: SyntheticFrontier, cfg: TrainConfig) -> tuple[list[LinearSoftmaxPolicy], np.ndarray]:
    """The status quo: N independent runs, each with its own fixed weight vector.

    Each expert gets ``1 / n_experts`` of the budget, so the ensemble as a whole spends
    exactly what a single PaCE run spends. This is also the Rewarded-Soups setting minus
    the weight interpolation.
    """
    # n_experts is a *count*, not a Das-Dennis resolution. Passing it through as a
    # resolution happens to be right at m=2 and is wrong everywhere else: it would ask for
    # 45 experts at m=3 and 165 at m=4, splitting the budget that many ways and turning the
    # baseline into a strawman precisely when the comparison gets interesting.
    expert_dirs = das_dennis_at_most(env.n_objectives, cfg.n_experts)
    per_expert = max(1, cfg.steps // len(expert_dirs))
    policies = []
    for e, w in enumerate(expert_dirs):
        sub = TrainConfig(**{**cfg.__dict__, "steps": per_expert, "seed": cfg.seed + 100 * e})
        fixed = np.repeat(w[None, :], cfg.group_size, axis=0)

        class _Fixed:
            def sample(self, k):
                return np.zeros(k, dtype=int), fixed

            def update(self, idx, vals):
                pass

            def probabilities(self):
                return np.ones(1)

        policy, _ = _train_conditioned(env, sub, linear_scalar_advantages, _Fixed())
        policies.append(policy)
    return policies, expert_dirs


# --------------------------------------------------------------------------------------
# Method registry
# --------------------------------------------------------------------------------------


def _pace_fn(config: PaCEConfig):
    return lambda R, W: pace_advantages(R, W, config)


METHODS: dict[str, str] = {
    "pace": "Full PaCE: Tchebycheff + cross-direction LOO + frontier shaping + coverage bandit",
    "pace_uniform": "PaCE with uniform direction sampling (ablates the coverage bandit)",
    "pace_no_front": "PaCE with lambda_front=0 (ablates dominance + crowding shaping)",
    "pace_linear": "PaCE with linear scalarization (ablates Tchebycheff)",
    "cond_linear": "Homogeneous groups, weighted-sum advantage (conditioned GRPO as usually applied)",
    "cond_mognorm": "Homogeneous groups, MO-GRPO-style per-objective z-score then sum",
    "cond_dominance": "Homogeneous groups, Pareto-rank-only advantage (PRPO-flavoured)",
    "fixed_scalar": "N independent runs at fixed weights, equal total budget (status quo)",
}


def run_method(method: str, env: SyntheticFrontier, cfg: TrainConfig) -> dict:
    """Train one method and return its evaluated frontier plus metrics."""
    eval_dirs = das_dennis(env.n_objectives, cfg.eval_partitions)
    grid = das_dennis(env.n_objectives, cfg.n_partitions)
    ref = env.reference_point()

    if method == "fixed_scalar":
        policies, expert_dirs = _train_fixed_scalar(env, cfg)
        # Each expert can only be queried at the weight it was trained for. Asking for a
        # trade-off between two experts means another training run -- that is the cost
        # this baseline is meant to expose.
        F = np.stack([evaluate(env, p, w[None, :])[0] for p, w in zip(policies, expert_dirs)])
        used_dirs = expert_dirs
        n_models = len(policies)
    else:
        if method.startswith("pace"):
            config = PaCEConfig(**cfg.pace.__dict__)
            if method == "pace_no_front":
                config.lambda_front = 0.0
            if method == "pace_linear":
                config.scalarization = "linear"
            adv_fn = _pace_fn(config)
        elif method == "cond_linear":
            adv_fn = linear_scalar_advantages
        elif method == "cond_mognorm":
            adv_fn = mo_grpo_advantages
        elif method == "cond_dominance":
            adv_fn = dominance_advantages
        else:
            raise ValueError(f"unknown method: {method!r}")

        bandit_cls = UniformDirections if method != "pace" else CoverageBandit
        if method in ("pace_uniform",):
            bandit_cls = UniformDirections
        if method.startswith("cond_"):
            bandit_cls = UniformDirections
        sampler = bandit_cls(grid, config=cfg.pace.bandit, seed=cfg.seed + 7)

        heterogeneous = method.startswith("pace")
        policy, history = _train_conditioned(env, cfg, adv_fn, sampler, heterogeneous=heterogeneous)
        F = evaluate(env, policy, eval_dirs)
        used_dirs = eval_dirs
        n_models = 1

    summary = frontier_summary(F, ref, used_dirs)
    summary["n_models"] = float(n_models)
    summary["rollouts"] = float(total_rollouts(cfg))
    return {"method": method, "frontier": F, "directions": used_dirs, "metrics": summary}
