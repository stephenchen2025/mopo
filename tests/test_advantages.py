"""Tests for the PaCE advantage estimator.

The properties tested here are the ones the method's correctness rests on: the baseline
must not see the rollout it is a baseline for, dominated rollouts must be penalized, and
the whole thing must be invariant to monotone rescaling of any objective.
"""

import numpy as np
import pytest

from pace.core.advantages import (
    dominance_advantages,
    linear_scalar_advantages,
    mo_grpo_advantages,
    pace_advantages,
)
from pace.core.config import PaCEConfig
from pace.core.scalarization import achievement_matrix


def _group(seed=0, K=8, m=2):
    rng = np.random.default_rng(seed)
    R = rng.random((K, m))
    W = rng.dirichlet(np.ones(m), size=K)
    return R, W


def test_output_shapes():
    R, W = _group()
    out = pace_advantages(R, W)
    assert out.advantages.shape == (8,)
    assert out.matrix.shape == (8, 8)
    assert out.utilities.shape == R.shape


def test_mismatched_shapes_raise():
    R, W = _group()
    with pytest.raises(ValueError, match="one direction per rollout"):
        pace_advantages(R, W[:4])


def test_singleton_group_gives_zero_advantage():
    """A group of one carries no relative information; anything else would be invented."""
    out = pace_advantages(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))
    assert out.advantages == pytest.approx(0.0)


def test_leave_one_out_baseline_excludes_the_rollout_it_scores():
    """The control-variate property, tested where it actually holds.

    Given *fixed* utilities, the baseline for rollout k is built only from the other
    rollouts' achievements under direction w_k, so perturbing rollout k's own utility
    leaves its baseline untouched. This is the RLOO argument, and it is what stops the
    baseline from correlating with the sampled action.
    """
    rng = np.random.default_rng(0)
    U = rng.random((8, 2))
    W = rng.dirichlet(np.ones(2), size=8)
    S = achievement_matrix(U, W)

    def baseline(mat, k):
        return (mat[:, k].sum() - mat[k, k]) / (mat.shape[0] - 1)

    U2 = U.copy()
    U2[0] = [0.99, 0.01]
    S2 = achievement_matrix(U2, W)
    assert baseline(S, 0) == pytest.approx(baseline(S2, 0), abs=1e-12)


def test_within_group_normalization_couples_rollouts_documented_caveat():
    """The corresponding caveat, asserted rather than hidden.

    The leave-one-out construction removes the *direct* dependence of the baseline on
    rollout k's own score. It does not make the estimator strictly unbiased, because the
    within-group normalization that produces the utilities is itself a function of every
    rollout including k: moving k's raw reward can shift the other rollouts' ranks and
    therefore its own baseline. This is not specific to PaCE -- every group-relative
    method that normalizes within the group inherits it, standard GRPO included.
    """
    R, W = _group()
    cfg = PaCEConfig(rank_blend=1.0)
    out = pace_advantages(R, W, cfg)
    before = out.achievement[0] - out.achievement_advantage[0]

    R2 = R.copy()
    R2[0] = [50.0, -50.0]  # large enough to move the group's min-max range
    out2 = pace_advantages(R2, W, cfg)
    after = out2.achievement[0] - out2.achievement_advantage[0]
    assert before != pytest.approx(
        after, abs=1e-9
    ), "if this ever passes, the coupling is gone and the caveat can be dropped"


def test_dominated_rollout_is_penalized():
    R = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6], [0.05, 0.05]])
    W = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
    out = pace_advantages(R, W)
    assert out.fronts[3] > 1
    assert out.advantages[3] == out.advantages.min()


def test_pace_is_invariant_to_monotone_rescaling_but_mo_grpo_is_not():
    """Rank normalization buys invariance that a per-objective z-score does not have."""
    R, W = _group(seed=3, K=10)
    base = pace_advantages(R, W).advantages
    baseline_before = mo_grpo_advantages(R, W)

    R2 = R.copy()
    R2[:, 1] = np.exp(4 * R2[:, 1])  # strictly increasing: the Pareto set is unchanged

    np.testing.assert_allclose(pace_advantages(R2, W).advantages, base, atol=1e-9)
    assert not np.allclose(mo_grpo_advantages(R2, W), baseline_before, atol=1e-3)


def test_lambda_front_zero_drops_the_shaping_term():
    R, W = _group(seed=5)
    out = pace_advantages(R, W, PaCEConfig(lambda_front=0.0))
    a_ach = out.achievement_advantage
    expected = (a_ach - a_ach.mean()) / (a_ach.std() + 1e-8)
    np.testing.assert_allclose(out.advantages, expected, atol=1e-6)


def test_advantages_are_clipped():
    R = np.array([[100.0, 0.0]] + [[0.0, 1.0]] * 7)
    W = np.repeat([[0.9, 0.1]], 8, axis=0)
    out = pace_advantages(R, W, PaCEConfig(advantage_clip=1.5))
    assert out.advantages.max() <= 1.5 and out.advantages.min() >= -1.5


def test_constant_rewards_give_no_signal():
    R = np.ones((6, 2))
    W = np.random.default_rng(0).dirichlet(np.ones(2), size=6)
    np.testing.assert_allclose(pace_advantages(R, W).advantages, 0.0, atol=1e-9)


def test_cross_direction_matrix_is_not_symmetric_and_uses_every_pair():
    """The K x K matrix is the mechanism; a rank-one or symmetric S means it collapsed."""
    R, W = _group(seed=7)
    S = pace_advantages(R, W).matrix
    assert not np.allclose(S, S.T)
    assert np.linalg.matrix_rank(S, tol=1e-8) > 1


def test_homogeneous_directions_collapse_the_mechanism():
    """Sanity check on the documented requirement that groups be heterogeneous.

    With one shared direction every column of S is identical, so the cross-direction
    baseline degenerates to an ordinary leave-one-out group baseline. This is the failure
    mode the TRL adapter warns about.
    """
    R, _ = _group(seed=9)
    W = np.repeat([[0.5, 0.5]], R.shape[0], axis=0)
    S = pace_advantages(R, W).matrix
    np.testing.assert_allclose(S, S[:, [0]] @ np.ones((1, S.shape[1])), atol=1e-12)


def test_baselines_run_and_are_zero_mean():
    R, W = _group(seed=11)
    for fn in (linear_scalar_advantages, mo_grpo_advantages, dominance_advantages):
        a = fn(R, W)
        assert a.shape == (8,)
        assert a.mean() == pytest.approx(0.0, abs=1e-9)


def test_alignment_term_is_zero_for_a_collapsed_policy():
    """The property the term exists for.

    A policy that ignores its conditioning emits the same behaviour at every direction, so
    every row of S is identical. Double-centering then gives exactly zero: collapse earns
    nothing. This matters because hypervolume *rewards* collapse once the policy can move
    the frontier outward (results/HEADROOM_RESULTS.md), so the advantage must not.
    """
    W = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])
    collapsed = np.tile([0.5, 0.5], (5, 1))
    np.testing.assert_allclose(pace_advantages(collapsed, W).align_advantage, 0.0, atol=1e-12)


def test_alignment_term_rewards_matching_and_penalizes_anti_matching():
    W = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])
    matched = W.copy()  # rollout k is exactly what direction k asks for
    align_matched = pace_advantages(matched, W).align_advantage
    align_anti = pace_advantages(matched[::-1], W).align_advantage

    assert align_matched[0] > 0 and align_matched[-1] > 0
    assert align_anti[0] < 0 and align_anti[-1] < 0
    assert align_matched.sum() > align_anti.sum()


def test_alignment_term_is_off_by_default_and_additive_when_on():
    """Off by default so previously published results reproduce unchanged."""
    R, W = _group(seed=4)
    base = pace_advantages(R, W).advantages
    np.testing.assert_allclose(pace_advantages(R, W, PaCEConfig(lambda_align=0.0)).advantages, base)
    assert not np.allclose(pace_advantages(R, W, PaCEConfig(lambda_align=0.5)).advantages, base)


def test_alignment_is_invariant_to_a_constant_shift_of_any_row_or_column():
    """Double-centering removes exactly the main effects it is meant to remove.

    Making one rollout uniformly better, or one direction uniformly easier, must not change
    the alignment signal -- only the interaction should.
    """
    from pace.core.scalarization import achievement_matrix, rank_normalize

    R, W = _group(seed=6)
    S = achievement_matrix(rank_normalize(R), W)

    def align(mat):
        return np.diag(mat - mat.mean(axis=1, keepdims=True) - mat.mean(axis=0, keepdims=True) + mat.mean())

    shifted_row = S.copy()
    shifted_row[2, :] += 3.0
    shifted_col = S.copy()
    shifted_col[:, 1] += 3.0
    np.testing.assert_allclose(align(shifted_row), align(S), atol=1e-9)
    np.testing.assert_allclose(align(shifted_col), align(S), atol=1e-9)
