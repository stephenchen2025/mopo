"""Tests for reward normalization and scalarization.

The central claim being tested here is the one the whole method rests on: a weighted sum
cannot select the interior of a concave Pareto front, and Tchebycheff can.
"""

import numpy as np
import pytest

from pace.core.scalarization import (
    achievement_matrix,
    direction_scale,
    linear,
    rank_normalize,
    smooth_tchebycheff,
    tchebycheff,
)


def test_rank_normalize_range_and_order():
    R = np.array([[10.0, 1.0], [2.0, 9.0], [5.0, 5.0]])
    U = rank_normalize(R)
    assert U.min() >= 0.0 and U.max() <= 1.0
    assert U[0, 0] == 1.0 and U[1, 0] == 0.0  # best/worst on objective 0


def test_rank_normalize_is_invariant_to_any_monotone_transform():
    """The property that distinguishes rank normalization from a z-score."""
    rng = np.random.default_rng(0)
    R = rng.random((16, 3)) + 0.1
    U = rank_normalize(R)
    for fn in (np.exp, lambda x: x**5, np.log, lambda x: 1000 * x + 7):
        R2 = R.copy()
        R2[:, 1] = fn(R2[:, 1])
        np.testing.assert_allclose(rank_normalize(R2), U, atol=1e-12)


def test_rank_normalize_ties_get_average_rank():
    R = np.array([[1.0], [1.0], [3.0], [4.0]])
    U = rank_normalize(R)
    assert U[0, 0] == U[1, 0] == pytest.approx(0.5 / 3.0)


def test_rank_normalize_constant_objective_is_neutral():
    R = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    U = rank_normalize(R)
    np.testing.assert_allclose(U[:, 0], 0.5)  # carries no signal, as it should


def test_rank_blend_recovers_minmax():
    R = np.array([[0.0], [1.0], [10.0]])
    np.testing.assert_allclose(rank_normalize(R, blend=1.0)[:, 0], [0.0, 0.1, 1.0])


def test_smooth_tchebycheff_converges_to_exact_as_mu_goes_to_zero():
    rng = np.random.default_rng(1)
    U = rng.random((6, 2))
    W = np.array([[1.0, 0.0], [0.5, 0.5], [0.2, 0.8]])
    np.testing.assert_allclose(smooth_tchebycheff(U, W, mu=1e-7, rho=0.0), tchebycheff(U, W), atol=1e-5)


def test_linear_scalarization_cannot_select_interior_of_a_concave_front():
    """The geometric fact that motivates the algorithm.

    A concave front parameterized by ``r = (1 - (1-t)^0.5, 1 - t^0.5)``. Under a weighted
    sum, *every* weight vector picks an endpoint. Under Tchebycheff, the argmax sweeps
    smoothly across the interior.
    """
    t = np.linspace(0, 1, 11)
    F = np.stack([1 - np.sqrt(1 - t), 1 - np.sqrt(t)], axis=1)
    W = np.stack([np.linspace(0, 1, 11), 1 - np.linspace(0, 1, 11)], axis=1)

    linear_choices = set(linear(F, W).argmax(axis=0).tolist())
    assert linear_choices <= {0, 10}, "weighted sum should only ever pick the extremes"

    U = (F - F.min(0)) / (F.max(0) - F.min(0))
    tch_choices = tchebycheff(U, W).argmax(axis=0)
    assert len(set(tch_choices.tolist())) >= 7, "Tchebycheff should sweep the interior"
    assert np.all(np.diff(tch_choices) >= 0), "and should do so monotonically in w"


def test_linear_scalarization_is_fine_on_a_convex_front():
    """The control: the failure above is about geometry, not about weighted sums per se."""
    t = np.linspace(0, 1, 11)
    F = np.stack([1 - (1 - t) ** 2, 1 - t**2], axis=1)
    W = np.stack([np.linspace(0, 1, 11), 1 - np.linspace(0, 1, 11)], axis=1)
    assert len(set(linear(F, W).argmax(axis=0).tolist())) >= 7


def test_achievement_matrix_shape_and_diagonal():
    U = np.random.default_rng(3).random((5, 2))
    W = np.random.default_rng(4).dirichlet(np.ones(2), size=5)
    S = achievement_matrix(U, W)
    assert S.shape == (5, 5)
    np.testing.assert_allclose(np.diag(S), [smooth_tchebycheff(U[[k]], W[[k]])[0, 0] for k in range(5)])


def test_direction_scale_equalizes_achievement_range_across_directions():
    """Without this correction the coverage bandit thinks extremes are always weakest."""
    W = np.array([[1.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
    worst = np.zeros((1, 2))  # u = 0 everywhere is the worst possible utility
    s_worst = tchebycheff(worst, W)[0] / direction_scale(W)
    np.testing.assert_allclose(s_worst, -1.0)  # every direction bottoms out at -1


def test_rank_normalization_linearizes_front_geometry_and_tchebycheff_breaks_the_tie():
    """How the two mechanisms actually divide the labour -- found by measurement.

    Rank normalization maps any monotone two-objective front onto evenly spaced ranks, so
    a *concave* front becomes exactly linear in rank space. That removes the weighted
    sum's pull toward the extremes, but it replaces it with something else: under a
    balanced weight every point ties exactly, so a weighted sum is perfectly *indifferent*
    and supplies no signal about which trade-off to pick. Tchebycheff breaks that tie
    strictly in favour of the balanced point, which is what makes it a usable gradient.

    Both halves matter, and neither alone is enough: without ranks you get collapse to an
    extreme, without Tchebycheff you get an undirected drift.
    """
    t = np.linspace(0, 1, 8)
    F = np.stack([1 - np.sqrt(1 - t), 1 - np.sqrt(t)], axis=1)  # strongly concave
    assert F[len(F) // 2].sum() < 0.7 * F[0].sum(), "front should be concave in reward space"

    U = rank_normalize(F)
    np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-12)  # linear in rank space

    balanced = np.array([[0.5, 0.5]])
    linear_scores = linear(U, balanced).ravel()
    assert np.allclose(linear_scores, linear_scores[0]), "weighted sum is fully indifferent"

    tch_choice = int(tchebycheff(U, balanced).argmax(axis=0)[0])
    assert 2 <= tch_choice <= 5, "Tchebycheff should strictly select a balanced point"


def test_rank_linearization_is_a_two_objective_accident_and_does_not_generalize():
    """Scope check on the interaction documented above -- it holds only for m = 2.

    With two objectives on a monotone front the ranks of objective 0 are a permutation of
    the uniform grid and the ranks of objective 1 are exactly its reverse, so ``u_0 + u_1``
    is identically 1: the front is linear in rank space and a weighted sum is indifferent.
    Nothing forces that at m >= 3 -- three rank permutations are not mutually reversed --
    so rank space keeps the front's curvature and a weighted sum is *not* indifferent.

    This matters for how the method is configured: at m = 2 Tchebycheff's job is mostly
    breaking a tie the ranks created, while at m >= 3 the concavity survives normalization
    and Tchebycheff is load-bearing again. The shipped LLM configs are all m = 3.
    """
    from pace.core.directions import das_dennis

    t = np.linspace(0, 1, 8)
    F2 = np.stack([1 - np.sqrt(1 - t), 1 - np.sqrt(t)], axis=1)
    sums2 = rank_normalize(F2).sum(axis=1)
    np.testing.assert_allclose(sums2, 1.0, atol=1e-12)

    T = das_dennis(3, 6)
    F3 = 1 - np.power(np.clip(1 - T, 0.0, 1.0), 0.5)
    sums3 = rank_normalize(F3).sum(axis=1)
    assert sums3.max() - sums3.min() > 0.1, "m=3 rank vectors should not be co-planar"

    balanced = np.array([[1 / 3, 1 / 3, 1 / 3]])
    scores = linear(rank_normalize(F3), balanced).ravel()
    assert scores.max() - scores.min() > 1e-3, "a weighted sum is not indifferent at m=3"
