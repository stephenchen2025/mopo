"""Tests for the frontier primitives."""

import numpy as np
import pytest

from pace.core.pareto import (
    _hv_inclusion_exclusion,
    _nondominated_mask_2d,
    _nondominated_mask_large,
    crowding_distance,
    dominates,
    hypervolume,
    nondominated_mask,
    nondominated_sort,
    normalized_crowding,
)


def test_dominates_basic():
    assert dominates([2, 2], [1, 1])
    assert dominates([2, 1], [1, 1])
    assert not dominates([1, 1], [1, 1])  # equal does not dominate
    assert not dominates([2, 0], [1, 1])  # trade-off: neither dominates


def test_nondominated_mask_simple():
    F = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0], [0.5, 0.5]])
    np.testing.assert_array_equal(nondominated_mask(F), [True, True, True, False])


def test_nondominated_mask_duplicates_all_survive():
    F = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert nondominated_mask(F).all()


def test_nondominated_sort_layers():
    F = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    np.testing.assert_array_equal(nondominated_sort(F), [1, 2, 3])


def test_nondominated_sort_partitions_every_point():
    rng = np.random.default_rng(0)
    F = rng.random((60, 3))
    fronts = nondominated_sort(F)
    assert (fronts >= 1).all()
    # Front 1 must agree with the direct mask, and every point must be assigned.
    np.testing.assert_array_equal(fronts == 1, nondominated_mask(F))


@pytest.mark.parametrize("m", [2, 3, 4])
def test_scalable_paths_agree_with_dense(m):
    """The 2-D sweep and the incremental filter must match the O(n^2) matrix exactly."""
    rng = np.random.default_rng(m)
    F = rng.random((500, m))
    dense = nondominated_mask(F, dense_limit=10_000)
    np.testing.assert_array_equal(_nondominated_mask_large(F), dense)
    if m == 2:
        np.testing.assert_array_equal(_nondominated_mask_2d(F), dense)


def test_hypervolume_2d_known_value():
    # Union of [0,3]x[0,1] and [0,1]x[0,3]: 3 + 3 - 1 overlap = 5.
    F = np.array([[3.0, 1.0], [1.0, 3.0]])
    assert hypervolume(F, np.zeros(2)) == pytest.approx(5.0)


def test_hypervolume_2d_matches_inclusion_exclusion():
    rng = np.random.default_rng(1)
    F = rng.random((12, 2))
    ref = np.zeros(2)
    assert hypervolume(F, ref) == pytest.approx(_hv_inclusion_exclusion(F, ref), rel=1e-9)


def test_hypervolume_3d_monte_carlo_matches_exact():
    rng = np.random.default_rng(2)
    F = rng.random((8, 3))
    ref = np.zeros(3)
    exact = _hv_inclusion_exclusion(F, ref)
    mc = hypervolume(F, ref, method="mc", n_samples=400_000, seed=0)
    assert mc == pytest.approx(exact, rel=0.02)


def test_hypervolume_ignores_points_below_reference():
    F = np.array([[2.0, 2.0], [-1.0, 5.0]])
    assert hypervolume(F, np.zeros(2)) == pytest.approx(4.0)


def test_hypervolume_is_monotone_under_adding_a_dominating_point():
    F = np.array([[2.0, 2.0]])
    ref = np.zeros(2)
    before = hypervolume(F, ref)
    after = hypervolume(np.vstack([F, [3.0, 3.0]]), ref)
    assert after > before


def test_crowding_boundaries_are_infinite():
    F = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
    d = crowding_distance(F, np.ones(3, dtype=int))
    assert np.isinf(d[0]) and np.isinf(d[2])
    assert np.isfinite(d[1])


def test_normalized_crowding_ranks_sparse_above_crowded():
    # Two points nearly on top of each other, one well separated.
    F = np.array([[4.0, 0.0], [3.0, 1.0], [2.9, 1.05], [1.0, 3.0], [0.0, 4.0]])
    c = normalized_crowding(F, nondominated_sort(F))
    assert c[0] == 1.0 and c[4] == 1.0  # boundaries stay at the top
    assert c[1] < c[3]  # the crowded pair scores below the isolated point
    assert c[2] < c[3]


def test_steerability_does_not_saturate_or_report_spurious_correlation():
    """Why the slope metric replaced Spearman over per-direction aggregates.

    Spearman sees only D points (9 here) after averaging away the per-problem spread. It
    saturates at 1.0 for any monotone response, so it cannot distinguish a strongly steerable
    policy from a weakly steerable one, and on a *collapsed* policy it reports whatever rank
    order the residual noise happens to produce -- a large spurious magnitude for a policy
    that is doing nothing at all. Both failure modes inflate the variance of the measurement,
    which is what made 3-seed comparisons unresolvable (results/ALIGNMENT_RESULTS.md).
    """
    from pace.core.metrics import controllability, steerability

    W = np.stack([np.linspace(0, 1, 9), 1 - np.linspace(0, 1, 9)], axis=1)
    rng = np.random.default_rng(0)

    steerable = np.stack(
        [
            np.stack([W[d, 0] * 0.8 + rng.normal(0, 0.15, 24), 0.8 - W[d, 0] * 0.6 + rng.normal(0, 0.15, 24)], axis=1)
            for d in range(9)
        ]
    )
    collapsed = np.stack([np.stack([rng.normal(0.5, 0.15, 24), rng.normal(0.5, 0.15, 24)], axis=1) for _ in range(9)])

    # Spearman saturates on the steerable case, losing all gradation.
    assert np.allclose(controllability(W, steerable.mean(axis=1)), 1.0)
    # The slope does not, and stays well inside (0, 1).
    slope_good = steerability(W, steerable)
    assert np.all(slope_good > 0.2) and np.all(slope_good < 0.95)

    # On a collapsed policy the slope stays near zero where Spearman does not.
    slope_bad = steerability(W, collapsed)
    assert np.all(np.abs(slope_bad) < 0.15)
    assert np.all(np.abs(slope_bad) < np.abs(slope_good))


def test_steerability_validates_its_input_shape():
    from pace.core.metrics import steerability

    W = np.stack([np.linspace(0, 1, 4), 1 - np.linspace(0, 1, 4)], axis=1)
    with pytest.raises(ValueError, match=r"expected \(D, P, m\)"):
        steerability(W, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="incompatible"):
        steerability(W, np.zeros((3, 5, 2)))
