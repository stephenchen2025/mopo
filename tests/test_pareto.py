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


def test_steerability_is_bounded_and_does_not_explode_under_the_null():
    """Two discarded designs are the reason this specific property gets its own test.

    A slope normalized by the raw observation range dilutes any high-variance objective
    (simulated: an identical injected effect read as -0.04 for a Bernoulli-like reward
    against +0.33 for a continuous one -- an 8x spurious gap from noise character alone,
    not from steerability). A slope normalized by the range of per-direction means, or by a
    variance-decomposition debiased version of it, fixes the dilution but can explode under
    the null: the sampled between-direction spread is itself noisy and can be small by
    chance, and dividing by it is unbounded (measured up to 9.28 under a genuine null with
    D=9, repeated 200 times). Pearson correlation on the means is bounded by construction and
    cannot repeat either failure -- this is the property that makes it usable at all, so it
    is asserted directly and repeatedly rather than trusted from the formula alone.
    """
    from pace.core.metrics import steerability

    W = np.stack([np.linspace(0, 1, 9), 1 - np.linspace(0, 1, 9)], axis=1)
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(200):
        Y = rng.binomial(1, 0.5, size=(9, 48)).astype(float)  # genuine null, noisy objective
        s = steerability(W, np.stack([Y, Y], axis=-1))
        worst = max(worst, float(np.abs(s).max()))
    assert worst <= 1.0, f"steerability must never leave [-1, 1]; saw {worst}"


def test_steerability_is_not_diluted_by_high_variance_objectives():
    """The dilution failure of the first discarded design, checked directly.

    Inject the identical direction-dependent mean shift into a Bernoulli-like reward (large
    per-problem variance, as a 0/1 accuracy check is) and a continuous one (small per-problem
    variance). A metric normalized by the raw observation range reports roughly an 8x smaller
    magnitude for the noisy objective purely from its noise character. Pearson-on-means must
    not reproduce that gap by anywhere near that factor, since both objectives carry the same
    true population correlation once averaged.
    """
    from pace.core.metrics import steerability

    D, P = 9, 200  # large P so each per-direction mean is precise, isolating the true effect
    w = np.linspace(0, 1, D)
    W = np.stack([w, 1 - w], axis=1)
    rng = np.random.default_rng(1)
    shift = 0.15
    mean_by_dir = 0.5 + shift * (w - 0.5)

    noisy = rng.binomial(1, np.clip(mean_by_dir, 0.05, 0.95)[:, None], size=(D, P)).astype(float)
    clean = mean_by_dir[:, None] + rng.normal(0, 0.05, size=(D, P))

    s_noisy = steerability(W, np.stack([noisy, noisy], axis=-1))[0]
    s_clean = steerability(W, np.stack([clean, clean], axis=-1))[0]
    assert s_noisy > 0 and s_clean > 0
    assert s_noisy / s_clean > 0.25, (
        f"noisy-objective steerability ({s_noisy:.3f}) should not be diluted far below the "
        f"clean-objective reading ({s_clean:.3f}) for the same true effect"
    )


def test_spearman_ties_from_quantization_rarely_force_exactly_zero():
    """A correction to this repo's own earlier reasoning, checked so it cannot drift back.

    An earlier version of this module's docstring justified replacing Spearman partly by
    claiming it "returns exactly 0.0 whenever achieved rewards happen to be constant, which
    happened in 10 of 24 runs". Investigating that claim here shows it was imprecise: with
    quantized per-direction means (accuracy = k/48, as real evaluation produces), near-ties
    across the 9 directions are common under a small true effect -- 479/500 simulated trials
    had at least one tied mean -- but scipy's average-rank tie handling means Spearman
    returns exactly 0.0 only rarely from that (4/500). The exactly-0.000 values seen in real
    training runs were most likely genuine policy collapse (every direction producing a
    literally identical mean, from deterministic greedy decoding), which is the *correct*
    reading, not a Spearman artifact -- and Pearson-on-means would report the same 0.0 in
    that exact case, since a genuinely constant array has no correlation with anything.
    """
    from scipy import stats as st

    rng = np.random.default_rng(5)
    D, P = 9, 48
    w = np.linspace(0, 1, D)
    exact_zero = 0
    for _ in range(500):
        shift = rng.uniform(0.0, 0.1)
        mean_by_dir = 0.5 + shift * (w - 0.5)
        Y = rng.binomial(1, np.clip(mean_by_dir, 0.02, 0.98)[:, None], size=(D, P)).astype(float)
        if st.spearmanr(w, Y.mean(axis=1)).correlation == 0.0:
            exact_zero += 1
    assert exact_zero < 25, "quantization ties should only rarely force Spearman to exactly 0"


def test_steerability_returns_zero_for_genuinely_flat_response():
    """The one case where returning exactly 0.0 is correct rather than an artifact.

    A policy whose greedy output is byte-identical regardless of direction -- the collapse
    failure mode in results/HEADROOM_RESULTS.md -- gives means that are not just close but
    exactly equal. Zero is the right answer here, unlike Spearman's zero-on-any-tie problem.
    """
    from pace.core.metrics import steerability

    W = np.stack([np.linspace(0, 1, 9), 1 - np.linspace(0, 1, 9)], axis=1)
    identical_row = np.random.default_rng(0).normal(0.6, 0.1, 48)
    Y = np.tile(identical_row, (9, 1))
    s = steerability(W, np.stack([Y, Y], axis=-1))
    np.testing.assert_allclose(s, 0.0)


def test_steerability_accepts_pre_aggregated_means_directly():
    """The (D, m) input path, for callers that already have per-direction means."""
    from pace.core.metrics import steerability

    W = np.stack([np.linspace(0, 1, 9), 1 - np.linspace(0, 1, 9)], axis=1)
    rng = np.random.default_rng(3)
    means = W * 0.6 + rng.normal(0, 0.02, size=W.shape)
    s = steerability(W, means)
    assert s.shape == (2,)
    assert s[0] > 0.8  # near-noiseless linear relationship


def test_steerability_validates_its_input_shape():
    from pace.core.metrics import steerability

    W = np.stack([np.linspace(0, 1, 4), 1 - np.linspace(0, 1, 4)], axis=1)
    with pytest.raises(ValueError, match="incompatible"):
        steerability(W, np.zeros((3, 5, 2)))
    with pytest.raises(ValueError, match=r"expected \(D, P, m\) or \(D, m\)"):
        steerability(W, np.zeros((4,)))
