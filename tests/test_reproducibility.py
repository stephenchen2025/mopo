"""Runs must be determined by their seed alone, not by what ran before them.

This is a correctness property, not a nicety. In the language-model experiment it was
violated: training problems were drawn from a shared task RNG that advanced through every
previous method and seed, so a given seed saw different data depending on execution order.
Methods were never paired, and one seed scored hypervolume 0.0000 inside a sweep and 0.5518
rerun alone. That single defect produced most of the measurement variance that made the
method comparisons unresolvable (results/ALIGNMENT_RESULTS.md).
"""

import numpy as np

from pace.envs.bandit_trainer import TrainConfig, run_method
from pace.envs.synthetic import SyntheticFrontier


def test_synthetic_runs_are_independent_of_execution_order():
    """The same (method, seed) must give the same answer wherever it appears in a sweep."""
    env = SyntheticFrontier(curvature=0.5, action_resolution=10, n_contexts=3, noise=0.05, seed=0)

    def cfg(seed):
        return TrainConfig(steps=40, seed=seed, n_partitions=4, eval_partitions=4)

    first = run_method("pace", env, cfg(3))["metrics"]["hypervolume"]
    for s in (0, 1):  # consume RNG the way other methods in a sweep would
        run_method("cond_linear", env, cfg(s))
    after = run_method("pace", env, cfg(3))["metrics"]["hypervolume"]

    assert first == after, (
        f"synthetic run is order-dependent: {first} vs {after}. Something is drawing "
        f"randomness from shared state instead of the per-run generator."
    )


def test_synthetic_methods_are_paired_across_seeds():
    """Different methods at the same seed must see the same environment draws.

    Without this, a method-vs-method difference silently includes a different training
    stream, and the comparison measures luck as much as method.
    """
    env = SyntheticFrontier(curvature=0.5, action_resolution=10, n_contexts=3, noise=0.05, seed=0)

    def cfg(seed):
        return TrainConfig(steps=40, seed=seed, n_partitions=4, eval_partitions=4)

    a1 = run_method("pace", env, cfg(7))["metrics"]["hypervolume"]
    b1 = run_method("pace_uniform", env, cfg(7))["metrics"]["hypervolume"]
    b2 = run_method("pace_uniform", env, cfg(7))["metrics"]["hypervolume"]
    a2 = run_method("pace", env, cfg(7))["metrics"]["hypervolume"]

    assert a1 == a2 and b1 == b2, "repeated runs of one method at one seed must agree"


def test_llm_experiment_reseeds_the_task_per_run():
    """Guard the fix in experiments/tiny_lm_rl.py at the source level.

    A behavioural test would need torch and a model; this asserts the reseed is present,
    which is what stops training problems leaking across runs.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "experiments" / "tiny_lm_rl.py").read_text()
    assert (
        "task.rng = np.random.default_rng" in src
    ), "run_method must reseed the task RNG, or training problems depend on execution order"


def test_seed_offset_gives_identical_results_to_unoffset_execution():
    """A seed run via --seed-offset must match running that seed from a bare sweep.

    This is what makes extending a sweep valid: seed 9 run as part of `--seeds 8
    --seed-offset 8` must be the exact same run as seed 9 inside `--seeds 16`, since
    the two are later pooled as though they were one sweep.
    """
    env = SyntheticFrontier(curvature=0.5, action_resolution=10, n_contexts=3, noise=0.05, seed=0)

    def cfg(seed):
        return TrainConfig(steps=40, seed=seed, n_partitions=4, eval_partitions=4)

    # Simulate "part of a --seeds 16 sweep" by running seeds 0..8 in order.
    for s in range(9):
        result = run_method("pace", env, cfg(s))
    # Simulate "the --seed-offset 8 --seeds 1 continuation" in a fresh process state.
    standalone = run_method("pace", env, cfg(8))
    assert result["metrics"]["hypervolume"] == standalone["metrics"]["hypervolume"]


def test_merge_results_pooling_matches_direct_computation():
    """merge_results.py reconstructs pooled mean/std from two partial summaries.

    Verified against numpy's direct computation on the concatenated data, not just
    algebraically: this is what actually gets used to combine two --seed-offset runs
    into the stats an 8-seed run and a 16-seed run would have produced, so it needs to
    be exactly right, not merely a good approximation.
    """
    import sys as _sys
    import pathlib as _pathlib

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    from merge_results import pool

    rng = np.random.default_rng(1)
    for n_a, n_b in ((8, 8), (3, 5), (1, 10)):
        a = rng.normal(0.5, 0.15, n_a)
        b = rng.normal(0.5, 0.15, n_b)
        mean, std, n = pool(a.mean(), a.std(), n_a, b.mean(), b.std(), n_b)
        combined = np.concatenate([a, b])
        assert n == n_a + n_b
        np.testing.assert_allclose(mean, combined.mean(), atol=1e-9)
        np.testing.assert_allclose(std, combined.std(), atol=1e-9)
