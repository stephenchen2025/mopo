#!/usr/bin/env python3
"""Validation of the PaCE advantage estimator on the synthetic multi-objective bandit.

Runs four experiments and writes both machine-readable and human-readable results:

1. **Main comparison** -- every method against every baseline, on a concave front (where
   linear scalarization provably fails) and a convex one (where it does not). Equal
   rollout budget throughout.
2. **Ablations** -- which PaCE component is carrying the result.
3. **Scale invariance** -- apply a strictly increasing transform to one objective, which
   leaves the Pareto set untouched, and check which methods notice.
4. **Coverage bandit** -- tested under the condition it is designed for (frontier regions
   of uneven difficulty) and the condition it is not.

This validates the optimizer's frontier behaviour. It says nothing about language models.

Usage::

    python experiments/synthetic_frontier.py --seeds 16 --steps 1500
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pace.core import hypervolume  # noqa: E402
from pace.core.config import PaCEConfig  # noqa: E402
from pace.core.directions import BanditConfig  # noqa: E402
from pace.envs.bandit_trainer import METHODS, TrainConfig, run_method, total_rollouts  # noqa: E402
from pace.envs.synthetic import SyntheticFrontier  # noqa: E402

METRICS = ("hypervolume", "spacing", "max_spread", "controllability_mean", "n_models")


def _one_seed(job):
    """Top-level so it survives pickling into a worker process."""
    method, env, cfg = job
    return run_method(method, env, cfg)["metrics"]


_WORKERS = 1


def aggregate(env, method, cfg_fn, seeds):
    """Run one method across seeds and return mean/std of each metric.

    Seeds are independent, so they fan out across processes. This matters more than it
    looks: the full grid is ~40 cells and a serial run takes hours, which is long enough
    that nobody re-runs it after changing anything.
    """
    jobs = [(method, env, cfg_fn(seed)) for seed in range(seeds)]
    if _WORKERS > 1:
        with ProcessPoolExecutor(max_workers=_WORKERS) as pool:
            rows = list(pool.map(_one_seed, jobs))
    else:
        rows = [_one_seed(job) for job in jobs]
    out = {}
    for key in METRICS:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = float(vals.mean())
        out[key + "_std"] = float(vals.std())
    out["hypervolume_all"] = [float(r["hypervolume"]) for r in rows]
    return out


def experiment_main(args) -> dict:
    """Every method, both front geometries, equal budget."""
    results = {}
    for curvature, label in ((0.5, "concave"), (2.0, "convex")):
        env = SyntheticFrontier(
            n_objectives=args.objectives,
            curvature=curvature,
            action_resolution=args.action_resolution,
            n_contexts=args.contexts,
            noise=args.noise,
            seed=0,
        )
        oracle = hypervolume(env.oracle_frontier(), env.reference_point())
        block = {"oracle_hypervolume": float(oracle), "methods": {}}
        for method in METHODS:
            cfg_fn = lambda s: TrainConfig(  # noqa: E731
                steps=args.steps,
                lr=args.lr,
                entropy_coef=args.entropy,
                n_partitions=args.partitions,
                seed=s,
            )
            stats = aggregate(env, method, cfg_fn, args.seeds)
            stats["pct_of_oracle"] = 100.0 * stats["hypervolume"] / oracle
            block["methods"][method] = stats
            print(
                f"  [{label}] {method:16s} HV={stats['hypervolume']:.4f}"
                f" +-{stats['hypervolume_std']:.4f} ({stats['pct_of_oracle']:5.1f}% of oracle)"
                f" ctrl={stats['controllability_mean']:+.3f}",
                flush=True,
            )
        results[label] = block
    return results


def experiment_ablation(args) -> dict:
    """Which component is doing the work, on the concave front."""
    env = SyntheticFrontier(
        curvature=0.5,
        action_resolution=args.action_resolution,
        n_contexts=args.contexts,
        noise=args.noise,
        seed=0,
    )
    oracle = hypervolume(env.oracle_frontier(), env.reference_point())
    variants = {
        "full": PaCEConfig(),
        "lambda_front=0.0": PaCEConfig(lambda_front=0.0),
        "lambda_front=0.6": PaCEConfig(lambda_front=0.6),
        "lambda_crowd=0.0": PaCEConfig(lambda_crowd=0.0),
        "rank_blend=1.0 (minmax)": PaCEConfig(rank_blend=1.0),
        "mu=0.02 (sharp)": PaCEConfig(mu=0.02),
        "mu=0.5 (soft)": PaCEConfig(mu=0.5),
        "loo_shrinkage=1.0": PaCEConfig(loo_shrinkage=1.0),
    }
    out = {"oracle_hypervolume": float(oracle), "variants": {}}
    for name, pace_cfg in variants.items():
        cfg_fn = lambda s, c=pace_cfg: TrainConfig(  # noqa: E731
            steps=args.steps,
            lr=args.lr,
            entropy_coef=args.entropy,
            n_partitions=args.partitions,
            seed=s,
            pace=c,
        )
        stats = aggregate(env, "pace_uniform", cfg_fn, args.seeds)
        stats["pct_of_oracle"] = 100.0 * stats["hypervolume"] / oracle
        out["variants"][name] = stats
        print(
            f"  [ablation] {name:24s} HV={stats['hypervolume']:.4f}"
            f" +-{stats['hypervolume_std']:.4f} ({stats['pct_of_oracle']:5.1f}%)",
            flush=True,
        )
    return out


def experiment_scale_invariance(args) -> dict:
    """Apply a strictly increasing map to one objective and see who notices.

    The transform does not move the Pareto set by even one point, so any change in a
    method's result is pure sensitivity to reward scale.
    """
    out = {}
    for tag, transforms in (
        ("identity", None),
        ("obj1 -> exp(4x)", (None, "exp4")),
        ("obj1 -> x^5", (None, "pow5")),
    ):
        env = SyntheticFrontier(
            curvature=0.5,
            action_resolution=args.action_resolution,
            n_contexts=args.contexts,
            noise=args.noise,
            objective_transforms=transforms,
            seed=0,
        )
        # The reference point and oracle stay in the untransformed space: rewards are only
        # transformed on the way *into* the learner, and evaluation uses true rewards.
        oracle = hypervolume(env.oracle_frontier(), env.reference_point())
        block = {"oracle_hypervolume": float(oracle), "methods": {}}
        for method in ("pace_uniform", "cond_mognorm", "cond_linear"):
            cfg_fn = lambda s: TrainConfig(  # noqa: E731
                steps=args.steps,
                lr=args.lr,
                entropy_coef=args.entropy,
                n_partitions=args.partitions,
                seed=s,
            )
            stats = aggregate(env, method, cfg_fn, args.seeds)
            block["methods"][method] = stats
            print(
                f"  [scale:{tag:14s}] {method:14s} HV={stats['hypervolume']:.4f}" f" +-{stats['hypervolume_std']:.4f}",
                flush=True,
            )
        out[tag] = block
    return out


def experiment_bandit(args) -> dict:
    """The coverage bandit, under the condition it is designed for and the one it is not.

    The bandit can only pay off when there is something to reallocate: frontier regions of
    genuinely uneven difficulty, and a direction grid meaningfully larger than the group
    size. With ``D`` close to ``K`` and sampling without replacement, a group covers almost
    the whole grid regardless of the bandit's probabilities, and the mechanism is inert by
    construction.
    """
    out = {}
    for skew, label in ((0.0, "uniform difficulty"), (8.0, "skewed difficulty")):
        env = SyntheticFrontier(
            curvature=0.5,
            action_resolution=args.action_resolution,
            n_contexts=args.contexts,
            noise=args.noise,
            noise_skew=skew,
            seed=0,
        )
        oracle = hypervolume(env.oracle_frontier(), env.reference_point())
        block = {"oracle_hypervolume": float(oracle), "grids": {}}
        for partitions in (8, 40):
            cell = {}
            for method in ("pace", "pace_uniform"):
                cfg_fn = lambda s, p=partitions: TrainConfig(  # noqa: E731
                    steps=args.steps,
                    lr=args.lr,
                    entropy_coef=args.entropy,
                    n_partitions=p,
                    seed=s,
                    pace=PaCEConfig(n_partitions=p, bandit=BanditConfig()),
                )
                cell[method] = aggregate(env, method, cfg_fn, args.seeds)
            block["grids"][f"D={partitions + 1}"] = cell
            print(
                f"  [bandit:{label:18s} D={partitions + 1:3d}]"
                f" bandit={cell['pace']['hypervolume']:.4f}+-{cell['pace']['hypervolume_std']:.4f}"
                f"  uniform={cell['pace_uniform']['hypervolume']:.4f}"
                f"+-{cell['pace_uniform']['hypervolume_std']:.4f}",
                flush=True,
            )
        out[label] = block
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--partitions", type=int, default=20)
    parser.add_argument("--action-resolution", type=int, default=20)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument(
        "--objectives",
        type=int,
        default=2,
        help="Number of objectives m. The shipped LLM configs use 3, and "
        "the method ordering differs between m=2 and m=3 -- see "
        "results/RESULTS.md section 5.",
    )
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="results/synthetic_results.json")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Processes to fan seeds across."
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "main", "ablation", "scale", "bandit"],
    )
    args = parser.parse_args()

    global _WORKERS
    _WORKERS = max(1, args.workers)

    budget = total_rollouts(TrainConfig(steps=args.steps))
    print(
        f"PaCE synthetic validation: {args.seeds} seeds, {args.steps} steps, "
        f"{budget} rollouts per run, {_WORKERS} workers\n"
    )

    results = {
        "config": vars(args),
        "rollouts_per_run": budget,
        "method_descriptions": METHODS,
    }
    started = time.time()
    if args.only in ("all", "main"):
        print("[1/4] main comparison")
        results["main"] = experiment_main(args)
    if args.only in ("all", "ablation"):
        print("[2/4] ablations")
        results["ablation"] = experiment_ablation(args)
    if args.only in ("all", "scale"):
        print("[3/4] scale invariance")
        results["scale_invariance"] = experiment_scale_invariance(args)
    if args.only in ("all", "bandit"):
        print("[4/4] coverage bandit")
        results["bandit"] = experiment_bandit(args)
    results["elapsed_seconds"] = time.time() - started

    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path} in {results['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
