#!/usr/bin/env python3
"""Merge two tiny_lm_rl.py result files from disjoint --seed-offset runs.

Pools mean/std correctly (not by averaging the reported stds) by reconstructing each
run's sum and sum-of-squares from its reported mean, std, and n. Use this to extend a
sweep -- e.g. run --seeds 8 then --seeds 8 --seed-offset 8 -- instead of re-running seeds
already computed, which matters once resolving a method difference needs 15-20 seeds
rather than the 3-8 a first pass typically uses.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def pool(mean_a, std_a, n_a, mean_b, std_b, n_b):
    """Combine two (mean, population-std, n) summaries into the stats of the union."""
    n = n_a + n_b
    mean = (mean_a * n_a + mean_b * n_b) / n
    # E[X^2] for each part, then combine, then recover the pooled variance.
    ex2_a = std_a**2 + mean_a**2
    ex2_b = std_b**2 + mean_b**2
    ex2 = (ex2_a * n_a + ex2_b * n_b) / n
    var = max(ex2 - mean**2, 0.0)
    return mean, var**0.5, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="Two or more tiny_lm_rl.py JSON outputs to merge")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    docs = [json.loads(pathlib.Path(f).read_text()) for f in args.inputs]
    base = docs[0]
    for d in docs[1:]:
        for method, stats in d["methods"].items():
            if method not in base["methods"]:
                base["methods"][method] = dict(stats)
                base["methods"][method]["_n"] = d["config"]["seeds"]
                continue
            acc = base["methods"][method]
            n_a = acc.pop("_n", base["config"]["seeds"])
            n_b = d["config"]["seeds"]
            merged = {}
            for key in stats:
                if key.endswith("_std"):
                    continue
                base_key = key
                std_key = key + "_std"
                if std_key in acc and std_key in stats:
                    m, s, n = pool(acc[base_key], acc[std_key], n_a, stats[base_key], stats[std_key], n_b)
                    merged[base_key] = m
                    merged[std_key] = s
                else:
                    merged[base_key] = acc[base_key]  # non-numeric / non-stat field
            merged["_n"] = n_a + n_b
            base["methods"][method] = merged

    for method, stats in base["methods"].items():
        stats["n_seeds"] = stats.pop("_n", None)

    base["config"]["merged_from"] = args.inputs
    pathlib.Path(args.output).write_text(json.dumps(base, indent=2))
    print(f"wrote {args.output}")
    for method, stats in base["methods"].items():
        print(
            f"  {method:18s} n={stats.get('n_seeds')}  HV={stats['hypervolume']:.4f}"
            f"+-{stats['hypervolume_std']:.4f}  ctrl={stats['controllability_mean']:+.4f}"
            f"+-{stats['controllability_mean_std']:.4f}"
        )


if __name__ == "__main__":
    main()
