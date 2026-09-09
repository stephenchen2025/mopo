# The direction-alignment term does not work — and the testbed cannot resolve it anyway

Two results. The first is a negative result about a mechanism designed to fix a measured
failure. The second is more important and retroactively limits what this repository's
language-model experiments can claim at all.

## 1. The alignment term does not prevent collapse

[`HEADROOM_RESULTS.md`](HEADROOM_RESULTS.md) measured RL doubling hypervolume while every
method became less steerable — collapsing to one behaviour emitted at every direction. The
direction-alignment term (`lambda_align`, `pace/core/advantages.py`) was designed against
that failure: the diagonal of the double-centered achievement matrix, which is exactly zero
for a collapsed group and positive for a direction-matched one. The mechanism is verified on
constructed cases in `tests/test_advantages.py`.

It does not help. The controlled comparison — `pace_no_front` vs `pace_align_only`, identical
but for `lambda_align`, 8 seeds each:

| | collapsed runs | controllability | hypervolume |
|---|---|---|---|
| alignment **off** | **2/8** | −0.025 ± 0.303 | 0.441 ± 0.182 |
| alignment **on** | **4/8** | +0.042 ± 0.211 | 0.384 ± 0.212 |

At three seeds across the wider grid, alignment looked good: 1/6 collapses against 9/18
without it. At eight seeds on the clean contrast it is 4/8 against 2/8 — pointing the other
way. The three-seed signal was noise, and it happened to favour the mechanism I had just
built.

`lambda_align` stays at its default of 0.0. The code and tests remain because the mechanism
is sound in principle and cheap, but nothing here justifies turning it on.

## 2. The testbed cannot resolve the differences being reported

The eight-seed run exposed the real per-seed spread, which three seeds had badly
underestimated. Identical configurations differing in one flag produced hypervolumes from
**0.0000 to 0.6209** and controllabilities from **−0.725 to +0.303**.

| measured at n = 8 | per-seed SD |
|---|---|
| controllability | 0.257 |
| hypervolume | 0.197 |

Seeds per arm for 80% power at α = 0.05:

| to detect | controllability | hypervolume |
|---|---|---|
| 0.20 | 26 | 16 |
| 0.10 | 104 | 61 |
| 0.05 | 415 | 244 |

At ~3.6 minutes per run, resolving a 0.10 controllability difference costs **12.5 hours for a
single two-arm contrast**.

**Three seeds resolve a difference of about 0.42. The method differences reported from
three-seed runs were 0.03 to 0.15** — between three and fourteen times smaller than the
design could detect.

### What this retracts

The method orderings in [`TINY_LM_RESULTS.md`](TINY_LM_RESULTS.md) and
[`HEADROOM_RESULTS.md`](HEADROOM_RESULTS.md) are **not supported**. Specifically:

- "Full PaCE finishes last of six, and its own ablations beat it" — the spread between those
  six methods (0.4647 to 0.5577) is well inside noise at n = 3.
- "All four PaCE variants keep positive controllability; both plain baselines go negative" —
  not resolvable. `pace_no_front` came out at +0.156 with three seeds and −0.025 with eight.
- "The cross-direction matrix moves the result by 0.2 standard errors" — the direction is
  right (no detectable effect) but the standard error itself was underestimated.

Those documents now carry this caveat inline. The qualitative findings in them stand; only
the between-method rankings fall.

### What survives

Effects large enough to clear the resolution limit, or that are not between-method
comparisons at all:

- **Verbal vs numeric preference conditioning: +0.860 against +0.000** on real Qwen3-0.6B
  ([`REAL_MODEL_CONDITIONING.md`](REAL_MODEL_CONDITIONING.md)). An effect of 0.86 against a
  0.42 resolution, greedy-decoded, and mechanistically explained.
- **The collapse phenomenon itself.** Brevity identical at all nine directions, nine
  directions producing three distinct points. A property of one policy, not a comparison.
- **Hypervolume rewards collapse.** This follows from the metric's definition; the experiment
  demonstrates it rather than establishing it statistically.
- **Everything from the synthetic environment**, which was run at 20 seeds with per-seed SD
  of 0.02–0.07 — an order of magnitude tighter than the LM testbed.
- **All four bug findings** (policy dropout, `extract_answer`, evaluation problem drift, the
  token-weighted conditioning trap), which are deterministic.

### What to do about it

Do not add more seeds to this testbed hoping to resolve method rankings; the arithmetic above
says that is not affordable. Reduce the variance instead:

1. **More evaluation problems.** 24 problems at 9 directions is a small sample, and the
   metric is a rank correlation over 9 points, which is inherently coarse — several runs
   returned controllability of exactly 0.000 because achieved rewards were constant.
2. **Longer or better-regularized RL.** Hypervolumes spanning 0.0 to 0.62 across seeds means
   some runs diverge outright; that instability is most of the variance.
3. **A finer controllability measure.** Spearman over 9 directions saturates and quantizes.
   A regression slope of achieved reward on requested weight, using per-problem rather than
   per-direction aggregates, would be far less coarse.

Until then, this testbed can support qualitative claims about *whether* a policy collapses,
and cannot support claims about which advantage estimator collapses less.
