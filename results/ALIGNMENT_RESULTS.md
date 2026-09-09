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

## n=16 merged result (paired data, 48 eval problems): what held up

`experiments/merge_results.py` combined the two paired 8-seed sweeps (seeds 0-7 and 8-15,
`--seed-offset`) into n=16 per method. Welch's t-test on each metric:

| | `pace_no_front` | `pace_align_only` | diff | t | df | p |
|---|---|---|---|---|---|---|
| hypervolume | 0.5679 ± 0.0606 | 0.4928 ± 0.0712 | −0.0751 | −3.21 | 29.3 | **0.003** |
| controllability (Spearman) | +0.0243 ± 0.2568 | +0.2157 ± 0.1662 | +0.1914 | 2.50 | 25.7 | **0.019** |

**Hypervolume: `pace_no_front` is significantly higher.** This is the more mechanically
legible of the two results — the alignment term is a penalty on top of the achievement
objective, so trading away some raw hypervolume for whatever the penalty rewards is exactly
what adding a penalty should do. Unsurprising, but worth having measured rather than assumed.

**Controllability: `pace_align_only` is significantly higher, but treat this cautiously.**
The `pace_no_front` controllability SD nearly doubled between the two halves of this exact
sweep — 0.1324 at n=8 (seeds 0-7) to 0.2568 at n=16 (the full seeds 0-15) — meaning the n=8
estimate had already understated the true spread before any seeds were added. A single
Welch test does not diagnose this by itself; it is a caution to weight this p=0.019 less
than its face value, not a reason to discard the point estimate.

## Addendum: the steerability metric itself was broken, twice, before it was fixed

The n=16 merge above compared `controllability` (Spearman) against `steerability`. They
disagreed sharply: Spearman found a significant difference favoring `pace_align_only`
(p=0.019); the slope-based `steerability` found none (p=0.957, means 0.0108 vs 0.0106,
essentially identical). Since `steerability` was built specifically because Spearman was
distrusted, a disagreement between them needed resolving, not reporting as-is.

It resolved against `steerability`, not against Spearman. Two things were wrong with it:

1. **It diluted high-variance objectives.** Normalizing the slope by the raw observation
   range means an objective with large per-problem noise (a 0/1 accuracy check) gets
   divided by a large denominator that has nothing to do with steerability. Simulated: the
   identical injected direction-dependent shift read as −0.04 for a Bernoulli-like reward
   against +0.33 for a continuous one with the same effect — an 8x spurious gap from noise
   character alone. Since accuracy is one of the two objectives being averaged into
   `steerability_mean`, this alone could explain a near-zero aggregate reading.
2. **A first attempted fix (normalizing by the range of per-direction means, or a
   variance-decomposition debiased version of it) traded dilution for explosion.** The
   between-direction spread is itself a noisy estimate at only 9 directions, and it can be
   small by chance under the null. Dividing by it is unbounded: simulated under a genuine
   null, repeated 200 times, this version returned values up to 9.28.

The fix that actually works: **Pearson correlation on the per-direction means** — bounded to
`[-1, 1]` by construction, not diluted (correlation is computed on means, each already
averaged over every problem), and cannot explode. `pace/core/metrics.py::steerability` now
implements this. It is verified by five tests: boundedness under 200 null trials, resistance
to the dilution failure mode, correct behaviour on a genuinely flat (collapsed) response,
support for pre-aggregated input, and shape validation.

**One claim from earlier in this investigation is also retracted.** The original
justification for replacing Spearman included "it returns exactly 0.0 whenever achieved
rewards happen to be constant, which happened in 10 of 24 runs" as though this were a
Spearman-specific artifact. Checked directly: quantized per-direction means tie often (479
of 500 simulated trials had at least one tied mean) but scipy's average-rank tie handling
means Spearman returns exactly 0.0 from that only rarely (4 of 500). The exactly-0.000
values seen in real training runs were most likely genuine policy collapse — every
direction producing a literally identical mean under deterministic greedy decoding — which
is the *correct* reading, not a measurement artifact, and Pearson-on-means would report the
same 0.0 in that exact case for the same reason: a constant array has no correlation with
anything, regardless of which formula computes it.

**What this means for the numbers already reported.** All `steerability_mean` values in
this document and in `TINY_LM_RESULTS.md` used the flawed (diluted) formula and should be
disregarded — not reinterpreted, disregarded. The `controllability` (Spearman) numbers are
not implicated by this specific bug, though the general small-D sampling-noise caveat below
still applies to them.

**A re-measurement with the corrected metric has not yet been run.** Observations are not
persisted from a completed sweep — recomputing a corrected metric on old data is not
possible without retraining, since training is what produces the model whose outputs get
evaluated. Whether Pearson-on-means agrees or disagrees with Spearman's p=0.019 finding on
real training data is open. Simulation at realistic noise levels found the two correlate
similarly rather than one saturating well before the other, so the base rate expectation is
agreement rather than another reversal — but that is a prediction from synthetic data, not
a measurement, and is reported as exactly that.

## A second addendum: correlation-type controllability metrics have an intrinsic floor at D=9

Independent of which correlation formula is used, repeating a genuine null 200 times at
`D = 9` directions gives a standard error around 0.35–0.40 for *any* correlation coefficient
computed over 9 points — this is the sampling distribution of a correlation coefficient at
small `n`, not a property of Spearman, Pearson, or any normalization choice.

This means the earlier recommendation to add more evaluation *problems* (which this
document's parent experiment did, 24 → 48) sharpens each of the 9 per-direction means but
does not touch this floor. Only more evaluation *directions* — increasing `--eval-partitions`
— would. This is now the more precisely targeted next step, in place of the more generic
"more evaluation problems" recommendation given earlier.

## Resolution: re-verified on real data at D=17, and it holds

Both open items from the previous addenda were checked together: the corrected
Pearson-on-means `steerability` re-run on real training data (never done before — the
original comparison used the flawed v1 formula), and more evaluation directions
(`--eval-partitions 16`, giving `D = 17` instead of 9) to address the small-D noise floor.
Same models, same seeds 0-7, only the evaluation grid changed.

### The two metrics now agree

| | `pace_no_front` | `pace_align_only` | diff | t | p |
|---|---|---|---|---|---|
| controllability (Spearman) | +0.025 ± 0.134 | +0.210 ± 0.144 | +0.185 | 2.67 | **0.018** |
| steerability (Pearson, corrected) | +0.086 ± 0.103 | +0.212 ± 0.120 | +0.126 | 2.26 | **0.041** |

Both point the same direction, both are significant, and for `pace_align_only` the two
numbers are close enough to be the same measurement (+0.210 vs +0.212). At `D = 9` with the
broken formula these read +0.0243 and +0.0106 — not just noisy, but wrong. The prediction
made two addenda ago — that a correctly computed steerability would most likely agree with
Spearman rather than disagree again — held up against real data, not just simulation.

### More directions tightened the estimate, confirmed on real data, even at half the seeds

| controllability SD | D=9, n=16 (merged) | D=17, n=8 (fresh) |
|---|---|---|
| `pace_no_front` | 0.257 | **0.134** |
| `pace_align_only` | 0.166 | 0.144 |

Both dropped despite using *half* as many seeds. Spending measurement budget on more
directions was more efficient here than spending it on more seeds — consistent with the
theoretical floor argument, now checked against trained policies rather than only simulated
nulls.

### Why hypervolume was exactly unchanged, and that's not a coincidence

`pace_no_front` and `pace_align_only` both post the identical hypervolume at D=17 as they
did at D=9, to four decimal places, for the same seeds. This is because the tiny LM's
conditioning channel is five discrete preference markers (`PREF_MARKERS = "VWXYZ"`) — every
requested `w_accuracy`, however finely sampled, rounds to one of five buckets before it
reaches the model. Both a 9-direction and a 17-direction grid query the exact same five
underlying behaviours; the 17-direction grid just re-samples them with more independent
draws. That is precisely why the extra directions sharpen the correlation-based metrics
(more independent looks at the same five-point relationship) without changing the
achievable frontier at all (there is nothing more to achieve than those five points already
capture). A future experiment wanting finer-grained conditioning resolution than this would
need more preference markers, not a finer evaluation grid.

### Where this leaves the alignment term

Hypervolume cost is established at n=16 (previous addendum, p=0.003 — and unaffected by
today's change, per the point above). Controllability benefit is now established two ways
at n=8/D=17 (p=0.018 and p=0.041, agreeing). **The trade is real in both directions**, not
an artifact of an underpowered or broken measurement on either side. `lambda_align` remains
defaulted to 0.0 pending a decision on whether that trade is worth taking for a given use
case; the honest summary is that it is a genuine, working preference-diversity mechanism
that costs the raw-hypervolume-maximizing behaviour it is designed to prevent.
