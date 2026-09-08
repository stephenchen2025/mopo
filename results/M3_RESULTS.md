# PaCE at three objectives — where the two-objective headline does not hold

Every experiment in [`RESULTS.md`](RESULTS.md) uses **two** objectives. Every shipped LLM
config uses **three** (`accuracy`, `brevity`, `format`). That gap turned out to matter, so
it is recorded here rather than left for someone to discover after booking a GPU.

Reproduce:

```bash
python experiments/synthetic_frontier.py --objectives 3 --seeds 8 --steps 600 \
    --action-resolution 8 --partitions 6 --only main
```

## Measured

8 seeds, 600 steps, equal rollout budget, greedy decoding, 45 actions, 4 contexts.

### Concave front (oracle HV = 0.1458)

| method | hypervolume | % of oracle | controllability |
|---|---|---|---|
| `cond_dominance` | 0.0426 ± 0.0203 | 29.2% | **+0.038** |
| `cond_linear` | 0.0396 ± 0.0021 | 27.1% | +0.852 |
| `pace_linear` | 0.0313 ± 0.0046 | 21.5% | +0.800 |
| `cond_mognorm` | 0.0281 ± 0.0064 | 19.3% | +0.822 |
| `pace_uniform` | 0.0255 ± 0.0047 | 17.5% | +0.800 |
| `pace` | 0.0251 ± 0.0040 | 17.2% | +0.803 |
| `pace_no_front` | 0.0205 ± 0.0031 | 14.0% | +0.748 |
| `fixed_scalar` | 0.0100 ± 0.0044 | 6.9% | +0.824 |

### Convex front (oracle HV = 0.4420)

| method | hypervolume | % of oracle | controllability |
|---|---|---|---|
| `cond_linear` | 0.2527 ± 0.0244 | 57.2% | +0.772 |
| `pace_no_front` | 0.2293 ± 0.0127 | 51.9% | +0.754 |
| `cond_mognorm` | 0.2141 ± 0.0338 | 48.4% | +0.784 |
| `pace` | 0.2069 ± 0.0213 | 46.8% | +0.836 |
| `pace_uniform` | 0.2015 ± 0.0278 | 45.6% | +0.821 |
| `cond_dominance` | 0.1961 ± 0.0216 | 44.4% | **−0.155** |
| `pace_linear` | 0.1462 ± 0.0454 | 33.1% | +0.802 |
| `fixed_scalar` | 0.0833 ± 0.0168 | 18.8% | +0.970 |

## What this says

**PaCE's specific machinery does not lead at three objectives.** A plain conditioned
weighted sum (`cond_linear`) wins in both geometries. At `m = 2` PaCE led on the concave
front (58.4% vs 54.5%); that lead does not survive the extra objective.

**It is not undertraining.** Quadrupling the budget leaves the ordering flat:

| | 600 steps | 2400 steps |
|---|---|---|
| `cond_linear` | 26.8% | 25.1% |
| `pace` | 17.4% | 18.8% |
| `pace_uniform` | 17.2% | 17.1% |
| `fixed_scalar` | 6.7% | 8.6% |

**What does survive, at both `m = 2` and `m = 3`, is the claim about conditioning.**
`fixed_scalar` — N independent fixed-weight runs, the status quo — is worst by a wide
margin everywhere (6.9% and 18.8% here, 26.7% at `m = 2`). One conditioned policy beats N
separately-trained models at equal rollout budget, and the margin *widens* with more
objectives, which is what you would expect: the simplex needs exponentially more grid
points to cover as `m` grows, so the fixed-weight ensemble's coverage degrades fastest.

That is a real result. But it is a result about **conditioning**, not about PaCE's
advantage estimator — `cond_linear` is conditioned too, and at `m = 3` it is better.

**A prediction of mine that the data contradicts.** I argued from geometry (`DESIGN.md`
§2.2) that because rank normalization only linearizes the front at `m = 2`, Tchebycheff
should be *more* load-bearing at `m = 3`. The geometric fact holds — rank-vector sums
spread 1.22–1.67 at `m = 3` against a constant 1.0 at `m = 2`. The predicted consequence
does not: `pace_linear` beats `pace` on the concave front here (21.5% vs 17.2%), so
Tchebycheff is *hurting*. The inference from geometry to training outcome was wrong, and
the geometric fact alone does not license it.

**Controllability keeps earning its place.** `cond_dominance` tops the concave table at
29.2% with a preference correlation of **+0.038** — the knob does nothing. On the convex
front its correlation is **−0.155**: the knob runs backwards. Ranked on hypervolume alone
it would look like the best method on the board.

## What would have to change for PaCE to be worth its complexity

On this evidence, honestly: either it needs to win somewhere `cond_linear` cannot follow,
or it should be cut back to the parts that pay.

1. **Adversarial reward scaling is the one place it clearly wins.** Rank normalization is
   the component with a robust, general advantage (`RESULTS.md` §3): a monotone transform
   that does not move the Pareto set at all collapses `cond_linear` 5× and leaves PaCE
   untouched. Real reward suites mix verifiers, learned reward models and cost terms with
   incommensurable scales, which is exactly that regime — but this environment's rewards
   are already well-scaled, so the benefit never shows up in the headline numbers.
2. **The LLM setting is still untested and is where two mechanisms become testable at
   all** — the cross-direction advantage matrix is a variance-reduction device, and this
   environment has almost no gradient variance to reduce.
3. **If neither pays off, the honest conclusion is that conditioning plus a weighted sum is
   the right default**, and PaCE's contribution narrows to rank normalization.
