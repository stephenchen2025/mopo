# PaCE on a small language model — the result is negative

This is the experiment the synthetic bandit could not run: a 5.9M-parameter Qwen3 trained
from scratch, with sampled sequences, token-level credit assignment, and deliberately
incommensurable rewards (a 0/1 verifier against an exponential length term). It exists to
test the two mechanisms [`RESULTS.md`](RESULTS.md) records as unestablished.

**It does not support PaCE.** Full PaCE finishes last of six methods, its own ablations beat
it, and a simple MO-GRPO-style baseline wins. Reproduce:

```bash
python experiments/tiny_lm_rl.py --seeds 3 --steps 200 --pretrain-steps 500
```

## Setup

The policy is pretrained for 500 steps into the frontier window (see
`pace/envs/tiny_lm.py`), then each method trains from that same checkpoint at equal rollout
budget: 3 seeds x 200 steps x 8 rollouts x 2 prompts. Conditioning uses graded preference
markers the model learns during pretraining — the character-level model has no English, so
the verbal conditioning validated on real Qwen3-0.6B is unavailable to it.

The pretrained policy already traces a clean monotone frontier: **HV 0.5701,
controllability +0.975**, accuracy sweeping 0.167 → 1.000 as brevity falls 0.800 → 0.380.

## Results

| method | hypervolume | % of pretrained | controllability |
|---|---|---|---|
| *pretrained baseline (no RL)* | **0.5701** | 100% | **+0.975** |
| `cond_mognorm` | 0.5577 ± 0.0203 | 97.8% | +0.886 |
| `pace_uniform` | 0.5119 ± 0.0422 | 89.8% | +0.860 |
| `cond_linear` | 0.5063 ± 0.0216 | 88.8% | +0.876 |
| `pace_no_front` | 0.5018 ± 0.0194 | 88.0% | +0.748 |
| `pace_no_crossdir` | 0.4719 ± 0.0524 | 82.8% | +0.941 |
| **`pace` (full)** | **0.4647 ± 0.0620** | **81.5%** | +0.936 |

## What this says

**1. No method improves on the policy it started from.** Every one loses hypervolume, and
every one loses controllability. The best merely comes close to breaking even. RL is not
finding a better frontier here; at best it is not damaging the one pretraining produced.

**2. Full PaCE is the worst of the six, and its ablations beat it.** Removing the coverage
bandit (`pace_uniform`, 0.5119) beats the full method (0.4647). Removing frontier shaping
(`pace_no_front`, 0.5018) also beats it. When a method's ablations outperform it, the
components are not merely unproven — they are, on this evidence, actively costly.

**3. The cross-direction advantage matrix is still unestablished, now for a different
reason.** This environment was built to test it, since the synthetic bandit had no gradient
variance for it to reduce. Ablating it changes the result by **−0.0073, or 0.2 standard
errors** — statistically indistinguishable from zero. It neither helps nor hurts. The
mechanism costs `K²m` flops and a good deal of the design's complexity; nothing measured so
far justifies it.

**4. The coverage bandit shows a negative effect** (−0.0472, 1.1 SE). Not significant at
n = 3, but it is the third setting in a row where the bandit fails to earn its keep, and the
first where the point estimate is against it.

**5. A simple baseline wins again.** `cond_mognorm` — per-objective z-scoring then a
weighted sum — tops the table at 97.8% of baseline. This is the second independent setting
where a simple conditioned baseline beats PaCE's machinery; the first was three objectives
in the synthetic environment ([`M3_RESULTS.md`](M3_RESULTS.md)).

## Two harness bugs found on the way, one of which invalidated a whole run

**A missing KL anchor.** The first version of this experiment used a bare REINFORCE loss
while the shipped `PaCETrainer` uses a KL penalty (`kl_coef=0.02`) and ratio clipping
(`clip_range=0.2`). Without an anchor the policy drifts freely from a starting point that
already traces a good frontier, and **all six methods collapsed** — HV 0.5701 → 0.458–0.543,
controllability +0.975 → +0.75–0.94.

That run is not reported as a comparison, and it is worth being explicit about why: its
rankings were *favourable* to PaCE. The coverage bandit led uniform sampling by ~2.3 SE and
the cross-direction matrix led its ablation by ~1.5 SE — which would have been the first
positive evidence for either. Rankings measured under a broken optimizer are not evidence
about the estimators, and publishing them because they pointed the right way would have been
precisely the wrong instinct.

The one thing worth keeping from it: **a good frontier is fragile.** Without a KL anchor, RL
destroyed it in every configuration within 200 steps.

**Degradation returns with more steps.** With the anchor in place, a 60-step check put three
of four methods *above* baseline. At 200 steps they are all below it. So the anchor slows the
erosion rather than preventing it.

## The honest limitation

The pretrained policy is close to the best this task allows. The frontier was *created* by
pretraining on a mixture of worked and direct answers, so both ends of the trade-off are
already reachable and well-calibrated — accuracy already hits 1.000 at the accuracy end.
There is little headroom for RL to find and a lot to lose.

That makes this a valid test of *"can RL improve an already-good frontier"* and a poor test
of *"can RL discover a frontier"*. PaCE is designed for the second. A fair test needs a task
where the pretrained policy is genuinely mediocre — which is a different experiment, not a
re-tuning of this one, and it is the obvious next thing to build.

What the result does establish, and what should be weighed against any future positive
finding: across two independent settings with real headroom for the comparison
(three objectives synthetic, and this one), **PaCE's advantage estimator has never beaten a
simple conditioned baseline**, and in both it has been beaten by one.
