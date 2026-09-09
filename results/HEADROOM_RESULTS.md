# When RL has a frontier to discover, it collapses it instead

The [first language-model experiment](TINY_LM_RESULTS.md) was a poor test: pretraining had
already built the frontier, so RL could only preserve or damage it. This one fixes that. The
start policy is competent (accuracy ~0.42) but **weakly conditioned** — it produces nearly
identical behaviour at every direction, controllability **+0.187**. A separately trained,
strongly-conditioned reference reaches controllability **+0.975**. RL therefore has to *learn
the direction→style routing from reward alone*, which is what PaCE claims to do.

Reproduce:

```bash
python experiments/tiny_lm_rl.py --n-digits 2 --pretrain-steps 500 \
    --start-floor 0.45 --start-span 0.10 --ceiling-steps 500 \
    --seeds 3 --steps 200 --eval-problems 24
```

## The headline result

| | hypervolume | controllability |
|---|---|---|
| start policy | 0.2580 | **+0.187** |
| supervised reference (strong conditioning) | 0.2996 | **+0.975** |
| `cond_linear` | **0.6312 ± 0.0325** | **−0.055** |
| `pace_uniform` | 0.6204 ± 0.0277 | +0.101 |
| `pace` | 0.5860 ± 0.0282 | +0.029 |
| `cond_mognorm` | 0.5860 ± 0.0743 | **−0.118** |
| `pace_no_front` | 0.5742 ± 0.0322 | **+0.156** |
| `pace_no_crossdir` | 0.5285 ± 0.0426 | +0.062 |

Read the first column alone and this is a triumph: RL **more than doubles** hypervolume,
0.2580 → 0.53–0.63, far past the supervised reference. Read the second column and it is a
failure: **every method ends less steerable than it started**, and two end steerable in the
*wrong direction*.

## What actually happened

Per-direction frontier for `pace` before and after 200 RL steps:

| `w_acc` | before: accuracy / brevity | after: accuracy / brevity |
|---|---|---|
| 0.00 | 0.417 / 0.526 | 0.583 / **0.828** |
| 0.25 | 0.458 / 0.546 | 0.583 / **0.828** |
| 0.50 | 0.375 / 0.567 | 0.625 / **0.828** |
| 0.75 | 0.458 / 0.546 | 0.625 / **0.828** |
| 1.00 | 0.458 / 0.526 | 0.667 / **0.828** |

**Brevity is identical at every direction. Nine directions produce three distinct points.**
The policy found one good behaviour and applied it everywhere. Hypervolume rose because a
single strong point (0.62 × 0.83) dominates a spread of weak ones (0.42 × 0.55) — the
"frontier" after training is a dot.

## The methodological finding

**Hypervolume is not a sufficient objective for frontier coverage when the policy can move
the frontier outward.** It rewards collapsing to one dominant operating point over
maintaining a well-spread but individually weaker set. This is not a quirk of the metric's
implementation; it follows from its definition, and it applies to any multi-objective RL
result that reports hypervolume without a coverage or steerability measure alongside.

It did not surface in the synthetic bandit because the policy there had little absolute
headroom, so spreading along the frontier was the *only* way to gain hypervolume. Give RL
room to improve absolutely and the incentive inverts.

Controllability caught it. Hypervolume alone would have reported a large win. That is now
three separate settings where controllability exposed a failure the other metrics
celebrated — `cond_dominance` in both synthetic experiments, and every method here.

## What this says about PaCE

A modest point in its favour, and it should not be oversold: **all four PaCE variants keep
positive controllability (+0.029 to +0.156); both plain baselines go negative** (−0.055,
−0.118). The frontier-shaping and conditioning machinery does resist collapse better than
plain scalarization does.

But the effect is small, the seed spread is wide (a single-seed rerun of `pace` gave +0.468),
and **no method retains even the weak conditioning it started with**, let alone approaches
the +0.975 the supervised reference reaches. Whatever PaCE's advantage estimator is doing, it
is not enough to keep a policy steerable under reward pressure.

`pace_no_front` — the ablation with frontier shaping *removed* — has the best controllability
of the six. That is the opposite of the mechanism's stated purpose.

## Caveats

- Three seeds, wide spreads. The controllability ordering within the PaCE variants is not
  resolved at this sample size; the PaCE-positive / baseline-negative split is the part that
  looks robust.
- The "supervised reference" is not an upper bound — every method exceeded its hypervolume.
  It is a reference point for *what conditioned supervision achieves*, and the interesting
  comparison is against its controllability (+0.975), which no RL method came close to.
- 200 steps. Longer training might recover conditioning, though the trend across the first
  experiment (60 steps above baseline, 200 steps below) points the other way.
- The obvious fix to try next is an explicit coverage or spread term in the objective, so
  that collapsing to a point is penalized rather than rewarded. PaCE's crowding term is meant
  to do this within a group; it evidently does not survive to the deployed policy.
