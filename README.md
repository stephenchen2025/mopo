# PaCE — Pareto Coverage Exploration

**A group-relative RL algorithm that trains one policy to cover a Pareto frontier, instead
of N runs to sketch N points on it.**

Standard post-training RL collapses a vector of objectives — correctness, brevity, safety,
latency, cost — into one scalar with fixed weights and optimizes that. You get one
trade-off per training run, whichever objective has the loudest reward tends to eat the
gradient, and if the Pareto front is concave, a weighted sum *provably cannot reach its
interior* no matter how you tune the weights. PaCE conditions one policy on a preference
direction `w`, makes each rollout group heterogeneous in `w`, and replaces the scalarized
z-score advantage with a rank-based Tchebycheff achievement compared across every
direction in the group.

Full algorithm spec, the math, and honest positioning against MO-GRPO / PRPO / Panacea /
dynamic reward weighting: **[DESIGN.md](DESIGN.md)**.
Measured results: **[results/RESULTS.md](results/RESULTS.md)**.

> The repository is `mopo` (multi-objective policy optimization); the algorithm and the
> importable package are `pace`. Rename the package if you would rather they match.

---

## Status — what is verified and what is not

Being precise about this, because the repo contains both.

| Component | Status |
|---|---|
| `pace/core/*` — frontier math, scalarization, directions, advantages | **Tested.** 57 tests, including hypervolume against a brute-force oracle and finite-difference gradient checks |
| `pace/envs/*` — synthetic environment + trainer | **Runs.** Full experiment reproduces on CPU in minutes; exact oracle cross-checked against brute-force enumeration |
| `experiments/synthetic_frontier.py` | **Run.** Numbers in `results/` come from it |
| `pace/llm/rewards.py`, `pace/llm/conditioning.py` | **Tested.** Pure Python, no GPU needed |
| `pace/llm/trainer.py` | **Executed, not GPU-verified.** Driven end-to-end against a 37k-parameter randomly-initialized model with a stub tokenizer (`tests/test_llm_trainer.py`, in CI). That found and fixed a real bug: policy dropout was live during both log-prob passes, making the importance ratio noise. Never run against a real checkpoint |
| `pace/llm/trl_adapter.py`, `scripts/train_llm.py` | **Not executed.** Config parsing *is* tested |
| Gemma 4 / Muse Glimmer results | **None exist.** No real LLM has been trained with this |

The synthetic results validate the **optimizer's frontier behaviour**. They say nothing
about language models.

## Install and run

```bash
pip install -e .            # core only: numpy
pip install -e '.[dev]'     # + pytest
pip install -e '.[llm]'     # + torch, transformers, peft, datasets

pytest                                            # 57 tests (5 need torch)
python experiments/synthetic_frontier.py --seeds 16 --steps 1500
python scripts/train_llm.py --config configs/gemma4_e4b.yaml --dry-run
```

## What the numbers say

Three settings have been measured. Read them together, because they do not agree.

| setting | winner | PaCE's position |
|---|---|---|
| Synthetic bandit, **2 objectives**, concave front | `pace` | **best** (58% of oracle vs 27% for N fixed-weight runs) |
| Synthetic bandit, **3 objectives** ([M3](results/M3_RESULTS.md)) | `cond_linear` | beaten in both geometries |
| **Small language model** ([tiny-LM](results/TINY_LM_RESULTS.md)) | `cond_mognorm` | **last of six**, beaten by its own ablations |

**The honest summary: PaCE's advantage estimator has never beaten a simple conditioned
baseline outside the two-objective synthetic case, and in the two settings closest to a real
LLM it has been beaten by one.** On the small language model the full method finishes last
of six, and removing either the coverage bandit or the frontier shaping *improves* it.

**What does hold up:**

- **Conditioning beats N separate runs.** One preference-conditioned policy beats N
  independently-trained fixed-weight models at equal budget, everywhere it was measured, and
  the margin widens with more objectives. This is a claim about *conditioning*, not about
  PaCE — `cond_linear` and `cond_mognorm` are conditioned too.
- **Rank normalization is scale-invariant, and that is worth something.** Applying `exp(4x)`
  to one objective moves the Pareto set by zero points; it leaves PaCE bit-identical and
  collapses a weighted-sum baseline 5×, identically on all 20 seeds. Real reward suites mix
  verifiers, learned reward models and cost terms with incommensurable scales.
- **Verbal conditioning works and numeric conditioning does not.** On real Qwen3-0.6B,
  `<preference>accuracy=0.70 brevity=0.30</preference>` gives **+0.000** controllability;
  the same weights rendered as instructions give **+0.860**
  ([details](results/REAL_MODEL_CONDITIONING.md)).

**A metric warning that outranks the rest.** In the one setting where RL had real room to
improve ([headroom experiment](results/HEADROOM_RESULTS.md)), every method **more than
doubled hypervolume while becoming less steerable than it started** — converging to a single
operating point with the trade-off dimension pinned constant, nine directions collapsing to
three points. Hypervolume rewards that: one dominant point beats a well-spread weaker set.
**Do not report hypervolume for a multi-objective policy without a steerability measure
beside it.** Controllability has now caught a failure that other metrics celebrated in three
separate settings.

PaCE variants resist the collapse better than plain baselines (all four keep positive
controllability; both baselines go negative), which is a real if modest point in their
favour. None retains the conditioning it started with.

**How much of this is resolvable.** An 8-seed controlled comparison
([ALIGNMENT_RESULTS.md](results/ALIGNMENT_RESULTS.md)) found the language-model testbed has a
per-seed SD of 0.257 in controllability and 0.197 in hypervolume — identical configurations
producing hypervolumes from 0.0000 to 0.6209. Three seeds resolve a difference of ~0.42;
the method differences reported from 3-seed runs were 0.03–0.15. **The LM method rankings in
this repo are not statistically supported**, and resolving a 0.10 controllability difference
would need 104 seeds per arm (12.5 hours per contrast). The synthetic results, run at 20
seeds with an order of magnitude less spread, are unaffected. A direction-alignment term
built to prevent the collapse was tested and **did not help** (4/8 collapses with it, 2/8
without); it ships defaulted off.

**What is not established:** the cross-direction advantage matrix (ablating it moves the
small-LM result by 0.2 standard errors) and the coverage bandit (no benefit in three
settings, and a negative point estimate in the most realistic one).

## The algorithm in one page## The algorithm in one page

One policy `π_θ(y | x, w)` conditioned on a preference direction `w` on the simplex. Per
prompt, generate `K` rollouts under `K` *different* directions, then:

1. **Rank-normalize** rewards within the group, per objective. Invariant to *any* strictly
   increasing rescaling of an objective — not just affine, which is all a z-score survives.
   Removes the "loudest reward wins" failure mode by construction rather than by tuning.

2. **Smooth Tchebycheff achievement** instead of a weighted sum:
   `s_μ(u; w) = −μ·log Σ_j exp(w_j(1−u_j)/μ) + ρ·Σ_j w_j u_j`.
   Every Pareto-optimal point maximizes this for some `w`, including points a weighted sum
   can never select.

3. **Cross-direction advantage matrix.** Scalarizing an already-computed reward *vector* is
   free, so score every rollout under every direction: `S[k,l] = s_μ(u_k; w_l)`, a `K×K`
   matrix for `K²m` flops and zero extra sampling. Rollout `k`'s advantage is its own
   achievement minus the leave-one-out mean of the *others* under the *same* direction:

   ```
   A_ach[k] = S[k,k] − (1/(K−1))·Σ_{l≠k} S[l,k]
   ```

   A rollout generated to be *short* still supplies baseline evidence for the *accurate*
   direction. Ordinary groups discard that.

4. **Frontier shaping** — non-dominated rank plus crowding distance within the group's
   first front, z-scored and mixed in with weight `λ_f`.

5. **Coverage bandit** over the direction simplex — UCB over weakness, improvement rate,
   and visit count, with an ε-uniform floor so no direction starves.

## Which base model — and why not Muse Spark

**Use Gemma 4** (Apache 2.0; E2B / E4B / 12B / 26B-A4B / 31B). E2B or E4B for iteration,
12B for the headline number.

**Muse Spark cannot be used as a training baseline.** It is Meta's proprietary, API-only
frontier model. Policy-gradient RL needs per-token log-probabilities and a gradient path
through the policy; an inference API provides neither. There is no version of GRPO — or
any RL algorithm in this family — that you can run against it. It has exactly one usable
role here: an LLM-judge reward for subjective objectives, which is supported but off by
default, because paying per call is a poor fit for a method that needs `K` rollouts per
prompt per step.

**Muse Glimmer 30B** (Meta, Apache 2.0) *is* trainable and is the right external-validity
check once the method works — its agent tuning suits tool-use trade-offs where
accuracy/latency/cost tension is real. But 30B dense makes every rollout expensive, and
PaCE is rollout-hungry by construction, so it is the wrong place to iterate.

Fuller reasoning in [DESIGN.md §4](DESIGN.md#4-which-base-model).

## Layout

```
pace/core/          numpy-only algorithm: pareto, scalarization, directions, advantages, metrics
pace/envs/          synthetic multi-objective bandit with tunable front curvature
pace/llm/           conditioning, reward suite, trainer, TRL adapter
experiments/        synthetic_frontier.py — the validation runs
configs/            gemma4_e4b, gemma4_12b, muse_glimmer_30b
tests/              57 tests; the LLM-trainer ones skip without torch
results/            measured output
.github/workflows/  CI: tests on 3.10-3.12, black, and a short run of the real experiment
```

## Using PaCE inside an existing TRL setup

```python
from pace.llm.trl_adapter import make_bandit, sample_group_directions, pace_advantage_fn

bandit = make_bandit(n_objectives=3)
idx, W = sample_group_directions(bandit, group_size=8)   # one direction PER ROLLOUT
# ... generate one completion per direction, collect rewards as a (K, m) matrix ...
advantages = pace_advantage_fn(rewards, W)
```

Two requirements, or it silently degrades to an ordinary group baseline:

- The `K` rollouts must use `K` **different** directions. Share one direction across the
  group and `S` becomes rank-one — the mechanism is gone. `tests/test_advantages.py`
  asserts exactly this degeneration.
- Rewards must arrive as a `(K, m)` **matrix**, not a pre-summed scalar. Collapsing reward
  functions into a scalar throws away the information PaCE runs on.

## License

MIT.
