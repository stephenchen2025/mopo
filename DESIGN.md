# PaCE: Pareto Coverage Exploration

**A group-relative RL algorithm that trains one policy to cover a Pareto frontier, instead of
one point on it.**

Status: algorithm specified, numpy core implemented and unit-tested, validated on a synthetic
multi-objective environment (CPU). The LLM training path is written but **not** GPU-verified —
see `README.md` for the honest status table.

---

## 1. The problem

Standard post-training RL (PPO, GRPO, RLOO, GSPO) collapses a vector of objectives
`r = (r_1, ..., r_m)` — correctness, brevity, safety, latency, style, cost — into one scalar,
usually with fixed weights `w`:

```
r_scalar = Σ_j w_j r_j
```

and then optimizes it. Three things go wrong.

**(a) You get a point, not a frontier.** One run buys one trade-off. Want a different
accuracy/cost balance? Retrain. The frontier is discovered by brute force: N runs, N models,
N deployments.

**(b) Linear scalarization is structurally blind to concave frontier regions.** A weighted sum
can only ever find solutions that lie on a supporting hyperplane of the attainable set. If the
Pareto front is concave (dented inward), *no* weight vector recovers its interior — the optimizer
snaps to an extreme. This is not a tuning problem, it is a geometric fact, and concave fronts are
common exactly where objectives are sharply antagonistic. This is why the middle of a trade-off
so often feels unreachable: "either it's accurate or it's short, nothing in between."

**(c) Whichever objective has the largest reward variance eats the gradient.** With raw scalar
sums, the loudest reward dominates and the others become decoration — a well-documented
multi-objective reward-hacking mode.

## 2. What PaCE does

One policy, conditioned on a preference direction: `π_θ(y | x, w)` where `w ∈ Δ^{m-1}`.
One training run. At inference you dial `w` and move along the frontier.

Five mechanisms, in the order they apply to a batch:

| # | Mechanism | Fixes |
|---|-----------|-------|
| 1 | Heterogeneous groups: each rollout in a group gets its *own* direction `w_k` | (a) one run covers the frontier |
| 2 | Rank-space reward normalization (within group, per objective) | (c) monotone-invariant, not just affine |
| 3 | Smooth Tchebycheff achievement instead of a weighted sum | (b) concave regions become reachable |
| 4 | Cross-direction advantage matrix (K×K leave-one-out) | variance — every rollout informs every direction |
| 5 | Coverage bandit over the direction simplex | training budget goes to weak/improvable frontier regions |

### 2.1 Heterogeneous groups

Group-based RL already generates `K` rollouts per prompt. Standard practice: all `K` share one
scalarization. PaCE samples `K` *different* directions `w_1..w_K` and generates
`y_k ~ π_θ(· | x, w_k)`. The group stops being `K` samples of one behavior and becomes a
`K`-point sketch of the model's local frontier for this prompt.

### 2.2 Rank-space normalization

For objective `j`, replace the raw reward with its within-group normalized rank:

```
u_k[j] = (rank of r_k[j] among {r_1[j] .. r_K[j]}) / (K - 1)  ∈ [0, 1]      (ties: average rank)
```

Per-objective z-scoring (the MO-GRPO fix) is invariant to *affine* rescaling of a reward.
Ranks are invariant to *any strictly increasing* transform. That matters because real reward
functions are not affine-comparable: a log-probability head, a 0/1 verifier, and a
`-tokens/1000` cost term have incommensurable and non-Gaussian scales, and one of them being
heavy-tailed is enough to swamp a z-score.

The cost is real and deliberate: ranks discard magnitude *within* a group, so "barely correct"
and "emphatically correct" look the same. `rank_blend ∈ [0,1]` mixes rank with min-max scaling
if you want some magnitude back.

**An interaction worth knowing about, found by measuring rather than by design.** Rank
normalization maps any monotone two-objective front onto evenly spaced ranks, which means a
*concave* front becomes exactly linear in rank space. So ranks already remove the weighted sum's
pull toward the extremes — but they replace it with a different problem: under a balanced weight
every point then ties exactly, and a weighted sum is perfectly *indifferent*, supplying no signal
about which trade-off to select. Tchebycheff breaks that tie strictly in favour of the balanced
point. The two mechanisms are therefore not redundant and neither is sufficient alone: without
ranks you get collapse to an extreme, without Tchebycheff you get an undirected drift. The drift
is visible in the experiments as `pace_linear` having the widest seed-to-seed spread of any
variant. Asserted in `tests/test_scalarization.py`.

### 2.3 Smooth Tchebycheff achievement

With `u ∈ [0,1]^m` and ideal point `z* = 1` (the best attainable rank), the weighted Tchebycheff
achievement to **maximize** is

```
s(u; w) = − max_j  w_j · (1 − u_j)
```

Unlike a weighted sum, every Pareto-optimal point is the unique maximizer of `s(·; w)` for some
`w` — including points in concave regions. The `max` is non-smooth, so we use the log-sum-exp
smoothing (Lin et al., *Smooth Tchebycheff Scalarization*), plus an augmentation term with small
`ρ` that rules out weakly-Pareto solutions:

```
s_μ(u; w) = −μ · log Σ_j exp( w_j (1 − u_j) / μ )   +   ρ · Σ_j w_j u_j
```

`μ → 0` recovers exact Tchebycheff; `μ` large behaves more like a smooth mean. Default `μ = 0.1`,
`ρ = 0.05`.

### 2.4 Cross-direction advantage matrix

Here is the piece that makes heterogeneous groups pay for themselves.

Rewards are *vectors*, and scalarizing a vector is free. So having generated `K` rollouts under
`K` directions, we can evaluate **every rollout under every direction** at no additional sampling
cost:

```
S[k, l] = s_μ( u_k ; w_l )        # K × K achievement matrix
```

Rollout `k` was generated under `w_k`, so its own achievement is the diagonal `S[k,k]`. Its
baseline is the leave-one-out mean of the *other* rollouts scored under the *same* direction:

```
A_ach[k] = S[k, k] − (1 / (K−1)) · Σ_{l ≠ k} S[l, k]
```

This is a control variate in the RLOO sense: the baseline is computed only from rollouts other
than `k`, so it does not contain `y_k`'s own achievement, which is the dominant source of
correlation between a baseline and the action it scores.

**It is not strictly unbiased, and the reason is worth stating plainly.** The utilities `u` come
from a normalization computed over the whole group, rollout `k` included. Perturbing `y_k`'s raw
reward can shift the other rollouts' ranks and therefore move `k`'s own baseline. So the
independence holds *conditional on the normalization*, not unconditionally. This is not specific
to PaCE — every group-relative method that normalizes within the group inherits exactly this
coupling, standard GRPO's within-group z-score included, and there is a growing literature on the
resulting bias. Both halves of this are asserted in `tests/test_advantages.py`
(`test_leave_one_out_baseline_excludes_the_rollout_it_scores` and
`test_within_group_normalization_couples_rollouts_documented_caveat`) so the claim cannot quietly
drift. If strict unbiasedness matters more to you than scale invariance, set `rank_blend = 0` and
normalize with statistics from a *frozen* running estimate rather than the current group — the
machinery for that is already in `core/normalizer.py`.

Note what this buys: in a standard group, a rollout is compared only against siblings that were
trying to do the same thing. Here, a rollout that was generated to be *short* still supplies
evidence about what is achievable under the *accurate* direction. Off-direction samples become
baseline signal instead of being discarded. The whole `K × K` matrix costs `K²·m` flops.

One caveat, stated plainly: the estimator is unbiased for each direction's objective, but the
`K−1` off-direction rollouts are drawn from `π(·|x,w_l≠k)`, not from `π(·|x,w_k)`. The baseline
is therefore a *different-distribution* control variate — still unbiased (it does not depend on
`y_k`), but its variance-reduction benefit degrades as the conditional policies diverge. Early in
training, when conditioning is weak and all `w` produce similar behavior, it is nearly ideal.
Late in training it degrades gracefully toward a constant baseline. `loo_shrinkage` interpolates
toward the same-direction-only baseline if you want to control this.

### 2.5 Frontier shaping

Achievement alone drives each rollout toward its direction's optimum but says nothing about the
*shape* of the group. Two NSGA-II terms, computed on the rank vectors `u_k`:

- **Non-dominated front index** `f_k` (1 = non-dominated). Rewards being on the group's frontier.
- **Crowding distance** `c_k` within the front, boundary points set to the max. Rewards being in
  a *sparse* region of the frontier — this is the explicit anti-collapse pressure that keeps the
  policy from piling every direction onto the same knee point.

```
A_front[k] = z(−f_k) + λ_c · z(c_k)
A[k]       = z(A_ach[k]) + λ_f · A_front[k]
```

Both terms are z-scored within the group, so `λ_f` is a clean dimensionless mixing weight
(default 0.3). `λ_f = 0` recovers pure conditional Tchebycheff RL — that ablation is in the
experiment.

### 2.6 Coverage bandit over directions

Uniform sampling of `w` spends the same budget on frontier regions that are already saturated
and on regions that are starving. PaCE keeps a Das–Dennis grid of `D` reference directions on the
simplex and scores each one:

```
score_d  =  α · (1 − achievement_d)      # weak regions
          + β · improvement_d            # regions that are still moving (learnable)
          + c · sqrt( log N / n_d )      # under-explored regions (UCB)

p  =  (1 − ε) · softmax(score / τ)  +  ε · uniform
```

`achievement_d` and `improvement_d` are EMAs of the achievement attained by rollouts at direction
`d`. The `improvement` term is what stops the bandit from dumping budget into regions that are
weak because they are *infeasible* rather than because they are neglected. The `ε`-uniform floor
(default 0.15) guarantees no direction is ever starved, which is what keeps the frontier from
silently losing an arm.

Two implementation details are not optional, and both were found by the bandit misbehaving:

- **The signal fed to the bandit must be absolute, not group-relative.** The advantage's own
  achievements come from within-group ranks and are relative by construction — they say who beat
  whom, not how good the group was — so as a coverage signal they are nearly constant across
  directions and the bandit ends up chasing noise. `core/normalizer.py` supplies the absolute
  scale.
- **Achievement must be divided by `max_j w_j` before directions are compared.** `s(u; w)` ranges
  over `[-max_j w_j, 0]`, so `(1, 0)` spans `[-1, 0]` while `(0.5, 0.5)` spans only `[-0.5, 0]`.
  Without the correction the bandit concludes that extreme directions are permanently
  underperforming and pours budget into them forever. See `scalarization.direction_scale`.

**Status of this component: not established.** Even with both fixes, the measured effect is
within one standard error in every condition tested (`results/RESULTS.md` §4). The direction is at
least consistent with the mechanism — it helps when the direction grid is much larger than the
group and not when they are comparable, which is what you would predict, since a group sampling
8 of 9 grid directions without replacement covers the grid whatever the bandit prefers — but
consistency inside the noise is not a result. Of the five mechanisms this is the one I would drop
first if it does not earn its keep on a real task.

### 2.7 The full update

```
for each step:
    x ~ D
    w_1..w_K ~ CoverageBandit                       # §2.6
    y_k ~ π_θ(· | x, w_k)                           # §2.1
    r_k = R(x, y_k) ∈ R^m
    u   = rank_normalize(r)                         # §2.2
    S   = smooth_tchebycheff(u, w)                  # §2.3, K×K
    A   = cross_direction_advantage(S) + λ_f·front  # §2.4, §2.5
    θ  ← θ + ∇ clipped_surrogate(A)                 # PPO-style ratio clipping + optional KL
    CoverageBandit.update(w, S.diagonal())
```

## 3. Relation to prior work

I want to be precise about what is borrowed, because several of these ingredients exist
individually. (Caveat: arxiv.org is blocked from this sandbox, so the characterizations below come
from abstracts and secondary summaries, not full-text reads. Verify before making claims in a
paper.)

| Prior work | What it does | What PaCE takes / changes |
|---|---|---|
| **MO-GRPO** (Ichihara et al., 2025) | Per-objective z-score *then* sum, to stop the highest-variance reward dominating | Same diagnosis. PaCE uses **rank** normalization: invariant to any monotone transform, not just affine. Implemented as a baseline (`mognorm`). |
| **PRPO** — Pareto Ranking Policy Optimization (2026) | Non-dominated sorting over a batch to compute advantages, for tool-integrated agents | PaCE uses dominance rank + crowding as a *shaping* term only (§2.5), not as the whole advantage — dominance rank alone is very coarse at `K ≈ 8`. PRPO trains one operating point; PaCE trains a conditioned family. |
| **Dynamic reward weighting** (2025) | Hypervolume-guided adaptation of scalarization weights during training; two-stage global/local | Closest relative to the coverage bandit. Difference: they adapt *one* weight vector over time (still one policy, one endpoint); PaCE keeps a *distribution* over directions and conditions the policy on the sample, so the frontier is available simultaneously at inference. |
| **Panacea** (NeurIPS 2024) | Preference-conditioned LLM via SVD-LoRA, traverses the simplex, Tchebycheff-style | Same "one model, whole frontier" goal and the same reason for Tchebycheff. Panacea is preference-model/DPO-flavored; PaCE is online group-RL with programmatic/verifiable rewards, and adds the cross-direction advantage matrix and coverage bandit. Panacea's SVD-LoRA is a strictly better conditioning mechanism than prompt conditioning and is the natural upgrade (§5). |
| **Rewarded Soups / model merging** | Train per-objective experts, interpolate weights | No training-time interaction between objectives; interpolated points are never trained on. PaCE trains *at* intermediate directions. Implemented as the `fixed-scalar` baseline. |
| **DPA, RiC, MOD, CLP** | Inference-time preference conditioning / decoding | Shared goal of steerability. PaCE differs in the training signal, not the interface. |

**The claim, stated narrowly:** the novelty is (i) the K×K cross-direction leave-one-out advantage
matrix, which is only available once groups are heterogeneous in direction and which is, as far as
I can tell, not in the literature; (ii) rank-space normalization for monotone invariance in
multi-objective group RL; (iii) putting conditioning + non-convex-capable scalarization + coverage
exploration into a single online group-RL update rather than across separate stages or runs. The
individual ideas of dominance-ranked advantages, hypervolume-guided weighting, and preference
conditioning are **not** new and I am not claiming them.

## 4. Which base model

Short version: **Gemma 4 E4B for iteration, Gemma 4 12B for the headline result.** Muse Spark is
the wrong tool and the reason is not a close call.

**Muse Spark (Meta, MSL).** Proprietary and API-only. Muse Spark 1.1 shipped 2026-07-09, 1.3 on
2026-09-02; open weights have been signalled as coming but have not landed. RL post-training
requires per-token log-probabilities and a gradient path through the policy. An inference API
gives you neither. You cannot run GRPO — or any policy-gradient method — against a model you
cannot backprop through. Muse Spark is usable in this project in exactly one role: as an
**LLM-judge reward model** for the subjective objectives (helpfulness, style), where API access is
sufficient. That role is supported in the reward suite but off by default, because a judge you pay
per call is a bad fit for a method that needs `K` rollouts per prompt per step.

**Muse Glimmer (Meta, 30B dense, Apache 2.0, 2026-08-10).** This *is* trainable and the license is
clean. Two problems for our purposes: 30B dense means every rollout is expensive, and PaCE is
rollout-hungry by construction (`K` rollouts per prompt, and the whole point is covering a frontier
rather than a point). At research-iteration scale that is the wrong trade. It is a good
**scale-check** model once the method works — the agentic tuning also makes it a good fit for
multi-objective tool-use tasks, which is where trade-offs like accuracy/latency/cost get real.

**Gemma 4 (Google, 2026-04-02, Apache 2.0).** Sizes E2B (~2.3B effective), E4B (~4.5B effective),
12B, 26B-A4B (MoE), 31B. This is the right family:

- **E4B** — fits comfortably on a single 24–48GB GPU with LoRA, fast enough to run the ~5–10
  ablations PaCE actually needs (bandit on/off, λ_f sweep, Tchebycheff vs linear, K sweep).
- **12B** — the credible headline number, single 80GB node with LoRA.
- **26B-A4B** — MoE, 4B active. If throughput is the binding constraint rather than memory, this
  is the better scale-up than 31B dense.
- Apache 2.0, so no license asterisk on released checkpoints.
- Gemma 4's own post-training already used RLHF + RL-from-verifier-feedback, so the instruct
  checkpoints respond sensibly to system-level control text — which is precisely what prompt-based
  preference conditioning (§5) depends on.

Recommendation: develop on **E2B** (fastest possible loop, sanity checks), report on **E4B** and
**12B**, and keep **Muse Glimmer 30B** as an optional external-validity check. Nothing in the
algorithm is Gemma-specific.

## 5. Preference conditioning: how `w` gets into the model

Two mechanisms, in increasing order of strength and invasiveness:

1. **Prompt conditioning (default, implemented).** A control line in the system turn:
   `<preference>accuracy=0.70 brevity=0.30</preference>`. Zero architecture change, works with any
   chat model, and the policy learns to read it because the reward depends on it. Weakness: the
   model can learn to ignore it, and nothing structurally prevents that — which is exactly why the
   **controllability metric** (§6) is a first-class part of the evaluation and not an afterthought.
2. **Conditioned LoRA (planned).** `w` modulates the LoRA update — Panacea's SVD-LoRA embeds the
   preference vector into the singular values of the adapter. Structural, so it cannot be ignored,
   at the cost of a custom adapter.

Start with (1) because it isolates the contribution of the *advantage estimator*, which is the
part of PaCE that is actually novel. If controllability is weak on a task, move to (2).

## 6. Evaluation

Optimizing a frontier requires frontier metrics. Reporting mean reward defeats the purpose.

- **Hypervolume** w.r.t. a fixed reference point — the standard scalar summary of frontier quality.
  Exact for `m ≤ 3`, Monte-Carlo above.
- **Spacing** (Schott) and **maximum spread** — is the frontier *covered*, or is it three points
  and a wish?
- **Controllability**: Spearman correlation between the requested weight `w_j` and the achieved
  reward `r_j`, per objective. Does the knob turn? A model that ignores `w` can still post a
  respectable hypervolume by sitting on the knee. This metric catches that, and it is the one I
  would look at first when the method appears to be working.
- **Per-direction dominance rate** vs. the fixed-weight ensemble at equal rollout budget — the
  fair comparison, since PaCE's whole pitch is budget efficiency.

## 7. What is validated, and what is not

The synthetic environment (`pace/envs/synthetic.py`) is a contextual bandit whose Pareto front has
tunable curvature, so the concave case (where linear scalarization provably fails) can be tested
directly. It is torch-free and runs on CPU in seconds.

This validates the **optimizer's frontier behavior** — that the advantage estimator finds and
spreads across a frontier, including a concave one. It says nothing about LLM performance. The
LLM path is written against Gemma 4 but has not been run: no GPU in this environment. Every claim
in `results/` comes from the synthetic environment and is labelled as such.


## 8. What the experiments actually showed

Full numbers and interpretation: [`results/RESULTS.md`](results/RESULTS.md). The short version,
including the parts that did not go the way the design predicted:

- **The headline held, with a condition.** On a concave front at equal rollout budget, PaCE
  reaches ~58% of oracle hypervolume with one model; nine independent fixed-weight runs reach
  ~27%. On a *convex* front the ordering reverses — the fixed-weight ensemble reaches 94% and
  PaCE 71%. PaCE's advantage is conditional on front geometry and costs something when the
  geometry is benign. That belongs in the abstract, not in a footnote.
- **Scale invariance was the most robust finding.** A strictly increasing transform of one
  objective, which does not move the Pareto set at all, leaves PaCE bit-identical and collapses a
  weighted-sum baseline by 5×, identically across all 20 seeds. MO-GRPO-style z-scoring degrades
  gracefully but is not invariant, exactly as the affine-vs-monotone argument predicts.
- **Frontier shaping carried the most weight** of any component (removing it costs ~12 points of
  oracle).
- **The cross-direction advantage matrix was not shown to matter here**, and the environment is
  the reason: a tiny action space with low gradient variance is where a variance-reduction device
  has least to offer. It needs expensive, high-variance rollouts to be evaluated at all.
- **The coverage bandit was not established** (§2.6).
- **Controllability earned its place as a metric.** A dominance-only advantage posted
  near-top-of-table hypervolume with a preference knob that does essentially nothing
  (correlation +0.07). Judged on hypervolume alone it looks like one of the better methods.

The two unestablished mechanisms are the reason the LLM experiment matters rather than being a
victory lap: it is the setting where they are testable at all.
