#!/usr/bin/env python3
"""PaCE vs baselines on a small real Qwen3 trained from scratch.

This is the experiment the synthetic bandit could not run. It has what that environment
lacks: sampled sequences, token-level credit assignment, and genuinely incommensurable
rewards (a 0/1 verifier against an exponential length term). Those are the conditions under
which the cross-direction advantage matrix and rank normalization's scale invariance are
testable at all -- ``results/RESULTS.md`` records both as unestablished precisely because
the bandit is too small and too well-scaled to test them.

Stages: pretrain the policy into the frontier window, confirm the frontier is real, then
run each method at equal rollout budget and compare achieved frontiers.

Usage::

    python experiments/tiny_lm_rl.py --seeds 3 --steps 250
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pace.core.advantages import linear_scalar_advantages, mo_grpo_advantages, pace_advantages  # noqa: E402
from pace.core.config import PaCEConfig  # noqa: E402
from pace.core.directions import CoverageBandit, UniformDirections, das_dennis  # noqa: E402
from pace.core.metrics import frontier_summary  # noqa: E402
from pace.core.normalizer import RunningMinMax  # noqa: E402
from pace.core.scalarization import achievement_matrix, direction_scale  # noqa: E402
from pace.envs.tiny_lm import ArithmeticTask, CharTokenizer, build_tiny_qwen, pretrain  # noqa: E402
from pace.llm.rewards import extract_answer  # noqa: E402

OBJECTIVES = ("accuracy", "brevity")
TARGET_TOKENS = 24


def rewards_for(texts, golds, prompt_lens):
    """(n, 2) rewards: a 0/1 verifier and an exponential length term.

    Deliberately incommensurable -- one is a Bernoulli indicator, the other a smooth
    positive quantity. That mismatch is the regime rank normalization exists for and the
    one the synthetic environment's tidy [0, 1] rewards never exercise.
    """
    out = []
    for text, gold, n_tok in zip(texts, golds, prompt_lens):
        answer = extract_answer(text.split("\n")[0])
        acc = 1.0 if (answer is not None and answer.strip() == str(gold)) else 0.0
        out.append([acc, float(np.exp(-n_tok / TARGET_TOKENS))])
    return np.asarray(out, dtype=float)


def generate_group(model, tok, torch, a, b, W, max_new_tokens, temperature, rng_seed):
    """One rollout per direction. Returns texts, token counts, and the tensors for scoring."""
    prompts = [ArithmeticTask.question(a, b, float(w[0])) for w in W]
    enc = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            pad_token_id=tok.pad_token_id,
        )
    completion = out[:, enc["input_ids"].shape[1] :]
    texts = tok.batch_decode(completion)
    # Count tokens up to the newline the model uses to end a row.
    lens = []
    for row in completion:
        ids = [int(i) for i in row]
        nl = tok.stoi["\n"]
        lens.append(ids.index(nl) + 1 if nl in ids else len(ids))
    return texts, np.array(lens, dtype=float), enc["input_ids"], completion


def token_logprobs(model, torch, tok, prompt_ids, completion_ids, grad: bool):
    ids = torch.cat([prompt_ids, completion_ids], dim=1)
    mask = (ids != tok.pad_token_id).long()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=ids, attention_mask=mask).logits
    logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
    lp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(lp, 2, completion_ids.unsqueeze(-1)).squeeze(-1)


def evaluate(model, tok, torch, task, dirs, n_problems, max_new_tokens, seed=0):
    """Greedy decode per direction; returns the achieved reward vector for each.

    Evaluation problems come from a locally seeded generator, NOT from ``task.sample()``.
    That distinction is the whole point: the task's own RNG advances through pretraining and
    through every RL step, so drawing eval problems from it hands each policy a *different*
    exam, and comparisons then mix real differences with problem-set luck.

    This was a real bug. It made the ceiling reference -- evaluated after 500 steps of
    pretraining had already consumed the task RNG -- score 0.3600 where the identical
    configuration on a fresh task scored 0.5948. That inverted the measured headroom to
    negative and would have sunk the experiment. It also inflated the seed-to-seed spread in
    the earlier tiny-LM run, where every method and every seed drew its own eval set.
    """
    rng = np.random.default_rng(seed)
    problems = [
        (int(rng.integers(task.lo, task.hi + 1)), int(rng.integers(task.lo, task.hi + 1))) for _ in range(n_problems)
    ]
    rows = []
    for w in dirs:
        texts, lens, golds = [], [], []
        for a, b in problems:
            prompt = ArithmeticTask.question(a, b, float(w[0]))
            enc = tok([prompt], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            comp = out[:, enc["input_ids"].shape[1] :]
            ids = [int(i) for i in comp[0]]
            nl = tok.stoi["\n"]
            lens.append(ids.index(nl) + 1 if nl in ids else len(ids))
            texts.append(tok.decode(comp[0]))
            golds.append(a + b)
        rows.append(rewards_for(texts, golds, lens).mean(axis=0))
    return np.stack(rows)


def run_method(method, cfg, args, base_state, tok, task, torch):
    """Train one method from the shared pretrained checkpoint and evaluate its frontier."""
    torch.manual_seed(cfg["seed"])
    model = build_tiny_qwen(tok, hidden=args.hidden, layers=args.layers)
    model.load_state_dict(base_state)
    from pace.llm.trainer import _disable_dropout

    _disable_dropout(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # A frozen copy of the pretrained policy, as the KL reference. Without this anchor the
    # policy is free to drift arbitrarily far from a starting point that already traces a
    # good frontier, and RL reliably destroys it -- which is exactly what the first version
    # of this experiment measured, because it used a bare REINFORCE loss while the shipped
    # PaCETrainer uses both a KL penalty and ratio clipping. Testing a different algorithm
    # than the one the repository ships is not a test of the repository.
    ref = build_tiny_qwen(tok, hidden=args.hidden, layers=args.layers)
    ref.load_state_dict(base_state)
    _disable_dropout(ref)
    ref.eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)

    pace_cfg = PaCEConfig(**{k: v for k, v in cfg.get("pace", {}).items()})
    grid = das_dennis(2, args.partitions)
    sampler_cls = CoverageBandit if cfg["bandit"] else UniformDirections
    sampler = sampler_cls(grid, config=pace_cfg.bandit, seed=cfg["seed"] + 7)
    norm = RunningMinMax(2)
    rng = np.random.default_rng(cfg["seed"])

    for step in range(args.steps):
        opt.zero_grad()
        for _ in range(args.prompts_per_step):
            a, b = task.sample()
            if cfg["heterogeneous"]:
                idx, W = sampler.sample(args.group)
            else:
                i1, w1 = sampler.sample(1)
                idx = np.repeat(i1, args.group)
                W = np.repeat(w1, args.group, axis=0)

            texts, lens, prompt_ids, comp_ids = generate_group(
                model, tok, torch, a, b, W, args.max_new_tokens, args.temperature, cfg["seed"]
            )
            R = rewards_for(texts, [a + b] * len(texts), lens)

            result = cfg["advantage"](R, W)
            adv = result.advantages if hasattr(result, "advantages") else result

            norm.update(R)
            sampler.update(
                idx,
                np.diagonal(
                    achievement_matrix(
                        norm.normalize(R), W, kind=pace_cfg.scalarization, mu=pace_cfg.mu, rho=pace_cfg.rho
                    )
                )
                / direction_scale(W),
            )

            mask = (comp_ids != tok.pad_token_id).float()
            a_t = torch.tensor(adv, dtype=torch.float32).unsqueeze(1)

            old_lp = token_logprobs(model, torch, tok, prompt_ids, comp_ids, grad=False)
            ref_lp = token_logprobs(ref, torch, tok, prompt_ids, comp_ids, grad=False)
            lp = token_logprobs(model, torch, tok, prompt_ids, comp_ids, grad=True)

            ratio = torch.exp(lp - old_lp)
            unclipped = ratio * a_t
            clipped = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * a_t
            pg = -(torch.min(unclipped, clipped) * mask).sum() / mask.sum().clamp(min=1)

            # k3 estimator: non-negative, lower variance than (ref - new).
            delta = ref_lp - lp
            kl = ((torch.exp(delta) - delta - 1.0) * mask).sum() / mask.sum().clamp(min=1)

            loss = (pg + args.kl_coef * kl) / args.prompts_per_step
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    eval_dirs = das_dennis(2, args.eval_partitions)
    F = evaluate(model, tok, torch, task, eval_dirs, args.eval_problems, args.max_new_tokens)
    summary = frontier_summary(F, np.zeros(2), eval_dirs)
    return summary, F


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--pretrain-steps", type=int, default=500)
    ap.add_argument("--n-digits", type=int, default=2)
    ap.add_argument(
        "--start-floor",
        type=float,
        default=0.15,
        help="Worked-style probability at the brevity end of the START policy.",
    )
    ap.add_argument(
        "--start-span",
        type=float,
        default=0.70,
        help="How much the marker shifts that probability. Small values leave the "
        "start policy weakly conditioned, so RL must LEARN the routing.",
    )
    ap.add_argument(
        "--ceiling-floor",
        type=float,
        default=0.15,
    )
    ap.add_argument(
        "--ceiling-span",
        type=float,
        default=0.70,
        help="Conditioning strength of the ceiling reference. Do not push this to the "
        "extreme: teacher-forced loss is token-weighted and the worked style carries ~73% "
        "of all tokens, so an over-strong setting lets the model minimize loss by always "
        "emitting worked and ignoring the marker -- which produces a ceiling with "
        "near-zero controllability and no usable target.",
    )
    ap.add_argument(
        "--ceiling-steps",
        type=int,
        default=0,
        help="If >0, train a strongly-conditioned reference policy too and report "
        "each method as the fraction of the (ceiling - start) gap it closes. "
        "Without it, results read only relative to the start policy, hiding "
        "whether headroom existed -- the flaw in the first tiny-LM run.",
    )
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.02, help="KL anchor to the pretrained policy")
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--max-new-tokens", type=int, default=28)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--partitions", type=int, default=8)
    ap.add_argument("--eval-partitions", type=int, default=8)
    ap.add_argument("--eval-problems", type=int, default=24)
    ap.add_argument("--output", default="results/tiny_lm_results.json")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    tok = CharTokenizer()
    task = ArithmeticTask(n_digits=args.n_digits, seed=0)
    dirs = das_dennis(2, args.eval_partitions)

    print(f"[1/3] pretraining {args.pretrain_steps} steps into the frontier window", flush=True)
    t0 = time.time()
    model = build_tiny_qwen(tok, hidden=args.hidden, layers=args.layers)
    pretrain(
        model,
        tok,
        task,
        steps=args.pretrain_steps,
        batch_size=48,
        lr=3e-3,
        log_every=250,
        floor=args.start_floor,
        span=args.start_span,
    )
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"      done in {(time.time()-t0)/60:.1f} min", flush=True)

    ceiling_hv = None
    if args.ceiling_steps > 0:
        print(
            f"[1b] CEILING reference: {args.ceiling_steps} steps, strong conditioning -- "
            f"what this architecture reaches by supervision alone",
            flush=True,
        )
        t1 = time.time()
        ceil_model = build_tiny_qwen(tok, hidden=args.hidden, layers=args.layers)
        pretrain(
            ceil_model,
            tok,
            task,
            steps=args.ceiling_steps,
            batch_size=48,
            lr=3e-3,
            log_every=1000,
            floor=0.02,
            span=0.96,
        )
        Fc = evaluate(ceil_model, tok, torch, task, dirs, args.eval_problems, args.max_new_tokens)
        sc = frontier_summary(Fc, np.zeros(2), dirs)
        ceiling_hv = sc["hypervolume"]
        print(
            f"      ceiling HV {ceiling_hv:.4f}  ctrl {sc['controllability_mean']:+.3f} "
            f"({(time.time() - t1) / 60:.1f} min)",
            flush=True,
        )
        del ceil_model

    print("[2/3] frontier check on the pretrained policy", flush=True)
    F0 = evaluate(model, tok, torch, task, dirs, args.eval_problems, args.max_new_tokens)
    s0 = frontier_summary(F0, np.zeros(2), dirs)
    for w, r in zip(dirs, F0):
        print(f"      w_acc={w[0]:.2f} -> accuracy {r[0]:.3f}  brevity {r[1]:.3f}", flush=True)
    print(f"      controllability {s0['controllability_mean']:+.3f}  HV {s0['hypervolume']:.4f}", flush=True)
    if ceiling_hv is not None:
        gap = ceiling_hv - s0["hypervolume"]
        print(
            f"      HEADROOM: ceiling {ceiling_hv:.4f} - start " f"{s0['hypervolume']:.4f} = {gap:+.4f}",
            flush=True,
        )
        if gap <= 0.02:
            print(
                "      WARNING: almost no headroom. RL can only preserve or damage this "
                "frontier, so the comparison will not test frontier discovery.",
                flush=True,
            )
    if s0["controllability_mean"] < 0.2:
        print(
            "      WARNING: pretrained policy barely responds to the marker; RL starts "
            "from almost no conditioning signal.",
            flush=True,
        )

    methods = {
        "pace": dict(advantage=lambda R, W: pace_advantages(R, W, PaCEConfig()), heterogeneous=True, bandit=True),
        "pace_uniform": dict(
            advantage=lambda R, W: pace_advantages(R, W, PaCEConfig()), heterogeneous=True, bandit=False
        ),
        "pace_no_front": dict(
            advantage=lambda R, W: pace_advantages(R, W, PaCEConfig(lambda_front=0.0)), heterogeneous=True, bandit=False
        ),
        "pace_no_crossdir": dict(
            advantage=lambda R, W: pace_advantages(R, W, PaCEConfig(loo_shrinkage=1.0)),
            heterogeneous=True,
            bandit=False,
        ),
        "cond_linear": dict(advantage=linear_scalar_advantages, heterogeneous=False, bandit=False),
        "cond_mognorm": dict(advantage=mo_grpo_advantages, heterogeneous=False, bandit=False),
    }

    print(f"[3/3] RL: {len(methods)} methods x {args.seeds} seeds x {args.steps} steps", flush=True)
    results = {
        "config": vars(args),
        "pretrained": {"frontier": F0.tolist(), **s0},
        "ceiling_hypervolume": ceiling_hv,
        "methods": {},
    }
    for name, spec in methods.items():
        runs = []
        for seed in range(args.seeds):
            t = time.time()
            cfg = dict(spec, seed=seed)
            summary, F = run_method(name, cfg, args, base_state, tok, task, torch)
            runs.append(summary)
            print(
                f"      {name:18s} seed{seed} HV={summary['hypervolume']:.4f} "
                f"ctrl={summary['controllability_mean']:+.3f} ({(time.time()-t)/60:.1f} min)",
                flush=True,
            )
        agg = {}
        for k in ("hypervolume", "controllability_mean", "spacing", "max_spread"):
            v = np.array([r[k] for r in runs], dtype=float)
            agg[k], agg[k + "_std"] = float(v.mean()), float(v.std())
        if ceiling_hv is not None:
            gap = ceiling_hv - s0["hypervolume"]
            agg["gap_closed"] = float((agg["hypervolume"] - s0["hypervolume"]) / gap) if gap > 1e-9 else float("nan")
        results["methods"][name] = agg
        extra = f" gap_closed={agg['gap_closed']:+.1%}" if "gap_closed" in agg else ""
        print(
            f"      -> {name:18s} HV={agg['hypervolume']:.4f} +-{agg['hypervolume_std']:.4f} "
            f"ctrl={agg['controllability_mean']:+.3f}{extra}",
            flush=True,
        )

    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
