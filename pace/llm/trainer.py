"""A self-contained PaCE trainer for causal language models.

Written against ``transformers`` directly rather than subclassing a trainer from another
library, because the piece that has to change -- how a group of rollouts becomes a vector
of advantages -- is exactly the piece those trainers keep private. Vendoring the ~200 lines
of rollout/log-prob/surrogate loop costs less than tracking another project's internals.
See ``trl_adapter.py`` if you would rather keep your existing TRL setup.

**Status: this module has not been executed.** It was developed in an environment with no
GPU and no torch installed. The algorithmic core it calls (``pace.core``) is tested and
validated; this glue is not. Treat it as a reviewed starting point, not as working code:
expect to fix shape and dtype details on first run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.advantages import pace_advantages
from ..core.config import PaCEConfig
from ..core.directions import CoverageBandit, das_dennis
from ..core.metrics import FrontierArchive
from ..core.normalizer import RunningMinMax
from ..core.scalarization import achievement_matrix, direction_scale
from .conditioning import build_prompt
from .rewards import RewardSuite

__all__ = ["LLMTrainConfig", "PaCETrainer"]


def _disable_dropout(model) -> int:
    """Zero every dropout probability in the policy. Returns how many modules changed.

    This is not a tuning preference, it is a correctness requirement. The surrogate below
    compares log-probabilities from two forward passes over the same tokens. With dropout
    active those passes sample different masks, so ``exp(logprobs - old_logprobs)`` is not
    1 even on the first inner epoch over a fresh batch -- it is noise. Two things then go
    wrong at once: PPO clipping starts firing on dropout randomness rather than on genuine
    policy movement, and the KL penalty charges the policy for that randomness, injecting a
    gradient that has nothing to do with the objective.

    Measured on a 2-layer test model at the library default of 0.1: the two passes differed
    by up to 0.28 in log-probability. Standard RLHF implementations disable policy dropout
    during RL for exactly this reason.

    Setting the config attributes alone is not enough -- modules are already constructed by
    then -- so walk the module tree.
    """
    import torch.nn as nn

    changed = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p != 0.0:
            module.p = 0.0
            changed += 1
    for attr in (
        "dropout",
        "attention_dropout",
        "resid_pdrop",
        "attn_pdrop",
        "embd_pdrop",
        "hidden_dropout",
        "activation_dropout",
        "classifier_dropout",
    ):
        if hasattr(model.config, attr) and isinstance(getattr(model.config, attr), (int, float)):
            setattr(model.config, attr, 0.0)
    return changed


@dataclass
class LLMTrainConfig:
    model_name: str = "google/gemma-4-E4B-it"
    objectives: tuple[str, ...] = ("accuracy", "brevity", "format")

    group_size: int = 8
    """Rollouts per prompt. Also the number of distinct directions per group -- PaCE gets
    its cross-direction signal from within the group, so very small K weakens it."""

    prompts_per_step: int = 4
    steps: int = 1000
    lr: float = 1e-6
    kl_coef: float = 0.02
    clip_range: float = 0.2
    max_new_tokens: int = 512
    temperature: float = 1.0
    target_tokens: int = 256
    grad_accum: int = 1
    max_grad_norm: float = 1.0

    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    seed: int = 0
    pace: PaCEConfig = field(default_factory=PaCEConfig)
    log_every: int = 10
    eval_every: int = 100
    eval_partitions: int = 8


class PaCETrainer:
    """Group-relative RL over a preference-conditioned causal LM."""

    def __init__(self, config: LLMTrainConfig, train_dataset, eval_dataset=None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError("The LLM path needs torch and transformers: pip install 'pace[llm]'") from exc

        self.torch = torch
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # generation needs left padding

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _disable_dropout(self.model)

        if config.use_lora:
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    task_type="CAUSAL_LM",
                    target_modules="all-linear",
                ),
            )
            # LoRA introduces its own dropout modules, so sweep again after wrapping.
            # lora_dropout is left in the config as a knob for supervised fine-tuning;
            # during RL it has to be off for the same reason as the base model's.
            _disable_dropout(self.model)

        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=config.lr)
        self.rewards = RewardSuite(
            objectives=config.objectives,
            target_tokens=config.target_tokens,
            tokenizer=self.tokenizer,
        )
        m = len(config.objectives)
        self.bandit = CoverageBandit(
            das_dennis(m, config.pace.n_partitions),
            config=config.pace.bandit,
            seed=config.seed,
        )
        self.normalizer = RunningMinMax(m)
        self.archive = FrontierArchive()
        self.rng = np.random.default_rng(config.seed)

    # -- rollout ------------------------------------------------------------------

    def _generate(self, question: str, W: np.ndarray) -> tuple[list[str], "object", "object"]:
        """Generate one completion per direction. Returns texts, prompt ids, completion ids."""
        torch = self.torch
        prompts = [
            self.tokenizer.apply_chat_template(
                build_prompt(question, w, self.config.objectives),
                tokenize=False,
                add_generation_prompt=True,
            )
            for w in W
        ]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion_ids = out[:, enc["input_ids"].shape[1] :]
        texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return texts, enc["input_ids"], completion_ids

    def _logprobs(self, prompt_ids, completion_ids, requires_grad: bool = True):
        """Per-token log-probabilities of ``completion_ids`` under the current policy."""
        torch = self.torch
        ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = (ids != self.tokenizer.pad_token_id).long()
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            logits = self.model(input_ids=ids, attention_mask=attention_mask).logits
        # Token t is predicted by the logits at position t-1.
        logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        return torch.gather(logprobs, 2, completion_ids.unsqueeze(-1)).squeeze(-1)

    # -- training -----------------------------------------------------------------

    def train_step(self, batch) -> dict:
        """One optimizer step over ``prompts_per_step`` prompts."""
        torch = self.torch
        cfg = self.config
        self.optimizer.zero_grad()
        logs: list[dict] = []

        for example in batch:
            idx, W = self.bandit.sample(cfg.group_size)
            texts, prompt_ids, completion_ids = self._generate(example["question"], W)
            R = self.rewards(texts, [example["answer"]] * len(texts))

            out = pace_advantages(R, W, cfg.pace)
            advantages = torch.tensor(out.advantages, dtype=torch.float32, device=self.model.device)

            # Bandit coverage signal must be absolute and direction-scale corrected; the
            # advantage's own achievements are group-relative and will not do. See
            # pace/core/normalizer.py.
            self.normalizer.update(R)
            self.bandit.update(
                idx,
                np.diagonal(
                    achievement_matrix(
                        self.normalizer.normalize(R),
                        W,
                        kind=cfg.pace.scalarization,
                        mu=cfg.pace.mu,
                        rho=cfg.pace.rho,
                    )
                )
                / direction_scale(W),
            )

            # Old log-probs are the behaviour policy's. With a single inner epoch the
            # ratio is exactly 1 and the clipping is inert; it starts to matter as soon as
            # you take more than one gradient step per batch of rollouts. That exactness
            # depends on dropout being off -- see _disable_dropout.
            with torch.no_grad():
                old_logprobs = self._logprobs(prompt_ids, completion_ids, requires_grad=False)
            logprobs = self._logprobs(prompt_ids, completion_ids)

            mask = (completion_ids != self.tokenizer.pad_token_id).float()
            ratio = torch.exp(logprobs - old_logprobs)
            adv = advantages.unsqueeze(1)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv
            pg_loss = -(torch.min(unclipped, clipped) * mask).sum() / mask.sum().clamp(min=1)

            # k3 estimator: non-negative and lower variance than (old - new).
            if cfg.kl_coef > 0:
                delta = old_logprobs - logprobs
                kl = torch.exp(delta) - delta - 1.0
                kl = (kl * mask).sum() / mask.sum().clamp(min=1)
            else:
                kl = torch.zeros((), device=self.model.device)

            loss = (pg_loss + cfg.kl_coef * kl) / len(batch)
            loss.backward()

            logs.append(
                {
                    "reward_mean": R.mean(axis=0),
                    "pg_loss": float(pg_loss.detach()),
                    "kl": float(kl.detach()),
                    "advantage_std": float(np.std(out.advantages)),
                    "nondominated_frac": float((out.fronts == 1).mean()),
                    "completion_tokens": float(mask.sum(dim=1).mean()),
                }
            )

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        merged = {"pg_loss": float(np.mean([l["pg_loss"] for l in logs]))}
        for key in ("kl", "advantage_std", "nondominated_frac", "completion_tokens"):
            merged[key] = float(np.mean([l[key] for l in logs]))
        for j, name in enumerate(cfg.objectives):
            merged[f"reward/{name}"] = float(np.mean([l["reward_mean"][j] for l in logs]))
        return merged

    @property
    def _eval_directions(self) -> np.ndarray:
        return das_dennis(len(self.config.objectives), self.config.eval_partitions)

    def evaluate(self, dataset, n_prompts: int = 64) -> dict:
        """Sweep the direction grid and measure the achieved frontier.

        Greedy decoding on purpose: sampling and then averaging rewards would let a policy
        that flips between extremes look like one that learned the middle. See the note in
        ``pace/envs/synthetic.py::oracle_frontier``.
        """
        from ..core.metrics import frontier_summary

        torch = self.torch
        cfg = self.config
        dirs = self._eval_directions
        points = []
        for w in dirs:
            rows = []
            for example in list(dataset)[:n_prompts]:
                prompt = self.tokenizer.apply_chat_template(
                    build_prompt(example["question"], w, cfg.objectives),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(
                        **enc,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                text = self.tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
                rows.append(self.rewards([text], [example["answer"]])[0])
            points.append(np.mean(rows, axis=0))
        F = np.stack(points)
        for w, r in zip(dirs, F):
            self.archive.add(w, r)
        # Reference point at zero: every reward in the suite is already in [0, 1] and 0 is
        # the true worst, so no margin is needed here.
        return frontier_summary(F, np.zeros(len(cfg.objectives)), dirs)
