"""A small real Qwen3 language model, trained from scratch on CPU, as an RL environment.

The synthetic bandit in ``synthetic.py`` validates the frontier mathematics but cannot test
two of PaCE's mechanisms: the cross-direction advantage matrix is a variance-reduction
device and that environment has almost no gradient variance, and rank normalization's scale
invariance never matters because its rewards are already tidy ``[0, 1]`` values. Both need a
setting with sampled sequences, token-level credit assignment, and genuinely incommensurable
rewards. This module builds the smallest thing that has all three.

It is the real Qwen3 architecture from ``transformers`` -- RoPE, SwiGLU, RMSNorm, grouped
query attention -- at a few million parameters, randomly initialized and pretrained here.
It is **not** Qwen's released weights: huggingface.co is blocked by this sandbox's egress
policy, so nothing is downloaded. What transfers from a real Qwen is the architecture and
the shape of the training loop, not any pretrained knowledge.

The task is two-digit addition, posed in two answer styles the model learns during
pretraining:

* **worked** -- digit-by-digit with an explicit carry, then the answer
* **direct** -- the answer alone

These are meant to trade off: the worked form is compositional and a small model learns it
from fewer examples, while the direct form requires memorizing a 100x100 table. That makes
accuracy genuinely depend on how many tokens the model spends, which is the frontier PaCE is
supposed to navigate. Whether the gap actually materializes is an empirical question, not an
assumption -- ``measure_style_gap`` checks it, and if it comes back flat the frontier is not
real and the experiment on top of it means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CharTokenizer", "ArithmeticTask", "QWEN3_REFERENCE", "build_tiny_qwen", "pretrain", "measure_style_gap"]

VOCAB = list("0123456789+=;#? cary\n") + ["<pad>", "<eos>", "<bos>"]


class CharTokenizer:
    """Character-level tokenizer. Deterministic, tiny, and needs no training corpus.

    Character level is the right choice here rather than a compromise: the reward depends
    on exact digit sequences, so a subword vocabulary would put the thing being scored
    behind an extra layer of tokenization noise.
    """

    def __init__(self) -> None:
        self.itos = list(VOCAB)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_token_id = self.stoi["<pad>"]
        self.eos_token_id = self.stoi["<eos>"]
        self.bos_token_id = self.stoi["<bos>"]
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.padding_side = "left"

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, **kw) -> list[int]:
        unk = self.stoi["?"]
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i >= len(self.itos):
                continue
            tok = self.itos[i]
            if tok in ("<pad>", "<eos>", "<bos>"):
                if tok == "<eos>":
                    break
                continue
            out.append(tok)
        return "".join(out)

    def batch_decode(self, seqs, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(s, skip_special_tokens) for s in seqs]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True) -> str:
        # No chat template: the "prompt" is the preference line plus the problem, and a
        # character model should not have to spend capacity on role scaffolding.
        return "".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors=None, padding=False, **kw):
        import torch

        texts = [text] if isinstance(text, str) else list(text)
        seqs = [self.encode(t) for t in texts]
        width = max(len(s) for s in seqs)
        ids = [[self.pad_token_id] * (width - len(s)) + s for s in seqs]  # left padding
        mask = [[0] * (width - len(s)) + [1] * len(s) for s in seqs]

        class _Batch(dict):
            def to(self, *a, **k):
                return self

        return _Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))

    def save_pretrained(self, path):
        pass


@dataclass
class ArithmeticTask:
    """Two-digit addition, rendered in a long ("worked") or short ("direct") style."""

    max_operand: int = 99
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def sample(self) -> tuple[int, int]:
        return int(self.rng.integers(10, self.max_operand + 1)), int(self.rng.integers(10, self.max_operand + 1))

    @staticmethod
    def question(a: int, b: int) -> str:
        return f"{a}+{b}="

    @staticmethod
    def worked(a: int, b: int) -> str:
        """Digit-by-digit with an explicit carry, then the answer after '#'."""
        ones = a % 10 + b % 10
        carry = ones // 10
        tens = a // 10 + b // 10 + carry
        return f"{a % 10}+{b % 10}={ones};c{carry};{a // 10}+{b // 10}+{carry}={tens};#{a + b}"

    @staticmethod
    def direct(a: int, b: int) -> str:
        return f"#{a + b}"

    def corpus(self, n: int, worked_fraction: float = 0.5) -> list[str]:
        """Pretraining corpus mixing both styles, so RL has both modes available."""
        rows = []
        for _ in range(n):
            a, b = self.sample()
            body = self.worked(a, b) if self.rng.random() < worked_fraction else self.direct(a, b)
            rows.append(self.question(a, b) + body + "\n")
        return rows


# Architectural ratios read from the real Qwen/Qwen3-0.6B config.json via the Hugging Face
# MCP connector (the weights themselves are a 1.5GB LFS blob and are not reachable from
# this sandbox, so only the shape transfers, not the pretrained knowledge).
QWEN3_REFERENCE = {
    "num_attention_heads": 16,
    "num_key_value_heads": 8,  # 2:1 grouped-query attention
    "head_dim": 128,  # deliberately decoupled from hidden_size, as in the real model
    "intermediate_ratio": 3,  # intermediate_size / hidden_size = 3072 / 1024
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "tie_word_embeddings": True,
}


def build_tiny_qwen(tokenizer: CharTokenizer, hidden: int = 256, layers: int = 6):
    """A small Qwen3 with random weights, mirroring the real Qwen3-0.6B's proportions.

    Nothing is downloaded. What is borrowed from the real model is its *shape*: the 2:1
    grouped-query ratio, a head dimension decoupled from the hidden size, the 3x MLP
    expansion, SwiGLU, RMSNorm at 1e-6, tied embeddings, and the large RoPE theta. Those
    are the choices that make attention and positional behaviour representative; guessing
    them (as an earlier version of this function did) produces a generic transformer
    wearing the Qwen name.

    The vocabulary is emphatically *not* Qwen's. The real tokenizer has 151936 entries,
    which at this scale would put almost every parameter into an embedding table for
    tokens the arithmetic task never emits.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    ref = QWEN3_REFERENCE
    heads = min(ref["num_attention_heads"], max(2, hidden // 32))
    kv_heads = max(1, heads // 2)  # preserve the real 2:1 GQA ratio

    cfg = AutoConfig.for_model(
        "qwen3",
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=ref["head_dim"] // 2,  # scaled down, still decoupled from hidden_size
        intermediate_size=hidden * ref["intermediate_ratio"],
        max_position_embeddings=256,
        rope_theta=ref["rope_theta"],
        rms_norm_eps=ref["rms_norm_eps"],
        hidden_act=ref["hidden_act"],
        attention_bias=ref["attention_bias"],
        tie_word_embeddings=ref["tie_word_embeddings"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def pretrain(model, tokenizer, task, steps=3000, batch_size=64, lr=3e-3, seq_len=48, log_every=500):
    """Teacher-forced pretraining so RL starts from a policy with something to shape.

    Without this the model emits noise, every accuracy reward is 0, and the RL run measures
    nothing: a frontier needs the policy to already reach both ends of the trade-off some
    of the time.
    """
    import torch

    torch.set_num_threads(4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        rows = task.corpus(batch_size)
        seqs = [tokenizer.encode(r)[:seq_len] for r in rows]
        width = max(len(s) for s in seqs)
        ids = torch.tensor([s + [tokenizer.pad_token_id] * (width - len(s)) for s in seqs])
        labels = ids.clone()
        labels[ids == tokenizer.pad_token_id] = -100
        out = model(input_ids=ids, attention_mask=(ids != tokenizer.pad_token_id).long(), labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.detach().item())
        if log_every and step % log_every == 0:
            print(f"  pretrain step {step:5d} loss {np.mean(losses[-100:]):.4f}", flush=True)
    return losses


def measure_style_gap(model, tokenizer, task, n: int = 200, max_new_tokens: int = 40):
    """Does the worked style actually beat the direct style on accuracy?

    This is the premise the whole tiny-LM experiment rests on. If the two styles are
    equally accurate there is no accuracy/brevity frontier here, only a length knob, and
    any 'frontier' the RL run appears to find is an artifact.
    """
    import torch

    from ..llm.rewards import extract_answer

    model.eval()
    results = {}
    for style in ("worked", "direct"):
        correct = 0
        lengths = []
        for _ in range(n):
            a, b = task.sample()
            # Prime the style by prefilling its first characters, so the comparison is
            # between styles rather than between whatever the model felt like emitting.
            prefix = task.question(a, b) + ("" if style == "worked" else "#")
            enc = tokenizer([prefix], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = prefix + tokenizer.decode(out[0, enc["input_ids"].shape[1] :])
            text = text.split("\n")[0]
            pred = extract_answer(text.split("#")[-1]) if "#" in text else None
            lengths.append(len(text) - len(task.question(a, b)))
            if pred is not None and pred.strip() == str(a + b):
                correct += 1
        results[style] = {"accuracy": correct / n, "mean_length": float(np.mean(lengths))}
    return results
