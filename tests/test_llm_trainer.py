"""End-to-end execution of the LLM training path on a tiny local model.

The LLM path cannot be checked against a real checkpoint here -- that needs a GPU and hub
access -- but almost none of what goes wrong in an RL trainer needs a big model to expose.
Shapes, masking, gradient flow, and the importance-ratio invariant all show up on a 37k
parameter randomly-initialized GPT-2 with a stub tokenizer, and running it caught a real
bug (see ``test_kl_is_exactly_zero_on_a_single_inner_epoch``).

Skipped unless torch and transformers are installed, so the core test suite stays
dependency-free.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from pace.core.directions import CoverageBandit, das_dennis  # noqa: E402
from pace.core.metrics import FrontierArchive  # noqa: E402
from pace.core.normalizer import RunningMinMax  # noqa: E402
from pace.llm.rewards import RewardSuite  # noqa: E402
from pace.llm.trainer import LLMTrainConfig, PaCETrainer, _disable_dropout  # noqa: E402

VOCAB = 128


class StubTokenizer:
    """Minimal byte-level tokenizer covering exactly the surface PaCETrainer uses."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1
    padding_side = "left"

    def encode(self, text, **kw):
        return [max(2, (b % (VOCAB - 2)) + 2) for b in text.encode()[:64]] or [2]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (int(i) % 26)) for i in ids if int(i) > 1)

    def batch_decode(self, seqs, skip_special_tokens=True):
        return [self.decode(s, skip_special_tokens) for s in seqs]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None, padding=False, **kw):
        texts = [text] if isinstance(text, str) else text
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


def tiny_model():
    cfg = AutoConfig.for_model(
        "gpt2",
        vocab_size=VOCAB,
        n_positions=256,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.config.pad_token_id = 0
    return model.to(torch.float32)


def build_trainer():
    """A real PaCETrainer wired to the tiny model, bypassing only the hub download."""
    cfg = LLMTrainConfig(
        objectives=("accuracy", "brevity", "format"),
        group_size=6,
        prompts_per_step=2,
        max_new_tokens=16,
        use_lora=False,
        lr=1e-3,
    )
    cfg.pace.n_partitions = 4

    tr = PaCETrainer.__new__(PaCETrainer)
    tr.torch = torch
    tr.config = cfg
    tr.tokenizer = StubTokenizer()
    tr.model = tiny_model()
    _disable_dropout(tr.model)
    tr.optimizer = torch.optim.AdamW(tr.model.parameters(), lr=cfg.lr)
    tr.rewards = RewardSuite(cfg.objectives, cfg.target_tokens, tr.tokenizer)
    m = len(cfg.objectives)
    tr.bandit = CoverageBandit(das_dennis(m, cfg.pace.n_partitions), config=cfg.pace.bandit, seed=0)
    tr.normalizer = RunningMinMax(m)
    tr.archive = FrontierArchive()
    tr.rng = np.random.default_rng(0)
    return tr


BATCH = [
    {"question": "what is 2+2?", "answer": "4"},
    {"question": "what is 7*6?", "answer": "42"},
]


def test_disable_dropout_makes_the_policy_deterministic():
    model = tiny_model()
    assert model.training, "a freshly built model is in train mode, so dropout is live"
    enc = StubTokenizer()(["user: hi"], return_tensors="pt", padding=True)

    def logprobs():
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        return torch.log_softmax(logits.float(), dim=-1)

    assert not torch.allclose(logprobs(), logprobs()), "dropout should make passes differ"
    assert _disable_dropout(model) > 0
    assert torch.allclose(logprobs(), logprobs())


def test_train_step_runs_and_moves_the_policy():
    tr = build_trainer()
    before = [p.detach().clone() for p in tr.model.parameters()]
    logs = tr.train_step(BATCH)

    assert np.isfinite(logs["pg_loss"])
    assert any(not torch.equal(b, p) for b, p in zip(before, tr.model.parameters())), "no gradient reached the policy"
    for name in tr.config.objectives:
        assert f"reward/{name}" in logs
    assert 0.0 <= logs["nondominated_frac"] <= 1.0


def test_kl_is_exactly_zero_on_a_single_inner_epoch():
    """Regression guard for a bug this suite was written to find.

    With one gradient step per batch of fresh rollouts, the old and new log-probabilities
    come from identical weights, so the importance ratio is exactly 1 and the KL term is
    exactly 0. It was not: dropout was live during both forward passes, so the ratio was
    sampling noise, PPO clipping fired on that noise, and the KL penalty charged the policy
    for it. If this assertion ever fails again, something has re-enabled stochasticity
    between the two passes.
    """
    tr = build_trainer()
    for _ in range(3):
        logs = tr.train_step(BATCH)
        assert logs["kl"] == pytest.approx(
            0.0, abs=1e-9
        ), f"KL should be exactly 0 with one inner epoch, got {logs['kl']}"


def test_bandit_and_normalizer_receive_updates():
    tr = build_trainer()
    tr.train_step(BATCH)
    assert tr.bandit.counts.sum() == tr.config.group_size * tr.config.prompts_per_step
    assert tr.normalizer.initialized


def test_evaluate_sweeps_directions_and_fills_the_archive():
    tr = build_trainer()
    metrics = tr.evaluate(BATCH, n_prompts=2)
    expected = len(das_dennis(len(tr.config.objectives), tr.config.eval_partitions))
    assert len(tr.archive) == expected
    assert tr.archive.rewards.shape == (expected, len(tr.config.objectives))
    for key in ("hypervolume", "spacing", "max_spread", "controllability_mean"):
        assert key in metrics and np.isfinite(metrics[key])
