"""Multi-objective reward functions for the LLM task.

The default suite is accuracy / brevity / format on verifiable math, chosen because the
objectives genuinely fight: the shortest correct answer to a hard problem is usually
wrong, so the accuracy-brevity frontier is real rather than manufactured. All three are
programmatic -- no reward model, no judge API, no per-rollout inference cost -- which
matters because PaCE spends ``K`` rollouts per prompt per step.

Every function returns a value in ``[0, 1]``, higher is better. Note that PaCE does not
*need* commensurable rewards -- rank normalization handles arbitrary monotone rescaling --
but keeping them in a common range makes the logs readable.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "extract_answer",
    "accuracy_reward",
    "brevity_reward",
    "format_reward",
    "RewardSuite",
]

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_HASHES = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_LAST_NUMBER = re.compile(r"(-?[\d,]*\.?\d+)")


def extract_answer(text: str) -> str | None:
    """Pull a final answer out of a completion, preferring explicit delimiters.

    The ``####`` marker needs care. The GSM8K convention puts it *last*
    (``reasoning...\n#### 95``), but instruction-tuned models routinely put it first and
    the answer at the end (``#### 37 + 58 = 95``). Capturing the number that immediately
    follows the marker gets the first operand in that second case -- so a completion that
    is entirely correct scores zero on accuracy.

    That failure is invisible in training curves: the accuracy objective just looks
    stubbornly hard. It was caught by generating from a real model and noticing that
    ``#### 37 + 58 = 95`` scored 0 against a gold answer of 95.

    So: after the last marker, take the *last* number, which is right under both
    conventions.
    """
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip().replace(",", "")

    marker = text.rfind("####")
    if marker != -1:
        tail = _LAST_NUMBER.findall(text[marker + 4 :])
        if tail:
            return tail[-1].strip().replace(",", "")

    matches = _LAST_NUMBER.findall(text)
    return matches[-1].strip().replace(",", "") if matches else None


def _numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a.strip() == b.strip()


def accuracy_reward(completion: str, gold: str) -> float:
    """1.0 if the extracted final answer matches the reference, else 0.0."""
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    return 1.0 if _numeric_equal(pred, str(gold)) else 0.0


def brevity_reward(completion: str, target_tokens: int = 256, tokenizer=None) -> float:
    """Smoothly decreasing in length: ``exp(-n / target)``, in ``[0, 1]``.

    Exponential rather than a hard cap so there is always a gradient toward shorter: a
    clipped ``1 - n/max`` gives every over-length completion the same zero reward and
    stops distinguishing "slightly too long" from "runaway", which is the difference
    worth learning.
    """
    n = len(tokenizer.encode(completion)) if tokenizer is not None else len(completion.split())
    return float(np.exp(-n / max(target_tokens, 1)))


def format_reward(completion: str) -> float:
    """Rewards a completion that shows work *and* marks its final answer.

    Deliberately antagonistic to brevity -- an empty completion scores 0 here and 1.0 on
    brevity -- because a frontier is only interesting where the objectives disagree.
    """
    has_answer = bool(_BOXED.search(completion) or _HASHES.search(completion))
    has_reasoning = len(completion.split()) >= 20
    return 0.5 * float(has_answer) + 0.5 * float(has_reasoning)


class RewardSuite:
    """Evaluates a list of completions into a ``(n, m)`` reward matrix.

    Args:
        objectives: names, in the order they index the reward vector and the direction.
        target_tokens: brevity scale.
        tokenizer: optional; if given, brevity counts real tokens instead of whitespace
            words. Worth passing -- whitespace words undercount code and math badly.
    """

    def __init__(
        self,
        objectives: tuple[str, ...] = ("accuracy", "brevity", "format"),
        target_tokens: int = 256,
        tokenizer=None,
    ) -> None:
        unknown = set(objectives) - {"accuracy", "brevity", "format"}
        if unknown:
            raise ValueError(f"unknown objectives: {sorted(unknown)}")
        self.objectives = objectives
        self.target_tokens = target_tokens
        self.tokenizer = tokenizer

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    def __call__(self, completions: list[str], golds: list[str]) -> np.ndarray:
        if len(completions) != len(golds):
            raise ValueError("completions and golds must be the same length")
        rows = []
        for completion, gold in zip(completions, golds):
            row = []
            for name in self.objectives:
                if name == "accuracy":
                    row.append(accuracy_reward(completion, gold))
                elif name == "brevity":
                    row.append(brevity_reward(completion, self.target_tokens, self.tokenizer))
                else:
                    row.append(format_reward(completion))
            rows.append(row)
        return np.asarray(rows, dtype=float)
