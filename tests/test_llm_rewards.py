"""Tests for the LLM-side reward suite and preference conditioning.

These are pure Python and run without torch, a GPU, or a model -- which is the point:
the reward definitions are where multi-objective RL runs go quietly wrong, so they should
be testable in isolation.
"""

import numpy as np
import pytest

from pace.llm.conditioning import build_prompt, format_preference, parse_preference
from pace.llm.rewards import (
    RewardSuite,
    accuracy_reward,
    brevity_reward,
    extract_answer,
    format_reward,
)


def test_extract_answer_prefers_boxed():
    assert extract_answer(r"first 3 then \boxed{42}") == "42"
    assert extract_answer("reasoning\n#### 17") == "17"
    assert extract_answer("the total is 7") == "7"
    assert extract_answer("no digits at all") is None


def test_extract_answer_strips_thousands_separators():
    assert extract_answer(r"\boxed{1,234}") == "1234"


def test_accuracy_reward():
    assert accuracy_reward(r"\boxed{42}", "42") == 1.0
    assert accuracy_reward(r"\boxed{41}", "42") == 0.0
    assert accuracy_reward("nothing here", "42") == 0.0


def test_brevity_reward_is_monotonically_decreasing_in_length():
    short = brevity_reward("a b c", target_tokens=10)
    long = brevity_reward(" ".join(["word"] * 100), target_tokens=10)
    assert 0.0 <= long < short <= 1.0


def test_format_reward_components():
    assert format_reward("") == 0.0
    assert format_reward(r"\boxed{1}") == 0.5  # answer but no visible work
    assert format_reward(" ".join(["w"] * 25) + r" \boxed{1}") == 1.0


def test_objectives_actually_conflict():
    """If brevity and format did not disagree there would be no frontier to explore."""
    terse = r"\boxed{42}"
    verbose = " ".join(["step"] * 200) + r" \boxed{42}"
    assert brevity_reward(terse) > brevity_reward(verbose)
    assert format_reward(terse) < format_reward(verbose)


def test_reward_suite_shape_and_range():
    suite = RewardSuite(objectives=("accuracy", "brevity", "format"))
    R = suite([r"\boxed{42}", "wrong"], ["42", "42"])
    assert R.shape == (2, 3)
    assert (R >= 0).all() and (R <= 1).all()
    assert R[0, 0] == 1.0 and R[1, 0] == 0.0


def test_reward_suite_rejects_unknown_objective():
    with pytest.raises(ValueError, match="unknown objectives"):
        RewardSuite(objectives=("accuracy", "vibes"))


def test_preference_round_trips():
    names = ("accuracy", "brevity")
    w = np.array([0.7, 0.3])
    text = format_preference(w, names)
    np.testing.assert_allclose(parse_preference(text, names), w, atol=1e-6)


def test_parse_preference_renormalizes_and_handles_junk():
    names = ("accuracy", "brevity")
    np.testing.assert_allclose(
        parse_preference("<preference>accuracy=2.0 brevity=2.0</preference>", names),
        [0.5, 0.5],
    )
    assert parse_preference("no control line", names) is None
    assert parse_preference("<preference>accuracy=1.0</preference>", names) is None


def test_build_prompt_carries_the_direction():
    msgs = build_prompt("2+2?", np.array([0.6, 0.4]), ("accuracy", "brevity"))
    assert msgs[0]["role"] == "system" and msgs[1]["content"] == "2+2?"
    assert parse_preference(msgs[0]["content"], ("accuracy", "brevity")) is not None


def test_every_shipped_config_parses():
    """Config parsing is the one part of the LLM path testable without a GPU, so test it.

    This catches the failure mode where a YAML key drifts away from the dataclass field it
    is supposed to fill and nobody notices until a GPU has been booked.
    """
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from train_llm import load_config

    configs = sorted((pathlib.Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    assert configs, "no configs found"
    for path in configs:
        cfg = load_config(str(path))
        assert cfg.group_size >= 2
        assert len(cfg.objectives) >= 2
        n_directions = len(
            __import__("pace.core.directions", fromlist=["das_dennis"]).das_dennis(
                len(cfg.objectives), cfg.pace.n_partitions
            )
        )
        assert n_directions > cfg.group_size, (
            f"{path.name}: direction grid ({n_directions}) must exceed group_size "
            f"({cfg.group_size}) or the coverage bandit is inert -- a group sampled "
            f"without replacement would cover most of the grid regardless of its choices"
        )
