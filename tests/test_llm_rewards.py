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
    """The default is now verbal conditioning, which is what steers a real model.

    Numeric weights measured +0.000 controllability zero-shot on Qwen3-0.6B against +0.860
    for the verbal form, so the numeric round-trip is asserted through the explicit
    ``style="numeric"`` path rather than through the default.
    """
    names = ("accuracy", "brevity")
    msgs = build_prompt("2+2?", np.array([0.6, 0.4]), names)
    assert msgs[0]["role"] == "system" and msgs[1]["content"] == "2+2?"
    assert parse_preference(msgs[0]["content"], names) is None  # verbal, not numeric

    numeric = build_prompt("2+2?", np.array([0.6, 0.4]), names, style="numeric")
    np.testing.assert_allclose(parse_preference(numeric[0]["content"], names), [0.6, 0.4], atol=1e-6)


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


def test_extract_answer_handles_marker_first_completions():
    """Regression guard for a bug found by generating from a real model.

    Qwen3-0.6B answers `What is 37+58?` with `#### 37 + 58 = 95`: marker first, answer
    last. Capturing the number straight after `####` returns 37, so a fully correct
    completion scores zero accuracy -- and nothing in a training curve would reveal it,
    the accuracy objective would simply look impossible.
    """
    assert extract_answer("#### 37 + 58 = 95") == "95"
    assert accuracy_reward("#### 37 + 58 = 95", "95") == 1.0
    # The GSM8K convention (marker last) must keep working.
    assert extract_answer("some reasoning here\n#### 95") == "95"
    assert accuracy_reward("lots of working\n#### 95", "95") == 1.0
    # A boxed answer still wins over any marker.
    assert extract_answer(r"#### 37 + 58 = \boxed{95}") == "95"
    # Multiple markers: the last one governs.
    assert extract_answer("#### 12 first try\n#### 5 + 90 = 95") == "95"


def test_verbalized_conditioning_is_coherent_across_the_simplex():
    """Guards the banding fix in verbalize_preference.

    Bands are relative to a uniform direction, not an absolute 0.5. With an absolute
    threshold and m=2 the balanced direction lands in the "high" band for *both*
    objectives and emits contradictory instructions, which is worse than no conditioning.
    """
    import numpy as np

    from pace.llm.conditioning import OBJECTIVE_PHRASES, verbalize_preference

    names = ("accuracy", "brevity")
    acc_high, _, acc_low = OBJECTIVE_PHRASES["accuracy"]
    brev_high, _, brev_low = OBJECTIVE_PHRASES["brevity"]

    extreme_acc = verbalize_preference(np.array([1.0, 0.0]), names)
    assert acc_high in extreme_acc and brev_low in extreme_acc

    extreme_brev = verbalize_preference(np.array([0.0, 1.0]), names)
    assert brev_high in extreme_brev and acc_low in extreme_brev

    balanced = verbalize_preference(np.array([0.5, 0.5]), names)
    assert (
        acc_high not in balanced and brev_high not in balanced
    ), "a balanced direction must not demand both extremes at once"

    # Three objectives: a uniform direction is balanced on all of them.
    names3 = ("accuracy", "brevity", "format")
    uniform = verbalize_preference(np.array([1 / 3, 1 / 3, 1 / 3]), names3)
    for obj in names3:
        assert OBJECTIVE_PHRASES[obj][0] not in uniform


def test_build_prompt_supports_both_conditioning_styles():
    import numpy as np

    from pace.llm.conditioning import build_prompt

    w, names = np.array([0.7, 0.3]), ("accuracy", "brevity")
    verbal = build_prompt("2+2?", w, names)[0]["content"]
    numeric = build_prompt("2+2?", w, names, style="numeric")[0]["content"]
    assert "<preference>" not in verbal and "<preference>" in numeric
    with pytest.raises(ValueError, match="unknown conditioning style"):
        build_prompt("2+2?", w, names, style="telepathy")
