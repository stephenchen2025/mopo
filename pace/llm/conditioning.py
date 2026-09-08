"""Getting the preference direction ``w`` into a language model, and reading it back out.

The default mechanism is a control line in the system turn. It costs nothing, works with
any chat model, and keeps the experiment focused on the advantage estimator -- which is
the part of PaCE that is actually new. Its weakness is that nothing *structurally* forces
the model to attend to it, which is exactly why
:func:`pace.core.metrics.controllability` is a first-class metric rather than a footnote.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "OBJECTIVE_PRESETS",
    "OBJECTIVE_PHRASES",
    "format_preference",
    "verbalize_preference",
    "parse_preference",
    "build_prompt",
]

OBJECTIVE_PRESETS: dict[str, tuple[str, ...]] = {
    "accuracy_brevity": ("accuracy", "brevity"),
    "accuracy_brevity_format": ("accuracy", "brevity", "format"),
    "helpful_harmless_brief": ("helpfulness", "harmlessness", "brevity"),
}


def format_preference(w: np.ndarray, names: tuple[str, ...], precision: int = 2) -> str:
    """Render a direction as the control line the policy is conditioned on.

    Example: ``<preference>accuracy=0.70 brevity=0.30</preference>``

    Rendering at fixed precision quantizes the direction the model sees. That is
    deliberate: it bounds how many distinct conditioning strings exist, which makes the
    conditioning easier to learn, and it costs nothing in frontier resolution because two
    directions that agree to two decimals target the same trade-off anyway.
    """
    w = np.asarray(w, dtype=float)
    if w.shape[0] != len(names):
        raise ValueError(f"got {w.shape[0]} weights for {len(names)} objectives")
    parts = " ".join(f"{n}={v:.{precision}f}" for n, v in zip(names, w))
    return f"<preference>{parts}</preference>"


def parse_preference(text: str, names: tuple[str, ...]) -> np.ndarray | None:
    """Recover a direction from a control line. Returns ``None`` if absent or malformed.

    Used to verify that prompts round-trip, and to score generations logged without their
    directions attached.
    """
    match = re.search(r"<preference>(.*?)</preference>", text, re.DOTALL)
    if not match:
        return None
    found = dict(re.findall(r"([A-Za-z_]+)\s*=\s*([0-9]*\.?[0-9]+)", match.group(1)))
    try:
        w = np.array([float(found[n]) for n in names], dtype=float)
    except KeyError:
        return None
    total = w.sum()
    return w / total if total > 0 else None


DEFAULT_SYSTEM = "You are a careful assistant. Follow the instructions below exactly."


def build_prompt(
    question: str,
    w: np.ndarray,
    names: tuple[str, ...],
    system: str = DEFAULT_SYSTEM,
    style: str = "verbal",
) -> list[dict[str, str]]:
    """Build a chat-format prompt carrying the preference direction.

    Args:
        style: ``"verbal"`` (default) renders the direction as instructions, which is the
            only form measured to steer a real instruct model -- see
            :func:`verbalize_preference`. ``"numeric"`` emits the weight vector and is kept
            for ablations and for models fine-tuned to read it.
    """
    if style == "verbal":
        control = verbalize_preference(w, names)
    elif style == "numeric":
        control = format_preference(w, names)
    else:
        raise ValueError(f"unknown conditioning style: {style!r}")
    return [
        {"role": "system", "content": f"{system}\n{control}"},
        {"role": "user", "content": question},
    ]


# --------------------------------------------------------------------------------------
# Verbalized conditioning
# --------------------------------------------------------------------------------------

OBJECTIVE_PHRASES: dict[str, tuple[str, str, str]] = {
    # objective -> (high weight, medium weight, low weight)
    "accuracy": (
        "Work through the problem step by step and check your arithmetic before answering.",
        "Show the key steps of your reasoning.",
        "Do not spend effort explaining your reasoning.",
    ),
    "brevity": (
        "Answer with the result only. No working, no explanation, no preamble.",
        "Keep the answer short.",
        "Length is not a concern; be as thorough as you like.",
    ),
    "format": (
        "Follow the requested output format exactly.",
        "Use the requested output format.",
        "The output format is not important.",
    ),
    "helpfulness": (
        "Be as genuinely useful as possible, anticipating what the user needs next.",
        "Be helpful.",
        "Do not elaborate beyond what was asked.",
    ),
    "harmlessness": (
        "Refuse anything unsafe and add caveats wherever there is any risk.",
        "Note any relevant safety caveats.",
        "Safety caveats are not needed here.",
    ),
}


def verbalize_preference(w: np.ndarray, names: tuple[str, ...]) -> str:
    """Render a direction as natural-language instructions rather than numeric weights.

    **This is the conditioning format that actually works, and the difference is not
    marginal.** Measured zero-shot on Qwen3-0.6B over two-digit multiplication, the numeric
    form (``<preference>accuracy=1.00 brevity=0.00</preference>``) produced *identical*
    behaviour at every point on the simplex -- 8 words and rank correlation +0.000 between
    requested weight and achieved reward. The same model swung from 3 words to 68, and from
    0.50 to 0.75 accuracy, in response to plain instructions ("answer with the number only"
    vs "show every step of your working").

    So the model was never the problem; the encoding was. An instruction-tuned model has
    seen a great deal of text telling it how to behave and essentially none pairing a
    decimal weight vector with a behaviour.

    This matters most at the *start* of RL. PaCE's cross-direction advantage matrix needs
    the rollouts in a group to actually differ by direction; if the policy ignores ``w`` at
    initialization, that matrix is rank-one and the mechanism has nothing to bootstrap
    from (``tests/test_advantages.py`` asserts that degenerate case directly). Numeric
    conditioning could in principle be *learned* from the reward, but it starts from no
    signal at all, which is a far worse place to begin.

    Weights are bucketed into three bands rather than mapped continuously: the phrasing is
    what the model responds to, and interpolating text does not produce interpolated
    behaviour. Frontier resolution comes from *which* objectives are emphasized, not from
    fine gradations in how emphatically.
    """
    w = np.asarray(w, dtype=float)
    if w.shape[0] != len(names):
        raise ValueError(f"got {w.shape[0]} weights for {len(names)} objectives")
    lines = []
    for weight, name in zip(w, names):
        phrases = OBJECTIVE_PHRASES.get(name)
        if phrases is None:
            lines.append(f"Weight on {name}: {'high' if weight >= 0.5 else 'low'}.")
            continue
        # Bands are relative to a uniform direction (1/m), not to an absolute 0.5.
        # With an absolute threshold and m=2, the balanced direction w=(0.5, 0.5) lands in
        # the "high" band for *both* objectives and emits contradictory instructions
        # ("work through it step by step" alongside "answer with the result only"), which
        # is worse than no conditioning at all.
        uniform = 1.0 / len(names)
        band = 0 if weight >= 1.5 * uniform else (1 if weight >= 0.5 * uniform else 2)
        lines.append(phrases[band])
    return " ".join(lines)
