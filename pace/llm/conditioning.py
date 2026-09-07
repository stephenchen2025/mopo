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
    "format_preference",
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


DEFAULT_SYSTEM = (
    "You are a careful assistant. The <preference> line states how the user weights each "
    "objective for this request; weights sum to 1. Satisfy the high-weight objectives "
    "first, and trade away the low-weight ones as needed."
)


def build_prompt(
    question: str,
    w: np.ndarray,
    names: tuple[str, ...],
    system: str = DEFAULT_SYSTEM,
) -> list[dict[str, str]]:
    """Build a chat-format prompt carrying the preference direction."""
    return [
        {"role": "system", "content": f"{system}\n{format_preference(w, names)}"},
        {"role": "user", "content": question},
    ]
