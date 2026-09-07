"""Use the PaCE advantage inside an existing TRL ``GRPOTrainer`` setup.

TRL's internals move between releases, so this deliberately exposes only a pure function
over ``(rewards, directions)`` -- no subclassing, nothing that breaks on upgrade. Wire it
in wherever your trainer turns a group's rewards into advantages.

Two things must be true for this to be PaCE rather than decoration:

1. The ``K`` rollouts of a group must be generated under ``K`` *different* directions --
   use :func:`sample_group_directions` and put each direction in its own prompt. If every
   rollout in the group shares one direction, the cross-direction matrix is rank-one and
   the whole mechanism reduces to an ordinary group baseline.
2. ``rewards`` must arrive as a ``(K, m)`` matrix, not a pre-summed scalar. TRL's usual
   pattern of collapsing multiple reward functions into a scalar throws away exactly the
   information PaCE runs on.

**Status: not executed against a live TRL install.**
"""

from __future__ import annotations

import numpy as np

from ..core.advantages import pace_advantages
from ..core.config import PaCEConfig
from ..core.directions import CoverageBandit, das_dennis

__all__ = ["pace_advantage_fn", "sample_group_directions", "make_bandit"]


def make_bandit(n_objectives: int, config: PaCEConfig | None = None, seed: int = 0):
    """Build the coverage bandit to draw group directions from."""
    config = config or PaCEConfig()
    return CoverageBandit(das_dennis(n_objectives, config.n_partitions), config=config.bandit, seed=seed)


def sample_group_directions(bandit, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``group_size`` distinct directions for one prompt's rollouts."""
    return bandit.sample(group_size)


def pace_advantage_fn(rewards: np.ndarray, directions: np.ndarray, config: PaCEConfig | None = None) -> np.ndarray:
    """``(K, m)`` rewards + ``(K, m)`` directions -> ``(K,)`` advantages."""
    return pace_advantages(rewards, directions, config or PaCEConfig()).advantages
