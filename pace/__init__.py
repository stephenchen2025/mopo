"""PaCE -- Pareto Coverage Exploration.

A group-relative RL algorithm that trains a single preference-conditioned policy to cover a
Pareto frontier, rather than N separate runs to sketch N points on it.

The ``pace.core`` subpackage is numpy-only and carries the whole algorithm; ``pace.envs``
holds the synthetic validation environment; ``pace.llm`` holds the language-model training
path and needs the ``llm`` extra.
"""

from .core.advantages import pace_advantages
from .core.config import PaCEConfig

__version__ = "0.1.0"
__all__ = ["PaCEConfig", "pace_advantages", "__version__"]
