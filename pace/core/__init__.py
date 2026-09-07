"""Torch-free core of PaCE: frontier math, scalarization, direction sampling, advantages."""

from .advantages import (
    AdvantageOutput,
    dominance_advantages,
    linear_scalar_advantages,
    mo_grpo_advantages,
    pace_advantages,
)
from .config import PaCEConfig
from .directions import BanditConfig, CoverageBandit, UniformDirections, das_dennis
from .pareto import (
    crowding_distance,
    dominates,
    hypervolume,
    nondominated_mask,
    nondominated_sort,
    normalized_crowding,
)
from .scalarization import (
    achievement_matrix,
    linear,
    rank_normalize,
    smooth_tchebycheff,
    tchebycheff,
)

__all__ = [
    "AdvantageOutput",
    "BanditConfig",
    "CoverageBandit",
    "PaCEConfig",
    "UniformDirections",
    "achievement_matrix",
    "crowding_distance",
    "das_dennis",
    "dominance_advantages",
    "dominates",
    "hypervolume",
    "linear",
    "linear_scalar_advantages",
    "mo_grpo_advantages",
    "nondominated_mask",
    "nondominated_sort",
    "normalized_crowding",
    "pace_advantages",
    "rank_normalize",
    "smooth_tchebycheff",
    "tchebycheff",
]
