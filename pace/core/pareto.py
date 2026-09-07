"""Pareto-set primitives: dominance, non-dominated sorting, crowding, hypervolume.

Convention throughout this package: **all objectives are maximized**. A reward vector
``f`` dominates ``g`` iff ``all(f >= g) and any(f > g)``.

Everything here is numpy-only and torch-free on purpose: the frontier math is the part
of PaCE that needs to be unit-testable without a GPU or a model.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "dominates",
    "nondominated_mask",
    "nondominated_sort",
    "crowding_distance",
    "normalized_crowding",
    "hypervolume",
]


def dominates(f: np.ndarray, g: np.ndarray) -> bool:
    """True if ``f`` Pareto-dominates ``g`` (maximization)."""
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    return bool(np.all(f >= g) and np.any(f > g))


def _domination_matrix(F: np.ndarray) -> np.ndarray:
    """``D[i, j]`` is True iff point ``i`` dominates point ``j``. O(n^2 m)."""
    ge = np.all(F[:, None, :] >= F[None, :, :], axis=-1)
    gt = np.any(F[:, None, :] > F[None, :, :], axis=-1)
    return ge & gt


def _nondominated_mask_2d(F: np.ndarray) -> np.ndarray:
    """O(n log n) sweep for two objectives."""
    n = F.shape[0]
    # Sort by objective 0 descending, objective 1 descending as tie-break. Sweeping in
    # that order, a point survives iff no earlier point has an objective-1 value >= its
    # own (an earlier point already has objective 0 >= this one's).
    order = np.lexsort((-F[:, 1], -F[:, 0]))
    mask = np.zeros(n, dtype=bool)
    best_y = -np.inf
    prev = None
    for i in order:
        x, y = F[i]
        if y > best_y:
            mask[i] = True
            best_y = y
            prev = (x, y)
        elif prev is not None and x == prev[0] and y == prev[1]:
            mask[i] = True  # exact duplicates of a surviving point are also non-dominated
    return mask


def _nondominated_mask_large(F: np.ndarray) -> np.ndarray:
    """Incremental filter for point sets too large for an n^2 comparison matrix."""
    n = F.shape[0]
    order = np.argsort(-F.sum(axis=1), kind="stable")  # promising points first
    front_pts = np.empty((0, F.shape[1]))
    front_idx: list[int] = []
    for i in order:
        f = F[i]
        if front_pts.shape[0]:
            if np.any(np.all(front_pts >= f, axis=1) & np.any(front_pts > f, axis=1)):
                continue
            keep = ~(np.all(f >= front_pts, axis=1) & np.any(f > front_pts, axis=1))
            if not keep.all():
                front_pts = front_pts[keep]
                front_idx = [front_idx[k] for k in np.flatnonzero(keep)]
        front_pts = np.vstack([front_pts, f])
        front_idx.append(int(i))
    mask = np.zeros(n, dtype=bool)
    mask[front_idx] = True
    return mask


def nondominated_mask(F: np.ndarray, dense_limit: int = 4000) -> np.ndarray:
    """Boolean mask of the non-dominated points of ``F`` (shape ``(n, m)``).

    Uses the O(n^2 m) comparison matrix for small inputs, an O(n log n) sweep for two
    objectives, and an incremental filter otherwise -- the dense matrix is 70 GB at
    n = 200k, which is reachable when filtering an enumerated achievable set.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    n = F.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n <= dense_limit:
        return ~np.any(_domination_matrix(F), axis=0)
    if F.shape[1] == 2:
        return _nondominated_mask_2d(F)
    return _nondominated_mask_large(F)


def nondominated_sort(F: np.ndarray) -> np.ndarray:
    """Non-dominated sorting. Returns a 1-based front index per point.

    Front 1 is the non-dominated set, front 2 is what is non-dominated once front 1 is
    removed, and so on. This is the NSGA-II layering, written the slow-but-obvious way
    (group sizes here are ~8-64, so the fast version buys nothing).
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    n = F.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)

    D = _domination_matrix(F)
    fronts = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    current = 1
    while remaining.any():
        # Among the points still unassigned, find those dominated by nobody unassigned.
        dominated_by_remaining = np.any(D[np.ix_(remaining, remaining)], axis=0)
        idx = np.flatnonzero(remaining)
        layer = idx[~dominated_by_remaining]
        if layer.size == 0:  # pragma: no cover - impossible for a strict partial order
            layer = idx
        fronts[layer] = current
        remaining[layer] = False
        current += 1
    return fronts


def crowding_distance(F: np.ndarray, fronts: np.ndarray | None = None) -> np.ndarray:
    """NSGA-II crowding distance, computed within each front.

    Boundary points of a front get ``inf`` (the standard NSGA-II convention: extremes are
    maximally valuable for spread). Use :func:`normalized_crowding` to get a bounded
    version suitable for mixing into an advantage.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    n, m = F.shape
    if fronts is None:
        fronts = np.ones(n, dtype=int)
    dist = np.zeros(n, dtype=float)

    for front_id in np.unique(fronts):
        members = np.flatnonzero(fronts == front_id)
        if members.size <= 2:
            dist[members] = np.inf
            continue
        for j in range(m):
            order = members[np.argsort(F[members, j], kind="stable")]
            lo, hi = F[order[0], j], F[order[-1], j]
            spread = hi - lo
            if spread <= 0:
                # Objective is constant on this front, so no point is extreme in it. The
                # boundary-is-infinite convention must not fire here: with every value
                # equal, argsort order is arbitrary and marking its ends would manufacture
                # a spread signal out of tie-breaking noise.
                continue
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            interior = order[1:-1]
            dist[interior] += (F[order[2:], j] - F[order[:-2], j]) / spread
    return dist


def normalized_crowding(F: np.ndarray, fronts: np.ndarray | None = None) -> np.ndarray:
    """Crowding distance mapped to ``[0, 1]``, with infinities pinned to the top.

    A point in a sparse region of its front scores near 1; a point crowded by neighbours
    scores near 0. This is the spread pressure in the PaCE advantage.
    """
    d = crowding_distance(F, fronts)
    finite = np.isfinite(d)
    if not finite.any():
        return np.ones_like(d)
    hi = float(d[finite].max())
    if hi <= 0:
        hi = 1.0
    # Infinities (front boundaries) are pinned to twice the largest finite distance, so
    # after scaling the boundaries sit at 1.0 and the sparsest interior point at 0.5.
    # Boundaries staying strictly above every interior point is the point of the
    # convention -- they are the extremes that hold the frontier open.
    out = np.where(finite, d, 2.0 * hi) / (2.0 * hi)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# Hypervolume
# --------------------------------------------------------------------------------------


def _hv_exact_2d(F: np.ndarray, ref: np.ndarray) -> float:
    keep = np.all(F >= ref, axis=1)
    F = F[keep]
    if F.shape[0] == 0:
        return 0.0
    F = F[nondominated_mask(F)]
    # Sort by objective 0 descending; for a non-dominated set this makes obj 1 ascending.
    order = np.argsort(-F[:, 0], kind="stable")
    F = F[order]
    total = 0.0
    prev_y = ref[1]
    for x, y in F:
        if y <= prev_y:
            continue
        total += (x - ref[0]) * (y - prev_y)
        prev_y = y
    return float(total)


def _hv_inclusion_exclusion(F: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume by inclusion-exclusion. O(2^n) -- test oracle only."""
    from itertools import combinations

    keep = np.all(F >= ref, axis=1)
    F = F[keep]
    n = F.shape[0]
    if n == 0:
        return 0.0
    if n > 20:
        raise ValueError(f"inclusion-exclusion is exponential; refusing n={n} > 20")
    total = 0.0
    for size in range(1, n + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for combo in combinations(range(n), size):
            corner = np.min(F[list(combo)], axis=0)  # intersection of boxes
            total += sign * float(np.prod(np.maximum(corner - ref, 0.0)))
    return total


def _hv_monte_carlo(F: np.ndarray, ref: np.ndarray, n_samples: int, seed: int) -> float:
    keep = np.all(F >= ref, axis=1)
    F = F[keep]
    if F.shape[0] == 0:
        return 0.0
    F = F[nondominated_mask(F)]
    hi = F.max(axis=0)
    box = float(np.prod(hi - ref))
    if box <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    pts = rng.uniform(ref, hi, size=(n_samples, F.shape[1]))
    # A sample counts if it is dominated by at least one front point.
    inside = np.zeros(n_samples, dtype=bool)
    for f in F:  # loop over front points to keep peak memory at O(n_samples)
        inside |= np.all(pts <= f, axis=1)
    return box * float(inside.mean())


def hypervolume(
    F: np.ndarray,
    ref: np.ndarray,
    method: str = "auto",
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """Hypervolume of the region dominated by ``F`` and bounded below by ``ref``.

    Args:
        F: ``(n, m)`` reward vectors (maximization).
        ref: ``(m,)`` reference point; must be worse than the points that should count.
            Points not dominating ``ref`` in every objective are dropped.
        method: ``"auto"`` uses the exact sweep for ``m == 2`` and Monte Carlo above.
            ``"exact"`` forces inclusion-exclusion (exponential, tests only).

    The reference point is a modelling choice, not a detail: hypervolume comparisons are
    only meaningful between runs that share one. Fix it per task and record it.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    ref = np.asarray(ref, dtype=float)
    if F.shape[0] == 0:
        return 0.0
    if F.shape[1] != ref.shape[0]:
        raise ValueError(f"objective mismatch: F has {F.shape[1]}, ref has {ref.shape[0]}")

    if method == "auto":
        method = "exact2d" if F.shape[1] == 2 else "mc"
    if method == "exact2d":
        if F.shape[1] != 2:
            raise ValueError("exact2d requires exactly 2 objectives")
        return _hv_exact_2d(F, ref)
    if method == "exact":
        return _hv_inclusion_exclusion(F, ref)
    if method == "mc":
        return _hv_monte_carlo(F, ref, n_samples, seed)
    raise ValueError(f"unknown hypervolume method: {method!r}")
