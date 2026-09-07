"""A minimal preference-conditioned softmax policy for the synthetic environment.

Deliberately linear. The mapping from a direction ``w`` to the best action is a piecewise
sweep across actions, and the upper envelope of linear logits reproduces exactly that, so
this policy class *is* expressive enough to cover the whole frontier. That matters for the
experiment's validity: if a method fails to cover the front here, the cause is the
objective it optimizes, not a capacity limit.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LinearSoftmaxPolicy"]


class LinearSoftmaxPolicy:
    """``pi(a | context, w) = softmax(theta @ phi(context, w))``."""

    def __init__(
        self,
        n_actions: int,
        n_contexts: int,
        n_objectives: int,
        seed: int = 0,
        init_scale: float = 0.0,
    ) -> None:
        self.n_actions = n_actions
        self.n_contexts = n_contexts
        self.n_objectives = n_objectives
        self.n_features = n_contexts + n_objectives + n_contexts * n_objectives + 1
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0.0, init_scale, size=(n_actions, self.n_features))

    def features(self, context: int, W: np.ndarray) -> np.ndarray:
        """``(K, m) -> (K, n_features)``. Context one-hot, direction, their interaction, bias."""
        W = np.atleast_2d(np.asarray(W, dtype=float))
        K = W.shape[0]
        ctx = np.zeros((K, self.n_contexts))
        ctx[:, context] = 1.0
        inter = (ctx[:, :, None] * W[:, None, :]).reshape(K, -1)
        return np.concatenate([ctx, W, inter, np.ones((K, 1))], axis=1)

    def probs(self, context: int, W: np.ndarray) -> np.ndarray:
        """Action probabilities. ``(K, n_actions)``."""
        phi = self.features(context, W)
        z = phi @ self.theta.T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def sample(self, context: int, W: np.ndarray, rng: np.random.Generator):
        """Sample one action per row of ``W``. Returns ``(actions, probs, features)``."""
        phi = self.features(context, W)
        z = phi @ self.theta.T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        cum = p.cumsum(axis=1)
        u = rng.random(size=(p.shape[0], 1))
        actions = (u > cum).sum(axis=1).clip(0, self.n_actions - 1)
        return actions, p, phi

    def policy_gradient(
        self, actions: np.ndarray, p: np.ndarray, phi: np.ndarray, advantages: np.ndarray
    ) -> np.ndarray:
        """``sum_k A_k * grad_theta log pi(a_k | phi_k)``, averaged over the group."""
        onehot = np.zeros_like(p)
        onehot[np.arange(len(actions)), actions] = 1.0
        dz = (onehot - p) * np.asarray(advantages, dtype=float)[:, None]  # (K, A)
        return dz.T @ phi / len(actions)

    def entropy_gradient(self, p: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Gradient of mean action entropy, for an exploration bonus."""
        logp = np.log(np.clip(p, 1e-12, None))
        H = -(p * logp).sum(axis=1, keepdims=True)
        dz = -p * (logp + H)
        return dz.T @ phi / p.shape[0]
