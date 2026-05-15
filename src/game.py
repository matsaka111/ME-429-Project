"""
Cournot game model and closed-form Nash equilibrium / social optimum.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Fixed parameters (held constant across all experiments)
# ---------------------------------------------------------------------------

A = 150.0      # choke price [EUR/MWh]
B = 0.03       # demand slope
C = 50.0       # marginal cost [EUR/MWh]
Q_BAR = 1e5    # per-player capacity (large enough that it never binds)


# ---------------------------------------------------------------------------
# Cournot game model
# ---------------------------------------------------------------------------

class CournotGame:
    """
    Symmetric Cournot game with N producers.

        J^i(q_i, q_{-i}) = c * q_i - q_i * P(S),   S = sum_j q_j
        P(S) = max(a - b * S^alpha, 0)

    Linear cost: c_i(q_i) = c * q_i, so c''_i = 0.
    """

    def __init__(self, N, alpha=1.0, a=A, b=B, c=C, q_bar=Q_BAR):
        self.N = N
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.q_bar = q_bar

    def P(self, S):
        return max(self.a - self.b * (S ** self.alpha), 0.0)

    def dP(self, S):
        # P'(S) = -b * alpha * S^(alpha - 1)
        if S <= 0:
            return 0.0 if self.alpha >= 1 else -1e12
        return -self.b * self.alpha * (S ** (self.alpha - 1))

    def d2P(self, S):
        # P''(S) = -b * alpha * (alpha - 1) * S^(alpha - 2)
        if self.alpha == 1:
            return 0.0
        if S <= 0:
            return 0.0
        return -self.b * self.alpha * (self.alpha - 1) * (S ** (self.alpha - 2))

    def grad_i(self, q_i, S):
        # d J^i / d q_i = c - P(S) - q_i * P'(S)
        return self.c - self.P(S) - q_i * self.dP(S)


# ---------------------------------------------------------------------------
# Closed-form Nash equilibrium and social optimum (symmetric, power-law P)
# ---------------------------------------------------------------------------

def closed_form_NE(game):
    """
    Symmetric NE. Each player's FOC c - P(S) - (S/N) P'(S) = 0 gives
        S* = ( (a - c) / (b * (1 + alpha/N)) )^(1/alpha)
    """
    if game.a <= game.c:
        return np.zeros(game.N)
    denom = game.b * (1.0 + game.alpha / game.N)
    S_star = ((game.a - game.c) / denom) ** (1.0 / game.alpha)
    return np.full(game.N, S_star / game.N)


def closed_form_social_opt(game):
    """
    Symmetric social optimum. Planner FOC c - P(S) - S P'(S) = 0 gives
        S^o = ( (a - c) / (b * (1 + alpha)) )^(1/alpha)
    """
    if game.a <= game.c:
        return np.zeros(game.N)
    denom = game.b * (1.0 + game.alpha)
    S_opt = ((game.a - game.c) / denom) ** (1.0 / game.alpha)
    return np.full(game.N, S_opt / game.N)


def total_loss(game, q):
    """J(q) = sum_i J^i(q) = sum_i (c * q_i - q_i * P(S))."""
    S = q.sum()
    return sum(game.c * q[i] - q[i] * game.P(S) for i in range(game.N))


def price_of_anarchy(game):
    """PoA = J(q*) / J(q^o)."""
    q_star = closed_form_NE(game)
    q_opt = closed_form_social_opt(game)
    J_star = total_loss(game, q_star)
    J_opt = total_loss(game, q_opt)
    if abs(J_opt) < 1e-9:
        return np.nan
    return J_star / J_opt