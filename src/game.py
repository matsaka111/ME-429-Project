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
Q_BAR = 1e5    # per-player capacity
BETA = 0       #  marginal cost slope, default 0 = linear


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

    def __init__(self, N, alpha=1.0, a=A, b=B, c=C, q_bar=Q_BAR, beta=BETA):
        self.N = N
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
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
        return self.dc_i(q_i) - self.P(S) - q_i * self.dP(S)
    
    def c_i(self, q_i):
        """Production cost c_i(q_i) = c * q_i + 0.5 beta * q_i^2"""
        return self.c * q_i + 0.5 * self.beta*q_i**2
    
    def dc_i(self, q_i):
        """Marginal cost c_i'(q_i) = c + beta * q_i"""
        return self.c + self.beta * q_i
    
    def d2c_i(self, q_i):
        """Cost curvature c_i''(q_i) = beta"""
        return self.beta


# ---------------------------------------------------------------------------
# Closed-form Nash equilibrium and social optimum (symmetric, power-law P)
# ---------------------------------------------------------------------------

from scipy.optimize import brentq

def closed_form_NE(game):
    if game.a <= game.c:
        return np.zeros(game.N)

    if game.alpha == 1.0:
        q = (game.a - game.c) / ((game.N + 1) * game.b + game.beta)
        return np.full(game.N, q)

    def foc(q):
        S = game.N * q
        return game.dc_i(q) - game.P(S) - q * game.dP(S)

    # Extend bracket until FOC changes sign, or give up
    q_lo = 1e-8
    q_hi = max(game.q_bar * 10, 1e6)
    f_lo = foc(q_lo)
    for _ in range(20):
        f_hi = foc(q_hi)
        if f_lo * f_hi < 0:
            break
        q_hi *= 10
    else:
        # No sign change found — unconstrained NE is enormous,
        # equilibrium is on the boundary
        return np.full(game.N, q_hi)

    try:
        q_star = brentq(foc, q_lo, q_hi, xtol=1e-10)
    except Exception as e:
        print(f"  [closed_form_NE] alpha={game.alpha} N={game.N} "
              f"brentq failed unexpectedly: {type(e).__name__}: {e}")
        q_star = q_hi
    return np.full(game.N, q_star)


def closed_form_social_opt(game):
    if game.a <= game.c:
        return np.zeros(game.N)

    if game.alpha == 1.0:
        q = (game.a - game.c) / (2 * game.b * game.N + game.beta)
        return np.full(game.N, q)

    def planner_foc(q):
        S = game.N * q
        return game.dc_i(q) - game.P(S) - S * game.dP(S)

    q_lo = 1e-8
    q_hi = max(game.q_bar * 10, 1e6)
    f_lo = planner_foc(q_lo)
    for _ in range(20):
        f_hi = planner_foc(q_hi)
        if f_lo * f_hi < 0:
            break
        q_hi *= 10
    else:
        return np.full(game.N, q_hi)

    try:
        q_opt = brentq(planner_foc, q_lo, q_hi, xtol=1e-10)
    except Exception as e:
        print(f"  [closed_form_social_opt] alpha={game.alpha} N={game.N} "
              f"brentq failed unexpectedly: {type(e).__name__}: {e}")
        q_opt = q_hi
    return np.full(game.N, q_opt)


def total_loss(game, q):
    """J(q) = sum_i [c_i(q_i) - q_i * P(S)]."""
    S = q.sum()
    return sum(game.c_i(q[i]) - q[i] * game.P(S) for i in range(game.N))


def price_of_anarchy(game):
    """PoA = J(q*) / J(q^o)."""
    q_star = closed_form_NE(game)
    q_opt = closed_form_social_opt(game)
    J_star = total_loss(game, q_star)
    J_opt = total_loss(game, q_opt)
    if abs(J_opt) < 1e-9:
        return np.nan
    return J_star / J_opt