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
    """
    Symmetric Nash equilibrium. Each player's FOC is
        c + beta * q - P(Nq) - q * P'(Nq) = 0.
    Affine demand has a clean closed form; power-law needs numerical FOC.
    """
    if game.a <= game.c:
        return np.zeros(game.N)
    
    if game.alpha == 1.0:
        # P(S) = a -bS, P'(S) = -b
        q = (game.a - game.c)/((game.N + 1) * game.b + game.beta)
        return np.full(game.N, q)
    
    # Power law case, slove FOC numerically
    def foc(q):
        S = game.N * q
        return game.dc_i(q) - game.P(S) - q*game.dP(S)
    
    # Search bracket: FOC is negative at q -> 0 and eventually positive
    q_lo, q_hi = 1e-8, game.q_bar * 10
    try:
        q_star = brentq(foc, q_lo, q_hi, xtol=1e-10)
    except ValueError:
        #Bracket failed - FOC  has no sign change, equilibrium lies above.
        #Return upper bound
        return np.full(game.N, q_hi)


def closed_form_social_opt(game):
    """
    Symmetric social optimum. Planner's FOC is
        c + beta * q - P(Nq) - Nq * P'(Nq) = 0.
    """
    if game.a <= game.c:
        return np.zeros(game.N)

    if game.alpha == 1.0:
        #Affine: c + beta q - a + bNq + bNq = 0
        q = (game.a - game.c)/(2* game.b * game.N + game.beta)
        return np.full(game.N, q)
    
    def planner_foc(q):
        S = game.N * q
        return game.dc_i(q) - game.P(S) - S*game.dP(S)
    
    q_lo, q_hi = 1e-8, game.q_bar * 10
    try:
        q_opt = brentq(planner_foc, q_lo, q_hi, xtol=1e-10)
    except ValueError:
        return np.full(game.N, q_hi)



def total_loss(game, q):
    """J(q) = sum_i [c_i(q_i) - q_i * P(S)]."""
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