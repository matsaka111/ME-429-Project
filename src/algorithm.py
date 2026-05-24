"""
Projected pseudo-gradient algorithm and step-size selection.
"""

import numpy as np
from game import closed_form_NE


# ---------------------------------------------------------------------------
# Projected pseudo-gradient
# ---------------------------------------------------------------------------

def projected_pseudo_gradient(game, q0, gamma, T_max=5000, tol=1e-9):
    """
    Algorithm 1 from the report:
        q^{t+1}_i = Pi_[0, q_bar]( q^t_i - gamma * grad_i(q^t_i, S^t) )

    Returns the trajectory as an array of shape (T+1, N).
    """
    q = np.array(q0, dtype=float).copy()
    traj = [q.copy()]
    for t in range(T_max):
        S = q.sum()
        g = np.array([game.grad_i(q[i], S) for i in range(game.N)])
        q_new = np.clip(q - gamma * g, 0.0, game.q_bar)
        traj.append(q_new.copy())
        if np.linalg.norm(q_new - q) < tol:
            q = q_new
            break
        q = q_new
    return np.array(traj)


# ---------------------------------------------------------------------------
# Step-size selection via Jacobian probing
# ---------------------------------------------------------------------------

def jacobian_F(game, q):
    """
    Jacobian of the game map F_Gamma at q.
        J_ij = c''_i - 2 P'(S) - q_i P''(S)  if i == j
             =        - P'(S) - q_i P''(S)  if i != j
    With linear costs, c''_i = 0.
    """
    N = game.N
    S = q.sum()
    Pp = game.dP(S)
    Ppp = game.d2P(S)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                J[i, j] = game.d2c_i(q[i]) -2 * Pp - q[i] * Ppp
            else:
                J[i, j] = -Pp - q[i] * Ppp
    return J


def estimate_mu_L(game, probe_fracs=(0.5, 0.8, 1.0, 1.2, 1.5)):
    """
    Estimate strong-monotonicity mu and Lipschitz L by probing the
    Jacobian at symmetric points q = frac * q*. Probing near the NE
    avoids the S -> 0 singularity for alpha < 1.
    """
    q_star = closed_form_NE(game)
    if np.all(q_star == 0):
        return -1.0, np.inf
    mus, Ls = [], []
    for frac in probe_fracs:
        q = np.clip(frac * q_star, 1e-6, game.q_bar)
        J = jacobian_F(game, q)
        sym = 0.5 * (J + J.T)
        eig_sym = np.linalg.eigvalsh(sym)
        mus.append(eig_sym.min())
        Ls.append(np.linalg.norm(J, 2))
    return min(mus), max(Ls)


def choose_step_size(game, safety=0.9):
    """
    Pick gamma inside the contraction range (0, 2 mu / L^2).
    Returns (gamma, mu, L) or (None, mu, L) if the bound is degenerate.
    """
    mu, L = estimate_mu_L(game)
    if mu <= 0 or L <= 0 or not np.isfinite(L):
        return None, mu, L
    return safety * 2 * mu / (L ** 2), mu, L