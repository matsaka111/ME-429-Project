"""
Five experiments E1 through E5.

Each experiment uses the projected pseudo-gradient algorithm and
isolates the effect of one parameter:
    E1: sanity check vs closed-form NE (affine).
    E2: convergence rate vs N (affine).
    E3: convergence under power-law demand for several (alpha, N).
    E4: phase diagram of contraction rate in (N, alpha).
    E5: price of anarchy vs N, for several alpha.
"""

import numpy as np
import matplotlib.pyplot as plt

from game import (CournotGame, closed_form_NE, closed_form_social_opt,
                  total_loss, price_of_anarchy)
from algorithm import projected_pseudo_gradient, choose_step_size
from utils import empirical_rate, plot_convergence, style_convergence_axis


# Each E3 / E4 panel uses an iteration window matched to the timescale of
# convergence for that alpha. Without this, the slow-alpha cases look flat
# and the fast-alpha cases look like a vertical line at t = 0.
PLOT_WINDOWS = {0.5: 30, 1.0: 200, 1.5: 800, 2.0: 4500}


def E1_affine_baseline():
    """Verify the algorithm reaches the closed-form NE for affine P."""
    print("=" * 60)
    print("E1: Affine baseline, compare to closed-form NE")
    print("=" * 60)
    for N in [2, 5, 10]:
        game = CournotGame(N, alpha=1.0)
        gamma, mu, L = choose_step_size(game)
        q_star = closed_form_NE(game)
        q0 = np.full(N, q_star[0] * 0.3)
        traj = projected_pseudo_gradient(game, q0, gamma)
        q_T = traj[-1]
        err = np.max(np.abs(q_T - q_star))
        S_star = q_star.sum()
        poa = price_of_anarchy(game)
        print(f"  N={N:3d}  gamma={gamma:.4f}  q*={q_star[0]:.3f}  "
              f"q_T={q_T[0]:.3f}  err={err:.2e}  iters={len(traj)-1}")
        print(f"          S*={S_star:.1f}  P(S*)={game.P(S_star):.2f}  "
              f"PoA={poa:.3f}")


def E2_convergence_vs_N():
    """Convergence trajectories for several N. Affine P."""
    print("=" * 60)
    print("E2: Convergence rate vs N (affine P)")
    print("=" * 60)
    Ns = [2, 5, 10, 50, 100]
    fig, ax = plt.subplots(figsize=(7, 5))
    for N in Ns:
        game = CournotGame(N, alpha=1.0)
        gamma, mu, L = choose_step_size(game)
        q_star = closed_form_NE(game)
        q0 = np.full(N, q_star[0] * 0.3)
        traj = projected_pseudo_gradient(game, q0, gamma, T_max=2000)
        tau_emp = empirical_rate(traj, q_star)
        tau_th = np.sqrt(max(0.0, 1 - mu * mu / (L * L)))
        print(f"  N={N:3d}  tau_theory={tau_th:.4f}  "
              f"tau_empirical={tau_emp:.4f}  iters={len(traj)-1}")
        plot_convergence(ax, traj, q_star, label=f"N={N}")
    ax.set_ylabel("||q^t - q*||")
    style_convergence_axis(ax, "E2: Convergence vs population size, affine P")
    plt.tight_layout()
    plt.savefig("E2_convergence_vs_N.png", dpi=140)
    plt.close()


def E3_nonlinear_demand():
    """Convergence under power-law P, several (alpha, N)."""
    print("=" * 60)
    print("E3: Non-linear demand, power-law P")
    print("=" * 60)
    alphas = [0.5, 1.0, 1.5, 2.0]
    Ns = [2, 10, 100]

    fig, axes = plt.subplots(1, len(alphas), figsize=(16, 4))
    for ax, alpha in zip(axes, alphas):
        T_show = PLOT_WINDOWS[alpha]
        for N in Ns:
            game = CournotGame(N, alpha=alpha)
            gamma, mu, L = choose_step_size(game)
            if gamma is None:
                print(f"  alpha={alpha}  N={N:3d}  step-size FAILED")
                continue
            q_star = closed_form_NE(game)
            q0 = np.full(N, q_star[0] * 0.3)
            traj = projected_pseudo_gradient(game, q0, gamma, T_max=5000)
            q_T = traj[-1]
            err_final = np.max(np.abs(q_T - q_star))
            print(f"  alpha={alpha}  N={N:3d}  gamma={gamma:.3e}  "
                  f"iters={len(traj)-1}  q*={q_star[0]:.3f}  "
                  f"q_T={q_T[0]:.3f}  err={err_final:.2e}")
            plot_convergence(ax, traj, q_star, label=f"N={N}", T_show=T_show)
        style_convergence_axis(
            ax, f"alpha = {alpha}  (first {T_show} iters)")
    axes[0].set_ylabel("||q^t - q*||")
    plt.suptitle("E3: Convergence under power-law demand")
    plt.tight_layout()
    plt.savefig("E3_nonlinear_demand.png", dpi=140)
    plt.close()


def E4_phase_diagram():
    """Heatmap of empirical contraction rate over (N, alpha)."""
    print("=" * 60)
    print("E4: Phase diagram in (N, alpha)")
    print("=" * 60)
    Ns = [2, 5, 10, 20, 50, 100, 200]
    alphas = np.linspace(0.4, 2.2, 10)
    rates = np.full((len(alphas), len(Ns)), np.nan)
    for ia, alpha in enumerate(alphas):
        for iN, N in enumerate(Ns):
            game = CournotGame(N, alpha=alpha)
            gamma, mu, L = choose_step_size(game)
            if gamma is None:
                continue
            q_star = closed_form_NE(game)
            if np.any(q_star > game.q_bar):
                continue  # unconstrained NE exceeds capacity
            q0 = np.full(N, q_star[0] * 0.3)
            traj = projected_pseudo_gradient(game, q0, gamma, T_max=3000)
            tau = empirical_rate(traj, q_star)
            if np.isfinite(tau) and tau < 1.0:
                rates[ia, iN] = tau

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rates, aspect="auto", origin="lower",
                   extent=[0, len(Ns), alphas[0], alphas[-1]],
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(Ns)) + 0.5)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel("N")
    ax.set_ylabel("alpha")
    ax.set_title("E4: Empirical contraction rate (lower = faster)")
    plt.colorbar(im, ax=ax, label="empirical tau")
    plt.tight_layout()
    plt.savefig("E4_phase_diagram.png", dpi=140)
    plt.close()


def E5_price_of_anarchy():
    """PoA from closed-form NE and social optimum."""
    print("=" * 60)
    print("E5: Price of anarchy (closed form)")
    print("=" * 60)
    alphas = [0.5, 1.0, 1.5, 2.0]
    Ns = [2, 5, 10, 20, 50, 100, 200]
    fig, ax = plt.subplots(figsize=(7, 5))
    for alpha in alphas:
        poa_vals = []
        for N in Ns:
            game = CournotGame(N, alpha=alpha)
            poa_vals.append(price_of_anarchy(game))
        print(f"  alpha={alpha}: " +
              " ".join(f"N={n}:PoA={p:.3f}" for n, p in zip(Ns, poa_vals)))
        ax.plot(Ns, poa_vals, marker="o", label=f"alpha={alpha}")
    ax.set_xlabel("N")
    ax.set_ylabel("Price of Anarchy = J(q*) / J(q^o)")
    ax.set_title("E5: PoA vs population size, by demand curvature")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    plt.tight_layout()
    plt.savefig("E5_price_of_anarchy.png", dpi=140)
    plt.close()