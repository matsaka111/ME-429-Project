"""
Six experiments E1 through E6.

Each experiment tests one specific theoretical claim from Section 2 of
the report. The principle is: every experiment is the simulation
counterpart of a named statement in the theory.

    E1: Sanity check vs closed-form NE (affine, linear cost).
    E2: kappa(N) = O(N) under affine P.
    E3: Effect of demand curvature alpha on convergence at fixed N.
    E4: Phase diagram of (N, alpha): contraction rate and PoA.
    E5: Cost curvature beta rescues convex-demand equilibrium.
    E6: Cost curvature beta accelerates convergence (affine).
"""

import numpy as np
import matplotlib.pyplot as plt

from game import (CournotGame, closed_form_NE, closed_form_social_opt,
                  total_loss, price_of_anarchy)
from algorithm import projected_pseudo_gradient, choose_step_size
from utils import empirical_rate, plot_convergence, style_convergence_axis


# Each panel uses an iteration window matched to the timescale of
# convergence for that alpha. Without this, slow-alpha cases look flat
# and fast-alpha cases look like a vertical line at t = 0.
PLOT_WINDOWS = {
    0.5: 30, 0.7: 80, 1.0: 200, 1.3: 400,
    1.5: 800, 1.8: 2500, 2.0: 4500,
}


# ---------------------------------------------------------------------------
# E1: Affine baseline
# ---------------------------------------------------------------------------

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
        poa = price_of_anarchy(game)
        print(f"  N={N:3d}  gamma={gamma:.4f}  q*={q_star[0]:.3f}  "
              f"q_T={q_T[0]:.3f}  err={err:.2e}  "
              f"iters={len(traj)-1}  PoA={poa:.3f}")


# ---------------------------------------------------------------------------
# E2: Convergence rate vs N, affine P
# ---------------------------------------------------------------------------

def E2_convergence_vs_N():
    """Convergence trajectories for several N. Affine P, linear cost."""
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


# ---------------------------------------------------------------------------
# E3: Effect of alpha at fixed N
# ---------------------------------------------------------------------------

def E3_alpha_sweep():
    """
    Isolate the effect of demand curvature alpha on convergence,
    holding N fixed.
    """
    print("=" * 60)
    print("E3: Effect of demand curvature alpha at fixed N = 10")
    print("=" * 60)
    N = 10
    alphas = [0.7, 1.0, 1.3, 1.5, 1.8, 2.0]
    fig, ax = plt.subplots(figsize=(7, 5))
    for alpha in alphas:
        game = CournotGame(N, alpha=alpha)
        gamma, mu, L = choose_step_size(game)
        if gamma is None:
            print(f"  alpha={alpha}  step-size FAILED")
            continue
        q_star = closed_form_NE(game)
        if np.any(q_star > game.q_bar):
            print(f"  alpha={alpha}  q* exceeds capacity, skipping")
            continue
        q0 = np.full(N, q_star[0] * 0.3)
        traj = projected_pseudo_gradient(game, q0, gamma, T_max=5000)
        tau_emp = empirical_rate(traj, q_star)
        print(f"  alpha={alpha}  gamma={gamma:.3e}  tau={tau_emp:.4f}  "
              f"iters={len(traj)-1}")
        plot_convergence(ax, traj, q_star, label=f"alpha={alpha}",
                         T_show=PLOT_WINDOWS.get(alpha, 2000))
    ax.set_ylabel("||q^t - q*||")
    style_convergence_axis(ax, f"E3: Convergence vs demand curvature, N = {N}")
    plt.tight_layout()
    plt.savefig("E3_alpha_sweep.png", dpi=140)
    plt.close()


# ---------------------------------------------------------------------------
# E4: Phase diagram in (N, alpha) + PoA
# ---------------------------------------------------------------------------

def E4_phase_diagram_and_poa():
    """
    Joint (N, alpha) behaviour: empirical contraction rate (left panel)
    and price of anarchy (right panel).
    """
    print("=" * 60)
    print("E4: Phase diagram in (N, alpha) and PoA")
    print("=" * 60)
    Ns = [2, 5, 10, 20, 50, 100, 200]
    alphas_rate = np.linspace(0.4, 2.2, 10)
    alphas_poa = [1.0, 1.25, 1.5, 1.75, 2.0]

    # --- Contraction rates ---
    rates = np.full((len(alphas_rate), len(Ns)), np.nan)
    for ia, alpha in enumerate(alphas_rate):
        for iN, N in enumerate(Ns):
            game = CournotGame(N, alpha=alpha)
            gamma, mu, L = choose_step_size(game)
            if gamma is None:
                continue
            q_star = closed_form_NE(game)
            if np.any(q_star > game.q_bar):
                continue
            q0 = np.full(N, q_star[0] * 0.3)
            traj = projected_pseudo_gradient(game, q0, gamma, T_max=3000)
            tau = empirical_rate(traj, q_star)
            if np.isfinite(tau) and tau < 1.0:
                rates[ia, iN] = tau

    # --- PoA values ---
    poa_table = {}
    for alpha in alphas_poa:
        poa_vals = []
        for N in Ns:
            game = CournotGame(N, alpha=alpha)
            poa_vals.append(price_of_anarchy(game))
        poa_table[alpha] = poa_vals
        print(f"  alpha={alpha}: " +
              " ".join(f"N={n}:PoA={p:.3f}" for n, p in zip(Ns, poa_vals)))

    # --- Plot side-by-side ---
    fig, (ax_rate, ax_poa) = plt.subplots(1, 2, figsize=(13, 5))

    im = ax_rate.imshow(rates, aspect="auto", origin="lower",
                        extent=[0, len(Ns), alphas_rate[0], alphas_rate[-1]],
                        cmap="viridis", vmin=0, vmax=1)
    ax_rate.set_xticks(np.arange(len(Ns)) + 0.5)
    ax_rate.set_xticklabels([str(N) for N in Ns])
    ax_rate.set_xlabel("N")
    ax_rate.set_ylabel("alpha")
    ax_rate.set_title("Empirical contraction rate (lower = faster)")
    plt.colorbar(im, ax=ax_rate, label="empirical tau")

    for alpha, poa_vals in poa_table.items():
        ax_poa.plot(Ns, poa_vals, marker="o", label=f"alpha={alpha}")
    ax_poa.set_xscale("log")
    ax_poa.set_xlabel("N")
    ax_poa.set_ylabel("Price of Anarchy = J(q*) / J(q^o)")
    ax_poa.set_title("PoA vs N, by demand curvature")
    ax_poa.grid(True, which="both", alpha=0.3)
    ax_poa.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_poa.legend()

    plt.tight_layout()
    plt.savefig("E4_phase_diagram_and_poa.png", dpi=140)
    plt.close()


# ---------------------------------------------------------------------------
# E5: Cost curvature rescues convex-demand equilibrium
# ---------------------------------------------------------------------------

def E5_cost_curvature_rescue():
    """
    Convex demand (alpha = 0.5) has no interior NE under linear costs.
    Show that quadratic cost (beta > 0) restores a finite interior NE
    by providing the internal market-power restraint that flat
    marginal cost lacks.
    """
    print("=" * 60)
    print("E5: Cost curvature rescue, convex demand alpha = 0.5")
    print("=" * 60)
    N = 10
    alpha = 0.5
    betas = [0.0, 1e-3, 1e-2, 1e-1, 1.0]

    fig, (ax_conv, ax_q) = plt.subplots(1, 2, figsize=(13, 5))
    q_finals = []
    for beta in betas:
        game = CournotGame(N, alpha=alpha, beta=beta)
        gamma, mu, L = choose_step_size(game)
        q_star = closed_form_NE(game)
        if gamma is None:
            print(f"  beta={beta:.1e}  step-size FAILED")
            q_finals.append(np.nan)
            continue
        q0 = np.full(N, min(q_star[0] * 0.3, game.q_bar * 0.3))
        traj = projected_pseudo_gradient(game, q0, gamma, T_max=5000)
        q_T = traj[-1]
        q_finals.append(q_T[0])
        on_boundary = q_T[0] >= 0.999 * game.q_bar
        print(f"  beta={beta:.1e}  q*_NE={q_star[0]:.3e}  q_T={q_T[0]:.3e}  "
              f"on_boundary={on_boundary}  iters={len(traj)-1}")
        plot_convergence(ax_conv, traj, q_star,
                         label=f"beta={beta:.0e}", T_show=200)
    ax_conv.set_ylabel("||q^t - q*||")
    style_convergence_axis(ax_conv,
                           f"Convergence, alpha={alpha}, varying beta")

    ax_q.plot(betas, q_finals, marker="o")
    ax_q.axhline(CournotGame(N, alpha=alpha).q_bar, color="red",
                 linestyle="--", label="capacity q_bar")
    ax_q.set_xscale("symlog", linthresh=1e-4)
    ax_q.set_yscale("log")
    ax_q.set_xlabel("beta (cost curvature)")
    ax_q.set_ylabel("final iterate q^T")
    ax_q.set_title("Final iterate vs cost curvature")
    ax_q.grid(True, which="both", alpha=0.3)
    ax_q.legend()

    plt.tight_layout()
    plt.savefig("E5_cost_curvature_rescue.png", dpi=140)
    plt.close()


# ---------------------------------------------------------------------------
# E6: Cost curvature accelerates convergence (affine demand)
# ---------------------------------------------------------------------------

def E6_cost_curvature_accelerates():
    """
    With affine P, the condition number is kappa = (b(N+1) + beta) / (b + beta),
    which decreases as beta grows. Show empirically that increasing
    cost curvature reduces the iteration count.
    """
    print("=" * 60)
    print("E6: Cost curvature accelerates convergence, affine P")
    print("=" * 60)
    N = 50
    betas = [0.0, 0.1, 0.5, 1.0, 5.0]

    fig, ax = plt.subplots(figsize=(7, 5))
    iters_list, kappa_list = [], []
    for beta in betas:
        game = CournotGame(N, alpha=1.0, beta=beta)
        gamma, mu, L = choose_step_size(game)
        q_star = closed_form_NE(game)
        q0 = np.full(N, q_star[0] * 0.3)
        traj = projected_pseudo_gradient(game, q0, gamma, T_max=5000)
        kappa = (game.b * (N + 1) + beta) / (game.b + beta)
        iters_list.append(len(traj) - 1)
        kappa_list.append(kappa)
        print(f"  beta={beta:.2f}  kappa={kappa:.2f}  iters={len(traj)-1}")
        plot_convergence(ax, traj, q_star, label=f"beta={beta}")
    ax.set_ylabel("||q^t - q*||")
    style_convergence_axis(ax,
                           f"E6: Convergence under varying cost curvature, N = {N}")

    plt.tight_layout()
    plt.savefig("E6_cost_curvature_accelerates.png", dpi=140)
    plt.close()