"""
Plotting helpers and empirical contraction-rate estimation.
"""

import numpy as np
import matplotlib.pyplot as plt


def empirical_rate(traj, q_star):
    """
    Linear regression of log ||q^t - q*|| vs t on the middle portion.
    Returns the contraction factor tau such that ||q^t - q*|| ~ tau^t.
    """
    err = np.linalg.norm(traj - q_star, axis=1)
    err = np.where(err > 1e-14, err, 1e-14)
    log_err = np.log(err)
    n = len(log_err)
    start = min(5, n // 4)
    end = max(start + 5, int(0.8 * n))
    end = min(end, n)
    if end - start < 3:
        return np.nan
    ts = np.arange(start, end)
    seg = log_err[start:end]
    if not np.all(np.isfinite(seg)) or len(ts) != len(seg):
        return np.nan
    slope, _ = np.polyfit(ts, seg, 1)
    return float(np.exp(slope))


def plot_convergence(ax, traj, q_star, label, T_show=None):
    """
    Plot ||q^t - q*|| on a log scale on the given axis.
    If T_show is given, plot only the first T_show iterations.
    """
    err = np.linalg.norm(traj - q_star, axis=1)
    err = np.maximum(err, 1e-15)
    if T_show is not None:
        err = err[:T_show + 1]
    ax.semilogy(err, label=label)


def style_convergence_axis(ax, title, xlabel="iteration t"):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()