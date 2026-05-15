# Cournot Electricity Market Simulations

This is the code that accompanies the ME-429 project on learning equilibria in electricity markets. It runs the projected pseudo-gradient algorithm on a symmetric Cournot game with power-law inverse demand, and reproduces the figures in the report.

## How to run

Requirements: Python 3.9+, numpy, matplotlib.

Install dependencies if needed:

    pip install numpy matplotlib

From the project folder, run:

    python main.py

This runs all five experiments (E1 through E5) and saves four PNG plots in the current folder:

- E2_convergence_vs_N.png
- E3_nonlinear_demand.png
- E4_phase_diagram.png
- E5_price_of_anarchy.png

Console output lists, for each experiment, the parameters tested, the empirical contraction rate, the final iterate, and the error against the closed-form Nash equilibrium.

## File structure

- main.py: entry point. Imports and runs the experiments.
- game.py: the CournotGame class and closed-form Nash equilibrium / social optimum.
- algorithm.py: the projected pseudo-gradient iteration and step-size selection.
- experiments.py: the five experiments E1 through E5.
- utils.py: plotting and empirical-rate helpers.

## What the experiments do

- E1: Verify the algorithm reaches the closed-form NE for the affine case.
- E2: Convergence rate vs population size N, affine demand.
- E3: Convergence under power-law demand P(S) = a - b * S^alpha, for several alpha and N.
- E4: Heatmap of the empirical contraction rate over a grid of (N, alpha).
- E5: Price of anarchy as a function of N, for several alpha.

## Parameters

All parameters are defined at the top of game.py:

- A = 150: choke price (EUR/MWh)
- B = 0.03: demand slope
- C = 50: marginal cost (EUR/MWh)
- Q_BAR = 1e5: per-player capacity
