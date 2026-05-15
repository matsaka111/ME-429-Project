"""
Entry point. Runs all five experiments and saves the plots.
"""

from experiments import (E1_affine_baseline, E2_convergence_vs_N,
                         E3_nonlinear_demand, E4_phase_diagram,
                         E5_price_of_anarchy)


def main():
    E1_affine_baseline()
    E2_convergence_vs_N()
    E3_nonlinear_demand()
    E4_phase_diagram()
    E5_price_of_anarchy()
    print("\nAll experiments complete. Plots saved as PNGs.")


if __name__ == "__main__":
    main()