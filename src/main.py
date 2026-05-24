from experiments import (E1_affine_baseline, E2_convergence_vs_N,
                         E3_alpha_sweep, E4_phase_diagram_and_poa,
                         E5_cost_curvature_rescue,
                         E6_cost_curvature_accelerates)

def main():
    E1_affine_baseline()
    E2_convergence_vs_N()
    E3_alpha_sweep()
    E4_phase_diagram_and_poa()
    E5_cost_curvature_rescue()
    E6_cost_curvature_accelerates()
    print("\nAll experiments complete. Plots saved as PNGs.")

if __name__ == "__main__":
    main()