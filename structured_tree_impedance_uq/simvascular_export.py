import numpy as np
import pandas as pd
from config import default_objects, SV_DIR
from core_impedance import build_time_array, synthetic_inflow_waveform, structured_tree_spectrum, fit_rcr_to_tree, TreeParams
blood, wall, base_tree, config = default_objects()
t = build_time_array(config)
q = synthetic_inflow_waveform(t, config.cycle_length_s)
np.savetxt(SV_DIR / 'inlet_flow.txt', np.column_stack([t, q]), fmt='%.8e', header='time_s flow_cm3_per_s', comments='')

def compute_case(label, r_min_cm):
    tree = TreeParams(root_radius_cm=base_tree.root_radius_cm, r_min_cm=r_min_cm, length_to_radius=base_tree.length_to_radius, asymmetry_g=base_tree.asymmetry_g, radius_exponent_j=base_tree.radius_exponent_j, z_leaf=base_tree.z_leaf, include_viscosity=base_tree.include_viscosity, max_generations=base_tree.max_generations)
    freqs = np.linspace(0, config.f_max_hz, config.n_freqs)
    z_tree = structured_tree_spectrum(freqs, tree, blood, wall)
    R_total = float(np.real(z_tree[0]))
    Rp, C, Rd = fit_rcr_to_tree(freqs, z_tree, config.rcr_fit_fmax_hz)
    return {'case': label, 'r_min_cm': r_min_cm, 'R_eff_total': R_total, 'R_each_outlet': 2 * R_total, 'Rp_total': Rp, 'C_total': C, 'Rd_total': Rd, 'Rp_each_outlet': 2 * Rp, 'C_each_outlet': 0.5 * C, 'Rd_each_outlet': 2 * Rd}
rows = [compute_case('baseline', 0.02), compute_case('low_load_larger_rmin', 0.06), compute_case('high_load_smaller_rmin', 0.01)]
pd.DataFrame(rows).to_csv(SV_DIR / 'simvascular_bc_values.csv', index=False)
readme = 'SimVascular validation inputs\n\nUse inlet_flow.txt as the prescribed inlet flow waveform.\n\nSuggested 1D bifurcation cases:\n1. Resistance baseline:\n   Use R_each_outlet from the baseline row at both daughter outlets.\n\n2. RCR baseline:\n   Use Rp_each_outlet, C_each_outlet, Rd_each_outlet from the baseline row.\n\n3. Low-load case:\n   Use values from low_load_larger_rmin.\n\n4. High-load case:\n   Use values from high_load_smaller_rmin.\n\nIf the bifurcation has two identical daughter outlets, these per-outlet values are scaled\nso that the two branches in parallel approximate the total outlet load from the Python model.\n'
(SV_DIR / 'README_simvascular_inputs.txt').write_text(readme)
print('SimVascular input files exported to:', SV_DIR)
