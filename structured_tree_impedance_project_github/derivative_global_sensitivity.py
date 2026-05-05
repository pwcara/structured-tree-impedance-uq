from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, PARAM_RANGES, FIG_DIR, DATA_DIR
from io_utils import finalize_figure
from core_impedance import make_tree_wall_from_values, compute_pressure_solution, build_time_array, synthetic_inflow_waveform
from sampling_sensitivity import latin_hypercube, scale_lhs_to_ranges

def clamp(value, lo, hi):
    return min(max(value, lo), hi)

def main():
    blood, base_wall, base_tree, config = default_objects()
    rng = np.random.default_rng(config.random_seed + 999)
    names = list(PARAM_RANGES.keys())
    n_samples = 60
    rel_step = 0.02
    X_unit = latin_hypercube(n_samples, len(names), rng)
    X_phys, names = scale_lhs_to_ranges(X_unit, PARAM_RANGES)
    t = build_time_array(config)
    q = synthetic_inflow_waveform(t, config.cycle_length_s)
    rows = []
    deriv_records = []
    for i in range(n_samples):
        base_values = dict(zip(names, X_phys[i, :]))
        tree0, wall0 = make_tree_wall_from_values(base_values, base_tree, base_wall)
        sol0 = compute_pressure_solution(tree0, wall0, blood, config, t=t, q=q)
        p0 = sol0['peak_pressure_mmhg']
        for name in names:
            lo, hi = PARAM_RANGES[name]
            theta0 = base_values[name]
            h = rel_step * theta0
            theta_minus = clamp(theta0 - h, lo, hi)
            theta_plus = clamp(theta0 + h, lo, hi)
            if theta_plus == theta_minus:
                continue
            vals_minus = dict(base_values)
            vals_plus = dict(base_values)
            vals_minus[name] = theta_minus
            vals_plus[name] = theta_plus
            tree_m, wall_m = make_tree_wall_from_values(vals_minus, base_tree, base_wall)
            tree_p, wall_p = make_tree_wall_from_values(vals_plus, base_tree, base_wall)
            pm = compute_pressure_solution(tree_m, wall_m, blood, config, t=t, q=q)['peak_pressure_mmhg']
            pp = compute_pressure_solution(tree_p, wall_p, blood, config, t=t, q=q)['peak_pressure_mmhg']
            dP = (pp - pm) / (theta_plus - theta_minus)
            dlog = (np.log(pp) - np.log(pm)) / (np.log(theta_plus) - np.log(theta_minus))
            deriv_records.append({'sample_id': i, 'parameter': name, 'theta0': theta0, 'p0': p0, 'dP_dtheta': dP, 'local_log_sensitivity': dlog, 'dP_dtheta_sq': dP ** 2, 'local_log_sensitivity_sq': dlog ** 2})
    deriv_df = pd.DataFrame(deriv_records)
    deriv_df.to_csv(DATA_DIR / 'derivative_samples.csv', index=False)
    summary_rows = []
    for name, group in deriv_df.groupby('parameter'):
        summary_rows.append({'parameter': name, 'mean_dP_dtheta': group['dP_dtheta'].mean(), 'mean_abs_dP_dtheta': group['dP_dtheta'].abs().mean(), 'DGSM_E_dP_dtheta_sq': group['dP_dtheta_sq'].mean(), 'mean_log_sensitivity': group['local_log_sensitivity'].mean(), 'mean_abs_log_sensitivity': group['local_log_sensitivity'].abs().mean(), 'DGSM_E_log_sens_sq': group['local_log_sensitivity_sq'].mean()})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_DIR / 'derivative_global_sensitivity.csv', index=False)
    plot_df = summary.sort_values('mean_abs_log_sensitivity', ascending=True)
    plt.figure(figsize=(7, 4.5))
    plt.barh(plot_df['parameter'], plot_df['mean_abs_log_sensitivity'])
    plt.xlabel('Mean absolute local log sensitivity')
    plt.title('Derivative-based global sensitivity: log scale')
    plt.grid(axis='x', alpha=0.25)
    finalize_figure(FIG_DIR / '20_derivative_log_sensitivity.png')
    plot_df = summary.sort_values('DGSM_E_log_sens_sq', ascending=True)
    plt.figure(figsize=(7, 4.5))
    plt.barh(plot_df['parameter'], plot_df['DGSM_E_log_sens_sq'])
    plt.xlabel('E[(∂log Pmax / ∂log θ)²]')
    plt.title('DGSM-style derivative sensitivity')
    plt.grid(axis='x', alpha=0.25)
    finalize_figure(FIG_DIR / '21_derivative_dgsm.png')
    print('Derivative-based sensitivity done.')
    print(summary)
if __name__ == '__main__':
    main()
