from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, PARAM_RANGES, FIG_DIR, DATA_DIR, DYN_PER_CM2_TO_MMHG
from io_utils import finalize_figure
from core_impedance import make_tree_wall_from_values, compute_pressure_solution, structured_tree_spectrum, reflection_coefficient_spectrum, build_time_array, synthetic_inflow_waveform
XI_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
PARAM_ORDER = ['r_min_cm', 'length_to_radius', 'asymmetry_g', 'k1']
PARAM_LABELS = {'r_min_cm': '$r_{\\min}$', 'length_to_radius': '$\\ell_{rr}$', 'asymmetry_g': '$g$', 'k1': '$k_1$'}

def theta_from_xi(param_name: str, xi: float) -> float:
    lo, hi = PARAM_RANGES[param_name]
    return lo + xi * (hi - lo)

def repeat_cycles(t_one, y_one, n_cycles=4):
    T = t_one[-1] + (t_one[1] - t_one[0])
    t_list = []
    y_list = []
    for k in range(n_cycles):
        t_list.append(t_one + k * T)
        y_list.append(y_one)
    return (np.concatenate(t_list), np.concatenate(y_list))

def compute_solution_for_param(param_name, xi, blood, base_tree, base_wall, config, t, q):
    value = theta_from_xi(param_name, xi)
    tree_i, wall_i = make_tree_wall_from_values({param_name: value}, base_tree, base_wall)
    sol = compute_pressure_solution(tree_i, wall_i, blood, config, t=t, q=q)
    return (value, tree_i, wall_i, sol)

def plot_pressure_waveform_sweeps():
    blood, base_wall, base_tree, config = default_objects()
    t = build_time_array(config)
    q = synthetic_inflow_waveform(t, config.cycle_length_s)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axs = axs.ravel()
    rows = []
    for ax, param in zip(axs, PARAM_ORDER):
        for xi in XI_VALUES:
            value, tree_i, wall_i, sol = compute_solution_for_param(param, xi, blood, base_tree, base_wall, config, t, q)
            t_rep, p_rep = repeat_cycles(t, sol['p'], n_cycles=4)
            ax.plot(t_rep, p_rep, linewidth=1.8, label=f'$\\xi={xi:.2f}$')
            rows.append({'parameter': param, 'xi': xi, 'parameter_value': value, 'peak_pressure_mmhg': sol['peak_pressure_mmhg'], 'mean_pressure_mmhg': sol['mean_pressure_mmhg'], 'pulse_pressure_mmhg': sol['pulse_pressure_mmhg']})
        ax.set_title(f'Pressure response varying {PARAM_LABELS[param]}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Pressure [mmHg]')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.suptitle('PCE-style parameter sweeps: proximal pressure waveforms', fontsize=15)
    finalize_figure(FIG_DIR / '34_pce_style_pressure_waveform_sweeps.png')
    pd.DataFrame(rows).to_csv(DATA_DIR / 'pce_style_parameter_sweep_pressure.csv', index=False)

def plot_impedance_sweeps():
    blood, base_wall, base_tree, config = default_objects()
    freqs = np.linspace(0.0, config.f_max_hz, config.n_freqs)
    mask = freqs > 0
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axs = axs.ravel()
    rows = []
    for ax, param in zip(axs, PARAM_ORDER):
        for xi in XI_VALUES:
            value = theta_from_xi(param, xi)
            tree_i, wall_i = make_tree_wall_from_values({param: value}, base_tree, base_wall)
            z = structured_tree_spectrum(freqs, tree_i, blood, wall_i)
            ax.plot(freqs[mask], np.abs(z[mask]), linewidth=1.8, label=f'$\\xi={xi:.2f}$')
            for h in [1.0, 2.0, 3.0]:
                idx = int(np.argmin(np.abs(freqs - h)))
                rows.append({'parameter': param, 'xi': xi, 'parameter_value': value, 'frequency_hz': float(freqs[idx]), 'Z_mag': float(np.abs(z[idx])), 'Z_phase_deg': float(np.angle(z[idx], deg=True))})
        ax.set_title(f'Impedance varying {PARAM_LABELS[param]}')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('$|Z_{\\mathrm{tree}}(\\omega)|$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.suptitle('PCE-style parameter sweeps: structured-tree impedance', fontsize=15)
    finalize_figure(FIG_DIR / '35_pce_style_impedance_sweeps.png')
    pd.DataFrame(rows).to_csv(DATA_DIR / 'pce_style_parameter_sweep_impedance.csv', index=False)

def plot_reflection_sweeps():
    blood, base_wall, base_tree, config = default_objects()
    freqs = np.linspace(0.0, config.f_max_hz, config.n_freqs)
    mask = freqs > 0
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axs = axs.ravel()
    rows = []
    for ax, param in zip(axs, PARAM_ORDER):
        for xi in XI_VALUES:
            value = theta_from_xi(param, xi)
            tree_i, wall_i = make_tree_wall_from_values({param: value}, base_tree, base_wall)
            z = structured_tree_spectrum(freqs, tree_i, blood, wall_i)
            gamma = reflection_coefficient_spectrum(freqs, z, tree_i, blood, wall_i)
            ax.plot(freqs[mask], np.abs(gamma[mask]), linewidth=1.8, label=f'$\\xi={xi:.2f}$')
            rows.append({'parameter': param, 'xi': xi, 'parameter_value': value, 'mean_abs_gamma': float(np.mean(np.abs(gamma[mask]))), 'max_abs_gamma': float(np.max(np.abs(gamma[mask])))})
        ax.set_title(f'Reflection varying {PARAM_LABELS[param]}')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('$|\\Gamma(\\omega)|$')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    plt.suptitle('PCE-style parameter sweeps: reflection coefficient', fontsize=15)
    finalize_figure(FIG_DIR / '36_pce_style_reflection_sweeps.png')
    pd.DataFrame(rows).to_csv(DATA_DIR / 'pce_style_parameter_sweep_reflection.csv', index=False)

def plot_peak_pressure_vs_xi():
    blood, base_wall, base_tree, config = default_objects()
    t = build_time_array(config)
    q = synthetic_inflow_waveform(t, config.cycle_length_s)
    plt.figure(figsize=(8, 5))
    rows = []
    for param in PARAM_ORDER:
        peaks = []
        for xi in XI_VALUES:
            value, tree_i, wall_i, sol = compute_solution_for_param(param, xi, blood, base_tree, base_wall, config, t, q)
            peaks.append(sol['peak_pressure_mmhg'])
            rows.append({'parameter': param, 'xi': xi, 'parameter_value': value, 'peak_pressure_mmhg': sol['peak_pressure_mmhg']})
        plt.plot(XI_VALUES, peaks, marker='o', linewidth=2, label=PARAM_LABELS[param])
    plt.xlabel('Normalized uncertainty coordinate $\\xi$')
    plt.ylabel('Peak pressure [mmHg]')
    plt.title('Peak pressure response along one-parameter $\\xi$ sweeps')
    plt.legend()
    plt.grid(alpha=0.25)
    finalize_figure(FIG_DIR / '37_peak_pressure_vs_xi_sweeps.png')
    pd.DataFrame(rows).to_csv(DATA_DIR / 'pce_style_peak_pressure_vs_xi.csv', index=False)

def main():
    print('Creating PCE-style multi-panel parameter sweep plots...')
    plot_pressure_waveform_sweeps()
    plot_impedance_sweeps()
    plot_reflection_sweeps()
    plot_peak_pressure_vs_xi()
    print('Done.')
    print('Created:')
    print('  figures/34_pce_style_pressure_waveform_sweeps.png')
    print('  figures/35_pce_style_impedance_sweeps.png')
    print('  figures/36_pce_style_reflection_sweeps.png')
    print('  figures/37_peak_pressure_vs_xi_sweeps.png')
if __name__ == '__main__':
    main()
