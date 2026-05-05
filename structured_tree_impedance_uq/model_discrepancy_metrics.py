from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, FIG_DIR, DATA_DIR, DYN_PER_CM2_TO_MMHG
from io_utils import finalize_figure
from core_impedance import structured_tree_spectrum, resistance_spectrum, rcr_spectrum, fit_rcr_to_tree, build_time_array, synthetic_inflow_waveform, pressure_from_impedance

def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-14))

def main():
    blood, wall, tree, config = default_objects()
    freqs = np.linspace(0, config.f_max_hz, config.n_freqs)
    mask = freqs > 0
    z_tree = structured_tree_spectrum(freqs, tree, blood, wall)
    R_dc = float(np.real(z_tree[0]))
    z_R = resistance_spectrum(freqs, R_dc)
    Rp, C, Rd = fit_rcr_to_tree(freqs, z_tree, config.rcr_fit_fmax_hz)
    z_RCR = rcr_spectrum(freqs, Rp, C, Rd)
    t = build_time_array(config)
    q = synthetic_inflow_waveform(t, config.cycle_length_s)
    fft_freqs = np.fft.rfftfreq(len(t), d=config.dt)
    z_tree_td = structured_tree_spectrum(fft_freqs, tree, blood, wall)
    z_R_td = resistance_spectrum(fft_freqs, R_dc)
    z_RCR_td = rcr_spectrum(fft_freqs, Rp, C, Rd)
    p_tree = pressure_from_impedance(t, q, z_tree_td) * DYN_PER_CM2_TO_MMHG
    p_R = pressure_from_impedance(t, q, z_R_td) * DYN_PER_CM2_TO_MMHG
    p_RCR = pressure_from_impedance(t, q, z_RCR_td) * DYN_PER_CM2_TO_MMHG
    rows = []
    for name, z, p in [('R', z_R, p_R), ('RCR', z_RCR, p_RCR)]:
        rows.append({'model': name, 'rel_L2_error_Z_magnitude': rel_l2(np.abs(z[mask]), np.abs(z_tree[mask])), 'rel_L2_error_phase_deg': rel_l2(np.angle(z[mask], deg=True), np.angle(z_tree[mask], deg=True)), 'rel_L2_error_pressure_waveform': rel_l2(p, p_tree), 'peak_pressure_error_mmhg': float(np.max(p) - np.max(p_tree)), 'pulse_pressure_error_mmhg': float(np.max(p) - np.min(p) - (np.max(p_tree) - np.min(p_tree))), 'mean_pressure_error_mmhg': float(np.mean(p) - np.mean(p_tree))})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / 'model_discrepancy_metrics.csv', index=False)
    metrics = ['rel_L2_error_Z_magnitude', 'rel_L2_error_phase_deg', 'rel_L2_error_pressure_waveform']
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 4.8))
    for j, model in enumerate(df['model']):
        vals = [df.loc[df['model'] == model, m].values[0] for m in metrics]
        plt.bar(x + (j - 0.5) * width, vals, width, label=model)
    plt.xticks(x, ['|Z|', 'phase', 'P(t)'], rotation=0)
    plt.ylabel('Relative L2 error vs structured tree')
    plt.title('Model discrepancy relative to structured tree')
    plt.legend()
    plt.grid(axis='y', alpha=0.25)
    finalize_figure(FIG_DIR / '28_model_discrepancy_bar.png')
    print('Model discrepancy metrics done.')
    print(df)
if __name__ == '__main__':
    main()
