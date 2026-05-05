from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, FIG_DIR, DATA_DIR
from io_utils import finalize_figure
from core_impedance import structured_tree_spectrum, resistance_spectrum, rcr_spectrum, fit_rcr_to_tree, reflection_coefficient_spectrum

def main():
    blood, wall, tree, config = default_objects()
    harmonics = np.array([1, 2, 3, 4, 5], dtype=float)
    freqs = np.linspace(0, config.f_max_hz, config.n_freqs)
    z_tree_full = structured_tree_spectrum(freqs, tree, blood, wall)
    R_dc = float(np.real(z_tree_full[0]))
    Rp, C, Rd = fit_rcr_to_tree(freqs, z_tree_full, config.rcr_fit_fmax_hz)
    z_R = resistance_spectrum(harmonics, R_dc)
    z_RCR = rcr_spectrum(harmonics, Rp, C, Rd)
    z_tree = structured_tree_spectrum(harmonics, tree, blood, wall)
    gamma_tree = reflection_coefficient_spectrum(harmonics, z_tree, tree, blood, wall)
    rows = []
    for i, f in enumerate(harmonics):
        for name, z in [('R', z_R[i]), ('RCR', z_RCR[i]), ('structured_tree', z_tree[i])]:
            rows.append({'frequency_hz': f, 'model': name, 'Z_mag': float(abs(z)), 'Z_phase_deg': float(np.angle(z, deg=True)), 'Gamma_mag_tree_only': float(abs(gamma_tree[i])) if name == 'structured_tree' else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / 'harmonic_diagnostics.csv', index=False)
    pivot = df.pivot(index='frequency_hz', columns='model', values='Z_mag')
    x = np.arange(len(harmonics))
    width = 0.25
    plt.figure(figsize=(8, 4.8))
    for j, model in enumerate(['R', 'RCR', 'structured_tree']):
        plt.bar(x + (j - 1) * width, pivot[model].values, width, label=model)
    plt.xticks(x, [f'{int(h)} Hz' for h in harmonics])
    plt.ylabel('|Z|')
    plt.title('Impedance modulus at cardiac harmonics')
    plt.legend()
    plt.grid(axis='y', alpha=0.25)
    finalize_figure(FIG_DIR / '25_harmonic_impedance_modulus.png')
    pivot_phase = df.pivot(index='frequency_hz', columns='model', values='Z_phase_deg')
    plt.figure(figsize=(8, 4.8))
    for j, model in enumerate(['R', 'RCR', 'structured_tree']):
        plt.bar(x + (j - 1) * width, pivot_phase[model].values, width, label=model)
    plt.xticks(x, [f'{int(h)} Hz' for h in harmonics])
    plt.ylabel('Phase [deg]')
    plt.title('Impedance phase at cardiac harmonics')
    plt.legend()
    plt.grid(axis='y', alpha=0.25)
    finalize_figure(FIG_DIR / '26_harmonic_impedance_phase.png')
    plt.figure(figsize=(7, 4.5))
    plt.plot(harmonics, abs(gamma_tree), marker='o', linewidth=2.2)
    plt.xlabel('Harmonic frequency [Hz]')
    plt.ylabel('|Γ|')
    plt.title('Structured-tree reflection at cardiac harmonics')
    plt.grid(alpha=0.25)
    finalize_figure(FIG_DIR / '27_harmonic_reflection_tree.png')
    print('Harmonic diagnostics done.')
    print(df)
if __name__ == '__main__':
    main()
