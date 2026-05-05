import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, PARAM_RANGES, FIG_DIR, DATA_DIR
from io_utils import finalize_figure, save_csv_dicts, save_json
from core_impedance import make_tree_wall_from_values, compute_pressure_solution, build_time_array, synthetic_inflow_waveform, structured_tree_spectrum, reflection_coefficient_spectrum
from sampling_sensitivity import latin_hypercube, scale_lhs_to_ranges
blood, base_wall, base_tree, config = default_objects()
rng = np.random.default_rng(config.random_seed)
names = list(PARAM_RANGES.keys())
X_unit = latin_hypercube(config.n_lhs, len(names), rng)
X_phys, names = scale_lhs_to_ranges(X_unit, PARAM_RANGES)
t = build_time_array(config)
q = synthetic_inflow_waveform(t, config.cycle_length_s)
fft_freqs = np.fft.rfftfreq(len(t), d=config.dt)
rows = []
pressure_samples = np.zeros((config.n_lhs, len(t)))
for i in range(config.n_lhs):
    values = dict(zip(names, X_phys[i, :]))
    tree_i, wall_i = make_tree_wall_from_values(values, base_tree, base_wall)
    sol = compute_pressure_solution(tree_i, wall_i, blood, config, t=t, q=q)
    z = sol['z']
    gamma = reflection_coefficient_spectrum(fft_freqs, z, tree_i, blood, wall_i)
    row = {'sample_id': i}
    row.update(values)
    row.update({'peak_pressure_mmhg': sol['peak_pressure_mmhg'], 'min_pressure_mmhg': sol['min_pressure_mmhg'], 'mean_pressure_mmhg': sol['mean_pressure_mmhg'], 'pulse_pressure_mmhg': sol['pulse_pressure_mmhg'], 'low_frequency_impedance': sol['low_frequency_impedance']})
    for harmonic in [1.0, 2.0, 3.0]:
        idx = int(np.argmin(np.abs(fft_freqs - harmonic)))
        row[f'Zmag_{harmonic:g}Hz'] = float(np.abs(z[idx]))
        row[f'Zphase_{harmonic:g}Hz_deg'] = float(np.angle(z[idx], deg=True))
        row[f'Gamma_{harmonic:g}Hz'] = float(np.abs(gamma[idx]))
    rows.append(row)
    pressure_samples[i, :] = sol['p']
df = pd.DataFrame(rows)
df.to_csv(DATA_DIR / 'lhs_uq_summary.csv', index=False)
peak = df['peak_pressure_mmhg'].values
pulse = df['pulse_pressure_mmhg'].values
save_json({'n_samples': int(config.n_lhs), 'mean_peak_pressure_mmhg': float(np.mean(peak)), 'std_peak_pressure_mmhg': float(np.std(peak)), 'cv_peak_pressure': float(np.std(peak) / max(np.mean(peak), 1e-14)), 'mean_pulse_pressure_mmhg': float(np.mean(pulse)), 'std_pulse_pressure_mmhg': float(np.std(pulse))}, DATA_DIR / 'uq_statistics.json')
plt.figure(figsize=(7, 4.5))
plt.hist(peak, bins=18, alpha=0.8, color='#3C78A8')
plt.xlabel('Peak pressure [mmHg]')
plt.ylabel('Count')
plt.title('LHS distribution of peak proximal pressure')
plt.grid(alpha=0.25)
finalize_figure(FIG_DIR / '07_lhs_peak_pressure_histogram.png')
lo = np.percentile(pressure_samples, 5, axis=0)
md = np.percentile(pressure_samples, 50, axis=0)
hi = np.percentile(pressure_samples, 95, axis=0)
plt.figure(figsize=(7.5, 4.5))
plt.fill_between(t, lo, hi, alpha=0.25, color='#3C78A8', label='5–95% band')
plt.plot(t, md, color='#0B3C5D', linewidth=2.5, label='Median')
plt.xlabel('Time [s]')
plt.ylabel('Pressure [mmHg]')
plt.title('Pressure uncertainty band')
plt.legend()
plt.grid(alpha=0.3)
finalize_figure(FIG_DIR / '08_pressure_uncertainty_band.png')
for name in names:
    plt.figure(figsize=(6.5, 4.2))
    plt.scatter(df[name], df['peak_pressure_mmhg'], s=32, alpha=0.75)
    plt.xlabel(name)
    plt.ylabel('Peak pressure [mmHg]')
    plt.title(f'Peak pressure vs {name}')
    plt.grid(alpha=0.25)
    finalize_figure(FIG_DIR / f'09_scatter_peak_vs_{name}.png')
print('LHS UQ study done.')
