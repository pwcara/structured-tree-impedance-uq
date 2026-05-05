import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict
from config import default_objects, FIG_DIR, DATA_DIR
from io_utils import finalize_figure, save_json
from core_impedance import build_time_array, synthetic_inflow_waveform, structured_tree_spectrum, compute_pressure_solution
blood, wall, tree, config = default_objects()
t = build_time_array(config)
q = synthetic_inflow_waveform(t, config.cycle_length_s)
sol = compute_pressure_solution(tree, wall, blood, config, t=t, q=q)
freqs = np.linspace(0, config.f_max_hz, config.n_freqs)
z_tree = structured_tree_spectrum(freqs, tree, blood, wall)
mask = freqs > 0
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
axs[0, 0].plot(t, q, color='#1f77b4')
axs[0, 0].set_title('Baseline inlet flow')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Flow [cm$^3$/s]')
axs[0, 0].grid(alpha=0.3)
axs[0, 1].plot(t, sol['p'], color='#d62728')
axs[0, 1].set_title('Baseline structured-tree pressure')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Pressure [mmHg]')
axs[0, 1].grid(alpha=0.3)
axs[1, 0].plot(freqs[mask], abs(z_tree[mask]), color='#2ca02c')
axs[1, 0].set_title('Structured-tree impedance modulus')
axs[1, 0].set_xlabel('Frequency [Hz]')
axs[1, 0].set_ylabel('$|Z(\\omega)|$')
axs[1, 0].grid(alpha=0.3)
axs[1, 1].plot(freqs[mask], np.angle(z_tree[mask], deg=True), color='#9467bd')
axs[1, 1].set_title('Structured-tree impedance phase')
axs[1, 1].set_xlabel('Frequency [Hz]')
axs[1, 1].set_ylabel('Phase [deg]')
axs[1, 1].grid(alpha=0.3)
finalize_figure(FIG_DIR / '01_baseline_dashboard.png')
save_json({'baseline_peak_pressure_mmhg': sol['peak_pressure_mmhg'], 'baseline_min_pressure_mmhg': sol['min_pressure_mmhg'], 'baseline_mean_pressure_mmhg': sol['mean_pressure_mmhg'], 'baseline_pulse_pressure_mmhg': sol['pulse_pressure_mmhg'], 'blood': asdict(blood), 'wall': asdict(wall), 'tree': {'root_radius_cm': tree.root_radius_cm, 'r_min_cm': tree.r_min_cm, 'length_to_radius': tree.length_to_radius, 'asymmetry_g': tree.asymmetry_g, 'radius_exponent_j': tree.radius_exponent_j, 'max_generations': tree.max_generations}, 'config': asdict(config)}, DATA_DIR / 'baseline_summary.json')
print('Baseline done.')
print('Figure:', FIG_DIR / '01_baseline_dashboard.png')
print('Summary:', DATA_DIR / 'baseline_summary.json')
