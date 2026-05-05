import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, PARAM_RANGES, FIG_DIR, DATA_DIR
from io_utils import finalize_figure, save_json
from core_impedance import make_tree_wall_from_values, compute_pressure_solution, build_time_array, synthetic_inflow_waveform
from sampling_sensitivity import latin_hypercube, scale_lhs_to_ranges

def P1(x):
    return x

def P2(x):
    return 0.5 * (3 * x ** 2 - 1)

def build_features(X_scaled, names):
    n, d = X_scaled.shape
    feats = [np.ones(n)]
    labels = ['constant']
    for j, name in enumerate(names):
        feats.append(P1(X_scaled[:, j]))
        labels.append(f'P1({name})')
    for j, name in enumerate(names):
        feats.append(P2(X_scaled[:, j]))
        labels.append(f'P2({name})')
    for i, j in itertools.combinations(range(d), 2):
        feats.append(P1(X_scaled[:, i]) * P1(X_scaled[:, j]))
        labels.append(f'P1({names[i]})*P1({names[j]})')
    return (np.column_stack(feats), labels)
blood, base_wall, base_tree, config = default_objects()
rng = np.random.default_rng(config.random_seed + 123)
names = list(PARAM_RANGES.keys())
n_samples = max(config.n_lhs, 250)
X_unit = latin_hypercube(n_samples, len(names), rng)
X_phys, names = scale_lhs_to_ranges(X_unit, PARAM_RANGES)
X_scaled = 2 * X_unit - 1
t = build_time_array(config)
q = synthetic_inflow_waveform(t, config.cycle_length_s)
y = np.zeros(n_samples)
rows = []
for i in range(n_samples):
    values = dict(zip(names, X_phys[i, :]))
    tree_i, wall_i = make_tree_wall_from_values(values, base_tree, base_wall)
    sol = compute_pressure_solution(tree_i, wall_i, blood, config, t=t, q=q)
    y[i] = sol['peak_pressure_mmhg']
    row = values.copy()
    row['peak_pressure_mmhg'] = float(y[i])
    rows.append(row)
pd.DataFrame(rows).to_csv(DATA_DIR / 'polynomial_samples.csv', index=False)
Phi, labels = build_features(X_scaled, names)
coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
yhat = Phi @ coeffs
ss_res = np.sum((y - yhat) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
coef_df = pd.DataFrame({'term': labels, 'coefficient_mmhg': coeffs, 'abs_coefficient_mmhg': np.abs(coeffs)}).sort_values('abs_coefficient_mmhg', ascending=False)
coef_df.to_csv(DATA_DIR / 'polynomial_coefficients.csv', index=False)
plot_df = coef_df[coef_df['term'] != 'constant'].head(12).iloc[::-1]
plt.figure(figsize=(8.5, 5))
plt.barh(plot_df['term'], plot_df['coefficient_mmhg'], color='#5B8C5A')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Coefficient [mmHg]')
plt.title('Largest polynomial surrogate coefficients')
plt.grid(axis='x', alpha=0.25)
finalize_figure(FIG_DIR / '15_polynomial_coefficients.png')
plt.figure(figsize=(5.5, 5.5))
plt.scatter(y, yhat, alpha=0.65)
lo = min(y.min(), yhat.min())
hi = max(y.max(), yhat.max())
plt.plot([lo, hi], [lo, hi], 'k--')
plt.xlabel('True peak pressure [mmHg]')
plt.ylabel('Polynomial surrogate prediction [mmHg]')
plt.title(f'Polynomial surrogate fit: R² = {r2:.3f}')
plt.grid(alpha=0.25)
finalize_figure(FIG_DIR / '14_polynomial_prediction.png')
save_json({'n_samples': int(n_samples), 'r2': float(r2), 'mean_peak_pressure_mmhg': float(np.mean(y)), 'std_peak_pressure_mmhg': float(np.std(y))}, DATA_DIR / 'polynomial_summary.json')
norm_file = DATA_DIR / 'normalized_oat_sweeps.csv'
log_file = DATA_DIR / 'local_log_sensitivity.csv'
if norm_file.exists() and log_file.exists():
    oat = pd.read_csv(norm_file)
    log_df = pd.read_csv(log_file)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for name, group in oat.groupby('parameter'):
        axs[0, 0].plot(group['normalized_value'], group['normalized_peak_pressure'], marker='o', label=name)
    axs[0, 0].axhline(1, color='black', linestyle='--', linewidth=1)
    axs[0, 0].axvline(1, color='black', linestyle='--', linewidth=1)
    axs[0, 0].set_title('Normalized one-at-a-time response')
    axs[0, 0].set_xlabel('$\\theta/\\theta_0$')
    axs[0, 0].set_ylabel('$P_{\\max}/P_{\\max,0}$')
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(alpha=0.25)
    axs[0, 1].bar(log_df['parameter'], log_df['log_sensitivity'], color='#6A8EAE')
    axs[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axs[0, 1].set_title('Local log sensitivities')
    axs[0, 1].set_ylabel('$\\partial \\log P_{\\max}/\\partial \\log \\theta$')
    axs[0, 1].tick_params(axis='x', rotation=25)
    axs[0, 1].grid(axis='y', alpha=0.25)
    lhs_file = DATA_DIR / 'lhs_uq_summary.csv'
    if lhs_file.exists():
        lhs = pd.read_csv(lhs_file)
        axs[1, 0].scatter(lhs['r_min_cm'], lhs['peak_pressure_mmhg'], s=24, alpha=0.65)
        axs[1, 0].set_title('Dominant trend: $r_{\\min}$')
        axs[1, 0].set_xlabel('$r_{\\min}$ [cm]')
        axs[1, 0].set_ylabel('Peak pressure [mmHg]')
        axs[1, 0].grid(alpha=0.25)
    top = coef_df[coef_df['term'] != 'constant'].head(8).iloc[::-1]
    axs[1, 1].barh(top['term'], top['coefficient_mmhg'], color='#5B8C5A')
    axs[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
    axs[1, 1].set_title(f'Polynomial coefficients, R²={r2:.2f}')
    axs[1, 1].set_xlabel('Coefficient [mmHg]')
    axs[1, 1].grid(axis='x', alpha=0.25)
    finalize_figure(FIG_DIR / '16_sensitivity_summary_slide.png')
print('Polynomial surrogate done.')
