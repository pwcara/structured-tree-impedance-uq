import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, PARAM_RANGES, FIG_DIR, DATA_DIR
from io_utils import finalize_figure, save_csv_dicts, save_matrix_csv
from core_impedance import make_tree_wall_from_values, compute_pressure_solution, build_time_array, synthetic_inflow_waveform
from sampling_sensitivity import standardized_regression_coefficients, partial_rank_correlation_coefficients
blood, base_wall, base_tree, config = default_objects()
uq_file = DATA_DIR / 'lhs_uq_summary.csv'
if not uq_file.exists():
    raise FileNotFoundError('Run 04_uq_lhs.py before 05_sensitivity.py')
df = pd.read_csv(uq_file)
input_names = list(PARAM_RANGES.keys())
output_names = ['peak_pressure_mmhg', 'pulse_pressure_mmhg', 'low_frequency_impedance']
corr_vars = input_names + output_names
C = df[corr_vars].corr().values
save_matrix_csv(DATA_DIR / 'correlation_matrix.csv', corr_vars, corr_vars, C)
plt.figure(figsize=(8, 6.5))
im = plt.imshow(C, vmin=-1, vmax=1, cmap='coolwarm')
plt.colorbar(im, label='Correlation')
plt.xticks(range(len(corr_vars)), corr_vars, rotation=45, ha='right')
plt.yticks(range(len(corr_vars)), corr_vars)
plt.title('Correlation matrix')
finalize_figure(FIG_DIR / '10_correlation_heatmap.png')
X = df[input_names].values
sensitivity_rows = []
for out in output_names:
    y = df[out].values
    src = standardized_regression_coefficients(X, y)
    prcc = partial_rank_correlation_coefficients(X, y)
    for j, name in enumerate(input_names):
        pearson = np.corrcoef(df[name].values, y)[0, 1]
        spearman = df[[name, out]].corr(method='spearman').iloc[0, 1]
        sensitivity_rows.append({'input': name, 'output': out, 'pearson': float(pearson), 'spearman': float(spearman), 'standardized_regression': float(src[j]), 'prcc': float(prcc[j])})
save_csv_dicts(sensitivity_rows, DATA_DIR / 'sensitivity_summary.csv')
sens_df = pd.DataFrame(sensitivity_rows)
peak = sens_df[sens_df['output'] == 'peak_pressure_mmhg'].copy()
peak['abs_prcc'] = peak['prcc'].abs()
peak = peak.sort_values('abs_prcc', ascending=True)
plt.figure(figsize=(7, 4.5))
plt.barh(peak['input'], peak['prcc'], color='#466A9F')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('PRCC')
plt.title('Global sensitivity: PRCC for peak pressure')
plt.grid(axis='x', alpha=0.25)
finalize_figure(FIG_DIR / '11_prcc_peak_pressure.png')
t = build_time_array(config)
q = synthetic_inflow_waveform(t, config.cycle_length_s)
base_sol = compute_pressure_solution(base_tree, base_wall, blood, config, t=t, q=q)
p0 = base_sol['peak_pressure_mmhg']
oat_rows = []
plt.figure(figsize=(7.5, 5.0))
for name, (lo, hi) in PARAM_RANGES.items():
    base_val = getattr(base_tree, name, None)
    if base_val is None:
        base_val = getattr(base_wall, name)
    vals = np.linspace(lo, hi, config.oat_points)
    responses = []
    for v in vals:
        tree_i, wall_i = make_tree_wall_from_values({name: float(v)}, base_tree, base_wall)
        sol = compute_pressure_solution(tree_i, wall_i, blood, config, t=t, q=q)
        responses.append(sol['peak_pressure_mmhg'])
        oat_rows.append({'parameter': name, 'value': float(v), 'normalized_value': float(v / base_val), 'peak_pressure_mmhg': sol['peak_pressure_mmhg'], 'normalized_peak_pressure': sol['peak_pressure_mmhg'] / p0})
    plt.plot(vals / base_val, np.array(responses) / p0, marker='o', label=name)
plt.axhline(1, color='black', linestyle='--', linewidth=1)
plt.axvline(1, color='black', linestyle='--', linewidth=1)
plt.xlabel('Normalized input $\\theta/\\theta_0$')
plt.ylabel('Normalized output $P_{\\max}/P_{\\max,0}$')
plt.title('Normalized one-at-a-time sensitivity')
plt.legend(fontsize=9)
plt.grid(alpha=0.25)
finalize_figure(FIG_DIR / '12_normalized_oat_sweeps.png')
pd.DataFrame(oat_rows).to_csv(DATA_DIR / 'normalized_oat_sweeps.csv', index=False)
rel_step = 0.05
log_rows = []
for name in input_names:
    theta0 = getattr(base_tree, name, None)
    if theta0 is None:
        theta0 = getattr(base_wall, name)
    minus = theta0 * (1 - rel_step)
    plus = theta0 * (1 + rel_step)
    tree_m, wall_m = make_tree_wall_from_values({name: minus}, base_tree, base_wall)
    tree_p, wall_p = make_tree_wall_from_values({name: plus}, base_tree, base_wall)
    sol_m = compute_pressure_solution(tree_m, wall_m, blood, config, t=t, q=q)
    sol_p = compute_pressure_solution(tree_p, wall_p, blood, config, t=t, q=q)
    pm = sol_m['peak_pressure_mmhg']
    pp = sol_p['peak_pressure_mmhg']
    dP_dtheta = (pp - pm) / (plus - minus)
    log_sens = (np.log(pp) - np.log(pm)) / (np.log(plus) - np.log(minus))
    log_rows.append({'parameter': name, 'theta_baseline': theta0, 'peak_minus_mmhg': pm, 'peak_plus_mmhg': pp, 'dP_dtheta': dP_dtheta, 'log_sensitivity': log_sens})
log_df = pd.DataFrame(log_rows)
log_df.to_csv(DATA_DIR / 'local_log_sensitivity.csv', index=False)
plt.figure(figsize=(7, 4.5))
plt.bar(log_df['parameter'], log_df['log_sensitivity'], color='#6A8EAE')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.ylabel('$\\partial \\log(P_{\\max})/\\partial \\log(\\theta)$')
plt.title('Local log sensitivities at baseline')
plt.xticks(rotation=25)
plt.grid(axis='y', alpha=0.25)
finalize_figure(FIG_DIR / '13_local_log_sensitivities.png')
print('Sensitivity analyses done.')
