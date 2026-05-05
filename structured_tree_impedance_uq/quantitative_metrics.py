from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import default_objects, FIG_DIR, DATA_DIR, DYN_PER_CM2_TO_MMHG
from io_utils import finalize_figure, save_json
from core_impedance import structured_tree_spectrum, resistance_spectrum, rcr_spectrum, fit_rcr_to_tree, reflection_coefficient_spectrum, build_time_array, synthetic_inflow_waveform, pressure_from_impedance, make_tree_wall_from_values

def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-14))

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def percentile_summary(x):
    return {'p05': float(np.percentile(x, 5)), 'p50': float(np.percentile(x, 50)), 'p95': float(np.percentile(x, 95)), 'mean': float(np.mean(x)), 'std': float(np.std(x)), 'cv': float(np.std(x) / max(np.mean(x), 1e-14)), 'iqr': float(np.percentile(x, 75) - np.percentile(x, 25)), 'width_90': float(np.percentile(x, 95) - np.percentile(x, 5))}

def compute_deterministic_metrics():
    blood, wall, tree, config = default_objects()
    freqs = np.linspace(0.0, config.f_max_hz, config.n_freqs)
    z_tree = structured_tree_spectrum(freqs, tree, blood, wall)
    R_dc = float(np.real(z_tree[0]))
    z_R = resistance_spectrum(freqs, R_dc)
    Rp, C, Rd = fit_rcr_to_tree(freqs, z_tree, config.rcr_fit_fmax_hz)
    z_RCR = rcr_spectrum(freqs, Rp, C, Rd)
    gamma_tree = reflection_coefficient_spectrum(freqs, z_tree, tree, blood, wall)
    t = build_time_array(config)
    q = synthetic_inflow_waveform(t, config.cycle_length_s)
    fft_freqs = np.fft.rfftfreq(len(t), d=config.dt)
    qhat = np.fft.rfft(q)
    z_tree_td = structured_tree_spectrum(fft_freqs, tree, blood, wall)
    z_R_td = resistance_spectrum(fft_freqs, R_dc)
    z_RCR_td = rcr_spectrum(fft_freqs, Rp, C, Rd)
    p_tree = pressure_from_impedance(t, q, z_tree_td) * DYN_PER_CM2_TO_MMHG
    p_R = pressure_from_impedance(t, q, z_R_td) * DYN_PER_CM2_TO_MMHG
    p_RCR = pressure_from_impedance(t, q, z_RCR_td) * DYN_PER_CM2_TO_MMHG
    mask = freqs > 0
    rows = []
    for model_name, z, z_td, p in [('R', z_R, z_R_td, p_R), ('RCR', z_RCR, z_RCR_td, p_RCR)]:
        pressure_spectral_error = rel_l2((z_td - z_tree_td) * qhat, z_tree_td * qhat)
        rows.append({'model': model_name, 'rel_L2_error_complex_Z': rel_l2(z[mask], z_tree[mask]), 'rel_L2_error_Z_magnitude': rel_l2(np.abs(z[mask]), np.abs(z_tree[mask])), 'RMSE_phase_deg': rmse(np.angle(z[mask], deg=True), np.angle(z_tree[mask], deg=True)), 'rel_L2_pressure_waveform_error': rel_l2(p, p_tree), 'RMSE_pressure_waveform_mmhg': rmse(p, p_tree), 'max_abs_pressure_error_mmhg': float(np.max(np.abs(p - p_tree))), 'peak_pressure_error_mmhg': float(np.max(p) - np.max(p_tree)), 'mean_pressure_error_mmhg': float(np.mean(p) - np.mean(p_tree)), 'pulse_pressure_error_mmhg': float(np.max(p) - np.min(p) - (np.max(p_tree) - np.min(p_tree))), 'flow_weighted_spectral_pressure_error': pressure_spectral_error})
    disc_df = pd.DataFrame(rows)
    disc_df.to_csv(DATA_DIR / 'quantified_model_discrepancy.csv', index=False)
    gamma_abs = np.abs(gamma_tree[mask])
    freqs_pos = freqs[mask]
    reflection_metrics = {'mean_abs_gamma': float(np.mean(gamma_abs)), 'max_abs_gamma': float(np.max(gamma_abs)), 'freq_integrated_abs_gamma': float(np.trapz(gamma_abs, freqs_pos)), 'freq_average_abs_gamma': float(np.trapz(gamma_abs, freqs_pos) / (freqs_pos[-1] - freqs_pos[0]))}
    harmonic_rows = []
    for h in [1, 2, 3, 4, 5]:
        idx = int(np.argmin(np.abs(freqs - h)))
        harmonic_rows.append({'frequency_hz': float(freqs[idx]), 'Z_tree_mag': float(np.abs(z_tree[idx])), 'Z_tree_phase_deg': float(np.angle(z_tree[idx], deg=True)), 'Gamma_mag': float(np.abs(gamma_tree[idx]))})
    pd.DataFrame(harmonic_rows).to_csv(DATA_DIR / 'quantified_harmonic_reflection.csv', index=False)
    plt.figure(figsize=(7.5, 4.8))
    x = np.arange(len(disc_df))
    width = 0.24
    plt.bar(x - width, disc_df['rel_L2_pressure_waveform_error'], width, label='rel L2 P(t)')
    plt.bar(x, disc_df['flow_weighted_spectral_pressure_error'], width, label='flow-weighted spectral')
    plt.bar(x + width, disc_df['rel_L2_error_Z_magnitude'], width, label='rel L2 |Z|')
    plt.xticks(x, disc_df['model'])
    plt.ylabel('Relative error vs structured tree')
    plt.title('Quantified discrepancy relative to structured tree')
    plt.legend()
    plt.grid(axis='y', alpha=0.25)
    finalize_figure(FIG_DIR / '29_quantified_model_discrepancy.png')
    plt.figure(figsize=(7.5, 4.6))
    plt.plot(freqs_pos, gamma_abs, linewidth=2.2, label='$|\\Gamma(\\omega)|$')
    plt.axhline(reflection_metrics['freq_average_abs_gamma'], color='black', linestyle='--', label='frequency average')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('$|\\Gamma(\\omega)|$')
    plt.title('Reflection burden of structured-tree outlet')
    plt.legend()
    plt.grid(alpha=0.25)
    finalize_figure(FIG_DIR / '30_reflection_burden.png')
    deterministic_summary = {'R_dc': R_dc, 'Rp': Rp, 'C': C, 'Rd': Rd, 'tree_peak_pressure_mmhg': float(np.max(p_tree)), 'tree_mean_pressure_mmhg': float(np.mean(p_tree)), 'tree_pulse_pressure_mmhg': float(np.max(p_tree) - np.min(p_tree)), 'reflection_metrics': reflection_metrics}
    return (deterministic_summary, disc_df, reflection_metrics)

def compute_uq_metrics():
    lhs_file = DATA_DIR / 'lhs_uq_summary.csv'
    if not lhs_file.exists():
        print('lhs_uq_summary.csv not found. Run 04_uq_lhs.py first.')
        return None
    blood, base_wall, base_tree, config = default_objects()
    df = pd.read_csv(lhs_file)
    scalar_metrics = {}
    for col in ['peak_pressure_mmhg', 'pulse_pressure_mmhg', 'mean_pressure_mmhg', 'low_frequency_impedance']:
        if col in df.columns:
            scalar_metrics[col] = percentile_summary(df[col].values)
    t = build_time_array(config)
    q = synthetic_inflow_waveform(t, config.cycle_length_s)
    pressure_samples = np.zeros((len(df), len(t)))
    for i, row in df.iterrows():
        values = {'r_min_cm': row['r_min_cm'], 'length_to_radius': row['length_to_radius'], 'asymmetry_g': row['asymmetry_g'], 'k1': row['k1']}
        tree_i, wall_i = make_tree_wall_from_values(values, base_tree, base_wall)
        sol = None
        from core_impedance import compute_pressure_solution
        sol = compute_pressure_solution(tree_i, wall_i, blood, config, t=t, q=q)
        pressure_samples[i, :] = sol['p']
    p05 = np.percentile(pressure_samples, 5, axis=0)
    p50 = np.percentile(pressure_samples, 50, axis=0)
    p95 = np.percentile(pressure_samples, 95, axis=0)
    width = p95 - p05
    T = config.cycle_length_s
    avg_width = float(np.trapz(width, t) / T)
    max_width = float(np.max(width))
    t_max_width = float(t[np.argmax(width)])
    systolic_mask = t <= 0.35 * T
    diastolic_mask = ~systolic_mask
    systolic_avg_width = float(np.mean(width[systolic_mask]))
    diastolic_avg_width = float(np.mean(width[diastolic_mask]))
    baseline_peak = None
    det_file = DATA_DIR / 'deterministic_summary.json'
    if det_file.exists():
        import json
        with open(det_file) as f:
            det = json.load(f)
        baseline_peak = det.get('tree_peak_pressure_mmhg')
    peak = df['peak_pressure_mmhg'].values
    exceedance = {}
    if baseline_peak is not None:
        exceedance['prob_peak_exceeds_baseline_tree_peak'] = float(np.mean(peak > baseline_peak))
        exceedance['prob_peak_exceeds_baseline_plus_5mmHg'] = float(np.mean(peak > baseline_peak + 5.0))
        exceedance['prob_peak_below_baseline_minus_5mmHg'] = float(np.mean(peak < baseline_peak - 5.0))
    waveform_metrics = {'avg_90pct_band_width_mmhg': avg_width, 'max_90pct_band_width_mmhg': max_width, 'time_of_max_band_width_s': t_max_width, 'systolic_avg_90pct_width_mmhg': systolic_avg_width, 'diastolic_avg_90pct_width_mmhg': diastolic_avg_width, 'systolic_to_diastolic_width_ratio': float(systolic_avg_width / max(diastolic_avg_width, 1e-14))}
    uq_metrics = {'scalar_metrics': scalar_metrics, 'waveform_uncertainty_metrics': waveform_metrics, 'exceedance_metrics': exceedance}
    save_json(uq_metrics, DATA_DIR / 'quantified_uq_metrics.json')
    pd.DataFrame({'time_s': t, 'p05_mmhg': p05, 'p50_mmhg': p50, 'p95_mmhg': p95, 'width_90pct_mmhg': width}).to_csv(DATA_DIR / 'quantified_pressure_band_width.csv', index=False)
    plt.figure(figsize=(7.5, 4.6))
    plt.plot(t, width, linewidth=2.2)
    plt.axvline(t_max_width, color='red', linestyle='--', label=f'max width at t={t_max_width:.2f}s')
    plt.xlabel('Time [s]')
    plt.ylabel('90% pressure band width [mmHg]')
    plt.title('Quantified waveform uncertainty width')
    plt.legend()
    plt.grid(alpha=0.25)
    finalize_figure(FIG_DIR / '31_pressure_uncertainty_width.png')
    metrics_to_plot = [('peak 90% width', scalar_metrics['peak_pressure_mmhg']['width_90']), ('pulse 90% width', scalar_metrics['pulse_pressure_mmhg']['width_90']), ('avg waveform width', avg_width), ('max waveform width', max_width)]
    plt.figure(figsize=(8, 4.8))
    plt.bar([m[0] for m in metrics_to_plot], [m[1] for m in metrics_to_plot])
    plt.xticks(rotation=20, ha='right')
    plt.ylabel('Width [mmHg]')
    plt.title('Quantified pressure uncertainty metrics')
    plt.grid(axis='y', alpha=0.25)
    finalize_figure(FIG_DIR / '32_pressure_uncertainty_metrics.png')
    return uq_metrics

def pull_sensitivity_summaries():
    output = {}
    sens_file = DATA_DIR / 'sensitivity_summary.csv'
    if sens_file.exists():
        sens = pd.read_csv(sens_file)
        peak = sens[sens['output'] == 'peak_pressure_mmhg'].copy()
        if len(peak) > 0 and 'prcc' in peak.columns:
            peak['abs_prcc'] = peak['prcc'].abs()
            peak = peak.sort_values('abs_prcc', ascending=False)
            output['top_prcc_peak_pressure'] = peak.iloc[0].to_dict()
            peak.to_csv(DATA_DIR / 'quantified_peak_pressure_sensitivity_rank.csv', index=False)
    pce_file = DATA_DIR / 'pce_sobol_indices.csv'
    if pce_file.exists():
        pce = pd.read_csv(pce_file).copy()
        if 'total_index' in pce.columns:
            pce = pce.sort_values('total_index', ascending=False)
            output['top_pce_total_index'] = pce.iloc[0].to_dict()
    return output

def make_one_page_quant_summary(det_summary, disc_df, uq_metrics, sens_summary):
    lines = []
    lines.append('Quantitative Summary')
    lines.append('')
    lines.append(f"Structured-tree peak pressure: {det_summary['tree_peak_pressure_mmhg']:.2f} mmHg")
    lines.append(f"Structured-tree pulse pressure: {det_summary['tree_pulse_pressure_mmhg']:.2f} mmHg")
    lines.append(f"Mean |Γ| over frequency: {det_summary['reflection_metrics']['mean_abs_gamma']:.3f}")
    lines.append(f"Max |Γ| over frequency: {det_summary['reflection_metrics']['max_abs_gamma']:.3f}")
    if uq_metrics:
        peak = uq_metrics['scalar_metrics']['peak_pressure_mmhg']
        wf = uq_metrics['waveform_uncertainty_metrics']
        lines.append('')
        lines.append(f"Peak pressure mean ± std: {peak['mean']:.2f} ± {peak['std']:.2f} mmHg")
        lines.append(f"Peak pressure 90% width: {peak['width_90']:.2f} mmHg")
        lines.append(f"Average waveform band width: {wf['avg_90pct_band_width_mmhg']:.2f} mmHg")
        lines.append(f"Max waveform band width: {wf['max_90pct_band_width_mmhg']:.2f} mmHg")
        lines.append(f"Systolic/diastolic width ratio: {wf['systolic_to_diastolic_width_ratio']:.2f}")
    if sens_summary.get('top_prcc_peak_pressure'):
        top = sens_summary['top_prcc_peak_pressure']
        lines.append('')
        lines.append(f"Top PRCC driver of peak pressure: {top['input']} (PRCC={top['prcc']:.3f})")
    if sens_summary.get('top_pce_total_index'):
        top = sens_summary['top_pce_total_index']
        lines.append(f"Top PCE total-index driver: {top['parameter']} (ST={top['total_index']:.3f})")
    lines.append('')
    for _, row in disc_df.iterrows():
        lines.append(f"{row['model']} pressure waveform error vs tree: {row['rel_L2_pressure_waveform_error']:.3f}")
    plt.figure(figsize=(8.5, 6))
    plt.axis('off')
    plt.text(0.02, 0.98, '\n'.join(lines), va='top', ha='left', fontsize=11, family='monospace')
    plt.title('Quantified project metrics', loc='left', fontsize=14)
    finalize_figure(FIG_DIR / '33_quantitative_dashboard.png')

def main():
    print('Computing deterministic quantitative metrics...')
    det_summary, disc_df, refl = compute_deterministic_metrics()
    print('Computing UQ quantitative metrics...')
    uq_metrics = compute_uq_metrics()
    print('Pulling sensitivity summaries...')
    sens_summary = pull_sensitivity_summaries()
    combined = {'deterministic_summary': det_summary, 'uq_metrics': uq_metrics, 'sensitivity_summary': sens_summary, 'interpretation': {'model_discrepancy': 'Quantifies how R and RCR differ from structured tree in impedance and pressure.', 'reflection_burden': 'Quantifies average and maximum reflection produced by the structured-tree outlet.', 'waveform_uncertainty': 'Quantifies how wide the pressure uncertainty band is over time.', 'exceedance': 'Quantifies probability of exceeding baseline pressure thresholds under distal uncertainty.'}}
    save_json(combined, DATA_DIR / 'quantitative_metrics_summary.json')
    make_one_page_quant_summary(det_summary, disc_df, uq_metrics, sens_summary)
    print('Quantitative metrics complete.')
    print('Main outputs:')
    print('  data/quantitative_metrics_summary.json')
    print('  data/quantified_model_discrepancy.csv')
    print('  data/quantified_uq_metrics.json')
    print('  figures/29_quantified_model_discrepancy.png')
    print('  figures/30_reflection_burden.png')
    print('  figures/31_pressure_uncertainty_width.png')
    print('  figures/32_pressure_uncertainty_metrics.png')
    print('  figures/33_quantitative_dashboard.png')
if __name__ == '__main__':
    main()
