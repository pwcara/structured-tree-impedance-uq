from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from config import DATA_DIR

def load_json(name):
    path = DATA_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def main():
    lines = []
    lines.append('# Structured Tree Impedance Project: Results Summary\n')
    det = load_json('deterministic_summary.json')
    if det:
        lines.append('## Deterministic outlet comparison\n')
        lines.append(f"- Structured-tree peak pressure: {det['tree_peak_pressure_mmhg']:.3f} mmHg")
        lines.append(f"- R peak pressure: {det['R_peak_pressure_mmhg']:.3f} mmHg")
        lines.append(f"- RCR peak pressure: {det['RCR_peak_pressure_mmhg']:.3f} mmHg")
        lines.append(f"- Structured-tree pulse pressure: {det['tree_pulse_pressure_mmhg']:.3f} mmHg")
        lines.append('')
    uq = load_json('uq_statistics.json')
    if uq:
        lines.append('## Uncertainty propagation\n')
        lines.append(f"- Mean peak pressure: {uq['mean_peak_pressure_mmhg']:.3f} mmHg")
        lines.append(f"- Std peak pressure: {uq['std_peak_pressure_mmhg']:.3f} mmHg")
        lines.append(f"- CV peak pressure: {uq['cv_peak_pressure']:.3f}")
        lines.append(f"- Mean pulse pressure: {uq['mean_pulse_pressure_mmhg']:.3f} mmHg")
        lines.append('')
    sens_path = DATA_DIR / 'sensitivity_summary.csv'
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        peak = sens[sens['output'] == 'peak_pressure_mmhg'].copy()
        if 'prcc' in peak.columns:
            peak['abs_prcc'] = peak['prcc'].abs()
            peak = peak.sort_values('abs_prcc', ascending=False)
        lines.append('## Sensitivity ranking for peak pressure\n')
        for _, row in peak.iterrows():
            lines.append(f"- {row['input']}: PRCC = {row.get('prcc', float('nan')):.3f}, Pearson = {row.get('pearson', float('nan')):.3f}")
        lines.append('')
    pce_path = DATA_DIR / 'pce_sobol_indices.csv'
    if pce_path.exists():
        pce = pd.read_csv(pce_path).sort_values('total_index', ascending=False)
        lines.append('## PCE/Sobol-style indices\n')
        for _, row in pce.iterrows():
            lines.append(f"- {row['parameter']}: first-order = {row['first_order_index']:.3f}, total = {row['total_index']:.3f}")
        lines.append('')
    disc_path = DATA_DIR / 'model_discrepancy_metrics.csv'
    if disc_path.exists():
        disc = pd.read_csv(disc_path)
        lines.append('## Model discrepancy relative to structured tree\n')
        for _, row in disc.iterrows():
            lines.append(f"- {row['model']}: rel. |Z| error = {row['rel_L2_error_Z_magnitude']:.3f}, pressure waveform error = {row['rel_L2_error_pressure_waveform']:.3f}")
        lines.append('')
    lines.append('## Mechanistic interpretation\n')
    lines.append('The main mechanism is geometric. r_min controls how deep the distal tree extends; ell_rr controls vessel length scaling. Both alter the effective downstream impedance Z_tree(omega), which changes reflected waves and the pressure response P_hat(omega)=Z(omega)Q_hat(omega).')
    out = DATA_DIR / 'report_summary.md'
    out.write_text('\n'.join(lines))
    print(out)
    print('\n'.join(lines))
if __name__ == '__main__':
    main()
