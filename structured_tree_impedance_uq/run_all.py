import subprocess
import sys

scripts = [
    'baseline.py',
    'deterministic_outlets.py',
    'convergence.py',
    'uq_lhs.py',
    'sensitivity.py',
    'polynomial_surrogate.py',
    'simvascular_export.py',
    'pce_sobol_indices.py',
    'derivative_global_sensitivity.py',
    'inverse_mcmc.py',
    'harmonic_diagnostics.py',
    'model_discrepancy_metrics.py',
    'make_report_summary.py',
    'quantitative_metrics.py',
    'pce_style_response_curves.py',

]

for script in scripts:
    print(f"\nRunning {script}")
    subprocess.run([sys.executable, script], check=True)

print("\nAll analyses completed.")
