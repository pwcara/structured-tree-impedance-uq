# Structured Tree Impedance Project

Python workflow for structured-tree outlet impedance, pressure reconstruction, uncertainty propagation, and sensitivity analysis.

## Setup

```bash
pip install -r requirements.txt
```

## Run everything

```bash
python run_all.py
```

## Core workflow

```bash
python baseline.py
python deterministic_outlets.py
python convergence.py
python uq_lhs.py
python sensitivity.py
python polynomial_surrogate.py
python simvascular_export.py
```

## Extensions

```bash
python pce_sobol_indices.py
python derivative_global_sensitivity.py
python inverse_mcmc.py
python harmonic_diagnostics.py
python model_discrepancy_metrics.py
python quantitative_metrics.py
python pce_style_response_curves.py
python make_report_summary.py
```

Outputs are written to `structured_tree_modular_outputs/`.
