from __future__ import annotations
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import DATA_DIR, FIG_DIR, PARAM_RANGES
from io_utils import finalize_figure, save_json

def legendre(n: int, x: np.ndarray) -> np.ndarray:
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    if n == 2:
        return 0.5 * (3 * x ** 2 - 1)
    if n == 3:
        return 0.5 * (5 * x ** 3 - 3 * x)
    raise ValueError('This script supports degree <= 3.')

def legendre_norm(n: int) -> float:
    return 1.0 / (2 * n + 1)

def build_total_degree_multiindices(dim: int, degree: int):
    terms = []
    for alpha in itertools.product(range(degree + 1), repeat=dim):
        if sum(alpha) <= degree:
            terms.append(alpha)
    return terms

def build_design_matrix(X_scaled: np.ndarray, terms: list[tuple[int, ...]]):
    Phi = np.ones((X_scaled.shape[0], len(terms)))
    for k, alpha in enumerate(terms):
        val = np.ones(X_scaled.shape[0])
        for j, deg in enumerate(alpha):
            val *= legendre(deg, X_scaled[:, j])
        Phi[:, k] = val
    return Phi

def scale_to_minus_one_one(df: pd.DataFrame, names: list[str]):
    X = np.zeros((len(df), len(names)))
    for j, name in enumerate(names):
        lo, hi = PARAM_RANGES[name]
        X[:, j] = 2.0 * (df[name].values - lo) / (hi - lo) - 1.0
    return X

def term_label(alpha, names):
    pieces = []
    for deg, name in zip(alpha, names):
        if deg > 0:
            pieces.append(f'P{deg}({name})')
    return 'constant' if not pieces else '*'.join(pieces)

def main():
    lhs_file = DATA_DIR / 'lhs_uq_summary.csv'
    if not lhs_file.exists():
        raise FileNotFoundError('Run 04_uq_lhs.py first; lhs_uq_summary.csv not found.')
    df = pd.read_csv(lhs_file)
    names = list(PARAM_RANGES.keys())
    y = df['peak_pressure_mmhg'].values
    X_scaled = scale_to_minus_one_one(df, names)
    degree = 3
    terms = build_total_degree_multiindices(dim=len(names), degree=degree)
    Phi = build_design_matrix(X_scaled, terms)
    coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    yhat = Phi @ coeffs
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    rows = []
    total_var = 0.0
    for alpha, c in zip(terms, coeffs):
        if sum(alpha) == 0:
            norm = 1.0
            var_contrib = 0.0
        else:
            norm = np.prod([legendre_norm(a) for a in alpha])
            var_contrib = c ** 2 * norm
            total_var += var_contrib
        rows.append({'term': term_label(alpha, names), 'multi_index': str(alpha), 'coefficient_mmhg': float(c), 'basis_norm': float(norm), 'variance_contribution': float(var_contrib), 'abs_coefficient_mmhg': float(abs(c)), 'total_degree': int(sum(alpha))})
    coef_df = pd.DataFrame(rows).sort_values('variance_contribution', ascending=False)
    coef_df.to_csv(DATA_DIR / 'pce_coefficients.csv', index=False)
    sobol_rows = []
    for j, name in enumerate(names):
        first = 0.0
        total = 0.0
        for alpha, c in zip(terms, coeffs):
            if sum(alpha) == 0:
                continue
            norm = np.prod([legendre_norm(a) for a in alpha])
            vc = c ** 2 * norm
            nonzero_vars = [idx for idx, a in enumerate(alpha) if a > 0]
            if nonzero_vars == [j]:
                first += vc
            if j in nonzero_vars:
                total += vc
        sobol_rows.append({'parameter': name, 'first_order_index': float(first / total_var) if total_var > 0 else np.nan, 'total_index': float(total / total_var) if total_var > 0 else np.nan})
    sobol_df = pd.DataFrame(sobol_rows)
    sobol_df.to_csv(DATA_DIR / 'pce_sobol_indices.csv', index=False)
    save_json({'degree': degree, 'n_terms': len(terms), 'r2': float(r2), 'rmse_mmhg': rmse, 'total_surrogate_variance': float(total_var), 'output': 'peak_pressure_mmhg', 'interpretation': 'Sobol-style indices are approximate and are computed from the Legendre polynomial surrogate.'}, DATA_DIR / 'pce_summary.json')
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y, yhat, alpha=0.7, s=30)
    lo = min(y.min(), yhat.min())
    hi = max(y.max(), yhat.max())
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5)
    plt.xlabel('True peak pressure [mmHg]')
    plt.ylabel('PCE surrogate prediction [mmHg]')
    plt.title(f'PCE surrogate prediction (R² = {r2:.3f})')
    plt.grid(alpha=0.25)
    finalize_figure(FIG_DIR / '17_pce_prediction.png')
    x = np.arange(len(sobol_df))
    width = 0.36
    plt.figure(figsize=(7.3, 4.6))
    plt.bar(x - width / 2, sobol_df['first_order_index'], width, label='First-order')
    plt.bar(x + width / 2, sobol_df['total_index'], width, label='Total')
    plt.xticks(x, sobol_df['parameter'], rotation=20)
    plt.ylabel('Sobol-style index')
    plt.title('Approximate variance decomposition from PCE')
    plt.legend()
    plt.grid(axis='y', alpha=0.25)
    finalize_figure(FIG_DIR / '18_pce_sobol_indices.png')
    top = coef_df[coef_df['term'] != 'constant'].head(12).iloc[::-1]
    plt.figure(figsize=(8.5, 5.2))
    plt.barh(top['term'], top['variance_contribution'])
    plt.xlabel('Variance contribution')
    plt.title('Top PCE variance-contributing terms')
    plt.grid(axis='x', alpha=0.25)
    finalize_figure(FIG_DIR / '19_pce_top_terms.png')
    print('PCE/Sobol analysis done.')
    print('R2:', r2)
    print(sobol_df)
if __name__ == '__main__':
    main()
