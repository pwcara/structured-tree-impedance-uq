from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

def latin_hypercube(n_samples: int, n_dim: int, rng: np.random.Generator):
    X = np.zeros((n_samples, n_dim))
    for j in range(n_dim):
        cut = np.linspace(0, 1, n_samples + 1)
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        X[:, j] = u
    return X

def scale_lhs_to_ranges(X_unit, ranges: dict):
    names = list(ranges.keys())
    X_phys = np.zeros_like(X_unit)
    for j, name in enumerate(names):
        lo, hi = ranges[name]
        X_phys[:, j] = lo + X_unit[:, j] * (hi - lo)
    return (X_phys, names)

def standardized_regression_coefficients(X, y):
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    yz = (y - y.mean()) / y.std(ddof=1)
    A = np.column_stack([np.ones(len(yz)), Xz])
    coeffs, *_ = np.linalg.lstsq(A, yz, rcond=None)
    return coeffs[1:]

def partial_rank_correlation_coefficients(X, y):
    Xr = np.apply_along_axis(lambda v: np.argsort(np.argsort(v)).astype(float), 0, X)
    yr = np.argsort(np.argsort(y)).astype(float)
    n, d = Xr.shape
    prcc = np.zeros(d)
    for j in range(d):
        others = [k for k in range(d) if k != j]
        A = np.column_stack([np.ones(n), Xr[:, others]])
        beta_x, *_ = np.linalg.lstsq(A, Xr[:, j], rcond=None)
        res_x = Xr[:, j] - A @ beta_x
        beta_y, *_ = np.linalg.lstsq(A, yr, rcond=None)
        res_y = yr - A @ beta_y
        prcc[j] = np.corrcoef(res_x, res_y)[0, 1]
    return prcc

def make_correlation_matrix(dataframe, variables):
    return dataframe[variables].corr().values
