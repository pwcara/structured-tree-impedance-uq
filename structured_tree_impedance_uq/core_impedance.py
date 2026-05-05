from __future__ import annotations
from dataclasses import asdict
import numpy as np
from scipy.optimize import least_squares
from config import BloodProps, WallLaw, TreeParams, RunConfig, DYN_PER_CM2_TO_MMHG

def eh_over_r0(radius_cm: float, wall: WallLaw) -> float:
    return wall.k1 * np.exp(wall.k2 * radius_cm) + wall.k3

def wave_speed_cm_s(radius_cm: float, blood: BloodProps, wall: WallLaw) -> float:
    val = 2.0 / (3.0 * blood.rho) * eh_over_r0(radius_cm, wall)
    return np.sqrt(max(val, 1e-14))

def daughter_scale_factors(g: float, j: float) -> tuple[float, float]:
    a = (1.0 + g ** (j / 2.0)) ** (-1.0 / j)
    b = a * np.sqrt(g)
    return (a, b)

def segment_length_cm(radius_cm: float, length_to_radius: float) -> float:
    return length_to_radius * radius_cm

def poiseuille_resistance(radius_cm: float, length_cm: float, blood: BloodProps) -> float:
    mu = blood.rho * blood.nu
    r = max(radius_cm, 1e-12)
    return 8.0 * mu * length_cm / (np.pi * r ** 4)

def parallel_impedance(z1: complex, z2: complex) -> complex:
    eps = 1e-14
    if abs(z1) < eps or abs(z2) < eps:
        return 0.0 + 0j
    denom = 1.0 / z1 + 1.0 / z2
    if abs(denom) < eps:
        return np.inf + 0j
    return 1.0 / denom

def vessel_input_impedance(omega: float, radius_cm: float, length_cm: float, z_terminal: complex, blood: BloodProps, wall: WallLaw, include_viscosity: bool=True) -> complex:
    area = np.pi * radius_cm ** 2
    c0 = wave_speed_cm_s(radius_cm, blood, wall)
    if abs(omega) < 1e-14:
        return poiseuille_resistance(radius_cm, length_cm, blood) + z_terminal
    if include_viscosity:
        f0 = 8.0 * np.pi * blood.nu / max(area, 1e-14)
        lam = np.sqrt((-omega ** 2 + 1j * omega * f0) / max(c0 ** 2, 1e-14))
    else:
        lam = 1j * omega / max(c0, 1e-14)
    if np.real(lam) < 0:
        lam = -lam
    z0 = blood.rho * c0 ** 2 * lam / (1j * omega * max(area, 1e-14))
    denom = z0 + z_terminal
    gamma = (z0 - z_terminal) / denom if abs(denom) > 1e-14 else 1.0 + 0j
    exp_term = np.exp(-2.0 * lam * length_cm)
    denom2 = 1.0 + gamma * exp_term
    if abs(denom2) < 1e-14:
        denom2 = 1e-14 + 0j
    return z0 * (1.0 - gamma * exp_term) / denom2

def make_radius_lookup(tree: TreeParams):
    a, b = daughter_scale_factors(tree.asymmetry_g, tree.radius_exponent_j)

    def radius_at(gen: int, n_b: int) -> float:
        n_a = gen - n_b
        return tree.root_radius_cm * a ** n_a * b ** n_b
    return radius_at

def tree_input_impedance(omega: float, tree: TreeParams, blood: BloodProps, wall: WallLaw) -> complex:
    radius_at = make_radius_lookup(tree)
    cache: dict[tuple[int, int], complex] = {}

    def recurse(gen: int, n_b: int) -> complex:
        key = (gen, n_b)
        if key in cache:
            return cache[key]
        r = radius_at(gen, n_b)
        L = segment_length_cm(r, tree.length_to_radius)
        terminal = r < tree.r_min_cm or gen >= tree.max_generations
        if terminal:
            zi = vessel_input_impedance(omega, r, L, tree.z_leaf, blood, wall, tree.include_viscosity)
            cache[key] = zi
            return zi
        z1 = recurse(gen + 1, n_b)
        z2 = recurse(gen + 1, n_b + 1)
        zT = parallel_impedance(z1, z2)
        zi = vessel_input_impedance(omega, r, L, zT, blood, wall, tree.include_viscosity)
        cache[key] = zi
        return zi
    return recurse(0, 0)

def structured_tree_spectrum(freqs_hz, tree: TreeParams, blood: BloodProps, wall: WallLaw):
    z = np.zeros_like(freqs_hz, dtype=complex)
    for i, f in enumerate(freqs_hz):
        z[i] = tree_input_impedance(2.0 * np.pi * f, tree, blood, wall)
    return z

def resistance_spectrum(freqs_hz, R: float):
    return np.full_like(freqs_hz, R + 0j, dtype=complex)

def rcr_spectrum(freqs_hz, Rp: float, C: float, Rd: float):
    omega = 2.0 * np.pi * freqs_hz
    z_parallel = Rd / (1.0 + 1j * omega * Rd * C)
    return Rp + z_parallel

def fit_rcr_to_tree(freqs_hz, z_target, f_fit_max_hz=4.0):
    mask = (freqs_hz >= 0) & (freqs_hz <= f_fit_max_hz)
    f = freqs_hz[mask]
    z = z_target[mask]
    Rtot = max(float(np.real(z_target[0])), 1e-12)
    x0 = np.log([0.1 * Rtot, 0.5 / (0.9 * Rtot), 0.9 * Rtot])
    scale = max(np.median(np.abs(z)), 1e-12)

    def residual(log_params):
        Rp, C, Rd = np.exp(log_params)
        zm = rcr_spectrum(f, Rp, C, Rd)
        res = (zm - z) / scale
        return np.concatenate([np.real(res), np.imag(res)])
    sol = least_squares(residual, x0, max_nfev=5000)
    return tuple(map(float, np.exp(sol.x)))

def root_characteristic_impedance(freqs_hz, tree: TreeParams, blood: BloodProps, wall: WallLaw):
    r = tree.root_radius_cm
    A = np.pi * r ** 2
    c0 = wave_speed_cm_s(r, blood, wall)
    z0 = np.zeros_like(freqs_hz, dtype=complex)
    for i, f in enumerate(freqs_hz):
        omega = 2.0 * np.pi * f
        if abs(omega) < 1e-14:
            z0[i] = blood.rho * c0 / max(A, 1e-14)
        else:
            f0 = 8.0 * np.pi * blood.nu / max(A, 1e-14)
            lam = np.sqrt((-omega ** 2 + 1j * omega * f0) / max(c0 ** 2, 1e-14))
            if np.real(lam) < 0:
                lam = -lam
            z0[i] = blood.rho * c0 ** 2 * lam / (1j * omega * max(A, 1e-14))
    return z0

def reflection_coefficient_spectrum(freqs_hz, z_terminal, tree: TreeParams, blood: BloodProps, wall: WallLaw):
    z0 = root_characteristic_impedance(freqs_hz, tree, blood, wall)
    denom = z0 + z_terminal
    gamma = np.zeros_like(z_terminal, dtype=complex)
    mask = np.abs(denom) > 1e-14
    gamma[mask] = (z0[mask] - z_terminal[mask]) / denom[mask]
    return gamma

def build_time_array(config: RunConfig):
    return np.arange(0.0, config.cycle_length_s, config.dt)

def synthetic_inflow_waveform(t, T):
    x = t / T
    return 5.0 + 2.2 * np.sin(2.0 * np.pi * x) + 0.8 * np.sin(4.0 * np.pi * x - 0.25) + 0.25 * np.sin(6.0 * np.pi * x + 0.3)

def pressure_from_impedance(t, q, z_fft):
    q_hat = np.fft.rfft(q)
    p_hat = z_fft * q_hat
    return np.fft.irfft(p_hat, n=len(t))

def compute_pressure_solution(tree, wall, blood, config, t=None, q=None):
    if t is None:
        t = build_time_array(config)
    if q is None:
        q = synthetic_inflow_waveform(t, config.cycle_length_s)
    fft_freqs = np.fft.rfftfreq(len(t), d=config.dt)
    z = structured_tree_spectrum(fft_freqs, tree, blood, wall)
    p = pressure_from_impedance(t, q, z) * DYN_PER_CM2_TO_MMHG
    return {'t': t, 'q': q, 'z': z, 'p': p, 'peak_pressure_mmhg': float(np.max(p)), 'min_pressure_mmhg': float(np.min(p)), 'mean_pressure_mmhg': float(np.mean(p)), 'pulse_pressure_mmhg': float(np.max(p) - np.min(p)), 'low_frequency_impedance': float(np.abs(z[1])) if len(z) > 1 else float(np.abs(z[0]))}

def make_tree_wall_from_values(values, base_tree: TreeParams, base_wall: WallLaw):
    tree = TreeParams(root_radius_cm=base_tree.root_radius_cm, r_min_cm=values.get('r_min_cm', base_tree.r_min_cm), length_to_radius=values.get('length_to_radius', base_tree.length_to_radius), asymmetry_g=values.get('asymmetry_g', base_tree.asymmetry_g), radius_exponent_j=base_tree.radius_exponent_j, z_leaf=base_tree.z_leaf, include_viscosity=base_tree.include_viscosity, max_generations=base_tree.max_generations)
    wall = WallLaw(k1=values.get('k1', base_wall.k1), k2=base_wall.k2, k3=base_wall.k3)
    return (tree, wall)
