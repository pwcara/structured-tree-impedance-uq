from dataclasses import dataclass
from pathlib import Path
SHOW_PLOTS = False
OUT_DIR = Path('structured_tree_modular_outputs')
FIG_DIR = OUT_DIR / 'figures'
DATA_DIR = OUT_DIR / 'data'
SV_DIR = OUT_DIR / 'simvascular_inputs'
for folder in [OUT_DIR, FIG_DIR, DATA_DIR, SV_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
DYN_PER_CM2_TO_MMHG = 1.0 / 1333.22

@dataclass
class BloodProps:
    rho: float = 1.06
    nu: float = 0.046

@dataclass
class WallLaw:
    k1: float = 20000000.0
    k2: float = -22.53
    k3: float = 865000.0

@dataclass
class TreeParams:
    root_radius_cm: float = 0.25
    r_min_cm: float = 0.02
    length_to_radius: float = 50.0
    asymmetry_g: float = 0.41
    radius_exponent_j: float = 2.76
    z_leaf: complex = 0.0 + 0j
    include_viscosity: bool = True
    max_generations: int = 16

@dataclass
class RunConfig:
    f_max_hz: float = 12.0
    n_freqs: int = 250
    dt: float = 0.004
    cycle_length_s: float = 1.0
    n_lhs: int = 200
    random_seed: int = 42
    rcr_fit_fmax_hz: float = 4.0
    oat_points: int = 21
PARAM_RANGES = {'r_min_cm': (0.01, 0.06), 'length_to_radius': (30.0, 80.0), 'asymmetry_g': (0.3, 0.55), 'k1': (14000000.0, 26000000.0)}

def default_objects():
    return (BloodProps(), WallLaw(), TreeParams(), RunConfig())
