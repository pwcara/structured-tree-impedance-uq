from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt
from config import default_objects, FIG_DIR, DATA_DIR, TreeParams
from io_utils import finalize_figure, save_csv_dicts, save_json
from core_impedance import structured_tree_spectrum
blood, wall, tree, config = default_objects()
freqs = np.linspace(0, config.f_max_hz, config.n_freqs)
depths = [8, 10, 12, 14, 16, 18, 20]
spectra = []
for depth in depths:
    tdict = asdict(tree)
    tdict['max_generations'] = depth
    tree_d = TreeParams(**tdict)
    z = structured_tree_spectrum(freqs, tree_d, blood, wall)
    spectra.append(z)
z_ref = spectra[-1]
errors = []
rows = []
for depth, z in zip(depths, spectra):
    err = np.linalg.norm(abs(z) - abs(z_ref)) / max(np.linalg.norm(abs(z_ref)), 1e-14)
    errors.append(float(err))
    rows.append({'max_generations': depth, 'relative_error': float(err)})
save_csv_dicts(rows, DATA_DIR / 'convergence_generations.csv')
save_json({'max_generations': depths, 'relative_errors': errors}, DATA_DIR / 'convergence_summary.json')
plt.figure(figsize=(7, 4.5))
plt.semilogy(depths, errors, marker='o', linewidth=2)
plt.xlabel('Maximum generations')
plt.ylabel('Relative impedance-modulus error')
plt.title('Structured-tree convergence with generation depth')
plt.grid(alpha=0.3)
finalize_figure(FIG_DIR / '06_convergence_generations.png')
print('Convergence study done.')
