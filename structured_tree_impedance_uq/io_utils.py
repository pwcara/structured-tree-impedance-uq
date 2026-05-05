import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
from config import SHOW_PLOTS

def finalize_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def save_json(obj: dict, path: Path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_csv_dicts(rows: list[dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def save_matrix_csv(path: Path, row_labels, col_labels, matrix):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + list(col_labels))
        for label, row in zip(row_labels, matrix):
            w.writerow([label] + list(row))

def save_spectrum_csv(path: Path, freqs_hz, named_spectra):
    import numpy as np
    headers = ['frequency_hz']
    cols = [freqs_hz]
    for name, z in named_spectra:
        headers += [f'{name}_real', f'{name}_imag', f'{name}_mag', f'{name}_phase_deg']
        cols += [np.real(z), np.imag(z), np.abs(z), np.angle(z, deg=True)]
    arr = np.column_stack(cols)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(arr)
