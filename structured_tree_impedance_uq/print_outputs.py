from pathlib import Path
import json
import pandas as pd
base = Path('structured_tree_modular_outputs')
print('All output files:')
for file in sorted(base.rglob('*')):
    if file.is_file():
        print(file)
print('\nJSON summaries:')
for file in sorted(base.rglob('*.json')):
    print('\n' + '=' * 80)
    print(file)
    print('=' * 80)
    with open(file) as f:
        print(json.dumps(json.load(f), indent=2))
print('\nCSV previews:')
for file in sorted(base.rglob('*.csv')):
    print('\n' + '=' * 80)
    print(file)
    print('=' * 80)
    try:
        print(pd.read_csv(file).head(10))
    except Exception as e:
        print('Could not read:', e)
try:
    from IPython.display import Image, display
    print('\nDisplaying PNG figures:')
    for file in sorted(base.rglob('*.png')):
        print('\n' + file.name)
        display(Image(str(file)))
except Exception:
    print('\nNot in IPython/Colab display environment; figures are saved in output folders.')
