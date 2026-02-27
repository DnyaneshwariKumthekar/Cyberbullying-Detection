import pandas as pd
from pathlib import Path
p = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'processed_data.pkl'
print('processed file exists:', p.exists())
if not p.exists():
    raise SystemExit('processed_data.pkl not found')
df = pd.read_pickle(p)
print('shape:', df.shape)
for col in ['is_cyberbullying', 'label']:
    if col in df.columns:
        print('\nColumn:', col)
        print(df[col].value_counts(dropna=False).head(50))
        print('unique values (up to 50):', df[col].unique()[:50])
print('\nSample rows (first 10):')
print(df.head(10).to_string(index=False))
