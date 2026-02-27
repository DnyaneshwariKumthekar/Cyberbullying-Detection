"""
Sanity check for project CSV data files.

Run this to print columns, dtypes, missing-value counts, label candidates,
and a few sample rows from each CSV in `data/`.

Usage:
    python scripts/sanity_check.py
"""
from pathlib import Path
import pandas as pd
import json


DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
CSV_FILES = [
    'Aggressive_All.csv',
    'Non_Aggressive_All.csv',
    'CB_Labels.csv',
    'users_data.csv',
    'peerness_values.csv',
    'Communication_Data_Among_Users.csv',
]


def inspect_file(p: Path):
    print('\n' + '='*60)
    print(f'File: {p.name}')
    if not p.exists():
        print('  NOT FOUND')
        return

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        print('  Error reading CSV:', e)
        return

    print(f'  Rows: {len(df):,}  Columns: {len(df.columns)}')
    print('  Columns:')
    for c in df.columns:
        dtype = str(df[c].dtype)
        non_null = df[c].notna().sum()
        print(f'    - {c}  ({dtype})  non-null: {non_null:,}')

    # Show candidate text/label columns
    text_candidates = [c for c in df.columns if c.lower() in ('text','tweet','tweet_text','message','content','post')]
    label_candidates = [c for c in df.columns if c.lower() in ('label','class','cb_label','annotation','target')]
    if text_candidates:
        print('  Text candidates:', text_candidates)
    if label_candidates:
        print('  Label candidates:', label_candidates)

    # Show top values for any small cardinality columns (up to 10)
    for c in df.columns:
        try:
            nunique = df[c].nunique(dropna=True)
        except Exception:
            nunique = None
        if nunique is not None and nunique <= 20:
            print(f'    Sample value counts for {c}:')
            print(df[c].value_counts(dropna=False).head(10).to_dict())

    print('\n  Sample rows:')
    print(df.head(3).to_dict(orient='records'))


def main():
    print('\nSanity check - data CSVs in project')
    print('='*60)
    for fname in CSV_FILES:
        # Allow files that are prefixed (e.g., "1. users_data.csv") by matching suffix
        candidates = list(DATA_DIR.glob(f'*{fname}'))
        if candidates:
            inspect_file(candidates[0])
        else:
            inspect_file(DATA_DIR / fname)

    print('\nSummary saved: none (printed to stdout)')


if __name__ == '__main__':
    main()
