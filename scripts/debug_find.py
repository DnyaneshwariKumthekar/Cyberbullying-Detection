import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocessing import find_text_column
import pandas as pd

paths = [
    'data/3. Aggressive_All.csv',
    'data/4. Non_Aggressive_All.csv',
    'data/6. CB_Labels.csv'
]

for p in paths:
    print('\n---', p)
    df = pd.read_csv(p, low_memory=False)
    print('columns ->', df.columns.tolist())
    print('dtypes ->', df.dtypes.to_dict())
    # show string dtype check per column
    import pandas as _pd
    for c in df.columns:
        print('  is_string_dtype for', c, '->', _pd.api.types.is_string_dtype(df[c]))
    txt_col = find_text_column(df)
    print('find_text_column ->', txt_col)
    if txt_col is not None:
        print('sample text column head:', df[txt_col].head(3).tolist())
