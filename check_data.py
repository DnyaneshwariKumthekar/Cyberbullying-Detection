import joblib
import sys
sys.path.append('src')
import config

data = joblib.load(config.PROCESSED_DATA_DIR / config.PROCESSED_DATA_FILE)
print('Keys in processed data:', list(data.keys()))
print('Type of data:', type(data))

if isinstance(data, dict):
    for k, v in data.items():
        print(f'{k}: {type(v)} - shape: {getattr(v, "shape", "no shape")}')
elif hasattr(data, 'shape'):
    print('Shape:', data.shape)
    print('Columns:', list(data.columns) if hasattr(data, 'columns') else 'No columns')