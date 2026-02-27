import joblib
import sys
sys.path.append('src')
import config

df = joblib.load(config.PROCESSED_DATA_DIR / config.PROCESSED_DATA_FILE)
print('Label distribution:')
print(df['label'].value_counts())
print()
print('First few examples:')
for i in range(5):
    text = df.iloc[i]['text'][:50] + '...' if len(df.iloc[i]['text']) > 50 else df.iloc[i]['text']
    print(f'{df.iloc[i]["label"]}: {text}')