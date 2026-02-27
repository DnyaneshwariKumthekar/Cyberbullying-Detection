import pandas as pd

df = pd.read_pickle('data/processed/processed_data.pkl')

print('Aggressive examples:')
aggressive = df[df['label'] == 'aggressive'].head(3)
for idx, row in aggressive.iterrows():
    print(f'  "{row["text"][:100]}..."')

print()
print('Non-aggressive examples:')
non_aggressive = df[df['label'] == 'non_aggressive'].head(3)
for idx, row in non_aggressive.iterrows():
    print(f'  "{row["text"][:100]}..."')