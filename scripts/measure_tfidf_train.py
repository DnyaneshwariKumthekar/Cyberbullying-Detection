import time
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import src.config as config

print('Loading processed data...')
df = pd.read_pickle(Path('data') / 'processed' / 'processed_data.pkl')
X = df['clean_text']

X_train, X_test = train_test_split(X, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df.get('is_cyberbullying') if 'is_cyberbullying' in df.columns else None)
print('Training size:', len(X_train))

vec = TfidfVectorizer(
    max_features=config.TFIDF_MAX_FEATURES,
    ngram_range=config.TFIDF_NGRAM_RANGE,
    min_df=config.TFIDF_MIN_DF,
    max_df=config.TFIDF_MAX_DF,
    use_idf=config.TFIDF_USE_IDF,
    smooth_idf=config.TFIDF_SMOOTH_IDF,
    sublinear_tf=config.TFIDF_SUBLINEAR_TF,
)

start = time.time()
X_train_tfidf = vec.fit_transform(X_train)
end = time.time()
print('Done. Time:', end - start)
print('Shape:', X_train_tfidf.shape)
