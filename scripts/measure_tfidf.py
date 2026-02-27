import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import src.config as config

print('Loading processed data...')
df = pd.read_pickle(Path('data') / 'processed' / 'processed_data.pkl')
X = df['clean_text']

print('Creating vectorizer...')
vec = TfidfVectorizer(
    max_features=config.TFIDF_MAX_FEATURES,
    ngram_range=config.TFIDF_NGRAM_RANGE,
    min_df=config.TFIDF_MIN_DF,
    max_df=config.TFIDF_MAX_DF,
    use_idf=config.TFIDF_USE_IDF,
    smooth_idf=config.TFIDF_SMOOTH_IDF,
    sublinear_tf=config.TFIDF_SUBLINEAR_TF,
)

print('Starting fit_transform...')
start = time.time()
X_train_tfidf = vec.fit_transform(X)
end = time.time()
print('Done. Time:', end - start)
print('Shape:', X_train_tfidf.shape)
