import time
from pathlib import Path
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))
import src.config as config
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def label_to_binary(label):
    return 1 if label == 'aggressive' else 0

print('Loading processed data...')
df = pd.read_pickle(Path('data') / 'processed' / 'processed_data.pkl')
X = df['clean_text']
y = df['label'].apply(label_to_binary)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=config.RANDOM_STATE)
print('Train size:', len(X_train))

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
print('TF-IDF fit time (s):', end - start)
print('TF-IDF shape:', X_train_tfidf.shape)

# Use a small subset for model timing to speed up
small_size = min(100, X_train_tfidf.shape[0])
# Take random sample to ensure balanced classes
np.random.seed(42)
sample_indices = np.random.choice(len(y_train), size=small_size, replace=False)
X_train_tfidf_small = X_train_tfidf[sample_indices]
y_train_small = y_train.iloc[sample_indices]

# Random Forest
rf = RandomForestClassifier(
    n_estimators=10,  # reduced for faster timing
    max_depth=config.RF_MAX_DEPTH,
    min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
    min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
    class_weight=config.RF_CLASS_WEIGHT,
    random_state=config.RANDOM_STATE,
    n_jobs=1,  # single thread to avoid issues
)
start = time.time()
rf.fit(X_train_tfidf_small, y_train_small)  # real labels for accurate timing
end = time.time()
print('RandomForest fit time (s) [real labels, small subset]:', end - start)

# Measure prediction time
start = time.time()
rf.predict(X_train_tfidf_small[:10])  # predict on 10 samples
end = time.time()
print('RandomForest predict time (s) [10 samples]:', end - start)

# LogisticRegression
lr = LogisticRegression(max_iter=100, random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=1)
start = time.time()
lr.fit(X_train_tfidf_small, y_train_small)
end = time.time()
print('LogisticRegression fit time (s) [real labels, small subset]:', end - start)

# Measure prediction time
start = time.time()
lr.predict(X_train_tfidf_small[:10])  # predict on 10 samples
end = time.time()
print('LogisticRegression predict time (s) [50 samples]:', end - start)

# XGBoost (optional)
try:
    import importlib
    xgb_mod = importlib.import_module('xgboost')
    XGBClassifier = getattr(xgb_mod, 'XGBClassifier', None)
    if XGBClassifier is not None:
        xgb = XGBClassifier(n_estimators=10, n_jobs=1, random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
        start = time.time()
        xgb.fit(X_train_tfidf_small, y_train_small, verbose=False)
        end = time.time()
        print('XGBoost fit time (s) [real labels, small subset]:', end - start)

        # Measure prediction time
        start = time.time()
        xgb.predict(X_train_tfidf_small[:10])  # predict on 10 samples
        end = time.time()
        print('XGBoost predict time (s) [50 samples]:', end - start)
    else:
        print('XGBoost not available; skipping')
except Exception as e:
    print('XGBoost not available or failed to fit (skipping):', e)

print('Timing test completed successfully!')
