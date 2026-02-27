import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import src.config as config
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

N = 20000
print('Loading processed data...')
df = pd.read_pickle(Path('data') / 'processed' / 'processed_data.pkl')
X = df['clean_text']

X_train, _ = train_test_split(X, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
print('Train size full:', len(X_train))
X_sample = X_train.sample(n=min(N, len(X_train)), random_state=1)
print('Sample size:', len(X_sample))

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
X_train_tfidf = vec.fit_transform(X_sample)
end = time.time()
print('TF-IDF fit time (s) on sample:', end - start)
print('TF-IDF shape (sample):', X_train_tfidf.shape)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=config.RF_N_ESTIMATORS,
    max_depth=config.RF_MAX_DEPTH,
    min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
    min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
    class_weight=config.RF_CLASS_WEIGHT,
    random_state=config.RANDOM_STATE,
    n_jobs=-1,
)
start = time.time()
n = X_train_tfidf.shape[0]
# create alternating labels to allow realistic training
labels = [i % 2 for i in range(n)]
try:
    start = time.time()
    rf.fit(X_train_tfidf, labels)
    end = time.time()
    print('RandomForest fit time (s) on sample [alt labels]:', end - start)
except Exception as e:
    print('RandomForest failed to fit on sample (skipping):', e)

# LogisticRegression
lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1)
try:
    start = time.time()
    lr.fit(X_train_tfidf, labels)
    end = time.time()
    print('LogisticRegression fit time (s) on sample [alt labels]:', end - start)
except Exception as e:
    print('LogisticRegression failed to fit on sample (skipping):', e)

# XGBoost (optional)
try:
    import importlib
    xgb_mod = importlib.import_module('xgboost')
    XGBClassifier = getattr(xgb_mod, 'XGBClassifier', None)
    if XGBClassifier is not None:
        xgb = XGBClassifier(n_estimators=config.XGB_N_ESTIMATORS, n_jobs=-1, random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
        start = time.time()
        xgb.fit(X_train_tfidf, [0]*X_train_tfidf.shape[0], verbose=False)
        end = time.time()
        print('XGBoost fit time (s) on sample [dummy labels]:', end - start)
    else:
        print('XGBoost not available; skipping')
except Exception as e:
    print('XGBoost not available or failed to fit (skipping):', e)
