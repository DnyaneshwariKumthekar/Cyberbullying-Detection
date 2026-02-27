from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / 'models'
SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'
VECTORIZERS_DIR = MODELS_DIR / 'vectorizers'
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
VECTORIZERS_DIR.mkdir(parents=True, exist_ok=True)

# Processed data file
PROCESSED_DATA_FILE = 'processed_data.pkl'

# Train/test split
TRAIN_SIZE = 0.7
TEST_SIZE = 0.3
RANDOM_STATE = 42

# TF-IDF settings
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.95
TFIDF_USE_IDF = True
TFIDF_SMOOTH_IDF = True
TFIDF_SUBLINEAR_TF = True

# BoW settings
BOW_MAX_FEATURES = 10000
BOW_NGRAM_RANGE = (1, 2)
BOW_MIN_DF = 3
BOW_MAX_DF = 0.95

# Naive Bayes
NB_ALPHA = 1.0

# SVM
SVM_C = 1.0
SVM_MAX_ITER = 10000

# Random Forest - Optimized for text classification with aggressive regularization
RF_N_ESTIMATORS = 50  # Reduced from 100
RF_MAX_DEPTH = 10     # Reduced from 20
RF_MIN_SAMPLES_SPLIT = 50  # Increased from 20
RF_MIN_SAMPLES_LEAF = 25   # Increased from 10
RF_MAX_FEATURES = 'sqrt'  # Use sqrt(n_features) features
RF_CLASS_WEIGHT = 'balanced'
RF_BOOTSTRAP = True
RF_RANDOM_STATE = 42

# XGBoost
# Disable XGBoost by default; set to True to enable if xgboost is installed
ENABLE_XGBOOST = False
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

# Ensemble
ENSEMBLE_VOTING = 'soft'
ENSEMBLE_WEIGHTS = [1, 2]

# Other
CLASSIFICATION_THRESHOLD = 0.5

# Severity thresholds
SEVERITY_THRESHOLDS = {
    'mild': 0.6,
    'moderate': 0.8,
    'severe': 0.95
}

# Lexicons / raw data
RAW_DATA_DIR = DATA_DIR
LEXICONS_DIR = DATA_DIR / 'lexicons'
LEXICONS_DIR.mkdir(parents=True, exist_ok=True)
PROFANITY_LIST_FILE = 'profanity_list.txt'
