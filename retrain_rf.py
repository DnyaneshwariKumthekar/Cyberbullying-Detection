"""
Retrain Random Forest model with updated hyperparameters
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

import logging
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def retrain_random_forest():
    """Retrain only the Random Forest model with updated hyperparameters"""

    # Load processed data
    logger.info("Loading processed data...")
    data_file = config.PROCESSED_DATA_DIR / config.PROCESSED_DATA_FILE
    if not data_file.exists():
        raise FileNotFoundError(f"Processed data not found at {data_file}")

    df = pd.read_pickle(data_file)
    logger.info(f"✓ Loaded {len(df):,} records")

    # Convert labels to binary
    def label_to_binary(v):
        s = str(v).lower().strip()
        if s.startswith('non') or s.startswith('not ') or s in ('0', 'false', 'no'):
            return 0
        if 'non_' in s or 'non ' in s:
            return 0
        if 'aggress' in s or 'cyber' in s or s in ('1', 'true', 'yes'):
            return 1
        return 0

    df['is_cyberbullying'] = df['label'].apply(label_to_binary)
    logger.info(f"✓ Target distribution:\n{df['is_cyberbullying'].value_counts().to_string()}")

    # Prepare data
    df_sample = df.sample(n=min(50000, len(df)), random_state=config.RANDOM_STATE)
    X = df_sample['clean_text']
    y = df_sample['is_cyberbullying']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")

    # Create TF-IDF features
    logger.info('Creating TF-IDF features...')
    vec = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        use_idf=config.TFIDF_USE_IDF,
        smooth_idf=config.TFIDF_SMOOTH_IDF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )

    X_train_tfidf = vec.fit_transform(X_train.astype(str))
    X_test_tfidf = vec.transform(X_test.astype(str))

    logger.info(f"✓ TF-IDF shape: {X_train_tfidf.shape}")

    # Train Random Forest with updated hyperparameters
    logger.info('Training Random Forest with updated hyperparameters...')
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        class_weight=config.RF_CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train_tfidf, y_train.to_numpy())
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test.to_numpy(), y_pred)
    logger.info(f"✓ Updated Random Forest Accuracy: {acc*100:.2f}%")

    # Save the updated model
    joblib.dump(model, config.SAVED_MODELS_DIR / 'random_forest_model.pkl')
    logger.info("✓ Updated Random Forest model saved")

    # Print detailed metrics
    logger.info("\nDetailed Classification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=['Not Cyberbullying', 'Cyberbullying']))

    return model, acc

if __name__ == '__main__':
    retrain_random_forest()