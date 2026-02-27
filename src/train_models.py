"""Clean training module for cyberbullying detection.

Provides `CyberbullyingModelTrainer` and a runnable `main()`.
XGBoost is optional; if unavailable, training will skip it with a clear warning.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing import get_sentiment
# XGBoost is intentionally skipped by default; enable by setting `ENABLE_XGBOOST = True` in config.py
XGBClassifier = None  # not imported at module import time


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


from typing import Optional

class CyberbullyingModelTrainer:
    """Train and evaluate multiple models for cyberbullying detection."""

    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        # Explicit types for static checkers
        self.X_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.Series] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        Path(config.SAVED_MODELS_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.VECTORIZERS_DIR).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        logger.info(" Loading preprocessed data...")
        data_file = config.PROCESSED_DATA_DIR / config.PROCESSED_DATA_FILE
        if not data_file.exists():
            raise FileNotFoundError(f"Preprocessed data not found at {data_file}. Run preprocessing first.")

        df = pd.read_pickle(data_file)
        logger.info(f"  ✓ Loaded {len(df):,} records")

        if 'clean_text' not in df.columns and 'cleaned_text' in df.columns:
            df['clean_text'] = df['cleaned_text']

        if 'clean_text' not in df.columns:
            raise ValueError('Processed dataframe must contain `clean_text` column')

        if 'is_cyberbullying' not in df.columns:
            if 'label' in df.columns:
                def label_to_binary(v):
                    s = str(v).lower().strip()
                    # explicit negatives first (e.g. 'non_aggressive', 'not abusive', 'no')
                    if s.startswith('non') or s.startswith('not ') or s in ('0', 'false', 'no'):
                        return 0
                    if 'non_' in s or 'non ' in s:
                        return 0
                    # positive indicators
                    if 'aggress' in s or 'cyber' in s or s in ('1', 'true', 'yes'):
                        return 1
                    # fallback: treat unknown as non-cyberbullying
                    return 0

                df['is_cyberbullying'] = df['label'].apply(label_to_binary)
                logger.info(f"  ✓ Derived target distribution:\n{df['is_cyberbullying'].value_counts().to_string()}")
            else:
                raise ValueError('Processed dataframe must contain `label` column to derive target')

        return df

    def prepare_data(self, df):
        logger.info('\n Preparing train/test split...')
        # Take a larger sample for better training
        df = df.sample(n=min(50000, len(df)), random_state=config.RANDOM_STATE)
        X = df['clean_text']
        y = df['is_cyberbullying']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )

        # Defensive checks for static analyzers and runtime safety
        assert self.X_train is not None and self.X_test is not None, 'Unexpected empty train/test split'
        logger.info(f"  Training set: {len(self.X_train):,} samples")
        logger.info(f"  Test set: {len(self.X_test):,} samples")

    def create_tfidf_features(self):
        logger.info('\n Creating TF-IDF features...')
        vec = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            use_idf=config.TFIDF_USE_IDF,
            smooth_idf=config.TFIDF_SMOOTH_IDF,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        )

        # Defensive checks for static analyzers and runtime
        assert self.X_train is not None and self.X_test is not None, 'Train/test splits not prepared'
        X_train_tfidf = vec.fit_transform(self.X_train.astype(str))
        logger.info('  ✓ TF-IDF fit completed')
        X_test_tfidf = vec.transform(self.X_test.astype(str))

        # Add sentiment features
        # sentiment_train = self.X_train.apply(get_sentiment).values.reshape(-1, 1)
        # X_train_combined = hstack([X_train_tfidf, sentiment_train])
        # sentiment_test = self.X_test.apply(get_sentiment).values.reshape(-1, 1)
        # X_test_combined = hstack([X_test_tfidf, sentiment_test])

        X_train_combined = X_train_tfidf
        X_test_combined = X_test_tfidf

        self.vectorizers['tfidf'] = vec
        joblib.dump(vec, config.VECTORIZERS_DIR / 'tfidf_vectorizer.pkl')

        logger.info(f"  ✓ TF-IDF shape: {X_train_tfidf.shape}")
        logger.info(f"  ✓ Combined features shape: {X_train_combined.shape}")
        return X_train_combined, X_test_combined

    def train_naive_bayes(self, X_train, X_test):
        logger.info('\n Training Naive Bayes...')
        model = MultinomialNB(alpha=config.NB_ALPHA)
        assert self.y_train is not None and self.y_test is not None, 'Targets not prepared'
        model.fit(X_train, self.y_train.to_numpy())
        y_pred = model.predict(X_test)
        acc = accuracy_score(self.y_test.to_numpy(), y_pred)
        logger.info(f"  ✓ Accuracy: {acc*100:.2f}%")
        self.models['naive_bayes'] = model
        joblib.dump(model, config.SAVED_MODELS_DIR / 'naive_bayes_model.pkl')
        return model, y_pred

    def train_svm(self, X_train, X_test):
        logger.info('\n Training SVM...')
        model = LinearSVC(C=config.SVM_C, max_iter=config.SVM_MAX_ITER, random_state=config.RANDOM_STATE)
        assert self.y_train is not None and self.y_test is not None, 'Targets not prepared'
        model.fit(X_train, self.y_train.to_numpy())
        y_pred = model.predict(X_test)
        acc = accuracy_score(self.y_test.to_numpy(), y_pred)
        logger.info(f"  ✓ Accuracy: {acc*100:.2f}%")
        self.models['svm'] = model
        joblib.dump(model, config.SAVED_MODELS_DIR / 'svm_model.pkl')
        return model, y_pred

    def train_random_forest(self, X_train, X_test):
        logger.info('\n Training Random Forest...')
        model = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            max_features=config.RF_MAX_FEATURES,
            class_weight=config.RF_CLASS_WEIGHT,
            bootstrap=config.RF_BOOTSTRAP,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
        )
        assert self.y_train is not None and self.y_test is not None, 'Targets not prepared'
        model.fit(X_train, self.y_train.to_numpy())
        y_pred = model.predict(X_test)
        acc = accuracy_score(self.y_test.to_numpy(), y_pred)
        logger.info(f"  ✓ Accuracy: {acc*100:.2f}%")
        self.models['random_forest'] = model
        joblib.dump(model, config.SAVED_MODELS_DIR / 'random_forest_model.pkl')
        return model, y_pred

    def train_xgboost(self, X_train, X_test):
        logger.info('\n Training XGBoost...')
        # Training XGBoost is disabled by default. Enable via config.ENABLE_XGBOOST = True
        if not getattr(config, 'ENABLE_XGBOOST', False):
            raise RuntimeError('XGBoost training is disabled in configuration.')

        if XGBClassifier is None:
            # If someone enables XGBoost but it's not installed, provide clear error
            raise RuntimeError('XGBoost not installed. Install with `pip install xgboost` to enable this model.')

        pos = (self.y_train == 1).sum()
        neg = (self.y_train == 0).sum()
        scale_pos_weight = neg / max(1, pos)

        model = XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
        )
        assert self.y_train is not None and self.y_test is not None, 'Targets not prepared'
        model.fit(X_train, self.y_train.to_numpy(), verbose=False)
        y_pred = model.predict(X_test)
        acc = accuracy_score(self.y_test.to_numpy(), y_pred)
        logger.info(f"  ✓ Accuracy: {acc*100:.2f}%")
        self.models['xgboost'] = model
        joblib.dump(model, config.SAVED_MODELS_DIR / 'xgboost_model.pkl')
        return model, y_pred

    def train_logistic_regression(self, X_train, X_test):
        logger.info('\n Training Logistic Regression...')
        model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, class_weight='balanced')
        assert self.y_train is not None and self.y_test is not None, 'Targets not prepared'
        model.fit(X_train, self.y_train.to_numpy())
        y_pred = model.predict(X_test)
        acc = accuracy_score(self.y_test.to_numpy(), y_pred)
        logger.info(f"  ✓ Accuracy: {acc*100:.2f}%")
        self.models['logistic_regression'] = model
        joblib.dump(model, config.SAVED_MODELS_DIR / 'logistic_regression_model.pkl')
        return model, y_pred

    def train_ensemble(self, X_train, X_test):
        logger.info('\n Training Ensemble Model...')
        ensemble = VotingClassifier(
            estimators=[
                ('nb', MultinomialNB(alpha=config.NB_ALPHA)),
                ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
                # ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=config.RANDOM_STATE)),
            ],
            voting='soft' if config.ENSEMBLE_VOTING == 'soft' else 'hard',
            weights=config.ENSEMBLE_WEIGHTS,
            n_jobs=-1,
        )
        assert self.y_train is not None and self.y_test is not None, 'Targets not prepared'
        ensemble.fit(X_train, self.y_train.to_numpy())
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(self.y_test.to_numpy(), y_pred)
        logger.info(f"  ✓ Accuracy: {acc*100:.2f}%")
        self.models['ensemble'] = ensemble
        joblib.dump(ensemble, config.SAVED_MODELS_DIR / 'ensemble_model.pkl')
        return ensemble, y_pred

    def print_evaluation_report(self, model_name, y_pred):
        logger.info(f"\n Evaluation Report: {model_name}")
        logger.info('=' * 60)
        assert self.y_test is not None, 'No test labels available'

        y_true = self.y_test.to_numpy()
        labels = np.unique(y_true)
        # Build target names dynamically to match labels
        target_names = [ 'Cyberbullying' if int(l) == 1 else 'Not Cyberbullying' for l in labels ]
        if labels.size == 1:
            logger.warning(f"Only a single class ({int(labels[0])}) present in test labels; adapting report to avoid errors.")

        # Use labels and matching target_names to avoid classification_report errors when only one class is present
        report = classification_report(y_true, y_pred, labels=labels.tolist(), target_names=target_names)
        logger.info('\n' + str(report))

        cm = confusion_matrix(y_true, y_pred, labels=labels.tolist())
        logger.info('Confusion Matrix:')
        # If binary 2x2 matrix, print as TN/FP/FN/TP; otherwise handle single-class cases
        if cm.shape == (2, 2):
            logger.info(f"  TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}, TP: {cm[1][1]}")
        else:
            if labels.size == 1 and int(labels[0]) == 0:
                logger.info(f"  TN: {cm[0][0]}, FP: 0, FN: 0, TP: 0")
            elif labels.size == 1 and int(labels[0]) == 1:
                logger.info(f"  TN: 0, FP: 0, FN: 0, TP: {cm[0][0]}")
            else:
                logger.info(f"  Confusion matrix:\n{cm}")


def main():
    logger.info('=' * 60)
    logger.info('  CYBERBULLYING DETECTION - MODEL TRAINING')
    logger.info('=' * 60)

    trainer = CyberbullyingModelTrainer()
    df = trainer.load_data()
    trainer.prepare_data(df)
    X_train_tfidf, X_test_tfidf = trainer.create_tfidf_features()

    results = {}

    nb_model, nb_pred = trainer.train_naive_bayes(X_train_tfidf, X_test_tfidf)
    trainer.print_evaluation_report('Naive Bayes', nb_pred)

    svm_model, svm_pred = trainer.train_svm(X_train_tfidf, X_test_tfidf)
    trainer.print_evaluation_report('SVM', svm_pred)

    lr_model, lr_pred = trainer.train_logistic_regression(X_train_tfidf, X_test_tfidf)
    trainer.print_evaluation_report('Logistic Regression', lr_pred)

    rf_model, rf_pred = trainer.train_random_forest(X_train_tfidf, X_test_tfidf)
    trainer.print_evaluation_report('Random Forest', rf_pred)

    # Ensure y_test exists before computing summary metrics
    assert trainer.y_test is not None, 'No test labels to compute summary metrics'
    results['Naive Bayes'] = accuracy_score(trainer.y_test.to_numpy(), nb_pred)
    results['SVM'] = accuracy_score(trainer.y_test.to_numpy(), svm_pred)
    results['Logistic Regression'] = accuracy_score(trainer.y_test.to_numpy(), lr_pred)
    results['Random Forest'] = accuracy_score(trainer.y_test.to_numpy(), rf_pred)

    # XGBoost training is optional; only run if enabled in config
    if getattr(config, 'ENABLE_XGBOOST', False):
        try:
            xgb_model, xgb_pred = trainer.train_xgboost(X_train_tfidf, X_test_tfidf)
            trainer.print_evaluation_report('XGBoost', xgb_pred)            # y_test is asserted earlier            results['XGBoost'] = accuracy_score(trainer.y_test, xgb_pred)
        except RuntimeError as e:
            logger.warning(f"XGBoost skipped: {e}")
    else:
        logger.info('Skipping XGBoost training (disabled in project configuration).')

    ensemble_model, ensemble_pred = trainer.train_ensemble(X_train_tfidf, X_test_tfidf)
    trainer.print_evaluation_report('Ensemble', ensemble_pred)
    results['Ensemble'] = accuracy_score(trainer.y_test.to_numpy(), ensemble_pred)

    logger.info('\n' + '=' * 60)
    logger.info('  MODEL COMPARISON SUMMARY')
    logger.info('=' * 60)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{name:<25} {acc*100:6.2f}%")

    logger.info('\n Training completed')
    logger.info(f"Models saved to: {config.SAVED_MODELS_DIR}")
    logger.info(f"Vectorizers saved to: {config.VECTORIZERS_DIR}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception('Unhandled exception during training')
        import traceback
        traceback.print_exc()
        raise