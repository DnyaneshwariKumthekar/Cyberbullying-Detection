import pandas as pd
import joblib
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
sys.path.append('src')
import config
from src.preprocessing import TextPreprocessor

def retrain_random_forest():
    print("Retraining Random Forest with aggressive regularization...")

    # Load processed data
    processed_data_path = config.PROCESSED_DATA_DIR / config.PROCESSED_DATA_FILE
    if not processed_data_path.exists():
        print(f"Processed data not found at {processed_data_path}")
        return

    df = joblib.load(processed_data_path)

    # Load or create vectorizer
    vectorizer_path = config.VECTORIZERS_DIR / 'tfidf_vectorizer.pkl'
    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
        print("Loaded existing vectorizer")
    else:
        print("Creating new vectorizer")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            use_idf=config.TFIDF_USE_IDF,
            smooth_idf=config.TFIDF_SMOOTH_IDF,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF
        )

    # Vectorize the clean text
    X = vectorizer.fit_transform(df['clean_text'])

    # Convert labels to binary like the original training
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

    y = df['label'].apply(label_to_binary)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels distribution: {y_train.value_counts()}")
    print(f"Test labels distribution: {y_test.value_counts()}")

    # Create and train model with aggressive regularization
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,      # 50
        max_depth=config.RF_MAX_DEPTH,            # 10
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,  # 50
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,    # 25
        max_features=config.RF_MAX_FEATURES,      # 'sqrt'
        class_weight=config.RF_CLASS_WEIGHT,      # 'balanced'
        bootstrap=config.RF_BOOTSTRAP,            # True
        random_state=config.RF_RANDOM_STATE,      # 42
        n_jobs=-1,
    )

    print(f"\nTraining with parameters:")
    print(f"n_estimators: {config.RF_N_ESTIMATORS}")
    print(f"max_depth: {config.RF_MAX_DEPTH}")
    print(f"min_samples_split: {config.RF_MIN_SAMPLES_SPLIT}")
    print(f"min_samples_leaf: {config.RF_MIN_SAMPLES_LEAF}")
    print(f"max_features: {config.RF_MAX_FEATURES}")
    print(f"class_weight: {config.RF_CLASS_WEIGHT}")

    model.fit(X_train, y_train.to_numpy())

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test.to_numpy(), y_pred)

    print(f"\nRandom Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test.to_numpy(), y_pred, target_names=['Not Cyberbullying', 'Cyberbullying']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test.to_numpy(), y_pred)
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")

    # Save model
    model_path = config.SAVED_MODELS_DIR / 'random_forest_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    return accuracy

if __name__ == "__main__":
    retrain_random_forest()