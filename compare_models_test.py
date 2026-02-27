import joblib
import sys
sys.path.append('src')
import config
from src.preprocessing import TextPreprocessor

# Load all models and vectorizer
models = {}
for model_name in ['naive_bayes', 'svm', 'logistic_regression', 'random_forest']:
    models[model_name] = joblib.load(config.SAVED_MODELS_DIR / f'{model_name}_model.pkl')

vectorizer = joblib.load(config.VECTORIZERS_DIR / 'tfidf_vectorizer.pkl')
preprocessor = TextPreprocessor()

# Test texts
test_texts = [
    'You are so stupid and ugly!',
    'Have a great day!',
    'Nobody likes you, just disappear',
    'I love this weather'
]

print('Model Comparison:')
print('=' * 80)
for text in test_texts:
    cleaned = preprocessor.clean_text(text)
    X = vectorizer.transform([cleaned])
    print(f'Input: "{text}"')
    print(f'Cleaned: "{cleaned}"')
    print()

    for model_name, model in models.items():
        pred = model.predict(X)[0]
        try:
            proba = model.predict_proba(X)[0]
            proba_str = f'[Not: {proba[0]:.3f}, Cyber: {proba[1]:.3f}]'
        except AttributeError:
            # SVM doesn't have predict_proba by default
            proba_str = '[Probabilities not available]'
        print(f'{model_name.upper()}: {pred} ({"Cyberbullying" if pred == 1 else "Not Cyberbullying"}) - {proba_str}')
    print('-' * 80)