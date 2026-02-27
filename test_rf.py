import joblib
import sys
from pathlib import Path
sys.path.append('src')
import config
from src.preprocessing import TextPreprocessor

# Load model and vectorizer
rf_model = joblib.load(config.SAVED_MODELS_DIR / 'random_forest_model.pkl')
vectorizer = joblib.load(config.VECTORIZERS_DIR / 'tfidf_vectorizer.pkl')
preprocessor = TextPreprocessor()

# Test texts
test_texts = [
    'You are so stupid and ugly!',
    'Have a great day!',
    'Nobody likes you, just disappear',
    'I love this weather'
]

print('Random Forest Predictions:')
print('=' * 50)
for text in test_texts:
    cleaned = preprocessor.clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = rf_model.predict(X)[0]
    proba = rf_model.predict_proba(X)[0]
    print(f'Input: "{text}"')
    print(f'Cleaned: "{cleaned}"')
    print(f'Prediction: {pred} ({"Cyberbullying" if pred == 1 else "Not Cyberbullying"})')
    print(f'Probabilities: [Not: {proba[0]:.3f}, Cyber: {proba[1]:.3f}]')
    print('-' * 30)