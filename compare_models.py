import joblib
import sys
sys.path.append('src')
import config
from src.preprocessing import TextPreprocessor

# Test other models to see if they work better
models_to_test = ['naive_bayes', 'svm', 'logistic_regression', 'random_forest']
vectorizer = joblib.load(config.VECTORIZERS_DIR / 'tfidf_vectorizer.pkl')
preprocessor = TextPreprocessor()

test_texts = [
    'You are so stupid and ugly!',
    'Have a great day!',
    'Nobody likes you, just disappear',
    'I love this weather'
]

print('Comparing all models:')
print('=' * 60)

for text in test_texts:
    cleaned = preprocessor.clean_text(text)
    X = vectorizer.transform([cleaned])
    print(f'Input: "{text}"')
    print(f'Cleaned: "{cleaned}"')

    for model_name in models_to_test:
        try:
            model = joblib.load(config.SAVED_MODELS_DIR / f'{model_name}_model.pkl')
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
            result = 'Cyberbullying' if pred == 1 else 'Not Cyberbullying'
            conf = proba[pred] if len(proba) > pred else 0.5
            print(f'  {model_name}: {result} ({conf:.3f})')
        except Exception as e:
            print(f'  {model_name}: Error - {e}')

    print('-' * 40)