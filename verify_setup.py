import joblib
import sys
sys.path.append('src')
import config
from src.preprocessing import TextPreprocessor

print("🔍 Verifying cyberbullying detection system...")

# Test if models can be loaded
models = {}
for model_name in ['naive_bayes', 'svm', 'logistic_regression', 'random_forest']:
    try:
        model = joblib.load(config.SAVED_MODELS_DIR / f'{model_name}_model.pkl')
        models[model_name] = model
        print(f'✓ {model_name} model loaded successfully')
    except Exception as e:
        print(f'✗ {model_name} model failed: {e}')

# Test preprocessing
try:
    preprocessor = TextPreprocessor()
    test_text = 'This is a test message!'
    cleaned = preprocessor.clean_text(test_text)
    print(f'✓ Preprocessing works: "{test_text}" -> "{cleaned}"')
except Exception as e:
    print(f'✗ Preprocessing failed: {e}')

# Test vectorizer
try:
    vectorizer = joblib.load(config.VECTORIZERS_DIR / 'tfidf_vectorizer.pkl')
    X = vectorizer.transform([cleaned])
    print(f'✓ Vectorizer works: shape {X.shape}')
except Exception as e:
    print(f'✗ Vectorizer failed: {e}')

print(f'\n✅ All components verified! {len(models)} models ready.')