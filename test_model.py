import sys
sys.path.append('.')

from dashboard.streamlit_app import CyberbullyingDetector

detector = CyberbullyingDetector()
print('Models loaded:', list(detector.models.keys()))
print('Vectorizers loaded:', list(detector.vectorizers.keys()))

# Test prediction
test_cases = [
    'you are stupid and ugly',
    'your songs are so good',
    'this is amazing work',
    'i hate you so much'
]

for text in test_cases:
    print(f'\nText: "{text}"')
    results = detector.predict_all_models(text)
    for model, result in results.items():
        print(f'  {model}: {result["prediction"]} (conf: {result["confidence"]:.2f})')