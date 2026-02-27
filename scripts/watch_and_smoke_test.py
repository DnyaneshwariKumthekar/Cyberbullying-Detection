"""Watch for model files and run a dashboard smoke test automatically when ready.

Usage:
    python scripts/watch_and_smoke_test.py

The script polls `models/saved_models` for the expected model files (excluding XGBoost).
When all are present, it imports `dashboard.streamlit_app.get_detector`, loads models, runs a couple
of sample predictions with `CyberbullyingDetector` and prints the results. Exits with code 0 on
success, non-zero otherwise.
"""
import time
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

SAVED = ROOT / 'models' / 'saved_models'
EXPECTED_MODELS = [
    'naive_bayes_model.pkl',
    'svm_model.pkl',
    'random_forest_model.pkl',
    'logistic_regression_model.pkl',
    'ensemble_model.pkl',
]
POLL_INTERVAL = 15  # seconds
TIMEOUT = 60 * 60  # 1 hour


def check_models():
    present = {p.name for p in SAVED.iterdir() if p.is_file()}
    missing = [m for m in EXPECTED_MODELS if m not in present]
    return missing


def ready_for_incremental_check():
    """Return (bool, present_models) indicating whether TF-IDF exists and at least one model is present."""
    present = {p.name for p in SAVED.iterdir() if p.is_file()}
    tfidf_path = ROOT / 'models' / 'vectorizers' / 'tfidf_vectorizer.pkl'
    has_tfidf = tfidf_path.exists()
    has_any_model = any(m in present for m in EXPECTED_MODELS)
    return has_tfidf and has_any_model, present


def run_smoke_test(minimal=False, present_models=None):
    """Run dashboard smoke test.

    If minimal=True, only test with a single available model (first found in present_models).
    """
    print('Running dashboard smoke test...' + (' (minimal)' if minimal else ''))
    try:
        import importlib
        app = importlib.import_module('dashboard.streamlit_app')

        detector = app.get_detector()
        print('  ✓ Detector loaded')

        sample_texts = [
            "You're an amazing person!",
            "I hate you and everything about you",
        ]

        if minimal and present_models:
            # pick the first available model
            available = [m for m in EXPECTED_MODELS if m in present_models]
            model_name = None
            if available:
                # map filename back to display name used by detector.models keys
                # filenames like 'naive_bayes_model.pkl' -> 'Naive Bayes'
                fname = available[0]
                model_name = {
                    'naive_bayes_model.pkl': 'Naive Bayes',
                    'svm_model.pkl': 'SVM',
                    'random_forest_model.pkl': 'Random Forest',
                    'logistic_regression_model.pkl': 'Logistic Regression',
                    'ensemble_model.pkl': 'Ensemble',
                }.get(fname, None)
            if model_name is None:
                print('No usable model found for minimal smoke test'); return False

            for t in sample_texts:
                pred, conf = detector.predict(t, model_name=model_name)
                print(f"  • Text: {t[:60]!r} -> prediction: {pred}, confidence: {conf:.3f}")

            return True

        # Full smoke test (all expected models should be present)
        full_sample_texts = [
            "You're an amazing person!",
            "I hate you and everything about you",
            "That was rude and unnecessary"
        ]

        for t in full_sample_texts:
            pred, conf = detector.predict(t, model_name='Ensemble' if 'Ensemble' in detector.models else list(detector.models.keys())[0])
            print(f"  • Text: {t[:60]!r} -> prediction: {pred}, confidence: {conf:.3f}")

        # run predict_all_models
        all_results = detector.predict_all_models('You are stupid')
        print('  ✓ All models produced predictions:')
        for k, v in all_results.items():
            print(f"    - {k}: {v['prediction']} ({v['confidence']*100:.1f}%)")

        return True
    except Exception as e:
        print('Smoke test FAILED')
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print(f'Watching for model files in {SAVED} (timeout {TIMEOUT}s)')
    start = time.time()
    incremental_run_done = False
    while True:
        missing = check_models()
        # Full set check
        if not missing:
            print('All expected models present; starting full smoke test')
            ok = run_smoke_test(minimal=False)
            sys.exit(0 if ok else 2)

        # Incremental check (TF-IDF + any model)
        ready, present = ready_for_incremental_check()
        if ready and not incremental_run_done:
            print('TF-IDF + at least one model found; starting incremental smoke test')
            ok = run_smoke_test(minimal=True, present_models=present)
            incremental_run_done = True
            if not ok:
                print('Incremental smoke test failed; continuing to wait for full set')

        elapsed = time.time() - start
        print(f"Models missing ({len(missing)}): {', '.join(missing)} — elapsed: {int(elapsed)}s")
        if elapsed > TIMEOUT:
            print('Timeout waiting for models; exiting with non-zero status')
            sys.exit(3)
        time.sleep(POLL_INTERVAL)
