"""Lightweight smoke test for the dashboard module.

This imports `dashboard.streamlit_app` (without starting Streamlit), calls `get_detector()`
and runs a few sample predictions. Exit code 0 on success (module imported and method callable),
non-zero on failure.
"""
import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path so package imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

try:
    import importlib
    app = importlib.import_module('dashboard.streamlit_app')

    detector = app.get_detector()
    print('  ✓ Detector factory returned')

    sample_texts = [
        "You're an amazing person!",
        "I hate you and everything about you",
    ]

    for t in sample_texts:
        try:
            # Attempt predict_all_models (fallbacks ensure function exists)
            all_results = detector.predict_all_models(t)
            print(f"  • Prediction for: {t[:40]!r} -> {all_results}")
        except Exception as e:
            print('  ✗ Prediction call failed')
            traceback.print_exc()
            sys.exit(2)

    print('\nSmoke test PASSED')
    sys.exit(0)
except Exception as e:
    print('Smoke test FAILED to import or initialize dashboard')
    traceback.print_exc()
    sys.exit(3)
