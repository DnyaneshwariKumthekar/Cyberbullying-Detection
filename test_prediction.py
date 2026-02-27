"""
Test prediction with sentiment
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dashboard.streamlit_app import CyberbullyingDetector

def test_prediction():
    detector = CyberbullyingDetector()
    text = "This is a positive message"
    print(f"Testing: {text}")
    results = detector.predict_all_models(text)
    print("Results:", results)

if __name__ == '__main__':
    test_prediction()