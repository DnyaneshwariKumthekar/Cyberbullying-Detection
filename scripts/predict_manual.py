"""
Manual Cyberbullying Prediction Script
Enter text and get predictions from all trained models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dashboard.streamlit_app import CyberbullyingDetector

def main():
    print("🛡️  Cyberbullying Detection - Manual Prediction")
    print("=" * 50)

    # Initialize detector
    try:
        detector = CyberbullyingDetector()
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    print("\n📝 Enter text to analyze (or 'quit' to exit):")

    while True:
        try:
            text = input("\nYour text: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not text:
                print("⚠️  Please enter some text.")
                continue

            print(f"\n🔍 Analyzing: '{text}'")
            print("-" * 40)

            # Debug: show cleaned text
            from src.preprocessing import TextPreprocessor
            preprocessor = TextPreprocessor()
            cleaned = preprocessor.clean_text(text)
            print(f"🧹 Cleaned text: '{cleaned}'")
            print()

            # Get predictions
            results = detector.predict_all_models(text)

            print("📊 Predictions:")
            for model_name, prediction in results.items():
                pred = prediction['prediction']
                conf = prediction['confidence']
                status = "🚨 CYBERBULLYING" if pred == "Cyberbullying" else "✅ NOT CYBERBULLYING"
                print(f"  {model_name}: {status} (confidence: {conf:.3f})")

            # Overall assessment using Ensemble model with confidence threshold
            ensemble_pred = results['Ensemble']['prediction']
            ensemble_conf = results['Ensemble']['confidence']
            
            CONFIDENCE_THRESHOLD = 0.7
            if ensemble_pred == "Cyberbullying" and ensemble_conf > CONFIDENCE_THRESHOLD:
                print("\n🎯 Overall: 🚨 CYBERBULLYING")
            else:
                print("\n🎯 Overall: ✅ NOT CYBERBULLYING")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()