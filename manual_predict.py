"""
Manual Cyberbullying Prediction Script
Enter text and get predictions from all trained models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dashboard.streamlit_app import CyberbullyingDetector

def main():
    # Initialize detector
    try:
        detector = CyberbullyingDetector()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    while True:
        try:
            text = input("Your text: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                break

            if not text:
                continue

            print(f"\n🔍 Analyzing: '{text}'")

            # Get Random Forest prediction (more accurate for positive texts)
            results = detector.predict_all_models(text)
            if 'Random Forest' in results:
                prediction = results['Random Forest']['prediction']
                if prediction == 'Not Cyberbullying':
                    print("prediction: Not cyberbullying")
                else:
                    print("prediction: cyberbullying")
            else:
                print("prediction: error")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()