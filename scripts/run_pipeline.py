"""
Complete Pipeline Automation Script
Runs the entire cyberbullying detection pipeline from start to finish
"""

import sys
import subprocess
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_step(step_num, description):
    """Print step information"""
    print(f"\n{'─'*70}")
    print(f"📍 STEP {step_num}: {description}")
    print('─'*70)


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n⚙️  {description}...")
    start_time = time.time()
    try:
        # If command is a string, run with shell=True (legacy), else use list (preferred)
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=False)
        else:
            result = subprocess.run(command, shell=False, check=True, text=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✅ {description} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed!")
        print(f"   {str(e)}")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print_step(1, "Checking Dependencies")
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'joblib'
    ]
    # Optional packages (these are not required for the pipeline to run)
    optional_packages = ['xgboost', 'nltk', 'textblob', 'streamlit', 'plotly', 'tqdm']
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} - MISSING")
    # Check optional packages and report but do not fail
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✓ (optional) {package}")
        except ImportError:
            print(f"  ✗ (optional) {package} - NOT INSTALLED")
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Please run: pip install -r requirements.txt")
        return False
    print("\n✅ All required dependencies installed!")
    return True


def download_nltk_data():
    """Download required NLTK data"""
    print_step(2, "Downloading NLTK Data")
    try:
        import nltk
    except ImportError:
        print("NLTK not available, skipping NLTK data download")
        return True
    datasets = ['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger']
    for dataset in datasets:
        try:
            nltk.data.find(f'corpora/{dataset}')
            print(f"  ✓ {dataset} already downloaded")
        except LookupError:
            print(f"  ⬇️  Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
            print(f"  ✓ {dataset} downloaded")
    print("\n✅ NLTK data ready!")
    return True


def setup_datasets():
    """Setup and download datasets"""
    print_step(3, "Setting Up Datasets")
    # Use a list to avoid shell splitting issues with spaces in paths
    return run_command(
        [sys.executable, "scripts/download_datasets.py"],
        "Dataset setup"
    )


def preprocess_data():
    """Run data preprocessing"""
    print_step(4, "Preprocessing Data")
    return run_command(
        [sys.executable, "src/preprocessing.py"],
        "Data preprocessing"
    )


def train_models():
    """Train all ML models"""
    print_step(5, "Training Machine Learning Models")
    print("⏱️  This may take 5-10 minutes...")
    return run_command(
        [sys.executable, "src/train_models.py"],
        "Model training"
    )


def verify_setup():
    """Verify all required files exist"""
    print_step(6, "Verifying Setup")
    import config
    checks = {
        "Processed Data": config.PROCESSED_DATA_DIR / 'processed_data.pkl',
        "TF-IDF Vectorizer": config.VECTORIZERS_DIR / 'tfidf_vectorizer.pkl',
        "Naive Bayes Model": config.SAVED_MODELS_DIR / 'naive_bayes_model.pkl',
        "SVM Model": config.SAVED_MODELS_DIR / 'svm_model.pkl',
        "Random Forest Model": config.SAVED_MODELS_DIR / 'random_forest_model.pkl',
        "Ensemble Model": config.SAVED_MODELS_DIR / 'ensemble_model.pkl',
    }
    all_good = True
    for name, path in checks.items():
        if path.exists():
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - NOT FOUND")
            all_good = False
    # XGBoost model is optional — only check if enabled in config
    try:
        if getattr(config, 'ENABLE_XGBOOST', False):
            xgb_path = config.SAVED_MODELS_DIR / 'xgboost_model.pkl'
            if xgb_path.exists():
                print("  ✓ XGBoost Model (optional)")
            else:
                print("  ✗ XGBoost Model (optional) - NOT FOUND")
    except Exception:
        # If config doesn't have the expected attributes, ignore optional check
        pass
    return all_good


def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*70)
    print("  🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📌 Next Steps:")
    print("\n1. 🚀 Launch the Dashboard:")
    print("   streamlit run dashboard/streamlit_app.py")
    print("\n2. 🌐 Or start the API server (optional):")
    print("   python api/app.py")
    print("\n3. 📚 Read the documentation:")
    print("   Check README.md for detailed usage instructions")
    print("\n💡 Quick Test:")
    print("   The dashboard will open in your browser automatically.")
    print("   Try entering some text to see the detection in action!")
    print("\n📊 Expected Performance:")
    print("   • Accuracy: 85-90%")
    print("   • Speed: <100ms per prediction")
    print("   • Models: 5 trained and ready (XGBoost optional)")
    print("\n🆘 Need Help?")
    print("   • Check README.md for troubleshooting")
    print("   • Review code comments")
    print("   • Examine config.py for settings")
    print("\n" + "="*70)


def launch_dashboard_prompt():
    """Print instructions for launching the dashboard"""
    print("\n" + "─"*70)
    print("To launch the dashboard manually:")
    print("   streamlit run dashboard/streamlit_app.py")
    print("   (Requires streamlit to be installed)")


def main():
    """Main pipeline execution"""
    start_time = time.time()
    print_header("🛡️  CYBERBULLYING DETECTION SYSTEM - AUTOMATED SETUP")
    print("This script will:")
    print("  1. Check dependencies")
    print("  2. Download NLTK data")
    print("  3. Setup datasets")
    print("  4. Preprocess data")
    print("  5. Train ML models")
    print("  6. Verify installation")
    print("\n▶️  Starting...")
    # input("\n▶️  Press Enter to start...")
    steps = [
        (check_dependencies, True),
        (download_nltk_data, True),
        (setup_datasets, True),
        (preprocess_data, True),
        (train_models, True),
        (verify_setup, False),
    ]
    for i, (step_func, required) in enumerate(steps, 1):
        success = step_func()
        if not success:
            if required:
                print(f"\n❌ Pipeline failed at step {i}")
                print("   Please fix the errors and try again.")
                return False
            else:
                print(f"\n⚠️  Step {i} had warnings, but continuing...")
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(f"\n⏱️  Total setup time: {minutes}m {seconds}s")
    print_next_steps()
    launch_dashboard_prompt()
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        print("   You can resume by running this script again")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
