# Project-level shim to re-export src.config for tools that expect `import config`
try:
    from src.config import *  # noqa: F401,F403
except Exception:
    # If src/config.py is unavailable, provide minimal defaults to avoid import errors
    from pathlib import Path
    BASE = Path(__file__).resolve().parent
    RAW_DATA_DIR = BASE / 'data'
    LEXICONS_DIR = RAW_DATA_DIR / 'lexicons'
    PROFANITY_LIST_FILE = 'profanity_list.txt'
    # keep a small placeholder for other config keys used in the project
    PROCESSED_DATA_DIR = BASE / 'data' / 'processed'
    SAVED_MODELS_DIR = BASE / 'models' / 'saved_models'
    VECTORIZERS_DIR = BASE / 'models' / 'vectorizers'
