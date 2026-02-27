"""
Automated Dataset  Script
Downloads cyberbullying datasets from Kaggle and other sources
"""

import sys
from pathlib import Path
import logging

# ensure src on path for config import
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
try:
    from config import RAW_DATA_DIR, LEXICONS_DIR, PROFANITY_LIST_FILE
except Exception:
    # fallback: use data/ under project root
    BASE = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = BASE / 'data'
    LEXICONS_DIR = RAW_DATA_DIR / 'lexicons'
    PROFANITY_LIST_FILE = 'profanity_list.txt'

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Optional import for Kaggle
import subprocess
KAGGLE_AVAILABLE = False
try:
    # Check if kaggle can be imported without warnings
    result = subprocess.run([sys.executable, '-c', 'import kaggle'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        import kaggle  # type: ignore
        KAGGLE_AVAILABLE = True
    else:
        logging.info("Kaggle import failed. Using sample data instead.")
except Exception as e:
    logging.info(f"Kaggle not available: {type(e).__name__}. Using sample data instead.")


def create_sample_dataset():
    """Create a synthetic cyberbullying / normal dataset for development."""
    logging.info("📝 Creating sample cyberbullying dataset...")

    cyberbullying_samples = [
        ("You're so stupid and ugly!", "cyberbullying"),
        ("Nobody likes you, just disappear", "cyberbullying"),
        ("Go kill yourself, loser", "cyberbullying"),
        ("You're too old for this, grandpa", "age"),
        ("Fat pig, you should stop eating", "cyberbullying"),
        ("Get out of our country", "ethnicity"),
        ("Your religion is a joke", "religion"),
        ("Women are so dumb", "gender"),
    ]

    normal_samples = [
        ("Have a great day!", "not_cyberbullying"),
        ("Thanks for your help!", "not_cyberbullying"),
        ("That's an interesting perspective", "not_cyberbullying"),
        ("I appreciate your work", "not_cyberbullying"),
    ]

    all_samples = []
    for text, label in (cyberbullying_samples * 500):
        all_samples.append((text, label))
    for text, label in (normal_samples * 500):
        all_samples.append((text, label))

    df = pd.DataFrame(all_samples, columns=['text', 'label'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DATA_DIR / 'cyberbullying_tweets.csv'
    df.to_csv(out, index=False)
    logging.info(f"✅ Created sample dataset: {out} ({len(df):,} rows)")
    return df


def create_hate_speech_dataset():
    logging.info("📝 Creating sample hate-speech dataset...")
    hate_samples = [
        ("I hate all people from that country", 0),
        ("They should all be eliminated", 0),
        ("Disgusting people, all of them", 0),
    ]
    neutral = [("What a beautiful day!", 2), ("I love this song", 2)]
    offensive = [("That's so stupid and dumb", 1), ("What a bunch of idiots", 1)]

    all_samples = (hate_samples * 500) + (offensive * 500) + (neutral * 500)
    df = pd.DataFrame(all_samples, columns=['text', 'class'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DATA_DIR / 'hate_speech_dataset.csv'
    df.to_csv(out, index=False)
    logging.info(f"✅ Created hate speech dataset: {out} ({len(df):,} rows)")
    return df


def create_profanity_list():
    logging.info("📝 Creating profanity list...")
    words = [
        'stupid', 'idiot', 'dumb', 'ugly', 'hate', 'kill', 'die', 'loser', 'fat', 'worthless'
    ]
    LEXICONS_DIR.mkdir(parents=True, exist_ok=True)
    out = LEXICONS_DIR / PROFANITY_LIST_FILE
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(set(words))))
    logging.info(f"✅ Profanity list saved: {out} ({len(words)} words)")
    return out


def download_from_kaggle():
    """Try to download datasets from Kaggle if kaggle package and credentials exist."""
    if not KAGGLE_AVAILABLE:
        logging.info("Kaggle API not available; skipping Kaggle download.")
        return False

    try:
        logging.info("📥 Attempting Kaggle download...")
        kaggle.api.dataset_download_files(
            'andrewmvd/cyberbullying-classification',
            path=str(RAW_DATA_DIR),
            unzip=True
        )
        logging.info("✅ Kaggle download finished")
        return True
    except Exception as e:
        logging.warning(f"Kaggle download failed: {e}")
        return False


def main():
    logging.info('='*60)
    logging.info('  CYBERBULLYING DETECTION - DATASET SETUP')
    logging.info('='*60)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not download_from_kaggle():
        logging.info('\nKaggle not used or failed — generating sample datasets')
        create_sample_dataset()
        create_hate_speech_dataset()

    create_profanity_list()

    logging.info('\n✅ Dataset setup completed')
    logging.info(f'   Datasets location: {RAW_DATA_DIR}')


if __name__ == '__main__':
    main()