"""
Preprocessing for Cyberbullying Detection

This script reads available CSVs from the project's `data/` folder,
merges labeled text sources, applies text cleaning and normalization,
and writes a processed DataFrame to `data/processed/processed_data.pkl` and CSV.

Usage:
    python src/preprocessing.py

The script is defensive: if some data files are missing it will work
with whatever labeled text it can find and print clear instructions.
"""
from pathlib import Path
import re
import sys
import logging
import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:
    nltk = None

logging.basicConfig(level=logging.INFO, format="%(message)s")
HERE = Path(__file__).resolve().parent.parent
DATA_DIR = HERE / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_nltk():
    global nltk
    if nltk is None:
        try:
            import nltk as _n
            nltk = _n
        except Exception:
            logging.warning('NLTK not available; continuing without lemmatization/stopwords')
            return None, None

    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        logging.info('Downloading NLTK stopwords/wordnet (only first run)')
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer


def find_text_column(df: pd.DataFrame):
    candidates = ['text', 'tweet', 'tweet_text', 'message', 'content', 'post']
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: pick first column with textual content (contains alphabetic characters)
    for c in df.columns:
        try:
            sample = df[c].dropna().astype(str).head(100)
            if sample.str.contains('[A-Za-z]', regex=True).any():
                return c
        except Exception:
            continue
    return None


def clean_text(s: str, stop_words=None, lemmatizer=None):
    if not isinstance(s, str):
        return ''
    s = s.strip()
    # normalize spaces and basic unicode
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('\u200b', '')
    # lower
    s = s.lower()
    # remove urls and emails
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r'\S+@\S+', ' ', s)
    # remove mentions but keep text (or remove entirely)
    s = re.sub(r'@\w+', ' ', s)
    # keep hashtags text without the #
    s = re.sub(r'#(\w+)', r'\1', s)
    # remove punctuation (keep intra-word apostrophes removed earlier by lowercasing)
    s = re.sub(r"[^\w\s]", ' ', s)
    # remove numbers (optional: keep if you want)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    tokens = s.split()
    if lemmatizer is not None:
        try:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            pass

    if stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words]

    return ' '.join(tokens)


def load_and_label_files(max_samples_per_class=5000):
    """Load and label data files, with optional sampling to reduce processing time."""
    dfs = []

    # Support prefixed filenames (e.g., '3. Aggressive_All.csv') by matching suffix
    candidates = list(DATA_DIR.glob('*Aggressive_All.csv'))
    aggressive = candidates[0] if candidates else DATA_DIR / 'Aggressive_All.csv'
    candidates = list(DATA_DIR.glob('*Non_Aggressive_All.csv'))
    non_aggressive = candidates[0] if candidates else DATA_DIR / 'Non_Aggressive_All.csv'

    if aggressive.exists():
        df_a = pd.read_csv(aggressive, encoding='utf-8', low_memory=False)
        txt_col = find_text_column(df_a)
        if txt_col is None:
            logging.warning('No text column found in Aggressive_All.csv; skipping')
        else:
            df_a = df_a[[txt_col]].rename(columns={txt_col: 'text'})
            df_a['label'] = 'aggressive'
            df_a['source'] = 'Aggressive_All'
            # Sample to reduce processing time
            if len(df_a) > max_samples_per_class:
                df_a = df_a.sample(n=max_samples_per_class, random_state=42)
                logging.info(f'Sampled {max_samples_per_class} rows from Aggressive_All.csv')
            dfs.append(df_a)
    else:
        logging.info('Aggressive_All.csv not found')

    if non_aggressive.exists():
        df_n = pd.read_csv(non_aggressive, encoding='utf-8', low_memory=False)
        txt_col = find_text_column(df_n)
        if txt_col is None:
            logging.warning('No text column found in Non_Aggressive_All.csv; skipping')
        else:
            df_n = df_n[[txt_col]].rename(columns={txt_col: 'text'})
            df_n['label'] = 'non_aggressive'
            df_n['source'] = 'Non_Aggressive_All'
            # Sample to reduce processing time
            if len(df_n) > max_samples_per_class:
                df_n = df_n.sample(n=max_samples_per_class, random_state=42)
                logging.info(f'Sampled {max_samples_per_class} rows from Non_Aggressive_All.csv')
            dfs.append(df_n)
    else:
        logging.info('Non_Aggressive_All.csv not found')

    # Attempt to use CB_Labels.csv if present (mapping or samples)
    candidates = list(DATA_DIR.glob('*CB_Labels.csv'))
    cb_labels = candidates[0] if candidates else DATA_DIR / 'CB_Labels.csv'
    if cb_labels.exists() and not dfs:
        # if we didn't find aggressive/non_aggressive, try to read cb labels as main dataset
        df_cb = pd.read_csv(cb_labels, encoding='utf-8', low_memory=False)
        txt_col = find_text_column(df_cb)
        lbl_col = None
        # try to find label column
        for c in ['label', 'class', 'cb_label']:
            if c in df_cb.columns:
                lbl_col = c
                break
        if txt_col is not None and lbl_col is not None:
            df_cb = df_cb[[txt_col, lbl_col]].rename(columns={txt_col: 'text', lbl_col: 'label'})
            df_cb['source'] = 'CB_Labels'
            dfs.append(df_cb)
        else:
            logging.info('CB_Labels.csv present but could not auto-detect text/label columns')

    if not dfs:
        logging.error('No labeled text files found in data/. Please place Aggressive_All.csv and/or Non_Aggressive_All.csv (or a labeled CSV) in the data/ folder.')
        return pd.DataFrame(columns=['text', 'label', 'source'])

    df = pd.concat(dfs, ignore_index=True, sort=False)
    return df


def preprocess_and_save():
    stop_words, lemmatizer = _ensure_nltk()

    df = load_and_label_files()
    if df.empty:
        logging.error('No data to process. Exiting.')
        return None

    # Drop rows with missing text
    df['text'] = df['text'].astype(str)
    df = df.dropna(subset=['text'])

    # Clean
    logging.info('Cleaning text...')
    try:
        from tqdm import tqdm
        df['clean_text'] = [clean_text(s, stop_words=stop_words, lemmatizer=lemmatizer) 
                           for s in tqdm(df['text'], desc='Cleaning texts')]
    except ImportError:
        df['clean_text'] = df['text'].apply(lambda s: clean_text(s, stop_words=stop_words, lemmatizer=lemmatizer))
        logging.info('Text cleaning completed (no progress bar - tqdm not available)')

    # Drop empty cleaned rows
    before = len(df)
    df = df[df['clean_text'].str.strip().astype(bool)]
    after = len(df)
    logging.info(f'Dropped {before - after} empty/invalid texts after cleaning')

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=['clean_text'])
    logging.info(f'Removed {before - len(df)} duplicate texts')

    # Re-index
    df = df.reset_index(drop=True)

    # Basic label normalization
    df['label'] = df['label'].astype(str).str.strip()

    # Save
    out_pkl = PROCESSED_DIR / 'processed_data.pkl'
    out_csv = PROCESSED_DIR / 'processed_data.csv'
    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)

    logging.info(f'Processed dataset saved: {out_pkl} ({len(df):,} rows)')
    logging.info(f'Also saved CSV: {out_csv}')

    # Print quick summary
    logging.info('\nLabel distribution:')
    logging.info(df['label'].value_counts(dropna=False).to_string())
    logging.info('\nSample rows:')
    logging.info(df[['clean_text', 'label', 'source']].head(5).to_string(index=False))

    return df


def main():
    logging.info('Starting preprocessing...')
    df = preprocess_and_save()
    if df is None:
        logging.error('Preprocessing did not produce a dataset')
        sys.exit(1)
    logging.info('Preprocessing finished successfully')


if __name__ == '__main__':
    main()


class TextPreprocessor:
    """Small wrapper used by the dashboard for single-text cleaning and lexicons."""
    def __init__(self):
        self.stop_words, self.lemmatizer = _ensure_nltk()
        # load profanity list if available
        try:
            from config import LEXICONS_DIR, PROFANITY_LIST_FILE
            pfile = LEXICONS_DIR / PROFANITY_LIST_FILE
            if pfile.exists():
                with open(pfile, 'r', encoding='utf-8') as f:
                    words = [w.strip().lower() for w in f.read().splitlines() if w.strip()]
                self.profanity_words = set(words)
            else:
                self.profanity_words = set()
        except Exception:
            self.profanity_words = set()

    def clean_text(self, s: str):
        return clean_text(s, stop_words=self.stop_words, lemmatizer=self.lemmatizer)


def get_sentiment(text: str) -> float:
    """Get sentiment polarity score for text using TextBlob."""
    try:
        from textblob import TextBlob
        return TextBlob(text).sentiment.polarity
    except ImportError:
        # Fallback if TextBlob not available
        return 0.0

