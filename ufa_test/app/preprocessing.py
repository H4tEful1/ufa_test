import re
import string
from typing import List, Optional
import spacy
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from config.logs_config import *

# Load spaCy models for both English and Russian
try:
    nlp_en = spacy.load("en_core_web_sm")
    nlp_ru = spacy.load("ru_core_news_sm")
except OSError:
    # If models aren't downloaded, install them first
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    subprocess.run(["python", "-m", "spacy", "download", "ru_core_news_sm"])
    nlp_en = spacy.load("en_core_web_sm")
    nlp_ru = spacy.load("ru_core_news_sm")

# Load stopwords
nltk.download('stopwords')
ru_stopwords = set(stopwords.words('russian'))
en_stopwords = ENGLISH_STOP_WORDS


def detect_language(text: str) -> Optional[str]:
    """Simple language detection (checks for Cyrillic chars)."""
    if re.search(r'[а-яА-ЯёЁ]', text):
        return "ru"
    elif re.search(r'[a-zA-Z]', text):
        return "en"
    return None


def clean_text(text: str) -> str:
    try:
        logger_interface.debug("Cleaning text")
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return text
    except Exception as e:
        logger_interface.error(f"Error in clean_text: {e}", exc_info=True)
        return text


def tokenize_with_spacy(text: str, lang: str) -> List[str]:
    """Tokenize using spaCy (better for both English & Russian)."""
    doc = nlp_ru(text) if lang == "ru" else nlp_en(text)
    return [token.text for token in doc if not token.is_punct]


def remove_stopwords(tokens: List[str], lang: str) -> List[str]:
    try:
        stopwords_set = ru_stopwords if lang == "ru" else en_stopwords
        return [t for t in tokens if t not in stopwords_set]
    except Exception as e:
        logger_interface.error(f"Error in remove_stopwords: {e}", exc_info=True)
        return tokens


def preprocess(text: str) -> str:
    try:
        if not text or len(text.strip()) < 5:
            logger_interface.warning("Preprocessing skipped: text too short or empty")
            return ""

        logger_interface.info(f"Starting preprocessing for text: {text[:50]}...")

        # Detect language (fallback to English if uncertain)
        lang = detect_language(text) or "en"
        logger_interface.debug(f"Detected language: {lang}")

        text = clean_text(text)
        tokens = tokenize_with_spacy(text, lang)
        tokens = remove_stopwords(tokens, lang)

        processed = " ".join(tokens)
        logger_interface.debug(f"Preprocessed text: {processed[:50]}...")
        return processed

    except Exception as e:
        logger_interface.error(f"Error in preprocess: {e}", exc_info=True)
        return ""
