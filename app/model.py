import joblib
import pandas as pd
from config.logs_config import *


pipeline = joblib.load(settings.model_path)

TEXT_COL = 'segmented'
CATEGORICAL_COLS = ['lang']
NUMERIC_COLS = ['token_len', 'char_len', 'sent_len',
                'exclaimation_num', 'questionmark_num', 'url_num',
                'hash_num', 'mention_num']
ALL_FEATURES = [TEXT_COL] + CATEGORICAL_COLS + NUMERIC_COLS


def predict(text: str, lang: str = 'en') -> dict:
    try:
        logger_interface.debug(f"Starting prediction for text: {text[:100]}...")

        if not text or len(text.strip()) < 5:
            logger_interface.warning("Input validation failed: text too short or empty")
            return {"error": "Text too short or empty"}
        if len(text) > 5000:
            logger_interface.warning("Input validation failed: text too long")
            return {"error": "Text too long"}

        # Example dummy values
        dummy_values: dict[str, int] = {
            'token_len': len(text.split()),
            'char_len': len(text),
            'sent_len': text.count('.') + text.count('!') + text.count('?'),
            'exclaimation_num': text.count('!'),
            'questionmark_num': text.count('?'),
            'url_num': text.count('http'),
            'hash_num': text.count('#'),
            'mention_num': text.count('@'),
        }

        row = {
            'segmented': text.lower().strip(),
            'lang': lang,
            **dummy_values
        }

        df = pd.DataFrame([row])[ALL_FEATURES]

        logger_interface.debug("Input DataFrame constructed")
        logger_interface.debug(f"DataFrame preview:\n{df}")

        proba = pipeline.predict_proba(df)[0]
        classes = pipeline.classes_
        pred_index = proba.argmax()

        sentiment_labels = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

        result = {
            "sentiment": sentiment_labels[pred_index],
            "confidence": round(float(proba[pred_index]), 4)
        }

        logger_interface.info(f"Prediction successful: {result}")
        return result

    except Exception as e:
        logger_interface.error(f"Error during prediction: {e}", exc_info=True)
        return {"error": "Internal model error"}
