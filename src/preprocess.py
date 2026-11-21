import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """Normalize SMS/email text while keeping phishing cues like URLs and urgency tokens."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Replace explicit URLs and email addresses with normalized tokens
    text = re.sub(r"https?://\S+|www\.\S+", " url_token ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " email_token ", text)
    # Collapse standalone numbers/codes (OTPs, amounts, tracking IDs) into a token
    text = re.sub(r"\b\d+\b", " num_token ", text)
    # Keep word characters and spaces
    text = re.sub(r"[^a-z0-9_\s]", " ", text)

    tokens = [
        tok for tok in text.split()
        if tok not in ENGLISH_STOP_WORDS and len(tok) > 1
    ]
    return " ".join(tokens)