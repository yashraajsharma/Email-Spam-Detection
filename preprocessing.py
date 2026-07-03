"""
Shared text preprocessing used by both the training script and the
Streamlit app, so the exact same cleaning logic is applied at training
time and at prediction time (this consistency matters — a model trained
on cleaned text will perform poorly if the app feeds it raw text).
"""

import re
import string

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "with",
    "this", "that", "these", "those", "it", "its", "i", "you", "your",
    "we", "our", "they", "them", "he", "she", "his", "her", "as", "by",
    "from", "up", "so", "if", "not", "no", "do", "does", "did", "have",
    "has", "had", "will", "would", "can", "could", "just", "about",
}


def preprocess(text: str) -> str:
    """Lowercase, strip URLs/punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)
