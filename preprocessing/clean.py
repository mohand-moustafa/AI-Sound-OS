"""Text cleaning. Owner: Member 3."""
import re


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse spaces.

    Example: "Open Chrome!" -> "open chrome"
    """
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cleaned_text_column(frame, source="text"):
    """Return the canonical cleaned text used by training and inference."""
    if source not in frame.columns:
        raise KeyError(f"Missing text column: {source}")
    return frame[source].astype(str).map(clean_text)
