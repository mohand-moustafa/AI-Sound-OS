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
