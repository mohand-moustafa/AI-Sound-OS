"""Screenshot and OCR actions.

Screenshot capture uses the Python ``mss`` package. OCR uses Tesseract through
``pytesseract`` and therefore also needs the Ubuntu ``tesseract-ocr`` package.
"""
from datetime import datetime
from pathlib import Path

import config


def screenshot_path() -> Path:
    config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.SCREENSHOTS_DIR / f"screenshot_{stamp}.png"


def capture_screen(output=None) -> Path:
    try:
        import mss
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Vision requires Python packages mss and Pillow. "
            "Run: pip install -r requirements.txt"
        ) from error

    destination = Path(output) if output else screenshot_path()
    if destination.parent != config.SCREENSHOTS_DIR:
        destination = config.SCREENSHOTS_DIR / destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with mss.mss() as monitor:
        image = monitor.grab(monitor.monitors[0])
        Image.frombytes("RGB", image.size, image.rgb).save(destination)
    return destination


def take_screenshot() -> str:
    return f"Screenshot saved to {capture_screen()}"


def ocr_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "OCR requires pytesseract and Pillow. Run: pip install -r requirements.txt"
        ) from error
    try:
        return pytesseract.image_to_string(Image.open(path)).strip()
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract is not installed. Run: sudo apt install tesseract-ocr"
        ) from error


def read_screen() -> str:
    text = ocr_image(capture_screen())
    return f"Screen text:\n{text}" if text else "No readable text was found on the screen."


def describe_screen() -> str:
    text = ocr_image(capture_screen())
    if not text:
        return "The screen was captured, but no readable text was found."
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    preview = "\n".join(lines[:20])
    suffix = f"\n... and {len(lines) - 20} more lines" if len(lines) > 20 else ""
    return f"Screen summary ({len(lines)} text lines):\n{preview}{suffix}"
