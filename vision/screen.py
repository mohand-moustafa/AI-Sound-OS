"""Screenshot and OCR actions.

Screenshot capture uses the Python ``mss`` package. OCR uses Tesseract through
``pytesseract`` and therefore also needs the Ubuntu ``tesseract-ocr`` package.
"""
from datetime import datetime
from pathlib import Path
import os
import shutil
import subprocess

import config


def screenshot_path() -> Path:
    config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.SCREENSHOTS_DIR / f"screenshot_{stamp}.png"


def capture_screen(output=None) -> Path:
    destination = Path(output) if output else screenshot_path()
    if destination.parent != config.SCREENSHOTS_DIR:
        destination = config.SCREENSHOTS_DIR / destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type == "wayland":
        for command in (
            ["gnome-screenshot", "-f", str(destination)],
            ["spectacle", "-b", "-n", "-o", str(destination)],
            ["grim", str(destination)],
        ):
            if shutil.which(command[0]):
                completed = subprocess.run(command, capture_output=True, timeout=15)
                if completed.returncode == 0 and destination.is_file():
                    break

    if not destination.is_file():
        try:
            import mss
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Vision requires Python packages mss and Pillow. "
                "Run: pip install -r requirements.txt"
            ) from error
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
        image = Image.open(path).convert("RGB")
        image = image.resize((image.width * 2, image.height * 2))
        from PIL import ImageOps
        image = ImageOps.autocontrast(ImageOps.grayscale(image))

        candidates = []
        for psm in ("11", "6"):
            candidates.append(pytesseract.image_to_string(
                image, lang="eng", config=f"--psm {psm}",
            ).strip())
        text = max(candidates, key=len)
        return text
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract is not installed. Run: sudo apt install tesseract-ocr"
        ) from error


def active_window() -> dict:
    """Get the active window title and application without guessing."""
    result = {"title": "", "app": "", "session": os.environ.get("XDG_SESSION_TYPE", "")}
    xdotool = shutil.which("xdotool")
    if xdotool:
        try:
            window_id = subprocess.run(
                [xdotool, "getactivewindow"], capture_output=True, text=True, timeout=3,
            ).stdout.strip()
            if window_id:
                title = subprocess.run(
                    [xdotool, "getwindowname", window_id],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip()
                pid_text = subprocess.run(
                    [xdotool, "getwindowpid", window_id],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip()
                result["title"] = title
                if pid_text.isdigit():
                    exe = Path(f"/proc/{pid_text}/exe")
                    if exe.exists():
                        result["app"] = exe.resolve().name
        except (OSError, subprocess.SubprocessError):
            pass

    if not result["title"]:
        gdbus = shutil.which("gdbus")
        if gdbus:
            script = (
                "global.display.focus_window ? "
                "(global.display.focus_window.get_title() || '') : ''"
            )
            try:
                output = subprocess.run(
                    [gdbus, "call", "--session", "--dest", "org.gnome.Shell",
                     "--object-path", "/org/gnome/Shell", "--method",
                     "org.gnome.Shell.Eval", script],
                    capture_output=True, text=True, timeout=3,
                ).stdout
                if output.startswith("(true,"):
                    value = output.split(",", 1)[1].strip().rstrip(")")
                    result["title"] = value.strip(" '\"")
            except (OSError, subprocess.SubprocessError):
                pass
    return result


def infer_application(window: dict) -> str:
    title = window.get("title", "").lower()
    app = window.get("app", "").lower()
    combined = f"{app} {title}"
    mappings = (
        (("code", "visual studio code", "vscode"), "Visual Studio Code"),
        (("firefox",), "Firefox"),
        (("chrome", "chromium", "google chrome"), "Google Chrome"),
        (("terminal", "gnome-terminal", "konsole", "kitty"), "Terminal"),
        (("gedit", "text editor", "mousepad"), "Text Editor"),
        (("nautilus", "files"), "Files"),
        (("calc", "calculator"), "Calculator"),
        (("libreoffice", "writer"), "LibreOffice"),
    )
    for identifiers, name in mappings:
        if any(identifier in combined for identifier in identifiers):
            return name
    return app or "Unknown application"


def infer_activity(window: dict, text: str) -> str:
    title = window.get("title", "").lower()
    app = infer_application(window).lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = "\n".join(lines[:20]).lower()

    if "visual studio code" in app or "code" in app:
        if lines:
            return "Viewing or editing source code in Visual Studio Code."
        return "Using Visual Studio Code."
    if "terminal" in app:
        if lines:
            return "A terminal session is open and command or shell text is visible."
        return "A terminal session is open."
    if "firefox" in app or "chrome" in app:
        if "github" in title or "github" in content:
            return "Browsing GitHub or a repository webpage."
        if "youtube" in title or "youtube" in content:
            return "Viewing YouTube in the web browser."
        if "settings" in title or "preferences" in title:
            return "Viewing browser settings or preferences."
        return "Browsing a webpage in the active web browser."
    if "files" in app:
        return "Browsing files and folders."
    if "text editor" in app:
        return "Viewing or editing text in a text editor."
    if "libreoffice" in app:
        return "Working in a LibreOffice document."
    if "github" in title or "github" in content:
        return "Viewing GitHub or repository-related content."
    if lines:
        return "An application is open with readable interface or document text."
    return "An application is open, but its purpose could not be determined reliably."


def read_screen() -> str:
    text = ocr_image(capture_screen())
    return f"Screen text:\n{text}" if text else "No readable text was found on the screen."


def describe_screen() -> str:
    path = capture_screen()
    window = active_window()
    text = ocr_image(path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    application = infer_application(window)

    parts = [f"Active application: {application}"]
    if window.get("title"):
        parts.append(f"Window title: {window['title']}")
    parts.append(f"Activity: {infer_activity(window, text)}")
    if lines:
        preview = "\n".join(lines[:12])
        suffix = f"\n... and {len(lines) - 12} more text lines" if len(lines) > 12 else ""
        parts.append(f"Visible text:\n{preview}{suffix}")
    else:
        parts.append("Visible text: No readable text was detected.")
    parts.append(f"Screenshot: {path}")
    return "\n".join(parts)
