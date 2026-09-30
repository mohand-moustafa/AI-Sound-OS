"""Find the details in the sentence (app name, folder name, location).
Owner: Member 5.

This part uses simple rules, not ML. It reads the ORIGINAL text
(not the cleaned one) so names like "AI Project" keep their capital letters.
"""
import difflib
import re

APP_ALIASES = {
    "chrome": ["chrome", "google chrome", "browser"],
    "firefox": ["firefox", "fire fox"],
    "vscode": ["vs code", "vscode", "visual studio code", "code editor", "code"],
    "terminal": ["terminal", "console", "command line"],
    "calculator": ["calculator"],
    "text_editor": ["text editor", "notepad", "gedit"],
}
ALIAS_TO_APP = {alias: app for app, aliases in APP_ALIASES.items() for alias in aliases}

LOCATIONS = ["downloads", "documents", "desktop", "home"]
_LOC = "|".join(LOCATIONS)


def find_app(text):
    t = text.lower()
    # 1) exact alias in the sentence (longest alias first)
    for alias in sorted(ALIAS_TO_APP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return ALIAS_TO_APP[alias]
    # 2) fuzzy match on the last word, to survive small speech mistakes ("crome")
    words = t.replace(".", " ").split()
    if words:
        close = difflib.get_close_matches(words[-1], ALIAS_TO_APP, n=1, cutoff=0.75)
        if close:
            return ALIAS_TO_APP[close[0]]
    return None


def find_location(text):
    for loc in LOCATIONS:
        if re.search(rf"\b{loc}\b", text.lower()):
            return loc
    return "home"


def find_name(text):
    """Get the folder/file name. Returns None if the user did not say one."""
    text = text.strip().rstrip(".!?")
    patterns = [
        r"(?:called|named|name it|with the name)\s+(.+)$",
        r"(?:folder|directory|file)\s+(.+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        name = re.sub(r"^(the|a|an)\s+", "", m.group(1).strip(), flags=re.I)
        # remove "in downloads", "inside the documents folder", ...
        name = re.sub(rf"\s*\b(in|inside|under|on)\s+(the\s+|my\s+)?({_LOC})(\s+folder)?$",
                      "", name, flags=re.I).strip()
        if name and not re.match(rf"^(in|inside|under|on)\b", name, re.I):
            return name
    return None


def extract(intent, text):
    """Return a dict with the details needed for this intent."""
    if intent in ("OPEN_APPLICATION", "CLOSE_APPLICATION"):
        return {"app": find_app(text)}
    if intent in ("CREATE_FOLDER", "CREATE_FILE"):
        return {"name": find_name(text), "location": find_location(text)}
    if intent in ("LIST_FILES", "OPEN_FOLDER"):
        return {"location": find_location(text)}
    return {}
