"""Run the Linux action for each intent. Owner: Member 7.

Safety rules:
  - never use shell=True
  - only run apps from OPEN_COMMANDS
  - only create files/folders inside the home folder
  - names cannot contain "/" or ".."
"""
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Result:
    success: bool
    message: str


# The first program found on the computer is used.
OPEN_COMMANDS = {
    "chrome": ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"],
    "firefox": ["firefox"],
    "vscode": ["code"],
    "terminal": ["gnome-terminal", "x-terminal-emulator"],
    "calculator": ["gnome-calculator", "kcalc"],
    "text_editor": ["gnome-text-editor", "gedit", "mousepad"],
}
# Process names for closing apps. Check them with `pgrep -l` on the demo laptop!
# "terminal" is NOT here on purpose: closing it would close our own program.
PROCESS_NAMES = {"chrome": "chrome", "firefox": "firefox", "vscode": "code"}

FOLDERS = {"home": "", "downloads": "Downloads", "documents": "Documents", "desktop": "Desktop"}
SAFE_NAME = re.compile(r"^[\w\- .]{1,50}$")


def location_path(location):
    return Path.home() / FOLDERS.get(location or "home", "")


def safe_path(base, name):
    """Check the name and make sure the result stays inside `base`."""
    if not SAFE_NAME.match(name) or name.strip(".") == "":
        raise ValueError(f"Invalid name: {name!r}")
    path = (base / name).resolve()
    if base.resolve() not in path.parents:
        raise ValueError("The path is outside the allowed folder")
    return path


def _spawn(cmd):
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def open_app(p):
    app = p.get("app")
    for program in OPEN_COMMANDS.get(app, []):
        if shutil.which(program):
            _spawn([program])
            return Result(True, f"Opening {app}")
    return Result(False, f"I can't find this application: {app}")


def close_app(p):
    app = p.get("app")
    process = PROCESS_NAMES.get(app)
    if not process:
        return Result(False, f"I can't close this application: {app}")
    code = subprocess.run(["pkill", "-x", process]).returncode
    if code == 0:
        return Result(True, f"Closed {app}")
    return Result(False, f"{app} is not running")


def create_folder(p):
    path = safe_path(location_path(p.get("location")), p.get("name") or "New Folder")
    path.mkdir(parents=True, exist_ok=True)
    return Result(True, f"Created folder: {path}")


def create_file(p):
    path = safe_path(location_path(p.get("location")), p.get("name") or "new_file.txt")
    path.touch()
    return Result(True, f"Created file: {path}")


def list_files(p):
    folder = location_path(p.get("location"))
    names = sorted(x.name for x in folder.iterdir())
    shown = "\n".join(names[:30]) or "(empty)"
    more = f"\n... and {len(names) - 30} more" if len(names) > 30 else ""
    return Result(True, f"Files in {folder}:\n{shown}{more}")


def open_folder(p):
    folder = location_path(p.get("location"))
    _spawn(["xdg-open", str(folder)])
    return Result(True, f"Opening {folder}")


def system_info(p):
    lines = [f"OS: {platform.platform()}", f"CPU cores: {os.cpu_count()}"]
    try:
        with open("/proc/meminfo") as f:
            mem = {line.split(":")[0]: int(line.split()[1]) for line in f}
        lines.append(f"RAM: {mem['MemTotal'] / 1024 / 1024:.1f} GB total, "
                     f"{mem['MemAvailable'] / 1024 / 1024:.1f} GB available")
    except (OSError, KeyError, ValueError):
        pass
    return Result(True, "\n".join(lines))


def show_date_time(p):
    return Result(True, datetime.now().strftime("It is %A, %d %B %Y, %H:%M"))


HANDLERS = {
    "OPEN_APPLICATION": open_app,
    "CLOSE_APPLICATION": close_app,
    "CREATE_FOLDER": create_folder,
    "CREATE_FILE": create_file,
    "LIST_FILES": list_files,
    "OPEN_FOLDER": open_folder,
    "SYSTEM_INFO": system_info,
    "SHOW_DATE_TIME": show_date_time,
}


def execute(intent, params, dry_run=False):
    """Run the action for `intent`. UNKNOWN does nothing."""
    if intent not in HANDLERS:
        return Result(False, "Sorry, I did not understand that command.")
    if dry_run:
        return Result(True, f"[dry-run] would run {intent} with {params}")
    try:
        return HANDLERS[intent](params)
    except (ValueError, OSError) as error:
        return Result(False, f"Error: {error}")
