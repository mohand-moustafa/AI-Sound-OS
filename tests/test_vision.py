from pathlib import Path

import vision.screen as screen


def test_screenshot_path_is_timestamped_inside_screenshot_directory():
    path = screen.screenshot_path()
    assert path.parent.name == "screenshots"
    assert path.suffix == ".png"
    assert Path(path.name).name == path.name


def test_infer_application_from_window_title():
    window = {"app": "firefox", "title": "GitHub - AI-Sound-OS"}
    assert screen.infer_application(window) == "Firefox"
    assert "GitHub" in screen.infer_activity(window, "")


def test_infer_terminal_activity():
    window = {"app": "gnome-terminal", "title": "Terminal"}
    text = "mohand@mohand-VirtualPlatform:~/AI-Sound-OS$"
    assert screen.infer_application(window) == "Terminal"
    assert "terminal session" in screen.infer_activity(window, text).lower()


def test_describe_screen_reports_app_window_activity(monkeypatch, tmp_path):
    image = tmp_path / "screen.png"
    image.touch()
    monkeypatch.setattr(screen, "capture_screen", lambda: image)
    monkeypatch.setattr(screen, "active_window", lambda: {
        "app": "code",
        "title": "classifier.py - AI-Sound-OS - Visual Studio Code",
    })
    monkeypatch.setattr(screen, "ocr_image", lambda _path:
                        "def describe_screen():\nreturn description")

    description = screen.describe_screen()
    assert "Active application: Visual Studio Code" in description
    assert "classifier.py" in description
    assert "source code" in description
    assert "Visible text:" in description
