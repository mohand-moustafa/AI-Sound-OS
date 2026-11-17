from pathlib import Path

from vision.screen import screenshot_path


def test_screenshot_path_is_timestamped_inside_screenshot_directory():
    path = screenshot_path()
    assert path.parent.name == "screenshots"
    assert path.suffix == ".png"
    assert Path(path.name).name == path.name
