from gui.app import process_command


def test_gui_process_command_reuses_pipeline():
    result = process_command("Create a folder called GUI Demo", dry_run=True)
    assert result["intent"] == "CREATE_FOLDER"
    assert result["params"]["name"] == "GUI Demo"
    assert result["success"]
    assert "[dry-run]" in result["message"]
