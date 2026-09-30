from nlp.extractor import extract


def test_app_name():
    assert extract("OPEN_APPLICATION", "Please open Firefox") == {"app": "firefox"}


def test_app_alias():
    assert extract("OPEN_APPLICATION", "Start VS Code")["app"] == "vscode"


def test_folder_name_keeps_capitals():
    result = extract("CREATE_FOLDER", "Create a folder called AI Project")
    assert result["name"] == "AI Project"


def test_folder_name_with_location():
    result = extract("CREATE_FOLDER", "Create a folder named Reports in documents")
    assert result == {"name": "Reports", "location": "documents"}


def test_no_name_given():
    assert extract("CREATE_FOLDER", "Make a new folder")["name"] is None


def test_location():
    assert extract("LIST_FILES", "Show the files in Downloads") == {"location": "downloads"}
