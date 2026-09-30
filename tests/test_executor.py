from commands.executor import execute


def test_create_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))          # use a temporary home folder
    result = execute("CREATE_FOLDER", {"name": "AI Project", "location": "home"})
    assert result.success
    assert (tmp_path / "AI Project").is_dir()


def test_create_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert execute("CREATE_FILE", {"name": "notes.txt"}).success
    assert (tmp_path / "notes.txt").is_file()


def test_unsafe_names_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert not execute("CREATE_FOLDER", {"name": "../evil"}).success
    assert not execute("CREATE_FOLDER", {"name": "a/b"}).success


def test_list_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "hello.txt").touch()
    result = execute("LIST_FILES", {"location": "home"})
    assert result.success and "hello.txt" in result.message


def test_unknown_does_nothing():
    assert not execute("UNKNOWN", {}).success


def test_dry_run_does_not_create(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    execute("CREATE_FOLDER", {"name": "Nope"}, dry_run=True)
    assert not (tmp_path / "Nope").exists()
