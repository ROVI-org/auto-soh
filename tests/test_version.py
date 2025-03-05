from moirae.version import load_git_version


def test_git():
    version = load_git_version()
    assert isinstance(version, str)
    assert len(version) >= 2
