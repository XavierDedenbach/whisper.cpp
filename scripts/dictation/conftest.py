"""Never let a regression-test daemon recover the operator's real sessions."""

import pytest


@pytest.fixture(autouse=True)
def isolated_dictation_home(tmp_path, monkeypatch):
    home = tmp_path / "operator-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
