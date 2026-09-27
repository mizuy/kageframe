"""The README quick start runs as written."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def _python_block_after(heading: str) -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split(heading, 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def test_quickstart_runs(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(_python_block_after("## クイックスタート"), namespace)
    assert "all checks passed" in capsys.readouterr().out
    assert (tmp_path / "profile.json").exists()
    assert namespace["report"].check()
