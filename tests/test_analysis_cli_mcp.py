"""``spinescope analyze`` and the MCP ``analyze`` tool on the phantom: JSON, no input paths, unknown names refused."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from spinescope.ai.mcp_server import SpineTools, ToolError
from spinescope.cli import main


def test_cli_analyze_prints_json_without_input_path(phantom_dir: dict, tmp_path: Path) -> None:
    root = str(phantom_dir["root"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["analyze", root, "--only", "discs", "--out", str(tmp_path / "out")])
    assert rc == 0
    text = buf.getvalue()
    assert root not in text
    doc = json.loads(text)
    assert set(doc["analyses"]) == {"discs"}
    assert doc["analyses"]["discs"]["saved"]  # file names only
    assert all("/" not in name for name in doc["analyses"]["discs"]["saved"])
    assert (tmp_path / "out" / "discs.csv").is_file()


def test_cli_analyze_refuses_unknown_name(phantom_dir: dict, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["analyze", str(phantom_dir["root"]), "--only", "nonsense"]) == 2
    assert "unknown analyses: nonsense" in capsys.readouterr().err


def test_mcp_analyze_returns_rows_flags_and_numbering(phantom_dir: dict, tmp_path: Path) -> None:
    tools = SpineTools(output_dir=tmp_path / "png")
    sid = tools.open_study(str(phantom_dir["root"]))["study_id"]
    out = tools.analyze(sid, only=["endplates"])
    text = json.dumps(out)
    assert str(phantom_dir["root"]) not in text
    entry = out["analyses"]["endplates"]
    assert entry["rows"] and "warnings" in entry and "numbering" in out
    assert entry["saved_to"] and all(Path(p).is_file() for p in entry["saved_to"])


def test_mcp_analyze_unknown_name_is_a_tool_error(phantom_dir: dict) -> None:
    tools = SpineTools()
    sid = tools.open_study(str(phantom_dir["root"]))["study_id"]
    with pytest.raises(ToolError):
        tools.analyze(sid, only=["bogus"])
