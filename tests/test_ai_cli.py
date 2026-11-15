"""CLI subcommands for the AI workflow: brief, tasks, viewer, mcp (phantom only)."""

from __future__ import annotations

import json

import pytest

from spinescope.cli import main


def test_brief_to_stdout_and_file(phantom_dir, capsys, tmp_path):
    assert main(["brief", str(phantom_dir["root"]), "--reader", "reader_c"]) == 0
    out = capsys.readouterr().out
    assert "SPINE MRI READING BRIEF" in out and "Reader name: reader_c." in out
    assert str(phantom_dir["root"]) not in out
    dest = tmp_path / "brief.txt"
    assert main(["brief", str(phantom_dir["root"]), "--no-anchors", "--out", str(dest)]) == 0
    assert "No anchors were supplied" in dest.read_text(encoding="utf-8")


def test_tasks_prints_json_and_writes_prompts(phantom_dir, capsys, tmp_path):
    assert main(["tasks", str(phantom_dir["root"]), "--no-anchors"]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert {t["kind"] for t in tasks} == {"structure", "holistic", "verify"}
    assert all("prompt" in t for t in tasks)

    outdir = tmp_path / "tasks"
    assert main(["tasks", str(phantom_dir["root"]), "--outdir", str(outdir)]) == 0
    index = json.loads((outdir / "tasks.json").read_text(encoding="utf-8"))
    assert all("prompt" not in t for t in index)
    for t in index:
        assert (outdir / "prompts" / f"{t['reader']}.txt").exists()


def test_viewer_writes_html(phantom_dir, capsys, tmp_path):
    assert main(["viewer", str(phantom_dir["root"]), str(tmp_path / "v"), "--no-anchors"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("spinescope_viewer.html")
    assert (tmp_path / "v" / "spinescope_viewer.html").exists()


def test_subcommands_are_listed_in_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    for cmd in ("brief", "tasks", "viewer", "mcp"):
        assert cmd in text
