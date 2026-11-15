"""Outputs of the synthesis: ``spinescope analyze --report``, the MCP ``analyze`` tool and the AI brief, on the phantom.
Checks: the files exist and the findings JSON validates against the schema; no input path reaches stdout or the
returned JSON; the brief gets its section only when a synthesis is passed."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from spinescope.ai.briefs import brief
from spinescope.ai.mcp_server import SpineTools
from spinescope.cli import main
from spinescope.report.validate import validate
from spinescope.study import load_study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study


@pytest.fixture(scope="module")
def phantom_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    ph = SpinePhantom(PhantomSpec(degenerated={"L4-L5": 0.2}, herniation={"L4-L5": ("central", 6.0)}))
    d = write_standard_study(ph, tmp_path_factory.mktemp("syn_phantom"))
    return Path(d["root"])


def test_cli_analyze_report_writes_three_files_and_validates(phantom_root: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            ["analyze", str(phantom_root), "--only", "discs,endplates", "--out", str(out_dir), "--report"]
        )
    assert rc == 0
    text = buf.getvalue()
    assert str(phantom_root) not in text
    doc = json.loads(text)
    assert set(doc["report_files"]) == {"findings.json", "report.md", "report.html"}
    assert "levels" in doc["synthesis"] and doc["synthesis"]["disclaimer"]
    check = validate(out_dir / "findings.json")
    assert check.ok, check.error_messages()
    assert check.warnings == []
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert str(phantom_root) not in md and "Research software" in md
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html  # thumbnails are embedded, so the page works on its own
    assert str(out_dir) not in html


def test_cli_report_needs_out(phantom_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["analyze", str(phantom_root), "--only", "discs", "--report"]) == 2
    assert "--report writes into the --out folder" in capsys.readouterr().err


def test_mcp_analyze_returns_synthesis_and_saves_the_bundle(phantom_root: Path, tmp_path: Path) -> None:
    tools = SpineTools(output_dir=tmp_path / "png")
    sid = tools.open_study(str(phantom_root))["study_id"]
    out = tools.analyze(sid, only=["discs", "endplates"])
    text = json.dumps(out)
    assert str(phantom_root) not in text
    syn = out["synthesis"]
    assert {"levels", "global", "pain_evidence", "disclaimer", "formula"} <= set(syn)
    lv = next(x for x in syn["levels"] if x["level"] == "L4-L5")
    assert {"name_status", "findings", "not_assessed", "strength"} <= set(lv)
    assert out["analyses"]["discs"]["rows"]  # per-analysis detail is still there
    names = {Path(p).name for p in out["report_files"]}
    assert {"findings.json", "report.md", "report.html"} <= names
    assert validate(next(Path(p) for p in out["report_files"] if p.endswith("findings.json"))).ok


def test_brief_has_the_synthesis_section_only_when_given(phantom_root: Path) -> None:
    from spinescope.analysis.base import run_all
    from spinescope.anatomy import label_spine
    from spinescope.report.synthesis import build_synthesis

    study = load_study(phantom_root)
    anchors = label_spine(study)
    plain = brief(study, anchors)
    assert "6. Your scope" not in plain and "7. Findings format" in plain
    assert "Automatic measurements" not in plain
    syn = build_synthesis(study, anchors, run_all(study, anchors, names=["discs"]))
    text = brief(study, anchors, synthesis=syn)
    assert "6. Automatic measurements (synthesis; validation status per field)" in text
    assert "8. Findings format" in text and "7. Findings format" not in text
    assert "Consider correlation with symptoms and examination" in text
