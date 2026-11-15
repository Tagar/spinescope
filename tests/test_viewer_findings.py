"""Viewer findings panel: the PHI-free items built from analysis output, the page structure, the CLI (phantom only)."""

from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path

import pytest

from spinescope.analysis.base import Result
from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.cli import main
from spinescope.viewer import build_viewer, render_html
from spinescope.viewer.build import _json_for_script
from spinescope.viewer.export import build_payload
from spinescope.viewer.findings_view import (
    FINDINGS_FORMAT,
    MAX_ITEMS,
    collect,
    load_findings_file,
    results_from_analyze_json,
)


def _anchors() -> Anchors:
    """Three discs (superior first in the dict order does not matter) and one vertebra, LPS mm."""
    a = Anchors()
    a.discs["L3-L4"] = Disc(center=[0.0, 10.0, -20.0], normal=[0.0, 0.0, 1.0], confidence=0.9)
    a.discs["L4-L5"] = Disc(center=[0.0, 12.0, -40.0], normal=[0.0, 0.0, 1.0], confidence=0.6)
    a.discs["L5-S1"] = Disc(center=[0.0, 14.0, -60.0], normal=[0.0, 0.0, 1.0], confidence=0.4)
    a.vertebrae["L4"] = Vertebra(center=[0.0, 11.0, -30.0], confidence=0.9)
    a.numbering = {"status": "probable", "confidence": 0.7}
    return a


def _result(name: str = "demo", **kw: object) -> Result:
    return Result(name=name, **kw)  # type: ignore[arg-type]


# ------------------------------------------------------------------ items from flags and rows
def test_flags_become_items_with_level_kind_value_and_confidence() -> None:
    res = _result(
        flags=[
            {
                "kind": "bulge",
                "level": "L4-L5",
                "class": "bulge",
                "confidence": 0.82,
                "note": "check the image",
            },
            {
                "kind": "modic_suggestion",
                "vertebra": "L4",
                "adjacent_disc": "L4-L5",
                "side": "left",
                "pattern": "Modic 1",
                "endplate": "inferior",
                "confidence": 0.3,
            },
        ]
    )
    sec = collect({"demo": res}, _anchors())
    assert sec["format"] == FINDINGS_FORMAT
    assert sec["numbering"] == "probable"
    bulge, modic = sec["items"]
    assert bulge["level"] == "L4-L5" and bulge["kind"] == "bulge" and bulge["value"] == "bulge"
    assert bulge["confidence"] == 0.82 and bulge["confidence_word"] == "high"
    assert (
        bulge["notes"] == "check the image" and bulge["analysis"] == "demo" and bulge["structure"] == "demo"
    )
    # an endplate flag is placed at the disc it borders, not at the vertebra
    assert modic["level"] == "L4-L5" and modic["side"] == "left" and modic["value"] == "Modic 1"
    assert modic["pos"] == [0.0, 12.0, -40.0] and modic["pos_source"] == "anchors"


def test_level_status_suffix_from_analysis_names_is_dropped() -> None:
    res = _result(flags=[{"kind": "retrolisthesis", "level": "L4-L5 (probable)", "confidence": 0.3}])
    (item,) = collect({"alignment": res}, _anchors())["items"]
    assert item["level"] == "L4-L5"  # the status comes from the anchors, not from the name
    assert item["pos"] == [0.0, 12.0, -40.0]


def test_positions_come_from_the_flag_before_the_anchors() -> None:
    res = _result(flags=[{"kind": "hit", "level": "L4-L5", "position": [1.25, 2.0, -3.5], "confidence": 0.5}])
    (item,) = collect({"x": res}, _anchors())["items"]
    assert item["pos"] == [1.25, 2.0, -3.5] and item["pos_source"] == "analysis"


def test_a_flag_without_any_position_is_listed_but_not_placed() -> None:
    res = _result(flags=[{"kind": "lesion", "level": "L9-L10", "confidence": "medium"}])
    (item,) = collect({"x": res}, _anchors())["items"]
    assert item["pos"] is None and item["confidence"] is None and item["confidence_word"] == "medium"


def test_rows_are_findings_only_when_they_opt_in() -> None:
    rows = [
        {"scope": "disc", "disc": "L4-L5", "height_mm": 7.1},  # a measurement: not listed
        {"scope": "disc", "disc": "L4-L5", "finding": "narrowing", "value": "mild", "confidence": 0.6},
    ]
    sec = collect({"discs": _result("discs", rows=rows)}, _anchors())
    assert [(i["kind"], i["value"]) for i in sec["items"]] == [("narrowing", "mild")]


def test_experimental_from_flag_or_from_analysis_status() -> None:
    meta = {"status": "experimental", "endplate_defect_status": "experimental", "method": "x"}
    res = _result(
        "endplates",
        meta=meta,
        flags=[
            {"kind": "modic_suggestion", "level": "L4-L5", "confidence": 0.4},
            {"kind": "endplate_defect_candidate", "level": "L4-L5", "confidence": 0.3},
        ],
    )
    other = _result(
        "alignment",
        flags=[
            {"kind": "retrolisthesis", "level": "L3-L4", "confidence": 0.3},
            {"kind": "bulge", "level": "L3-L4", "experimental": True, "confidence": 0.9},
        ],
    )
    items = collect({"endplates": res, "alignment": other}, _anchors())["items"]
    flags = {i["kind"]: i["experimental"] for i in items}
    assert flags == {
        "modic_suggestion": True,
        "endplate_defect_candidate": True,
        "retrolisthesis": False,
        "bulge": True,
    }


def test_validation_text_from_flag_or_meta_by_kind() -> None:
    res = _result(
        meta={"validation": {"bulge": "AUC 0.82 on SPIDER validation"}},
        flags=[
            {"kind": "bulge", "level": "L4-L5", "confidence": 0.8},
            {"kind": "other", "level": "L4-L5", "confidence": 0.8, "validation": "not validated"},
            {"kind": "none", "level": "L4-L5", "confidence": 0.8},
        ],
    )
    got = {i["kind"]: i["validation"] for i in collect({"h": res}, _anchors())["items"]}
    assert got == {"bulge": "AUC 0.82 on SPIDER validation", "other": "not validated", "none": ""}


def test_only_documented_keys_and_scalars_reach_the_section() -> None:
    res = _result(
        meta={"_private": "/home/someone/scan", "method": "m"},
        flags=[
            {
                "kind": "k",
                "level": "L4-L5",
                "confidence": 0.5,
                "path": "/home/someone/scan/1.dcm",
                "nested": {"a": 1},
                "value": {"not": "a scalar"},
                "note": "x" * 1000,
            }
        ],
    )
    sec = collect({"n": res}, _anchors())
    text = json.dumps(sec)
    assert "/home/someone" not in text and "nested" not in text and "_private" not in text
    (item,) = sec["items"]
    assert item["value"] is None and len(item["notes"]) <= 400
    assert set(item) == {
        "id",
        "level",
        "side",
        "structure",
        "analysis",
        "kind",
        "value",
        "confidence",
        "confidence_word",
        "experimental",
        "validation",
        "notes",
        "pos",
        "pos_source",
    }


def test_items_are_ordered_superior_first() -> None:
    flags = [{"kind": "k", "level": lab, "confidence": 0.5} for lab in ("L9-L10", "L5-S1", "L3-L4")]
    levels = [i["level"] for i in collect({"a": _result(flags=flags)}, _anchors())["items"]]
    assert levels == ["L3-L4", "L5-S1", "L9-L10"]


def test_the_list_is_capped_and_the_cap_is_said() -> None:
    flags = [{"kind": "k", "level": "L3-L4", "confidence": 0.5, "value": i} for i in range(MAX_ITEMS + 5)]
    sec = collect({"a": _result(flags=flags)}, _anchors())
    assert len(sec["items"]) == MAX_ITEMS
    assert [i["id"] for i in sec["items"]] == list(range(MAX_ITEMS))
    assert any("5 findings not shown" in w for w in sec["warnings"])


def test_collect_accepts_the_dict_form_of_a_result() -> None:
    res = _result(flags=[{"kind": "k", "level": "L4-L5", "confidence": 0.5}])
    a = collect({"a": res}, _anchors())
    b = collect({"a": res.to_dict()}, _anchors())
    assert a["items"] == b["items"]


def test_collect_without_anchors_has_no_numbering_or_positions() -> None:
    sec = collect({"a": _result(flags=[{"kind": "k", "level": "L4-L5"}])}, None)
    assert sec["numbering"] == "" and sec["items"][0]["pos"] is None


# ------------------------------------------------------------------ analyze files
def _analyze_doc(levels: list[str]) -> dict[str, object]:
    return {
        "numbering": "probable",
        "levels": levels,
        "analyses": {
            "discs": _result("discs", flags=[{"kind": "k", "level": "L4-L5", "confidence": 0.5}]).to_dict()
        },
    }


def test_analyze_output_is_read_back(tmp_path: Path) -> None:
    path = tmp_path / "findings.json"
    path.write_text(json.dumps(_analyze_doc(["L3-L4", "L4-L5", "L5-S1"])))
    results, notes = load_findings_file(path, _anchors())
    assert list(results) == ["discs"] and notes == []
    (item,) = collect(results, _anchors())["items"]
    assert item["level"] == "L4-L5"


def test_mismatched_levels_are_a_note_not_an_error(tmp_path: Path) -> None:
    path = tmp_path / "findings.json"
    path.write_text(json.dumps(_analyze_doc(["L1-L2"])))
    _, notes = load_findings_file(path, _anchors())
    assert notes and "other levels" in notes[0]


def test_a_file_that_is_not_an_analyze_output_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError, match="analyses"):
        load_findings_file(path, _anchors())
    assert results_from_analyze_json({"analyses": {"a": "not a dict"}}) == {}


# ------------------------------------------------------------------ payload and page
@pytest.fixture(scope="module")
def page_payload(phantom_study) -> dict:
    """The phantom payload with a hand-made findings section (no analyses run: fast and deterministic)."""
    res = {"demo": _result(flags=[{"kind": "bulge", "level": "L4-L5", "confidence": 0.8}])}
    return build_payload(phantom_study, _anchors(), title="t", findings=res)


def test_payload_has_findings_only_when_given(phantom_study) -> None:
    without = build_payload(phantom_study, None, title="t")
    assert without["findings"] is None
    with_f = build_payload(phantom_study, _anchors(), title="t", findings={})
    assert with_f["findings"]["items"] == [] and with_f["findings"]["format"] == FINDINGS_FORMAT


def test_findings_section_is_strict_json(page_payload) -> None:
    json.dumps(page_payload, allow_nan=False)
    assert page_payload["findings"]["items"][0]["pos"] == [0.0, 12.0, -40.0]


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.labels_for: list[str] = []
        self.controls: list[dict[str, str | None]] = []
        self.tags: list[str] = []
        self.external: list[str] = []
        self.th_scope_missing = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        self.tags.append(tag)
        if a.get("id"):
            self.ids.append(a["id"] or "")
        if tag == "label" and a.get("for"):
            self.labels_for.append(a["for"] or "")
        if tag in ("input", "select", "textarea"):
            self.controls.append(a)
        if tag == "th" and "scope" not in a:
            self.th_scope_missing += 1
        for k in ("src", "href", "action"):
            v = a.get(k) or ""
            if re.match(r"^(https?:)?//", v):
                self.external.append(v)


def _parse(html: str) -> _Page:
    p = _Page()
    p.feed(html)
    return p


def test_page_structure_with_findings(page_payload) -> None:
    html = render_html(page_payload)
    p = _parse(html)
    assert len(p.ids) == len(set(p.ids)), "duplicate element ids"
    for target in p.labels_for:
        assert target in p.ids, f"label for missing id {target}"
    for c in p.controls:
        assert c.get("id") or c.get("name"), f"form field without id or name: {c}"
    assert {"findings", "findbtn", "fx-structure", "fx-body", "fx-msg", "fx-legend", "fx-warn"} <= set(p.ids)
    assert p.th_scope_missing == 0
    assert p.external == []
    assert "<caption" in html and 'aria-labelledby="h-findings"' in html


def test_page_without_findings_still_has_a_valid_structure(phantom_study) -> None:
    p = _parse(render_html(build_payload(phantom_study, None, title="t")))
    assert len(p.ids) == len(set(p.ids))
    for c in p.controls:
        assert c.get("id") or c.get("name")


def test_payload_is_embedded_safely(page_payload) -> None:
    html = render_html(page_payload)
    m = re.search(r"window\.SPINESCOPE_DATA = (\{.*?\});</script>", html, re.S)
    assert m is not None
    assert json.loads(m.group(1))["findings"]["format"] == FINDINGS_FORMAT
    evil = _json_for_script({"s": "</script><script>alert(1)</script>"})
    assert "</script>" not in evil and "<" not in evil


def test_findings_add_a_small_amount_to_the_page(phantom_study) -> None:
    """Size budget: a finding is about 250 bytes of JSON; 500 findings add under 150 kB to a page."""
    flags = [
        {
            "kind": f"kind_{i % 7}",
            "level": "L4-L5",
            "value": i % 5,
            "confidence": 0.5,
            "note": "check the image",
        }
        for i in range(500)
    ]
    base = render_html(build_payload(phantom_study, _anchors(), title="t"))
    more = render_html(
        build_payload(phantom_study, _anchors(), title="t", findings={"a": _result(flags=flags)})
    )
    assert len(more) - len(base) < 150_000


def test_build_viewer_writes_the_panel_and_no_network_urls(phantom_study, tmp_path: Path) -> None:
    res = {"discs": _result("discs", flags=[{"kind": "k", "level": "L4-L5", "confidence": 0.5}])}
    path = build_viewer(phantom_study, tmp_path / "v", anchors=_anchors(), findings=res)
    html = path.read_text(encoding="utf-8")
    assert "fx-body" in html and not re.search(r"""(src|href)=["']https?://""", html)
    assert "https://" not in html.split("<script>window.SPINESCOPE_DATA")[0]


# ------------------------------------------------------------------ CLI
def test_viewer_findings_needs_labels(
    phantom_dir, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["viewer", str(phantom_dir["root"]), str(tmp_path / "v"), "--no-anchors", "--findings"])
    assert rc == 2
    assert "needs the level labels" in capsys.readouterr().err


def test_viewer_findings_from_an_analyze_file(phantom_dir, tmp_path: Path) -> None:
    root = str(phantom_dir["root"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["analyze", root, "--only", "discs"]) == 0
    analyze = tmp_path / "analyze.json"
    analyze.write_text(buf.getvalue())
    out = tmp_path / "v"
    assert main(["viewer", root, str(out), "--findings", str(analyze)]) == 0
    html = (out / "spinescope_viewer.html").read_text(encoding="utf-8")
    m = re.search(r"window\.SPINESCOPE_DATA = (\{.*?\});</script>", html, re.S)
    assert m is not None
    findings = json.loads(m.group(1))["findings"]
    assert findings is not None and findings["analyses"][0]["name"] == "discs"


def test_viewer_without_the_flag_has_no_findings(phantom_dir, tmp_path: Path) -> None:
    out = tmp_path / "plain"
    assert main(["viewer", str(phantom_dir["root"]), str(out), "--no-anchors"]) == 0
    html = (out / "spinescope_viewer.html").read_text(encoding="utf-8")
    m = re.search(r"window\.SPINESCOPE_DATA = (\{.*?\});</script>", html, re.S)
    assert m is not None and json.loads(m.group(1))["findings"] is None
