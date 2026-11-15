"""Canary scan of every output surface: no identifying text from a study may reach an output.

The study (``_canary.write_canary_study``) carries a unique canary in every text attribute, a private tag, a
folder name, and the raw UIDs. Each surface below is run on it, and every byte it produces (text, JSON, CSV, HTML,
PNG pixels and PNG text chunks, log records, exception messages) is scanned for the canaries.

Not covered: burned-in text in the pixels (we do not OCR). See ``docs/privacy.md``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout, suppress
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _canary import PATH_CANARY, TEXT_CANARIES, canary_strings, write_canary_study
from report_factory import findings, l45_disc
from spinescope import load_study
from spinescope.anatomy import label_spine
from spinescope.anatomy.summary import labelling_summary, summary_lines
from spinescope.cli import main as cli_main
from spinescope.io.dicom import list_studies as io_list_studies
from spinescope.privacy import public_text
from spinescope.report.merge import merge
from spinescope.report.render import render_html, render_markdown
from spinescope.study import list_studies

PILImage = pytest.importorskip("PIL.Image")
from PIL import ImageDraw  # noqa: E402

pytest.importorskip("mcp")

from spinescope.ai.briefs import brief, reading_tasks  # noqa: E402
from spinescope.ai.mcp_server import _TOOL_NAMES, SpineTools, ToolError, _as_content  # noqa: E402
from spinescope.analysis import discs as discs_mod  # noqa: E402
from spinescope.analysis import endplates as endplates_mod  # noqa: E402
from spinescope.viewer import build_viewer  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:The value length")

Blobs = dict[str, bytes]


@pytest.fixture(scope="module")
def canary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return write_canary_study(tmp_path_factory.mktemp("canary_study"))


@pytest.fixture(scope="module")
def root(canary: dict[str, Any]) -> str:
    return str(canary["root"])


@pytest.fixture(scope="module")
def needles(canary: dict[str, Any]) -> list[str]:
    """Every string that must not appear: canaries, the folder name, the raw UIDs."""
    return sorted({*canary_strings(), PATH_CANARY, *canary["uids"]})


def _leaks(blob: bytes, needles: list[str]) -> list[str]:
    return [n for n in needles if n.encode() in blob]


def _png_blobs(data: bytes) -> list[bytes]:
    """Raw PNG bytes plus every text chunk PIL decodes (tEXt, zTXt, iTXt, EXIF, ...), as bytes."""
    out = [data]
    with PILImage.open(io.BytesIO(data)) as im:
        for k, v in im.info.items():
            out.append(f"{k}={v}".encode("utf-8", "replace"))
    return out


def _dump(obj: Any) -> bytes:
    return json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8")


def _files(directory: Path) -> Blobs:
    out: Blobs = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(directory))] = p.read_bytes()
    return out


_B64_RE = re.compile(rb"base64,([A-Za-z0-9+/=]{64,})")


def _check(blobs: Blobs, needles: list[str], surface: str) -> None:
    """Fail with the surface, the artefact and the canary for every leak.

    Embedded base64 images (the viewer and the report embed thumbnails) are decoded and scanned as PNG; the base64
    text itself is not scanned, since random base64 can spell any short digit string.
    """
    bad: list[str] = []
    for name, data in blobs.items():
        blob_list = _png_blobs(data) if name.lower().endswith(".png") else [data]
        for m in _B64_RE.finditer(data):
            with suppress(ValueError, OSError):
                blob_list += _png_blobs(base64.b64decode(m.group(1)))
        text_only = _B64_RE.sub(b"base64,", data)
        blob_list = [text_only if b is data else b for b in blob_list]
        for blob in blob_list:
            for n in _leaks(blob, needles):
                bad.append(f"{name}: {n[:40]}")
    assert not bad, f"{surface} leaks: {bad[:20]}"


# --------------------------------------------------------------------------- the surfaces
@pytest.fixture(scope="module")
def study(root: str):
    return load_study(root)


@pytest.fixture(scope="module")
def anchors(study):
    return label_spine(study)


def _surface_study(study) -> Blobs:
    blobs: Blobs = {
        "study.summary": study.summary().encode(),
        "study.info": _dump(study.info),
        "study.report": _dump(study.report),
        "study.keys": _dump(study.keys()),
    }
    for key in study:
        st = study[key]
        blobs[f"stack.summary[{key}]"] = st.summary().encode()
        # meta["dicom"] holds the raw header text in memory for the classifier; it is not an output. Outputs are
        # what the library serialises: role and geometry.
        blobs[f"stack.role[{key}]"] = _dump({k: st.meta.get(k) for k in ("role", "geometry")})
    return blobs


def _surface_list_studies(root: str) -> Blobs:
    return {
        "list_studies": _dump([s.as_dict() for s in list_studies(root)]),
        "io.list_studies": _dump([s.as_dict() for s in io_list_studies(root)]),
    }


def _surface_labelling(study, anchors) -> Blobs:
    doc = anchors.to_dict()
    return {
        "anchors.to_dict": _dump(doc),
        "anchors.to_json": anchors.to_json().encode(),
        "labelling_summary": _dump(labelling_summary(anchors)),
        "summary_lines": "\n".join(summary_lines(anchors)).encode(),
    }


def _surface_analysis(study, anchors, tmp: Path) -> Blobs:
    blobs: Blobs = {}
    for name, mod in (("discs", discs_mod), ("endplates", endplates_mod)):
        res = mod.analyze(study, anchors)
        out = tmp / name
        res.save(out)
        blobs.update({f"analysis.{name}:{k}": v for k, v in _files(out).items()})
        blobs[f"analysis.{name}.to_dict"] = _dump(res.to_dict())
    return blobs


def _surface_viewer(study, anchors, tmp: Path) -> Blobs:
    path = build_viewer(study, tmp / "viewer", anchors=anchors)
    return {"viewer.html": Path(path).read_bytes()}


def _surface_viewer_findings(study, anchors, tmp: Path) -> Blobs:
    """The findings panel, from the analyses run in memory and from an ``analyze`` file read back."""
    from spinescope.analysis.base import run_all
    from spinescope.viewer.findings_view import load_findings_file

    results = run_all(study, anchors)
    path = build_viewer(study, tmp / "viewer_findings", anchors=anchors, findings=results)
    analyze = tmp / "analyze.json"
    analyze.write_text(
        json.dumps(
            {"levels": anchors.disc_labels(), "analyses": {k: r.to_dict() for k, r in results.items()}}
        )
    )
    read_back, notes = load_findings_file(analyze, anchors)
    path2 = build_viewer(
        study, tmp / "viewer_from_file", anchors=anchors, findings=read_back, findings_notes=notes
    )
    return {
        "viewer_findings.html": Path(path).read_bytes(),
        "viewer_from_file.html": Path(path2).read_bytes(),
    }


def _surface_report() -> Blobs:
    a = findings("reader_a", levels={"L4-L5": {"disc": l45_disc(3)}})
    b = findings("reader_b", levels={"L4-L5": {"disc": l45_disc(4)}})
    c = merge([a, b])
    return {
        "report.markdown": render_markdown(c).encode(),
        "report.html": render_html(c).encode(),
        "report.consensus": _dump(c.to_dict()),
    }


def _surface_briefs(study, anchors) -> Blobs:
    blobs: Blobs = {"brief": brief(study, anchors).encode()}
    for t in reading_tasks(study, anchors):
        blobs[f"task[{t.reader}]"] = _dump(t.to_dict())
        blobs[f"task-prompt[{t.reader}]"] = t.prompt.encode()
    return blobs


def _surface_cli(root: str, tmp: Path) -> Blobs:
    """Every CLI command, stdout and stderr captured, artefacts read back."""
    blobs: Blobs = {}

    def run(name: str, argv: list[str]) -> None:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                rc = cli_main(argv)
            except SystemExit as exc:  # argparse exits on bad options
                rc = exc.code
        blobs[f"cli.{name}.stdout"] = out.getvalue().encode()
        blobs[f"cli.{name}.stderr"] = err.getvalue().encode()
        assert rc in (0, 2, None)

    # Labelling costs a few seconds per run, so the labelling commands run once each; the brief, tasks and viewer
    # commands run without it here (the anchors-driven text is covered by the briefs and MCP surfaces).
    run("inspect", ["inspect", root])
    run("inspect_json", ["inspect", root, "--json"])
    run("label_json", ["label", root, "--json"])
    run("label_files", ["label", root, "--out", str(tmp / "anchors.json"), "--overlay", str(tmp / "qa.png")])
    run("brief", ["brief", root, "--no-anchors", "--out", str(tmp / "brief.md")])
    run("tasks", ["tasks", root, "--no-anchors", "--outdir", str(tmp / "tasks")])
    run("viewer", ["viewer", root, str(tmp / "viewer_cli"), "--no-anchors"])
    run("verify_labels", ["verify-labels", root, "--outdir", str(tmp / "packet")])
    run("bad_path", ["inspect", str(tmp / f"{PATH_CANARY}_missing")])
    blobs.update({f"cli.files/{k}": v for k, v in _files(tmp).items()})
    return blobs


def _surface_mcp(root: str, tmp: Path) -> Blobs:
    """Every MCP tool with its arguments: JSON results, PNG images (pixels and text chunks), saved PNGs."""
    tools = SpineTools(output_dir=tmp / "png_out")
    blobs: Blobs = {}

    def record(name: str, value: Any) -> None:
        content = _as_content(value)  # images come back as Image objects, everything else as JSON or text
        for i, item in enumerate(content if isinstance(content, list) else [content]):
            data = getattr(item, "data", None)
            if isinstance(data, bytes):
                blobs[f"mcp.{name}#{i}.png"] = data
            elif isinstance(item, str):
                blobs[f"mcp.{name}#{i}"] = item.encode()
            else:
                blobs[f"mcp.{name}#{i}"] = _dump(item)

    opened = tools.open_study(root)
    sid = opened["study_id"]
    record("open_study", opened)
    record("list_series", tools.list_series(sid))
    record("get_anchors", tools.get_anchors(sid))
    record("level_slices", tools.level_slices(sid, "L4-L5") if _has(tools, sid, "L4-L5") else {})
    packet = tools.verify_labels(sid)
    record("verify_labels", packet)
    record(
        "apply_label_verification",
        tools.apply_label_verification(
            sid, json.dumps({"lowest_disc_is": "L5-S1", "missing": [], "extra": [], "transitional": None})
        ),
    )
    record("render_slice", tools.render_slice(sid, "sag_t2", 6))
    record("zoom", tools.zoom(sid, "sag_t2", 6, [60, 60], fov_mm=30))
    record("linked_views", tools.linked_views(sid, [0.0, 0.0, 0.0], crop_mm=60))
    record("reference_lines", tools.reference_lines(sid, "sag_t2", 6, ["ax_t2"]))
    record("measure_distance", tools.measure_distance(sid, "sag_t2", 6, [10, 10], [20, 20]))
    record("roi_stats", tools.roi_stats(sid, "sag_t2", 6, [60, 60], 4.0))
    record("reformat_sag", tools.reformat(sid, "sag"))
    record("reformat_cor", tools.reformat(sid, "cor"))
    record("reformat_ax", tools.reformat(sid, "ax"))
    record("build_viewer", tools.build_viewer(sid, str(tmp / "mcp_viewer")))
    record("brief", tools.brief(sid, reader="reader_z"))
    record("tasks", tools.tasks(sid))
    record("validate_findings", tools.validate_findings(json.dumps(findings("reader_a"))))
    fa = tmp / "reader_a.json"
    fb = tmp / f"{PATH_CANARY}_reader_b.json"
    fa.write_text(json.dumps(findings("reader_a", levels={"L4-L5": {"disc": l45_disc(3)}})))
    fb.write_text(json.dumps(findings("reader_b", levels={"L4-L5": {"disc": l45_disc(4)}})))
    record("merge_findings", tools.merge_findings([str(fa), str(fb)], outdir=str(tmp / "merged")))
    record("analyze", tools.analyze(sid, only=["discs", "endplates"]))
    blobs.update({f"mcp.files/{k}": v for k, v in _files(tmp / "png_out").items()})
    blobs.update({f"mcp.merged/{k}": v for k, v in _files(tmp / "merged").items()})
    blobs.update({f"mcp.mcp_viewer/{k}": v for k, v in _files(tmp / "mcp_viewer").items()})
    return blobs


def _has(tools: SpineTools, sid: str, level: str) -> bool:
    return level in tools.get_anchors(sid).get("discs", {})


def _surface_errors(root: str, tmp: Path) -> Blobs:
    """Forced errors: the messages (and the CLI's stderr) must not echo the path or any canary."""
    blobs: Blobs = {}
    tools = SpineTools()
    attempts: list[Callable[[], Any]] = [
        lambda: tools.open_study(str(tmp / PATH_CANARY / "nothing")),
        lambda: tools.open_study(str(tmp)),  # a folder with no DICOM
        lambda: tools.render_slice("0" * 12, "sag_t2", 0),
        lambda: tools.merge_findings([str(tmp / f"{PATH_CANARY}_absent.json")]),
        lambda: tools.apply_label_verification("0" * 12, "{}"),
    ]
    for i, fn in enumerate(attempts):
        try:
            fn()
        except ToolError as exc:
            blobs[f"error.tool#{i}"] = str(exc).encode()
        except Exception as exc:
            blobs[f"error.other#{i}"] = f"{type(exc).__name__}: {exc}".encode()
    # Library errors surface through the CLI: exercised with the canary path in argv.
    err = io.StringIO()
    with redirect_stderr(err):
        cli_main(["inspect", str(tmp / PATH_CANARY / "nothing")])
    blobs["error.cli"] = err.getvalue().encode()
    return blobs


# --------------------------------------------------------------------------- tests
def test_canary_study_really_carries_the_canaries(root: str, needles: list[str]) -> None:
    """The fixture is not vacuous: the canaries are in the input files (or the scan below proves nothing)."""
    import pydicom

    first = sorted(Path(root).rglob("*.dcm"))[0]
    ds = pydicom.dcmread(first, stop_before_pixels=True)
    for name, value in TEXT_CANARIES.items():
        if name in ("SeriesDescription", "ProtocolName"):
            assert value.split()[-1] in str(ds.get(name, "")), name
        else:
            assert str(ds.get(name, "")) == value, name
    assert PATH_CANARY in root
    assert len(needles) > 30


#: Descriptions the series classifier and the level-name readers understand: none of their words may be hidden.
CLASSIFIER_EXAMPLES = (
    "SAG T2 FSE",
    "AX T2 FS",
    "SAG T1 BRAVO",
    "SAG STIR",
    "T2 SPAIR-IR",
    "3D DIXON WATER",
    "AX DIXON IP",
    "SAG T1 MPRAGE",
    "SAG T2* GRE",
    "AX DWI ADC",
    "T1 FLAIR",
    "AX T1+C POST GD",
    "SAG T1 NON-CONTRAST",
    "SAG T2 TIRM",
    "T2 CHESS",
    "COR T1FS",
    "PROTON PD",
    "MEDIC",
    "EP2D_DIFF",
    "L4-5 OBL AX",
    "AX T1 SPGR",
    "FLASH 3D",
    "T2W FSE OBL",
    "3-PL LOC",
)


@pytest.mark.parametrize("text", CLASSIFIER_EXAMPLES)
def test_classifier_vocabulary_survives_scrubbing(text: str) -> None:
    assert "*" not in public_text(text).split(), public_text(text)  # "*" alone marks a dropped token


def test_public_text_keeps_protocol_words_and_drops_the_rest() -> None:
    assert public_text("SAG T2 ZQCANARY_SeriesDescription") == "SAG T2 *"
    assert public_text("OBL AX T2 L4-5 3-PL") == "OBL AX T2 L4-5 3-PL"
    assert public_text("Smithson_Jon T2 STIR 20010203") == "* T2 STIR *"
    assert public_text("L4-L5 T2* FS 3.5mm") == "L4-L5 T2* FS 3.5MM"
    assert public_text(None) == "" and public_text("") == ""
    assert len(public_text("T2 " * 100)) <= 80


@pytest.mark.parametrize(
    "surface",
    [
        "study",
        "list_studies",
        "labelling",
        "analysis",
        "viewer",
        "viewer_findings",
        "report",
        "briefs",
        "cli",
        "mcp",
        "errors",
    ],
)
def test_no_canary_in_surface(
    surface: str,
    root: str,
    study: Any,
    anchors: Any,
    needles: list[str],
    canary: dict[str, Any],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.DEBUG)  # DEBUG: the most verbose logs must not carry identifying text either
    # Text drawn onto images cannot be read back from the PNG, so record every string PIL draws instead.
    drawn: list[str] = []
    original_text = ImageDraw.ImageDraw.text

    def spy(self: ImageDraw.ImageDraw, xy: Any, text: Any, *args: Any, **kwargs: Any) -> Any:
        drawn.append(str(text))
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    if surface == "study":
        blobs = _surface_study(study)
    elif surface == "list_studies":
        blobs = _surface_list_studies(root)
    elif surface == "labelling":
        blobs = _surface_labelling(study, anchors)
    elif surface == "analysis":
        blobs = _surface_analysis(study, anchors, tmp_path)
    elif surface == "viewer":
        blobs = _surface_viewer(study, anchors, tmp_path)
    elif surface == "viewer_findings":
        blobs = _surface_viewer_findings(study, anchors, tmp_path)
    elif surface == "report":
        blobs = _surface_report()
    elif surface == "briefs":
        blobs = _surface_briefs(study, anchors)
    elif surface == "cli":
        blobs = _surface_cli(root, tmp_path)
    elif surface == "mcp":
        blobs = _surface_mcp(root, tmp_path)
    else:
        blobs = _surface_errors(root, tmp_path)
    blobs["drawn text on images"] = "\n".join(drawn).encode()
    if surface in ("mcp", "analysis"):  # the spy must really see the text these surfaces draw
        assert drawn, f"{surface}: no text was recorded, the drawing spy is not working"
    _check(blobs, needles, surface)
    logs = "\n".join(f"{r.levelname} {r.name} {r.getMessage()}" for r in caplog.records)
    assert not _leaks(logs.encode(), needles), f"log records of {surface} leak"


def test_needle_list_has_the_path_and_the_uids(needles: list[str]) -> None:
    """Sanity of the needle list itself: the folder name and the raw study UIDs are included."""
    assert PATH_CANARY in needles
    assert any(n.startswith("1.2.") for n in needles)


def test_every_mcp_tool_is_covered() -> None:
    """A new MCP tool must get a canary check: the scan above calls each name in _TOOL_NAMES."""
    covered = {
        "open_study",
        "list_series",
        "get_anchors",
        "level_slices",
        "verify_labels",
        "apply_label_verification",
        "render_slice",
        "zoom",
        "linked_views",
        "reference_lines",
        "measure_distance",
        "roi_stats",
        "reformat",
        "build_viewer",
        "brief",
        "tasks",
        "validate_findings",
        "merge_findings",
        "analyze",
    }
    assert set(_TOOL_NAMES) == covered


def test_render_reformat_png_has_no_text_metadata(root: str) -> None:
    tools = SpineTools()
    sid = tools.open_study(root)["study_id"]
    png = tools.reformat(sid, "ax").png
    with PILImage.open(io.BytesIO(png)) as im:
        assert {k: v for k, v in im.info.items() if k not in ("dpi", "gamma", "transparency")} == {}
    assert np.frombuffer(png[:8], dtype=np.uint8).tolist() == [137, 80, 78, 71, 13, 10, 26, 10]


def test_scanner_catches_a_planted_leak(needles: list[str]) -> None:
    """Negative control: the scanner must fail on a canary in text, in base64-embedded PNG and in a PNG text chunk."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    with pytest.raises(AssertionError):
        _check({"planted.json": b'{"series": "ZQCANARY_SeriesDescription"}'}, needles, "planted")
    img = io.BytesIO()
    meta = PngInfo()
    meta.add_text("Comment", "ZQCANARY_ImageComments")
    Image.new("RGB", (4, 4)).save(img, format="PNG", pnginfo=meta)
    with pytest.raises(AssertionError):
        _check({"planted.png": img.getvalue()}, needles, "planted png")
    embedded = base64.b64encode(img.getvalue()).decode()
    with pytest.raises(AssertionError):
        _check(
            {"planted.html": f'<img src="data:image/png;base64,{embedded}">'.encode()},
            needles,
            "planted html",
        )


def test_benchmark_error_text_drops_paths() -> None:
    from spinescope.benchmark import error_text

    text = error_text(ValueError("cannot read /Users/someone/data/ZQCANARY_Smith/a.mha: bad header"))
    assert "ZQCANARY" not in text and "/Users" not in text and "bad header" in text
