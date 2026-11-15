"""MCP server tools (spinescope.ai.mcp_server): called directly on the phantom, plus one stdio smoke test."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from report_factory import findings, l45_disc

pytest.importorskip("mcp")

from spinescope.ai.mcp_server import _TOOL_NAMES, RenderResult, SpineTools, ToolError, build_server

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PHI = ("PHANTOM^SPINE", "PHANTOM-0001", "DOE^JANE", "Phantom General Hospital")


@pytest.fixture(scope="module")
def root(phantom_dir) -> str:
    return str(phantom_dir["root"])


@pytest.fixture(scope="module")
def tools(tmp_path_factory) -> SpineTools:
    return SpineTools(output_dir=tmp_path_factory.mktemp("mcp_out"))


@pytest.fixture(scope="module")
def sid(tools: SpineTools, root: str) -> str:
    return tools.open_study(root)["study_id"]


def _assert_clean(obj: object, root: str) -> None:
    text = json.dumps(obj, default=str)
    assert root not in text
    for s in PHI:
        assert s not in text


def test_open_study_returns_opaque_id_and_public_summary(tools, root):
    out = tools.open_study(root)
    assert len(out["study_id"]) == 12 and int(out["study_id"], 16) >= 0
    assert out["n_series"] == len(out["series"]) >= 10
    assert set(out["info"]) <= {
        "Modality",
        "Manufacturer",
        "ManufacturerModelName",
        "MagneticFieldStrength",
        "BodyPartExamined",
        "PatientPosition",
    }
    _assert_clean(out, root)


def test_open_study_errors_never_echo_the_path(tools, tmp_path):
    missing = str(tmp_path / "does-not-exist-PATIENTNAME")
    with pytest.raises(ToolError) as e1:
        tools.open_study(missing)
    assert "PATIENTNAME" not in str(e1.value) and missing not in str(e1.value)
    junk = tmp_path / "junk.nii"
    junk.write_bytes(b"not an image")
    with pytest.raises(ToolError) as e2:
        tools.open_study(str(junk))
    assert str(junk) not in str(e2.value)
    assert "could not read the study" in str(e2.value)


def test_unknown_study_and_series_are_tool_errors(tools, sid):
    with pytest.raises(ToolError, match="unknown study_id"):
        tools.list_series("000000000000")
    with pytest.raises(ToolError, match="unknown series key"):
        tools.render_slice(sid, "nope", 0)
    with pytest.raises(ToolError, match="out of range"):
        tools.render_slice(sid, "sag_t2", 999)


def test_list_series_and_anchors(tools, sid, root):
    rows = tools.list_series(sid)
    assert {"sag_t2", "ax_t2"} <= {r["key"] for r in rows}
    anchors = tools.get_anchors(sid)
    assert anchors["source_stack"] == "sag_t2"
    assert anchors["discs"] and all("confidence" in d for d in anchors["discs"].values())
    assert tools.get_anchors(sid) == anchors  # cached: the same anchors on a repeat call
    _assert_clean(anchors, root)


def test_render_slice_returns_png_and_saves_it(tools, sid):
    r = tools.render_slice(sid, "sag_t2", 6)
    assert isinstance(r, RenderResult) and r.png.startswith(PNG_MAGIC)
    assert r.saved_to and Path(r.saved_to).read_bytes() == r.png
    assert r.meta["key"] == "sag_t2" and r.meta["k"] == 6
    r2 = tools.render_slice(sid, "ax_t2", 10, window=[0, 500])
    assert r2.meta["window"] == [0.0, 500.0]
    with pytest.raises(ToolError):
        tools.render_slice(sid, "sag_t2", 6, window=[1, 2, 3])


def test_zoom_linked_views_and_reference_lines(tools, sid):
    z = tools.zoom(sid, "sag_t2", 6, [96, 96], fov_mm=60)
    assert z.png.startswith(PNG_MAGIC) and z.meta["fov_mm"] == 60
    with pytest.raises(ToolError):
        tools.zoom(sid, "sag_t2", 6, [9999, 0])
    lv = tools.linked_views(sid, [0.0, 0.0, 0.0])
    assert lv.png.startswith(PNG_MAGIC)
    keys = {v["key"] for v in lv.meta["views"]}
    assert {"sag_t2", "ax_t2"} <= keys
    ref = tools.reference_lines(sid, "sag_t2", 6, ["ax_t2"])
    assert ref.png.startswith(PNG_MAGIC)
    with pytest.raises(ToolError):
        tools.reference_lines(sid, "sag_t2", 6, [])


def test_measure_distance_is_exact_in_mm(tools, sid):
    st = tools._study(sid)["sag_t2"]
    d = tools.measure_distance(sid, "sag_t2", 6, [10, 10], [10, 20])
    assert d["distance_mm"] == pytest.approx(10 * st.col_mm, abs=0.01)
    diag = tools.measure_distance(sid, "sag_t2", 6, [0, 0], [3, 4])
    assert diag["distance_mm"] == pytest.approx(5 * st.row_mm, abs=0.02)  # 3 rows and 4 cols: 5 px in-plane
    with pytest.raises(ToolError):
        tools.measure_distance(sid, "sag_t2", 6, [1], [2, 3])


def test_roi_stats_matches_numpy_and_rejects_empty_roi(tools, sid):
    st = tools._study(sid)["sag_t2"]
    out = tools.roi_stats(sid, "sag_t2", 6, [96, 96], 6.0)
    rows, cols = np.mgrid[0 : st.shape[1], 0 : st.shape[2]]
    mask = np.hypot((rows - 96) * st.row_mm, (cols - 96) * st.col_mm) <= 6.0
    vals = st.vol[6][mask].astype(np.float64)
    assert out["n"] == vals.size
    assert out["mean"] == pytest.approx(vals.mean(), abs=1e-2)
    assert out["units"].startswith("raw MR signal")
    with pytest.raises(ToolError):
        tools.roi_stats(sid, "sag_t2", 6, [96, 96], 0.0)


@pytest.mark.parametrize(
    "args",
    [
        {"plane": "sag"},
        {"plane": "cor"},
        {"plane": "ax"},
        {"plane": "sag", "at_mm": 0.0, "window": "disc/CSF"},
        {"plane": "disc", "label": "L4-L5"},
    ],
)
def test_reformat_planes_return_png(tools, sid, args):
    r = tools.reformat(sid, **args)
    assert r.png.startswith(PNG_MAGIC)
    assert r.meta["plane"] == args["plane"] and r.meta["source_key"]
    assert 0.0 < r.meta["fraction_inside"] <= 1.0


def test_reformat_argument_errors(tools, sid):
    with pytest.raises(ToolError):
        tools.reformat(sid, "oblique")
    with pytest.raises(ToolError, match="needs label"):
        tools.reformat(sid, "disc")
    with pytest.raises(ToolError, match="no disc"):
        tools.reformat(sid, "disc", label="L9-S9")


def test_build_viewer_writes_html_without_phi(tools, sid, root, tmp_path):
    out = tools.build_viewer(sid, str(tmp_path / "viewer"))
    text = Path(out["viewer"]).read_text(encoding="utf-8")
    assert "<html" in text.lower() and root not in text
    for s in PHI:
        assert s not in text


def test_validate_findings_and_merge_two_readers(tools, tmp_path):
    good = findings("reader_a", levels={"L4-L5": {"disc": l45_disc(3)}})
    res = tools.validate_findings(json.dumps(good))
    assert res["ok"] is True and res["errors"] == []
    bad = tools.validate_findings(json.dumps({"schema_version": "1.0", "reader": "x"}))
    assert bad["ok"] is False and bad["errors"]
    assert tools.validate_findings("{not json")["ok"] is False

    a = tmp_path / "reader_a.json"
    b = tmp_path / "reader_b.json"
    a.write_text(json.dumps(good))
    b.write_text(json.dumps(findings("reader_b", levels={"L4-L5": {"disc": l45_disc(4)}})))
    merged = tools.merge_findings([str(a), str(b)], outdir=str(tmp_path / "merged"))
    assert sorted(merged["consensus"]["readers"][i]["reader"] for i in range(2)) == ["reader_a", "reader_b"]
    assert "reader_a" in merged["markdown"] and str(tmp_path) not in merged["markdown"]
    assert (tmp_path / "merged" / "consensus.json").exists()
    assert str(tmp_path) not in json.dumps(merged["consensus"])
    assert merged["consensus"]["disagreements"]  # pfirrmann 3 vs 4 disagrees
    with pytest.raises(ToolError):
        tools.merge_findings([str(tmp_path / "missing.json")])


def test_brief_and_tasks_tools(tools, sid, root):
    text = tools.brief(sid, reader="reader_z")
    assert "Reader name: reader_z." in text
    _assert_clean(text, root)
    tasks = tools.tasks(sid)
    assert any(t["kind"] == "verify" and t["depends_on"] == ["merge"] for t in tasks)
    _assert_clean(tasks, root)


def test_server_registers_every_tool():
    server = build_server(SpineTools())
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == set(_TOOL_NAMES)


def test_stdio_smoke(root):
    """Spawn the server over stdio, use the real MCP client, open a study and render one slice."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "spinescope.ai.mcp_server"])

    async def run() -> tuple[set[str], str, str]:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            opened = await session.call_tool("open_study", {"path": root})
            sid = json.loads(opened.content[0].text)["study_id"]
            img = await session.call_tool("render_slice", {"study_id": sid, "key": "sag_t2", "k": 6})
            return {t.name for t in tools.tools}, sid, img.content[0].type

    names, sid, kind = asyncio.run(asyncio.wait_for(run(), timeout=180))
    assert set(_TOOL_NAMES) <= names
    assert len(sid) == 12
    assert kind == "image"
