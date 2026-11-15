"""Viewer: planning of reformats, payload/export, privacy scan and build variants (phantom only)."""

from __future__ import annotations

import base64
import re
import zlib

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors
from spinescope.geometry import Stack
from spinescope.study import Study
from spinescope.testing.phantom import FAKE_PHI
from spinescope.viewer import build_viewer, render_html
from spinescope.viewer.build import _json_for_script
from spinescope.viewer.export import build_payload, encode_volume
from spinescope.viewer.reformat import (
    body_box,
    choose_source,
    inside_mask,
    plan_reformat,
    plan_reformats,
)


def _study_without(study: Study, *keys: str) -> Study:
    return Study(stacks={k: v for k, v in study.stacks.items() if k not in keys}, info=dict(study.info))


def _payload_of(html: str) -> dict:
    import json

    m = re.search(r"window\.SPINESCOPE_DATA = (\{.*?\});</script>", html, re.S)
    assert m, "payload not found in html"
    return json.loads(m.group(1))


# ------------------------------------------------------------------ reformat planning
def test_coronal_reformat_comes_from_axial_with_t2_preference(phantom_study):
    # the phantom has only a coronal localizer; ax_t2 and ax_t1 tie on coverage, T2 wins
    src, cov = choose_source(phantom_study, "cor")
    assert src.key == "ax_t2"
    assert 0.3 < cov <= 1.0


def test_reformat_geometry_is_exact_and_inside_body_box(phantom_study):
    ref = plan_reformat(phantom_study, "cor")
    assert ref is not None
    stk = ref.as_stack()
    assert stk.plane == "cor"
    assert np.all(np.diff(stk.positions) > 0)  # slices ascending along the normal
    main = [s for s in phantom_study.stacks.values() if not s.meta["role"]["localizer"]]
    lo, hi = body_box(main)
    corners = np.vstack([stk.corners(0), stk.corners(stk.n - 1)])
    tol = stk.row_mm  # pixel edges reach half a pixel past the centre grid
    assert np.all(corners >= lo - tol) and np.all(corners <= hi + tol)
    # the grid's own pixel centres map back to themselves
    pts = stk.pix_to_lps(3, 10.0, 20.0)
    k, r, c = stk.lps_to_pix(pts)
    assert (round(k, 6), round(r, 6), round(c, 6)) == (3.0, 10.0, 20.0)


def test_reformat_only_for_missing_planes(phantom_study):
    refs = plan_reformats(phantom_study)
    assert [r.plane for r in refs] == ["cor"]  # sagittal and axial exist in the phantom


def test_reformat_appears_for_missing_axial_and_sagittal(phantom_study):
    no_ax = _study_without(phantom_study, *[k for k in phantom_study if k.startswith("ax_")])
    planes = [r.plane for r in plan_reformats(no_ax)]
    assert "ax" in planes
    ax = plan_reformat(no_ax, "ax")
    assert ax is not None and ax.source_key.startswith("sag_")  # only sagittal data covers the body


def test_reformat_refused_below_min_coverage(phantom_study):
    from spinescope.config import Config

    strict = Config(viewer_mpr_min_coverage=1.01)
    assert plan_reformat(phantom_study, "cor", config=strict) is None


def test_inside_mask_matches_contains(phantom_study):
    rng = np.random.default_rng(0)
    for st in phantom_study.stacks.values():
        lo, hi = st.corners(0).min(axis=0), st.corners(st.n - 1).max(axis=0)
        pts = lo + (hi - lo) * rng.uniform(-0.2, 1.2, size=(120, 3))
        expect = np.array([st.contains(p) for p in pts])
        assert np.array_equal(inside_mask(st, pts), expect), st.key


# ------------------------------------------------------------------ payload / export
def test_default_panes_show_three_different_planes(phantom, phantom_study):
    payload = build_payload(phantom_study, Anchors.from_truth(phantom.truth), title="t")
    by_key = {s["key"]: s for s in payload["series"]}
    planes = [by_key[k]["plane"] for k in payload["panes"][:3]]
    assert sorted(planes) == ["ax", "cor", "sag"]
    assert len(payload["panes"]) == 4
    assert all(k in by_key for k in payload["panes"])


def test_voxel_encoding_roundtrip_integer_and_float():
    vol = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) * 3 - 7
    enc = encode_volume(vol)
    raw = zlib.decompress(base64.b64decode(enc["data"]))
    n = vol.size
    lo, hi = raw[:n], raw[n:]
    q = np.frombuffer(lo, np.uint8).astype(np.uint16) | (np.frombuffer(hi, np.uint8).astype(np.uint16) << 8)
    assert enc["scale"] == 1.0
    assert np.array_equal(q.reshape(vol.shape).astype(np.float64) + enc["vmin"], vol.astype(np.float64))
    # non-integer data is quantised to 16 bits over the range
    f = (np.linspace(-1.0, 3.0, 2 * 3 * 3)).reshape(2, 3, 3).astype(np.float32)
    enc_f = encode_volume(f)
    assert enc_f["scale"] == pytest.approx(4.0 / 65535)


def test_anchors_add_disc_labels_and_none_gives_none(phantom, phantom_study):
    with_a = build_payload(phantom_study, Anchors.from_truth(phantom.truth), title="t")
    without = build_payload(phantom_study, None, title="t")
    labels = [d["label"] for d in with_a["discs"]]
    assert sorted(labels) == sorted(phantom.discs.keys())
    zs = [d["center"][2] for d in with_a["discs"]]
    assert zs == sorted(zs, reverse=True)  # superior first
    assert without["discs"] == []
    # disc normals are unit and centres sit at the truth
    d0 = with_a["discs"][0]
    assert np.linalg.norm(d0["normal"]) == pytest.approx(1.0)
    assert np.allclose(d0["center"], phantom.truth["discs"][d0["label"]]["center"], atol=1e-6)


def test_payload_contains_no_private_meta(phantom_study):
    payload = build_payload(phantom_study, None, title="t")
    assert set(payload["info"]) <= {"Manufacturer", "ManufacturerModelName", "MagneticFieldStrength"}
    text = str(payload)
    assert "_sources" not in text and "series_uid_hash" not in text
    for s in payload["series"]:
        assert set(s["acq"]) <= {"RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "SliceThickness"}


def test_empty_study_is_an_error():
    with pytest.raises(ValueError):
        build_payload(Study(stacks={}), None, title="t")


# ------------------------------------------------------------------ build and privacy
def test_build_writes_one_offline_file(phantom, phantom_study, tmp_path):
    path = build_viewer(
        phantom_study, tmp_path / "v", anchors=Anchors.from_truth(phantom.truth), title="Demo"
    )
    html = path.read_text(encoding="utf-8")
    assert path.suffix == ".html"
    assert "<script src" not in html and "<link " not in html
    assert "http://" not in html and "https://" not in html
    assert "fetch(" not in html and "import(" not in html
    assert "Research and education only. Not for diagnosis." in html


def test_page_contains_no_phi(phantom, phantom_study, phantom_dir, tmp_path):
    path = build_viewer(phantom_study, tmp_path / "v", anchors=Anchors.from_truth(phantom.truth))
    html = path.read_text(encoding="utf-8")
    for tag, value in FAKE_PHI.items():
        if len(value) >= 4:
            assert value not in html, f"{tag} value leaked into the viewer"
    # no DICOM UIDs, no file-system paths of the input
    assert not re.search(r"\d+\.\d+\.\d+\.\d+\.\d+", html)
    assert str(phantom_dir["root"]) not in html
    assert "/tmp/" not in html and "/Users/" not in html


def test_title_is_escaped(phantom_study, tmp_path):
    html = build_viewer(phantom_study, tmp_path / "v", title="<script>alert(1)</script>").read_text("utf-8")
    assert "<script>alert(1)" not in html
    assert "&lt;script&gt;" in html


def test_json_for_script_cannot_close_the_tag():
    s = _json_for_script({"a": "</script><b>&"})
    assert "<" not in s and ">" not in s and "&" not in s


def test_build_is_deterministic(phantom, phantom_study, tmp_path):
    a = build_viewer(phantom_study, tmp_path / "a", anchors=Anchors.from_truth(phantom.truth)).read_bytes()
    b = build_viewer(phantom_study, tmp_path / "b", anchors=Anchors.from_truth(phantom.truth)).read_bytes()
    assert a == b


# ------------------------------------------------------------------ incomplete inputs
def test_single_sagittal_volume_still_gets_four_panes(phantom_study, tmp_path):
    only = Study(stacks={"sag_t2": phantom_study["sag_t2"]}, info={})
    html = build_viewer(only, tmp_path / "one").read_text("utf-8")
    payload = _payload_of(html)
    keys = {s["key"] for s in payload["series"]}
    assert len(payload["panes"]) == 4
    assert set(payload["panes"]) <= keys
    assert {s["plane"] for s in payload["series"] if s["reformat"]} == {"ax", "cor"}


def test_no_axial_no_coronal_study(phantom_study, tmp_path):
    study = _study_without(phantom_study, *[k for k in phantom_study if k.startswith(("ax_", "loc_"))])
    payload = _payload_of(build_viewer(study, tmp_path / "noax").read_text("utf-8"))
    keys = {s["key"] for s in payload["series"]}
    assert set(payload["panes"]) <= keys
    assert any(s["reformat"] and s["plane"] == "ax" for s in payload["series"])


def test_flipped_acquisition_still_builds(phantom_study, tmp_path):
    # reversed slice order and a flipped in-plane axis must not change the planning
    st = phantom_study["sag_t2"]
    flipped = Stack(
        key="sag_t2",
        vol=st.vol[::-1, :, ::-1].copy(),
        ipp=st.ipp[::-1].copy(),
        row_dir=-st.row_dir,
        col_dir=st.col_dir,
        row_mm=st.row_mm,
        col_mm=st.col_mm,
        thickness=st.thickness,
        meta=dict(st.meta),
    )
    study = Study(
        stacks={**{k: v for k, v in phantom_study.stacks.items() if k != "sag_t2"}, "sag_t2": flipped}
    )
    html = build_viewer(study, tmp_path / "flip").read_text("utf-8")
    assert "mpr_cor" in html


def test_render_html_accepts_minimal_payload():
    html = render_html({"title": "x", "series": [], "discs": [], "panes": [], "info": {}, "disclaimer": "d"})
    assert "<title>x</title>" in html
