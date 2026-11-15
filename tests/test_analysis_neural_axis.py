"""Neural axis analysis (spinescope.analysis.neural_axis): conus surfacing, cysts, sac ratio and filum on the phantom.

The phantom tests use an oracle sagittal segmentation built from the phantom's analytic labels (the labeller is not
trained on the phantom; see docs/labeling/lab16_conus.md). No network, no weights needed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from spinescope.analysis import base
from spinescope.analysis import neural_axis as na
from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.anatomy.conus import refresh_conus
from spinescope.anatomy.ml.grid import intensity_window, plane_grid
from spinescope.anatomy.ml.labeller import SegmentedStack
from spinescope.config import DEFAULT
from spinescope.study import Study, from_stacks, load_study
from spinescope.testing.phantom import (
    ANNULUS,
    CORD,
    CORTEX,
    CSF,
    MARROW,
    MODIC1,
    NUCLEUS,
    PhantomSpec,
    SpinePhantom,
    write_standard_study,
)

_TO_CLASS = {MARROW: 1, CORTEX: 1, MODIC1: 1, ANNULUS: 2, NUCLEUS: 2, CSF: 3, CORD: 3}


def _oracle(phantom: SpinePhantom, contrast: str) -> SegmentedStack:
    """Canal (thecal sac incl. cord and any cyst) and body/disc classes from the phantom labels, on the 1 mm grids."""
    stack = phantom.render("sag", contrast)
    grids = [plane_grid(stack, k, 1.0) for k in range(stack.n)]
    h, w = grids[0].shape
    probs = np.zeros((stack.n, 4, h, w), np.float32)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for k, g in enumerate(grids):
        tissue = phantom.labels(g.lps_of_index(ii, jj).reshape(-1, 3)).reshape(h, w)
        cls = np.zeros((h, w), np.int64)
        for t, c in _TO_CLASS.items():
            cls[tissue == t] = c
        for c in range(4):
            probs[k, c] = cls == c
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=intensity_window(stack.vol))


def _anchors(ph: SpinePhantom, *, body_ap: float | None = None) -> Anchors:
    """Disc and body anchors from the phantom truth (no labeller). ``body_ap`` overrides the body AP depth (mm)."""
    discs = {
        n: Disc(
            center=[float(v) for v in d["center"]],
            normal=[float(v) for v in d["normal"]],
            height_mm=float(ph.spec.disc_height),
        )
        for n, d in ph.discs.items()
    }
    depth = 2.0 * ph.spec.body_half_depth if body_ap is None else body_ap
    vert = {
        n: Vertebra(center=[float(v) for v in ph.bodies[n]["center"]], body={"ap_depth_mm": depth})
        for n in ph.bodies
        if n[0] in "LTS"
    }
    return Anchors(discs=discs, vertebrae=vert, numbering={"status": "confirmed"})


def _truth_cyst_centre(ph: SpinePhantom, lat_mm: float) -> np.ndarray:
    sac = ph.bodies[ph._sacrum]
    return sac["center"] + sac["ap"] * ph.spec.canal_offset - sac["normal"] * 12.0 + sac["lat"] * lat_mm


def _axial(ph: SpinePhantom, contrast: str):  # type: ignore[no-untyped-def]
    centre = (np.asarray(ph.discs["L1-L2"]["center"]) + np.asarray(ph.discs["L5-S1"]["center"])) / 2
    return ph.render(
        "ax",
        contrast,
        n=36,
        spacing=5.0,
        thickness=4.0,
        rows=128,
        cols=128,
        px=1.4,
        center=centre + np.array([0.0, 10.0, 0.0]),
    )


def _l5s1_centre(ph: SpinePhantom, seg: SegmentedStack, anchors: Anchors) -> np.ndarray:
    centre = na.canal_point_at(seg, float(anchors.discs["L5-S1"].center[2]), DEFAULT)
    assert centre is not None
    return centre


# ------------------------------------------------------------------------------ conus
def _conus_anchors(z: float, level: str, conf: float, low: bool | None, fov_end: bool = False) -> Anchors:
    return Anchors(
        conus={
            "z": z,
            "level": level,
            "position": "middle third",
            "confidence": conf,
            "low_lying": low,
            "fov_end": fov_end,
            "experimental": True,
        },
        numbering={"status": "confirmed"},
    )


def test_conus_row_surfaces_the_experimental_locator_without_a_flag_below_threshold() -> None:
    rows, flags = na.conus_rows(_conus_anchors(-40.0, "L3", 0.45, True), DEFAULT)
    assert rows[0]["scope"] == "conus" and rows[0]["experimental"] is True
    assert rows[0]["level"] == "L3" and rows[0]["low_lying"] is True
    assert flags == []  # below the 0.6 threshold: the warning from the conus module says "too low to call"


def test_low_lying_flag_needs_a_tip_below_l2_l3_and_good_confidence() -> None:
    _, flags = na.conus_rows(_conus_anchors(-40.0, "L3", 0.7, True), DEFAULT)
    assert [f["kind"] for f in flags] == ["possibly low-lying conus"]
    _, flags_normal = na.conus_rows(_conus_anchors(60.0, "L1", 0.9, False), DEFAULT)
    assert flags_normal == []


def test_conus_not_located_gives_a_row_without_a_tip() -> None:
    rows, flags = na.conus_rows(Anchors(), DEFAULT)
    assert rows == [{"scope": "conus", "located": False, "experimental": True}] and flags == []


def test_low_lying_flag_is_dormant_under_the_experimental_cap() -> None:
    """The locator's confidence is capped at conus_max_confidence, below the flag threshold: no flag today."""
    anchors = Anchors(
        discs={"L2-L3": Disc(center=[0.0, 0.0, 30.0], normal=[0.0, 0.0, 1.0], height_mm=10.0)},
        conus={"z": 0.0, "confidence_raw": 0.95, "confidence": 0.95, "level": "L3"},
        numbering={"status": "confirmed"},
    )
    refresh_conus(anchors, DEFAULT)
    assert anchors.conus is not None
    assert (
        anchors.conus["confidence"] <= DEFAULT.conus_max_confidence < DEFAULT.conus_low_lying_min_confidence
    )
    _, flags = na.conus_rows(anchors, DEFAULT)
    assert flags == []


# ------------------------------------------------------------------------------ 2D blob rules
def _disc_mask(shape: tuple[int, int], r: int, c: int, rad: float) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - r) ** 2 + (xx - c) ** 2 <= rad**2


def test_blob_rules_keep_an_oval_and_reject_a_streak_and_an_edge_blob() -> None:
    shape = (60, 60)
    csf_like = _disc_mask(shape, 30, 30, 6.0)  # 12 mm disc
    csf_like[5:9, 5:45] = True  # streak: thin and elongated
    csf_like[40:55, 0:10] = True  # touches the left edge of the grid
    eligible = np.ones(shape, bool)
    blobs = na.cyst_blobs_2d(csf_like, eligible, np.zeros(shape, bool), 1.0, DEFAULT)
    assert len(blobs) == 1
    _centroid, major, _minor, sol, _mask = blobs[0]
    assert 10.0 <= major <= 14.0 and sol > 0.8


def test_blob_rules_exclude_the_sac_region() -> None:
    shape = (40, 40)
    csf_like = _disc_mask(shape, 20, 20, 6.0)
    excl = _disc_mask(shape, 20, 20, 8.0)
    assert na.cyst_blobs_2d(csf_like, np.ones(shape, bool), excl, 1.0, DEFAULT) == []


# ------------------------------------------------------------------------------ 2. perineural cysts
def test_planted_cyst_is_found_with_its_size_and_position() -> None:
    ph = SpinePhantom(PhantomSpec(perineural_cysts=((12.0, 10.0),)))
    seg = _oracle(ph, "stir")
    rows, warns, kept = na.sagittal_cysts(seg, _anchors(ph), DEFAULT, contrast="stir")
    assert len(rows) == 1 and kept
    row = rows[0]
    assert abs(row["size_mm"] - 10.0) <= 2.5
    truth = _truth_cyst_centre(ph, 12.0)
    assert abs(row["z_mm"] - truth[2]) <= 3.0
    assert row["large"] is False and row["confidence"] <= DEFAULT.neural_axis_max_confidence
    assert row["wording"] == "possible perineural cyst, likely incidental"
    assert warns == []


def test_default_phantom_has_no_cyst_call() -> None:
    ph = SpinePhantom(PhantomSpec())
    rows, warns, _ = na.sagittal_cysts(_oracle(ph, "stir"), _anchors(ph), DEFAULT, contrast="stir")
    assert rows == []
    assert any("no perineural cyst candidate" in w for w in warns)


def test_large_cyst_is_worded_for_correlation() -> None:
    ph = SpinePhantom(PhantomSpec(perineural_cysts=((12.0, 24.0),)))
    rows, _, _ = na.sagittal_cysts(_oracle(ph, "stir"), _anchors(ph), DEFAULT, contrast="stir")
    assert rows and rows[0]["large"] is True
    assert "consider correlation" in rows[0]["wording"]


def test_cysts_are_not_assessed_when_csf_and_marrow_look_alike() -> None:
    ph = SpinePhantom(PhantomSpec(perineural_cysts=((12.0, 10.0),)))
    cfg = dataclasses.replace(DEFAULT, neural_axis_cyst_min_contrast_frac=0.99)
    rows, warns, _ = na.sagittal_cysts(_oracle(ph, "stir"), _anchors(ph), cfg, contrast="stir")
    assert rows == [] and any("look alike" in w for w in warns)


def test_cysts_need_the_l5_s1_anchor() -> None:
    ph = SpinePhantom(PhantomSpec())
    anchors = _anchors(ph)
    anchors.discs.pop("L5-S1")
    rows, warns, _ = na.sagittal_cysts(_oracle(ph, "stir"), anchors, DEFAULT, contrast="stir")
    assert rows == [] and any("no L5-S1 disc" in w for w in warns)


# ------------------------------------------------------------------------------ dural sac ratio
@pytest.mark.parametrize(("canal_radius", "above"), [(6.0, False), (9.0, True)])
def test_dural_sac_ratio_follows_the_sac_width_and_the_limit(canal_radius: float, above: bool) -> None:
    ph = SpinePhantom(PhantomSpec(canal_radius=canal_radius))
    rows, warns = na.dural_sac_ratios(_oracle(ph, "stir"), _anchors(ph), DEFAULT)
    l5 = next(r for r in rows if r["level"] == "L5")
    # sac AP diameter is about twice the canal radius; the body AP depth is 2 x body_half_depth (32 mm)
    assert abs(l5["ratio"] - 2 * canal_radius / 32.0) <= 0.08
    assert l5["above_limit"] is above and l5["limit"] == DEFAULT.neural_axis_dsr_l5
    assert warns == [] or all("S1" in w for w in warns)


def test_dural_sac_ratio_without_body_depth_is_not_assessed() -> None:
    ph = SpinePhantom(PhantomSpec())
    anchors = _anchors(ph)
    for v in anchors.vertebrae.values():
        v.body = None
    rows, warns = na.dural_sac_ratios(_oracle(ph, "stir"), anchors, DEFAULT)
    assert rows == [] and any("no body AP depth" in w for w in warns)


# ------------------------------------------------------------------------------ 3. filum (axial)
@pytest.mark.parametrize(("radius", "thickened"), [(1.0, None), (2.5, True)])
def test_filum_thickness_on_axial_t2(radius: float, thickened: bool | None) -> None:
    """A 2 mm filum is within one pixel of the 2 mm limit: indeterminate (None), not thickened."""
    ph = SpinePhantom(PhantomSpec(filum_radius=radius))
    seg = _oracle(ph, "stir")
    anchors = _anchors(ph)
    row, _outline, reason, _patch = na.filum_at_l5s1(
        _axial(ph, "t2"), _l5s1_centre(ph, seg, anchors), DEFAULT, fat=False
    )
    assert row is not None, reason
    assert abs(row["diameter_mm"] - 2 * radius) <= 1.5
    assert row["thickened"] is thickened


def test_fatty_filum_is_found_on_t1_and_cord_like_filum_is_not() -> None:
    fatty = SpinePhantom(PhantomSpec(filum_radius=1.5, filum_fat=True))
    row, _, _, _ = na.filum_at_l5s1(
        _axial(fatty, "t1"), _l5s1_centre(fatty, _oracle(fatty, "stir"), _anchors(fatty)), DEFAULT, fat=True
    )
    assert row is not None and abs(row["diameter_mm"] - 3.0) <= 1.5
    cord = SpinePhantom(PhantomSpec(filum_radius=1.0))  # cord-like filum is brighter than CSF on T1 too
    row_cord, _, reason, _ = na.filum_at_l5s1(
        _axial(cord, "t1"), _l5s1_centre(cord, _oracle(cord, "stir"), _anchors(cord)), DEFAULT, fat=True
    )
    assert row_cord is None and "no fatty filum candidate" in reason


def test_filum_off_the_axial_slab_is_not_assessed() -> None:
    ph = SpinePhantom(PhantomSpec(filum_radius=1.0))
    seg = _oracle(ph, "stir")
    far = _l5s1_centre(ph, seg, _anchors(ph)) + np.array([0.0, 0.0, 200.0])
    row, _, reason, _ = na.filum_at_l5s1(_axial(ph, "t2"), far, DEFAULT, fat=False)
    assert row is None and "outside the axial" in reason


# ------------------------------------------------------------------------------ analysis wiring
def _study(tmp_path: Path, spec: PhantomSpec) -> tuple[Study, SpinePhantom]:
    ph = SpinePhantom(spec)
    root = write_standard_study(ph, tmp_path / "dcm", slab_contrasts=("t2",))["root"]
    return load_study(root), ph


def test_analyze_on_a_phantom_study_reports_cyst_filum_and_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = PhantomSpec(perineural_cysts=((12.0, 10.0),), filum_radius=2.5)
    study, ph = _study(tmp_path, spec)
    seg = _oracle(ph, "stir")
    monkeypatch.setattr(na, "_sagittal_segmentation", lambda study, cfg, warns: (seg, "stir"))
    anchors = _anchors(ph)
    anchors.conus = {
        "z": float(ph.conus_z),
        "level": "L1",
        "position": "middle third",
        "confidence": 0.4,
        "low_lying": False,
        "experimental": True,
    }
    res = na.analyze(study, anchors, DEFAULT)
    scopes = {r["scope"] for r in res.rows}
    assert {"conus", "perineural_cyst", "dural_sac_ratio", "filum"} <= scopes
    assert "cyst_sagittal" in res.figures and "filum_t2" in res.figures
    assert any("syrinx" in w for w in res.warnings)
    json.dumps(res.to_dict(), allow_nan=False)  # JSON-clean (no NaN, no numpy types)


def test_analyze_saves_csv_json_and_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study, ph = _study(tmp_path, PhantomSpec(perineural_cysts=((12.0, 10.0),)))
    seg = _oracle(ph, "stir")
    monkeypatch.setattr(na, "_sagittal_segmentation", lambda study, cfg, warns: (seg, "stir"))
    paths = na.analyze(study, _anchors(ph), DEFAULT).save(tmp_path / "out")
    names = {Path(p).name for p in paths}
    assert "neural_axis.csv" in names and "neural_axis.json" in names
    assert any(n.startswith("neural_axis_cyst_sagittal") for n in names)


def test_missing_sagittal_series_warns_and_does_not_raise(tmp_path: Path) -> None:
    ph = SpinePhantom(PhantomSpec())
    study = from_stacks([_axial(ph, "t2")])
    res = na.analyze(study, _anchors(ph), DEFAULT)
    assert any("no sagittal STIR or T2 series" in w for w in res.warnings)
    assert all(r.get("scope") != "perineural_cyst" for r in res.rows)


def test_neural_axis_is_registered_and_runs_through_the_registry(tmp_path: Path) -> None:
    assert "neural_axis" in base.available()
    study, ph = _study(tmp_path, PhantomSpec())
    out = base.run_all(study, _anchors(ph), DEFAULT, names=["neural_axis"])
    assert out["neural_axis"].name == "neural_axis"
    assert not any(w.startswith("analysis failed") for w in out["neural_axis"].warnings)
