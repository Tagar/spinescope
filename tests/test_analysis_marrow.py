from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from _studies import keep_contrasts, phantom_study, stacks_flipped
from spinescope.analysis import base, marrow
from spinescope.analysis.base import Result
from spinescope.anatomy import Anchors
from spinescope.config import DEFAULT
from spinescope.geometry import Stack
from spinescope.study import Study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

PLANTED = (
    ("L3", "haemangioma", (0.0, 0.0, 0.0), 10.0),
    ("L1", "focal_fat", (0.0, 0.0, 0.0), 10.0),
    ("L2", "bone_island", (0.0, 0.0, 0.0), 8.0),
    ("L4", "atypical", (0.0, 0.0, 0.0), 10.0),
)


def _focus(res: Result, vertebra: str) -> dict:
    hits = [r for r in res.rows if r["scope"] == "focus" and r["vertebra"] == vertebra]
    assert len(hits) == 1, (
        vertebra,
        [(r["vertebra"], r["class"]) for r in res.rows if r["scope"] == "focus"],
    )
    return hits[0]


def _foci(res: Result) -> list[dict]:
    return [r for r in res.rows if r["scope"] == "focus"]


def _run(spec: PhantomSpec, **study_kw) -> tuple[SpinePhantom, Result]:
    ph = SpinePhantom(spec)
    res = marrow.analyze(phantom_study(ph, **study_kw), Anchors.from_truth(ph.truth))
    return ph, res


@pytest.fixture(scope="module")
def planted() -> tuple[SpinePhantom, Result]:
    return _run(PhantomSpec(marrow_lesions=list(PLANTED), basivertebral_vein=True, diffuse_low_t1=("L5",)))


# ----------------------------------------------------------------------- planted lesions (ground truth)
def test_each_planted_lesion_gets_its_class(planted):
    _, res = planted
    assert _focus(res, "L3")["class"] == "haemangioma-like"
    assert _focus(res, "L1")["class"] == "focal fat-like"
    assert _focus(res, "L2")["class"] == "bone island-like"
    assert _focus(res, "L4")["class"] == "atypical"
    assert len(_foci(res)) == 4  # nothing else, including the basivertebral vein in every body


def test_sizes_match_the_planted_diameters(planted):
    _, res = planted
    # size is the largest bounding-box side of the thresholded focus. The 1 mm smoothing and the 1.2 / 0.8 ratio cut
    # trim the edge of a round focus and a pixel grid of 1.2 mm adds one, so the sizes land within about 2 mm
    for vertebra, dia in (("L3", 10.0), ("L1", 10.0), ("L2", 8.0), ("L4", 10.0)):
        size = _focus(res, vertebra)["size_mm"]
        assert abs(size - dia) <= 2.5, (vertebra, size)


def test_signal_ratios_follow_the_phantom_intensities(planted):
    _, res = planted
    # phantom ratios to marrow (720 T1, 470 T2, 130 STIR): haemangioma 950/720, 900/470, 200/130
    haem = _focus(res, "L3")
    assert haem["T1_ratio"] == pytest.approx(950 / 720, abs=0.08)
    assert haem["T2_ratio"] == pytest.approx(900 / 470, abs=0.15)
    assert haem["STIR_ratio"] == pytest.approx(200 / 130, abs=0.2)
    # fat STIR 50 / 130 is suppressed, which is what separates focal fat from haemangioma without fat fraction
    assert _focus(res, "L1")["STIR_ratio"] < DEFAULT.marrow_stir_fat_ratio
    assert _focus(res, "L1")["class"] == "focal fat-like"


def test_focal_flags_are_off_by_default_and_the_rows_remain(planted):
    _, res = planted
    assert not [f for f in res.flags if f["kind"] == "atypical_focal_lesion"]
    assert _focus(res, "L4")["class"] == "atypical"
    assert res.meta["focal_flags"].startswith("off")


def test_atypical_flag_when_switched_on_is_capped_and_worded(planted):
    ph, _ = planted
    on = dataclasses.replace(DEFAULT, marrow_focal_flags=True)
    res = marrow.analyze(phantom_study(ph), Anchors.from_truth(ph.truth), on)
    atyp = [f for f in res.flags if f["kind"] == "atypical_focal_lesion"]
    assert [f["vertebra"] for f in atyp] == ["L4"]
    assert atyp[0]["note"] == marrow.ATYPICAL_TEXT
    assert atyp[0]["size_mm"] >= DEFAULT.marrow_min_diameter_mm
    assert all(f["confidence"] <= DEFAULT.marrow_focal_max_confidence for f in res.flags)
    assert res.meta["focal_flags"] == "on"


def test_truth_records_the_planted_lesions(planted):
    ph, _ = planted
    lesions = ph.truth["marrow"]["lesions"]
    assert [les["kind"] for les in lesions] == ["haemangioma", "focal_fat", "bone_island", "atypical"]
    assert ph.truth["marrow"]["basivertebral_vein"] is True
    assert ph.truth["marrow"]["diffuse_low_t1"] == ["L5"]


# ----------------------------------------------------------------------- negatives (must not be called)
def test_normal_marrow_gives_no_focus_and_no_flag():
    _, res = _run(PhantomSpec())
    assert _foci(res) == []
    assert [f for f in res.flags if f["kind"] == "atypical_focal_lesion"] == []


def test_normal_marrow_robust_to_noise_lordosis_and_field_strength():
    variants = [
        PhantomSpec(noise=0.05, seed=3),
        PhantomSpec(segment_lordosis=(5.0, 8.0, 10.0, 12.0, 14.0, 18.0), field_strength=3.0),
    ]
    for spec in variants:
        _, res = _run(spec)
        assert _foci(res) == [], spec
        assert not [f for f in res.flags if f["kind"] == "atypical_focal_lesion"], spec


def test_basivertebral_vein_and_modic_band_are_not_called():
    # the vein is in every body; the Modic-type band (T1 low, T2 high) sits at the L4 inferior endplate
    _, res = _run(PhantomSpec(basivertebral_vein=True, modic1=[("L4-L5", "inf", "L")]))
    assert _foci(res) == []


def test_schmorl_node_is_not_called():
    _, res = _run(PhantomSpec(schmorl=[("L3", "inf"), ("L2", "sup")]))
    assert _foci(res) == []


def test_partial_volume_on_thick_slices_is_not_called():
    # 8 mm slices with 8 mm thickness average the body edges and the endplates into the marrow
    ph = SpinePhantom(PhantomSpec())
    stacks = [
        ph.render("sag", c, n=9, spacing=8.0, thickness=8.0, rows=192, cols=192, px=1.2)
        for c in ("t2", "stir", "t1")
    ]
    from spinescope.study import from_stacks

    res = marrow.analyze(from_stacks(stacks), Anchors.from_truth(ph.truth))
    assert _foci(res) == []


def test_diffuse_low_t1_only_in_the_altered_body(planted):
    _, res = planted
    diffuse = [r for r in res.rows if r["scope"] == "body"]
    low = [r["vertebra"] for r in diffuse if r["T1_marrow_over_disc"] < DEFAULT.marrow_diffuse_disc_ratio]
    assert low == ["L5"]
    normal = [r["T1_marrow_over_disc"] for r in diffuse if r["vertebra"] != "L5"]
    assert normal and min(normal) > 1.5  # marrow sits well above the disc in normal bodies (Vande Berg 2005)
    flags = [f for f in res.flags if f["kind"] == "diffuse_t1_low"]
    assert [f["vertebra"] for f in flags] == ["L5"]
    assert flags[0]["note"] == marrow.DIFFUSE_TEXT
    assert flags[0]["confidence"] <= DEFAULT.marrow_diffuse_max_confidence


# ----------------------------------------------------------------------- robustness of the input
def test_flipped_acquisition_gives_the_same_foci(planted):
    ph, res = planted
    flipped = stacks_flipped(phantom_study(ph))
    res_f = marrow.analyze(flipped, Anchors.from_truth(ph.truth))
    assert sorted((r["vertebra"], r["class"]) for r in _foci(res_f)) == sorted(
        (r["vertebra"], r["class"]) for r in _foci(res)
    )


def test_missing_t2_warns_and_calls_nothing():
    ph = SpinePhantom(PhantomSpec(marrow_lesions=list(PLANTED)))
    res = marrow.analyze(keep_contrasts(phantom_study(ph), ("t1", "stir")), Anchors.from_truth(ph.truth))
    assert any("no sagittal T2" in w for w in res.warnings)
    assert _foci(res) == []


def test_missing_t1_warns_and_measures_no_diffuse_signal():
    ph = SpinePhantom(PhantomSpec(diffuse_low_t1=("L5",)))
    res = marrow.analyze(keep_contrasts(phantom_study(ph), ("t2", "stir")), Anchors.from_truth(ph.truth))
    assert any("no sagittal T1" in w for w in res.warnings)
    assert not [r for r in res.rows if r["scope"] == "body"]


def test_no_sagittal_or_no_discs_warns(phantom):
    res = marrow.analyze(Study(stacks={}, info={}), Anchors.from_truth(phantom.truth))
    assert res.warnings and not res.rows
    ph = SpinePhantom(PhantomSpec())
    res2 = marrow.analyze(phantom_study(ph), Anchors(discs={}, vertebrae={}, midline_x=0.0))
    assert any("no discs" in w for w in res2.warnings)


def test_dixon_fat_fraction_is_measured_per_body():
    ph = SpinePhantom(PhantomSpec())
    study = phantom_study(ph)
    t2 = study.stacks[next(k for k, s in study.stacks.items() if s.meta["role"]["contrast"] == "t2")]
    shape = t2.vol.shape

    def dixon(comp: str, value: float) -> Stack:
        return Stack(
            key=f"sag_dixon_{comp}",
            vol=np.full(shape, value),
            ipp=t2.ipp,
            row_dir=t2.row_dir,
            col_dir=t2.col_dir,
            row_mm=t2.row_mm,
            col_mm=t2.col_mm,
            thickness=t2.thickness,
            meta={"role": {"plane": "sag", "contrast": "dixon", "dixon": comp, "localizer": False}},
        )

    stacks = dict(study.stacks)
    stacks["sag_dixon_w"] = dixon("w", 100.0)
    stacks["sag_dixon_f"] = dixon("f", 300.0)  # fat share 0.75 everywhere
    res = marrow.analyze(Study(stacks=stacks, info={}), Anchors.from_truth(ph.truth))
    assert res.meta["dixon"] is True
    body = [r for r in res.rows if r["scope"] == "body"]
    assert body and all(r["fat_fraction_pct"] == pytest.approx(75.0, abs=0.5) for r in body)


def test_analysis_is_registered_and_deterministic(planted):
    assert "marrow" in base.available()
    ph, res = planted
    again = marrow.analyze(phantom_study(ph), Anchors.from_truth(ph.truth))
    assert again.rows == res.rows and again.flags == res.flags


def test_result_carries_experimental_status(planted):
    _, res = planted
    assert res.meta["status"] == "experimental"
    assert res.meta["muscle_reference"].startswith("not measured")
    assert not any(k.startswith("_") for k in res.meta)


# ----------------------------------------------------------------------- unit functions
def test_classify_decides_by_fat_evidence_first():
    cfg = DEFAULT
    nan = float("nan")
    assert marrow._classify(marrow.PAT_BRIGHT, 0.8, nan, cfg) == marrow.CLASS_FAT
    assert marrow._classify(marrow.PAT_BRIGHT, 0.1, 0.1, cfg) == marrow.CLASS_HAEM  # Dixon overrides STIR
    assert marrow._classify(marrow.PAT_BRIGHT, nan, 0.4, cfg) == marrow.CLASS_FAT
    assert marrow._classify(marrow.PAT_BRIGHT, nan, 1.5, cfg) == marrow.CLASS_HAEM
    assert marrow._classify(marrow.PAT_BRIGHT, nan, nan, cfg) == marrow.CLASS_UNDECIDED
    assert marrow._classify(marrow.PAT_DARK, nan, nan, cfg) == marrow.CLASS_ISLAND
    assert marrow._classify(marrow.PAT_ATYPICAL, nan, nan, cfg) == marrow.CLASS_ATYPICAL


def test_shape_separates_thin_channels_from_round_foci():
    row, col = 1.0, 1.0
    # a 3 x 8 mm channel: width 3 mm, length 8 mm
    rr, cc = np.mgrid[0:3, 0:8]
    width, length, _ = marrow._shape_mm(rr.ravel(), cc.ravel(), row, col)
    assert width <= DEFAULT.marrow_linear_max_width_mm
    assert length >= DEFAULT.marrow_linear_aspect * width
    # a 10 x 10 mm disc is not linear
    rr, cc = np.mgrid[0:10, 0:10]
    width, length, size = marrow._shape_mm(rr.ravel(), cc.ravel(), row, col)
    assert width > DEFAULT.marrow_linear_max_width_mm
    assert length < DEFAULT.marrow_linear_aspect * width
    assert size == pytest.approx(10.0)


def test_pooled_reference_is_not_set_by_a_focus():
    rng = np.random.default_rng(0)
    t1 = np.concatenate([rng.normal(700, 10, 200), np.full(20, 950.0)])  # 20 focus pixels at 950
    t2 = np.concatenate([rng.normal(470, 10, 200), np.full(20, 900.0)])
    ref = marrow._pooled_reference({"t1": t1, "t2": t2}, DEFAULT)
    assert ref["t1"] == pytest.approx(700, abs=5)
    assert ref["t2"] == pytest.approx(470, abs=5)
    # a tiny sample keeps the plain median (no reliable exclusion)
    small = marrow._pooled_reference({"t1": np.array([700.0, 950.0, 950.0])}, DEFAULT)
    assert math.isfinite(small["t1"])


def test_phantom_marrow_options_are_validated():
    with pytest.raises(ValueError):
        SpinePhantom(PhantomSpec(marrow_lesions=[("L3", "cyst", (0.0, 0.0, 0.0), 8.0)]))
    with pytest.raises(ValueError):
        SpinePhantom(PhantomSpec(realism="realistic", basivertebral_vein=True))
