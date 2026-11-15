from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from _studies import keep_contrasts, phantom_study, stacks_flipped
from spinescope.analysis import base, endplates
from spinescope.analysis.base import Result
from spinescope.anatomy import Anchors
from spinescope.config import DEFAULT
from spinescope.study import Study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom


def _rows(res: Result, vertebra: str, endplate: str, side: str) -> dict:
    hits = [
        r
        for r in res.rows
        if r["scope"] == "endplate"
        and r["vertebra"] == vertebra
        and r["endplate"] == endplate
        and r["side"] == side
    ]
    assert hits, (vertebra, endplate, side)
    return hits[0]


def _modic_flags(res: Result) -> list[dict]:
    return [f for f in res.flags if f["kind"] == "modic_suggestion"]


def test_modic1_on_the_left_inferior_endplate_of_L4_only(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    left = _rows(res, "L4", "inferior", "left")
    right = _rows(res, "L4", "inferior", "right")
    central = _rows(res, "L4", "inferior", "central")
    assert left["pattern"] == "Modic 1"
    assert right["pattern"] == "none"
    assert central["pattern"] == "none"
    assert left["extent_pct"] > 80.0
    overall = _rows(res, "L4", "inferior", "all")
    assert overall["pattern"] == "Modic 1" and "left side" in overall["note"]


def test_no_false_modic_anywhere_else(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    flags = _modic_flags(res)
    assert len(flags) == 1
    assert (flags[0]["vertebra"], flags[0]["endplate"], flags[0]["side"]) == ("L4", "inferior", "left")
    others = [
        r
        for r in res.rows
        if r["scope"] == "endplate"
        and r["side"] != "all"
        and not (r["vertebra"] == "L4" and r["endplate"] == "inferior" and r["side"] == "left")
    ]
    assert others and all(r["pattern"] == "none" for r in others)


def test_signal_ratios_match_the_phantom_truth(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    left = _rows(res, "L4", "inferior", "left")
    # phantom band / mid-body: T1 360/720 = 0.5, T2 900/470 = 1.9, STIR 900/130 = 6.9
    assert abs(left["T1_ratio"] - 0.5) < 0.05
    assert abs(left["T2_ratio"] - 1.9) < 0.2
    assert left["STIR_ratio"] > 5.0
    normal = _rows(res, "L2", "superior", "all")
    assert abs(normal["T1_ratio"] - 1.0) < 0.05 and abs(normal["T2_ratio"] - 1.0) < 0.05


def test_extent_and_depth_of_the_band(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    left = _rows(res, "L4", "inferior", "left")
    # the phantom band lies 1-9 mm inside the endplate; the depth statistic is the 95th percentile
    assert 6.0 <= left["depth_mm"] <= 10.5
    assert np.isnan(_rows(res, "L2", "superior", "all")["depth_mm"])


def test_healthiest_reference_excludes_the_affected_body(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    assert "L4" not in res.meta["healthy_bodies"]
    assert "L2" in res.meta["healthy_bodies"]
    assert abs(_rows(res, "L2", "superior", "all")["T1_vs_healthy"] - 1.0) < 0.1


def test_side_classification_uses_slice_position(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    left = _rows(res, "L4", "inferior", "left")
    assert left["x_mm"] > 0 and all(k in (4, 5) for k in left["slices_k"])


def test_triptych_figure_for_the_flagged_endplate(phantom, phantom_study, tmp_path):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    assert list(res.figures) == ["triptych_L4_inferior"]
    fig = res.figures["triptych_L4_inferior"]
    assert fig.dtype == np.uint8 and fig.ndim == 3 and fig.shape[2] == 3
    assert fig.shape[1] > fig.shape[0]  # three panels side by side
    paths = res.save(tmp_path)
    assert any(p.endswith("endplates_triptych_L4_inferior.png") for p in paths)
    assert json.loads((tmp_path / "endplates.json").read_text())["name"] == "endplates"


def test_no_lesion_candidates_on_the_phantom(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    assert not [f for f in res.flags if f["kind"] == "marrow_lesion_candidate"]


def test_missing_stir_warns_and_t1_t2_still_type_modic1(phantom, phantom_study):
    res = endplates.analyze(keep_contrasts(phantom_study, ("t1", "t2")), Anchors.from_truth(phantom.truth))
    assert any("STIR" in w for w in res.warnings)
    assert _rows(res, "L4", "inferior", "left")["pattern"] == "Modic 1"
    assert (
        _rows(res, "L4", "inferior", "left")["STIR_ratio"]
        != _rows(res, "L4", "inferior", "left")["STIR_ratio"]
    )


def test_missing_t1_warns_and_does_not_assign_modic_type(phantom, phantom_study):
    res = endplates.analyze(keep_contrasts(phantom_study, ("t2", "stir")), Anchors.from_truth(phantom.truth))
    assert any("T1" in w for w in res.warnings)
    assert not _modic_flags(res)
    assert all(
        r["pattern"] in ("not assessed (no T1)", "presence only (no T1)", "not assessed")
        for r in res.rows
        if r["scope"] == "endplate" and r["side"] != "all"
    )


def test_flipped_acquisition_gives_the_same_suggestions(phantom, phantom_study):
    anchors = Anchors.from_truth(phantom.truth)
    a = endplates.analyze(phantom_study, anchors)
    b = endplates.analyze(stacks_flipped(phantom_study), anchors)
    assert [(f["vertebra"], f["endplate"], f["side"]) for f in _modic_flags(a)] == [
        (f["vertebra"], f["endplate"], f["side"]) for f in _modic_flags(b)
    ]


def test_missing_sagittal_stacks_are_warnings():
    res = endplates.analyze(Study(stacks={}), Anchors())
    assert res.warnings and not res.rows


def test_graceful_with_lordosis_noise_and_cropped_fov():
    spec = PhantomSpec(
        segment_lordosis=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
        noise=0.04,
        degenerated={"L5-S1": 0.6},
        modic1=[("L3-L4", "sup", "R")],
        seed=5,
    )
    ph = SpinePhantom(spec)
    res = endplates.analyze(phantom_study(ph, rows=160), Anchors.from_truth(ph.truth))
    flags = _modic_flags(res)
    # phantom 'sup' on disc L3-L4 is the superior endplate of the body below it, i.e. L4
    assert [(f["vertebra"], f["endplate"], f["side"]) for f in flags] == [("L4", "superior", "right")]


def test_registry_runs_endplates_on_a_study(phantom, phantom_study):
    out = base.run("endplates", phantom_study, Anchors.from_truth(phantom.truth))
    assert out.name == "endplates" and out.rows


def test_missing_t2_gives_presence_only_and_no_modic_type(phantom, phantom_study):
    res = endplates.analyze(keep_contrasts(phantom_study, ("t1", "stir")), Anchors.from_truth(phantom.truth))
    assert any("T2" in w for w in res.warnings)
    assert not _modic_flags(res)
    rows = [r for r in res.rows if r["scope"] == "endplate" and r["side"] != "all"]
    assert rows
    assert all(not r["pattern"].startswith("Modic") for r in rows)
    presence = [f for f in res.flags if f["kind"] == "marrow_change_presence"]
    assert presence, "the phantom Modic 1 must show as a presence-only change without T2"
    assert all(f["confidence"] <= 0.3 for f in presence)


def test_pattern_rule_needs_t1_and_t2():
    cfg = endplates.DEFAULT
    nan = float("nan")
    assert endplates._pattern(0.5, nan, nan, cfg) == endplates.PRESENCE_NO_T2
    assert endplates._pattern(1.0, nan, nan, cfg) == "none"
    assert endplates._pattern(nan, 1.3, nan, cfg) == endplates.PRESENCE_NO_T1
    assert endplates._pattern(nan, 1.0, nan, cfg) == endplates.NOT_ASSESSED_NO_T1
    assert endplates._pattern(0.5, 1.3, nan, cfg) == "Modic 1"
    assert endplates._pattern(0.5, 1.3, 1.5, cfg) == "Modic 1"
    assert endplates._pattern(0.5, 1.0, 1.5, cfg) == "Modic 1-like"
    assert endplates._pattern(0.5, 1.0, 1.0, cfg) == "indeterminate"
    assert endplates._pattern(0.5, 0.5, nan, cfg) == "Modic 3"
    assert endplates._pattern(1.3, 1.0, nan, cfg) == "Modic 2"
    assert endplates._pattern(1.0, 1.0, nan, cfg) == "none"


def test_width_pct_is_share_of_endplate_width_with_abnormal_marrow(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    r = _rows(res, "L4", "inferior", "left")
    assert 0.0 < r["width_pct"] <= 100.0
    healthy = _rows(res, "L2", "superior", "all")
    assert healthy["width_pct"] == healthy["width_pct"]  # defined (not NaN) on a normal endplate
    assert healthy["width_pct"] <= 1.0


def _trace(bot: np.ndarray, top: np.ndarray | None = None, step: float = 1.0) -> SimpleNamespace:
    """Minimal disc-trace stand-in for the defect measure (columns 1 mm apart, all traced)."""
    s = np.arange(bot.size, dtype=float) * step
    return SimpleNamespace(
        s=s,
        bot=bot,
        top=bot if top is None else top,
        ok=np.ones(bot.size, dtype=bool),
    )


def test_defect_flat_contour_has_no_concavity():
    conc, irr = endplates._defect_profile(_trace(np.full(40, 10.0)), "bot", 20.0)
    assert conc == 0.0 and irr == 0.0


def test_defect_notch_into_body_is_measured_in_mm():
    bot = np.full(60, 10.0)
    bot[25:31] -= 3.0  # superior endplate of the body below dips 3 mm into the body (Schmorl-like intrusion)
    conc, irr = endplates._defect_profile(_trace(bot), "bot", 20.0)
    assert 2.5 <= conc <= 3.2
    assert irr > 0.0


def test_defect_upward_notch_for_top_endplate_and_sign():
    top = np.full(60, 10.0)
    top[25:31] += 3.0  # inferior endplate of the body above: intrusion raises the line
    conc, _ = endplates._defect_profile(_trace(np.full(60, 10.0), top), "top", 20.0)
    assert 2.5 <= conc <= 3.2
    # the same downward notch on a "top" endplate is not an intrusion into the body
    top_down = np.full(60, 10.0)
    top_down[25:31] -= 3.0
    conc_down, _ = endplates._defect_profile(_trace(np.full(60, 10.0), top_down), "top", 20.0)
    assert conc_down == 0.0


def test_defect_too_few_columns_is_not_assessed():
    conc, irr = endplates._defect_profile(_trace(np.full(3, 10.0)), "bot", 20.0)
    assert np.isnan(conc) and np.isnan(irr)


def test_typed_modic_is_experimental_and_capped(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    assert res.meta["modic_status"] == "experimental"
    assert "experimental" in res.meta["modic_note"]
    typed = [r for r in res.rows if r["scope"] == "endplate" and r["pattern"] in endplates.MODIC_PATTERNS]
    assert typed
    assert all(r["confidence"] <= DEFAULT.endplate_modic_max_confidence + 1e-9 for r in typed)
    flags = _modic_flags(res)
    assert flags and all(f["confidence"] <= DEFAULT.endplate_modic_max_confidence + 1e-9 for f in flags)
    assert all("experimental" in f["note"] for f in flags)


def test_defect_flags_are_off_by_default_and_rows_always_written(phantom, phantom_study):
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    assert res.meta["endplate_defect_status"] == "experimental"
    assert not [f for f in res.flags if f["kind"] == "endplate_defect_candidate"]
    assert any(r["scope"] == "endplate" and r["side"] == "all" for r in res.rows)
    assert all("concavity_mm" in r for r in res.rows if r["scope"] == "endplate")


def test_defect_flags_on_when_the_switch_is_set(phantom, phantom_study):
    from dataclasses import replace

    cfg = replace(
        DEFAULT, endplate_defect_flags=True, endplate_defect_mm=-1.0
    )  # every finite concavity flags
    res = endplates.analyze(phantom_study, Anchors.from_truth(phantom.truth), cfg)
    flags = [f for f in res.flags if f["kind"] == "endplate_defect_candidate"]
    assert flags
    assert all("experimental" in f["note"] for f in flags)
