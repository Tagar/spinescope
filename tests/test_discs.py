from __future__ import annotations

import json

import numpy as np

from _studies import keep_contrasts, phantom_study, stacks_flipped
from spinescope.analysis import base, discs
from spinescope.analysis.base import Result, available, run_all
from spinescope.anatomy import Anchors
from spinescope.study import Study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

ALL_DISCS = ("T11-T12", "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")


def _primary(res: Result) -> dict[str, dict]:
    return {r["disc"]: r for r in res.rows if r["scope"] == "slice" and r["primary"]}


def _summary(res: Result) -> dict[str, dict]:
    return {r["disc"]: r for r in res.rows if r["scope"] == "disc"}


def _measured(res: Result) -> list[str]:
    return sorted(_primary(res))


def test_mid_disc_height_within_1mm_of_phantom_truth(phantom, phantom_study):
    res = discs.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    prim = _primary(res)
    assert len(prim) >= 6
    for lab, row in prim.items():
        assert abs(row["mid_h_mm"] - phantom.spec.disc_height) <= 1.0, (lab, row["mid_h_mm"])
        assert row["valid_fraction"] >= 0.5


def test_degenerated_disc_has_the_lowest_nucleus_ratio(phantom, phantom_study):
    res = discs.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    summ = _summary(res)
    assert min(summ, key=lambda d: summ[d]["nuc_over_csf"]) == "L4-L5"
    assert min(summ, key=lambda d: summ[d]["nuc_over_brightest"]) == "L4-L5"
    assert summ["L4-L5"]["nuc_over_csf"] < 0.5 * summ["L1-L2"]["nuc_over_csf"]


def test_pfirrmann_suggestion_is_labelled_and_highest_at_degenerated_disc(phantom, phantom_study):
    res = discs.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    summ = _summary(res)
    grades = {d: s["pfirrmann_suggestion"] for d, s in summ.items() if s["pfirrmann_suggestion"] is not None}
    assert grades["L4-L5"] == max(grades.values()) and grades["L4-L5"] >= 4
    for s in summ.values():
        assert s["suggestion_only"] is True
        assert 0.0 <= s["pfirrmann_confidence"] <= 0.75
    assert any(f["kind"] == "pfirrmann_suggestion" and f["level"] == "L4-L5" for f in res.flags)


def test_no_hiz_candidates_on_the_phantom(phantom, phantom_study):
    res = discs.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    assert not [f for f in res.flags if f["kind"] == "hiz_candidate"]


def test_heights_come_from_the_image_not_the_anchor(phantom, phantom_study):
    anchors = Anchors.from_truth(phantom.truth)
    for d in anchors.discs.values():
        d.height_mm = 3.0  # wrong on purpose: the measurement must not read it
    res = discs.analyze(phantom_study, anchors)
    prim = _primary(res)
    assert all(abs(r["mid_h_mm"] - phantom.spec.disc_height) <= 1.0 for r in prim.values())


def test_t1_only_gives_heights_and_a_warning(phantom):
    # T1 endplates need finer pixels in the phantom (its 1 mm cortex is missed at 1.2 mm)
    t1 = phantom_study(phantom, contrasts=("t1",), px=0.8, rows=288)
    res = discs.analyze(t1, Anchors.from_truth(phantom.truth))
    assert any("T1" in w for w in res.warnings)
    prim = _primary(res)
    assert len(prim) >= 6
    assert all(abs(r["mid_h_mm"] - phantom.spec.disc_height) <= 1.0 for r in prim.values())
    assert all(np.isnan(r["nuc_mean"]) for r in prim.values())


def test_missing_stir_warns_and_t2_analysis_still_runs(phantom, phantom_study):
    no_stir = keep_contrasts(phantom_study, ("t2", "t1"))
    res = discs.analyze(no_stir, Anchors.from_truth(phantom.truth))
    assert any("STIR" in w for w in res.warnings)
    assert _measured(res)


def test_flipped_acquisition_gives_the_same_heights(phantom, phantom_study):
    anchors = Anchors.from_truth(phantom.truth)
    a = _primary(discs.analyze(phantom_study, anchors))
    b = _primary(discs.analyze(stacks_flipped(phantom_study), anchors))
    assert set(a) == set(b)
    for lab in a:
        assert abs(a[lab]["mid_h_mm"] - b[lab]["mid_h_mm"]) < 0.2


def test_lordosis_noise_and_cropped_fov_still_measured():
    spec = PhantomSpec(
        segment_lordosis=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
        noise=0.05,
        degenerated={"L3-L4": 0.5},
        seed=3,
    )
    ph = SpinePhantom(spec)
    study = phantom_study(ph, rows=150)  # cropped field of view
    res = discs.analyze(study, Anchors.from_truth(ph.truth))
    prim = _primary(res)
    assert len(prim) >= 3
    for lab, row in prim.items():
        assert abs(row["mid_h_mm"] - spec.disc_height) <= 1.0, (lab, row["mid_h_mm"])


def test_disc_strip_figure_and_saved_outputs(tmp_path, phantom, phantom_study):
    res = discs.analyze(phantom_study, Anchors.from_truth(phantom.truth))
    assert "disc_strip" in res.figures
    fig = res.figures["disc_strip"]
    assert fig.dtype == np.uint8 and fig.ndim == 3 and fig.shape[2] == 3
    paths = res.save(tmp_path)
    assert any(p.endswith("discs.csv") for p in paths)
    assert any(p.endswith("discs_disc_strip.png") for p in paths)
    data = json.loads((tmp_path / "discs.json").read_text())
    assert data["name"] == "discs" and data["figures"] == ["disc_strip"]
    assert data.get("rows")


def test_missing_margins_are_assumed_with_a_warning(phantom, phantom_study):
    anchors = Anchors.from_truth(phantom.truth)
    for d in anchors.discs.values():
        d.anterior = None
        d.posterior = None
    res = discs.analyze(phantom_study, anchors)
    assert any("assumed" in w for w in res.warnings)
    assert _measured(res)


def test_no_sagittal_stack_is_a_warning_not_an_error(phantom):
    res = discs.analyze(Study(stacks={}), Anchors.from_truth(phantom.truth))
    assert res.warnings and not res.rows


def test_registry_lists_builtins_and_run_all_contains_failures(phantom):
    assert {"discs", "endplates"} <= set(available())
    out = run_all(Study(stacks={}), Anchors.from_truth(phantom.truth))
    assert set(out) >= {"discs", "endplates"}
    assert all(isinstance(r, Result) for r in out.values())
    assert all(r.warnings for r in out.values())


def test_result_to_dict_drops_private_keys_and_pixels():
    r = Result(
        name="x", rows=[{"a": np.float64(1.5), "nan": float("nan"), "_hidden": 1}], meta={"_private": 2}
    )
    r.figures["f"] = np.zeros((2, 2, 3), dtype=np.uint8)
    d = base._jsonable(r.to_dict())
    assert d["rows"] == [{"a": 1.5, "nan": None}]
    assert d["meta"] == {}
    assert d["figures"] == ["f"]
    json.dumps(d, allow_nan=False)


def test_run_is_deterministic(phantom, phantom_study):
    anchors = Anchors.from_truth(phantom.truth)
    a = discs.analyze(phantom_study, anchors).rows
    b = discs.analyze(phantom_study, anchors).rows
    assert json.dumps(base._jsonable(a), sort_keys=True) == json.dumps(base._jsonable(b), sort_keys=True)
