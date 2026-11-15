"""Sudirman scorer and loader units: synthetic arrays and anchors only (no dataset needed)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from spinescope.anatomy._validate import TruthDisc
from spinescope.anatomy._validate_sudirman import score_levels, summarise_levels, truth_discs
from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.datasets import REGISTRY, get_spec
from spinescope.datasets.sudirman import (
    DISC_LEVELS,
    IVD_VALUE,
    LevelTruth,
    match_levels_arrays,
    ncc,
    read_radiologist_notes,
)
from spinescope.geometry import Stack
from spinescope.series import DISPLAY_AXES
from spinescope.study import from_stacks


def _truth(level: str, centre: list[float], normal=(0.0, 0.0, 1.0)) -> LevelTruth:
    d = {v: k for k, v in DISC_LEVELS.items()}[level]
    return LevelTruth(
        patient="0001",
        dlabel=d,
        level=level,
        stack_key="ax_t1_slab1",
        slice_index=0,
        center=np.array(centre, dtype=np.float64),
        normal=np.array(normal, dtype=np.float64),
        ivd_pixels=100,
        match_ncc=1.0,
    )


def _anchors(discs: dict[str, list[float]]) -> Anchors:
    return Anchors(
        discs={name: Disc(center=c, normal=[0.0, 0.0, 1.0], confidence=0.9) for name, c in discs.items()}
    )


def test_perfect_prediction_scores_perfectly():
    truths = [_truth("L3-L4", [0, 60, -50]), _truth("L4-L5", [0, 70, -75]), _truth("L5-S1", [0, 80, -100])]
    anchors = _anchors(
        {"L3-L4": [0, 61, -49], "L4-L5": [0, 69, -74], "L5-S1": [0, 83, -100], "L2-L3": [0, 50, -25]}
    )
    rows = score_levels(anchors, truths)
    assert all(r.matched and r.name_ok for r in rows)
    s = summarise_levels(rows)
    assert s["recall"] == 1.0 and s["label_accuracy"] == 1.0
    assert s["centre_err_median_mm"] == pytest.approx(
        np.median([np.hypot(1, 1), np.hypot(1, 1), np.hypot(3, 0)])
    )
    assert s["normal_err_median_deg"] == pytest.approx(0.0)


def test_neighbour_name_is_matched_but_wrong():
    # The disc at the L4-L5 position carries the name L3-L4: in-plane close, name wrong.
    truths = [_truth("L4-L5", [0, 70, -75])]
    rows = score_levels(_anchors({"L3-L4": [0, 70, -77]}), truths)
    assert rows[0].matched and not rows[0].name_ok
    s = summarise_levels(rows)
    assert s["recall"] == 1.0 and s["label_accuracy"] == 0.0 and s["label_accuracy_any_distance"] == 0.0


def test_far_prediction_is_not_matched_but_still_reported():
    truths = [_truth("L5-S1", [0, 80, -100])]
    rows = score_levels(_anchors({"L5-S1": [0, 80, -150]}), truths)  # 50 mm away
    assert not rows[0].matched and rows[0].name_ok
    s = summarise_levels(rows)
    assert s["recall"] == 0.0 and np.isnan(s["centre_err_median_mm"])
    assert s["label_accuracy_any_distance"] == 1.0


def test_no_predicted_discs_gives_no_match():
    rows = score_levels(_anchors({}), [_truth("L3-L4", [0, 60, -50])])
    assert rows[0].predicted is None and not rows[0].matched and not rows[0].name_ok
    assert summarise_levels(rows)["recall"] == 0.0


def test_truth_discs_use_spider_label_values_and_names():
    truths = truth_discs([_truth("L3-L4", [0, 1, 2]), _truth("L5-S1", [0, 1, 3])])
    assert [t.label for t in truths] == [203, 201]
    assert all(isinstance(t, TruthDisc) for t in truths)
    assert [t.name for t in truths] == ["L3-L4", "L5-S1"]


def test_ncc_is_gain_and_offset_invariant():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(32, 32))
    assert ncc(a, 3.0 * a + 7.0) == pytest.approx(1.0)
    assert abs(ncc(a, rng.normal(size=(32, 32)))) < 0.1


def _axial_stack(key: str, z0: float, rng: np.random.Generator, n: int = 4) -> Stack:
    rd, cd = DISPLAY_AXES["ax"]
    ipp = [np.array([-70.0, -70.0, z0 + 4.0 * i]) for i in range(n)]
    return Stack(
        key=key,
        vol=rng.uniform(0, 255, size=(n, 32, 32)),
        ipp=ipp,
        row_dir=rd,
        col_dir=cd,
        row_mm=4.0,
        col_mm=4.0,
        thickness=4.0,
        description="t1_tse_tra",
        meta={"dicom": {"SeriesDescription": "T1_TSE_TRA", "RepetitionTime": 715, "EchoTime": 11}},
    )


def test_match_levels_finds_the_slice_and_its_centre():
    rng = np.random.default_rng(1)
    stacks = [_axial_stack("a", 0.0, rng, 3), _axial_stack("b", 60.0, rng, 4)]
    study = from_stacks(stacks)
    assert {"ax_t1", "ax_t1_2"} == set(study.stacks)
    target = study.stacks["ax_t1_2"]
    k = 2
    ref = target.vol[k].copy()
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:14, 20:24] = IVD_VALUE
    truths = match_levels_arrays(study, {"D5": ref}, {"D5": mask}, patient="9999")
    assert len(truths) == 1
    t = truths[0]
    assert (t.dlabel, t.level, t.stack_key, t.slice_index) == ("D5", "L5-S1", "ax_t1_2", k)
    assert t.match_ncc == pytest.approx(1.0)
    assert t.ivd_pixels == 16
    expected = target.pix_to_lps(k, 11.5, 21.5)
    assert np.allclose(t.center, expected)
    assert np.allclose(t.normal, np.abs(t.normal) * np.sign(t.normal[2]))
    assert t.normal[2] > 0


def test_match_levels_rejects_unmatched_reference():
    rng = np.random.default_rng(2)
    study = from_stacks([_axial_stack("a", 0.0, rng, 3)])
    ref = rng.uniform(0, 255, size=(32, 32))
    mask = np.full((32, 32), IVD_VALUE, dtype=np.uint8)
    assert match_levels_arrays(study, {"D4": ref}, {"D4": mask}, min_ncc=0.98) == []


def test_match_levels_rejects_mismatched_mask_shape():
    rng = np.random.default_rng(3)
    study = from_stacks([_axial_stack("a", 0.0, rng, 2)])
    ref = study.stacks["ax_t1"].vol[0].copy()
    with pytest.raises(ValueError, match="does not match"):
        match_levels_arrays(study, {"D3": ref}, {"D3": np.zeros((5, 5), dtype=np.uint8)})


def test_read_radiologist_notes_parses_shared_strings(tmp_path: Path):
    path = tmp_path / "raw" / "Radiologists Report.xlsx"
    path.parent.mkdir(parents=True)
    shared = (
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<si><t>Patient ID</t></si><si><t>Clinician's Notes</t></si><si><t>L4-5 bulge</t></si></sst>"
    )
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
        '<row r="2"><c r="A2"><v>7</v></c><c r="B2" t="s"><v>2</v></c></row>'
        "</sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", shared)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    assert read_radiologist_notes(tmp_path) == {7: "L4-5 bulge"}


def test_registry_has_the_sudirman_label_and_notes_sets():
    labels = get_spec("lumbar_spine_mri_sudirman_labels")
    notes = get_spec("lumbar_spine_mri_sudirman_notes")
    assert labels.doi == "10.17632/zbf6b4pttk.2" and labels.licence == "CC-BY-4.0"
    assert notes.doi == "10.17632/s6bgczr8s2.2" and notes.licence == "CC-BY-4.0"
    # Both fit under the default 2 GiB download cap.
    assert labels.total_size < 2 * 1024**3 and notes.total_size < 2 * 1024**3
    for spec in (labels, notes):
        for f in spec.files:
            assert f.sha256 and f.resolver == "mendeley-api" and f.size > 0
    assert {"lumbar_spine_mri_sudirman_labels", "lumbar_spine_mri_sudirman_notes"} <= set(REGISTRY)
