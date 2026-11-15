"""Axial cue (lab20): abstention, the gated slab vote, phantom T1 slabs, and the unchanged default path.

Synthetic and phantom-based; needs no dataset. The disc-likeness model is the shipped ``axial_v1.json``.
Appearance tests on the phantom are about mechanics (coverage, matching, abstention): the phantom is a cartoon
and the calibration was fitted on Sudirman, so the LLR values are not asserted against the phantom.
"""

from __future__ import annotations

import copy
import tempfile

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.axial import (
    AxialReport,
    SlabMatch,
    assess_axial,
    axial_slabs,
    load_axial_model,
)
from spinescope.anatomy.fusion import FusionConfig, fuse_numbering
from spinescope.anatomy.numbering_checks import AXIAL_SLAB_CUE, OFFSETS, axial_slab_vote
from spinescope.anatomy.pipeline import label_spine_details
from spinescope.study import Study, load_study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study


def _report(slabs: list[SlabMatch], available: bool = True) -> AxialReport:
    return AxialReport(available=available, contrast="t1", slabs=slabs)


# ------------------------------------------------------------ shipped model


def test_shipped_model_loads_with_stable_feature_order() -> None:
    model = load_axial_model()
    assert model is not None, "axial_v1.json must ship next to axial.py"
    assert len(model.mean_disc) == len(model.var_disc) == len(model.mean_non) == len(model.var_non) == 3
    assert all(v > 0 for v in model.var_disc + model.var_non)
    assert np.isfinite(model.missed_llr) and np.isfinite(model.support_llr)


def test_missing_calibration_file_abstains() -> None:
    assert load_axial_model("/nonexistent/axial_v1.json") is None


# --------------------------------------------------------------- the vote


def test_vote_none_without_report_or_slab() -> None:
    anchors = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    assert axial_slab_vote(None, anchors) is None
    assert axial_slab_vote(AxialReport(available=False), anchors) is None
    assert axial_slab_vote(_report([]), anchors) is None


def test_vote_none_when_lowest_slab_disc_not_supported() -> None:
    anchors = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    rep = _report([SlabMatch(slab="s", rank=0, disc="L5-S1", llr=-3.0, supported=False)])
    assert axial_slab_vote(rep, anchors) is None


def test_vote_none_when_lowest_slab_covers_no_disc() -> None:
    anchors = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    rep = _report([SlabMatch(slab="s", rank=0, disc=None)])
    assert axial_slab_vote(rep, anchors) is None


def test_vote_lowest_disc_supported_favours_offset_zero() -> None:
    anchors = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    rep = _report([SlabMatch(slab="s", rank=0, disc="L5-S1", llr=5.0, supported=True)])
    v = axial_slab_vote(rep, anchors)
    assert v is not None and v.cue == AXIAL_SLAB_CUE
    assert v.best == 0
    assert set(v.likelihood) == set(OFFSETS)


def test_vote_disc_one_level_up_favours_offset_minus_one() -> None:
    # Lowest slab angled to the disc above the lowest provisional disc: the lowest disc is one level lower.
    anchors = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    upper = anchors.disc_labels()[-2]  # L4-L5 in the truth anchors
    rep = _report([SlabMatch(slab="s", rank=0, disc=upper, llr=5.0, supported=True)])
    v = axial_slab_vote(rep, anchors)
    assert v is not None and v.best == -1


# -------------------------------------------------------- study-level cases


def _sagittal_only(study: Study) -> Study:
    """The same study with the axial stacks removed (the SPIDER situation: sagittal series only)."""
    return Study(
        stacks={k: s for k, s in study.stacks.items() if s.plane != "ax"},
        info=dict(study.info),
        report=dict(study.report),
    )


@pytest.fixture(scope="module")
def phantom_t1_study() -> Study:
    ph = SpinePhantom(PhantomSpec())
    root = tempfile.mkdtemp(prefix="lab20_axial_")
    paths = write_standard_study(ph, root, slab_contrasts=("t2", "t1"))
    return load_study(paths["root"])


def test_t1_slabs_are_selected_and_ordered_lowest_first(phantom_t1_study: Study) -> None:
    slabs = axial_slabs(phantom_t1_study)
    assert slabs, "phantom has T1 oblique slabs"
    assert all(s.meta["role"]["contrast"] == "t1" for s in slabs)
    zs = [float(s.ipp[:, 2].mean()) for s in slabs]
    assert zs == sorted(zs)


def test_sagittal_only_study_abstains(phantom_t1_study: Study) -> None:
    truth = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    rep = assess_axial(_sagittal_only(phantom_t1_study), truth)
    assert rep.available is False
    assert rep.warnings and "no axial slab" in rep.warnings[0]
    assert axial_slab_vote(rep, truth) is None


def test_t2_only_slabs_abstain_for_t1_cue() -> None:
    ph = SpinePhantom(PhantomSpec())
    root = tempfile.mkdtemp(prefix="lab20_t2_")
    st = load_study(write_standard_study(ph, root, slab_contrasts=("t2",))["root"])
    rep = assess_axial(st, Anchors.from_truth(ph.truth))
    assert rep.available is False


def test_phantom_t1_coverage_and_matches(phantom_t1_study: Study) -> None:
    truth = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    rep = assess_axial(phantom_t1_study, truth)
    assert rep.available is True
    assert rep.contrast == "t1"
    assert rep.slabs, "one match record per T1 slab"
    assert [s.rank for s in rep.slabs] == list(range(len(rep.slabs)))
    # The lowest slab is angled to L5-S1 in the phantom: covered, LLR finite.
    low = rep.slabs[0]
    assert low.disc == "L5-S1"
    assert low.llr is not None and np.isfinite(low.llr)
    assert rep.discs["L5-S1"].covered is True
    # Discs the phantom's slabs do not reach are not scored (no evidence either way).
    uncovered = [d for d in rep.discs.values() if not d.covered]
    assert all(d.supported is None for d in uncovered)


def test_predicted_disc_removed_does_not_crash_and_is_not_covering(phantom_t1_study: Study) -> None:
    truth = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    partial = copy.deepcopy(truth)
    partial.discs.pop("L5-S1")
    rep = assess_axial(phantom_t1_study, partial)
    assert "L5-S1" not in rep.discs
    assert rep.available is True


# -------------------------------------------------------------- fusion path


def test_default_fusion_is_unchanged_by_axial_report(phantom_t1_study: Study) -> None:
    """With use_axial off (the default) the numbering is the same whether or not a report is passed."""
    truth = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    rep = assess_axial(phantom_t1_study, truth)
    base = fuse_numbering(copy.deepcopy(truth), study=phantom_t1_study)
    with_rep = fuse_numbering(copy.deepcopy(truth), study=phantom_t1_study, axial=rep)
    assert base.offset == with_rep.offset
    assert base.posterior == with_rep.posterior
    assert with_rep.axial is None
    assert AXIAL_SLAB_CUE not in [v["method"] for v in with_rep.votes]


def test_axial_on_keeps_truth_offset_zero(phantom_t1_study: Study) -> None:
    truth = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    rep = assess_axial(phantom_t1_study, truth)
    res = fuse_numbering(
        copy.deepcopy(truth),
        study=phantom_t1_study,
        axial=rep,
        cfg=FusionConfig(use_axial=True),
    )
    assert res.offset == 0 and not res.applied
    assert res.axial is not None and res.axial["available"] is True
    methods = [v["method"] for v in res.votes]
    # Exactly one slab vote: the axial one replaces the unchecked slab cue (or is absent when unsupported).
    assert methods.count("slab") == 0


def test_label_spine_axial_switch_default_off_and_on(phantom_t1_study: Study) -> None:
    det_off = label_spine_details(phantom_t1_study, calibrate=False)
    assert "axial" not in det_off.anchors.numbering
    det_on = label_spine_details(phantom_t1_study, calibrate=False, fusion=FusionConfig(use_axial=True))
    assert det_on.anchors.discs
    rec = det_on.anchors.numbering.get("axial")
    assert rec is not None and rec["available"] is True


def test_sagittal_only_with_switch_on_is_identical_to_off(phantom_t1_study: Study) -> None:
    """SPIDER-like input (no axials): turning the cue on must change nothing."""
    st = _sagittal_only(phantom_t1_study)
    off = label_spine_details(st, calibrate=False)
    on = label_spine_details(st, calibrate=False, fusion=FusionConfig(use_axial=True))
    assert sorted(off.anchors.discs) == sorted(on.anchors.discs)
    assert off.anchors.numbering.get("offset") == on.anchors.numbering.get("offset")
    assert off.anchors.numbering.get("confidence") == on.anchors.numbering.get("confidence")
    assert "axial" not in on.anchors.numbering or on.anchors.numbering["axial"].get("available") is False


def test_report_to_dict_has_no_paths(phantom_t1_study: Study) -> None:
    truth = Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)
    d = assess_axial(phantom_t1_study, truth).to_dict()
    text = repr(d)
    assert "/tmp" not in text and "lab20_axial_" not in text
