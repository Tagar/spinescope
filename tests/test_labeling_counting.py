"""Top-chain check (lab34): candidate geometry, plausibility gates, abstention, and evidence-only recording.

Geometry tests are synthetic (no image data). The phantom tests check mechanics and the record, not accuracy: the
phantom is out of distribution for the learned labeller (see docs/model_card.md), so no flag value is asserted there.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from spinescope.anatomy._validate import TruthDisc
from spinescope.anatomy._validate_counting import top_chain_error
from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.anatomy.counting import (
    DEFAULT_COUNTING,
    NO_LANDMARK_REASON,
    analyse_candidate,
    candidate_stacks,
    chain_spacing,
    count_evidence,
)
from spinescope.anatomy.pipeline import label_spine_details
from spinescope.study import Study, from_stacks, load_study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study

# ----------------------------------------------------------------- synthetic chains (LPS, z in mm)

#: A lumbar chain, bottom-up, spaced 36 mm (a plausible spine).
PRIMARY_Z = {"L5-S1": -100.0, "L4-L5": -64.0, "L3-L4": -28.0, "L2-L3": 8.0, "L1-L2": 44.0}
#: Span of the primary's stack: covers its discs with room above and below.
PRIMARY_SPAN = (-160.0, 110.0)


def _chain(zs: dict[str, float], x: float = 0.0, y: float = 20.0, height: float = 8.0) -> Anchors:
    a = Anchors(method="test")
    for lab, z in zs.items():
        a.discs[lab] = Disc(center=[x, y, z], normal=[0.0, 0.0, 1.0], height_mm=height, confidence=0.9)
    return a


def _primary() -> Anchors:
    return _chain(PRIMARY_Z)


def test_chain_spacing_median_and_largest() -> None:
    med, big = chain_spacing(_primary())
    assert med == pytest.approx(36.0)
    assert big == pytest.approx(36.0)
    assert chain_spacing(_chain({"L5-S1": 0.0})) == (None, None)


def test_identical_candidate_agrees_at_the_top() -> None:
    ev = analyse_candidate(_primary(), PRIMARY_SPAN, _primary(), (-160.0, 150.0), key="loc", localizer=True)
    assert ev.overlap_discs == 5
    assert ev.overlap_matched == 5
    assert ev.top_disc_seen is True
    assert ev.extra_above_top == []
    assert ev.pitch_plausible is True
    assert ev.top_disagrees is False


def test_extra_disc_above_the_primary_top_is_flagged() -> None:
    zs = {
        **PRIMARY_Z,
        "T12-L1": 80.0,
    }  # one more disc, 36 mm above the top primary disc, inside the primary's span
    ev = analyse_candidate(_primary(), PRIMARY_SPAN, _chain(zs), (-160.0, 150.0), key="wide", localizer=False)
    assert ev.extra_above_top == ["T12-L1"]
    assert ev.top_disagrees is True


def test_disc_above_primary_coverage_is_not_an_extra_disc() -> None:
    zs = {**PRIMARY_Z, "T12-L1": 80.0}
    ev = analyse_candidate(
        _primary(), (-160.0, 60.0), _chain(zs), (-160.0, 150.0), key="wide", localizer=False
    )
    assert (
        ev.extra_above_top == []
    )  # z = 80 lies above the primary's coverage (60): no evidence against the primary


def test_top_disc_absent_from_candidate_is_flagged() -> None:
    zs = {k: v for k, v in PRIMARY_Z.items() if k != "L1-L2"}  # candidate lacks the primary's top disc
    ev = analyse_candidate(_primary(), PRIMARY_SPAN, _chain(zs), (-160.0, 150.0), key="wide", localizer=False)
    assert ev.top_disc_seen is False
    assert ev.top_disagrees is True


def test_top_disc_outside_candidate_coverage_is_not_judged() -> None:
    zs = {k: v for k, v in PRIMARY_Z.items() if k != "L1-L2"}
    ev = analyse_candidate(
        _primary(), PRIMARY_SPAN, _chain(zs), (-160.0, 20.0), key="narrow", localizer=False
    )
    assert ev.top_disc_seen is None
    assert ev.top_disagrees is False


def test_implausible_spacing_cannot_flag() -> None:
    # A localizer chain at 20 mm pitch (below level_spacing_mm[0]) with an extra disc above: recorded, not used.
    zs = {f"d{i}": -100.0 + 20.0 * i for i in range(9)}
    ev = analyse_candidate(_primary(), PRIMARY_SPAN, _chain(zs), (-160.0, 200.0), key="loc", localizer=True)
    assert ev.pitch_plausible is False
    assert ev.extra_above_top  # the disagreement is recorded ...
    assert ev.top_disagrees is False  # ... but cannot raise the flag


def test_single_missed_disc_gap_cannot_flag() -> None:
    zs = {
        "L5-S1": -100.0,
        "L4-L5": -64.0,
        "L3-L4": 60.0,
        "L2-L3": 96.0,
        "L1-L2": 130.0,
    }  # 124 mm gap in the middle
    ev = analyse_candidate(_primary(), PRIMARY_SPAN, _chain(zs), (-160.0, 150.0), key="loc", localizer=True)
    assert ev.pitch_plausible is False
    assert ev.top_disagrees is False


def test_reach_is_measured_against_the_primary_coverage() -> None:
    ev = analyse_candidate(_primary(), PRIMARY_SPAN, _primary(), (-200.0, 150.0), key="k", localizer=False)
    assert ev.reach_above_mm == pytest.approx(40.0)
    assert ev.reach_below_mm == pytest.approx(40.0)


def test_default_reach_threshold_is_one_level() -> None:
    assert DEFAULT_COUNTING.reach_min_mm == 25.0


# ----------------------------------------------------------------- validation helper (top-chain truth)


def _truth(zs: list[float]) -> list[TruthDisc]:
    names = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2", "T12-L1", "T11-T12"]
    return [
        TruthDisc(
            label=201 + i, name=names[i], center=np.array([0.0, 20.0, z]), normal=np.array([0.0, 0.0, 1.0])
        )
        for i, z in enumerate(zs)
    ]


def test_top_chain_error_is_false_for_matching_chain() -> None:
    truth = _truth([-100.0, -64.0, -28.0, 8.0, 44.0, 80.0])
    pred = _chain(
        {"L5-S1": -100.0, "L4-L5": -64.0, "L3-L4": -28.0, "L2-L3": 8.0, "L1-L2": 44.0, "T12-L1": 80.0}
    )
    assert top_chain_error(pred, truth) == (False, [])


def test_top_chain_error_sees_extra_and_missed_discs_above_the_fourth() -> None:
    truth = _truth([-100.0, -64.0, -28.0, 8.0, 44.0, 80.0])
    extra = _chain(
        {
            "L5-S1": -100.0,
            "L4-L5": -64.0,
            "L3-L4": -28.0,
            "L2-L3": 8.0,
            "L1-L2": 44.0,
            "T12-L1": 80.0,
            "T11-T12": 116.0,
        }
    )
    wrong, bad = top_chain_error(extra, truth)
    assert wrong and any(b.startswith("extra") for b in bad)
    missed = _chain({"L5-S1": -100.0, "L4-L5": -64.0, "L3-L4": -28.0, "L2-L3": 8.0})
    wrong, bad = top_chain_error(missed, truth)
    assert wrong and any(b.startswith("missed") for b in bad)


def test_top_chain_error_ignores_errors_below_the_fourth_disc() -> None:
    truth = _truth([-100.0, -64.0, -28.0, 8.0, 44.0, 80.0])
    # the lowest disc is missed: below the fourth truth disc, so it is a bottom error, not a top-chain error
    pred = _chain({"L4-L5": -64.0, "L3-L4": -28.0, "L2-L3": 8.0, "L1-L2": 44.0, "T12-L1": 80.0})
    assert top_chain_error(pred, truth) == (False, [])


# ----------------------------------------------------------------- phantom: selection and mechanics


def _two_series(ph: SpinePhantom) -> tuple[Study, str]:
    """A narrow sagittal series (the primary, ~230 mm FOV) and a wide one (~460 mm) of the same phantom.

    Returns the study and the key of the narrow (primary) series, found by its in-plane spacing.
    """
    narrow = ph.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    wide = ph.render("sag", "t1", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=2.4)
    study = from_stacks([narrow, wide])
    primary_key = next(k for k, st in study.stacks.items() if float(st.col_mm) < 1.5)
    return study, primary_key


def test_candidate_stacks_selects_only_the_wider_series() -> None:
    study, primary_key = _two_series(SpinePhantom(PhantomSpec()))
    primary = study.stacks[primary_key]
    cands = candidate_stacks(study, primary)
    assert [c.key for c in cands] == [k for k in study.stacks if k != primary_key]
    assert candidate_stacks(study, primary, cfg=type(DEFAULT_COUNTING)(reach_min_mm=10_000.0)) == []


def test_count_evidence_without_wider_series_abstains() -> None:
    ph = SpinePhantom(PhantomSpec())
    study = from_stacks(
        [ph.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)]
    )
    res = count_evidence(study, _primary(), primary_stack=study.best("sag"))
    assert res.available is False
    assert res.vote is None and res.landmark is None
    assert "no sagittal series reaches beyond" in res.reason


def test_count_evidence_abstains_without_a_landmark() -> None:
    ph = SpinePhantom(PhantomSpec())
    study, primary_key = _two_series(ph)
    primary = study.stacks[primary_key]
    res = count_evidence(study, _primary(), primary_stack=primary)
    assert res.landmark is None
    assert res.vote is None
    assert res.reason == NO_LANDMARK_REASON
    json.dumps(res.to_dict())  # serialisable, plain values


def test_count_evidence_with_no_primary_is_unavailable() -> None:
    res = count_evidence(Study(stacks={}), Anchors(method="test"), primary_stack=None)
    assert res.available is False and res.vote is None


def test_label_spine_records_counting_evidence_and_keeps_the_names() -> None:
    import tempfile

    ph = SpinePhantom(PhantomSpec())
    with tempfile.TemporaryDirectory() as tmp:
        write_standard_study(ph, tmp)  # the standard protocol has a 3-plane localizer with a wide FOV
        study = load_study(tmp)
        with_c = label_spine_details(study, counting=True).anchors
        without = label_spine_details(study, counting=False).anchors
    rec = with_c.numbering.get("counting")
    assert isinstance(rec, dict)
    assert rec["landmark"] is None and rec["vote"] is None
    assert "counting" not in without.numbering
    # evidence only: the names and the status do not depend on it
    assert with_c.disc_labels() == without.disc_labels()
    assert with_c.numbering.get("status") == without.numbering.get("status")
