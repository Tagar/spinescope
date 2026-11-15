"""Lumbosacral transitional vertebra check (lab28): cues, Castellvi grading, naming suggestion, pipeline wiring.

Most tests use synthetic anchors (exact numbers, no image). A few run the labeller on the phantom to prove the
hook and the record shape; those check contracts (no crash, names unchanged, record present), not accuracy.
"""

from __future__ import annotations

import json

import pytest

from _studies import phantom_study
from spinescope.anatomy import summary
from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.anatomy.lstv import (
    DEFAULT_LSTV,
    LUMBAR,
    NONE,
    OFFSET_OF,
    SACRAL,
    STATUS_NOT_ASSESSED,
    STATUS_NOT_SUSPECTED,
    STATUS_POSSIBLE,
    STATUS_SUSPECTED,
    UNRESOLVED,
    LstvConfig,
    assess_lstv,
    attach_lstv,
    body_cues,
    castellvi_grade,
    combine_cues,
    disc_cue,
    lstv_line,
    naming_record,
    numbering_cues,
    side_grade,
)
from spinescope.anatomy.pipeline import label_spine
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

# ------------------------------------------------------------------ helpers


def _anchors(
    heights: list[float],
    *,
    bodies: dict[int, dict[str, float]] | None = None,
    numbering: dict | None = None,
) -> Anchors:
    """Discs top-down named L1-L2 ... with the given heights (the last one is the lowest, named L5-S1 by count).

    ``bodies``: discs_below -> {"anterior", "posterior", "mean"} heights (mm) for the body record.
    """
    names = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    labels = names[-len(heights) :]
    discs = {}
    for i, (lab, h) in enumerate(zip(labels, heights, strict=True)):
        z = 60.0 - 40.0 * i  # superior first; the lowest disc has the smallest z
        discs[lab] = Disc(center=[0.0, 0.0, z], normal=[0.0, 0.0, 1.0], height_mm=h, confidence=0.9)
    verts: dict[str, Vertebra] = {}
    for j, hb in (bodies or {}).items():
        name = {0: "S1", 1: "L5", 2: "L4", 3: "L3", 4: "L2"}[j]
        body = {
            "discs_below": j,
            "height_mm": {
                "anterior": hb.get("anterior"),
                "middle": hb.get("mean"),
                "posterior": hb.get("posterior"),
                "mean": hb.get("mean"),
            },
        }
        verts[name] = Vertebra(center=[0.0, 0.0, -100.0 + 40 * j], confidence=0.9, body=body)
    return Anchors(
        discs=discs,
        vertebrae=verts,
        numbering=numbering
        if numbering is not None
        else {"posterior": {"-1": 0.075, "0": 0.85, "1": 0.075}, "applied_offset": 0},
        method="test",
    )


def _post(minus: float, zero: float, plus: float) -> dict[str, float]:
    return {"-1": minus, "0": zero, "1": plus}


# ------------------------------------------------------------ Castellvi


@pytest.mark.parametrize(
    ("left", "right", "grade"),
    [
        ({"tp_cc_mm": 21.0, "contact": "none"}, {"tp_cc_mm": 8.0, "contact": "none"}, "Ia"),
        ({"tp_cc_mm": 21.0, "contact": "none"}, {"tp_cc_mm": 22.0, "contact": "none"}, "Ib"),
        ({"tp_cc_mm": 12.0, "contact": "pseudoarticulation"}, {"tp_cc_mm": 8.0, "contact": "none"}, "IIa"),
        (
            {"tp_cc_mm": 12.0, "contact": "pseudoarticulation"},
            {"tp_cc_mm": 12.0, "contact": "pseudoarticulation"},
            "IIb",
        ),
        ({"tp_cc_mm": 12.0, "contact": "fusion"}, {"tp_cc_mm": 8.0, "contact": "none"}, "IIIa"),
        ({"tp_cc_mm": 12.0, "contact": "fusion"}, {"tp_cc_mm": 12.0, "contact": "fusion"}, "IIIb"),
        ({"tp_cc_mm": 12.0, "contact": "pseudoarticulation"}, {"tp_cc_mm": 12.0, "contact": "fusion"}, "IV"),
        ({"tp_cc_mm": 8.0, "contact": "none"}, {"tp_cc_mm": 9.0, "contact": "none"}, "normal"),
    ],
)
def test_castellvi_grades(left: dict, right: dict, grade: str) -> None:
    rec = castellvi_grade({"left": left, "right": right})
    assert rec["status"] == "measured"
    assert rec["grade"] == grade


def test_castellvi_threshold_is_19_mm_craniocaudal() -> None:
    assert side_grade({"tp_cc_mm": 19.0, "contact": "none"}) == "I"
    assert side_grade({"tp_cc_mm": 18.9, "contact": "none"}) == "normal"
    assert DEFAULT_LSTV.tp_enlarged_mm == 19.0


def test_castellvi_not_assessed_without_record() -> None:
    rec = castellvi_grade(None)
    assert rec["status"] == "not_assessed" and rec["grade"] is None


def test_castellvi_one_side_unknown_is_noted() -> None:
    rec = castellvi_grade({"left": {"tp_cc_mm": 12.0, "contact": "fusion"}, "right": None})
    assert rec["grade"] == "IIIa"
    assert "right side not assessed" in rec["notes"]


def test_side_grade_unknown_contact_is_not_assessed() -> None:
    assert side_grade({"tp_cc_mm": 25.0, "contact": "unknown"}) is None
    assert side_grade({"contact": "none"}) is None  # no size, no grade


# ------------------------------------------------------------- sagittal cues


def test_disc_cue_hypoplastic_thin_and_normal() -> None:
    hypo = disc_cue(_anchors([9.0, 9.5, 9.0, 4.5]))  # 4.5 / 9.17 = 0.49
    assert (
        hypo.like is not None and hypo.like[SACRAL] == DEFAULT_LSTV.lr_hypoplastic and hypo.like[NONE] == 1.0
    )
    thin = disc_cue(_anchors([9.0, 9.5, 9.0, 6.5]))  # 0.71
    assert thin.like is not None and thin.like[SACRAL] == DEFAULT_LSTV.lr_thin
    normal = disc_cue(_anchors([9.0, 9.5, 9.0, 9.0]))
    assert normal.like is not None and normal.like[SACRAL] == 1.0 and normal.like[NONE] > 1.0


def test_disc_cue_needs_three_discs() -> None:
    c = disc_cue(_anchors([9.0, 9.5]))
    assert c.like is None


def test_body_cues_fused_block_and_wedge() -> None:
    bodies = {
        0: {"anterior": 30.0, "posterior": 30.0, "mean": 40.0},
        1: {"anterior": 27.0, "posterior": 22.0, "mean": 25.0},
        2: {"anterior": 25.0, "posterior": 25.0, "mean": 25.0},
        3: {"anterior": 25.0, "posterior": 25.0, "mean": 25.0},
    }
    cues = {c.name: c for c in body_cues(_anchors([9.0, 9.0, 9.0, 9.0], bodies=bodies))}
    assert (
        cues["s1_block_height"].like is not None
        and cues["s1_block_height"].like[SACRAL] == DEFAULT_LSTV.lr_fused_block
    )
    assert cues["l5_wedge"].like is not None and cues["l5_wedge"].like[SACRAL] == DEFAULT_LSTV.lr_wedge


def test_body_cues_not_measured_without_bodies() -> None:
    cues = {c.name: c for c in body_cues(_anchors([9.0, 9.0, 9.0, 9.0]))}
    assert cues["s1_block_height"].like is None and cues["l5_wedge"].like is None


def test_numbering_cues_neutral_without_a_candidate_shift() -> None:
    cues = {
        c.name: c
        for c in numbering_cues(
            _anchors(
                [9.0] * 4,
                numbering={
                    "posterior": _post(0.075, 0.85, 0.075),
                    "applied_offset": 0,
                    "lumbosacral": {"status": "drop", "applied_drop": 1},
                },
            )
        )
    }
    # the fusion posterior sits at its prior (no shift candidate): not evidence, so not counted
    assert cues["numbering_posterior"].like is None
    assert (
        cues["lumbosacral_check"].like is not None
        and cues["lumbosacral_check"].like[LUMBAR] == DEFAULT_LSTV.lr_lumbosacral_drop
    )


def test_numbering_cues_candidate_shift_counts() -> None:
    cues = {
        c.name: c for c in numbering_cues(_anchors([9.0] * 4, numbering={"posterior": _post(0.1, 0.3, 0.6)}))
    }
    like = cues["numbering_posterior"].like
    assert like is not None and like[SACRAL] == DEFAULT_LSTV.lr_fusion_candidate
    assert like[LUMBAR] == 1.0 and like[NONE] == 1.0


def test_numbering_cues_absent_record() -> None:
    cues = {c.name: c for c in numbering_cues(_anchors([9.0] * 4, numbering={}))}
    assert cues["numbering_posterior"].like is None and cues["lumbosacral_check"].like is None


def test_combine_without_cues_returns_prior() -> None:
    post = combine_cues([])
    assert post[NONE] == pytest.approx(1 - DEFAULT_LSTV.prior_transitional)
    assert post[SACRAL] == pytest.approx(DEFAULT_LSTV.prior_transitional / 2)


# ------------------------------------------------------------- assess


def test_sacral_pattern_is_suspected_and_suggested_up_one_level() -> None:
    bodies = {
        0: {"anterior": 30.0, "posterior": 30.0, "mean": 42.0},
        1: {"anterior": 25.0, "posterior": 25.0, "mean": 25.0},
        2: {"anterior": 25.0, "posterior": 25.0, "mean": 25.0},
        3: {"anterior": 25.0, "posterior": 25.0, "mean": 25.0},
    }
    a = _anchors(
        [9.0, 9.5, 9.0, 4.5],
        bodies=bodies,
        numbering={
            "posterior": _post(0.02, 0.3, 0.68),
            "applied_offset": 0,
            "lumbosacral": {"status": "keep", "applied_drop": 0},
        },
    )
    res = assess_lstv(a)
    assert res.status == STATUS_SUSPECTED and res.kind == SACRAL
    assert res.p_transitional >= 0.5
    sug = {d["count"]: d["suggested"] for d in res.naming["suggested_names"]}
    assert sug["L5-S1"] == "L4-L5"
    assert sug["L4-L5"] == "L3-L4" and sug["L2-L3"] == "L1-L2"  # every count name moves one level cranially
    # the lowest disc is the one that moves to L4-L5 (count L5-S1 -> anatomical L4-L5)
    assert res.naming["suggested_names"][-1] == {"count": "L5-S1", "suggested": "L4-L5"}
    assert res.naming["applied"] is False
    assert a.disc_labels()[-1] == "L5-S1"  # nothing renamed


def test_lumbarised_pattern_suggests_only_lowest_disc_renamed() -> None:
    a = _anchors(
        [9.0, 9.0, 9.0, 9.0],
        numbering={
            "posterior": _post(0.6, 0.2, 0.2),
            "applied_offset": 0,
            "lumbosacral": {"status": "drop", "applied_drop": 1},
        },
    )
    res = assess_lstv(a)
    assert res.kind == LUMBAR and res.status in (STATUS_SUSPECTED, STATUS_POSSIBLE)
    suggested = res.naming["suggested_names"]
    assert suggested[-1] == {"count": "L5-S1", "suggested": "L5-L6"}
    assert all(d["suggested"] == d["count"] for d in suggested[:-1])
    assert OFFSET_OF[LUMBAR] == -1


def test_normal_pattern_not_suspected() -> None:
    a = _anchors(
        [9.0, 9.0, 9.0, 9.0],
        numbering={
            "posterior": _post(0.03, 0.94, 0.03),
            "applied_offset": 0,
            "lumbosacral": {"status": "keep", "applied_drop": 0},
        },
    )
    res = assess_lstv(a)
    assert res.status == STATUS_NOT_SUSPECTED and res.kind is None
    assert res.naming["suggested_names"] is None


def test_equal_kinds_are_unresolved_not_picked() -> None:
    # a hypoplastic lowest disc (sacral) and a lumbosacral removal (lumbar) with equal ratios: a tie, not a pick
    a = _anchors(
        [9.0, 9.0, 9.0, 4.5],
        numbering={
            "posterior": _post(0.1, 0.8, 0.1),
            "applied_offset": 0,
            "lumbosacral": {"status": "drop", "applied_drop": 1},
        },
    )
    res = assess_lstv(a, cfg=LstvConfig(lr_lumbosacral_drop=DEFAULT_LSTV.lr_hypoplastic))
    assert res.kind == UNRESOLVED
    assert res.naming["suggested_names"] is None


def test_no_discs_not_assessed() -> None:
    res = assess_lstv(Anchors())
    assert res.status == STATUS_NOT_ASSESSED and res.kind is None
    assert res.naming["applied"] is False


def test_already_shifted_by_fusion_gives_no_second_suggestion() -> None:
    a = _anchors([9.0, 9.0, 9.0, 4.5], numbering={"posterior": _post(0.02, 0.3, 0.68), "applied_offset": 1})
    res = assess_lstv(a)
    assert res.naming["suggested_names"] is None
    assert "already shifted" in res.naming["note"]


def test_castellvi_record_moves_the_posterior_and_is_recorded() -> None:
    a = _anchors([9.0, 9.0, 9.0, 9.0], numbering={"posterior": _post(0.1, 0.8, 0.1), "applied_offset": 0})
    without = assess_lstv(a)
    with_ct = assess_lstv(
        a,
        transverse={
            "left": {"tp_cc_mm": 12.0, "contact": "pseudoarticulation"},
            "right": {"tp_cc_mm": 8.0, "contact": "none"},
        },
    )
    assert with_ct.castellvi["grade"] == "IIa"
    assert with_ct.p_transitional > without.p_transitional
    assert without.castellvi["status"] == "not_assessed"


def test_sagittal_only_study_never_reports_a_castellvi_grade() -> None:
    a = _anchors([9.0, 9.0, 9.0, 4.5], numbering={"posterior": _post(0.02, 0.3, 0.68), "applied_offset": 0})
    res = assess_lstv(a)
    assert res.castellvi["grade"] is None
    assert any("Castellvi grade not assessed" in w for w in res.warnings)


def test_record_is_json_safe_and_has_sequences() -> None:
    res = assess_lstv(
        _anchors([9.0, 9.0, 9.0, 4.5], numbering={"posterior": _post(0.02, 0.3, 0.68), "applied_offset": 0})
    )
    text = json.dumps(res.to_dict())
    rec = json.loads(text)
    assert {
        "status",
        "kind",
        "p_transitional",
        "castellvi",
        "evidence",
        "naming",
        "not_evaluated",
        "warnings",
    } <= set(rec)
    assert all("sequence" in e for e in rec["evidence"])
    assert rec["naming"]["convention"].startswith("Count convention")


# ------------------------------------------------------------ attach / summary


def test_attach_sets_field_and_warns_once() -> None:
    a = _anchors(
        [9.0, 9.0, 9.0, 4.5],
        numbering={
            "posterior": _post(0.02, 0.3, 0.68),
            "applied_offset": 0,
            "lumbosacral": {"status": "keep", "applied_drop": 0},
        },
    )
    attach_lstv(a)
    n = len(a.warnings)
    attach_lstv(a)
    assert isinstance(a.lstv, dict) and a.lstv["status"] in (
        STATUS_SUSPECTED,
        STATUS_POSSIBLE,
        STATUS_NOT_SUSPECTED,
    )
    if a.lstv["status"] == STATUS_SUSPECTED:
        assert len(a.warnings) == n  # not duplicated on re-run
        assert any("lumbosacral transitional vertebra" in w for w in a.warnings)


def test_anchors_round_trip_keeps_lstv() -> None:
    a = _anchors([9.0, 9.0, 9.0, 9.0], numbering={"posterior": _post(0.03, 0.94, 0.03), "applied_offset": 0})
    attach_lstv(a)
    back = Anchors.from_dict(json.loads(a.to_json()))
    assert back.lstv == json.loads(json.dumps(a.lstv))


def test_summary_carries_the_check() -> None:
    a = _anchors(
        [9.0, 9.0, 9.0, 4.5],
        numbering={"posterior": _post(0.02, 0.3, 0.68), "applied_offset": 0, "status": "unconfirmed"},
    )
    attach_lstv(a)
    s = summary.labelling_summary(a)
    assert s["lstv"] is not None and s["lstv"]["status"] == a.lstv["status"]
    assert "transitional check" in s["lstv"]["line"]
    assert any("transitional check" in line for line in summary.summary_lines(a))
    assert lstv_line(a) == s["lstv"]["line"]


def test_summary_without_check_is_none() -> None:
    assert summary.labelling_summary(_anchors([9.0] * 4))["lstv"] is None


def test_naming_record_without_discs() -> None:
    rec = naming_record(Anchors(), NONE, STATUS_NOT_ASSESSED)
    assert rec["count_names"] == [] and rec["suggested_names"] is None


# ------------------------------------------------------------ phantom contracts


@pytest.mark.parametrize("transitional", [None, "lumbarised_s1", "sacralised_l5"])
def test_pipeline_attaches_check_without_renaming(transitional: str | None) -> None:
    ph = SpinePhantom(PhantomSpec(transitional=transitional))
    anchors = label_spine(phantom_study(ph))
    assert isinstance(anchors.lstv, dict)
    assert anchors.lstv["status"] in (
        STATUS_SUSPECTED,
        STATUS_POSSIBLE,
        STATUS_NOT_SUSPECTED,
        STATUS_NOT_ASSESSED,
    )
    assert anchors.lstv["naming"]["applied"] is False
    assert anchors.lstv["castellvi"]["grade"] is None  # a sagittal-only phantom study cannot grade Castellvi
    # the labels are the labeller's: the suggestion (if any) is a separate list, never written into the discs
    assert [d["count"] for d in anchors.lstv["naming"]["count_names"]] == anchors.disc_labels()


def test_unconfirmed_numbering_withholds_the_verdict_but_keeps_cues() -> None:
    a = _anchors(
        [9.0, 9.0, 9.0, 4.5],
        numbering={"posterior": _post(0.02, 0.3, 0.68), "applied_offset": 0, "status": "unconfirmed"},
    )
    res = assess_lstv(a)
    assert res.status == STATUS_NOT_ASSESSED and res.kind is None
    assert any("numbering unconfirmed" in w for w in res.warnings)
    assert any(c.name == "disc_height" and c.like is not None for c in res.cues)
    assert res.naming["suggested_names"] is None
