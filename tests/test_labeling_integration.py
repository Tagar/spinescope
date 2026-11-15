"""lab37 integration review: regression tests for the cross-stage fixes in the labelling pipeline.

Each test names the defect it guards. Phantom studies only (no public data needed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from _studies import phantom_study
from spinescope.anatomy import ml as ml_pkg
from spinescope.anatomy import pipeline
from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.anatomy.ml import labeller
from spinescope.anatomy.numbering import STATUS_CONFIRMED, STATUS_PROBABLE, STATUS_UNCONFIRMED
from spinescope.anatomy.verify import apply_answer
from spinescope.config import DEFAULT


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture()
def segmentation_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the stack key of every network pass (call through)."""
    calls: list[str] = []
    real = labeller.segment_stack

    def rec(stack: Any, cfg: Any = DEFAULT, session: Any = None, **kw: Any) -> Any:
        calls.append(stack.key)
        return real(stack, cfg, session=session, **kw)

    monkeypatch.setattr(labeller, "segment_stack", rec)
    monkeypatch.setattr(ml_pkg, "segment_stack", rec)
    return calls


@pytest.fixture()
def stage_order(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the order of the final caps and the conus derivation."""
    from spinescope.anatomy import conus as conus_mod

    order: list[str] = []

    def wrap(module: Any, name: str, label: str) -> None:
        real = getattr(module, name)

        def inner(*args: Any, **kwargs: Any) -> Any:
            order.append(label)
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, inner)

    wrap(pipeline, "_quality_stage", "quality")
    wrap(pipeline, "_lstv_stage", "lstv")
    wrap(conus_mod, "refresh_conus", "conus_refresh")
    return order


def _t2_t1_study(phantom: Any) -> Any:
    """Sagittal T2 and T1 (the second series is T1, the conus cross-contrast is T1 too)."""
    return phantom_study(phantom, contrasts=("t2", "t1"))


def _disc(z: float) -> Disc:
    return Disc(center=[0.0, 45.0, z], normal=[0.0, 0.0, 1.0], height_mm=8.0, confidence=0.9)


def _labelled(status: str | None, confidence: float = 0.95) -> Anchors:
    a = Anchors(method="synthetic")
    a.discs = {"L4-L5": _disc(-60.0), "L5-S1": _disc(-100.0)}
    a.numbering = {"method": "synthetic", "confidence": confidence}
    if status is not None:
        a.numbering["status"] = status
    a.applicability = {"unconfirmed_reasons": []}
    return a


# ------------------------------------------------------------------------------- repeated network work
def test_second_series_is_segmented_once_when_it_is_also_the_conus_contrast(
    phantom: Any, segmentation_calls: list[str]
) -> None:
    """Defect: the T1 series was run through the network twice, once as the second series and once as the conus
    cross-contrast check (``locate_conus`` ignored the outputs already computed). Each stack runs once now."""
    study = _t2_t1_study(phantom)
    pipeline.label_spine_details(study, backend="ml", calibrate=True)
    assert segmentation_calls, "the learned labeller must run"
    assert len(segmentation_calls) == len(set(segmentation_calls)), f"segmented twice: {segmentation_calls}"


# ------------------------------------------------------------------------------ stage order of the caps
def test_conus_is_derived_after_every_status_cap(phantom: Any, stage_order: list[str]) -> None:
    """Defect: the conus confidence and its 'numbering <status>' evidence were computed before the image-quality and
    transitional-vertebra caps, so a status capped afterwards was still reported as the uncapped one."""
    det = pipeline.label_spine_details(phantom_study(phantom), backend="ml", calibrate=True)
    if not det.anchors.discs:
        pytest.skip("phantom study not labelled")
    assert stage_order == ["quality", "lstv", "conus_refresh"], stage_order


def test_conus_status_factor_matches_the_final_status(phantom: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cap from the image-quality stage must reach the conus record (same status word, same factor)."""
    real_conus = pipeline._conus_stage
    real_quality = pipeline._quality_stage

    def conus_below_discs(study: Any, anchors: Anchors, cfg: Any, seg: Any, segs: Any = None) -> None:
        real_conus(study, anchors, cfg, seg, segs)
        if anchors.discs:  # a located tip below the lowest disc (the phantom has no cord to find)
            z = min(d.center[2] for d in anchors.discs.values()) - 10.0
            anchors.conus = {
                "z": z,
                "center": [0.0, 45.0, z],
                "level": None,
                "confidence": 0.8,
                "evidence": [],
            }

    def cap_to_probable(study: Any, anchors: Anchors, cfg: Any) -> None:
        real_quality(study, anchors, cfg)
        if anchors.numbering.get("status") == STATUS_CONFIRMED:
            anchors.numbering["status"] = STATUS_PROBABLE

    monkeypatch.setattr(pipeline, "_conus_stage", conus_below_discs)
    monkeypatch.setattr(pipeline, "_quality_stage", cap_to_probable)
    det = pipeline.label_spine_details(phantom_study(phantom), backend="ml", calibrate=True)
    conus = det.anchors.conus
    assert isinstance(conus, dict) and conus.get("z") is not None
    final = det.anchors.numbering["status"]
    ev = [e for e in conus.get("evidence", []) if e.startswith("level names:")]
    assert ev and ev[-1] == f"level names: numbering {final}", ev


# -------------------------------------------------------------------------------- status warnings
def _status_lines(warnings: list[str]) -> list[str]:
    return [w for w in warnings if w.startswith("numbering ") and "level names need confirmation" in w]


def test_one_status_summary_matching_the_final_status(phantom: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: each stage wrote its own status sentence, so a status that a later cap changed was announced with the
    wrong word (and the labeller's "numbering unconfirmed" line stood under a probable status). One summary, written
    from the final status."""
    real_quality = pipeline._quality_stage

    def cap_to_probable(study: Any, anchors: Anchors, cfg: Any) -> None:
        real_quality(study, anchors, cfg)
        if anchors.discs:
            anchors.numbering["status"] = STATUS_PROBABLE

    monkeypatch.setattr(pipeline, "_quality_stage", cap_to_probable)
    det = pipeline.label_spine_details(phantom_study(phantom), backend="ml", calibrate=True)
    if not det.anchors.discs:
        pytest.skip("phantom study not labelled")
    assert det.anchors.numbering["status"] == STATUS_PROBABLE
    conf = float(det.anchors.numbering["confidence"])
    assert _status_lines(det.anchors.warnings) == [
        f"numbering {STATUS_PROBABLE} (P={conf:.2f}): level names need confirmation"
    ]
    assert not any(w.startswith("numbering unconfirmed") for w in det.anchors.warnings)


# ---------------------------------------------------------------------------- AI reader trust (verify)
def test_human_answer_is_probable_never_confirmed() -> None:
    """Defect: a human answer to the verification packet set 'confirmed' on its own (the viewer's explicit Confirm
    is the only confirm; the file reloads as probable without it, so the two disagreed)."""
    a = _labelled(STATUS_UNCONFIRMED, confidence=0.99)
    apply_answer(a, {"lowest_disc_is": "L5-S1"}, who="Reviewer", source="human")
    assert a.numbering["status"] == STATUS_PROBABLE


def test_ai_answer_cannot_raise_a_missing_status() -> None:
    """Defect: an AI answer on anchors with no status key took 'probable' (the missing prior was treated as no cap).
    A missing or unknown prior status counts as unconfirmed, and a missing confidence as zero."""
    a = _labelled(None, confidence=0.99)
    a.numbering.pop("confidence")
    apply_answer(a, {"lowest_disc_is": "L5-S1"}, who="reader", source="ai_reader")
    assert a.numbering["status"] == STATUS_UNCONFIRMED
    assert float(a.numbering["confidence"]) == 0.0


def test_ai_answer_still_caps_a_confirmed_status_at_probable() -> None:
    a = _labelled(STATUS_CONFIRMED, confidence=0.99)
    apply_answer(a, {"lowest_disc_is": "L5-S1"}, who="reader", source="ai_reader")
    assert a.numbering["status"] == STATUS_PROBABLE


# ------------------------------------------------------------------------------ image-quality ghosting
def test_pearson_matches_numpy_corrcoef() -> None:
    """The ghosting detector's per-lag Pearson r equals np.corrcoef (same value, less overhead)."""
    from spinescope.anatomy.artifacts import _pearson

    rng = np.random.default_rng(7)
    for _ in range(20):
        a = rng.normal(size=400)
        b = 0.4 * a + rng.normal(size=400)
        assert _pearson(a, b) == pytest.approx(float(np.corrcoef(a, b)[0, 1]), abs=1e-12)
    assert _pearson(np.ones(60), rng.normal(size=60)) is None  # a constant vector has no correlation


# ------------------------------------------------------------------------------ transitional-vertebra note
def test_suspected_transitional_message_is_kept_when_the_kind_is_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defect: attach_lstv kept warnings[0] as the suspicion, so an unresolved kind ("sacralised or lumbarised, about
    equally likely") replaced the message that says a transitional vertebra is possible."""
    from spinescope.anatomy import lstv

    suspected = (
        "possible lumbosacral transitional vertebra (unresolved, P=0.80): level names are count-based; "
        "confirm on axial or coronal images before the names are used"
    )
    res = lstv.LstvResult(
        status=lstv.STATUS_SUSPECTED,
        kind="unresolved",
        p_transitional=0.8,
        p_kind=0.5,
        posterior={},
        cues=[],
        castellvi={},
        naming={},
        warnings=[
            "transitional kind unresolved: sacralised L5 and lumbarised S1 about equally likely",
            suspected,
        ],
    )
    monkeypatch.setattr(lstv, "assess_lstv", lambda anchors, transverse=None: res)
    a = _labelled(STATUS_CONFIRMED)
    lstv.attach_lstv(a)
    assert suspected in a.warnings


# ------------------------------------------------------------------------------ opt-in augmentation guard
def test_failing_test_time_augmentation_warns_and_keeps_the_labels(
    phantom: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect: an exception in the opt-in test-time augmentation escaped label_spine. It is additional, so it warns."""
    from dataclasses import replace

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("variant failed")

    monkeypatch.setattr(pipeline, "analyse_tta", boom)
    cfg = replace(DEFAULT, ml_tta=True)
    det = pipeline.label_spine_details(phantom_study(phantom), cfg, backend="ml", calibrate=True)
    assert any("test-time augmentation not measured (RuntimeError)" in w for w in det.anchors.warnings)
    assert det.tta == {}
