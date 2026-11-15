"""lab22: numbering evidence before the calibrated numbering decision (pipeline stage order, signals, caps, v3).

Synthetic features and anchors (every expected value is known), plus one phantom run that records the order of the
pipeline stages and checks that no stack is segmented twice. The calibration v3 acceptance numbers come from the
held-out tool (``tools/calibrate_labeling.py fit-v3``), not from the tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from spinescope.anatomy import conus as conus_mod
from spinescope.anatomy import pipeline
from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.anatomy.calibration import (
    DEFAULT_CALIBRATION_VERSION,
    SACRAL_FEATURES_V3,
    SIGNAL_FEATURES_V3,
    TARGET_PRECISION_V3,
    UNREACHABLE_THRESHOLD,
    VOLUME_FEATURES_V2,
    calibration_path,
    choose_thresholds,
    load_calibration,
    numbering_signals,
    sacral_signals,
    volume_feature_map,
)
from spinescope.anatomy.conus import refresh_conus
from spinescope.anatomy.fusion import relabel_anchors
from spinescope.anatomy.numbering import (
    STATUS_CONFIRMED,
    STATUS_PROBABLE,
    STATUS_UNCONFIRMED,
    numbering_status,
)
from spinescope.config import DEFAULT


def _disc(z: float, height: float = 8.0) -> Disc:
    return Disc(center=[0.0, 45.0, z], normal=[0.0, 0.0, 1.0], height_mm=height, confidence=0.9)


def _three_discs() -> Anchors:
    a = Anchors(method="synthetic")
    a.discs = {"L3-L4": _disc(-20.0), "L4-L5": _disc(-60.0), "L5-S1": _disc(-100.0)}
    return a


# ----------------------------------------------------------------------------------------- numbering signals
def test_v3_signals_present_finite_and_neutral_when_missing():
    a = _three_discs()
    v2 = numbering_signals(a, lumbo_p=[0.95, 0.04, 0.01], votes=[], p_shift=0.0, coverage_low_z=None)
    full = {
        **v2,
        **sacral_signals(s1_below_mm=12.0, s1_ap_ratio=0.5, t1t2_bottom=True),
    }
    assert set(SIGNAL_FEATURES_V3) == set(full)
    assert set(SACRAL_FEATURES_V3) <= set(full) and set(v2) <= set(full)
    assert all(np.isfinite(v) for v in full.values())
    assert full["s1_present"] == 1.0 and full["s1_below_mm"] == 12.0
    assert full["s1_ap_ratio"] == 0.5 and full["t1t2_bottom"] == 1.0

    none = sacral_signals(s1_below_mm=None, s1_ap_ratio=None, t1t2_bottom=False)
    assert none["s1_present"] == 0.0 and none["s1_below_mm"] == 0.0
    assert none["s1_ap_ratio"] == 1.0 and none["t1t2_bottom"] == 0.0


def test_s1_present_uses_the_visibility_threshold():
    thr = DEFAULT.sacrum_min_visible_mm
    below = sacral_signals(s1_below_mm=thr - 0.1, s1_ap_ratio=None, t1t2_bottom=False)
    at = sacral_signals(s1_below_mm=thr, s1_ap_ratio=None, t1t2_bottom=False)
    assert below["s1_present"] == 0.0 and at["s1_present"] == 1.0


def test_volume_feature_map_keeps_the_v1_names_for_old_calibrations():
    feats = volume_feature_map(
        0.9, {"L5-S1": 0.8}, sacrum_below=True, second_present=False, t1t2_fraction=0.0, classical_agree=False
    )
    assert set(VOLUME_FEATURES_V2) - {"lumbo_margin", "gap_bottom_dev"} <= set(feats)


# ----------------------------------------------------------------------------------------- thresholds
def test_confirmed_threshold_is_found_when_precision_0_99_is_reached():
    p = np.linspace(0.5, 1.0, 200)
    correct = (p > 0.9).astype(float)  # every volume above 0.9 is right
    thr = choose_thresholds(
        p,
        correct,
        targets=TARGET_PRECISION_V3,
        fallback={STATUS_CONFIRMED: UNREACHABLE_THRESHOLD, STATUS_PROBABLE: UNREACHABLE_THRESHOLD},
    )
    assert thr[STATUS_CONFIRMED] <= 0.92
    assert thr[STATUS_PROBABLE] <= thr[STATUS_CONFIRMED]
    assert numbering_status(0.99, thr) == STATUS_CONFIRMED


def test_unreachable_target_never_announces_anything():
    p = np.linspace(0.5, 1.0, 200)
    correct = (np.arange(200) % 2).astype(float)  # half wrong everywhere: no threshold reaches 0.97 or 0.99
    thr = choose_thresholds(
        p,
        correct,
        targets=TARGET_PRECISION_V3,
        fallback={STATUS_CONFIRMED: UNREACHABLE_THRESHOLD, STATUS_PROBABLE: UNREACHABLE_THRESHOLD},
    )
    assert thr[STATUS_CONFIRMED] == UNREACHABLE_THRESHOLD and thr[STATUS_PROBABLE] == UNREACHABLE_THRESHOLD
    assert numbering_status(1.0, thr) == STATUS_UNCONFIRMED


# ----------------------------------------------------------------------------------------- caps and recap
def test_confirmed_needs_bone_below_and_no_relabel():
    caps = pipeline._apply_status_caps
    assert caps(
        STATUS_CONFIRMED, sacrum_below=True, applied=False, second_extra=False, lumbo_status="ok"
    ) == (STATUS_CONFIRMED)
    assert caps(STATUS_CONFIRMED, sacrum_below=False, applied=False, second_extra=False) == STATUS_PROBABLE
    assert caps(STATUS_CONFIRMED, sacrum_below=True, applied=True, second_extra=False) == STATUS_PROBABLE
    assert caps(
        STATUS_CONFIRMED, sacrum_below=True, applied=False, second_extra=False, lumbo_status="unsure"
    ) == (STATUS_UNCONFIRMED)


def test_recap_undoes_a_recomputed_status_that_skips_the_caps():
    a = _three_discs()
    a.numbering = {
        "labelled": True,
        "status": STATUS_CONFIRMED,  # what apply_applicability recomputes from the confidence alone
        "sacrum_below": False,
        "relabelled": False,
        "second_series": {},
        "lumbosacral": {"status": "ok"},
    }
    pipeline._recap_status(a)
    assert a.numbering["status"] == STATUS_PROBABLE
    assert any("capped to probable" in w for w in a.warnings)


def test_recap_leaves_unlabelled_studies_alone():
    a = Anchors(method="synthetic")
    a.numbering = {"status": STATUS_CONFIRMED, "sacrum_below": False}
    pipeline._recap_status(a)
    assert a.numbering["status"] == STATUS_CONFIRMED and not a.warnings


# ----------------------------------------------------------------------------------------- calibration versions
def test_calibration_versions_stay_loadable():
    v1 = load_calibration(version=1)
    v2 = load_calibration(version=2)
    assert v1 is not None and v1.version == 1
    assert v2 is not None and v2.version == 2
    assert load_calibration() is not None
    assert DEFAULT_CALIBRATION_VERSION == 2  # v3 is a candidate only (lab22 report)
    assert load_calibration(version=3) is None  # not packaged
    assert calibration_path(3).name == "calibration_v3.json"
    assert (
        calibration_path(2).name == "calibration_v2.json"
        and calibration_path(1).name == "calibration_v1.json"
    )


def test_v2_volume_is_insensitive_to_the_sacrum_feature():
    """Version 2 was fitted with sacrum_below constant (weight exactly 0), so the newly varying S1 presence cannot
    change its confidence."""
    v2 = load_calibration(version=2)
    assert v2 is not None
    base = {k: 0.5 for k in v2.volume.names}
    base.update({"logit_offset": 1.0, "min_logit_raw": 1.0, "lumbo_margin": 0.5, "gap_bottom_dev": 0.1})
    on, off = dict(base, sacrum_below=1.0), dict(base, sacrum_below=0.0)
    p_on = v2.volume.predict(v2.volume_vector(on))[0]
    p_off = v2.volume.predict(v2.volume_vector(off))[0]
    assert p_on == pytest.approx(p_off, abs=1e-12)


# ----------------------------------------------------------------------------------------- geometry through relabelling
def test_relabelling_moves_pre_numbering_geometry_with_the_names():
    a = _three_discs()
    a.discs = {"L4-L5": _disc(-60.0), "L5-S1": _disc(-100.0)}
    a.vertebrae = {
        "S1": Vertebra(
            center=[0, 45, -115.0], body={"level": "S1", "discs_below": 0, "below_lowest_disc_mm": 14.0}
        ),
        "L5": Vertebra(center=[0, 45, -80.0], body={"level": "L5", "discs_below": 1}),
    }
    a.canal = {"per_level": {"L5-S1": {"z_mm": -100.0}, "L5": {"z_mm": -80.0}, "L4-L5": {"z_mm": -60.0}}}
    relabel_anchors(a, 1)  # the lowest disc was one level higher than counted: every name moves cranially
    assert set(a.discs) == {"L3-L4", "L4-L5"}
    assert set(a.vertebrae) == {"L5", "L4"}
    assert a.vertebrae["L5"].body["level"] == "L5" and a.vertebrae["L4"].body["level"] == "L4"
    # the S1 body (0 discs below) is now named L5 and keeps its measurement; the L5 body is now L4
    assert (
        a.vertebrae["L5"].body["discs_below"] == 0 and a.vertebrae["L5"].body["below_lowest_disc_mm"] == 14.0
    )
    assert a.vertebrae["L4"].body["discs_below"] == 1
    assert set(a.canal["per_level"]) <= set(a.discs) | set(a.vertebrae)
    assert "L4-L5" in a.canal["per_level"]


# ----------------------------------------------------------------------------------------- conus
def _conus(z: float, raw: float, level: str | None = None) -> dict[str, Any]:
    return {
        "z": z,
        "xyz": [0.0, 45.0, z],
        "center": [0.0, 45.0, z],
        "level": level,
        "position": None,
        "confidence": raw,
        "confidence_raw": raw,
        "low_lying": None,
        "evidence": ["cord-like signal inside the thecal sac ends here"],
    }


def test_refresh_conus_applies_status_factor_cap_and_names():
    a = _three_discs()
    a.discs["L2-L3"] = _disc(20.0)
    a.conus = _conus(z=-5.0, raw=0.9)  # below the L2-L3 margin (z < 16): low-lying
    a.numbering = {"status": STATUS_UNCONFIRMED}
    refresh_conus(a, DEFAULT)
    assert a.conus["low_lying"] is True
    assert a.conus["confidence"] <= DEFAULT.conus_max_confidence  # experimental cap
    assert a.conus["experimental"] is True
    assert any(w.startswith("possible low-lying conus") for w in a.warnings)
    # Names change (a relabelling): the level follows the new names and the stale warning is rewritten.
    relabel_anchors(a, 1)
    a.numbering = {"status": STATUS_PROBABLE}
    refresh_conus(a, DEFAULT)
    # the reference disc L2-L3 is renamed, so the low-lying flag is recomputed; the warning follows the flag
    assert sum(w.startswith("possible low-lying conus") for w in a.warnings) == (
        0 if not a.conus["low_lying"] else 1
    )


def test_refresh_conus_without_a_tip_is_a_no_op():
    a = _three_discs()
    a.conus = None
    refresh_conus(a, DEFAULT)
    assert a.conus is None


# ----------------------------------------------------------------------------------------- phantom: stage order
@pytest.fixture()
def recorded_order(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Wrap the pipeline stages (call through) and record the order of their calls and every segmentation."""
    from spinescope.anatomy import ml as ml_pkg
    from spinescope.anatomy.ml import labeller

    order: list[str] = []
    real_segment = labeller.segment_stack

    def seg_rec(stack: Any, cfg: Any = DEFAULT, session: Any = None) -> Any:
        order.append(f"segment:{stack.key}")
        return real_segment(stack, cfg, session=session)

    monkeypatch.setattr(labeller, "segment_stack", seg_rec)
    monkeypatch.setattr(ml_pkg, "segment_stack", seg_rec)

    def wrap(name: str, label: str, module: Any = pipeline) -> None:
        real = getattr(module, name)

        def inner(*args: Any, **kwargs: Any) -> Any:
            order.append(label)
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, inner)

    wrap("_lumbosacral_step", "lumbosacral")
    wrap("_geometry_step", "geometry")
    wrap("_conus_stage", "conus")
    wrap("fuse_numbering", "fusion")
    wrap("apply_applicability", "applicability")
    wrap("_recap_status", "recap")
    wrap("refresh_conus", "conus_refresh", conus_mod)
    return order


def test_phantom_stage_order_and_single_segmentation(phantom_study, recorded_order: list[str]) -> None:
    det = pipeline.label_spine_details(phantom_study, backend="ml", calibrate=True)
    order = recorded_order
    segs = [e for e in order if e.startswith("segment:")]
    assert segs, "the learned labeller must run on the primary stack"
    assert len(segs) == len(set(segs)), f"a stack was segmented twice: {segs}"
    if det.anchors.discs:
        stages = ["lumbosacral", "geometry", "conus", "fusion", "applicability", "recap", "conus_refresh"]
        present = [e for e in order if e in stages]
        assert present == stages, f"stage order {present}"
        assert order.index(segs[0]) < order.index("geometry")
    # The numbering status is never announced above what the caps allow (no bone below and not confirmed).
    if det.anchors.discs and not det.anchors.numbering.get("sacrum_below", False):
        assert det.anchors.numbering.get("status") != STATUS_CONFIRMED
