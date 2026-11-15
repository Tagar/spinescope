"""Numbering reliability (lab18): version 2 signals, calibration v2 and the v1 fallback. Synthetic, no datasets.

The signals are computed from hand-built anchors (a sacral-like short bottom disc, a truncated field of view, an
absent lumbosacral check), so each rule is checked on a known input. The packaged-calibration checks skip when the
file is missing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.anatomy.calibration import (
    SIGNAL_FEATURES,
    SIGNAL_FEATURES_V3,
    VOLUME_FEATURES,
    VOLUME_FEATURES_V1,
    VOLUME_FEATURES_V2,
    Calibration,
    Logistic,
    calibration_path,
    load_calibration,
    numbering_signals,
    volume_feature_map,
    volume_feature_vector,
)
from spinescope.anatomy.numbering import STATUS_CONFIRMED, STATUS_PROBABLE


def _disc(z: float, h: float | None, conf: float = 0.9) -> Disc:
    return Disc(center=[0.0, 0.0, z], normal=[0.0, 0.0, 1.0], height_mm=h, confidence=conf)


def _anchors(zs_heights: list[tuple[float, float | None]]) -> Anchors:
    """Discs at the given (z, height) pairs, named by position (the signals do not use the names)."""
    a = Anchors()
    for i, (z, h) in enumerate(zs_heights):
        a.discs[f"D{i}"] = _disc(z, h)
    return a


def _regular() -> Anchors:
    """Five discs, 30 mm apart, 10 mm high: a regular lumbar chain."""
    return _anchors([(-120.0 + 30.0 * i, 10.0) for i in range(5)])


def test_signal_names_match_the_returned_keys() -> None:
    sig = numbering_signals(_regular(), lumbo_p=None, votes=[], p_shift=0.0, coverage_low_z=None)
    assert tuple(sig) == SIGNAL_FEATURES
    assert all(math.isfinite(v) for v in sig.values())


def test_regular_chain_has_neutral_signals() -> None:
    sig = numbering_signals(_regular(), lumbo_p=None, votes=[], p_shift=0.02, coverage_low_z=None)
    assert sig["h_bottom"] == pytest.approx(1.0)
    assert sig["h_second"] == pytest.approx(1.0)
    assert sig["gap_bottom_dev"] == pytest.approx(0.0, abs=1e-9)
    assert sig["n_discs"] == 5.0
    assert sig["lumbo_available"] == 0.0
    assert sig["vote_shift_frac"] == 0.0
    assert sig["p_shift"] == pytest.approx(0.02)


def test_short_bottom_disc_and_wide_bottom_gap_are_visible() -> None:
    # Lowest disc 5 mm high (sacral-like) and 45 mm below the next one (versus 30 mm spacing above).
    a = _anchors([(-120.0, 5.0), (-75.0, 10.0), (-45.0, 10.0), (-15.0, 10.0)])
    sig = numbering_signals(a, lumbo_p=None, votes=[], p_shift=0.0, coverage_low_z=None)
    assert sig["h_bottom"] < 0.6
    assert sig["h_second"] == pytest.approx(1.0)
    assert sig["gap_bottom_dev"] > 0.3


def test_lumbosacral_margin_uses_the_discs_that_remain() -> None:
    sig = numbering_signals(_regular(), lumbo_p=[1.0, 0.1, 0.0], votes=[], p_shift=0.0, coverage_low_z=None)
    assert sig["lumbo_available"] == 1.0
    assert sig["lumbo_p0"] == pytest.approx(1.0)
    assert sig["lumbo_p1"] == pytest.approx(0.1)
    assert sig["lumbo_margin"] == pytest.approx(0.9)
    # a sacral remnant the CNN also scores high: the margin collapses
    tied = numbering_signals(_regular(), lumbo_p=[0.99, 0.98], votes=[], p_shift=0.0, coverage_low_z=None)
    assert tied["lumbo_margin"] == pytest.approx(0.01)


def test_fov_margin_below_the_lowest_disc() -> None:
    a = _regular()  # lowest disc at z = -120
    sig = numbering_signals(a, lumbo_p=None, votes=[], p_shift=0.0, coverage_low_z=-150.0)
    assert sig["fov_below_mm"] == pytest.approx(30.0)
    trunc = numbering_signals(a, lumbo_p=None, votes=[], p_shift=0.0, coverage_low_z=-121.0)
    assert trunc["fov_below_mm"] == pytest.approx(1.0)


def test_vote_fraction_and_shift_posterior() -> None:
    votes = [{"method": "slab", "best": 1}, {"method": "tilt", "best": 0}, {"method": "conus", "best": 0}]
    sig = numbering_signals(_regular(), lumbo_p=None, votes=votes, p_shift=0.4, coverage_low_z=None)
    assert sig["vote_shift_frac"] == pytest.approx(1.0 / 3.0)
    assert sig["p_shift"] == pytest.approx(0.4)
    none = numbering_signals(_regular(), lumbo_p=None, votes=[], p_shift=0.0, coverage_low_z=None)
    assert none["vote_shift_frac"] == 0.0


def test_empty_anchors_are_finite() -> None:
    sig = numbering_signals(Anchors(), lumbo_p=[], votes=[], p_shift=0.0, coverage_low_z=None)
    assert all(math.isfinite(v) for v in sig.values())
    assert sig["n_discs"] == 0.0


def test_volume_feature_map_matches_the_v1_vector() -> None:
    kw = dict(sacrum_below=False, second_present=True, t1t2_fraction=0.5, classical_agree=True)
    raw = {"A": 0.9, "B": 0.4}
    feats = volume_feature_map(0.8, raw, **kw)
    assert tuple(feats) == VOLUME_FEATURES
    np.testing.assert_allclose(volume_feature_vector(0.8, raw, **kw), [feats[k] for k in VOLUME_FEATURES])
    assert VOLUME_FEATURES_V2[: len(VOLUME_FEATURES)] == VOLUME_FEATURES
    assert set(VOLUME_FEATURES_V2) - set(VOLUME_FEATURES) <= set(SIGNAL_FEATURES)
    assert VOLUME_FEATURES_V1 == VOLUME_FEATURES


def test_calibration_selects_its_own_volume_features() -> None:
    names = ("min_logit_raw", "h_bottom")
    vol = Logistic(names=names, mean=[0.0, 1.0], scale=[1.0, 0.5], weights=[1.0, -1.0], bias=0.0)
    disc = Logistic(names=("logit_raw",), mean=[0.0], scale=[1.0], weights=[1.0], bias=0.0)
    cal = Calibration(
        disc=disc, volume=vol, thresholds={STATUS_CONFIRMED: 0.9, STATUS_PROBABLE: 0.6}, version=2
    )
    vec = cal.volume_vector({"min_logit_raw": 2.0, "h_bottom": 0.5, "unused": 9.0})
    np.testing.assert_allclose(vec, [2.0, 0.5])
    back = Calibration.from_dict(cal.to_dict())
    assert back.version == 2 and back.volume.names == names


def test_v1_file_round_trips_and_stays_loadable() -> None:
    v1 = load_calibration(version=1)
    if v1 is None:
        pytest.skip("calibration_v1.json not packaged")
    assert v1.version == 1
    assert v1.volume.names == VOLUME_FEATURES_V1
    assert calibration_path(1).name == "calibration_v1.json"


def test_packaged_v2_is_consistent() -> None:
    cal = load_calibration(version=2)  # the lab18 file stays loadable (the default is v3 since lab22)
    if cal is None:
        pytest.skip("no packaged v2 calibration")
    assert cal.version == 2
    assert set(cal.volume.names) <= set(VOLUME_FEATURES_V2)
    assert cal.thresholds[STATUS_CONFIRMED] >= cal.thresholds[STATUS_PROBABLE] > 0.0
    x = np.zeros((3, len(cal.volume.names)))
    p = cal.volume.predict(x)
    assert np.all((p > 0) & (p < 1))


def test_label_spine_details_carry_signals_on_a_phantom(phantom_study) -> None:  # type: ignore[no-untyped-def]
    """The default pipeline records the signals and the named volume features for every labelled study."""
    from spinescope.anatomy.pipeline import label_spine_details

    det = label_spine_details(phantom_study, calibrate=False)
    if not det.anchors.discs:
        pytest.skip("phantom not labelled by this backend")
    assert set(det.signals) == set(SIGNAL_FEATURES_V3)  # v2 keys plus the lab22 sacral signals
    assert set(VOLUME_FEATURES) <= set(det.volume_feature_dict)
    assert all(math.isfinite(v) for v in det.signals.values())
