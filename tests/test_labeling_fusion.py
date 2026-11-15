"""Numbering fusion and calibration: deterministic unit tests on synthetic anchors and evidence, plus phantom runs.

The fusion logic is driven by hand-built evidence: anchors from the phantom's ground truth, renamed to a known
offset, and oblique slabs built in code. Nothing here needs SPIDER or the ONNX model.
"""

from __future__ import annotations

import numpy as np
import pytest

from _labels import label_spine
from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.anatomy.calibration import (
    DISC_FEATURES,
    VOLUME_FEATURES,
    Calibration,
    Logistic,
    choose_thresholds,
    disc_feature_rows,
    load_calibration,
    logit,
    spacing_deviation,
    volume_feature_vector,
)
from spinescope.anatomy.fusion import (
    FusionConfig,
    description_votes,
    fuse_numbering,
    geometric_agreement,
    match_discs,
    relabel_anchors,
    stack_z_range,
)
from spinescope.anatomy.numbering import (
    DEFAULT_STATUS_THRESHOLDS,
    STATUS_CONFIRMED,
    STATUS_PROBABLE,
    STATUS_UNCONFIRMED,
    disc_names,
    numbering_status,
    shift_vertebra_name,
)
from spinescope.geometry import Stack
from spinescope.study import from_stacks
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

# ----------------------------------------------------------------- helpers


def _truth_anchors(seed: int = 0) -> tuple[SpinePhantom, Anchors]:
    ph = SpinePhantom(PhantomSpec(seed=seed))
    return ph, Anchors.from_truth(ph.truth)


def _renamed_without_bottom(ph: SpinePhantom) -> Anchors:
    """Truth discs with the lowest one (L5-S1) removed, provisionally renamed by count from the bottom.

    The lowest remaining disc is truly L4-L5: a sacralised L5 that a count from the bottom mislabels as L5-S1.
    """
    truth = Anchors.from_truth(ph.truth)
    order = sorted(truth.discs, key=lambda lab: truth.discs[lab].center[2])  # bottom-up by height
    assert order[0] == "L5-S1"
    kept = order[1:]
    provisional = disc_names(len(kept))
    discs = {prov: truth.discs[lab] for prov, lab in zip(provisional, kept, strict=True)}
    vertebrae = {k: v for k, v in truth.vertebrae.items() if k != "S1"}
    return Anchors(
        discs=discs, vertebrae=vertebrae, method="test", numbering={"method": "test", "confidence": 0.5}
    )


def _slab(center: np.ndarray, normal: np.ndarray, description: str, key: str = "ax_oblique") -> Stack:
    """A one-slice axial-like oblique slab through ``center`` with slice normal ``normal``."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = a - (a @ n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    corner = np.asarray(center, dtype=np.float64) - 20.0 * u - 20.0 * v
    return Stack(
        key=key,
        vol=np.zeros((1, 41, 41), dtype=np.float32),
        ipp=corner[None, :],
        row_dir=u,
        col_dir=v,
        row_mm=1.0,
        col_mm=1.0,
        thickness=4.0,
        description=description,
    )


def _lowest_centre(anchors: Anchors) -> tuple[np.ndarray, np.ndarray]:
    lab = anchors.disc_labels()[-1]
    d = anchors.discs[lab]
    return np.asarray(d.center), np.asarray(d.normal)


# ----------------------------------------------------------------- status policy


def test_numbering_status_thresholds_and_default() -> None:
    th = {"confirmed": 0.9, "probable": 0.6}
    assert numbering_status(0.95, th) == STATUS_CONFIRMED
    assert numbering_status(0.9, th) == STATUS_CONFIRMED
    assert numbering_status(0.7, th) == STATUS_PROBABLE
    assert numbering_status(0.59, th) == STATUS_UNCONFIRMED
    assert numbering_status(0.8) == STATUS_PROBABLE  # default thresholds
    assert DEFAULT_STATUS_THRESHOLDS["confirmed"] > DEFAULT_STATUS_THRESHOLDS["probable"]


def test_shift_vertebra_name_rules() -> None:
    assert shift_vertebra_name("L5", 1) == "L4"
    assert shift_vertebra_name("S1", 1) == "L5"
    assert shift_vertebra_name("S1", -1) == "L6"
    assert shift_vertebra_name("L5", -1) == "L5"  # lumbarised: only the body below the lowest disc moves
    assert shift_vertebra_name("L3", 0) == "L3"
    assert shift_vertebra_name("V9", 1) == "V9"  # unknown names are left alone


# ----------------------------------------------------------------- relabelling


def test_relabel_plus_one_shifts_discs_vertebrae_and_conus() -> None:
    ph, _ = _truth_anchors()
    anchors = _renamed_without_bottom(ph)
    before = dict(anchors.discs)
    before_vertebrae = list(anchors.vertebrae)
    anchors.conus = {"z": 0.0, "level": "L2", "confidence": 0.5}
    mapping = relabel_anchors(anchors, +1)
    assert mapping["L5-S1"] == "L4-L5"  # the old lowest name is one level up after a sacralised L5
    assert "L5-S1" not in anchors.discs
    assert set(anchors.discs) == {k for k in ph.truth["discs"] if k != "L5-S1"}  # now the truth names
    assert anchors.conus["level"] == "L1"  # the conus level moves with the bodies
    assert set(anchors.vertebrae) == {shift_vertebra_name(k, 1) for k in before_vertebrae if k != "S1"}
    assert len(anchors.discs) == len(before)


def test_relabel_minus_one_renames_only_the_lowest_disc() -> None:
    anchors = Anchors(
        discs={
            "L4-L5": Disc(center=[0, 0, 40.0], normal=[0, 0, 1.0]),
            "L5-S1": Disc(center=[0, 0, 10.0], normal=[0, 0, 1.0]),
        },
        vertebrae={"S1": Vertebra(center=[0, 0, 0.0]), "L5": Vertebra(center=[0, 0, 25.0])},
    )
    mapping = relabel_anchors(anchors, -1)
    assert mapping == {"L5-S1": "L5-L6", "L4-L5": "L4-L5"}
    assert set(anchors.discs) == {"L4-L5", "L5-L6"}
    assert set(anchors.vertebrae) == {"L6", "L5"}


def test_relabel_zero_is_identity() -> None:
    ph, _ = _truth_anchors()
    anchors = _renamed_without_bottom(ph)
    names = sorted(anchors.discs)
    assert relabel_anchors(anchors, 0) == {k: k for k in names}
    assert sorted(anchors.discs) == names


# ----------------------------------------------------------------- votes and decision


def test_consistent_truth_is_not_relabelled() -> None:
    _ph, anchors = _truth_anchors()
    res = fuse_numbering(anchors, study=None, slabs=[])
    assert res.offset == 0 and not res.applied and not res.transitional
    assert res.posterior[0] > 0.8
    assert not any(v["method"] == "slab_description" for v in res.votes)


def test_one_named_slab_is_a_candidate_not_applied() -> None:
    ph, _ = _truth_anchors()
    anchors = _renamed_without_bottom(ph)
    c, n = _lowest_centre(anchors)
    res = fuse_numbering(anchors, study=None, slabs=[_slab(c, n, "L4-5 OBL AX")])
    assert res.posterior[1] > 0.5  # the name points to +1 ...
    assert not res.applied  # ... but one slab is below the apply threshold (0.9)
    assert res.transitional_candidate and not res.transitional
    assert any("transitional candidate" in n for n in res.notes)
    assert any("possible transitional" in w for w in res.warnings)


def test_two_agreeing_slabs_apply_the_shift_and_record_both_votes() -> None:
    ph, _ = _truth_anchors()
    anchors = _renamed_without_bottom(ph)
    c, n = _lowest_centre(anchors)
    slabs = [_slab(c, n, "L4-5 OBL AX", key="ax_a"), _slab(c, n, "L4-5 OBL AX", key="ax_b")]
    res = fuse_numbering(anchors, study=None, slabs=slabs)
    assert res.offset == 1 and res.applied
    assert res.transitional and res.transitional_kind == "sacralized"
    assert res.posterior[1] >= 0.9
    desc = [v for v in res.votes if v["method"] == "slab_description"]
    assert len(desc) == 2 and all(len(v["evidence"]) == 1 for v in desc)
    # the shift itself is applied by relabel_anchors, names then match the truth
    relabel_anchors(anchors, res.offset)
    assert "L4-L5" in anchors.discs and "L5-S1" not in anchors.discs
    assert any("names shifted" in w for w in res.warnings)


def test_slab_naming_the_lowest_level_gives_offset_zero() -> None:
    _ph, anchors = _truth_anchors()
    c, n = _lowest_centre(anchors)
    slabs = [_slab(c, n, "L5-S1 OBL AX", key="ax_a"), _slab(c, n, "L5-S1 OBL AX", key="ax_b")]
    res = fuse_numbering(anchors, study=None, slabs=slabs)
    assert res.offset == 0 and not res.applied


def test_description_vote_ignores_slabs_that_miss_the_lowest_disc() -> None:
    ph, _ = _truth_anchors()
    anchors = _renamed_without_bottom(ph)
    c, n = _lowest_centre(anchors)
    far = np.asarray(c) + np.array([0.0, 0.0, 120.0])  # 12 cm above the lowest disc
    assert description_votes([_slab(far, n, "L4-5 OBL AX")], anchors, FusionConfig()) == []


def test_description_vote_strength_is_configurable() -> None:
    ph, _ = _truth_anchors()
    anchors = _renamed_without_bottom(ph)
    c, n = _lowest_centre(anchors)
    slab = _slab(c, n, "L4-5 OBL AX")
    default = fuse_numbering(anchors, study=None, slabs=[slab])
    strong = fuse_numbering(
        anchors, study=None, slabs=[slab], cfg=FusionConfig(name_p_match=0.99, name_p_other=0.01)
    )
    assert strong.posterior[1] > default.posterior[1]


def test_second_series_extra_disc_below_is_a_warning_and_blocks_the_shift() -> None:
    ph, anchors = _truth_anchors()
    second = Anchors.from_truth(ph.truth)
    lowest = second.discs["L5-S1"]
    extra = Disc(
        center=[lowest.center[0], lowest.center[1], lowest.center[2] - 30.0],
        normal=lowest.normal,
        height_mm=9.0,
        confidence=0.9,
    )
    second.discs["L6-S1"] = extra
    cov = (lowest.center[2] - 80.0, lowest.center[2] + 80.0)
    res = fuse_numbering(anchors, study=None, slabs=[], second=second, second_key="sag_t1", coverage_z=cov)
    assert res.second is not None
    assert res.second["extra_below"] == ["L6-S1"]
    assert res.second["agree"] == len(anchors.discs)
    assert any("shows 1 disc(s) below" in w for w in res.warnings)
    assert not res.applied


def test_second_series_extra_disc_outside_coverage_is_not_counted() -> None:
    ph, anchors = _truth_anchors()
    second = Anchors.from_truth(ph.truth)
    lowest = second.discs["L5-S1"]
    second.discs["L6-S1"] = Disc(
        center=[lowest.center[0], lowest.center[1], lowest.center[2] - 30.0],
        normal=lowest.normal,
        height_mm=9.0,
    )
    res = fuse_numbering(
        anchors, study=None, slabs=[], second=second, coverage_z=(lowest.center[2] - 5.0, 500.0)
    )
    assert res.second is not None and res.second["extra_below"] == []


def test_classical_weak_vote_only_when_geometry_agrees() -> None:
    ph, anchors = _truth_anchors()
    agree = Anchors.from_truth(ph.truth)
    res = fuse_numbering(anchors, study=None, slabs=[], classical=agree)
    assert res.classical is not None and res.classical["agrees"]
    assert any(v["method"] == "classical_tilt" and v["weight"] == 0.5 for v in res.votes)
    # a classical chain that is far away is recorded as disagreeing and contributes no vote
    far = Anchors(
        discs={k: Disc(center=[v + 300.0 for v in d.center], normal=d.normal) for k, d in agree.discs.items()}
    )
    res2 = fuse_numbering(anchors, study=None, slabs=[], classical=far)
    assert res2.classical is not None and not res2.classical["agrees"]
    assert not any(v["method"] == "classical_tilt" for v in res2.votes)


def test_no_discs_is_unconfirmed_and_unfused() -> None:
    res = fuse_numbering(Anchors(), study=None, slabs=[])
    assert res.offset == 0 and not res.applied and res.votes == []
    assert res.warnings and "no discs" in res.warnings[0]


def test_match_discs_and_geometric_agreement() -> None:
    ph, anchors = _truth_anchors()
    same = Anchors.from_truth(ph.truth)
    m = match_discs(anchors, same, 6.0)
    assert all(v == k for k, v in m.items())
    ok, frac = geometric_agreement(anchors, same)
    assert ok and frac == 1.0
    empty_ok, _ = geometric_agreement(anchors, Anchors())
    assert not empty_ok


def test_stack_z_range_covers_the_slices() -> None:
    # a sagittal stack: rows run along z (41 mm), columns along y, slices along x
    st = Stack(
        key="sag",
        vol=np.zeros((2, 41, 41), dtype=np.float32),
        ipp=np.array([[0.0, 0.0, 5.0], [3.0, 0.0, 5.0]]),
        row_dir=np.array([0.0, 1.0, 0.0]),
        col_dir=np.array([0.0, 0.0, 1.0]),
        row_mm=1.0,
        col_mm=1.0,
        thickness=3.0,
    )
    lo, hi = stack_z_range(st)
    assert lo == pytest.approx(5.0) and hi == pytest.approx(45.0)


# ----------------------------------------------------------------- calibration features and model


def test_spacing_deviation_is_zero_for_regular_spacing_and_positive_otherwise() -> None:
    regular = Anchors(discs={f"D{i}": Disc(center=[0, 0, 30.0 * i], normal=[0, 0, 1.0]) for i in range(5)})
    assert all(v == pytest.approx(0.0) for v in spacing_deviation(regular).values())
    z = [0.0, 30.0, 60.0, 105.0]  # one gap 45 mm long
    irregular = Anchors(
        discs={f"D{i}": Disc(center=[0, 0, zz], normal=[0, 0, 1.0]) for i, zz in enumerate(z)}
    )
    dev = spacing_deviation(irregular)
    assert dev["D2"] > 0.2 and dev["D3"] > 0.2


def test_disc_and_volume_feature_shapes() -> None:
    _ph, anchors = _truth_anchors()
    raw = {k: 0.9 for k in anchors.discs}
    rows = disc_feature_rows(
        anchors, raw, {k: True for k in anchors.discs}, second_present=True, sacrum_below=True
    )
    assert set(rows) == set(anchors.discs)
    assert all(v.shape == (len(DISC_FEATURES),) for v in rows.values())
    assert all(v[4] == 1.0 and v[3] == 1.0 for v in rows.values())  # second series present and agreeing
    vec = volume_feature_vector(
        0.9, raw, sacrum_below=False, second_present=False, t1t2_fraction=0.0, classical_agree=True
    )
    assert vec.shape == (len(VOLUME_FEATURES),)
    assert vec[1] == 0.0 and vec[4] == 1.0
    assert logit(0.5) == pytest.approx(0.0)


def test_logistic_fit_orders_probabilities_and_round_trips() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + 0.2 * rng.normal(size=400) > 0).astype(float)
    model = Logistic.fit(X, y, ("a", "b"), l2=1.0)
    p = model.predict(np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]))
    assert p[0] < p[1] < p[2] and np.all((p > 0) & (p < 1))
    again = Logistic.from_dict(model.to_dict())
    assert np.allclose(again.predict(X[:5]), model.predict(X[:5]))
    # the same data gives the same model (deterministic)
    assert Logistic.fit(X, y, ("a", "b"), l2=1.0).weights == model.weights


def test_choose_thresholds_uses_training_precision() -> None:
    p = np.linspace(0.0, 1.0, 401)
    correct = (p >= 0.8).astype(float)  # precision 1.0 above 0.8, 0 below
    th = choose_thresholds(p, correct)
    assert 0.79 <= th[STATUS_CONFIRMED] <= 0.81
    assert th[STATUS_PROBABLE] <= th[STATUS_CONFIRMED]
    # nothing reaches the target: fall back to the documented defaults
    none_ok = choose_thresholds(p, np.zeros_like(p))
    assert none_ok == DEFAULT_STATUS_THRESHOLDS


def test_calibration_save_and_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, len(DISC_FEATURES)))
    y = (X[:, 0] > 0).astype(float)
    disc = Logistic.fit(X, y, DISC_FEATURES)
    Xv = rng.normal(size=(200, len(VOLUME_FEATURES)))
    yv = (Xv[:, 0] > 0).astype(float)
    vol = Logistic.fit(Xv, yv, VOLUME_FEATURES)
    cal = Calibration(disc=disc, volume=vol, thresholds={"confirmed": 0.9, "probable": 0.6}, source="test")
    path = tmp_path / "cal.json"
    cal.save(path)
    back = load_calibration(str(path))
    assert back is not None
    assert back.thresholds == cal.thresholds
    assert np.allclose(back.disc.predict(X[:4]), disc.predict(X[:4]))


# ----------------------------------------------------------------- phantom end to end (classical backend)


@pytest.fixture(scope="module")
def phantom_single() -> object:
    ph = SpinePhantom(PhantomSpec(seed=2))
    st = ph.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    return from_stacks([st])


@pytest.fixture(scope="module")
def phantom_two_series() -> object:
    ph = SpinePhantom(PhantomSpec(seed=2))
    t2 = ph.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    t1 = ph.render("sag", "t1", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    return from_stacks([t2, t1])


def test_label_spine_numbering_record_carries_status_and_votes(phantom_single) -> None:  # type: ignore[no-untyped-def]
    anc = label_spine(phantom_single, backend="classical")
    num = anc.numbering
    assert num["status"] in {STATUS_CONFIRMED, STATUS_PROBABLE, STATUS_UNCONFIRMED}
    assert num["method"].startswith("fusion/1")
    assert num["offset"] == 0 and num["relabelled"] is False and num["transitional"] is False
    assert num["votes"] and all(
        {"method", "best", "probs", "weight", "evidence"} <= set(v) for v in num["votes"]
    )
    assert num["calibrated"] is False  # the classical backend is not calibrated
    assert 0.0 <= num["confidence"] <= 1.0
    assert num["second_series"] is None


def test_classical_second_opinion_is_opt_in_for_the_learned_labeller(phantom_single) -> None:  # type: ignore[no-untyped-def]
    """Default: no classical opinion in the learned numbering (its cost, lab14_perf); opt in with the flag."""
    pytest.importorskip("onnxruntime")
    from spinescope.config import Config

    assert Config().classical_second_opinion is False
    off = label_spine(phantom_single, backend="ml")
    assert off.numbering["classical"] is None
    on = label_spine(phantom_single, Config(classical_second_opinion=True), backend="ml")
    assert on.numbering["classical"] is not None


def test_label_spine_second_series_is_recorded(phantom_two_series) -> None:  # type: ignore[no-untyped-def]
    anc = label_spine(phantom_two_series, backend="classical")
    rec = anc.numbering["second_series"]
    assert rec is not None and rec["key"] and rec["agree"] <= len(anc.discs)
    assert anc.numbering["second_series_agreement"] == rec["agree"]


def test_uncalibrated_numbering_is_capped_below_confirmed_without_bone() -> None:
    from spinescope.anatomy.pipeline import _apply_status_caps

    assert (
        _apply_status_caps(STATUS_CONFIRMED, sacrum_below=False, applied=False, second_extra=False)
        == STATUS_PROBABLE
    )
    assert (
        _apply_status_caps(STATUS_CONFIRMED, sacrum_below=True, applied=True, second_extra=False)
        == STATUS_PROBABLE
    )
    assert (
        _apply_status_caps(STATUS_CONFIRMED, sacrum_below=True, applied=False, second_extra=True)
        == STATUS_PROBABLE
    )
    assert (
        _apply_status_caps(STATUS_CONFIRMED, sacrum_below=True, applied=False, second_extra=False)
        == STATUS_CONFIRMED
    )
    assert (
        _apply_status_caps(STATUS_UNCONFIRMED, sacrum_below=True, applied=False, second_extra=False)
        == STATUS_UNCONFIRMED
    )


def test_no_sagittal_series_is_unconfirmed() -> None:
    from spinescope.study import Study

    anc = label_spine(Study(stacks={}, info={}), backend="classical")
    assert anc.numbering["status"] == STATUS_UNCONFIRMED
    assert anc.discs == {}
