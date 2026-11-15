"""Applicability (spinescope.anatomy.ood): when the labeller must not label.

Pure-logic tests need no network. Tests that run the learned labeller need onnxruntime and the packaged weights and
are skipped otherwise. Inputs: the synthetic phantom and its robustness protocols (FOV crops, STIR, PD, 3-D
isotropic), phantom crops with no spine, rendered random structures and a brain-like ellipse image. The SPIDER
regression guard runs only when the public SPIDER data are present.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spinescope.anatomy import Anchors
from spinescope.anatomy.anchors import Disc
from spinescope.anatomy.ml import model_path
from spinescope.anatomy.ood import (
    LABELLED_VERDICTS,
    LUMBAR,
    NO_SAGITTAL,
    NOT_LUMBAR,
    NOT_SPINE,
    THORACOLUMBAR,
    UNCERTAIN,
    WHOLE_SPINE,
    Applicability,
    acquisition_cues,
    apply_applicability,
    assess_applicability,
    disc_region,
    disc_regions,
    fov_coverage,
    region_from_text,
)
from spinescope.config import DEFAULT
from spinescope.geometry import Stack
from spinescope.study import Study, from_stacks, load_study
from spinescope.testing.phantom import FAKE_PHI, PhantomSpec, SpinePhantom
from spinescope.testing.robustness import SCENARIOS, Protocol, Scenario, build_case


def _ml_ready() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_path().exists()


requires_ml = pytest.mark.skipif(not _ml_ready(), reason="needs onnxruntime and the packaged labeller")


def _scenario(name: str) -> Scenario:
    return next(s for s in SCENARIOS if s.name == name)


def _sag_stack(
    vol: np.ndarray, *, key: str = "sag_t2", desc: str = "SAG T2", meta: dict | None = None
) -> Stack:
    """A sagittal stack on the phantom's geometry (x normal, z along the rows, 5 mm slices, 1.2 mm pixels)."""
    n = vol.shape[0]
    ipp = np.array([[-30.0 + 5.0 * k, 0.0, 100.0] for k in range(n)])
    dicom = {"SeriesDescription": desc, "RepetitionTime": 3000.0, "EchoTime": 100.0}
    dicom.update((meta or {}).get("dicom", {}))
    return Stack(
        key=key,
        vol=vol.astype(np.float32),
        ipp=ipp,
        row_dir=np.array([0.0, 1.0, 0.0]),
        col_dir=np.array([0.0, 0.0, -1.0]),
        row_mm=1.2,
        col_mm=1.2,
        thickness=4.0,
        meta={"dicom": dicom, **{k: v for k, v in (meta or {}).items() if k != "dicom"}},
    )


def _random_structures(seed: int, n: int = 13, rows: int = 192, cols: int = 192) -> np.ndarray:
    """Random ellipses and rectangles of random intensity on each slice, with noise (seeded)."""
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[0:rows, 0:cols]
    vol = np.zeros((n, rows, cols))
    for k in range(n):
        img = np.zeros((rows, cols))
        for _ in range(12):
            r0, c0 = rng.uniform(0, rows), rng.uniform(0, cols)
            a, b = rng.uniform(5, 40, 2)
            v = rng.uniform(0.2, 1.0)
            if rng.random() < 0.5:
                m = ((rr - r0) / a) ** 2 + ((cc - c0) / b) ** 2 <= 1
            else:
                m = (np.abs(rr - r0) <= a) & (np.abs(cc - c0) <= b)
            img[m] = v
        vol[k] = img + rng.normal(0, 0.03, img.shape)
    return np.clip(vol, 0, None)


def _brain_like(n: int = 13, rows: int = 192, cols: int = 192) -> np.ndarray:
    """Skull, brain and ventricle ellipses, no spine."""
    rr, cc = np.mgrid[0:rows, 0:cols]
    skull = ((rr - 96) / 80.0) ** 2 + ((cc - 96) / 70.0) ** 2 <= 1
    brain = ((rr - 96) / 70.0) ** 2 + ((cc - 96) / 60.0) ** 2 <= 1
    vent = ((rr - 80) / 12.0) ** 2 + ((cc - 96) / 25.0) ** 2 <= 1
    img = 0.3 * skull + 0.5 * brain - 0.3 * vent
    rng = np.random.default_rng(1)
    return np.clip(np.stack([img + rng.normal(0, 0.02, img.shape) for _ in range(n)]), 0, None)


# ----------------------------------------------------------------------------- pure logic


@pytest.mark.parametrize(
    ("text", "region"),
    [
        ("LSPINE", "lumbar"),
        ("L-SPINE", "lumbar"),
        ("MRI LWK", "lumbar"),
        ("T2 SAG LUMBAR SPINE", "lumbar"),
        ("CSPINE", "cervical"),
        ("HWK", "cervical"),
        ("TSPINE", "thoracic"),
        ("BWK", "thoracic"),
        ("TL SPINE", "thoracolumbar"),
        ("WHOLE SPINE", "whole_spine"),
        ("BRAIN", "brain"),
        ("KNEE", "knee"),
        ("PELVIS", "pelvis"),
        ("BRAIN LSPINE", "brain"),  # a non-spine body part wins over a spine word
        ("SAG T2 L4-5 OBL AX", None),  # disc labels are not a region
        ("T2_space_sag_c4", None),
        ("", None),
    ],
)
def test_region_from_text(text: str, region: str | None) -> None:
    got, tokens = region_from_text(text)
    assert got == region
    assert (got is None) == (tokens == [])


@pytest.mark.parametrize(
    ("height", "spacings", "neighbour", "expected"),
    [
        (10.0, [36.0], False, "lumbar"),  # tall disc with a lumbar spacing
        (8.0, [36.0, 36.0], False, "lumbar"),
        (5.0, [28.0, 28.0], False, "thoracic"),  # thin disc, thoracic spacing
        (5.0, [19.0, 19.0], False, "cervical"),  # thin, two short spacings
        (9.0, [19.0, 19.0], False, "unknown"),  # short spacings but not thin: not cervical
        (5.0, [19.0], False, "unknown"),  # one short spacing alone is not a cervical column
        (5.0, [19.0], True, "cervical"),  # end disc next to a cervical disc
        (6.0, [19.0, 36.0], False, "lumbar"),  # any lumbar spacing makes a lumbar disc (spurious neighbour)
        (None, [], False, "unknown"),
        (10.0, [], False, "lumbar"),  # lowest disc, tall
        (5.0, [], False, "unknown"),  # lowest disc, thin: lumbar or thoracic, unknown from one disc
    ],
)
def test_disc_region_rules(
    height: float | None, spacings: list[float], neighbour: bool, expected: str
) -> None:
    assert disc_region(height, spacings, DEFAULT, neighbour_cervical=neighbour) == expected


def _anchors_from_centres(zs: list[float], heights: list[float], x: float = 0.0) -> Anchors:
    """Anchors with discs at the given z (superior first) and heights; names are placeholders."""
    names = [f"D{i}" for i in range(len(zs))]
    discs = {
        n: Disc(center=[x, 0.0, z], normal=[0.0, 0.0, 1.0], height_mm=h, confidence=0.9)
        for n, z, h in zip(names, zs, heights, strict=True)
    }
    return Anchors(discs=discs, numbering={"status": "confirmed", "confidence": 0.95})


def test_disc_regions_spurious_neighbour_keeps_lumbar_disc() -> None:
    # Lumbar column (spacings 36) with a spurious thin disc 19 mm below one real disc. The real discs stay lumbar.
    a = _anchors_from_centres([80.0, 44.0, 25.0, 6.0], [10.0, 10.0, 6.0, 10.0])
    regions = disc_regions(a)
    assert regions["D0"] == "lumbar"
    assert regions["D1"] == "lumbar"
    assert regions["D3"] != "cervical"


def test_disc_regions_cervical_run() -> None:
    a = _anchors_from_centres([60.0, 42.0, 24.0, 6.0], [4.0, 4.0, 4.0, 4.0])
    assert set(disc_regions(a).values()) == {"cervical"}


@pytest.mark.parametrize(
    ("desc", "flag"),
    [
        ("SAG T2", None),
        ("SAG STIR", "STIR"),
        ("SAG PD", "PD"),
    ],
)
def test_acquisition_cues_contrast(desc: str, flag: str | None) -> None:
    meta = {"dicom": {"EchoTime": 20.0}} if desc == "SAG PD" else None  # short TE, long TR: proton density
    st = from_stacks([_sag_stack(np.ones((13, 16, 16)), desc=desc, meta=meta)])
    cues = acquisition_cues(st.best("sag"), DEFAULT)
    if flag is None:
        assert cues == []
    else:
        assert any(flag in c for c in cues)


def test_acquisition_cues_dixon_postcontrast_lowfield_iso() -> None:
    base = _sag_stack(np.ones((13, 16, 16)))
    st = from_stacks([base])
    sag = st.best("sag")
    sag.meta["role"]["dixon"] = "w"
    assert any("Dixon" in c for c in acquisition_cues(sag, DEFAULT))
    sag.meta["role"]["dixon"] = None
    sag.meta["role"]["post_contrast"] = True
    sag.meta["role"]["fat_suppressed"] = True
    sag.meta["role"]["contrast"] = "t1"
    assert any("post-contrast fat-suppressed T1" in c for c in acquisition_cues(sag, DEFAULT))
    sag.meta["role"].update(post_contrast=False, fat_suppressed=False, contrast="t2")
    sag.meta["dicom"]["MagneticFieldStrength"] = "0.36"
    assert any("low field" in c for c in acquisition_cues(sag, DEFAULT))
    sag.meta["dicom"]["MagneticFieldStrength"] = "3.0"
    assert not any("low field" in c for c in acquisition_cues(sag, DEFAULT))
    # 3-D isotropic 1 mm voxels: flagged. SPIDER T2 SPACE (0.47 x 0.47 x 0.9 mm) is anisotropic: not flagged.
    iso = replace(sag, row_mm=1.0, col_mm=1.0, thickness=1.0)
    assert any("3-D isotropic" in c for c in acquisition_cues(iso, DEFAULT))
    space = replace(sag, row_mm=0.47, col_mm=0.47, thickness=0.9)
    assert not any("3-D isotropic" in c for c in acquisition_cues(space, DEFAULT))


def test_fov_coverage_uses_ground_truth_sacrum_position() -> None:
    """A FOV that ends at L5 has less bone below L5-S1 than the full protocol (truth anchors, no network)."""
    cut = build_case(
        Scenario("fov_L1_L5", "fov", "L1 to L5 only", protocol=Protocol(contrasts=("t2",), fov=("L1", "L5")))
    )
    cov = fov_coverage(cut.study.best("sag"), Anchors.from_truth(cut.phantom.truth), DEFAULT)
    assert cov["sacrum_covered"] is False
    full = build_case(_scenario("baseline"))
    cov_full = fov_coverage(full.study.best("sag"), Anchors.from_truth(full.phantom.truth), DEFAULT)
    assert cov_full["bone_below_lowest_disc_mm"] > cov["bone_below_lowest_disc_mm"]
    assert cov_full["top_level_in_view"] is not None
    assert "L3" in cov_full["levels_in_fov"]


def test_fov_coverage_reports_top_level_for_fov_starting_at_l2() -> None:
    """A FOV that starts at L2 (truth anchors): the level at its superior edge is named, levels above are outside."""
    case = build_case(
        Scenario("fov_L2_S1_t", "fov", "L2 to S1", protocol=Protocol(contrasts=("t2",), fov=("L2", "S1")))
    )
    cov = fov_coverage(case.study.best("sag"), Anchors.from_truth(case.phantom.truth), DEFAULT)
    assert cov["top_level_in_view"] is not None
    assert cov["superior_margin_mm"] is not None
    assert "T11" not in cov["levels_in_fov"]  # nothing above L2 is in view


def test_assess_without_sagittal_is_not_labelled() -> None:
    ph = SpinePhantom(PhantomSpec())
    axial = ph.render("ax", "t2", n=10, spacing=5.0, thickness=4.0, rows=96, cols=96, px=1.4)
    study = from_stacks([axial])
    app = assess_applicability(study, Anchors.from_truth(ph.truth), cfg=DEFAULT)
    assert app.verdict == NO_SAGITTAL
    assert app.labelled is False


def test_assess_too_short_sagittal_is_not_usable() -> None:
    from spinescope.anatomy.ood import NOT_USABLE

    short = _sag_stack(np.ones((2, 16, 16)))
    study = from_stacks([short])
    app = assess_applicability(study, _anchors_from_centres([40.0, 0.0], [10.0, 10.0]), cfg=DEFAULT)
    assert app.verdict == NOT_USABLE
    assert app.labelled is False


def test_apply_not_labelled_clears_labels() -> None:
    a = _anchors_from_centres([40.0, 0.0], [10.0, 10.0])
    a.conus = {"z": 10.0, "confidence": 0.5}
    app = Applicability(
        verdict=NOT_LUMBAR,
        region="brain",
        labelled=False,
        summary="not a lumbar spine study",
        warnings=["applicability: not a lumbar spine study (brain)"],
    )
    apply_applicability(a, app)
    assert a.discs == {}
    assert a.conus is None
    assert a.numbering["status"] == "unconfirmed"
    assert a.numbering["confidence"] == 0.0
    assert a.numbering["labelled"] is False
    assert a.applicability["verdict"] == NOT_LUMBAR
    assert any("applicability" in w for w in a.warnings)


def test_apply_scales_confidence_caps_thoracic_and_forces_unconfirmed_without_sacrum() -> None:
    a = _anchors_from_centres([120.0, 84.0, 48.0, 12.0], [5.0, 10.0, 10.0, 10.0])
    a.discs = {
        "T10-T11": Disc(center=[0, 0, 120.0], normal=[0, 0, 1], height_mm=5.0, confidence=0.9),
        "L1-L2": Disc(center=[0, 0, 84.0], normal=[0, 0, 1], height_mm=10.0, confidence=0.9),
        "L4-L5": Disc(center=[0, 0, 48.0], normal=[0, 0, 1], height_mm=10.0, confidence=0.9),
        "L5-S1": Disc(center=[0, 0, 12.0], normal=[0, 0, 1], height_mm=10.0, confidence=0.9),
    }
    a.numbering = {"status": "confirmed", "confidence": 0.95}
    app = Applicability(
        verdict=THORACOLUMBAR,
        region="thoracolumbar",
        labelled=True,
        summary="thoracolumbar",
        factor=0.8,
        cap=0.3,
        capped=["T10-T11"],
        dropped=[],
        sacrum_covered=False,
        unconfirmed_reasons=["sacrum not in the field of view"],
    )
    apply_applicability(a, app, thresholds={"confirmed": 0.9, "probable": 0.6})
    assert a.discs["T10-T11"].confidence == pytest.approx(0.3)
    assert a.discs["L1-L2"].confidence == pytest.approx(0.72)
    assert a.numbering["status"] == "unconfirmed"
    assert a.numbering["confidence"] == pytest.approx(0.76)
    assert a.numbering["confidence_before_applicability"] == pytest.approx(0.95)


def test_apply_drops_cervical_discs_and_keeps_lumbar() -> None:
    a = _anchors_from_centres([120.0, 84.0, 12.0], [10.0, 10.0, 10.0])
    a.discs = {
        "C6-C7": Disc(center=[0, 0, 120.0], normal=[0, 0, 1], height_mm=4.0, confidence=0.9),
        "L4-L5": Disc(center=[0, 0, 84.0], normal=[0, 0, 1], height_mm=10.0, confidence=0.9),
        "L5-S1": Disc(center=[0, 0, 12.0], normal=[0, 0, 1], height_mm=10.0, confidence=0.9),
    }
    app = Applicability(
        verdict=WHOLE_SPINE, region="whole spine", labelled=True, summary="", dropped=["C6-C7"], factor=1.0
    )
    apply_applicability(a, app)
    assert set(a.discs) == {"L4-L5", "L5-S1"}


def test_applicability_round_trips_through_anchors_dict() -> None:
    a = _anchors_from_centres([40.0, 0.0], [10.0, 10.0])
    app = Applicability(
        verdict=LUMBAR, region="lumbar", labelled=True, summary="lumbar", coverage={"z_min_mm": -1.0}
    )
    apply_applicability(a, app)
    text = json.dumps(a.to_dict())
    back = Anchors.from_json(text)
    assert back.applicability["verdict"] == LUMBAR
    assert back.applicability["coverage"]["z_min_mm"] == -1.0


def test_verdict_vocabulary() -> None:
    assert set(LABELLED_VERDICTS) == {LUMBAR, THORACOLUMBAR, WHOLE_SPINE, UNCERTAIN}
    assert NOT_SPINE not in LABELLED_VERDICTS


# ----------------------------------------------------------------------------- labeller end to end


@requires_ml
@pytest.mark.xfail(
    strict=False,
    reason="the cartoon phantom is out of distribution for the learned labeller: since the canal-median midline "
    "fix (lab09) the network places the lowest disc at the sacral end of the phantom FOV, so the sacrum check "
    "fails; real-data behaviour is covered by the SPIDER acceptance run",
)
def test_lumbar_phantom_is_labelled_without_caps() -> None:
    from spinescope.anatomy import label_spine

    case = build_case(_scenario("baseline"))
    a = label_spine(case.study)
    ap = a.applicability
    assert ap["verdict"] == LUMBAR
    assert ap["labelled"] is True
    assert ap["factor"] == pytest.approx(1.0)
    assert ap["sacrum_covered"] is True
    assert {"L4-L5", "L5-S1"} <= set(a.disc_labels())
    assert ap["capped"] == []


@requires_ml
def test_strong_non_spine_metadata_is_not_labelled() -> None:
    from spinescope.anatomy import label_spine

    case = build_case(_scenario("baseline"))
    case.study.best("sag").meta["dicom"]["BodyPartExamined"] = "BRAIN"
    a = label_spine(case.study)
    assert a.discs == {}
    assert a.applicability["verdict"] == NOT_LUMBAR
    assert a.applicability["region"] == "brain"
    assert a.numbering["status"] == "unconfirmed"
    assert any("not a lumbar spine study" in w.lower() for w in a.warnings)


@requires_ml
def test_strong_thoracic_metadata_does_not_override_lumbar_anatomy() -> None:
    from spinescope.anatomy import label_spine

    case = build_case(_scenario("baseline"))
    case.study.best("sag").meta["dicom"]["BodyPartExamined"] = "TSPINE"
    a = label_spine(case.study)
    assert a.applicability["labelled"] is True
    assert "L5-S1" in a.disc_labels()
    assert any("BodyPartExamined names thoracic" in w for w in a.warnings)


@requires_ml
def test_brain_like_image_is_not_spine() -> None:
    from spinescope.anatomy import label_spine

    study = from_stacks([_sag_stack(_brain_like())])
    a = label_spine(study)
    assert a.discs == {}
    assert a.applicability["verdict"] in (NOT_SPINE, NOT_LUMBAR)
    assert a.applicability["labelled"] is False


@requires_ml
def test_phantom_crop_without_spine_is_not_labelled() -> None:
    from spinescope.anatomy import label_spine

    ph = SpinePhantom(PhantomSpec())
    c = ph.bodies["L3"]["center"]
    center = np.array([0.0, c[1] + 10.0 - 150.0, c[2] - 15.0])  # anterior to the body: abdomen only
    study = from_stacks(
        [ph.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2, center=center)]
    )
    a = label_spine(study)
    assert a.discs == {}
    assert a.applicability["labelled"] is False


@requires_ml
@pytest.mark.parametrize("seed", [0, 1])
def test_random_structures_are_not_confident(seed: int) -> None:
    from spinescope.anatomy import label_spine

    a = label_spine(from_stacks([_sag_stack(_random_structures(seed))]))
    # Random blobs must never yield confident lumbar labels: not labelled, or the numbering is not confirmed.
    assert a.numbering.get("status") != "confirmed"


@requires_ml
@pytest.mark.parametrize(
    ("name", "spec", "protocol"),
    [
        (
            "cervical_like",
            {"body_height": 14.0, "disc_height": 5.0, "body_half_depth": 12.0},
            Protocol(contrasts=("t2",)),
        ),
        (
            "thoracic_like_crop",
            {"body_height": 22.0, "disc_height": 5.0, "thoracic_top": 9},
            Protocol(contrasts=("t2",), fov=("T10", "T12"), fov_pad_top=10.0),
        ),
        (
            "thoracic_like",
            {"body_height": 22.0, "disc_height": 5.0, "thoracic_top": 9},
            Protocol(contrasts=("t2",)),
        ),
    ],
)
def test_non_lumbar_phantom_variants_are_not_labelled(name: str, spec: dict, protocol: Protocol) -> None:
    from spinescope.anatomy import label_spine

    a = label_spine(build_case(Scenario(name, "x", name, spec=spec, protocol=protocol)).study)
    assert a.applicability["labelled"] is False, a.applicability.get("reasons")
    assert a.applicability["verdict"] in (NOT_LUMBAR, NOT_SPINE)
    assert a.discs == {}


@requires_ml
def test_fov_without_sacrum_is_not_confirmed() -> None:
    from spinescope.anatomy import label_spine

    a = label_spine(
        build_case(
            Scenario("fov_L1_L5", "fov", "x", protocol=Protocol(contrasts=("t2",), fov=("L1", "L5")))
        ).study
    )
    assert a.numbering["status"] != "confirmed"


@requires_ml
def test_fov_starting_at_l2_reports_top_level_and_coverage() -> None:
    from spinescope.anatomy import label_spine

    a = label_spine(build_case(_scenario("fov_L2_S1")).study)
    cov = a.applicability["coverage"]
    assert cov["top_level_in_view"] is not None
    assert cov["z_max_mm"] > cov["z_min_mm"]


@pytest.mark.xfail(
    strict=False,
    reason="cartoon phantom is out of distribution for the learned labeller (after the lab09 midline fix the "
    "thoracolumbar phantom is judged not usable); real-data behaviour is covered by dataset runs",
)
@requires_ml
def test_thoracolumbar_fov_caps_thoracic_discs_above_t10() -> None:
    from spinescope.anatomy import label_spine

    a = label_spine(build_case(_scenario("fov_T10_sacrum")).study)
    ap = a.applicability
    assert ap["labelled"] is True
    assert "T10-T11" in ap["capped"]
    assert a.discs["T10-T11"].confidence <= DEFAULT.ood_thoracic_cap + 1e-9
    assert "L5-S1" in a.disc_labels()  # lumbar labels remain


@requires_ml
@pytest.mark.xfail(
    strict=True,
    reason="realistic phantom, learned labeller (measured): the lumbar verdict and the L5-S1 labels hold, but the "
    "sacrum is not confirmed below L5-S1 (sacrum_covered is False: SPIDER has no sacrum class), and a spurious upper "
    "disc T10-T11 is capped by the thoracic rule",
)
def test_realistic_lumbar_phantom_is_labelled_without_caps() -> None:
    from spinescope.anatomy import label_spine

    case = build_case(_scenario("baseline"), "realistic")
    a = label_spine(case.study)
    ap = a.applicability
    assert ap["verdict"] == LUMBAR
    assert ap["labelled"] is True
    assert ap["factor"] == pytest.approx(1.0)
    assert ap["sacrum_covered"] is True
    assert {"L4-L5", "L5-S1"} <= set(a.disc_labels())
    assert ap["capped"] == []


@requires_ml
def test_realistic_thoracolumbar_fov_caps_thoracic_discs_above_t10() -> None:
    from spinescope.anatomy import label_spine

    a = label_spine(build_case(_scenario("fov_T10_sacrum"), "realistic").study)
    ap = a.applicability
    assert ap["labelled"] is True
    assert "T10-T11" in ap["capped"]
    assert a.discs["T10-T11"].confidence <= DEFAULT.ood_thoracic_cap + 1e-9
    assert "L5-S1" in a.disc_labels()  # lumbar labels remain


@requires_ml
@pytest.mark.parametrize(
    ("name", "cue"), [("stir_only", "STIR"), ("pd_only", "PD"), ("iso_1mm_3d", "3-D isotropic")]
)
def test_untrained_inputs_are_warned_and_lowered(name: str, cue: str) -> None:
    from spinescope.anatomy import label_spine

    a = label_spine(build_case(_scenario(name)).study)
    assert any(cue in w for w in a.warnings), a.warnings
    assert a.applicability["factor"] < 1.0


@requires_ml
def test_low_field_and_postcontrast_and_dixon_are_warned() -> None:
    from spinescope.anatomy import label_spine

    case = build_case(_scenario("baseline"))
    sag = case.study.best("sag")
    sag.meta["dicom"]["MagneticFieldStrength"] = "0.36"
    a = label_spine(case.study)
    assert any("low field" in w for w in a.warnings)

    case = build_case(_scenario("baseline"))
    sag = case.study.best("sag")
    sag.meta["role"]["dixon"] = "w"
    a = label_spine(case.study)
    assert any("Dixon" in w for w in a.warnings)

    case = build_case(_scenario("baseline"))
    sag = case.study.best("sag")
    sag.meta["role"].update(post_contrast=True, fat_suppressed=True, contrast="t1")
    # The labeller selects non-post-contrast series only: with no other sagittal series there is nothing to label.
    only_post = Study(stacks={"sag_t1_post": sag}, info={})
    a = label_spine(only_post)
    assert a.applicability["verdict"] == NO_SAGITTAL
    assert a.discs == {}


@requires_ml
def test_classical_backend_records_applicability_without_network_cues() -> None:
    from spinescope.anatomy.pipeline import label_spine_details

    det = label_spine_details(build_case(_scenario("baseline")).study, backend="classical", calibrate=False)
    ap = det.anchors.applicability
    assert ap["verdict"] in (*LABELLED_VERDICTS, NOT_LUMBAR, NOT_SPINE)
    assert "entropy" not in ap["cues"]


@requires_ml
def test_applicability_record_has_no_patient_attributes(phantom_dir: dict) -> None:
    from spinescope.anatomy import label_spine

    study = load_study(phantom_dir["root"])
    a = label_spine(study)
    text = json.dumps(a.applicability)
    for value in FAKE_PHI.values():
        assert value not in text
    assert str(phantom_dir["root"]) not in text


def _spider_ready() -> bool:
    base = Path(os.environ.get("SPINESCOPE_DATA", "~/data/public")).expanduser() / "spider" / "images"
    return base.is_dir()


@requires_ml
@pytest.mark.skipif(not _spider_ready(), reason="SPIDER not found under $SPINESCOPE_DATA/spider")
def test_spider_validation_sample_is_not_rejected() -> None:
    """Regression guard: a few SPIDER validation studies are lumbar and must stay labelled."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _spider import load_t2_t1, paired_study_ids
    from spinescope.anatomy.pipeline import label_spine_details

    ids = paired_study_ids()[:3]
    rejected = []
    for sid in ids:
        loaded = load_t2_t1(sid)
        if loaded is None:
            continue
        study, _t2, _mask = loaded
        det = label_spine_details(study, backend="ml")
        if not det.anchors.applicability.get("labelled"):
            rejected.append(sid)
    assert rejected == []
