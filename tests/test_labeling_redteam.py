"""Red-team regression tests for ``load_study`` + ``label_spine`` (lab23).

Each test is one attack from docs/labeling/lab23_redteam.md: nasty but realistic DICOM or volume inputs built from
the phantom and then perturbed with pydicom. The expected behaviour is one of: a graceful error that names the
cause, a warning, an abstention (no level names), or the same labels as the clean reference. Matrices are small
(160 px, 13 slices) so the file stays fast; the whole-spine case is marked ``slow``.

Phantom caveat: the learned labeller is out of distribution on the cartoon phantom (its lowest disc sits one level
low, see docs/labeling/lab23_redteam.md), so the reference is the clean phantom run through the same pipeline and
attacks are checked for invariance against it, not against the phantom truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import pytest
from pydicom.uid import SecondaryCaptureImageStorage, generate_uid

from spinescope import load_study
from spinescope.anatomy import label_spine
from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.ml.grid import network_input_volume
from spinescope.anatomy.ml.labeller import label_spine_ml_segmented
from spinescope.anatomy.numbering import disc_names
from spinescope.anatomy.ood import _contradicts_contrast, contrast_polarity
from spinescope.config import DEFAULT
from spinescope.geometry import Stack
from spinescope.io.dicom import _geometry_problem, _signed_offset
from spinescope.study import _pixel_qa
from spinescope.testing.phantom import FAKE_PHI, PhantomSpec, SpinePhantom, write_dicom_series

PH = SpinePhantom(PhantomSpec())
STUDY = generate_uid()
FRAME = generate_uid()
ROWS = 192
PX = 1.2  # 230 mm in plane: the field of view of the default phantom protocol (the labeller is pixel-size sensitive)


def _sag(
    contrast: str = "t2", n: int = 13, spacing: float = 5.0, rows: int = ROWS, px: float = PX, **kw: Any
) -> Stack:
    return PH.render("sag", contrast, n=n, spacing=spacing, thickness=4.0, rows=rows, cols=rows, px=px, **kw)


def _write(stack: Stack, root: Path, *, series: int = 1, contrast: str = "t2", **kw: Any) -> list[str]:
    return write_dicom_series(
        stack, str(root), series_number=series, contrast=contrast, study_uid=STUDY, frame_uid=FRAME, **kw
    )


def _tamper(paths: list[str], fn: Any) -> None:
    for k, p in enumerate(paths):
        ds = pydicom.dcmread(p)
        fn(ds, k)
        ds.save_as(p)


def _set_pixels(ds: pydicom.Dataset, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        if "PixelData" in ds:
            del ds["PixelData"]
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 32, 32, 31, 0
        ds.add_new(0x7FE00008, "OF", arr.astype("<f4").tobytes())
    else:
        ds.PixelData = arr.astype(f"<u{max(1, ds.BitsAllocated // 8)}").tobytes()


def _rotate_iop_in_plane(iop: list[float], deg: float) -> list[str]:
    a, b = np.array(iop[:3]), np.array(iop[3:])
    n = np.cross(a, b)
    th = np.deg2rad(deg)
    a2 = a * np.cos(th) + np.cross(n, a) * np.sin(th)
    b2 = b * np.cos(th) + np.cross(n, b) * np.sin(th)
    return [f"{x:.6f}" for x in (*a2, *b2)]


def _centres(anc: Anchors) -> dict[str, np.ndarray]:
    return {lab: np.asarray(d.center, dtype=np.float64) for lab, d in anc.discs.items()}


def _assert_same_as(anc: Anchors, ref: Anchors, tol_mm: float = 2.0) -> None:
    got, want = _centres(anc), _centres(ref)
    assert sorted(got) == sorted(want), f"labels differ: {sorted(got)} vs {sorted(want)}"
    for lab in want:
        dist = float(np.linalg.norm(got[lab] - want[lab]))
        assert dist <= tol_mm, f"{lab} moved {dist:.1f} mm"


def _no_levels(anc: Anchors) -> None:
    assert not anc.disc_labels(), f"expected abstention, got {anc.disc_labels()}"


def _assert_unconfirmed(anc: Anchors) -> None:
    assert anc.numbering.get("status") != "confirmed", anc.numbering


def _warn(study: Any, needle: str) -> None:
    msgs = list(study.report.get("warnings", []))
    assert any(needle in m for m in msgs), f"no load warning containing {needle!r}: {msgs}"


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Anchors:
    root = tmp_path_factory.mktemp("clean")
    _write(_sag(), root)
    anc = label_spine(load_study(root))
    assert anc.disc_labels(), "the clean reference must be labelled"
    return anc


def _study_from(stack: Stack, tmp_path: Path, **kw: Any) -> Any:
    _write(stack, tmp_path, **kw)
    return load_study(tmp_path)


# ------------------------------------------------------------------ missing and odd tags
def test_missing_pixel_spacing_is_an_explicit_error(tmp_path: Path) -> None:
    _write(_sag(), tmp_path)
    _tamper([str(p) for p in tmp_path.iterdir()], lambda ds, k: ds.__delitem__("PixelSpacing"))
    with pytest.raises(FileNotFoundError, match="missing_pixel_spacing"):
        load_study(tmp_path)


def test_missing_slice_thickness_does_not_matter(tmp_path: Path, clean: Anchors) -> None:
    _write(_sag(), tmp_path)
    _tamper([str(p) for p in tmp_path.iterdir()], lambda ds, k: ds.__delitem__("SliceThickness"))
    _assert_same_as(label_spine(load_study(tmp_path)), clean)


def test_some_slices_without_position_are_reported(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def drop(ds: pydicom.Dataset, k: int) -> None:
        if k in (0, 1, 2, 6):
            del ds["ImagePositionPatient"]

    _tamper(paths, drop)
    study = load_study(tmp_path)
    _warn(study, "ignored for missing or invalid")
    assert study.report["scan"]["ignored"].get("missing_position") == 4
    _assert_unconfirmed(label_spine(study))


def test_no_slice_has_a_position_is_an_explicit_error(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)
    _tamper(paths, lambda ds, k: ds.__delitem__("ImagePositionPatient"))
    with pytest.raises(FileNotFoundError, match="missing_position"):
        load_study(tmp_path)


def test_duplicate_instance_numbers_do_not_change_labels(tmp_path: Path, clean: Anchors) -> None:
    paths = _write(_sag(), tmp_path)
    _tamper(paths, lambda ds, k: setattr(ds, "InstanceNumber", 1))
    _assert_same_as(label_spine(load_study(tmp_path)), clean)


def test_interleaved_instance_order_does_not_change_labels(tmp_path: Path, clean: Anchors) -> None:
    paths = _write(_sag(), tmp_path)
    order = [0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11]
    _tamper(paths, lambda ds, k: setattr(ds, "InstanceNumber", order[k] + 1))
    _assert_same_as(label_spine(load_study(tmp_path)), clean)


# ------------------------------------------------------------------ orientation
def test_one_slice_with_a_rotated_iop_is_warned_and_not_confident(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def rotate(ds: pydicom.Dataset, k: int) -> None:
        if k == 6:
            iop = [float(x) for x in ds.ImageOrientationPatient]
            ds.ImageOrientationPatient = _rotate_iop_in_plane(iop, 5.0)

    _tamper(paths, rotate)
    study = load_study(tmp_path)
    _warn(study, "inconsistent ImageOrientationPatient")
    _assert_unconfirmed(label_spine(study))


def test_one_slice_with_a_negated_iop_is_warned(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def flip(ds: pydicom.Dataset, k: int) -> None:
        if k == 6:
            iop = [float(x) for x in ds.ImageOrientationPatient]
            ds.ImageOrientationPatient = [f"{-x:.6f}" for x in iop[:3]] + [f"{x:.6f}" for x in iop[3:]]

    _tamper(paths, flip)
    _warn(load_study(tmp_path), "180.0 deg away")  # a reversed slice, not a 0-degree one


def test_scaled_iop_is_excluded_not_normalised(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def scale(ds: pydicom.Dataset, k: int) -> None:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        ds.ImageOrientationPatient = [f"{1.3 * x:.6f}" for x in iop[:3]] + [f"{x:.6f}" for x in iop[3:]]

    _tamper(paths, scale)
    with pytest.raises(FileNotFoundError, match="bad_orientation"):
        load_study(tmp_path)


def test_skewed_iop_is_excluded(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def skew(ds: pydicom.Dataset, k: int) -> None:
        a = np.array([float(x) for x in ds.ImageOrientationPatient[:3]])
        b = np.array([float(x) for x in ds.ImageOrientationPatient[3:]])
        b2 = b + 0.25 * a
        b2 /= np.linalg.norm(b2)
        ds.ImageOrientationPatient = [f"{x:.6f}" for x in (*a, *b2)]

    _tamper(paths, skew)
    with pytest.raises(FileNotFoundError, match="bad_orientation"):
        load_study(tmp_path)


def test_zero_iop_is_an_explicit_error_not_a_crash(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)
    _tamper(paths, lambda ds, k: setattr(ds, "ImageOrientationPatient", [0, 0, 0, 0, 0, 0]))
    with pytest.raises(FileNotFoundError, match="bad_orientation"):
        load_study(tmp_path)


def test_lr_flipped_iop_abstains(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def flip(ds: pydicom.Dataset, k: int) -> None:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        ds.ImageOrientationPatient = [f"{-x:.6f}" for x in iop[:3]] + [f"{x:.6f}" for x in iop[3:]]

    _tamper(paths, flip)
    _no_levels(label_spine(load_study(tmp_path)))


def test_mirrored_pixels_with_unchanged_iop_abstain(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)
    _tamper(paths, lambda ds, k: _set_pixels(ds, np.asarray(ds.pixel_array)[:, ::-1].copy()))
    _no_levels(label_spine(load_study(tmp_path)))


@pytest.mark.parametrize("degrees", [35.0, 45.0])
def test_extreme_obliquity_abstains(tmp_path: Path, degrees: float) -> None:
    th = np.deg2rad(degrees)
    n = np.array([np.cos(th), 0.0, np.sin(th)])
    _write(_sag(normal=n, row_dir=np.array([0.0, 1.0, 0.0])), tmp_path)
    _no_levels(label_spine(load_study(tmp_path)))


# ------------------------------------------------------------------ pixel content
def test_nan_pixels_are_replaced_and_warned(tmp_path: Path, clean: Anchors) -> None:
    paths = _write(_sag(), tmp_path)

    def nan(ds: pydicom.Dataset, k: int) -> None:
        if k in (4, 5):
            arr = np.asarray(ds.pixel_array).astype(np.float32)
            arr[60:100, 60:100] = np.nan
            _set_pixels(ds, arr)

    _tamper(paths, nan)
    study = load_study(tmp_path)
    _warn(study, "non-finite pixel")
    vol = study["sag_t2"].vol
    assert np.isfinite(vol).all()
    anc = label_spine(study)
    assert all(np.isfinite(c).all() for c in _centres(anc).values())
    _assert_same_as(anc, clean)


def test_inf_pixels_are_replaced_and_warned(tmp_path: Path, clean: Anchors) -> None:
    paths = _write(_sag(), tmp_path)

    def inf(ds: pydicom.Dataset, k: int) -> None:
        if k == 5:
            arr = np.asarray(ds.pixel_array).astype(np.float32)
            arr[:, :10] = np.inf
            arr[100, 100] = -np.inf
            _set_pixels(ds, arr)

    _tamper(paths, inf)
    study = load_study(tmp_path)
    _warn(study, "non-finite pixel")
    assert np.isfinite(study["sag_t2"].vol).all()
    _assert_same_as(label_spine(study), clean)


def test_constant_slices_are_warned(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def zero(ds: pydicom.Dataset, k: int) -> None:
        if k < 4:
            _set_pixels(ds, np.zeros_like(np.asarray(ds.pixel_array)))

    _tamper(paths, zero)
    study = load_study(tmp_path)
    _warn(study, "4 of 13 slice(s) are constant")
    _assert_unconfirmed(label_spine(study))


def test_all_zero_volume_abstains(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)
    _tamper(paths, lambda ds, k: _set_pixels(ds, np.zeros_like(np.asarray(ds.pixel_array))))
    study = load_study(tmp_path)
    _warn(study, "13 of 13 slice(s) are constant")
    _no_levels(label_spine(study))


def test_saturated_constant_slices_are_warned(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def sat(ds: pydicom.Dataset, k: int) -> None:
        if 5 <= k <= 7:
            _set_pixels(ds, np.full_like(np.asarray(ds.pixel_array), 4095))

    _tamper(paths, sat)
    _warn(load_study(tmp_path), "constant (every voxel equal")


def test_signed_int16_with_negative_background_matches_unsigned(tmp_path: Path, clean: Anchors) -> None:
    _write(_sag(), tmp_path, signed=True)
    study = load_study(tmp_path)
    # the loader keeps the stored values; the recorded offset is the signed background (-2048 less Rician noise)
    assert study["sag_t2"].meta["signed_offset"] == pytest.approx(-2048.0, abs=50.0)
    # the network sees the background at zero: only the noisiest background voxels stay below it
    assert (network_input_volume(study["sag_t2"]) < 0).mean() < 0.01
    _assert_same_as(label_spine(study), clean)


def test_uint8_pixels_match_the_clean_reference(tmp_path: Path, clean: Anchors) -> None:
    paths = _write(_sag(), tmp_path)

    def to_u8(ds: pydicom.Dataset, k: int) -> None:
        arr = np.asarray(ds.pixel_array).astype(np.float64)
        arr = np.clip(arr / max(arr.max(), 1.0) * 255, 0, 255).astype(np.uint8)
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        ds.PixelData = arr.tobytes()

    _tamper(paths, to_u8)
    _assert_same_as(label_spine(load_study(tmp_path)), clean, tol_mm=3.0)


def test_secondary_capture_is_not_anatomy(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def sc(ds: pydicom.Dataset, k: int) -> None:
        g = np.asarray(ds.pixel_array).astype(np.float64)
        g = np.clip(g / max(g.max(), 1.0) * 255, 0, 255).astype(np.uint8)
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.Modality = "SC"
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        ds.PixelData = np.stack([g, g, g], axis=-1).tobytes()
        if "ImagePositionPatient" in ds:
            del ds["ImagePositionPatient"]

    _tamper(paths, sc)
    with pytest.raises(FileNotFoundError, match="secondary_capture"):
        load_study(tmp_path)


def test_colour_mr_is_not_anatomy(tmp_path: Path) -> None:
    paths = _write(_sag(), tmp_path)

    def rgb(ds: pydicom.Dataset, k: int) -> None:
        g = np.asarray(ds.pixel_array).astype(np.float64)
        g = np.clip(g / max(g.max(), 1.0) * 255, 0, 255).astype(np.uint8)
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        ds.PixelData = np.stack([g, g, g], axis=-1).tobytes()

    _tamper(paths, rgb)
    with pytest.raises(FileNotFoundError, match="colour_or_screenshot"):
        load_study(tmp_path)


# ------------------------------------------------------------------ matrix and slice count
def test_large_matrix_proxy_labels_consistently(tmp_path: Path, clean: Anchors) -> None:
    # Scaled-down stand-in for a 1024 x 1024 matrix (same field of view, 2.7x the pixels of the reference).
    _write(_sag(rows=256, px=PX * ROWS / 256), tmp_path)
    anc = label_spine(load_study(tmp_path))
    common = set(_centres(anc)) & set(_centres(clean))
    assert common, "the large matrix must label the same discs as the reference"
    for lab in common:
        dist = float(np.linalg.norm(_centres(anc)[lab] - _centres(clean)[lab]))
        assert dist <= 3.0, f"{lab} moved {dist:.1f} mm at a larger matrix"


def test_tiny_matrix_abstains_without_crashing(tmp_path: Path) -> None:
    _write(_sag(rows=64, px=4.0), tmp_path)
    _no_levels(label_spine(load_study(tmp_path)))


def test_single_slice_abstains(tmp_path: Path) -> None:
    _write(_sag(n=1), tmp_path)
    _no_levels(label_spine(load_study(tmp_path)))


def test_two_slices_abstain(tmp_path: Path) -> None:
    _write(_sag(n=2), tmp_path)
    _no_levels(label_spine(load_study(tmp_path)))


def test_many_slices_do_not_crash(tmp_path: Path) -> None:
    _write(_sag(n=60, spacing=1.0, rows=96, px=2.4), tmp_path)
    anc = label_spine(load_study(tmp_path))
    assert anc.to_dict() is not None


# ------------------------------------------------------------------ several studies, series and descriptions
def test_mixed_patients_warn_and_do_not_leak_the_other_patient(tmp_path: Path) -> None:
    _write(_sag(), tmp_path, series=1)
    other_study = generate_uid()
    other = write_dicom_series(
        _sag(rows=144, px=1.6),
        str(tmp_path),
        series_number=9,
        contrast="t2",
        study_uid=other_study,
        frame_uid=generate_uid(),
    )

    def rename(ds: pydicom.Dataset, k: int) -> None:
        ds.PatientName = "OTHER^PATIENT"
        ds.PatientID = "OTHER-7777"

    _tamper(other, rename)
    study = load_study(tmp_path)
    _warn(study, "holds 2 studies")
    _warn(study, "choice is arbitrary")  # both studies have 13 images
    dump = json.dumps(label_spine(study).to_dict(), default=str)
    for token in (
        "OTHER^PATIENT",
        "OTHER-7777",
        *(v for v in FAKE_PHI.values() if len(v) >= 6),
        str(tmp_path),
    ):
        assert token not in dump


def test_two_sagittal_rescans_with_different_fov_are_not_confident(tmp_path: Path) -> None:
    _write(_sag(), tmp_path, series=1)
    write_dicom_series(
        _sag(rows=112, px=2.0),
        str(tmp_path),
        series_number=2,
        contrast="t2",
        study_uid=STUDY,
        frame_uid=FRAME,
        description="SAG T2 RESCAN",
    )
    anc = label_spine(load_study(tmp_path))
    assert any("second sagittal series" in w for w in anc.warnings)
    _assert_unconfirmed(anc)


def test_t1_labelled_as_t2_is_flagged_by_the_pixels(tmp_path: Path) -> None:
    _write(_sag(contrast="t1"), tmp_path, contrast="t2", description="SAG T2 FSE")
    study = load_study(tmp_path)
    _anc, seg = label_spine_ml_segmented(study)
    pol = contrast_polarity(seg) if seg is not None else None
    assert pol is not None and pol < 0, "canal must read darker than bone on a T1 image"
    assert any("contradict the declared T2 contrast" in w for w in label_spine(study).warnings)


def test_t2_polarity_is_positive_on_the_phantom(tmp_path: Path) -> None:
    _write(_sag(), tmp_path)
    _anc, seg = label_spine_ml_segmented(load_study(tmp_path))
    assert seg is not None
    pol = contrast_polarity(seg)
    assert pol is not None and pol > DEFAULT.ood_contrast_polarity_min


def test_stir_description_on_t2_pixels_does_not_change_labels(tmp_path: Path, clean: Anchors) -> None:
    _write(_sag(), tmp_path, description="SAG STIR")
    _assert_same_as(label_spine(load_study(tmp_path)), clean)


def test_localizer_description_leaves_no_sagittal(tmp_path: Path) -> None:
    _write(_sag(), tmp_path, description="3-PL LOC")
    study = load_study(tmp_path)
    assert study.find("sag", localizer=False) == []
    anc = label_spine(study)
    _no_levels(anc)
    assert any("no sagittal series" in w for w in anc.warnings)


def test_scout_only_study_abstains_with_a_reason(tmp_path: Path) -> None:
    write_dicom_series(
        PH.render("sag", "t2", n=3, spacing=15.0, thickness=8.0, rows=96, cols=96, px=3.6),
        str(tmp_path),
        series_number=1,
        contrast="t2",
        study_uid=STUDY,
        frame_uid=FRAME,
        description="3-PL LOC",
        image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"),
    )
    anc = label_spine(load_study(tmp_path))
    _no_levels(anc)
    assert any("no sagittal series" in w for w in anc.warnings)


def test_non_spine_noise_abstains(tmp_path: Path) -> None:
    base = _sag()
    rng = np.random.default_rng(0)
    noise = Stack(
        key="",
        vol=rng.normal(500, 40, base.vol.shape).astype(np.float32),
        ipp=base.ipp,
        row_dir=base.row_dir,
        col_dir=base.col_dir,
        row_mm=base.row_mm,
        col_mm=base.col_mm,
        thickness=base.thickness,
        description="NOISE",
    )
    _write(noise, tmp_path)
    _no_levels(label_spine(load_study(tmp_path)))


def test_spine_out_of_field_of_view_abstains(tmp_path: Path) -> None:
    _write(_sag(center=np.array([0.0, -220.0, 0.0])), tmp_path)
    _no_levels(label_spine(load_study(tmp_path)))


def test_cervical_only_field_of_view_abstains(tmp_path: Path) -> None:
    c = np.asarray(PH.truth["discs"]["T11-T12"]["center"], dtype=float) + np.array([0.0, 0.0, 90.0])
    _write(_sag(center=c), tmp_path)
    _no_levels(label_spine(load_study(tmp_path)))


def test_gap_in_slices_is_warned(tmp_path: Path) -> None:
    base = _sag()
    keep = [i for i in range(base.n) if i not in (4, 5, 6)]
    gapped = Stack(
        key="",
        vol=base.vol[keep],
        ipp=base.ipp[keep],
        row_dir=base.row_dir,
        col_dir=base.col_dir,
        row_mm=base.row_mm,
        col_mm=base.col_mm,
        thickness=base.thickness,
        description="GAP",
    )
    study = _study_from(gapped, tmp_path)
    _warn(study, "gap(s) in slice positions")
    _assert_unconfirmed(label_spine(study))


def test_anisotropic_pixels_are_labelled_at_their_true_size(tmp_path: Path) -> None:
    # The same anatomy at 0.72 x 0.72 mm, then every third column kept and labelled as 0.72 x 2.16 mm.
    # The reference is the isotropic run at 0.72 mm (the labeller's naming depends on pixel size, so 1.44 mm
    # would not be a like-for-like reference).
    iso_dir, aniso_dir = tmp_path / "iso", tmp_path / "aniso"
    iso = _sag(rows=320, px=0.72)
    _write(iso, iso_dir)
    aniso = Stack(
        key="",
        vol=iso.vol[:, :, ::3],
        ipp=iso.ipp,
        row_dir=iso.row_dir,
        col_dir=iso.col_dir,
        row_mm=0.72,
        col_mm=0.72 * 3,
        thickness=iso.thickness,
        description="ANISO",
    )
    _write(aniso, aniso_dir)
    ref = label_spine(load_study(iso_dir))
    anc = label_spine(load_study(aniso_dir))
    assert ref.disc_labels(), "the isotropic reference must be labelled"
    # Name-agnostic on purpose: the phantom's sacral artefact shifts names by one level between pixel sizes
    # (see the module docstring), so only the disc positions are compared: each anisotropic disc must sit on
    # a reference disc. Positions are checked in mm, so a wrong pixel spacing would show up here.
    ref_centres = np.array(list(_centres(ref).values()))
    got = _centres(anc)
    assert len(got) >= 3
    for lab, c in got.items():
        dist = float(np.min(np.linalg.norm(ref_centres - c, axis=1)))
        assert dist <= 4.0, f"{lab} is {dist:.1f} mm from every reference disc"


# ------------------------------------------------------------------ helpers and units
def test_geometry_problem_reasons() -> None:
    iop = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    assert _geometry_problem([0, 0, 0], iop, [1.0, 1.0]) is None
    assert _geometry_problem(None, iop, [1.0, 1.0]) == "missing_position"
    assert _geometry_problem([0, 0, 0], None, [1.0, 1.0]) == "missing_orientation"
    assert _geometry_problem([0, 0, 0], iop, None) == "missing_pixel_spacing"
    assert _geometry_problem([0, 0, 0], [0] * 6, [1.0, 1.0]) == "bad_orientation"
    assert _geometry_problem([0, 0, 0], [1.3, 0, 0, 0, 0, -1], [1.0, 1.0]) == "bad_orientation"
    assert _geometry_problem([0, 0, 0], [1.0, 0.25, 0, 0, 1.0, 0], [1.0, 1.0]) == "bad_orientation"
    assert _geometry_problem([0, 0, 0], iop, [0.0, 1.0]) == "bad_pixel_spacing"
    assert _geometry_problem([np.nan, 0, 0], iop, [1.0, 1.0]) == "bad_position"


def test_signed_offset_is_zero_for_unsigned_and_negative_for_signed() -> None:
    unsigned = np.zeros((2, 8, 8), np.float32)
    unsigned[:, 2:6, 2:6] = 900.0
    assert _signed_offset(unsigned) == 0.0
    assert _signed_offset(unsigned - 2048.0) == pytest.approx(-2048.0)


def test_disc_names_beyond_the_table_do_not_raise() -> None:
    names = disc_names(30)
    assert names[0] == "L5-S1" and len(names) == 30 and len(set(names)) == 30


@pytest.mark.parametrize(
    ("declared", "pol", "expect"),
    [
        ("t2", 0.2, False),
        ("t2", -0.2, True),
        ("t1", -0.2, False),
        ("t1", 0.2, True),
        ("unknown", -0.9, False),
    ],
)
def test_contrast_contradiction_truth_table(declared: str, pol: float, expect: bool) -> None:
    assert _contradicts_contrast(declared, pol, DEFAULT) is expect


def test_pixel_qa_reports_nonfinite_constant_and_saturated() -> None:
    vol = np.zeros((4, 8, 8), np.float32)
    vol[1] = 5.0
    vol[2, :, :7] = 10.0  # most voxels at the maximum: clipped
    stack = Stack(
        key="",
        vol=vol,
        ipp=np.array([[0, 0, float(k) * 2] for k in range(4)]),
        row_dir=[0, 1, 0],
        col_dir=[0, 0, -1],
        row_mm=1.0,
        col_mm=1.0,
    )
    stack.meta["nonfinite_voxels"] = 3
    report: dict[str, Any] = {}
    _pixel_qa("s", stack, report)
    text = " | ".join(report["warnings"])
    assert "3 non-finite" in text
    assert "3 of 4 slice(s) are constant" in text  # slices 0, 1 and 3 are flat
    assert "1 slice(s) look saturated" in text  # slice 2 has 56 of 64 voxels at the maximum


@pytest.mark.slow
def test_whole_spine_field_of_view_does_not_crash(tmp_path: Path) -> None:
    # 60 slices of 384 px at 3 mm: a 1150 mm field of view. Used to raise IndexError in the disc names.
    _write(_sag(n=60, rows=384, px=3.0), tmp_path)
    anc = label_spine(load_study(tmp_path))
    assert anc.to_dict() is not None
