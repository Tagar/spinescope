from __future__ import annotations

import numpy as np
import pytest

from spinescope.geometry import Stack
from spinescope.series import DISPLAY_AXES, classify, to_display_orientation


def _stack(plane="sag", **dicom):
    rd, cd = DISPLAY_AXES[plane]
    return Stack(
        key="",
        vol=np.zeros((10, 8, 8)),
        ipp=[i * np.cross(rd, cd) * 4 for i in range(10)],
        row_dir=rd,
        col_dir=cd,
        row_mm=1,
        col_mm=1,
        thickness=4,
        meta={"dicom": dicom},
    )


@pytest.mark.parametrize(
    ("dicom", "contrast", "flags"),
    [
        ({"RepetitionTime": 550, "EchoTime": 8, "SeriesDescription": "SAG T1"}, "t1", {}),
        ({"RepetitionTime": 4000, "EchoTime": 95, "SeriesDescription": "SAG T2 FSE"}, "t2", {}),
        (
            {
                "RepetitionTime": 5000,
                "EchoTime": 46,
                "InversionTime": 150,
                "ScanningSequence": "IR",
                "SeriesDescription": "SAG IR",
            },
            "stir",
            {},
        ),
        ({"RepetitionTime": 4500, "EchoTime": 50, "SeriesDescription": "t2_tirm_sag"}, "stir", {}),
        (
            {"RepetitionTime": 9000, "EchoTime": 120, "InversionTime": 2500, "SeriesDescription": "AX FLAIR"},
            "flair",
            {},
        ),
        ({"RepetitionTime": 3000, "EchoTime": 30, "SeriesDescription": "SAG PD"}, "pd", {}),
        (
            {"RepetitionTime": 3500, "EchoTime": 90, "SeriesDescription": "SAG T2 FS", "ScanOptions": ["FS"]},
            "t2",
            {"fat_suppressed": True},
        ),
        (
            {
                "RepetitionTime": 600,
                "EchoTime": 10,
                "SeriesDescription": "SAG T1 +C",
                "ContrastBolusAgent": "GADAVIST",
            },
            "t1",
            {"post_contrast": True},
        ),
        (
            {
                "RepetitionTime": 4000,
                "EchoTime": 90,
                "SeriesDescription": "T2 DIXON W",
                "ImageType": ["DERIVED", "PRIMARY", "DIXON", "WATER"],
            },
            "t2",
            {"dixon": "w"},
        ),
        (
            {
                "RepetitionTime": 500,
                "EchoTime": 15,
                "SeriesDescription": "3-PL LOC",
                "ImageType": ["ORIGINAL", "PRIMARY", "LOCALIZER"],
            },
            None,
            {"localizer": True},
        ),
        (
            {
                "RepetitionTime": 5000,
                "EchoTime": 70,
                "SeriesDescription": "ep2d_diff_tra",
                "ImageType": ["ORIGINAL", "PRIMARY", "DIFFUSION"],
            },
            "dwi",
            {},
        ),
    ],
)
def test_classify(dicom, contrast, flags):
    role = classify(_stack(**dicom))
    if contrast is not None:
        assert role.contrast == contrast, role.evidence
    for k, v in flags.items():
        assert getattr(role, k) == v, role.evidence
    assert role.evidence


def test_spatial_saturation_is_not_fat_suppression():
    role = classify(
        _stack(RepetitionTime=4000, EchoTime=95, SeriesDescription="SAG T2", ScanOptions=["SAT_GEMS", "FC"])
    )
    assert not role.fat_suppressed


@pytest.mark.parametrize("plane", ["sag", "ax", "cor"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("flip_r", [False, True])
@pytest.mark.parametrize("flip_c", [False, True])
def test_display_orientation_is_canonical_and_exact(plane, transpose, flip_r, flip_c):
    rng = np.random.default_rng(0)
    rd, cd = (v.copy() for v in DISPLAY_AXES[plane])
    if transpose:
        rd, cd = cd, rd
    if flip_c:
        rd = -rd
    if flip_r:
        cd = -cd
    normal = np.cross(rd, cd)
    vol = rng.normal(size=(5, 12, 9)).astype(np.float32)
    st = Stack(
        key="x",
        vol=vol,
        ipp=[np.array([3.0, -7.0, 11.0]) + i * 4 * normal for i in range(5)],
        row_dir=rd,
        col_dir=cd,
        row_mm=0.7,
        col_mm=1.3,
    )
    probe = [(st.pix_to_lps(k, r, c), st.vol[k, r, c]) for k, r, c in [(0, 0, 0), (4, 11, 8), (2, 5, 3)]]
    out = to_display_orientation(st)
    trd, tcd = DISPLAY_AXES[plane]
    assert np.allclose(out.row_dir, trd) and np.allclose(out.col_dir, tcd)
    for xyz, val in probe:
        k, r, c = (round(v) for v in out.lps_to_pix(xyz))
        assert out.vol[k, r, c] == pytest.approx(val)
