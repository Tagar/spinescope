"""Series classification from vendor protocol names (no TR/TE), and from full acquisition headers."""

from __future__ import annotations

import numpy as np
import pytest

from spinescope.geometry import Stack
from spinescope.series import DISPLAY_AXES, classify


def _stack(plane: str = "sag", **dicom: object) -> Stack:
    rd, cd = DISPLAY_AXES[plane]
    return Stack(
        key="",
        vol=np.zeros((12, 8, 8)),
        ipp=[i * np.cross(rd, cd) * 4 for i in range(12)],
        row_dir=rd,
        col_dir=cd,
        row_mm=1,
        col_mm=1,
        thickness=4,
        meta={"dicom": dict(dicom)},
    )


# Protocol / series names as they appear on common lumbar protocols (Siemens, GE, Philips, Canon/Toshiba,
# Hitachi/Fujifilm). Only the name is given, so classification must work from the text alone.
NAME_CASES = [
    # (SeriesDescription, expected contrast, fat_suppressed, post_contrast, dixon)
    ("t2_tse_sag", "t2", False, False, None),  # Siemens-style protocol name
    ("T2W_TSE", "t2", False, False, None),  # Philips-style T2W
    ("SAG T2 FRFSE", "t2", False, False, None),  # GE fast-recovery FSE
    ("Sag T2 FSE", "t2", False, False, None),  # GE / Toshiba
    ("T2 TSE SAG", "t2", False, False, None),
    ("SAG T2 FSE", "t2", False, False, None),
    ("Ax T2 FS", "t2", True, False, None),
    ("T2FS SAG", "t2", True, False, None),  # compound name
    ("T2 SPAIR", "t2", True, False, None),  # Siemens spectral fat sat
    ("T2W TSE SPIR", "t2", True, False, None),
    ("T2 SPACE", "t2", False, False, None),  # 3-D isotropic T2 (Siemens SPACE)
    ("3D SPACE T2", "t2", False, False, None),
    ("3D T2 CUBE", "t2", False, False, None),  # GE CUBE
    ("3D T2 VISTA", "t2", False, False, None),  # Philips VISTA
    ("STIR", "stir", False, False, None),
    ("SAG STIR FSE", "stir", False, False, None),
    ("t2_tirm_sag", "stir", False, False, None),  # Siemens TIRM (STIR-type)
    ("SAG STIR/TIRM", "stir", False, False, None),
    ("T1 TSE", "t1", False, False, None),
    ("SAG T1 FSE", "t1", False, False, None),
    ("t1_tse_sag", "t1", False, False, None),
    ("T1W_SE", "t1", False, False, None),
    ("T1 FS SAG POST GAD", "t1", True, True, None),
    ("T1 SAG POST CONTRAST", "t1", False, True, None),
    ("T1 NON-CONTRAST", "t1", False, False, None),  # negative contrast words must not flag post-contrast
    ("T2 MEDIC", "gre", False, False, None),  # gradient echo with T2* in the name
    ("MEDIC", "gre", False, False, None),
    ("T2_STAR", "gre", False, False, None),
    ("FFE", "gre", False, False, None),
    ("3D SPGR T1", "t1", False, False, None),  # gradient echo, but explicitly T1-weighted
    ("DIXON W", None, False, False, "w"),  # Philips Dixon components
    ("DIXON F", None, False, False, "f"),
    ("DIXON IP", None, False, False, "ip"),
    ("DIXON OP", None, False, False, "op"),
    ("Dixon_W", None, False, False, "w"),
    ("IDEAL WATER", None, False, False, "w"),
    ("T1 DIXON W", "t1", False, False, "w"),
    ("ENHANCED MR", None, False, False, None),  # "enhanced" describes the object, not the contrast
    ("3-PL LOC", None, False, False, None),
]


@pytest.mark.parametrize(("name", "contrast", "fat", "post", "dixon"), NAME_CASES)
def test_vendor_name_classification(name, contrast, fat, post, dixon):
    role = classify(_stack(SeriesDescription=name))
    if contrast is not None:
        assert role.contrast == contrast, role.evidence
    assert role.fat_suppressed is fat, role.evidence
    assert role.post_contrast is post, role.evidence
    assert role.dixon == dixon, role.evidence


def test_localizer_by_name_and_by_geometry():
    assert classify(_stack(SeriesDescription="AAHScout")).localizer
    assert classify(_stack(SeriesDescription="3-PL LOC")).localizer
    assert classify(_stack(SeriesDescription="SAG T2 FSE")).localizer is False


@pytest.mark.parametrize(
    ("dicom", "contrast"),
    [
        # GE-style header: SE with long TR and TE, name is vendor-specific
        ({"SeriesDescription": "Sag T2 FRFSE", "RepetitionTime": 4200, "EchoTime": 102}, "t2"),
        # Philips-style: name says T1 but the header agrees with T1 (short TR/TE)
        ({"SeriesDescription": "T1 TSE SAG", "RepetitionTime": 550, "EchoTime": 10}, "t1"),
        # Siemens STIR: inversion time in the 150-180 ms range
        (
            {
                "SeriesDescription": "t2_tirm_sag",
                "RepetitionTime": 5000,
                "EchoTime": 50,
                "InversionTime": 160,
            },
            "stir",
        ),
        # Canon/Toshiba: header wins over a misleading name without T tokens
        ({"SeriesDescription": "SAG FSE", "RepetitionTime": 3000, "EchoTime": 100}, "t2"),
        ({"SeriesDescription": "SAG FSE", "RepetitionTime": 500, "EchoTime": 12}, "t1"),
        # Hitachi/Fujifilm: name only, no timing parameters at all
        ({"SeriesDescription": "SAG T2 FSE FAT SAT"}, "t2"),
    ],
)
def test_header_and_name_agree(dicom, contrast):
    role = classify(_stack(**dicom))
    assert role.contrast == contrast, role.evidence


def test_fat_sat_from_scan_options_only():
    role = classify(_stack(SeriesDescription="SAG T2 FSE", ScanOptions=["FS"]))
    assert role.fat_suppressed and role.contrast == "t2"
