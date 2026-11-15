"""Volume orientation: slice axis, obliquity and plane label for rotated direction matrices.

Each case writes a small uncompressed .mha with SimpleITK whose direction matrix is a sagittal layout rotated
0-90 deg about one patient axis (L = x, P = y, S = z). Checks: the geometry of every voxel against SimpleITK's own
index-to-point transform, the slice-axis decision and recorded obliquity, the plane label through the full
``load_study`` path, and agreement of the built-in MetaImage reader with SimpleITK. Two base layouts: ``std`` (slices
along index 0, SPIDER's usual files) and ``spider_rot`` (slices along index 2, directions P+, S-, L-, SPIDER's
rotated files). Nothing here needs data on disk.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from spinescope import load_study
from spinescope.geometry import obliquity_deg, plane_margin_deg
from spinescope.io.volumes import _read_metaimage, choose_slice_axis, read_label_volume, read_volume

sitk = pytest.importorskip("SimpleITK")

ANGLES = (0.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0)
AXES = ("x", "y", "z")  # rotation about patient x = L, y = P, z = S
L_PLUS, S_MINUS, P_PLUS, L_MINUS = (1, 0, 0), (0, 0, -1), (0, 1, 0), (-1, 0, 0)

# Columns of each layout are the LPS directions of index axes i, j, k (ITK convention); det = +1 for both.
LAYOUTS = {
    # slices along i (thick, 4 mm, direction L), rows along j (S-), columns along k (P+)
    "std": {"cols": (L_PLUS, S_MINUS, P_PLUS), "spacing": (4.0, 0.6, 0.8), "sizes": (5, 12, 14), "thick": 0},
    # slices along k (thick, 4 mm, direction L-), rows along j (S-), columns along i (P+)
    "spider_rot": {
        "cols": (P_PLUS, S_MINUS, L_MINUS),
        "spacing": (0.6, 0.8, 4.0),
        "sizes": (14, 12, 5),
        "thick": 2,
    },
}


def _rot(axis: str, deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _direction(layout: str, axis: str | None, deg: float) -> np.ndarray:
    d0 = np.column_stack([np.asarray(c, dtype=float) for c in LAYOUTS[layout]["cols"]])
    return d0 if axis is None else _rot(axis, deg) @ d0


def _expected(axis: str, deg: float) -> tuple[str, float]:
    """(plane label, obliquity of the slice normal) for a sagittal layout rotated about ``axis``."""
    if axis == "x":
        return "sag", 0.0
    obl = min(deg, 90.0 - deg)
    if axis == "y":
        return ("sag" if deg < 45 else "ax"), obl
    return ("sag" if deg < 45 else "cor"), obl


def _write(path: Path, layout: str, axis: str, deg: float) -> None:
    spec = LAYOUTS[layout]
    ni, nj, nk = spec["sizes"]
    arr = np.arange(ni * nj * nk, dtype=np.float32).reshape(
        nk, nj, ni
    )  # value = flat index, decodes to voxel
    img = sitk.GetImageFromArray(arr)  # SimpleITK arrays are [z, y, x]
    img.SetSpacing(tuple(float(v) for v in spec["spacing"]))
    img.SetOrigin((12.0, -30.0, 45.0))
    img.SetDirection(tuple(float(v) for v in _direction(layout, axis, deg).reshape(-1)))
    sitk.WriteImage(img, str(path))


CASES = [(lay, ax, deg) for lay in LAYOUTS for ax in AXES for deg in ANGLES]
IDS = [f"{lay}-{ax}{int(deg)}" for lay, ax, deg in CASES]


@pytest.mark.parametrize(("layout", "axis", "deg"), CASES, ids=IDS)
def test_voxel_geometry_matches_simpleitk(tmp_path, layout, axis, deg):
    path = tmp_path / "vol_t2.mha"
    _write(path, layout, axis, deg)
    stack = read_volume(str(path))
    img = sitk.ReadImage(str(path))
    ni, nj, _nk = LAYOUTS[layout]["sizes"]
    for s in range(stack.n):
        for r in range(stack.shape[1]):
            for c in range(stack.shape[2]):
                flat = round(float(stack.vol[s, r, c]))
                ii, jj, kk = flat % ni, (flat // ni) % nj, flat // (ni * nj)
                expect = np.array(img.TransformIndexToPhysicalPoint((ii, jj, kk)))
                assert np.allclose(stack.pix_to_lps(s, r, c), expect, atol=1e-3)


@pytest.mark.parametrize(("layout", "axis", "deg"), CASES, ids=IDS)
def test_slice_axis_and_obliquity_are_recorded(tmp_path, layout, axis, deg):
    path = tmp_path / "vol_t2.mha"
    _write(path, layout, axis, deg)
    stack = read_volume(str(path))
    _plane, obl = _expected(axis, deg)
    info = stack.meta["volume"]
    assert info["slice_axis"] == LAYOUTS[layout]["thick"]
    assert info["slice_rule"] == "largest_spacing" and info["slice_ambiguous"] is False
    assert info["obliquity_deg"] == pytest.approx(obl, abs=0.05)
    assert stack.obliquity == pytest.approx(obl, abs=0.05)
    if obl not in (20.0, 70.0):  # the alignment boundary itself is not tested
        assert info["slice_aligned"] is (obl <= 20.0)
    assert stack.thickness == pytest.approx(LAYOUTS[layout]["spacing"][LAYOUTS[layout]["thick"]])


@pytest.mark.parametrize(("layout", "axis", "deg"), CASES, ids=IDS)
def test_plane_label_and_ambiguity_through_load_study(tmp_path, layout, axis, deg):
    path = tmp_path / "vol_t2.mha"
    _write(path, layout, axis, deg)
    study = load_study(str(path), load_pixels=True)
    (stack,) = study.stacks.values()
    plane, obl = _expected(axis, deg)
    role = stack.meta["role"]
    assert role["obliquity_deg"] == pytest.approx(obl, abs=0.05)
    margin = 90.0 - 2.0 * deg if axis != "x" else 90.0  # gap between the two nearest canonical axes
    ambiguous = abs(margin) < 15.0 if axis != "x" else False
    assert role["plane_ambiguous"] is ambiguous
    if deg != 45.0:  # at exactly 45 deg either label is acceptable; the ambiguity flag is the contract
        assert stack.plane == plane
    else:
        assert stack.plane in ("sag", "cor", "ax")
    if plane == "sag" and deg != 45.0:
        assert study.best("sag") is not None


@pytest.mark.parametrize(("layout", "axis", "deg"), CASES, ids=IDS)
def test_builtin_metaimage_reader_agrees_with_simpleitk(tmp_path, layout, axis, deg):
    path = tmp_path / "vol_t2.mha"
    _write(path, layout, axis, deg)
    arr_b, sp_b, org_b, dir_b = _read_metaimage(str(path))
    img = sitk.ReadImage(str(path))
    arr_s = sitk.GetArrayFromImage(img)
    assert np.array_equal(arr_b, arr_s)
    assert np.allclose(sp_b, img.GetSpacing())
    assert np.allclose(org_b, img.GetOrigin())
    assert np.allclose(dir_b, np.array(img.GetDirection()).reshape(3, 3))


@pytest.mark.parametrize(("layout", "axis", "deg"), CASES, ids=IDS)
def test_label_volume_is_on_the_image_grid(tmp_path, layout, axis, deg):
    path = tmp_path / "mask_t2.mha"
    _write(path, layout, axis, deg)
    lab = read_label_volume(str(path))
    stack = read_volume(str(path))
    assert lab.shape == stack.shape
    assert np.array_equal(lab, stack.vol)


def test_isotropic_voxels_are_decided_by_canonical_alignment():
    ident = np.column_stack([np.asarray(c, dtype=float) for c in LAYOUTS["std"]["cols"]])
    # Unrotated, all three axes are equally thick and canonical: the sagittal (L) normal wins the tie.
    ch = choose_slice_axis((0.7, 0.7, 0.7), ident)
    assert (ch.axis, ch.rule, ch.ambiguous, ch.aligned) == (0, "isotropic_aligned", True, True)
    # Rotated 30 deg about z, the S-directed axis is still exactly canonical: it is the most aligned choice.
    rotated = _rot("z", 30.0) @ ident
    ch = choose_slice_axis((0.7, 0.7, 0.7), rotated)
    assert (ch.axis, ch.rule, ch.ambiguous) == (1, "isotropic_aligned", True)
    # Within the 15 % isotropic band spacing does not decide either.
    assert choose_slice_axis((1.0, 1.1, 1.0), ident).rule == "isotropic_aligned"


def test_clear_spacing_beats_alignment_for_oblique_slabs():
    """A tilted sagittal slab is still sliced across its thick axis; the obliquity is recorded, not overridden."""
    tilted = _rot("z", 30.0) @ np.column_stack([np.asarray(c, dtype=float) for c in LAYOUTS["std"]["cols"]])
    ch = choose_slice_axis((4.0, 0.6, 0.8), tilted)
    assert (ch.axis, ch.rule, ch.aligned, ch.ambiguous) == (0, "largest_spacing", False, False)
    assert ch.deviation_deg == pytest.approx(30.0, abs=1e-3)


def test_plane_margin_separates_canonical_from_45_degrees():
    assert plane_margin_deg((1.0, 0.0, 0.0)) == pytest.approx(90.0)
    assert plane_margin_deg((1.0, 1.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    assert plane_margin_deg((math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0)) == pytest.approx(
        30.0
    )
    assert obliquity_deg((1.0, 1.0, 0.0)) == pytest.approx(45.0)


def test_non_volume_input_is_rejected(tmp_path):
    path = tmp_path / "flat.mha"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((8, 9), dtype=np.float32)), str(path))
    with pytest.raises(ValueError, match="3-D"):
        read_volume(str(path))


def _spider_case_path(sid: str, seq: str = "t2") -> Path | None:
    """Path of a SPIDER image on disk, or None (the dataset is optional)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _spider import spider_root

    root = spider_root()
    if root is None:
        return None
    path = root / "images" / f"{sid}_{seq}.mha"
    return path if path.exists() else None


# Case 35 is a rotated (slices along y) validation study; 107 is a rotated study too. Both were coronal on the
# paired path and 0 discs (35) before the fix; the single-file path labels 3 discs for 35.
@pytest.mark.dataset
@pytest.mark.parametrize("sid", ["35", "107"])
def test_spider_rotated_pair_is_sagittal_and_matches_single_file(sid):
    import sys

    path = _spider_case_path(sid)
    if path is None:
        pytest.skip("SPIDER not on disk (set SPINESCOPE_DATA)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _spider import load_t2_t1

    loaded = load_t2_t1(sid)
    assert loaded is not None
    study, t2, mask = loaded
    single = load_study(str(path))
    assert t2.plane == "sag" and study.best("sag") is not None
    assert np.array_equal(t2.vol, single["sag_t2"].vol)
    assert np.allclose(t2.ipp, single["sag_t2"].ipp)
    assert mask.shape == t2.vol.shape


@pytest.mark.dataset
@pytest.mark.slow
def test_spider_case_35_discs_match_on_both_paths():
    pytest.importorskip("onnxruntime")
    import sys

    if _spider_case_path("35") is None:
        pytest.skip("SPIDER not on disk (set SPINESCOPE_DATA)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _spider import load_t2_t1
    from spinescope.anatomy.pipeline import label_spine

    pair, _t2, _mask = load_t2_t1("35")
    single = load_study(str(_spider_case_path("35")))
    n_pair = len(label_spine(pair, backend="ml").discs)
    n_single = len(label_spine(single, backend="ml").discs)
    assert n_single >= 3
    assert n_pair == n_single
