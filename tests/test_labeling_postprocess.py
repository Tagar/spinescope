"""Post-processing fixes for the learned labeller (lab09_errors), tested on synthetic probability maps.

No network: the disc probability is painted by hand on the phantom's sagittal grids, so each rule is
tested in isolation (8-connected instances, tilt plausibility, caudal outliers, 3-D centroid and normal).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spinescope.anatomy.anchors import Disc
from spinescope.anatomy.ml.grid import plane_grid
from spinescope.anatomy.ml.labeller import DISC, SegmentedStack
from spinescope.anatomy.ml.postprocess import (
    drop_caudal_outliers,
    drop_duplicate_discs,
    extract_discs,
    refine_disc_3d,
)
from spinescope.config import DEFAULT
from spinescope.testing.phantom import SpinePhantom


def _seg(n: int = 1, rows: int = 160, cols: int = 160) -> SegmentedStack:
    """A stack of ``n`` phantom sagittal slices with empty probability maps on 1 mm grids."""
    stack = SpinePhantom().render("sag", "t2", n=n, rows=rows, cols=cols)
    grids = [plane_grid(stack, k, 1.0) for k in range(stack.n)]
    h, w = grids[0].shape
    probs = np.zeros((stack.n, 4, h, w), np.float32)
    probs[:, 0] = 1.0  # background everywhere, so only the disc channel is painted below
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=(0.0, 1.0))


def _paint(seg: SegmentedStack, k: int, rows: slice, cols: slice) -> None:
    """Set the disc class to 1 on a rectangle of slice ``k`` (grid pixel indices)."""
    seg.probs[k, DISC, rows, cols] = 1.0
    seg.probs[k, 0, rows, cols] = 0.0


def _centre_of(seg: SegmentedStack, k: int) -> tuple[int, int]:
    h, w = seg.grids[k].shape
    return h // 2, w // 2


# -------------------------------------------------------------------------- instances (8-connectivity)
def test_diagonal_seam_is_one_disc() -> None:
    """Two blocks that touch only at a corner are one disc, not two levels."""
    seg = _seg()
    r, c = _centre_of(seg, 0)
    _paint(seg, 0, slice(r - 4, r + 1), slice(c - 15, c + 1))  # upper-left block
    _paint(seg, 0, slice(r + 1, r + 6), slice(c + 1, c + 16))  # lower-right block, corner contact
    discs, _ = extract_discs(seg, 0)
    assert len(discs) == 1
    assert discs[0].height_mm is not None and discs[0].height_mm >= 5.0


def test_separated_blocks_stay_separate() -> None:
    """A gap of one pixel along an edge keeps two components (8-connectivity does not bridge it)."""
    seg = _seg()
    r, c = _centre_of(seg, 0)
    _paint(seg, 0, slice(r - 12, r - 3), slice(c - 15, c + 15))  # disc 1
    _paint(seg, 0, slice(r + 4, r + 13), slice(c - 15, c + 15))  # disc 2, 3 px below
    discs, _ = extract_discs(seg, 0)
    assert len(discs) == 2


# -------------------------------------------------------------------------- tilt plausibility
def test_vertical_strip_is_rejected_by_tilt() -> None:
    """A narrow vertical strip has a horizontal normal: it is not a disc, even though its aspect passes."""
    seg = _seg()
    r, c = _centre_of(seg, 0)
    _paint(seg, 0, slice(r - 6, r + 7), slice(c, c + 5))
    discs, warns = extract_discs(seg, 0)
    assert discs == []
    assert any("tilted" in w for w in warns), warns


def test_horizontal_disc_is_kept_with_flat_normal() -> None:
    seg = _seg()
    r, c = _centre_of(seg, 0)
    _paint(seg, 0, slice(r - 4, r + 5), slice(c - 15, c + 16))
    discs, warns = extract_discs(seg, 0)
    assert len(discs) == 1 and not warns
    assert abs(discs[0].normal[2]) > 0.99  # superior-inferior normal


# -------------------------------------------------------------------------- caudal outliers
def _disc_at(z: float) -> Disc:
    return Disc(center=[0.0, 0.0, z], normal=[0.0, 0.0, 1.0], height_mm=8.0, confidence=0.9)


def test_isolated_caudal_disc_is_dropped() -> None:
    """A disc 60 mm below the rest of the column (sacral false positive) is dropped and reported."""
    discs = [_disc_at(-60.0), _disc_at(2.0), _disc_at(36.0), _disc_at(70.0)]
    kept, warns = drop_caudal_outliers(discs, DEFAULT)
    assert [round(d.center[2]) for d in kept] == [2, 36, 70]
    assert len(warns) == 1 and "dropped" in warns[0]


def test_normal_level_spacing_is_kept() -> None:
    discs = [_disc_at(-30.0), _disc_at(2.0), _disc_at(36.0)]
    kept, warns = drop_caudal_outliers(discs, DEFAULT)
    assert len(kept) == 3 and not warns


def test_single_disc_is_never_dropped() -> None:
    kept, warns = drop_caudal_outliers([_disc_at(-100.0)], DEFAULT)
    assert len(kept) == 1 and not warns


# -------------------------------------------------------------------------- 3-D centroid and normal
def _slab(seg: SegmentedStack, k: int, shift_per_slice: float, span: int) -> None:
    """A flat disc (9 mm high, 30 mm AP) on slices k-span..k+span whose centre slides by ``shift_per_slice``
    mm per slice away from the mid-plane (in the superior-inferior direction is avoided: we shift in AP)."""
    r, c = _centre_of(seg, k)
    for kk in range(k - span, k + span + 1):
        g = seg.grids[kk]
        dc = round(abs(kk - k) * shift_per_slice)  # pixels (1 mm grid)
        seg.probs[kk, DISC, r - 4 : r + 5, c - 15 + dc : c + 16 + dc] = 1.0
        seg.probs[kk, 0, r - 4 : r + 5, c - 15 + dc : c + 16 + dc] = 0.0
        assert g.shape[0] > r + 5


def test_refine_uses_the_3d_centroid_across_slices() -> None:
    """The disc's centroid is the mean over the slices it spans, so a disc that drifts with the slice index
    moves its 3-D centre away from the mid-plane centre (here the mean of |offset| over the span)."""
    seg = _seg(n=9)
    k = seg.stack.n // 2
    _slab(seg, k, shift_per_slice=1.0, span=3)
    discs, _ = extract_discs(seg, k)
    assert len(discs) == 1
    mid_centre = np.asarray(discs[0].center)
    refined = refine_disc_3d(seg, k, discs[0], DEFAULT)
    moved = np.asarray(refined.center) - mid_centre
    expected = np.mean([0, 1, 1, 2, 2, 3, 3])  # mm, averaged over the seven slices (1 mm per slice)
    # Posterior is +y in LPS; the drift is in the column direction (AP), so the shift is along y.
    assert abs(abs(moved[1]) - expected) < 0.6, (moved, expected)
    assert "3-D disc" in refined.evidence[-1]


def test_refine_keeps_a_flat_disc_in_place_and_normal_superior() -> None:
    seg = _seg(n=9)
    k = seg.stack.n // 2
    _slab(seg, k, shift_per_slice=0.0, span=4)
    disc = extract_discs(seg, k)[0][0]
    refined = refine_disc_3d(seg, k, disc, DEFAULT)
    assert np.allclose(refined.center, disc.center, atol=0.6)
    assert refined.normal[2] > 0.99


def test_refine_returns_disc_unchanged_when_seen_on_few_slices() -> None:
    seg = _seg(n=9)
    k = seg.stack.n // 2
    _slab(seg, k, shift_per_slice=0.0, span=0)  # mid-plane only
    disc = extract_discs(seg, k)[0][0]
    before = list(disc.center), list(disc.normal)
    refined = refine_disc_3d(seg, k, disc, DEFAULT)
    assert (list(refined.center), list(refined.normal)) == before
    assert not any("3-D disc" in e for e in refined.evidence)


@pytest.mark.parametrize("span", [2, 4])
def test_refine_normal_of_a_tilted_disc(span: int) -> None:
    """A disc tilted 20 deg in the sagittal plane: the 3-D normal recovers the tilt."""
    seg = _seg(n=9, rows=200, cols=200)
    k = seg.stack.n // 2
    r, c = _centre_of(seg, k)
    for kk in range(k - span, k + span + 1):
        for dc in range(-18, 19):
            # One row of disc per column, rising 0.36 px per px in AP (about 20 degrees).
            top = round(r - 4 + 0.36 * dc)
            seg.probs[kk, DISC, top : top + 7, c + dc] = 1.0
            seg.probs[kk, 0, top : top + 7, c + dc] = 0.0
    discs, _ = extract_discs(seg, k)
    assert len(discs) == 1
    refined = refine_disc_3d(seg, k, discs[0], replace(DEFAULT, ml_disc_normal_3d=True))
    # Normal tilt from the superior-inferior axis is near 20 degrees; the 2-D estimate agrees here too.
    tilt = float(np.degrees(np.arccos(min(1.0, abs(refined.normal[2])))))
    assert 12.0 <= tilt <= 28.0, tilt


# -------------------------------------------------------------------------- duplicates after refinement
def _disc_yz(y: float, z: float, conf: float) -> Disc:
    return Disc(center=[0.0, y, z], normal=[0.0, 0.0, 1.0], height_mm=8.0, confidence=conf)


def test_duplicate_after_refinement_keeps_the_more_confident_disc() -> None:
    """Two estimates of one disc 11 mm apart in the sagittal plane: the more confident one is kept."""
    weak = _disc_yz(38.9, 49.1, 0.88)
    strong = _disc_yz(27.1, 49.2, 0.92)
    kept, warns = drop_duplicate_discs([weak, _disc_yz(14.0, 14.0, 0.9), strong], DEFAULT)
    assert len(kept) == 2 and strong in kept and weak not in kept
    assert kept[0].center[1] == 14.0 and kept[1] is strong  # input order is kept
    assert len(warns) == 1 and "duplicate" in warns[0]


def test_adjacent_discs_at_level_spacing_are_both_kept() -> None:
    a, b = _disc_yz(30.0, 0.0, 0.9), _disc_yz(30.0, 30.0, 0.8)
    kept, warns = drop_duplicate_discs([a, b], DEFAULT)
    assert kept == [a, b] and not warns


def test_default_keeps_the_mid_plane_normal() -> None:
    """The 3-D normal is opt-in: by default only the centre moves to the 3-D centroid."""
    seg = _seg(n=9, rows=200, cols=200)
    k = seg.stack.n // 2
    r, c = _centre_of(seg, k)
    for kk in range(k - 3, k + 4):
        for dc in range(-18, 19):
            top = round(r - 4 + 0.36 * dc)
            seg.probs[kk, DISC, top : top + 7, c + dc] = 1.0
            seg.probs[kk, 0, top : top + 7, c + dc] = 0.0
    disc = extract_discs(seg, k)[0][0]
    refined = refine_disc_3d(seg, k, disc, DEFAULT)
    assert refined.normal == disc.normal
    assert "centre from the 3-D disc" in refined.evidence[-1]
