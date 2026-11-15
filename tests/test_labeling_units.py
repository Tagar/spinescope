"""Unit tests for the anatomy building blocks (no study needed)."""

from __future__ import annotations

import numpy as np

from spinescope.anatomy.column import Line, chain_lines, conus_tip, find_lines
from spinescope.anatomy.numbering import disc_names, level_names_in, vertebra_name
from spinescope.anatomy.sagittal import CanalTrack, SagImage, csf_polarities
from spinescope.config import DEFAULT


def test_disc_names_count_up_from_sacrum():
    names = disc_names(7)
    assert names[:3] == ["L5-S1", "L4-L5", "L3-L4"]
    assert names[-1] == "T11-T12"
    assert vertebra_name(0) == "S1" and vertebra_name(1) == "L5"


def test_level_names_in_descriptions():
    assert level_names_in("L4-5 OBL AX T2") == ["L4-L5"]
    assert level_names_in("L5S1 axial") == ["L5-S1"]
    assert level_names_in("SAG T2 TSE") == []
    assert level_names_in("T2 sag L4") == []  # a single level is not a pair


def test_csf_polarity_by_contrast():
    assert csf_polarities("t2") == (1,)
    assert csf_polarities("t1") == (-1,)
    assert csf_polarities("unknown") == (1, -1)


def test_find_lines_detects_dark_bands_and_chain_alternates():
    arc = np.arange(0, 120, 0.5)
    prof = np.ones_like(arc) * 0.6
    # Dark endplate bands (cortex) at 10, 14 (disc), 40 (body), 46 (disc), 80 (body), 86 (disc)
    for c in (10, 14, 40, 46, 80, 86):
        prof -= 0.4 * np.exp(-0.5 * ((arc - c) / 0.6) ** 2)
    lines = find_lines(prof, arc, DEFAULT)
    assert len(lines) >= 6
    _total, segs = chain_lines(lines, DEFAULT)
    kinds = [s.kind for s in segs]
    assert kinds[:5] == ["disc", "body", "disc", "body", "disc"]
    assert abs(segs[0].lo - 10) <= 3 and abs(segs[0].hi - 14) <= 3


def test_chain_rejects_spurious_intervals_that_break_alternation():
    lines = [Line(0.0, 1.0), Line(5.0, 1.0), Line(10.0, 1.0)]  # disc, disc: not allowed
    _, segs = chain_lines(lines, DEFAULT)
    assert all(s.kind != "disc" or s.hi - s.lo >= 4 for s in segs)
    assert len({s.kind for s in segs}) <= 2


def _synthetic_sag(cord: bool) -> SagImage:
    """A 2-D sagittal image with a bright CSF tube (optional darker cord core) at a = 50."""
    s = np.arange(-100.0, 0.0, 1.0)
    a = np.arange(0.0, 100.0, 1.0)
    img = np.full((len(s), len(a)), 0.3)
    for i in range(len(s)):
        img[i, np.abs(a - 50) <= 8] = 1.0
        if cord:
            img[i, np.abs(a - 50) <= 4] = 0.35
    return SagImage(
        img=img,
        a=a,
        s=s,
        origin=np.zeros(3),
        ap=np.array([0.0, 1.0, 0.0]),
        sup=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        stack_key="synthetic",
        slice_index=0,
        res=1.0,
    )


def test_conus_found_where_cord_stops():
    sag = _synthetic_sag(cord=True)
    # cord present along the whole track, so the tip is at the lower end of the track
    s = sag.s
    track = CanalTrack(
        s=s,
        a=np.full_like(s, 50.0),
        coef=np.array([0.0, 0.0, 0.0, 50.0]),
        score=1.0,
        polarity=1,
    )
    tip = conus_tip(sag, track, DEFAULT)
    assert tip is not None
    assert abs(tip[0] - s[0]) <= 2.0


def test_no_conus_without_cord_signal():
    sag = _synthetic_sag(cord=False)
    s = sag.s
    track = CanalTrack(
        s=s, a=np.full_like(s, 50.0), coef=np.array([0.0, 0.0, 0.0, 50.0]), score=1.0, polarity=1
    )
    assert conus_tip(sag, track, DEFAULT) is None
