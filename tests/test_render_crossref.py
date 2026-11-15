from __future__ import annotations

import logging

import numpy as np
import pytest

from spinescope.anatomy import Anchors
from spinescope.config import Config
from spinescope.geometry import intersection_segment
from spinescope.render import linked_views, reference_lines, slice_frame
from spinescope.render.annotate import PALETTE
from spinescope.render.frame import display_scale, min_display_px


@pytest.fixture(scope="module")
def anchors(phantom):
    return Anchors.from_truth(phantom.truth)


def test_axial_line_on_sagittal_lands_at_its_row(phantom_study, anchors):
    sag, ax = phantom_study["sag_t2"], phantom_study["ax_t2"]
    k_a, kb = 7, 10
    seg = intersection_segment(sag, k_a, ax, kb)
    assert seg is not None
    p0 = sag.pix_to_lps(k_a, *seg[0])
    assert p0[2] == pytest.approx(ax.ipp[kb][2], abs=1e-6)  # the line lies on the axial plane
    native = slice_frame(sag, k_a)
    expect_row = native.to_px(ax.ipp[kb])[1]
    out = reference_lines(sag, k_a, [(ax, [kb])], anchors=anchors, config=Config())
    _base, frame = display_scale(np.zeros((*native.shape, 3), np.uint8), native, min_display_px(Config()))
    title_h = out.shape[0] - frame.shape[0]
    colour = np.array(PALETTE[0], dtype=np.uint8)
    counts = (out == colour).all(-1).sum(axis=1)
    row = int(np.argmax(counts))
    assert counts[row] > 0.5 * frame.shape[1]  # a full-width line in that colour
    assert abs((row - title_h) - expect_row * frame.shape[0] / native.shape[0]) <= 2


def test_each_reference_stack_gets_its_colour(phantom_study):
    sag, ax, stir = phantom_study["sag_t2"], phantom_study["ax_t2"], phantom_study["sag_stir"]
    out = reference_lines(sag, 7, [(ax, [5]), (stir, [])])
    assert (out == np.array(PALETTE[0], dtype=np.uint8)).all(-1).any()
    assert not (out == np.array(PALETTE[1], dtype=np.uint8)).all(-1).any()


def test_reference_lines_no_intersection_is_reported(phantom_study):
    sag, loc = phantom_study["sag_t2"], phantom_study["sag_t1"]
    out = reference_lines(sag, 7, [(loc, [])])
    assert out.ndim == 3 and out.shape[2] == 3


def test_linked_views_map_back_to_input_point(phantom_study, phantom):
    point = np.asarray(phantom.discs["L4-L5"]["center"]) + np.array([0.0, 2.0, 0.0])
    stacks = {k: phantom_study[k] for k in ("sag_t2", "sag_t1", "ax_t2")}
    img, recs = linked_views(point, stacks)
    assert img.shape[2] == 3 and len(recs) == 3
    for rec in recs:
        st = stacks[rec["key"]]
        p_on_plane = st.pix_to_lps(rec["k"], rec["native_row"], rec["native_col"])
        n = st.normal
        projected = point - ((point - st.ipp[rec["k"]]) @ n) * n
        assert np.linalg.norm(p_on_plane - projected) < 1e-3
        assert rec["signed_mm"] == pytest.approx(float((point - st.ipp[rec["k"]]) @ n), abs=1e-2)


def test_linked_views_flag_off_slab_and_outside_fov(phantom_study, phantom):
    sag = phantom_study["sag_t2"]
    k = 6
    # the 5 mm sagittal spacing leaves gaps between 4 mm slabs: 2.5 mm off the plane is off-slab
    point_gap = sag.ipp[k] + 2.5 * sag.normal + np.array([0.0, 0.0, 0.0])
    _, recs = linked_views(point_gap, {"sag_t2": sag})
    assert "OFF SLAB" in recs[0]["flags"] and recs[0]["in_slab"] is False
    small = phantom.render("sag", "t2", n=3, rows=20, cols=20, px=1.0, key="small")
    far = small.pix_to_lps(1, 10, 10) + 80.0 * small.row_dir
    _, recs2 = linked_views(far, {"small": small})
    assert "OUTSIDE FOV" in recs2[0]["flags"] and recs2[0]["inside_fov"] is False


def test_linked_views_skip_missing_and_need_one(phantom_study):
    sag = phantom_study["sag_t2"]
    _img, recs = linked_views(sag.ipp[6], [sag, None])
    assert len(recs) == 1
    with pytest.raises(ValueError):
        linked_views(sag.ipp[6], [None])


def test_linked_views_title_flags_visible(phantom_study):
    sag = phantom_study["sag_t2"]
    _, recs = linked_views(sag.ipp[6] + 2.5 * sag.normal, [sag])
    assert recs[0]["flags"] == ["OFF SLAB"]
    assert recs[0]["dist_mm"] == pytest.approx(2.5, abs=1e-2)


def test_reference_lines_warn_nothing_for_matching_stack(phantom_study, caplog):
    with caplog.at_level(logging.WARNING):
        reference_lines(phantom_study["sag_t2"], 7, [(phantom_study["sag_stir"], [6])])
    assert "not co-registered" not in caplog.text
