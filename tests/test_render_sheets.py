from __future__ import annotations

import logging

import numpy as np
import pytest

from spinescope.anatomy import Anchors
from spinescope.config import Config
from spinescope.geometry import Stack
from spinescope.render import contact_sheet, slice_frame, slice_label_lines, triptych, zoom_crop
from spinescope.render.sheets import _discs_in_slab
from spinescope.study import Study


@pytest.fixture(scope="module")
def anchors(phantom):
    return Anchors.from_truth(phantom.truth)


def test_contact_sheet_fits_png_limit(phantom_study, anchors):
    for key in ("sag_t2", "ax_t2", "loc_cor"):
        for sheet in contact_sheet(phantom_study[key], anchors=anchors):
            assert sheet.dtype == np.uint8 and sheet.ndim == 3 and sheet.shape[2] == 3
            assert max(sheet.shape[:2]) <= 2000


def test_contact_sheet_splits_when_limit_is_small(phantom_study, anchors):
    small = Config(max_png_side_px=400)
    sheets = contact_sheet(phantom_study["ax_t2"], anchors=anchors, config=small)
    assert len(sheets) > 1
    assert all(max(s.shape[:2]) <= 400 for s in sheets)


def test_contact_sheet_default_is_single_sheet_for_small_stacks(phantom_study):
    assert len(contact_sheet(phantom_study["sag_t2"])) == 1


def test_sagittal_labels_have_position_and_side(phantom_study, anchors):
    st = phantom_study["sag_t2"]
    lines = slice_label_lines(st, 0, anchors)
    assert lines[0].startswith("k=00 x=") and lines[0].endswith(("L", "R"))
    mid = slice_label_lines(st, st.n // 2, anchors)[0]
    assert mid.endswith("mid")  # the midline slice is labelled as such via the anchors' midline


def test_axial_labels_report_signed_disc_distance(phantom_study, anchors):
    st = phantom_study["ax_t2"]
    for k in (0, 10, 20):
        lines = slice_label_lines(st, k, anchors)
        lab, s = anchors.nearest_disc(slice_frame(st, k).centre)
        assert lines[1].startswith(lab) and f"{s:+.1f}" in lines[1]


def test_sagittal_tiles_list_discs_in_slab(phantom_study, anchors):
    st = phantom_study["sag_t2"]
    assert _discs_in_slab(st, st.n // 2, anchors)  # midline: discs are in the slab
    lateral = _discs_in_slab(st, 0, anchors)  # 30 mm off the midline: none
    assert lateral == []


def test_triptych_shape_and_warning(phantom_study, caplog):
    trio = [phantom_study["sag_t2"], phantom_study["sag_stir"], phantom_study["sag_t1"]]
    img = triptych(trio, 7)
    assert img.shape[2] == 3 and img.shape[1] <= 2000 and img.shape[0] <= 2000
    # a stack that does not cover the reference slice position is not co-registered
    shifted = Stack(
        key="shifted",
        vol=trio[1].vol.copy(),
        ipp=trio[1].ipp + 200.0 * trio[1].normal,
        row_dir=trio[1].row_dir,
        col_dir=trio[1].col_dir,
        row_mm=trio[1].row_mm,
        col_mm=trio[1].col_mm,
        thickness=trio[1].thickness,
        meta=dict(trio[1].meta),
    )
    with caplog.at_level(logging.WARNING):
        triptych([trio[0], shifted], 7)
    assert "not co-registered" in caplog.text


def test_triptych_resamples_different_orientation(phantom_study):
    img = triptych([phantom_study["sag_t2"], phantom_study["ax_t2"]], 7)
    assert img.ndim == 3 and img.shape[2] == 3


def test_triptych_needs_stacks():
    with pytest.raises(ValueError):
        triptych([])


def test_zoom_crop_frame_maps_centre_to_point(phantom_study):
    st = phantom_study["sag_t2"]
    centre = np.asarray(phantom_study["sag_t2"].ipp[6]) + np.array([0.0, 4.0, -6.0])
    img, fr = zoom_crop(st, centre, 40.0, k=6)
    assert img.dtype == np.uint8 and img.shape[2] == 3
    x, y, off = fr.to_px(centre)
    assert (x, y) == pytest.approx(((fr.shape[1] - 1) / 2, (fr.shape[0] - 1) / 2), abs=1e-6)
    assert abs(off) < 1e-9
    # the crosshair pixel maps back to the input LPS, and the native pixel is the same point
    k, r, c = st.lps_to_pix(fr.from_px(x, y))
    assert k == pytest.approx(6, abs=1e-6)
    assert np.allclose(st.pix_to_lps(6, r, c), centre, atol=0.05)


def test_zoom_crop_auto_upscale_gives_legible_size(phantom_study):
    from spinescope.render.frame import min_display_px

    st = phantom_study["ax_t2"]
    _, fr = zoom_crop(st, st.ipp[10], 20.0)
    assert min(fr.shape) >= min_display_px(Config())


def test_zoom_crop_explicit_upscale_sets_pitch(phantom_study):
    st = phantom_study["sag_t2"]
    _, fr4 = zoom_crop(st, st.ipp[6], 30.0, k=6, upscale=4.0)
    assert fr4.px_x == pytest.approx(st.col_mm / 4.0)
    _, fr1 = zoom_crop(st, st.ipp[6], 30.0, k=6, upscale=1.0)
    assert fr1.px_x == pytest.approx(st.col_mm)


def test_zoom_crop_rejects_bad_upscale(phantom_study):
    with pytest.raises(ValueError):
        zoom_crop(phantom_study["sag_t2"], phantom_study["sag_t2"].ipp[6], 30.0, upscale=0.0)


def test_zoom_crop_without_pixel_labels_and_rectangular_size(phantom_study):
    st = phantom_study["sag_t2"]
    img, fr = zoom_crop(st, st.ipp[6], (40.0, 20.0), k=6, pixel_coords=False)
    assert fr.extent_mm[0] == pytest.approx(40.0, abs=st.col_mm) and fr.extent_mm[1] == pytest.approx(
        20.0, abs=st.row_mm
    )
    assert img.ndim == 3


def test_study_without_stir_still_renders(phantom_study, anchors):
    stacks = {k: v for k, v in phantom_study.stacks.items() if k != "sag_stir"}
    study = Study(stacks=stacks, info=phantom_study.info)
    assert "sag_stir" not in study
    assert contact_sheet(study["sag_t2"], anchors=anchors)
    assert triptych([study["sag_t2"], study["sag_t1"]], 7).ndim == 3


def test_flipped_stack_contact_sheet_letters(phantom):
    flipped = phantom.render("sag", "t2", n=3, rows=48, cols=48, row_dir=(0, -1, 0))
    sheets = contact_sheet(flipped)
    assert len(sheets) == 1 and sheets[0].shape[2] == 3
