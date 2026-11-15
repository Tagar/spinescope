from __future__ import annotations

import numpy as np
import pytest

from spinescope.render import annotate, direction_letter, nice_step, slice_frame
from spinescope.render.frame import Frame, display_scale


def test_frame_round_trip_and_off_plane(phantom_study):
    st = phantom_study["sag_t2"]
    fr = slice_frame(st, 3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x, y = rng.uniform(0, fr.shape[1] - 1), rng.uniform(0, fr.shape[0] - 1)
        p = fr.from_px(x, y)
        x2, y2, off = fr.to_px(p)
        assert (x2, y2) == pytest.approx((x, y), abs=1e-9) and abs(off) < 1e-9
    # a point 3 mm off the plane reports that offset, with the same in-plane pixel
    x, y, off = fr.to_px(fr.from_px(10, 20) + 3.0 * fr.normal)
    assert (x, y) == pytest.approx((10, 20), abs=1e-9) and off == pytest.approx(3.0)


def test_slice_frame_matches_stack_pixel_mapping(phantom_study):
    st = phantom_study["ax_t2"]
    fr = slice_frame(st, 5)
    for r, c in [(0, 0), (17, 60), (127, 3)]:
        assert np.allclose(fr.from_px(c, r), st.pix_to_lps(5, r, c))


def test_orientation_letters_by_plane(phantom_study):
    assert slice_frame(phantom_study["sag_t2"], 0).edge_letters() == {
        "right": "P",
        "left": "A",
        "top": "S",
        "bottom": "I",
    }
    assert slice_frame(phantom_study["ax_t2"], 0).edge_letters() == {
        "right": "L",
        "left": "R",
        "top": "A",
        "bottom": "P",
    }


def test_flipped_acquisition_letters_follow_geometry(phantom):
    flipped = phantom.render("sag", "t2", n=2, rows=32, cols=32, row_dir=(0, -1, 0))
    fr = slice_frame(flipped, 0)
    assert fr.edge_letters()["right"] == "A" and fr.edge_letters()["left"] == "P"


def test_direction_letter_and_nice_step():
    assert direction_letter((1, 0, 0)) == "L" and direction_letter((0, 0, -0.9)) == "I"
    assert direction_letter((0.1, -1, 0)) == "A"
    assert nice_step(0.9) == 1.0 and nice_step(2.6) == 2.0 and nice_step(70) == 50.0
    assert nice_step(0) == 1.0


def test_scaled_frame_is_consistent_with_native(phantom_study):
    st = phantom_study["sag_t2"]
    fr = slice_frame(st, 4)
    big = fr.scaled(3)
    assert big.shape == (fr.shape[0] * 3, fr.shape[1] * 3)
    p = fr.from_px(20.3, 40.7)
    x, y, _ = big.to_px(p)
    assert (x, y) == pytest.approx(((20.3 + 0.5) * 3 - 0.5, (40.7 + 0.5) * 3 - 0.5), abs=1e-9)


def test_display_scale_upscales_only_when_small(phantom_study):
    fr0 = slice_frame(phantom_study["sag_t2"], 4)
    fr = Frame(fr0.origin, fr0.u, fr0.v, fr0.px_x, fr0.px_y, (40, 50))
    small = np.zeros((40, 50, 3), dtype=np.uint8)
    out, f2 = display_scale(small, fr, 120)
    assert min(out.shape[:2]) >= 120 and f2.shape == out.shape[:2]
    same, f3 = display_scale(out, f2, 120)
    assert same is out and f3 is f2


def _blank(h: int = 120, w: int = 160) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_annotations_are_pure_rgb(phantom_study):
    fr = slice_frame(phantom_study["sag_t2"], 4)
    base = _blank(fr.shape[0], fr.shape[1])
    snapshot = base.copy()
    p0 = fr.from_px(10, 10)
    p1 = fr.from_px(60, 40)
    outs = [
        annotate.line(base, fr, p0, p1),
        annotate.arrow(base, fr, p0, p1, text="a"),
        annotate.circle(base, fr, p0, 4.0),
        annotate.crosshair(base, fr, p0),
        annotate.text_at(base, fr, p0, "x"),
        annotate.orientation_letters(base, fr),
        annotate.scale_bar(base, fr),
        annotate.label(base, (3, 3), "hello"),
    ]
    for o in outs:
        assert o.dtype == np.uint8 and o.shape == base.shape and o.ndim == 3
    assert np.array_equal(base, snapshot)  # inputs are not modified


def test_line_drawn_between_lps_points(phantom_study):
    fr = slice_frame(phantom_study["sag_t2"], 4)
    img = _blank(fr.shape[0], fr.shape[1])
    out = annotate.line(img, fr, fr.from_px(20, 30), fr.from_px(100, 30), (255, 0, 0), show_length=False)
    rows, cols = np.nonzero((out == (255, 0, 0)).all(-1))
    assert cols.min() <= 21 and cols.max() >= 99
    assert np.all(np.abs(rows - 30) <= 2)


def test_scale_bar_length_matches_request(phantom_study):
    fr = slice_frame(phantom_study["sag_t2"], 4)
    img = _blank(fr.shape[0] + 40, fr.shape[1] + 40)
    out = annotate.scale_bar(img, fr, length_mm=10.0, color=(255, 255, 255))
    white = (out == 255).all(-1)
    # the bar is the longest run of white pixels in its row
    best = 0
    for row in white:
        run = best_run = 0
        for v in row:
            run = run + 1 if v else 0
            best_run = max(best_run, run)
        best = max(best, best_run)
    assert best == pytest.approx(10.0 / fr.px_x, abs=2)


def test_crosshair_centred_on_point(phantom_study):
    fr = slice_frame(phantom_study["ax_t2"], 3)
    img = _blank(fr.shape[0], fr.shape[1])
    p = fr.from_px(50.0, 60.0)
    out = annotate.crosshair(img, fr, p, (255, 220, 0))
    ys, xs = np.nonzero((out == (255, 220, 0)).all(-1))
    assert abs(xs.mean() - 50.0) < 1.0 and abs(ys.mean() - 60.0) < 1.0


def test_rulers_pad_the_image_and_are_centred(phantom_study):
    fr = slice_frame(phantom_study["ax_t2"], 3)
    img = _blank(fr.shape[0], fr.shape[1])
    out = annotate.add_rulers(img, fr)
    assert out.shape[0] > img.shape[0] and out.shape[1] > img.shape[1]


def test_pixel_labels_are_sparse(phantom_study):
    from spinescope.render import zoom_crop

    st = phantom_study["sag_t2"]
    centre = slice_frame(st, 6).centre
    img, _ = zoom_crop(st, centre, 60.0)  # auto upscale, so labels are small relative to the crop
    cyan = float(((img == annotate.CYAN).all(-1)).mean())
    assert 0 < cyan < 0.05  # sparse labels (dots and text) over a small fraction of the image
