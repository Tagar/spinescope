from __future__ import annotations

import logging

import numpy as np
import pytest
from PIL import Image

from spinescope.geometry import Stack
from spinescope.render import grey_rgb, presets, resolve_window, save_png, series_window, to_uint8
from spinescope.render.window import foreground_values


def _stack(vol: np.ndarray, contrast: str | None = None) -> Stack:
    n = vol.shape[0]
    meta = {"role": {"contrast": contrast}} if contrast else {}
    return Stack(
        key="t",
        vol=vol,
        ipp=np.array([[0.0, 0.0, 5.0 * k] for k in range(n)]),
        row_dir=[1, 0, 0],
        col_dir=[0, 1, 0],
        row_mm=1.0,
        col_mm=1.0,
        meta=meta,
    )


def test_series_window_ignores_zero_fill_background():
    rng = np.random.default_rng(0)
    vol = np.zeros((4, 64, 64), dtype=np.float32)
    vol[:, 16:48, 16:48] = rng.uniform(100, 1000, (4, 32, 32))  # 75% zero-filled background
    lo, hi = series_window(vol)
    naive_lo = float(np.percentile(vol, 0.5))
    assert naive_lo == 0.0  # the naive window would be dragged down to zero
    assert lo > 90 and hi <= 1000 * 1.001
    tissue = vol[vol > 0]
    assert lo == pytest.approx(float(np.percentile(tissue, 0.5)), rel=0.02)


def test_series_window_all_zero_and_flat_are_safe():
    assert series_window(np.zeros((2, 8, 8))) == (0.0, 1.0)
    lo, hi = series_window(np.full((2, 8, 8), 7.0))
    assert hi > lo


def test_foreground_drops_zeros_and_noise_floor():
    v = np.array([0.0, 0.5, 2.0, 500.0, 1000.0, np.nan])
    fg = foreground_values(v)
    assert 0.0 not in fg and 0.5 not in fg and np.isfinite(fg).all()


def test_to_uint8_clips_and_maps_nan_to_black():
    img = np.array([[np.nan, -5.0, 50.0, 500.0]])
    out = to_uint8(img, (0.0, 100.0))
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 0, 127, 255]]


def test_grey_rgb_is_three_channel_uint8():
    out = grey_rgb(np.ones((5, 7)), (0.0, 2.0))
    assert out.shape == (5, 7, 3) and out.dtype == np.uint8
    assert np.all(out[..., 0] == out[..., 2])


def test_presets_follow_contrast_role():
    rng = np.random.default_rng(1)
    vol = rng.uniform(0, 1000, (3, 16, 16))
    t1 = presets(_stack(vol, "t1"))
    t2 = presets(_stack(vol, "t2"))
    stir = presets(_stack(vol, "stir"))
    assert {"global", "marrow"} <= set(t1)
    assert "disc/CSF" in t2 and "marrow" not in t2
    assert "edema" in stir
    assert t1["marrow"][0] > t1["global"][0]  # marrow preset starts higher in the intensity range


def test_resolve_window_variants(caplog):
    st = _stack(np.random.default_rng(2).uniform(0, 100, (2, 8, 8)), "t2")
    assert resolve_window(st) == series_window(st)
    assert resolve_window(st, (1.0, 2.0)) == (1.0, 2.0)
    with pytest.raises(ValueError):
        resolve_window(st, (2.0, 2.0))
    with caplog.at_level(logging.WARNING):
        assert resolve_window(st, "no-such-preset") == series_window(st)
    assert "no-such-preset" in caplog.text


def test_save_png_roundtrip(tmp_path):
    img = np.zeros((6, 9, 3), dtype=np.uint8)
    img[2, 3] = (255, 220, 0)
    p = save_png(img, tmp_path / "sub" / "a.png")
    back = np.asarray(Image.open(p).convert("RGB"))
    assert back.shape == img.shape and np.array_equal(back, img)


def test_save_png_rejects_float_input(tmp_path):
    with pytest.raises(TypeError):
        save_png(np.zeros((4, 4, 3)), tmp_path / "x.png")
