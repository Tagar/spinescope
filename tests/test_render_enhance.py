from __future__ import annotations

import logging

import numpy as np
import pytest

from spinescope.anatomy import Anchors
from spinescope.render import enhance
from spinescope.render.enhance import (
    ENHANCEMENT_WARNING,
    background_mask,
    bias_correct,
    body_mask,
    compare_panel,
    denoise,
    fusion,
    fusion_slice,
    noise_sigma,
    ratio_image,
    t2_t1_ratio,
)
from spinescope.testing.phantom import PhantomSpec, SpinePhantom


@pytest.fixture(autouse=True)
def _reset_warning():
    enhance._warned = False
    yield


def _block_phantom(shape=(96, 96), sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    disc = ((yy - shape[0] / 2) / (shape[0] * 0.35)) ** 2 + ((xx - shape[1] / 2) / (shape[1] * 0.4)) ** 2 <= 1
    img = np.where(disc, 500.0, 0.0)
    if sigma:
        img = img + np.where(disc, rng.normal(0, sigma, shape), 0.0)
    return img, disc


def test_background_and_body_masks():
    img, disc = _block_phantom()
    air = background_mask(img)
    assert air[0, 0] and not air[48, 48]
    body = body_mask(img)
    assert body[48, 48] and not body[0, 0]
    assert (body == disc).mean() > 0.95


def test_noise_sigma_is_the_right_order_of_magnitude():
    rng = np.random.default_rng(3)
    sigma = 5.0
    img = np.abs(rng.normal(0, sigma, (128, 128)))  # magnitude noise in air, tissue absent
    img[40:90, 40:90] = 600.0
    est = noise_sigma(img)
    assert 0.3 * sigma < est < 2.5 * sigma


def test_denoise_reduces_noise_inside_tissue():
    img, disc = _block_phantom(sigma=25.0, seed=4)
    before = float(np.std(img[disc]))
    out = denoise(img, "nlm", sigma=25.0)
    assert out.dtype == np.float32 and out.shape == img.shape
    assert float(np.std(out[disc])) < 0.7 * before
    for method in ("bilateral", "tv"):
        assert float(np.std(denoise(img, method, sigma=25.0)[disc])) < before
    with pytest.raises(ValueError):
        denoise(img, "median")


def test_bias_correction_flattens_a_coil_gradient():
    shape = (96, 96)
    _, disc = _block_phantom(shape)
    gradient = 0.6 + 0.8 * np.linspace(0, 1, shape[1])[None, :] * np.ones((shape[0], 1))
    img = np.where(disc, 500.0 * gradient, 0.0)
    raw_cv = np.std(img[disc]) / np.mean(img[disc])
    out = bias_correct(img)
    cv = np.std(out[disc]) / np.mean(out[disc])
    assert cv < 0.6 * raw_cv
    assert np.array_equal(out[~disc], img[~disc])  # air untouched


def test_fusion_tints_only_bright_overlay():
    base = np.full((32, 32), 500.0)
    overlay = np.zeros((32, 32))
    overlay[10:20, 10:20] = 1000.0
    rgb = fusion(base, overlay, (0.0, 1000.0), (0.0, 1000.0))
    red = rgb[15, 15].astype(int)
    grey = rgb[2, 2].astype(int)
    assert red[0] > red[1] + 40  # tinted
    assert abs(grey[0] - grey[1]) < 2  # untinted grey


def test_t2_t1_ratio_is_one_in_marrow_and_nan_in_weak_t1(phantom):
    clean = SpinePhantom(PhantomSpec(noise=0.0))
    lumbar_mid = (clean.discs["L1-L2"]["center"] + clean.discs["L5-S1"]["center"]) / 2
    kw = dict(
        n=36,
        spacing=5.0,
        thickness=4.0,
        rows=128,
        cols=128,
        px=1.4,
        center=lumbar_mid + np.array([0, 10.0, 0]),
        key="ax",
    )
    t2 = clean.render("ax", "t2", **kw)
    t1 = clean.render("ax", "t1", **kw)
    anchors = Anchors.from_truth(clean.truth)
    res = t2_t1_ratio(t2, 20, t1, anchors)
    assert res is not None
    ratio, frame = res
    body = np.asarray(clean.bodies["L3"]["center"])
    x, y, _ = frame.to_px(body)
    assert ratio[round(float(y)), round(float(x))] == pytest.approx(1.0, abs=1e-3)
    assert np.isnan(ratio).any()  # CSF is weak on T1, so the ratio is withheld there


def test_t2_t1_ratio_without_t1_is_none(phantom_study, caplog):
    with caplog.at_level(logging.WARNING):
        assert t2_t1_ratio(phantom_study["ax_t2"], 3, None) is None
    assert "needs both" in caplog.text


def test_fusion_slice_and_ratio_image_are_banner_marked(phantom_study):
    img = fusion_slice(phantom_study["sag_t1"], 7, phantom_study["sag_stir"])
    assert img is not None and img.shape[2] == 3
    assert fusion_slice(None, 7, phantom_study["sag_stir"]) is None
    rimg = ratio_image(phantom_study["sag_t2"], 7, phantom_study["sag_t1"])
    assert rimg is not None and rimg.shape[2] == 3


def test_compare_panel_shape(phantom_study):
    out = compare_panel(phantom_study["sag_t2"], 7)
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_enhancement_warning_logged_once(caplog):
    img, _ = _block_phantom(sigma=10.0)
    with caplog.at_level(logging.WARNING, logger="spinescope.render.enhance"):
        denoise(img, sigma=10.0)
        denoise(img, sigma=10.0)
        bias_correct(img)
    hits = [r for r in caplog.records if ENHANCEMENT_WARNING in r.getMessage()]
    assert len(hits) == 1
