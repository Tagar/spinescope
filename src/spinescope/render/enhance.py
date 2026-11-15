"""Display and analysis enhancement: denoising, surface-coil bias correction, normalised T2/T1 ratio and
STIR-on-T1 fusion.

.. warning::
   Enhancement can create or remove structure. Smoothing can blur a fine edge into a plausible one, a bias
   field can leak anatomy, and a fused overlay tints wherever a threshold is crossed. Every enhanced image
   produced here carries a banner, and every function logs the warning once. Compare with the raw image
   (:func:`compare_panel` shows raw beside each step) before relying on any enhanced detail. Nothing here
   is a diagnosis.

Rule: every transform is a smoothing, a monotone remap, or a colour overlay of a measured threshold.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from skimage import restoration

from ..config import DEFAULT, Config
from ..geometry import Stack, unit
from .annotate import orientation_letters
from .frame import Frame, display_scale, min_display_px, slice_frame
from .layout import hstack, with_title
from .mpr import resample_onto, sample_points
from .text import label_px
from .window import Window, contrast_of, grey_rgb, resolve_window, series_window

log = logging.getLogger(__name__)

RGB = NDArray[np.uint8]

#: One-line warning shown on every enhanced image and logged once per process.
ENHANCEMENT_WARNING = (
    "ENHANCED: smoothing, bias correction, ratio and fusion can create or remove structure. "
    "Check against the raw image. Not for diagnosis."
)

#: Control points of a perceptual dark-to-bright colour map (magma-like), for ratio images.
_HEAT = np.array(
    [
        [0.0, 0, 0, 4],
        [0.25, 80, 18, 123],
        [0.5, 182, 54, 121],
        [0.75, 251, 136, 97],
        [1.0, 252, 253, 191],
    ]
)

_warned = False


def _warn_once() -> None:
    global _warned
    if not _warned:
        log.warning(ENHANCEMENT_WARNING)
        _warned = True


def _banner(img: RGB, extra: list[str] | None = None, cfg: Config = DEFAULT) -> RGB:
    """Image with the enhancement warning as its title band (plus optional extra lines)."""
    return with_title(img, [ENHANCEMENT_WARNING, *(extra or [])], cfg=cfg)


def _rel_px(shape: tuple[int, ...], frac: float) -> float:
    """A smoothing scale in pixels that is a fraction of the image side (resolution-independent)."""
    return max(0.5, frac * min(shape[0], shape[1]))


def background_mask(img: ArrayLike, cfg: Config = DEFAULT) -> NDArray[np.bool_]:
    """Air: smoothed voxels below ``cfg.air_frac`` of the 99.5th percentile, connected to the image border."""
    x = np.asarray(img, dtype=np.float64)
    pos = x[x > 0]
    ref = float(np.percentile(pos, 99.5)) if pos.size else 0.0
    low = ndi.gaussian_filter(x, _rel_px(x.shape, cfg.smooth_frac)) < cfg.air_frac * ref
    lab, _ = ndi.label(low)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge > 0]
    return np.isin(lab, edge)


def body_mask(img: ArrayLike, cfg: Config = DEFAULT) -> NDArray[np.bool_]:
    """Tissue: everything that is not border-connected air, holes filled, largest blob kept."""
    body = ndi.binary_fill_holes(~background_mask(img, cfg))
    body = ndi.binary_opening(body, iterations=max(1, round(_rel_px(np.shape(img), cfg.smooth_frac) * 2)))
    lab, n = ndi.label(body)
    if n == 0:
        return body
    sizes = ndi.sum(body, lab, range(1, n + 1))
    return np.asarray(lab == (int(np.argmax(sizes)) + 1))


def noise_sigma(img: ArrayLike, cfg: Config = DEFAULT) -> float:
    """Noise sigma from air: MAD of differences between neighbouring non-zero air voxels.

    Approximate. Zero-clipped air breaks the Rayleigh model, so RMS and Laplacian estimates can disagree
    by a large factor. Falls back to the whole image when there is too little air.
    """
    x = np.asarray(img, dtype=np.float64)
    air = background_mask(x, cfg)
    dh = np.diff(x, axis=1)
    mh = air[:, 1:] & air[:, :-1] & (x[:, 1:] > 0) & (x[:, :-1] > 0)
    dv = np.diff(x, axis=0)
    mv = air[1:] & air[:-1] & (x[1:] > 0) & (x[:-1] > 0)
    d = np.concatenate([dh[mh], dv[mv]])
    if d.size < cfg.min_foreground_frac * x.size:
        d = np.concatenate([dh.ravel(), dv.ravel()])
    if d.size == 0:
        return 0.0
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2))


def denoise(
    img: ArrayLike, method: str = "nlm", sigma: float | None = None, cfg: Config = DEFAULT
) -> NDArray[np.float32]:
    """Edge-preserving denoising scaled to the air-estimated noise sigma. Output is in the input's units.

    ``method``: 'nlm' (non-local means), 'bilateral' or 'tv' (Chambolle total variation). The filters assume
    Gaussian noise; magnitude MR noise is Rician, so they are approximate where SNR is low.
    """
    _warn_once()
    x = np.asarray(img, dtype=np.float64)
    s = float(np.percentile(x, 99.5)) or 1.0
    y = x / s
    sn = max((noise_sigma(x, cfg) if sigma is None else sigma) / s, 1e-4)
    if method == "nlm":
        out = restoration.denoise_nl_means(
            y,
            h=cfg.nlm_h_per_sigma * sn,
            sigma=sn,
            patch_size=cfg.nlm_patch_size,
            patch_distance=cfg.nlm_patch_distance,
            fast_mode=False,
            preserve_range=True,
        )
    elif method == "bilateral":
        out = restoration.denoise_bilateral(
            y,
            sigma_color=cfg.bilateral_sigma_color * sn,
            sigma_spatial=_rel_px(x.shape, cfg.smooth_frac),
            channel_axis=None,
        )
    elif method == "tv":
        out = restoration.denoise_tv_chambolle(y, weight=sn, channel_axis=None)
    else:
        raise ValueError(f"unknown denoising method {method!r}")
    return np.asarray(out * s, dtype=np.float32)


def bias_field(img: ArrayLike, mask: ArrayLike | None = None, cfg: Config = DEFAULT) -> NDArray[np.float64]:
    """Smooth multiplicative field from normalised convolution inside tissue (median 1 there).

    A surface-coil sensitivity proxy. Anatomy leaks into a heavy smoothing, so treat it as approximate.
    """
    x = np.asarray(img, dtype=np.float64)
    m = body_mask(x, cfg) if mask is None else np.asarray(mask, dtype=bool)
    sigma = cfg.bias_sigma_frac * min(x.shape)
    w = m.astype(np.float64)
    num = ndi.gaussian_filter(np.where(m, x, 0.0), sigma, mode="constant")
    den = ndi.gaussian_filter(w, sigma, mode="constant")
    f = num / np.maximum(den, 1e-6)
    med = float(np.median(f[m])) if m.any() else 1.0
    f = f / (med or 1.0)
    return np.clip(f, cfg.bias_clip[0], cfg.bias_clip[1])


def bias_correct(img: ArrayLike, mask: ArrayLike | None = None, cfg: Config = DEFAULT) -> NDArray[np.float32]:
    """Divide by the smooth coil field inside tissue; air is left unchanged."""
    _warn_once()
    x = np.asarray(img, dtype=np.float32)
    m = body_mask(x, cfg) if mask is None else np.asarray(mask, dtype=bool)
    out = np.asarray(x / bias_field(x, m, cfg).astype(np.float32), dtype=np.float32)
    out[~m] = x[~m]
    return out


def ratio_from_medians(
    t2: ArrayLike, t1: ArrayLike, med_t2: float, med_t1: float, cfg: Config = DEFAULT
) -> NDArray[np.float64]:
    """Normalised T2/T1: (T2 / med_t2) / (T1 / med_t1). NaN where normalised T1 < ``cfg.ratio_min_t1``.

    Fluid and oedema read high (bright on T2, dark on T1). ``1`` means reference-marrow-like. The NaN rule
    stops noise being amplified where T1 is weak (air, low-signal tissue).
    """
    n2 = np.asarray(t2, dtype=np.float64) / med_t2
    n1 = np.asarray(t1, dtype=np.float64) / med_t1
    out = n2 / np.maximum(n1, cfg.ratio_min_t1)
    out[n1 < cfg.ratio_min_t1] = np.nan
    return out


def ratio_map(
    t2: ArrayLike,
    t1: ArrayLike,
    ref_t2: ArrayLike,
    ref_t1: ArrayLike,
    cfg: Config = DEFAULT,
) -> NDArray[np.float64]:
    """Normalised T2/T1 on one grid, with reference ROI masks (e.g. marrow) on that grid."""
    _warn_once()
    return ratio_from_medians(
        t2,
        t1,
        float(np.median(np.asarray(t2)[np.asarray(ref_t2, bool)])),
        float(np.median(np.asarray(t1)[np.asarray(ref_t1, bool)])),
        cfg,
    )


def _frame_points(frame: Frame) -> NDArray[np.float64]:
    h, w = frame.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.asarray(
        frame.origin + (x * frame.px_x)[..., None] * frame.u + (y * frame.px_y)[..., None] * frame.v
    )


def _reference_medians(
    t2: Stack, t1: Stack, centre: NDArray[np.float64], radius: float
) -> tuple[float, float] | None:
    """Medians of T2 and of T1 (sampled at the same points) inside a sphere around ``centre``."""
    a: list[NDArray[np.float64]] = []
    b: list[NDArray[np.float64]] = []
    n = unit(t2.normal)
    reach = radius + 0.5 * (t2.thickness or t2.spacing or 0.0)
    for k in range(t2.n):
        if abs(float((centre - t2.ipp[k]) @ n)) > reach:
            continue
        pts = _frame_points(slice_frame(t2, k))
        m = np.linalg.norm(pts - centre, axis=-1) <= radius
        if not m.any():
            continue
        a.append(t2.vol[k][m].astype(np.float64))
        b.append(sample_points(t1, pts[m]).astype(np.float64))
    if not a:
        return None
    va, vb = np.concatenate(a), np.concatenate(b)
    vb = vb[np.isfinite(vb)]
    if va.size == 0 or vb.size == 0:
        return None
    return float(np.median(va)), float(np.median(vb))


def t2_t1_ratio(
    t2: Stack,
    k: int,
    t1: Stack,
    anchors: Any = None,
    cfg: Config = DEFAULT,
) -> tuple[NDArray[np.float64], Frame] | None:
    """Normalised T2/T1 on slice ``k`` of ``t2``, with T1 resampled onto the T2 grid.

    The reference is a sphere (``cfg.ratio_ref_radius_mm``) in the vertebral body nearest the middle of the
    T2 stack, from ``anchors``. Without anchors the reference is the whole-stack foreground median (less
    specific). Returns None with a warning if T1 or T2 are missing.
    """
    _warn_once()
    if t2 is None or t1 is None:
        log.warning("T2/T1 ratio needs both a T2 and a T1 stack; skipped")
        return None
    med: tuple[float, float] | None = None
    if anchors is not None and getattr(anchors, "vertebrae", None):
        mid = t2.corners(t2.n // 2).mean(0)
        lab = min(
            anchors.vertebrae,
            key=lambda v: float(np.linalg.norm(np.asarray(anchors.vertebrae[v].center) - mid)),
        )
        med = _reference_medians(
            t2, t1, np.asarray(anchors.vertebrae[lab].center, dtype=np.float64), cfg.ratio_ref_radius_mm
        )
    if med is None:
        log.warning("%s: no vertebral reference for the T2/T1 ratio; using whole-stack medians", t2.key)
        med = (
            float(np.median(t2.vol[np.isfinite(t2.vol) & (t2.vol > 0)])),
            float(np.median(t1.vol[np.isfinite(t1.vol) & (t1.vol > 0)])),
        )
    t1_on = resample_onto(t1, t2, k).astype(np.float64)
    ratio = ratio_from_medians(t2.vol[k], t1_on, med[0], med[1], cfg)
    return ratio, slice_frame(t2, k)


def _heat(values: NDArray[np.float64], vmin: float, vmax: float) -> RGB:
    """Map values through the colour map; NaN is black."""
    t = np.clip((values - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    out = np.zeros((*values.shape, 3), dtype=np.float64)
    for ch in range(3):
        out[..., ch] = np.interp(t, _HEAT[:, 0], _HEAT[:, 1 + ch])
    out[~np.isfinite(values)] = 0
    return out.astype(np.uint8)


def ratio_image(t2: Stack, k: int, t1: Stack, anchors: Any = None, cfg: Config = DEFAULT) -> RGB | None:
    """Colour image of the normalised T2/T1 ratio on slice ``k`` (black where not shown), with banner."""
    res = t2_t1_ratio(t2, k, t1, anchors, cfg)
    if res is None:
        return None
    ratio, frame = res
    img, frame = display_scale(_heat(ratio, 0.0, cfg.ratio_display_max), frame, min_display_px(cfg))
    img = orientation_letters(img, frame, cfg=cfg)
    return _banner(
        img, [f"{t2.key} k={k:02d}  normalised T2/T1 (1 = reference marrow-like; black = T1 too weak)"], cfg
    )


def fusion(
    base: ArrayLike,
    overlay: ArrayLike,
    base_window: Window,
    overlay_window: Window,
    *,
    cfg: Config = DEFAULT,
    colour: tuple[float, float, float] = (1.0, 0.15, 0.15),
    alpha: float = 0.6,
) -> RGB:
    """Greyscale base with overlay-bright tissue tinted where the normalised overlay >= ``cfg.fusion_threshold``.

    Both arrays must be on the same grid (use :func:`spinescope.render.mpr.resample_onto`). The tint is
    limited to tissue of the base image, and the overlay is lightly smoothed to suppress speckle.
    """
    _warn_once()
    b = np.nan_to_num(np.asarray(base, dtype=np.float64))
    s = np.nan_to_num(np.asarray(overlay, dtype=np.float64))
    g = np.clip((b - base_window[0]) / max(base_window[1] - base_window[0], 1e-6), 0, 1)
    o = np.clip((s - overlay_window[0]) / max(overlay_window[1] - overlay_window[0], 1e-6), 0, 1)
    o = ndi.gaussian_filter(o, _rel_px(o.shape, cfg.smooth_frac))
    m = (o >= cfg.fusion_threshold) & body_mask(b, cfg)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.asarray(colour, dtype=np.float64)
    return np.asarray(np.clip(rgb * 255, 0, 255), dtype=np.uint8)


def fusion_slice(t1: Stack, k: int, stir: Stack, *, cfg: Config = DEFAULT) -> RGB | None:
    """STIR tinted on the T1 image of slice ``k`` (STIR resampled onto the T1 grid), with banner."""
    if t1 is None or stir is None:
        log.warning("STIR-on-T1 fusion needs both a T1 and a STIR stack; skipped")
        return None
    overlay = resample_onto(stir, t1, k)
    base_w = resolve_window(t1, "global", cfg)
    over_w = resolve_window(stir, "edema" if contrast_of(stir) == "stir" else "global", cfg)
    rgb = fusion(t1.vol[k], overlay, base_w, over_w, cfg=cfg)
    rgb, frame = display_scale(rgb, slice_frame(t1, k), min_display_px(cfg))
    img = orientation_letters(rgb, frame, cfg=cfg)
    return _banner(
        img,
        [f"{t1.key} k={k:02d} with {stir.key} tinted (threshold {cfg.fusion_threshold:g} of its window)"],
        cfg,
    )


def compare_panel(stack: Stack, k: int, *, window: Window | str | None = None, cfg: Config = DEFAULT) -> RGB:
    """Raw | denoised (NLM) | bias-corrected for slice ``k``, all in the raw window, with the banner."""
    _warn_once()
    raw = np.asarray(stack.vol[k], dtype=np.float32)
    win = resolve_window(stack, window, cfg) if window is not None else series_window(stack, cfg)
    steps = [
        ("raw", raw),
        ("denoised (NLM, sigma from air)", denoise(raw, "nlm", cfg=cfg)),
        ("bias-corrected (smooth coil field)", bias_correct(raw, cfg=cfg)),
    ]
    native = slice_frame(stack, k)
    panels = []
    for title, img in steps:
        scaled, frame = display_scale(grey_rgb(img, win), native, min_display_px(cfg))
        panels.append(with_title(orientation_letters(scaled, frame, cfg=cfg), [title], cfg=cfg))
    sig = noise_sigma(raw, cfg)
    gap = max(2, round(label_px(raw.shape, cfg) * 0.4))
    body = hstack(panels, gap=gap)
    return _banner(
        body,
        [f"{stack.key} k={k:02d}  identical window {win[0]:.0f}-{win[1]:.0f}  air noise sigma ~{sig:.1f}"],
        cfg,
    )


__all__ = [
    "ENHANCEMENT_WARNING",
    "background_mask",
    "bias_correct",
    "bias_field",
    "body_mask",
    "compare_panel",
    "denoise",
    "fusion",
    "fusion_slice",
    "noise_sigma",
    "ratio_from_medians",
    "ratio_image",
    "ratio_map",
    "t2_t1_ratio",
]
