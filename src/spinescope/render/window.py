"""Display windows (grey-level mapping), 8-bit conversion and PNG output.

Windows are percentiles of *foreground* voxels. Exact zeros and anything below
``config.background_frac`` of the 99.5th percentile (zero-filled air, noise floor) are dropped before
the percentiles, so a zero-filled background does not set the window. A series window is computed once
per stack, so every slice of that series is shown with the same mapping. Contrast presets come from
``config.window_presets`` and are selected by ``stack.meta['role']['contrast']``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from ..config import DEFAULT, Config
from ..geometry import Stack

log = logging.getLogger(__name__)

Window = tuple[float, float]


def contrast_of(stack: Stack) -> str:
    """Contrast of a stack from its role ('t1', 't2', 'stir', ...), or 'unknown'."""
    role = stack.meta.get("role") or {}
    return str(role.get("contrast", "unknown"))


def foreground_values(vol: ArrayLike, cfg: Config = DEFAULT) -> NDArray[np.float32]:
    """Voxel values that are not background (zero fill, air, noise floor)."""
    v = np.asarray(vol, dtype=np.float32).ravel()
    v = v[np.isfinite(v)]
    pos = v[v > 0]
    if pos.size == 0:
        return np.zeros(1, dtype=np.float32)
    ref = float(np.percentile(pos, 99.5))
    fg = v[v > cfg.background_frac * ref]
    if fg.size < max(1.0, cfg.min_foreground_frac * v.size):
        fg = pos
    return fg


def _pct_window(values: NDArray[np.float32], pct: tuple[float, float]) -> Window:
    lo, hi = (float(x) for x in np.percentile(values, pct))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def series_window(source: Stack | ArrayLike, cfg: Config = DEFAULT) -> Window:
    """Series-wide window (``cfg.window_pct`` percentiles of foreground voxels)."""
    vol = source.vol if isinstance(source, Stack) else source
    return _pct_window(foreground_values(vol, cfg), cfg.window_pct)


def presets(stack: Stack, cfg: Config = DEFAULT) -> dict[str, Window]:
    """Named windows for this stack: ``global`` plus the contrast presets matching its role."""
    values = foreground_values(stack.vol, cfg)
    out = {"global": _pct_window(values, cfg.window_pct)}
    contrast = contrast_of(stack)
    for c, name, lo, hi in cfg.window_presets:
        if c == contrast:
            out[name] = _pct_window(values, (lo, hi))
    return out


def resolve_window(stack: Stack, window: Window | str | None = None, cfg: Config = DEFAULT) -> Window:
    """Turn ``None`` (series window), a preset name, or an explicit (lo, hi) into (lo, hi)."""
    if window is None:
        return series_window(stack, cfg)
    if isinstance(window, str):
        found = presets(stack, cfg)
        if window in found:
            return found[window]
        log.warning(
            "no %r preset for %s (%s); using the series window", window, stack.key, contrast_of(stack)
        )
        return series_window(stack, cfg)
    lo, hi = float(window[0]), float(window[1])
    if not hi > lo:
        raise ValueError(f"window needs hi > lo, got {(lo, hi)}")
    return lo, hi


def to_uint8(img: ArrayLike, window: Window) -> NDArray[np.uint8]:
    """Map a 2-D array onto 0-255 through ``window``; NaN and out-of-window values clip to 0 / 255."""
    a = np.asarray(img, dtype=np.float32)
    lo, hi = window
    g = (a - lo) / max(hi - lo, 1e-6) * 255.0
    g = np.where(np.isfinite(g), g, 0.0)
    return np.clip(g, 0, 255).astype(np.uint8)


def grey_rgb(img: ArrayLike, window: Window) -> NDArray[np.uint8]:
    """2-D array -> (H, W, 3) uint8 greyscale through ``window`` (NaN shown black)."""
    g = to_uint8(img, window)
    return np.repeat(g[..., None], 3, axis=-1)


def save_png(img: ArrayLike, path: str | os.PathLike[str]) -> str:
    """Write an (H, W, 3) or (H, W) uint8 image as PNG, creating the directory. Returns the path."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        raise TypeError(f"save_png needs uint8 pixels, got {arr.dtype}; convert with to_uint8/grey_rgb")
    if arr.ndim == 3 and arr.shape[2] not in (3, 4):
        raise ValueError(f"expected (H, W, 3) RGB, got {arr.shape}")
    p = os.fspath(path)
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    Image.fromarray(arr).save(p)
    return p


def rgb_array(obj: Any) -> NDArray[np.uint8]:
    """Coerce a PIL image or array to a contiguous (H, W, 3) uint8 array."""
    arr = np.asarray(obj.convert("RGB") if isinstance(obj, Image.Image) else obj)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3].astype(np.uint8))
