"""Contact sheets, co-registered triptychs and zoom crops with rulers and pixel coordinates.

* :func:`contact_sheet` tiles every slice of a stack, labelled with its index and position in mm, and
  the nearest disc when anchors are given. Sheets split automatically so no PNG side exceeds
  ``config.max_png_side_px``.
* :func:`triptych` shows co-registered stacks (same slice positions) side by side at one slice.
* :func:`zoom_crop` shows a mm-sized crop of a slice, with mm rulers and sparse native ``row,col``
  labels, so a reader can pick pixel coordinates and then measure. Any upscaling factor.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..geometry import Stack, sample_plane, unit
from .annotate import YELLOW, Canvas, add_rulers, crosshair, orientation_letters, pixel_labels, scale_bar
from .frame import Frame, auto_upscale, min_display_px, slice_frame
from .layout import gap_px, grid, hstack, title_band, vstack, with_title
from .mpr import resample_onto
from .text import label_px
from .window import Window, grey_rgb, resolve_window

log = logging.getLogger(__name__)

RGB = NDArray[np.uint8]

#: A contact-sheet tile is at most 1/TILE_DIVISOR of the PNG side limit (keeps tiles legible).
TILE_DIVISOR = 8
#: Position of the 'mid' side label or a co-registration warning is judged within this fraction of a slice.
SIDE_TOL_SLICES = 0.5


def slice_label_lines(stack: Stack, k: int, anchors: Anchors | None = None) -> list[str]:
    """Label text for slice ``k``: its index and position, plus side (sagittal) or nearest disc (axial)."""
    ctr = slice_frame(stack, k).centre
    head = f"k={k:02d}"
    if stack.plane == "sag":
        x = float(ctr[0])
        side = "L" if x > 0 else "R"
        if anchors is not None and anchors.midline_x is not None:
            dx = x - anchors.midline_x
            tol = SIDE_TOL_SLICES * (stack.spacing or stack.thickness or 1.0)
            side = "mid" if abs(dx) <= tol else ("L" if dx > 0 else "R")
        lines = [f"{head} x={x:+.1f}mm {side}"]
        if anchors is not None and anchors.discs:
            lines.append(_disc_summary(_discs_in_slab(stack, k, anchors)))
        return lines
    if stack.plane == "cor":
        y = float(ctr[1])
        return [f"{head} y={y:+.1f}mm {'P' if y > 0 else 'A'}"]
    lines = [f"{head} z={float(ctr[2]):+.1f}mm"]
    if anchors is not None and anchors.discs:
        lab, s = anchors.nearest_disc(ctr)
        lines.append(f"{lab} {s:+.1f}mm")
    return lines


def _disc_summary(labels: list[str]) -> str:
    """'discs: none', the full list for up to three discs, or 'N discs: first .. last' (fits a tile)."""
    if not labels:
        return "discs: none"
    if len(labels) <= 3:
        return "discs: " + ", ".join(labels)
    return f"{len(labels)} discs: {labels[0]} .. {labels[-1]}"


def _discs_in_slab(stack: Stack, k: int, anchors: Anchors) -> list[str]:
    """Discs whose centre lies in the slab of slice ``k`` (within half a slice thickness of its plane)."""
    n = unit(stack.normal)
    half = 0.5 * (stack.thickness or stack.spacing or 0.0)
    out = []
    for lab in anchors.disc_labels():
        d = anchors.discs[lab]
        off = float((np.asarray(d.center) - stack.ipp[k]) @ n)
        if abs(off) <= half:
            out.append(lab)
    return out


def _resize(img: RGB, h: int, w: int) -> RGB:
    im = Image.fromarray(img).resize((max(1, w), max(1, h)), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def _fit_square(gray: RGB, tile: int) -> RGB:
    """Fit an RGB image inside a ``tile`` x ``tile`` square (aspect kept), centred on black."""
    h, w = gray.shape[:2]
    f = tile / max(h, w)
    nh, nw = max(1, round(h * f)), max(1, round(w * f))
    out = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0, x0 = (tile - nh) // 2, (tile - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = _resize(gray, nh, nw)
    return out


def _label_block(lines: Sequence[str], width: int, font_px: int) -> RGB:
    """Plain yellow text lines on black (no legibility box), used under contact-sheet tiles."""
    line_h = round(font_px * 1.3)
    pad = gap_px(font_px)
    c = Canvas(np.zeros((len(lines) * line_h + 2 * pad, max(1, width), 3), dtype=np.uint8))
    for i, s in enumerate(lines):
        c.text((pad, pad + i * line_h), s, YELLOW, font_px, bg=False)
    return c.array()


def _tile(stack: Stack, k: int, window: Window, tile: int, anchors: Anchors | None, cfg: Config) -> RGB:
    """One contact-sheet tile: slice k in the window, orientation letters, labels under it."""
    img = orientation_letters(
        _fit_square(grey_rgb(stack.vol[k], window), tile), slice_frame(stack, k), cfg=cfg
    )
    fp = max(6, round(label_px((tile, tile), cfg) * 0.9))
    body = vstack([img, _label_block(slice_label_lines(stack, k, anchors), tile, fp)])
    body[0, :, :] = 90  # thin grey border so tiles stay apart on a black sheet
    body[:, 0, :] = 90
    return body


def _sheet_geometry(n: int, cfg: Config) -> tuple[int, int, int]:
    """(tile px, columns, gap px) for ``n`` tiles, so that one sheet fits the PNG side limit."""
    side = cfg.max_png_side_px
    cols = max(1, math.ceil(math.sqrt(n)))
    gap = max(2, round(side / 400))
    tile = min(side // TILE_DIVISOR, (side - gap * (cols + 1)) // cols)
    return max(16, tile), cols, gap


def contact_sheet(
    stack: Stack,
    *,
    anchors: Anchors | None = None,
    window: Window | str | None = None,
    config: Config | None = None,
) -> list[RGB]:
    """Labelled grid of every slice of ``stack``. Returns one or more sheets, each <= the PNG side limit.

    Each tile shows slice k in the window (default: series-wide, so every slice is comparable), its
    orientation letters and position: sagittal x (mm, side), coronal y, axial z with the nearest disc
    (signed mm, + superior to the disc) when ``anchors`` is given.
    """
    cfg = config or DEFAULT
    win = resolve_window(stack, window, cfg)
    tile, cols, gap = _sheet_geometry(stack.n, cfg)
    fp = max(6, round(label_px((tile, tile), cfg) * 0.9))
    header_h = 2 * round(fp * 1.35) + 2 * gap_px(fp)
    label_h = max(len(slice_label_lines(stack, k, anchors)) for k in range(stack.n)) * round(fp * 1.3)
    cell_h = tile + label_h + 2 * gap_px(fp) + gap
    rows_per_sheet = max(1, (cfg.max_png_side_px - header_h - gap) // cell_h)
    per_sheet = cols * rows_per_sheet
    n_sheets = math.ceil(stack.n / per_sheet)
    sheets: list[RGB] = []
    for s in range(n_sheets):
        ks = list(range(s * per_sheet, min(stack.n, (s + 1) * per_sheet)))
        tiles = [_tile(stack, k, win, tile, anchors, cfg) for k in ks]
        title = f"{stack.key}  n={stack.n}  {stack.plane} plane  window {win[0]:.0f}-{win[1]:.0f}" + (
            f"  sheet {s + 1}/{n_sheets}" if n_sheets > 1 else ""
        )
        sheets.append(with_title(grid(tiles, cols, gap=gap), [title], cfg=cfg))
    return sheets


def _panel(img: RGB, frame: Frame, lines: Sequence[str], cfg: Config) -> RGB:
    """A triptych panel: edge letters and a title band."""
    img = orientation_letters(img, frame, cfg=cfg)
    return vstack([title_band(lines, img.shape[1], font_px=label_px(img.shape, cfg), cfg=cfg), img])


def triptych(
    stacks: Sequence[Stack],
    k: int = 0,
    *,
    anchors: Anchors | None = None,
    windows: Sequence[Window | str | None] | None = None,
    config: Config | None = None,
) -> RGB:
    """Co-registered stacks side by side at one slice.

    Slice ``k`` of ``stacks[0]`` is the reference. Every other stack shows its native slice nearest to that
    position, or its values resampled onto the reference plane when its plane is not parallel. Each panel
    has its own window (default: series-wide). A warning is logged for stacks whose slice positions differ
    from the reference by more than half a slice.
    """
    cfg = config or DEFAULT
    if not stacks:
        raise ValueError("triptych needs at least one stack")
    wins = list(windows) if windows is not None else [None] * len(stacks)
    if len(wins) != len(stacks):
        raise ValueError("windows must match stacks one-to-one")
    ref = stacks[0]
    ref_frame = slice_frame(ref, k)
    panels: list[RGB] = []
    for st, w in zip(stacks, wins, strict=True):
        win = resolve_window(st, w, cfg)
        parallel = abs(float(np.dot(unit(st.normal), ref.normal))) > 1 - 1e-6
        if st is ref:
            kk, note = k, "reference"
            img, frame = grey_rgb(st.vol[k], win), ref_frame
        elif parallel:
            kk, dist = st.nearest_slice(ref_frame.centre)
            note = f"slice {kk}, {dist:+.1f} mm from reference"
            if abs(dist) > SIDE_TOL_SLICES * (ref.spacing or ref.thickness or 1.0):
                log.warning(
                    "%s has no slice within half a %s slice of %s slice %d (nearest is %.1f mm away); "
                    "the panel is not co-registered",
                    st.key,
                    ref.key,
                    ref.key,
                    k,
                    abs(dist),
                )
            img, frame = grey_rgb(st.vol[kk], win), slice_frame(st, kk)
        else:
            kk, note = -1, "resampled onto reference plane (different orientation)"
            img, frame = grey_rgb(resample_onto(st, ref, k), win), ref_frame
        lines = [f"{st.key}  {note}"]
        if kk >= 0:
            lines += slice_label_lines(st, kk, anchors)
        panels.append(_panel(img, frame, lines, cfg))
    target_h = cfg.max_png_side_px // 2
    scaled = [_resize(p, target_h, round(p.shape[1] * target_h / p.shape[0])) for p in panels]
    row = hstack(scaled, gap=max(2, round(cfg.max_png_side_px / 400)), align="top")
    if row.shape[1] > cfg.max_png_side_px:
        f = cfg.max_png_side_px / row.shape[1]
        row = _resize(row, round(row.shape[0] * f), cfg.max_png_side_px)
    return row


def _crop_size(size_mm: float | tuple[float, float] | None, default: float) -> tuple[float, float]:
    """(width, height) in mm from a side length, a (width, height) pair, or the default."""
    if size_mm is None:
        return default, default
    if isinstance(size_mm, tuple):
        return float(size_mm[0]), float(size_mm[1])
    return float(size_mm), float(size_mm)


def zoom_crop(
    stack: Stack,
    centre: Any,
    size_mm: float | tuple[float, float] | None = None,
    *,
    k: int | None = None,
    upscale: float | None = None,
    window: Window | str | None = None,
    anchors: Anchors | None = None,
    pixel_coords: bool = True,
    crosshair_on: bool = True,
    order: int = 1,
    config: Config | None = None,
) -> tuple[RGB, Frame]:
    """Zoom on an LPS point: a ``size_mm`` crop of slice ``k`` (default: nearest slice), in its plane.

    ``size_mm`` is a side or (width, height) in mm (default ``config.zoom_mm``). ``upscale`` multiplies the
    native resolution of the crop, by any factor; ``order=0`` gives nearest-neighbour pixels. The image
    has mm rulers (offsets from the crop centre, + right and + down), a scale bar and, when
    ``pixel_coords``, sparse native ``row,col`` labels of slice ``k``. Returns the RGB image and the crop's
    :class:`Frame`: ``frame.to_px`` maps an LPS point into the image and ``frame.from_px`` maps back.
    """
    cfg = config or DEFAULT
    if upscale is not None and upscale <= 0:
        raise ValueError("upscale must be > 0")
    c_xyz = np.asarray(centre, dtype=np.float64)
    if k is None:
        k, _ = stack.nearest_slice(c_xyz)
    w_mm, h_mm = _crop_size(size_mm, cfg.zoom_mm)
    nrm = stack.normal
    on_plane = c_xyz - ((c_xyz - stack.ipp[k]) @ nrm) * nrm
    native_px = min(stack.row_mm, stack.col_mm)
    up = upscale if upscale is not None else auto_upscale(native_px, (w_mm, h_mm), min_display_px(cfg))
    px = native_px / up
    win = resolve_window(stack, window, cfg)
    samp = sample_plane(
        stack, on_plane, stack.row_dir, stack.col_dir, size_mm=(h_mm, w_mm), px_mm=px, order=order
    )
    frame = Frame.centred(on_plane, stack.row_dir, stack.col_dir, samp.shape, px, px)
    img = grey_rgb(samp, win)
    if crosshair_on:
        img = crosshair(img, frame, on_plane, YELLOW, cfg=cfg)
    if pixel_coords:
        img = pixel_labels(img, frame, stack, k, cfg=cfg)
    img = scale_bar(img, frame, cfg=cfg)
    img = orientation_letters(img, frame, cfg=cfg)
    img = add_rulers(img, frame, cfg=cfg)
    lines = [
        f"{stack.key} k={k:02d}  centre LPS ({c_xyz[0]:+.1f}, {c_xyz[1]:+.1f}, {c_xyz[2]:+.1f}) mm",
        f"crop {w_mm:g} x {h_mm:g} mm at x{up:.3g} native; "
        + "; ".join(slice_label_lines(stack, k, anchors))
        + ("; labels = native row,col" if pixel_coords else ""),
    ]
    return with_title(img, lines, cfg=cfg), frame
