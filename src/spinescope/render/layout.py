"""Composing rendered panels: title bands, horizontal / vertical stacks and grids.

All gaps and paddings are fractions of the label font height, so layouts scale with the image.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ..config import DEFAULT, Config
from .annotate import BLACK, WHITE, YELLOW
from .text import font
from .window import rgb_array

RGB = NDArray[np.uint8]


def gap_px(font_px: int) -> int:
    return max(2, round(font_px * 0.4))


MIN_TITLE_PX = 8  # smallest title font before lines are wrapped instead


def fit_font(lines: Sequence[str], width: int, font_px: int) -> int:
    """Largest font size <= ``font_px`` (and >= MIN_TITLE_PX) at which the longest line fits ``width``."""
    f = max(MIN_TITLE_PX, font_px)
    while f > MIN_TITLE_PX:
        longest = max((font(f).getlength(s) for s in lines), default=0.0)
        if longest <= width - 2 * gap_px(f):
            break
        f -= 1
    return f


def wrap_lines(lines: Sequence[str], width: int, font_px: int) -> list[str]:
    """Word-wrap each line so it fits ``width`` at ``font_px`` (a single word longer than that is kept)."""
    f = font(font_px)
    max_w = width - 2 * gap_px(font_px)
    out: list[str] = []
    for s in lines:
        if f.getlength(s) <= max_w:
            out.append(s)
            continue
        cur = ""
        for word in s.split(" "):
            cand = f"{cur} {word}".strip()
            if not cur or f.getlength(cand) <= max_w:
                cur = cand
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
    return out


def title_band(lines: Sequence[str], width: int, *, font_px: int, cfg: Config = DEFAULT) -> RGB:
    """Black band of text lines (first line white, the rest yellow), ``width`` px wide.

    The font is shrunk (never below 6 px) so that the longest line fits the width.
    """
    font_px = fit_font(lines, width, font_px)
    lines = wrap_lines(lines, width, font_px)
    line_h = round(font_px * 1.35)
    pad = gap_px(font_px)
    h = len(lines) * line_h + 2 * pad
    im = Image.new("RGB", (max(1, width), max(1, h)), BLACK)
    d = ImageDraw.Draw(im)
    f = font(font_px)
    for i, s in enumerate(lines):
        d.text((pad, pad + i * line_h), s, font=f, fill=WHITE if i == 0 else YELLOW)
    return np.asarray(im, dtype=np.uint8)


def pad_to(img: RGB, width: int, height: int, *, align: str = "left") -> RGB:
    """Pad ``img`` with black to (height, width); ``align`` is 'left', 'centre' or 'right' horizontally."""
    h, w = img.shape[:2]
    out = np.zeros((max(height, h), max(width, w), 3), dtype=np.uint8)
    if align == "centre":
        x0 = (out.shape[1] - w) // 2
    elif align == "right":
        x0 = out.shape[1] - w
    else:
        x0 = 0
    out[:h, x0 : x0 + w] = img
    return out


def vstack(panels: Sequence[RGB], *, gap: int = 0, align: str = "left") -> RGB:
    """Stack panels top to bottom with a black gap; narrower panels are padded."""
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    width = max(p.shape[1] for p in panels)
    height = sum(p.shape[0] for p in panels) + gap * (len(panels) - 1)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y = 0
    for p in panels:
        x0 = {"centre": (width - p.shape[1]) // 2, "right": width - p.shape[1]}.get(align, 0)
        out[y : y + p.shape[0], x0 : x0 + p.shape[1]] = p
        y += p.shape[0] + gap
    return out


def hstack(panels: Sequence[RGB], *, gap: int = 0, align: str = "top") -> RGB:
    """Place panels left to right with a black gap; shorter panels are padded at the top or centre."""
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    height = max(p.shape[0] for p in panels)
    width = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x = 0
    for p in panels:
        y0 = (height - p.shape[0]) // 2 if align == "centre" else 0
        out[y0 : y0 + p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1] + gap
    return out


def grid(tiles: Sequence[RGB], cols: int, *, gap: int = 0, cell: tuple[int, int] | None = None) -> RGB:
    """Tiles in row-major order, ``cols`` per row, each in a cell (default: the largest tile's size)."""
    if not tiles:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    cols = max(1, min(cols, len(tiles)))
    rows = -(-len(tiles) // cols)
    ch, cw = cell or (max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles))
    out = np.zeros((rows * ch + gap * (rows - 1), cols * cw + gap * (cols - 1), 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y0, x0 = r * (ch + gap), c * (cw + gap)
        a = rgb_array(t)[:ch, :cw]
        out[y0 : y0 + a.shape[0], x0 : x0 + a.shape[1]] = a
    return out


def title_px(shape: tuple[int, ...], cfg: Config = DEFAULT) -> int:
    """Title font height for an image of ``shape`` (a fraction of its shorter side)."""
    return max(6, round(cfg.title_frac * min(shape[0], shape[1])))


def with_title(img: RGB, lines: Sequence[str], *, cfg: Config = DEFAULT) -> RGB:
    """Prepend a title band (sized from the image) to ``img``."""
    return vstack([title_band(lines, img.shape[1], font_px=title_px(img.shape, cfg), cfg=cfg), img])


def fit_max_side(img: RGB, max_side: int) -> RGB:
    """Shrink ``img`` (keeping aspect) only if its longest side exceeds ``max_side``."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    f = max_side / max(h, w)
    nh, nw = max(1, round(h * f)), max(1, round(w * f))
    return np.asarray(Image.fromarray(img).resize((nw, nh), Image.Resampling.LANCZOS), dtype=np.uint8)
