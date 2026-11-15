"""Annotations on rendered images: mm lines, arrows, circles, crosshairs, text, orientation letters,
scale bars, rulers and sparse native pixel-coordinate labels.

Every annotation takes a :class:`Frame` (the plane the image shows), so positions are given in patient
LPS mm and converted to pixels here. Each function returns a new RGB uint8 array; inputs are not
modified. Stroke widths, dot sizes, insets and font sizes are fractions of the image side.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw

from ..config import DEFAULT, Config
from ..geometry import Stack
from .frame import Frame, nice_step
from .text import font, label_px
from .window import rgb_array

RGB = NDArray[np.uint8]

YELLOW = (255, 220, 0)
WHITE = (255, 255, 255)
CYAN = (0, 220, 255)
RED = (255, 80, 80)
GREEN = (90, 255, 120)
BLACK = (0, 0, 0)
GREY = (110, 110, 110)
#: Categorical colours for series (cross-reference lines, linked panels); distinct on black.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 90, 90),
    (90, 200, 255),
    (255, 200, 60),
    (140, 255, 120),
    (230, 130, 255),
    (255, 150, 200),
    (120, 255, 220),
    (255, 255, 140),
)


def _stroke(shape: tuple[int, ...]) -> int:
    return max(1, round(0.004 * min(shape[0], shape[1])))


class Canvas:
    """A PIL RGB image with text helpers that draw a legibility box behind labels."""

    def __init__(self, img: ArrayLike) -> None:
        self.im = Image.fromarray(rgb_array(img))
        self.draw = ImageDraw.Draw(self.im)
        self.shape: tuple[int, int] = (self.im.height, self.im.width)

    def array(self) -> RGB:
        return np.array(self.im, dtype=np.uint8)

    def text(
        self,
        xy: tuple[float, float],
        s: str,
        color: tuple[int, int, int],
        size: int,
        anchor: str = "la",
        bg: bool = True,
    ) -> None:
        f = font(size)
        if bg:
            x0, y0, x1, y1 = self.draw.textbbox(xy, s, font=f, anchor=anchor)
            pad = max(1, size // 5)
            self.draw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=BLACK)
        self.draw.text(xy, s, font=f, fill=color, anchor=anchor)

    def line(self, p0: tuple[float, float], p1: tuple[float, float], color: Any, width: int) -> None:
        self.draw.line([p0, p1], fill=color, width=width)


def label(
    img: ArrayLike,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int] = YELLOW,
    anchor: str = "la",
    size: int | None = None,
) -> RGB:
    """Text at pixel position ``xy`` (anchor as in Pillow, e.g. 'la' = left-ascender)."""
    c = Canvas(img)
    c.text(xy, text, color, size or label_px(c.shape), anchor=anchor)
    return c.array()


def line(
    img: ArrayLike,
    frame: Frame,
    p0: ArrayLike,
    p1: ArrayLike,
    color: tuple[int, int, int] = YELLOW,
    *,
    show_length: bool = True,
    text: str | None = None,
    cfg: Config = DEFAULT,
) -> RGB:
    """Straight line between two LPS points, labelled with its length in mm (or ``text``)."""
    c = Canvas(img)
    a = frame.to_px(p0)[:2]
    b = frame.to_px(p1)[:2]
    c.line(a, b, color, _stroke(c.shape))
    if show_length or text:
        mm = float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float)))
        s = text if text is not None else f"{mm:.1f} mm"
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        c.text((mx + 4, my - 4), s, color, label_px(c.shape, cfg), anchor="lb")
    return c.array()


def arrow(
    img: ArrayLike,
    frame: Frame,
    tail: ArrayLike,
    head: ArrayLike,
    color: tuple[int, int, int] = YELLOW,
    *,
    text: str | None = None,
    cfg: Config = DEFAULT,
) -> RGB:
    """Arrow from ``tail`` to ``head`` (LPS points); head drawn as a filled triangle."""
    c = Canvas(img)
    a = np.asarray(frame.to_px(tail)[:2], float)
    b = np.asarray(frame.to_px(head)[:2], float)
    w = _stroke(c.shape)
    c.line(tuple(a), tuple(b), color, w)
    d = b - a
    n = float(np.hypot(*d))
    if n > 0:
        u = d / n
        perp = np.array([-u[1], u[0]])
        hl = min(0.05 * min(c.shape), 0.3 * n)
        base = b - u * hl
        pts = [tuple(b), tuple(base + perp * hl * 0.5), tuple(base - perp * hl * 0.5)]
        c.draw.polygon(pts, fill=color)
    if text:
        c.text((b[0] + 4, b[1] - 4), text, color, label_px(c.shape, cfg), anchor="lb")
    return c.array()


def circle(
    img: ArrayLike,
    frame: Frame,
    centre: ArrayLike,
    radius_mm: float,
    color: tuple[int, int, int] = YELLOW,
    *,
    text: str | None = None,
    cfg: Config = DEFAULT,
) -> RGB:
    """Circle of ``radius_mm`` on the frame plane around an LPS point."""
    c = Canvas(img)
    x, y, _ = frame.to_px(centre)
    rx, ry = radius_mm / frame.px_x, radius_mm / frame.px_y
    c.draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=color, width=_stroke(c.shape))
    if text:
        c.text((x + rx + 2, y), text, color, label_px(c.shape, cfg), anchor="lm")
    return c.array()


def crosshair(
    img: ArrayLike,
    frame: Frame,
    xyz: ArrayLike,
    color: tuple[int, int, int] = YELLOW,
    *,
    cfg: Config = DEFAULT,
) -> RGB:
    """Gapped crosshair with a small ring at an LPS point (projected onto the frame)."""
    c = Canvas(img)
    x, y, _ = frame.to_px(xyz)
    h, w = c.shape
    arm = 0.05 * min(h, w)
    gap = 0.01 * min(h, w)
    s = _stroke(c.shape)
    c.line((x - arm, y), (x - gap, y), color, s)
    c.line((x + gap, y), (x + arm, y), color, s)
    c.line((x, y - arm), (x, y - gap), color, s)
    c.line((x, y + gap), (x, y + arm), color, s)
    r = max(2.0, 2 * gap)
    c.draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=s)
    return c.array()


def text_at(
    img: ArrayLike,
    frame: Frame,
    xyz: ArrayLike,
    text: str,
    color: tuple[int, int, int] = YELLOW,
    *,
    anchor: str = "la",
    cfg: Config = DEFAULT,
) -> RGB:
    """Text anchored at an LPS point."""
    c = Canvas(img)
    x, y, _ = frame.to_px(xyz)
    c.text((x, y), text, color, label_px(c.shape, cfg), anchor=anchor)
    return c.array()


def orientation_letters(img: ArrayLike, frame: Frame, *, cfg: Config = DEFAULT) -> RGB:
    """R/L, A/P, S/I letters at the four edge midpoints, from the frame axes (see Frame.edge_letters)."""
    c = Canvas(img)
    h, w = c.shape
    size = label_px(c.shape, cfg)
    inset = size * 0.6
    letters = frame.edge_letters()
    c.text((w - inset, h / 2), letters["right"], YELLOW, size, anchor="rm")
    c.text((inset, h / 2), letters["left"], YELLOW, size, anchor="lm")
    c.text((w / 2, inset), letters["top"], YELLOW, size, anchor="ma")
    c.text((w / 2, h - inset), letters["bottom"], YELLOW, size, anchor="md")
    return c.array()


def scale_bar(
    img: ArrayLike,
    frame: Frame,
    length_mm: float | None = None,
    color: tuple[int, int, int] = WHITE,
    *,
    cfg: Config = DEFAULT,
) -> RGB:
    """Scale bar at the bottom left, about a fifth of the image width unless ``length_mm`` is given."""
    c = Canvas(img)
    h, w = c.shape
    width_mm, _ = frame.extent_mm
    if length_mm is None:
        length_mm = nice_step(0.2 * width_mm)
    length_px = length_mm / frame.px_x
    x0 = 0.05 * w
    y1 = h - 0.05 * h
    stroke = _stroke(c.shape)
    c.draw.rectangle((x0, y1 - stroke, x0 + length_px, y1), fill=color)
    size = label_px(c.shape, cfg)
    s = f"{length_mm:g} mm"
    c.text((x0, y1 - stroke - size * 0.3), s, color, size, anchor="ls")
    return c.array()


def pixel_labels(
    img: ArrayLike,
    frame: Frame,
    stack: Stack,
    k: int,
    *,
    grid_mm: float | None = None,
    color: tuple[int, int, int] = CYAN,
    cfg: Config = DEFAULT,
) -> RGB:
    """Sparse labels 'r,c' at native pixel indices of slice ``k`` of ``stack`` that fall on the frame.

    The frame must lie in the plane of slice ``k`` (e.g. a zoom crop of that slice). Labels sit on a grid
    of about ``grid_mm``, so a reader can pick coordinates in the native image and then measure.
    """
    c = Canvas(img)
    h, w = c.shape
    grid = cfg.grid_mm if grid_mm is None else grid_mm
    step = max(1, round(grid / min(stack.row_mm, stack.col_mm)))
    corners = [
        frame.from_px(x, y) for x, y in ((-0.5, -0.5), (w - 0.5, -0.5), (w - 0.5, h - 0.5), (-0.5, h - 0.5))
    ]
    rc = np.array([stack.lps_to_pix(p)[1:] for p in corners])
    r_lo, c_lo = np.floor(rc.min(0) / step).astype(int) * step
    r_hi, c_hi = rc.max(0)
    size = max(6, round(label_px(c.shape, cfg) * 0.8))
    dot = max(1.0, _stroke(c.shape) * 1.5)
    r = int(r_lo)
    while r <= r_hi + step:
        cc = int(c_lo)
        while cc <= c_hi + step:
            x, y, _ = frame.to_px(stack.pix_to_lps(k, r, cc))
            if frame.inside(x, y):
                c.draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=color)
                c.text((x + 2 * dot, y), f"{r},{cc}", color, size, anchor="lm")
            cc += step
        r += step
    return c.array()


def add_rulers(
    img: ArrayLike,
    frame: Frame,
    *,
    step_mm: float | None = None,
    color: tuple[int, int, int] = WHITE,
    cfg: Config = DEFAULT,
) -> RGB:
    """Pad the image with a mm ruler on the left and bottom edges.

    Ticks are offsets from the image centre in mm (+ right, + down, along the frame axes). The step is a
    1-2-5 number giving about six ticks across the image unless ``step_mm`` is given.
    """
    arr = rgb_array(img)
    h, w = arr.shape[:2]
    size = label_px(arr.shape, cfg)
    ml = round(size * 3.2)
    mb = round(size * 2.2)
    width_mm, height_mm = frame.extent_mm
    out = Image.new("RGB", (w + ml, h + mb), BLACK)
    out.paste(Image.fromarray(arr), (ml, 0))
    c = Canvas(np.asarray(out))
    tick = max(2, round(size * 0.5))
    cx, cy = (w - 1) / 2 + ml, (h - 1) / 2

    def _fmt(v: float, step: float) -> str:
        dec = max(0, -math.floor(math.log10(step)))
        return "0" if abs(v) < 1e-9 else f"{v:+.{dec}f}"

    step_x = step_mm or nice_step(width_mm / 6)
    step_y = step_mm or nice_step(height_mm / 6)
    k = -math.floor((w / 2) * frame.px_x / step_x)
    while k * step_x <= (w / 2) * frame.px_x:
        off = k * step_x
        x = cx + off / frame.px_x
        if ml <= x <= ml + w:
            c.line((x, h), (x, h + tick), WHITE, 1)
            c.text((x, h + tick + 1), _fmt(off, step_x), WHITE, size, anchor="ma", bg=False)
        k += 1
    k = -math.floor((h / 2) * frame.px_y / step_y)
    while k * step_y <= (h / 2) * frame.px_y:
        off = k * step_y
        y = cy + off / frame.px_y
        if 0 <= y <= h:
            c.line((ml - tick, y), (ml, y), WHITE, 1)
            c.text((ml - tick - 1, y), _fmt(off, step_y), WHITE, size, anchor="rm", bg=False)
        k += 1
    return c.array()
