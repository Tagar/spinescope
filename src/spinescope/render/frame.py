"""Image planes: the pixel <-> patient LPS mapping shared by every renderer and annotation.

A :class:`Frame` describes a 2-D image lying in a plane: ``origin`` is the LPS point at the centre of
pixel (0, 0), ``u`` / ``v`` are unit LPS vectors for increasing column (image right) and increasing row
(image down), and ``px_x`` / ``px_y`` are the mm per pixel along them. Pixel coordinates are ``(x, y)``
= (column, row) with pixel centres at integers, so ``x`` pairs with ``u`` and ``y`` with ``v``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import DEFAULT, Config
from ..geometry import Stack, unit

FloatArray = NDArray[np.float64]

_LETTERS = {(0, 1): "L", (0, -1): "R", (1, 1): "P", (1, -1): "A", (2, 1): "S", (2, -1): "I"}


def direction_letter(vec: Any) -> str:
    """Radiological letter (R/L, A/P, I/S) of the dominant LPS direction of ``vec``."""
    v = np.asarray(vec, dtype=np.float64)
    i = int(np.argmax(np.abs(v)))
    return _LETTERS[(i, 1 if v[i] >= 0 else -1)]


def nice_step(target: float) -> float:
    """The 1-2-5 x 10^n number closest (in log scale) to ``target`` (> 0)."""
    if not target > 0 or not math.isfinite(target):
        return 1.0
    exp = math.floor(math.log10(target))
    cands = [m * 10.0**e for e in (exp - 1, exp, exp + 1) for m in (1.0, 2.0, 5.0)]
    return min(cands, key=lambda c: abs(math.log(c / target)))


@dataclass(frozen=True, eq=False)
class Frame:
    """An image plane: origin (LPS of pixel (0, 0) centre), unit axes u (right) / v (down), mm per pixel."""

    origin: FloatArray
    u: FloatArray
    v: FloatArray
    px_x: float
    px_y: float
    shape: tuple[int, int]  # (rows, cols)

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=np.float64).reshape(3))
        object.__setattr__(self, "u", unit(self.u))
        object.__setattr__(self, "v", unit(self.v))
        object.__setattr__(self, "shape", (int(self.shape[0]), int(self.shape[1])))

    @classmethod
    def centred(cls, centre: Any, u: Any, v: Any, shape: tuple[int, int], px_x: float, px_y: float) -> Frame:
        """Frame whose image centre sits at ``centre`` (pixel centres symmetric about it)."""
        uu, vv = unit(u), unit(v)
        rows, cols = int(shape[0]), int(shape[1])
        origin = (
            np.asarray(centre, dtype=np.float64) - ((cols - 1) / 2) * px_x * uu - ((rows - 1) / 2) * px_y * vv
        )
        return cls(origin, uu, vv, px_x, px_y, (rows, cols))

    @property
    def normal(self) -> FloatArray:
        return unit(np.cross(self.u, self.v))

    @property
    def extent_mm(self) -> tuple[float, float]:
        """(width, height) of the image in mm."""
        return self.shape[1] * self.px_x, self.shape[0] * self.px_y

    @property
    def centre(self) -> FloatArray:
        """LPS of the image centre."""
        return self.from_px((self.shape[1] - 1) / 2, (self.shape[0] - 1) / 2)

    def to_px(self, xyz: Any) -> tuple[float, float, float]:
        """LPS point -> (x, y, off_plane_mm). Off-plane is the signed distance along the frame normal."""
        rel = np.asarray(xyz, dtype=np.float64) - self.origin
        return float(rel @ self.u / self.px_x), float(rel @ self.v / self.px_y), float(rel @ self.normal)

    def from_px(self, x: float, y: float) -> FloatArray:
        """Pixel (x, y) -> LPS point on the frame plane (fractional allowed)."""
        return self.origin + x * self.px_x * self.u + y * self.px_y * self.v

    def inside(self, x: float, y: float) -> bool:
        """True if pixel coordinate (x, y) lies on the image."""
        return bool(-0.5 <= x <= self.shape[1] - 0.5 and -0.5 <= y <= self.shape[0] - 0.5)

    def edge_letters(self) -> dict[str, str]:
        """Orientation letters at the image edges: right, left, top, bottom."""
        return {
            "right": direction_letter(self.u),
            "left": direction_letter(-self.u),
            "top": direction_letter(-self.v),
            "bottom": direction_letter(self.v),
        }

    def scaled(self, factor: float) -> Frame:
        """The same plane sampled ``factor`` times more finely (pixel centres of the new grid)."""
        f = float(factor)
        rows, cols = self.shape
        # native pixel index of the first new pixel centre is 0.5/f - 0.5
        x0 = 0.5 / f - 0.5
        origin = self.origin + x0 * self.px_x * self.u + x0 * self.px_y * self.v
        return Frame(
            origin,
            self.u,
            self.v,
            self.px_x / f,
            self.px_y / f,
            (max(1, round(rows * f)), max(1, round(cols * f))),
        )


def slice_frame(stack: Stack, k: int) -> Frame:
    """The native pixel grid of slice ``k`` in display axes (image right = ``row_dir``)."""
    rows, cols = stack.vol.shape[1:]
    return Frame(stack.ipp[k], stack.row_dir, stack.col_dir, stack.col_mm, stack.row_mm, (rows, cols))


def display_scale(img: NDArray[np.uint8], frame: Frame, min_side: int) -> tuple[NDArray[np.uint8], Frame]:
    """Upscale an RGB image (bilinear) so its shorter side reaches ``min_side`` px, for legibility.

    Returns the image and the matching frame (pixel size divided by the same factor), so annotations
    drawn afterwards stay in mm. Images already large enough are returned unchanged.
    """
    from PIL import Image

    h, w = img.shape[:2]
    short = min(h, w)
    if short <= 0 or short >= min_side:
        return img, frame
    f = min_side / short
    nh, nw = max(1, round(h * f)), max(1, round(w * f))
    out = np.asarray(Image.fromarray(img).resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.uint8)
    return out, frame.scaled(f)


def auto_upscale(px_mm: float, size_mm: tuple[float, float], min_side: int) -> float:
    """Sampling upscale factor (>= 1) so that a crop of ``size_mm`` at native pitch ``px_mm`` has a shorter
    side of at least ``min_side`` pixels."""
    native_side = min(size_mm) / max(px_mm, 1e-9)
    return max(1.0, min_side / max(native_side, 1e-9))


def min_display_px(cfg: Config = DEFAULT) -> int:
    """Smallest shorter side (px) at which a crop or reformat is shown; coarser grids are upscaled."""
    return max(16, round(cfg.max_png_side_px * cfg.display_min_frac))
