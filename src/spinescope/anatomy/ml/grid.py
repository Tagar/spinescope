"""Fixed-mm sagittal plane grid shared by training and inference (no torch dependency).

Each sagittal slice is resampled onto a regular grid of ``res_mm`` pixels whose rows run from
superior to inferior and whose columns run from anterior to posterior, whatever the acquisition
orientation. Grid coordinates are "plane mm": ``down`` (mm inferior from the slice origin) and
``post`` (mm posterior), so ``LPS = ipp_k + down * e_down + post * e_post`` exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ...geometry import Stack

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlaneGrid:
    """A regular in-plane grid of one slice of a :class:`~spinescope.geometry.Stack`."""

    k: int  # slice index in the stack
    ipp: FloatArray  # (3,) LPS of the slice origin (ImagePositionPatient)
    e_down: FloatArray  # (3,) unit LPS vector towards inferior (row index increases along it)
    e_post: FloatArray  # (3,) unit LPS vector towards posterior (column index increases along it)
    down0: float  # mm coordinate of the top-left pixel edge
    post0: float  # mm coordinate of the anterior pixel edge
    res_mm: float
    shape: tuple[int, int]  # (rows, cols), both multiples of the requested block size

    def to_lps(self, down: object, post: object) -> FloatArray:
        """Plane mm (down, post) -> LPS mm. Arrays broadcast; the result has a trailing axis of 3."""
        d = np.asarray(down, dtype=np.float64)[..., None]
        p = np.asarray(post, dtype=np.float64)[..., None]
        return self.ipp + d * self.e_down + p * self.e_post

    def mm_of(self, xyz: object) -> FloatArray:
        """LPS mm -> plane mm ``(down, post)`` (last axis of size 2) for points on this slice's plane."""
        rel = np.asarray(xyz, dtype=np.float64) - self.ipp
        return np.stack([rel @ self.e_down, rel @ self.e_post], axis=-1)

    def index_of_mm(self, down: object, post: object) -> tuple[FloatArray, FloatArray]:
        """Plane mm -> fractional pixel index (row, col); pixel centres are at integer indices."""
        i = (np.asarray(down, dtype=np.float64) - self.down0) / self.res_mm - 0.5
        j = (np.asarray(post, dtype=np.float64) - self.post0) / self.res_mm - 0.5
        return i, j

    def mm_of_index(self, i: object, j: object) -> tuple[FloatArray, FloatArray]:
        """Fractional pixel index (row, col) -> plane mm."""
        down = self.down0 + (np.asarray(i, dtype=np.float64) + 0.5) * self.res_mm
        post = self.post0 + (np.asarray(j, dtype=np.float64) + 0.5) * self.res_mm
        return down, post

    def lps_of_index(self, i: object, j: object) -> FloatArray:
        """Pixel index (row, col) -> LPS mm."""
        down, post = self.mm_of_index(i, j)
        return self.to_lps(down, post)


def _ceil_to(n: int, multiple: int) -> int:
    return int(math.ceil(max(n, 1) / multiple) * multiple)


def plane_grid(stack: Stack, k: int, res_mm: float, multiple: int = 16) -> PlaneGrid:
    """Grid covering slice ``k`` at ``res_mm``, with the shape padded (symmetrically) to ``multiple``."""
    rows, cols = int(stack.vol.shape[1]), int(stack.vol.shape[2])
    ax_r, ax_c = stack.col_dir, stack.row_dir  # direction of increasing row / column index
    # The axis with the larger z component is the superior-inferior axis of the plane.
    if abs(ax_r[2]) >= abs(ax_c[2]):
        si, ap = ax_r, ax_c
    else:
        si, ap = ax_c, ax_r
    e_down = -si if si[2] > 0 else si
    e_post = ap if ap[1] >= 0 else -ap
    downs: list[float] = []
    posts: list[float] = []
    for r in (-0.5, rows - 0.5):
        for c in (-0.5, cols - 0.5):
            d = c * stack.col_mm * stack.row_dir + r * stack.row_mm * stack.col_dir
            downs.append(float(d @ e_down))
            posts.append(float(d @ e_post))
    d_lo, d_hi = min(downs), max(downs)
    p_lo, p_hi = min(posts), max(posts)
    h = _ceil_to(math.ceil((d_hi - d_lo) / res_mm), multiple)
    w = _ceil_to(math.ceil((p_hi - p_lo) / res_mm), multiple)
    d_start = 0.5 * (d_lo + d_hi) - 0.5 * h * res_mm
    p_start = 0.5 * (p_lo + p_hi) - 0.5 * w * res_mm
    return PlaneGrid(
        k=k,
        ipp=np.asarray(stack.ipp[k], dtype=np.float64),
        e_down=np.asarray(e_down, dtype=np.float64),
        e_post=np.asarray(e_post, dtype=np.float64),
        down0=d_start,
        post0=p_start,
        res_mm=float(res_mm),
        shape=(h, w),
    )


def resample_slice(
    stack: Stack, grid: PlaneGrid, array: NDArray[np.floating] | None = None, order: int = 1
) -> NDArray[np.float32]:
    """Sample ``array`` (default ``stack.vol``, shape (n, rows, cols)) on ``grid``; outside the slice is 0."""
    src = stack.vol if array is None else array
    h, w = grid.shape
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    down, post = grid.mm_of_index(ii, jj)
    d = down[..., None] * grid.e_down + post[..., None] * grid.e_post
    r = (d @ stack.col_dir) / stack.row_mm
    c = (d @ stack.row_dir) / stack.col_mm
    out = ndimage.map_coordinates(
        np.asarray(src[grid.k], dtype=np.float32), [r, c], order=order, mode="constant", cval=0.0
    )
    return np.asarray(out, dtype=np.float32)


def network_input_volume(stack: Stack) -> NDArray[np.floating]:
    """The volume the network and its window see: ``stack.vol``, shifted for signed pixel data.

    Signed DICOM (PixelRepresentation 1) with a background below zero records ``signed_offset`` in its metadata
    (see :mod:`spinescope.io.dicom`); the window assumes a background near zero, so the offset is removed here.
    Every other stack (unsigned DICOM, NIfTI, MetaImage such as SPIDER) is returned unchanged.
    """
    off = float(stack.meta.get("signed_offset", 0.0) or 0.0)
    if off == 0.0:
        return stack.vol
    return (stack.vol - np.float32(off)).astype(np.float32)


def intensity_window(
    vol: NDArray[np.floating], pct: tuple[float, float] = (0.5, 99.5), background_frac: float = 0.02
) -> tuple[float, float]:
    """Per-volume intensity window: percentiles of the foreground (voxels above ``background_frac`` of the 99.5th percentile)."""
    v = np.asarray(vol, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    ref = float(np.percentile(v, 99.5))
    fg = v[v > background_frac * ref] if ref > 0 else v
    if fg.size < 100:
        fg = v
    lo, hi = (float(x) for x in np.percentile(fg, pct))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def normalise(img: NDArray[np.floating], window: tuple[float, float]) -> NDArray[np.float32]:
    """Map intensities to [0, 1] with a per-volume window (clipped)."""
    lo, hi = window
    return np.clip((np.asarray(img, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
