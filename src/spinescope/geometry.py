"""Patient-space geometry for stacks of MR slices.

Coordinates are DICOM patient LPS in millimetres:
  +x -> patient Left, +y -> Posterior, +z -> Head (superior).

A :class:`Stack` is a set of parallel slices with one orientation. Voxels are indexed
``vol[k, r, c]`` (slice, row, column), and pixel ``(r, c)`` on slice ``k`` sits at::

    ipp[k] + c * col_mm * row_dir + r * row_mm * col_dir

(DICOM quirk: ``row_dir`` is the direction of increasing *column* index.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

PLANES = ("sag", "cor", "ax")


def unit(v: Any) -> FloatArray:
    """Return ``v`` as a float64 unit vector."""
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n == 0:
        raise ValueError("zero-length vector")
    return a / n


def plane_of(normal: Any) -> str:
    """Nearest canonical plane ('sag', 'cor' or 'ax') for a slice normal."""
    return PLANES[int(np.argmax(np.abs(np.asarray(normal, dtype=np.float64))))]


def obliquity_deg(normal: Any) -> float:
    """Angle in degrees between a slice normal and the nearest canonical axis."""
    n = unit(normal)
    return float(np.degrees(np.arccos(np.clip(np.max(np.abs(n)), -1.0, 1.0))))


@dataclass(eq=False)
class Stack:
    """Parallel slices sharing one orientation, with exact patient-space geometry."""

    key: str
    vol: NDArray[np.float32]  # (n, rows, cols)
    ipp: FloatArray  # (n, 3) ImagePositionPatient of each slice's first pixel centre
    row_dir: FloatArray  # (3,) direction of increasing column index
    col_dir: FloatArray  # (3,) direction of increasing row index
    row_mm: float  # spacing between rows (vertical pixel size)
    col_mm: float  # spacing between columns (horizontal pixel size)
    thickness: float = 0.0
    description: str = ""
    series_number: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vol = np.asarray(self.vol, dtype=np.float32)
        if self.vol.ndim != 3:
            raise ValueError(f"vol must be 3-D (n, rows, cols), got shape {self.vol.shape}")
        self.ipp = np.asarray(self.ipp, dtype=np.float64).reshape(-1, 3)
        if len(self.ipp) != self.vol.shape[0]:
            raise ValueError("one ImagePositionPatient per slice is required")
        self.row_dir = unit(self.row_dir)
        self.col_dir = unit(self.col_dir)
        self.normal = unit(np.cross(self.row_dir, self.col_dir))
        order = np.argsort(self.ipp @ self.normal, kind="stable")
        if not np.array_equal(order, np.arange(len(order))):
            self.vol = self.vol[order]
            self.ipp = self.ipp[order]
            if "_sources" in self.meta:
                self.meta["_sources"] = [self.meta["_sources"][i] for i in order]
        self.positions: FloatArray = self.ipp @ self.normal

    # ----------------------------------------------------------------- properties
    @property
    def n(self) -> int:
        return int(self.vol.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.vol.shape[0]), int(self.vol.shape[1]), int(self.vol.shape[2]))

    @property
    def plane(self) -> str:
        return plane_of(self.normal)

    @property
    def obliquity(self) -> float:
        return obliquity_deg(self.normal)

    @property
    def spacing(self) -> float:
        """Median centre-to-centre slice spacing in mm (0 for a single slice)."""
        return float(np.median(np.abs(np.diff(self.positions)))) if self.n > 1 else 0.0

    def extent_mm(self) -> tuple[float, float]:
        """In-plane field of view (height, width) in mm."""
        return self.vol.shape[1] * self.row_mm, self.vol.shape[2] * self.col_mm

    # ---------------------------------------------------------------- transforms
    def _origin(self, k: float) -> FloatArray:
        k = float(k)
        if self.n == 1:
            return self.ipp[0] + (k * self.thickness) * self.normal
        k0 = int(np.clip(np.floor(k), 0, self.n - 2))
        f = k - k0
        return self.ipp[k0] * (1 - f) + self.ipp[k0 + 1] * f

    def pix_to_lps(self, k: float, r: float, c: float) -> FloatArray:
        """Slice ``k`` (fractional allowed, linearly extrapolated), row, column -> LPS mm."""
        return self._origin(k) + c * self.col_mm * self.row_dir + r * self.row_mm * self.col_dir

    def pix_to_lps_many(self, k: float, rc: Any) -> FloatArray:
        """Vectorised :meth:`pix_to_lps` for an (m, 2) array of (r, c) on slice ``k``."""
        rc = np.asarray(rc, dtype=np.float64).reshape(-1, 2)
        o = self._origin(k)
        return (
            o
            + np.outer(rc[:, 1] * self.col_mm, self.row_dir)
            + np.outer(rc[:, 0] * self.row_mm, self.col_dir)
        )

    def slice_index(self, xyz: Any) -> float:
        """Fractional slice index of a point (may be < 0 or > n-1 when outside the stack)."""
        d = float(np.asarray(xyz, dtype=np.float64) @ self.normal)
        if self.n == 1:
            return (d - float(self.positions[0])) / (self.thickness or 1.0)
        idx = np.arange(self.n, dtype=np.float64)
        if d <= self.positions[0]:
            step = self.positions[1] - self.positions[0]
            return float((d - self.positions[0]) / step)
        if d >= self.positions[-1]:
            step = self.positions[-1] - self.positions[-2]
            return float(self.n - 1 + (d - self.positions[-1]) / step)
        return float(np.interp(d, self.positions, idx))

    def lps_to_pix(self, xyz: Any) -> tuple[float, float, float]:
        """LPS mm -> (k, r, c). ``k`` is fractional and NOT clamped; check :meth:`contains`."""
        xyz = np.asarray(xyz, dtype=np.float64)
        k = self.slice_index(xyz)
        rel = xyz - self._origin(k)
        return k, float(rel @ self.col_dir / self.row_mm), float(rel @ self.row_dir / self.col_mm)

    def contains(self, xyz: Any, slab_margin_mm: float | None = None) -> bool:
        """True if ``xyz`` lies within the stack's slab (± half a slice) and in-plane field of view."""
        k, r, c = self.lps_to_pix(xyz)
        margin = (
            0.5
            if slab_margin_mm is None
            else slab_margin_mm / max(self.spacing or self.thickness or 1.0, 1e-6)
        )
        rows, cols = self.vol.shape[1:]
        return (
            (-margin <= k <= self.n - 1 + margin) and (-0.5 <= r <= rows - 0.5) and (-0.5 <= c <= cols - 0.5)
        )

    def nearest_slice(self, xyz: Any) -> tuple[int, float]:
        """Index of the slice plane closest to ``xyz`` and the out-of-plane distance in mm."""
        d = float(np.asarray(xyz, dtype=np.float64) @ self.normal)
        k = int(np.argmin(np.abs(self.positions - d)))
        return k, float(abs(self.positions[k] - d))

    def plane_at(self, k: int) -> tuple[FloatArray, FloatArray]:
        """(origin, unit normal) of slice ``k``."""
        return self.ipp[k].copy(), self.normal.copy()

    def corners(self, k: int) -> FloatArray:
        """LPS of the four image corners of slice ``k`` (pixel edges, not centres)."""
        rows, cols = self.vol.shape[1:]
        rc = np.array([[-0.5, -0.5], [-0.5, cols - 0.5], [rows - 0.5, cols - 0.5], [rows - 0.5, -0.5]])
        return self.pix_to_lps_many(k, rc)

    def summary(self) -> str:
        return (
            f"{self.key}: '{self.description}' n={self.n} {self.vol.shape[1]}x{self.vol.shape[2]} "
            f"px={self.row_mm:.3f}x{self.col_mm:.3f}mm thk={self.thickness:g} spacing={self.spacing:.2f}mm "
            f"plane={self.plane} oblique={self.obliquity:.1f}deg"
        )


def intersection_segment(
    a: Stack, k_a: int, b: Stack, k_b: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Where slice ``k_b`` of ``b`` cuts slice ``k_a`` of ``a``, as a pixel segment on ``a``.

    The segment is clipped to both images' in-plane fields of view. Returns ``None`` for
    parallel planes or when the planes don't meet inside both images.
    """
    oa, na = a.plane_at(k_a)
    ob, nb = b.plane_at(k_b)
    d = np.cross(na, nb)
    if np.linalg.norm(d) < 1e-6:
        return None
    d = unit(d)
    # A point on both planes: solve [na; nb; d] p = [na.oa; nb.ob; d.oa]
    m = np.vstack([na, nb, d])
    p0 = np.linalg.solve(m, np.array([na @ oa, nb @ ob, d @ oa]))

    def clip(stack: Stack, k: int, t0: float, t1: float) -> tuple[float, float] | None:
        rows, cols = stack.vol.shape[1:]
        o = stack.ipp[k]
        # pixel coordinates along the line: r(t), c(t) are affine in t
        pr = (p0 - o) @ stack.col_dir / stack.row_mm
        pc = (p0 - o) @ stack.row_dir / stack.col_mm
        dr = d @ stack.col_dir / stack.row_mm
        dc = d @ stack.row_dir / stack.col_mm
        for p, dp, lo, hi in ((pr, dr, -0.5, rows - 0.5), (pc, dc, -0.5, cols - 0.5)):
            if abs(dp) < 1e-12:
                if not lo <= p <= hi:
                    return None
                continue
            ta, tb = sorted(((lo - p) / dp, (hi - p) / dp))
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return None
        return t0, t1

    span = clip(a, k_a, -1e9, 1e9)
    if span is None:
        return None
    span = clip(b, k_b, *span)
    if span is None:
        return None
    pts = []
    for t in span:
        _k, r, c = a.lps_to_pix(p0 + t * d)
        pts.append((r, c))
    return (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1])


def sample_plane(
    stack: Stack,
    origin: Any,
    u: Any,
    v: Any,
    size_mm: tuple[float, float],
    px_mm: float,
    order: int = 1,
) -> NDArray[np.float32]:
    """Resample ``stack`` on an arbitrary plane ``origin + s*u + t*v`` (trilinear by default).

    Returns an (nt, ns) image covering ``t`` in [-h/2, h/2] (rows) and ``s`` in [-w/2, w/2]
    (columns); points outside the stack are NaN.
    """
    from scipy.ndimage import map_coordinates

    u, v = unit(u), unit(v)
    h, w = size_mm
    nt, ns = max(round(h / px_mm), 1), max(round(w / px_mm), 1)
    tt = (np.arange(nt) - (nt - 1) / 2) * px_mm
    ss = (np.arange(ns) - (ns - 1) / 2) * px_mm
    S, T = np.meshgrid(ss, tt)
    pts = np.asarray(origin, dtype=np.float64) + S[..., None] * u + T[..., None] * v
    flat = pts.reshape(-1, 3)
    d = flat @ stack.normal
    if stack.n > 1:
        k = np.interp(d, stack.positions, np.arange(stack.n), left=np.nan, right=np.nan)
        half = 0.5 * (stack.spacing or stack.thickness)
        k = np.where(d < stack.positions[0] - half, np.nan, k)
        k = np.where(d > stack.positions[-1] + half, np.nan, k)
        k = np.where((d >= stack.positions[0] - half) & (d < stack.positions[0]), 0.0, k)
        k = np.where((d <= stack.positions[-1] + half) & (d > stack.positions[-1]), stack.n - 1.0, k)
    else:
        k = np.where(np.abs(d - stack.positions[0]) <= max(stack.thickness, 1.0) / 2, 0.0, np.nan)
    o = stack.ipp[0]
    rel = flat - o
    r = rel @ stack.col_dir / stack.row_mm
    c = rel @ stack.row_dir / stack.col_mm
    ok = np.isfinite(k)
    out = np.full(len(flat), np.nan, dtype=np.float32)
    if ok.any():
        # in-plane offset of each slice relative to slice 0 (stacks may shear slightly)
        k_ok = k[ok]
        k0 = np.clip(np.floor(k_ok).astype(int), 0, stack.n - 1)
        shift = stack.ipp[k0] - o
        r_ok = r[ok] - shift @ stack.col_dir / stack.row_mm
        c_ok = c[ok] - shift @ stack.row_dir / stack.col_mm
        rows, cols = stack.vol.shape[1:]
        inside = (r_ok >= -0.5) & (r_ok <= rows - 0.5) & (c_ok >= -0.5) & (c_ok <= cols - 0.5)
        vals = map_coordinates(stack.vol, [k_ok, r_ok, c_ok], order=order, mode="nearest")
        vals = np.where(inside, vals, np.nan)
        out[ok] = vals
    return out.reshape(nt, ns)
