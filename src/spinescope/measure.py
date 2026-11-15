"""Measurements on stacks: distances, angles, intensity profiles, ROI statistics and edges.

Conventions: in-plane points are pixel ``(row, col)`` on slice ``vol[k]``; LPS points are patient mm.
Distances use ``row_mm`` / ``col_mm``, so they are valid on any slice of a stack.

Note on :func:`edge_to_edge`: it reports the two steepest intensity transitions along a segment. On a
dark endplate line those are the outer margins of the line, which over-reads a disc height by about a
millimetre per side (the prototype read 2-3 mm high). Disc heights therefore come from
:mod:`spinescope.analysis.discs`, which places the endplate on the disc side of the dark line.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.signal import find_peaks

from .geometry import FloatArray, Stack

__all__ = [
    "angle_deg",
    "dist_lps",
    "dist_mm",
    "edge_to_edge",
    "poly_stats",
    "profile",
    "roi_stats",
]


def dist_lps(a: Any, b: Any) -> float:
    """Euclidean distance in mm between two LPS points."""
    return float(np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)))


def dist_mm(stack: Stack, p0: Any, p1: Any) -> float:
    """In-plane distance in mm between two pixel points ``(r, c)`` (same on every slice)."""
    dr = (float(p1[0]) - float(p0[0])) * stack.row_mm
    dc = (float(p1[1]) - float(p0[1])) * stack.col_mm
    return float(np.hypot(dr, dc))


def angle_deg(line1: Any, line2: Any, stack: Stack | None = None) -> float:
    """Acute angle (0-90 degrees) between two lines given as ``(p0, p1)``.

    With ``stack``, points are pixel ``(r, c)`` and are scaled to mm first; otherwise they are taken
    as-is (LPS 3-vectors or mm 2-vectors).
    """

    def vec(line: Any) -> FloatArray:
        a = np.asarray(line[0], dtype=np.float64)
        b = np.asarray(line[1], dtype=np.float64)
        if stack is not None:
            sc = np.array([stack.row_mm, stack.col_mm])
            a, b = a * sc, b * sc
        return b - a

    u, v = vec(line1), vec(line2)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0 or nv == 0:
        raise ValueError("degenerate line (zero length)")
    cosang = abs(float(u @ v)) / (nu * nv)
    return float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))


def _rc_to_lps(stack: Stack, k: int, rc: NDArray[np.float64]) -> FloatArray:
    """(..., 2) pixel (r, c) on slice k -> (..., 3) LPS mm."""
    origin = stack.ipp[k]
    rc = np.asarray(rc, dtype=np.float64)
    return origin + rc[..., 1:2] * stack.col_mm * stack.row_dir + rc[..., 0:1] * stack.row_mm * stack.col_dir


def profile(
    stack: Stack,
    k: int,
    p0: Any,
    p1: Any,
    width_mm: float = 0.0,
    step_mm: float | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Intensity profile along the segment ``p0 -> p1`` (pixel ``(r, c)``) on slice ``k``.

    ``width_mm`` > 0 averages across a band of that width perpendicular to the segment. Samples are
    bilinear and clamp to the image edge.

    Returns ``(s_mm, values, xyz)``: distance from ``p0`` in mm, intensity, and the LPS position (n, 3)
    of each sample (band centre).
    """
    step = float(step_mm) if step_mm else 0.25 * min(stack.row_mm, stack.col_mm)
    sc = np.array([stack.row_mm, stack.col_mm])
    a = np.asarray(p0, dtype=np.float64) * sc
    b = np.asarray(p1, dtype=np.float64) * sc
    length = float(np.linalg.norm(b - a))
    if length == 0:
        raise ValueError("p0 == p1")
    n = int(np.ceil(length / step)) + 1
    t = np.linspace(0.0, 1.0, n)
    u = (b - a) / length
    perp = np.array([-u[1], u[0]])
    px = min(stack.row_mm, stack.col_mm)
    nb = max(1, 2 * int(width_mm / (2 * px)) + 1) if width_mm > 0 else 1
    offs = (np.arange(nb) - (nb - 1) / 2) * px
    pts = (
        a[None, None, :]
        + t[:, None, None] * (b - a)[None, None, :]
        + offs[None, :, None] * perp[None, None, :]
    )
    rc = pts / sc
    vals = ndimage.map_coordinates(stack.vol[k], [rc[..., 0], rc[..., 1]], order=1, mode="nearest")
    vals = vals.mean(axis=1)
    xyz = _rc_to_lps(stack, k, rc.mean(axis=1))
    return t * length, vals, xyz


def _stats(vals: NDArray[np.float32], area_mm2: float) -> dict[str, float | int]:
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "sd": float("nan"), "n": 0, "area_mm2": 0.0}
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "n": int(v.size),
        "area_mm2": float(area_mm2),
    }


def roi_stats(stack: Stack, k: int, center_rc: Any, radius_mm: float) -> dict[str, float | int]:
    """Mean, median, sd, n and area (mm2) of a circular ROI of radius ``radius_mm`` on slice ``k``."""
    rows, cols = stack.vol.shape[1:]
    rr, cc = np.mgrid[0:rows, 0:cols]
    d2 = ((rr - center_rc[0]) * stack.row_mm) ** 2 + ((cc - center_rc[1]) * stack.col_mm) ** 2
    m = d2 <= radius_mm**2
    return _stats(stack.vol[k][m], float(m.sum()) * stack.row_mm * stack.col_mm)


def poly_stats(stack: Stack, k: int, poly_rc: Any) -> dict[str, float | int]:
    """Mean, median, sd, n and area (mm2) inside a polygon given as ``[(r, c), ...]`` on slice ``k``."""
    rows, cols = stack.vol.shape[1:]
    pts = np.asarray(poly_rc, dtype=np.float64)
    from skimage.draw import polygon as _sk_polygon  # lazy: scikit-image is slow to import

    rr, cc = _sk_polygon(pts[:, 0], pts[:, 1], shape=(rows, cols))
    return _stats(stack.vol[k][rr, cc], float(len(rr)) * stack.row_mm * stack.col_mm)


def _edge_candidates(
    stack: Stack, k: int, p0: Any, p1: Any, min_sep_mm: float, smooth_mm: float
) -> list[dict[str, Any]]:
    step = 0.25 * min(stack.row_mm, stack.col_mm)
    s, v, xyz = profile(stack, k, p0, p1, step_mm=step)
    vs = ndimage.gaussian_filter1d(v, smooth_mm / step) if smooth_mm > 0 else v
    g = np.gradient(vs, step)
    ag = np.abs(g)
    peaks, _ = find_peaks(ag, distance=max(1, round(min_sep_mm / step)))
    p0a, p1a = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    length = float(s[-1])
    out: list[dict[str, Any]] = []
    for i in peaks[np.argsort(ag[peaks])[::-1]]:
        t = float(s[i]) / length
        rc = p0a + t * (p1a - p0a)
        out.append(
            {
                "s_mm": float(s[i]),
                "rc": (float(rc[0]), float(rc[1])),
                "lps": [round(float(x), 2) for x in xyz[i]],
                "strength": float(ag[i]),
                "sign": int(np.sign(g[i])),
            }
        )
    return out


def edge_to_edge(
    stack: Stack,
    k: int,
    p0: Any,
    p1: Any,
    min_sep_mm: float = 2.0,
    smooth_mm: float = 0.4,
) -> dict[str, Any]:
    """Distance between the two strongest intensity edges along ``p0 -> p1`` on slice ``k``.

    Edges are peaks of ``|d intensity / ds|`` at least ``min_sep_mm`` apart. Returns a dict with
    ``ok``, ``dist_mm`` (nan if fewer than two edges), ``edges`` (the two chosen, ``s_mm`` ascending,
    each with ``s_mm``, ``rc``, ``lps``, ``strength``, ``sign``; sign +1 means intensity rises from
    ``p0`` to ``p1``), ``peaks`` (all candidates, strongest first) and ``n_peaks``.
    Semi-automatic: check the overlay before trusting it on dark lines (see module docstring).
    """
    peaks = _edge_candidates(stack, k, p0, p1, min_sep_mm, smooth_mm)
    res: dict[str, Any] = {
        "ok": False,
        "dist_mm": float("nan"),
        "edges": [],
        "peaks": peaks,
        "n_peaks": len(peaks),
    }
    if len(peaks) < 2:
        return res
    two = sorted(peaks[:2], key=lambda e: e["s_mm"])
    res.update(ok=True, dist_mm=two[1]["s_mm"] - two[0]["s_mm"], edges=two)
    return res
