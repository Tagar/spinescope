"""Plan browser-side reformats (MPR) for planes the study has no usable series for.

A reformat is a regular grid in one of the display planes (sagittal, coronal, axial) whose voxels
the browser resamples from a source volume. This module only decides *which* source to use and
the exact patient-space geometry of the grid; the voxel values are computed in the viewer.

Source choice is measured on the data: the study's body box is sampled on a grid and each
candidate volume is scored by the fraction of those points it covers (its through-plane and
in-plane coverage together). Ties within ``Config.viewer_mpr_coverage_tie`` go to the contrast
preference (T2, then T1, STIR, PD), then to the thinner through-plane spacing.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import DEFAULT, Config
from ..geometry import Stack, unit
from ..privacy import public_text
from ..series import DISPLAY_AXES
from ..study import Study

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: Contrast preference for tie-breaks (lower is better).
CONTRAST_RANK = {"t2": 0, "t1": 1, "stir": 2, "pd": 3}
#: Points per axis of the body-box coverage grid (15^3 = 3375 points, vectorised per candidate).
COVERAGE_GRID = 15


@dataclass(frozen=True)
class Reformat:
    """Geometry of one browser-computed plane. ``ipp[k]`` is the first pixel centre of slice ``k``."""

    key: str
    plane: str
    source_key: str
    coverage: float
    px_mm: float
    row_dir: FloatArray
    col_dir: FloatArray
    normal: FloatArray
    ipp: FloatArray  # (n, 3)
    rows: int
    cols: int
    contrast: str
    source_description: str

    @property
    def n(self) -> int:
        return int(self.ipp.shape[0])

    @property
    def positions(self) -> FloatArray:
        return self.ipp @ self.normal

    @property
    def description(self) -> str:
        return f"{self.plane.upper()} reformat from {self.source_key}"

    def as_stack(self) -> Stack:
        """A zero-voxel :class:`Stack` with this geometry (for geometry checks; voxels are not stored)."""
        return Stack(
            key=self.key,
            vol=np.zeros((self.n, self.rows, self.cols), dtype=np.float32),
            ipp=self.ipp,
            row_dir=self.row_dir,
            col_dir=self.col_dir,
            row_mm=self.px_mm,
            col_mm=self.px_mm,
            thickness=self.px_mm,
        )


def body_box(stacks: list[Stack]) -> tuple[FloatArray, FloatArray]:
    """Axis-aligned LPS box enclosing the outer pixel corners of the first and last slices."""
    pts = np.vstack([np.vstack([s.corners(0), s.corners(s.n - 1)]) for s in stacks])
    return pts.min(axis=0), pts.max(axis=0)


def inside_mask(stack: Stack, points: FloatArray) -> NDArray[np.bool_]:
    """Vectorised :meth:`Stack.contains` (half-slice slab margin): which (m, 3) LPS points are inside."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    d = pts @ stack.normal
    pos = stack.positions
    if stack.n == 1:
        k = (d - pos[0]) / (stack.thickness or 1.0)
        origin = stack.ipp[0][None, :] + (k * stack.thickness)[:, None] * stack.normal[None, :]
    else:
        idx = np.arange(stack.n, dtype=np.float64)
        k = np.interp(d, pos, idx)
        lo_ext = d <= pos[0]
        hi_ext = d >= pos[-1]
        k = np.where(lo_ext, (d - pos[0]) / (pos[1] - pos[0]), k)
        k = np.where(hi_ext, stack.n - 1 + (d - pos[-1]) / (pos[-1] - pos[-2]), k)
        k0 = np.clip(np.floor(k), 0, stack.n - 2).astype(int)
        f = (k - k0)[:, None]
        origin = stack.ipp[k0] * (1 - f) + stack.ipp[k0 + 1] * f
    rel = pts - origin
    r = rel @ stack.col_dir / stack.row_mm
    c = rel @ stack.row_dir / stack.col_mm
    rows, cols = stack.vol.shape[1:]
    return (
        (k >= -0.5) & (k <= stack.n - 0.5) & (r >= -0.5) & (r <= rows - 0.5) & (c >= -0.5) & (c <= cols - 0.5)
    )


def coverage(stack: Stack, points: FloatArray) -> float:
    """Fraction of ``points`` (m, 3) LPS that lie inside ``stack``'s slab and in-plane field of view."""
    if len(points) == 0:
        return 0.0
    return float(np.mean(inside_mask(stack, points)))


def _grid(lo: FloatArray, hi: FloatArray, m: int) -> FloatArray:
    axes = [np.linspace(lo[i], hi[i], m) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _candidates(study: Study, plane: str) -> list[Stack]:
    """Non-localizer stacks in other planes; localizers only if nothing else exists."""
    others = [s for s in study.stacks.values() if s.plane != plane]
    main = [s for s in others if not s.meta.get("role", {}).get("localizer")]
    return main or others


def choose_source(study: Study, plane: str, *, config: Config = DEFAULT) -> tuple[Stack, float] | None:
    """Best source volume for a reformat in ``plane``, with its coverage, or ``None``.

    Returns ``None`` (and logs a warning) when no candidate covers at least
    ``config.viewer_mpr_min_coverage`` of the body box.
    """
    if plane not in DISPLAY_AXES:
        raise ValueError(f"unknown plane {plane!r}")
    cands = _candidates(study, plane)
    if not cands:
        return None
    main = [s for s in study.stacks.values() if not s.meta.get("role", {}).get("localizer")] or list(
        study.stacks.values()
    )
    lo, hi = body_box(main)
    pts = _grid(lo, hi, COVERAGE_GRID)
    scored = [(coverage(s, pts), s) for s in cands]
    best_cov = max(c for c, _ in scored)
    if best_cov < config.viewer_mpr_min_coverage:
        log.warning("no source covers the body enough for a %s reformat (best %.2f)", plane, best_cov)
        return None
    tied = [(c, s) for c, s in scored if c >= best_cov - config.viewer_mpr_coverage_tie]

    def rank(item: tuple[float, Stack]) -> tuple[int, float, str]:
        _c, s = item
        contrast = str(s.meta.get("role", {}).get("contrast", ""))
        return (CONTRAST_RANK.get(contrast, 9), s.spacing or s.thickness or 1e9, s.key)

    cov, src = min(tied, key=rank)
    return src, float(cov)


def plan_reformat(study: Study, plane: str, *, config: Config = DEFAULT) -> Reformat | None:
    """Exact grid for a reformat in ``plane`` covering the study's body box, or ``None``."""
    chosen = choose_source(study, plane, config=config)
    if chosen is None:
        return None
    src, cov = chosen
    rd, cd = (np.asarray(v, dtype=np.float64) for v in DISPLAY_AXES[plane])
    nrm = unit(np.cross(rd, cd))
    main = [s for s in study.stacks.values() if not s.meta.get("role", {}).get("localizer")] or list(
        study.stacks.values()
    )
    lo, hi = body_box(main)
    corners = np.array(list(itertools.product(*zip(lo, hi, strict=True))), dtype=np.float64)
    s_proj, t_proj, d_proj = corners @ rd, corners @ cd, corners @ nrm
    px = min(src.row_mm, src.col_mm)
    extent = max(np.ptp(s_proj), np.ptp(t_proj), np.ptp(d_proj))
    if px <= 0:
        raise ValueError(f"source {src.key} has no pixel spacing")
    px = max(px, extent / max(config.viewer_mpr_max_dim - 1, 1))
    s0, t0, d0 = float(np.min(s_proj)), float(np.min(t_proj)), float(np.min(d_proj))
    cols = math.floor(np.ptp(s_proj) / px + 1e-9) + 1
    rows = math.floor(np.ptp(t_proj) / px + 1e-9) + 1
    n = math.floor(np.ptp(d_proj) / px + 1e-9) + 1
    d_k = d0 + np.arange(n) * px
    ipp = s0 * rd + t0 * cd + d_k[:, None] * nrm[None, :]
    role = src.meta.get("role", {})
    return Reformat(
        key=f"mpr_{plane}",
        plane=plane,
        source_key=src.key,
        coverage=cov,
        px_mm=float(px),
        row_dir=rd,
        col_dir=cd,
        normal=nrm,
        ipp=ipp,
        rows=rows,
        cols=cols,
        contrast=str(role.get("contrast", "")),
        source_description=public_text(src.description),
    )


def plan_reformats(study: Study, *, config: Config = DEFAULT) -> list[Reformat]:
    """Reformats for every plane that has no non-localizer series in the study."""
    out: list[Reformat] = []
    for plane in DISPLAY_AXES:
        if any(
            s.plane == plane and not s.meta.get("role", {}).get("localizer") for s in study.stacks.values()
        ):
            continue
        ref = plan_reformat(study, plane, config=config)
        if ref is not None:
            log.info("viewer: %s reformat from %s (coverage %.2f)", plane, ref.source_key, ref.coverage)
            out.append(ref)
    return out
