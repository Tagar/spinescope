"""Spinal canal geometry from the learned labeller's canal class: centreline x/y(z) and AP diameters.

Pure numpy/scipy functions on probability maps, a plane grid, or LPS point clouds, so the validation tool can
apply the very same functions to SPIDER canal masks (label 100) and compare like with like.

Definitions (LPS mm; "plane" = the mid-sagittal grid, columns run anterior->posterior):

- centreline: canal pixels of every slice pooled into z bins (``Config.canal_z_bin_mm``); each bin gives the
  mean x and y of its pixels. Bins seen on fewer than ``canal_min_bin_slices`` slices are dropped. The centreline
  therefore follows the canal where the slices cover it and does not extrapolate.
- AP diameter at a level z: on the mid-sagittal canal mask, the median over the rows within
  ``canal_level_band_mm`` of z of the row's AP chord (last minus first canal column, in mm). It is a
  sagittal dural-sac width, not a lateral or spinal-canal (bony) diameter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..config import DEFAULT, Config
from .ml.grid import PlaneGrid
from .ml.labeller import CANAL, SegmentedStack

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def largest_component(mask: BoolArray) -> BoolArray:
    """The largest 8-connected component of ``mask`` (``mask`` itself when it has at most one)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.asarray(lab == int(np.argmax(sizes)), dtype=bool)


def row_heights_mm(grid: PlaneGrid) -> FloatArray:
    """Mean LPS z of each row of ``grid`` (rows run inferior, so this decreases down the grid)."""
    h, w = grid.shape
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    down, post = grid.mm_of_index(ii, jj)
    return np.asarray(grid.to_lps(down, post)[..., 2].mean(axis=1), dtype=np.float64)


def canal_mask_plane(seg: SegmentedStack, k: int, cfg: Config = DEFAULT) -> BoolArray:
    """Canal pixels on slice ``k`` (largest component of the canal class above threshold)."""
    return largest_component(seg.probs[k, CANAL] > cfg.ml_prob_threshold)


def ap_chord_mm(mask: BoolArray, grid: PlaneGrid, z_mm: float, band_mm: float) -> float | None:
    """Median AP chord (mm) of ``mask`` over the rows within ``band_mm`` of height ``z_mm``; None if no canal there."""
    zr = row_heights_mm(grid)
    rows = np.nonzero(np.abs(zr - z_mm) <= band_mm)[0]
    chords: list[float] = []
    for r in rows:
        cols = np.nonzero(mask[r])[0]
        if cols.size:
            chords.append(float((cols.max() - cols.min() + 1) * grid.res_mm))
    return float(np.median(chords)) if chords else None


def posterior_canal_edge(mask: BoolArray, row: int) -> int | None:
    """Most posterior canal column on ``row`` (the posterior dural margin on that row); None if absent."""
    cols = np.nonzero(mask[row])[0]
    return int(cols.max()) if cols.size else None


def canal_points_lps(seg: SegmentedStack, cfg: Config = DEFAULT) -> tuple[FloatArray, NDArray[np.intp]]:
    """LPS (n, 3) of every canal pixel above the threshold, and the slice index of each point."""
    pts: list[FloatArray] = []
    ids: list[NDArray[np.intp]] = []
    for k, grid in enumerate(seg.grids):
        ii, jj = np.nonzero(seg.probs[k, CANAL] > cfg.ml_prob_threshold)
        if ii.size == 0:
            continue
        down, post = grid.mm_of_index(ii, jj)
        pts.append(np.asarray(grid.to_lps(down, post), dtype=np.float64))
        ids.append(np.full(ii.size, k, dtype=np.intp))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.intp)
    return np.concatenate(pts, axis=0), np.concatenate(ids)


def centreline_from_points(
    pts: FloatArray,
    z_lo: float,
    z_hi: float,
    cfg: Config = DEFAULT,
    slice_ids: NDArray[np.intp] | None = None,
) -> list[list[float]]:
    """Centreline points ``[x, y, z]`` (superior first) from canal points, in z bins over ``[z_lo, z_hi]``.

    Within a bin, each sagittal slice contributes its canal pixels only when it carries at least
    ``canal_bin_slice_fraction`` of the bin's largest slice count, so stray canal pixels on lateral slices do not
    pull the centre. A bin is kept when at least ``canal_min_bin_slices`` slices pass. ``slice_ids`` identifies
    the slice of each point (default: the rounded x coordinate). ``pts`` may come from a prediction or a truth mask.
    """
    if pts.size == 0 or z_hi <= z_lo:
        return []
    b = cfg.canal_z_bin_mm
    sel = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
    p = pts[sel]
    if p.size == 0:
        return []
    sid = np.round(p[:, 0]).astype(np.int64) if slice_ids is None else np.asarray(slice_ids)[sel]
    idx = np.floor((p[:, 2] - z_lo) / b).astype(int)
    out: list[list[float]] = []
    for bi in np.unique(idx):
        in_bin = idx == bi
        q, qs = p[in_bin], sid[in_bin]
        slices, counts = np.unique(qs, return_counts=True)
        keep = slices[counts >= cfg.canal_bin_slice_fraction * float(np.max(counts))]
        if keep.size < cfg.canal_min_bin_slices:
            continue
        m = np.isin(qs, keep)
        out.append([float(q[m, 0].mean()), float(q[m, 1].mean()), float(z_lo + (bi + 0.5) * b)])
    return sorted(out, key=lambda t: -t[2])


def canal_geometry(
    seg: SegmentedStack,
    k_default: int,
    disc_z: list[tuple[str, float]],
    body_mids: list[tuple[str, float]],
    cfg: Config = DEFAULT,
    *,
    confidence: float = 1.0,
) -> dict[str, Any]:
    """``Anchors.canal`` for the labelled study.

    ``disc_z`` is ``[(label, z)]`` of the discs (any order); ``body_mids`` is ``[(vertebra name, z)]`` of the
    mid-body heights. The canal runs laterally across a sagittal stack (scoliosis, rotation), so each level is
    measured on the sagittal slice nearest the centreline at that height (``k_default`` only when the centreline is
    empty). Returns ``points`` (centreline ``[x, y, z]``, superior first), ``per_level`` (label -> ``{"z_mm",
    "ap_mm", "kind", "slice"}``), ``confidence`` and ``evidence``; ``ap_mm`` is None where the canal is not on
    the chosen slice.
    """
    zs = [z for _, z in disc_z]
    z_lo = (min(zs) - cfg.canal_zone_margin_mm) if zs else 0.0
    z_hi = (max(zs) + cfg.canal_zone_margin_mm) if zs else 0.0
    pts, ids = canal_points_lps(seg, cfg)
    if pts.size:
        z_lo = max(z_lo, float(np.min(pts[:, 2])))
        z_hi = min(z_hi, float(np.max(pts[:, 2])))
    points = centreline_from_points(pts, z_lo, z_hi, cfg, ids)
    normal = np.asarray(seg.stack.normal, dtype=np.float64)
    positions = np.asarray(seg.stack.positions, dtype=np.float64)
    cz = np.array([p[2] for p in points][::-1])  # ascending for interpolation
    cu = np.array([float(np.dot(p, normal)) for p in points][::-1])

    def slice_at(z: float) -> int:
        if cz.size == 0:
            return k_default
        u = float(np.interp(z, cz, cu))
        return int(np.argmin(np.abs(positions - u)))

    masks: dict[int, BoolArray] = {}
    per_level: dict[str, dict[str, Any]] = {}
    for kind, items in (("disc", disc_z), ("mid_body", body_mids)):
        for lab, z in items:
            k = slice_at(float(z))
            if k not in masks:
                masks[k] = canal_mask_plane(seg, k, cfg)
            chord = ap_chord_mm(masks[k], seg.grids[k], float(z), cfg.canal_level_band_mm)
            per_level[lab] = {
                "z_mm": float(z),
                "ap_mm": None if chord is None else round(max(chord - cfg.canal_ap_bias_mm, 0.0), 3),
                "kind": kind,
                "slice": k,
            }
    measured = sum(1 for v in per_level.values() if v["ap_mm"] is not None)
    frac = measured / len(per_level) if per_level else 0.0
    ev = [
        f"centreline from {len(points)} z bins of {cfg.canal_z_bin_mm:.0f} mm; "
        f"AP diameters on the sagittal slice nearest the centreline at {measured}/{len(per_level)} levels"
    ]
    return {
        "points": points,
        "per_level": per_level,
        "confidence": round(float(np.clip(confidence * frac, 0.0, 1.0)), 3),
        "method": "canal/1 centreline from canal class pooled in z bins; AP chord on the nearest sagittal slice",
        "evidence": ev,
    }
