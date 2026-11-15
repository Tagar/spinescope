"""Vertebral bodies, posterior-element estimates and disc margins from the learned labeller's segmentation.

``attach_geometry(study, anchors)`` runs after :func:`~spinescope.anatomy.label_spine` (the learned labeller) and
fills three additive fields:

- ``Vertebra.body`` (in ``anchors.vertebrae[name]``): body centre (3-D, pooled over every slice), mid-sagittal
  superior and inferior endplate lines (anterior and posterior corners, angle), body height at the anterior,
  middle and posterior thirds, AP depth and, when the slices span enough, the 3-D lateral width.
- ``Anchors.canal``: centreline and AP diameters at each disc and mid-body (:mod:`.canal_ml`).
- ``Anchors.posterior_elements``: lamina proxy and spinous tip estimate on the mid-sagittal plane. These are
  approximations with no training labels; see ``docs/labeling/lab15_vertebrae.md`` for the limits.

Conventions. LPS mm everywhere. Body ``j`` has ``j`` discs below it (j = 1 is the body above the lowest disc,
L5 when the lowest disc is L5-S1; j = 0 is the body below the lowest disc, S1), the convention of
:func:`~spinescope.anatomy.numbering.vertebra_name`. Endplate geometry is measured on the mid-sagittal plane in
the plane's (down, post) mm coordinates; the 3-D centre and width pool the body class over all slices.

Only the learned labeller provides body and canal probabilities. For the classical labeller the fields stay None.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..config import DEFAULT, Config
from ..study import Study
from .anchors import Anchors, Vertebra, split_disc
from .canal_ml import (
    canal_geometry,
    canal_mask_plane,
    largest_component,
    posterior_canal_edge,
    row_heights_mm,
)
from .ml.grid import PlaneGrid, resample_slice
from .ml.labeller import BODY, SegmentedStack, segment_stack
from .ml.postprocess import mid_sagittal

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

GEOMETRY_METHOD = (
    "vertebrae/1 body endplates, heights and 3-D centre; canal/1 centreline and AP; posterior approximations"
)

#: What the posterior-element fields do and do not represent (also written into the output).
POSTERIOR_LIMITS: tuple[str, ...] = (
    "no training labels for posterior elements: lamina and spinous tip are not validated",
    "lamina proxy is the posterior edge of the canal at mid-body height, not the lamina surface",
    "spinous tip is the posterior-most intensity edge within the search depth; it can be the skin-fat boundary",
    "only the mid-sagittal plane is used; off-midline elements and facets are not measured",
)


# ---------------------------------------------------------------------------- 2-D body geometry (pure)
def _fit_line(p: FloatArray, d: FloatArray) -> tuple[float, float]:
    """Least-squares line ``d = a + b p``; returns ``(a, b)``."""
    b, a = np.polyfit(p, d, 1)
    return float(a), float(b)


def body_plane_geometry(
    mask: BoolArray, grid: PlaneGrid, sup: bool, inf: bool, cfg: Config = DEFAULT
) -> dict[str, Any] | None:
    """Endplate lines, corners, heights and AP depth of one body on a mid-sagittal plane.

    ``mask`` is the body on ``grid`` (largest component). ``sup`` / ``inf`` say whether a disc bounds the body
    above / below, i.e. whether its top / bottom boundary is an endplate. Points are returned in the plane's
    (down, post) mm coordinates; use :meth:`PlaneGrid.to_lps` for LPS. None when the body is too small.
    """
    if np.count_nonzero(mask) < 3:
        return None
    present = np.nonzero(mask.any(axis=0))[0]
    top = mask[:, present].argmax(axis=0)
    bot = mask.shape[0] - 1 - mask[::-1][:, present].argmax(axis=0)
    d_top, post = grid.mm_of_index(top, present)
    d_bot, _ = grid.mm_of_index(bot, present)
    p_lo, p_hi = float(np.min(post)), float(np.max(post))
    span = p_hi - p_lo
    ap = span + grid.res_mm
    if ap < cfg.vert_min_ap_mm:
        return None
    core = (post >= p_lo + cfg.vert_core_fraction * span) & (post <= p_hi - cfg.vert_core_fraction * span)
    if core.sum() < 3:
        core = np.ones_like(post, dtype=bool)
    out: dict[str, Any] = {"ap_mm": float(ap), "n_columns": int(present.size)}
    lines: dict[str, tuple[float, float]] = {}
    if sup:
        lines["sup"] = _fit_line(post[core], d_top[core])
    if inf:
        lines["inf"] = _fit_line(post[core], d_bot[core])
    for key, (a, b) in lines.items():
        out[key] = {
            "anterior": [a + b * p_lo, p_lo],
            "posterior": [a + b * p_hi, p_hi],
            "angle_deg": float(np.degrees(np.arctan(b))),
        }
    if "sup" in lines:  # AP depth = length of the superior endplate between its corners
        s_b = lines["sup"][1]
        out["ap_mm"] = float(np.hypot(span, s_b * span))
    if "sup" in lines and "inf" in lines:
        s_a, s_b = lines["sup"]
        i_a, i_b = lines["inf"]
        cos_t = float(np.cos(np.arctan(s_b)))
        heights = {}
        for name, frac in (("anterior", 1.0 / 6.0), ("middle", 0.5), ("posterior", 5.0 / 6.0)):
            p = p_lo + frac * span
            gap = (i_a + i_b * p) - (s_a + s_b * p)  # vertical distance, inferior below superior
            heights[name] = float(gap * cos_t)  # perpendicular to the superior endplate
        out["height_mm"] = heights
    return out


def _plane_point_lps(grid: PlaneGrid, pt: list[float]) -> list[float]:
    return [float(v) for v in grid.to_lps(pt[0], pt[1])]


def _body_mask(seg: SegmentedStack, k: int, zone: NDArray[np.intp], j: int, cfg: Config) -> BoolArray:
    """Body pixels of body ``j`` on slice ``k`` (zone = number of discs below each pixel), largest component."""
    m = (seg.probs[k, BODY] > cfg.ml_prob_threshold) & (zone == j)
    if m.sum() * seg.grids[k].res_mm ** 2 < cfg.ml_body_min_area_mm2:
        return np.zeros_like(m, dtype=bool)
    return largest_component(m)


def _slice_zones(seg: SegmentedStack, disc_z: FloatArray) -> list[NDArray[np.intp]]:
    """Per slice, the number of discs (ascending ``disc_z``) lying below each pixel centre."""
    out = []
    for g in seg.grids:
        ii, jj = np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1]), indexing="ij")
        down, post = g.mm_of_index(ii, jj)
        z = g.to_lps(down, post)[..., 2]
        out.append(np.searchsorted(disc_z, z))
    return out


def pooled_body(
    seg: SegmentedStack, zones: list[NDArray[np.intp]], j: int, cfg: Config
) -> tuple[FloatArray, FloatArray, NDArray[np.intp], dict[int, int]]:
    """Body-class pixels of body ``j`` over every slice.

    Returns the LPS (n, 3) points, their body probabilities, the slice index of each point and the pixel count of
    each slice that has any. All zone pixels are used: a largest-component cut or an area floor per slice dropped
    lateral body parts and biased the 3-D centre by 4-6 mm (SPIDER check, lab15).
    """
    pts: list[FloatArray] = []
    probs: list[FloatArray] = []
    ids: list[NDArray[np.intp]] = []
    counts: dict[int, int] = {}
    for k, g in enumerate(seg.grids):
        m = (seg.probs[k, BODY] > cfg.ml_prob_threshold) & (zones[k] == j)
        if not m.any():
            continue
        ii, jj = np.nonzero(m)
        down, post = g.mm_of_index(ii, jj)
        pts.append(np.asarray(g.to_lps(down, post), dtype=np.float64))
        probs.append(seg.probs[k, BODY][ii, jj].astype(np.float64))
        ids.append(np.full(ii.size, k, dtype=np.intp))
        counts[k] = int(ii.size)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.intp), {}
    return np.concatenate(pts), np.concatenate(probs), np.concatenate(ids), counts


def _body_width_mm(
    pts: FloatArray,
    ids: NDArray[np.intp],
    counts: dict[int, int],
    normal: FloatArray,
    seg: SegmentedStack,
    cfg: Config,
) -> tuple[float | None, str]:
    """3-D lateral width of a body from the slices that carry at least ``vert_width_slice_fraction`` of the largest
    slice count (lateral slices hold the spurious body probability), needing a slice span of ``vert_width_min_span_mm``."""
    if not counts:
        return None, "no body"
    top = max(counts.values())
    keep = [k for k, c in counts.items() if c >= cfg.vert_width_slice_fraction * top]
    pos = np.asarray(seg.stack.positions, dtype=np.float64)[keep]
    span = float(np.max(pos) - np.min(pos)) if len(keep) > 1 else 0.0
    if span < cfg.vert_width_min_span_mm:
        return None, f"slices span {span:.0f} mm (need {cfg.vert_width_min_span_mm:.0f} mm)"
    sel = np.isin(ids, np.asarray(keep, dtype=np.intp))
    lat = pts[sel] @ normal
    return float(np.percentile(lat, 99.5) - np.percentile(lat, 0.5)), f"{len(keep)} slices over {span:.0f} mm"


def body_name(j: int, disc_labels_bottom_up: list[str]) -> str:
    """Name of body ``j`` (discs below it = j) from the final disc labels, bottom-up.

    The body below the lowest disc is the lower vertebra of that disc (S1 for L5-S1); the body above disc ``i``
    (bottom-up index) is the upper vertebra of that disc, which is the body between discs ``i`` and ``i+1``.
    """
    if not disc_labels_bottom_up:
        raise ValueError("no discs")
    if j == 0:
        return split_disc(disc_labels_bottom_up[0])[1]
    return split_disc(disc_labels_bottom_up[j - 1])[0]


def _body_record(
    name: str,
    j: int,
    plane: dict[str, Any] | None,
    grid: PlaneGrid,
    pooled: tuple[FloatArray, FloatArray, NDArray[np.intp], dict[int, int]],
    stack_normal: FloatArray,
    seg: SegmentedStack,
    cfg: Config,
) -> dict[str, Any] | None:
    pts, probs, ids, counts = pooled
    if pts.shape[0] == 0 and plane is None:
        return None
    centre = [float(v) for v in pts.mean(axis=0)] if pts.shape[0] else None
    width, width_note = (
        (None, "no body pixels")
        if not pts.shape[0]
        else _body_width_mm(pts, ids, counts, stack_normal, seg, cfg)
    )
    rec: dict[str, Any] = {
        "level": name,
        "discs_below": j,
        "centre": centre,
        "centre_plane": None,
        "superior_endplate": None,
        "inferior_endplate": None,
        "height_mm": None,
        "ap_depth_mm": None,
        "width_mm": width,
        "complete": False,
        "confidence": 0.0,
        "evidence": [],
    }
    if plane is not None:
        for key, out_key in (("sup", "superior_endplate"), ("inf", "inferior_endplate")):
            if key in plane:
                e = plane[key]
                rec[out_key] = {
                    "anterior": _plane_point_lps(grid, e["anterior"]),
                    "posterior": _plane_point_lps(grid, e["posterior"]),
                    "angle_deg": round(e["angle_deg"], 3),
                }
        if "height_mm" in plane:
            rec["height_mm"] = {key: round(v, 3) for key, v in plane["height_mm"].items()}
            rec["height_mm"]["mean"] = round(float(np.mean(list(plane["height_mm"].values()))), 3)
        rec["ap_depth_mm"] = round(float(plane["ap_mm"]), 3)
        rec["complete"] = bool("sup" in plane and "inf" in plane)
        rec["evidence"].append(
            f"mid-sagittal body: {plane['n_columns']} columns, "
            f"{'both endplates' if rec['complete'] else 'one endplate (the body touches the field edge or a missing disc)'}"
        )
    mean_p = float(probs.mean()) if probs.size else 0.0
    rec["confidence"] = round(mean_p * (1.0 if rec["complete"] else 0.5), 3)
    rec["evidence"].append(
        f"3-D centre pooled over {pts.shape[0]} body pixels on the slices (body probability {mean_p:.2f})"
    )
    if width is None:
        rec["evidence"].append(f"3-D width not measured: {width_note}")
    return rec


# ------------------------------------------------------------------------ posterior elements (approx.)
def posterior_elements(
    seg: SegmentedStack, mids: list[tuple[str, float, int]], cfg: Config = DEFAULT
) -> dict[str, Any]:
    """Lamina proxy and spinous tip estimate per body (approximations, see ``POSTERIOR_LIMITS``).

    ``mids`` is ``[(body name, z of the mid-body height, slice index of that body)]``. On the body's own sagittal
    slice the lamina proxy is the posterior canal edge on the mid-height row. The spinous tip is the posterior-most
    strong intensity edge between ``spinous_gap_mm`` and ``spinous_search_mm`` behind that edge, along the
    window-normalised intensity of the row (mean of three rows, Gaussian-smoothed).
    """
    lo, hi = seg.window  # the same per-volume window the network saw
    res = seg.grids[0].res_mm
    sig = max(cfg.spinous_smooth_mm / res, 0.1)
    laminae: dict[str, Any] = {}
    tips: dict[str, Any] = {}
    masks: dict[int, BoolArray] = {}
    for name, z, k in mids:
        grid = seg.grids[k]
        if k not in masks:
            masks[k] = canal_mask_plane(seg, k, cfg)
        zr = row_heights_mm(grid)
        row = int(np.argmin(np.abs(zr - z)))
        edge = posterior_canal_edge(masks[k], row)
        if edge is None:
            continue
        down, post = grid.mm_of_index(np.array([row]), np.array([edge]))
        laminae[name] = {
            "point": _plane_point_lps(grid, [float(down[0]), float(post[0])]),
            "slice": k,
            "row_z_mm": float(z),
        }
        inten = resample_slice(seg.stack, grid, order=1).astype(np.float64)
        inten = np.clip((inten - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        r0, r1 = max(row - 1, 0), min(row + 2, grid.shape[0])
        profile = ndimage.gaussian_filter1d(inten[r0:r1].mean(axis=0), sig)
        grad = np.gradient(profile) / res
        start = edge + round(cfg.spinous_gap_mm / res)
        stop = min(edge + round(cfg.spinous_search_mm / res), grid.shape[1] - 1)
        if stop <= start:
            continue
        cand = np.nonzero(np.abs(grad[start : stop + 1]) >= cfg.spinous_edge_per_mm)[0]
        if cand.size == 0:
            continue
        col = start + int(cand.max())
        d2, p2 = grid.mm_of_index(np.array([row]), np.array([col]))
        tips[name] = {
            "point": _plane_point_lps(grid, [float(d2[0]), float(p2[0])]),
            "slice": k,
            "depth_behind_canal_mm": round(float((col - edge) * res), 2),
            "confidence": 0.2,
        }
    ev = [f"lamina proxy at {len(laminae)} body mid-height(s); spinous tip estimated at {len(tips)}"]
    return {
        "lamina_proxy": laminae,
        "spinous_tip": tips,
        "confidence": round(0.5 if laminae else 0.0, 3),
        "limits": list(POSTERIOR_LIMITS),
        "method": "posterior/1 intensity-edge approximation on each body's sagittal slice (unvalidated)",
        "evidence": ev,
    }


# ------------------------------------------------------------------------------------- the hook
def measure_geometry(seg: SegmentedStack, anchors: Anchors, cfg: Config = DEFAULT) -> Anchors:
    """Fill ``vertebrae[*].body``, ``canal`` and ``posterior_elements`` of ``anchors`` from a segmentation.

    Needs the final disc labels in ``anchors`` (after numbering and fusion); names come from those labels. Each
    body is measured on the sagittal slice nearest its own pooled centre (the spine is laterally displaced across a
    stack, so one mid-plane is not enough). Missing bodies or canal give fewer entries, never an exception.
    """
    if not anchors.discs:
        anchors.warnings.append("vertebral geometry not measured: no discs")
        return anchors
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        anchors.warnings.append("vertebral geometry not measured: no spinal canal segmented")
        return anchors
    order = anchors.disc_labels()[::-1]  # bottom-up (ascending z)
    disc_z_sorted = np.sort(np.array([anchors.discs[lab].center[2] for lab in order], dtype=np.float64))
    n = len(order)
    zones = _slice_zones(seg, disc_z_sorted)
    normal = np.asarray(seg.stack.normal, dtype=np.float64)
    positions = np.asarray(seg.stack.positions, dtype=np.float64)
    body_mids: list[tuple[str, float, int]] = []
    for j in range(n + 1):
        name = body_name(j, order)
        pooled = pooled_body(seg, zones, j, cfg)
        pts = pooled[0]
        if pts.shape[0] == 0:
            continue
        u_c = float((pts @ normal).mean())
        kb = int(np.argmin(np.abs(positions - u_c)))  # the sagittal slice through this body
        grid = seg.grids[kb]
        m_plane = _body_mask(seg, kb, zones[kb], j, cfg)
        sup, inf = j < n, j >= 1
        plane = body_plane_geometry(m_plane, grid, sup, inf, cfg) if m_plane.any() else None
        rec = _body_record(name, j, plane, grid, pooled, normal, seg, cfg)
        if rec is None:
            continue
        rec["slice"] = kb
        if plane is not None:
            rec["centre_plane"] = _plane_point_lps(grid, [float(v) for v in _plane_centre(m_plane, grid)])
        if rec["complete"]:
            zb = float(0.5 * (disc_z_sorted[j - 1] + disc_z_sorted[j]))
            body_mids.append((name, zb, kb))
        if name in anchors.vertebrae:
            anchors.vertebrae[name].body = rec
        else:
            c = rec["centre"] if rec["centre"] is not None else rec.get("centre_plane")
            if c is None:
                continue
            anchors.vertebrae[name] = Vertebra(
                center=[float(v) for v in c],
                confidence=float(rec["confidence"]),
                evidence=["body measured from the learned labeller (no earlier centre)"],
                body=rec,
            )
    disc_pairs = [(lab, float(anchors.discs[lab].center[2])) for lab in anchors.discs]
    anchors.canal = canal_geometry(
        seg, mid.k, disc_pairs, [(nm, z) for nm, z, _ in body_mids], cfg, confidence=float(mid.confidence)
    )
    anchors.posterior_elements = posterior_elements(seg, body_mids, cfg)
    return anchors


def _plane_centre(mask: BoolArray, grid: PlaneGrid) -> FloatArray:
    ii, jj = np.nonzero(mask)
    down, post = grid.mm_of_index(ii, jj)
    return np.array([float(down.mean()), float(post.mean())])


def attach_geometry(study: Study, anchors: Anchors, config: Config | None = None) -> Anchors:
    """Run :func:`measure_geometry` on the study's best sagittal series (the learned labeller's own input).

    Failures add a warning to ``anchors`` and leave the labels unchanged.
    """
    cfg = config or DEFAULT
    stack = study.best("sag")
    if stack is None:
        return anchors
    try:
        seg = segment_stack(stack, cfg)
        measure_geometry(seg, anchors, cfg)
    except Exception as exc:  # the geometry is an addition: never let it sink the labels
        log.warning("vertebral geometry failed: %s", exc)
        anchors.warnings.append(f"vertebral geometry not measured ({type(exc).__name__})")
    return anchors


def geometry_summary(anchors: Anchors) -> dict[str, Any]:
    """Compact view of the geometry fields (for reports and tests)."""
    bodies = {n: v.body for n, v in anchors.vertebrae.items() if v.body}
    return {
        "bodies": sorted(bodies),
        "canal_points": len((anchors.canal or {}).get("points", [])),
        "canal_levels": sorted((anchors.canal or {}).get("per_level", {})),
        "posterior_tips": sorted((anchors.posterior_elements or {}).get("spinous_tip", {})),
    }


__all__ = [
    "GEOMETRY_METHOD",
    "POSTERIOR_LIMITS",
    "attach_geometry",
    "body_plane_geometry",
    "geometry_summary",
    "measure_geometry",
    "pooled_body",
    "posterior_elements",
]
