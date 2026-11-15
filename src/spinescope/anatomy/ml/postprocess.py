"""Turn network probabilities into :class:`~spinescope.anatomy.anchors.Anchors`.

Every step works on the probability maps and the plane geometry only; thresholds come from
:class:`~spinescope.config.Config`. Pure numpy/scipy, so it is unit-tested with synthetic maps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ...config import DEFAULT, Config
from ..anchors import Anchors, Disc, Vertebra
from ..numbering import disc_names, vertebra_name
from .grid import PlaneGrid, resample_slice
from .labeller import BODY, CANAL, DISC, METHOD, SACRUM, SegmentedStack

log = logging.getLogger(__name__)


@dataclass
class MidSagittal:
    k: int  # analysed slice nearest to the plane
    x_lps: float  # LPS x of the plane (parabolic refinement of the canal-area peak)
    confidence: float
    evidence: str


def canal_areas_mm2(seg: SegmentedStack, cfg: Config = DEFAULT) -> NDArray[np.float64]:
    """Canal area (mm^2) on every slice of the segmented stack."""
    thr = cfg.ml_prob_threshold
    px = cfg.ml_resolution_mm**2
    return np.array([float((p[CANAL] > thr).sum()) * px for p in seg.probs])


def mid_sagittal(seg: SegmentedStack, cfg: Config = DEFAULT) -> MidSagittal | None:
    """Mid-sagittal slice from the canal area across slices; None if no canal is segmented."""
    area = canal_areas_mm2(seg, cfg)
    if area.size == 0 or float(np.max(area)) <= 0.0:
        return None
    stack = seg.stack
    pos = np.asarray(stack.positions, dtype=np.float64)  # plane coordinate along the stack normal
    k = int(np.argmax(area))
    x = float(pos[k])
    if 0 < k < len(area) - 1 and area[k - 1] > 0 and area[k + 1] > 0 and pos[k + 1] != pos[k - 1]:
        xs = pos[k - 1 : k + 2]
        ys = area[k - 1 : k + 2]
        a, b, _c = np.polyfit(xs - pos[k], ys, 2)
        if a < 0:  # concave: a true peak
            vertex = -b / (2 * a)
            x = float(pos[k] + np.clip(vertex, xs[0] - pos[k], xs[-1] - pos[k]))
    # Confidence: how much the canal area drops away from the peak (slices 2+ steps away).
    far = np.abs(np.arange(len(area)) - k) >= 2
    conf_shape = float(np.clip(1.0 - area[far].mean() / area[k], 0.0, 1.0)) if far.any() else 0.5
    normal = np.asarray(stack.normal, dtype=np.float64)
    point = stack.ipp[k] + (x - pos[k]) * normal
    ev = (
        f"canal area peak {area[k]:.0f} mm2 on slice {k} of {len(area)}; "
        f"plane refined to x={point[0]:.1f} mm (LPS)"
    )
    return MidSagittal(k=k, x_lps=float(point[0]), confidence=conf_shape, evidence=ev)


def _components(mask: NDArray[np.bool_], grid: PlaneGrid, min_area_mm2: float) -> list[NDArray[np.intp]]:
    """Index arrays (rows, cols) of each connected component with at least ``min_area_mm2``."""
    lab, n = ndimage.label(mask)
    out: list[NDArray[np.intp]] = []
    px = grid.res_mm**2
    for lid in range(1, n + 1):
        rr, cc = np.nonzero(lab == lid)
        if rr.size * px >= min_area_mm2:
            out.append(np.stack([rr, cc]))
    return out


def _disc_from_component(
    idx: NDArray[np.intp], prob: NDArray[np.float32], grid: PlaneGrid, cfg: Config
) -> tuple[Disc, float, float] | None:
    """Centre, principal normal and AP margins of one disc component. Returns (disc, height, aspect)."""
    res = grid.res_mm
    down, post = grid.mm_of_index(idx[0], idx[1])
    pts = np.stack([down, post], axis=1)
    centre = pts.mean(axis=0)
    rel = pts - centre
    if len(pts) < 3:
        return None
    _w, vecs = np.linalg.eigh(np.cov(rel.T))
    minor, major = vecs[:, 0], vecs[:, 1]  # eigh: ascending eigenvalues
    if minor[0] > 0:  # superior = negative "down" coordinate
        minor = -minor
    if major[1] < 0:  # posterior = positive "post" coordinate
        major = -major
    s = rel @ minor
    a = rel @ major
    height = float(s.max() - s.min()) + res
    length = float(a.max() - a.min()) + res
    aspect = length / max(height, res)
    band = np.abs(s) <= max(height / 4.0, res)  # mid-height band for the margins
    a_band = a[band] if band.any() else a
    ant = centre + float(a_band.min()) * major
    post_pt = centre + float(a_band.max()) * major
    n_lps = minor[0] * grid.e_down + minor[1] * grid.e_post
    if n_lps[2] < 0:
        n_lps = -n_lps
    n_lps = n_lps / np.linalg.norm(n_lps)
    mean_p = float(prob[idx[0], idx[1]].mean())
    disc = Disc(
        center=[float(v) for v in grid.to_lps(centre[0], centre[1])],
        normal=[float(v) for v in n_lps],
        anterior=[float(v) for v in grid.to_lps(ant[0], ant[1])],
        posterior=[float(v) for v in grid.to_lps(post_pt[0], post_pt[1])],
        height_mm=height,
        confidence=mean_p,
        evidence=[f"disc class probability {mean_p:.2f}; {int(idx.shape[1])} px; aspect {aspect:.2f}"],
    )
    return disc, height, aspect


def extract_discs(seg: SegmentedStack, k: int, cfg: Config = DEFAULT) -> tuple[list[Disc], list[str]]:
    """Disc instances on slice ``k`` (bottom-up order is applied by the caller). Returns (discs, warnings)."""
    grid = seg.grids[k]
    prob = seg.probs[k, DISC]
    warns: list[str] = []
    comps = _components(prob > cfg.ml_prob_threshold, grid, cfg.ml_disc_min_area_mm2)
    found: list[Disc] = []
    lo_h, hi_h = cfg.disc_height_mm
    for idx in sorted(comps, key=lambda c: -c.shape[1]):
        got = _disc_from_component(idx, prob, grid, cfg)
        if got is None:
            continue
        disc, height, aspect = got
        if aspect < cfg.ml_disc_min_aspect:
            warns.append(f"component of {idx.shape[1]} px rejected: not disc-shaped (aspect {aspect:.2f})")
            continue
        if not (lo_h <= height <= hi_h):
            disc.confidence *= 0.5
            disc.evidence.append(
                f"height {height:.1f} mm outside {lo_h:.0f}-{hi_h:.0f} mm: confidence halved"
            )
        # Fragments of one disc: a smaller component whose centre is close to a kept disc is dropped.
        c = np.asarray(disc.center)
        if any(np.linalg.norm(c - np.asarray(f.center)) < cfg.level_spacing_mm[0] / 2 for f in found):
            warns.append("duplicate disc fragment merged away")
            continue
        found.append(disc)
    return found, warns


def extract_bodies(
    seg: SegmentedStack, k: int, cfg: Config = DEFAULT
) -> list[tuple[NDArray[np.float64], float, int]]:
    """Vertebral body components on slice ``k``: (LPS centroid, mean probability, pixel count)."""
    grid = seg.grids[k]
    prob = seg.probs[k, BODY]
    out = []
    for idx in _components(prob > cfg.ml_prob_threshold, grid, cfg.ml_body_min_area_mm2):
        down, post = grid.mm_of_index(idx[0], idx[1])
        cen = grid.to_lps(float(down.mean()), float(post.mean()))
        out.append((cen, float(prob[idx[0], idx[1]].mean()), int(idx.shape[1])))
    return out


@dataclass
class SacrumEvidence:
    """Sacrum class on the analysed slice, below the lowest disc of that slice."""

    area_mm2: float
    centre_lps: NDArray[np.float64] | None
    fov_below_disc_mm: float | None  # field of view below the lowest disc (None when no disc on the slice)
    evidence: list[str] = field(default_factory=list)


def _lowest_disc_component(seg: SegmentedStack, k: int, cfg: Config) -> NDArray[np.intp] | None:
    """Index array of the most inferior disc component on slice ``k`` (None when there is none)."""
    grid = seg.grids[k]
    comps = _components(seg.probs[k, DISC] > cfg.ml_prob_threshold, grid, cfg.ml_disc_min_area_mm2)
    if not comps:
        return None
    return max(comps, key=lambda c: float(c[0].mean()))


def fov_coverage_rows(seg: SegmentedStack, k: int) -> NDArray[np.bool_]:
    """Boolean (rows, cols) mask of the grid of slice ``k`` that lies inside the acquired field of view."""
    stack = seg.stack
    grid0 = replace(seg.grids[k], k=0)
    ones = np.ones((1, *stack.vol.shape[1:]), np.float32)
    return np.asarray(resample_slice(stack, grid0, array=ones, order=1) > 0.5)


def sacrum_below_lowest_disc(seg: SegmentedStack, k: int, cfg: Config = DEFAULT) -> SacrumEvidence | None:
    """Sacrum class below the lowest disc of slice ``k``; None when the network has no sacrum channel.

    The area counts sacrum pixels more inferior than the lowest disc component's inferior edge, in the
    columns of that disc (the sacrum lies behind the lowest disc's inferior endplate, not beside it).
    ``fov_below_disc_mm`` is how far the acquired field of view extends below that edge in the same columns,
    so a value below ``ml_fov_below_disc_min_mm`` means the sacrum is not in view.
    """
    if seg.probs.shape[1] <= SACRUM:
        return None
    grid = seg.grids[k]
    px = grid.res_mm**2
    disc = _lowest_disc_component(seg, k, cfg)
    if disc is None:
        return SacrumEvidence(
            0.0, None, None, ["no disc on this slice: sacrum below the lowest disc not assessed"]
        )
    d_bot = int(disc[0].max())
    c0, c1 = int(disc[1].min()), int(disc[1].max())
    rows = np.arange(grid.shape[0])
    below = rows > d_bot
    sac = seg.probs[k, SACRUM] > cfg.ml_prob_threshold
    box = np.zeros(grid.shape, bool)
    box[below, c0 : c1 + 1] = True
    mask = sac & box
    area = float(mask.sum()) * px
    centre = None
    if mask.any():
        ii, jj = np.nonzero(mask)
        down, post = grid.mm_of_index(ii.mean(), jj.mean())
        centre = grid.to_lps(float(down), float(post))
    cov = fov_coverage_rows(seg, k)[:, c0 : c1 + 1]
    inside = np.nonzero(cov[below].mean(axis=1) > 0.5)[0] if below.any() else np.array([], dtype=np.intp)
    fov_mm = float(np.max(inside) + 1) * grid.res_mm if inside.size else 0.0
    ev = [f"sacrum class {area:.0f} mm2 below the lowest disc; field of view {fov_mm:.0f} mm below it"]
    return SacrumEvidence(area, centre, fov_mm, ev)


def discs_near_midline(seg: SegmentedStack, k: int, cfg: Config = DEFAULT) -> tuple[list[Disc], list[str]]:
    """Discs of the mid-sagittal slice ``k``, plus discs seen only on slices up to ``ml_disc_slice_window`` away.

    A disc at the edge of the field can be absent from the exact plane but present on a neighbour; the
    neighbour's disc is kept when no accepted disc lies within half a level spacing in the sagittal plane.
    Candidates on the mid-sagittal slice are preferred.
    """
    warns: list[str] = []
    cands: list[tuple[int, float, Disc]] = []
    w = cfg.ml_disc_slice_window
    for kk in range(max(0, k - w), min(len(seg.grids), k + w + 1)):
        ds, dw = extract_discs(seg, kk, cfg)
        if kk == k:
            warns.extend(dw)
        for d in ds:
            cands.append((abs(kk - k), -d.confidence, d))
    cands.sort(key=lambda t: (t[0], t[1]))
    kept: list[Disc] = []
    min_sep = cfg.level_spacing_mm[0] / 2
    for dk, _c, d in cands:
        cz = np.asarray(d.center)[[1, 2]]  # (y, z): distance within the sagittal plane
        if any(np.linalg.norm(cz - np.asarray(f.center)[[1, 2]]) < min_sep for f in kept):
            continue
        if dk > 0:
            d.evidence.append(f"seen on a slice {dk} away from the mid-sagittal slice (absent from it)")
        kept.append(d)
    return kept, warns


def anchors_from_segmentation(seg: SegmentedStack, cfg: Config = DEFAULT) -> Anchors:
    """Mid-sagittal discs, bodies and numbering from a :class:`SegmentedStack`."""
    anchors = Anchors(method=METHOD, source_stack=seg.stack.key)
    anchors.warnings.extend(seg.evidence)
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        anchors.warnings.append(f"{seg.stack.key}: no spinal canal segmented; spine anatomy not labelled")
        anchors.numbering = {"method": "none", "confidence": 0.0, "notes": "no canal"}
        return anchors
    anchors.midline_x = mid.x_lps
    discs, warns = discs_near_midline(seg, mid.k, cfg)
    anchors.warnings.extend(warns)
    if not discs:
        anchors.warnings.append(f"{seg.stack.key}: no disc found on the mid-sagittal slice")
        anchors.numbering = {"method": "none", "confidence": 0.0, "notes": "no disc"}
        return anchors
    # Bottom-up: discs sorted by centre z, ascending (inferior first).
    discs.sort(key=lambda d: d.center[2])
    names = disc_names(len(discs))
    labelled: dict[str, Disc] = {}
    for name, d in zip(names, discs, strict=True):
        d.evidence.append(f"level {name} by count from the lowest disc")
        labelled[name] = d
    anchors.discs = labelled
    bodies = extract_bodies(seg, mid.k, cfg)
    centres_z = np.array([d.center[2] for d in discs])
    body_by_index: dict[int, tuple[NDArray[np.float64], float, int]] = {}
    for cen, prob, npx in bodies:
        n_below = int(np.sum(centres_z < cen[2]))
        if n_below not in body_by_index or npx > body_by_index[n_below][2]:
            body_by_index[n_below] = (cen, prob, npx)
    for n_below, (cen, prob, _npx) in sorted(body_by_index.items()):
        name = vertebra_name(n_below)
        anchors.vertebrae[name] = Vertebra(
            center=[float(v) for v in cen],
            confidence=prob,
            evidence=[f"body with {n_below} disc(s) below it on the mid-sagittal slice"],
        )
    body_below = 0 in body_by_index
    sac = sacrum_below_lowest_disc(seg, mid.k, cfg)
    sacrum_seen = sac is not None and sac.area_mm2 >= cfg.ml_sacrum_min_area_mm2
    fov_mm = sac.fov_below_disc_mm if sac is not None else None
    # A field of view that ends just below the lowest disc cannot show an S1 body, whatever the network says there.
    fov_short = fov_mm is not None and fov_mm < cfg.ml_fov_below_disc_min_mm
    confirmed = (body_below or sacrum_seen) and not fov_short
    if sacrum_seen and not body_below and not fov_short and sac is not None and sac.centre_lps is not None:
        anchors.vertebrae[vertebra_name(0)] = Vertebra(
            center=[float(v) for v in sac.centre_lps],
            confidence=float(min(1.0, sac.area_mm2 / (4 * cfg.ml_sacrum_min_area_mm2))),
            evidence=["sacrum class below the lowest disc (learned sacrum channel)"],
        )
    mean_conf = float(np.mean([d.confidence for d in discs]))
    numbering_conf = (0.9 if confirmed else 0.4) * (0.5 + 0.5 * min(1.0, mean_conf))
    if fov_short:
        notes = (
            f"field of view ends {fov_mm:.0f} mm below the lowest disc, above the sacrum: "
            "lowest disc named L5-S1 without confirmation"
        )
    elif body_below:
        notes = "vertebral body visible below the lowest disc: lowest disc taken as L5-S1"
    elif sacrum_seen:
        notes = "sacrum visible below the lowest disc: lowest disc taken as L5-S1"
    else:
        notes = (
            "no vertebral body or sacrum visible below the lowest disc: lowest disc named L5-S1 unconfirmed"
        )
    anchors.numbering = {
        "method": "ml sacrum-up count",
        "confidence": float(numbering_conf),
        "notes": notes,
        "midline_confidence": mid.confidence,
        "midline_evidence": mid.evidence,
        "sacrum_confirmed": bool(sacrum_seen and not fov_short),
        "fov_below_lowest_disc_mm": fov_mm,
    }
    if sac is not None:
        anchors.numbering["sacrum_evidence"] = list(sac.evidence)
    if not confirmed:
        if fov_short:
            anchors.warnings.append(
                "numbering unconfirmed: field of view ends "
                f"{fov_mm:.0f} mm below the lowest disc (sacrum not in view)"
            )
        else:
            anchors.warnings.append(
                "numbering unconfirmed: no vertebral body or sacrum below the lowest disc"
            )
    return anchors


__all__ = [
    "MidSagittal",
    "anchors_from_segmentation",
    "canal_areas_mm2",
    "discs_near_midline",
    "extract_bodies",
    "extract_discs",
    "mid_sagittal",
]
