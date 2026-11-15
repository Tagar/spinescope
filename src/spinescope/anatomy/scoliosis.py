"""Scoliosis-aware labelling: the lateral curve of the spine, and disc sampling planes that follow it.

A sagittal stack samples the body at fixed left-right positions x. With a lateral curve (degenerative scoliosis,
or axial rotation of the column) the canal and the discs at the apex lie off the mid-sagittal plane: a disc can be
absent from it, and its normal tilts laterally. This module:

1. Measures the lateral position of the canal, x(z), from the canal segmentation of ALL sagittal slices. At each
   height the canal's mass-weighted centroid over the slices that contain it gives x(z), binned every
   ``Config.scoliosis_knot_mm``. A Cobb-like angle (the spread of the canal's tangent angle), an apex and a
   confidence come from that curve (:func:`measure_curve`). The record is stored as
   ``Anchors.alignment['coronal']``.
2. When the curve is significant, measures every disc on the sagittal slice nearest to its own lateral centre
   (:func:`discs_on_curve`). Each disc gets a 3-D centre and a 3-D normal, so a disc at the apex is not lost
   and a lateral tilt is not read as a mid-plane tilt.

When the curve is negligible the labeller keeps its single mid-sagittal plane path. Every threshold is in
:class:`~spinescope.config.Config` (``scoliosis_*``).

Not done here: the labeller segments sagittal stacks only, so axial and coronal localizers give no lateral cue,
and the vertebral-body search stays on the mid-sagittal plane.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..config import DEFAULT, Config
from .anchors import Disc

if TYPE_CHECKING:  # pragma: no cover
    from .ml.labeller import SegmentedStack

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


@dataclass
class LateralCurve:
    """Lateral curve of the canal, x(z) in LPS mm, with a Cobb-like angle, its apex and a confidence."""

    valid: bool
    knots_z: FloatArray  # height of each knot (LPS z, mm), ascending
    knots_x: FloatArray  # canal centroid x at each knot (LPS mm), raw
    smooth_x: FloatArray  # the same knots after gap filling and smoothing, on a regular grid
    grid_z: FloatArray  # regular height grid of ``smooth_x``
    support: IntArray  # sagittal slices that contain the canal at each knot
    coverage: float  # fraction of the canal's height range that has a knot
    cobb_deg: float | None
    apex_z: float | None
    apex_x: float | None
    amplitude_mm: float  # largest displacement of the curve from the chord between its ends
    direction: str  # "left" or "right" (LPS +x is the patient's left) at the apex, or ""
    confidence: float
    active: bool  # significant: Cobb, amplitude and confidence all clear their thresholds
    method: str = "canal centroid over sagittal slices"
    evidence: list[str] = field(default_factory=list)

    def x_at(self, z: float | FloatArray) -> FloatArray:
        """Canal x (LPS mm) at height ``z`` from the smoothed curve (clamped at the ends); NaN when not valid."""
        zz = np.asarray(z, dtype=np.float64)
        if not self.valid:
            return np.full(zz.shape, np.nan)
        return np.interp(zz, self.grid_z, self.smooth_x)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record (no arrays): the form stored in ``Anchors.alignment['coronal']``."""

        def _r(v: float | None, nd: int = 2) -> float | None:
            return None if v is None or not np.isfinite(v) else round(float(v), nd)

        return {
            "valid": bool(self.valid),
            "active": bool(self.active),
            "method": self.method,
            "cobb_deg": _r(self.cobb_deg, 1),
            "apex_z_mm": _r(self.apex_z, 1),
            "apex_x_mm": _r(self.apex_x, 1),
            "amplitude_mm": _r(self.amplitude_mm, 1),
            "direction": self.direction,
            "confidence": _r(self.confidence, 3),
            "coverage": _r(self.coverage, 3),
            "knots": [
                [_r(z, 1), _r(x, 2), int(s)]
                for z, x, s in zip(self.knots_z, self.knots_x, self.support, strict=True)
            ],
            "evidence": list(self.evidence),
        }


def _empty(method: str, why: str) -> LateralCurve:
    e = np.zeros(0)
    return LateralCurve(
        valid=False,
        knots_z=e,
        knots_x=e,
        smooth_x=e,
        grid_z=e,
        support=np.zeros(0, dtype=np.intp),
        coverage=0.0,
        cobb_deg=None,
        apex_z=None,
        apex_x=None,
        amplitude_mm=0.0,
        direction="",
        confidence=0.0,
        active=False,
        method=method,
        evidence=[why],
    )


def slice_spacing_mm(seg: SegmentedStack) -> float:
    """Median distance between neighbouring sagittal slices (mm); 5 mm when the stack has a single slice."""
    pos = np.asarray(seg.stack.ipp, dtype=np.float64) @ np.asarray(seg.stack.normal, dtype=np.float64)
    if pos.size < 2:
        return 5.0
    return float(np.median(np.abs(np.diff(np.sort(pos)))))


def canal_samples(
    seg: SegmentedStack, cfg: Config = DEFAULT
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray]:
    """Canal rows of every sagittal slice: (z, x, weight, slice index) with LPS mm and canal pixel counts.

    A row is one height of one slice. Its weight is the number of canal pixels in it, which is proportional to
    the canal's cross-section in that slice, so the slice nearest the canal's centre carries the most weight.
    """
    from .ml.labeller import CANAL  # local: the ml package imports postprocess, which imports this module

    thr = cfg.ml_prob_threshold
    zs: list[FloatArray] = []
    xs: list[FloatArray] = []
    ws: list[FloatArray] = []
    ks: list[IntArray] = []
    for k, g in enumerate(seg.grids):
        mask = seg.probs[k, CANAL] > thr
        cnt = mask.sum(axis=1)
        rows = np.nonzero(cnt)[0]
        if rows.size == 0:
            continue
        mid_col = np.full(rows.size, mask.shape[1] // 2, dtype=np.intp)
        lps = g.lps_of_index(rows, mid_col)
        zs.append(np.asarray(lps[:, 2], dtype=np.float64))
        xs.append(np.asarray(lps[:, 0], dtype=np.float64))
        ws.append(cnt[rows].astype(np.float64))
        ks.append(np.full(rows.size, k, dtype=np.intp))
    if not zs:
        empty = np.zeros(0)
        return empty, empty, empty, np.zeros(0, dtype=np.intp)
    return np.concatenate(zs), np.concatenate(xs), np.concatenate(ws), np.concatenate(ks)


def knots_from_samples(
    z: FloatArray, x: FloatArray, w: FloatArray, k: IntArray, cfg: Config = DEFAULT
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Canal x(z) knots: per height bin of ``scoliosis_knot_mm``, the weighted centroid of x over the rows.

    The centroid is the same statistic as the mask truth (the centroid of the canal voxels in a height bin), so the
    network curve and the mask curve are comparable. A bin needs ``scoliosis_min_slices`` distinct slices: one slice
    gives only its own position, not a lateral centre. (A log-parabola sub-slice vertex was tried and not kept: it
    matched the mask curve less well on SPIDER validation.)

    Returns (knot heights, knot x, slice count per knot), ascending in height.
    """
    if z.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.intp)
    step = cfg.scoliosis_knot_mm
    z0 = np.floor(float(np.min(z)) / step) * step
    idx = np.floor((z - z0) / step).astype(np.intp)
    nb = int(np.max(idx)) + 1
    sw = np.bincount(idx, weights=w, minlength=nb)
    swx = np.bincount(idx, weights=w * x, minlength=nb)
    pairs = np.unique(np.stack([idx, k.astype(np.intp)], axis=1), axis=0)
    support = np.bincount(pairs[:, 0], minlength=nb)
    keep = (support >= cfg.scoliosis_min_slices) & (sw > 0)
    centres = z0 + (np.arange(nb) + 0.5) * step
    return centres[keep], swx[keep] / sw[keep], support[keep].astype(np.intp)


def cobb_from_knots(zk: FloatArray, xk: FloatArray, cfg: Config = DEFAULT) -> dict[str, Any] | None:
    """Cobb-like angle, apex and amplitude of a lateral curve given by its knots; None with fewer than 4 knots.

    The knots are gap-filled on a regular grid and smoothed (``scoliosis_smooth_mm``). The tangent angle theta(z)
    of the curve is measured against the vertical; the Cobb-like angle is the spread of theta over the interior of
    the curve (the ``smooth`` margin at each end is dropped), which is the angle between the two most tilted
    end vertebrae. The apex is the point farthest from the chord joining the two ends of the curve.
    """
    if zk.size < 4:
        return None
    order = np.argsort(zk)
    zk, xk = zk[order], xk[order]
    step = cfg.scoliosis_knot_mm
    grid = np.arange(zk[0], zk[-1] + 0.5 * step, step)
    xs = np.interp(grid, zk, xk)
    sigma = cfg.scoliosis_smooth_mm / step
    xsm = ndimage.gaussian_filter1d(xs, sigma, mode="nearest") if sigma > 0 else xs
    theta = np.degrees(np.arctan(np.gradient(xsm, grid)))
    trim = int(np.ceil(cfg.scoliosis_smooth_mm / step))
    if grid.size <= 2 * trim + 3:
        return None
    inner = slice(trim, grid.size - trim)
    g, xi, th = grid[inner], xsm[inner], theta[inner]
    cobb = float(th.max() - th.min())
    chord = xi[0] + (xi[-1] - xi[0]) * (g - g[0]) / max(g[-1] - g[0], 1e-9)
    dev = xi - chord
    i = int(np.argmax(np.abs(dev)))
    amp = float(abs(dev[i]))
    return {
        "cobb_deg": cobb,
        "apex_z": float(g[i]),
        "apex_x": float(xi[i]),
        "amplitude_mm": amp,
        "direction": "left" if dev[i] > 0 else "right",
        "grid_z": grid,
        "smooth_x": xsm,
    }


def measure_curve(seg: SegmentedStack, cfg: Config = DEFAULT) -> LateralCurve:
    """Lateral curve of the canal from all sagittal slices of the segmentation. Never raises: a canal too short or
    seen on too few slices gives an invalid, inactive curve with the reason in ``evidence``.

    Confidence = coverage x (1 - residual / slice spacing): the canal's knots must cover its height range, and
    scatter about the smoothed curve must stay below one slice spacing (the resolution of the measurement).
    The curve is active when Cobb >= ``scoliosis_min_cobb_deg``, amplitude >= ``scoliosis_min_amplitude_mm`` and
    confidence >= ``scoliosis_min_confidence``.
    """
    z, x, w, k = canal_samples(seg, cfg)
    if z.size == 0:
        return _empty("canal centroid over sagittal slices", "no canal in any sagittal slice")
    zk, xk, sup = knots_from_samples(z, x, w, k, cfg)
    if zk.size < 4:
        return _empty(
            "canal centroid over sagittal slices",
            f"{zk.size} canal knots with {cfg.scoliosis_min_slices}+ slices: too few for a lateral curve",
        )
    res = cobb_from_knots(zk, xk, cfg)
    if res is None:
        return _empty("canal centroid over sagittal slices", "canal too short for a curve")
    grid = res["grid_z"]
    full = max(1, int(np.ceil((float(np.max(z)) - float(np.min(z))) / cfg.scoliosis_knot_mm)) + 1)
    coverage = float(min(1.0, zk.size / full))
    fit = np.interp(zk, grid, res["smooth_x"])
    rms = float(np.sqrt(np.mean((xk - fit) ** 2)))
    spacing = slice_spacing_mm(seg)
    confidence = float(coverage * np.clip(1.0 - rms / max(spacing, 1e-6), 0.0, 1.0))
    cobb = float(res["cobb_deg"])
    amp = float(res["amplitude_mm"])
    active = (
        cobb >= cfg.scoliosis_min_cobb_deg
        and amp >= cfg.scoliosis_min_amplitude_mm
        and confidence >= cfg.scoliosis_min_confidence
    )
    ev = [
        f"canal x(z) from {zk.size} knots every {cfg.scoliosis_knot_mm:.0f} mm over "
        f"{int(np.min(sup))}-{int(np.max(sup))} slices each; slice spacing {spacing:.1f} mm",
        f"Cobb-like {cobb:.1f} deg; apex {amp:.1f} mm from the chord at z={res['apex_z']:.0f} mm "
        f"({res['direction']})",
        f"residual {rms:.1f} mm about the smoothed curve; coverage {coverage:.2f}",
    ]
    if not active:
        ev.append(
            "negligible or unreliable: single mid-sagittal plane path kept "
            f"(needs Cobb >= {cfg.scoliosis_min_cobb_deg:.0f} deg, amplitude >= "
            f"{cfg.scoliosis_min_amplitude_mm:.0f} mm, confidence >= {cfg.scoliosis_min_confidence:.2f})"
        )
    return LateralCurve(
        valid=True,
        knots_z=zk,
        knots_x=xk,
        smooth_x=res["smooth_x"],
        grid_z=grid,
        support=sup,
        coverage=coverage,
        cobb_deg=cobb,
        apex_z=float(res["apex_z"]),
        apex_x=float(res["apex_x"]),
        amplitude_mm=amp,
        direction=str(res["direction"]),
        confidence=confidence,
        active=bool(active),
        evidence=ev,
    )


def _pooled_3d(
    seg: SegmentedStack, kk: int, centre: FloatArray, height_mm: float, cfg: Config = DEFAULT
) -> tuple[FloatArray, FloatArray, int] | None:
    """3-D centroid and normal of the disc at ``centre`` (on slice ``kk``), pooled over the slices within
    ``ml_disc_3d_half_width_mm`` of slice ``kk``. The disc is evaluated on the slice through its own lateral centre,
    so this window is symmetric about the disc, as the mid-plane pooling is about the mid-plane. Returns (centre LPS,
    unit normal with positive z, slices used) or None when fewer than three slices contain the disc.
    """
    from .ml.labeller import DISC  # local import, as in canal_samples

    thr = cfg.ml_prob_threshold
    x_slices = np.asarray(seg.stack.ipp, dtype=np.float64)[:, 0]
    down_c, post_c = seg.grids[kk].mm_of(centre)
    half_h = height_mm / 2.0 + 3.0
    pts: list[FloatArray] = []
    for jj, g in enumerate(seg.grids):
        if abs(float(x_slices[jj]) - float(x_slices[kk])) > cfg.ml_disc_3d_half_width_mm:
            continue
        ii, jc = np.nonzero(seg.probs[jj, DISC] > thr)
        if ii.size == 0:
            continue
        dn, pn = g.mm_of_index(ii, jc)
        sel = (np.abs(dn - down_c) <= half_h) & (np.abs(pn - post_c) <= cfg.ml_disc_3d_ap_half_mm)
        if np.any(sel):
            pts.append(g.to_lps(dn[sel], pn[sel]))
    if len(pts) < 3:
        return None
    allp = np.concatenate(pts, axis=0)
    cen = allp.mean(axis=0)
    _w, vecs = np.linalg.eigh(np.cov((allp - cen).T))
    nrm = vecs[:, 0]  # thinnest direction: the disc normal
    if nrm[2] < 0:
        nrm = -nrm
    return cen, nrm / np.linalg.norm(nrm), len(pts)


def _pick_slice(x_slices: FloatArray, target_x: float, cands: list[tuple[int, Disc]]) -> int:
    """Index in ``cands`` of the candidate on the slice whose x is nearest ``target_x``."""
    offs = [abs(float(x_slices[kk]) - target_x) for kk, _d in cands]
    return int(np.argmin(offs))


def discs_on_curve(
    seg: SegmentedStack, curve: LateralCurve, cfg: Config = DEFAULT, mid_k: int | None = None
) -> tuple[list[Disc], list[str]]:
    """Discs measured on the sagittal slice through each disc's own lateral centre (3-D centre and normal).

    Candidates are the discs of every sagittal slice that lies within ``scoliosis_disc_reach_mm`` of the canal at the
    disc's height. Pass 1 keeps, per disc, the candidate on the slice nearest the canal; candidates are the same disc
    when their sagittal (y, z) centres are within half a level spacing (the mid-plane path's test). Pass 2 moves the
    disc to the candidate on the slice nearest the 3-D lateral centre of the disc, pooled over the slices within
    ``ml_disc_3d_half_width_mm``. A disc seen on fewer than three slices is dropped. Its centre and normal are then
    the 3-D disc (as :func:`~spinescope.anatomy.ml.postprocess.refine_disc_3d`, with the normal too), and its height
    is the largest height it shows on any slice (a lateral slice cuts only part of the disc). Last, discs closer than
    one level spacing in height are resolved in favour of the more confident one.
    Discs that the mid-sagittal slice ``mid_k`` also shows (the mid-plane path's discs) are never dropped by the last
    step: only a disc the mid-plane does not show can be a lateral blob that clashes with another. Returns (discs,
    warnings); the caller sorts them and applies the usual disc filters.
    """
    from .ml.postprocess import discs_near_midline, extract_discs  # local: postprocess imports this module

    warns: list[str] = []
    x_slices = np.asarray(seg.stack.ipp, dtype=np.float64)[:, 0]
    cands: list[tuple[int, Disc]] = []
    for kk in range(len(seg.grids)):
        ds, _w = extract_discs(seg, kk, cfg)
        cands.extend((kk, d) for d in ds)

    def off(c: tuple[int, Disc]) -> float:
        return abs(float(x_slices[c[0]]) - float(curve.x_at(c[1].center[2])))

    # A slice far from the canal at this height cannot cut the disc: such candidates are partial blobs.
    cands = [c for c in cands if off(c) <= cfg.scoliosis_disc_reach_mm]
    if not cands:
        return [], warns
    order = sorted(range(len(cands)), key=lambda i: (off(cands[i]), -cands[i][1].confidence))
    min_sep = cfg.level_spacing_mm[0] / 2.0

    def same_disc(a: Disc, b: Disc) -> bool:
        # In-plane (sagittal y, z), as the mid-plane path: a posterior blob at the height of a disc is another structure.
        return float(np.linalg.norm(np.asarray(a.center)[[1, 2]] - np.asarray(b.center)[[1, 2]])) < min_sep

    accepted: list[int] = []
    for i in order:
        if any(same_disc(cands[i][1], cands[j][1]) for j in accepted):
            continue
        accepted.append(i)

    out: list[Disc] = []
    moved = 0.0
    dropped = 0
    for i in accepted:
        kk, d = cands[i]
        same = [c for c in cands if same_disc(c[1], d)]
        h = max(float(c[1].height_mm or 0.0) for c in same) or cfg.disc_height_mm[0]
        pooled = _pooled_3d(seg, kk, np.asarray(d.center, dtype=np.float64), h, cfg)
        if pooled is not None:  # pass 2: the candidate on the slice nearest the disc's own 3-D lateral centre
            kk2, d2 = same[_pick_slice(x_slices, float(pooled[0][0]), same)]
            if kk2 != kk:
                kk, d = kk2, d2
                h = max(float(c[1].height_mm or 0.0) for c in same) or cfg.disc_height_mm[0]
                pooled = _pooled_3d(seg, kk, np.asarray(d.center, dtype=np.float64), h, cfg)
        lat_off = abs(float(x_slices[kk]) - float(curve.x_at(d.center[2])))
        moved = max(moved, lat_off)
        if pooled is None:  # seen on fewer than three slices: a partial blob, not a disc (3-D convention)
            dropped += 1
            continue
        d.height_mm = h
        cen, nrm, n_sl = pooled
        shift = float(np.linalg.norm(cen - np.asarray(d.center)))
        d.center = [float(v) for v in cen]
        turn = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(nrm, d.normal))), 0.0, 1.0))))
        if cfg.scoliosis_normal_3d:
            d.normal = [float(v) for v in nrm]
            what = "centre and normal"
        else:  # the 2-D normal of the own plane, as the labeller uses everywhere else (see Config)
            what = "centre"
        d.evidence.append(
            f"{what} from the 3-D disc over {n_sl} slices, on the slice x={x_slices[kk]:.1f} mm nearest its lateral "
            f"centre (curve active; mid-plane centre moved {shift:.1f} mm; 3-D normal would turn {turn:.1f} deg)"
        )
        out.append(d)
    # Physical check: discs are at least one level spacing apart in height. Each mid-sagittal disc protects its nearest
    # relocated disc (one to one), so a disc the mid plane shows is kept; any other disc that lies this close to another
    # is a lateral blob or an artefact, and is dropped.
    gap_min = cfg.level_spacing_mm[0]
    ref = discs_near_midline(seg, mid_k, cfg)[0] if mid_k is not None else []

    def inplane(a: Disc, b: Disc) -> float:
        return float(np.linalg.norm(np.asarray(a.center)[[1, 2]] - np.asarray(b.center)[[1, 2]]))

    protected: dict[int, Disc] = {}  # index in ``out`` -> the mid-sagittal disc it matches
    for r in ref:
        near = [(inplane(d, r), j) for j, d in enumerate(out) if inplane(d, r) < min_sep]
        if near:
            _dist, j = min(near)
            if j not in protected or inplane(out[j], r) < inplane(out[j], protected[j]):
                protected[j] = r
    kept_idx: list[int] = []
    clash = 0
    for j in sorted(range(len(out)), key=lambda q: (q not in protected, -out[q].confidence)):
        if j not in protected and any(abs(out[j].center[2] - out[k].center[2]) < gap_min for k in kept_idx):
            clash += 1
            continue
        kept_idx.append(j)
    # A disc the mid-sagittal slice shows keeps the mid-plane normal, as on main: its own-plane section is a lateral
    # cut, not a central one, so its normal is the less reliable one (the 3-D flag overrides this, see Config).
    if not cfg.scoliosis_normal_3d:
        for j in kept_idx:
            if j in protected:
                out[j].normal = list(protected[j].normal)
                out[j].evidence.append(
                    "normal from the mid-sagittal slice, as the mid-plane path (disc also seen there)"
                )
    kept = [out[j] for j in kept_idx]
    out = sorted(kept, key=lambda q: q.center[2])
    warns.append(
        f"lateral curve (Cobb-like {curve.cobb_deg or 0.0:.0f} deg, apex {curve.amplitude_mm:.0f} mm {curve.direction}): "
        f"{len(out)} discs measured on their own sagittal planes (largest offset from the canal {moved:.1f} mm"
        + (f"; {dropped} candidate(s) seen on fewer than 3 slices dropped" if dropped else "")
        + (f"; {clash} closer than {gap_min:.0f} mm to a more confident disc dropped" if clash else "")
        + ")"
    )
    return out, warns


__all__ = [
    "LateralCurve",
    "canal_samples",
    "cobb_from_knots",
    "discs_on_curve",
    "knots_from_samples",
    "measure_curve",
    "slice_spacing_mm",
]
