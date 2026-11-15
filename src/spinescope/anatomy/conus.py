"""Conus medullaris: the caudal end of the cord inside the thecal sac, located on a sagittal series.

Inputs: the sagittal study (T2 preferred, then STIR, then T1), the learned canal segmentation
(:mod:`spinescope.anatomy.ml`, public ``segment_stack``) and the disc / vertebra anchors for the level name.

Method v2 (lab26, see docs/labeling/lab26_conus_v2.md). All geometry in LPS mm on the 1 mm plane grid:

1. Sac slices: every sagittal slice of the segmented stack is scored by its canal area. The sac's mid-portion
   lies on the slices whose canal area is at least ``conus_sac_area_frac`` of the largest (the mid-sagittal
   plane is the canal-area median, :func:`spinescope.anatomy.ml.postprocess.mid_sagittal`).
2. Interpolation: the cord evidence of the two sac slices that bracket the mid-sagittal plane is blended by
   their distance to that plane, so the tracking runs on one profile at the plane itself. Each sac slice is also
   tracked on its own, for the consistency vote.
3. Cord evidence per row: the canal centreline and half-width; core (cord, within ``conus_core_half_mm``) against
   CSF flank (from ``conus_flank_inner_frac`` of the half-width outwards), window-relative. Cord-like rows need
   the CSF bright on T2/STIR (dark on T1) and a contrast above ``conus_cord_contrast_min``.
4. Tracking from the top: the first cord run (at least ``conus_min_run_mm``) is followed caudally. Gaps up to
   ``conus_stop_mm`` are bridged; the cord ends where a non-cord span reaches ``conus_stop_mm`` (cord width
   collapses to roots and filum). Reaching the lowest canal row means the tip is a lower bound (``fov_end``).
5. Consistency: the tip of the interpolated profile must agree (within ``conus_agree_mm``) with the sac slices'
   own tips; with a second sagittal contrast in the study (T1 next to T2), its tip must agree too.
6. Level: the disc whose margin contains the tip, else the vertebra between two discs (position: upper, middle or
   lower third of its body). Low-lying flag: tip below the L2-L3 disc (Wilson & Prince, AJNR 1989).
7. Confidence: cord contrast and run length, times the agreeing share of sac slices, times the cross-contrast and
   numbering-status factors. Not calibrated: on the lab26 reference no tip was within half a vertebra
   (docs/labeling/lab26_conus_v2.md), so the conus stays experimental.

Thresholds live in :class:`spinescope.config.Config` with their sources. The v1 slice-by-slice method is
documented in docs/labeling/lab16_conus.md.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .anchors import Anchors, split_disc
from .ml.grid import PlaneGrid
from .ml.labeller import SegmentedStack
from .numbering import VERTEBRA_NAMES

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
METHOD = "conus/2 cord tracked caudally in the thecal sac over bracketing sac slices; T1/T2 consistency; level from disc anchors"
#: Sequence preference for the conus: CSF bright on T2 and STIR, dark on T1.
CONUS_CONTRAST_ORDER = ("t2", "stir", "t1")
#: Low-lying threshold reference (Wilson & Prince 1989, AJNR 10(2):259): tip at or below L3 is abnormal-to-indeterminate.
LOW_LYING_REFERENCE = "Wilson & Prince, AJNR 1989; Saifuddin et al., Spine 1998 (normal distribution)"


@dataclass
class SliceTip:
    """Cord tip on one sagittal slice (or on the interpolated profile at the mid-sagittal plane)."""

    k: int  # -1 for the interpolated profile
    z: float
    xyz: FloatArray
    confidence: float
    run_mm: float
    reached_fov_end: bool
    score: float = 0.0  # mean cord contrast along the run (0..1 window-relative scale)


@dataclass
class ConusResult:
    """Located conus, or the reason it was not located. ``to_dict`` gives the ``Anchors.conus`` form."""

    found: bool
    z: float | None = None
    xyz: FloatArray | None = None
    level: str | None = None
    position: str | None = (
        None  # "upper third" | "middle third" | "lower third" of the vertebra, or "at disc"
    )
    confidence: float = 0.0
    #: Confidence before the numbering-status factor (kept so that :func:`refresh_conus` can re-apply the factor).
    confidence_raw: float | None = None
    low_lying: bool | None = None  # True: below the L2-L3 disc; None: unknown (tip at the field-of-view end)
    fov_end: bool = False
    n_slices: int = 0
    n_agree: int = 0
    contrast: str = ""
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cross_z: float | None = None  # tip on a second sagittal contrast, when the study has one
    cross_contrast: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for ``Anchors.conus`` (JSON-safe; no file paths, no patient data)."""
        if not self.found or self.xyz is None or self.z is None:
            return {}
        xyz = [round(float(v), 2) for v in self.xyz]
        return {
            "z": round(float(self.z), 2),
            "xyz": xyz,
            "center": xyz,
            "level": self.level,
            "position": self.position,
            "confidence": round(float(self.confidence), 3),
            "confidence_raw": None if self.confidence_raw is None else round(float(self.confidence_raw), 3),
            "low_lying": self.low_lying,
            "fov_end": self.fov_end,
            "slices": self.n_slices,
            "agreeing_slices": self.n_agree,
            "contrast": self.contrast,
            "cross_contrast": self.cross_contrast or None,
            "cross_z": None if self.cross_z is None else round(float(self.cross_z), 2),
            "evidence": list(self.evidence),
        }


@dataclass
class _Evidence:
    """Cord evidence of one sagittal slice, on the rows of its canal (superior first, z descending)."""

    k: int
    x: float  # LPS x of the slice plane
    z: FloatArray  # mm, descending
    y: FloatArray  # LPS y of the canal centreline per row
    lps: FloatArray  # (N, 3) LPS of the canal centreline per row (for the tip position)
    score: FloatArray  # cord-likeness per row (NaN where the canal is absent)
    csf_ok: NDArray[np.bool_]  # CSF in the expected half of the window
    half_mm: FloatArray  # canal half-width per row


# ------------------------------------------------------------------------------ slice analysis
def _largest_component(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())[1:]
    return np.asarray(lab == (int(np.argmax(sizes)) + 1), dtype=np.bool_)


def _longest_run(flags: NDArray[np.bool_], min_len: int) -> tuple[int, int] | None:
    """(start, end) inclusive of the longest True run of at least ``min_len`` samples, or None."""
    best: tuple[int, int] | None = None
    i, n = 0, len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        if j - i + 1 >= min_len and (best is None or j - i > best[1] - best[0]):
            best = (i, j)
        i = j + 1
    return best


def _slice_evidence(
    grid_probs: NDArray[np.float32],
    grid: PlaneGrid,
    img: NDArray[np.float32],
    x_plane: float,
    polarity: int,
    window: tuple[float, float],
    cfg: Config,
) -> _Evidence | None:
    """Cord-likeness per canal row on one slice; None when the canal is too short to analyse."""
    res = cfg.ml_resolution_mm
    mask = _largest_component(grid_probs >= cfg.ml_prob_threshold)
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size < max(3, round(cfg.conus_min_run_mm / res)):
        return None
    first = np.argmax(mask, axis=1).astype(np.float64)
    last = (mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)).astype(np.float64)
    r_idx = np.arange(rows[0], rows[-1] + 1)
    c_r = np.interp(r_idx, rows, (0.5 * (first + last))[rows])
    h_r = np.interp(r_idx, rows, (0.5 * (last - first + 1) * res)[rows])
    m = max(1, round(cfg.conus_smooth_mm / res))
    c_r = ndimage.median_filter(c_r, size=m, mode="nearest")
    # Profiles across the canal, in mm, bilinear on the slice image.
    half_max = float(np.max(h_r))
    offs = np.arange(-half_max, half_max + 0.5 * res, res)
    cols = c_r[:, None] + offs[None, :] / res
    rr = np.broadcast_to(r_idx[:, None].astype(np.float64), cols.shape)
    vals = ndimage.map_coordinates(np.asarray(img, dtype=np.float64), [rr, cols], order=1, mode="nearest")
    ab = np.abs(offs)[None, :]
    core_m = np.broadcast_to(ab <= cfg.conus_core_half_mm, vals.shape)
    flank_m = (ab >= cfg.conus_flank_inner_frac * h_r[:, None]) & (ab <= h_r[:, None] - 0.5)
    core = np.full(len(r_idx), np.nan)
    flank = np.full(len(r_idx), np.nan)
    has_core = core_m.any(axis=1)
    core[has_core] = np.nanmean(np.where(core_m, vals, np.nan)[has_core], axis=1)
    has_flank = flank_m.any(axis=1)
    flank[has_flank] = np.nanmedian(np.where(flank_m, vals, np.nan)[has_flank], axis=1)
    # Window-relative levels (0 = window floor, 1 = window top): the volumes carry offsets, so ratios of raw
    # values are meaningless.
    lo, hi = window
    span = max(float(hi) - float(lo), 1e-6)
    core_rel = (core - lo) / span
    flank_rel = (flank - lo) / span
    eps = 0.05
    if polarity > 0:  # T2/STIR: CSF bright, cord darker than CSF
        score = 1.0 - np.maximum(core_rel, eps) / np.maximum(flank_rel, eps)
        csf_ok = flank_rel >= cfg.conus_csf_min_rel
    else:  # T1: CSF dark, cord brighter than CSF
        score = 1.0 - np.maximum(flank_rel, eps) / np.maximum(core_rel, eps)
        csf_ok = flank_rel <= cfg.conus_csf_min_rel
    xyz = np.asarray(grid.lps_of_index(r_idx.astype(np.float64), c_r), dtype=np.float64)
    if not np.all(np.isfinite(xyz)):
        return None
    return _Evidence(
        k=int(grid.k),
        x=float(x_plane),
        z=xyz[:, 2].copy(),
        y=xyz[:, 1].copy(),
        lps=xyz.copy(),
        score=np.where(np.isfinite(score), score, np.nan),
        csf_ok=np.nan_to_num(csf_ok, nan=0.0).astype(bool),
        half_mm=h_r,
    )


# ------------------------------------------------------------------------------ profiles and tracking
def _common_grid(evs: list[_Evidence], res: float) -> FloatArray:
    """Descending z grid (1 mm) spanning every slice's canal rows."""
    zmax = max(float(np.max(e.z)) for e in evs)
    zmin = min(float(np.min(e.z)) for e in evs)
    n = int(np.floor((zmax - zmin) / res)) + 1
    return np.asarray(zmax - res * np.arange(n), dtype=np.float64)


def _on_grid(
    e: _Evidence, zg: FloatArray, res: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """(present, score, csf_ok, y) of one slice interpolated onto the descending grid ``zg``.

    Present is 1 inside the slice's canal rows and 0 outside; the other arrays are NaN outside.
    """
    zr = e.z[::-1]
    lo, hi = float(zr[0]), float(zr[-1])
    inside = (zg >= lo - 0.5 * res) & (zg <= hi + 0.5 * res)
    present = inside.astype(np.float64)
    score = np.full(zg.shape, np.nan)
    csf = np.full(zg.shape, np.nan)
    y = np.full(zg.shape, np.nan)
    if not np.any(inside):
        return present, score, csf, y
    zq = zg[inside]
    score[inside] = np.interp(zq, zr, np.nan_to_num(e.score[::-1], nan=-1.0))
    csf[inside] = np.interp(zq, zr, e.csf_ok[::-1].astype(np.float64))
    y[inside] = np.interp(zq, zr, e.y[::-1])
    return present, score, csf, y


def _cord_flags(present: FloatArray, score: FloatArray, csf: FloatArray, cfg: Config) -> NDArray[np.bool_]:
    ok_score = np.nan_to_num(score, nan=-1.0) >= cfg.conus_cord_contrast_min
    return (present >= 0.5) & ok_score & (np.nan_to_num(csf, nan=0.0) >= 0.5)


@dataclass
class _Track:
    """Cord run followed caudally from the top of the canal (indices on the descending grid)."""

    start: int
    tip: int
    reached_end: bool


def _track(flags: NDArray[np.bool_], cfg: Config, res: float) -> _Track | None:
    """Follow the first cord run from the top: bridge short gaps; stop where a non-cord span reaches stop length."""
    m = max(1, round(cfg.conus_smooth_mm / res))
    f = ndimage.median_filter(flags.astype(np.uint8), size=m, mode="nearest").astype(bool)
    min_n = max(1, round(cfg.conus_min_run_mm / res))
    stop_n = max(1, round(cfg.conus_stop_mm / res))
    n = len(f)
    i = 0
    while i < n:
        if not f[i]:
            i += 1
            continue
        last, gap, k = i, 0, i
        stopped = False
        while k < n:
            if f[k]:
                last, gap = k, 0
            else:
                gap += 1
                if gap >= stop_n:
                    stopped = True
                    break
            k += 1
        if last - i + 1 >= min_n:
            reached = not stopped and last >= n - 1 - round(2.0 / res)
            return _Track(start=i, tip=last, reached_end=bool(reached))
        i = last + 1
    return None


def _tip_from_track(
    tr: _Track,
    zg: FloatArray,
    score: FloatArray,
    y: FloatArray,
    ev: _Evidence,
    res: float,
    k: int,
) -> SliceTip:
    """Tip from a track: z on the grid, LPS position from the slice's own canal centreline at that height."""
    run_mm = (tr.tip - tr.start + 1) * res
    seg = slice(tr.start, tr.tip + 1)
    mean_score = float(np.nanmean(score[seg])) if np.any(np.isfinite(score[seg])) else 0.0
    z = float(zg[tr.tip])
    j = int(np.argmin(np.abs(ev.z - z)))
    xyz = np.asarray(ev.lps[j], dtype=np.float64).copy()
    xyz[2] = z
    return SliceTip(
        k=k,
        z=z,
        xyz=xyz,
        confidence=0.0,
        run_mm=run_mm,
        reached_fov_end=tr.reached_end,
        score=mean_score,
    )


def _slice_confidence(tip: SliceTip, cfg: Config) -> float:
    """Per-slice confidence from the cord contrast and run length (before agreement and status)."""
    contrast_term = float(np.clip(tip.score / (2.0 * cfg.conus_cord_contrast_min), 0.0, 1.0))
    length_term = float(np.clip(tip.run_mm / cfg.conus_full_run_mm, 0.0, 1.0))
    return contrast_term * (0.5 + 0.5 * length_term)


def _bracket_weights(xs: list[float], x_mid: float) -> list[tuple[int, float]]:
    """Index/weight pairs of the sac slices that bracket the plane ``x_mid`` (linear in x)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    below = [i for i in order if xs[i] <= x_mid]
    above = [i for i in order if xs[i] > x_mid]
    if below and above:
        a, b = below[-1], above[0]
        span = xs[b] - xs[a]
        wa = 1.0 if span < 1e-6 else (xs[b] - x_mid) / span
        wa = float(np.clip(wa, 0.0, 1.0))
        return [(a, wa), (b, 1.0 - wa)]
    if below:
        return [(below[-1], 1.0)]
    return [(above[0], 1.0)]


@dataclass
class _Located:
    """Cord tip of one sagittal stack (interpolated at the mid-sagittal plane) with its slice tips."""

    tip: SliceTip
    slice_tips: list[SliceTip]
    n_sac_slices: int
    polarity: int


def _locate_on_stack(
    seg: SegmentedStack, cfg: Config, contrast: str, mid_x: float | None = None
) -> tuple[_Located | None, str]:
    """Cord tip for one segmented stack. Returns (result or None, reason when None)."""
    from .ml.labeller import CANAL
    from .ml.postprocess import canal_areas_mm2, mid_sagittal

    stack = seg.stack
    window = (float(seg.window[0]), float(seg.window[1]))
    if seg.probs.shape[0] == 0:
        return None, "no slices to analyse"
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        return None, "conus not located: no spinal canal in the sagittal segmentation"
    # Plane coordinates: the sac slices and the mid-sagittal plane share the stack's ``positions`` axis.
    x_mid = float(stack.positions[mid.k]) if mid_x is None else mid_x
    area = canal_areas_mm2(seg, cfg)
    peak = float(np.max(area))
    sac = [k for k in range(stack.n) if area[k] >= cfg.conus_sac_area_frac * peak and area[k] > 0]
    if mid.k not in sac:
        sac.append(mid.k)
    sac = sorted(set(sac), key=lambda k: abs(float(stack.positions[k]) - x_mid))[: cfg.conus_max_slices]
    res = cfg.ml_resolution_mm
    best: tuple[float, int, _Located] | None = None
    for pol in _polarities(contrast):
        evs: list[_Evidence] = []
        for k in sac:
            img = _resample(stack, seg.grids[k])
            ev = _slice_evidence(
                seg.probs[k, CANAL], seg.grids[k], img, float(stack.positions[k]), pol, window, cfg
            )
            if ev is not None:
                evs.append(ev)
        if not evs:
            continue
        zg = _common_grid(evs, res)
        # Interpolated profile at the mid-sagittal plane (bracketing sac slices).
        xs = [e.x for e in evs]
        wts = _bracket_weights(xs, x_mid)
        present = np.zeros(zg.shape)
        score = np.zeros(zg.shape)
        csf = np.zeros(zg.shape)
        y = np.zeros(zg.shape)
        wsum_s = np.zeros(zg.shape)
        for idx, w in wts:
            p_, s_, c_, y_ = _on_grid(evs[idx], zg, res)
            present += w * p_
            ok = np.isfinite(s_)
            score[ok] += w * s_[ok]
            wsum_s[ok] += w
            csf += w * np.nan_to_num(c_, nan=0.0)
            y += w * np.nan_to_num(y_, nan=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(wsum_s > 0, score / np.where(wsum_s > 0, wsum_s, 1.0), np.nan)
        flags = _cord_flags(present, score, csf, cfg)
        tr = _track(flags, cfg, res)
        if tr is None:
            continue
        nearest = min(evs, key=lambda e: abs(e.x - x_mid))
        ip_tip = _tip_from_track(tr, zg, score, y, nearest, res, -1)
        if ip_tip is None:
            continue
        # Per sac slice: its own profile, its own track.
        slice_tips: list[SliceTip] = []
        for ev in evs:
            p_, s_, c_, y_ = _on_grid(ev, zg, res)
            tr_k = _track(_cord_flags(p_, s_, c_, cfg), cfg, res)
            if tr_k is None:
                continue
            tk = _tip_from_track(tr_k, zg, s_, y_, ev, res, ev.k)
            if tk is not None:
                tk.confidence = _slice_confidence(tk, cfg)
                slice_tips.append(tk)
        ip_tip.confidence = _slice_confidence(ip_tip, cfg)
        ip_tip.reached_fov_end = tr.reached_end
        total = ip_tip.confidence + sum(t.confidence for t in slice_tips)
        cand = _Located(tip=ip_tip, slice_tips=slice_tips, n_sac_slices=len(evs), polarity=pol)
        if best is None or total > best[0]:
            best = (total, pol, cand)
    if best is None:
        return None, f"conus not located on {contrast or 'sagittal'}: no cord signal inside the thecal sac"
    return best[2], ""


def _resample(stack: Stack, grid: PlaneGrid) -> NDArray[np.float32]:
    from .ml.grid import resample_slice

    return resample_slice(stack, grid)


# ------------------------------------------------------------------------------ levels
def level_at(z: float, anchors: Anchors) -> tuple[str | None, str | None, list[str]]:
    """(level, position, evidence) of a superior-inferior height ``z`` in the disc/vertebra anchors.

    Level is the disc label when ``z`` lies within half a disc height of its centre; otherwise the vertebra
    between two adjacent discs, with the third of its body. Above or below all discs the vertebra at the
    end is named and no third is given.
    """
    discs = sorted(
        ((lab, float(d.center[2]), float(d.height_mm or 8.0)) for lab, d in anchors.discs.items()),
        key=lambda t: -t[1],
    )
    if not discs:
        return None, None, ["no disc anchors: level not named"]
    for lab, zc, h in discs:
        if abs(z - zc) <= h / 2.0:
            return lab, "at disc", [f"tip within the {lab} disc"]
    top, bottom = discs[0], discs[-1]
    if z > top[1]:
        vert = split_disc(top[0])[0]
        return vert, None, [f"tip above the highest disc ({top[0]}): body {vert} named by count"]
    if z < bottom[1]:
        vert = split_disc(bottom[0])[1]
        return vert, None, [f"tip below the lowest disc ({bottom[0]}): body {vert} named by count"]
    for (u, zu, hu), (lo, zl, hl) in itertools.pairwise(discs):
        if not (zl < z < zu):
            continue
        up_name, lo_name = split_disc(u)[1], split_disc(lo)[0]
        if up_name != lo_name:
            return None, None, [f"gap in disc labels between {u} and {lo}: level not named"]
        top_edge, bot_edge = zu - hu / 2.0, zl + hl / 2.0
        frac = (top_edge - z) / max(top_edge - bot_edge, 1e-6)
        third = "upper third" if frac < 1 / 3 else ("middle third" if frac < 2 / 3 else "lower third")
        return up_name, third, [f"tip in body {up_name} between {u} and {lo}, {third} (fraction {frac:.2f})"]
    return None, None, ["tip not inside the disc range: level not named"]


def _vertebra_index(name: str) -> int | None:
    return VERTEBRA_NAMES.index(name) if name in VERTEBRA_NAMES else None


def low_lying(z: float, level: str | None, anchors: Anchors, cfg: Config = DEFAULT) -> bool | None:
    """True when the tip lies below the L2-L3 disc (its inferior margin); False when not; None if unknown."""
    ref = anchors.discs.get(cfg.conus_low_lying_disc)
    if ref is not None:
        margin = float(ref.center[2]) - float(ref.height_mm or 8.0) / 2.0
        return bool(z < margin)
    if level is None:
        return None
    if "-" in level:
        lower = split_disc(level)[1]
        idx = _vertebra_index(lower)
        return None if idx is None else idx <= 2  # disc L3-L4 or below (lower name L4 or below)
    idx = _vertebra_index(level)
    return None if idx is None else idx <= 3  # vertebra L3 or below


def _with_status(raw: float, status: str, cfg: Config) -> tuple[float, float]:
    """Confidence after the numbering-status factor (``conus_status_factor``), and the factor used.

    An empty status (the conus was located before the numbering was final) leaves the confidence unchanged.
    """
    if not status:
        return float(np.clip(raw, 0.0, 1.0)), 1.0
    factor = {
        "confirmed": cfg.conus_status_factor[0],
        "probable": cfg.conus_status_factor[1],
        "unconfirmed": cfg.conus_status_factor[2],
    }.get(status, cfg.conus_status_factor[2])
    return float(np.clip(raw * factor, 0.0, 1.0)), factor


_LOW_LYING_MESSAGES = ("low-lying conus:", "possible low-lying conus at")
_NAME_EVIDENCE = (
    "tip ",
    "gap in disc",
    "no disc anchors",
    "level names",
)  # evidence lines that depend on names


def refresh_conus(anchors: Anchors, cfg: Config = DEFAULT) -> None:
    """Re-derive the names and status-dependent fields of ``anchors.conus`` once the numbering is final.

    The conus is located before the numbering fusion (so its tip can vote), in the pre-fusion names. After the
    fusion and the applicability check, its level, position, low-lying flag and confidence (status factor) are
    recomputed from the stored tip height ``z``; the low-lying warnings are rewritten to match. The experimental
    confidence cap (``conus_max_confidence``) is applied here as well.
    """
    c = anchors.conus
    if not isinstance(c, dict) or c.get("z") is None:
        return
    z = float(c["z"])
    raw = c.get("confidence_raw")
    raw = float(c.get("confidence", 0.0)) if raw is None else float(raw)
    status = str(anchors.numbering.get("status", "") or "")
    level, position, lev_ev = level_at(z, anchors)
    conf, _factor = _with_status(raw, status, cfg)
    conf = min(conf, cfg.conus_max_confidence)
    low = low_lying(z, level, anchors, cfg)
    anchors.warnings = [w for w in anchors.warnings if not w.startswith(_LOW_LYING_MESSAGES)]
    if low:
        if conf >= cfg.conus_low_lying_min_confidence:
            msg = (
                f"low-lying conus: tip at {level or 'an unnamed level'} ({position or 'no third'}), below the "
                f"{cfg.conus_low_lying_disc} disc (confidence {conf:.2f}; {LOW_LYING_REFERENCE}); "
                "confirm on the sagittal T2 before reporting"
            )
        else:
            msg = (
                f"possible low-lying conus at {level or 'an unnamed level'}: confidence {conf:.2f} is too low to "
                "call it; verify manually on the sagittal T2"
            )
        anchors.warnings.append(msg)
    c.update(
        {
            "level": level,
            "position": position,
            "confidence": round(conf, 3),
            "low_lying": low,
            "experimental": True,
            "evidence": [e for e in c.get("evidence", []) if not e.startswith(_NAME_EVIDENCE)]
            + lev_ev
            + ([f"level names: numbering {status}"] if status else []),
        }
    )


# ------------------------------------------------------------------------------ public API
def conus_from_segmentation(
    seg: Any,
    anchors: Anchors,
    cfg: Config = DEFAULT,
    *,
    contrast: str = "t2",
    cross: _Located | None = None,
    cross_contrast: str = "",
) -> ConusResult:
    """Locate the conus from a segmented sagittal stack (``SegmentedStack``) and disc anchors.

    ``contrast`` gives the CSF polarity: T2/STIR (CSF bright) or T1 (CSF dark). ``cross`` is the tip found on a
    second sagittal contrast of the same study (see :func:`locate_conus`); it is used only for the consistency
    check and the confidence. Pure function of its inputs, so tests can pass an oracle segmentation.
    """
    found, why = _locate_on_stack(seg, cfg, contrast)
    if found is None:
        return ConusResult(found=False, contrast=contrast, warnings=[why] if why else [])
    tip = found.tip
    slice_tips = found.slice_tips
    n_sac = found.n_sac_slices
    # Agreement: slice tips within conus_agree_mm of the interpolated tip.
    agree = [t for t in slice_tips if abs(t.z - tip.z) <= cfg.conus_agree_mm]
    share = len(agree) / len(slice_tips) if slice_tips else 0.0
    if len(slice_tips) >= 2 and share <= cfg.conus_min_agree_share:
        return ConusResult(
            found=False,
            contrast=contrast,
            n_slices=len(slice_tips),
            warnings=[
                f"conus not located on {contrast or 'sagittal'}: cord tips on the sac slices disagree in z"
            ],
        )
    z = float(tip.z)
    xyz = np.asarray(tip.xyz, dtype=np.float64)
    if not slice_tips:
        share = 1.0  # the interpolated profile alone: no slice-level vote
    conf = _slice_confidence(tip, cfg) * (0.5 + 0.5 * share)
    if len(slice_tips) < 2:
        conf *= cfg.conus_single_slice_factor
    # Second contrast: agreement raises nothing, disagreement halves the confidence, absence is a small penalty.
    cross_z = float(cross.tip.z) if cross is not None else None
    if cross is not None and cross_z is not None:
        conf *= (
            cfg.conus_xc_agree_factor
            if abs(cross_z - z) <= cfg.conus_agree_mm
            else cfg.conus_xc_disagree_factor
        )
    else:
        conf *= cfg.conus_xc_absent_factor
    level, position, lev_ev = level_at(z, anchors)
    status = str(anchors.numbering.get("status", "") or "")
    fov_end = bool(tip.reached_fov_end)
    raw = float(np.clip(conf, 0.0, 1.0))
    if fov_end:
        raw = min(raw, 0.5)
    conf, status_factor = _with_status(raw, status, cfg)
    low = low_lying(z, level, anchors, cfg)
    ev = [
        f"cord-like signal inside the thecal sac ends at z={z:.1f} mm on the mid-sagittal profile "
        f"({contrast or 'unknown contrast'}, CSF polarity {found.polarity:+d}; {n_sac} sac slice(s))",
        f"cord run {tip.run_mm:.0f} mm; {len(agree)}/{len(slice_tips)} sac slice tips agree within "
        f"{cfg.conus_agree_mm:.0f} mm",
        *lev_ev,
    ]
    if cross is not None and cross_z is not None:
        ev.append(
            f"second contrast ({cross_contrast or 'other'}) tip at z={cross_z:.1f} mm "
            f"({'agrees' if abs(cross_z - z) <= cfg.conus_agree_mm else 'disagrees'})"
        )
    if status:
        ev.append(f"level names: numbering {status} (confidence factor {status_factor:.2f})")
    warns: list[str] = []
    if fov_end:
        ev.append("cord signal continues to the lowest canal row in the field of view")
        warns.append(
            "conus at or below the field of view: the tip is a lower bound; a low-lying conus cannot be excluded"
        )
    result = ConusResult(
        found=True,
        z=z,
        xyz=xyz,
        level=level,
        position=position,
        confidence=conf,
        confidence_raw=raw,
        low_lying=low,
        fov_end=fov_end,
        n_slices=len(slice_tips),
        n_agree=len(agree),
        contrast=contrast,
        evidence=ev,
        warnings=warns,
        cross_z=cross_z,
        cross_contrast=cross_contrast,
    )
    if low:
        if conf >= cfg.conus_low_lying_min_confidence:
            msg = (
                f"low-lying conus: tip at {level or 'an unnamed level'} ({position or 'no third'}), below the "
                f"{cfg.conus_low_lying_disc} disc (confidence {conf:.2f}; {LOW_LYING_REFERENCE}); "
                "confirm on the sagittal T2 before reporting"
            )
            if level and level.startswith("L3"):
                msg += "; L3 is an indeterminate zone in the literature"
        else:
            msg = (
                f"possible low-lying conus at {level or 'an unnamed level'}: confidence {conf:.2f} is too low to "
                "call it; verify manually on the sagittal T2"
            )
        result.warnings.append(msg)
    return result


def _polarities(contrast: str) -> tuple[int, ...]:
    from .sagittal import csf_polarities

    return tuple(csf_polarities(contrast)) or (1, -1)


def pick_conus_stack(study: Study) -> Stack | None:
    """Sagittal stack for the conus: T2, else STIR, else T1 (non-localizer), else the best sagittal."""
    for c in CONUS_CONTRAST_ORDER:
        hits = study.find("sag", c)
        hits = [s for s in hits if not s.meta.get("role", {}).get("localizer")]
        if hits:
            return hits[0]
    return study.best("sag")


def _cross_stack(study: Study, primary: Stack) -> Stack | None:
    """A second sagittal contrast for the consistency check: T1 when the primary is T2/STIR (and the reverse)."""
    from ..render.window import contrast_of

    want = "t1" if contrast_of(primary) in ("t2", "stir") else "t2"
    for s in study.find("sag", want):
        if s is primary or s.meta.get("role", {}).get("localizer"):
            continue
        return s
    return None


def locate_conus(
    study: Study,
    anchors: Anchors,
    cfg: Config = DEFAULT,
    *,
    seg: SegmentedStack | None = None,
    segs: Mapping[str, SegmentedStack] | None = None,
) -> ConusResult:
    """Locate the conus on the study's sagittal series (see module doc). Missing inputs give warnings.

    ``seg`` is reused when it is the network output of the picked stack; otherwise the network runs here. When
    the study has a second sagittal contrast (T1 next to T2, or T2 next to T1) and ``Config.conus_cross_contrast``
    is on, its tip is located too and used for the consistency check. ``segs`` maps stack keys to network outputs
    already computed for the study: the contrast stack is taken from there when present (the labeller segments
    the second sagittal series once, and it is often this one), so the network does not run twice.
    """
    from ..render.window import contrast_of
    from .ml.labeller import segment_stack

    stack = pick_conus_stack(study)
    if stack is None:
        return ConusResult(found=False, warnings=["conus not located: no sagittal series"])
    contrast = contrast_of(stack)
    if seg is None or seg.stack is not stack:
        seg = segment_stack(stack, cfg)
    cross: _Located | None = None
    cross_contrast = ""
    if cfg.conus_cross_contrast:
        other = _cross_stack(study, stack)
        if other is not None:
            cross_contrast = contrast_of(other)
            known = segs.get(other.key) if segs is not None else None
            cross_seg = known if known is not None and known.stack is other else segment_stack(other, cfg)
            cross, _why = _locate_on_stack(cross_seg, cfg, cross_contrast)
    return conus_from_segmentation(
        seg, anchors, cfg, contrast=contrast, cross=cross, cross_contrast=cross_contrast
    )


def attach_conus(
    study: Study,
    anchors: Anchors,
    cfg: Config = DEFAULT,
    *,
    seg: SegmentedStack | None = None,
    segs: Mapping[str, SegmentedStack] | None = None,
) -> ConusResult:
    """Locate the conus and write it to ``anchors.conus`` with its warnings.

    A failed search keeps an existing conus (for example the classical tip) and says so; otherwise the
    conus is ``None`` and a warning is added. ``seg`` and ``segs`` are passed on to :func:`locate_conus`.
    """
    res = locate_conus(study, anchors, cfg, seg=seg, segs=segs)
    if res.found:
        anchors.conus = res.to_dict()
        anchors.warnings.extend(res.warnings)
        return res
    anchors.warnings.extend(res.warnings)
    if anchors.conus and anchors.conus.get("z") is not None:
        anchors.warnings.append(
            "conus: only the classical tip estimate is available (no level or confidence)"
        )
    else:
        anchors.conus = None
    return res


__all__ = [
    "METHOD",
    "ConusResult",
    "attach_conus",
    "conus_from_segmentation",
    "level_at",
    "locate_conus",
    "low_lying",
    "pick_conus_stack",
    "refresh_conus",
]
