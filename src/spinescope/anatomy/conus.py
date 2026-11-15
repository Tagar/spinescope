"""Conus medullaris: the caudal end of the cord inside the thecal sac, located on a sagittal series.

Inputs: the sagittal study (T2 preferred, then STIR, then T1), the learned canal segmentation
(:mod:`spinescope.anatomy.ml`, public ``segment_stack``) and the disc / vertebra anchors for the level name.

Method (all geometry in LPS mm, samples on the network's 1 mm plane grid):

1. Midline slice from the canal area (``mid_sagittal``). Sagittal slices within
   ``Config.conus_slice_halfwidth_mm`` of it are analysed separately.
2. Canal centreline per slice: the middle of the canal mask in each row (largest component),
   median-smoothed along the canal. Half-width of the canal per row from the same mask.
3. Profiles across the canal: a "core" (samples within ``conus_core_half_mm`` of the centreline, where
   the cord is) and a "CSF flank" (samples from ``conus_flank_inner_frac`` of the half-width outwards).
   Cord-likeness per row: ``1 - core/CSF`` on T2/STIR (CSF bright, cord darker); ``1 - CSF/core`` on T1
   (CSF dark, cord brighter).
4. Cord-like rows are median-filtered along z; the longest run of at least ``conus_min_run_mm`` gives the
   cord, and its inferior end is the conus tip. A run that reaches the lowest canal row in the field of
   view means the tip lies at or below the field of view.
5. Slices agreeing with the median tip (within ``conus_agree_mm``) give the position; confidence is the
   per-slice contrast and run length, times the agreeing share of slices, times the numbering-status factor.
6. Level: the disc whose margin contains the tip, else the vertebra between two discs (position: upper,
   middle or lower third of its body). Low-lying flag: tip below the L2-L3 disc (Wilson & Prince, AJNR 1989).

The result is judgement-based (no conus ground truth exists in the datasets used here); thresholds live
in :class:`spinescope.config.Config` with their sources. See docs/labeling/lab16_conus.md.
"""

from __future__ import annotations

import itertools
import logging
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
from .numbering import VERTEBRA_NAMES

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
METHOD = "conus/1 cord-in-thecal-sac profiles on the learned canal mask; level from disc anchors"
#: Sequence preference for the conus: CSF bright on T2 and STIR, dark on T1.
CONUS_CONTRAST_ORDER = ("t2", "stir", "t1")
#: Low-lying threshold reference (Wilson & Prince 1989, AJNR 10(2):259): tip at or below L3 is abnormal-to-indeterminate.
LOW_LYING_REFERENCE = "Wilson & Prince, AJNR 1989; Saifuddin et al., Spine 1998 (normal distribution)"


@dataclass
class SliceTip:
    """Cord tip on one sagittal slice."""

    k: int
    z: float
    xyz: FloatArray
    confidence: float
    run_mm: float
    reached_fov_end: bool


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
    low_lying: bool | None = None  # True: below the L2-L3 disc; None: unknown (tip at the field-of-view end)
    fov_end: bool = False
    n_slices: int = 0
    n_agree: int = 0
    contrast: str = ""
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

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
            "low_lying": self.low_lying,
            "fov_end": self.fov_end,
            "slices": self.n_slices,
            "agreeing_slices": self.n_agree,
            "contrast": self.contrast,
            "evidence": list(self.evidence),
        }


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


def _slice_tip(
    grid_probs: NDArray[np.float32],
    grid: PlaneGrid,
    img: NDArray[np.float32],
    polarity: int,
    window: tuple[float, float],
    cfg: Config,
) -> SliceTip | None:
    """Cord tip on one slice from its canal probabilities (``grid_probs``) and resampled image."""
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
    # Profiles: offsets across the canal in mm, sampled bilinearly on the slice image.
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
    # values are meaningless. The floor is the background; cord and CSF are measured above it.
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
    cord = (np.nan_to_num(score, nan=-1.0) >= cfg.conus_cord_contrast_min) & np.nan_to_num(csf_ok, nan=False)
    cord = ndimage.median_filter(cord.astype(np.uint8), size=m, mode="nearest").astype(bool)
    run = _longest_run(cord, round(cfg.conus_min_run_mm / res))
    if run is None:
        return None
    i0, i1 = run
    run_mm = (i1 - i0 + 1) * res
    mean_score = float(np.nanmean(score[i0 : i1 + 1]))
    contrast_term = float(np.clip(mean_score / (2.0 * cfg.conus_cord_contrast_min), 0.0, 1.0))
    length_term = float(np.clip(run_mm / cfg.conus_full_run_mm, 0.0, 1.0))
    conf = contrast_term * (0.5 + 0.5 * length_term)
    row_end = int(r_idx[i1])
    col_end = float(c_r[i1])
    xyz = np.asarray(grid.lps_of_index(float(row_end), col_end), dtype=np.float64)
    reached = i1 >= len(r_idx) - 1 - round(2.0 / res)
    return SliceTip(
        k=int(grid.k),
        z=float(xyz[2]),
        xyz=xyz,
        confidence=conf,
        run_mm=run_mm,
        reached_fov_end=bool(reached),
    )


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


# ------------------------------------------------------------------------------ public API
def conus_from_segmentation(
    seg: Any,
    anchors: Anchors,
    cfg: Config = DEFAULT,
    *,
    contrast: str = "t2",
) -> ConusResult:
    """Locate the conus from a segmented sagittal stack (``SegmentedStack``) and disc anchors.

    ``contrast`` gives the CSF polarity: T2/STIR (CSF bright) or T1 (CSF dark). Pure function of its inputs,
    so tests can pass an oracle segmentation.
    """
    from .ml.grid import resample_slice
    from .ml.labeller import CANAL
    from .ml.postprocess import mid_sagittal

    stack = seg.stack
    window = (float(seg.window[0]), float(seg.window[1]))
    if seg.probs.shape[0] == 0:
        return ConusResult(found=False, contrast=contrast, warnings=["no slices to analyse"])
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        return ConusResult(
            found=False,
            contrast=contrast,
            warnings=["conus not located: no spinal canal in the sagittal segmentation"],
        )
    x_mid = float(stack.positions[mid.k])
    ks = [
        k
        for k in range(stack.n)
        if abs(float(stack.positions[k]) - x_mid) <= cfg.conus_slice_halfwidth_mm
        and seg.probs[k, CANAL].max() > 0
    ]
    if mid.k not in ks:
        ks.append(mid.k)
    ks = sorted(set(ks), key=lambda k: abs(float(stack.positions[k]) - x_mid))[:5]
    polarities = _polarities(contrast)
    best: tuple[float, int, list[SliceTip]] | None = None
    for pol in polarities:
        tips: list[SliceTip] = []
        for k in ks:
            img = resample_slice(stack, seg.grids[k])
            tip = _slice_tip(seg.probs[k, CANAL], seg.grids[k], img, pol, window, cfg)
            if tip is not None:
                tips.append(tip)
        total = float(sum(t.confidence for t in tips))
        if best is None or total > best[0]:
            best = (total, pol, tips)
    assert best is not None
    _, pol, tips = best
    if not tips:
        return ConusResult(
            found=False,
            contrast=contrast,
            n_slices=len(ks),
            warnings=[f"conus not located on {contrast or 'sagittal'}: no cord signal inside the thecal sac"],
        )
    zs = np.array([t.z for t in tips])
    z_med = float(np.median(zs))
    agree = [t for t in tips if abs(t.z - z_med) <= cfg.conus_agree_mm]
    z = float(np.mean([t.z for t in agree]))
    xyz = np.mean([t.xyz for t in agree], axis=0)
    xyz[2] = z
    # Agreement among the slices that show a cord tip: slices that miss the cord (it is ~8 mm wide) are expected.
    share = len(agree) / len(tips)
    conf = float(np.mean([t.confidence for t in agree])) * share
    if len(agree) < 2:
        conf *= cfg.conus_single_slice_factor
    fov_end = bool(np.mean([t.reached_fov_end for t in agree]) >= 0.5)
    if fov_end:
        conf = min(conf, 0.5)
    level, position, lev_ev = level_at(z, anchors)
    status = str(anchors.numbering.get("status", "") or "")
    status_factor = {
        "confirmed": cfg.conus_status_factor[0],
        "probable": cfg.conus_status_factor[1],
        "unconfirmed": cfg.conus_status_factor[2],
    }.get(status, cfg.conus_status_factor[2])
    if status:
        conf *= status_factor
    conf = float(np.clip(conf, 0.0, 1.0))
    low = low_lying(z, level, anchors, cfg)
    ev = [
        f"cord-like signal inside the thecal sac ends at z={z:.1f} mm on {len(agree)}/{len(ks)} sagittal slice(s) "
        f"({contrast or 'unknown contrast'}, CSF polarity {pol:+d})",
        f"cord run {float(np.mean([t.run_mm for t in agree])):.0f} mm (mean over agreeing slices)",
        *lev_ev,
    ]
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
        xyz=np.asarray(xyz, dtype=np.float64),
        level=level,
        position=position,
        confidence=conf,
        low_lying=low,
        fov_end=fov_end,
        n_slices=len(ks),
        n_agree=len(agree),
        contrast=contrast,
        evidence=ev,
        warnings=warns,
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


def locate_conus(study: Study, anchors: Anchors, cfg: Config = DEFAULT) -> ConusResult:
    """Locate the conus on the study's sagittal series (see module doc). Missing inputs give warnings."""
    from ..render.window import contrast_of
    from .ml.labeller import segment_stack

    stack = pick_conus_stack(study)
    if stack is None:
        return ConusResult(found=False, warnings=["conus not located: no sagittal series"])
    contrast = contrast_of(stack)
    seg = segment_stack(stack, cfg)
    res = conus_from_segmentation(seg, anchors, cfg, contrast=contrast)
    return res


def attach_conus(study: Study, anchors: Anchors, cfg: Config = DEFAULT) -> ConusResult:
    """Locate the conus and write it to ``anchors.conus`` with its warnings.

    A failed search keeps an existing conus (for example the classical tip) and says so; otherwise the
    conus is ``None`` and a warning is added.
    """
    res = locate_conus(study, anchors, cfg)
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
]
