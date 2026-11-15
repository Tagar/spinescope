"""Image quality and hardware per study and per level: ``assess_quality`` and ``apply_quality`` (lab31).

Post-operative spines (pedicle screws, rods, interbody cages, disc arthroplasty, vertebroplasty cement) and image
artefacts (susceptibility blooming, motion ghosting, phase wrap, severe bias field, failed fat suppression on STIR,
low SNR) degrade labelling and every measurement that follows. This module looks for them *after* labelling, using
the disc positions of the anchors, and reports:

* ``quality['levels'][label]``: per disc level, a flag confidence in [0, 1] for each flag kind and the flags that
  reach ``quality_flag_min_confidence``;
* ``quality['study']``: a study score in [0, 1] (1 = no evidence of hardware or artefact) and one record per flag
  kind with its evidence;
* ``quality['warnings']``: plain-text warnings, also appended to ``anchors.warnings``.

:func:`apply_quality` then lowers the confidence of flagged levels (the raw calibrated value is kept, with the cap
and its reason in the evidence and in ``quality['caps']``) and caps a confirmed numbering to probable when a flagged
level is in view. Calibrated values are never overwritten silently.

All thresholds are relative to the study's own tissue intensity (``Config.quality_*``); nothing is in absolute
signal units. Detectors are classical image analysis (connected components, lag autocorrelation, intensity
profiles); they are judgement-based and not calibrated. Validation numbers: docs/labeling/lab31_artifacts.md.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .anchors import Anchors
from .numbering import STATUS_CONFIRMED, STATUS_PROBABLE, STATUS_UNCONFIRMED

log = logging.getLogger(__name__)

METHOD = "artifacts/1 per-level hardware and artefact flags (relative intensity, classical detectors)"
#: Flag kinds, in report order.
FLAGS: tuple[str, ...] = (
    "metal",
    "fusion_or_cage",
    "motion_ghost",
    "wrap",
    "fat_sat_failure",
    "low_snr",
    "bias_field",
)
FloatArray = NDArray[np.float64]


def _window_mask(stack: Stack, disc_pix: list[tuple[float, float]], cfg: Config) -> NDArray[np.bool_]:
    """Pixels within ~20 mm (rows) and 25 mm (columns) of a disc: the spinal window, in-plane for every slice."""
    rows, cols = stack.vol.shape[1], stack.vol.shape[2]
    m = np.zeros((rows, cols), dtype=bool)
    rr, cc = np.mgrid[0:rows, 0:cols]
    for r0, c0 in disc_pix:
        dr = np.abs(rr - r0) * stack.row_mm <= 20.0
        dc = np.abs(cc - c0) * stack.col_mm <= 25.0
        m |= dr & dc
    return m


def _in_plane(stack: Stack, xyz: FloatArray) -> tuple[float, float]:
    """(row, column) of an LPS point projected onto the stack's plane (fractional pixel coordinates)."""
    d = xyz - np.asarray(stack.ipp[0], dtype=np.float64)
    r = float(d @ np.asarray(stack.col_dir, dtype=np.float64)) / stack.row_mm
    c = float(d @ np.asarray(stack.row_dir, dtype=np.float64)) / stack.col_mm
    return r, c


def _disc_pixels(stack: Stack, anchors: Anchors) -> list[tuple[str, float, float]]:
    return [
        (lab, *_in_plane(stack, np.asarray(d.center, dtype=np.float64))) for lab, d in anchors.discs.items()
    ]


def _nearest_disc(anchors: Anchors, z: float) -> tuple[str, float]:
    """Disc label whose centre is nearest in z, and the distance in mm."""
    best = min(anchors.discs.items(), key=lambda kv: abs(kv[1].center[2] - z))
    return best[0], abs(best[1].center[2] - z)


def _body_ref(vol: NDArray[np.float64], window: NDArray[np.bool_]) -> float:
    """Tissue reference: median intensity inside the spinal window over all slices (0 when empty)."""
    vals = vol[:, window]
    return float(np.median(vals)) if vals.size else 0.0


def _metal_candidates(
    stack: Stack, window: NDArray[np.bool_], ref: float, cfg: Config
) -> list[dict[str, Any]]:
    """Compact dark regions inside the spinal window of each slice: one record per region.

    A region is a void when its core is below ``quality_void_fraction`` of the tissue reference, its area is within
    ``quality_void_area_mm2`` (screws and cages; a canal or dropout is larger), it is at least ~2 mm thick, and it sits
    inside tissue (more than half of the 6 mm ring around it is above 0.2 of the reference, so air at the edge of the
    field of view does not count). The rim ratio (median of a 2-3 pixel ring over the reference) is the blooming
    evidence: a dark bloom (below 1) or a bright pile-up (above 1).
    """
    out: list[dict[str, Any]] = []
    if ref <= 0:
        return out
    px_area = stack.row_mm * stack.col_mm
    lo_a, hi_a = cfg.quality_void_area_mm2
    for k in range(stack.vol.shape[0]):
        img = ndimage.gaussian_filter(stack.vol[k].astype(np.float64), 0.7)
        dark = (img < cfg.quality_void_fraction * ref) & window
        lab, n = ndimage.label(dark)
        if n == 0:
            continue
        thick = ndimage.distance_transform_edt(dark)
        for j, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            comp = lab[sl] == j
            area = float(comp.sum()) * px_area
            if not (lo_a <= area <= hi_a):
                continue
            if (
                float(thick[sl][comp].max()) * min(stack.row_mm, stack.col_mm)
                < cfg.quality_void_min_thickness_mm
            ):
                continue  # a thin line (cortex, endplate), not a compact object
            pad = 6
            r0, r1 = max(0, sl[0].start - pad), min(stack.vol.shape[1], sl[0].stop + pad)
            c0, c1 = max(0, sl[1].start - pad), min(stack.vol.shape[2], sl[1].stop + pad)
            sub = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            sub[sl[0].start - r0 : sl[0].stop - r0, sl[1].start - c0 : sl[1].stop - c0] = comp
            ring6 = ndimage.binary_dilation(
                sub, iterations=max(1, round(6.0 / min(stack.row_mm, stack.col_mm)))
            )
            ring6 &= ~sub
            patch = img[r0:r1, c0:c1]
            tissue_frac = float((patch[ring6] > 0.2 * ref).mean()) if ring6.any() else 0.0
            if tissue_frac <= 0.5:
                continue  # air at the edge of the field of view, not a structure inside the body
            rim_ring = ndimage.binary_dilation(sub, iterations=3) & ~sub
            rim = float(np.median(patch[rim_ring]) / ref) if rim_ring.any() else 1.0
            core = float(np.median(img[sl][comp]) / ref)
            rr, cc = np.nonzero(comp)
            r_c = float(rr.mean()) + sl[0].start
            c_c = float(cc.mean()) + sl[1].start
            out.append(
                {
                    "slice": k,
                    "row": r_c,
                    "col": c_c,
                    "area_mm2": round(area, 1),
                    "core_ratio": round(core, 3),
                    "rim_ratio": round(rim, 3),
                }
            )
    return out


def detect_metal(stack: Stack, anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Metal and blooming: compact signal voids inside the spine, seen in several sagittal slices.

    Returns ``{"levels": {disc: conf}, "kinds": {"metal": n_body, "disc_space": n}, "voids": n, "rim_bright": n}``.
    ``metal`` (screws, rods, cement in a body) and ``disc_space`` (a void in a disc interval: a cage or a block) are
    counted separately; a void counts for a level only when it is seen in at least ``quality_void_min_slices``
    slices. Confidence grows with the number of slices and with a rim that deviates from the tissue.
    """
    empty: dict[str, Any] = {
        "levels": {},
        "kinds": {"metal": 0, "disc_space": 0},
        "voids": 0,
        "rim_bright": 0,
    }
    if not anchors.discs:
        return empty
    discs = _disc_pixels(stack, anchors)
    window = _window_mask(stack, [(r, c) for _, r, c in discs], cfg)
    ref = _body_ref(stack.vol.astype(np.float64), window)
    cands = _metal_candidates(stack, window, ref, cfg)
    if not cands:
        return empty
    by_disc: dict[str, dict[str, Any]] = {}
    kinds = {"metal": 0, "disc_space": 0}
    rim_bright = 0
    for cnd in cands:
        xyz = _pix_to_lps(stack, cnd["slice"], cnd["row"], cnd["col"])
        z = float(xyz[2])
        lab, dz = _nearest_disc(anchors, z)
        h = anchors.discs[lab].height_mm or 8.0
        in_disc = dz <= h / 2.0
        kind = "disc_space" if in_disc else "metal"
        kinds[kind] += 1
        if cnd["rim_ratio"] >= 1.0 + cfg.quality_rim_deviation:
            rim_bright += 1
        targets = [lab] if in_disc else _targets_of_body(anchors, z)
        for t in targets:
            rec = by_disc.setdefault(t, {"slices": set(), "kinds": set(), "rim": []})
            rec["slices"].add(cnd["slice"])
            rec["kinds"].add(kind)
            rec["rim"].append(cnd["rim_ratio"])
    levels: dict[str, float] = {}
    for lab, rec in by_disc.items():
        n = len(rec["slices"])
        if n < cfg.quality_void_min_slices:
            continue
        rim_dev = np.mean([abs(r - 1.0) for r in rec["rim"]]) if rec["rim"] else 0.0
        levels[lab] = float(min(1.0, 0.5 + 0.2 * (n - cfg.quality_void_min_slices + 1) + rim_dev))
    return {
        "levels": levels,
        "kinds": kinds,
        "voids": len(cands),
        "rim_bright": rim_bright,
        "by_disc_kind": {lab: sorted(rec["kinds"]) for lab, rec in by_disc.items() if lab in levels},
    }


def _pix_to_lps(stack: Stack, k: int, r: float, c: float) -> FloatArray:
    return np.asarray(stack.pix_to_lps(k, r, c), dtype=np.float64)


def _targets_of_body(anchors: Anchors, z: float) -> list[str]:
    """Discs adjacent to the vertebral body at height ``z`` (for a void inside a body)."""
    # the body lies between two discs; the void is in the body between the discs either side of ``z``
    lo_disc = [d for d in anchors.discs if anchors.discs[d].center[2] < z]
    hi_disc = [d for d in anchors.discs if anchors.discs[d].center[2] >= z]
    out: list[str] = []
    if lo_disc:
        out.append(max(lo_disc, key=lambda d: anchors.discs[d].center[2]))
    if hi_disc:
        out.append(min(hi_disc, key=lambda d: anchors.discs[d].center[2]))
    return out


def detect_ghosting(stack: Stack, anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Motion ghosting: repeated structures along the phase (row) axis, found as a peak in the lag autocorrelation.

    The high-pass residual of each slice (image minus a 4 px Gaussian) is correlated with itself shifted by ``L`` rows,
    over the spinal window. Ghost copies of the anatomy add a peak at the ghost lag; anatomy alone gives a smooth decay
    and a broad peak at its own vertebral period. For every lag ``L`` in ``[3, rows/4)`` the prominence is the peak
    height above the largest value at least 3 lags away within +-4 lags. The strongest prominence is the ghost evidence
    (``quality_ghost_prominence``). Nothing assumes a ghost spacing: the lag is searched. Validated on the motion
    scenario (prominence about 0.24 at 16 rows, against at most 0.09 on the other phantom scenarios).
    """
    out: dict[str, Any] = {"present": False, "prominence": 0.0, "lag_rows": None, "slices": 0}
    if not anchors.discs:
        return out
    discs = _disc_pixels(stack, anchors)
    win = _window_mask(stack, [(r, c) for _, r, c in discs], cfg)
    vol = stack.vol.astype(np.float64)
    rows = vol.shape[1]
    lags = np.arange(3, rows // 4)
    if lags.size < 9 or win.sum() < 500:
        return out
    per_slice: list[NDArray[np.float64]] = []
    for k in range(vol.shape[0]):
        hp = vol[k] - ndimage.gaussian_filter(vol[k], 4.0)
        if hp[win].std() <= 0:
            continue
        row = np.full(lags.size, np.nan)
        for j, lag in enumerate(lags):
            a = hp[:-lag][win[:-lag]]
            b = hp[lag:][win[:-lag]]
            if a.size < 50 or a.std() <= 0 or b.std() <= 0:
                continue
            row[j] = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(row).any():
            per_slice.append(row)
    if not per_slice:
        return out
    mat = np.stack(per_slice)
    n_used = mat.shape[0]
    r = np.nanmean(mat, axis=0)

    def prominence(curve: NDArray[np.float64], j: int) -> float:
        ring = [curve[i] for i in range(j - 4, j + 5) if abs(i - j) >= 3 and np.isfinite(curve[i])]
        return float(curve[j] - max(ring)) if ring and np.isfinite(curve[j]) else 0.0

    # Repeated anatomy (the vertebral period, from the labelled disc spacing) is not a ghost: lags within a quarter of
    # the spacing of a multiple of it are excluded from the search. The evidence is the median over slices at the lag.
    zs = sorted(d.center[2] for d in anchors.discs.values())
    gaps = np.diff(zs)
    gap_rows = float(np.median(gaps)) / stack.row_mm if gaps.size else 0.0
    banned = np.zeros(lags.size, dtype=bool)
    if gap_rows > 0:
        for m in (1, 2, 3):
            banned |= np.abs(lags - m * gap_rows) <= 0.25 * gap_rows
    scores = [prominence(r, j) if not banned[j] else -1.0 for j in range(4, lags.size - 4)]
    if not scores or max(scores) <= -1.0:
        return out
    j_best = int(np.argmax(scores)) + 4
    lag = int(lags[j_best])
    per = [prominence(row, j_best) for row in mat]
    prom = float(np.median(per))
    present = prom >= cfg.quality_ghost_prominence
    out.update({"present": bool(present), "prominence": round(prom, 3), "lag_rows": lag, "slices": n_used})
    return out


def detect_wrap(stack: Stack, anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Phase wrap: an edge band of the field of view that is much brighter than the body of the image.

    Wrapped anatomy from beyond the far edge is folded onto the near edge, so the whole band (not only the spine)
    carries the background of the far region on top of its own: the mean row intensity of the outer
    ``quality_wrap_band_fraction`` of the rows is compared with the median row intensity of the middle third. A ratio at
    or above ``quality_wrap_edge_ratio`` flags that edge; the disc nearest that edge is the flagged level. Validated
    on the aliasing scenario of both phantom looks (ratio about 1.9-2.0) against the baseline and six other scenarios
    (0.9-1.05). Returns the edge, the ratio and the flagged level.
    """
    out: dict[str, Any] = {"present": False, "edge": None, "ratio": 0.0, "levels": {}}
    if not anchors.discs:
        return out
    vol = stack.vol.astype(np.float64)
    rowp = vol.mean(axis=(0, 2))
    n = rowp.size
    b = max(1, round(cfg.quality_wrap_band_fraction * n))
    mid = float(np.median(rowp[n // 3 : 2 * n // 3]))
    if mid <= 0:
        return out
    ratios = {"superior": float(rowp[:b].mean() / mid), "inferior": float(rowp[-b:].mean() / mid)}
    edge = max(ratios, key=lambda e: ratios[e])
    ratio = ratios[edge]
    if ratio < cfg.quality_wrap_edge_ratio:
        out["ratio"] = round(ratio, 3)
        return out
    if edge == "superior":
        outer = max(anchors.discs, key=lambda d: anchors.discs[d].center[2])
    else:
        outer = min(anchors.discs, key=lambda d: anchors.discs[d].center[2])
    conf = float(min(1.0, (ratio - 1.0) / max(cfg.quality_wrap_edge_ratio - 1.0, 1e-6) * 0.5 + 0.5))
    out.update({"present": True, "edge": edge, "ratio": round(ratio, 3), "levels": {outer: conf}})
    return out


def detect_fat_sat(stir: Stack, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Failed fat suppression on STIR: the subcutaneous rim is as bright as muscle.

    On a STIR image fat is dark. The body is the pixels above 0.15 of the 90th percentile; the rim is the body minus
    its 15 mm erosion, the interior is the erosion minus its 25 mm erosion. Returns the rim-over-interior ratio;
    ``present`` when it reaches ``quality_stir_fat_ratio``. Study-level only (fat is not one level).
    """
    vol = stir.vol.astype(np.float64)
    ref = float(np.percentile(vol, 90))
    if ref <= 0:
        return {"present": False, "ratio": None}
    rs = max(1, round(15.0 / stir.row_mm))
    ks = max(1, round(25.0 / stir.row_mm))
    ratios: list[float] = []
    for k in range(vol.shape[0]):
        b = ndimage.gaussian_filter(vol[k], 1.0) > 0.15 * ref
        b = ndimage.binary_fill_holes(b)
        if b.sum() < 500:
            continue
        erode1 = ndimage.binary_erosion(b, iterations=rs)
        erode2 = ndimage.binary_erosion(b, iterations=ks)
        rim = b & ~erode1
        interior = erode2
        if rim.sum() < 50 or interior.sum() < 50:
            continue
        ratios.append(float(np.median(vol[k][rim]) / np.median(vol[k][interior])))
    if not ratios:
        return {"present": False, "ratio": None}
    ratio = float(np.median(ratios))
    return {"present": bool(ratio >= cfg.quality_stir_fat_ratio), "ratio": round(ratio, 3)}


def detect_snr(stack: Stack, anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Tissue reference over background noise (robust sigma of the air). Study-level."""
    vol = stack.vol.astype(np.float64)
    ref = float(np.percentile(vol, 90))
    air = vol[vol < 0.08 * ref]
    if air.size < 500 or ref <= 0:
        return {"present": False, "snr": None}
    sigma = 1.4826 * float(np.median(np.abs(air - np.median(air))))
    if sigma <= 0:
        return {"present": False, "snr": None}
    snr = ref / sigma
    return {"present": bool(snr < cfg.quality_min_snr), "snr": round(float(snr), 2)}


def detect_bias(stack: Stack, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Severe intensity bias: 90th over 10th percentile of tissue intensity along the anterior-posterior profile.

    Tissue is the pixels above 0.3 of the 90th percentile, in columns that carry tissue in at least 20 % of rows.
    The ratio is the profile contrast; a surface coil gradient makes it large. Study-level.
    """
    vol = stack.vol.astype(np.float64)
    ref = float(np.percentile(vol, 90))
    if ref <= 0:
        return {"present": False, "ratio": None}
    tissue = vol > 0.3 * ref
    frac = tissue.mean(axis=(0, 1))
    cols = np.nonzero(frac >= 0.2)[0]
    if cols.size < 8:
        return {"present": False, "ratio": None}
    prof = np.array([float(np.median(vol[:, :, c][tissue[:, :, c]])) for c in cols])
    lo = float(np.percentile(prof, 10))
    ratio = float(np.percentile(prof, 90) / lo) if lo > 0 else float("inf")
    return {"present": bool(ratio >= cfg.quality_bias_ratio), "ratio": round(ratio, 3)}


def _sagittal_primary(study: Study) -> Stack | None:
    return study.best("sag")


def _second_sag_stir(study: Study) -> Stack | None:
    for st in study.find("sag"):
        if "stir" in st.key or "stir" in str(st.description).lower():
            return st
    return None


def assess_quality(study: Study, anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Hardware and artefact flags of a labelled study, per level and per study.

    Pure: does not change ``anchors``. Runs on the primary sagittal stack the labels came from (and the STIR stack for
    fat suppression when one is present); missing inputs give warnings and no flags. Returns the ``quality`` record
    (see the module docstring); :func:`apply_quality` turns it into caps.
    """
    rec: dict[str, Any] = {
        "method": METHOD,
        "study": {"score": 1.0, "flags": {}},
        "levels": {},
        "caps": [],
        "warnings": [],
    }
    stack = _sagittal_primary(study) if study is not None else None
    if stack is None or not anchors.discs:
        rec["warnings"].append("quality not assessed: no sagittal stack or no labelled levels")
        return rec
    metal = detect_metal(stack, anchors, cfg)
    ghost = detect_ghosting(stack, anchors, cfg)
    wrap = detect_wrap(stack, anchors, cfg)
    snr = detect_snr(stack, anchors, cfg)
    bias = detect_bias(stack, cfg)
    stir_stack = _second_sag_stir(study)
    fat = detect_fat_sat(stir_stack, cfg) if stir_stack is not None else {"present": False, "ratio": None}
    if stir_stack is None:
        rec.setdefault("notes", []).append("fat suppression not assessed: no STIR sagittal series")

    level_flags: dict[str, dict[str, float]] = {lab: {} for lab in anchors.discs}

    def add(kind: str, lab: str, conf: float) -> None:
        if lab in level_flags and conf > 0:
            level_flags[lab][kind] = max(level_flags[lab].get(kind, 0.0), float(min(1.0, conf)))

    kinds = metal.get("kinds", {})
    by_kind = metal.get("by_disc_kind", {})
    for lab, conf in metal.get("levels", {}).items():
        kinds_here = by_kind.get(lab, [])
        if "metal" in kinds_here:
            add("metal", lab, conf)
        if "disc_space" in kinds_here:
            add("fusion_or_cage", lab, conf)
    for lab, conf in wrap.get("levels", {}).items():
        add("wrap", lab, conf)
    study_flags: dict[str, dict[str, Any]] = {
        "metal": {
            "present": bool(metal.get("levels")),
            "confidence": round(max(metal.get("levels", {}).values(), default=0.0), 3),
            "detail": kinds,
        },
        "fusion_or_cage": {
            "present": any("disc_space" in v for v in by_kind.values()),
            "confidence": round(
                max(
                    (c for lab, c in metal.get("levels", {}).items() if "disc_space" in by_kind.get(lab, [])),
                    default=0.0,
                ),
                3,
            ),
        },
        "motion_ghost": {
            "present": bool(ghost["present"]),
            "confidence": round(min(1.0, ghost["prominence"] / 0.3), 3) if ghost["present"] else 0.0,
            "detail": ghost,
        },
        "wrap": {
            "present": bool(wrap["present"]),
            "confidence": round(max(wrap["levels"].values(), default=0.0), 3),
            "detail": wrap,
        },
        # Advisory: the STIR rim rule is not validated (the phantom skin is soft tissue), so it is reported, never capped.
        "fat_sat_failure": {
            "present": False,
            "advisory": True,
            "would_flag": bool(fat["present"]),
            "confidence": 0.0,
            "detail": fat,
        },
        "low_snr": {
            "present": bool(snr["present"]),
            "confidence": 0.9 if snr["present"] else 0.0,
            "detail": snr,
        },
        # Advisory: the anatomy confounds the profile (an iso 1 mm phantom reads as biased, the bias phantom barely),
        # so it is reported with its ratio and never flagged or capped (docs/labeling/lab31_artifacts.md).
        "bias_field": {
            "present": False,
            "advisory": True,
            # Never a hit: the ratio fires on most real studies (SPIDER 66 of 87, Sudirman 448 of 513) and is kept as detail.
            "would_flag": False,
            "confidence": 0.0,
            "detail": bias,
        },
    }
    # A study-level flag that covers every level (ghosting, low SNR, bias) is applied to each level at its confidence.
    for kind in ("motion_ghost", "low_snr"):
        if study_flags[kind]["present"]:
            for lab in anchors.discs:
                add(kind, lab, float(study_flags[kind]["confidence"]))

    cfgmin = cfg.quality_flag_min_confidence
    capping = set(cfg.quality_capping_kinds)
    for lab, flags in level_flags.items():
        flagged = sorted(k for k, v in flags.items() if k in capping and v >= cfgmin)
        advisory = sorted(k for k, v in flags.items() if k not in capping and v >= cfgmin)
        rec["levels"][lab] = {
            "flags": {k: round(v, 3) for k, v in flags.items()},
            "flagged": flagged,
            "advisory": advisory,
            "confidence": round(max((v for k, v in flags.items() if k in capping), default=0.0), 3),
        }
    # Advisory kinds are reported (present False, would_flag True when the evidence passes) and never cap or score.
    for kind, f in study_flags.items():
        raw = bool(f.get("present") or f.get("would_flag"))
        if kind in capping:
            f["present"] = raw
        else:
            f["advisory"] = True
            f["present"] = False
            f["would_flag"] = raw
    rec["study"]["flags"] = study_flags
    worst = max((v["confidence"] for v in rec["levels"].values()), default=0.0)
    rec["study"]["score"] = round(float(1.0 - worst), 3)
    for kind, f in study_flags.items():
        if kind in capping and f["present"] and f["confidence"] >= cfgmin:
            rec["warnings"].append(
                f"image quality: {kind.replace('_', ' ')} (confidence {f['confidence']:.2f})"
            )
        elif kind not in capping and f.get("would_flag"):
            note = f"image quality, advisory (not capped, unvalidated): {kind.replace('_', ' ')}"
            # Advisory detectors are not specific on real data: keep them out of the user-facing warnings
            # unless asked for (Config.quality_show_advisory).
            (rec["warnings"] if cfg.quality_show_advisory else rec.setdefault("notes", [])).append(note)
    return rec


def apply_quality(anchors: Anchors, quality: dict[str, Any], cfg: Config = DEFAULT) -> dict[str, Any]:
    """Store the quality record on the anchors and cap what the flags cover.

    * a flagged level (flag confidence at or above ``quality_flag_min_confidence``) gets its disc confidence capped at
      ``quality_disc_cap``; the raw value is kept in the disc's evidence, and the cap in ``quality['caps']``;
    * a confirmed numbering becomes probable when any flagged level exists (its names need confirming there);
    * a study score below 0.3 caps the numbering to unconfirmed.

    The numbering confidence number itself is not changed: only the status and a record of the cap. Returns the
    record as stored. Idempotent for the same record.
    """
    anchors.quality = quality
    cfgmin = cfg.quality_flag_min_confidence
    caps: list[dict[str, Any]] = []
    flagged_levels = []
    capping = set(cfg.quality_capping_kinds)
    for lab, lv in quality.get("levels", {}).items():
        kinds = [k for k in lv.get("flagged", []) if k in capping]
        if lab not in anchors.discs or not kinds or lv.get("confidence", 0.0) < cfgmin:
            continue
        flagged_levels.append(lab)
        d = anchors.discs[lab]
        raw = float(d.confidence)
        if raw > cfg.quality_disc_cap:
            reason = ", ".join(kinds)
            d.confidence = float(cfg.quality_disc_cap)
            d.evidence.append(
                f"confidence capped {raw:.2f} -> {cfg.quality_disc_cap:.2f} by image quality ({reason}); "
                "the calibrated value is kept in quality caps"
            )
            caps.append(
                {
                    "level": lab,
                    "raw": round(raw, 3),
                    "capped": round(cfg.quality_disc_cap, 3),
                    "reason": reason,
                }
            )
    quality["caps"] = caps
    num = anchors.numbering
    if num and anchors.discs:
        status = str(num.get("status", "") or "")
        study_score = float(quality.get("study", {}).get("score", 1.0))
        new = status
        reason = ""
        if flagged_levels and status == STATUS_CONFIRMED:
            new = STATUS_PROBABLE
            reason = f"image quality flags on {', '.join(flagged_levels)}"
        if study_score < 0.3 and status in (STATUS_CONFIRMED, STATUS_PROBABLE):
            new = STATUS_UNCONFIRMED
            reason = "study image quality score below 0.3"
        if new != status:
            num["status"] = new
            num["quality_cap"] = {"from": status, "to": new, "reason": reason}
            anchors.warnings.append(f"numbering capped to {new} by image quality: {reason}")
    anchors.warnings.extend(w for w in quality.get("warnings", []) if w not in anchors.warnings)
    return quality


__all__ = [
    "FLAGS",
    "METHOD",
    "apply_quality",
    "assess_quality",
    "detect_bias",
    "detect_fat_sat",
    "detect_ghosting",
    "detect_metal",
    "detect_snr",
    "detect_wrap",
]
