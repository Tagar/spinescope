"""Paraspinal and psoas muscles on the axial slices of the L3-L4, L4-L5 and L5-S1 levels.

For each level the axial slab (or axial stack) that cuts the disc is taken from :func:`spinescope.anatomy.level_map`.
On its slice through the disc, three regions per side are outlined from anatomy alone, in millimetres from the
midline (patient left positive) and from the body centre (posterior positive), scaled by the level's body
geometry:

* multifidus (MF): medial, behind the canal and lamina;
* erector spinae (ES): lateral to the multifidus, behind the canal level;
* psoas (PS): lateral to the body, around its middle.

Positions come from the slice's own LPS coordinates, so the outline does not depend on the image orientation, the
row order or a flipped acquisition. Intensity references are per slice: the subcutaneous fat (a ring inside the skin)
and the lean psoas (or erector spinae if the psoas is mostly fat). A pixel brighter than the midpoint of the fat and
muscle references is fat-bright. Outputs per muscle and side: CSA (whole outline), functional CSA (lean, outline minus
fat-bright pixels), fat fraction (fat-bright share of the outline), CSA normalised to the body's cross-section, and the
Kjaer 2007 three-grade multifidus fat infiltration. Left/right asymmetry is given per level and series.

Oblique slabs: the measured area is that of a cut through the muscle at the slab angle. The perpendicular area is
``area * cos(tilt)`` with the tilt between the slab normal and the disc normal (the spine axis). Above
``muscles_oblique_tilt_max_deg`` the confidence is lowered. Normalised CSA uses the same slab area for muscle and body,
so it does not need the tilt correction.

Nothing here diagnoses. Muscle outlines are approximations (no fascia is segmented), and the thresholds are local
to each slice (no age or sex normalisation: no demographics are used). See ``docs/analysis/muscles.md``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from ..anatomy.anchors import Anchors
from ..anatomy.correspondence import CorrespondenceConfig, LevelEntry, LevelMap, level_map
from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .base import Result, register

log = logging.getLogger(__name__)

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]

#: Levels analysed (disc labels). L5-S1 needs an axial slab through the sacral disc; absent slabs give a warning.
LEVELS: tuple[str, ...] = ("L3-L4", "L4-L5", "L5-S1")
#: Muscle codes: multifidus, erector spinae, psoas.
MUSCLES: tuple[str, ...] = ("MF", "ES", "PS")
MUSCLE_NAME: dict[str, str] = {"MF": "multifidus", "ES": "erector spinae", "PS": "psoas"}
#: Patient sides ('L' = patient left, +x in LPS).
SIDES: tuple[str, ...] = ("L", "R")

#: Result name and the method string stored in ``meta``.
METHOD = (
    "axial slice through the disc (level_map slab), anatomy windows in mm from the midline and body centre, "
    "per-slice fat and lean references, fat-bright share at the fat/muscle midpoint"
)


# ------------------------------------------------------------------ small pure helpers
def kjaer_grade(fat_fraction: float, bounds: tuple[float, float] = DEFAULT.muscles_kjaer_bounds) -> int:
    """Kjaer et al. 2007 multifidus grade from a fat fraction in [0, 1]: 0 below 10 %, 1 up to 50 %, 2 above.

    The paper's ranges (0-10 %, 10-50 %, >50 %) overlap at 10 %; 10 % itself is grade 1 here.
    """
    lo, hi = bounds
    if fat_fraction < lo:
        return 0
    if fat_fraction <= hi:
        return 1
    return 2


def perpendicular_area(area_mm2: float, tilt_deg: float) -> float:
    """Area of the cut perpendicular to the spine axis: ``area * cos(tilt)`` for a slab tilted ``tilt_deg`` from it."""
    return float(area_mm2 * math.cos(math.radians(min(abs(tilt_deg), 89.0))))


def asymmetry_pct(left: float, right: float) -> float:
    """Left minus right as a percentage of their mean (NaN if either is NaN or both are zero)."""
    if not (math.isfinite(left) and math.isfinite(right)):
        return float("nan")
    mean = 0.5 * (left + right)
    if mean == 0.0:
        return float("nan")
    return 100.0 * (left - right) / mean


def largest_component(mask: BoolArray) -> BoolArray:
    """Largest 8-connected component of ``mask``, holes filled (empty in, empty out)."""
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n > 1:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(mask)


def air_threshold(img: FloatArray, n_sd: float = 3.0, corner_frac: float = 0.12) -> float:
    """Tissue threshold from the measured background: mean + ``n_sd`` SD of the two darkest corner boxes (taken as air).

    Muscle on T2 can sit only a few times above the air noise (measured on a Sudirman T2: muscle ~40-60, noise ~20),
    so a fat-relative threshold drops it. The two darker corners are used because the posterior corners can hold
    the back. NaN if even the darker corners are not air (bright relative to the slice).
    """
    h, w = img.shape
    dh, dw = max(round(corner_frac * h), 2), max(round(corner_frac * w), 2)
    boxes = [img[:dh, :dw], img[:dh, -dw:], img[-dh:, :dw], img[-dh:, -dw:]]
    boxes.sort(key=lambda b: float(np.mean(b)))
    air = np.concatenate([b.ravel() for b in boxes[:2]])
    bg = float(np.mean(air))
    sd = float(np.std(air))
    hi = float(np.percentile(img, 90.0))
    if not (hi > bg and sd > 0.0) or bg + n_sd * sd >= 0.5 * hi:
        return float("nan")
    return bg + n_sd * sd


def body_outline(img: FloatArray, threshold_rel: float = DEFAULT.muscles_body_threshold_rel) -> BoolArray:
    """Largest tissue component of the smoothed slice, holes filled.

    Threshold: the air threshold (corner noise, :func:`air_threshold`); if the corners carry signal, the fraction
    ``threshold_rel`` of the way from the 5th to the 90th percentile.
    """
    sm = ndi.gaussian_filter(img, 1.0)
    thr = air_threshold(sm)
    if not math.isfinite(thr):
        lo, hi = (float(x) for x in np.percentile(sm, [5.0, 90.0]))
        if not hi > lo:
            return np.zeros(img.shape, dtype=bool)
        thr = lo + threshold_rel * (hi - lo)
    m = ndi.binary_opening(sm > thr, iterations=2)
    return largest_component(m)


# ------------------------------------------------------------------ geometry of one slice
@dataclass(frozen=True)
class LevelFrame:
    """Millimetre coordinates of every pixel of one axial slice, relative to the level's midline and body centre.

    ``lat``: patient-left positive, from the midline. ``ap``: posterior positive, from the body centre. Both (H, W).
    """

    lat: FloatArray
    ap: FloatArray
    row_mm: float
    col_mm: float
    half_width_mm: float
    half_depth_mm: float
    measured: bool  # True when the body size comes from the labeller's body geometry, not the defaults


def level_frame(
    stack: Stack,
    k: int,
    centre_xyz: FloatArray,
    midline_x: float | None,
    half_width_mm: float,
    half_depth_mm: float,
    measured: bool,
) -> LevelFrame:
    """Pixel coordinates in mm for slice ``k`` of an axial ``stack``. The LPS geometry decides left/right and AP."""
    h, w = stack.vol.shape[1:]
    rr, cc = np.mgrid[0:h, 0:w]
    xyz = stack.pix_to_lps_many(float(k), np.column_stack([rr.ravel(), cc.ravel()]).astype(np.float64))
    x0 = (
        float(midline_x)
        if midline_x is not None and math.isfinite(float(midline_x))
        else float(centre_xyz[0])
    )
    lat = (xyz[:, 0] - x0).reshape(h, w)
    ap = (xyz[:, 1] - float(centre_xyz[1])).reshape(h, w)
    return LevelFrame(
        lat=lat,
        ap=ap,
        row_mm=float(stack.row_mm),
        col_mm=float(stack.col_mm),
        half_width_mm=float(half_width_mm),
        half_depth_mm=float(half_depth_mm),
        measured=measured,
    )


def muscle_windows(frame: LevelFrame, cfg: Config) -> dict[tuple[str, str], BoolArray]:
    """Anatomy windows ``(muscle, side) -> mask`` before intensity or the body outline are applied."""
    hw, hd = frame.half_width_mm, frame.half_depth_mm
    out: dict[tuple[str, str], BoolArray] = {}
    lat_frac = {"MF": cfg.muscles_mf_lat_frac, "ES": cfg.muscles_es_lat_frac, "PS": cfg.muscles_ps_lat_frac}
    ap_frac = {"MF": cfg.muscles_mf_ap_frac, "ES": cfg.muscles_es_ap_frac, "PS": cfg.muscles_ps_ap_frac}
    for mu in MUSCLES:
        (l0, l1), (a0, a1) = lat_frac[mu], ap_frac[mu]
        in_ap = (frame.ap >= a0 * hd) & (frame.ap <= a1 * hd)
        for side in SIDES:
            sign = 1.0 if side == "L" else -1.0
            lat_abs = sign * frame.lat  # positive on this side
            out[(mu, side)] = (lat_abs >= l0 * hw) & (lat_abs <= l1 * hw) & in_ap
    return out


# ------------------------------------------------------------------ per-slice measurement
@dataclass
class SliceMeasure:
    """Everything measured on one slice: outlines, references and the body outline."""

    masks: dict[tuple[str, str], BoolArray]
    fat: dict[tuple[str, str], BoolArray]
    body: BoolArray
    f_ref: float
    m_ref: float
    threshold: float
    reference: str  # "psoas" | "erector spinae" | "none"


def fat_reference(sm: FloatArray, body: BoolArray, frame: LevelFrame, cfg: Config) -> float:
    """Subcutaneous fat level F on this slice: median of the bright pixels of the skin ring. NaN if there is none."""
    dist = ndi.distance_transform_edt(body, sampling=(frame.row_mm, frame.col_mm))
    ring = body & (dist > 0) & (dist < cfg.muscles_subcut_ring_mm)
    if not ring.any():
        return float("nan")
    fat_ring = ring & (sm > 0.5 * float(np.percentile(sm[ring], 90.0)))
    return float(np.median(sm[fat_ring])) if fat_ring.any() else float("nan")


def measure_slice(img: FloatArray, frame: LevelFrame, cfg: Config = DEFAULT) -> SliceMeasure:
    """Outlines, references and the fat-bright masks for one slice (``img`` is the raw slice, float)."""
    sm = ndi.gaussian_filter(img, 1.0)
    body = body_outline(img, cfg.muscles_body_threshold_rel)
    f_ref = fat_reference(sm, body, frame, cfg)
    dist = ndi.distance_transform_edt(body, sampling=(frame.row_mm, frame.col_mm))
    ring = body & (dist > 0) & (dist < cfg.muscles_subcut_ring_mm)
    windows = muscle_windows(frame, cfg)
    masks: dict[tuple[str, str], BoolArray] = {}
    for key, win in windows.items():
        roi = win & body & ~ring
        masks[key] = largest_component(roi) if roi.any() else roi

    # Lean muscle reference: psoas core that is not fat-bright; erector spinae if the psoas is mostly fat.
    m_ref = float("nan")
    reference = "none"
    if math.isfinite(f_ref):
        ps_core = np.zeros_like(body)
        for side in SIDES:
            ps_core |= masks[("PS", side)]
        ps_core = ndi.binary_erosion(ps_core, iterations=2) if ps_core.any() else ps_core
        if ps_core.any():
            lean_ps = ps_core & (sm < cfg.muscles_lean_fraction * f_ref)
            share_fat = float(np.mean(sm[ps_core] >= cfg.muscles_lean_fraction * f_ref))
            if share_fat <= cfg.muscles_psoas_fat_max_share and lean_ps.sum() >= cfg.muscles_min_roi_pixels:
                m_ref = float(np.median(sm[lean_ps]))
                reference = "psoas"
        if not math.isfinite(m_ref):
            es = np.zeros_like(body)
            for side in SIDES:
                es |= masks[("ES", side)]
            lean_es = es & (sm < cfg.muscles_lean_fraction * f_ref)
            if lean_es.sum() >= cfg.muscles_min_roi_pixels:
                m_ref = float(np.median(sm[lean_es]))
                reference = "erector spinae"
    thr = 0.5 * (f_ref + m_ref) if math.isfinite(f_ref) and math.isfinite(m_ref) else float("nan")
    fat: dict[tuple[str, str], BoolArray] = {}
    for key, m in masks.items():
        fat[key] = (m & (sm > thr)) if math.isfinite(thr) else np.zeros_like(m)
    return SliceMeasure(
        masks=masks, fat=fat, body=body, f_ref=f_ref, m_ref=m_ref, threshold=thr, reference=reference
    )


def _truncated(m: BoolArray) -> bool:
    return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


# ------------------------------------------------------------------ slice selection per level
@dataclass(frozen=True)
class LevelSlice:
    """One axial slice through a level, with the stack that holds it and its tilt to the disc normal."""

    level: str
    stack_key: str
    contrast: str
    k: int
    tilt_deg: float
    out_of_plane_mm: float
    entry: LevelEntry


def _contrast(key: str) -> str:
    for c in ("t1", "t2", "stir"):
        if c in key:
            return c
    return "other"


def select_axial_slices(lm: LevelMap, level: str) -> list[LevelSlice]:
    """Axial slices through ``level``: the preferred slab first, then the same slab in the other contrast if covered.

    Empty when no axial stack shows the disc in its slab and field of view.
    """
    entry = lm.discs.get(level)
    if entry is None:
        return []
    cands: list[tuple[int, str]] = []
    for key, h in entry.stacks.items():
        if h.plane != "ax" or not (h.in_slab and h.in_fov):
            continue
        rank = 0 if key == entry.slab else 1
        cands.append((rank, key))
    out: list[LevelSlice] = []
    seen_contrast: set[str] = set()
    for _rank, key in sorted(cands, key=lambda t: (t[0], abs(entry.stacks[t[1]].out_of_plane_mm), t[1])):
        h = entry.stacks[key]
        c = _contrast(key)
        if c in seen_contrast:
            continue
        seen_contrast.add(c)
        out.append(
            LevelSlice(
                level=level,
                stack_key=key,
                contrast=c,
                k=int(h.slice),
                tilt_deg=float(h.tilt_deg),
                out_of_plane_mm=float(h.out_of_plane_mm),
                entry=entry,
            )
        )
    return out


def _body_dims(entry: LevelEntry) -> tuple[float, float, bool]:
    """Half width, half AP depth (mm) of the level's body and whether they were measured (else correspondence defaults)."""
    body = entry.geometry.get("body") or {}
    hw = body.get("half_width_mm")
    hd = body.get("half_depth_mm")
    if hw and hd:
        return float(hw), float(hd), body.get("source") == "measured"
    dflt = CorrespondenceConfig()
    return (
        dflt.default_body_width_mm / 2.0,
        dflt.default_body_width_mm * dflt.body_depth_to_width / 2.0,
        False,
    )


def _level_centre(lm: LevelMap, level: str) -> FloatArray:
    return np.asarray(lm.discs[level].centre, dtype=np.float64)


# ------------------------------------------------------------------ rows and analysis
def _slice_rows(
    ls: LevelSlice,
    stack: Stack,
    meas: SliceMeasure,
    frame: LevelFrame,
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Per-muscle rows for one slice, and the per-(muscle, side) values used for asymmetry."""
    area_px = frame.row_mm * frame.col_mm
    tilt = ls.tilt_deg
    cos_t = math.cos(math.radians(min(abs(tilt), 89.0)))
    body_area_slab = math.pi * frame.half_width_mm * frame.half_depth_mm
    rows: list[dict[str, Any]] = []
    vals: dict[tuple[str, str], dict[str, Any]] = {}
    for mu in MUSCLES:
        for side in SIDES:
            key = (mu, side)
            m = meas.masks.get(key, np.zeros(stack.vol.shape[1:], dtype=bool))
            n = int(m.sum())
            base: dict[str, Any] = {
                "scope": "muscle",
                "level": ls.level,
                "series": ls.stack_key,
                "contrast": ls.contrast,
                "k": int(ls.k),
                "muscle": mu,
                "muscle_name": MUSCLE_NAME[mu],
                "side": side,
                "tilt_deg": round(tilt, 2),
                "out_of_plane_mm": round(ls.out_of_plane_mm, 2),
                "body_csa_mm2": round(body_area_slab * cos_t, 1),
                "geometry": "measured" if frame.measured else "default",
            }
            reason = ""
            if n < cfg.muscles_min_roi_pixels:
                reason = "outline too small or outside the window"
            elif not math.isfinite(meas.threshold):
                reason = "no fat or lean muscle reference on this slice"
            if reason:
                rows.append({**base, "n_pixels": n, "value": None, "reason": reason, "confidence": 0.0})
                continue
            fat_n = int(meas.fat[key].sum())
            area_slab = n * area_px
            fat_pct = 100.0 * fat_n / n
            truncated = _truncated(m)
            conf = 0.5 if frame.measured else 0.35
            notes: list[str] = []
            if abs(tilt) > cfg.muscles_oblique_tilt_max_deg:
                conf *= 0.6
                notes.append(
                    f"oblique slab ({tilt:.0f} deg to the disc normal): CSA corrected by cos(tilt), lower confidence"
                )
            if truncated:
                conf *= 0.6
                notes.append("outline touches the field-of-view edge")
            if meas.reference != "psoas":
                notes.append(f"lean reference from {meas.reference}, not psoas")
            fat_frac = fat_n / n
            row = {
                **base,
                "n_pixels": n,
                "csa_slab_mm2": round(area_slab, 1),
                "csa_mm2": round(perpendicular_area(area_slab, tilt), 1),
                "functional_csa_mm2": round(perpendicular_area((n - fat_n) * area_px, tilt), 1),
                "fat_fraction_pct": round(fat_pct, 1),
                "normalised_csa": round(area_slab / body_area_slab, 4),
                "truncated": truncated,
                "f_ref": round(meas.f_ref, 2),
                "m_ref": round(meas.m_ref, 2),
                "reference": meas.reference,
                "threshold": round(meas.threshold, 2),
                "value": "measured",
                "reason": "; ".join(notes),
                "confidence": round(min(conf, cfg.muscles_max_confidence), 3),
            }
            if mu == "MF":
                row["kjaer_grade"] = kjaer_grade(fat_frac, cfg.muscles_kjaer_bounds)
            rows.append(row)
            vals[key] = row
    return rows, vals


def _asymmetry_rows(
    ls: LevelSlice, vals: dict[tuple[str, str], dict[str, Any]], cfg: Config
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mu in MUSCLES:
        left, right = vals.get((mu, "L")), vals.get((mu, "R"))
        if left is None or right is None:
            continue
        out.append(
            {
                "scope": "asymmetry",
                "level": ls.level,
                "series": ls.stack_key,
                "contrast": ls.contrast,
                "muscle": mu,
                "csa_asym_pct": round(asymmetry_pct(left["csa_mm2"], right["csa_mm2"]), 1),
                "functional_csa_asym_pct": round(
                    asymmetry_pct(left["functional_csa_mm2"], right["functional_csa_mm2"]), 1
                ),
                "fat_asym_pct": round(asymmetry_pct(left["fat_fraction_pct"], right["fat_fraction_pct"]), 1),
                "flag": abs(asymmetry_pct(left["csa_mm2"], right["csa_mm2"]))
                > cfg.muscles_asymmetry_flag_pct,
            }
        )
    return out


def _overlay(
    img: FloatArray, masks: dict[tuple[str, str], BoolArray], zoom: int = 3, margin: int = 8
) -> NDArray[np.uint8]:
    """Grey crop of the slice around the outlines, zoomed, with each outline drawn in its muscle colour."""
    colour = {"MF": (0, 255, 0), "ES": (0, 160, 255), "PS": (255, 128, 0)}
    union = np.zeros(img.shape, dtype=bool)
    for m in masks.values():
        union |= m
    if not union.any():
        union[:] = True
    rr, cc = np.nonzero(union)
    r0 = max(int(np.min(rr)) - margin, 0)
    r1 = min(int(np.max(rr)) + margin + 1, img.shape[0])
    c0 = max(int(np.min(cc)) - margin, 0)
    c1 = min(int(np.max(cc)) + margin + 1, img.shape[1])
    crop = img[r0:r1, c0:c1]
    lo, hi = np.percentile(crop, [0.5, 99.5])
    g = np.clip((crop - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    g8 = (255 * g).astype(np.uint8)
    rgb = np.repeat(np.repeat(np.stack([g8, g8, g8], -1), zoom, axis=0), zoom, axis=1)
    for (mu, _side), m in masks.items():
        mm = m[r0:r1, c0:c1]
        edge = mm & ~ndi.binary_erosion(mm)
        e = np.repeat(np.repeat(edge, zoom, axis=0), zoom, axis=1)
        rgb[e] = colour[mu]
    return rgb


@register("muscles")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Paraspinal and psoas CSA, functional CSA, fat fraction, normalised CSA, asymmetry and Kjaer grade per level."""
    cfg = config or DEFAULT
    res = Result(
        name="muscles",
        meta={
            "method": METHOD,
            "levels": list(LEVELS),
            "muscles": list(MUSCLES),
            "kjaer_bounds": list(cfg.muscles_kjaer_bounds),
            "oblique_tilt_max_deg": cfg.muscles_oblique_tilt_max_deg,
            "asymmetry_flag_pct": cfg.muscles_asymmetry_flag_pct,
            "max_confidence": cfg.muscles_max_confidence,
            "status": "experimental: see docs/analysis/muscles.md",
        },
    )
    if not any(s.plane == "ax" for s in study.stacks.values()):
        res.warnings.append("no axial stack: paraspinal and psoas muscles not assessed")
        return res
    if not anchors.discs:
        res.warnings.append("no discs in anchors: muscles not assessed")
        return res
    try:
        lm = level_map(study, anchors, measure_geometry=False)
    except Exception as exc:  # the level map is additive for this module; report, do not raise
        log.exception("level map failed")
        res.warnings.append(f"level map failed: {type(exc).__name__}: {exc}")
        return res
    for level in LEVELS:
        if level not in lm.discs:
            res.warnings.append(f"{level}: no such disc in the labelled spine: not assessed")
            continue
        slices = select_axial_slices(lm, level)
        if not slices:
            res.warnings.append(f"{level}: no axial slab covers the disc: not assessed")
            continue
        hw, hd, measured = _body_dims(lm.discs[level])
        centre = _level_centre(lm, level)
        for ls in slices:
            stack = study.stacks[ls.stack_key]
            img = stack.vol[ls.k].astype(np.float64)
            frame = level_frame(stack, ls.k, centre, anchors.midline_x, hw, hd, measured)
            meas = measure_slice(img, frame, cfg)
            rows, vals = _slice_rows(ls, stack, meas, frame, cfg)
            res.rows.extend(rows)
            asym = _asymmetry_rows(ls, vals, cfg)
            res.rows.extend(asym)
            for r in rows:
                if r.get("kjaer_grade") is not None and r["kjaer_grade"] >= 1:
                    res.flags.append(
                        {
                            "kind": "multifidus_fat",
                            "level": level,
                            "series": ls.stack_key,
                            "side": r["side"],
                            "kjaer_grade": r["kjaer_grade"],
                            "fat_fraction_pct": r["fat_fraction_pct"],
                            "confidence": r["confidence"],
                        }
                    )
            for a in asym:
                if a["flag"]:
                    res.flags.append(
                        {
                            "kind": "paraspinal_asymmetry",
                            "level": level,
                            "series": ls.stack_key,
                            "muscle": a["muscle"],
                            "csa_asym_pct": a["csa_asym_pct"],
                            "confidence": round(min(0.5, cfg.muscles_max_confidence), 3),
                        }
                    )
            if ls is slices[0]:
                res.figures[f"ax_{level}"] = _overlay(img, meas.masks)
    if not res.rows:
        res.warnings.append("no paraspinal outline could be measured")
    return res
