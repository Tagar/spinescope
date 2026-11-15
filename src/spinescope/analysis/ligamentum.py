"""Ligamentum flavum thickness and dorsal epidural fat on the axial plane, at each disc level (experimental).

What is measured, per labelled disc:

* **Ligamentum flavum (LF) thickness, per side.** The axial slice nearest the disc (T2 preferred, T1 accepted at lower
  confidence) is sampled along rays that leave the dural sac posterolaterally (lateral sweep about the posterior
  direction of the disc, see ``config.ligamentum_ray_deg``). Along a ray the LF is the intermediate-dark band that lies
  between the bright dural sac and the posterior elements. Its thickness is the length of the run around the band's
  minimum where the (smoothed) signal stays below the midpoint between the minimum and the brighter tissue behind it.
  A run that is not closed on both sides, or whose contrast is below ``ligamentum_min_cnr`` noise sigmas, is not
  measured. The level value is the maximum over the rays of a side, i.e. the largest thickness found along a ray.
* **Dorsal epidural fat (midline, T1 only).** Along the posterior midline ray, the fat is the bright run that starts
  at the dural edge and lies above the midpoint between the dural (CSF) level and the fat level of the slice (95th
  percentile). On T2 the CSF and the fat are both bright, so the fat is not assessed there.

Method notes and limits (also in ``docs/analysis/ligamentum.md``):

* The dural edge is the labeller's canal AP diameter (``Anchors.canal``) halved. Without canal geometry nothing is
  measured (warning). The sagittal fallback is not implemented: this module needs an axial slice through the disc.
* Thickness along a ray approximates the perpendicular thickness; a lateral ray can overestimate a band that runs
  oblique to it, so the sweep is kept small and the maximum over rays is reported as an upper estimate.
* The findings schema has a graded ``ligamentum_flavum.thickening`` field (none, mild, moderate, severe). This module
  emits only ``none`` or ``mild``: the grades above mild have no validated cut-offs. Epidural fat has no schema field,
  so it is reported as rows and flags only.
* The module measures and suggests. It does not diagnose. Every confidence is capped at ``ligamentum_max_confidence``
  because the thresholds are conventions and the validation is weak (``docs/analysis/ligamentum.md``).

The pixel-space core (:func:`measure_rays`) takes any 2-D image with its pixel spacing, so the same code runs on a
labelled study and on the Sudirman reference slices (``tools/eval_ligamentum.py``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..geometry import Stack, unit
from ..study import Study
from .base import Result, register
from .findings import Candidate, FieldRule, register_field

log = logging.getLogger(__name__)

#: Axial contrasts used, in preference order (CSF bright on T2; the dural edge is clearest there).
CONTRASTS: tuple[str, ...] = ("t2", "t1")
#: Figure: half width of the crop around the dural sac (mm) and the display scale (px per mm).
FIG_HALF_MM = 22.0
FIG_PX_PER_MM = 6.0

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RayResult:
    """One ray leaving the dural sac: its angle, the LF band (mm) if one was measured, and the contrast."""

    angle_deg: float
    thickness_mm: float | None  # LF band length along the ray; None when not measured
    cnr: float  # band depth in noise sigmas
    reason: str  # "ok" or why the band was not measured
    start_mm: float | None = None  # band edges along the ray, from the dural edge outward
    end_mm: float | None = None


@dataclass(frozen=True)
class AxialMeasure:
    """Everything measured on one axial slice at one level."""

    rays: list[RayResult]
    noise: float
    fat_mm: float | None  # midline dorsal epidural fat, None when not assessed (T2) or not closed
    fat_reason: str


# ------------------------------------------------------------------------------- pure signal functions
def robust_noise(img: NDArray[np.float64]) -> float:
    """Noise sigma of a 2-D image from its horizontal first differences (MAD, robust to edges)."""
    d = np.diff(np.asarray(img, dtype=np.float64), axis=1).ravel()
    if d.size == 0:
        return 0.0
    mad = float(np.median(np.abs(d - np.median(d))))
    return 1.4826 * mad / math.sqrt(2.0)


def _smooth(values: FloatArray, step_mm: float, cfg: Config) -> FloatArray:
    sigma = max(cfg.ligamentum_smooth_mm / step_mm, 0.0)
    return np.asarray(ndi.gaussian_filter1d(values, sigma), dtype=np.float64) if sigma > 0 else values


def _run_around(v: FloatArray, j: int, thr: float) -> tuple[int, int, bool]:
    """Inclusive run of samples <= thr around index ``j``; flag True when the run reaches either end of ``v``."""
    lo = j
    while lo - 1 >= 0 and v[lo - 1] <= thr:
        lo -= 1
    hi = j
    while hi + 1 < len(v) and v[hi + 1] <= thr:
        hi += 1
    open_ = lo == 0 or hi == len(v) - 1
    return lo, hi, open_


def band_from_profile(
    values: FloatArray, step_mm: float, noise: float, cfg: Config = DEFAULT
) -> tuple[float | None, float, float | None, float | None, str]:
    """LF band along one ray, sampled from the dural edge outward.

    Returns ``(thickness_mm, cnr, start_mm, end_mm, reason)``. The band is the run around the minimum of the smoothed
    profile where the signal stays at or below the midpoint between that minimum and the highest value behind it.
    """
    vals = np.asarray(values, dtype=np.float64)
    if vals.size < 3 or not np.all(np.isfinite(vals)):
        return None, 0.0, None, None, "no profile"
    v = _smooth(vals, step_mm, cfg)
    j = int(np.argmin(v))
    hi = float(np.max(v[j:]))
    depth = hi - float(v[j])
    cnr = depth / noise if noise > 0 else float("inf")
    if cnr < cfg.ligamentum_min_cnr:
        return None, cnr, None, None, "low contrast"
    thr = float(v[j]) + 0.5 * depth
    lo, hi_idx, open_ = _run_around(v, j, thr)
    if open_:
        return None, cnr, None, None, "band not closed within the search window"
    thick = (hi_idx - lo + 1) * step_mm
    return float(thick), cnr, lo * step_mm, (hi_idx + 1) * step_mm, "ok"


def fat_from_profile(
    values: FloatArray, step_mm: float, csf_level: float, fat_level: float, cfg: Config = DEFAULT
) -> tuple[float | None, str]:
    """Dorsal fat length from the dural edge outward: the run that starts at index 0 and stays above the midpoint."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size < 3 or not np.all(np.isfinite(vals)):
        return None, "no profile"
    if not fat_level > csf_level:
        return None, "fat is not brighter than the dural sac on this contrast"
    v = _smooth(vals, step_mm, cfg)
    thr = 0.5 * (csf_level + fat_level)
    if v[0] < thr:
        return 0.0, "no fat at the dural edge"
    n = 0
    while n < v.size and v[n] >= thr:
        n += 1
    if n >= v.size:
        return None, "fat run not closed within the search window"
    return float(n * step_mm), "ok"


def ray_angles(cfg: Config = DEFAULT) -> list[float]:
    """Lateral ray angles in degrees: 0 (midline) and +/- each configured angle, as a sorted list."""
    out = {0.0}
    for a in cfg.ligamentum_ray_deg:
        out.add(float(a))
        out.add(-float(a))
    return sorted(out)


# ------------------------------------------------------------------------------- pixel-space core
def _sample(img: FloatArray, rr: FloatArray, cc: FloatArray) -> FloatArray:
    return np.asarray(ndi.map_coordinates(img, [rr, cc], order=1, mode="nearest"), dtype=np.float64)


def measure_rays(
    img: NDArray[np.float64],
    row_mm: float,
    col_mm: float,
    centre_rc: tuple[float, float],
    rays_mm: list[tuple[float, tuple[float, float]]],
    radius_mm: float,
    cfg: Config = DEFAULT,
    *,
    fat_contrast: bool = False,
) -> AxialMeasure:
    """Measure the LF band along each ray and, optionally, the midline dorsal fat.

    ``img`` is one axial slice (rows, cols); ``row_mm``/``col_mm`` its pixel size; ``centre_rc`` the dural sac centre
    in pixels. ``rays_mm`` is a list of ``(angle_deg, (dr_mm, dc_mm))``: a unit direction in millimetres of (row, col),
    with angle 0 the midline posterior ray. ``radius_mm`` is the dural radius. ``fat_contrast`` says whether the
    contrast makes the fat brighter than the dural sac (T1); when False, the fat is not assessed.
    """
    arr = np.asarray(img, dtype=np.float64)
    noise = robust_noise(arr)
    step = cfg.ligamentum_step_mm
    r0 = radius_mm
    s = np.arange(0.0, r0 + cfg.ligamentum_search_mm + step / 2, step)
    i_edge = round(r0 / step)
    csf_n = max(round(cfg.ligamentum_csf_frac * r0 / step), 1)
    rc_r, rc_c = centre_rc
    rays: list[RayResult] = []
    fat_mm: float | None = None
    fat_reason = "not assessed (T2: the fat and the dural sac are both bright)"
    for angle, (dr, dc) in rays_mm:
        rr = rc_r + s * dr / row_mm
        cc = rc_c + s * dc / col_mm
        prof = _sample(arr, rr, cc)
        if angle == 0.0 and fat_contrast:
            csf_level = float(np.median(prof[:csf_n]))
            # the brightest tissue just behind the dural edge (fat is the brightest tissue on T1 in this window)
            fat_level = float(np.max(prof[i_edge : i_edge + round(cfg.ligamentum_search_mm / step)]))
            fat_mm, fat_reason = fat_from_profile(prof[i_edge:], step, csf_level, fat_level, cfg)
        thick, cnr, a, b, reason = band_from_profile(prof[i_edge:], step, noise, cfg)
        rays.append(
            RayResult(
                angle_deg=float(angle),
                thickness_mm=thick,
                cnr=float(cnr),
                reason=reason,
                start_mm=None if a is None else float(a),
                end_mm=None if b is None else float(b),
            )
        )
    return AxialMeasure(rays=rays, noise=noise, fat_mm=fat_mm, fat_reason=fat_reason)


def side_max(rays: list[RayResult], side_of: dict[float, str], side: str) -> tuple[float | None, int]:
    """Largest measured LF thickness over the rays of one side, and how many rays gave a value."""
    vals = [r.thickness_mm for r in rays if side_of.get(r.angle_deg) == side and r.thickness_mm is not None]
    if not vals:
        return None, 0
    return float(max(vals)), len(vals)


# ------------------------------------------------------------------------------- stack geometry
def _canal_at(anchors: Anchors, z: float) -> FloatArray | None:
    """Canal centreline point (LPS) at height ``z`` and the canal AP radius there; None if not measured."""
    canal = anchors.canal or {}
    pts = canal.get("points") or []
    if len(pts) < 2:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    order = np.argsort(arr[:, 2])
    zs = arr[order, 2]
    if not zs[0] <= z <= zs[-1]:
        return None
    centre = np.array(
        [float(np.interp(z, zs, arr[order, 0])), float(np.interp(z, zs, arr[order, 1])), float(z)]
    )
    return centre


def _ap_radius(anchors: Anchors, label: str) -> float | None:
    per = (anchors.canal or {}).get("per_level") or {}
    entry = per.get(label) or {}
    ap = entry.get("ap_mm")
    if ap is None or not math.isfinite(float(ap)) or float(ap) <= 0:
        return None
    return float(ap) / 2.0


def _axial_stack(study: Study, centre: FloatArray, cfg: Config) -> tuple[Stack, float] | None:
    """Best axial stack cutting the level: T2 before T1, then the closest to the disc centre along its normal."""
    best: tuple[int, float, Stack] | None = None
    for st in study.stacks.values():
        role = st.meta.get("role", {})
        if st.plane != "ax" or role.get("localizer"):
            continue
        contrast = role.get("contrast")
        if contrast not in CONTRASTS:
            continue
        _k, dist = st.nearest_slice(centre)
        limit = max(st.thickness, st.spacing or 0.0) / 2.0 + 1.0
        if dist > limit:
            continue
        key = (CONTRASTS.index(contrast), dist)
        if best is None or key < (best[0], best[1]):
            best = (key[0], key[1], st)
    if best is None:
        return None
    return best[2], best[1]


def _rays_for_stack(
    st: Stack, centre: FloatArray, post: FloatArray, cfg: Config
) -> list[tuple[float, str, tuple[float, float]]]:
    """Rays from the dural centre in the stack's plane: (angle, side, direction in mm of (row, col))."""
    n = np.asarray(st.normal, dtype=np.float64)
    u = post - (post @ n) * n
    if np.linalg.norm(u) < 1e-6:
        return []
    u = unit(u)
    w = unit(np.cross(n, u))
    r0, c0 = st.lps_to_pix(centre)[1:]
    out: list[tuple[float, str, tuple[float, float]]] = []
    for angle in ray_angles(cfg):
        th = math.radians(angle)
        d = math.cos(th) * u + math.sin(th) * w
        # patient left is +x in LPS
        side = "mid" if abs(angle) < 1e-9 else ("L" if float(d[0]) > 0 else "R")
        p1 = centre + 1.0 * d
        r1, c1 = st.lps_to_pix(p1)[1:]
        dr, dc = (r1 - r0) * st.row_mm, (c1 - c0) * st.col_mm
        norm = math.hypot(dr, dc)
        if norm < 1e-9:
            continue
        out.append((angle, side, (dr / norm, dc / norm)))
    return out


def _figure(
    img: NDArray[np.float64],
    st: Stack,
    centre_rc: tuple[float, float],
    rays: list[tuple[float, tuple[float, float]]],
    results: list[RayResult],
    cfg: Config,
) -> NDArray[np.uint8]:
    """Crop around the dural sac with the rays drawn: red where a band was measured, grey where not."""
    half_r = round(FIG_HALF_MM / st.row_mm)
    half_c = round(FIG_HALF_MM / st.col_mm)
    r_c, c_c = round(centre_rc[0]), round(centre_rc[1])
    lo, hi = np.percentile(img, [0.5, 99.5])
    scale = 255.0 / max(hi - lo, 1e-9)
    crop = np.clip(
        (img[max(r_c - half_r, 0) : r_c + half_r, max(c_c - half_c, 0) : c_c + half_c] - lo) * scale, 0, 255
    )
    pil = Image.fromarray(crop.astype(np.uint8)).convert("RGB")
    pil = pil.resize(
        (int(pil.width * FIG_PX_PER_MM * st.col_mm), int(pil.height * FIG_PX_PER_MM * st.row_mm))
    )
    draw = ImageDraw.Draw(pil)
    y0, x0 = max(r_c - half_r, 0), max(c_c - half_c, 0)
    for (_angle, (dr, dc)), res in zip(rays, results, strict=False):
        length = cfg.ligamentum_search_mm + 1.0
        p_end = (centre_rc[0] + length * dr / st.row_mm, centre_rc[1] + length * dc / st.col_mm)
        p0 = (
            (centre_rc[1] - x0) * st.col_mm * FIG_PX_PER_MM,
            (centre_rc[0] - y0) * st.row_mm * FIG_PX_PER_MM,
        )
        p1 = ((p_end[1] - x0) * st.col_mm * FIG_PX_PER_MM, (p_end[0] - y0) * st.row_mm * FIG_PX_PER_MM)
        colour = (255, 60, 60) if res.thickness_mm is not None else (150, 150, 150)
        draw.line([p0, p1], fill=colour, width=1)
    return np.asarray(pil, dtype=np.uint8)


# ------------------------------------------------------------------------------- analysis
@register("ligamentum")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Ligamentum flavum thickness per side and midline dorsal epidural fat, per disc level, from the axial slices."""
    cfg = config or DEFAULT
    res = Result(name="ligamentum")
    res.meta.update(
        {
            "method": "axial rays from the dural sac: LF band (half-maximum run) per side; midline dorsal fat (T1)",
            "contrasts": list(CONTRASTS),
            "ray_deg": list(cfg.ligamentum_ray_deg),
            "thick_mm": cfg.ligamentum_thick_mm,
            "epidural_fat_mm": cfg.ligamentum_epidural_fat_mm,
            "max_confidence": cfg.ligamentum_max_confidence,
            "status": "experimental",
        }
    )
    if not anchors.canal:
        res.warnings.append("ligamentum flavum not assessed: no canal geometry (learned labeller only)")
        return res
    if not any(st.plane == "ax" for st in study.stacks.values()):
        res.warnings.append(
            "ligamentum flavum not assessed: no axial stack (sagittal fallback not implemented)"
        )
        return res
    series_used: list[str] = []
    for label in anchors.disc_labels():
        disc = anchors.discs[label]
        centre = _canal_at(anchors, float(disc.center[2]))
        radius = _ap_radius(anchors, label)
        if centre is None or radius is None:
            res.rows.append(_row(label, "mid", None, "canal not measured at this level"))
            continue
        found = _axial_stack(study, centre, cfg)
        if found is None:
            res.rows.append(_row(label, "mid", None, "no axial T1/T2 slice through this level"))
            continue
        st, off = found
        k, _ = st.nearest_slice(centre)
        _, r_c, c_c = st.lps_to_pix(centre)
        rays_info = _rays_for_stack(st, centre, np.asarray(disc.ap_dir, dtype=np.float64), cfg)
        if not rays_info:
            res.rows.append(_row(label, "mid", None, "posterior direction not defined in this slice"))
            continue
        contrast = str(st.meta.get("role", {}).get("contrast"))
        img = np.asarray(st.vol[k], dtype=np.float64)
        rays = [(a, d) for a, _, d in rays_info]
        meas = measure_rays(
            img, st.row_mm, st.col_mm, (r_c, c_c), rays, radius, cfg, fat_contrast=contrast == "t1"
        )
        side_of = {a: s for a, s, _ in rays_info}
        series_used.append(st.key)
        for side in ("L", "R"):
            top, n_ok = side_max(meas.rays, side_of, side)
            res.rows.append(_lf_row(label, side, top, n_ok, meas, contrast, st.key, off, cfg))
        res.rows.append(_fat_row(label, meas, contrast, st.key, cfg))
        res.figures[f"lf_{label}"] = _figure(img, st, (r_c, c_c), rays, meas.rays, cfg)
    if series_used:
        res.meta["series_key"] = max(set(series_used), key=series_used.count)
    if not res.rows:
        res.warnings.append("ligamentum flavum not assessed: no level could be measured")
    return res


def _row(label: str, side: str, _value: float | None, reason: str) -> dict[str, Any]:
    return {
        "scope": "disc",
        "disc": label,
        "side": side,
        "metric": "not_assessed",
        "value_mm": None,
        "status": "not_assessed",
        "reason": reason,
    }


def _lf_row(
    label: str,
    side: str,
    top: float | None,
    n_ok: int,
    meas: AxialMeasure,
    contrast: str,
    series: str,
    off: float,
    cfg: Config,
) -> dict[str, Any]:
    measured = top is not None
    cap_note = "experimental: thresholds are conventions with weak literature support"
    reason = "ok" if measured else "no ray gave a closed band with enough contrast"
    flag = bool(measured and top is not None and top >= cfg.ligamentum_thick_mm)
    return {
        "scope": "disc",
        "disc": label,
        "side": side,
        "metric": "ligamentum_thickness",
        "value_mm": round(top, 2) if top is not None else None,
        "n_rays_measured": n_ok,
        "rays_total": len(meas.rays),
        "contrast": contrast,
        "series": series,
        "slice_distance_mm": round(off, 2),
        "status": "measured" if measured else "not_assessed",
        "reason": reason,
        "suggests_thickening": flag,
        "threshold_mm": cfg.ligamentum_thick_mm,
        "confidence": cfg.ligamentum_max_confidence if measured else 0.0,
        "notes": cap_note,
    }


def _fat_row(label: str, meas: AxialMeasure, contrast: str, series: str, cfg: Config) -> dict[str, Any]:
    fat = meas.fat_mm
    flag = bool(fat is not None and fat > cfg.ligamentum_epidural_fat_mm)
    return {
        "scope": "disc",
        "disc": label,
        "side": "mid",
        "metric": "dorsal_epidural_fat",
        "value_mm": None if fat is None else round(fat, 2),
        "contrast": contrast,
        "series": series,
        "status": "measured" if fat is not None else "not_assessed",
        "reason": meas.fat_reason,
        "possible_lipomatosis": flag,
        "threshold_mm": cfg.ligamentum_epidural_fat_mm,
        "confidence": cfg.ligamentum_max_confidence if fat is not None else 0.0,
        "notes": "dorsal fat only (Borre 2003 grades need ventral and dorsal fat); experimental",
    }


# ------------------------------------------------------------------------------- findings wiring
def _extract_thickening(res: Result, level: str) -> Candidate | None:
    """Schema grade for a level: ``mild`` at or above the threshold, ``none`` below it, None when not measured."""
    rows = [
        r
        for r in res.rows
        if r.get("scope") == "disc" and r.get("disc") == level and r.get("metric") == "ligamentum_thickness"
    ]
    if not rows or all(r.get("status") != "measured" for r in rows):
        return None
    vals = [float(r["value_mm"]) for r in rows if r.get("status") == "measured"]
    top = max(vals)
    value = "mild" if top >= DEFAULT.ligamentum_thick_mm else "none"
    sides = ", ".join(f"{r['side']} {r['value_mm']} mm" for r in rows if r.get("status") == "measured")
    return Candidate(
        value,
        DEFAULT.ligamentum_max_confidence,
        f"axial maximum over rays: {sides}; threshold {DEFAULT.ligamentum_thick_mm} mm is a convention (see docs)",
        slices=(),
    )


register_field(
    FieldRule(
        "ligamentum_flavum",
        "thickening",
        "ligamentum",
        _extract_thickening,
        figure=lambda level: f"lf_{level}",
        notes="experimental: axial measurement, convention threshold; never above low confidence",
    )
)
