"""Disc contour abnormality (bulge, protrusion, extrusion) on the sagittal plane, with its excursion and size.

Nomenclature: Fardon DF, Williams AL, Dohring EJ, et al. Lumbar disc nomenclature version 2.0 (NASS, ASSR, ASNR).
Spine J 2014;14:2525-45. Bulge: symmetric, broad-based extension, under 3 mm beyond the ring apophyses. Protrusion: the
base is wider than the displaced material extends beyond the disc space. Extrusion: the displaced material is wider than
its base, or migrates beyond the disc space. Sequestration needs a separate fragment and is not called here.

Method, per disc and per sagittal slice of the disc (disc frame, endplate traces and CSF reference from
:mod:`spinescope.analysis.discs`):

1. Reference line. The posterior corners of the body above (inferior endplate) and of the body below (superior endplate)
   are joined in the slice plane: the posterior margin of the vertebral column at this level. The corners come from the
   labeller's body records (``Vertebra.body``). Without them the disc is not assessed. Corners whose AP position is far
   from the disc's posterior margin are treated as unreliable.
2. Posterior edge. For each height row inside the disc, the search starts a few millimetres anterior to the line (inside
   the posterior annulus) and runs posteriorly. The edge is where the signal reaches a fixed fraction of the way from the
   annulus level to the thecal sac (CSF) level, interpolated on the smoothed profile.
3. Excursion. At each row, excursion = edge - line (mm, positive is posterior). ``e_max`` is the maximum over the rows;
   ``base`` is the height over which the excursion stays at or above the tolerance around that maximum (where the
   displaced material attaches to the disc).
4. Classes (``e_max`` in mm): below the tolerance, normal; from the tolerance up to 3 mm, bulge; above 3 mm, protrusion
   when the base is at least ``e_max``, extrusion otherwise. A sagittal view cannot establish the circumferential extent
   of a bulge, and migration beyond the disc space is not measured: an extrusion here is a lower bound.

Position. Only sagittal slices are used. A finding on the slice nearest the disc's midline is labelled ``central`` (a
sagittal-only candidate). Side and the subarticular, foraminal and extraforaminal zones need the axial plane, which this
module does not measure, so they are not called. Nerve roots (possible, anatomical): a central finding at a level can
involve the root that travels down the canal, which exits one level lower (L5 for L4-L5).

The module measures position and size. It does not diagnose. Confidence reflects measurement quality (traced endplates,
contrast against the noise, corner consistency, number of slices), not disease.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..anatomy.anchors import Anchors, split_disc
from ..config import DEFAULT, Config
from ..geometry import FloatArray
from ..study import Study
from .alignment import endplate_from_record
from .base import Result, register, sagittal_stacks
from .discs import _csf_reference, _disc_masks, _stack_noise, _traced, background, trace_disc
from .evaluate import register_predictor
from .findings import Candidate, FieldRule, register_field

log = logging.getLogger(__name__)

#: Fardon 2014 classes this module calls, in increasing severity (the findings schema enum minus ``sequestration``).
SEVERITY: tuple[str, ...] = ("normal", "bulge", "protrusion", "extrusion")
#: Sagittal contrasts in which the CSF is bright against the disc (needed for the posterior edge). T1 is not used.
EDGE_CONTRASTS: tuple[str, ...] = ("t2", "stir")
#: Rows closer than this to an endplate are not used (dark-line partial volume).
ROW_MARGIN_MM = 1.0
#: Contrast (in noise sigmas) at which the contrast term of a confidence reaches 1.
FULL_CONTRAST_SIGMA = 6.0
#: Figure scale and crop (AP from the disc centre, + posterior; height half-range), mm.
FIG_PX_PER_MM = 12.0
FIG_AP_MM = (-8.0, 22.0)
FIG_H_MM = 9.0


@dataclass(frozen=True)
class LineCorners:
    """The two posterior corners of the reference line, LPS mm."""

    up: FloatArray  # posterior-inferior corner of the body above the disc
    low: FloatArray  # posterior-superior corner of the body below the disc
    source: str  # "wall" (labeller wall corners) or "global" (global posterior extent of the endplate)


# ------------------------------------------------------------------------------------------ pure geometry
def line_s_at(
    h: NDArray[np.float64], s_up: float, h_up: float, s_low: float, h_low: float
) -> NDArray[np.float64]:
    """AP coordinate (mm) of the line through (s_up, h_up) and (s_low, h_low) at normal heights ``h``.

    NaN when the two corners sit at (almost) the same height, since no line is defined.
    """
    dh = h_low - h_up
    hh = np.asarray(h, dtype=np.float64)
    if not abs(dh) >= 1.0:
        return np.full_like(hh, np.nan)
    return s_up + (hh - h_up) / dh * (s_low - s_up)


def posterior_edge_index(profile: NDArray[np.float64], thr: float) -> int | None:
    """Index of the first sample where ``profile`` reaches ``thr`` and stays there for one more sample; None if never."""
    above = profile >= thr
    for i in range(len(profile) - 1):
        if above[i] and above[i + 1]:
            return i
    return None


def interp_crossing(s: NDArray[np.float64], profile: NDArray[np.float64], thr: float, i: int) -> float:
    """Position of the threshold crossing between samples ``i - 1`` and ``i`` (``s[0]`` when ``i`` is 0)."""
    if i <= 0:
        return float(s[0])
    p0, p1 = float(profile[i - 1]), float(profile[i])
    if p1 == p0:
        return float(s[i])
    frac = min(max((thr - p0) / (p1 - p0), 0.0), 1.0)
    return float(s[i - 1] + frac * (s[i] - s[i - 1]))


def contiguous_run(values: NDArray[np.float64], tol: float) -> tuple[int, int] | None:
    """Inclusive index range of the run of finite values >= ``tol`` that contains the maximum; None if there is none."""
    finite = np.isfinite(values)
    if not finite.any():
        return None
    top = int(np.argmax(np.where(finite, values, -np.inf)))
    if not values[top] >= tol:
        return None
    lo = top
    while lo - 1 >= 0 and finite[lo - 1] and values[lo - 1] >= tol:
        lo -= 1
    hi = top
    while hi + 1 < len(values) and finite[hi + 1] and values[hi + 1] >= tol:
        hi += 1
    return lo, hi


def classify_sagittal(e_max: float, base_mm: float, cfg: Config = DEFAULT) -> str:
    """Class from the mid-sagittal maximal posterior excursion (after the baseline offset) and its base (both mm)."""
    if not math.isfinite(e_max):
        raise ValueError("excursion is not finite")
    if e_max < cfg.herniation_tolerance_mm:
        return "normal"
    if e_max < cfg.herniation_beyond_apophysis_mm:
        return "bulge"
    return "protrusion" if base_mm >= e_max else "extrusion"


def corners_consistent(s_up: float, s_low: float, s_post: float, cfg: Config = DEFAULT) -> bool:
    """The two reference corners lie near the disc's posterior margin (AP), so the line describes this disc."""
    if not all(math.isfinite(v) for v in (s_up, s_low, s_post)):
        return False
    return abs(0.5 * (s_up + s_low) - s_post) <= cfg.herniation_corner_max_ap_mm


# ------------------------------------------------------------------------------------- labeller records
def posterior_corners(anchors: Anchors, upper: str, lower: str) -> LineCorners | None:
    """Posterior corners of the bodies bounding a disc, from the labeller's body records; None when missing."""
    vu = anchors.vertebrae.get(upper)
    vl = anchors.vertebrae.get(lower)
    if vu is None or vl is None or not vu.body or not vl.body:
        return None
    rec_u = vu.body.get("inferior_endplate")
    rec_l = vl.body.get("superior_endplate")
    ep_u = endplate_from_record(rec_u)
    ep_l = endplate_from_record(rec_l)
    if ep_u is None or ep_l is None:
        return None
    wall = bool(rec_u and "posterior_wall" in rec_u) and bool(rec_l and "posterior_wall" in rec_l)
    return LineCorners(
        up=np.asarray(ep_u.posterior, dtype=np.float64),
        low=np.asarray(ep_l.posterior, dtype=np.float64),
        source="wall" if wall else "global",
    )


# ------------------------------------------------------------------------------------ sagittal slice
@dataclass
class SliceMeasure:
    """Posterior excursion of one disc on one sagittal slice (mm)."""

    e_max: float  # raw maximal excursion (mm, before the baseline offset)
    e_adj: float  # maximal excursion minus the training baseline offset (mm): the value that is classified
    h_at_max: float
    base_mm: float  # height of the adjusted excursion at or above the tolerance, around its maximum (mm)
    valid_fraction: float
    thr: float
    csf: float
    ann: float
    contrast_sigma: float  # (csf - annulus) in noise sigmas
    geometry: str  # "traced" | "anchor"
    h_rows: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    edge_s: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    line_s: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))


def measure_slice(
    tr: Any,
    corners: LineCorners,
    noise: float,
    cfg: Config = DEFAULT,
    disc_height_mm: float | None = None,
) -> SliceMeasure | str:
    """Posterior excursion of one disc on one trace (``DiscTrace``). Returns the reason when it cannot be measured."""
    origin, u, n = tr.origin, tr.u, tr.n
    su, hu = float((corners.up - origin) @ u), float((corners.up - origin) @ n)
    sl, hl = float((corners.low - origin) @ u), float((corners.low - origin) @ n)
    if not corners_consistent(su, sl, float(tr.s_post), cfg):
        return "posterior corners not at the disc's posterior margin"
    if not abs(hl - hu) >= 1.0:
        return "posterior corners at one height"
    if _traced(tr, cfg):
        if np.count_nonzero(tr.ok) < 2:
            return "endplates not traced"
        mid_s = 0.5 * (tr.s_ant + tr.s_post)
        central = tr.ok & (np.abs(tr.s - mid_s) <= 0.25 * (tr.s_post - tr.s_ant))
        sel = central if central.any() else tr.ok
        h_top = float(np.median(tr.top[sel]))
        h_bot = float(np.median(tr.bot[sel]))
        geometry = "traced"
    else:
        if disc_height_mm is None or not math.isfinite(disc_height_mm) or disc_height_mm <= 0:
            return "endplates not traced and no anchored height"
        h_top, h_bot = 0.5 * disc_height_mm, -0.5 * disc_height_mm
        geometry = "anchor"
    rows = (tr.h >= h_bot + ROW_MARGIN_MM) & (tr.h <= h_top - ROW_MARGIN_MM)
    if not rows.any():
        return "disc too thin for the search"
    ann_v = tr.img[_disc_masks(tr, cfg)["annulus"]]
    ann_v = ann_v[np.isfinite(ann_v)]
    csf, _bright, _xyz = _csf_reference(tr, cfg)
    if ann_v.size == 0 or not math.isfinite(csf):
        return "no thecal sac boundary found"
    ann = float(np.median(ann_v))
    contrast_sigma = (csf - ann) / noise if noise > 0 else float("inf")
    if not contrast_sigma >= cfg.herniation_min_contrast_sigma:
        return "low contrast between disc and thecal sac"
    thr = ann + cfg.herniation_edge_fraction * (csf - ann)
    filled = np.where(np.isfinite(tr.img), tr.img, ann)
    sm = ndi.gaussian_filter1d(filled, sigma=1.0, axis=1)  # smoothing along AP only
    h_rows = tr.h[rows]
    line = line_s_at(h_rows, su, hu, sl, hl)
    edge = np.full(h_rows.shape, np.nan)
    hi_d = cfg.csf_search_mm[1]
    for r_i in range(h_rows.size):
        s_line = line[r_i]
        if not math.isfinite(s_line):
            continue
        row_idx = int(np.argmin(np.abs(tr.h - h_rows[r_i])))
        cols = np.nonzero((tr.s >= s_line - cfg.herniation_search_start_mm) & (tr.s <= s_line + hi_d))[0]
        if cols.size < 3:
            continue
        i = posterior_edge_index(sm[row_idx, cols], thr)
        if i is not None:
            edge[r_i] = interp_crossing(tr.s[cols], sm[row_idx, cols], thr, i)
    excess = edge - line
    valid = np.isfinite(excess)
    if valid.mean() < 0.5:
        return "posterior edge found on fewer than half of the rows"
    e_max = float(np.max(excess[valid]))
    adj = excess - cfg.herniation_baseline_mm
    h_at = float(h_rows[int(np.argmax(np.where(valid, excess, -np.inf)))])
    step = float(np.median(np.diff(tr.h))) if tr.h.size > 1 else cfg.disc_grid_mm
    run = contiguous_run(adj, cfg.herniation_tolerance_mm)
    base = 0.0 if run is None else float((run[1] - run[0] + 1) * abs(step))
    return SliceMeasure(
        e_max=e_max,
        e_adj=e_max - cfg.herniation_baseline_mm,
        h_at_max=h_at,
        base_mm=base,
        valid_fraction=float(valid.mean()),
        thr=thr,
        csf=csf,
        ann=ann,
        contrast_sigma=float(contrast_sigma),
        geometry=geometry,
        h_rows=h_rows,
        edge_s=edge,
        line_s=line,
    )


# ------------------------------------------------------------------------------------ rows
def _slice_row(
    lab: str, k: int, x_mm: float, centre_x: float, m: SliceMeasure, corners: LineCorners
) -> dict[str, Any]:
    return {
        "scope": "slice",
        "disc": lab,
        "slice_k": k,
        "slice_x_mm": round(x_mm, 2),
        "offset_mm": round(x_mm - centre_x, 2),
        "e_raw_mm": round(m.e_max, 2),
        "e_adj_mm": round(m.e_adj, 2),
        "h_at_max_mm": round(m.h_at_max, 2),
        "base_mm": round(m.base_mm, 2),
        "valid_fraction": round(m.valid_fraction, 2),
        "geometry": m.geometry,
        "line_source": corners.source,
        "contrast_sigma": round(m.contrast_sigma, 2),
        "threshold": round(m.thr, 2),
        "csf_level": round(m.csf, 2),
        "annulus_level": round(m.ann, 2),
    }


def _confidence(m: SliceMeasure, corners: LineCorners, n_slices: int) -> float:
    """Measurement quality in [0, 1]: traced endplates, wall corners, edge rows found, contrast, number of slices."""
    q_geom = 1.0 if m.geometry == "traced" else 0.6
    q_line = 1.0 if corners.source == "wall" else 0.75
    q_rows = min(1.0, m.valid_fraction / 0.8)
    q_contrast = min(1.0, m.contrast_sigma / FULL_CONTRAST_SIGMA)
    q_slices = 1.0 if n_slices >= 2 else 0.8
    return q_geom * q_line * q_rows * q_contrast * q_slices


def _location(contour: str | None, zone: str | None) -> str | None:
    """Schema location value of a herniation: ``none`` for a normal contour or a bulge (a bulge is not a herniation,
    Fardon et al. 2014); the zone for a protrusion or extrusion; None when no zone is known."""
    if contour is None:
        return None
    if contour in ("normal", "bulge"):
        return "none"
    return zone


def _possible_roots(label: str) -> str:
    """Anatomy with "possibly" wording: a central finding can involve the root travelling down the canal (lower body)."""
    _upper, lower = split_disc(label)
    return f"possibly {lower} (traversing root in the central canal; mid-sagittal finding)"


def _disc_row(
    lab: str,
    cfg: Config,
    *,
    sag: SliceMeasure | None = None,
    sag_cls: str | None = None,
    sag_offset_mm: float | None = None,
    n_slices: int = 0,
    e_any: float = float("nan"),
    confidence: float = 0.0,
    reason: str = "",
) -> dict[str, Any]:
    """Per-disc summary row. Predictor fields: ``herniation_suggestion``, ``bulging_suggestion`` and ``e_max_mm``."""
    row: dict[str, Any] = {"scope": "disc", "disc": lab, "n_slices": n_slices}
    if sag is None or sag_cls is None:
        row.update(
            contour=None,
            sag_class=None,
            e_raw_mm=float("nan"),
            e_max_mm=float("nan"),
            e_any_mm=float("nan"),
            base_mm=float("nan"),
            herniation_suggestion=None,
            bulging_suggestion=None,
            herniation_location=None,
            herniation_size_mm=None,
            confidence=0.0,
            zone=None,
            possible_roots="",
            reason=reason or "no measurable mid-sagittal slice",
        )
        return row
    e = float(sag.e_adj)
    zone: str | None = None
    midline_band = cfg.disc_half_width_mm * cfg.disc_mid_fraction
    if sag_cls != "normal" and sag_offset_mm is not None and abs(sag_offset_mm) <= midline_band:
        zone = "central"  # sagittal only: a mid-sagittal finding; lateral zones need the axial plane
    row.update(
        contour=sag_cls,
        sag_class=sag_cls,
        e_raw_mm=round(float(sag.e_max), 2),
        e_max_mm=round(e, 2),
        e_any_mm=round(float(e_any), 2) if math.isfinite(float(e_any)) else float("nan"),
        base_mm=round(sag.base_mm, 2),
        herniation_suggestion=1 if sag_cls in ("protrusion", "extrusion") else 0,
        bulging_suggestion=1 if e >= cfg.herniation_tolerance_mm else 0,
        herniation_location=_location(sag_cls, zone),
        herniation_size_mm=round(max(e, 0.0), 2),
        confidence=round(float(min(1.0, confidence)), 3),
        zone=zone,
        possible_roots=_possible_roots(lab) if zone == "central" else "",
        reason="",
    )
    return row


# ------------------------------------------------------------------------------------ figures
def _to_u8(img: NDArray[np.float64]) -> NDArray[np.uint8]:
    fin = img[np.isfinite(img)]
    if fin.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(fin, (1.0, 99.5))
    scale = (img - lo) / (hi - lo) if hi > lo else img * 0.0
    return (np.clip(np.nan_to_num(scale, nan=0.0), 0.0, 1.0) * 255).astype(np.uint8)


def _sag_figure(tr: Any, meas: SliceMeasure, corners: LineCorners, origin: FloatArray) -> NDArray[np.uint8]:
    """Disc crop in the disc frame (superior up, posterior right): the posterior line (yellow), the edge per row (red)."""
    s_sel = (tr.s >= FIG_AP_MM[0]) & (tr.s <= FIG_AP_MM[1])
    h_sel = np.abs(tr.h) <= FIG_H_MM
    crop = _to_u8(tr.img[np.ix_(h_sel, s_sel)])[::-1]
    w = round((FIG_AP_MM[1] - FIG_AP_MM[0]) * FIG_PX_PER_MM)
    hgt = round(2 * FIG_H_MM * FIG_PX_PER_MM)
    im = Image.fromarray(crop).convert("RGB").resize((w, hgt), Image.Resampling.BILINEAR)
    d = ImageDraw.Draw(im)

    def xy(s: float, hh: float) -> tuple[float, float]:
        return (s - FIG_AP_MM[0]) * FIG_PX_PER_MM, (FIG_H_MM - hh) * FIG_PX_PER_MM

    su, hu = float((corners.up - origin) @ tr.u), float((corners.up - origin) @ tr.n)
    sl, hl = float((corners.low - origin) @ tr.u), float((corners.low - origin) @ tr.n)
    d.line([xy(su, hu), xy(sl, hl)], fill=(255, 220, 0), width=2)
    for hh, se in zip(meas.h_rows, meas.edge_s, strict=True):
        if np.isfinite(se):
            x, y = xy(float(se), float(hh))
            d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 40, 40))
    return np.asarray(im, dtype=np.uint8)


# ------------------------------------------------------------------------------------ analysis
@register("herniation")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Posterior disc contour (bulge, protrusion, extrusion) and its excursion, measured on the sagittal slices."""
    cfg = config or DEFAULT
    res = Result(
        name="herniation",
        meta={
            "method": "posterior excursion beyond the body-corner line, sagittal slices (no axial contour in this module)",
            "tolerance_mm": cfg.herniation_tolerance_mm,
            "apophysis_mm": cfg.herniation_beyond_apophysis_mm,
            "edge_fraction": cfg.herniation_edge_fraction,
            "baseline_mm": cfg.herniation_baseline_mm,
        },
    )
    stacks = sagittal_stacks(study)
    edge_key = next((c for c in EDGE_CONTRASTS if c in stacks), None)
    if edge_key is None:
        res.warnings.append("no sagittal T2 or STIR: posterior disc edge not measured (not assessed)")
        return res
    ref = stacks[edge_key]
    if ref.plane != "sag":
        res.warnings.append("reference stack is not sagittal: herniation not assessed")
        return res
    res.meta.update(contrast=edge_key, series_key=ref.key)
    if "t2" not in stacks:
        res.warnings.append(f"no sagittal T2: posterior edge measured on {edge_key.upper()}")
    if any(s.plane == "ax" for s in study.stacks.values()):
        res.warnings.append("axial contour, side and lateral zones are not measured by this analysis")
    labels = anchors.disc_labels()
    if not labels:
        res.warnings.append("no discs in anchors")
        return res
    floor = background(ref)
    noise = _stack_noise(ref)
    for lab in labels:
        disc = anchors.discs[lab]
        upper, lower = split_disc(lab)
        corners = posterior_corners(anchors, upper, lower)
        if corners is None:
            res.rows.append(_disc_row(lab, cfg, reason="no posterior corners in the labeller's body records"))
            continue
        if corners.source == "global":
            res.warnings.append(f"{lab}: body corners are global-extent, not wall corners: lower confidence")
        centre_x = float(np.asarray(disc.center) @ ref.normal)
        ks = [k for k in range(ref.n) if abs(float(ref.positions[k]) - centre_x) <= cfg.disc_half_width_mm]
        measured: list[tuple[float, SliceMeasure]] = []
        reasons: list[str] = []
        first: tuple[Any, SliceMeasure] | None = None
        for k in sorted(ks, key=lambda kk: abs(float(ref.positions[kk]) - centre_x)):
            tr = trace_disc(ref, k, lab, disc, cfg, floor=floor)
            m = measure_slice(tr, corners, noise, cfg, disc_height_mm=disc.height_mm)
            if isinstance(m, str):
                reasons.append(m)
                continue
            x_mm = float(ref.positions[k])
            measured.append((x_mm - centre_x, m))
            res.rows.append(_slice_row(lab, k, x_mm, centre_x, m, corners))
            if first is None:
                first = (tr, m)
        if not measured:
            res.rows.append(
                _disc_row(lab, cfg, reason="; ".join(sorted(set(reasons))) or "no slice measured")
            )
            continue
        off0, prim = measured[0]
        sag_cls = classify_sagittal(prim.e_adj, prim.base_mm, cfg)
        row = _disc_row(
            lab,
            cfg,
            sag=prim,
            sag_cls=sag_cls,
            sag_offset_mm=off0,
            n_slices=len(measured),
            e_any=max(m.e_adj for _o, m in measured),
            confidence=_confidence(prim, corners, len(measured)),
        )
        if row["contour"] in ("protrusion", "extrusion"):
            row["confidence"] = round(min(float(row["confidence"]), cfg.herniation_class_max_confidence), 3)
            row["reason"] = "; ".join(
                x
                for x in (
                    str(row.get("reason") or ""),
                    "focal (protrusion/extrusion) calls are not validated on public data: check the axial images",
                )
                if x
            )
        res.rows.append(row)
        if first is not None:
            res.figures[f"sag_{lab}"] = _sag_figure(
                first[0], first[1], corners, np.asarray(disc.center, dtype=float)
            )
        if row["contour"] not in (None, "normal"):
            res.flags.append(
                {
                    "kind": "disc_contour",
                    "level": lab,
                    "class": row["contour"],
                    "e_max_mm": row["e_max_mm"],
                    "confidence": row["confidence"],
                }
            )
    return res


# ------------------------------------------------------------------------------------ findings fields
def _row(res: Result, level: str) -> dict[str, Any] | None:
    return next((r for r in res.rows if r.get("scope") == "disc" and r.get("disc") == level), None)


def _extract_contour(res: Result, level: str) -> Candidate | None:
    r = _row(res, level)
    if r is None or r.get("contour") is None:
        return None
    return Candidate(value=r["contour"], confidence=float(r["confidence"]), notes=str(r.get("reason") or ""))


def _extract_location(res: Result, level: str) -> Candidate | None:
    r = _row(res, level)
    if r is None or r.get("herniation_location") is None:
        return None
    return Candidate(value=r["herniation_location"], confidence=float(r["confidence"]), notes="sagittal only")


def _extract_size(res: Result, level: str) -> Candidate | None:
    r = _row(res, level)
    if r is None or r.get("herniation_size_mm") is None:
        return None
    return Candidate(
        value=float(r["herniation_size_mm"]), confidence=float(r["confidence"]), notes="sagittal excursion"
    )


def _figure_name(level: str) -> str:
    return f"sag_{level}"


register_field(FieldRule("disc", "contour", "herniation", _extract_contour, figure=_figure_name))
register_field(FieldRule("disc", "herniation_location", "herniation", _extract_location, figure=_figure_name))
register_field(FieldRule("disc", "herniation_size_mm", "herniation", _extract_size, figure=_figure_name))


# ------------------------------------------------------------------------------------ harness predictors
def _flag_prediction(res: Result, anchors: Anchors, key: str) -> dict[str, tuple[Any, float | None]]:
    out: dict[str, tuple[Any, float | None]] = {}
    for lab in anchors.discs:
        r = _row(res, lab)
        val = None if r is None else r.get(key)
        e = None if r is None else r.get("e_max_mm")
        score = float(e) if e is not None and math.isfinite(float(e)) else None
        out[lab] = (None if val is None else int(val), score)
    return out


def herniation_prediction(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
    """SPIDER ``disc_herniation``: 1 when the mid-sagittal class is protrusion or extrusion (excursion beyond 3 mm).
    Score = excursion (AUC)."""
    return _flag_prediction(res, anchors, "herniation_suggestion")


def bulging_prediction(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
    """SPIDER ``disc_bulging``: 1 when the mid-sagittal excursion reaches the 1 mm tolerance. Score = excursion (AUC)."""
    return _flag_prediction(res, anchors, "bulging_suggestion")


register_predictor("herniation", "disc_herniation", herniation_prediction)
register_predictor("herniation", "disc_bulging", bulging_prediction)
