"""Central canal: sagittal AP diameter and axial dural sac area per level, with stenosis thresholds (dx06).

Measurements, per level (disc levels and mid-body levels of the labelled spine):

1. **Sagittal canal AP** (mm). The labeller's canal mask gives it: ``anchors.canal["per_level"]`` holds the AP chord
   of the canal on the sagittal slice nearest the canal centreline at that height (see :mod:`spinescope.anatomy.
   canal_ml`). It is a dural sac width, not a bony canal diameter. No CSF-signal fallback is implemented: without a
   learned canal (classical labeller) the AP is not assessed and a warning says so.
2. **Axial dural sac area** (mm^2) at each disc level. The axial slab angled to the disc (:func:`spinescope.anatomy.
   correspondence.level_map`) is sampled on the slice nearest the disc plane. The canal centre is the centreline point
   at the disc height, projected onto that slice. The sac is the CSF-bright (T2, STIR, PD) or CSF-dark (T1) region
   connected to that centre, inside a square window, with an Otsu threshold on the window. The area is corrected for
   slab obliquity: ``area = observed * cos(angle between slab normal and disc normal)``. Above
   ``canal_axial_obliquity_max_deg`` no area is reported.
3. **Relative value**: the area divided by the study's reference area (``canal_reference_quantile`` of the measured
   disc levels, a proxy for the non-stenotic sac).
4. **Thresholds** (literature, :mod:`spinescope.config`): area below 100 mm^2 relative and below 75 mm^2 absolute
   (Hamanishi et al. 1994); AP below 10 mm absolute and below 13 mm relative (Steurer et al. 2011). The stenosis
   suggestion per level is the most severe criterion met: ``severe`` when both the area and the AP are absolute,
   ``moderate`` for one absolute criterion, ``mild`` for relative criteria only, else ``none``. The mapping is a
   convention of this module, not a validated severity grading.

Observed contributors (disc height from the anchors) are reported as geometry only. Ligamentum flavum thickness is
owned by another analysis (dx09); facet and foraminal stenosis are not measured here.

Wording: these are measurements and suggestions. A grade is a prompt for a reader, never a diagnosis.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.filters import threshold_otsu

from ..anatomy.anchors import Anchors, split_disc
from ..anatomy.correspondence import CorrespondenceConfig, LevelMap, level_map
from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .base import Result, register
from .findings import Candidate, FieldRule, register_field

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Stenosis suggestion words, in increasing severity (the findings schema enum).
GRADES: tuple[str, ...] = ("none", "mild", "moderate", "severe")
#: Contrasts in which CSF is dark (T1). Other contrasts (T2, STIR, PD) are read with bright CSF.
DARK_CSF_CONTRASTS: frozenset[str] = frozenset({"t1"})
#: Figure scales: axial crops are drawn at this many pixels per mm, and span +-this many mm about the canal centre.
FIG_PX_PER_MM = 5
FIG_HALF_MM = 20.0


@dataclass
class SacOutline:
    """The dural sac found in one axial slice. ``window`` and ``mask`` share the window grid (r0, c0 origin)."""

    area_mm2: float  # observed area, before the obliquity correction
    pixels: int
    threshold: float
    truncated: bool  # the sac touches the search window: the area is a lower bound
    bright_csf: bool
    origin: tuple[int, int]  # (row, column) of the window's first pixel in the slice
    window: NDArray[np.float64]  # slice intensities in the window
    mask: BoolArray  # sac pixels, same shape as ``window``


def sac_outline(
    plane: NDArray[np.floating],
    centre_rc: tuple[float, float],
    row_mm: float,
    col_mm: float,
    bright_csf: bool,
    cfg: Config = DEFAULT,
) -> SacOutline | None:
    """Dural sac on one axial slice, seeded at ``centre_rc`` (pixel row, column).

    Otsu threshold on a square window of ``canal_axial_halfwindow_mm``; the connected component of the threshold
    class nearest the seed (within ``canal_axial_centre_tol_mm``) is the sac, holes filled. None when the seed is off
    the slice, the window holds too little data, or no CSF-like component lies at the seed.
    """
    img = np.asarray(plane, dtype=np.float64)
    h, w = img.shape
    rc_r, rc_c = float(centre_rc[0]), float(centre_rc[1])
    if not (0.0 <= rc_r <= h - 1 and 0.0 <= rc_c <= w - 1):
        return None
    hr = max(2, round(cfg.canal_axial_halfwindow_mm / row_mm))
    hc = max(2, round(cfg.canal_axial_halfwindow_mm / col_mm))
    r0, r1 = max(0, round(rc_r) - hr), min(h, round(rc_r) + hr + 1)
    c0, c1 = max(0, round(rc_c) - hc), min(w, round(rc_c) + hc + 1)
    patch = img[r0:r1, c0:c1]
    finite = np.isfinite(patch)
    if finite.sum() < 20:
        return None
    lo, hi = np.percentile(patch[finite], [0.5, 99.5])
    if hi <= lo:
        return None
    filled = np.clip(np.where(finite, patch, lo), lo, hi)
    thr = float(threshold_otsu(filled))
    mask = (filled > thr) if bright_csf else (filled < thr)
    mask &= finite
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    rr, cc = np.nonzero(mask)
    dist = np.hypot((rr + r0 - rc_r) * row_mm, (cc + c0 - rc_c) * col_mm)
    j = int(np.argmin(dist))
    if float(dist[j]) > cfg.canal_axial_centre_tol_mm:
        return None
    comp = ndimage.binary_fill_holes(labels == labels[rr[j], cc[j]])
    truncated = bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any())
    pixels = int(comp.sum())
    return SacOutline(
        area_mm2=float(pixels * row_mm * col_mm),
        pixels=pixels,
        threshold=thr,
        truncated=truncated,
        bright_csf=bright_csf,
        origin=(r0, c0),
        window=filled,
        mask=comp,
    )


def obliquity_deg(slab_normal: Any, disc_normal: Any) -> float:
    """Angle (deg) between the axial slab normal and the disc normal; sign of either is irrelevant."""
    a = np.asarray(slab_normal, dtype=np.float64)
    b = np.asarray(disc_normal, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def obliquity_factor(angle_deg: float) -> float:
    """Area correction: a plane at ``angle_deg`` to a tube's axis cuts it in ``A / cos(angle)``; this returns cos."""
    return float(math.cos(math.radians(angle_deg)))


def ap_grade(ap_mm: float | None, cfg: Config = DEFAULT) -> str | None:
    """``absolute`` below ``canal_ap_absolute_mm``, ``relative`` below ``canal_ap_relative_mm``, else ``none``."""
    if ap_mm is None:
        return None
    if ap_mm < cfg.canal_ap_absolute_mm:
        return "absolute"
    if ap_mm < cfg.canal_ap_relative_mm:
        return "relative"
    return "none"


def csa_grade(area_mm2: float | None, cfg: Config = DEFAULT) -> str | None:
    """``absolute`` below ``canal_csa_absolute_mm2``, ``relative`` below ``canal_csa_relative_mm2``, else ``none``."""
    if area_mm2 is None:
        return None
    if area_mm2 < cfg.canal_csa_absolute_mm2:
        return "absolute"
    if area_mm2 < cfg.canal_csa_relative_mm2:
        return "relative"
    return "none"


def stenosis_grade(csa: str | None, ap: str | None) -> str | None:
    """Suggested grade from the two criteria (``absolute`` / ``relative`` / ``none`` / None = not assessed).

    ``severe``: both absolute. ``moderate``: one absolute. ``mild``: relative only. ``none``: no criterion met.
    None when neither criterion was assessed.
    """
    if csa is None and ap is None:
        return None
    crit = (csa, ap)
    if crit == ("absolute", "absolute"):
        return "severe"
    if "absolute" in crit:
        return "moderate"
    if "relative" in crit:
        return "mild"
    return "none"


def reference_area(areas: list[float], cfg: Config = DEFAULT) -> float | None:
    """The study's reference sac area: the ``canal_reference_quantile`` of the measured disc areas, or None (too few)."""
    if len(areas) < cfg.canal_reference_min_levels:
        return None
    return float(np.quantile(np.asarray(areas, dtype=np.float64), cfg.canal_reference_quantile))


def _contrast(st: Stack) -> str | None:
    value = st.meta.get("role", {}).get("contrast")
    return None if value is None else str(value)


def _project(st: Stack, k: int, p: Any) -> FloatArray:
    v = np.asarray(p, dtype=np.float64)
    return v - ((v - st.ipp[k]) @ st.normal) * st.normal


def _axial_confidence(obliquity: float, truncated: bool, cfg: Config) -> float:
    conf = 0.8
    if obliquity > cfg.canal_axial_obliquity_lower_deg:
        conf = 0.6
    if truncated:
        conf = min(conf, 0.4)
    return conf


def _level_confidence(ap_ok: bool, canal_conf: float, csa_conf: float | None) -> float | None:
    """Both criteria: the lower of the two; area only: its confidence; AP only: half the canal confidence."""
    if csa_conf is not None and ap_ok:
        return round(min(csa_conf, canal_conf), 3)
    if csa_conf is not None:
        return round(csa_conf, 3)
    if ap_ok:
        return round(0.5 * canal_conf, 3)
    return None


@dataclass
class _AxialDisc:
    """Axial measurement of one disc level (internal; the outline is kept for the figure)."""

    slab: str
    slice_index: int
    obliquity_deg: float
    outline: SacOutline
    centre_rc: tuple[float, float]
    area_mm2: float  # corrected for obliquity
    confidence: float
    note: str


def _axial_disc(
    study: Study, lm: LevelMap, label: str, disc_normal: Any, cfg: Config
) -> tuple[_AxialDisc | None, str]:
    """Axial sac of disc ``label``; (None, reason) when it cannot be measured."""
    entry = lm.discs.get(label)
    if entry is None or entry.slab is None:
        return None, "no axial slab through this disc"
    st = study.stacks[entry.slab]
    hit = entry.stacks.get(entry.slab)
    if hit is None:
        return None, "no slice of the axial slab for this disc"
    contrast = _contrast(st)
    if contrast is None:
        return None, "axial contrast unknown: CSF polarity not set"
    canal_pt = entry.targets.get("canal", {}).get("point")
    if canal_pt is None:
        return None, "no canal centre at this level"
    k = int(hit.slice)
    obl = obliquity_deg(st.normal, disc_normal)
    if obl > cfg.canal_axial_obliquity_max_deg:
        return None, f"slab {obl:.0f} deg from the disc normal: too oblique"
    p = _project(st, k, canal_pt)
    _k, r, c = st.lps_to_pix(p)
    if not (0 <= r <= st.vol.shape[1] - 1 and 0 <= c <= st.vol.shape[2] - 1):
        return None, "canal centre outside the axial field of view"
    bright = contrast not in DARK_CSF_CONTRASTS
    out = sac_outline(st.vol[k], (r, c), st.row_mm, st.col_mm, bright, cfg)
    if out is None:
        return None, "no CSF-like region at the canal centre"
    area = out.area_mm2 * obliquity_factor(obl)
    if not (cfg.canal_csa_min_mm2 <= area <= cfg.canal_csa_max_mm2):
        return None, f"area {area:.0f} mm2 outside the plausible range: not measured"
    conf = _axial_confidence(obl, out.truncated, cfg)
    if not cfg.canal_axial_validated:
        conf = min(conf, cfg.canal_axial_unvalidated_cap)
    notes = [f"axial {entry.slab} slice {k}, obliquity {obl:.1f} deg (area x cos)"]
    if out.truncated:
        notes.append("sac touches the search window: lower bound")
    return (
        _AxialDisc(
            slab=entry.slab,
            slice_index=k,
            obliquity_deg=obl,
            outline=out,
            centre_rc=(float(r), float(c)),
            area_mm2=float(area),
            confidence=conf,
            note="; ".join(notes),
        ),
        "",
    )


#: Levels graded for stenosis: the lumbar canal (discs L1-L2 to L5-S1, bodies L1 to L5). The thoracic canal is
#: narrower and carries the cord, so the lumbar AP and area cut-offs do not apply there.
LUMBAR_BODIES: frozenset[str] = frozenset({"L1", "L2", "L3", "L4", "L5"})
LUMBAR_DISC_LOWER: frozenset[str] = frozenset({"L2", "L3", "L4", "L5", "S1"})


def in_grading_range(level: str) -> bool:
    """True for a level inside the lumbar grading range (see :data:`LUMBAR_BODIES`, :data:`LUMBAR_DISC_LOWER`)."""
    if "-" in level:
        try:
            return split_disc(level)[1] in LUMBAR_DISC_LOWER
        except ValueError:
            return False
    return level in LUMBAR_BODIES


def _sagittal_rows(anchors: Anchors) -> list[dict[str, Any]]:
    canal = anchors.canal or {}
    rows: list[dict[str, Any]] = []
    for lab, entry in (canal.get("per_level") or {}).items():
        ap = entry.get("ap_mm")
        rows.append(
            {
                "scope": "disc" if entry.get("kind") == "disc" else "mid_body",
                "level": lab,
                "z_mm": round(float(entry.get("z_mm", 0.0)), 2),
                "sagittal_slice": entry.get("slice"),
                "ap_mm": None if ap is None else round(float(ap), 2),
                "ap_grade": ap_grade(None if ap is None else float(ap)),
            }
        )
    return rows


@register("canal")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Sagittal AP and axial dural sac area per level, with the stenosis suggestion (see module docstring)."""
    cfg = config or DEFAULT
    res = Result(name="canal")
    res.meta = {
        "method": "AP from the labeller canal mask; area from the axial slab through each disc (Otsu sac in a window)",
        "thresholds": {
            "csa_absolute_mm2": cfg.canal_csa_absolute_mm2,
            "csa_relative_mm2": cfg.canal_csa_relative_mm2,
            "ap_absolute_mm": cfg.canal_ap_absolute_mm,
            "axial_contrasts": list(cfg.canal_axial_contrasts),
            "ap_relative_mm": cfg.canal_ap_relative_mm,
        },
        "reference": f"quantile {cfg.canal_reference_quantile} of disc areas",
    }
    rows = _sagittal_rows(anchors)
    canal = anchors.canal or {}
    canal_conf = float(canal.get("confidence", 0.0) or 0.0)
    if not canal:
        res.warnings.append("no canal geometry (learned labeller not used): sagittal AP not assessed")
    if not study.stacks or not any(s.plane == "ax" for s in study.stacks.values()):
        res.warnings.append(
            "no axial stack: dural sac area not assessed (sagittal AP only, lower confidence)"
        )

    axial: dict[str, _AxialDisc] = {}
    lm: LevelMap | None = None
    if anchors.discs and any(s.plane == "ax" for s in study.stacks.values()):
        try:
            lm = level_map(
                study,
                anchors,
                CorrespondenceConfig(slab_contrasts=cfg.canal_axial_contrasts),
                measure_geometry=False,
            )
        except Exception as exc:  # the axial part is optional; the sagittal AP still stands
            log.warning("canal: level correspondence failed: %s", exc)
            res.warnings.append(f"level correspondence failed: {type(exc).__name__}: {exc}")
    reasons: dict[str, str] = {}
    if lm is not None:
        for lab, disc in anchors.discs.items():
            got, why = _axial_disc(study, lm, lab, disc.normal, cfg)
            if got is not None:
                axial[lab] = got
            else:
                reasons[lab] = why

    areas = [a.area_mm2 for a in axial.values()]
    ref = reference_area(areas, cfg)
    if areas and ref is None:
        res.warnings.append(
            f"fewer than {cfg.canal_reference_min_levels} measured disc levels: no relative area value"
        )

    out_rows: list[dict[str, Any]] = []
    seen_disc = set()
    for row in rows:
        lab = row["level"]
        is_disc = row["scope"] == "disc"
        ax = axial.get(lab) if is_disc else None
        if is_disc:
            seen_disc.add(lab)
            disc_h = anchors.discs[lab].height_mm if lab in anchors.discs else None
            row["disc_height_mm"] = round(float(disc_h), 2) if disc_h else None
            chosen = lm.discs.get(lab) if lm is not None else None
            row["axial_slab_chosen"] = chosen.slab if chosen is not None else None
            row["axial_slice_chosen"] = (
                chosen.stacks[chosen.slab].slice if chosen is not None and chosen.slab else None
            )
            row["axial_slab"] = ax.slab if ax else None
            row["axial_slice"] = ax.slice_index if ax else None
            row["obliquity_deg"] = round(ax.obliquity_deg, 1) if ax else None
            row["csa_observed_mm2"] = round(ax.outline.area_mm2, 1) if ax else None
            row["csa_mm2"] = round(ax.area_mm2, 1) if ax else None
            row["csa_truncated"] = bool(ax.outline.truncated) if ax else None
            row["csa_ratio"] = round(ax.area_mm2 / ref, 3) if ax and ref else None
            row["csa_grade"] = csa_grade(ax.area_mm2 if ax else None, cfg)
            if ax is None and lab in reasons:
                row["csa_note"] = reasons[lab]
        else:
            row["csa_mm2"] = None
            row["csa_grade"] = None
        ap_ok = row["ap_mm"] is not None
        csa_conf = ax.confidence if ax else None
        conf = _level_confidence(ap_ok, canal_conf, csa_conf)
        in_range = in_grading_range(lab)
        row["grading_range"] = in_range
        row["grade"] = stenosis_grade(row["csa_grade"], row["ap_grade"]) if in_range else None
        row["confidence"] = conf if row["grade"] is not None else None
        notes = []
        if not in_range:
            notes.append("outside the lumbar grading range: measured, not graded")
        if ax is not None:
            notes.append(ax.note)
        if not ap_ok and not canal:
            notes.append("AP not assessed")
        elif not ap_ok:
            notes.append("AP not on the sagittal slice at this level")
        if is_disc and ax is None:
            notes.append("area not assessed: " + reasons.get(lab, "no axial"))
        if ax is None and ap_ok and is_disc:
            notes.append("sagittal only: lower confidence")
        row["note"] = "; ".join(notes)
        out_rows.append(row)
    for lab in sorted(set(anchors.discs) - seen_disc):  # discs the canal did not cover at all
        out_rows.append(
            {
                "scope": "disc",
                "level": lab,
                "ap_mm": None,
                "ap_grade": None,
                "csa_mm2": None,
                "csa_grade": None,
                "grade": None,
                "confidence": None,
                "note": "no canal measurement at this level",
            }
        )
    res.rows = out_rows
    disc_rows = [r for r in out_rows if r["scope"] == "disc" and r.get("grade")]
    for r in disc_rows:
        if r["grade"] in ("moderate", "severe", "mild"):
            res.flags.append(
                {
                    "level": r["level"],
                    "grade": r["grade"],
                    "confidence": r["confidence"],
                    "note": r["note"],
                }
            )
    if axial:
        res.figures["axial"] = _axial_sheet(axial, anchors, ref, cfg)
    if any(r.get("ap_mm") is not None for r in out_rows):
        res.figures["profile"] = _profile_figure(out_rows, cfg)
    res.meta["reference_area_mm2"] = None if ref is None else round(ref, 1)
    # Report evidence cites the series the labels came from (the sagittal stack); the profile is drawn from it.
    res.meta["series_key"] = str(anchors.source_stack or "")
    return res


# ----------------------------------------------------------------------- figures
def _scale_gray(a: NDArray[np.float64]) -> NDArray[np.uint8]:
    lo, hi = np.percentile(a, [1.0, 99.0])
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.asarray(np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0, dtype=np.uint8)


def _axial_sheet(
    axial: dict[str, _AxialDisc], anchors: Anchors, ref: float | None, cfg: Config
) -> NDArray[np.uint8]:
    """One crop per measured disc (levels superior to inferior): the sac outline in red and its area."""
    labels = sorted(axial, key=lambda lab: -anchors.discs[lab].center[2])
    size = int(2 * FIG_HALF_MM * FIG_PX_PER_MM)
    tile_w = size + 4
    tiles: list[Image.Image] = []
    for lab in labels:
        ax = axial[lab]
        out = ax.outline
        img = _scale_gray(out.window)
        pil = (
            Image.fromarray(img)
            .convert("RGB")
            .resize(
                (out.window.shape[1] * FIG_PX_PER_MM, out.window.shape[0] * FIG_PX_PER_MM),
                Image.Resampling.NEAREST,
            )
        )
        arr = np.asarray(pil).copy()
        m = np.kron(
            out.mask.astype(np.uint8), np.ones((FIG_PX_PER_MM, FIG_PX_PER_MM), dtype=np.uint8)
        ).astype(bool)
        edge = m & ~ndimage.binary_erosion(m, iterations=1)
        arr[edge] = (255, 40, 40)
        canvas = Image.new("RGB", (size, size + 34), (20, 20, 20))
        canvas.paste(Image.fromarray(arr), (0, 34))
        d = ImageDraw.Draw(canvas)
        rel = "" if ref is None else f" ratio {ax.area_mm2 / ref:.2f}"
        flag = " (lower bound)" if out.truncated else ""
        d.text((4, 2), f"{lab}  {ax.area_mm2:.0f} mm2{flag}", fill=(255, 255, 255))
        d.text((4, 16), f"obl {ax.obliquity_deg:.0f} deg{rel}", fill=(200, 200, 200))
        tiles.append(canvas)
    sheet = Image.new("RGB", (tile_w * len(tiles), size + 34), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * tile_w, 0))
    return np.asarray(sheet, dtype=np.uint8)


def _profile_figure(rows: list[dict[str, Any]], cfg: Config) -> NDArray[np.uint8]:
    """Sagittal AP by level (levels superior to inferior) with the absolute and relative thresholds."""
    pts = [r for r in rows if r.get("ap_mm") is not None]
    pts.sort(key=lambda r: -float(r["z_mm"]))
    W, H, pad_l, pad_b = 110 + 46 * max(len(pts), 1), 260, 46, 36
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    top, bottom = 14, H - pad_b
    ymax = max(20.0, max(float(r["ap_mm"]) for r in pts) + 2.0)

    def y_of(v: float) -> int:
        return int(bottom - (v / ymax) * (bottom - top))

    for thr, colour in (
        (cfg.canal_ap_absolute_mm, (200, 30, 30)),
        (cfg.canal_ap_relative_mm, (230, 150, 30)),
    ):
        yy = y_of(thr)
        d.line([(pad_l, yy), (W - 60, yy)], fill=colour, width=1)
        d.text((W - 58, yy - 12), f"{thr:g} mm", fill=colour)
    d.line([(pad_l, top), (pad_l, bottom)], fill=(0, 0, 0))
    d.line([(pad_l, bottom), (W - 60, bottom)], fill=(0, 0, 0))
    for i, r in enumerate(pts):
        x = pad_l + 14 + i * 46
        y = y_of(float(r["ap_mm"]))
        colour = (60, 110, 200) if r["scope"] == "disc" else (150, 150, 150)
        d.rectangle([x - 12, y, x + 12, bottom], fill=colour)
        d.text((x - 12, y - 12), f"{float(r['ap_mm']):.1f}", fill=(0, 0, 0))
        d.text((x - 14, bottom + 6), str(r["level"]), fill=(0, 0, 0))
    d.text((4, 2), "sagittal canal AP (mm); blue: disc level, grey: mid-body", fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)


# ----------------------------------------------------------------------- findings wiring
def _disc_row(res: Result, level: str) -> dict[str, Any] | None:
    for r in res.rows:
        if r.get("scope") == "disc" and r.get("level") == level:
            return r
    return None


def _extract_stenosis(res: Result, level: str) -> Candidate | None:
    r = _disc_row(res, level)
    if r is None or not r.get("grade"):
        return None
    parts = []
    if r.get("csa_mm2") is not None:
        parts.append(f"sac area {r['csa_mm2']:.0f} mm2 ({r.get('csa_grade')})")
    if r.get("ap_mm") is not None:
        parts.append(f"sagittal AP {r['ap_mm']:.1f} mm ({r.get('ap_grade')})")
    return Candidate(
        value=str(r["grade"]),
        confidence=float(r["confidence"] or 0.0),
        notes="; ".join(parts) + ". " + str(r.get("note", "")),
    )


# Only the suggested grade goes into the report schema. The area is not written there while it is unvalidated (the
# axial figure is for review); the per-level area is in the rows and the CSV.
register_field(FieldRule("central_canal", "stenosis", "canal", _extract_stenosis, figure="profile"))
