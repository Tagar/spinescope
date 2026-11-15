"""Focal vertebral marrow lesions and diffuse marrow signal on sagittal T1 and T2 (Dixon fat fraction when present).

What it does, per vertebral body on every reference slice:

1. Takes the body geometry from the disc frames (the same traced endplates as :mod:`.endplates`) and keeps the body
   **core**: at least ``marrow_focal_endplate_mm`` (the top of the endplate marrow band) from both endplates and clear of
   the 1.5 mm cortical rim. Endplate changes and Schmorl nodes therefore never enter the focal search.
2. Expresses T1 and T2 (and STIR when present) as a ratio to the same body's mid-body marrow on the same slice.
3. Segments focal foci by contrast pattern: T1 and T2 bright (haemangioma-like, or fat-like when a fat-sensitive
   sequence says so), T1 and T2 dark (bone island-like), and T1 dark relative to the adjacent disc without the bone
   island pattern (atypical). Thin linear structures (the basivertebral vein) are excluded as not focal.
4. Groups the foci across slices and reports size in mm. Only the atypical pattern is flagged for a reader; the other
   patterns are rows. Nothing here names a malignancy: an atypical focus is a prompt for a dedicated evaluation.

Diffuse marrow: the mid-body marrow T1 is compared with the adjacent discs' nucleus T1 (per body, median over slices).
Marrow darker than the discs is flagged as diffusely low T1 marrow. The paraspinal-muscle reference is not measured here:
it needs a parasagittal or axial stack, which this analysis does not use (see docs/analysis/marrow.md).

Dixon fat fraction (water and fat stacks, ``role.dixon``) is measured per body and per focus when both are present.

Thresholds are in :class:`spinescope.config.Config` (``marrow_*``), relative to the same body's marrow. They are
judgement values, uncalibrated, and every flag carries at most ``marrow_focal_max_confidence``.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..study import Study
from .base import Result, register, sagittal_stacks
from .discs import DISC_CONTRAST_ORDER, DiscTrace, _body_height, background, trace_disc
from .endplates import _body_geometry, _interp_endplate, _resample_to

log = logging.getLogger(__name__)

MARROW_NOTE = (
    "focal and diffuse marrow measurements are experimental: the rules are not validated on public gradings; "
    "check T1 and T2 at this level"
)
ATYPICAL_TEXT = "atypical focal marrow lesion; consider dedicated evaluation"
DIFFUSE_TEXT = (
    "diffusely low T1 marrow signal; can reflect marrow reconversion or infiltration; consider correlation"
)
CLASS_HAEM = "haemangioma-like"
CLASS_FAT = "focal fat-like"
CLASS_UNDECIDED = "T1 and T2 bright: haemangioma or fat (no fat-sensitive sequence)"
CLASS_ISLAND = "bone island-like"
CLASS_ATYPICAL = "atypical"
#: pattern names used internally (before the fat decision)
PAT_BRIGHT = "bright T1 and T2"
PAT_DARK = "dark T1 and T2"
PAT_ATYPICAL = "dark T1 below disc, not dark T2"
_PATTERNS = (PAT_BRIGHT, PAT_DARK, PAT_ATYPICAL)


def _smooth(img: NDArray[np.float64], sigma_px: float) -> NDArray[np.float64]:
    """Gaussian smoothing with NaN (outside the slab) filled by the median, as in :mod:`.endplates`."""
    fill = float(np.nanmedian(img)) if np.isfinite(img).any() else 0.0
    return ndi.gaussian_filter(np.where(np.isfinite(img), img, fill), sigma=sigma_px)


def _nucleus_median(tr: DiscTrace, pts: NDArray[np.float64], img: NDArray[np.float64], cfg: Config) -> float:
    """Median of ``img`` over the central nucleus box of one traced disc (box as in :mod:`.discs`)."""
    s = (pts - tr.origin) @ tr.u
    h = (pts - tr.origin) @ tr.n
    bot, in_b = _interp_endplate(tr, s, "bot")
    top, in_t = _interp_endplate(tr, s, "top")
    mid_s = 0.5 * (tr.s_ant + tr.s_post)
    length = tr.s_post - tr.s_ant
    with np.errstate(invalid="ignore"):
        hmid = 0.5 * (top + bot)
        hh = top - bot
        box = (
            in_b
            & in_t
            & (np.abs(s - mid_s) <= cfg.nucleus_ap_fraction * length)
            & (np.abs(h - hmid) <= cfg.nucleus_h_fraction * hh)
            & (h < top)
            & (h > bot)
        )
    vals = img[box]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def _shape_mm(
    rr: NDArray[np.intp], cc: NDArray[np.intp], row_mm: float, col_mm: float
) -> tuple[float, float, float]:
    """(width, length, size) of a 2-D pixel set in mm. Width and length are 4 standard deviations along the minor and
    major axes (full width of a uniform ellipse); size is the larger bounding-box side (the largest extent)."""
    x = cc * col_mm
    y = rr * row_mm
    ap_ext = float(np.max(cc) - np.min(cc) + 1) * col_mm
    up_ext = float(np.max(rr) - np.min(rr) + 1) * row_mm
    if rr.size < 2:
        return 0.0, 0.0, max(ap_ext, up_ext)
    cov = np.cov(np.vstack([x, y]))
    vals = np.sort(np.linalg.eigvalsh(cov))
    width = 4.0 * math.sqrt(max(float(vals[0]), 0.0))
    length = 4.0 * math.sqrt(max(float(vals[1]), 0.0))
    return width, length, max(ap_ext, up_ext)


def _classify(pattern: str, ff: float, stir: float, cfg: Config) -> str:
    """Focal class from the pattern and the fat evidence (Dixon fat fraction first, then STIR)."""
    if pattern == PAT_DARK:
        return CLASS_ISLAND
    if pattern == PAT_ATYPICAL:
        return CLASS_ATYPICAL
    if math.isfinite(ff):
        fat = ff >= cfg.marrow_fat_fraction_focal
    elif math.isfinite(stir):
        fat = stir <= cfg.marrow_stir_fat_ratio
    else:
        return CLASS_UNDECIDED
    return CLASS_FAT if fat else CLASS_HAEM


def _cluster(cands: list[dict[str, Any]], link_mm: float) -> list[list[dict[str, Any]]]:
    """Group per-slice components of one body and pattern into foci (centres within ``link_mm`` in x-y and z)."""
    clusters: list[list[dict[str, Any]]] = []
    for cand in sorted(cands, key=lambda d: (-d["area_mm2"], d["k"])):
        for cl in clusters:
            rep = cl[0]
            if (
                rep["vertebra"] == cand["vertebra"]
                and rep["pattern"] == cand["pattern"]
                and math.hypot(rep["x_mm"] - cand["x_mm"], rep["y_mm"] - cand["y_mm"]) < link_mm
                and abs(rep["z_mm"] - cand["z_mm"]) < link_mm
            ):
                cl.append(cand)
                break
        else:
            clusters.append([cand])
    return clusters


@dataclass
class _View:
    """One traced body on one reference slice: masks on the reference grid and the smoothed sequences."""

    body: str
    mid: NDArray[np.bool_]
    core: NDArray[np.bool_]
    img: dict[str, NDArray[np.float64]]
    disc_t1: float
    ff_mid: float


def _pooled_reference(vals: dict[str, NDArray[np.float64]], cfg: Config) -> dict[str, float]:
    """Marrow reference of one body from its pooled mid-body pixels (two passes: first median, then the median of the
    pixels that stay within ``marrow_reference_exclude_ratio`` of it in T1 and T2)."""
    first = {c: float(np.median(v)) for c, v in vals.items()}
    n = next(iter(vals.values())).size
    dev = np.zeros(n, dtype=bool)
    for c in ("t1", "t2"):
        if c in vals and first[c] != 0.0:
            dev |= np.abs(vals[c] / first[c] - 1.0) > cfg.marrow_reference_exclude_ratio
    keep = ~dev if (~dev).sum() >= 10 else np.ones(n, dtype=bool)
    return {c: float(np.median(v[keep])) for c, v in vals.items()}


@register("marrow")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Focal marrow lesion candidates (typical haemangioma-like, fat-like, bone island-like, atypical) and diffuse T1."""
    cfg = config or DEFAULT
    res = Result(
        name="marrow",
        meta={
            "method": "disc-frame body geometry; body-core foci by T1/T2 ratio to the same body's marrow; "
            "diffuse T1 marrow over adjacent disc nucleus",
            "status": "experimental",
            "note": MARROW_NOTE,
            "muscle_reference": "not measured (no parasagittal or axial stack in this analysis)",
            "linear_excluded": 0,
            "focal_flags": "on"
            if cfg.marrow_focal_flags
            else "off: failed pre-registered validation (docs/analysis/marrow.md)",
        },
    )
    stacks = sagittal_stacks(study)
    if not stacks:
        res.warnings.append("no sagittal stack in study: marrow analysis skipped")
        return res
    ref_name = next(c for c in DISC_CONTRAST_ORDER if c in stacks)
    ref = stacks[ref_name]
    if ref.plane != "sag":
        res.warnings.append("reference stack is not sagittal: marrow analysis skipped")
        return res
    seq = {c: stacks[c] for c in ("t1", "t2", "stir") if c in stacks}
    have_t1, have_t2 = "t1" in seq, "t2" in seq
    dixon = {
        comp: next(
            (
                st
                for st in study.stacks.values()
                if st.plane == "sag" and st.meta.get("role", {}).get("dixon") == comp
            ),
            None,
        )
        for comp in ("w", "f")
    }
    have_dixon = dixon["w"] is not None and dixon["f"] is not None
    res.meta.update(reference=ref_name, sequences=sorted(seq), series_key=ref.key, dixon=have_dixon)
    if not have_t1:
        res.warnings.append("no sagittal T1: focal lesions not characterised and diffuse T1 not measured")
    if not have_t2:
        res.warnings.append("no sagittal T2: focal lesions not characterised (T1-only foci are not called)")
    if not have_dixon:
        res.warnings.append("no Dixon water and fat stacks: fat fraction not measured")
    if not have_t1 and not have_t2:
        return res
    labels = anchors.disc_labels()
    if not labels:
        res.warnings.append("no discs in anchors: marrow analysis skipped")
        return res

    upper_of = {lab.split("-")[1]: lab for lab in labels}  # body -> disc above it
    lower_of = {lab.split("-")[0]: lab for lab in labels}  # body -> disc below it
    body_names = sorted({*upper_of, *lower_of})

    floors = {c: background(st) for c, st in seq.items()}
    ref_floor = background(ref)
    traces = {
        k: {lab: trace_disc(ref, k, lab, anchors.discs[lab], cfg, floor=ref_floor) for lab in labels}
        for k in range(ref.n)
    }
    heights = [
        _body_height(tr[up], tr[lo])
        for tr in traces.values()
        for up, lo in itertools.pairwise(labels)
        if up.split("-")[1] == lo.split("-")[0]
    ]
    heights = [h for h in heights if np.isfinite(h)]
    h_nominal = float(np.median(heights)) if heights else float(sum(cfg.body_height_mm) / 2)

    rows_n, cols_n = ref.vol.shape[1:]
    rr, cc = np.mgrid[0:rows_n, 0:cols_n].astype(np.float64)
    smooth_px = cfg.ratio_smooth_mm / ref.row_mm
    min_area_mm2 = math.pi * (cfg.marrow_min_diameter_mm / 2.0) ** 2
    link_mm = cfg.marrow_focus_link_mm

    def slice_views(k: int) -> list[_View]:
        """Smoothed sequences and the body masks of reference slice ``k`` (one view per traced body)."""
        tr_k = traces[k]
        pts = ref.ipp[k] + cc[..., None] * ref.col_mm * ref.row_dir + rr[..., None] * ref.row_mm * ref.col_dir
        img: dict[str, NDArray[np.float64]] = {}
        for c, st in seq.items():
            img[c] = _smooth(_resample_to(ref, k, st) - floors[c], smooth_px)
        if have_dixon:
            w = _resample_to(ref, k, dixon["w"])  # type: ignore[arg-type]
            f = _resample_to(ref, k, dixon["f"])  # type: ignore[arg-type]
            with np.errstate(invalid="ignore", divide="ignore"):
                ff = np.where(f + w > 0, f / (f + w), np.nan)
            img["ff"] = _smooth(ff, smooth_px)
        out: list[_View] = []
        for V in body_names:
            upper = tr_k.get(upper_of.get(V, ""))
            lower = tr_k.get(lower_of.get(V, ""))
            bg = _body_geometry(ref, k, V, upper, lower, h_nominal, pts, cfg)
            if bg is None or not bg.body.any():
                continue
            real = [t for t in (upper, lower) if t is not None]
            if not any(t.valid_fraction() >= 0.5 for t in real):
                continue
            ds, di = bg.depth_sup, bg.depth_inf
            with np.errstate(invalid="ignore"):
                mid_min = np.maximum(cfg.midbody_min_mm, 0.35 * (ds + di))
                mid = bg.body & (ds >= mid_min) & (di >= mid_min)
                inner = bg.body & (ds >= cfg.marrow_focal_endplate_mm) & (di >= cfg.marrow_focal_endplate_mm)
            if not mid.any():
                continue
            rim_it = max(1, round(cfg.marrow_cortex_rim_mm / min(ref.row_mm, ref.col_mm)))
            core = inner & ndi.binary_erosion(bg.body, iterations=rim_it)  # drop the 1.5 mm cortical rim
            disc_vals = [_nucleus_median(t, pts, img["t1"], cfg) for t in real] if have_t1 else []
            disc_t1 = float(np.nanmean(disc_vals)) if any(np.isfinite(disc_vals)) else float("nan")
            ff_mid = float(np.median(img["ff"][mid])) if have_dixon else float("nan")
            out.append(_View(body=V, mid=mid, core=core, img=img, disc_t1=disc_t1, ff_mid=ff_mid))
        return out

    # pass A: marrow reference per body, pooled over its slices. A focus covers one or two slices, so the pooled median
    # is not set by it; a per-slice reference would be (the lesion and a basivertebral vein can fill the mid band of one
    # slice). Two passes inside the pool: pixels deviating by more than marrow_reference_exclude_ratio are left out.
    pooled: dict[str, dict[str, list[NDArray[np.float64]]]] = {}
    views_by_k: list[list[_View]] = []
    for k in range(ref.n):
        views = slice_views(k)
        views_by_k.append(views)
        for view in views:
            d = pooled.setdefault(view.body, {c: [] for c in seq})
            for c in seq:
                d[c].append(view.img[c][view.mid])
    ref_by_body: dict[str, dict[str, float]] = {}
    for V, d in sorted(pooled.items()):
        vals = {c: np.concatenate(arrs) for c, arrs in d.items()}
        ref_by_body[V] = _pooled_reference(vals, cfg)

    # pass B: candidates per slice against the body's marrow reference
    cands: list[dict[str, Any]] = []
    body_obs: dict[
        str, list[tuple[float, float, float]]
    ] = {}  # body -> per slice (T1 marrow, T1 disc, Dixon FF)
    linear = 0
    for k, views in enumerate(views_by_k):
        for view in views:
            V = view.body
            if V not in ref_by_body:
                continue
            mid_vals = ref_by_body[V]
            img, core = view.img, view.core
            t1_marrow = mid_vals.get("t1", float("nan"))
            body_obs.setdefault(V, []).append((t1_marrow, view.disc_t1, view.ff_mid))
            if not (have_t1 and have_t2) or mid_vals["t1"] == 0.0 or mid_vals["t2"] == 0.0 or not core.any():
                continue
            ratio = {c: img[c] / mid_vals[c] for c in seq}
            r1, r2 = ratio["t1"], ratio["t2"]
            rs = ratio.get("stir")
            disc_t1 = view.disc_t1
            with np.errstate(invalid="ignore"):
                b1 = r1 >= cfg.marrow_focal_t1_bright
                d1 = r1 <= cfg.marrow_focal_t1_dark
                b2 = r2 >= cfg.marrow_focal_t2_bright
                d2 = r2 <= cfg.marrow_focal_t2_dark
                if rs is not None:
                    s_not_bright = rs <= cfg.marrow_stir_not_bright_ratio
                else:
                    s_not_bright = np.ones(r1.shape, dtype=bool)
                t1_below_disc = (
                    (img["t1"] < disc_t1) if np.isfinite(disc_t1) else np.zeros(r1.shape, dtype=bool)
                )
            masks = {
                PAT_BRIGHT: core & b1 & b2,
                PAT_DARK: core & d1 & d2 & s_not_bright,
                PAT_ATYPICAL: core & d1 & t1_below_disc & ~(d2 & s_not_bright),
            }
            for pattern in _PATTERNS:
                lab_img, n = ndi.label(masks[pattern])
                for i in range(1, n + 1):
                    comp = lab_img == i
                    if comp.sum() * ref.row_mm * ref.col_mm < min_area_mm2:
                        continue
                    pr, pc = np.nonzero(comp)
                    width, length, size = _shape_mm(pr, pc, ref.row_mm, ref.col_mm)
                    if width <= cfg.marrow_linear_max_width_mm and length >= cfg.marrow_linear_aspect * max(
                        width, 1e-6
                    ):
                        linear += 1
                        continue
                    x, y, z = ref.pix_to_lps(k, float(pr.mean()), float(pc.mean()))

                    def med(
                        arr: NDArray[np.float64] | None, pr: NDArray[np.intp] = pr, pc: NDArray[np.intp] = pc
                    ) -> float:
                        if arr is None:
                            return float("nan")
                        return float(np.median(arr[pr, pc]))

                    cands.append(
                        {
                            "vertebra": V,
                            "pattern": pattern,
                            "k": k,
                            "x_mm": float(x),
                            "y_mm": float(y),
                            "z_mm": float(z),
                            "area_mm2": float(comp.sum()) * ref.row_mm * ref.col_mm,
                            "size_mm": size,
                            "T1_ratio": med(r1),
                            "T2_ratio": med(r2),
                            "STIR_ratio": med(rs),
                            "FF": med(img.get("ff")),
                            "disc_T1": disc_t1,
                        }
                    )
    res.meta["linear_excluded"] = linear

    for cl in _cluster(cands, link_mm):
        best = max(cl, key=lambda d: d["area_mm2"])
        n_sl = len({d["k"] for d in cl})
        pattern = best["pattern"]
        cls = _classify(pattern, best["FF"], best["STIR_ratio"], cfg)
        conf = min(0.5 if n_sl >= 2 else 0.3, cfg.marrow_focal_max_confidence)
        row: dict[str, Any] = {
            "scope": "focus",
            "vertebra": best["vertebra"],
            "class": cls,
            "pattern": pattern,
            "slices_k": sorted({d["k"] for d in cl}),
            "n_slices": n_sl,
            "x_mm": round(best["x_mm"], 1),
            "y_mm": round(best["y_mm"], 1),
            "z_mm": round(best["z_mm"], 1),
            "size_mm": round(best["size_mm"], 1),
            "area_mm2": round(best["area_mm2"], 1),
            "T1_ratio": round(best["T1_ratio"], 3),
            "T2_ratio": round(best["T2_ratio"], 3),
            "STIR_ratio": round(best["STIR_ratio"], 3) if math.isfinite(best["STIR_ratio"]) else None,
            "fat_fraction_pct": round(100 * best["FF"], 1) if math.isfinite(best["FF"]) else None,
            "confidence": round(conf, 2),
            "note": "candidate only; body core, at least 8 mm from both endplates, away from the cortical rim",
        }
        res.rows.append(row)
        if cls == CLASS_ATYPICAL and cfg.marrow_focal_flags:
            res.flags.append(
                {
                    "kind": "atypical_focal_lesion",
                    "vertebra": best["vertebra"],
                    "size_mm": row["size_mm"],
                    "x_mm": row["x_mm"],
                    "y_mm": row["y_mm"],
                    "z_mm": row["z_mm"],
                    "confidence": row["confidence"],
                    "note": ATYPICAL_TEXT,
                }
            )

    for V in sorted(body_obs):
        obs = body_obs[V]
        marrow = [t for t, d, _ in obs if math.isfinite(t) and math.isfinite(d) and d != 0]
        disc = [d for t, d, _ in obs if math.isfinite(t) and math.isfinite(d) and d != 0]
        ffs = [f for _, _, f in obs if math.isfinite(f)]
        if not marrow:
            continue
        marrow_over_disc = float(np.median([t / d for t, d in zip(marrow, disc, strict=True)]))
        row = {
            "scope": "body",
            "vertebra": V,
            "T1_marrow_over_disc": round(marrow_over_disc, 3),
            "n_slices": len(marrow),
            "fat_fraction_pct": round(100 * float(np.median(ffs)), 1) if ffs else None,
            "note": "diffuse T1 relative to the adjacent disc nucleus; age-unaware, no normative comparison",
        }
        res.rows.append(row)
        if marrow_over_disc < cfg.marrow_diffuse_disc_ratio:
            res.flags.append(
                {
                    "kind": "diffuse_t1_low",
                    "vertebra": V,
                    "T1_marrow_over_disc": row["T1_marrow_over_disc"],
                    "confidence": round(
                        min(0.5 if len(marrow) >= 3 else 0.3, cfg.marrow_diffuse_max_confidence), 2
                    ),
                    "note": DIFFUSE_TEXT,
                }
            )
    return res
