"""Endplates and marrow on sagittal T1 / T2 / STIR: Modic-type suggestions, extent, lesions, triptychs.

For every endplate adjacent to an anchored disc (the superior endplate of each body under a disc and the
inferior endplate of each body above one), this module:

1. Traces the endplate on the reference sagittal stack with the disc frames from :mod:`.discs`.
2. Builds depth maps (mm from the endplate, inside the body) on every sagittal slice that intersects the
   body. A slice intersects a body when its endplates are traced on at least half their AP columns, so
   the analysed width is measured from the data, not assumed. Sequences are resampled onto the reference
   grid, so a T1, T2 or STIR stack with different slice positions is still aligned.
3. Samples the marrow band 3-8 mm inside each endplate (clear of cortex) and the mid-body marrow of the
   same vertebra on the same slice. Ratios to the mid-body are the signal measure.
4. Reports a Modic-type suggestion for each endplate overall and for each side (left, central, right of
   the anchored midline). Side is decided by the slice position, so a one-sided change is visible.
5. Reports extent (fraction of the band that is abnormal) and depth (95th percentile of abnormal marrow).
   Ratios are also given against the study's healthiest bodies, meaning bodies whose endplates are normal
   on both sides.
6. Searches for focal marrow lesion candidates in the body core, away from endplates and cortex.

Everything is a suggestion with a confidence. Thresholds in :class:`spinescope.config.Config` are relative
to the vertebra's own mid-body marrow and are uncalibrated until validated on public gradings.
"""

from __future__ import annotations

import itertools
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
from ..geometry import Stack
from ..study import Study
from .base import Result, register, sagittal_stacks
from .discs import DISC_CONTRAST_ORDER, DiscTrace, _body_height, background, trace_disc

log = logging.getLogger(__name__)

MODIC_PATTERNS = ("Modic 1", "Modic 1-like", "Modic 2", "Modic 3")
SEQS = ("t1", "t2", "stir")
SIDES = ("left", "central", "right")
FIG_HALF_PX = 50  # half size of a triptych crop, native pixels
FIG_SCALE = 2


# ----------------------------------------------------------------------- resampling
def _resample_to(ref: Stack, k: int, other: Stack) -> NDArray[np.float64]:
    """Sequence ``other`` sampled on the pixel grid of slice ``k`` of ``ref`` (NaN outside its slab)."""
    rows, cols = ref.vol.shape[1:]
    rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float64)
    pts = ref.ipp[k] + cc[..., None] * ref.col_mm * ref.row_dir + rr[..., None] * ref.row_mm * ref.col_dir
    d = pts @ other.normal
    if other.n > 1:
        kf = np.interp(d, other.positions, np.arange(other.n, dtype=np.float64))
        half = 0.5 * (other.spacing or other.thickness or 1.0)
        outside = (d < other.positions[0] - half) | (d > other.positions[-1] + half)
    else:
        kf = np.zeros_like(d)
        outside = np.abs(d - other.positions[0]) > max(other.thickness, 1.0) / 2
    k0 = np.clip(np.floor(kf), 0, other.n - 1)
    k1 = np.clip(k0 + 1, 0, other.n - 1)
    f = kf - k0
    origin = other.ipp[k0.astype(int)] * (1 - f)[..., None] + other.ipp[k1.astype(int)] * f[..., None]
    rel = pts - origin
    r = rel @ other.col_dir / other.row_mm
    c = rel @ other.row_dir / other.col_mm
    vals = ndi.map_coordinates(other.vol.astype(np.float64), [kf, r, c], order=1, mode="nearest")
    vals = np.where(outside, np.nan, vals)
    return vals


# ----------------------------------------------------------------------- geometry per slice
@dataclass
class _Body:
    """One vertebral body on one reference slice: masks on the reference pixel grid."""

    name: str
    k: int
    x_mm: float
    sup_traced: bool  # superior endplate traced (False: nominal)
    inf_traced: bool
    body: NDArray[np.bool_]
    band_sup: NDArray[np.bool_]
    band_inf: NDArray[np.bool_]
    mid: NDArray[np.bool_]
    core: NDArray[np.bool_]
    depth_sup: NDArray[np.float64]
    depth_inf: NDArray[np.float64]
    height_mm: float
    intersects: bool
    ap_mm: NDArray[
        np.float64
    ]  # anterior-posterior position of every pixel along the endplate (mm, NaN outside)


def _defect_profile(tr: DiscTrace, which: str, window_mm: float) -> tuple[float, float]:
    """Endplate contour defect of one traced endplate: (concavity_mm, irregularity_mm).

    ``which`` is ``"bot"`` (the superior endplate of the body below the disc: intrusion lowers the line) or ``"top"``
    (the inferior endplate of the body above: intrusion raises it). The smooth contour is a running median over
    ``window_mm`` along AP. Concavity is the largest intrusion into the body relative to that contour (mm, >= 0);
    irregularity is the RMS deviation from it. Both are NaN when fewer than 5 columns are traced.
    """
    ok = tr.ok
    if ok.sum() < 5:
        return float("nan"), float("nan")
    s_ok = tr.s[ok]
    v = (tr.bot if which == "bot" else tr.top)[ok]
    step = float(np.median(np.diff(s_ok))) if s_ok.size > 1 else 1.0
    size = max(3, round(window_mm / max(step, 1e-6)) | 1)
    smooth = ndi.median_filter(v, size=size, mode="nearest")
    resid = v - smooth
    sign = -1.0 if which == "bot" else 1.0  # into the body
    concavity = float(max(0.0, np.max(sign * resid)))
    irregularity = float(np.sqrt(np.mean(resid**2)))
    return concavity, irregularity


def _interp_endplate(
    tr: DiscTrace, s: NDArray[np.float64], which: str
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Endplate height (bot or top) of a disc trace at AP positions ``s``; mask of positions inside the trace."""
    ok = tr.ok
    if ok.sum() < 2:
        return np.full(s.shape, np.nan), np.zeros(s.shape, dtype=bool)
    grid = tr.s[ok]
    vals = (tr.bot if which == "bot" else tr.top)[ok]
    inside = (s >= np.min(grid)) & (s <= np.max(grid))
    return np.interp(s, grid, vals), inside


def _body_geometry(
    ref: Stack,
    k: int,
    name: str,
    upper: DiscTrace | None,
    lower: DiscTrace | None,
    h_nominal: float,
    grid_pts: NDArray[np.float64],
    cfg: Config,
) -> _Body | None:
    """Depth maps and masks for one body on slice ``k``. ``upper``/``lower`` are the discs above/below."""
    if upper is None and lower is None:
        return None
    rows, cols = ref.vol.shape[1:]
    nanarr = np.full((rows, cols), np.nan)
    ap_ok = np.ones((rows, cols), dtype=bool)
    corner = cfg.corner_inset_mm
    if upper is not None:
        s_u = (grid_pts - upper.origin) @ upper.u
        h_u = (grid_pts - upper.origin) @ upper.n
        bot_u, in_u = _interp_endplate(upper, s_u, "bot")
        depth_sup = bot_u - h_u
        ap_ok &= (s_u >= upper.s_ant + corner) & (s_u <= upper.s_post - corner) & in_u
        ds_valid = in_u
        sup_traced = True
    else:
        depth_sup = nanarr.copy()
        ds_valid = np.zeros((rows, cols), dtype=bool)
        sup_traced = False
    if lower is not None:
        s_d = (grid_pts - lower.origin) @ lower.u
        h_d = (grid_pts - lower.origin) @ lower.n
        top_d, in_d = _interp_endplate(lower, s_d, "top")
        depth_inf = h_d - top_d
        ap_ok &= (s_d >= lower.s_ant + corner) & (s_d <= lower.s_post - corner) & in_d
        di_valid = in_d
        inf_traced = True
    else:
        depth_inf = nanarr.copy()
        di_valid = np.zeros((rows, cols), dtype=bool)
        inf_traced = False
    if not sup_traced and not inf_traced:
        return None
    if not sup_traced:
        depth_sup = h_nominal - depth_inf
        ds_valid = di_valid
    if not inf_traced:
        depth_inf = h_nominal - depth_sup
        di_valid = ds_valid
    valid = ds_valid & di_valid & ap_ok
    ds = np.where(valid, depth_sup, np.nan)
    di = np.where(valid, depth_inf, np.nan)
    with np.errstate(invalid="ignore"):
        body = valid & (ds >= 0) & (di >= 0)
        H = ds + di
        lo, hi = cfg.endplate_band_mm
        band_sup = body & (ds >= lo) & (ds <= hi)
        band_inf = body & (di >= lo) & (di <= hi)
        m = np.maximum(cfg.midbody_min_mm, 0.35 * H)
        mid = body & (ds >= m) & (di >= m)
        core = body & (ds >= cfg.lesion_endplate_mm) & (di >= cfg.lesion_endplate_mm)
    # posterior inset for the lesion core only (canal, posterior cortex)
    if upper is not None:
        s_post_ok = (grid_pts - upper.origin) @ upper.u <= upper.s_post - cfg.posterior_inset_mm
        core &= s_post_ok
    height = float(np.nanmedian(H[body])) if body.any() else float("nan")
    x_mm = float(ref.ipp[k] @ np.array([1.0, 0.0, 0.0]))
    ap = s_u if upper is not None else s_d
    return _Body(
        ap_mm=np.where(body, ap, np.nan),
        name=name,
        k=k,
        x_mm=x_mm,
        sup_traced=sup_traced,
        inf_traced=inf_traced,
        body=body,
        band_sup=band_sup,
        band_inf=band_inf,
        mid=mid,
        core=core,
        depth_sup=ds,
        depth_inf=di,
        height_mm=height,
        intersects=bool(body.sum() >= 1),
    )


# ----------------------------------------------------------------------- pattern rules
PRESENCE_NO_T1 = "presence only (no T1)"
PRESENCE_NO_T2 = "presence only (no T2)"
PRESENCE_PATTERNS = (PRESENCE_NO_T1, PRESENCE_NO_T2)
NOT_ASSESSED_NO_T1 = "not assessed (no T1)"


def _pattern(r1: float, r2: float, rs: float, cfg: Config) -> str:
    """Modic-type suggestion from band-to-mid-body ratios (NaN = sequence missing).

    A type needs T1 and T2: every Modic type changes T1, and T1 with T2 separate the types. With one of them missing
    the result is presence-only (a change is seen, the type is not assessed) or ``none`` when T1 is normal, because
    no Modic type exists with a normal T1. STIR decides Modic 1-like (T1 low, STIR high, T2 not high) when present.
    """
    have1, have2, haves = np.isfinite(r1), np.isfinite(r2), np.isfinite(rs)
    t1_lo = have1 and r1 < cfg.t1_low
    t1_hi = have1 and r1 > cfg.t1_high
    t2_lo = have2 and r2 < cfg.t2_low
    t2_hi = have2 and r2 > cfg.t2_high
    s_hi = haves and rs > cfg.stir_high
    if not have1:
        return PRESENCE_NO_T1 if (t2_lo or t2_hi or s_hi) else NOT_ASSESSED_NO_T1
    if not have2:
        return PRESENCE_NO_T2 if (t1_lo or t1_hi or s_hi) else "none"
    if t1_lo and t2_hi and (s_hi or not haves):
        return "Modic 1"
    if t1_lo and s_hi and not t2_hi:
        return "Modic 1-like"
    if t1_lo and t2_lo:
        return "Modic 3"
    if t1_hi and not s_hi and not t2_lo:
        return "Modic 2"
    if not (t1_lo or t1_hi or t2_lo or t2_hi or s_hi):
        return "none"
    return "indeterminate"


def _abnormal_pixels(
    r1: NDArray[np.float64], r2: NDArray[np.float64], rs: NDArray[np.float64], cfg: Config
) -> NDArray[np.bool_]:
    """Per-pixel marrow abnormality (Modic-like) from smoothed ratios; NaN ratios count as normal."""
    with np.errstate(invalid="ignore"):
        t1_lo = r1 < cfg.t1_low
        t1_hi = r1 > cfg.t1_high
        t2_hi = r2 > cfg.t2_high
        t2_lo = r2 < cfg.t2_low
        s_hi = rs > cfg.stir_high
        return (t1_lo & (t2_hi | s_hi)) | (t1_hi & ~s_hi & ~t2_lo) | (t1_lo & t2_lo)


# ----------------------------------------------------------------------- observations
@dataclass
class _Obs:
    """Measurements of one endplate band (one body, one endplate, one slice)."""

    k: int
    x_mm: float
    side: str
    band: dict[str, NDArray[np.float64]]  # seq -> band values (smoothed)
    ap_band: NDArray[np.float64]  # AP position of each band pixel (mm)
    mid: dict[str, float]  # seq -> mid-body median (smoothed)
    abn: NDArray[np.bool_]  # abnormal flag per band pixel
    depth_abn: NDArray[np.float64]  # depth of abnormal body pixels from this endplate (mm)


def _side_of(x_mm: float, midline: float, half_spacing: float) -> str:
    dx = x_mm - midline
    if abs(dx) < half_spacing:
        return "central"
    return "left" if dx > 0 else "right"


def _ratio_median(values: NDArray[np.float64], ref: float) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0 or not np.isfinite(ref) or ref == 0:
        return float("nan")
    return float(np.median(v)) / ref


def _mode(vals: list[str]) -> tuple[str, float]:
    if not vals:
        return "not assessed", 0.0
    uniq, counts = np.unique(np.array(vals), return_counts=True)
    i = int(np.argmax(counts))
    return str(uniq[i]), float(counts[i]) / len(vals)


def _confidence(n_seq: int, agree: float, slices: int) -> float:
    """Heuristic confidence: more sequences and more slices agreeing give higher values (capped)."""
    base = {0: 0.0, 1: 0.3, 2: 0.5}.get(n_seq, 0.7)
    if slices < 2:
        base *= 0.7
    return round(min(0.9, base * agree), 2)


# ----------------------------------------------------------------------- triptychs
def _window(img: NDArray[np.float64]) -> tuple[float, float]:
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    return float(np.percentile(v, 1)), float(np.percentile(v, 99.5))


def _to_rgb(img: NDArray[np.float64], win: tuple[float, float]) -> NDArray[np.uint8]:
    lo, hi = win
    g = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1) * 255
    return np.stack([g, g, g], axis=-1).astype(np.uint8)


def _overlay_mask(
    rgb: NDArray[np.uint8], mask: NDArray[np.bool_], color: tuple[int, int, int], alpha: float
) -> None:
    if not mask.any():
        return
    rgb[mask] = ((1 - alpha) * rgb[mask] + alpha * np.array(color)).astype(np.uint8)
    edge = mask & ~ndi.binary_erosion(mask)
    rgb[edge] = color


def _overlay_curve(
    rgb: NDArray[np.uint8],
    ref: Stack,
    k: int,
    tr: DiscTrace,
    which: str,
    r0: int,
    c0: int,
    color: tuple[int, int, int],
) -> None:
    ok = tr.ok
    if ok.sum() < 2:
        return
    ys = (tr.bot if which == "bot" else tr.top)[ok]
    pts = tr.origin + tr.s[ok][:, None] * tr.u + ys[:, None] * tr.n
    _, rr, cc = [np.asarray(v) for v in zip(*[ref.lps_to_pix(p) for p in pts], strict=False)]
    for r, c in zip(rr, cc, strict=False):
        ri, ci = round(r) - r0, round(c) - c0
        if 0 <= ri < rgb.shape[0] and 0 <= ci < rgb.shape[1]:
            rgb[ri, ci] = color


def _triptych(
    ref: Stack,
    seqs: dict[str, NDArray[np.float64]],
    k: int,
    body: _Body,
    band_masks: list[tuple[NDArray[np.bool_], tuple[int, int, int], float]],
    curves: list[tuple[DiscTrace, str]],
    title: str,
) -> NDArray[np.uint8]:
    rr, cc = np.nonzero(body.band_sup | body.band_inf | body.mid)
    rc, cc0 = (
        int(np.mean(rr)) if rr.size else ref.vol.shape[1] // 2,
        int(np.mean(cc)) if cc.size else ref.vol.shape[2] // 2,
    )
    rows, cols = ref.vol.shape[1:]
    r0, r1 = max(0, rc - FIG_HALF_PX), min(rows, rc + FIG_HALF_PX)
    c0, c1 = max(0, cc0 - FIG_HALF_PX), min(cols, cc0 + FIG_HALF_PX)
    panels = []
    for name, img in seqs.items():
        crop = img[r0:r1, c0:c1]
        rgb = _to_rgb(crop, _window(img))
        for mask, color, alpha in band_masks:
            _overlay_mask(rgb, mask[r0:r1, c0:c1], color, alpha)
        for tr, which in curves:
            _overlay_curve(rgb, ref, k, tr, which, r0, c0, (255, 40, 40))
        big = Image.fromarray(rgb).resize(
            (rgb.shape[1] * FIG_SCALE, rgb.shape[0] * FIG_SCALE), Image.Resampling.NEAREST
        )
        d = ImageDraw.Draw(big)
        d.rectangle([(0, 0), (big.width, 14)], fill=(0, 0, 0))
        d.text((3, 2), f"{name} {title}", fill=(255, 255, 255))
        panels.append(np.asarray(big.convert("RGB")))
    gap = 6
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)
    x = 0
    for p in panels:
        canvas[: p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1] + gap
    return canvas


# ----------------------------------------------------------------------- summaries
def _vs(raw: float, ref: float | None) -> float:
    if ref is None or not np.isfinite(ref) or ref == 0 or not np.isfinite(raw):
        return float("nan")
    return round(raw / ref, 3)


def _summarise(ob: list[_Obs], cfg: Config) -> dict[str, dict[str, Any]]:
    """Suggestion, ratios, extent and depth for one endplate: 'all' slices and each side separately."""
    groups: dict[str, list[_Obs]] = {"all": ob}
    for side in SIDES:
        g = [o for o in ob if o.side == side]
        if g:
            groups[side] = g
    out: dict[str, dict[str, Any]] = {}
    for side, grp in groups.items():
        seqs = {c for o in grp for c in o.band}
        ratios: dict[str, list[float]] = {c: [] for c in SEQS}
        raw: dict[str, list[float]] = {c: [] for c in SEQS}
        pats: list[str] = []
        n_abn = n_band = 0
        depths: list[float] = []
        bins_all: set[int] = set()
        bins_abn: set[int] = set()
        for o in grp:
            ap_ok = np.isfinite(o.ap_band)
            bins = np.floor(o.ap_band[ap_ok]).astype(int)
            bins_all.update(bins.tolist())
            bins_abn.update(bins[o.abn[ap_ok]].tolist())
            per: dict[str, float] = {}
            for c in SEQS:
                if c in o.band and o.mid.get(c) and np.isfinite(o.band[c]).any():
                    per[c] = _ratio_median(o.band[c], o.mid[c])
                    raw[c].append(float(np.nanmedian(o.band[c])))
                    ratios[c].append(per[c])
            pats.append(_pattern(per.get("t1", np.nan), per.get("t2", np.nan), per.get("stir", np.nan), cfg))
            n_abn += int(o.abn.sum())
            n_band += int(o.abn.size)
            depths.extend(o.depth_abn.tolist())
        _, agree = _mode(pats)  # share of slices agreeing with the per-slice pattern (confidence only)
        med = {c: (float(np.median(ratios[c])) if ratios[c] else float("nan")) for c in SEQS}
        # pre-registered rule: the type comes from the median band/mid-body ratios over slices
        pat = _pattern(med["t1"], med["t2"], med["stir"], cfg) if pats else "not assessed"
        raw_med = {c: (float(np.median(raw[c])) if raw[c] else float("nan")) for c in SEQS}
        n_seq = sum(1 for c in SEQS if np.isfinite(med[c]))
        note = f"{len(grp)} slice(s); {agree:.0%} agree"
        if "t1" not in seqs:
            note += "; no T1"
        if "stir" not in seqs:
            note += "; no STIR"
        out[side] = {
            "n_slices": len(grp),
            "slices": [o.k for o in grp],
            "x_mm": round(float(np.mean([o.x_mm for o in grp])), 1),
            "T1": round(med["t1"], 3),
            "T2": round(med["t2"], 3),
            "STIR": round(med["stir"], 3),
            "T1_raw": raw_med["t1"],
            "T2_raw": raw_med["t2"],
            "STIR_raw": raw_med["stir"],
            "pattern": pat,
            # presence-only (type not assessed, a sequence missing) is always low confidence
            "confidence": min(0.3, _confidence(n_seq, agree, len(grp)))
            if pat in PRESENCE_PATTERNS
            else _confidence(n_seq, agree, len(grp)),
            "extent_pct": round(100.0 * n_abn / n_band, 1) if n_band else float("nan"),
            "width_pct": round(100.0 * len(bins_abn) / len(bins_all), 1) if bins_all else float("nan"),
            "depth_mm": round(float(np.percentile(depths, 95)), 1)
            if depths and pat in MODIC_PATTERNS
            else float("nan"),
            "note": note,
        }
    # the endplate-level suggestion is the worst side when a Modic pattern is confined to one or two sides
    mod_sides = [sd for sd in SIDES if sd in out and out[sd]["pattern"] in MODIC_PATTERNS]
    if mod_sides and out["all"]["pattern"] not in MODIC_PATTERNS:
        best = max(mod_sides, key=lambda sd: out[sd]["extent_pct"])
        out["all"].update(
            pattern=out[best]["pattern"],
            confidence=out[best]["confidence"],
            depth_mm=out[best]["depth_mm"],
            note=out["all"]["note"] + f"; {out[best]['pattern']} on {', '.join(mod_sides)} side(s) only",
        )
    return out


def _is_normal(summ: dict[str, dict[str, Any]]) -> bool:
    return all(v["pattern"] == "none" for v in summ.values())


def _healthy_reference(obs: dict[tuple[str, str], list[_Obs]], healthy: set[str]) -> dict[str, float]:
    """Median mid-body marrow per sequence over the healthy bodies' slices."""
    vals: dict[str, list[float]] = {c: [] for c in SEQS}
    for (V, _ep), ob in obs.items():
        if V not in healthy:
            continue
        for o in ob:
            for c in SEQS:
                if o.mid.get(c):
                    vals[c].append(o.mid[c])
    return {c: float(np.median(v)) for c, v in vals.items() if v}


# ----------------------------------------------------------------------- lesions
def _lesions(
    inputs: list[tuple[int, str, _Body, dict[str, NDArray[np.float64]]]],
    ref: Stack,
    cfg: Config,
    have: set[str],
) -> list[dict[str, Any]]:
    """Focal marrow lesion candidates in body cores (away from endplates and cortex), grouped across slices."""
    if not {"t1", "stir"} <= have:
        return []
    min_px = max(1, round(cfg.min_region_mm2 / (ref.row_mm * ref.col_mm)))
    raw: list[dict[str, Any]] = []
    for k, V, bg, sm in inputs:
        if not bg.mid.any():
            continue
        ratio = {c: sm[c] / float(np.median(sm[c][bg.mid])) for c in sm}
        r1 = ratio["t1"]
        r2 = ratio.get("t2", np.full(r1.shape, np.nan))
        rs = ratio["stir"]
        cortex = ndi.binary_dilation(r1 < cfg.cortex_t1_ratio, iterations=max(1, round(2.0 / ref.row_mm)))
        core = bg.core & ~cortex
        patterns = {
            "T1 low + STIR high": (r1 < cfg.t1_low) & (rs > cfg.stir_high),
            "T1 low + T2 low": (r1 < cfg.t1_low) & (r2 < cfg.t2_low),
            "T1 high + T2 high": (r1 > cfg.t1_high) & (r2 > cfg.t2_high),
        }
        for name, m in patterns.items():
            lab, n = ndi.label(m & core)
            for i in range(1, n + 1):
                comp = lab == i
                if comp.sum() < min_px:
                    continue
                rr_, cc_ = np.nonzero(comp)
                x, y, z = ref.pix_to_lps(k, float(rr_.mean()), float(cc_.mean()))
                raw.append(
                    {
                        "vertebra": V,
                        "pattern": name,
                        "k": k,
                        "x_mm": float(x),
                        "y_mm": float(y),
                        "z_mm": float(z),
                        "area_mm2": float(comp.sum()) * ref.row_mm * ref.col_mm,
                        "T1_ratio": float(np.median(r1[comp])),
                        "T2_ratio": float(np.median(r2[comp])),
                        "STIR_ratio": float(np.median(rs[comp])),
                    }
                )
    clusters: list[list[dict[str, Any]]] = []
    for cand in sorted(raw, key=lambda d: -d["area_mm2"]):
        for cl in clusters:
            rep = cl[0]
            if (
                rep["vertebra"] == cand["vertebra"]
                and rep["pattern"] == cand["pattern"]
                and math.hypot(rep["x_mm"] - cand["x_mm"], rep["y_mm"] - cand["y_mm"]) < 8
                and abs(rep["z_mm"] - cand["z_mm"]) < 8
            ):
                cl.append(cand)
                break
        else:
            clusters.append([cand])
    out: list[dict[str, Any]] = []
    for cl in clusters:
        best = max(cl, key=lambda d: d["area_mm2"])
        slices = sorted({d["k"] for d in cl})
        out.append(
            {
                "scope": "lesion",
                "vertebra": best["vertebra"],
                "pattern": best["pattern"],
                "slices_k": slices,
                "x_mm": round(best["x_mm"], 1),
                "y_mm": round(best["y_mm"], 1),
                "z_mm": round(best["z_mm"], 1),
                "area_mm2": round(best["area_mm2"], 1),
                "T1_ratio": round(best["T1_ratio"], 3),
                "T2_ratio": round(best["T2_ratio"], 3),
                "STIR_ratio": round(best["STIR_ratio"], 3),
                "confidence": 0.5 if len(slices) >= 2 else 0.3,
                "note": "candidate only; body core, away from endplates and cortex",
            }
        )
    return out


# ----------------------------------------------------------------------- analysis entry point
@register("endplates")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Modic-type suggestions, marrow extent and focal lesion candidates on sagittal images."""
    cfg = config or DEFAULT
    res = Result(
        name="endplates", meta={"method": "disc-frame endplate depth maps; mid-body normalised ratios"}
    )
    stacks = sagittal_stacks(study)
    if not stacks:
        res.warnings.append("no sagittal stack in study: endplate analysis skipped")
        return res
    ref_name = next(c for c in DISC_CONTRAST_ORDER if c in stacks)
    ref = stacks[ref_name]
    if ref.plane != "sag":
        res.warnings.append("reference stack is not sagittal: endplate analysis skipped")
        return res
    seq_stacks = {c: stacks[c] for c in SEQS if c in stacks}
    have = set(seq_stacks)
    res.meta.update(reference=ref_name, sequences=sorted(have), series_key=ref.key)
    if "t1" not in have:
        res.warnings.append("no sagittal T1: Modic type cannot be assigned from T2/STIR alone")
    if "stir" not in have:
        res.warnings.append("no sagittal STIR: Modic typing relies on T1/T2 only (lower confidence)")
    if "t2" not in have:
        res.warnings.append("no sagittal T2: T2 ratios unavailable")
    if not have:
        res.warnings.append("no T1, T2 or STIR sagittal stack: endplate signal analysis skipped")
        return res
    labels = anchors.disc_labels()
    if not labels:
        res.warnings.append("no discs in anchors: endplate analysis skipped")
        return res

    upper_of = {lab.split("-")[1]: lab for lab in labels}  # body -> disc above it
    lower_of = {lab.split("-")[0]: lab for lab in labels}  # body -> disc below it
    body_names = sorted({*upper_of, *lower_of}, key=_body_rank)
    midline = anchors.midline_x
    if midline is None:
        midline = float(np.median([anchors.discs[lab].center[0] for lab in labels]))
        res.warnings.append("midline not in anchors: median disc centre x used")
    half_spacing = cfg.side_central_spacing_fraction * (ref.spacing or ref.thickness or 1.0)

    # disc traces on every reference slice; nominal body height from measured bodies
    floors = {c: background(st) for c, st in seq_stacks.items()}
    ref_floor = background(ref)
    traces_by_k = {
        k: {lab: trace_disc(ref, k, lab, anchors.discs[lab], cfg, floor=ref_floor) for lab in labels}
        for k in range(ref.n)
    }
    heights = [
        _body_height(tr[up], tr[lo])
        for tr in traces_by_k.values()
        for up, lo in itertools.pairwise(labels)
        if up.split("-")[1] == lo.split("-")[0]
    ]
    heights = [h for h in heights if np.isfinite(h)]
    h_nominal = float(np.median(heights)) if heights else float(sum(cfg.body_height_mm) / 2)
    if not heights:
        res.warnings.append("no body with both endplates traced: nominal body height used for missing planes")

    rr, cc = np.mgrid[0 : ref.vol.shape[1], 0 : ref.vol.shape[2]].astype(np.float64)
    smooth_px = cfg.ratio_smooth_mm / ref.row_mm
    obs: dict[tuple[str, str], list[_Obs]] = {}
    body_geo: dict[tuple[str, int], _Body] = {}
    lesion_inputs: list[tuple[int, str, _Body, dict[str, NDArray[np.float64]]]] = []
    defects: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for k in range(ref.n):
        tr_k = traces_by_k[k]
        pts = ref.ipp[k] + cc[..., None] * ref.col_mm * ref.row_dir + rr[..., None] * ref.row_mm * ref.col_dir
        smoothed: dict[str, NDArray[np.float64]] = {}
        for c, st in seq_stacks.items():
            img = _resample_to(ref, k, st) - floors[c]
            fill = np.nanmedian(img) if np.isfinite(img).any() else 0.0
            smoothed[c] = ndi.gaussian_filter(np.where(np.isfinite(img), img, fill), sigma=smooth_px)
        for V in body_names:
            upper = tr_k.get(upper_of.get(V, ""))
            lower = tr_k.get(lower_of.get(V, ""))
            bg = _body_geometry(ref, k, V, upper, lower, h_nominal, pts, cfg)
            if bg is None:
                continue
            real = [t for t in (upper, lower) if t is not None]
            bg.intersects = any(t.valid_fraction() >= 0.5 for t in real) and bool(bg.mid.any())
            if not bg.intersects:
                continue
            mid_vals = {c: float(np.median(img[bg.mid])) for c, img in smoothed.items()}
            if any(not np.isfinite(v) or v == 0 for v in mid_vals.values()):
                continue
            body_geo[(V, k)] = bg
            side = _side_of(bg.x_mm, midline, half_spacing)
            ratio = {c: smoothed[c] / mid_vals[c] for c in smoothed}
            shape = ref.vol.shape[1:]
            abn = _abnormal_pixels(
                ratio.get("t1", np.full(shape, np.nan)),
                ratio.get("t2", np.full(shape, np.nan)),
                ratio.get("stir", np.full(shape, np.nan)),
                cfg,
            )
            for ep, band, depth, traced in (
                ("sup", bg.band_sup, bg.depth_sup, bg.sup_traced),
                ("inf", bg.band_inf, bg.depth_inf, bg.inf_traced),
            ):
                if not traced or not band.any():
                    continue  # nominal planes are not assessed
                tr_ep = upper if ep == "sup" else lower
                if tr_ep is not None:
                    defects.setdefault((V, ep), []).append(
                        _defect_profile(tr_ep, "bot" if ep == "sup" else "top", cfg.endplate_defect_window_mm)
                    )
                # depth statistics: marrow between the band and mid-height, away from the cortex at both ends
                near = (
                    bg.body
                    & (depth >= cfg.endplate_band_mm[0])
                    & (depth <= 0.5 * (bg.depth_sup + bg.depth_inf))
                )
                ab_body = abn & near
                obs.setdefault((V, ep), []).append(
                    _Obs(
                        k=k,
                        x_mm=bg.x_mm,
                        side=side,
                        band={c: img[band] for c, img in smoothed.items()},
                        ap_band=bg.ap_mm[band],
                        mid=mid_vals,
                        abn=abn[band],
                        depth_abn=depth[ab_body] if ab_body.any() else np.zeros(0),
                    )
                )
            lesion_inputs.append((k, V, bg, smoothed))

    summaries = {key: _summarise(ob, cfg) for key, ob in obs.items()}
    # healthiest reference: bodies whose observed endplates are normal on the whole endplate
    seen_bodies = {V for V, _ in obs}
    healthy = {
        V
        for V in seen_bodies
        if all(_is_normal(summaries[(V, ep)]) for ep in ("sup", "inf") if (V, ep) in summaries)
    }
    if not healthy:
        res.warnings.append("no body with normal endplates: reference uses all bodies")
        healthy = seen_bodies
    healthy_ref = _healthy_reference(obs, healthy)
    res.meta["healthy_bodies"] = sorted(healthy, key=_body_rank)

    # rows and flags
    for V in body_names:
        for ep in ("sup", "inf"):
            if (V, ep) not in summaries:
                continue
            disc_lab = (upper_of.get(V) if ep == "sup" else lower_of.get(V)) or ""
            for side in ("all", *SIDES):
                s = summaries[(V, ep)].get(side)
                if s is None:
                    continue
                endplate = "superior" if ep == "sup" else "inferior"
                res.rows.append(
                    {
                        "scope": "endplate",
                        "vertebra": V,
                        "endplate": endplate,
                        "adjacent_disc": disc_lab,
                        "side": side,
                        "n_slices": s["n_slices"],
                        "slices_k": s["slices"],
                        "x_mm": s["x_mm"],
                        "T1_ratio": s["T1"],
                        "T2_ratio": s["T2"],
                        "STIR_ratio": s["STIR"],
                        "concavity_mm": _defect_median(defects.get((V, ep), []), 0)
                        if side == "all"
                        else float("nan"),
                        "irregularity_mm": _defect_median(defects.get((V, ep), []), 1)
                        if side == "all"
                        else float("nan"),
                        "T1_vs_healthy": _vs(s["T1_raw"], healthy_ref.get("t1")),
                        "T2_vs_healthy": _vs(s["T2_raw"], healthy_ref.get("t2")),
                        "STIR_vs_healthy": _vs(s["STIR_raw"], healthy_ref.get("stir")),
                        "pattern": s["pattern"],
                        "confidence": s["confidence"],
                        "extent_pct": s["extent_pct"],
                        "width_pct": s["width_pct"],
                        "depth_mm": s["depth_mm"],
                        "nominal_plane": False,
                        "note": s["note"],
                    }
                )
                if side != "all" and s["pattern"] in PRESENCE_PATTERNS:
                    res.flags.append(
                        {
                            "kind": "marrow_change_presence",
                            "vertebra": V,
                            "endplate": endplate,
                            "adjacent_disc": disc_lab,
                            "side": side,
                            "pattern": s["pattern"],
                            "confidence": s["confidence"],
                            "extent_pct": s["extent_pct"],
                            "width_pct": s["width_pct"],
                            "depth_mm": s["depth_mm"],
                            "note": "marrow change seen, Modic type not assessed (a sequence is missing)",
                        }
                    )
                if side != "all" and s["pattern"] in MODIC_PATTERNS:
                    res.flags.append(
                        {
                            "kind": "modic_suggestion",
                            "vertebra": V,
                            "endplate": endplate,
                            "adjacent_disc": disc_lab,
                            "side": side,
                            "pattern": s["pattern"],
                            "confidence": s["confidence"],
                            "extent_pct": s["extent_pct"],
                            "width_pct": s["width_pct"],
                            "depth_mm": s["depth_mm"],
                            "note": "suggestion for reader confirmation",
                        }
                    )

    for lr in _lesions(lesion_inputs, ref, cfg, have):
        res.rows.append(lr)
        res.flags.append(
            {"kind": "marrow_lesion_candidate", **{k2: v2 for k2, v2 in lr.items() if k2 != "scope"}}
        )

    # triptychs (T1 | T2 | STIR) for each flagged endplate, at the slice with the most abnormal marrow
    flagged = sorted({(f["vertebra"], f["endplate"]) for f in res.flags if f["kind"] == "modic_suggestion"})
    for V, endplate in flagged:
        ep = "sup" if endplate == "superior" else "inf"
        ob = obs.get((V, ep), [])
        if not ob:
            continue
        best = max(ob, key=lambda o: float(o.abn.sum()))
        bg = body_geo[(V, best.k)]
        tr_k = traces_by_k[best.k]
        curves: list[tuple[DiscTrace, str]] = []
        if upper_of.get(V) in tr_k:
            curves.append((tr_k[upper_of[V]], "bot"))
        if lower_of.get(V) in tr_k:
            curves.append((tr_k[lower_of[V]], "top"))
        seqs = {c.upper(): _resample_to(ref, best.k, seq_stacks[c]) for c in SEQS if c in seq_stacks}
        masks = [
            (bg.band_sup, (255, 220, 0), 0.35),
            (bg.band_inf, (0, 220, 255), 0.35),
            (bg.mid, (0, 255, 0), 0.2),
        ]
        title = f"{V} {endplate} k={best.k} x={best.x_mm:+.0f}mm"
        res.figures[f"triptych_{V}_{endplate}"] = _triptych(ref, seqs, best.k, bg, masks, curves, title)
    return res


def _defect_median(vals: list[tuple[float, float]], idx: int) -> float:
    """Median over slices of one endplate defect measure (index 0 concavity, 1 irregularity); NaN if none."""
    xs = [v[idx] for v in vals if np.isfinite(v[idx])]
    return round(float(np.median(xs)), 2) if xs else float("nan")


def _body_rank(v: str) -> int:
    order = ("T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "L6", "S1", "S2")
    return order.index(v) if v in order else len(order)
