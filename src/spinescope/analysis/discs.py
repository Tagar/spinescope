"""Intervertebral discs on sagittal MRI: heights, nucleus signal, HIZ candidates, Pfirrmann suggestion.

Method, per disc and sagittal slice (no fixed slice numbers, no per-study constants):

1. Disc frame. The anchored centre, superior normal and anterior/posterior margins are projected onto the
   slice plane. ``u`` is the anterior->posterior axis, ``n`` the superior normal, and AP coordinates
   ``s`` and normal heights ``h`` are in mm.
2. Slice selection. Sagittal slices whose plane lies within a fraction of the disc half width of the
   disc centre (the mid-portion) are analysed. Other slices are not reported.
3. Endplates. The image is resampled on the (s, h) grid. Above the disc the superior endplate is the
   darkest dark-line trace found by dynamic programming (smooth across AP columns). The endplate
   position is then refined to the disc-side half-maximum crossing of that dark line, so the boundary
   lies on the disc side of the cortex. Using the strongest gradient instead places it about a
   millimetre beyond the disc on each side, which is the bias the prototype showed. Columns where the
   dark line is not clearly darker than the disc (relative contrast below ``disc_endplate_min_contrast``)
   are dropped.
4. Heights are anterior, middle and posterior thirds of the traced AP extent, in mm along the normal.
5. Nucleus = central box, annulus = rest of the disc minus a rim at each endplate. Signal is
   normalised to the thecal sac (CSF) found posterior to the disc and to the brightest disc on the same
   slice. HIZ candidates are posterior-annulus foci brighter than the nucleus, not next to CSF.
6. Pfirrmann suggestion: a uncalibrated grade from four relative features, reported with confidence.

Only T2 (then STIR, PD, T1) is used for signal. T1 gives heights only.

Output rows, with ``scope = "slice"`` for each disc and mid-portion slice, and ``scope = "disc"``
summaries (median over slices, with the Pfirrmann suggestion).
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..anatomy.anchors import Anchors, Disc
from ..config import DEFAULT, Config
from ..geometry import FloatArray, Stack, unit
from ..study import Study
from .base import Result, register, sagittal_stacks

log = logging.getLogger(__name__)

DISC_CONTRAST_ORDER = ("t2", "stir", "pd", "t1")
NUCLEUS_CONTRASTS = ("t2", "stir", "pd")

#: Figure rendering: disc crops span this many mm (AP, normal) and are drawn at this many px per mm.
FIG_S_MM = 26.0
FIG_H_MM = 14.0
FIG_PX_PER_MM = 2.0


# ----------------------------------------------------------------------- geometry helpers
def _proj(stack: Stack, k: int, p: Any) -> FloatArray:
    """Project an LPS point onto the plane of slice ``k``."""
    v = np.asarray(p, dtype=np.float64)
    return v - ((v - stack.ipp[k]) @ stack.normal) * stack.normal


def _inplane_unit(v: Any, normal: FloatArray) -> FloatArray:
    """Unit vector of ``v`` projected into the plane with normal ``normal``."""
    a = np.asarray(v, dtype=np.float64)
    a = a - (a @ normal) * normal
    return unit(a)


def _sample_grid(
    stack: Stack,
    k: int,
    origin: FloatArray,
    u: FloatArray,
    n: FloatArray,
    s_grid: NDArray[np.float64],
    h_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bilinear samples of slice ``k`` on ``origin + s*u + h*n``; shape (len(h), len(s)); NaN outside."""
    S, H = np.meshgrid(s_grid, h_grid)
    pts = origin + S[..., None] * u + H[..., None] * n
    rel = pts - stack.ipp[k]
    r = rel @ stack.col_dir / stack.row_mm
    c = rel @ stack.row_dir / stack.col_mm
    rows, cols = stack.vol.shape[1:]
    inside = (r >= -0.5) & (r <= rows - 0.5) & (c >= -0.5) & (c <= cols - 0.5)
    vals = ndi.map_coordinates(stack.vol[k].astype(np.float64), [r, c], order=1, mode="nearest")
    return np.where(inside, vals, np.nan)


# ----------------------------------------------------------------------- tracing
@dataclass
class DiscTrace:
    """One disc traced on one sagittal slice, in its own (s, h) frame."""

    label: str
    k: int
    plane_mm: float  # position of the slice plane along the stack normal (mm)
    origin: FloatArray  # disc centre projected on the slice plane (LPS)
    u: FloatArray  # anterior -> posterior, in plane
    n: FloatArray  # superior normal, in plane
    s_ant: float
    s_post: float
    ap_from_anchors: bool
    s: NDArray[np.float64]  # AP grid (mm, from the centre, + posterior)
    h: NDArray[np.float64]  # normal grid (mm, + superior)
    img: NDArray[np.float64]  # (nh, ns) samples, NaN outside the image
    cols: NDArray[np.bool_]  # AP columns between the insets
    top: NDArray[np.float64]  # superior endplate height per column (NaN if not traced)
    bot: NDArray[np.float64]  # inferior endplate height per column (NaN if not traced)
    contrast: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))  # dark-line contrast

    @property
    def ok(self) -> NDArray[np.bool_]:
        """Columns with both endplates traced."""
        return self.cols & np.isfinite(self.top) & np.isfinite(self.bot)

    @property
    def ap_mm(self) -> float:
        return float(self.s_post - self.s_ant)

    def valid_fraction(self) -> float:
        n = int(self.cols.sum())
        return float(self.ok.sum()) / n if n else 0.0


def _dp(cost: NDArray[np.float64], h: NDArray[np.float64], lam: float) -> NDArray[np.intp]:
    """Minimum-cost smooth path through ``cost[h_index, column]``; returns the h index per column."""
    nh, nc = cost.shape
    pen = lam * np.abs(h[:, None] - h[None, :])
    acc = cost[:, 0].copy()
    back = np.zeros((nh, nc), dtype=np.intp)
    idx = np.arange(nh)
    for j in range(1, nc):
        tot = acc[None, :] + pen
        back[:, j] = np.argmin(tot, axis=1)
        acc = tot[idx, back[:, j]] + cost[:, j]
    path = np.zeros(nc, dtype=np.intp)
    path[-1] = int(np.argmin(acc))
    for j in range(nc - 1, 0, -1):
        path[j - 1] = back[path[j], j]
    return path


def _cross(
    col: NDArray[np.float64], h: NDArray[np.float64], i0: int, step: int, thr: float, reach: int
) -> float:
    """Height where the profile, walking from index ``i0`` by ``step``, first reaches ``thr``."""
    prev = i0
    for m in range(1, reach + 1):
        i = i0 + step * m
        if not (0 <= i < len(h)) or not np.isfinite(col[i]):
            return float("nan")
        if col[i] >= thr:
            v0, v1 = float(col[prev]), float(col[i])
            frac = 0.0 if v1 == v0 else (thr - v0) / (v1 - v0)
            return float(h[prev] + frac * (h[i] - h[prev]))
        prev = i
    return float("nan")


def _refine_edge(
    col: NDArray[np.float64], h: NDArray[np.float64], i0: int, side: int, cfg: Config
) -> tuple[float, float]:
    """Centre of the dark endplate line near index ``i0``, from its half-maximum edges.

    The darkest sample within 1 mm of ``i0`` is the core of the line. Each side is walked outward until
    the profile reaches halfway to the local plateau (median over ``endplate_plateau_mm``) on that side.
    The endplate is the midpoint of the two crossings, so it sits at the centre of the dark line and not
    at its outer margin. ``side`` is +1 for the superior endplate and -1 for the inferior one.
    Returns ``(h_edge, relative_contrast)``; ``h_edge`` is NaN when a crossing is not found.
    """
    grid = h[1] - h[0]
    w = max(1, round(1.0 / grid))
    lo, hi = max(0, i0 - w), min(len(h), i0 + w + 1)
    seg = col[lo:hi]
    if not np.isfinite(seg).any():
        return float("nan"), 0.0
    iv = lo + int(np.nanargmin(seg))
    i_v = float(col[iv])
    reach = max(1, round(cfg.endplate_plateau_mm / grid))
    toward = -side  # index step from the endplate toward the disc
    idx_d = [iv + toward * m for m in range(1, reach + 1) if 0 <= iv + toward * m < len(h)]
    idx_b = [iv - toward * m for m in range(1, reach + 1) if 0 <= iv - toward * m < len(h)]
    vd = col[idx_d] if idx_d else np.array([])
    vb = col[idx_b] if idx_b else np.array([])
    if vd.size == 0 or vb.size == 0 or not np.isfinite(vd).any() or not np.isfinite(vb).any():
        return float("nan"), 0.0
    p_d, p_b = float(np.nanmedian(vd)), float(np.nanmedian(vb))
    if p_d <= 0 or p_b <= 0:
        return float("nan"), 0.0
    contrast = (min(p_d, p_b) - i_v) / min(p_d, p_b)
    h_d = _cross(col, h, iv, toward, 0.5 * (i_v + p_d), reach)
    h_b = _cross(col, h, iv, -toward, 0.5 * (i_v + p_b), reach)
    if not (np.isfinite(h_d) and np.isfinite(h_b)):
        return float("nan"), contrast
    return 0.5 * (h_d + h_b), contrast


def _trace_side(
    sm: NDArray[np.float64],
    h: NDArray[np.float64],
    cols_idx: NDArray[np.intp],
    side: int,
    scale: float,
    cfg: Config,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Endplate height and dark-line contrast per AP column for one side of the disc."""
    search = (side * h >= cfg.endplate_min_h_mm) & (side * h <= cfg.disc_endplate_search_mm)
    sel = np.nonzero(search)[0]
    edge = np.full(len(cols_idx), np.nan)
    contrast = np.zeros(len(cols_idx))
    if sel.size < 3 or cols_idx.size < 3:
        return edge, contrast
    cost = sm[sel][:, cols_idx] / max(scale, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 10.0)
    path = _dp(cost, h[sel], cfg.endplate_smooth)
    for j, c in enumerate(cols_idx):
        e, cc = _refine_edge(sm[:, c], h, int(sel[path[j]]), side, cfg)
        edge[j], contrast[j] = e, cc
    return edge, contrast


def background(stack: Stack) -> float:
    """Background floor of a stack (0.5th percentile of the volume), subtracted before measuring.

    Magnitude MR images can be signed or offset (SPIDER volumes contain negative values), so relative
    contrast and signal ratios are only meaningful above a common floor. Air in the phantom is zero.
    """
    return float(np.nanpercentile(stack.vol, 0.5))


def trace_disc(
    stack: Stack, k: int, label: str, disc: Disc, cfg: Config = DEFAULT, floor: float | None = None
) -> DiscTrace:
    """Build the disc frame on slice ``k`` and trace both endplates. Requires a sagittal stack.

    ``floor`` is subtracted from the samples; pass :func:`background` of the stack once per study
    (computed on demand if omitted).
    """
    N = stack.normal
    origin = _proj(stack, k, disc.center)
    n = _inplane_unit(disc.normal, N)
    have_ap = disc.anterior is not None and disc.posterior is not None
    if have_ap:
        a = _proj(stack, k, disc.anterior)
        p = _proj(stack, k, disc.posterior)
        u = _inplane_unit(p - a, N)
    else:
        u = _inplane_unit(disc.ap_dir, N)
    u = unit(u - (u @ n) * n)
    if have_ap:
        s_ant, s_post = float((a - origin) @ u), float((p - origin) @ u)
    else:
        s_ant, s_post = -cfg.disc_ap_default_mm / 2, cfg.disc_ap_default_mm / 2
    if s_post <= s_ant:
        s_ant, s_post = -abs(s_ant), abs(s_post)

    grid = cfg.disc_grid_mm
    s_grid = np.arange(s_ant - 4.0, s_post + cfg.csf_search_mm[1] + 1.0 + grid / 2, grid)
    H = cfg.disc_endplate_search_mm + 1.0
    h_grid = np.arange(-H, H + grid / 2, grid)
    img = _sample_grid(stack, k, origin, u, n, s_grid, h_grid)
    img = img - (background(stack) if floor is None else floor)
    filled = np.where(np.isfinite(img), img, np.nanmedian(img) if np.isfinite(img).any() else 0.0)
    sig = 0.5 / grid  # 0.5 mm smoothing; the dark line is about 1 mm thick
    sm = ndi.gaussian_filter(filled, sigma=sig)

    cols = (s_grid >= s_ant + cfg.disc_inset_mm) & (s_grid <= s_post - cfg.disc_inset_mm)
    cols_idx = np.nonzero(cols)[0]
    scale = float(np.nanpercentile(sm[:, cols_idx], 95)) if cols_idx.size else 1.0
    top_idx, bot_idx = np.full(len(s_grid), np.nan), np.full(len(s_grid), np.nan)
    top_c, bot_c = np.zeros(len(s_grid)), np.zeros(len(s_grid))
    if cols_idx.size:
        te, tc = _trace_side(sm, h_grid, cols_idx, +1, scale, cfg)
        be, bc = _trace_side(sm, h_grid, cols_idx, -1, scale, cfg)
        ok_t = tc >= cfg.disc_endplate_min_contrast
        ok_b = bc >= cfg.disc_endplate_min_contrast
        top_idx[cols_idx] = np.where(ok_t, te, np.nan)
        bot_idx[cols_idx] = np.where(ok_b, be, np.nan)
        top_c[cols_idx], bot_c[cols_idx] = tc, bc
    ok_both = np.isfinite(top_idx) & np.isfinite(bot_idx)
    top_idx = np.where(ok_both, top_idx, np.nan)
    bot_idx = np.where(ok_both, bot_idx, np.nan)
    return DiscTrace(
        label=label,
        k=k,
        plane_mm=float(stack.ipp[k] @ N),
        origin=origin,
        u=u,
        n=n,
        s_ant=s_ant,
        s_post=s_post,
        ap_from_anchors=have_ap,
        s=s_grid,
        h=h_grid,
        img=img,
        cols=cols,
        top=top_idx,
        bot=bot_idx,
        contrast=np.minimum(top_c, bot_c),
    )


def _lps(tr: DiscTrace, s: NDArray[np.float64], h: NDArray[np.float64]) -> NDArray[np.float64]:
    return tr.origin + np.asarray(s)[..., None] * tr.u + np.asarray(h)[..., None] * tr.n


def _body_height(up: DiscTrace, lo: DiscTrace) -> float:
    """Vertebral body height (mm, along ``up.n``) between the bottom of ``up`` and the top of ``lo``."""
    okU = up.ok
    okL = lo.ok
    if okU.sum() < 2 or okL.sum() < 2:
        return float("nan")
    E = _lps(up, up.s[okU], up.bot[okU])
    sd = (E - lo.origin) @ lo.u
    inside = (sd >= np.min(lo.s[okL])) & (sd <= np.max(lo.s[okL]))
    topd = np.interp(sd, lo.s[okL], lo.top[okL])
    Ei = lo.origin + sd[:, None] * lo.u + topd[:, None] * lo.n
    hh = (E - Ei) @ up.n
    hh = hh[inside]
    return float(np.mean(hh)) if hh.size else float("nan")


def _thirds(tr: DiscTrace, values: NDArray[np.float64]) -> dict[str, float]:
    """Mean of ``values`` (per column) over the anterior, middle and posterior thirds of the AP length."""
    L = tr.s_post - tr.s_ant
    ok = tr.ok
    out: dict[str, float] = {}
    bounds = {
        "ant": (tr.s_ant, tr.s_ant + L / 3),
        "mid": (tr.s_ant + L / 3, tr.s_ant + 2 * L / 3),
        "post": (tr.s_ant + 2 * L / 3, tr.s_post + 1e-9),
    }
    for name, (lo, hi) in bounds.items():
        sel = ok & (tr.s >= lo) & (tr.s < hi)
        out[name] = float(np.mean(values[sel])) if sel.any() else float("nan")
    return out


# ----------------------------------------------------------------------- per-slice measures
def _disc_masks(tr: DiscTrace, cfg: Config) -> dict[str, NDArray[np.bool_]]:
    S, H = np.meshgrid(tr.s, tr.h)
    with np.errstate(invalid="ignore"):
        top = tr.top[None, :]
        bot = tr.bot[None, :]
        inside = (top >= H) & (bot <= H) & tr.ok[None, :]
        L = tr.s_post - tr.s_ant
        mid_s = 0.5 * (tr.s_ant + tr.s_post)
        hmid = 0.5 * (top + bot)
        hh = top - bot
        # central box (half axes: fraction of the AP length and of the local height); a box rather than an
        # ellipse, because an ellipse under-covers the central nucleus and leaks bright voxels into the annulus
        box = (np.abs(S - mid_s) <= cfg.nucleus_ap_fraction * L) & (
            np.abs(H - hmid) <= cfg.nucleus_h_fraction * hh
        )
        nucleus = box & inside
        rim = (np.abs(H - top) < cfg.endplate_rim_mm) | (np.abs(H - bot) < cfg.endplate_rim_mm)
        annulus = inside & ~nucleus & ~rim
        post = annulus & (tr.s_ant + cfg.posterior_annulus_fraction * L <= S)
    return {"disc": inside, "nucleus": nucleus, "annulus": annulus, "post_annulus": post}


def _csf_reference(tr: DiscTrace, cfg: Config) -> tuple[float, NDArray[np.bool_] | None, FloatArray | None]:
    """Thecal sac: brightest smoothed point posterior to the disc. Returns (value, bright mask, LPS)."""
    S, H = np.meshgrid(tr.s, tr.h)
    lo, hi = cfg.csf_search_mm
    region = (tr.s_post + lo <= S) & (tr.s_post + hi >= S) & (np.abs(H) <= cfg.csf_half_height_mm)
    if not region.any():
        return float("nan"), None, None
    filled = np.where(np.isfinite(tr.img), tr.img, 0.0)
    sm = ndi.gaussian_filter(filled, sigma=1.0 / cfg.disc_grid_mm)
    if not np.isfinite(tr.img[region]).any():
        return float("nan"), None, None
    pk = int(np.argmax(np.where(region, sm, -np.inf)))
    ih, is_ = np.unravel_index(pk, tr.img.shape)
    near = (S - S[ih, is_]) ** 2 + (H - H[ih, is_]) ** 2 <= 1.5**2
    vals = tr.img[near & np.isfinite(tr.img)]
    if vals.size == 0:
        return float("nan"), None, None
    csf = float(np.median(vals))
    xyz = _lps(tr, np.array([S[ih, is_]]), np.array([H[ih, is_]]))[0]
    bright = sm >= 0.8 * csf
    return csf, bright, xyz


def _measure_disc(tr: DiscTrace, stack: Stack, cfg: Config, nucleus_ok: bool) -> dict[str, Any]:
    """Heights and signal for one traced disc. Signal fields are NaN when ``nucleus_ok`` is False."""
    h_thirds = _thirds(tr, tr.top - tr.bot)
    thirds = [h_thirds["ant"], h_thirds["mid"], h_thirds["post"]]
    mean_h = float(np.mean(thirds)) if all(np.isfinite(thirds)) else float("nan")
    rec: dict[str, Any] = {
        "ap_mm": round(tr.ap_mm, 2),
        "ant_h_mm": h_thirds["ant"],
        "mid_h_mm": h_thirds["mid"],
        "post_h_mm": h_thirds["post"],
        "mean_h_mm": mean_h,
        "valid_fraction": tr.valid_fraction(),
        "ap_from_anchors": tr.ap_from_anchors,
        "hiz": [],
    }
    if not nucleus_ok or not np.isfinite(tr.img).any() or not tr.ok.any():
        return rec
    masks = _disc_masks(tr, cfg)
    img = tr.img
    nuc_v = img[masks["nucleus"]]
    ann_v = img[masks["annulus"]]
    post_v = img[masks["post_annulus"]]
    nuc_v = nuc_v[np.isfinite(nuc_v)]
    ann_v = ann_v[np.isfinite(ann_v)]
    post_v = post_v[np.isfinite(post_v)]
    nuc_mean = float(nuc_v.mean()) if nuc_v.size else float("nan")
    ann_mean = float(ann_v.mean()) if ann_v.size else float("nan")
    csf, bright, _csf_xyz = _csf_reference(tr, cfg)
    if np.isfinite(csf) and np.isfinite(nuc_mean) and csf < nuc_mean:
        csf = float("nan")  # the brightest point posterior to the disc is not a sac
    rec.update(
        nuc_mean=nuc_mean,
        nuc_cv=float(nuc_v.std() / nuc_mean) if nuc_v.size and nuc_mean else float("nan"),
        ann_mean=ann_mean,
        post_ann_mean=float(post_v.mean()) if post_v.size else float("nan"),
        csf_mean=csf,
        nuc_over_csf=(nuc_mean / csf) if np.isfinite(csf) and csf > 0 else float("nan"),
        nuc_ann_contrast=((nuc_mean - ann_mean) / (nuc_mean + ann_mean))
        if np.isfinite(nuc_mean) and np.isfinite(ann_mean) and nuc_mean + ann_mean > 0
        else float("nan"),
        nuc_over_ann=(nuc_mean / ann_mean) if np.isfinite(ann_mean) and ann_mean > 0 else float("nan"),
    )
    rec["hiz"] = _hiz_candidates(tr, stack, masks, nuc_mean, bright, cfg)
    return rec


def _hiz_candidates(
    tr: DiscTrace,
    stack: Stack,
    masks: dict[str, NDArray[np.bool_]],
    nuc_mean: float,
    csf_bright: NDArray[np.bool_] | None,
    cfg: Config,
) -> list[dict[str, Any]]:
    """Posterior-annulus foci brighter than ``hiz_nucleus_ratio`` x nucleus, away from CSF partial volume."""
    if not np.isfinite(nuc_mean) or nuc_mean <= 0 or not masks["post_annulus"].any():
        return []
    img = np.where(np.isfinite(tr.img), tr.img, 0.0)
    sm = ndi.gaussian_filter(img, sigma=0.5 / cfg.disc_grid_mm)
    region = masks["post_annulus"] & (sm >= cfg.hiz_nucleus_ratio * nuc_mean)
    if csf_bright is not None:
        iters = max(1, round(cfg.hiz_csf_exclusion_mm / cfg.disc_grid_mm))
        near_csf = ndi.binary_dilation(csf_bright, iterations=iters)
    else:
        near_csf = np.zeros(region.shape, dtype=bool)
    lab, nlab = ndi.label(region)
    cell = cfg.disc_grid_mm * cfg.disc_grid_mm
    out: list[dict[str, Any]] = []
    S, H = np.meshgrid(tr.s, tr.h)
    ann_mean = float(np.nanmean(tr.img[masks["annulus"]])) if masks["annulus"].any() else float("nan")
    for i in range(1, nlab + 1):
        comp = lab == i
        area = float(comp.sum()) * cell
        lo, hi = cfg.hiz_area_mm2
        if not (lo <= area <= hi):
            continue
        if (comp & near_csf).any():
            continue  # partial volume with the thecal sac
        peak_idx = int(np.argmax(np.where(comp, img, -np.inf)))
        ih, is_ = np.unravel_index(peak_idx, img.shape)
        xyz = _lps(tr, np.array([S[ih, is_]]), np.array([H[ih, is_]]))[0]
        _, r, c = stack.lps_to_pix(xyz)
        peak = float(img[ih, is_])
        out.append(
            {
                "x_mm": round(float(xyz[0]), 2),
                "y_mm": round(float(xyz[1]), 2),
                "z_mm": round(float(xyz[2]), 2),
                "row": round(float(r), 1),
                "col": round(float(c), 1),
                "area_mm2": round(area, 2),
                "peak": round(peak, 1),
                "peak_over_nucleus": round(peak / nuc_mean, 3),
                "peak_over_annulus": round(peak / ann_mean, 3)
                if np.isfinite(ann_mean) and ann_mean
                else float("nan"),
            }
        )
    return out


def _body_heights(traces: dict[str, DiscTrace], order: list[str]) -> dict[str, float]:
    """Height of each body between two consecutive traced discs (keyed by body name)."""
    out: dict[str, float] = {}
    for up_lab, lo_lab in itertools.pairwise(order):
        body = up_lab.split("-")[1]
        if lo_lab.split("-")[0] != body:
            continue  # not adjacent through a shared vertebra
        out[body] = _body_height(traces[up_lab], traces[lo_lab])
    return out


def _pfirrmann(features: dict[str, float], cfg: Config) -> tuple[int, float, dict[str, int]]:
    """Grade 1..5 suggestion from relative features, with agreement-based confidence (capped)."""

    def ladder_high(v: float, t: tuple[float, ...]) -> int:  # larger is better
        for g, th in enumerate(t, start=1):
            if v >= th:
                return g
        return len(t) + 1

    def ladder_low(v: float, t: tuple[float, ...]) -> int:  # smaller is better
        for g, th in enumerate(t, start=1):
            if v <= th:
                return g
        return len(t) + 1

    grades: dict[str, int] = {}
    if np.isfinite(features.get("nuc_over_csf", np.nan)):
        grades["brightness"] = ladder_high(features["nuc_over_csf"], cfg.pfirrmann_brightness)
    if np.isfinite(features.get("nuc_cv", np.nan)):
        grades["homogeneity"] = ladder_low(features["nuc_cv"], cfg.pfirrmann_cv)
    if np.isfinite(features.get("nuc_ann_contrast", np.nan)):
        grades["distinction"] = ladder_high(features["nuc_ann_contrast"], cfg.pfirrmann_contrast)
    if np.isfinite(features.get("height_ratio", np.nan)):
        grades["height"] = ladder_high(features["height_ratio"], cfg.pfirrmann_height)
    if not grades:
        return 0, 0.0, grades
    g = math.floor(float(np.median(list(grades.values()))) + 0.5)
    agree = sum(1 for v in grades.values() if v == g) / len(grades)
    return g, round(0.75 * agree, 2), grades  # 0.75 cap: thresholds are not yet calibrated


def _mid_slices(ref: Stack, origin_n: float, cfg: Config) -> tuple[list[int], bool]:
    """Slices whose plane is within the mid-portion of a disc (|offset| <= fraction x half width)."""
    offsets = ref.ipp @ ref.normal - origin_n
    w = cfg.disc_half_width_mm * cfg.disc_mid_fraction
    ks = [k for k in range(ref.n) if abs(float(offsets[k])) <= w]
    if ks:
        return sorted(ks, key=lambda k: abs(float(offsets[k]))), True
    k = int(np.argmin(np.abs(offsets)))
    if abs(float(offsets[k])) <= cfg.disc_half_width_mm:
        return [k], False
    return [], False


def _median(vals: list[float]) -> float:
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


# ----------------------------------------------------------------------- figures
def _on_grid(tr: DiscTrace, s: NDArray[np.float64], which: str) -> NDArray[np.float64]:
    """An endplate curve of a trace interpolated onto the AP grid ``s`` (NaN outside its traced columns)."""
    ok = tr.ok
    out = np.full(s.shape, np.nan)
    if ok.sum() < 2:
        return out
    grid = tr.s[ok]
    vals = (tr.top if which == "top" else tr.bot)[ok]
    inside = (s >= np.min(grid)) & (s <= np.max(grid))
    out[inside] = np.interp(s[inside], grid, vals)
    return out


def _panel(
    img: NDArray[np.float64],
    top: NDArray[np.float64],
    bot: NDArray[np.float64],
    s: NDArray[np.float64],
    h: NDArray[np.float64],
    window: tuple[float, float],
) -> NDArray[np.uint8]:
    lo, hi = window
    g = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1) * 255
    rgb = np.stack([g, g, g], axis=-1).astype(np.uint8)
    rgb = np.repeat(np.repeat(rgb, int(FIG_PX_PER_MM), axis=0), int(FIG_PX_PER_MM), axis=1)
    px = FIG_PX_PER_MM
    for j in range(len(s)):
        for edge in (top[j], bot[j]):
            if not np.isfinite(edge):
                continue
            ri = round((h[-1] - edge) / (h[1] - h[0]) * px)
            ci = round((s[j] - s[0]) / (s[1] - s[0]) * px)
            if 0 <= ri < rgb.shape[0] and 0 <= ci < rgb.shape[1]:
                rgb[ri, ci] = (255, 40, 40)
    return rgb


def _compose(
    rows: list[list[NDArray[np.uint8]]], row_names: list[str], col_names: list[str]
) -> NDArray[np.uint8]:
    pad, label_h, label_w = 4, 16, 70
    ph, pw = rows[0][0].shape[:2]
    H = label_h + len(rows) * (ph + pad)
    W = label_w + len(col_names) * (pw + pad)
    canvas = Image.new("RGB", (W, H), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    for ci, name in enumerate(col_names):
        draw.text((label_w + ci * (pw + pad) + 2, 2), name, fill=(255, 255, 0))
    for ri, row in enumerate(rows):
        y = label_h + ri * (ph + pad)
        draw.text((2, y + ph // 2), row_names[ri], fill=(255, 255, 0))
        for ci, panel in enumerate(row):
            canvas.paste(Image.fromarray(panel), (label_w + ci * (pw + pad), y))
    return np.asarray(canvas, dtype=np.uint8)


def _strip_figure(
    stacks: dict[str, Stack],
    traces: dict[str, DiscTrace],
    cfg: Config,
) -> NDArray[np.uint8] | None:
    """Disc crops in each disc's frame (reference slice): one row per available contrast."""
    s = np.arange(-FIG_S_MM, FIG_S_MM + cfg.disc_grid_mm / 2, cfg.disc_grid_mm)
    h = np.arange(-FIG_H_MM, FIG_H_MM + cfg.disc_grid_mm / 2, cfg.disc_grid_mm)
    labels = [lab for lab in traces]
    if not labels:
        return None
    contrasts = [c for c in DISC_CONTRAST_ORDER if c in stacks]
    rows: list[list[NDArray[np.uint8]]] = []
    names: list[str] = []
    for c in contrasts:
        st = stacks[c]
        crops: list[NDArray[np.float64]] = []
        for lab in labels:
            tr = traces[lab]
            kk, _ = st.nearest_slice(tr.origin)
            crops.append(_sample_grid(st, kk, tr.origin, tr.u, tr.n, s, h))
        allv = np.concatenate([x[np.isfinite(x)] for x in crops])
        window = (
            (float(np.percentile(allv, 1)), float(np.percentile(allv, 99.5))) if allv.size else (0.0, 1.0)
        )
        rows.append(
            [
                _panel(
                    crops[i], _on_grid(traces[lab], s, "top"), _on_grid(traces[lab], s, "bot"), s, h, window
                )
                for i, lab in enumerate(labels)
            ]
        )
        names.append(c.upper())
    if not rows:
        return None
    return _compose(rows, names, labels)


# ----------------------------------------------------------------------- analysis entry point
@register("discs")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Disc heights, nucleus signal, HIZ candidates and Pfirrmann suggestions on sagittal images."""
    cfg = config or DEFAULT
    res = Result(name="discs", meta={"method": "disc-frame DP dark-line endplates, half-max refinement"})
    stacks = sagittal_stacks(study)
    if not stacks:
        res.warnings.append("no sagittal stack in study: disc analysis skipped")
        return res
    contrast = next(c for c in DISC_CONTRAST_ORDER if c in stacks)
    ref = stacks[contrast]
    res.meta.update(contrast=contrast, sequences=sorted(stacks), series_key=ref.key)
    if contrast == "t1":
        res.warnings.append("no T2, STIR or PD sagittal stack: disc heights only, no nucleus signal (T1)")
    elif "t2" not in stacks:
        res.warnings.append(f"no sagittal T2: disc signal measured on {contrast.upper()} instead")
    if "stir" not in stacks:
        res.warnings.append("no sagittal STIR stack: STIR-based checks are not available")
    if ref.plane != "sag":
        res.warnings.append(f"reference stack plane is {ref.plane}, not sagittal: disc analysis skipped")
        return res
    nucleus_ok = contrast in NUCLEUS_CONTRASTS
    labels = anchors.disc_labels()
    if not labels:
        res.warnings.append("no discs in anchors")
        return res
    for lab in labels:
        a = anchors.discs[lab]
        if a.anterior is None or a.posterior is None:
            res.warnings.append(f"{lab}: no anterior/posterior margin in anchors; AP length assumed")

    # Mid-portion slices per disc, and every disc traced on each needed slice (for neighbours and brightest)
    centre_offset = {lab: float(np.asarray(anchors.discs[lab].center) @ ref.normal) for lab in labels}
    per_disc_slices: dict[str, list[int]] = {}
    for lab in labels:
        ks, in_mid = _mid_slices(ref, centre_offset[lab], cfg)
        if not ks:
            res.warnings.append(f"{lab}: outside the sagittal stack; not measured")
            continue
        if not in_mid:
            res.warnings.append(f"{lab}: no slice in the disc mid-portion; nearest slice used")
        per_disc_slices[lab] = ks
    needed = sorted({k for ks in per_disc_slices.values() for k in ks})
    ref_floor = background(ref)
    records: dict[int, dict[str, dict[str, Any]]] = {}
    traces_by_k: dict[int, dict[str, DiscTrace]] = {}
    for k in needed:
        traces = {lab: trace_disc(ref, k, lab, anchors.discs[lab], cfg, floor=ref_floor) for lab in labels}
        traces_by_k[k] = traces
        recs = {lab: _measure_disc(traces[lab], ref, cfg, nucleus_ok) for lab in labels}
        bodies = _body_heights(traces, labels)
        bright = [recs[lab]["nuc_mean"] for lab in labels if np.isfinite(recs[lab].get("nuc_mean", np.nan))]
        brightest = max(bright) if bright else float("nan")
        for lab in labels:
            r = recs[lab]
            ab = bodies.get(lab.split("-")[0], float("nan"))
            be = bodies.get(lab.split("-")[1], float("nan"))
            r["body_above_h_mm"] = ab
            r["body_below_h_mm"] = be
            bl = [x for x in (ab, be) if np.isfinite(x)]
            r["dhi"] = (
                (r["mean_h_mm"] / float(np.mean(bl))) if bl and np.isfinite(r["mean_h_mm"]) else float("nan")
            )
            r["nuc_over_brightest"] = (
                (r.get("nuc_mean", float("nan")) / brightest)
                if np.isfinite(brightest) and np.isfinite(r.get("nuc_mean", float("nan")))
                else float("nan")
            )
        records[k] = recs
    if not records:
        return res

    # Study-level height reference (median over discs of their median mean height), for height ratio
    slice_rows: list[dict[str, Any]] = []
    per_disc_vals: dict[str, dict[str, list[float]]] = {}
    for lab in labels:
        for k in per_disc_slices.get(lab, []):
            r = records[k][lab]
            tr = traces_by_k[k][lab]
            if tr.valid_fraction() < 0.5 or not np.isfinite(r["mean_h_mm"]):
                res.warnings.append(f"{lab} slice {k}: endplates not traced on enough columns; skipped")
                continue
            row = {
                "scope": "slice",
                "disc": lab,
                "slice_k": k,
                "slice_x_mm": round(float(ref.ipp[k] @ ref.normal), 2),
                "primary": False,
                "contrast": contrast,
                "ap_mm": round(r["ap_mm"], 2),
                "ant_h_mm": round(r["ant_h_mm"], 2),
                "mid_h_mm": round(r["mid_h_mm"], 2),
                "post_h_mm": round(r["post_h_mm"], 2),
                "mean_h_mm": round(r["mean_h_mm"], 2),
                "dhi": round(r["dhi"], 3) if np.isfinite(r["dhi"]) else float("nan"),
                "body_above_h_mm": round(r["body_above_h_mm"], 2)
                if np.isfinite(r["body_above_h_mm"])
                else float("nan"),
                "body_below_h_mm": round(r["body_below_h_mm"], 2)
                if np.isfinite(r["body_below_h_mm"])
                else float("nan"),
                "nuc_mean": r.get("nuc_mean", float("nan")),
                "nuc_cv": r.get("nuc_cv", float("nan")),
                "ann_mean": r.get("ann_mean", float("nan")),
                "post_ann_mean": r.get("post_ann_mean", float("nan")),
                "csf_mean": r.get("csf_mean", float("nan")),
                "nuc_over_csf": r.get("nuc_over_csf", float("nan")),
                "nuc_over_brightest": r.get("nuc_over_brightest", float("nan")),
                "nuc_ann_contrast": r.get("nuc_ann_contrast", float("nan")),
                "nuc_over_ann": r.get("nuc_over_ann", float("nan")),
                "hiz_n": len(r["hiz"]),
                "hiz_max_peak_over_nucleus": max(
                    [c["peak_over_nucleus"] for c in r["hiz"]], default=float("nan")
                ),
                "valid_fraction": round(r["valid_fraction"], 2),
            }
            slice_rows.append(row)
            for key in (
                "mean_h_mm",
                "dhi",
                "nuc_over_csf",
                "nuc_over_brightest",
                "nuc_cv",
                "nuc_ann_contrast",
                "nuc_over_ann",
            ):
                per_disc_vals.setdefault(lab, {}).setdefault(key, []).append(float(row[key]))
            for c in r["hiz"]:
                res.flags.append(
                    {
                        "kind": "hiz_candidate",
                        "level": lab,
                        "slice_k": k,
                        "confidence": 0.4,
                        **{kk: vv for kk, vv in c.items()},
                    }
                )
    if not slice_rows:
        res.warnings.append("no disc could be measured on this study")
        return res

    height_ref = _median([_median(per_disc_vals.get(lab, {}).get("mean_h_mm", [])) for lab in labels])
    primary_k: dict[str, int] = {}
    for lab in labels:
        lab_rows = [r for r in slice_rows if r["disc"] == lab]
        if lab_rows:
            best = min(
                lab_rows, key=lambda r: abs(float(ref.ipp[r["slice_k"]] @ ref.normal) - centre_offset[lab])
            )
            best["primary"] = True
            primary_k[lab] = int(best["slice_k"])
    res.rows.extend(slice_rows)

    # Per-disc summary with a Pfirrmann suggestion
    for lab in labels:
        vals = per_disc_vals.get(lab)
        if not vals:
            continue
        feats = {key: _median(vals.get(key, [])) for key in ("nuc_over_csf", "nuc_cv", "nuc_ann_contrast")}
        feats["nuc_over_csf"] = _median(vals.get("nuc_over_csf", []))
        mh = _median(vals.get("mean_h_mm", []))
        feats["height_ratio"] = (
            mh / height_ref if height_ref and np.isfinite(height_ref) and np.isfinite(mh) else float("nan")
        )
        grade, conf, parts = _pfirrmann(feats, cfg)
        summary = {
            "scope": "disc",
            "disc": lab,
            "n_slices": len(vals.get("mean_h_mm", [])),
            "contrast": contrast,
            "mean_h_mm": round(mh, 2),
            "height_ratio": round(feats["height_ratio"], 3)
            if np.isfinite(feats["height_ratio"])
            else float("nan"),
            "dhi": round(_median(vals.get("dhi", [])), 3),
            "nuc_over_csf": round(feats["nuc_over_csf"], 3)
            if np.isfinite(feats["nuc_over_csf"])
            else float("nan"),
            "nuc_over_brightest": round(_median(vals.get("nuc_over_brightest", [])), 3),
            "nuc_cv": round(feats["nuc_cv"], 3) if np.isfinite(feats["nuc_cv"]) else float("nan"),
            "nuc_ann_contrast": round(feats["nuc_ann_contrast"], 3)
            if np.isfinite(feats["nuc_ann_contrast"])
            else float("nan"),
            "pfirrmann_suggestion": grade if grade else None,
            "pfirrmann_confidence": conf,
            "pfirrmann_parts": parts,
            "suggestion_only": True,
        }
        res.rows.append(summary)
        if grade >= 4:
            res.flags.append(
                {
                    "kind": "pfirrmann_suggestion",
                    "level": lab,
                    "grade": grade,
                    "confidence": conf,
                    "note": "uncalibrated suggestion; a reader must confirm",
                }
            )
        if np.isfinite(feats["height_ratio"]) and feats["height_ratio"] < 0.6:
            res.flags.append(
                {
                    "kind": "low_disc_height",
                    "level": lab,
                    "height_ratio": round(feats["height_ratio"], 3),
                    "confidence": 0.5,
                }
            )

    # Figure: disc crops on the reference slice of the primary slice of the study
    if primary_k:
        mid_off = float(np.median(list(centre_offset.values())))
        k0 = min(traces_by_k, key=lambda k: abs(float(ref.ipp[k] @ ref.normal) - mid_off))
        fig_traces = {lab: tr for lab, tr in traces_by_k[k0].items() if tr.ok.any()}
        fig = _strip_figure(stacks, fig_traces, cfg)
        if fig is not None:
            res.figures["disc_strip"] = fig
    return res
