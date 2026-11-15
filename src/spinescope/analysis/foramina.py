"""Neural foramina per level and side: location, foraminal height, AP width, foraminal fat and perineural fat grade.

Everything is measured on the sagittal T1 stack, parasagittal planes only. No segmentation, no network, no fixed slice
index, no per-study constant (DESIGN.md principle 2). Conventions, in order:

1. Parasagittal plane. For each side, the sagittal slices whose lateral distance from the labeller's midline lies in
   ``foramen_lateral_mm`` are candidates (LPS: patient left = +x). Epidural fat is also bright on T1, so the window keeps
   the candidates clear of the canal. If no slice reaches the window, the side is **not assessed** (narrow coverage).
2. Centre. On each candidate plane the foramen centre is searched on a grid behind the posterior margin of the disc
   (``foramen_ap_search_mm``) and around the disc centre along its normal (``foramen_si_search_mm``). The centre is the
   candidate with the highest mean T1 fat index, fat index = intensity / series 99th percentile (relative, not absolute).
   The plane whose best centre scores highest is the foramen plane. ``score_margin`` is its lead over the median plane.
3. Foraminal height and AP width. Along the disc normal (height) and the disc AP axis (width) through the centre, the
   fat-to-dark transition of each side is placed at the half level between a fat reference (90th percentile) and a dark
   reference (10th percentile) of that profile: the pedicle borders. Height = superior minus inferior border, AP width =
   posterior minus anterior border, both in mm, both only when the centre is fat-bright and the profile contrast is at
   least ``foramen_min_rim_contrast``.
4. Foraminal fat area. Connected fat-bright region (half level of its own window) that contains the centre, in mm^2.
5. Perineural fat. Four rays from the centre (anterior, posterior, superior, inferior) of length equal to the measured
   half width or half height: a ray is **obliterated** when no fat-bright sample lies in its outer part
   (``foramen_ray_min_frac`` onwards). The count of obliterated directions gives the perineural status
   (0 preserved, 1-3 partially obliterated, 4 fully obliterated), the Lee et al. 2010 grade (0 none, 1 an opposing pair,
   2 all four; root morphology change, grade 3, is not measurable here and is not assigned) and the Wildermuth-style
   grade (0 none, 2 partial, 3 complete).

Level names: a foramen is named by the disc it lies behind; the exiting root is the one from the **upper** vertebra
(the L4-L5 foramen carries the L4 root). Anatomy, not a compression claim. The disc name carries the numbering status
of the anchors; rows and flags repeat it.

Outputs: one row per (disc, side) with ``scope = "foramen"``, flags for an opposing or wider obliteration and a height
at or below ``foramen_critical_height_mm``, and one overview figure. Missing inputs return warnings, never exceptions.
Validation (Sudirman reports, SPIDER geometry) is in docs/analysis/foramina.md and tools/eval_foramina.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..anatomy.anchors import Anchors, Disc, split_disc
from ..anatomy.summary import status_of
from ..config import DEFAULT, Config
from ..geometry import Stack, unit
from ..study import Study
from .base import Result, register
from .findings import Candidate, FieldRule, register_field

log = logging.getLogger(__name__)

SIDES: tuple[str, ...] = ("L", "R")  # patient left / right
DIRECTIONS: tuple[str, ...] = ("ant", "post", "sup", "inf")
#: Figure: each cell is this many mm (AP x SI) around the centre, drawn at ``FIG_PX_PER_MM``.
FIG_AP_MM = 50.0
FIG_SI_MM = 40.0
FIG_PX_PER_MM = 3.0
#: Sampling pitch of the rim profiles and rays, as a fraction of the pixel size (no effect on the result beyond interpolation).
_SUBPIXEL = 0.25
#: Fat reference on the fat index: the series 99th percentile is index 1.0 by construction, so fat is taken as 1.0.
#: A fixed reference keeps small foraminal fat regions from being scored against themselves (a 90th-percentile reference would).
FAT_INDEX = 1.0


# ------------------------------------------------------------------------------------------- geometry helpers
def side_sign(side: str) -> float:
    """+1 for patient left (LPS +x), -1 for patient right."""
    if side not in SIDES:
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    return 1.0 if side == "L" else -1.0


def lateral_mm(x: float, midline_x: float, side: str) -> float:
    """Lateral distance of plane x from the midline on ``side`` (positive = on that side)."""
    return side_sign(side) * (float(x) - float(midline_x))


def _pix(stack: Stack, k: int, pts: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(row, col) of LPS points projected onto slice ``k`` (the component along the slice normal is ignored)."""
    rel = np.atleast_2d(pts) - stack.ipp[k]
    c = rel @ stack.row_dir / stack.col_mm
    r = rel @ stack.col_dir / stack.row_mm
    return r, c


def sample(stack: Stack, k: int, pts: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    """Fat index (intensity / ``scale``, bilinear) at LPS points on slice ``k``; NaN outside the field of view."""
    r, c = _pix(stack, k, pts)
    rows, cols = stack.vol.shape[1], stack.vol.shape[2]
    inside = (r >= -0.5) & (r <= rows - 0.5) & (c >= -0.5) & (c <= cols - 0.5)
    v = ndi.map_coordinates(
        np.asarray(stack.vol[k], dtype=np.float64),
        [np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)],
        order=1,
        mode="nearest",
    )
    out = v / max(scale, 1e-9)
    out[~inside] = np.nan
    return out


def _unit_in_plane(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit vector with the slice-normal (x) component removed; sagittal planes have x as normal."""
    v = np.asarray(vec, dtype=np.float64).copy()
    v[0] = 0.0
    return unit(v)


@dataclass(frozen=True)
class DiscFrame:
    """Disc geometry used to place the foramen: centre, posterior margin, superior normal and AP axis (LPS)."""

    centre: NDArray[np.float64]
    posterior: NDArray[np.float64]
    normal: NDArray[np.float64]
    ap: NDArray[np.float64]  # anterior -> posterior, in the disc plane


def disc_frame(disc: Disc) -> DiscFrame | None:
    """Frame from the anchored disc, or None when the posterior margin or the normal is missing."""
    if disc.posterior is None or disc.center is None or disc.normal is None:
        return None
    n = unit(np.asarray(disc.normal, dtype=np.float64))
    ap = _unit_in_plane(disc.ap_dir)
    return DiscFrame(
        centre=np.asarray(disc.center, dtype=np.float64),
        posterior=np.asarray(disc.posterior, dtype=np.float64),
        normal=n,
        ap=ap,
    )


def _grid(lo: float, hi: float, step: float) -> NDArray[np.float64]:
    n = math.floor((hi - lo) / step + 1e-9)
    return lo + step * np.arange(n + 1, dtype=np.float64)


def _sym(half: float, step: float) -> NDArray[np.float64]:
    """Grid symmetric about zero, ``2 * round(half / step) + 1`` points, the centre at index ``len // 2``."""
    n = round(half / step)
    return step * np.arange(-n, n + 1, dtype=np.float64)


# ----------------------------------------------------------------------------------------- profile borders
def _borders(t: NDArray[np.float64], v: NDArray[np.float64], cfg: Config) -> tuple[float, float] | None:
    """Fat-to-dark borders on a profile through the centre (t = 0): the nearest superior/anterior and inferior/posterior
    half-level crossings, as (negative, positive) positions in mm. None when the centre is not fat-bright, the profile
    leaves the field of view, the contrast is too low, or a border is not found inside the profile."""
    if not np.all(np.isfinite(v)):
        return None
    dark = float(np.percentile(v, 10))
    if (FAT_INDEX - dark) / FAT_INDEX < cfg.foramen_min_rim_contrast:
        return None
    half = 0.5 * (FAT_INDEX + dark)
    i0 = int(np.argmin(np.abs(t)))
    if v[i0] < half:
        return None
    below = np.where(v < half)[0]
    up = below[below > i0]
    dn = below[below < i0]
    if up.size == 0 or dn.size == 0:
        return None
    return float(t[dn[-1]]), float(t[up[0]])


# -------------------------------------------------------------------------------------------- the site
@dataclass
class Site:
    """Chosen plane and centre for one (disc, side)."""

    k: int
    centre: NDArray[np.float64]
    ap_off_mm: float  # centre offset from the posterior disc margin along the AP axis (+ = posterior)
    si_off_mm: float  # centre offset along the disc normal
    score: float
    score_margin: float
    lateral_mm: float
    edge: bool  # plane lies within one slice of the lateral window edge


def search_site(
    stack: Stack, frame: DiscFrame, side: str, midline_x: float, scale: float, cfg: Config
) -> Site | None:
    """Best foramen plane and centre for one disc and side, or None when no candidate plane lies in the window."""
    lo, hi = cfg.foramen_lateral_mm
    pitch = min(stack.row_mm, stack.col_mm)
    cands: list[tuple[int, float]] = []
    for k in range(stack.n):
        d = lateral_mm(stack.ipp[k, 0], midline_x, side)
        if lo <= d <= hi:
            cands.append((k, d))
    if not cands:
        return None
    ap_grid = _grid(*cfg.foramen_ap_search_mm, pitch)
    si_grid = _grid(-cfg.foramen_si_search_mm, cfg.foramen_si_search_mm, pitch)
    box = _grid(-cfg.foramen_box_mm, cfg.foramen_box_mm, pitch)
    AA, SS = np.meshgrid(ap_grid, si_grid, indexing="ij")
    base = frame.posterior[None, :] + AA.reshape(-1, 1) * frame.ap + SS.reshape(-1, 1) * frame.normal
    BA, BS = np.meshgrid(box, box, indexing="ij")
    offs = BA.reshape(-1, 1) * frame.ap + BS.reshape(-1, 1) * frame.normal  # (q, 3)
    best: list[tuple[float, int, float, float, NDArray[np.float64]]] = []
    for k, d in cands:
        pts = base.copy()
        pts[:, 0] = stack.ipp[k, 0]
        allp = (pts[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        vals = sample(stack, k, allp, scale).reshape(pts.shape[0], -1)
        with np.errstate(all="ignore"):
            score = np.nanmean(vals, axis=1)
        if not np.any(np.isfinite(score)):
            continue
        score = np.where(np.isfinite(score), score, -np.inf)
        j = int(np.argmax(score))
        best.append((float(score[j]), k, d, float(AA.reshape(-1)[j]), pts[j]))
    if not best:
        return None
    top = max(best, key=lambda b: b[0])
    margin = top[0] - float(np.median([b[0] for b in best]))
    k_best, d_best = top[1], top[2]
    si_pitch = stack.spacing if stack.spacing > 0 else pitch
    edge = (d_best - lo) < si_pitch or (hi - d_best) < si_pitch
    centre = top[4]
    si_off = float(np.dot(centre - frame.posterior, frame.normal))
    return Site(
        k=k_best,
        centre=centre,
        ap_off_mm=float(top[3]),
        si_off_mm=si_off,
        score=top[0],
        score_margin=float(margin),
        lateral_mm=float(d_best),
        edge=bool(edge),
    )


@dataclass
class Measure:
    """Measurements through a site. NaN where not measured; ``reason`` says why."""

    height_mm: float
    ap_width_mm: float
    area_mm2: float
    centre_fat: bool
    ant_mm: float  # AP border positions (mm from the centre; NaN if not found)
    post_mm: float
    sup_mm: float
    inf_mm: float
    obliterated: tuple[str, ...]
    reason: str


def _ray_fat(
    stack: Stack,
    k: int,
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    length: float,
    scale: float,
    thr: float,
    cfg: Config,
) -> bool:
    """True if fat-bright tissue lies in the outer part of the ray (from ``foramen_ray_min_frac * length`` to length)."""
    step = min(stack.row_mm, stack.col_mm) * _SUBPIXEL
    t = _grid(cfg.foramen_ray_min_frac * length, length, step)
    if t.size == 0:
        return False
    pts = origin[None, :] + t[:, None] * direction[None, :]
    v = sample(stack, k, pts, scale)
    with np.errstate(all="ignore"):
        return bool(np.nanmax(v) >= thr) if np.any(np.isfinite(v)) else False


def measure_site(stack: Stack, frame: DiscFrame, site: Site, scale: float, cfg: Config) -> Measure:
    """Height, AP width, foraminal fat area and the four perineural rays through the chosen centre.

    The centre is fat-bright when its box mean (the search criterion) reaches the half level of the window around it.
    Pedicle borders are placed on the profiles through the centre; they can be missing (``rims_found`` False) while the
    centre is still fat-bright, and the rays need only the centre.
    """
    k, F = site.k, site.centre
    pitch = min(stack.row_mm, stack.col_mm)
    H = cfg.foramen_profile_half_mm
    # window half level, from the fat-to-dark spread of the window around the centre
    g = _sym(H, pitch)
    AA, SS = np.meshgrid(g, g, indexing="ij")
    win_pts = F[None, :] + AA.reshape(-1, 1) * frame.ap[None, :] + SS.reshape(-1, 1) * frame.normal[None, :]
    wv = sample(stack, k, win_pts, scale)
    fin = wv[np.isfinite(wv)]
    if fin.size < 0.9 * wv.size:
        return Measure(
            np.nan,
            np.nan,
            np.nan,
            False,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            (),
            "window outside the field of view",
        )
    dark_w = float(np.percentile(fin, 10))
    if (FAT_INDEX - dark_w) / FAT_INDEX < cfg.foramen_min_rim_contrast:
        return Measure(
            np.nan, np.nan, np.nan, False, np.nan, np.nan, np.nan, np.nan, (), "window contrast too low"
        )
    thr = 0.5 * (FAT_INDEX + dark_w)
    box = _sym(cfg.foramen_box_mm, pitch)
    BA, BS = np.meshgrid(box, box, indexing="ij")
    bpts = F[None, :] + BA.reshape(-1, 1) * frame.ap[None, :] + BS.reshape(-1, 1) * frame.normal[None, :]
    with np.errstate(all="ignore"):
        centre_mean = float(np.nanmean(sample(stack, k, bpts, scale)))
    centre_fat = bool(np.isfinite(centre_mean) and centre_mean >= thr)

    # pedicle borders on the profiles through the centre
    step = pitch * _SUBPIXEL
    t = _sym(H, step)
    v_si = sample(stack, k, F[None, :] + t[:, None] * frame.normal[None, :], scale)
    v_ap = sample(stack, k, F[None, :] + t[:, None] * frame.ap[None, :], scale)
    si_b = _borders(t, v_si, cfg)
    ap_b = _borders(t, v_ap, cfg)
    if si_b is not None:
        dn_si, up_si = si_b
    else:
        dn_si = up_si = float("nan")
    if ap_b is not None:
        ant, post = ap_b
    else:
        ant = post = float("nan")
    height = up_si - dn_si if np.isfinite(up_si) else float("nan")
    width = post - ant if np.isfinite(post) else float("nan")

    # foraminal fat area: connected fat-bright region of the window containing the centre, mm^2 (NaN if it opens
    # into the window edge)
    area = 0.0
    if centre_fat:
        mask = (np.nan_to_num(wv, nan=-1.0) >= thr).reshape(AA.shape)
        lab, _ = ndi.label(mask)
        c = len(g) // 2
        if lab[c, c] > 0:
            comp = lab == lab[c, c]
            area = float(comp.sum()) * pitch * pitch
            if comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any():
                area = float("nan")
        else:
            area = float("nan")

    obl: list[str] = []
    if centre_fat:
        half_w = 0.5 * width if np.isfinite(width) else cfg.foramen_ray_default_mm
        half_h = 0.5 * height if np.isfinite(height) else cfg.foramen_ray_default_mm
        rays = {
            "ant": (-frame.ap, half_w),
            "post": (frame.ap, half_w),
            "sup": (frame.normal, half_h),
            "inf": (-frame.normal, half_h),
        }
        for name in DIRECTIONS:
            d, length = rays[name]
            if not _ray_fat(stack, k, F, d, length, scale, thr, cfg):
                obl.append(name)
        reason = "measured" if np.isfinite(height) and np.isfinite(width) else "measured: rims incomplete"
    else:
        reason = "centre not fat-bright: foramen not assessed"
    return Measure(
        height_mm=height,
        ap_width_mm=width,
        area_mm2=area,
        centre_fat=centre_fat,
        ant_mm=ant,
        post_mm=post,
        sup_mm=up_si,
        inf_mm=dn_si,
        obliterated=tuple(obl),
        reason=reason,
    )


# -------------------------------------------------------------------------------------------- grading
def grade(obliterated: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Perineural status, Lee et al. 2010 grade and Wildermuth-style grade from the obliterated directions.

    Lee 2010 (AJR 194:1095-1098): 0 normal; 1 perineural fat obliterated in two opposing directions (superior and
    inferior, or anterior and posterior) with no root change; 2 obliterated in all four directions with no root change;
    3 nerve-root collapse or morphologic change, not assigned here. One direction, or two non-opposing directions, is
    outside the Lee scheme: ``lee_grade`` None. Wildermuth 1998 (Radiology 207:391-398, as reproduced in Lee 2010):
    0 normal, 1 deformed but completely surrounded, 2 partially surrounded, 3 complete obliteration. The ray test cannot
    separate 0 from 1, so this returns 0 for no obliteration and 2 or 3 otherwise.
    """
    o = set(obliterated)
    n = len(o)
    lee: int | None
    if n == 0:
        status, lee = "preserved", 0
    elif n == 4:
        status, lee = "fully obliterated", 2
    else:
        status = "partially obliterated"
        opposing = {"sup", "inf"} <= o or {"ant", "post"} <= o
        lee = 1 if (n == 2 and opposing) else None
    wild = 0 if n == 0 else (3 if n == 4 else 2)
    return {
        "perineural_status": status,
        "n_obliterated": n,
        "lee_grade": lee,
        "wildermuth_grade": wild,
        "root_morphology": "not assessed",
    }


def _rules_of(height: float, cfg: Config) -> dict[str, Any]:
    crit = cfg.foramen_critical_height_mm
    return {"below_critical_height": bool(np.isfinite(height) and height <= crit)}


def normal_height(level: str, cfg: Config = DEFAULT) -> tuple[float, float] | None:
    """(mean, SD) of the Hasegawa 1995 normal group for ``level`` (disc label), or None if not tabulated."""
    for name, mu, sd in cfg.foramen_normal_height_mm:
        if name == level:
            return mu, sd
    return None


# ----------------------------------------------------------------------------------------------- figures
def _crop(
    stack: Stack, k: int, F: NDArray[np.float64], frame: DiscFrame, scale: float, lo_hi: tuple[float, float]
) -> NDArray[np.uint8]:
    """Sagittal crop around F (superior up), normalised to ``lo_hi``."""
    na = round(FIG_AP_MM * FIG_PX_PER_MM)
    ns = round(FIG_SI_MM * FIG_PX_PER_MM)
    a = np.linspace(-FIG_AP_MM / 2, FIG_AP_MM / 2, na)
    s = np.linspace(FIG_SI_MM / 2, -FIG_SI_MM / 2, ns)  # top row = superior
    AA, SS = np.meshgrid(a, s)
    pts = F[None, :] + AA.reshape(-1, 1) * frame.ap[None, :] + SS.reshape(-1, 1) * frame.normal[None, :]
    v = sample(stack, k, pts, scale).reshape(ns, na)
    lo, hi = lo_hi
    img = np.clip((np.nan_to_num(v, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    return (img * 255).astype(np.uint8)


def _cell(
    stack: Stack,
    frame: DiscFrame | None,
    site: Site | None,
    meas: Measure | None,
    scale: float,
    lo_hi: tuple[float, float],
    label: str,
    detail: str,
) -> NDArray[np.uint8]:
    """One figure cell: sagittal crop with the foramen outline (measured borders) and the grade text."""
    na = round(FIG_AP_MM * FIG_PX_PER_MM)
    ns = round(FIG_SI_MM * FIG_PX_PER_MM)
    if frame is None or site is None:
        im = Image.new("RGB", (na, ns), (40, 40, 40))
        d = ImageDraw.Draw(im)
        d.text((6, 6), label, fill=(255, 255, 0))
        d.text((6, 22), detail, fill=(200, 200, 200))
        return np.asarray(im, dtype=np.uint8)
    base = _crop(stack, site.k, site.centre, frame, scale, lo_hi)
    im = Image.fromarray(base).convert("RGB")
    d = ImageDraw.Draw(im)

    def xy(a: float, s: float) -> tuple[float, float]:
        return ((a + FIG_AP_MM / 2) * FIG_PX_PER_MM, (FIG_SI_MM / 2 - s) * FIG_PX_PER_MM)

    if (
        meas is not None
        and np.isfinite(meas.ant_mm)
        and np.isfinite(meas.post_mm)
        and np.isfinite(meas.sup_mm)
        and np.isfinite(meas.inf_mm)
    ):
        d.rectangle([xy(meas.ant_mm, meas.sup_mm), xy(meas.post_mm, meas.inf_mm)], outline=(255, 80, 80))
    cx, cy = xy(0.0, 0.0)
    d.line([(cx - 5, cy), (cx + 5, cy)], fill=(0, 255, 0))
    d.line([(cx, cy - 5), (cx, cy + 5)], fill=(0, 255, 0))
    d.text((4, 2), label, fill=(255, 255, 0))
    d.text((4, ns - 14), detail, fill=(255, 255, 255))
    return np.asarray(im, dtype=np.uint8)


def _grid_figure(cells: list[list[NDArray[np.uint8]]], row_labels: list[str]) -> NDArray[np.uint8]:
    """Assemble cells (rows = discs, columns = L, R) with row labels on the left."""
    if not cells:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = cells[0][0].shape[:2]
    pad, head = 6, 18
    canvas = np.full((head + len(cells) * (h + pad), 60 + 2 * (w + pad), 3), 30, dtype=np.uint8)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((64, 4), "L (patient left)", fill=(255, 255, 255))
    d.text((64 + w + pad, 4), "R (patient right)", fill=(255, 255, 255))
    for i, row in enumerate(cells):
        y = head + i * (h + pad)
        d.text((4, y + h // 2), row_labels[i], fill=(255, 255, 255))
        for j, cell in enumerate(row):
            im.paste(Image.fromarray(cell), (60 + j * (w + pad), y))
    return np.asarray(im, dtype=np.uint8)


# ------------------------------------------------------------------------------------------ the analysis
def _study_sagittal_t1(study: Study) -> Stack | None:
    hits = study.find("sag", "t1", slabs=False)
    if not hits:
        return None
    return max(hits, key=lambda s: s.n)


def _row(disc: str, side: str, numbering: str, status: str, reason: str, **kw: Any) -> dict[str, Any]:
    up, _ = split_disc(disc)
    base: dict[str, Any] = {
        "scope": "foramen",
        "disc": disc,
        "level": disc,
        "side": side,
        "root": up,
        "numbering": numbering,
        "status": status,
        "reason": reason,
        # measurement columns are always present (None when not measured), so the CSV has one shape
        "foraminal_height_mm": None,
        "ap_width_mm": None,
        "foraminal_fat_area_mm2": None,
        "perineural_status": None,
        "lee_grade": None,
        "wildermuth_grade": None,
        "root_morphology": "not assessed",
    }
    base.update(kw)
    return base


def measure(study: Study, anchors: Anchors, cfg: Config = DEFAULT) -> Result:
    """Foramina for every disc in ``anchors`` on both sides, from the sagittal T1 stack."""
    res = Result(name="foramina")
    res.meta.update(
        {
            "method": "parasagittal T1 fat search, Lee 2010 / Wildermuth perineural grading, Hasegawa 1995 height norms",
            "series_key": "sag_t1",
            "lateral_window_mm": list(cfg.foramen_lateral_mm),
            "critical_height_mm": cfg.foramen_critical_height_mm,
        }
    )
    numbering = status_of(anchors)
    stack = _study_sagittal_t1(study)
    if stack is None:
        res.warnings.append("no sagittal T1 series: foramina not assessed")
        return res
    if anchors.midline_x is None:
        res.warnings.append("no midline: foramina not assessed")
        return res
    if not anchors.discs:
        res.warnings.append("no discs labelled: foramina not assessed")
        return res
    scale = float(np.percentile(stack.vol, 99))
    lo_hi = (float(np.percentile(stack.vol, 0.5)), float(np.percentile(stack.vol, 99.5)))
    lo_hi = (lo_hi[0] / scale, lo_hi[1] / scale)
    ordered = sorted(anchors.discs.items(), key=lambda kv: -float(kv[1].center[2]))
    cell_rows: list[list[NDArray[np.uint8]]] = []
    row_labels: list[str] = []
    for label, disc in ordered:
        try:
            split_disc(label)
        except ValueError:
            res.warnings.append(f"{label}: not a disc label, skipped")
            continue
        frame = disc_frame(disc)
        row_cells: list[NDArray[np.uint8]] = []
        for side in SIDES:
            if frame is None:
                res.rows.append(
                    _row(
                        label, side, numbering, "not_assessed", "no posterior margin or normal for this disc"
                    )
                )
                row_cells.append(
                    _cell(stack, None, None, None, scale, lo_hi, f"{label} {side}", "not assessed")
                )
                continue
            site = search_site(stack, frame, side, anchors.midline_x, scale, cfg)
            if site is None:
                lo, hi = cfg.foramen_lateral_mm
                reason = (
                    f"sagittal T1 planes do not reach the foramen window {lo:.0f}-{hi:.0f} mm from the midline on "
                    f"side {side} (coverage)"
                )
                res.rows.append(_row(label, side, numbering, "not_assessed", reason))
                row_cells.append(
                    _cell(stack, None, None, None, scale, lo_hi, f"{label} {side}", "not assessed: coverage")
                )
                continue
            meas = measure_site(stack, frame, site, scale, cfg)
            if not meas.centre_fat:  # no borders or areas for a centre that is not foraminal fat
                meas = replace(meas, height_mm=float("nan"), ap_width_mm=float("nan"), area_mm2=float("nan"))
            g = grade(meas.obliterated) if meas.centre_fat else {}
            conf = 0.6
            if site.edge:
                conf *= 0.5
            if not np.isfinite(meas.height_mm) or not np.isfinite(meas.ap_width_mm):
                conf *= 0.5
            if not meas.centre_fat:
                conf = min(conf, 0.1)
            conf = min(conf, float(disc.confidence) if disc.confidence else conf)
            norm = normal_height(label, cfg)
            z = (meas.height_mm - norm[0]) / norm[1] if (norm and np.isfinite(meas.height_mm)) else None
            status = "assessed" if meas.centre_fat else "not_assessed"
            row = _row(
                label,
                side,
                numbering,
                status,
                meas.reason,
                plane_k=site.k,
                rims_found=bool(np.isfinite(meas.height_mm) and np.isfinite(meas.ap_width_mm)),
                lateral_mm=round(site.lateral_mm, 2),
                edge_plane=site.edge,
                centre_ap_offset_mm=round(site.ap_off_mm, 2),
                centre_si_offset_mm=round(site.si_off_mm, 2),
                centre_fat_index=round(site.score, 3),
                score_margin=round(site.score_margin, 3),
                foraminal_height_mm=round(meas.height_mm, 2) if np.isfinite(meas.height_mm) else None,
                ap_width_mm=round(meas.ap_width_mm, 2) if np.isfinite(meas.ap_width_mm) else None,
                foraminal_fat_area_mm2=round(meas.area_mm2, 1) if np.isfinite(meas.area_mm2) else None,
                obliterated_directions=list(meas.obliterated),
                height_z_hasegawa=round(z, 2) if z is not None else None,
                confidence=round(conf, 3),
                **_rules_of(meas.height_mm, cfg),
                **g,
            )
            res.rows.append(row)
            if status == "assessed" and (g["n_obliterated"] >= 2 or row["below_critical_height"]):
                why = []
                if g["n_obliterated"] >= 2:
                    why.append(f"perineural fat obliterated in {g['n_obliterated']} directions")
                if row["below_critical_height"]:
                    why.append(
                        f"foraminal height {meas.height_mm:.1f} mm at or below {cfg.foramen_critical_height_mm:.0f} mm"
                    )
                res.flags.append(
                    {
                        "disc": label,
                        "level": label,
                        "side": side,
                        "root": row["root"],
                        "kind": "foramen_geometry",
                        "confidence": round(min(conf, cfg.foramen_flag_max_confidence), 3),
                        "notes": "; ".join(why)
                        + f"; level name {numbering}; a measurement, not a compression claim",
                    }
                )
            row_cells.append(
                _cell(
                    stack,
                    frame,
                    site,
                    meas,
                    scale,
                    lo_hi,
                    f"{label} {side} root {row['root']}  {g.get('perineural_status', 'n/a')}",
                    f"h {meas.height_mm:.1f} mm  AP {meas.ap_width_mm:.1f} mm  Lee {g.get('lee_grade')}"
                    if status == "assessed"
                    else "not assessed",
                )
            )
        cell_rows.append(row_cells)
        row_labels.append(label)
    # relative to the same study's other levels (same side, median over assessed foramina)
    for side in SIDES:
        hs = [
            r["foraminal_height_mm"]
            for r in res.rows
            if r["side"] == side and r.get("foraminal_height_mm") is not None
        ]
        if hs:
            med = float(np.median(hs))
            for r in res.rows:
                if r["side"] == side and r.get("foraminal_height_mm") is not None:
                    r["height_minus_side_median_mm"] = round(r["foraminal_height_mm"] - med, 2)
    assessed = [r for r in res.rows if r["status"] == "assessed"]
    if assessed:
        fi = [r["centre_fat_index"] for r in assessed]
        med_fi = float(np.median(fi))
        for r in assessed:
            r["fat_index_rel_study_median"] = round(r["centre_fat_index"] / med_fi, 3) if med_fi > 0 else None
    n_na = len(res.rows) - len(assessed)
    if n_na:
        res.warnings.append(
            f"{n_na} of {len(res.rows)} foramina not assessed (coverage, posterior margin or rims)"
        )
    res.figures["foramina_grid"] = _grid_figure(cell_rows, row_labels)
    res.meta["sites"] = len(assessed)
    return res


@register("foramina")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Neural foramina per level and side (see the module docstring). Warnings, not exceptions, when inputs are missing."""
    return measure(study, anchors, config or DEFAULT)


# ------------------------------------------------------------------------------------- findings fields
def _foramen_row(res: Result, level: str, side: str) -> dict[str, Any] | None:
    for r in res.rows:
        if r.get("scope") == "foramen" and r.get("disc") == level and r.get("side") == side:
            return r
    return None


def _extract_lee(side: str) -> Any:
    def extract(res: Result, level: str) -> Candidate | None:
        r = _foramen_row(res, level, side)
        if r is None or r.get("status") != "assessed":
            return None
        n = int(r.get("n_obliterated") or 0)
        lee = r.get("lee_grade")
        note = (
            f"perineural fat obliterated in {n} direction(s) ({', '.join(r.get('obliterated_directions') or []) or 'none'}); "
            "a geometric proxy from T1, root morphology not assessed (grade 3 not assigned)"
        )
        if lee is None:
            note += "; outside Lee's scheme (not one opposing pair, not all four)"
        conf = float(r.get("confidence") or 0.0)
        return Candidate(value=lee, confidence=conf, notes=note, slices=(int(r["plane_k"]),))

    return extract


register_field(
    FieldRule(
        group="foramen_R",
        name="lee_grade",
        analysis="foramina",
        extract=_extract_lee("R"),
        figure="foramina_grid",
        notes="proxy grade from T1 (docs/analysis/foramina.md); not a reader grade",
    )
)
register_field(
    FieldRule(
        group="foramen_L",
        name="lee_grade",
        analysis="foramina",
        extract=_extract_lee("L"),
        figure="foramina_grid",
        notes="proxy grade from T1 (docs/analysis/foramina.md); not a reader grade",
    )
)
