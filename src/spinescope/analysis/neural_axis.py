"""Neural axis findings on a lumbar MRI: conus level, perineural (Tarlov) cysts, dural sac ratio, filum terminale.

Everything here is a measurement with a confidence, never a diagnosis, and every output is capped and marked
experimental (``Config.neural_axis_max_confidence``): none of these is validated on a reference (docs/analysis/neural_axis.md).

1. **Conus.** The experimental conus locator (:mod:`spinescope.anatomy.conus`, lab16/lab26) is surfaced as a row, with its
   level, position and confidence. A "possible low-lying conus" flag needs the tip below the L2-L3 disc
   (Wilson DA, Prince JR. AJNR 1989;10:259) AND a confidence at or above ``conus_low_lying_min_confidence``. The conus
   record is already capped at ``conus_max_confidence`` (0.5), so the flag cannot fire on today's locator. That is
   deliberate: the cap stays until a reference shows the locator is accurate.
2. **Perineural (Tarlov) cysts.** On the sagittal STIR (else T2) plane, CSF-like blobs in the sacral canal (below the
   L5-S1 disc) that are separate from the thecal sac. Size is the blob's major axis in mm. Prevalence and size context:
   Paulsen RD, Call GA, Murtagh FR. AJNR 1994;15:293 (PMID 8192075): 4.6% of 500 lumbosacral MRIs; treated symptomatic
   cysts 1.6-3.0 cm. Cysts are reported as "likely incidental" unless they reach ``neural_axis_cyst_large_mm``.
   Limits: cysts touching the sac merge with it and are missed; lumbar root-sleeve cysts are not separated on a
   midline-plane sagittal; intensity-only, so sacral marrow that matches CSF on the contrast gives "not assessed".
3. **Dural sac ratio (dural ectasia pattern).** Midsagittal thecal sac AP diameter over the vertebral body AP depth at
   the same level (Oosterhof N, et al. Spine 2001, PMID 11477262, derived in Marfan and control adults). Limits: L5 0.48,
   S1 0.57. Overlap with normal subjects is substantial (later studies), so it is a measurement with a low-confidence flag.
4. **Filum terminale at L5-S1 (axial).** Thickness of the filum (dark on T2, inside the CSF of the axial sac) against the
   2 mm limit (Raghavan N, et al. AJNR 1989;10:27: normal <= 2 mm, thickened > 2 mm). Fatty filum: a T1-bright focus
   inside the sac centre. Fat saturation is not checked, so this is a candidate, not a fat-verified finding. Nerve
   roots of the cauda equina can mimic both at this level; the axial resolution (about 1 mm) is near the filum size.

Not implemented (see docs/analysis/neural_axis.md): syrinx (rare in a lumbar FOV, no reference), clumped nerve roots
(no robust measure; Delamarter classification needs a reader, not a threshold).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from ..anatomy.anchors import Anchors
from ..anatomy.ml.grid import intensity_window, resample_slice
from ..anatomy.ml.labeller import CANAL, SegmentedStack
from ..anatomy.ml.postprocess import canal_areas_mm2, mid_sagittal
from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .base import Result, register

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
#: Sagittal contrasts for the cyst search. STIR suppresses fat, so sacral marrow is dark and CSF-like cysts stand out.
CYST_CONTRASTS: tuple[str, ...] = ("stir", "t2")
#: Half-width (mm) of the sacral band around the canal, LPS y (AP), for cyst candidates.
SACRAL_BAND_MM = 25.0
#: Half-width (mm) of the sacral marrow reference band around the canal.
MARROW_BAND_MM = 15.0
#: Half-width (mm) of the axial patch and its sampling step.
PATCH_STEP_MM = 0.5
#: Voxel size of the figure upscaling.
FIG_SCALE = 8
#: Eligible cyst candidates touching the bottom of the field of view are not counted (their size is unknown).
METHOD = "neural_axis/1 conus tip; sagittal CSF-like blobs in the sacral canal; dural sac ratio; axial filum at L5-S1"


# ------------------------------------------------------------------------------ small helpers
def _largest(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.asarray(lab == int(np.argmax(sizes)))


def canal_mask(seg: SegmentedStack, k: int, cfg: Config = DEFAULT) -> NDArray[np.bool_]:
    """Thecal sac (largest canal component) on the 1 mm grid of slice ``k``."""
    return _largest(seg.probs[k, CANAL] >= cfg.ml_prob_threshold)


def _z_of_rows(grid: Any, n_rows: int) -> FloatArray:
    rows = np.arange(n_rows, dtype=np.float64)
    return np.asarray(grid.lps_of_index(rows, np.zeros(n_rows))[:, 2], dtype=np.float64)


def _y_grid(grid: Any, shape: tuple[int, int]) -> FloatArray:
    """LPS y (AP) of every pixel centre of a plane grid."""
    ii, jj = np.meshgrid(
        np.arange(shape[0], dtype=np.float64), np.arange(shape[1], dtype=np.float64), indexing="ij"
    )
    return np.asarray(grid.lps_of_index(ii, jj)[..., 1], dtype=np.float64)


def _axes_mm(coords: NDArray[np.float64], res: float) -> tuple[float, float]:
    """(major axis, minor axis) in mm of a pixel set, from the second moments (a disc of diameter D gives D)."""
    if coords.shape[0] < 2:
        return res, res
    cov = np.cov(coords.T)
    lam = np.clip(np.sort(np.linalg.eigvalsh(cov)), 0.0, None)
    return 4.0 * math.sqrt(float(lam[1])) * res, 4.0 * math.sqrt(float(lam[0])) * res


def _solidity(coords: NDArray[np.float64]) -> float:
    """Pixel count over convex-hull area (1 for a convex blob). 0 when the hull is degenerate."""
    if coords.shape[0] < 3:
        return 0.0
    try:
        hull_area = float(ConvexHull(coords).volume)
    except QhullError:
        return 0.0
    return float(coords.shape[0] / hull_area) if hull_area > 0 else 0.0


def _window_span(window: tuple[float, float]) -> float:
    return max(float(window[1]) - float(window[0]), 1e-6)


def _rgb(
    img: NDArray[np.float64], window: tuple[float, float], outline: NDArray[np.bool_] | None
) -> NDArray[np.uint8]:
    """Grey image scaled to the window, scaled up, with a red outline of ``outline`` (a same-shape mask)."""
    lo, span = float(window[0]), _window_span(window)
    g = np.clip((img - lo) / span, 0.0, 1.0)
    rgb = np.stack([g, g, g], axis=-1)
    if outline is not None and outline.any():
        edge = outline & ~ndimage.binary_erosion(outline)
        rgb[edge] = (1.0, 0.1, 0.1)
    out = np.repeat(np.repeat((rgb * 255).astype(np.uint8), FIG_SCALE, axis=0), FIG_SCALE, axis=1)
    return out


# ------------------------------------------------------------------------------ 1. conus
def conus_rows(anchors: Anchors, cfg: Config = DEFAULT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Conus row and the possible-low-lying flag, from ``anchors.conus`` (already capped and marked experimental)."""
    c = anchors.conus
    if not isinstance(c, dict) or c.get("z") is None:
        return [{"scope": "conus", "located": False, "experimental": True}], []
    conf = float(c.get("confidence") or 0.0)
    low = c.get("low_lying")
    status = str(anchors.numbering.get("status", "") or "") or None
    row = {
        "scope": "conus",
        "located": True,
        "z_mm": c.get("z"),
        "level": c.get("level"),
        "position": c.get("position"),
        "confidence": round(conf, 3),
        "experimental": True,
        "low_lying": low,
        "fov_end": bool(c.get("fov_end")),
        "level_names": status,
        "flag_min_confidence": cfg.conus_low_lying_min_confidence,
    }
    flags: list[dict[str, Any]] = []
    if low is True and conf >= cfg.conus_low_lying_min_confidence:
        note = (
            f"tip below the {cfg.conus_low_lying_disc} disc; experimental locator"
            + ("; tip is a lower bound (field-of-view end)" if row["fov_end"] else "")
            + "; consider correlation with the sagittal T2 and clinical findings"
        )
        flags.append(
            {
                "kind": "possibly low-lying conus",
                "level": c.get("level"),
                "confidence": round(conf, 3),
                "note": note,
            }
        )
    return [row], flags


# ------------------------------------------------------------------------------ 2. perineural cysts (sagittal)
@dataclass
class _Blob:
    k: int  # sagittal slice index
    z: float
    y: float  # LPS AP coordinate
    size_mm: float  # major axis
    minor_mm: float
    solidity: float
    aspect: float
    mask: NDArray[np.bool_]  # on the slice grid (for the figure)


def cyst_blobs_2d(
    csf_like: NDArray[np.bool_],
    eligible: NDArray[np.bool_],
    sac_excl: NDArray[np.bool_],
    res: float,
    cfg: Config = DEFAULT,
) -> list[tuple[NDArray[np.float64], float, float, float, NDArray[np.bool_]]]:
    """CSF-like blobs of one slice outside the sac. Returns (centroid (row, col), major, minor, solidity, mask).

    A blob is kept when it is separate from the sac (``sac_excl`` is the sac dilated by the margin), inside the
    eligible region, not cut by the edge of the grid, solid (``neural_axis_cyst_min_solidity``) and oval
    (``neural_axis_cyst_min_aspect``). Its size is the major axis; size filtering is left to the caller.
    """
    lab, n = ndimage.label(csf_like & eligible & ~sac_excl)
    out: list[tuple[NDArray[np.float64], float, float, float, NDArray[np.bool_]]] = []
    h, w = csf_like.shape
    for i in range(1, n + 1):
        blob = lab == i
        coords = np.argwhere(blob).astype(np.float64)
        if (
            np.min(coords[:, 0]) == 0
            or np.min(coords[:, 1]) == 0
            or np.max(coords[:, 0]) == h - 1
            or np.max(coords[:, 1]) == w - 1
        ):
            continue
        major, minor = _axes_mm(coords, res)
        sol = _solidity(coords)
        aspect = minor / major if major > 0 else 0.0
        if sol < cfg.neural_axis_cyst_min_solidity or aspect < cfg.neural_axis_cyst_min_aspect:
            continue
        out.append((coords.mean(axis=0), major, minor, sol, blob))
    return out


def _link_blobs(blobs: list[_Blob], cfg: Config) -> list[list[_Blob]]:
    """Group blobs on adjacent slices whose centres lie within the link distance (plus half their sizes)."""
    parent = list(range(len(blobs)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(blobs):
        for j in range(i + 1, len(blobs)):
            b = blobs[j]
            if abs(a.k - b.k) != 1:
                continue
            d = math.hypot(a.z - b.z, a.y - b.y)
            if d <= cfg.neural_axis_cyst_link_mm + 0.5 * (a.size_mm + b.size_mm):
                parent[find(i)] = find(j)
    groups: dict[int, list[_Blob]] = {}
    for i, b in enumerate(blobs):
        groups.setdefault(find(i), []).append(b)
    return [sorted(g, key=lambda t: t.k) for g in groups.values()]


def _cyst_confidence(n_slices: int, cfg: Config) -> float:
    """Judgement: 0.3 for one slice, +0.1 per extra slice up to three, capped at ``neural_axis_max_confidence``."""
    return min(cfg.neural_axis_max_confidence, 0.3 + 0.1 * min(n_slices, 3))


def sagittal_cysts(
    seg: SegmentedStack, anchors: Anchors, cfg: Config = DEFAULT, *, contrast: str = "t2"
) -> tuple[list[dict[str, Any]], list[str], list[_Blob]]:
    """Perineural cyst rows for one sagittal segmentation. Returns (rows, warnings, blobs kept for the figure)."""
    warns: list[str] = []
    disc = anchors.discs.get("L5-S1")
    if disc is None:
        return [], ["perineural cysts not assessed: no L5-S1 disc anchor (sacral region undefined)"], []
    margin_z = float(disc.center[2]) - float(disc.height_mm or 8.0) / 2.0
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        return [], ["perineural cysts not assessed: no spinal canal in the sagittal segmentation"], []
    stack = seg.stack
    window = (float(seg.window[0]), float(seg.window[1]))
    span = _window_span(window)
    res = cfg.ml_resolution_mm
    # Reference AP line: the sac's centre on the mid-sagittal plane at the sacral level (or at the mid canal).
    g_mid = seg.grids[mid.k]
    m_mid = canal_mask(seg, mid.k, cfg)
    z_mid = _z_of_rows(g_mid, m_mid.shape[0])
    sacral_rows = np.flatnonzero(m_mid.any(axis=1) & (z_mid < margin_z))
    use_rows = sacral_rows if sacral_rows.size else np.flatnonzero(m_mid.any(axis=1))
    if use_rows.size == 0:
        return [], ["perineural cysts not assessed: no thecal sac on the mid-sagittal plane"], []
    y_ref = float(
        np.mean(
            g_mid.lps_of_index(
                use_rows.astype(np.float64), np.array([np.flatnonzero(m_mid[r]).mean() for r in use_rows])
            )[:, 1]
        )
    )
    # Pass 1: CSF reference (lumbar sac above the margin) and sacral marrow reference (around the sacral canal).
    csf_vals: list[float] = []
    marrow_vals: list[float] = []
    imgs: dict[int, NDArray[np.float32]] = {}
    masks: dict[int, NDArray[np.bool_]] = {}
    margin_px = max(1, round(cfg.neural_axis_cyst_canal_margin_mm / res))
    areas = canal_areas_mm2(seg, cfg)
    sac_slice = areas >= cfg.neural_axis_sac_area_frac * float(np.max(areas))
    for k in range(stack.n):
        grid = seg.grids[k]
        m = canal_mask(seg, k, cfg)
        if not m.any():
            continue
        img = resample_slice(stack, grid)
        imgs[k], masks[k] = img, m
        z = _z_of_rows(grid, m.shape[0])
        lumbar = z >= margin_z
        csf_vals += list(img[m & lumbar[:, None]].astype(np.float64))
        yy = _y_grid(grid, m.shape)
        sac_excl = ndimage.binary_dilation(m, iterations=margin_px)
        sacral_band = (np.abs(yy - y_ref) <= MARROW_BAND_MM) & (z[:, None] < margin_z) & ~sac_excl
        marrow_vals += list(img[sacral_band].astype(np.float64))
    if not csf_vals:
        return [], ["perineural cysts not assessed: no lumbar thecal sac above the L5-S1 margin"], []
    if not marrow_vals:
        return [], ["perineural cysts not assessed: no sacral marrow in the segmentation's field of view"], []
    csf = float(np.median(csf_vals))
    marrow = float(np.median(marrow_vals))
    if abs(csf - marrow) < cfg.neural_axis_cyst_min_contrast_frac * span:
        return (
            [],
            [
                f"perineural cysts not assessed on {contrast or 'sagittal'}: CSF and sacral marrow look alike "
                "(a cyst would not stand out)"
            ],
            [],
        )
    tol = cfg.neural_axis_cyst_csf_tol_frac * span
    # Pass 2: blobs per slice.
    blobs: list[_Blob] = []
    for k, img in imgs.items():
        grid = seg.grids[k]
        m = masks[k]
        z = _z_of_rows(grid, m.shape[0])
        yy = _y_grid(grid, m.shape)
        eligible = (z[:, None] < margin_z) & (np.abs(yy - y_ref) <= SACRAL_BAND_MM)
        # The sac is excluded only on slices that carry it; on lateral slices the largest canal blob may be a cyst.
        sac_excl = (
            ndimage.binary_dilation(masks[k], iterations=margin_px) if sac_slice[k] else np.zeros_like(m)
        )
        csf_like = np.abs(img.astype(np.float64) - csf) <= tol
        for centroid, major, minor, sol, blob in cyst_blobs_2d(csf_like, eligible, sac_excl, res, cfg):
            if major < cfg.neural_axis_cyst_min_mm:
                continue
            xyz = grid.lps_of_index(float(centroid[0]), float(centroid[1]))
            blobs.append(
                _Blob(
                    k=k,
                    z=float(xyz[2]),
                    y=float(xyz[1]),
                    size_mm=major,
                    minor_mm=minor,
                    solidity=sol,
                    aspect=minor / major,
                    mask=blob,
                )
            )
    rows: list[dict[str, Any]] = []
    kept: list[_Blob] = []
    for grp in _link_blobs(blobs, cfg):
        best = max(grp, key=lambda b: b.size_mm)
        conf = _cyst_confidence(len({b.k for b in grp}), cfg)
        large = best.size_mm >= cfg.neural_axis_cyst_large_mm
        rows.append(
            {
                "scope": "perineural_cyst",
                "contrast": contrast,
                "z_mm": round(best.z, 1),
                "y_mm": round(best.y, 1),
                "size_mm": round(best.size_mm, 1),
                "minor_mm": round(best.minor_mm, 1),
                "slices": len({b.k for b in grp}),
                "large": bool(large),
                "confidence": round(conf, 3),
                "experimental": True,
                "wording": (
                    "large perineural cyst: consider correlation with symptoms"
                    if large
                    else "possible perineural cyst, likely incidental"
                ),
            }
        )
        kept.append(best)
    if not rows:
        warns.append(f"no perineural cyst candidate on {contrast or 'sagittal'} (sacral canal, below L5-S1)")
    return rows, warns, kept


def dural_sac_ratios(
    seg: SegmentedStack, anchors: Anchors, cfg: Config = DEFAULT
) -> tuple[list[dict[str, Any]], list[str]]:
    """Dural sac ratio at L5 and S1 (midsagittal sac AP diameter over the body AP depth at the same level)."""
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        return [], ["dural sac ratio not assessed: no spinal canal in the sagittal segmentation"]
    grid = seg.grids[mid.k]
    m = canal_mask(seg, mid.k, cfg)
    z = _z_of_rows(grid, m.shape[0])
    rows: list[dict[str, Any]] = []
    warns: list[str] = []
    for name, limit in (("L5", cfg.neural_axis_dsr_l5), ("S1", cfg.neural_axis_dsr_s1)):
        v = anchors.vertebrae.get(name)
        depth = (v.body or {}).get("ap_depth_mm") if v is not None else None
        if v is None or not depth:
            warns.append(f"dural sac ratio at {name} not assessed: no body AP depth from the labeller")
            continue
        r = int(np.argmin(np.abs(z - float(v.center[2]))))
        cols = np.flatnonzero(m[r])
        if cols.size == 0:
            warns.append(f"dural sac ratio at {name} not assessed: no thecal sac at that height")
            continue
        # extent, not count: a filum or a septum inside the sac must not shorten the measured diameter
        sac_ap = float(cols[-1] - cols[0] + 1) * cfg.ml_resolution_mm
        ratio = sac_ap / float(depth)
        rows.append(
            {
                "scope": "dural_sac_ratio",
                "level": name,
                "sac_ap_mm": round(sac_ap, 1),
                "body_ap_mm": round(float(depth), 1),
                "ratio": round(ratio, 3),
                "limit": limit,
                "above_limit": bool(ratio > limit),
                "confidence": round(min(cfg.neural_axis_max_confidence, 0.3), 3),
                "experimental": True,
                "wording": "dural sac ratio above the published limit: consider dural ectasia in correlation with morphology",
            }
        )
    return rows, warns


# ------------------------------------------------------------------------------ 3. filum terminale (axial)
def canal_point_at(seg: SegmentedStack, z_target: float, cfg: Config = DEFAULT) -> FloatArray | None:
    """LPS point at the centre of the sac on the mid-sagittal plane, at height ``z_target``; None if not in the sac."""
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        return None
    grid = seg.grids[mid.k]
    m = canal_mask(seg, mid.k, cfg)
    z = _z_of_rows(grid, m.shape[0])
    r = int(np.argmin(np.abs(z - z_target)))
    cols = np.flatnonzero(m[r])
    if cols.size == 0:
        return None
    return np.asarray(grid.lps_of_index(float(r), float(cols.mean())), dtype=np.float64)


def axial_patch(
    stack: Stack, k: int, centre: FloatArray, half_mm: float
) -> tuple[NDArray[np.float64], FloatArray]:
    """Axial image of slice ``k`` sampled on a 0.5 mm grid centred at ``centre`` (LPS). Returns (patch, offsets mm)."""
    offs = np.arange(-half_mm, half_mm + 0.5 * PATCH_STEP_MM, PATCH_STEP_MM)
    uu, vv = np.meshgrid(offs, offs, indexing="ij")
    origin = np.asarray(stack.ipp[k], dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    c = c - float((c - origin) @ stack.normal) * stack.normal  # project onto the slice plane
    pts = c[None, None, :] + uu[..., None] * stack.row_dir + vv[..., None] * stack.col_dir
    rel = pts - origin
    r = (rel @ stack.col_dir) / stack.row_mm
    cc = (rel @ stack.row_dir) / stack.col_mm
    patch = ndimage.map_coordinates(
        np.asarray(stack.vol[k], dtype=np.float64), [r, cc], order=1, mode="constant", cval=0.0
    )
    return patch, offs


def _axial_stack(study: Study, contrast: str) -> Stack | None:
    for s in study.find("ax", contrast):
        if not s.meta.get("role", {}).get("localizer"):
            return s
    return None


def filum_at_l5s1(
    stack: Stack, centre: FloatArray, cfg: Config = DEFAULT, *, fat: bool
) -> tuple[dict[str, Any] | None, NDArray[np.bool_] | None, str, FloatArray | None]:
    """Filum candidate on one axial slice through the L5-S1 canal centre.

    The CSF reference is the median in a disc of ``neural_axis_filum_csf_radius_mm`` around the canal centre (the sac
    around a thick filum still reads as CSF). The filum is the component whose centroid lies within
    ``neural_axis_filum_centre_mm`` of the centre and whose intensity is beyond the half-maximum between the CSF and
    the darkest (T2) or brightest (T1) pixel of the centre disk. Its diameter is the equivalent diameter at that
    half-maximum threshold. Returns (row or None, outline mask for the figure, reason, patch).
    """
    k, dist = stack.nearest_slice(centre)
    if abs(dist) > max(stack.thickness, 5.0):
        return None, None, "L5-S1 canal centre is outside the axial slab", None
    if not stack.contains(centre, slab_margin_mm=0.0):
        return None, None, "L5-S1 canal centre is outside the axial field of view", None
    half = cfg.neural_axis_filum_search_mm
    patch, offs = axial_patch(stack, k, centre, half)
    span = _window_span(intensity_window(stack.vol))
    uu, vv = np.meshgrid(offs, offs, indexing="ij")
    r2 = uu**2 + vv**2
    csf_disk = r2 <= cfg.neural_axis_filum_csf_radius_mm**2
    centre_disk = r2 <= 1.5**2
    csf = float(np.median(patch[csf_disk]))
    tol = cfg.neural_axis_cyst_csf_tol_frac * span
    if float((np.abs(patch[csf_disk] - csf) <= tol).mean()) < 0.5:
        return None, None, "no CSF-like thecal sac at the canal centre (contrast or slice not suited)", patch
    kind = "fatty" if fat else "thin"
    if fat:
        ref = float(np.max(patch[centre_disk]))
        # Fat is the brightest tissue on T1 (subcutaneous fat, marrow): the filum must reach a share of that level.
        # Cord-like tissue is brighter than CSF on T1 too, so a plain rise above CSF is not enough.
        fat_level = float(np.percentile(np.asarray(stack.vol[k], dtype=np.float64), 99.0))
        contrast_ok = (ref - csf >= cfg.neural_axis_fatty_contrast_frac * span) and (
            ref - csf >= cfg.neural_axis_fatty_relative_frac * max(fat_level - csf, 1e-6)
        )
        cand = patch >= 0.5 * (csf + ref)
    else:
        ref = float(np.min(patch[centre_disk]))
        contrast_ok = csf - ref >= cfg.neural_axis_fatty_contrast_frac * span
        cand = patch <= 0.5 * (csf + ref)
    if not contrast_ok:
        return None, None, f"no {kind} filum candidate inside the sac at the canal centre", patch
    cand &= r2 <= half**2
    lab, n = ndimage.label(cand)
    best: tuple[float, int] | None = None
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 2:  # single pixels are noise at 0.5 mm steps
            continue
        d = float(np.hypot(uu[comp].mean(), vv[comp].mean()))
        if d > cfg.neural_axis_filum_centre_mm:
            continue
        if best is None or d < best[0]:
            best = (d, i)
    if best is None:
        return None, None, f"no {kind} filum candidate inside the sac at the canal centre", patch
    comp = lab == best[1]
    diam = 2.0 * math.sqrt(float(comp.sum()) * PATCH_STEP_MM * PATCH_STEP_MM / math.pi)
    # Within one in-plane pixel of the limit the call is indeterminate (null), as the findings schema allows.
    pixel = max(float(stack.row_mm), float(stack.col_mm))
    limit = cfg.neural_axis_filum_thick_mm
    thick: bool | None = None if abs(diam - limit) <= pixel else bool(diam > limit)
    row = {
        "scope": "filum_fatty" if fat else "filum",
        "contrast": "t1" if fat else "t2",
        "diameter_mm": round(diam, 2),
        "thickened": None if fat else thick,
        "pixel_mm": round(pixel, 3),
        "confidence": round(min(cfg.neural_axis_max_confidence, 0.3), 3),
        "experimental": True,
        "wording": (
            "candidate fatty filum (T1 bright focus in the sac centre; fat saturation not checked): "
            "consider correlation with the sagittal T1 and fat-saturated images"
            if fat
            else (
                f"filum terminale "
                f"{'indeterminate (at the limit)' if thick is None else ('thickened' if thick else 'not thickened')} "
                f"({diam:.1f} mm on axial, pixel {pixel:.2f} mm; limit {limit:.0f} mm, Raghavan 1989)"
            )
        ),
    }
    return row, comp, "", patch


def filum_rows(
    study: Study, anchors: Anchors, seg: SegmentedStack | None, cfg: Config = DEFAULT
) -> tuple[list[dict[str, Any]], dict[str, NDArray[np.uint8]], list[dict[str, Any]], list[str]]:
    """Filum thickness (T2 axial) and fatty filum candidate (T1 axial) at L5-S1. Missing axials give warnings."""
    rows: list[dict[str, Any]] = []
    figs: dict[str, NDArray[np.uint8]] = {}
    flags: list[dict[str, Any]] = []
    warns: list[str] = []
    disc = anchors.discs.get("L5-S1")
    if disc is None or seg is None:
        return [], {}, [], ["filum not assessed: needs the L5-S1 disc anchor and the sagittal segmentation"]
    centre = canal_point_at(seg, float(disc.center[2]), cfg)
    if centre is None:
        return [], {}, [], ["filum not assessed: no thecal sac at L5-S1 on the sagittal segmentation"]
    for contrast, fat in (("t2", False), ("t1", True)):
        stack = _axial_stack(study, contrast)
        if stack is None:
            warns.append(
                f"filum ({'fatty, T1' if fat else 'thickness, T2'}) not assessed: no axial {contrast.upper()}"
            )
            continue
        row, outline, reason, patch = filum_at_l5s1(stack, centre, cfg, fat=fat)
        if row is None:
            warns.append(f"filum ({contrast.upper()} axial) not assessed: {reason}")
            continue
        rows.append(row)
        if fat:
            flags.append(
                {
                    "kind": "candidate fatty filum",
                    "level": "L5-S1",
                    "confidence": row["confidence"],
                    "note": row["wording"],
                }
            )
        elif row["thickened"]:
            flags.append(
                {
                    "kind": "possibly thickened filum",
                    "level": "L5-S1",
                    "confidence": row["confidence"],
                    "note": row["wording"],
                }
            )
        if patch is not None and outline is not None:
            figs[f"filum_{contrast}"] = _rgb(patch, _window_of(stack), outline)
    return rows, figs, flags, warns


def _window_of(stack: Stack) -> tuple[float, float]:
    lo, hi = intensity_window(stack.vol)
    return float(lo), float(hi)


# ------------------------------------------------------------------------------ analysis
def _sagittal_segmentation(study: Study, cfg: Config, warns: list[str]) -> tuple[SegmentedStack | None, str]:
    """Network segmentation of the first available sagittal series (STIR, then T2). Missing: warning."""
    from ..anatomy.ml.labeller import segment_stack

    for contrast in CYST_CONTRASTS:
        hits = [s for s in study.find("sag", contrast) if not s.meta.get("role", {}).get("localizer")]
        if not hits:
            continue
        try:
            return segment_stack(hits[0], cfg), contrast
        except Exception as exc:  # the segmentation needs the packaged weights and onnxruntime
            warns.append(
                f"sagittal segmentation failed ({type(exc).__name__}): cyst, sac ratio and filum not assessed"
            )
            return None, contrast
    warns.append("no sagittal STIR or T2 series: cyst, sac ratio and filum not assessed")
    return None, ""


@register("neural_axis")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Conus, perineural cysts, dural sac ratio and filum at L5-S1. Missing inputs give warnings, never exceptions."""
    cfg = config or DEFAULT
    res = Result(name="neural_axis")
    res.meta = {
        "method": METHOD,
        "experimental": True,
        "max_confidence": cfg.neural_axis_max_confidence,
        "cyst_min_mm": cfg.neural_axis_cyst_min_mm,
        "cyst_large_mm": cfg.neural_axis_cyst_large_mm,
        "filum_thick_mm": cfg.neural_axis_filum_thick_mm,
        "dsr_limits": {"L5": cfg.neural_axis_dsr_l5, "S1": cfg.neural_axis_dsr_s1},
    }
    rows, flags = conus_rows(anchors, cfg)
    res.rows += rows
    res.flags += flags
    seg, contrast = _sagittal_segmentation(study, cfg, res.warnings)
    res.meta["sagittal_contrast"] = contrast or None
    if seg is not None:
        cyst_rows, cyst_warns, kept = sagittal_cysts(seg, anchors, cfg, contrast=contrast)
        res.rows += cyst_rows
        res.warnings += cyst_warns
        # cyst candidates fail specificity on both public sets (docs/analysis/neural_axis.md): rows always, flags opt-in
        for row in cyst_rows if cfg.neural_axis_cyst_flags else []:
            res.flags.append(
                {
                    "kind": row["wording"],
                    "z_mm": row["z_mm"],
                    "size_mm": row["size_mm"],
                    "confidence": row["confidence"],
                }
            )
        if kept:
            best = max(kept, key=lambda b: b.size_mm)
            stack = seg.stack
            img = resample_slice(stack, seg.grids[best.k])
            res.figures["cyst_sagittal"] = _rgb(img.astype(np.float64), _window_of(stack), best.mask)
        dsr_rows, dsr_warns = dural_sac_ratios(seg, anchors, cfg)
        res.rows += dsr_rows
        res.warnings += dsr_warns
        for row in dsr_rows:
            if row["above_limit"]:
                res.flags.append(
                    {
                        "kind": row["wording"],
                        "level": row["level"],
                        "ratio": row["ratio"],
                        "confidence": row["confidence"],
                    }
                )
    f_rows, f_figs, f_flags, f_warns = filum_rows(study, anchors, seg, cfg)
    res.rows += f_rows
    res.flags += f_flags
    res.warnings += f_warns
    res.figures.update(f_figs)
    res.warnings.append("syrinx and clumped nerve roots are not assessed (see docs/analysis/neural_axis.md)")
    return res


__all__ = [
    "METHOD",
    "analyze",
    "axial_patch",
    "canal_mask",
    "canal_point_at",
    "conus_rows",
    "cyst_blobs_2d",
    "dural_sac_ratios",
    "filum_at_l5s1",
    "filum_rows",
    "sagittal_cysts",
]
