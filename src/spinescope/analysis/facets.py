"""Facet (zygapophyseal) joints per level and side: joint-space fluid on axial MRI, parasagittal fallback.

Method, per disc level and side (no fixed slice numbers, no per-study constants):

1. Site. The canal point at the disc level and the canal half width come from the level map
   (:func:`spinescope.anatomy.correspondence.level_map`, measured geometry where available). The joint site is
   ``facets_site_fraction`` of the way from the canal wall to the body edge, lateral to the canal and at the canal's
   level (judgement; see ``docs/analysis/facets.md``). Side ``L`` is patient left (+x in LPS), ``R`` patient right.
2. Stack. The first axial stack (STIR, then T2, then PD: fat-suppressed first) whose slab covers the site within half
   a slice. If none, a parasagittal stack whose slice lies within half a slice of the site (lower confidence). T1 is
   not used for fluid.
3. Sampling. A square of ``facets_roi_half_mm`` around the site is resampled in the stack's plane on a
   ``facets_px_mm`` grid (trilinear, :func:`spinescope.geometry.sample_plane`). Reference patches: CSF in the thecal
   sac at the canal point, muscle ``facets_muscle_offset_mm`` posterior to the site. Pixels within the canal (canal
   half width + 1 mm) are excluded so the thecal sac is not read as joint fluid.
4. Fluid. Threshold = muscle + ``facets_fluid_frac`` x (CSF - muscle) (half maximum between muscle and CSF). The
   connected component that reaches within ``facets_seed_mm`` of the site is the fluid. Width = maximal caliper of
   that component (mm); roundness = minimal / maximal caliper. A call is made only when the CSF-muscle contrast is at
   least ``facets_min_contrast`` of the sampled intensity range.
5. Calls. Only on fat-suppressed images (STIR, fat-sat T2); plain T2 is recorded but not called, because fat is as
   bright as fluid on it. A fluid component that reaches the window edge is not called (its width is a lower bound
   and such regions were not joints). Otherwise effusion present when the width is at least
   ``facets_effusion_min_mm``. A round collection of at least ``facets_cyst_min_mm`` is reported as a possible
   synovial cyst (wording "possible"). The fluid ratio (fluid vs muscle over CSF vs muscle) states how bright the fluid
   is relative to CSF.
6. Confidence. 0.8 for a window-contained fat-suppressed call; the parasagittal fallback is capped at
   ``facets_sagittal_max_confidence``.

Output rows carry ``scope = "joint"`` (fluid, one row per level and side) and ``scope = "joint_geometry"`` (dx21:
image-based joint line, facet angle and morphology proxies, see :func:`joint_geometry_rows`). The Weishaupt grade is
never assigned here (``grade_status = "not assessed"``); periarticular oedema is not measured.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..anatomy.anchors import Anchors
from ..anatomy.correspondence import LevelMap, level_map
from ..config import DEFAULT, Config
from ..geometry import Stack, sample_plane, unit
from ..study import Study
from .base import Result, register
from .findings import Candidate, Extract, FieldRule, register_field

log = logging.getLogger(__name__)

SIDES: tuple[str, str] = ("L", "R")
#: Contrasts used for fluid, in preference order. T1 is never used for fluid.
FLUID_CONTRASTS: tuple[str, ...] = ("stir", "t2", "pd")
#: Angles (deg) at which the maximal and minimal caliper are measured (5 deg steps, deterministic).
_CALIPER_ANGLES = np.deg2rad(np.arange(0.0, 180.0, 5.0))
_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class JointSite:
    """Where one facet joint is expected: LPS point, and the canal it is measured from."""

    level: str
    side: str  # "L" patient left, "R" patient right
    xyz: NDArray[np.float64]
    canal_xyz: NDArray[np.float64]
    canal_half_mm: float
    lateral_mm: float


@dataclass
class JointMeasure:
    """Fluid measurement in one stack. ``width_mm`` is None when no call could be made (no contrast)."""

    width_mm: float | None
    roundness: float | None
    roi_limited: bool
    fluid_ratio: float | None
    csf: float
    muscle: float
    threshold: float
    contrast: float
    fluid_mask: NDArray[np.bool_]  # ROI-shaped; the fluid component (empty when none)
    roi: NDArray[np.float64]  # ROI image (NaN outside the field of view)
    site_rc: tuple[float, float]  # joint site in ROI pixel coordinates (row, col)


# ----------------------------------------------------------------------------- sites
def joint_sites(lm: LevelMap, anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, dict[str, JointSite]]:
    """Joint site per disc label and side, in the disc plane (posterior to the canal, lateral to it).

    Posterior distance from the disc centre: half the body depth plus half the canal AP diameter (the level map's
    measured or flagged defaults). Lateral distance: ``canal_half + facets_site_fraction * (body_half - canal_half)``.
    The canal point is the same posterior point at zero lateral offset, so the thecal sac is sampled in the same
    slice as the joint. The site is not forced to disc height: the disc plane follows the segment's tilt.
    """
    out: dict[str, dict[str, JointSite]] = {}
    for label, entry in lm.discs.items():
        disc = anchors.discs.get(label)
        body = entry.geometry.get("body", {})
        half_w = body.get("half_width_mm")
        half_d = body.get("half_depth_mm")
        canal_half = entry.geometry.get("canal_half_width_mm")
        canal_ap = entry.geometry.get("canal_ap_mm")
        if disc is None or half_w is None or half_d is None or canal_half is None or canal_ap is None:
            continue
        n = unit(np.asarray(entry.normal, dtype=np.float64))
        lat = unit(_X - float(_X @ n) * n)  # +x: patient left
        ap = unit(np.asarray(disc.ap_dir, dtype=np.float64))
        c = np.asarray(entry.centre, dtype=np.float64)
        post = float(half_d) + 0.5 * float(canal_ap)
        canal = c + post * ap
        dist = float(canal_half) + cfg.facets_site_fraction * (float(half_w) - float(canal_half))
        per: dict[str, JointSite] = {}
        for side, sign in (("L", 1.0), ("R", -1.0)):
            per[side] = JointSite(
                level=label,
                side=side,
                xyz=canal + sign * dist * lat,
                canal_xyz=canal,
                canal_half_mm=float(canal_half),
                lateral_mm=dist,
            )
        out[label] = per
    return out


# ----------------------------------------------------------------------------- stacks
def _covers(st: Stack, xyz: NDArray[np.float64]) -> bool:
    """True when the point lies in the stack's field of view and within half a slice of its plane."""
    _, dist = st.nearest_slice(xyz)
    half = 0.5 * (st.spacing or st.thickness or 1.0)
    return dist <= half + 1e-6 and st.contains(xyz)


def _contrast_of(st: Stack) -> tuple[str, bool]:
    role = st.meta.get("role", {})
    contrast = str(role.get("contrast") or "")
    fat_sup = bool(role.get("fat_suppressed")) or contrast == "stir"
    return contrast, fat_sup


def choose_stack(
    study: Study, xyz: NDArray[np.float64], refs: tuple[NDArray[np.float64], ...] = ()
) -> tuple[Stack, str] | None:
    """(stack, plane) for a joint site, or None if no stack covers it.

    Order: fat-suppressed before not (STIR or fat-sat T2 can call fluid; plain T2 cannot, see :func:`analyze`), then
    axial before parasagittal, then the contrast preference, then the closest plane, then the key. ``refs`` are points
    the stack must also contain (the CSF and muscle reference patches).
    """
    cands: list[tuple[tuple[int, int, int, float, str], Stack, str]] = []
    for key, st in study.stacks.items():
        role = st.meta.get("role", {})
        if st.plane not in ("ax", "sag") or role.get("localizer") or role.get("post_contrast"):
            continue
        contrast, fat_sup = _contrast_of(st)
        if contrast not in FLUID_CONTRASTS or not _covers(st, xyz):
            continue
        if not all(st.contains(p) for p in refs):
            continue
        _, dist = st.nearest_slice(xyz)
        rank = (0 if fat_sup else 1, 0 if st.plane == "ax" else 1, FLUID_CONTRASTS.index(contrast))
        cands.append(((*rank, dist, key), st, st.plane))
    if not cands:
        return None
    _, best, plane = min(cands, key=lambda t: t[0])
    return best, plane


def _refs(site: JointSite, cfg: Config) -> tuple[NDArray[np.float64], ...]:
    """Reference points that the chosen stack must also contain: the canal point and the muscle patch."""
    return (site.canal_xyz, site.xyz + cfg.facets_muscle_offset_mm * _Y)


def _axes(plane: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """In-plane (u = screen right, v = screen down) directions: axial patient-left right and posterior down;
    sagittal posterior right and inferior down (anterior-left display)."""
    if plane == "ax":
        return _X, _Y
    return _Y, -_Z


# ----------------------------------------------------------------------------- measurement
def _calipers(pts: NDArray[np.float64]) -> tuple[float, float]:
    """(maximal, minimal) caliper width of a point set, in the units of ``pts``."""
    dirs = np.stack([np.cos(_CALIPER_ANGLES), np.sin(_CALIPER_ANGLES)], axis=1)
    proj = pts @ dirs.T
    widths = proj.max(axis=0) - proj.min(axis=0)
    return float(widths.max()), float(widths.min())


def _patch(
    st: Stack,
    origin: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    half: float,
    px: float,
) -> NDArray[np.float64]:
    return np.asarray(sample_plane(st, origin, u, v, (2 * half, 2 * half), px), dtype=np.float64)


def measure_joint(stack: Stack, plane: str, site: JointSite, cfg: Config = DEFAULT) -> JointMeasure:
    """Fluid in the joint at ``site`` on one stack (see the module docstring for the steps)."""
    u, v = _axes(plane)
    px = cfg.facets_px_mm
    roi = _patch(stack, site.xyz, u, v, cfg.facets_roi_half_mm, px)
    # reference patches: thecal sac at the canal point, paraspinal muscle posterior to the joint
    csf_img = _patch(stack, site.canal_xyz, u, v, cfg.facets_ref_radius_mm, px)
    muscle_img = _patch(
        stack, site.xyz + cfg.facets_muscle_offset_mm * _Y, u, v, cfg.facets_ref_radius_mm, px
    )
    csf = float(np.nanmedian(csf_img)) if np.isfinite(csf_img).any() else float("nan")
    muscle = float(np.nanmedian(muscle_img)) if np.isfinite(muscle_img).any() else float("nan")
    rng = float(np.nanpercentile(roi, 99.5) - np.nanpercentile(roi, 0.5)) if np.isfinite(roi).any() else 0.0
    nt, ns = roi.shape
    # ROI pixel -> mm offset (u, v) from the site, same convention as sample_plane
    tt = (np.arange(nt) - (nt - 1) / 2) * px
    ss = (np.arange(ns) - (ns - 1) / 2) * px
    site_rc = ((nt - 1) / 2, (ns - 1) / 2)
    empty = np.zeros(roi.shape, dtype=bool)
    if not (np.isfinite(csf) and np.isfinite(muscle)) or rng <= 0:
        return JointMeasure(None, None, False, None, csf, muscle, float("nan"), 0.0, empty, roi, site_rc)
    contrast = (csf - muscle) / rng
    thr = muscle + cfg.facets_fluid_frac * (csf - muscle)
    if contrast < cfg.facets_min_contrast:
        return JointMeasure(None, None, False, None, csf, muscle, thr, contrast, empty, roi, site_rc)
    # exclude the thecal sac: pixels within the canal half width (+1 mm) of the canal point
    dc = site.canal_xyz - site.xyz
    cs, ct = float(dc @ u), float(dc @ v)
    ggs, ggt = np.meshgrid(ss, tt)
    excl = np.hypot(ggs - cs, ggt - ct) <= site.canal_half_mm + 1.0
    finite = np.isfinite(roi)
    mask = (roi >= thr) & finite & ~excl
    lab, n_lab = ndi.label(mask)
    if n_lab == 0:
        return JointMeasure(0.0, None, False, None, csf, muscle, thr, contrast, empty, roi, site_rc)
    seed = np.hypot(ggs, ggt) <= cfg.facets_seed_mm
    ids, counts = np.unique(lab[seed & (lab > 0)], return_counts=True)
    if ids.size == 0:
        return JointMeasure(0.0, None, False, None, csf, muscle, thr, contrast, empty, roi, site_rc)
    comp = lab == int(ids[int(np.argmax(counts))])
    rr, cc = np.nonzero(comp)
    pts = np.stack([ss[cc], tt[rr]], axis=1)
    cmax, cmin = _calipers(pts)
    width = cmax + px
    roundness = float(cmin / cmax) if cmax > 0 else None
    roi_limited = bool(rr.min() == 0 or cc.min() == 0 or rr.max() == nt - 1 or cc.max() == ns - 1)
    fluid_med = float(np.median(roi[comp]))
    fluid_ratio = (fluid_med - muscle) / (csf - muscle) if csf != muscle else None
    return JointMeasure(
        width_mm=width,
        roundness=roundness,
        roi_limited=roi_limited,
        fluid_ratio=fluid_ratio,
        csf=csf,
        muscle=muscle,
        threshold=thr,
        contrast=contrast,
        fluid_mask=comp,
        roi=roi,
        site_rc=site_rc,
    )


def tropism_flag(left_deg: float, right_deg: float, threshold_deg: float | None = None) -> dict[str, Any]:
    """Left-right facet angle asymmetry (tropism) against one threshold (pure helper; see :func:`tropism_flags`).

    The default threshold is ``Config.facets_tropism_deg`` (7 degrees, a commonly quoted value; see the config comment).
    """
    thr = DEFAULT.facets_tropism_deg if threshold_deg is None else threshold_deg
    diff = abs(float(left_deg) - float(right_deg))
    return {"difference_deg": round(diff, 2), "threshold_deg": thr, "tropism": bool(diff > thr)}


# ----------------------------------------------------------------------------- analysis
def _confidence(stack_plane: str, cfg: Config) -> float:
    # fat-suppressed fluid call (the only kind made); validated on the phantom only, so capped (config source comment)
    conf = cfg.facets_max_confidence
    if stack_plane == "sag":
        conf = min(conf, cfg.facets_sagittal_max_confidence)
    return round(conf, 3)


def _figure(level: str, panels: dict[str, tuple[JointMeasure, str]], cfg: Config) -> NDArray[np.uint8] | None:
    """Two-panel crop (L | R) with the site, the fluid outline and the measured values."""
    if not panels:
        return None
    scale = 6  # display pixels per ROI pixel
    tiles: list[Image.Image] = []
    for side in SIDES:
        if side not in panels:
            tiles.append(Image.new("RGB", (48 * scale, 48 * scale), (0, 0, 0)))
            continue
        m, label = panels[side]
        img = m.roi
        lo, hi = np.nanpercentile(img, 0.5), np.nanpercentile(img, 99.5)
        g = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
        tile = Image.fromarray((g * 255).astype(np.uint8)).resize(
            (img.shape[1] * scale, img.shape[0] * scale), Image.Resampling.NEAREST
        )
        rgb = Image.new("RGB", tile.size)
        rgb.paste(tile)
        d = ImageDraw.Draw(rgb)
        r, c = m.site_rc
        d.line([(c * scale - 6, r * scale), (c * scale + 6, r * scale)], fill=(255, 0, 0))
        d.line([(c * scale, r * scale - 6), (c * scale, r * scale + 6)], fill=(255, 0, 0))
        if m.fluid_mask.any():
            edge = m.fluid_mask & ~ndi.binary_erosion(m.fluid_mask)
            for y, x in zip(*np.nonzero(edge), strict=True):
                d.rectangle(
                    [x * scale, y * scale, x * scale + scale - 1, y * scale + scale - 1],
                    outline=(255, 255, 0),
                )
        d.text((6, 4), f"{level} {side} facet", fill=(255, 255, 0))
        d.text((6, 18), label, fill=(0, 255, 0))
        tiles.append(rgb)
    h = max(t.size[1] for t in tiles)
    w = sum(t.size[0] for t in tiles) + 8
    canvas = Image.new("RGB", (w, h), (30, 30, 30))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0))
        x += t.size[0] + 8
    return np.asarray(canvas, dtype=np.uint8)


@register("facets")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Facet joint fluid per level and side: width, effusion call, possible synovial cyst (see module docstring)."""
    cfg = config or DEFAULT
    res = Result(name="facets", meta={"method": "joint fluid vs CSF and muscle, see docs/analysis/facets.md"})
    if not anchors.discs:
        res.warnings.append("no labelled discs: facet joints not located")
        return res
    if not study.find("ax", None, localizer=False) and not study.find("sag", None, localizer=False):
        res.warnings.append("no axial or sagittal stack: facet joints not assessed")
        return res
    try:
        lm = level_map(study, anchors)
    except Exception as exc:  # the level map is shared; one failure must not raise out of the analysis
        log.warning("facets: level map failed: %s", exc)
        res.warnings.append(f"level map failed: {type(exc).__name__}: {exc}")
        return res
    sites = joint_sites(lm, anchors, cfg)
    used: Counter[str] = Counter()
    panels: dict[str, dict[str, tuple[JointMeasure, str]]] = {}
    if not any(st.plane == "ax" for st in study.stacks.values()):
        res.warnings.append("no axial stack: parasagittal fallback only (low confidence)")
    for label in anchors.disc_labels():
        if label not in sites:
            res.warnings.append(f"{label}: no level-map site; facet joints not assessed")
            continue
        for side in SIDES:
            site = sites[label][side]
            found = choose_stack(study, site.xyz, _refs(site, cfg))
            if found is None:
                res.rows.append(
                    {
                        "scope": "joint",
                        "level": label,
                        "side": side,
                        "effusion_present": None,
                        "confidence": 0.0,
                        "notes": "no axial or parasagittal stack covers the joint",
                    }
                )
                res.warnings.append(f"{label} {side} facet: no stack covers the joint")
                continue
            st, plane = found
            contrast, fat_sup = _contrast_of(st)
            m = measure_joint(st, plane, site, cfg)
            used[st.key] += 1
            conf = _confidence(plane, cfg)
            width = m.width_mm
            present: bool | None
            cyst = False
            notes: list[str] = []
            if width is None:
                present = None
                conf = 0.0
                notes.append("no CSF-muscle contrast in the sample: no fluid call")
            elif not fat_sup:
                # Plain T2: fat is as bright as fluid, so no call is made. The extent is kept for review only.
                present = None
                conf = 0.0
                notes.append("fat not suppressed (plain T2): fluid not called; extent recorded for review")
            elif m.roi_limited:
                # Fluid reaching the window edge is not a joint-sized collection (it is CSF, fat or another structure
                # outside the joint): the width is only a lower bound, so no call is made.
                present = None
                conf = 0.0
                notes.append("fluid reaches the window edge: not a joint-sized collection; no call")
            else:
                present = width >= cfg.facets_effusion_min_mm
                cyst = bool(
                    present
                    and width >= cfg.facets_cyst_min_mm
                    and m.roundness is not None
                    and m.roundness >= cfg.facets_cyst_roundness_min
                )
                if m.roi_limited:
                    notes.append("fluid touches the window edge: width is a lower bound")
                if plane == "sag":
                    notes.append("parasagittal fallback")
            row: dict[str, Any] = {
                "scope": "joint",
                "level": label,
                "side": side,
                "plane": plane,
                "stack": st.key,
                "contrast": contrast,
                "fat_suppressed": fat_sup,
                "site_xyz": [round(float(x), 2) for x in site.xyz],
                "lateral_mm": round(site.lateral_mm, 2),
                "csf": round(m.csf, 2) if np.isfinite(m.csf) else None,
                "muscle": round(m.muscle, 2) if np.isfinite(m.muscle) else None,
                "fluid_threshold": round(m.threshold, 2) if np.isfinite(m.threshold) else None,
                "csf_muscle_contrast": round(m.contrast, 3),
                "effusion_width_mm": round(width, 2) if (width is not None and present is not None) else None,
                "fluid_extent_mm": round(width, 2) if width is not None else None,
                "effusion_present": present,
                "fluid_ratio": round(m.fluid_ratio, 3) if m.fluid_ratio is not None else None,
                "roundness": round(m.roundness, 3) if m.roundness is not None else None,
                "roi_limited": m.roi_limited,
                "synovial_cyst_possible": cyst,
                "confidence": conf,
                "notes": "; ".join(notes),
            }
            res.rows.append(row)
            if present:
                res.flags.append(
                    {
                        "kind": "facet_effusion",
                        "level": label,
                        "side": side,
                        "width_mm": row["effusion_width_mm"],
                        "confidence": conf,
                        "wording": "joint fluid measured; consider reviewing the joint on fluid-sensitive images",
                    }
                )
            if cyst:
                res.flags.append(
                    {
                        "kind": "synovial_cyst_possible",
                        "level": label,
                        "side": side,
                        "width_mm": row["effusion_width_mm"],
                        "confidence": min(conf, 0.5),
                        "wording": "possible synovial cyst (round fluid collection), consider review",
                    }
                )
            if width is None:
                shown = "no fluid call (no contrast)"
            elif not fat_sup:
                shown = f"plain T2, not called ({width:.1f} mm)"
            elif m.roi_limited:
                shown = "fluid at window edge, not called"
            else:
                call = "present" if present else "absent"
                shown = f"fluid {width:.1f} mm, {call}"
            panels.setdefault(label, {})[side] = (m, f"{contrast or plane}: {shown}")
    try:  # dx21: image-based joint line, facet angle and morphology proxies (unvalidated; capped, see the doc)
        res.rows.extend(joint_geometry_rows(study, lm, anchors, cfg))
    except Exception as exc:  # the geometry rows never break the fluid path
        log.warning("facets: joint geometry failed: %s", exc)
        res.warnings.append(f"joint geometry failed: {type(exc).__name__}: {exc}")
    for label, per in panels.items():
        fig = _figure(label, per, cfg)
        if fig is not None:
            res.figures[_figure_name(label)] = fig
    if used:
        res.meta["series_key"] = used.most_common(1)[0][0]
    res.meta["thresholds"] = {
        k: getattr(cfg, k)
        for k in (
            "facets_site_fraction",
            "facets_fluid_frac",
            "facets_effusion_min_mm",
            "facets_cyst_min_mm",
            "facets_min_contrast",
        )
    }
    return res


def _figure_name(level: str) -> str:
    return "joints_" + level.replace("-", "_")


# ----------------------------------------------------------------------------- report wiring
def _joint_row(res: Result, level: str, side: str) -> dict[str, Any] | None:
    for r in res.rows:
        if r.get("scope") == "joint" and r.get("level") == level and r.get("side") == side:
            return r
    return None


def _effusion_extract(side: str) -> Extract:
    def fn(res: Result, level: str) -> Candidate | None:
        row = _joint_row(res, level, side)
        if row is None or row.get("effusion_present") is None:
            return None
        if row.get("stack") != res.meta.get("series_key"):
            return None  # the findings evidence cites one series; a level on another stack is not reported
        width = float(row.get("effusion_width_mm") or 0.0)
        fat = "fat-suppressed" if row.get("fat_suppressed") else "fat not suppressed"
        notes = (
            f"fluid width {width:.1f} mm on {row.get('contrast')} ({fat}); {row.get('notes') or ''}".rstrip(
                "; "
            )
        )
        value = "present" if row["effusion_present"] else "absent"
        return Candidate(value, float(row.get("confidence") or 0.0), notes)

    return fn


def _figure_for(level: str) -> str:
    return _figure_name(level)


_FIG: Callable[[str], str | None] = _figure_for
register_field(FieldRule("facet_L", "effusion", "facets", _effusion_extract("L"), figure=_FIG))
register_field(FieldRule("facet_R", "effusion", "facets", _effusion_extract("R"), figure=_FIG))


# ----------------------------------------------------------------------------- joint localisation, angle, morphology (dx21)
@dataclass(frozen=True)
class JointLocalisation:
    """Joint line found in the image. Centre and direction are in the plane: (u, v) mm offsets from the ROI origin.

    ``direction`` is the unit major axis of the joint-line component; ``length_mm`` its maximal caliper and
    ``width_mm`` its minimal caliper (the joint-line thickness, a proxy for joint space, not a measured joint space).
    """

    ok: bool
    centre_uv: tuple[float, float] | None
    direction: tuple[float, float] | None
    length_mm: float | None
    width_mm: float | None
    peak: float
    pixels: int
    reason: str


def dark_ridge_score(img: NDArray[np.float64], px: float, sigma_mm: float) -> NDArray[np.float64]:
    """Dark-ridge (dark cortical line) score per pixel, scale-normalised and relative to the image intensity range.

    Hessian of the Gaussian-smoothed image: a dark line has a large positive second derivative across it (``l1``)
    and a small one along it. Score = max(l1, 0) - 2 max(l2, 0), so blobs (both eigenvalues large) score low.
    """
    fin = np.isfinite(img)
    if not fin.any():
        return np.zeros(img.shape, dtype=np.float64)
    f = np.where(fin, img, float(np.nanmedian(img)))
    rng = float(np.nanpercentile(img, 99.5) - np.nanpercentile(img, 0.5))
    if rng <= 0:
        return np.zeros(img.shape, dtype=np.float64)
    s = max(sigma_mm / px, 0.5)
    hrr = ndi.gaussian_filter(f, s, order=(2, 0))  # d2/dv2 (rows)
    hcc = ndi.gaussian_filter(f, s, order=(0, 2))  # d2/du2 (cols)
    hrc = ndi.gaussian_filter(f, s, order=(1, 1))
    tr = 0.5 * (hrr + hcc)
    root = np.sqrt((0.5 * (hrr - hcc)) ** 2 + hrc**2)
    l1 = tr + root
    l2 = tr - root
    score = (np.clip(l1, 0.0, None) - 2.0 * np.clip(l2, 0.0, None)) * (s * s) / rng
    score[~fin] = 0.0
    return np.asarray(score, dtype=np.float64)


def _fail(reason: str) -> JointLocalisation:
    return JointLocalisation(False, None, None, None, None, 0.0, 0, reason)


def localise_joint(roi: NDArray[np.float64], px: float, cfg: Config = DEFAULT) -> JointLocalisation:
    """Joint line near the centre of ``roi`` (the geometric site) from the image alone.

    Peak of the dark-ridge score within ``facets_loc_search_mm`` of the centre; the joint line is the connected set of
    ridge pixels above ``facets_loc_component_frac`` of the peak, within ``facets_loc_max_radius_mm`` of the centre.
    Its centroid is the joint centre and its principal axis the joint direction. No mask or label is used.
    """
    nt, ns = roi.shape
    if not np.isfinite(roi).any():
        return _fail("no image data in the window")
    score = dark_ridge_score(roi, px, cfg.facets_loc_sigma_mm)
    tt = (np.arange(nt) - (nt - 1) / 2) * px
    ss = (np.arange(ns) - (ns - 1) / 2) * px
    ggs, ggt = np.meshgrid(ss, tt)
    radius = np.hypot(ggs, ggt)
    # the peak must lie in the joint-line disc too (else its label is 0 and the component is background)
    search = (radius <= cfg.facets_loc_search_mm) & (radius <= cfg.facets_loc_max_radius_mm)
    s_in = np.where(search, score, -np.inf)
    idx = np.unravel_index(int(np.argmax(s_in)), s_in.shape)
    peak = float(score[idx])
    if not np.isfinite(peak) or peak <= 0:
        return _fail("no dark ridge within the search radius")
    # limited to a radius of the site (not of the peak: on a long line the peak can sit at the edge of the search)
    lab, _ = ndi.label(
        (score >= cfg.facets_loc_component_frac * peak) & (radius <= cfg.facets_loc_max_radius_mm)
    )
    if lab[idx] == 0:  # cannot happen: the peak is above the threshold inside the disc
        return _fail("peak outside the joint-line component")
    comp = lab == lab[idx]
    n = int(comp.sum())
    if n < cfg.facets_loc_min_pixels:
        return _fail(f"joint-line component too small ({n} px)")
    rr, cc = np.nonzero(comp)
    pts = np.stack([ss[cc], tt[rr]], axis=1)  # (u, v) mm
    centre = pts.mean(axis=0)
    if n > 2:
        w, vecs = np.linalg.eigh(np.cov((pts - centre).T))
        major = vecs[:, int(np.argmax(w))]
    else:
        major = np.array([1.0, 0.0])
    cmax, cmin = _calipers(pts)
    return JointLocalisation(
        ok=True,
        centre_uv=(float(centre[0]), float(centre[1])),
        direction=(float(major[0]), float(major[1])),
        length_mm=cmax + px,
        width_mm=cmin + px,
        peak=peak,
        pixels=n,
        reason="ok",
    )


def joint_angle_deg(direction: tuple[float, float]) -> float:
    """Facet joint angle (deg, 0 to 90) in the axial plane: acute angle between the joint line and the mid-sagittal
    (AP) direction. Karacan et al. (2004) convention: the angle between a midsagittal reference line and the facet line
    on an axial image. Smaller means more sagittal, larger means more coronal. NaN for a zero vector.
    """
    du, dv = float(direction[0]), float(direction[1])
    n = float(np.hypot(du, dv))
    if n == 0:
        return float("nan")
    return float(np.degrees(np.arccos(min(1.0, abs(dv) / n))))


def tropism_flags(left_deg: float, right_deg: float) -> dict[str, Any]:
    """Left-right facet angle difference with both reported thresholds (7 and 10 degrees; see config comments).

    Neither threshold is validated here: both flags are descriptive, and the difference is always reported.
    """
    diff = abs(float(left_deg) - float(right_deg))
    return {
        "difference_deg": round(diff, 2),
        "tropism_7deg": bool(diff > DEFAULT.facets_tropism_deg),
        "tropism_10deg": bool(diff > DEFAULT.facets_tropism_strict_deg),
        "threshold_deg": DEFAULT.facets_tropism_deg,
        "threshold_strict_deg": DEFAULT.facets_tropism_strict_deg,
    }


def roi_grid_mm(
    origin: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    shape: tuple[int, int],
    px: float,
) -> NDArray[np.float64]:
    """LPS points (nt, ns, 3) of an in-plane grid centred on ``origin`` (same convention as :func:`measure_joint`)."""
    nt, ns = shape
    tt = (np.arange(nt) - (nt - 1) / 2) * px
    ss = (np.arange(ns) - (ns - 1) / 2) * px
    return (
        np.asarray(origin, dtype=np.float64)[None, None, :]
        + ss[None, :, None] * np.asarray(u, dtype=np.float64)[None, None, :]
        + tt[:, None, None] * np.asarray(v, dtype=np.float64)[None, None, :]
    )


def sample_label_plane(
    st: Stack,
    k: int,
    label2d: NDArray[Any],
    origin: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    shape: tuple[int, int],
    px: float,
) -> NDArray[np.int64]:
    """Nearest-pixel label of a 2-D label map on slice ``k`` sampled on the in-plane grid of :func:`roi_grid_mm`.

    ``label2d`` must share the slice's pixel grid (true for the Sudirman label maps). Outside the grid: -1.
    """
    pts = roi_grid_mm(origin, u, v, shape, px)
    o = st.pix_to_lps(float(k), 0.0, 0.0)
    d = pts - o[None, None, :]
    c = np.rint(d @ np.asarray(st.row_dir, dtype=np.float64) / st.col_mm).astype(np.int64)
    r = np.rint(d @ np.asarray(st.col_dir, dtype=np.float64) / st.row_mm).astype(np.int64)
    rows, cols = label2d.shape
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    out = np.full(shape, -1, dtype=np.int64)
    out[inside] = label2d[r[inside], c[inside]].astype(np.int64)
    return out


def geometry_at(
    st: Stack, site: JointSite, cfg: Config = DEFAULT, slice_k: int | None = None
) -> dict[str, Any] | None:
    """Image-based joint localisation, facet angle and morphology proxies for one site on one axial stack.

    The site is projected onto the slice nearest to it; the ROI is sampled in that slice's plane. Returns None for a
    non-axial stack (joint lines and angles are defined on axial images only). ``slice_k`` evaluates on that slice
    instead of the nearest one (validation: the annotated slice).
    """
    if st.plane != "ax":
        return None
    k, dist = st.nearest_slice(site.xyz)
    if slice_k is not None:
        k = int(slice_k)
        dist = abs(
            float((site.xyz - st.pix_to_lps(float(k), 0.0, 0.0)) @ np.asarray(st.normal, dtype=np.float64))
        )
    n = np.asarray(st.normal, dtype=np.float64)
    o_k = st.pix_to_lps(float(k), 0.0, 0.0)
    origin = site.xyz - float((site.xyz - o_k) @ n) * n
    px = cfg.facets_px_mm
    roi = _patch(st, origin, _X, _Y, cfg.facets_roi_half_mm, px)
    loc = localise_joint(roi, px, cfg)
    row: dict[str, Any] = {
        "scope": "joint_geometry",
        "level": site.level,
        "side": site.side,
        "stack": st.key,
        "slice": int(k),
        "out_of_plane_mm": round(float(dist), 2),
        "contrast": _contrast_of(st)[0],
        "loc_ok": loc.ok,
        "loc_reason": loc.reason,
        "loc_peak": round(loc.peak, 3),
        "loc_pixels": loc.pixels,
        "origin_xyz": [round(float(x), 2) for x in origin],
        "joint_uv_mm": None,
        "joint_angle_deg": None,
        "joint_line_length_mm": None,
        "joint_line_width_mm": None,
        "weishaupt_grade": None,
        "grade_status": "not assessed",
        # experimental: unvalidated until the dx21 gates pass (docs/analysis/facets.md section 6.3); capped
        "experimental": True,
        "confidence": round(cfg.facets_morph_max_confidence, 3) if loc.ok else 0.0,
    }
    if loc.ok and loc.centre_uv is not None and loc.direction is not None:
        row["joint_uv_mm"] = [round(loc.centre_uv[0], 2), round(loc.centre_uv[1], 2)]
        row["joint_angle_deg"] = round(joint_angle_deg(loc.direction), 2)
        row["joint_line_length_mm"] = round(float(loc.length_mm or 0.0), 2)
        row["joint_line_width_mm"] = round(float(loc.width_mm or 0.0), 2)
    row["grade_reason"] = (
        "Weishaupt grade needs osteophyte, hypertrophy and erosion criteria that are not measured here; "
        "the joint-line measures are unvalidated proxies"
    )
    return row


def joint_geometry_rows(
    study: Study, lm: LevelMap, anchors: Anchors, cfg: Config = DEFAULT
) -> list[dict[str, Any]]:
    """Geometry rows (see :func:`geometry_at`) for every joint with an axial stack that covers it."""
    sites = joint_sites(lm, anchors, cfg)
    rows: list[dict[str, Any]] = []
    for label in anchors.disc_labels():
        for side in SIDES:
            if label not in sites or side not in sites[label]:
                continue
            site = sites[label][side]
            found = choose_stack(study, site.xyz, _refs(site, cfg))
            axial = [
                st
                for st in study.stacks.values()
                if st.plane == "ax" and _covers(st, site.xyz) and not st.meta.get("role", {}).get("localizer")
            ]
            if not axial:
                continue
            st = min(axial, key=lambda s: (_contrast_of(s)[0] != "t2", s.key))
            geo = geometry_at(st, site, cfg)
            if geo is not None:
                geo["fluid_stack_key"] = found[0].key if found is not None else None
                rows.append(geo)
    return rows
