"""Level correspondence: which slices of every stack show each labelled disc and vertebra.

Labelling (:func:`spinescope.anatomy.label_spine`) fixes where each disc and vertebra is, in patient LPS mm. Downstream
analysis (discs, endplates, canal, foramina, facets, muscles) and AI readers then need the right *slices*: the
sagittal slice through the level, the axial slab angled to the disc, the axial slices that cut through the disc's
thickness, and the parasagittal slices through the foramina and facets. :func:`level_map` computes that for every
stack of a study from the anchors and each stack's exact geometry (:mod:`spinescope.geometry`).

Nothing is fitted to one scan. Lengths come from the anchors when they are measured (vertebral body width and depth,
disc height, canal centreline and AP diameter, see :mod:`spinescope.anatomy.vertebrae`); otherwise they come from the
defaults in :class:`CorrespondenceConfig`, and the entry says so (``geometry_default``). The foramen and facet points
are anatomical approximations: no ground truth in the public data locates them, so they are checked for geometry
(the phantom) and not for anatomy. See ``docs/labeling/lab29_correspondence.md``.

Conventions. ``out_of_plane_mm`` is the target minus the slice plane along the stack normal: positive means the
target lies toward higher slice indices. ``pixel`` is ``[row, col]`` of the target projected onto the nearest slice (it
may lie outside the image; ``in_fov`` says whether it does). ``intersects`` lists the slices whose plane cuts the
target body (a cylinder: radius = half body width, thickness = disc or body height, axis = its normal).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import DEFAULT
from ..geometry import Stack
from ..study import Study
from .anchors import Anchors, Disc, Vertebra, split_disc

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: Schema of :meth:`LevelMap.to_dict`. Additive changes only; bump the minor part for new keys.
SCHEMA_VERSION = "1.0"

#: Flags that state coverage facts rather than quality problems (they do not change ``quality``).
INFO_FLAGS = frozenset({"no_axial_slab", "no_axial_slab_for_level", "no_parasagittal_stack"})

_X = np.array([1.0, 0.0, 0.0])


def _mid(pair: tuple[float, float]) -> float:
    return float(0.5 * (pair[0] + pair[1]))


@dataclass(frozen=True)
class CorrespondenceConfig:
    """Settings of :func:`level_map`. Geometric fractions are judgement defaults; each is documented below."""

    #: Stack-to-reference angle (deg) within which a stack counts as aligned with a level. Same value as the axial
    #: slab matcher (``AxialConfig.slab_match_angle_deg``); Sudirman slabs are angled to within a few degrees.
    tilt_tol_deg: float = 12.0
    #: Lumbar body width (mm) used only when the anchors carry no body geometry (flagged ``geometry_default``).
    #: Adult lumbar bodies are about 40 to 50 mm wide in anatomy texts; the midpoint is used.
    default_body_width_mm: float = 44.0
    #: Body AP depth / width when unmeasured. Adult lumbar bodies are slightly shallower than wide (about 0.7 to 0.8).
    body_depth_to_width: float = 0.75
    #: Canal transverse width / AP diameter (judgement: the lumbar canal is wider than it is deep, about 1.2 to 1.4).
    canal_transverse_to_ap: float = 1.3
    #: Canal half width / body half width when the canal AP diameter is unmeasured (judgement, about 0.4).
    canal_to_body_half_width: float = 0.4
    #: Foramen lateral position between the canal wall (0) and the body edge (1). 0.5: midway, the foramen lies
    #: between the canal and the pedicle (judgement).
    foramen_fraction: float = 0.5
    #: Facet joint lateral position between the canal wall (0) and the body edge (1). Facets lie close to the canal
    #: (judgement).
    facet_fraction: float = 0.2
    #: Foramen AP position as a fraction of the disc half depth from the disc centre; 2/3 is the posterior third.
    foramen_ap_fraction: float = 2.0 / 3.0
    #: Foramen superior offset from the disc centre as a fraction of disc height (the exiting root lies above the disc).
    foramen_z_fraction: float = 0.25
    #: Slab candidates are restricted to these contrasts when given. None: every axial slab.
    slab_contrasts: tuple[str, ...] | None = None
    #: Among slabs with the same geometry (see the two tolerances below), the earlier contrast here wins.
    contrast_preference: tuple[str, ...] = ("t2", "t1", "stir", "pd")
    #: A slab whose block contains the disc centre, but whose normal is off by more than ``tilt_tol_deg``, is still
    #: assigned when it is within this angle (twice the tolerance), with the flag ``slab_tilt_mismatch``. The labeller's
    #: disc normal, not the slab, is the likely error there (a slab through the disc centre is that disc's slab).
    #: Set to ``tilt_tol_deg`` to disable the fallback (strict mode).
    slab_fallback_deg: float = 24.0
    #: Two slabs have the same geometry when their normals differ by at most this many degrees ...
    slab_geometry_tol_deg: float = 1.0
    #: ... and their offsets along the normal by at most this many mm.
    slab_geometry_tol_mm: float = 2.0
    #: Anchor confidence below this is flagged ``low_confidence`` and the entry's quality is ``poor``.
    low_confidence: float = 0.5


#: Fallbacks taken from the ranges in :class:`spinescope.config.Config` (midpoints), used when unmeasured.
_FALLBACK_DISC_HEIGHT_MM = _mid(DEFAULT.disc_height_mm)
_FALLBACK_BODY_HEIGHT_MM = _mid(DEFAULT.body_height_mm)
_FALLBACK_CANAL_AP_MM = _mid(DEFAULT.canal_ap_mm)


@dataclass
class StackHit:
    """Where one target (disc, vertebra, canal, foramen or facet) sits in one stack."""

    key: str
    plane: str
    slice: int  # nearest slice index
    out_of_plane_mm: float  # signed, see module docs
    in_slab: bool  # target inside the stack's slab (slice range, half a slice of margin)
    in_fov: bool  # projected target inside the in-plane field of view
    pixel: list[float]  # [row, col] of the target on the nearest slice
    intersects: list[int]  # slices whose plane cuts the target body (discs and vertebrae only)
    tilt_deg: float  # angle between the stack normal and the target's reference axis
    tilt_ref: str  # "disc normal" | "disc lateral axis" | "disc AP axis"
    flags: list[str] = field(default_factory=list)


@dataclass
class LevelEntry:
    """A disc or vertebra of the map, with its hit in every stack and, for discs, its slab and foramen targets."""

    label: str
    kind: str  # "disc" | "vertebra"
    centre: list[float]
    normal: list[float]  # unit, superior
    radius_mm: float  # in-plane half width of the body (or disc) used for intersections
    thickness_mm: float  # disc height or body height
    confidence: float
    stacks: dict[str, StackHit] = field(default_factory=dict)
    axials: list[str] = field(default_factory=list)  # axial stacks whose slices cut the body
    slab: str | None = None  # discs: the axial slab angled to this disc (None if no slab covers it)
    slab_candidates: list[dict[str, Any]] = field(default_factory=list)  # discs: every slab, scored
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)  # discs: canal, foramina, facets
    geometry: dict[str, Any] = field(default_factory=dict)  # the measures used and where each came from
    flags: list[str] = field(default_factory=list)
    quality: str = "ok"  # ok | check | poor


@dataclass
class LevelMap:
    """Slice correspondence for one study. JSON-serialisable with :meth:`to_dict`; contains no paths or identifiers."""

    discs: dict[str, LevelEntry]
    vertebrae: dict[str, LevelEntry]
    stacks: dict[str, dict[str, Any]]  # technical description of every stack (no identifiers)
    slab_for_level: dict[str, str | None]
    level_for_slab: dict[str, str | None]
    midline: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def level(self, label: str) -> LevelEntry:
        """The disc or vertebra entry called ``label`` (``L4-L5`` or ``L4``); KeyError if absent."""
        if label in self.discs:
            return self.discs[label]
        if label in self.vertebrae:
            return self.vertebrae[label]
        raise KeyError(f"no level {label!r} in the map")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ small helpers


def _angle_deg(a: FloatArray, b: FloatArray) -> float:
    """Unsigned angle in degrees between two unit vectors (sign of the normal is irrelevant)."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


def _unit(v: FloatArray) -> FloatArray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _is_localizer(st: Stack) -> bool:
    return bool(st.meta.get("role", {}).get("localizer"))


def _is_slab(st: Stack) -> bool:
    return st.plane == "ax" and st.meta.get("role", {}).get("slab_index") is not None


def _contrast(st: Stack) -> str | None:
    value = st.meta.get("role", {}).get("contrast")
    return None if value is None else str(value)


def _stack_info(st: Stack) -> dict[str, Any]:
    """Technical description of a stack. No description text or identifiers."""
    role = st.meta.get("role", {})
    return {
        "plane": st.plane,
        "contrast": role.get("contrast"),
        "localizer": bool(role.get("localizer")),
        "slab_index": role.get("slab_index"),
        "slices": st.n,
        "spacing_mm": round(st.spacing, 2),
        "thickness_mm": round(st.thickness, 2),
        "fov_mm": [round(float(v), 1) for v in st.extent_mm()],
        "obliquity_deg": round(st.obliquity, 1),
        "normal": [round(float(v), 4) for v in st.normal],
    }


def _has_geometry(anchors: Anchors) -> bool:
    return anchors.canal is not None or any(v.body for v in anchors.vertebrae.values())


def _with_geometry(study: Study, anchors: Anchors, warnings: list[str]) -> Anchors:
    """Anchors with the vertebral and canal geometry measured on a copy (the input is never changed)."""
    from .vertebrae import attach_geometry

    measured = Anchors.from_dict(anchors.to_dict())
    try:
        attach_geometry(study, measured)
    except Exception as exc:  # geometry is optional: the map falls back to defaults and says so
        warnings.append(f"geometry not measured on demand ({type(exc).__name__}); defaults used")
        return anchors
    if not _has_geometry(measured):
        warnings.append("geometry not measured on demand; defaults used")
    return measured


def _midline(anchors: Anchors) -> tuple[float | None, str]:
    if anchors.midline_x is not None:
        return float(anchors.midline_x), "anchors"
    return None, "unknown (the level's own centre is used)"


def _body_dims(anchors: Anchors, names: tuple[str, ...], cfg: CorrespondenceConfig) -> dict[str, Any]:
    """Half width, half AP depth and height of the bodies in ``names``, measured where the anchors have them."""
    widths: list[float] = []
    depths: list[float] = []
    heights: list[float] = []
    for name in names:
        v = anchors.vertebrae.get(name)
        body = v.body if v is not None else None
        if not body:
            continue
        if body.get("width_mm"):
            widths.append(float(body["width_mm"]))
        if body.get("ap_depth_mm"):
            depths.append(float(body["ap_depth_mm"]))
        h = body.get("height_mm")
        if isinstance(h, dict) and h.get("mean"):
            heights.append(float(h["mean"]))
    width = float(np.mean(widths)) if widths else cfg.default_body_width_mm
    depth = float(np.mean(depths)) if depths else width * cfg.body_depth_to_width
    height = float(np.mean(heights)) if heights else _FALLBACK_BODY_HEIGHT_MM
    if widths and depths and heights:
        source = "measured"
    elif widths:
        source = "partly measured"
    else:
        source = "default"
    return {"half_width_mm": width / 2.0, "half_depth_mm": depth / 2.0, "height_mm": height, "source": source}


def _canal_point(anchors: Anchors, z: float) -> tuple[float, float, str] | None:
    """(x, y) of the canal centreline at height ``z`` (linear in z) and how it was obtained; None if no canal."""
    canal = anchors.canal or {}
    pts = np.asarray(canal.get("points") or [], dtype=np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return None
    order = np.argsort(pts[:, 2], kind="stable")
    zs, xs, ys = pts[order, 2], pts[order, 0], pts[order, 1]
    if zs[-1] - zs[0] <= 0:
        return None
    src = "centreline"
    if z < zs[0] or z > zs[-1]:
        src = "centreline (extrapolated)"
    return float(np.interp(z, zs, xs)), float(np.interp(z, zs, ys)), src


def _canal_ap(anchors: Anchors, label: str) -> float | None:
    per_level = (anchors.canal or {}).get("per_level", {})
    entry = per_level.get(label) or {}
    value = entry.get("ap_mm")
    return None if value is None else float(value)


def _adjacent_normal(anchors: Anchors, name: str) -> FloatArray:
    """Mean normal of the discs above and below a vertebra (superior-pointing); the body axis when no disc is near."""
    normals = []
    for lab, d in anchors.discs.items():
        try:
            upper, lower = split_disc(lab)
        except ValueError:
            continue
        if name in (upper, lower):
            normals.append(np.asarray(d.normal, dtype=np.float64))
    if not normals:
        return np.array([0.0, 0.0, 1.0])
    n = np.mean(normals, axis=0)
    return _unit(n)


# -------------------------------------------------------------------- hits


def _hit(
    st: Stack,
    p: FloatArray,
    frame: tuple[FloatArray, FloatArray, FloatArray],
    radius: float,
    thickness: float,
    *,
    cfg: CorrespondenceConfig,
    intersects: bool,
) -> StackHit:
    """Where point ``p`` sits in stack ``st``; ``frame`` is (normal, lateral, AP) of the target's own level."""
    n_t, lat, ap = frame
    ns = st.normal
    pos = st.positions
    dp = float(p @ ns)
    k = int(np.argmin(np.abs(pos - dp)))
    out = dp - float(pos[k])
    half = 0.5 * max(st.spacing, st.thickness, 1e-9)
    in_slab = bool(pos[0] - half <= dp <= pos[-1] + half)
    rows, cols = st.vol.shape[1:]

    def pixel_on(j: int) -> tuple[float, float]:
        proj = p - (dp - float(pos[j])) * ns
        rel = proj - st.ipp[j]
        return float(rel @ st.col_dir / st.row_mm), float(rel @ st.row_dir / st.col_mm)

    r, c = pixel_on(k)
    in_fov = bool(-0.5 <= r <= rows - 0.5 and -0.5 <= c <= cols - 0.5)
    cuts: list[int] = []
    if intersects:
        cos_t = abs(float(n_t @ ns))
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        support = 0.5 * thickness * cos_t + radius * sin_t  # half extent of the disc along the slice normal
        for j in range(st.n):
            e = float(pos[j]) - dp
            if abs(e) > support + 1e-9:
                continue
            rj, cj = pixel_on(j)
            pad_r = radius / st.row_mm if st.row_mm > 0 else 0.0
            pad_c = radius / st.col_mm if st.col_mm > 0 else 0.0
            if -0.5 - pad_r <= rj <= rows - 0.5 + pad_r and -0.5 - pad_c <= cj <= cols - 0.5 + pad_c:
                cuts.append(j)
    ref, ref_name = {
        "ax": (n_t, "disc normal"),
        "sag": (lat, "disc lateral axis"),
        "cor": (ap, "disc AP axis"),
    }.get(st.plane, (n_t, "disc normal"))
    tilt = _angle_deg(ns, ref)
    flags: list[str] = []
    if not in_slab:
        flags.append("off_slab")
    elif not in_fov:
        flags.append("off_fov")
    if intersects and st.plane == "ax" and not cuts:
        flags.append("no_intersection")
    if abs(out) > radius and not cuts:
        flags.append("slice_far")
    if in_slab and in_fov and tilt > cfg.tilt_tol_deg:
        flags.append("tilt_mismatch")
    return StackHit(
        key=st.key,
        plane=st.plane,
        slice=k,
        out_of_plane_mm=round(out, 2),
        in_slab=in_slab,
        in_fov=in_fov,
        pixel=[round(r, 1), round(c, 1)],
        intersects=cuts,
        tilt_deg=round(tilt, 1),
        tilt_ref=ref_name,
        flags=flags,
    )


def _hits_for(
    study: Study,
    p: FloatArray,
    frame: tuple[FloatArray, FloatArray, FloatArray],
    radius: float,
    thickness: float,
    cfg: CorrespondenceConfig,
    *,
    intersects: bool,
) -> dict[str, StackHit]:
    return {
        key: _hit(st, p, frame, radius, thickness, cfg=cfg, intersects=intersects)
        for key, st in study.stacks.items()
    }


# ----------------------------------------------------------------- slabs


def _slab_candidates(
    centre: FloatArray, normal: FloatArray, height: float, slabs: list[Stack], cfg: CorrespondenceConfig
) -> list[dict[str, Any]]:
    """Every slab scored against one disc: its angle to the disc normal, the centre's offset along the slab normal,
    whether the disc centre lies inside the slab block, and whether it therefore covers the disc."""
    rows: list[dict[str, Any]] = []
    for s in slabs:
        ang = _angle_deg(normal, s.normal)
        off = float(centre @ s.normal) - float(s.positions.mean())
        in_block = bool(s.contains(centre, slab_margin_mm=0.5 * height))
        rows.append(
            {
                "key": s.key,
                "angle_deg": ang,
                "offset_mm": off,
                "in_block": in_block,
                "covers": bool(ang <= cfg.tilt_tol_deg and in_block),
                "contrast": _contrast(s),
            }
        )
    return rows


def _pref_rank(contrast: str | None, cfg: CorrespondenceConfig) -> int:
    prefs = cfg.contrast_preference
    return prefs.index(contrast) if contrast in prefs else len(prefs)


def _choose_slab(cands: list[dict[str, Any]], cfg: CorrespondenceConfig) -> dict[str, Any] | None:
    """The covering slab: the best geometry, and among slabs with that same geometry the preferred contrast.

    With no covering slab, the slab whose block holds the disc centre within ``slab_fallback_deg`` is taken, marked
    ``fallback``. Its normal disagrees with the disc's; the caller flags it.
    """
    cov = [c for c in cands if c["covers"]]
    if not cov:
        near = [c for c in cands if c["in_block"] and c["angle_deg"] <= cfg.slab_fallback_deg]
        if not near:
            return None
        pick = min(near, key=lambda c: (c["angle_deg"], abs(c["offset_mm"])))
        return {**pick, "fallback": True}
    best = min(cov, key=lambda c: (c["angle_deg"], abs(c["offset_mm"])))
    group = [
        c
        for c in cov
        if c["angle_deg"] <= best["angle_deg"] + cfg.slab_geometry_tol_deg
        and abs(c["offset_mm"] - best["offset_mm"]) <= cfg.slab_geometry_tol_mm
    ]
    pick = min(group, key=lambda c: (_pref_rank(c["contrast"], cfg), c["angle_deg"], abs(c["offset_mm"])))
    return {**pick, "fallback": False}


# ----------------------------------------------------------------- entries


def _disc_frame(d: Disc) -> tuple[FloatArray, FloatArray, FloatArray]:
    n = np.asarray(d.normal, dtype=np.float64)
    n = _unit(n)
    lat = _unit(_X - float(_X @ n) * n)
    ap = _unit(np.asarray(d.ap_dir, dtype=np.float64))
    return n, lat, ap


def _disc_targets(
    study: Study,
    label: str,
    d: Disc,
    anchors: Anchors,
    cfg: CorrespondenceConfig,
    body: dict[str, Any],
    midline_x: float | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Canal, foramen and facet points at a disc level, each with its hits in the sagittal and axial stacks."""
    c = np.asarray(d.center, dtype=np.float64)
    n, lat, ap = _disc_frame(d)
    height = float(d.height_mm) if d.height_mm else _FALLBACK_DISC_HEIGHT_MM
    half_w, half_d = body["half_width_mm"], body["half_depth_mm"]
    x_mid = midline_x if midline_x is not None else float(c[0])
    geo: dict[str, Any] = {
        "midline_x_mm": round(x_mid, 2),
        "disc_height_mm": round(height, 2),
        "disc_height_source": "measured" if d.height_mm else "default",
    }
    canal_ap = _canal_ap(anchors, label)
    geo["canal_ap_mm"] = round(canal_ap if canal_ap is not None else _FALLBACK_CANAL_AP_MM, 2)
    geo["canal_ap_source"] = "measured" if canal_ap is not None else "default"
    # canal centre: centreline when measured, else the posterior body margin plus half the canal AP diameter
    cp = _canal_point(anchors, float(c[2]))
    if cp is not None:
        canal_xyz = np.array([cp[0], cp[1], float(c[2])])
        geo["canal_centre_source"] = cp[2]
    else:
        ap_len = half_d + (canal_ap if canal_ap is not None else _FALLBACK_CANAL_AP_MM) / 2.0
        canal_xyz = c + ap * ap_len
        canal_xyz[0] = x_mid
        geo["canal_centre_source"] = "formula (body depth and canal AP)"
    # canal half width: transverse width from the AP diameter when measured, else from the body width
    if canal_ap is not None:
        canal_half = 0.5 * canal_ap * cfg.canal_transverse_to_ap
        geo["canal_half_width_source"] = "AP diameter x transverse ratio"
    else:
        canal_half = half_w * cfg.canal_to_body_half_width
        geo["canal_half_width_source"] = "body width x ratio (default)"
    canal_half = min(canal_half, 0.9 * half_w)
    geo["canal_half_width_mm"] = round(canal_half, 2)
    geo["body_half_width_mm"] = round(half_w, 2)
    geo["body_source"] = body["source"]
    d_for = canal_half + cfg.foramen_fraction * (half_w - canal_half)
    d_fac = canal_half + cfg.facet_fraction * (half_w - canal_half)
    geo["foramen_lateral_mm"] = round(d_for, 2)
    geo["facet_lateral_mm"] = round(d_fac, 2)
    foramen_base = (
        np.array([x_mid, c[1], c[2]])
        + ap * (half_d * cfg.foramen_ap_fraction)
        + n * (height * cfg.foramen_z_fraction)
    )
    points: dict[str, tuple[FloatArray, float]] = {
        "canal": (canal_xyz, 0.0),
        "left_foramen": (foramen_base + lat * d_for, d_for),
        "right_foramen": (foramen_base - lat * d_for, d_for),
        "left_facet": (canal_xyz + lat * d_fac, d_fac),
        "right_facet": (canal_xyz - lat * d_fac, d_fac),
    }
    targets: dict[str, dict[str, Any]] = {}
    planar = {k: st for k, st in study.stacks.items() if st.plane in ("sag", "ax")}
    for name, (p, lateral) in points.items():
        hits = {
            key: _hit(st, p, (n, lat, ap), half_w, height, cfg=cfg, intersects=False)
            for key, st in planar.items()
        }
        # a sagittal slice covers the target plane when the target lies within half a slice thickness of it
        covered = {
            key: h.slice
            for key, h in hits.items()
            if study.stacks[key].plane == "sag"
            and not _is_localizer(study.stacks[key])
            and h.in_slab
            and h.in_fov
            and abs(h.out_of_plane_mm)
            <= 0.5 * (study.stacks[key].thickness or study.stacks[key].spacing or 1.0)
        }
        flags = [] if covered else ["no_parasagittal_stack"]
        entry: dict[str, Any] = {
            "point": [round(float(v), 2) for v in p],
            "lateral_mm": round(lateral, 2),
            "covered_by": dict(sorted(covered.items())),
            "flags": flags,
            "stacks": {k: asdict(h) for k, h in hits.items()},
        }
        targets[name] = entry
    return targets, geo


def _disc_entry(
    study: Study,
    label: str,
    d: Disc,
    anchors: Anchors,
    cfg: CorrespondenceConfig,
    midline_x: float | None,
    chosen: dict[str, Any] | None,
    slab_conflict: str | None,
    tilt_best: dict[str, Any] | None,
    has_slabs: bool,
) -> LevelEntry:
    c = np.asarray(d.center, dtype=np.float64)
    n, lat, ap = _disc_frame(d)
    height = float(d.height_mm) if d.height_mm else _FALLBACK_DISC_HEIGHT_MM
    try:
        upper, lower = split_disc(label)
    except ValueError:
        upper, lower = label.split("-")[0], label.split("-")[-1]
    body = _body_dims(anchors, (upper, lower), cfg)
    radius = body["half_width_mm"]
    targets, geo = _disc_targets(study, label, d, anchors, cfg, body, midline_x)
    geo["body"] = {k: body[k] for k in ("half_width_mm", "half_depth_mm", "height_mm", "source")}
    stacks = _hits_for(study, c, (n, lat, ap), radius, height, cfg, intersects=True)
    defaults = [
        body["source"] != "measured",
        geo["disc_height_source"] != "measured",
        geo["canal_ap_source"] != "measured",
        geo["canal_half_width_source"].endswith("(default)"),
        midline_x is None,
    ]
    axials = sorted(k for k, h in stacks.items() if study.stacks[k].plane == "ax" and h.intersects)
    flags: list[str] = []
    if d.confidence < cfg.low_confidence:
        flags.append("low_confidence")
    relevant = [
        h
        for k, h in stacks.items()
        if study.stacks[k].plane == "sag" and not _is_localizer(study.stacks[k]) and h.in_slab and h.in_fov
    ]
    if not relevant:
        flags.append("no_sagittal")
    if any(defaults):
        flags.append("geometry_default")
    if midline_x is None:
        flags.append("midline_unknown")
    if slab_conflict is not None:
        flags.append("slab_conflict")
    if (chosen is None or chosen.get("fallback")) and tilt_best is not None:
        flags.append("slab_tilt_mismatch")
    if not has_slabs:
        flags.append("no_axial_slab")
    elif chosen is None and slab_conflict is None:
        flags.append("no_axial_slab_for_level")
    if any("tilt_mismatch" in h.flags for h in relevant):
        flags.append("tilt_mismatch")
    if any("slice_far" in h.flags for h in relevant):
        flags.append("slice_far")
    quality = _quality(flags, poor=("low_confidence", "no_sagittal"))
    return LevelEntry(
        label=label,
        kind="disc",
        centre=[round(float(v), 2) for v in c],
        normal=[round(float(v), 4) for v in n],
        radius_mm=round(radius, 2),
        thickness_mm=round(height, 2),
        confidence=round(float(d.confidence), 3),
        stacks=stacks,
        axials=axials,
        slab=None if chosen is None else chosen["key"],
        slab_candidates=[],
        targets=targets,
        geometry=geo,
        flags=sorted(set(flags)),
        quality=quality,
    )


def _quality(flags: list[str], poor: tuple[str, ...]) -> str:
    if any(f in flags for f in poor):
        return "poor"
    if any(f not in INFO_FLAGS for f in flags):
        return "check"
    return "ok"


def _vertebra_entry(
    study: Study, name: str, v: Vertebra, anchors: Anchors, cfg: CorrespondenceConfig
) -> LevelEntry:
    c = np.asarray(v.center, dtype=np.float64)
    n = _adjacent_normal(anchors, name)
    lat = _unit(_X - float(_X @ n) * n)
    ap = _unit(np.cross(n, lat))
    body = _body_dims(anchors, (name,), cfg)
    stacks = _hits_for(study, c, (n, lat, ap), body["half_width_mm"], body["height_mm"], cfg, intersects=True)
    flags: list[str] = []
    if v.confidence < cfg.low_confidence:
        flags.append("low_confidence")
    relevant = [
        h
        for k, h in stacks.items()
        if study.stacks[k].plane == "sag" and not _is_localizer(study.stacks[k]) and h.in_slab and h.in_fov
    ]
    if not relevant:
        flags.append("no_sagittal")
    if body["source"] != "measured":
        flags.append("geometry_default")
    if any("tilt_mismatch" in h.flags for h in relevant):
        flags.append("tilt_mismatch")
    axials = sorted(k for k, h in stacks.items() if study.stacks[k].plane == "ax" and h.intersects)
    return LevelEntry(
        label=name,
        kind="vertebra",
        centre=[round(float(x), 2) for x in c],
        normal=[round(float(x), 4) for x in n],
        radius_mm=round(body["half_width_mm"], 2),
        thickness_mm=round(body["height_mm"], 2),
        confidence=round(float(v.confidence), 3),
        stacks=stacks,
        axials=axials,
        geometry={"body": {k: body[k] for k in ("half_width_mm", "half_depth_mm", "height_mm", "source")}},
        flags=sorted(set(flags)),
        quality=_quality(flags, poor=("low_confidence", "no_sagittal")),
    )


# ------------------------------------------------------------------- public


def level_map(
    study: Study,
    anchors: Anchors,
    config: CorrespondenceConfig | None = None,
    *,
    measure_geometry: bool = True,
) -> LevelMap:
    """Which slices of every stack show each labelled disc and vertebra.

    For each disc: the hit in every stack (nearest slice, out-of-plane distance, inside the slab and field of view,
    the disc centre's pixel, the slices that cut the disc); the axial slab angled to it (chosen one-to-one, the
    best-fitting geometry first, then the preferred contrast); the canal, left/right foramen and facet points with the
    sagittal stacks that cover them; and quality flags (``slice_far``, ``tilt_mismatch``, ``slab_tilt_mismatch``,
    ``slab_conflict``, ``no_sagittal``, ``geometry_default`` and others). Vertebrae get the same stack hits.

    ``measure_geometry``: when the anchors lack vertebral body and canal geometry (the classical labeller, or
    anchors read from an old file), measure it on a copy of the anchors first (runs the learned labeller once more;
    the input is never changed). Without it, the defaults in ``config`` are used and flagged. Never raises for a
    study without sagittal or axial stacks: those cases produce warnings and flags.
    """
    cfg = config or CorrespondenceConfig()
    warnings: list[str] = []
    if not anchors.discs:
        warnings.append("no discs in the anchors: the map has no disc entries")
    if measure_geometry and anchors.discs and not _has_geometry(anchors):
        anchors = _with_geometry(study, anchors, warnings)
    midline_x, midline_src = _midline(anchors)
    if midline_x is None:
        warnings.append("midline unknown: foramen and canal x positions use the level centre")
    slabs = [
        s
        for s in study.stacks.values()
        if _is_slab(s) and (cfg.slab_contrasts is None or _contrast(s) in cfg.slab_contrasts)
    ]
    has_slabs = any(_is_slab(s) for s in study.stacks.values())
    if not has_slabs:
        warnings.append("no axial slabs in the study: slab_for_level is empty (sagittal-only study)")

    # 1. slab choice per disc, then one slab per disc (greedy: covering slabs first, then by geometry quality)
    cands: dict[str, list[dict[str, Any]]] = {}
    chosen: dict[str, dict[str, Any] | None] = {}
    tilt_best: dict[str, dict[str, Any] | None] = {}
    discs_in = list(anchors.disc_labels())
    for lab in discs_in:
        d = anchors.discs[lab]
        height = float(d.height_mm) if d.height_mm else _FALLBACK_DISC_HEIGHT_MM
        rows = _slab_candidates(
            np.asarray(d.center, dtype=np.float64),
            _unit(np.asarray(d.normal, dtype=np.float64)),
            height,
            slabs,
            cfg,
        )
        cands[lab] = rows
        chosen[lab] = _choose_slab(rows, cfg)
        blocked = [r for r in rows if r["in_block"] and not r["covers"]]
        tilt_best[lab] = min(blocked, key=lambda r: r["angle_deg"]) if blocked else None
    owner: dict[str, str] = {}
    slab_conflict: dict[str, str] = {}

    def fit(lab: str) -> tuple[bool, float, float]:
        pick = chosen[lab]
        assert pick is not None
        return bool(pick.get("fallback")), float(pick["angle_deg"]), abs(float(pick["offset_mm"]))

    for lab in sorted((lab for lab in discs_in if chosen[lab] is not None), key=fit):
        pick = chosen[lab]
        assert pick is not None
        key = str(pick["key"])
        if key in owner:
            slab_conflict[lab] = owner[key]
            chosen[lab] = None
        else:
            owner[key] = lab

    # 2. entries
    discs: dict[str, LevelEntry] = {}
    for lab in discs_in:
        entry = _disc_entry(
            study,
            lab,
            anchors.discs[lab],
            anchors,
            cfg,
            midline_x,
            chosen[lab],
            slab_conflict.get(lab),
            tilt_best[lab],
            has_slabs,
        )
        entry.slab_candidates = [
            {**row, "angle_deg": round(row["angle_deg"], 1), "offset_mm": round(row["offset_mm"], 1)}
            for row in cands[lab]
            if row["in_block"] or row["angle_deg"] <= 2 * cfg.tilt_tol_deg
        ]
        discs[lab] = entry
    vertebrae = {
        name: _vertebra_entry(study, name, v, anchors, cfg) for name, v in sorted(anchors.vertebrae.items())
    }
    slab_for_level: dict[str, str | None] = {}
    for lab in discs_in:
        pick = chosen[lab]
        slab_for_level[lab] = None if pick is None else str(pick["key"])
    level_for_slab: dict[str, str | None] = {s.key: None for s in slabs}
    for lab in discs_in:
        slab_key = slab_for_level[lab]
        if slab_key is not None:
            level_for_slab[slab_key] = lab
    stacks_info = {k: _stack_info(st) for k, st in study.stacks.items()}
    if not study.stacks:
        warnings.append("study has no stacks")
    return LevelMap(
        discs=discs,
        vertebrae=vertebrae,
        stacks=stacks_info,
        slab_for_level=slab_for_level,
        level_for_slab=level_for_slab,
        midline={"x_mm": None if midline_x is None else round(midline_x, 2), "source": midline_src},
        warnings=warnings,
    )


def level_slices(lm: LevelMap, label: str) -> dict[str, Any]:
    """One level as a plain dict (JSON-ready): its entry, its slab and its per-stack hits. KeyError if absent."""
    entry = lm.level(label)
    out = asdict(entry)
    out["lines"] = slice_lines(lm, label)
    return out


def slice_table(lm: LevelMap, label: str) -> list[dict[str, Any]]:
    """Rows for the stacks that show ``label``: inside the slab and field of view, or cut by the body.

    Sorted: the slab for the level first, then the sagittal stacks by distance, then the other stacks.
    """
    entry = lm.level(label)
    rows: list[dict[str, Any]] = []
    for key, h in entry.stacks.items():
        shows = (h.in_slab and h.in_fov) or bool(h.intersects)
        if not shows:
            continue
        if key == entry.slab:
            rank = 0
        elif h.plane == "sag":
            rank = 1
        else:
            rank = 2
        rows.append(
            {
                "key": key,
                "plane": h.plane,
                "slice": h.slice,
                "out_of_plane_mm": h.out_of_plane_mm,
                "inside": h.in_slab and h.in_fov,
                "cut_slices": h.intersects,
                "tilt_deg": h.tilt_deg,
                "flags": h.flags,
                "role": "slab for level" if key == entry.slab else h.plane,
                "_rank": rank,
                "_dist": abs(h.out_of_plane_mm),
            }
        )
    rows.sort(key=lambda r: (r["_rank"], r["_dist"], r["key"]))
    for r in rows:
        r.pop("_rank")
        r.pop("_dist")
    return rows


def slice_lines(lm: LevelMap, label: str) -> list[str]:
    """Plain-text lines for one level (used by the reading briefs and the MCP tool)."""
    entry = lm.level(label)
    head = f"{label} ({entry.kind}, quality {entry.quality}"
    if entry.flags:
        head += "; flags: " + ", ".join(sorted(entry.flags))
    lines = [head + ")"]
    if entry.kind == "disc":
        if entry.slab is not None:
            cand = next((c for c in entry.slab_candidates if c["key"] == entry.slab), None)
            detail = ""
            if cand is not None:
                detail = (
                    f" ({cand['angle_deg']} deg to the disc normal, centre offset {cand['offset_mm']} mm)"
                )
            lines.append(f"  slab for this level: {entry.slab}{detail}")
        else:
            lines.append("  slab for this level: none covers this disc")
    sag = [
        f"{r['key']} slice {r['slice']} ({r['out_of_plane_mm']:+.1f} mm)"
        for r in slice_table(lm, label)
        if r["plane"] == "sag" and not lm.stacks.get(r["key"], {}).get("localizer")
    ]
    lines.append(
        "  sagittal: " + ("; ".join(sag) if sag else "none shows this level inside the field of view")
    )
    axial = [
        f"{key} slices {h.intersects}" for key, h in entry.stacks.items() if h.plane == "ax" and h.intersects
    ]
    if entry.kind == "disc":
        lines.append("  axial slices cutting the disc: " + ("; ".join(axial) if axial else "none"))
        for side in ("left_foramen", "right_foramen", "left_facet", "right_facet"):
            t = entry.targets.get(side, {})
            cov = t.get("covered_by") or {}
            if cov:
                lines.append(
                    f"  {side}: parasagittal slice in " + ", ".join(f"{k} slice {i}" for k, i in cov.items())
                )
            else:
                near = min(
                    (
                        abs(h["out_of_plane_mm"])
                        for k, h in t.get("stacks", {}).items()
                        if h["plane"] == "sag" and not lm.stacks.get(k, {}).get("localizer")
                    ),
                    default=None,
                )
                tail = (
                    "no sagittal stack covers it" if near is None else f"nearest sagittal {near:.1f} mm away"
                )
                lines.append(f"  {side}: {tail} (reformat from a volume, if one exists)")
    return lines
