"""Automatic spinal anatomy: ``label_spine(study) -> Anchors``.

Steps (each is a documented module function; the numbers come from :class:`spinescope.config.Config`):

1. Mid-sagittal slice and plane from the canal ridge (``midline.select_midline``), refined with
   axial CSF blobs when axial slices cross the canal.
2. Canal centreline (``sagittal.track_canal``) and the vertebral-body axis shifted anteriorly from it;
   the anterior offset is searched for the best endplate chain.
3. Dark endplate lines along the straightened image and an alternating disc/body chain
   (``column.find_lines``, ``column.chain_lines``), then per-disc endplate following for centre,
   margins, plane normal and height (``column.fit_disc``).
4. Numbering from the sacrum upward (``numbering``), cross-checked with a second sagittal series
   and with level names in slab descriptions.
5. Conus tip from the cord signal inside the thecal sac (``column.conus_tip``).

Failure is reported, not raised: missing sequences or an unfollowable column produce warnings and
low confidence in the returned :class:`Anchors`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .anchors import Anchors, Disc, Vertebra
from .column import DiscFit, Segment, chain_lines, conus_tip, endplate_profile, find_lines, fit_disc
from .midline import MidlineChoice, pols_for, refine_midline_axial, select_midline
from .numbering import disc_names, sacrum_check, slab_crosscheck, vertebra_name
from .sagittal import BodyAxis, CanalTrack, SagImage, build_body_axis, sagittal_image, straighten, track_canal

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
METHOD = "label_spine/1 automatic (mid-sagittal canal, endplate chain, sacrum-up numbering)"


@dataclass
class SliceAnalysis:
    """Result of the mid-sagittal analysis of one slice."""

    stack: Stack
    k: int
    sag: SagImage
    track: CanalTrack
    axis: BodyAxis
    delta: float
    t: FloatArray
    segments: list[Segment]
    discs: list[DiscFit]  # bottom-up (ascending arc)
    disc_segments: list[Segment]
    conus: tuple[float, float] | None  # (superior mm of the cord tip on the canal, confidence)
    chain_score: float
    warnings: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def _delta_grid(cfg: Config) -> FloatArray:
    lo, hi = cfg.body_axis_offset_mm
    return np.arange(lo, hi + 1e-9, 2.0)


def analyse_slice(stack: Stack, k: int, pol: int, cfg: Config = DEFAULT) -> SliceAnalysis | None:
    """Canal, body axis, endplate chain and disc geometry on sagittal slice ``k``."""
    sag = sagittal_image(stack, k, cfg.sag_resolution_mm)
    track = track_canal(sag, pol, cfg)
    if track is None:
        return None
    best: tuple[float, float, BodyAxis, FloatArray, FloatArray, list[Segment]] | None = None
    for delta in _delta_grid(cfg):
        axis = build_body_axis(sag, track, float(delta))
        half = float(np.clip(delta - 9.0, 12.0, 26.0))
        t = np.arange(-half, half + 1e-9, 0.5)
        straight = straighten(sag, axis, t)
        prof, valid = endplate_profile(straight, t, half)
        lines = find_lines(prof, axis.arc, cfg, valid)
        total, segs = chain_lines(lines, cfg)
        if best is None or total > best[0]:
            best = (total, float(delta), axis, t, straight, segs)
    assert best is not None
    total, delta, axis, t, straight, segs = best
    warns: list[str] = []
    ev = [f"canal track: {track.s[-1] - track.s[0]:.0f} mm, ridge score {track.score:.1f}"]
    ev.append(f"body axis offset {delta:.0f} mm anterior to the canal centre (best endplate chain)")
    disc_segs = [s for s in segs if s.kind == "disc"]
    if not disc_segs:
        warns.append("no disc found on the mid-sagittal slice")
    discs = [fit_disc(straight, axis.arc, t, s, cfg) for s in disc_segs]
    # A "disc" whose endplates are closer than a disc can be (a fused S1-S2 line, or two tracks on one
    # line) is not a disc.
    kept = [(s, d) for s, d in zip(disc_segs, discs, strict=True) if d.height >= cfg.disc_height_mm[0]]
    if len(kept) < len(discs):
        warns.append(f"{len(discs) - len(kept)} thin interval(s) rejected as not disc-like")
    disc_segs = [s for s, _ in kept]
    discs = [d for _, d in kept]
    # A vertebral body longer than a body plus a disc means a disc was missed: labels above are then unreliable.
    long_bodies = [g for g in segs if g.kind == "body" and g.hi - g.lo > cfg.body_height_mm[1] - 4.0]
    if long_bodies:
        warns.append(
            f"{len(long_bodies)} vertebral-body interval(s) are too long for one body: a disc may be missed, "
            "so levels above it may be shifted"
        )
    conus = conus_tip(sag, track, cfg)
    if conus is None:
        warns.append("conus not located (cord signal not found in the thecal sac)")
    return SliceAnalysis(
        stack=stack,
        k=k,
        sag=sag,
        track=track,
        axis=axis,
        delta=delta,
        t=t,
        segments=segs,
        discs=discs,
        disc_segments=disc_segs,
        conus=conus,
        chain_score=total,
        warnings=warns,
        evidence=ev,
    )


def _to_plane(p: FloatArray, normal: FloatArray, d: float) -> FloatArray:
    """Move point ``p`` along ``normal`` onto the plane ``normal . x = d``."""
    return p + (d - float(p @ normal)) * normal


def _disc_geometry(an: SliceAnalysis, fit: DiscFit, d_mid: float) -> dict[str, Any]:
    sag, axis = an.sag, an.axis
    c_a, c_s = axis.point(fit.arc, fit.t)
    center = _to_plane(sag.to_lps(float(c_a), float(c_s)), sag.normal, d_mid)
    ta, ts = axis.tangent(fit.arc)
    na, ns = axis.normal(fit.arc)
    nv_a, nv_s = float(ta - fit.slope * na), float(ts - fit.slope * ns)
    nrm = np.hypot(nv_a, nv_s)
    nv_a, nv_s = nv_a / nrm, nv_s / nrm
    if nv_s < 0:
        nv_a, nv_s = -nv_a, -nv_s
    normal3 = nv_a * sag.ap + nv_s * sag.sup
    out: dict[str, Any] = {
        "center": center,
        "normal": normal3,
        "height": fit.height,
        "anterior": None,
        "posterior": None,
        "ap_mm": None,
    }
    if fit.t_ant is not None and fit.t_post is not None:
        for key, tv in (("anterior", fit.t_ant), ("posterior", fit.t_post)):
            arc_m = fit.arc + fit.slope * (tv - fit.t)
            pa, ps = axis.point(arc_m, tv)
            out[key] = _to_plane(sag.to_lps(float(pa), float(ps)), sag.normal, d_mid)
        out["ap_mm"] = fit.t_ant - fit.t_post
    tilt = float(np.degrees(np.arctan2(abs(nv_a), nv_s)))
    out["tilt_deg"] = tilt
    return out


def _disc_confidence(fit: DiscFit, cfg: Config) -> float:
    if fit.t_ant is None or fit.t_post is None:
        return 0.3
    strength = min(1.0, 0.5 * (fit.lo_strength + fit.hi_strength) / 0.4)
    ap = min(1.0, (fit.t_ant - fit.t_post) / 25.0)
    return float(np.clip(0.5 * strength + 0.5 * ap, 0.0, 1.0))


def _level_at(an: SliceAnalysis, arc: float, names: list[str]) -> str | None:
    """Label of the disc or vertebra at ``arc`` (bottom-up lists, ``names`` for discs)."""
    discs = an.discs
    if not discs:
        return None
    for i, d in enumerate(discs):
        if abs(arc - d.arc) <= d.height / 2:
            return names[i]
    for i in range(len(discs) - 1):
        if discs[i].arc < arc < discs[i + 1].arc:
            return vertebra_name(i + 1)
    if arc < discs[0].arc:
        return vertebra_name(0)
    return vertebra_name(len(discs))


@dataclass
class StackResult:
    analysis: SliceAnalysis
    midline: MidlineChoice
    d_mid: float
    names: list[str]
    anchors_discs: dict[str, Disc]
    sacrum_conf: float
    numbering_notes: list[str]
    warnings: list[str]


def _analyse_stack(stack: Stack, cfg: Config, study: Study | None = None) -> StackResult | None:
    """Full analysis of one sagittal stack, with its own midline and numbering."""
    warns: list[str] = []
    mid = select_midline(stack, pols_for(stack), cfg)
    if mid is None:
        return None
    d_mid = mid.plane_coord
    ax_x: float | None = None
    if study is not None:
        ax_x, ax_ev, ax_w = refine_midline_axial(study, stack, mid, cfg)
        mid.evidence.extend(ax_ev)
        warns.extend(ax_w)
        if ax_x is not None:
            d_mid = ax_x * float(stack.normal[0])
    an = analyse_slice(stack, mid.slice_index, mid.polarity, cfg)
    if an is None:
        return None
    warns.extend(an.warnings)
    if not an.discs:
        return StackResult(an, mid, d_mid, [], {}, 0.0, [], warns)
    if len(an.discs) < 3:
        warns.append(f"only {len(an.discs)} disc(s) found; numbering is less certain")
    # Numbering from the bottom (lowest disc = L5-S1 when bone is visible below it).
    names = disc_names(len(an.discs))
    arc0 = float(an.axis.arc[0])
    normals = []
    for fit in an.discs:
        g = _disc_geometry(an, fit, d_mid)
        normals.append(g["tilt_deg"])
    low_angle = normals[0]
    next_angle = normals[1] if len(normals) > 1 else None
    sc = sacrum_check(an.discs[0].arc - an.discs[0].height / 2, arc0, low_angle, next_angle, cfg)
    warns.extend(sc.warnings)
    notes = list(sc.evidence)
    anchors_discs: dict[str, Disc] = {}
    for name, fit in zip(names, an.discs, strict=True):
        g = _disc_geometry(an, fit, d_mid)
        conf = _disc_confidence(fit, cfg) * (1.0 if sc.confidence >= 0.8 else 0.85)
        ev = list(fit.evidence)
        ev.append(f"height {fit.height:.1f} mm, tilt {g['tilt_deg']:.0f} deg from the sagittal superior axis")
        ev.append(f"numbering: {name} by counting up from the lowest disc (confidence {sc.confidence:.2f})")
        anchors_discs[name] = Disc(
            center=[float(v) for v in g["center"]],
            normal=[float(v) for v in g["normal"]],
            anterior=None if g["anterior"] is None else [float(v) for v in g["anterior"]],
            posterior=None if g["posterior"] is None else [float(v) for v in g["posterior"]],
            height_mm=float(fit.height),
            confidence=float(np.clip(conf, 0.0, 1.0)),
            evidence=ev,
        )
    if study is not None and study.find("sag") and len(an.discs) and mid.track is not None:
        pass
    return StackResult(an, mid, d_mid, names, anchors_discs, sc.confidence, notes, warns)


def _vertebrae(res: StackResult, cfg: Config) -> dict[str, Vertebra]:
    an = res.analysis
    out: dict[str, Vertebra] = {}
    discs = an.discs
    sag = an.sag
    arc_lo, arc_hi = float(an.axis.arc[0]), float(an.axis.arc[-1])
    names = res.names

    def add(name: str, arc: float, conf: float, ev: str) -> None:
        if not (arc_lo <= arc <= arc_hi):
            return
        a_, s_ = an.axis.point(arc, 0.0)
        c = _to_plane(sag.to_lps(float(a_), float(s_)), sag.normal, res.d_mid)
        out[name] = Vertebra(center=[float(v) for v in c], confidence=conf, evidence=[ev])

    for k in range(len(discs) - 1):
        mid_arc = 0.5 * (discs[k].arc + discs[k + 1].arc)
        add(vertebra_name(k + 1), mid_arc, 0.8 * res.sacrum_conf, "midway between adjacent disc centres")
    if discs:
        add(vertebra_name(0), discs[0].arc - 20.0, 0.4 * res.sacrum_conf, "20 mm below the lowest disc")
        add(vertebra_name(len(discs)), discs[-1].arc + 20.0, 0.4, "20 mm above the highest disc")
    del names
    return out


def label_spine(study: Study, config: Config | None = None) -> Anchors:
    """Detect discs, vertebrae and the conus of a lumbar MRI study and number them from the sacrum.

    Uses the best sagittal stack (``study.best('sag')``). Returns :class:`Anchors` in LPS mm with
    confidence and evidence on every anchor. Missing inputs give warnings instead of exceptions.
    """
    cfg = config or DEFAULT
    anchors = Anchors(method=METHOD)
    stack = study.best("sag")
    if stack is None:
        anchors.warnings.append("no sagittal series: spine anatomy cannot be labelled")
        anchors.numbering = {"method": "none", "confidence": 0.0, "notes": "no sagittal"}
        return anchors
    anchors.source_stack = stack.key
    res = _analyse_stack(stack, cfg, study)
    if res is None or not res.anchors_discs:
        anchors.warnings.append(f"{stack.key}: spinal canal or endplate column not found")
        anchors.numbering = {"method": "none", "confidence": 0.0, "notes": "no disc chain"}
        if res is not None:
            anchors.warnings.extend(res.warnings)
        return anchors
    anchors.discs = res.anchors_discs
    anchors.vertebrae = _vertebrae(res, cfg)
    anchors.midline_x = _midline_x(stack, res.d_mid)
    anchors.warnings.extend(res.warnings)
    conf_num = res.sacrum_conf
    notes = list(res.numbering_notes)
    # Cross-check with a second sagittal series (e.g. T1 against T2): agreement of labels.
    other = [
        s for s in study.find("sag") if s.key != stack.key and not s.meta.get("role", {}).get("localizer")
    ]
    agree_n = 0
    for st in other[:1]:
        res2 = _analyse_stack(st, cfg, study)
        if res2 is None or not res2.anchors_discs:
            notes.append(f"second sagittal series {st.key}: no disc chain")
            continue
        agree, total = _compare_labels(anchors.discs, res2.anchors_discs, cfg.disc_match_mm)
        agree_n = agree
        if total:
            notes.append(
                f"second sagittal series {st.key}: {agree}/{total} discs agree in label and position"
            )
            if agree == total:
                conf_num = min(1.0, conf_num + 0.1)
            else:
                anchors.warnings.append(
                    f"sagittal series {st.key} numbers {total - agree} disc(s) differently"
                )
                conf_num = max(0.0, conf_num - 0.2)
    # Slab cross-check: level names in oblique slab descriptions.
    slabs = study.slabs("ax")
    if slabs:
        centres = {k: np.asarray(v.center) for k, v in anchors.discs.items()}
        normals = {k: np.asarray(v.normal) for k, v in anchors.discs.items()}
        heights = {k: float(v.height_mm or 8.0) for k, v in anchors.discs.items()}
        ag, dis, sl_ev = slab_crosscheck(slabs, centres, normals, heights)
        notes.extend(sl_ev)
        if dis:
            anchors.warnings.append("oblique slab level names disagree with the numbering")
            conf_num = max(0.0, conf_num - 0.2)
        elif ag:
            conf_num = min(1.0, conf_num + 0.05)
    _attach_conus(anchors, res)
    anchors.numbering = {
        "method": "count up from the lowest disc (L5-S1 when bone is visible below it)",
        "confidence": round(float(conf_num), 3),
        "notes": "; ".join(notes),
        "second_series_agreement": agree_n,
    }
    if conf_num < 0.5:
        anchors.warnings.append("numbering confidence is low")
    return anchors


def _midline_x(stack: Stack, d_mid: float) -> float | None:
    if abs(float(stack.normal[0])) < 0.95:
        return None
    return float(d_mid / float(stack.normal[0]))


def _compare_labels(a: dict[str, Disc], b: dict[str, Disc], tol: float) -> tuple[int, int]:
    """Agreement of two disc sets: a disc of ``a`` agrees if the nearest disc of ``b`` (within tol mm in
    the sagittal plane) has the same label. Returns (agree, compared)."""
    agree = total = 0
    for lab, d in a.items():
        ca = np.asarray(d.center)
        best: tuple[float, str] | None = None
        for lb, e in b.items():
            cb = np.asarray(e.center)
            dd = float(np.linalg.norm((ca - cb)[1:]))
            if best is None or dd < best[0]:
                best = (dd, lb)
        if best is None or best[0] > tol:
            continue
        total += 1
        agree += int(best[1] == lab)
    return agree, total


def _attach_conus(anchors: Anchors, res: StackResult) -> None:
    an = res.analysis
    if an.conus is None:
        anchors.conus = None
        return
    s_tip, conf = an.conus
    a_c = float(an.track.a_at(s_tip))
    pt = _to_plane(an.sag.to_lps(a_c, s_tip), an.sag.normal, res.d_mid)
    arc_tip = float(np.interp(s_tip, an.axis.s, an.axis.arc))
    level = _level_at(an, arc_tip, res.names)
    anchors.conus = {
        "z": float(pt[2]),
        "center": [float(v) for v in pt],
        "level": level,
        "confidence": round(float(conf) * (0.9 if an.track.polarity else 0.8), 3),
        "evidence": ["cord signal inside the thecal sac ends here (longest run on the canal centreline)"],
    }


def label_spine_cli(path: str) -> dict[str, Any]:
    """Load a study from ``path`` (DICOM directory or volume file) and return the anchors as a dict.

    Intended for the CLI and the MCP server. Output carries no file paths or patient attributes.
    """
    from ..study import load_study

    study = load_study(path)
    d = label_spine(study).to_dict()
    return d
