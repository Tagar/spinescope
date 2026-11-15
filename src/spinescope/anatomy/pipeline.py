"""Automatic spinal anatomy: ``label_spine(study) -> Anchors``.

Steps (each is a documented module function; the numbers come from :class:`spinescope.config.Config`):

1. Mid-sagittal slice and plane from the canal ridge (``midline.select_midline``), refined with
   axial CSF blobs when axial slices cross the canal.
2. Canal centreline (``sagittal.track_canal``) and the vertebral-body axis shifted anteriorly from it;
   the anterior offset is searched for the best endplate chain.
3. Dark endplate lines along the straightened image and an alternating disc/body chain
   (``column.find_lines``, ``column.chain_lines``), then per-disc endplate following for centre,
   margins, plane normal and height (``column.fit_disc``).
4. Numbering from the sacrum upward (``numbering``) for the classical labeller. With the learned labeller
   (default) the discs come from the network; in both cases the numbering is then fused
   (:mod:`.fusion`): numbering cues, level names in slab descriptions, the classical labeller as a weak
   second opinion and the second sagittal series (T1 against T2) as a consistency check.
5. Conus tip from the cord signal inside the thecal sac (``column.conus_tip``).
6. Calibrated disc and numbering confidence and the numbering status ``confirmed`` / ``probable`` /
   ``unconfirmed`` (:mod:`.calibration`, :mod:`.numbering`; policy in docs/labeling/lab08_fusion.md).

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
from .calibration import disc_feature_rows, load_calibration, volume_feature_vector
from .column import DiscFit, Segment, chain_lines, conus_tip, endplate_profile, find_lines, fit_disc
from .fusion import FusionResult, fuse_numbering, fusion_record, relabel_anchors, stack_z_range
from .midline import MidlineChoice, pols_for, refine_midline_axial, select_midline
from .numbering import (
    DEFAULT_STATUS_THRESHOLDS,
    STATUS_CONFIRMED,
    STATUS_PROBABLE,
    STATUS_UNCONFIRMED,
    disc_names,
    numbering_status,
    sacrum_check,
    vertebra_name,
)
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


#: Labelling backends accepted by :func:`label_spine`. ``auto`` uses the learned (ONNX) labeller when
#: ONNX Runtime and the packaged model are available, otherwise the classical one.
BACKENDS = ("auto", "ml", "classical")
METHOD_NUMBERING = (
    "fusion/1 numbering: learned-labeller discs, slab/tilt/conus/slab-name votes, classical tilt (weak), "
    "second sagittal series (consistency), calibrated confidence and status"
)


def _ml_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        from .ml import model_path

        return model_path().exists()
    except Exception:  # pragma: no cover - import failures of optional runtime
        return False


@dataclass
class LabelDetails:
    """:func:`label_spine_details` output: the anchors plus what the fusion and calibration used.

    ``raw_disc_confidence`` and ``raw_numbering_confidence`` are the values before calibration (the numbering one
    is the labeller's own heuristic, as on main). ``disc_features`` / ``volume_features`` are the calibration
    inputs (see :mod:`spinescope.anatomy.calibration`). Used by the calibration tool; most callers want
    :func:`label_spine`.
    """

    anchors: Anchors
    backend: str  # "ml" | "classical" | "none"
    raw_disc_confidence: dict[str, float] = field(default_factory=dict)
    raw_numbering_confidence: float = 0.0
    disc_features: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    volume_features: NDArray[np.float64] | None = None
    fusion: FusionResult | None = None
    calibrated: bool = False
    unfused: Anchors | None = None  # the primary's anchors before the numbering was fused (for evaluation)


def _label_classical(study: Study, cfg: Config) -> Anchors:
    """Classical labelling of the best sagittal stack: discs, bodies, conus and the sacrum-check numbering."""
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
    _attach_conus(anchors, res)
    anchors.numbering = {
        "method": "classical: count up from the lowest disc, sacrum check",
        "confidence": round(float(res.sacrum_conf), 3),
        "notes": "; ".join(res.numbering_notes),
    }
    return anchors


def _classical_discs_of(stack: Stack, cfg: Config, study: Study) -> Anchors | None:
    """Classical disc set of one sagittal stack (used for a second series with the classical backend)."""
    res = _analyse_stack(stack, cfg, study)
    if res is None or not res.anchors_discs:
        return None
    a = Anchors(method=METHOD, source_stack=stack.key)
    a.discs = res.anchors_discs
    return a


def _ml_discs_of(stack: Stack, cfg: Config) -> Anchors:
    """Learned-labeller disc set of one sagittal stack (run on the stack itself, not ``study.best``)."""
    from .ml import anchors_from_segmentation, segment_stack

    return anchors_from_segmentation(segment_stack(stack, cfg), cfg)


def _second_sagittal(study: Study, primary_key: str) -> Stack | None:
    """The first non-localizer sagittal stack other than the primary (e.g. T1 when T2 is primary)."""
    others = [
        s for s in study.find("sag") if s.key != primary_key and not s.meta.get("role", {}).get("localizer")
    ]
    return others[0] if others else None


def _apply_status_caps(status: str, *, sacrum_below: bool, applied: bool, second_extra: bool) -> str:
    """Hard caps on the status: confirmed needs bone below the lowest disc, and a relabelled or
    second-series-contradicted study is at most probable."""
    if status == STATUS_CONFIRMED and (not sacrum_below or applied or second_extra):
        return STATUS_PROBABLE
    return status


def _finalise(study: Study, primary: Anchors, kind: str, cfg: Config, calibrate: bool) -> LabelDetails:
    """Second series, classical opinion, numbering fusion, relabelling, calibration and status."""
    det = LabelDetails(anchors=primary, backend=kind)
    if not primary.discs:
        primary.numbering = {**primary.numbering, "status": STATUS_UNCONFIRMED, "calibrated": False}
        return det
    stack = study.best("sag")
    if stack is None:  # cannot happen when discs were found; keep the numbering contract anyway
        primary.numbering = {**primary.numbering, "status": STATUS_UNCONFIRMED, "calibrated": False}
        return det
    z_cov = stack_z_range(stack)
    raw_num = float(primary.numbering.get("confidence", 0.0) or 0.0)
    det.raw_numbering_confidence = raw_num
    sacrum_below = (
        "S1" in primary.vertebrae and primary.vertebrae["S1"].confidence >= 0.5
        if kind == "ml"
        else raw_num
        >= 0.8  # classical: the sacrum check (>= 9 mm of bone below the lowest disc) is confident
    )

    second: Anchors | None = None
    second_stack = _second_sagittal(study, primary.source_stack)
    warn: list[str] = []
    if second_stack is not None:
        try:
            second = (
                _ml_discs_of(second_stack, cfg)
                if kind == "ml"
                else _classical_discs_of(second_stack, cfg, study)
            )
        except Exception as exc:  # a failing second series must not sink the primary labels
            log.warning("second sagittal series %s failed: %s", second_stack.key, exc)
            warn.append(f"second sagittal series {second_stack.key}: labelling failed ({type(exc).__name__})")
        if second is not None and not second.discs:
            warn.append(f"second sagittal series {second_stack.key}: no disc chain")
            second = None
    classical: Anchors | None = None
    if kind == "ml":
        try:
            classical = _label_classical(study, cfg)
        except Exception as exc:  # the classical opinion is optional: never let it sink the learned labels
            log.warning("classical second opinion failed: %s", exc)
            warn.append(f"classical second opinion failed ({type(exc).__name__}): not used")

    fus = fuse_numbering(
        primary,
        study=study,
        second=second,
        second_key=second_stack.key if second_stack is not None else "",
        classical=classical,
        coverage_z=z_cov,
        sacrum_below=sacrum_below,
    )
    det.fusion = fus
    # Features are taken before the names shift: the T1/T2 comparison is between count-based names.
    raw_pre = {lab: float(d.confidence) for lab, d in primary.discs.items()}
    det.unfused = Anchors.from_dict(primary.to_dict())
    mapping = relabel_anchors(primary, fus.offset if fus.applied else 0)
    det.raw_disc_confidence = {mapping[lab]: v for lab, v in raw_pre.items()}
    t1t2 = {mapping[lab]: v for lab, v in fus.t1t2_by_label.items()}
    second_present = second is not None
    det.disc_features = disc_feature_rows(
        primary, det.raw_disc_confidence, t1t2, second_present=second_present, sacrum_below=sacrum_below
    )
    det.volume_features = volume_feature_vector(
        fus.p_applied,
        det.raw_disc_confidence,
        sacrum_below=sacrum_below,
        second_present=second_present,
        t1t2_fraction=fus.t1t2_fraction,
        classical_agree=bool(fus.classical and fus.classical["agrees"]),
    )

    cal = load_calibration() if (calibrate and kind == "ml") else None
    fallback = fus.p_applied * (1.0 if sacrum_below else 0.9)
    if cal is not None:
        det.calibrated = True
        labels = list(det.disc_features)
        probs = cal.disc.predict(np.stack([det.disc_features[lab] for lab in labels]))
        for lab, p in zip(labels, probs, strict=True):
            d = primary.discs[lab]
            d.evidence.append(
                f"confidence {float(p):.2f} calibrated (raw model probability {det.raw_disc_confidence[lab]:.2f})"
            )
            d.confidence = float(p)
        conf_num = float(cal.volume.predict(det.volume_features)[0])
        thresholds = cal.thresholds
    else:
        conf_num = float(fallback)
        thresholds = DEFAULT_STATUS_THRESHOLDS
        warn.append("numbering confidence is uncalibrated (no calibration for this backend)")
    status = numbering_status(conf_num, thresholds)
    status = _apply_status_caps(
        status,
        sacrum_below=sacrum_below,
        applied=fus.applied,
        second_extra=bool(fus.second and fus.second["extra_below"]),
    )
    rec = fusion_record(fus)
    notes = list(fus.notes)
    if not sacrum_below:
        notes.append(
            "no vertebral bone visible below the lowest disc: the lowest name is not confirmed by bone"
        )
    midline_keys = {k: v for k, v in primary.numbering.items() if k.startswith("midline")}
    primary.numbering = {
        **midline_keys,
        **rec,
        "method": METHOD_NUMBERING,
        "confidence": round(conf_num, 4),
        "confidence_raw": round(raw_num, 4),
        "calibrated": det.calibrated,
        "status": status,
        "sacrum_below": sacrum_below,
        "notes": "; ".join(notes),
        "second_series_agreement": fus.second["agree"] if fus.second else 0,
    }
    primary.warnings.extend(fus.warnings)
    primary.warnings.extend(warn)
    if status != STATUS_CONFIRMED:
        primary.warnings.append(f"numbering {status} (P={conf_num:.2f}): level names need confirmation")
    if status == STATUS_UNCONFIRMED:
        primary.warnings.append("numbering confidence is low")
    return det


def label_spine_details(
    study: Study, config: Config | None = None, *, backend: str = "auto", calibrate: bool = True
) -> LabelDetails:
    """As :func:`label_spine`, but also returns the fusion record, raw confidences and calibration features."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown labelling backend {backend!r}; available: {', '.join(BACKENDS)}")
    if backend == "auto":
        backend = "ml" if _ml_available() else "classical"
        if backend == "classical":
            log.warning("learned labeller unavailable (pip install onnxruntime); using the classical backend")
    cfg = config or DEFAULT
    if backend == "ml":
        from .ml import label_spine_ml

        primary = label_spine_ml(study, cfg)
        return _finalise(study, primary, "ml", cfg, calibrate)
    primary = _label_classical(study, cfg)
    return _finalise(study, primary, "classical", cfg, calibrate)


def label_spine(study: Study, config: Config | None = None, *, backend: str = "auto") -> Anchors:
    """Detect discs, vertebrae and the conus of a lumbar MRI study and number them from the sacrum.

    Uses the best sagittal stack (``study.best('sag')``). Returns :class:`Anchors` in LPS mm with
    confidence and evidence on every anchor. Missing inputs give warnings instead of exceptions.

    ``backend``: ``"auto"`` (default: the learned labeller if available, else classical), ``"ml"``
    (learned; validated on public data, see docs/model_card.md) or ``"classical"`` (no model;
    much less accurate). Unknown names raise :class:`ValueError`.

    The numbering is fused from the labeller, its cues and a second sagittal series when the study has one
    (:mod:`spinescope.anatomy.fusion`). ``anchors.numbering['status']`` is ``confirmed``, ``probable`` or
    ``unconfirmed``; see docs/labeling/lab08_fusion.md for the abstain policy.
    """
    return label_spine_details(study, config, backend=backend).anchors


def _midline_x(stack: Stack, d_mid: float) -> float | None:
    if abs(float(stack.normal[0])) < 0.95:
        return None
    return float(d_mid / float(stack.normal[0]))


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


def labelling_keys(study: Study) -> list[str]:
    """Keys of the stacks :func:`label_spine` reads: the primary sagittal, the second sagittal series, and every
    axial stack (midline refinement and numbering cues). Localizers, coronal and other series are not needed."""
    keys: list[str] = []
    primary = study.best("sag")
    if primary is not None:
        keys.append(primary.key)
        second = _second_sagittal(study, primary.key)
        if second is not None:
            keys.append(second.key)
    keys += [st.key for st in study.stacks.values() if st.plane == "ax"]
    return list(dict.fromkeys(keys))


def load_labelling_study(path: str) -> Study:
    """Load a study from ``path`` with pixels only for the stacks :func:`label_spine` reads.

    A header-only pass classifies the series; then only :func:`labelling_keys` get their voxels read, so peak
    memory follows the labelled stacks rather than every series in the folder.
    """
    from ..study import load_study

    header = load_study(path, load_pixels=False)
    return load_study(path, keys=labelling_keys(header))


def label_spine_cli(path: str) -> dict[str, Any]:
    """Load a study from ``path`` (DICOM directory or volume file) and return the anchors as a dict.

    Intended for the CLI and the MCP server. Loads only the stacks the labeller reads
    (:func:`load_labelling_study`). Output carries no file paths or patient attributes.
    """
    study = load_labelling_study(path)
    d = label_spine(study).to_dict()
    return d
