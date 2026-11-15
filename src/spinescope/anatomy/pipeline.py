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
5. Conus tip: the classical tip (``column.conus_tip``) as a fallback; the learned-canal conus locator
   (:mod:`.conus`) runs after the labels are final and replaces it when it finds the cord.
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
from .axial import assess_axial
from .calibration import (
    disc_feature_rows,
    load_calibration,
    load_tta_calibration,
    numbering_signals,
    sacral_signals,
    volume_feature_map,
    volume_feature_vector,
)
from .column import DiscFit, Segment, chain_lines, conus_tip, endplate_profile, find_lines, fit_disc
from .fusion import (
    DEFAULT_FUSION,
    FusionConfig,
    FusionResult,
    fuse_numbering,
    fusion_record,
    relabel_anchors,
    stack_z_range,
)
from .lumbosacral import (
    PATCH_MM,
    Candidate,
    LumbosacralDecision,
    candidate_patches,
    decide,
    drop_lowest,
    lumbosacral_record,
)
from .lumbosacral import (
    load_model as load_lumbosacral_model,
)
from .midline import MidlineChoice, pols_for, refine_midline_axial, select_midline
from .ml.labeller import SegmentedStack
from .ml.tta import TTAResult
from .ml.tta import analyse as analyse_tta
from .ml.tta import disc_records as tta_disc_records
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
from .ood import apply_applicability, assess_applicability
from .review import stamp_study
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
    #: Version 2 numbering signals (:func:`calibration.numbering_signals`) and every named volume feature.
    signals: dict[str, float] = field(default_factory=dict)
    volume_feature_dict: dict[str, float] = field(default_factory=dict)
    fusion: FusionResult | None = None
    calibrated: bool = False
    unfused: Anchors | None = None  # the primary's anchors before the numbering was fused (for evaluation)
    status_thresholds: dict[str, float] | None = None  # thresholds the numbering status was computed with
    #: Test-time augmentation (lab35, ``Config.ml_tta`` only): the study record and the per-disc agreement, keyed by the
    #: final labels. Empty when the augmentation is off.
    tta: dict[str, Any] = field(default_factory=dict)
    tta_discs: dict[str, dict[str, float]] = field(default_factory=dict)


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


def _ml_discs_of(stack: Stack, cfg: Config, seg: SegmentedStack | None = None) -> Anchors:
    """Learned-labeller disc set of one sagittal stack. ``seg``: its network output, when already computed."""
    from .ml import anchors_from_segmentation, segment_stack

    return anchors_from_segmentation(seg if seg is not None else segment_stack(stack, cfg), cfg)


def _second_sagittal(study: Study, primary_key: str) -> Stack | None:
    """The first non-localizer sagittal stack other than the primary (e.g. T1 when T2 is primary)."""
    others = [
        s for s in study.find("sag") if s.key != primary_key and not s.meta.get("role", {}).get("localizer")
    ]
    return others[0] if others else None


def _apply_status_caps(
    status: str, *, sacrum_below: bool, applied: bool, second_extra: bool, lumbo_status: str = "not_assessed"
) -> str:
    """Hard caps on the status: confirmed needs bone below the lowest disc, and a relabelled or
    second-series-contradicted study is at most probable. A lumbosacral decision that is unsure (a lower disc may
    be a sacral remnant) caps the status at unconfirmed; a removed sacral disc caps it at probable."""
    if status == STATUS_CONFIRMED and (not sacrum_below or applied or second_extra or lumbo_status == "drop"):
        status = STATUS_PROBABLE
    if lumbo_status == "drop" and status == STATUS_CONFIRMED:
        status = STATUS_PROBABLE
    if lumbo_status == "unsure" and status in (STATUS_CONFIRMED, STATUS_PROBABLE):
        status = STATUS_UNCONFIRMED
    return status


def _lumbosacral_step(
    primary: Anchors, seg: SegmentedStack | None, cfg: Config
) -> tuple[LumbosacralDecision, int]:
    """Decide whether the lowest candidates are sacral (:mod:`.lumbosacral`) and remove them from ``primary``.

    Returns the decision and the number of discs removed. Without the packaged model or a sagittal segmentation
    the decision is ``not_assessed`` and nothing changes.
    """
    model = load_lumbosacral_model()
    cands: list[Candidate] = []
    patches = np.zeros((0, PATCH_MM, PATCH_MM), dtype=np.float32)
    if seg is not None and primary.discs and model is not None:
        cands, patches = candidate_patches(seg, primary, cfg)
    dec = decide(cands, patches, model)
    n = dec.drop if dec.status == "drop" else 0
    if n:
        drop_lowest(primary, n)
        primary.warnings.append(
            f"{n} lowest disc candidate(s) removed as sacral (S1-S2 remnant or sacral segment); "
            "level names counted up from the next disc"
        )
    if dec.status == "unsure":
        primary.warnings.extend(dec.warnings)
    return dec, n


def _geometry_step(primary: Anchors, seg: SegmentedStack, cfg: Config) -> None:
    """Vertebral bodies, canal and posterior elements from the labeller's own segmentation (lab22: before numbering).

    Measured with the disc names of this stage; :func:`~spinescope.anatomy.fusion.relabel_anchors` carries them
    through the numbering offset. A failure adds a warning and leaves the labels unchanged.
    """
    from .vertebrae import measure_geometry

    try:
        measure_geometry(seg, primary, cfg)
    except Exception as exc:  # the geometry is an addition: it must never sink the labels
        log.warning("vertebral geometry failed: %s", exc)
        primary.warnings.append(f"vertebral geometry not measured ({type(exc).__name__})")


def _s1_signals(primary: Anchors) -> tuple[float | None, float | None]:
    """Bone below the lowest disc (mm, mid-sagittal S1 body) and the S1 / L5 AP depth ratio, from the geometry."""
    bodies: dict[int, dict[str, Any]] = {}
    for v in primary.vertebrae.values():
        b = v.body
        if isinstance(b, dict) and b.get("discs_below") is not None:
            bodies[int(b["discs_below"])] = b
    s1 = bodies.get(0)
    below = (
        None if s1 is None or s1.get("below_lowest_disc_mm") is None else float(s1["below_lowest_disc_mm"])
    )
    ratio: float | None = None
    if s1 is not None and 1 in bodies:
        ap0, ap1 = s1.get("ap_depth_mm"), bodies[1].get("ap_depth_mm")
        if ap0 and ap1:
            ratio = float(ap0) / float(ap1)
    return below, ratio


def _lstv_stage(anchors: Anchors) -> None:
    """Transitional-vertebra check after the numbering is final (:mod:`.lstv`); suggests, never renames. A failure
    keeps the labels and warns."""
    from .lstv import attach_lstv

    try:
        attach_lstv(anchors)
    except Exception as exc:  # the check is additional: it must never sink the labels
        log.warning("transitional vertebra check failed: %s", exc)
        anchors.warnings.append(f"transitional vertebra check failed ({type(exc).__name__})")
        return
    # A suspected transitional vertebra makes the anatomical level names ambiguous even when the count is right:
    # never present such a study as "confirmed".
    lstv = anchors.lstv if isinstance(anchors.lstv, dict) else None
    if lstv and lstv.get("status") == "suspected" and anchors.numbering.get("status") == STATUS_CONFIRMED:
        anchors.numbering["status"] = STATUS_PROBABLE
        anchors.warnings.append(
            "numbering capped at probable: a transitional lumbosacral vertebra is suspected, so the anatomical "
            "level names may differ from the count from the sacrum"
        )


def _conus_stage(study: Study, anchors: Anchors, cfg: Config, seg: SegmentedStack | None) -> None:
    """Locate the conus before the numbering fusion (experimental; its tip is a weak vote). A failure keeps the
    labels and warns. The final names and the status factor are set by :func:`refresh_conus` after numbering."""
    from .conus import attach_conus

    try:
        attach_conus(study, anchors, cfg, seg=seg)  # reuses the primary network output
    except Exception as exc:  # the conus is additional: it must never sink the labels
        log.warning("conus locator failed: %s", exc)
        anchors.warnings.append(f"conus not located (locator failed: {type(exc).__name__})")
        anchors.conus = None
        return
    if isinstance(anchors.conus, dict):
        anchors.conus["experimental"] = True


def _finalise(
    study: Study,
    primary: Anchors,
    kind: str,
    cfg: Config,
    calibrate: bool,
    seg: SegmentedStack | None = None,
    lumbosacral: bool = True,
    fusion_cfg: FusionConfig | None = None,
    segs: dict[str, SegmentedStack] | None = None,
    tta: TTAResult | None = None,
) -> LabelDetails:
    """Every stage that precedes the calibrated numbering decision, in this order (lab22):

    lumbosacral check -> vertebral geometry (bodies incl. the bone below the lowest disc, canal) -> conus (weak vote)
    -> second series and classical opinion -> numbering fusion -> calibrated confidence and status.

    ``segs`` maps stack keys to network outputs already computed for the study (reused, never recomputed).
    """
    fcfg = fusion_cfg or DEFAULT_FUSION
    segs = segs if segs is not None else {}
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
    # The labeller's output before any numbering change (the "before" of every evaluation).
    det.unfused = Anchors.from_dict(primary.to_dict())

    # 1. lumbosacral check (drops sacral candidates from the bottom of the chain)
    lumbo_dec: LumbosacralDecision | None = None
    lumbo_drop = 0
    if kind == "ml" and lumbosacral:
        lumbo_dec, lumbo_drop = _lumbosacral_step(primary, seg, cfg)

    # 2. vertebral geometry and 3. conus, both before the numbering fusion
    if kind == "ml" and seg is not None:
        _geometry_step(primary, seg, cfg)
    s1_below, s1_ratio = _s1_signals(primary) if kind == "ml" else (None, None)
    if kind == "ml":
        _conus_stage(study, primary, cfg, seg)
    if kind == "ml":
        sacrum_below = s1_below is not None and s1_below >= cfg.sacrum_min_visible_mm
    else:
        sacrum_below = raw_num >= 0.8  # classical: the sacrum check (>= 9 mm of bone below the lowest disc)

    # 4. second sagittal series and the classical opinion (both optional; never sink the primary labels)
    second: Anchors | None = None
    second_stack = _second_sagittal(study, primary.source_stack)
    warn: list[str] = []
    if second_stack is not None:
        try:
            second = (
                _ml_discs_of(second_stack, cfg, segs.get(second_stack.key))
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
    if kind == "ml" and cfg.classical_second_opinion:  # opt-in (lab14_perf): off by default
        try:
            classical = _label_classical(study, cfg)
        except Exception as exc:  # the classical opinion is optional: never let it sink the learned labels
            log.warning("classical second opinion failed: %s", exc)
            warn.append(f"classical second opinion failed ({type(exc).__name__}): not used")

    # 5. numbering fusion (votes: slabs, tilt, conus, classical, second series, axial when enabled)
    axial_rep = assess_axial(study, primary) if fcfg.use_axial else None
    fus = fuse_numbering(
        primary,
        study=study,
        second=second,
        second_key=second_stack.key if second_stack is not None else "",
        classical=classical,
        coverage_z=z_cov,
        sacrum_below=sacrum_below,
        cfg=fcfg,
        axial=axial_rep,
    )
    det.fusion = fus
    low_pre = primary.disc_labels()[-1]  # the lowest disc before the names shift
    t1t2_bottom = bool(second is not None and fus.t1t2_by_label.get(low_pre, False))
    # Features are taken before the names shift: the T1/T2 comparison is between count-based names.
    raw_pre = {lab: float(d.confidence) for lab, d in primary.discs.items()}
    # Version 2/3 numbering signals, on the labeller's lowest discs after the sacral check (name-free geometry).
    lumbo_p = lumbo_dec.probs[lumbo_drop:] if lumbo_dec is not None and lumbo_dec.candidates else None
    det.signals = {
        **numbering_signals(
            primary,
            lumbo_p=lumbo_p,
            votes=fus.votes,
            p_shift=float(1.0 - fus.posterior.get(0, 1.0)),
            coverage_low_z=z_cov[0],
        ),
        **sacral_signals(s1_below_mm=s1_below, s1_ap_ratio=s1_ratio, t1t2_bottom=t1t2_bottom),
    }
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
    det.volume_feature_dict = {
        **volume_feature_map(
            fus.p_applied,
            det.raw_disc_confidence,
            sacrum_below=sacrum_below,
            second_present=second_present,
            t1t2_fraction=fus.t1t2_fraction,
            classical_agree=bool(fus.classical and fus.classical["agrees"]),
        ),
        **det.signals,
    }
    if tta is not None:  # lab35: study features of the variant agreement (only with Config.ml_tta)
        det.volume_feature_dict.update(tta.study)

    # 6. calibrated numbering confidence and status (caps as before)
    cal = load_calibration() if (calibrate and kind == "ml") else None
    if cal is not None and cfg.ml_tta and tta is not None:
        tta_cal = load_tta_calibration()
        if tta_cal is not None:
            cal = tta_cal
        else:
            warn.append(
                "test-time augmentation calibration not shipped: numbering confidence is calibration v2"
            )
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
        conf_num = float(cal.volume.predict(cal.volume_vector(det.volume_feature_dict))[0])
        thresholds = cal.thresholds
    else:
        conf_num = float(fallback)
        thresholds = DEFAULT_STATUS_THRESHOLDS
        warn.append("numbering confidence is uncalibrated (no calibration for this backend)")
    status = numbering_status(conf_num, thresholds)
    det.status_thresholds = thresholds
    lumbo_status = lumbo_dec.status if lumbo_dec is not None else "not_assessed"
    status = _apply_status_caps(
        status,
        sacrum_below=sacrum_below,
        applied=fus.applied,
        second_extra=bool(fus.second and fus.second["extra_below"]),
        lumbo_status=lumbo_status,
    )
    rec = fusion_record(fus)
    notes = list(fus.notes)
    if lumbo_dec is not None:
        notes.append(
            f"lumbosacral check: {lumbo_dec.status}"
            + (f" ({lumbo_drop} sacral disc(s) removed)" if lumbo_drop else "")
        )
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
        **({"lumbosacral": lumbosacral_record(lumbo_dec, lumbo_drop)} if lumbo_dec is not None else {}),
        **({"tta": {k: round(float(v), 4) for k, v in tta.study.items()}} if tta is not None else {}),
    }
    if tta is not None and tta.study.get("tta_chain_agree", 1.0) < 1.0:
        warn.append(
            f"test-time variants disagree on the disc chain: {tta.study['tta_chain_agree']:.0%} of the "
            f"{len(tta.variants)} variants ({', '.join(tta.variants)}) reproduce the base count and names"
        )
    primary.warnings.extend(fus.warnings)
    primary.warnings.extend(warn)
    if status != STATUS_CONFIRMED:
        primary.warnings.append(f"numbering {status} (P={conf_num:.2f}): level names need confirmation")
    if status == STATUS_UNCONFIRMED:
        primary.warnings.append("numbering confidence is low")
    return det


def _segmentations(study: Study, seg: SegmentedStack, cfg: Config) -> dict[str, SegmentedStack]:
    """Network outputs of this study by stack key: the primary (already computed) and the second sagittal series.

    The second series is segmented once here; a failure leaves it out (its anchors are then not used).
    """
    from .ml import segment_stack

    out: dict[str, SegmentedStack] = {seg.stack.key: seg}
    second = _second_sagittal(study, seg.stack.key)
    if second is not None:
        try:
            out[second.key] = segment_stack(second, cfg)
        except Exception as exc:  # the second series is optional
            log.warning("second sagittal series %s failed: %s", second.key, exc)
    return out


def _recap_status(anchors: Anchors) -> None:
    """Re-apply the status caps after applicability: :func:`apply_applicability` recomputes the status from the
    confidence alone, which would undo the caps (no bone below, relabelling, lumbosacral, second series)."""
    num = anchors.numbering
    if not anchors.discs or not num.get("labelled", False) or "status" not in num:
        return
    lumbo = num.get("lumbosacral") or {}
    second = num.get("second_series") or {}
    capped = _apply_status_caps(
        str(num["status"]),
        sacrum_below=bool(num.get("sacrum_below", False)),
        applied=bool(num.get("relabelled", False)),
        second_extra=bool(second.get("extra_below", False)),
        lumbo_status=str(lumbo.get("status", "not_assessed")),
    )
    if capped != num["status"]:
        reasons = []
        if not num.get("sacrum_below", False):
            reasons.append("no bone below the lowest disc")
        if num.get("relabelled", False):
            reasons.append("relabelled")
        if second.get("extra_below", False):
            reasons.append("second series has an extra disc below")
        if lumbo.get("status") in ("drop", "unsure"):
            reasons.append(f"lumbosacral check {lumbo.get('status')}")
        num["status"] = capped
        anchors.warnings.append(
            f"numbering capped to {capped} after applicability: {', '.join(reasons) or 'caps'}"
        )


def label_spine_details(
    study: Study,
    config: Config | None = None,
    *,
    backend: str = "auto",
    calibrate: bool = True,
    lumbosacral: bool = True,
    fusion: FusionConfig | None = None,
    counting: bool = False,
) -> LabelDetails:
    """As :func:`label_spine`, but also returns the fusion record, raw confidences and calibration features.

    Stage order (lab22): network segmentation (once per stack) -> anchors -> lumbosacral check -> vertebral geometry
    -> conus (weak vote) -> second series and fusion votes -> calibrated numbering -> applicability (may still cap).

    ``lumbosacral=False`` skips the sacral-disc check (:mod:`.lumbosacral`): the labeller-only numbering, used as
    the "before" in evaluations. The default check is on. ``fusion`` overrides the fusion settings (e.g.
    ``FusionConfig(use_axial=True)`` for the axial T1 slab cue); the default is :data:`fusion.DEFAULT_FUSION`.

    ``counting=True`` (opt-in: it found no usable landmark and casts no vote, docs/labeling/lab34_counting.md) records
    the top-chain evidence of wider sagittal series in ``anchors.numbering["counting"]``
    (:mod:`.counting`; evidence only, no name changes, no warnings added to the labels). ML backend only.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown labelling backend {backend!r}; available: {', '.join(BACKENDS)}")
    if backend == "auto":
        backend = "ml" if _ml_available() else "classical"
        if backend == "classical":
            log.warning("learned labeller unavailable (pip install onnxruntime); using the classical backend")
    cfg = config or DEFAULT
    seg: SegmentedStack | None = None
    if backend == "ml":
        from .ml import label_spine_ml_segmented

        primary, seg = label_spine_ml_segmented(study, cfg)
        # lab35: the variants are compared with the labeller's chain before any numbering step changes it.
        tta_res = analyse_tta(primary, seg, cfg) if (cfg.ml_tta and seg is not None) else None
        segs = _segmentations(study, seg, cfg) if seg is not None else {}
        det = _finalise(
            study,
            primary,
            "ml",
            cfg,
            calibrate,
            seg=seg,
            lumbosacral=lumbosacral,
            fusion_cfg=fusion,
            segs=segs,
            tta=tta_res,
        )
        if counting:
            _counting_stage(study, det.anchors, cfg, segs)
        if tta_res is not None:
            _attach_tta(det, tta_res)
    else:
        primary = _label_classical(study, cfg)
        # The lateral curve needs the learned labeller's canal segmentation (lab30): record that it was not measured.
        primary.alignment.setdefault(
            "coronal",
            {"valid": False, "active": False, "method": "classical backend", "evidence": ["not measured"]},
        )
        det = _finalise(study, primary, "classical", cfg, calibrate, fusion_cfg=fusion)
    # Applicability: when not to label (wrong region, unusable field of view, unseen input). Additive; see ood.py.
    app = assess_applicability(study, det.anchors, seg=seg, cfg=cfg)
    apply_applicability(det.anchors, app, thresholds=det.status_thresholds)
    _recap_status(det.anchors)
    if det.anchors.discs:
        if backend == "ml":
            from .conus import refresh_conus

            refresh_conus(det.anchors, cfg)  # level, low-lying flag and status factor in the final names
        else:
            _conus_after_labels(study, det.anchors, cfg)
        _quality_stage(study, det.anchors, cfg)
    _lstv_stage(det.anchors)
    stamp_study(det.anchors, study)  # binds the labels to this study (corrected files are checked against it)
    return det


def _counting_stage(study: Study, anchors: Anchors, cfg: Config, segs: dict[str, SegmentedStack]) -> None:
    """Record the top-chain evidence of the wider sagittal series in ``anchors.numbering["counting"]`` (never changes names)."""
    from .counting import count_evidence

    try:
        res = count_evidence(study, anchors, primary_stack=study.best("sag"), config=cfg, segs=segs)
    except Exception as exc:  # evidence only: a failure must never sink the labels
        log.warning("counting evidence failed: %s", exc)
        anchors.numbering["counting"] = {"available": False, "reason": f"failed ({type(exc).__name__})"}
        return
    anchors.numbering["counting"] = res.to_dict()


def _quality_stage(study: Study, anchors: Anchors, cfg: Config) -> None:
    """Hardware and image-artefact flags (:mod:`.artifacts`), last: they cap what the labels cover, never the labels.

    A failing detector adds a warning and leaves the labels and the status unchanged.
    """
    from .artifacts import apply_quality, assess_quality

    try:
        record = assess_quality(study, anchors, cfg)
    except Exception as exc:  # the quality check is additional: it must never sink the labels
        log.warning("image quality check failed: %s", exc)
        anchors.warnings.append(f"image quality not assessed ({type(exc).__name__})")
        return
    apply_quality(anchors, record, cfg)


def _attach_tta(det: LabelDetails, res: TTAResult) -> None:
    """Record the variant agreement on the details and on each final disc (lab35; evidence line per disc)."""
    det.tta = res.to_dict()
    det.tta_discs = tta_disc_records(res, det.anchors)
    for lab, rec in det.tta_discs.items():
        det.anchors.discs[lab].evidence.append(
            f"test-time variants ({len(res.variants)}): found in {rec['presence']:.0%}, same name in "
            f"{rec['name_agree']:.0%}, largest centre shift {rec['spread_mm']:.1f} mm"
        )


def _conus_after_labels(study: Study, anchors: Anchors, cfg: Config) -> None:
    """Classical backend: locate the conus once the anchors are final (:mod:`.conus`); a failure keeps the labels.

    The conus locator is EXPERIMENTAL (uncalibrated; a visual audit on public data could not verify most tips):
    its confidence is capped and the record is marked ``experimental``.
    """
    from .conus import attach_conus

    try:
        attach_conus(study, anchors, cfg)
    except Exception as exc:  # the conus is additional: it must never sink the labels
        log.warning("conus locator failed: %s", exc)
        anchors.warnings.append(f"conus not located (locator failed: {type(exc).__name__})")
        return
    if isinstance(anchors.conus, dict):
        c = anchors.conus.get("confidence")
        if isinstance(c, (int, float)):
            anchors.conus["confidence"] = min(float(c), cfg.conus_max_confidence)
        anchors.conus["experimental"] = True


def label_spine(
    study: Study, config: Config | None = None, *, backend: str = "auto", fusion: FusionConfig | None = None
) -> Anchors:
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
    return label_spine_details(study, config, backend=backend, fusion=fusion).anchors


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
    """Keys of the stacks :func:`label_spine` reads: the primary sagittal, the second sagittal series, every sagittal
    stack (localizers included: the top-chain check, :mod:`.counting`, reads them), and every axial stack (midline
    refinement and numbering cues). Coronal and other series are not needed."""
    keys: list[str] = []
    primary = study.best("sag")
    if primary is not None:
        keys.append(primary.key)
        second = _second_sagittal(study, primary.key)
        if second is not None:
            keys.append(second.key)
    keys += [st.key for st in study.stacks.values() if st.plane == "sag"]
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
