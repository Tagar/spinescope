"""Sagittal alignment: listhesis per level, lumbar lordosis, segmental (intervertebral) angles, lateral screening record.

Everything is measured on the vertebral body endplates that :mod:`spinescope.anatomy.vertebrae` attaches to the anchors
(``Vertebra.body``: superior and inferior endplate corners in LPS mm, AP depth). Nothing is re-segmented here, so the
analysis costs no network time. Without that geometry (classical backend, missing sequence, no canal) the rows are
absent and the result carries warnings, never an exception.

Conventions (LPS: +x patient left, +y posterior, +z superior):

* Tilt of an endplate: ``atan2(dz, dy)`` of its anterior->posterior corner vector; positive = posterior end higher.
  Lordosis = tilt(S1 superior endplate) - tilt(L1 superior endplate): positive for a lordotic spine.
* Segmental (intervertebral) angle of a disc: tilt(lower body's superior endplate) - tilt(upper body's inferior
  endplate), the same sign convention. Its sum over the lumbar discs is the lordosis taken between those endplates.
* Translation, as in the findings schema (``Listhesis.translation_mm``): **positive = anterior** displacement of the
  upper body on the lower one. Measured as the offset of the upper body's posterior-inferior corner from the lower
  body's posterior-superior corner, projected on the lower superior endplate (a posterior-wall displacement in the
  Meyerding tradition; not checked line by line against the published Taillard procedure).
* Percent slip (Meyerding): |translation| / AP depth of the lower body x 100. Grade I to V for anterolisthesis only.

What is not claimed: a supine study shows less lordosis than standing (see docs/analysis/alignment.md), and a slip
on one supine study is a measurement, not evidence of instability. Level names carry their numbering status: a
level whose name is probable or unconfirmed is reported with that status in its row and in its notes.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ..anatomy.anchors import Anchors, split_disc
from ..anatomy.summary import STATUS_CONFIRMED, status_of
from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .base import Result, register, sagittal_stacks
from .evaluate import register_predictor
from .findings import Candidate, FieldRule, register_field

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: Side of the figure crop, mm: AP and superior-inferior extent around the disc centre.
FIG_AP_MM = 60.0
FIG_SI_MM = 50.0
FIG_SCALE = 4  # figure pixels per stack pixel (upscaled for reading)

#: Ordered listhesis grades (Meyerding) above the first edge.
GRADES: tuple[str, ...] = ("I", "II", "III", "IV", "V")


# ------------------------------------------------------------------------------------------- pure geometry
@dataclass(frozen=True)
class Endplate:
    """One vertebral endplate in the sagittal plane, LPS mm: anterior and posterior corners."""

    anterior: FloatArray
    posterior: FloatArray

    @property
    def length_mm(self) -> float:
        """AP length of the endplate (the AP depth of its body when it is the superior endplate)."""
        return float(np.linalg.norm(self.posterior - self.anterior))

    @property
    def tilt_deg(self) -> float:
        """Tilt, positive when the posterior corner is higher than the anterior one (see module docstring)."""
        d = self.posterior - self.anterior
        return float(math.degrees(math.atan2(d[2], d[1])))


def endplate_from_record(rec: Mapping[str, Any] | None) -> Endplate | None:
    """Endplate from a ``Vertebra.body`` endplate record (``{"anterior": [x,y,z], "posterior": [x,y,z], ...}``).

    The corners are the wall corners (``anterior_wall`` / ``posterior_wall``) when the labeller measured them, else the
    global-extent corners, which are biased for a tilted body (see :func:`~spinescope.anatomy.vertebrae.body_plane_geometry`).
    """
    if not rec:
        return None
    try:
        ant_key = "anterior_wall" if "anterior_wall" in rec else "anterior"
        post_key = "posterior_wall" if "posterior_wall" in rec else "posterior"
        ant = np.asarray(rec[ant_key], dtype=np.float64)
        post = np.asarray(rec[post_key], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        ant.shape != (3,)
        or post.shape != (3,)
        or not np.all(np.isfinite(ant))
        or not np.all(np.isfinite(post))
    ):
        return None
    if float(np.linalg.norm(post - ant)) < 1.0:  # a degenerate line is not an endplate
        return None
    return Endplate(ant, post)


def meyerding_grade(percent: float, edges: Sequence[float] | None = None) -> str:
    """Meyerding grade I-V of a slip given as a percentage of the lower body's AP depth (V = above the last edge)."""
    e = tuple(edges) if edges is not None else DEFAULT.alignment_grade_edges_pct
    a = abs(float(percent))
    for grade, edge in zip(GRADES, e, strict=False):
        if a <= edge:
            return grade
    return "V"


def listhesis(upper_inf: Endplate, lower_sup: Endplate, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Translation of the upper body on the lower one at one disc (posterior wall method, see module docstring).

    Returns ``translation_mm`` (anterior positive), ``percent`` (of the lower body's AP depth), ``grade`` (Meyerding,
    anterolisthesis only, else None), ``direction`` (``anterolisthesis`` | ``retrolisthesis`` | ``none``, with the
    threshold ``Config.alignment_listhesis_min_mm``) and ``lower_ap_mm``.
    """
    ap = lower_sup.posterior - lower_sup.anterior
    ap = ap / float(np.linalg.norm(ap))
    posterior_offset = float((upper_inf.posterior - lower_sup.posterior) @ ap)  # + = upper body posterior
    translation = -posterior_offset  # schema convention: anterior positive
    depth = lower_sup.length_mm
    percent = abs(translation) / depth * 100.0 if depth > 0 else float("nan")
    minimum = cfg.alignment_listhesis_min_mm
    if translation >= minimum:
        direction = "anterolisthesis"
    elif translation <= -minimum:
        direction = "retrolisthesis"
    else:
        direction = "none"
    grade = (
        meyerding_grade(percent, cfg.alignment_grade_edges_pct) if direction == "anterolisthesis" else None
    )
    return {
        "translation_mm": round(translation, 2),
        "posterior_offset_mm": round(posterior_offset, 2),
        "percent": round(percent, 1),
        "grade": grade,
        "direction": direction,
        "lower_ap_mm": round(depth, 2),
    }


def segmental_angle_deg(upper_inf: Endplate, lower_sup: Endplate) -> float:
    """Intervertebral (disc wedge) angle: tilt of the lower superior endplate minus tilt of the upper inferior endplate."""
    return lower_sup.tilt_deg - upper_inf.tilt_deg


def lordosis_deg(start: Endplate, end: Endplate) -> float:
    """Lordosis between two endplates: tilt(end) - tilt(start). Start is the L1 superior endplate, end the S1 superior
    endplate (L1-S1) or the L5 inferior endplate (L1-L5)."""
    return end.tilt_deg - start.tilt_deg


# ------------------------------------------------------------------------------------------- anchors -> rows
def _body_records(anchors: Anchors) -> dict[str, dict[str, Any]]:
    return {name: v.body for name, v in anchors.vertebrae.items() if v.body}


def _level_name(lab: str, status: str) -> str:
    return lab if status == STATUS_CONFIRMED else f"{lab} ({status})"


def _body_conf(rec: Mapping[str, Any] | None) -> float:
    return float(rec.get("confidence") or 0.0) if rec else 0.0


#: The S1 superior endplate is not used. The labeller's endplate fit takes the line over the middle columns of the body's
#: global AP extent; for a steep sacrum that includes the sloped posterior wall, so the phantom's S1 endplate (truth 28
#: degrees) comes out at about 13 degrees. Ground-truth masks scored with the same fit cancel this bias, so SPIDER
#: comparisons cannot reveal it. Until the endplate fit is corrected (anatomy/vertebrae.py), lordosis is L1-L5 and the
#: lowest disc (L5-S1) is not assessed. Pinned here so the decision is visible in one place.
SACRAL_ENDPLATE_USABLE = False
SACRAL_NOTE = (
    "S1 superior endplate not used: the labeller's endplate fit is biased for a steep sacrum (phantom: about 13 "
    "degrees against 28 true); lordosis is reported as L1-L5"
)


def _lordosis(bodies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Lumbar lordosis from the L1 superior endplate to the L5 inferior endplate (L1-L5, see SACRAL_NOTE)."""
    l1 = endplate_from_record((bodies.get("L1") or {}).get("superior_endplate"))
    l5 = endplate_from_record((bodies.get("L5") or {}).get("inferior_endplate"))
    out: dict[str, Any] = {"value": None, "method": "L1-L5", "sacrum_used": False, "notes": [SACRAL_NOTE]}
    if l1 is None or l5 is None:
        out["notes"].append("L1 superior or L5 inferior endplate not measured: lordosis not assessed")
        out["method"] = None
        return out
    out["value"] = round(lordosis_deg(l1, l5), 1)
    return out


def _stack_for(study: Study, anchors: Anchors) -> Stack | None:
    """The sagittal stack the labeller measured (its key is ``anchors.source_stack``), else the best T2 sagittal."""
    stack = study.stacks.get(anchors.source_stack) if anchors.source_stack else None
    if stack is None:
        stack = sagittal_stacks(study).get("t2")
    return stack


def _window(vol: NDArray[np.floating]) -> tuple[float, float]:
    lo, hi = (float(v) for v in np.percentile(vol, (1.0, 99.5)))
    if not hi > lo:
        return float(np.min(vol)), float(np.max(vol)) + 1.0
    return lo, hi


def _figure(
    stack: Stack,
    k: int,
    centre: FloatArray,
    lines: Sequence[tuple[FloatArray, FloatArray, tuple[int, int, int]]],
) -> NDArray[np.uint8] | None:
    """Crop of slice ``k`` around ``centre`` (LPS) with the drawn ``lines`` (LPS point pairs and colours)."""
    if not (0 <= k < stack.n):
        return None
    img = np.asarray(stack.vol[k], dtype=np.float64)
    lo, hi = _window(img)
    gray = np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * 255.0
    rc = stack.lps_to_pix(centre)
    r0, c0 = rc[1], rc[2]
    half_r = FIG_SI_MM / stack.row_mm
    half_c = FIG_AP_MM / stack.col_mm
    r_lo, r_hi = max(int(r0 - half_r), 0), min(int(r0 + half_r), img.shape[0])
    c_lo, c_hi = max(int(c0 - half_c), 0), min(int(c0 + half_c), img.shape[1])
    if r_hi - r_lo < 4 or c_hi - c_lo < 4:
        return None
    crop = gray[r_lo:r_hi, c_lo:c_hi].astype(np.uint8)
    pil = Image.fromarray(crop, mode="L").resize(
        ((c_hi - c_lo) * FIG_SCALE, (r_hi - r_lo) * FIG_SCALE), Image.Resampling.BILINEAR
    )
    rgb = Image.merge("RGB", (pil, pil, pil))
    draw = ImageDraw.Draw(rgb)
    for a, b, colour in lines:
        pa = stack.lps_to_pix(a)
        pb = stack.lps_to_pix(b)
        xa = (pa[2] - c_lo) * FIG_SCALE
        ya = (pa[1] - r_lo) * FIG_SCALE
        xb = (pb[2] - c_lo) * FIG_SCALE
        yb = (pb[1] - r_lo) * FIG_SCALE
        draw.line([(xa, ya), (xb, yb)], fill=colour, width=2)
    return np.asarray(rgb, dtype=np.uint8)


def _level_figure(
    stack: Stack,
    lab: str,
    upper: Mapping[str, Any],
    lower: Mapping[str, Any],
    row: Mapping[str, Any],
    anchors: Anchors,
) -> NDArray[np.uint8] | None:
    up_inf = endplate_from_record(upper.get("inferior_endplate"))
    lo_sup = endplate_from_record(lower.get("superior_endplate"))
    if up_inf is None or lo_sup is None:
        return None
    k = int(lower.get("slice", upper.get("slice", 0)))
    disc = anchors.discs.get(lab)
    centre = (
        np.asarray(disc.center, dtype=np.float64)
        if disc is not None
        else 0.5 * (up_inf.posterior + lo_sup.posterior)
    )
    yellow, cyan, red = (255, 210, 0), (0, 220, 255), (255, 60, 60)
    lines: list[tuple[FloatArray, FloatArray, tuple[int, int, int]]] = [
        (lo_sup.anterior, lo_sup.posterior, yellow),
        (up_inf.anterior, up_inf.posterior, yellow),
        (up_inf.posterior, lo_sup.posterior, cyan),
    ]
    if row.get("translation_mm") is not None:
        # the translation, drawn from the lower posterior corner along the lower endplate
        ap = lo_sup.posterior - lo_sup.anterior
        ap = ap / float(np.linalg.norm(ap))
        proj = lo_sup.posterior + ap * float(row.get("posterior_offset_mm") or 0.0)
        lines.append((lo_sup.posterior, proj, red))
    return _figure(stack, k, centre, lines)


def _sup_tilt_row(name: str, bodies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    rec = bodies.get(name)
    if not rec:
        return None
    sup = endplate_from_record(rec.get("superior_endplate"))
    inf = endplate_from_record(rec.get("inferior_endplate"))
    return {
        "scope": "body",
        "body": name,
        "superior_tilt_deg": round(sup.tilt_deg, 2) if sup else None,
        "inferior_tilt_deg": round(inf.tilt_deg, 2) if inf else None,
        "ap_depth_mm": rec.get("ap_depth_mm"),
        "confidence": round(_body_conf(rec), 3),
        "slice": rec.get("slice"),
    }


def lateral_screening(anchors: Anchors) -> dict[str, Any]:
    """The lateral-curvature screening record of lab30 (``Anchors.alignment['coronal']``), passed through unchanged.

    Unvalidated: the amplitude in mm is the primary figure, the Cobb-like angle is uncalibrated. Nothing here upgrades
    that: the record keeps its own label and method.
    """
    rec = anchors.alignment.get("coronal") if anchors.alignment else None
    if not rec:
        return {"measured": False, "note": "lateral curvature not measured (no record on the anchors)"}
    out = {k: v for k, v in rec.items() if k != "knots"}
    out["measured"] = bool(rec.get("valid"))
    out.setdefault("label", "lateral curvature screening (unvalidated)")
    return out


def measure(study: Study, anchors: Anchors, cfg: Config = DEFAULT) -> Result:
    """Rows, flags and figures of the sagittal alignment (see :func:`analyze`, the registered entry point)."""
    res = Result(name="alignment")
    status = status_of(anchors)
    bodies = _body_records(anchors)
    stack = _stack_for(study, anchors)
    res.meta.update(
        method="endplate geometry of the labeller's bodies (Vertebra.body); posterior-wall listhesis",
        name_status=status,
        posture="supine",
        series_key=stack.key if stack is not None else "",
        thresholds={
            "listhesis_min_mm": cfg.alignment_listhesis_min_mm,
            "meyerding_edges_pct": list(cfg.alignment_grade_edges_pct),
        },
        slices_checked="mid-sagittal slice of each body (labeller); no neighbour slices kept in Anchors",
    )
    if not anchors.discs:
        res.warnings.append("alignment not measured: no discs")
        return res
    if not bodies:
        res.warnings.append("alignment not measured: no vertebral body endplates (learned labeller needed)")
    for lab in anchors.disc_labels():
        upper, lower = split_disc(lab)
        ub, lb = bodies.get(upper), bodies.get(lower)
        name = _level_name(lab, status)
        row: dict[str, Any] = {
            "scope": "level",
            "disc": lab,
            "level": name,
            "name_status": status,
            "upper": upper,
            "lower": lower,
            "translation_mm": None,
            "percent": None,
            "grade": None,
            "direction": None,
            "segmental_deg": None,
            "confidence": 0.0,
            "slice_upper": ub.get("slice") if ub else None,
            "slice_lower": lb.get("slice") if lb else None,
            "notes": "",
        }
        up_inf = endplate_from_record((ub or {}).get("inferior_endplate"))
        lo_sup = endplate_from_record((lb or {}).get("superior_endplate"))
        if lower == "S1" and not SACRAL_ENDPLATE_USABLE:
            row["notes"] = (
                "not assessed: the lowest disc needs the sacral superior endplate (see SACRAL_NOTE)"
            )
            res.rows.append(row)
            continue
        if up_inf is None or lo_sup is None:
            row["notes"] = "endplates not measured on both sides of the disc"
            res.rows.append(row)
            continue
        lst = listhesis(up_inf, lo_sup, cfg)
        seg = segmental_angle_deg(up_inf, lo_sup)
        complete = bool((ub or {}).get("complete")) and bool((lb or {}).get("complete"))
        conf = min(_body_conf(ub), _body_conf(lb))
        conf = conf if complete else conf * 0.5
        notes: list[str] = []
        if not complete:
            notes.append(
                "one endplate of a body is missing (the body touches the field edge or a missing disc)"
            )
        if status != STATUS_CONFIRMED:
            notes.append(f"level name {status}: the position is measured, the name is not confirmed")
        row.update(
            translation_mm=lst["translation_mm"],
            posterior_offset_mm=lst["posterior_offset_mm"],
            percent=lst["percent"],
            grade=lst["grade"],
            direction=lst["direction"],
            lower_ap_mm=lst["lower_ap_mm"],
            segmental_deg=round(seg, 2),
            upper_inferior_tilt_deg=round(up_inf.tilt_deg, 2),
            lower_superior_tilt_deg=round(lo_sup.tilt_deg, 2),
            confidence=round(float(conf), 3),
            notes="; ".join(notes),
        )
        res.rows.append(row)
        if lst["direction"] != "none":
            res.flags.append(
                {
                    "kind": lst["direction"],
                    "level": name,
                    "disc": lab,
                    "translation_mm": lst["translation_mm"],
                    "percent": lst["percent"],
                    "confidence": row["confidence"],
                    "note": "measured translation on one supine study; not a diagnosis of instability",
                }
            )
        if stack is not None:
            fig = _level_figure(stack, lab, ub or {}, lb or {}, row, anchors)
            if fig is not None:
                res.figures[f"listhesis_{lab}"] = fig
    for name in ("L1", "L2", "L3", "L4", "L5"):
        srow = _sup_tilt_row(name, bodies)
        if srow is not None:
            res.rows.append(srow)
    lord = _lordosis(bodies)
    res.rows.append(
        {
            "scope": "global",
            "name": "lumbar_lordosis_deg",
            "value": lord["value"],
            "method": lord["method"],
            "sacrum_used": lord["sacrum_used"],
            "name_status": status,
            "notes": "; ".join(lord["notes"]),
            "posture": "supine (underestimates standing lordosis)",
        }
    )
    res.warnings.append(SACRAL_NOTE)
    lat = lateral_screening(anchors)
    res.rows.append(
        {
            "scope": "global",
            "name": "lateral_curvature_screening",
            "value": lat.get("amplitude_mm"),
            "cobb_like_uncalibrated_deg": lat.get("cobb_like_uncalibrated_deg"),
            "direction": lat.get("direction"),
            "measured": lat.get("measured"),
            "label": lat.get("label", "lateral curvature screening (unvalidated)"),
            "notes": "screening record from lab30 (unvalidated); not a Cobb measurement",
        }
    )
    return res


# ------------------------------------------------------------------------------------------- registry entry
@register("alignment")
def analyze(study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Sagittal alignment on the labeller's body endplates: per-level listhesis (mm, %, Meyerding), segmental angles,
    lumbar lordosis, and the lab30 lateral screening record. Returns warnings, not exceptions, when geometry is missing."""
    return measure(study, anchors, config or DEFAULT)


# ------------------------------------------------------------------------------------------- findings fields
def _level_row(res: Result, level: str) -> dict[str, Any] | None:
    for r in res.rows:
        if r.get("scope") == "level" and r.get("disc") == level:
            return r
    return None


def _extract_direction(res: Result, level: str) -> Candidate | None:
    r = _level_row(res, level)
    if r is None or r.get("direction") is None:
        return None
    return Candidate(
        r["direction"],
        float(r["confidence"]),
        notes=f"measured translation {r['translation_mm']} mm (posterior wall method)",
        slices=tuple(int(s) for s in (r.get("slice_upper"), r.get("slice_lower")) if s is not None),
    )


def _extract_translation(res: Result, level: str) -> Candidate | None:
    r = _level_row(res, level)
    if r is None or r.get("translation_mm") is None:
        return None
    return Candidate(
        float(r["translation_mm"]),
        float(r["confidence"]),
        notes="positive = anterior; mid-sagittal body endplates",
        slices=tuple(int(s) for s in (r.get("slice_upper"), r.get("slice_lower")) if s is not None),
    )


def _extract_grade(res: Result, level: str) -> Candidate | None:
    r = _level_row(res, level)
    if r is None or r.get("direction") is None:
        return None
    grade = r.get("grade")
    if grade is None:
        value: Any = "none" if r["direction"] == "none" else None
        note = "not graded: Meyerding grades anterolisthesis only"
    else:
        value = grade
        note = f"{r['percent']} % of the lower body's AP depth"
    return Candidate(
        value,
        float(r["confidence"]),
        notes=note,
        slices=tuple(int(s) for s in (r.get("slice_upper"), r.get("slice_lower")) if s is not None),
    )


def _listhesis_figure(level: str) -> str:
    return f"listhesis_{level}"


register_field(FieldRule("listhesis", "direction", "alignment", _extract_direction, figure=_listhesis_figure))
register_field(
    FieldRule("listhesis", "translation_mm", "alignment", _extract_translation, figure=_listhesis_figure)
)
register_field(
    FieldRule("listhesis", "meyerding_grade", "alignment", _extract_grade, figure=_listhesis_figure)
)


# ------------------------------------------------------------------------------------------- harness predictor
def spondylolisthesis_prediction(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
    """SPIDER ``spondylolisthesis`` predictor: 1 when |translation| >= ``alignment_listhesis_min_mm`` (either direction),
    0 when measured below it, None when not measured. The score is |translation| (AUC)."""
    rows = {str(r.get("disc")): r for r in res.rows if r.get("scope") == "level"}
    threshold = DEFAULT.alignment_listhesis_min_mm
    out: dict[str, tuple[Any, float | None]] = {}
    for lab in anchors.discs:
        r = rows.get(lab)
        t = r.get("translation_mm") if r else None
        if t is None:
            out[lab] = (None, None)
            continue
        mag = abs(float(t))
        out[lab] = (1 if mag >= threshold else 0, mag)
    return out


register_predictor("alignment", "spondylolisthesis", spondylolisthesis_prediction)


def threshold_sweep(
    scores: Sequence[float],
    truth: Sequence[int],
    cases: Sequence[str],
    thresholds: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sensitivity, specificity and PPV of ``score >= threshold`` for each threshold, with 95 % percentile bootstrap
    intervals that resample **cases** (discs of one patient are not independent). ``scores`` are |translation| in mm.

    Youden's J (sensitivity + specificity - 1) is reported for each threshold so that a threshold can be chosen on one
    split and applied to another. Deterministic for a given seed.
    """
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(truth, dtype=np.int64)
    if x.shape != y.shape or x.shape[0] != len(cases):
        raise ValueError("scores, truth and cases must have the same length")
    case_ids = sorted(set(cases))
    idx_of = {c: np.nonzero(np.asarray(cases) == c)[0] for c in case_ids}
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(case_ids), size=len(case_ids)) for _ in range(n_boot)]

    def rates(sel: NDArray[np.intp], t: float) -> tuple[float, float, float]:
        pred = x[sel] >= t
        yy = y[sel] == 1
        tp = float(np.sum(pred & yy))
        fn = float(np.sum(~pred & yy))
        tn = float(np.sum(~pred & ~yy))
        fp = float(np.sum(pred & ~yy))
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        ppv = tp / (tp + fp) if tp + fp else float("nan")
        return sens, spec, ppv

    out: list[dict[str, Any]] = []
    everything = np.arange(x.shape[0])
    for t in thresholds:
        sens, spec, ppv = rates(everything, float(t))
        boots: dict[str, list[float]] = {"sensitivity": [], "specificity": [], "ppv": []}
        for d in draws:
            sel = np.concatenate([idx_of[case_ids[i]] for i in d]) if n_boot else everything
            s_b, p_b, v_b = rates(sel, float(t))
            for key, val in (("sensitivity", s_b), ("specificity", p_b), ("ppv", v_b)):
                if np.isfinite(val):
                    boots[key].append(val)

        def ci(vals: list[float]) -> list[float] | None:
            if not vals:
                return None
            lo, hi = np.percentile(vals, [2.5, 97.5])
            return [round(float(lo), 3), round(float(hi), 3)]

        out.append(
            {
                "threshold_mm": float(t),
                "tp": int(np.sum((x >= t) & (y == 1))),
                "fp": int(np.sum((x >= t) & (y == 0))),
                "fn": int(np.sum((x < t) & (y == 1))),
                "tn": int(np.sum((x < t) & (y == 0))),
                "sensitivity": round(sens, 3) if np.isfinite(sens) else None,
                "sensitivity_ci": ci(boots["sensitivity"]),
                "specificity": round(spec, 3) if np.isfinite(spec) else None,
                "specificity_ci": ci(boots["specificity"]),
                "ppv": round(ppv, 3) if np.isfinite(ppv) else None,
                "ppv_ci": ci(boots["ppv"]),
                "youden_j": round(sens + spec - 1.0, 3) if np.isfinite(sens) and np.isfinite(spec) else None,
            }
        )
    return out


__all__ = [
    "Endplate",
    "analyze",
    "endplate_from_record",
    "lateral_screening",
    "listhesis",
    "lordosis_deg",
    "measure",
    "meyerding_grade",
    "segmental_angle_deg",
    "spondylolisthesis_prediction",
    "threshold_sweep",
]
