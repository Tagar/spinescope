"""Lumbosacral transitional vertebrae (LSTV): is one suspected, of which kind, what Castellvi grade, and what it does to the names.

Two anatomical variants change the level names by one level without any change in the disc count:

* **sacralised L5**: L5 is partly or fully fused to the sacrum. The lowest visible disc is the true L4-L5, but the
  count names it L5-S1, so every name is one level too low. Anatomical offset +1.
* **lumbarised S1**: S1 is free (a lumbar-type body with a full disc below it). The lowest visible disc is the true
  L5-L6, but the count names it L5-S1. Anatomical offset -1 (only the lowest disc is renamed).

Naming convention, stated here and in every output (lab28): disc names count up from the lowest disc, which is
called L5-S1 (the SPIDER label 1 = lowest lumbar vertebra convention, and this pipeline's numbering). The anatomical
name of a disc is the lumbar-type vertebra above it. This module reports the count names that are stored in the
anchors and the anatomical suggestion next to them. It never renames anything; a reviewer or a verified
Castellvi/iliolumbar-ligament reading decides (Konin & Walz 2010; Hughes & Saifuddin 2004, 2006).

Evidence, one :class:`Cue` each (a likelihood ratio over ``none`` / ``sacralised_l5`` / ``lumbarised_s1``):

* ``disc_height`` (sagittal): the lowest disc is thin or hypoplastic against the discs above (a rudimentary disc is
  the sacralised sign in Konin & Walz 2010). Confounded by degeneration. Measured on nearly every study; thin or
  hypoplastic in about 5 % of the real-data spines of this project (lab28 audit).
* ``s1_block_height`` (sagittal): the body below the lowest disc is tall against the lumbar bodies (a fused L5-S1
  block). NOT measurable with the current labeller: it has no sacrum class, and no real-data record had the body
  below the lowest disc measured (0 of 385). Code kept for a labeller that segments the sacrum.
* ``l5_wedge`` (sagittal): the body above the lowest disc is anteriorly wedged. Reported with weight 1.0: a wedged
  L5 is in 59 % of the real-data spines (226 of 385), so it does not separate the variants.
* ``numbering_posterior`` (fusion): counted only when the fusion calls a shift a candidate (its best non-zero offset
  at least ``fusion_candidate_min``, below its apply threshold); the fusion's own weak votes sit at their floors and
  are not evidence (lab28 audit: counting them gave every real flag).
* ``lumbosacral_check``: a lower disc was removed as a sacral remnant. In a lumbarised S1 the removed disc can be
  the full S1-S2 disc (Carrino et al. 2011 found a complete S1-S2 disc was associated with LSTV).
* ``castellvi`` (axial or coronal, only when a reader or a future measurement supplies it): transverse process
  size and contact with the sacral ala. Castellvi grades describe the L5 transverse process, so they point to
  sacralisation. Not measured automatically here: no transverse-process segmentation exists, and no public data
  in this project carries the labels to validate one.

Likelihood ratios and the prior are judgement defaults in :class:`LstvConfig` (no LSTV label exists in SPIDER,
Sudirman or qx5rt); they are not calibrated. The status is ``suspected`` (P >= 0.5), ``possible`` (P >= 0.2) or
``not_suspected``; ``not_assessed`` when no cue could be measured.

Sequences: each cue names the sequence it needs (``Cue.sequence``); a sagittal-only study can give at most a
suspicion, never a Castellvi grade (Farshad-Amacker et al. 2014 used coronal MRI for reliable classification;
O'Driscoll et al. 1996 found sagittal MRI patterns do not map one-to-one to Castellvi types).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .anchors import Anchors, Vertebra
from .numbering_checks import OFFSETS, relabel_discs

log = logging.getLogger(__name__)

LSTV_METHOD = (
    "lstv/1 sagittal morphology, fused numbering posterior, lumbosacral check; Castellvi only from a supplied "
    "axial/coronal record; names are suggested, never applied"
)

NONE = "none"
SACRAL = "sacralised_l5"
LUMBAR = "lumbarised_s1"
UNRESOLVED = "unresolved"  # suspected, but the two kinds are about equally likely
KINDS: tuple[str, ...] = (NONE, SACRAL, LUMBAR)
#: Anatomical offset of the lowest disc (numbering_checks convention: +1 = L4-L5, -1 = L5-L6).
OFFSET_OF: dict[str, int] = {NONE: 0, SACRAL: 1, LUMBAR: -1}
#: Kind of each fusion offset (the fusion posterior uses offsets -1, 0, +1).
_KIND_OF_OFFSET: dict[int, str] = {-1: LUMBAR, 0: NONE, 1: SACRAL}

STATUS_SUSPECTED = "suspected"
STATUS_POSSIBLE = "possible"
STATUS_NOT_SUSPECTED = "not_suspected"
STATUS_NOT_ASSESSED = "not_assessed"

#: Sequence each cue needs, for the docs and for the output (``Cue.sequence``).
SEQ_SAGITTAL = "sagittal T1 or T2 through the lumbosacral junction, mid-sagittal slice, bone below the lowest disc in view"
SEQ_FUSION = "sagittal plus oblique axial slabs (T1) or a second sagittal series, as used by fusion"
SEQ_LSC = "sagittal (the lumbosacral CNN decision on the mid-sagittal slice)"
SEQ_AXIAL = (
    "axial T1 or T2 through L5 and S1 (3-5 mm, sacral ala and L5 transverse processes both in view), or coronal "
    "T1 or T2 covering both sacral ala and the L5 transverse processes"
)


@dataclass(frozen=True)
class LstvConfig:
    """Priors, thresholds and likelihood ratios. Every value is a judgement default; none is calibrated on data."""

    #: Prior probability of any transitional vertebra. Konin & Walz 2010 (doi 10.3174/ajnr.A2036) quote about
    #: 4-30 % in the literature they review; 15 % is the middle of that range (judgement). Split evenly.
    prior_transitional: float = 0.15
    #: Lowest disc height / median height of the discs above. At or below ``hypoplastic_ratio``: hypoplastic (the
    #: sagittal sign of a rudimentary disc in sacralisation). Judgement: an ordinary L5-S1 is often 0.8 to 0.9 of
    #: L4-L5, so 0.6 is a clear gap; the band up to ``thin_ratio`` is weaker.
    hypoplastic_ratio: float = 0.6
    thin_ratio: float = 0.75
    #: Height of the body below the lowest disc / median height of the lumbar bodies above it. At or above this:
    #: a fused block (judgement; a normal S1 segment is about one lumbar body high).
    fused_block_ratio: float = 1.5
    #: Anterior / posterior height of the body above the lowest disc. At or above this: wedged (weak, judgement).
    wedge_ratio: float = 1.2
    #: Castellvi type I: craniocaudal transverse process size, mm (Castellvi et al. 1984; Konin & Walz 2010: 19 mm).
    tp_enlarged_mm: float = 19.0
    #: Likelihood ratios against ``none`` for each cue (judgement; see the module docstring).
    lr_hypoplastic: float = 3.0
    lr_thin: float = 1.5
    lr_fused_block: float = 2.0
    #: Wedged L5 body (anterior / posterior height >= wedge_ratio) fires in about 59 % of the real-data spines of this
    #: project (lab28 audit: 226 of 385), so it does not separate the variants: reported, weight 1.0 (judgement).
    lr_wedge: float = 1.0
    #: Fusion posterior (best shift) at or above this is a candidate shift; then it counts with this ratio.
    fusion_candidate_min: float = 0.5
    lr_fusion_candidate: float = 3.0
    lr_lumbosacral_drop: float = 2.0  # lumbarised kind, when a lower disc was removed as sacral
    lr_normal_measured: float = 1.2  # none, for a measured normal sagittal cue (weak)
    lr_castellvi_i: float = 3.0
    lr_castellvi_ii: float = 6.0
    lr_castellvi_iii: float = 10.0
    lr_castellvi_iv: float = 10.0
    lr_castellvi_normal: float = 2.0  # none, both transverse processes measured normal
    #: The two kinds are told apart only when one is at least this many times the other (else "unresolved").
    kind_ratio_min: float = 1.5
    #: Status thresholds on P(transitional).
    suspect_min: float = 0.5
    possible_min: float = 0.2
    #: Floor for any likelihood before the product, so one cue cannot veto (as in numbering_checks).
    floor: float = 1e-3


DEFAULT_LSTV = LstvConfig()


@dataclass
class Cue:
    """One piece of evidence: a likelihood per kind (``None`` when the cue could not be measured)."""

    name: str
    like: dict[str, float] | None
    detail: str
    sequence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cue": self.name,
            "measured": self.like is not None,
            "likelihood": None
            if self.like is None
            else {k: round(float(v), 4) for k, v in self.like.items()},
            "detail": self.detail,
            "sequence": self.sequence,
        }


def _like(sacral: float = 1.0, none: float = 1.0, lumbar: float = 1.0) -> dict[str, float]:
    return {SACRAL: sacral, NONE: none, LUMBAR: lumbar}


def _kind_arg(kind: str) -> str:
    return {NONE: "none", SACRAL: "sacral", LUMBAR: "lumbar"}[kind]


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


# ------------------------------------------------------------------ sagittal cues


def disc_cue(anchors: Anchors, cfg: LstvConfig = DEFAULT_LSTV) -> Cue:
    """Height of the lowest disc against the median of the discs above it."""
    labels = anchors.disc_labels()  # superior -> inferior
    if len(labels) < 3:
        return Cue(
            "disc_height", None, "fewer than three discs: lowest-disc height not compared", SEQ_SAGITTAL
        )
    low = labels[-1]
    h_low = anchors.discs[low].height_mm
    above: list[float] = []
    for lab in labels[:-1]:
        hh = anchors.discs[lab].height_mm
        if hh:
            above.append(float(hh))
    if not h_low or len(above) < 2:
        return Cue("disc_height", None, "disc heights not measured", SEQ_SAGITTAL)
    ratio = float(h_low) / _median(above)
    ev = (
        f"lowest disc {low} {float(h_low):.1f} mm vs median {_median(above):.1f} mm above (ratio {ratio:.2f})"
    )
    if ratio <= cfg.hypoplastic_ratio:
        return Cue("disc_height", _like(sacral=cfg.lr_hypoplastic), ev + ": hypoplastic", SEQ_SAGITTAL)
    if ratio <= cfg.thin_ratio:
        return Cue("disc_height", _like(sacral=cfg.lr_thin), ev + ": thin", SEQ_SAGITTAL)
    return Cue("disc_height", _like(none=cfg.lr_normal_measured), ev + ": not thin", SEQ_SAGITTAL)


def _bodies_by_j(anchors: Anchors) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for v in anchors.vertebrae.values():
        b = v.body if isinstance(v, Vertebra) else None
        if isinstance(b, dict) and b.get("discs_below") is not None:
            out[int(b["discs_below"])] = b
    return out


def body_cues(anchors: Anchors, cfg: LstvConfig = DEFAULT_LSTV) -> list[Cue]:
    """Body below the lowest disc (fused block?) and body above it (wedged?). Needs the learned labeller's bodies."""
    bodies = _bodies_by_j(anchors)
    s1 = bodies.get(0)
    s1_h = None if s1 is None else (s1.get("height_mm") or {}).get("mean")
    ref = [
        float(b["height_mm"]["mean"])
        for j, b in bodies.items()
        if 1 <= j <= 4 and (b.get("height_mm") or {}).get("mean")
    ]
    cues: list[Cue] = []
    if s1_h is None or len(ref) < 2:
        cues.append(
            Cue(
                "s1_block_height",
                None,
                "bone below the lowest disc or lumbar bodies not measured",
                SEQ_SAGITTAL,
            )
        )
    else:
        ratio = float(s1_h) / _median(ref)
        ev = f"body below lowest disc {float(s1_h):.1f} mm vs median lumbar body {_median(ref):.1f} mm (ratio {ratio:.2f})"
        if ratio >= cfg.fused_block_ratio:
            cues.append(
                Cue("s1_block_height", _like(sacral=cfg.lr_fused_block), ev + ": tall block", SEQ_SAGITTAL)
            )
        else:
            cues.append(Cue("s1_block_height", _like(none=cfg.lr_normal_measured), ev, SEQ_SAGITTAL))
    l5 = bodies.get(1)
    hl5 = None if l5 is None else l5.get("height_mm") or {}
    if not hl5 or not hl5.get("anterior") or not hl5.get("posterior"):
        cues.append(
            Cue(
                "l5_wedge",
                None,
                "body above the lowest disc: anterior and posterior heights not measured",
                SEQ_SAGITTAL,
            )
        )
    else:
        w = float(hl5["anterior"]) / float(hl5["posterior"])
        ev = f"body above lowest disc: anterior/posterior height {w:.2f}"
        if w >= cfg.wedge_ratio:
            cues.append(Cue("l5_wedge", _like(sacral=cfg.lr_wedge), ev + ": wedged", SEQ_SAGITTAL))
        else:
            cues.append(
                Cue(
                    "l5_wedge",
                    None,
                    ev + ": not wedged (no weight: a squared or lordotic body is normal)",
                    SEQ_SAGITTAL,
                )
            )
    return cues


# ------------------------------------------------------------- numbering cues


def numbering_cues(anchors: Anchors, cfg: LstvConfig = DEFAULT_LSTV) -> list[Cue]:
    """The fused offset posterior and the lumbosacral check record, read from ``anchors.numbering``."""
    num = anchors.numbering or {}
    cues: list[Cue] = []
    post = num.get("posterior")
    if isinstance(post, dict) and post:
        parts = [f"{o:+d}: {float(post.get(str(o), 0.0)):.2f}" for o in OFFSETS]
        shifts = {o: float(post.get(str(o), 0.0)) for o in OFFSETS if o != 0}
        best = max(shifts, key=lambda o: shifts[o])
        if shifts[best] >= cfg.fusion_candidate_min:
            # The fusion itself calls a shift a candidate (below its apply threshold): a real, if weak, signal.
            cues.append(
                Cue(
                    "numbering_posterior",
                    _like(**{_kind_arg(_KIND_OF_OFFSET[best]): cfg.lr_fusion_candidate}),
                    "fused offset posterior " + ", ".join(parts) + f": shift {best:+d} is a candidate",
                    SEQ_FUSION,
                )
            )
        else:
            # Below the candidate level the fusion's votes are at their floors: their posterior sits near the prior.
            # Counting it would turn vote noise into evidence (found in the lab28 visual audit), so it is neutral.
            cues.append(
                Cue(
                    "numbering_posterior",
                    None,
                    "fused offset posterior " + ", ".join(parts) + ": no shift candidate (not used)",
                    SEQ_FUSION,
                )
            )
    else:
        cues.append(
            Cue("numbering_posterior", None, "no fused numbering posterior in the record", SEQ_FUSION)
        )
    lumbo = num.get("lumbosacral")
    if isinstance(lumbo, dict) and lumbo.get("status"):
        drop = int(lumbo.get("applied_drop", 0) or 0)
        status = str(lumbo.get("status"))
        if drop > 0:
            cues.append(
                Cue(
                    "lumbosacral_check",
                    _like(lumbar=cfg.lr_lumbosacral_drop),
                    f"{drop} lower disc(s) removed as sacral: can be the full S1-S2 disc of a lumbarised S1",
                    SEQ_LSC,
                )
            )
        else:
            cues.append(
                Cue("lumbosacral_check", None, f"lumbosacral check {status}: no sacral disc removed", SEQ_LSC)
            )
    else:
        cues.append(Cue("lumbosacral_check", None, "lumbosacral check not run", SEQ_LSC))
    return cues


# -------------------------------------------------------------- Castellvi


def side_grade(rec: dict[str, Any] | None, cfg: LstvConfig = DEFAULT_LSTV) -> str | None:
    """Castellvi grade of one side: ``I``, ``II``, ``III``, ``normal`` (measured, none), or None (not assessed).

    ``rec``: ``{"tp_cc_mm": float | None, "contact": "none" | "pseudoarticulation" | "fusion" | "unknown"}``.
    Contact decides II and III; the transverse process size decides I (craniocaudal, >= ``tp_enlarged_mm``).
    """
    if not rec:
        return None
    contact = str(rec.get("contact", "unknown"))
    if contact == "fusion":
        return "III"
    if contact == "pseudoarticulation":
        return "II"
    if contact == "none":
        tp = rec.get("tp_cc_mm")
        if tp is None:
            return None
        return "I" if float(tp) >= cfg.tp_enlarged_mm else "normal"
    return None


_RANK = {"I": 1, "II": 2, "III": 3}


def castellvi_grade(transverse: dict[str, Any] | None, cfg: LstvConfig = DEFAULT_LSTV) -> dict[str, Any]:
    """Grade from a left/right record: I, IIa/IIb, IIIa/IIIb, IV (II one side, III the other), normal, or not assessed.

    Subtype ``a`` when one side carries the grade, ``b`` when both do (Castellvi 1984; Konin & Walz 2010). When the
    sides differ in grade without II and III together, the more severe side gives the type. A side that is not
    assessed is noted, and the grade then holds for the other side only.
    """
    sides = {s: side_grade((transverse or {}).get(s), cfg) for s in ("left", "right")}
    known = {s: g for s, g in sides.items() if g not in (None, "normal")}
    unknown = [s for s, g in sides.items() if g is None]
    note = [f"{s} side not assessed" for s in unknown]
    if not known:
        if unknown or not transverse:
            status = "not_assessed"
            grade = None
        else:
            status = "measured"
            grade = "normal"
    else:
        status = "measured"
        present = set(known.values())
        if present == {"II", "III"}:
            grade = "IV"
        else:
            worst = max(present, key=lambda g: _RANK[g])
            n_worst = sum(1 for g in known.values() if g == worst)
            grade = worst + ("b" if n_worst == 2 else "a")
        if unknown:
            note.append("grade is from the assessed side(s) only")
    return {"grade": grade, "status": status, "sides": sides, "notes": note}


def castellvi_cue(
    transverse: dict[str, Any] | None, cfg: LstvConfig = DEFAULT_LSTV
) -> tuple[Cue, dict[str, Any]]:
    """The Castellvi cue (likelihoods from the grade) and the grade record."""
    rec = castellvi_grade(transverse, cfg)
    g = rec["grade"]
    base = str(g).rstrip("ab") if g else None
    if g is None:
        cue = Cue(
            "castellvi",
            None,
            "no axial or coronal transverse-process record (not measured automatically)",
            SEQ_AXIAL,
        )
    elif g == "normal":
        cue = Cue(
            "castellvi",
            _like(none=cfg.lr_castellvi_normal),
            "transverse processes measured normal on both sides",
            SEQ_AXIAL,
        )
    else:
        lr = {
            "I": cfg.lr_castellvi_i,
            "II": cfg.lr_castellvi_ii,
            "III": cfg.lr_castellvi_iii,
            "IV": cfg.lr_castellvi_iv,
        }
        key = "IV" if g == "IV" else str(base)
        cue = Cue("castellvi", _like(sacral=lr[key]), f"Castellvi {g} from the supplied record", SEQ_AXIAL)
    return cue, rec


# ------------------------------------------------------------- combination


def combine_cues(cues: list[Cue], cfg: LstvConfig = DEFAULT_LSTV) -> dict[str, float]:
    """Posterior over the three kinds: prior x product of the measured cues' likelihoods (cues taken independent)."""
    prior_t = cfg.prior_transitional
    prior = {NONE: 1.0 - prior_t, SACRAL: prior_t / 2.0, LUMBAR: prior_t / 2.0}
    logp = {k: math.log(prior[k]) for k in KINDS}
    for c in cues:
        if c.like is None:
            continue
        for k in KINDS:
            logp[k] += math.log(max(float(c.like[k]), cfg.floor))
    m = max(logp.values())
    w = {k: math.exp(logp[k] - m) for k in KINDS}
    tot = sum(w.values())
    return {k: w[k] / tot for k in KINDS}


# ------------------------------------------------------------------ naming


def naming_record(anchors: Anchors, kind: str, status: str) -> dict[str, Any]:
    """Count names and the anatomical suggestion (not applied). The suggestion needs an unshifted numbering."""
    labels = anchors.disc_labels()  # superior -> inferior, as stored (count names)
    applied = int((anchors.numbering or {}).get("applied_offset", 0) or 0)
    suggested: list[dict[str, str]] | None = None
    if status in (STATUS_SUSPECTED, STATUS_POSSIBLE) and kind in (SACRAL, LUMBAR) and labels and applied == 0:
        bottom_up = list(reversed(labels))
        mapped = relabel_discs(bottom_up, OFFSET_OF[kind])
        suggested = [
            {"count": c, "suggested": s} for c, s in zip(reversed(bottom_up), reversed(mapped), strict=True)
        ]
    if applied != 0:
        note = (
            f"numbering already shifted by the fusion (offset {applied:+d}); the names are not changed again"
        )
    elif status == STATUS_NOT_ASSESSED or not labels:
        note = "no name suggestion: the transitional check is not assessed"
    elif suggested is None:
        note = "no transitional name change suggested"
    else:
        note = "suggestion only: verify before any name is used clinically or in a report"
    return {
        "convention": CONVENTION,
        "count_names": [{"count": lab} for lab in labels],
        "suggested_names": suggested,
        "applied": False,
        "note": note,
    }


CONVENTION = (
    "Count convention (SPIDER and this pipeline): disc names count up from the lowest disc, called L5-S1. "
    "Anatomical names follow the lumbar-type vertebra above each disc. Sacralised L5: the lowest disc is L4-L5 and "
    "all names move up one level. Lumbarised S1: the lowest disc is L5-L6 and only that disc is renamed. Never "
    "applied automatically."
)


# ------------------------------------------------------------------- entry points


@dataclass
class LstvResult:
    status: str
    kind: str | None
    p_transitional: float
    p_kind: float
    posterior: dict[str, float]
    cues: list[Cue]
    castellvi: dict[str, Any]
    naming: dict[str, Any]
    not_evaluated: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": LSTV_METHOD,
            "status": self.status,
            "kind": self.kind,
            "p_transitional": round(self.p_transitional, 4),
            "p_kind": round(self.p_kind, 4),
            "posterior": {k: round(v, 4) for k, v in self.posterior.items()},
            "castellvi": self.castellvi,
            "evidence": [c.to_dict() for c in self.cues],
            "naming": self.naming,
            "not_evaluated": list(self.not_evaluated),
            "warnings": list(self.warnings),
        }


NOT_EVALUATED: tuple[str, ...] = (
    "iliolumbar ligament origin on axial images (vertebral numbering; Hughes & Saifuddin 2006, needs axial T2 or T1)",
    "transverse-process size and sacral-ala contact (Castellvi) unless a record is supplied (needs "
    f"{SEQ_AXIAL})",
    "coronal localizer rib or vertebra count from a landmark (not implemented)",
    "iliac crest and aortic bifurcation level (not implemented)",
    "complete S1-S2 disc as its own cue (Carrino et al. 2011); only the lumbosacral check's removal is used",
)


def assess_lstv(
    anchors: Anchors,
    *,
    transverse: dict[str, Any] | None = None,
    cfg: LstvConfig = DEFAULT_LSTV,
) -> LstvResult:
    """Suspicion, kind, Castellvi grade and naming for a labelled ``anchors``. Pure: nothing in ``anchors`` changes.

    ``transverse`` is an optional reader or measurement record ``{"left": {...}, "right": {...}}`` (see
    :func:`side_grade`). Without it the Castellvi grade is ``not_assessed``.
    """
    if not anchors.discs:
        cue, rec = castellvi_cue(None, cfg)
        return LstvResult(
            STATUS_NOT_ASSESSED,
            None,
            0.0,
            0.0,
            {k: 1.0 if k == NONE else 0.0 for k in KINDS},
            [cue],
            rec,
            naming_record(anchors, NONE, STATUS_NOT_ASSESSED),
            list(NOT_EVALUATED),
            ["no discs: transitional check not assessed"],
        )
    cues: list[Cue] = [disc_cue(anchors, cfg)]
    cues += body_cues(anchors, cfg)
    cues += numbering_cues(anchors, cfg)
    cue_c, rec_c = castellvi_cue(transverse, cfg)
    cues.append(cue_c)
    post = combine_cues(cues, cfg)
    p_t = 1.0 - post[NONE]
    measured = [c for c in cues if c.like is not None]
    warnings: list[str] = []
    if not measured:
        status = STATUS_NOT_ASSESSED
        warnings.append("no transitional cue could be measured")
    elif p_t >= cfg.suspect_min:
        status = STATUS_SUSPECTED
    elif p_t >= cfg.possible_min:
        status = STATUS_POSSIBLE
    else:
        status = STATUS_NOT_SUSPECTED
    # An unconfirmed numbering (e.g. a disc chain that runs the wrong way, or no bone below the lowest disc) makes
    # the names this check reasons about unreliable: withhold the verdict and keep the cues for transparency.
    if (anchors.numbering or {}).get("status") == "unconfirmed" and status != STATUS_NOT_ASSESSED:
        status = STATUS_NOT_ASSESSED
        warnings.append(
            "numbering unconfirmed: transitional check withheld (the level names it reasons about are not settled); "
            "the cues are kept in the record"
        )
    kind: str | None = None
    p_kind = 0.0
    if status in (STATUS_SUSPECTED, STATUS_POSSIBLE):
        hi, lo = max(post[SACRAL], post[LUMBAR]), min(post[SACRAL], post[LUMBAR])
        if hi >= cfg.kind_ratio_min * lo:
            kind = SACRAL if post[SACRAL] > post[LUMBAR] else LUMBAR
            p_kind = post[kind] / p_t if p_t > 0 else 0.0
        else:
            kind = UNRESOLVED  # the two kinds are about equally likely: say so, do not pick one
            p_kind = 0.5
            warnings.append(
                "transitional kind unresolved: sacralised L5 and lumbarised S1 about equally likely"
            )
    if status == STATUS_SUSPECTED and kind is not None:
        warnings.append(
            f"possible lumbosacral transitional vertebra ({kind}, P={p_t:.2f}): level names are count-based; "
            "confirm on axial or coronal images before the names are used"
        )
    if rec_c["status"] == "not_assessed" and status != STATUS_NOT_ASSESSED:
        warnings.append("Castellvi grade not assessed: needs " + SEQ_AXIAL)
    if not any(c.name == "disc_height" and c.like is not None for c in cues):
        warnings.append("lowest-disc height not compared: fewer than three discs with heights")
    return LstvResult(
        status=status,
        kind=kind,
        p_transitional=p_t,
        p_kind=p_kind,
        posterior=post,
        cues=cues,
        castellvi=rec_c,
        naming=naming_record(anchors, kind or NONE, status),
        not_evaluated=list(NOT_EVALUATED),
        warnings=warnings,
    )


_SUSPECTED_MESSAGES: tuple[str, ...] = (
    "possible lumbosacral transitional vertebra",
    "transitional kind unresolved",
)


def attach_lstv(
    anchors: Anchors,
    *,
    transverse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Set ``anchors.lstv`` (additive) and add one warning when a transitional vertebra is suspected."""
    res = assess_lstv(anchors, transverse=transverse)
    rec = res.to_dict()
    anchors.lstv = rec
    if res.status == STATUS_SUSPECTED:
        # Pick the messages by their text: the first warning can be the kind note ("kind unresolved"), not the suspicion.
        for msg in res.warnings:
            if msg.startswith(_SUSPECTED_MESSAGES) and msg not in anchors.warnings:
                anchors.warnings.append(msg)
    return rec


def lstv_line(anchors: Anchors) -> str:
    """One line for the summary: status, kind, probability and what is missing."""
    rec = anchors.lstv
    if not isinstance(rec, dict):
        return "transitional check: not run"
    status = str(rec.get("status"))
    if status == STATUS_NOT_ASSESSED:
        return "transitional check: not assessed"
    kind = rec.get("kind")
    p = float(rec.get("p_transitional", 0.0))
    cas = (rec.get("castellvi") or {}).get("grade")
    cas_txt = f"Castellvi {cas}" if cas else "Castellvi not assessed (no axial or coronal record)"
    if status == STATUS_NOT_SUSPECTED:
        return f"transitional check: not suspected (P {p:.2f}); {cas_txt}"
    return f"transitional check: {status} {kind} (P {p:.2f}); {cas_txt}; names are count-based, not changed"


__all__ = [
    "CONVENTION",
    "DEFAULT_LSTV",
    "KINDS",
    "LSTV_METHOD",
    "LUMBAR",
    "NONE",
    "SACRAL",
    "UNRESOLVED",
    "Cue",
    "LstvConfig",
    "LstvResult",
    "assess_lstv",
    "attach_lstv",
    "body_cues",
    "castellvi_cue",
    "castellvi_grade",
    "combine_cues",
    "disc_cue",
    "lstv_line",
    "naming_record",
    "numbering_cues",
    "side_grade",
]
