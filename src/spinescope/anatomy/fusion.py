"""Fusion of labeller outputs into the final numbering: one offset, one transitional flag, every vote recorded.

Inputs for one study (all optional except the primary anchors):

* the primary labeller's anchors (the learned labeller by default);
* the :mod:`numbering_checks` cues computed on the primary's discs: oblique slabs (geometry), sacral tilt and
  conus tip (the lowest disc is usually the most tilted one; the conus lies in the lower third of L1);
* the level names written in oblique-slab descriptions (``L4-5 OBL AX``), when a slab passes through the
  lowest disc: a named level is a direct statement by the technologist;
* the classical labeller as an independent second opinion. It is a weak vote: its tilt cue, with a tempered
  likelihood (exponent ``classical_weight``), and only when its discs agree geometrically with the primary's;
* the axial slabs (:mod:`axial`), when ``FusionConfig.use_axial`` is set: the lowest T1 slab's disc, if it shows
  disc-like appearance, replaces the unchecked slab cue; disc-support flags and missed-disc candidates are warnings.
* the second sagittal series of the same study (T1 against T2). Two sagittal series show the same anatomy, so
  they can only disagree about coverage or a missed disc. The check is therefore a consistency check, not an
  offset vote: it is recorded (neutral vote), feeds the calibrated confidence, and adds warnings.

The posterior over the lowest-disc offset (-1, 0, +1; see :mod:`numbering_checks`) is the product of the
floored cue likelihoods and the prior. A non-zero offset is applied (names shifted, ``transitional`` set) only
when its posterior is at least ``FusionConfig.relabel_min_p``. Below that the provisional names stay and a
non-zero best offset is recorded as a transitional candidate with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import DEFAULT
from ..geometry import Stack
from ..study import Study
from .anchors import Anchors
from .axial import AxialReport
from .numbering import VERTEBRA_NAMES, level_names_in, shift_vertebra_name
from .numbering_checks import (
    DEFAULT_NUMBERING_CONFIG,
    OFFSETS,
    NumberingConfig,
    Vote,
    axial_slab_vote,
    combine,
    numbering_evidence,
    relabel_discs,
    tilt_vote,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionConfig:
    """Fusion settings. Judgement defaults, documented in docs/labeling/lab08_fusion.md."""

    #: Apply a non-zero lowest-disc offset only when its posterior reaches this (judgement: about a 1-in-10
    #: chance of a wrong shift is accepted; below it the provisional names stay and a warning is raised).
    relabel_min_p: float = 0.9
    #: Likelihood exponent for the classical tilt vote. 0.5 halves its log-evidence: a weak vote.
    classical_weight: float = 0.5
    #: In-plane (y, z) distance for matching discs between two labeller outputs (Config.disc_match_mm).
    agree_mm: float = DEFAULT.disc_match_mm
    #: Two disc sets agree geometrically when this fraction of discs match and their counts differ by <= 1.
    agree_fraction: float = 0.7
    #: Likelihood of a slab-description name when it is the true level of the lowest disc (judgement: a
    #: protocol name is usually the intended level; the residual covers mis-typed or protocol-default names).
    name_p_match: float = 0.95
    #: Likelihood of that name for each other offset. With the 0.05 floor of numbering_checks.combine one named
    #: slab is worth a likelihood ratio of about 19 and gives P(shift) about 0.6; two independent slabs that
    #: name the same level give about 0.9 (applied).
    name_p_other: float = 0.02
    #: Use the axial T1 slabs (:mod:`axial`, additive, default OFF). Off keeps the numbers and the calibrated
    #: confidence exactly as shipped: the calibration was fitted without this vote. Turning it on changes the
    #: volume features, so it needs a new calibration fit before it is shipped (see docs/labeling/lab20_axial.md).
    use_axial: bool = False


DEFAULT_FUSION = FusionConfig()


@dataclass
class FusionResult:
    """Everything the fusion decided, plus the features the calibration needs (see calibration.py)."""

    offset: int  # the best offset (applied or not)
    applied: bool  # True when the names were shifted
    posterior: dict[int, float]
    votes: list[dict[str, Any]]  # every vote as a plain dict (method, best, probs, weight, evidence)
    transitional: bool
    transitional_kind: str | None
    transitional_candidate: bool
    p_applied: float  # posterior of the offset that the final names carry
    sacrum_below: bool
    second: dict[str, Any] | None  # second-series agreement record (None when there is no second series)
    classical: dict[str, Any] | None  # classical agreement record (None when there is no classical opinion)
    t1t2_by_label: dict[str, bool]  # per primary disc (pre-shift label): same label in the second series
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    axial: dict[str, Any] | None = None  # axial report summary (None when the axial cue was not used)

    @property
    def t1t2_fraction(self) -> float:
        """Share of the primary's discs confirmed (same label, same place) by the second series."""
        n = len(self.t1t2_by_label)
        return float(sum(self.t1t2_by_label.values())) / n if n else 0.0


# ----------------------------------------------------------------- geometry


def _yz(p: Any) -> np.ndarray:
    """Sagittal-plane (y, z) of an LPS point."""
    return np.asarray(p, dtype=np.float64)[1:]


def match_discs(a: Anchors, b: Anchors, mm: float) -> dict[str, str | None]:
    """For each disc of ``a``: the label of the nearest disc of ``b`` in the sagittal plane within ``mm``, or None."""
    out: dict[str, str | None] = {}
    for la, da in a.discs.items():
        best: tuple[float, str] | None = None
        for lb, db in b.discs.items():
            dist = float(np.linalg.norm(_yz(da.center) - _yz(db.center)))
            if best is None or dist < best[0]:
                best = (dist, lb)
        out[la] = best[1] if best is not None and best[0] <= mm else None
    return out


def geometric_agreement(a: Anchors, b: Anchors, cfg: FusionConfig = DEFAULT_FUSION) -> tuple[bool, float]:
    """(agrees, matched fraction). Agreement: most discs match in place and the counts differ by at most one."""
    if not a.discs or not b.discs:
        return False, 0.0
    matched = sum(1 for v in match_discs(a, b, cfg.agree_mm).values() if v is not None)
    frac = matched / max(len(a.discs), len(b.discs))
    ok = frac >= cfg.agree_fraction and abs(len(a.discs) - len(b.discs)) <= 1
    return ok, frac


def stack_z_range(stack: Stack) -> tuple[float, float]:
    """Lowest and highest LPS z covered by a sagittal stack (its four corner pixels, first and last slice)."""
    zs: list[float] = []
    for k in (0, stack.vol.shape[0] - 1):
        rows, cols = stack.vol.shape[1], stack.vol.shape[2]
        for r in (0, rows - 1):
            for c in (0, cols - 1):
                p = (
                    np.asarray(stack.ipp[k], dtype=np.float64)
                    + c * stack.col_mm * np.asarray(stack.row_dir, dtype=np.float64)
                    + r * stack.row_mm * np.asarray(stack.col_dir, dtype=np.float64)
                )
                zs.append(float(p[2]))
    return min(zs), max(zs)


# ----------------------------------------------------------------- cues


def description_votes(slabs: list[Stack], primary: Anchors, cfg: FusionConfig = DEFAULT_FUSION) -> list[Vote]:
    """One vote per slab whose description names a level for the lowest disc.

    A slab named ``L4-5`` that passes through the lowest provisional disc says that disc is L4-L5 (offset +1);
    ``L5-S1`` says offset 0. Only the lowest disc is used: a name for a higher disc would move only part of the
    chain, which the offset model does not represent. Each slab is a separate vote (each is floored separately
    in :func:`numbering_checks.combine`), so independent slabs strengthen each other.
    """
    if not slabs or not primary.discs:
        return []
    labels = primary.disc_labels()  # superior -> inferior
    lowest = labels[-1]
    d = primary.discs[lowest]
    out: list[Vote] = []
    for sl in slabs:
        names = level_names_in(sl.description or "")
        if not names:
            continue
        angle = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(sl.normal, d.normal))), -1.0, 1.0))))
        margin = (d.height_mm or 8.0) / 2.0
        if angle > 12.0 or not sl.contains(np.asarray(d.center), slab_margin_mm=margin):
            continue
        for name in names:
            lower = name.split("-")[1]
            if lower not in VERTEBRA_NAMES:
                continue
            t = VERTEBRA_NAMES.index(lower)  # true index of the lowest disc's lower body (S1 = 0, L5 = 1)
            if t not in (0, 1):
                continue
            like = {
                o: (cfg.name_p_match if o == t else cfg.name_p_other) for o in OFFSETS
            }  # provisional index 0
            out.append(
                Vote(
                    cue="slab_description",
                    likelihood=like,
                    evidence=[f"slab {sl.key} described as {name} passes through the lowest disc {lowest}"],
                )
            )
            break  # one name per slab
    return out


def _tempered(v: Vote, weight: float, cue: str) -> Vote:
    return Vote(
        cue=cue,
        likelihood={o: float(v.likelihood[o]) ** weight for o in OFFSETS},
        evidence=[f"weak vote (likelihood ** {weight:g}): {e}" for e in v.evidence],
    )


def _vote_record(v: Vote, weight: float = 1.0) -> dict[str, Any]:
    return {
        "method": v.cue,
        "best": int(v.best),
        "probs": {str(o): round(float(p), 4) for o, p in v.probs.items()},
        "weight": weight,
        "evidence": list(v.evidence),
    }


def _neutral_record(method: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "method": method,
        "best": 0,
        "probs": {str(o): round(1.0 / len(OFFSETS), 4) for o in OFFSETS},
        "weight": 0.0,
        "evidence": evidence,
    }


# ----------------------------------------------------------------- fusion


def fuse_numbering(
    primary: Anchors,
    *,
    study: Study | None,
    second: Anchors | None = None,
    second_key: str = "",
    classical: Anchors | None = None,
    coverage_z: tuple[float, float] | None = None,
    sacrum_below: bool = False,
    cfg: FusionConfig = DEFAULT_FUSION,
    ncfg: NumberingConfig = DEFAULT_NUMBERING_CONFIG,
    slabs: list[Stack] | None = None,
    axial: AxialReport | None = None,
) -> FusionResult:
    """Combine every numbering cue for ``primary`` into one offset decision (see the module docstring).

    ``slabs`` overrides the study's oblique axial slabs (used by the tests; default: ``study.slabs('ax')``).

    ``axial`` is the :func:`axial.assess_axial` report of ``primary``; used only when ``cfg.use_axial``.

    ``coverage_z`` is the (low, high) LPS z of the primary's sagittal stack: a disc of the second series below
    the primary's lowest disc is a "possible missed disc" only when it lies inside that coverage.
    ``sacrum_below``: the primary shows vertebral bone below its lowest disc (a sacrum confirmation feature).
    """
    warnings: list[str] = []
    notes: list[str] = []
    votes_raw: list[Vote] = []
    records: list[dict[str, Any]] = []
    if not primary.discs:
        post = dict(zip(OFFSETS, ncfg.prior_offset, strict=True))
        return FusionResult(
            offset=0,
            applied=False,
            posterior=post,
            votes=[],
            transitional=False,
            transitional_kind=None,
            transitional_candidate=False,
            p_applied=post[0],
            sacrum_below=sacrum_below,
            second=None,
            classical=None,
            t1t2_by_label={},
            warnings=["no discs: numbering not fused"],
        )

    use_slabs = list(slabs) if slabs is not None else (study.slabs("ax") if study is not None else [])
    ev = numbering_evidence(study, primary, ncfg, slabs=use_slabs)
    axial_on = bool(cfg.use_axial and axial is not None and axial.available)
    axial_vote = axial_slab_vote(axial, primary, ncfg) if axial_on else None
    axial_rec: dict[str, Any] | None = None
    if axial_on and axial is not None:
        axial_rec = axial.to_dict()
        warnings.extend(axial.warnings)
        if axial_vote is None:
            # The lowest slab is not disc-like or covers no disc: the unchecked slab cue is not used either.
            notes.append("axial lowest slab gives no disc-like support: slab cue not used")
        notes.append(f"axial cue: {len(axial.covered)} predicted disc(s) covered by a T1 slab")
    for v in ev.votes:
        if axial_on and v.cue == "slab":
            continue  # replaced by the disc-gated axial vote (or dropped when that abstains)
        votes_raw.append(v)
        records.append(_vote_record(v))
    if axial_vote is not None:
        votes_raw.append(axial_vote)
        records.append(_vote_record(axial_vote))
    warnings.extend(w for w in ev.warnings if "uncertain" in w)
    for dv in description_votes(use_slabs, primary, cfg):
        votes_raw.append(dv)
        records.append(_vote_record(dv))

    # Classical second opinion: a weak tilt vote, used only when its discs agree with the primary's.
    classical_rec: dict[str, Any] | None = None
    if classical is not None:
        agrees, frac = geometric_agreement(primary, classical, cfg)
        classical_rec = {
            "agrees": agrees,
            "matched_fraction": round(frac, 3),
            "n_discs": len(classical.discs),
        }
        if agrees:
            tv = tilt_vote(classical, ncfg)
            if tv is not None:
                weak = _tempered(tv, cfg.classical_weight, "classical_tilt")
                votes_raw.append(weak)
                records.append(_vote_record(weak, cfg.classical_weight))
                notes.append("classical labeller agrees geometrically: its tilt cue used as a weak vote")
        else:
            notes.append(
                f"classical labeller disagrees geometrically ({frac:.0%} of discs matched): not used"
            )

    # Second sagittal series: consistency of labels (count from the bottom) and coverage of missed discs.
    second_rec: dict[str, Any] | None = None
    t1t2: dict[str, bool] = {}
    if second is not None and second.discs:
        same = match_discs(primary, second, cfg.agree_mm)
        t1t2 = {lab: same[lab] == lab for lab in primary.discs}
        agree = sum(1 for v in t1t2.values() if v)
        matched = sum(1 for v in same.values() if v is not None)
        # Discs of the second series that the primary does not have, below its lowest disc.
        primary_lowest_z = min(float(d.center[2]) for d in primary.discs.values())
        s_matched = {v for v in same.values() if v is not None}
        extra = [
            lab
            for lab, d in second.discs.items()
            if lab not in s_matched
            and float(d.center[2]) < primary_lowest_z
            and (coverage_z is None or float(d.center[2]) > coverage_z[0] + (d.height_mm or 8.0) / 2.0)
        ]
        second_rec = {
            "key": second_key,
            "agree": agree,
            "matched": matched,
            "primary_discs": len(primary.discs),
            "second_discs": len(second.discs),
            "extra_below": extra,
        }
        if agree < len(primary.discs):
            warnings.append(
                f"second sagittal series {second_key} labels {len(primary.discs) - agree} of "
                f"{len(primary.discs)} disc(s) differently"
            )
        if extra:
            warnings.append(
                f"second sagittal series {second_key} shows {len(extra)} disc(s) below the lowest disc of this "
                "series: a disc may be missed here, so the lowest name is less certain"
            )
        records.append(
            _neutral_record(
                "second_series",
                [
                    f"{second_key}: {agree}/{len(primary.discs)} primary discs have the same label in place",
                    "consistency check, not an offset vote (both series show the same anatomy)",
                ],
            )
        )

    post = combine(votes_raw, ncfg)
    best = max(OFFSETS, key=lambda o: (round(post[o], 12), o == 0))
    p_best = post[best]
    applied = best != 0 and p_best >= cfg.relabel_min_p
    kind = {1: "sacralized", -1: "lumbarized"}.get(best) if best != 0 else None
    candidate = best != 0 and p_best >= 0.5 and not applied
    if applied:
        notes.append(
            f"offset {best:+d} applied (P={p_best:.2f}): names shifted; "
            f"{'sacralized L5 (lowest disc is L4-L5)' if best == 1 else 'lumbarized S1 (lowest disc is L5-L6)'}"
        )
        warnings.append(f"lowest-disc names shifted by offset {best:+d} (P={p_best:.2f}): check by eye")
    elif candidate:
        notes.append(
            f"offset {best:+d} is a transitional candidate (P={p_best:.2f}) below the apply threshold"
        )
        warnings.append(
            f"possible transitional vertebra (offset {best:+d}, P={p_best:.2f}): names not shifted"
        )
    if best == 0:
        notes.append(f"lowest disc named as provisional (P(offset 0)={p_best:.2f})")
    p_applied = p_best if applied or best == 0 else post[0]
    return FusionResult(
        offset=int(best),
        applied=bool(applied),
        posterior={int(o): float(post[o]) for o in OFFSETS},
        votes=records,
        transitional=bool(applied),
        transitional_kind=kind if applied else None,
        transitional_candidate=bool(candidate),
        p_applied=float(p_applied),
        sacrum_below=bool(sacrum_below),
        second=second_rec,
        classical=classical_rec,
        t1t2_by_label=t1t2,
        warnings=warnings,
        notes=notes,
        axial=axial_rec,
    )


# ----------------------------------------------------------------- applying


def relabel_anchors(anchors: Anchors, offset: int) -> dict[str, str]:
    """Shift disc, vertebra and conus names in place for ``offset``. Returns the disc name map (old -> new)."""
    if offset == 0 or not anchors.discs:
        return {lab: lab for lab in anchors.discs}
    order = anchors.disc_labels()  # superior -> inferior
    bottom_up = list(reversed(order))
    new_bottom_up = relabel_discs(bottom_up, offset)
    mapping = dict(zip(bottom_up, new_bottom_up, strict=True))
    vert_map = {name: shift_vertebra_name(name, offset) for name in anchors.vertebrae}
    anchors.discs = {mapping[lab]: anchors.discs[lab] for lab in order}
    anchors.vertebrae = {vert_map[name]: v for name, v in anchors.vertebrae.items()}
    # Geometry measured before the numbering (body levels, canal and posterior-element keys) follows the names.
    anchors._sync_geometry(mapping, vert_map)
    if anchors.conus and anchors.conus.get("level"):
        lvl = str(anchors.conus["level"])
        anchors.conus["level"] = mapping.get(lvl, shift_vertebra_name(lvl, offset))
    return mapping


def fusion_record(res: FusionResult) -> dict[str, Any]:
    """The serialisable part of a :class:`FusionResult` for ``anchors.numbering``."""
    return {
        "offset": res.offset,
        "applied_offset": res.offset if res.applied else 0,
        "posterior": {str(o): round(p, 4) for o, p in res.posterior.items()},
        "votes": res.votes,
        "transitional": res.transitional,
        "transitional_kind": res.transitional_kind,
        "transitional_candidate": res.transitional_candidate,
        "sacrum_below": res.sacrum_below,
        "second_series": res.second,
        "classical": res.classical,
        "relabelled": res.applied,
        **({"axial": res.axial} if res.axial is not None else {}),
    }


__all__ = [
    "DEFAULT_FUSION",
    "FusionConfig",
    "FusionResult",
    "description_votes",
    "fuse_numbering",
    "fusion_record",
    "geometric_agreement",
    "match_discs",
    "relabel_anchors",
    "stack_z_range",
]
