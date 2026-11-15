"""Independent evidence for vertebral numbering: which disc is L5-S1, and is there a transitional vertebra.

:func:`spinescope.anatomy.numbering.disc_names` counts discs up from the lowest one and calls it L5-S1.
That is wrong when the lowest visible disc is really L4-L5 (sacralized L5) or L5-L6 (lumbarized S1).
This module collects cues that do not depend on that count and combines them into a posterior over
the level shift of the lowest disc:

* ``offset`` (-1, 0, +1) is the true level of the lowest visible disc minus its provisional level, in
  lumbar steps: 0 = L5-S1, +1 = L4-L5 (sacralized L5), -1 = L5-L6 (lumbarized S1).
* ``transitional`` is True when the offset is not 0; ``transitional_kind`` is ``"sacralized"`` for +1
  and ``"lumbarized"`` for -1.

Shifting names. For +1 every provisional disc name moves one level up (the whole count was shifted).
For -1 only the lowest disc is renamed (the bodies above it keep their names). See
:func:`relabel_discs`.

Cues (each returns a :class:`Vote` with a likelihood over the three offsets):

* :func:`slab_vote`: the lowest oblique axial slab (technologists aim it at L5-S1, or at L4-L5 in some
  protocols) and the disc it passes through.
* :func:`tilt_vote`: the sacral slope; the lowest disc is usually the most tilted one.
* :func:`conus_vote`: the conus tip, usually in the lower third of L1 (weak prior).

Cues that need image analysis that is not implemented here (axial transverse-process/ala morphology,
iliac crest, aortic bifurcation, localizer rib counting, S1-S2 remnant disc) are reported in
``NumberingEvidence.warnings`` as not evaluated, with no vote. Weights and priors are documented in
:class:`NumberingConfig`; see ``docs/labeling/lab04_numbering.md`` for sources and calibration.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..geometry import Stack
from ..study import Study
from .anchors import Anchors
from .numbering import VERTEBRA_NAMES

if TYPE_CHECKING:  # the axial module imports nothing from here at module level; avoid an import cycle
    from .axial import AxialReport

log = logging.getLogger(__name__)

OFFSETS: tuple[int, ...] = (-1, 0, 1)


@dataclass(frozen=True)
class NumberingConfig:
    """Priors and cue parameters. Every value has a source or is marked as a judgement default."""

    # Prior over the lowest-disc offset. Transitional vertebrae occur in roughly 4-30% of people
    # (Konin & Walz, AJNR 2010, doi 10.3174/ajnr.A2036; the range is from the prior literature it
    # reviews). A 15% total is taken as a rough middle, split evenly between sacralized and lumbarized.
    prior_offset: tuple[float, float, float] = (0.075, 0.85, 0.075)  # offsets -1, 0, +1
    # Lowest oblique slab: the probability that it is aimed at L5-S1 (vs L4-L5) when it is the lowest
    # slab. Judgement default from the ACR MR adult spine protocol wording (L5-S1 disc-parallel) and the
    # usual two-protocol pattern (L1-L2..L5-S1 or L3-L4..L5-S1). Not measured on data.
    slab_target_p_l5s1: float = 0.7
    slab_target_p_l4l5: float = 0.2
    # A slab matches a disc when its normal is within this angle of the disc normal (deg) and the disc
    # centre lies within the slab block (plus half a disc height). Same tolerance as numbering.slab_crosscheck.
    slab_match_angle_deg: float = 12.0
    # Tilt cue: P(lowest disc is the most tilted by tilt_margin_deg | offset). Given offset 0 the SPIDER
    # oracle (masks, lowest disc assumed L5-S1) gives 0.971 (167/172 training cases; validation 0.921),
    # see docs/labeling/lab04_numbering.md. Set to 0.8 because the disc normals come from the classical
    # chain, which is noisier than the masks (judgement). Given a shifted offset there are no labelled
    # cases, so 0.5 (uninformative) is used.
    tilt_p_most_tilted_if_normal: float = 0.8
    tilt_p_most_tilted_if_shifted: float = 0.5
    tilt_margin_deg: float = 3.0  # lowest disc must exceed the disc above by this many degrees
    # Conus cue: conus tip position in the lowest-disc frame. Saifuddin et al., Spine 1998
    # (doi 10.1097/00007632-199807010-00005): mean in the lower third of L1, observed T12 mid .. L3 upper.
    # Mean (in vertebra index units, S1=0, L5=1, L1=5) and SD (judgement, about one level).
    conus_mean_index: float = 4.7
    conus_sd_levels: float = 1.0
    conus_mix_uniform: float = 0.75  # share of a uniform likelihood mixed in: a weak cue (judgement)
    # Every likelihood bin is floored at this value before combining, so one cue cannot veto.
    floor: float = 0.05
    # Report a warning when the posterior of the best offset is below this.
    uncertain_below: float = 0.6


DEFAULT_NUMBERING_CONFIG = NumberingConfig()


@dataclass
class Vote:
    """One cue's likelihood over the lowest-disc offsets, with its evidence."""

    cue: str
    likelihood: dict[int, float]  # offset -> relative likelihood (not normalised)
    evidence: list[str] = field(default_factory=list)

    @property
    def probs(self) -> dict[int, float]:
        total = sum(self.likelihood.values())
        return {o: self.likelihood[o] / total for o in OFFSETS}

    @property
    def best(self) -> int:
        # Ties (e.g. an uninformative cue) resolve to offset 0, the prior's choice.
        return max(OFFSETS, key=lambda o: (round(self.likelihood[o], 12), o == 0))

    @property
    def confidence(self) -> float:
        return self.probs[self.best]


@dataclass
class NumberingEvidence:
    """Combined numbering evidence. ``offset`` is the best lowest-disc offset; see module docstring."""

    offset: int
    offset_confidence: float
    posterior: dict[int, float]
    transitional: bool
    transitional_kind: str | None
    transitional_confidence: float
    votes: list[Vote]
    warnings: list[str]
    not_evaluated: list[str]

    @property
    def cues_used(self) -> list[str]:
        return [v.cue for v in self.votes]

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "offset_confidence": round(self.offset_confidence, 4),
            "posterior": {str(o): round(p, 4) for o, p in self.posterior.items()},
            "transitional": self.transitional,
            "transitional_kind": self.transitional_kind,
            "transitional_confidence": round(self.transitional_confidence, 4),
            "votes": [
                {
                    "cue": v.cue,
                    "best": v.best,
                    "probs": {str(o): round(p, 4) for o, p in v.probs.items()},
                    "evidence": list(v.evidence),
                }
                for v in self.votes
            ],
            "warnings": list(self.warnings),
            "not_evaluated": list(self.not_evaluated),
        }


# ------------------------------------------------------------------ helpers


def _index_of(name: str) -> float | None:
    """Vertebra index (S1 = 0, L5 = 1, ..., L1 = 5) of a level name or disc name ("L4-L5" -> 2.5)."""
    parts = name.upper().split("-")
    idx: list[float] = []
    for p in parts:
        if p in VERTEBRA_NAMES:
            idx.append(float(VERTEBRA_NAMES.index(p)))
    if not idx:
        return None
    return float(np.mean(idx))


def _name_of(index: int) -> str:
    """Vertebra name for an integer index (S1 = 0 upward; negative indices are lumbarized bodies)."""
    if index < 0:
        return f"L{5 - index}"  # -1 -> L6
    return VERTEBRA_NAMES[index] if index < len(VERTEBRA_NAMES) else f"V{index}"


def relabel_discs(provisional_bottom_up: Sequence[str], offset: int) -> list[str]:
    """Disc names after applying a lowest-disc offset.

    ``offset`` +1 (sacralized) moves every disc one level up; -1 (lumbarized) renames only the lowest
    disc, to one level below (L5-S1 -> L5-L6). 0 returns the names unchanged.
    """
    names = list(provisional_bottom_up)
    if offset == 0 or not names:
        return names
    if offset == 1:
        out = []
        for lab in names:
            upper, lower = lab.split("-")
            ui, li = _index_of(upper), _index_of(lower)
            if ui is None or li is None:
                out.append(lab)
                continue
            out.append(f"{_name_of(int(ui) + 1)}-{_name_of(int(li) + 1)}")
        return out
    if offset == -1:
        upper, _lower = names[0].split("-")
        names[0] = f"{upper}-{_name_of(-1)}"  # L5-S1 -> L5-L6
        return names
    raise ValueError(f"offset must be -1, 0 or +1, got {offset}")


def _normalise(like: dict[int, float], cfg: NumberingConfig) -> dict[int, float]:
    floored = {o: max(float(like[o]), cfg.floor) for o in OFFSETS}
    total = sum(floored.values())
    return {o: floored[o] / total for o in OFFSETS}


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0))))


# -------------------------------------------------------------------- cues


def slab_vote(
    slabs: Sequence[Stack], anchors: Anchors, cfg: NumberingConfig = DEFAULT_NUMBERING_CONFIG
) -> Vote | None:
    """Vote from the lowest oblique slab and the disc it is angled to.

    The slab is matched to a disc when its slice normal is within ``slab_match_angle_deg`` of the disc
    normal and the disc centre lies inside the slab block. If it matches the disc with provisional
    index ``k`` (0 = lowest disc) and that disc is truly L5-S1 (probability ``slab_target_p_l5s1``) then
    the lowest disc has offset -k; if it is truly L4-L5 (``slab_target_p_l4l5``) then the offset is 1-k.
    """
    if not slabs or not anchors.discs:
        return None
    slab = min(slabs, key=lambda s: float(s.ipp[:, 2].mean()))  # most inferior slab
    labels = anchors.disc_labels()  # superior -> inferior
    ordered = list(reversed(labels))  # bottom-up: index 0 = lowest
    best: tuple[int, str] | None = None
    for k, lab in enumerate(ordered):
        d = anchors.discs[lab]
        if _angle_deg(np.asarray(d.normal), slab.normal) > cfg.slab_match_angle_deg:
            continue
        margin = (d.height_mm or 8.0) / 2.0
        if not slab.contains(np.asarray(d.center), slab_margin_mm=margin):
            continue
        best = (k, lab)
        break
    if best is None:
        return Vote(
            cue="slab",
            likelihood={o: 1.0 for o in OFFSETS},
            evidence=[f"lowest oblique slab ({slab.key}) passes through no labelled disc"],
        )
    k, lab = best
    if k >= 2:
        # Matched two or more discs above the lowest: that needs a shift of two levels, which is outside
        # the offsets modelled here. Treat as uninformative and say so.
        return Vote(
            cue="slab",
            likelihood={o: 1.0 for o in OFFSETS},
            evidence=[
                f"lowest oblique slab ({slab.key}) is angled to disc {lab}, {k} levels above the lowest: "
                "disc chain or slab labels doubtful, not used"
            ],
        )
    like = {o: 0.0 for o in OFFSETS}
    # target L5-S1 (true index 0) or L4-L5 (true index 1) for the matched disc; lowest disc offset = t - k
    for t, p in ((0, cfg.slab_target_p_l5s1), (1, cfg.slab_target_p_l4l5)):
        o = t - k
        if o in like:
            like[o] += p
    ev = [f"lowest oblique slab ({slab.key}) is angled to disc {lab} (provisional lowest-disc index {k})"]
    return Vote(cue="slab", likelihood=like, evidence=ev)


#: Cue name of the gated axial slab vote (stable: fusion records and calibration refer to it).
AXIAL_SLAB_CUE = "axial_slab"


def axial_slab_vote(
    report: AxialReport | None, anchors: Anchors, cfg: NumberingConfig = DEFAULT_NUMBERING_CONFIG
) -> Vote | None:
    """Slab vote from the lowest axial T1 slab, using only a disc that shows disc-like appearance there.

    Same likelihood model as :func:`slab_vote`, with two differences: the slab must be a T1 slab (the calibrated
    sequence) and the disc it is angled to must pass the axial disc-likeness check (``SlabMatch.supported``). A
    slab whose disc is not disc-like, or one that covers no predicted disc, gives no vote (None). The lowest slab
    only: higher slabs carry a different protocol target. See :mod:`axial`.
    """
    if report is None or not report.available or not report.slabs:
        return None
    low = report.slabs[0]  # rank 0: the most inferior T1 slab
    if low.disc is None or not low.supported or low.disc not in anchors.discs:
        return None
    ordered = list(reversed(anchors.disc_labels()))  # bottom-up: index 0 = lowest provisional disc
    k = ordered.index(low.disc)
    if k >= 2:
        return Vote(
            cue=AXIAL_SLAB_CUE,
            likelihood={o: 1.0 for o in OFFSETS},
            evidence=[
                f"lowest axial slab ({low.slab}) is angled to disc {low.disc}, {k} levels above the lowest: "
                "not used"
            ],
        )
    like = {o: 0.0 for o in OFFSETS}
    for t, p in ((0, cfg.slab_target_p_l5s1), (1, cfg.slab_target_p_l4l5)):
        o = t - k
        if o in like:
            like[o] += p
    ev = [
        f"lowest axial T1 slab ({low.slab}) is angled to disc {low.disc} with disc-like appearance "
        f"(LLR {low.llr:.2f}; provisional lowest-disc index {k})"
    ]
    return Vote(cue=AXIAL_SLAB_CUE, likelihood=like, evidence=ev)


def tilt_vote(anchors: Anchors, cfg: NumberingConfig = DEFAULT_NUMBERING_CONFIG) -> Vote | None:
    """Vote from the sacral slope: is the lowest disc the most tilted of the chain.

    Tilt is the angle of the disc normal from the superior axis in the sagittal plane.
    """
    labels = anchors.disc_labels()
    if len(labels) < 2:
        return None
    tilts = {}
    for lab in labels:
        n = np.asarray(anchors.discs[lab].normal, dtype=np.float64)
        tilts[lab] = math.degrees(math.atan2(abs(n[1]), abs(n[2])))
    lowest, above = labels[-1], labels[-2]
    margin = tilts[lowest] - max(tilts[lab] for lab in labels[:-1])
    most_tilted = margin >= cfg.tilt_margin_deg
    p0 = cfg.tilt_p_most_tilted_if_normal if most_tilted else 1 - cfg.tilt_p_most_tilted_if_normal
    ps = cfg.tilt_p_most_tilted_if_shifted if most_tilted else 1 - cfg.tilt_p_most_tilted_if_shifted
    like = {0: p0, 1: ps, -1: ps}
    ev = [
        f"lowest disc {lowest} tilt {tilts[lowest]:.0f} deg vs {above} {tilts[above]:.0f} deg "
        f"({'most tilted' if most_tilted else 'not the most tilted'})"
    ]
    return Vote(cue="tilt", likelihood=like, evidence=ev)


def conus_vote(anchors: Anchors, cfg: NumberingConfig = DEFAULT_NUMBERING_CONFIG) -> Vote | None:
    """Weak prior from the conus tip level (Saifuddin et al. 1998: mean lower third of L1)."""
    if not anchors.conus or not anchors.conus.get("level"):
        return None
    x = _index_of(str(anchors.conus["level"]))
    if x is None:
        return None
    # Under +1 every provisional name moves up one level; under -1 and 0 the bodies keep their names.
    shifts = {1: 1.0, 0: 0.0, -1: 0.0}
    dens: dict[int, float] = {}
    for o in OFFSETS:
        t = x + shifts[o]
        z = (t - cfg.conus_mean_index) / cfg.conus_sd_levels
        dens[o] = math.exp(-0.5 * z * z)
    tot = sum(dens.values())
    like = {o: (1 - cfg.conus_mix_uniform) * dens[o] / tot + cfg.conus_mix_uniform / 3 for o in OFFSETS}
    ev = [f"conus tip provisionally at {anchors.conus['level']} (prior: lower third of L1)"]
    return Vote(cue="conus", likelihood=like, evidence=ev)


# ---------------------------------------------------------------- combine


def combine(
    votes: Sequence[Vote | None], cfg: NumberingConfig = DEFAULT_NUMBERING_CONFIG
) -> dict[int, float]:
    """Posterior over offsets: prior x product of each cue's floored likelihood (cues treated as independent)."""
    prior = dict(zip(OFFSETS, cfg.prior_offset, strict=True))
    logp = {o: math.log(prior[o]) for o in OFFSETS}
    for v in votes:
        if v is None:
            continue
        norm = _normalise(v.likelihood, cfg)
        for o in OFFSETS:
            logp[o] += math.log(norm[o])
    m = max(logp.values())
    w = {o: math.exp(logp[o] - m) for o in OFFSETS}
    tot = sum(w.values())
    return {o: w[o] / tot for o in OFFSETS}


def numbering_evidence(
    study: Study | None,
    anchors: Anchors,
    config: NumberingConfig = DEFAULT_NUMBERING_CONFIG,
    *,
    slabs: Sequence[Stack] | None = None,
) -> NumberingEvidence:
    """Combine every available numbering cue for ``anchors`` into a :class:`NumberingEvidence`.

    Uses ``study.slabs('ax')`` unless ``slabs`` is given. A missing study or missing cue is a warning,
    never an exception. With no cues, the result is offset 0 with the prior's confidence.
    """
    warnings: list[str] = []
    not_eval = [
        "axial transverse-process / sacral-ala morphology (Castellvi typing; Konin & Walz 2010 say sagittal "
        "alone can miss type II/III)",
        "iliac crest level, aortic bifurcation (no vessel or crest localisation implemented)",
        "localizer rib count from a landmark",
        "S1-S2 remnant disc and sacral canal ending on the sagittal (not implemented in this module)",
    ]
    if not anchors.discs:
        warnings.append("no discs: numbering evidence not computed")
        post = dict(zip(OFFSETS, config.prior_offset, strict=True))
        return _result(post, [], warnings, not_eval, config)
    use_slabs: Sequence[Stack]
    if slabs is not None:
        use_slabs = list(slabs)
    elif study is not None:
        use_slabs = study.slabs("ax")
    else:
        use_slabs = []
    votes: list[Vote | None] = []
    sv = slab_vote(use_slabs, anchors, config) if use_slabs else None
    if sv is None and not use_slabs:
        warnings.append("no oblique slabs in the study: slab cue not available")
    votes.append(sv)
    votes.append(tilt_vote(anchors, config))
    if votes[-1] is None:
        warnings.append("fewer than two discs: tilt cue not available")
    votes.append(conus_vote(anchors, config))
    if votes[-1] is None:
        warnings.append("conus level not located: conus cue not available")
    post = combine(votes, config)
    used = [v for v in votes if v is not None]
    if not used:
        warnings.append("no numbering cue available; offset 0 assumed from the prior")
    return _result(post, used, warnings, not_eval, config)


def _result(
    post: dict[int, float],
    votes: list[Vote],
    warnings: list[str],
    not_eval: list[str],
    config: NumberingConfig,
) -> NumberingEvidence:
    best = max(OFFSETS, key=lambda o: post[o])
    if post[best] < config.uncertain_below:
        warnings.append(
            f"lowest-disc offset uncertain (best {best:+d} at {post[best]:.2f}); check the level names by eye"
        )
    kind = {1: "sacralized", -1: "lumbarized"}.get(best) if best != 0 else None
    p_trans = 1.0 - post[0]
    return NumberingEvidence(
        offset=best,
        offset_confidence=float(post[best]),
        posterior={o: float(post[o]) for o in OFFSETS},
        transitional=best != 0 and post[best] >= 0.5,
        transitional_kind=kind if best != 0 and post[best] >= 0.5 else None,
        transitional_confidence=float(p_trans),
        votes=votes,
        warnings=warnings,
        not_evaluated=not_eval,
    )


__all__ = [
    "AXIAL_SLAB_CUE",
    "DEFAULT_NUMBERING_CONFIG",
    "OFFSETS",
    "NumberingConfig",
    "NumberingEvidence",
    "Vote",
    "axial_slab_vote",
    "combine",
    "conus_vote",
    "numbering_evidence",
    "relabel_discs",
    "slab_vote",
    "tilt_vote",
]
