"""Top-chain check from wide sagittal coverage: localizers and sagittal series that reach beyond the primary.

Bottom-of-spine cues (lumbosacral CNN, bottom gap, sacrum) cannot see an extra or missed disc higher up the chain.
Clinical practice counts from a landmark instead: C2 on a whole-spine localizer, or the thoracic count from the
lowest rib-bearing vertebra (T12) on a coronal image. This module gathers the evidence such a count would need:

1. **Candidates.** Sagittal localizers (``role.localizer``) and non-localizer sagittal series other than the primary
   that reach at least ``CountingConfig.reach_min_mm`` beyond the primary's coverage, above or below.
2. **Discs.** Each candidate is segmented by the same learned labeller as the primary (its own mid-sagittal plane,
   discs in LPS mm, cached network output reused when the study already has it). Names are not used.
3. **Geometry against the primary.** In the sagittal (y, z) plane within ``agree_mm``:
   * overlap agreement: primary discs inside the candidate's coverage that the candidate also shows;
   * discs the candidate shows above the primary's top disc, inside the primary's coverage and unmatched (a
     possible missed disc at the top of the primary chain);
   * whether the primary's top disc is seen by the candidate inside the candidate's coverage (absent: a possible
     extra disc at the top of the primary chain);
   * how far each candidate reaches above and below the primary.
4. **Landmark.** A count from the top needs a verifiable landmark: C2, or the rib-head count from T12. Neither has
   a detector here. The learned labeller is trained on lumbar and thoracic sagittal slices (SPIDER), and no public
   set in use has cervical or rib labels, so ``landmark`` stays None and no offset vote is produced. The result
   abstains; the fusion layer is unchanged unless a future landmark source supplies ``CountingResult.vote``.

Why no offset vote for top-chain errors: the fusion votes move the lowest disc's name by an offset. A missed or
extra disc at the top changes how many discs lie above the lowest one, not the lowest name. A count from C2 would
fix the lowest name only through the presacral count prior (23 discs from C2-3 to L5-S1 in 24 presacral bodies), and
it would not say which discs above are wrong. Top names would need a top-down relabel, which is not implemented.

The output is evidence only. It is stored in ``anchors.numbering["counting"]`` and changes no name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .anchors import Anchors
from .fusion import match_discs, stack_z_range
from .numbering_checks import Vote

if TYPE_CHECKING:  # pragma: no cover
    from .ml.labeller import SegmentedStack

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CountingConfig:
    """Settings of the top-chain check. Judgement defaults, documented in docs/labeling/lab34_counting.md."""

    #: A sagittal series counts as wider than the primary when it reaches this far beyond the primary's coverage.
    #: Source: one level (the shortest disc-centre spacing, config.level_spacing_mm[0]), so a candidate that only
    #: repeats the primary's top level is not a candidate.
    reach_min_mm: float = DEFAULT.level_spacing_mm[0]
    #: In-plane (y, z) tolerance for matching discs between two series (config.disc_match_mm).
    agree_mm: float = DEFAULT.disc_match_mm


DEFAULT_COUNTING = CountingConfig()


@dataclass
class CandidateEvidence:
    """What one candidate series shows about the primary chain's top and overlap."""

    key: str
    localizer: bool
    n_discs: int
    reach_above_mm: float  # candidate top minus primary top (positive: the candidate reaches higher)
    reach_below_mm: float  # primary bottom minus candidate bottom (positive: the candidate reaches lower)
    overlap_discs: int  # primary discs inside the candidate's coverage
    overlap_matched: int  # of those, discs the candidate also shows in place
    top_disc_seen: (
        bool | None
    )  # primary top disc: True seen, False absent, None outside the candidate's coverage
    extra_above_top: list[str] = field(
        default_factory=list
    )  # candidate discs above the primary top, unmatched
    pitch_mm: float | None = (
        None  # median disc-centre spacing of the candidate's chain (None: fewer than 2 discs)
    )
    pitch_plausible: bool = (
        False  # a spine's spacing: median >= level_spacing_mm[0], no gap > level_spacing_mm[1]
    )

    @property
    def top_disagrees(self) -> bool:
        """The candidate disagrees with the primary's top of chain. Only a plausible chain can disagree."""
        return self.pitch_plausible and (bool(self.extra_above_top) or self.top_disc_seen is False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "localizer": self.localizer,
            "n_discs": self.n_discs,
            "reach_above_mm": round(self.reach_above_mm, 1),
            "reach_below_mm": round(self.reach_below_mm, 1),
            "overlap_discs": self.overlap_discs,
            "overlap_matched": self.overlap_matched,
            "top_disc_seen": self.top_disc_seen,
            "extra_above_top": list(self.extra_above_top),
            "pitch_mm": None if self.pitch_mm is None else round(self.pitch_mm, 1),
            "pitch_plausible": self.pitch_plausible,
            "top_disagrees": self.top_disagrees,
        }


@dataclass
class CountingResult:
    """The top-chain evidence of one study. ``vote`` is None whenever the count abstains (the current state)."""

    available: bool
    reason: str
    candidates: list[CandidateEvidence] = field(default_factory=list)
    landmark: str | None = None  # "C2" or "T12" when a landmark was verified; None otherwise
    vote: Vote | None = None  # an offset vote for fusion (never produced without a landmark)
    warnings: list[str] = field(default_factory=list)

    @property
    def top_disagreement(self) -> bool:
        return any(c.top_disagrees for c in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "landmark": self.landmark,
            "vote": None if self.vote is None else {"cue": self.vote.cue, "best": self.vote.best},
            "top_disagreement": self.top_disagreement,
            "candidates": [c.to_dict() for c in self.candidates],
        }


NO_LANDMARK_REASON = (
    "no verifiable top landmark (C2 or the rib-head count from T12 needs a detector that is not implemented): "
    "counts from the top are not made, so no offset vote"
)


def _inside(z: float, span: tuple[float, float], height_mm: float) -> bool:
    """True when a disc centre at ``z`` lies inside a coverage span, with half a disc of margin at each end."""
    half = height_mm / 2.0
    return span[0] + half <= z <= span[1] - half


def _height(anchors: Anchors, label: str) -> float:
    return float(anchors.discs[label].height_mm or 8.0)


def chain_spacing(anchors: Anchors) -> tuple[float | None, float | None]:
    """(median, largest) centre-to-centre spacing of a chain in mm; (None, None) with fewer than two discs."""
    zs = sorted(float(d.center[2]) for d in anchors.discs.values())
    if len(zs) < 2:
        return None, None
    gaps = np.diff(zs)
    return float(np.median(gaps)), float(np.max(gaps))


def candidate_stacks(study: Study, primary: Stack, cfg: CountingConfig = DEFAULT_COUNTING) -> list[Stack]:
    """Sagittal stacks other than ``primary`` that reach at least ``cfg.reach_min_mm`` beyond the primary's coverage."""
    p_lo, p_hi = stack_z_range(primary)
    out: list[Stack] = []
    for st in study.stacks.values():
        if (
            st.plane != "sag"
            or st.key == primary.key
            or st.meta.get("role", {}).get("slab_index") is not None
        ):
            continue
        lo, hi = stack_z_range(st)
        if hi - p_hi >= cfg.reach_min_mm or p_lo - lo >= cfg.reach_min_mm:
            out.append(st)
    return sorted(out, key=lambda s: s.key)


def analyse_candidate(
    primary: Anchors,
    primary_span: tuple[float, float],
    cand: Anchors,
    cand_span: tuple[float, float],
    *,
    key: str,
    localizer: bool,
    cfg: CountingConfig = DEFAULT_COUNTING,
) -> CandidateEvidence:
    """Geometry of one candidate's disc set against the primary's (pure: no image data).

    A chain is a spine's only when its median disc spacing is at least ``config.level_spacing_mm[0]`` and no single
    spacing exceeds ``config.level_spacing_mm[1]`` (a missed disc). Otherwise the candidate is recorded but cannot
    raise the top-chain flag.
    """
    p_top_lab = max(primary.discs, key=lambda lab: primary.discs[lab].center[2])
    p_top_z = float(primary.discs[p_top_lab].center[2])
    match = match_discs(primary, cand, cfg.agree_mm)  # primary label -> candidate label, or None
    used = {v for v in match.values() if v is not None}
    overlap = [
        lab
        for lab in primary.discs
        if _inside(float(primary.discs[lab].center[2]), cand_span, _height(primary, lab))
    ]
    matched = sum(1 for lab in overlap if match[lab] is not None)
    top_seen: bool | None = None
    if _inside(p_top_z, cand_span, _height(primary, p_top_lab)):
        top_seen = match[p_top_lab] is not None
    extra: list[str] = []
    for lab, d in cand.discs.items():
        z = float(d.center[2])
        if lab in used or z <= p_top_z:
            continue
        if _inside(z, primary_span, _height(cand, lab)):
            extra.append(lab)
    pitch, largest = chain_spacing(cand)
    plausible = (
        pitch is not None
        and largest is not None
        and pitch >= DEFAULT.level_spacing_mm[0]
        and largest <= DEFAULT.level_spacing_mm[1]
    )
    return CandidateEvidence(
        key=key,
        localizer=localizer,
        n_discs=len(cand.discs),
        reach_above_mm=float(cand_span[1] - primary_span[1]),
        reach_below_mm=float(primary_span[0] - cand_span[0]),
        overlap_discs=len(overlap),
        overlap_matched=matched,
        top_disc_seen=top_seen,
        extra_above_top=sorted(extra),
        pitch_mm=pitch,
        pitch_plausible=bool(plausible),
    )


def _candidate_anchors(
    st: Stack, config: Config, session: Any, segs: dict[str, SegmentedStack] | None
) -> Anchors:
    from .ml.labeller import segment_stack
    from .ml.postprocess import anchors_from_segmentation

    seg = segs.get(st.key) if segs is not None else None
    if seg is None:
        seg = segment_stack(st, config, session=session)
    return anchors_from_segmentation(seg, config)


def count_evidence(
    study: Study,
    primary: Anchors,
    *,
    primary_stack: Stack | None,
    config: Config | None = None,
    cfg: CountingConfig = DEFAULT_COUNTING,
    session: Any = None,
    segs: dict[str, SegmentedStack] | None = None,
) -> CountingResult:
    """Top-chain evidence for ``primary`` from every wider sagittal series of ``study``.

    ``primary_stack`` is the sagittal stack the primary discs came from (``study.best('sag')``). ``segs`` maps
    stack keys to network output already computed for this study (reused, never recomputed). A failing
    candidate is skipped with a warning; the function never raises for a bad series.
    """
    if not primary.discs:
        return CountingResult(available=False, reason="no primary discs")
    if primary_stack is None:
        return CountingResult(available=False, reason="no sagittal primary stack")
    cands = candidate_stacks(study, primary_stack, cfg)
    if not cands:
        return CountingResult(
            available=False, reason="no sagittal series reaches beyond the primary's coverage (no wider view)"
        )
    cfg_run = config or DEFAULT
    primary_span = stack_z_range(primary_stack)
    evidence: list[CandidateEvidence] = []
    warnings: list[str] = []
    for st in cands:
        try:
            cand = _candidate_anchors(st, cfg_run, session, segs)
        except Exception as exc:  # a failing candidate must not sink the labels
            log.warning("counting: candidate %s failed: %s", st.key, exc)
            warnings.append(f"counting: candidate {st.key} not labelled ({type(exc).__name__})")
            continue
        if not cand.discs:
            warnings.append(f"counting: candidate {st.key} shows no disc chain")
            continue
        ev = analyse_candidate(
            primary,
            primary_span,
            cand,
            stack_z_range(st),
            key=st.key,
            localizer=bool(st.meta.get("role", {}).get("localizer")),
            cfg=cfg,
        )
        evidence.append(ev)
        if ev.pitch_mm is None or not ev.pitch_plausible:
            warnings.append(
                f"counting: candidate {st.key} disc chain is not a spine's (spacing or missed disc); "
                "not used for the top-chain check"
            )
        if ev.top_disagrees:
            warnings.append(
                f"top of chain disagrees with sagittal series {st.key}"
                + (
                    f" (extra disc(s) above the top disc: {', '.join(ev.extra_above_top)})"
                    if ev.extra_above_top
                    else ""
                )
                + (" (the top disc is absent there)" if ev.top_disc_seen is False else "")
                + ": a disc may be missed or extra at the top; names not changed"
            )
    if not evidence:
        return CountingResult(
            available=False, reason="wider series gave no usable disc chain", warnings=warnings
        )
    return CountingResult(
        available=True,
        reason=NO_LANDMARK_REASON,
        candidates=evidence,
        landmark=None,
        vote=None,
        warnings=warnings,
    )


__all__ = [
    "DEFAULT_COUNTING",
    "NO_LANDMARK_REASON",
    "CandidateEvidence",
    "CountingConfig",
    "CountingResult",
    "analyse_candidate",
    "candidate_stacks",
    "chain_spacing",
    "count_evidence",
]
