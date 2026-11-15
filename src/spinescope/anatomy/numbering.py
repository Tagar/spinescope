"""Level numbering: count upward from the sacrum and cross-check the result.

Discs are counted from the lowest one found. The lowest disc is called L5-S1 when a vertebral body
is visible below it (the sacrum or S1); its evidence is the visible length below the disc and whether
the disc is the most tilted one (the lumbosacral disc carries the largest sacral slope). The
fused numbering (:mod:`.fusion`) combines these with the other cues and sets the abstain status
(:func:`numbering_status`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..config import DEFAULT, Config
from ..geometry import Stack

#: Vertebra names from the bottom up (index 0 = S1). Thoracic names continue upward.
VERTEBRA_NAMES: tuple[str, ...] = (
    "S1",
    "L5",
    "L4",
    "L3",
    "L2",
    "L1",
    "T12",
    "T11",
    "T10",
    "T9",
    "T8",
    "T7",
    "T6",
    "T5",
    "T4",
    "T3",
    "T2",
    "T1",
)


def disc_names(n_discs: int) -> list[str]:
    """Disc labels from the bottom up for ``n_discs`` discs, with the lowest disc = L5-S1."""
    out = []
    for k in range(n_discs):
        upper = VERTEBRA_NAMES[k + 1] if k + 1 < len(VERTEBRA_NAMES) else f"V{k + 1}"
        lower = VERTEBRA_NAMES[k]
        out.append(f"{upper}-{lower}")
    return out


def vertebra_name(index: int) -> str:
    """Name of the vertebra at position ``index`` from the bottom (0 = S1)."""
    return VERTEBRA_NAMES[index] if index < len(VERTEBRA_NAMES) else f"V{index}"


@dataclass
class SacrumCheck:
    confidence: float
    visible_mm: float
    tilt_supports: bool | None
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sacrum_check(
    lowest_lo_arc: float,
    arc0: float,
    lowest_normal_angle: float | None,
    next_normal_angle: float | None,
    cfg: Config = DEFAULT,
) -> SacrumCheck:
    """Evidence that the lowest disc is L5-S1: body visible below it, and its tilt versus the disc above."""
    visible = float(lowest_lo_arc - arc0)
    ev: list[str] = []
    warns: list[str] = []
    if visible >= cfg.sacrum_min_visible_mm:
        conf = 0.85
        ev.append(f"vertebral body visible below the lowest disc ({visible:.0f} mm)")
    else:
        conf = 0.45
        warns.append(
            f"only {visible:.0f} mm of bone is visible below the lowest disc; it is labelled L5-S1 by default "
            "and may be L4-L5 if the field of view ends in L5"
        )
    tilt_ok: bool | None = None
    if lowest_normal_angle is not None and next_normal_angle is not None:
        tilt_ok = lowest_normal_angle >= next_normal_angle
        if tilt_ok:
            ev.append(
                f"lowest disc is the most tilted ({lowest_normal_angle:.0f} vs {next_normal_angle:.0f} deg above)"
            )
            conf = min(1.0, conf + 0.05)
        else:
            warns.append("lowest disc is less tilted than the disc above (unusual for L5-S1)")
            conf = max(0.0, conf - 0.15)
    return SacrumCheck(
        confidence=conf, visible_mm=visible, tilt_supports=tilt_ok, evidence=ev, warnings=warns
    )


_LEVEL = r"(T1[0-2]|T[1-9]|L[1-5]|S1)"
_PAIR_RE = re.compile(rf"\b{_LEVEL}\s*[-/_ ]?\s*(?:{_LEVEL}|([1-5]))\b", re.IGNORECASE)


def level_names_in(description: str) -> list[str]:
    """Disc labels written in a series description, e.g. 'L4-5 OBL AX' -> ['L4-L5'].

    Only pairs of adjacent levels are accepted (upper, lower), so 'T2 SAG L4' yields nothing.
    """
    out: list[str] = []
    for m in _PAIR_RE.finditer(description.upper()):
        upper = m.group(1)
        lower = m.group(2) or (f"{upper[0]}{m.group(3)}" if m.group(3) else None)
        if lower is None:
            continue
        ui, li = _level_index(upper), _level_index(lower)
        if ui is not None and li is not None and ui - li == 1:
            out.append(f"{upper}-{lower}")
    return out


def _level_index(name: str) -> int | None:
    """Position from the bottom (S1 = 0, L5 = 1, ...), or None for unknown names."""
    name = name.upper()
    if name in VERTEBRA_NAMES:
        return VERTEBRA_NAMES.index(name)
    return None


def slab_crosscheck(
    slabs: list[Stack],
    disc_centres: dict[str, np.ndarray],
    disc_normals: dict[str, np.ndarray],
    disc_heights: dict[str, float],
) -> tuple[int, int, list[str]]:
    """Compare angled slabs with the labelled discs. Returns (agree, disagree, evidence).

    Agreement: the level named in a slab description is the disc whose centre the slab passes
    through (within half a disc height) and whose normal matches the slab normal (within 12 deg).
    """
    agree = disagree = 0
    ev: list[str] = []
    for sl in slabs:
        names = level_names_in(sl.description or "")
        for lab in names:
            if lab not in disc_centres:
                continue
            c = disc_centres[lab]
            d = float(np.dot(c - sl.ipp.mean(axis=0), sl.normal))
            ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(sl.normal, disc_normals[lab])), -1, 1))))
            if abs(d) <= disc_heights[lab] / 2 + 3.0 and ang <= 12.0:
                agree += 1
            else:
                disagree += 1
                ev.append(
                    f"slab named {lab} does not pass through that disc (off-plane {d:.0f} mm, {ang:.0f} deg)"
                )
    if agree:
        ev.append(f"{agree} slab description(s) name the disc they pass through")
    return agree, disagree, ev


# ------------------------------------------------------------------ abstain policy
#: ``anchors.numbering['status']`` values. Downstream modules and briefs should treat anything other than
#: ``confirmed`` as an unconfirmed level name (show the name with its status, never as settled).
STATUS_CONFIRMED = "confirmed"
STATUS_PROBABLE = "probable"
STATUS_UNCONFIRMED = "unconfirmed"

#: Minimum calibrated probability that the whole numbering of the study is right, per status. Used only when
#: the packaged calibration is missing (uncalibrated fallback); the shipped values are fitted on SPIDER training
#: cases (see ``spinescope.anatomy.calibration`` and docs/labeling/lab08_fusion.md) and override these.
DEFAULT_STATUS_THRESHOLDS: dict[str, float] = {STATUS_CONFIRMED: 0.9, STATUS_PROBABLE: 0.6}


def numbering_status(confidence: float, thresholds: dict[str, float] | None = None) -> str:
    """``confirmed`` at or above the confirmed threshold, ``probable`` at or above the probable one, else
    ``unconfirmed``. ``confidence`` is the (calibrated) probability that the numbering is right."""
    th = thresholds or DEFAULT_STATUS_THRESHOLDS
    if confidence >= th[STATUS_CONFIRMED]:
        return STATUS_CONFIRMED
    if confidence >= th[STATUS_PROBABLE]:
        return STATUS_PROBABLE
    return STATUS_UNCONFIRMED


def shift_vertebra_name(name: str, offset: int) -> str:
    """Name of a vertebral body after a lowest-disc offset (+1 sacralized: every body moves up one level;
    -1 lumbarized: only the body below the lowest disc, S1, becomes L6). Other names are returned unchanged."""
    if name not in VERTEBRA_NAMES:
        return name
    idx = VERTEBRA_NAMES.index(name)
    if offset == 1:
        return VERTEBRA_NAMES[idx + 1] if idx + 1 < len(VERTEBRA_NAMES) else name
    if offset == -1:
        return "L6" if name == "S1" else name
    return name
