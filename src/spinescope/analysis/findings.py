"""Per-level report wiring: analysis results -> findings-schema fields ``{value, confidence, evidence[], notes}``.

Each analysis field that should appear in a structured read is registered once, as a :class:`FieldRule`:

* ``group`` and ``name`` are a level field of the findings schema (``data/findings.schema.json``, the ``Level``
  definition), for example ``disc.pfirrmann`` or ``endplate_above.modic``. Unknown groups or names are rejected
  when the rule is registered, so a typo cannot silently drop a field.
* ``analysis`` is the registered analysis that produces the value; ``extract(result, level)`` returns a
  :class:`Candidate` (value, the analysis' own confidence in [0, 1], notes, slice indices shown) or None.
* ``figure`` names the analysis figure that is cited as image evidence. The schema requires at least one image for
  every field, so a field without an image (no figure, or none saved) is **skipped** and listed in the output with
  the reason, never written without evidence.

:func:`level_findings` applies the level-name status (``confirmed`` | ``probable`` | ``unconfirmed``, from
:func:`spinescope.anatomy.summary.status_of`): the confidence is capped at medium for ``probable`` and at low for
``unconfirmed``, and the note says the level name is not confirmed. A finding about a disc whose name is uncertain
is still about a position in the spine, so it is kept, but it is never presented as firm.

Wording stays at "measured" or "suggests": these are measurements, not diagnoses.

To add a field: write an ``extract`` function next to your analysis, then ``register_field(FieldRule(...))``
from the analysis module's import (see ``docs/analysis/README.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..anatomy.anchors import Anchors
from ..anatomy.summary import status_of
from ..report.schema import level_fields
from .base import Result

#: Analysis confidence -> schema confidence word.
HIGH_AT = 0.7
MEDIUM_AT = 0.4
#: Level-name status -> highest confidence word allowed for findings on that level.
NUMBERING_CAP: dict[str, str] = {"confirmed": "high", "probable": "medium", "unconfirmed": "low"}
_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class Candidate:
    """What an ``extract`` function returns for one level."""

    value: Any
    confidence: float  # the analysis' own confidence, in [0, 1]
    notes: str = ""
    slices: tuple[int, ...] = ()  # slice indices shown in the evidence image; () = whole series


#: ``extract(result, level) -> Candidate | None``.
Extract = Callable[[Result, str], Candidate | None]


@dataclass(frozen=True)
class FieldRule:
    group: str
    name: str
    analysis: str
    extract: Extract
    #: Figure name in the analysis' ``Result.figures``, or a function of the level name giving it.
    figure: str | Callable[[str], str | None] | None = None
    notes: str = ""  # appended to every finding of this field (e.g. "uncalibrated suggestion")


_RULES: dict[tuple[str, str], FieldRule] = {}


def register_field(rule: FieldRule) -> FieldRule:
    """Register a field. Raises ValueError for an unknown schema field or a duplicate."""
    known = {(f.group, f.name) for f in level_fields()}
    if (rule.group, rule.name) not in known:
        raise ValueError(f"{rule.group}.{rule.name} is not a level field of the findings schema")
    if (rule.group, rule.name) in _RULES:
        raise ValueError(f"{rule.group}.{rule.name} is already registered")
    _RULES[(rule.group, rule.name)] = rule
    return rule


def registered_fields() -> list[FieldRule]:
    """Registered field rules, in schema order."""
    order = {(f.group, f.name): i for i, f in enumerate(level_fields())}
    return sorted(_RULES.values(), key=lambda r: order.get((r.group, r.name), 10**6))


def confidence_word(c: float) -> str:
    """``high`` / ``medium`` / ``low`` for an analysis confidence."""
    if c >= HIGH_AT:
        return "high"
    if c >= MEDIUM_AT:
        return "medium"
    return "low"


def cap_confidence(word: str, numbering: str) -> str:
    """Lower ``word`` to the cap for the numbering status (never raises it)."""
    cap = NUMBERING_CAP.get(numbering, "low")
    return word if _RANK[word] <= _RANK[cap] else cap


def _evidence(
    result: Result, figure: str | None, figures: Mapping[str, str], slices: tuple[int, ...]
) -> list[dict[str, Any]]:
    series = str(result.meta.get("series_key") or "")
    if not figure or figure not in figures or not series:
        return []
    return [{"image": figures[figure], "series": series, "slices": list(slices)}]


def level_findings(
    results: Mapping[str, Result],
    level: str,
    anchors: Anchors,
    *,
    figures: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Findings-schema level object for ``level``: ``{group: {name: {value, confidence, evidence, notes}}}``.

    ``results`` maps analysis name -> :class:`Result`; ``figures`` maps analysis name -> {figure name: image path}
    (paths of PNGs already saved, relative or absolute, as the findings schema expects). Fields with no candidate are
    omitted; fields without image evidence are omitted and reported under the key ``"_skipped"`` as
    ``{"field": "group.name", "reason": ...}``.
    """
    status = status_of(anchors)
    figs = figures or {}
    out: dict[str, Any] = {}
    skipped: list[dict[str, str]] = []
    for rule in registered_fields():
        res = results.get(rule.analysis)
        if res is None:
            continue
        cand = rule.extract(res, level)
        if cand is None:
            continue
        fig = rule.figure(level) if callable(rule.figure) else rule.figure
        ev = _evidence(res, fig, figs.get(rule.analysis, {}), cand.slices)
        if not ev:
            skipped.append({"field": f"{rule.group}.{rule.name}", "reason": "no image evidence saved"})
            continue
        word = cap_confidence(confidence_word(cand.confidence), status)
        notes = [n for n in (cand.notes, rule.notes) if n]
        if status != "confirmed":
            notes.append(f"level name {status}: the position is measured, the name is not confirmed")
        out.setdefault(rule.group, {})[rule.name] = {
            "value": cand.value,
            "confidence": word,
            "evidence": ev,
            "notes": "; ".join(notes),
        }
    if skipped:
        out["_skipped"] = skipped
    return out


# ============================================================================= built-in examples
def _discs_pfirrmann(res: Result, level: str) -> Candidate | None:
    for r in res.rows:
        if r.get("scope") == "disc" and r.get("disc") == level:
            grade = r.get("pfirrmann_suggestion")
            if not grade:
                return None
            # The thresholds are not calibrated yet, so the suggestion is never reported above medium.
            conf = min(float(r.get("pfirrmann_confidence") or 0.0), 0.6)
            return Candidate(int(grade), conf, "uncalibrated suggestion")
    return None


def _triptych(endplate: str) -> Callable[[str], str | None]:
    """Figure name of the endplate triptych for the body on the side of the endplate (see endplates.py)."""

    def name(level: str) -> str | None:
        upper, _, lower = level.partition("-")
        body = upper if endplate == "inferior" else lower
        return f"triptych_{body}_{endplate}" if body else None

    return name


def _endplate_modic(endplate: str) -> Extract:
    mapping = {"none": "0", "Modic 1": "1", "Modic 1-like": "1", "Modic 2": "2", "Modic 3": "3"}

    def fn(res: Result, level: str) -> Candidate | None:
        for r in res.rows:
            if (
                r.get("scope") == "endplate"
                and r.get("side") == "all"
                and r.get("adjacent_disc") == level
                and r.get("endplate") == endplate
            ):
                pat = str(r.get("pattern"))
                if pat.startswith("presence only"):
                    # marrow change seen, type not assessed (a sequence is missing): no Modic value, low confidence
                    return Candidate(
                        None, min(0.3, float(r.get("confidence") or 0.0)), f"{pat}; type not assessed"
                    )
                if pat not in mapping:
                    return None
                # same text as endplates.MODIC_EXPERIMENTAL_NOTE (kept literal to avoid an import cycle)
                note = (
                    "Modic typing is experimental: agreement with radiologists on public data is low; "
                    "check T1 and T2 at this endplate"
                )
                return Candidate(mapping[pat], float(r.get("confidence") or 0.0), f"pattern: {pat}; {note}")
        return None

    return fn


register_field(FieldRule("disc", "pfirrmann", "discs", _discs_pfirrmann, figure="disc_strip"))
# The inferior endplate of the body above a disc is its upper endplate; the superior endplate of the body below is
# the lower one (see endplates.py: rows carry the disc they are adjacent to).
register_field(
    FieldRule(
        "endplate_above", "modic", "endplates", _endplate_modic("inferior"), figure=_triptych("inferior")
    )
)
register_field(
    FieldRule(
        "endplate_below", "modic", "endplates", _endplate_modic("superior"), figure=_triptych("superior")
    )
)
