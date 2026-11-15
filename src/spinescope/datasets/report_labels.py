"""Rule-based parser for free-text lumbar MRI reports (Sudirman radiologist notes) into per-level labels.

The parser is deliberately simple and transparent: a fixed list of term patterns (disc contour, canal, foramen,
lateral recess, nerve root, thecal sac, listhesis, Modic, facet, ligamentum flavum, degeneration and a few others),
each with a negation rule and a level rule. It is written for the free-text style of the Sudirman notes, which has
frequent misspellings (``protruion``, ``dissicating``, ``modec``), abbreviations (``rt``, ``lt``) and clauses that
hold several levels. It is validated on synthetic sentences in ``tests/test_report_labels.py``, not on the notes.

Method, per clause (sentences split at full stops, semicolons, newlines and ``///``):

1. Level mentions are ``Lx-Ly``, ``Lx/Ly``, ``Lx-y``, ``Lx Ly`` or ``Lx over Ly``, normalised to a disc name and kept
   only if the two vertebrae are adjacent in the lumbosacral order (``L5-S1``, ``T12-L1``, ...).
2. Finding terms are matched in the clause. Each term is assigned the level mentions that fall between it and its
   neighbouring terms; if there are none, the nearest level mention in the clause is used; with no level mention the
   label is global (``level`` None), which is never scored per level.
3. Negation: a term is negated when a negation cue (``no``, ``not``, ``without``, ``adequate``, ...) precedes it in
   the same clause segment, with ``or``/``and`` inheriting the negation of the previous term, or, for canal and thecal
   sac terms, when a normal cue follows it. Negated terms give ``negative`` labels.
4. Side (``right``/``left``, ``rt``/``lt``) and zone (paracentral, foraminal, extraforaminal, subarticular, central)
   are taken from the nearest token within 40 characters.

Output is derived labels only. Report text is never stored in the result, except as generalised templates
(levels and numbers replaced) for the sentences no rule matched, so the coverage analysis can be shared.

Wording of positive labels follows the findings schema (``report/`` and ``data/findings.schema.json``), where
they are reduced to per-level fields by :func:`aggregate_levels`.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .gradings import SPIDER_BOTTOM_UP

log = logging.getLogger(__name__)

#: Positions of the lumbosacral order used to check that two vertebrae are adjacent.
_ORDER = {name: i for i, name in enumerate(SPIDER_BOTTOM_UP)}
_ROMAN = {"i": 1, "ii": 2, "iii": 3}
_LEVEL_RE = re.compile(
    r"(?<![a-z0-9])(t1[0-2]|t[6-9]|l[1-6]|s1)\s?(?:-|/|–|to|over|\s)\s?(l[1-6]|s1|[1-6])(?![a-z0-9])"
)
_CLAUSE_SPLIT = re.compile(r"\.|;|\n|/{3,}")
_CONJ_RE = re.compile(r"\b(?:but|however|although|while|with)\b")
_NEG_RE = re.compile(
    r"\b(?:no|not|without|nil|absent|negative|unremarkable|free of|adequate|normal|preserved|still adequate)\b"
)
_NORMAL_AFTER_RE = re.compile(r"\b(?:adequate|no significant|no |normal|preserved|unremarkable)\b")
_BOILERPLATE_RE = re.compile(
    r"^(?:lss|lumbosacral|lumbar spine|mri|nb|spine|findings?|impression|report|study|sagittal|axial|"
    r"t1|t2|stir|dr\b|features? of muscle spasm)\b"
)
_SIDE_RE = re.compile(r"\b(right|rt|left|lt|bilateral|both)\b")
_ZONE_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("extraforaminal", re.compile(r"extra\s?foramin\w*|far\s?lateral")),
    ("foraminal", re.compile(r"foramin\w*|foramen|exit\s+(?:neural\s+)?(?:canal|nerve|root)\w*|exiting")),
    ("subarticular", re.compile(r"subarticular")),
    ("paracentral", re.compile(r"para\s?cent\w*|para\s?med\w*|posterolat\w*|postero-lat\w*|posterolateral")),
    ("central", re.compile(r"\bcentral\b|\bposterocentral\b|\bcentrally\b")),
)
_EVENT_RE = re.compile(
    r"compress\w*|indent\w*|abut\w*|efface\w*|encroach\w*|impinge\w*|contact\w*|displac\w*"
)
_NARROW_RE = re.compile(r"narrow\w*|stenos\w*|encroach\w*|compress\w*|reduc\w*")
_SEVERITY_RE = re.compile(r"\b(mild|moderate|severe|marked|significant)\b")
_GRADE_RE = re.compile(r"grade\s+(i{1,3}|iv|v|[1-5])\b")
_MODIC_TYPE_RE = re.compile(r"\s*(?:type|tyoe|typ)?\s*[-:]?\s*(?:(i{1,3}|iv)\b|([1-4])\b)")

#: Contour strength order (weakest first), used to pick the strongest positive contour at a level.
CONTOUR_ORDER: tuple[str, ...] = ("bulge", "herniation", "protrusion", "extrusion", "sequestration")
HERNIATION_CONTOURS = frozenset({"herniation", "protrusion", "extrusion", "sequestration"})


@dataclass(frozen=True)
class ParsedLabel:
    """One finding found in one clause. ``value`` is a contour name, a grade, a Modic type or True/False."""

    field: str
    value: Any
    negated: bool
    level: str | None
    side: str | None = None
    zone: str | None = None
    clause: int = 0
    grade: str | None = None  # severity word or Meyerding grade (listhesis, stenosis)
    direction: str | None = None  # listhesis: anterolisthesis | retrolisthesis


@dataclass
class ReportParse:
    """Labels of one report plus the coverage bookkeeping (no report text)."""

    labels: list[ParsedLabel] = field(default_factory=list)
    clauses: int = 0  # all clauses
    substantive: int = 0  # clauses that are not boilerplate
    parsed: int = 0  # substantive clauses with at least one label
    unparsed_templates: Counter[str] = field(default_factory=Counter)  # generalised text of unparsed clauses

    @property
    def coverage(self) -> float:
        return self.parsed / self.substantive if self.substantive else float("nan")


# --------------------------------------------------------------------------- level mentions
def normalise_level(upper: str, lower: str) -> str | None:
    """Disc name from two vertebra tokens (``l4``, ``5``, ``s1``, ``t12``), or None if not adjacent."""
    u = upper.upper()
    low = (f"L{lower}" if u.startswith("L") else None) if lower.isdigit() else lower.upper()
    if low is None or u not in _ORDER or low not in _ORDER:
        return None
    if _ORDER[u] != _ORDER[low] + 1:
        return None
    return f"{u}-{low}"


def level_mentions(clause: str) -> list[tuple[int, int, str]]:
    """(start, end, disc name) of every adjacent-level mention in a lower-case clause."""
    out: list[tuple[int, int, str]] = []
    for m in _LEVEL_RE.finditer(clause):
        name = normalise_level(m.group(1), m.group(2))
        if name is not None:
            out.append((m.start(), m.end(), name))
    return out


# --------------------------------------------------------------------------- term handlers
@dataclass
class _Hit:
    start: int
    end: int
    field: str
    value: Any
    kind: str  # "disc" (side/zone used), "event" (needs an event cue), "normal_after" (normal cue after), ...


def _hits(clause: str) -> list[_Hit]:
    hits: list[_Hit] = []
    for m in re.finditer(r"sequest\w*|extru\w*|protru\w*|(?:her|he)ni?at\w*|hernia\b|bulg\w*", clause):
        word = m.group(0)
        value = (
            "sequestration"
            if word.startswith("sequest")
            else "extrusion"
            if word.startswith("extru")
            else "protrusion"
            if word.startswith("protru")
            else "herniation"
            if word.startswith("hern") or word.startswith("heni")
            else "bulge"
        )
        hits.append(_Hit(m.start(), m.end(), "contour", value, "disc"))
    for m in re.finditer(r"olisthes\w*|olithes\w*|listhes\w*|lithesis", clause):
        hits.append(_Hit(m.start(), m.end(), "listhesis", True, "listhesis"))
    for m in re.finditer(
        r"canal\s+(?:stenosis|narrow\w*)|stenos\w*|narrowing\s+of\s+(?:the\s+)?(?:spinal|central)\s+canal",
        clause,
    ):
        hits.append(_Hit(m.start(), m.end(), "stenosis", True, "stenosis"))
    for m in re.finditer(
        r"canal\s+(?:is\s+)?(?:still\s+)?(?:appear\w*\s+)?ad\w{1,3}ate|ad\w{1,3}ate\s+(?:spinal\s+|central\s+)?canal|"
        r"(?:spinal|central)\s+canal\s+(?:is\s+)?(?:patent|normal)",
        clause,
    ):
        hits.append(_Hit(m.start(), m.end(), "stenosis", False, "normal"))
    for m in re.finditer(r"ad\w{1,3}ate\s+thecal\s+sac", clause):
        hits.append(_Hit(m.start(), m.end(), "thecal_sac", False, "normal"))
    for m in re.finditer(
        r"narrow\w*\s+(?:of\s+)?(?:the\s+)?(?:(?:left|right|lt|rt)\s+)?(?:neural\s+)?foramen\w*|"
        r"foramen\s+narrow\w*|foramin\w*\s+(?:narrow\w*|stenos\w*)",
        clause,
    ):
        hits.append(_Hit(m.start(), m.end(), "stenosis", True, "stenosis"))
    for m in re.finditer(r"exit\w*\s+(?:neural\s+)?(?:canals?|nerve\s+roots?|roots?)", clause):
        hits.append(_Hit(m.start(), m.end(), "stenosis", True, "event"))
    for m in re.finditer(r"lateral\s+recess", clause):
        hits.append(_Hit(m.start(), m.end(), "lateral_recess", True, "event"))
    for m in re.finditer(r"nerve\s?roots?|nerveroots|neural\s+root\w*", clause):
        hits.append(_Hit(m.start(), m.end(), "nerve_root", True, "event"))
    for m in re.finditer(r"thecal\s*sac|dural\s+sac", clause):
        hits.append(_Hit(m.start(), m.end(), "thecal_sac", True, "event"))
    for m in re.finditer(r"mod[ie]c\w*", clause):
        hits.append(_Hit(m.start(), m.end(), "modic", None, "modic"))
    for m in re.finditer(r"facet\w*", clause):
        hits.append(_Hit(m.start(), m.end(), "facet", True, "facet"))
    for m in re.finditer(r"lig\w*\s+flavum", clause):
        hits.append(_Hit(m.start(), m.end(), "ligamentum_flavum", True, "event"))
    for m in re.finditer(r"osteophyt\w*", clause):
        hits.append(_Hit(m.start(), m.end(), "osteophytes", True, "plain"))
    for m in re.finditer(r"schmorl\w*", clause):
        hits.append(_Hit(m.start(), m.end(), "schmorl_node", True, "plain"))
    for m in re.finditer(
        r"annul\w*\s+(?:tear|fissure)|torn\s+annul\w*|tear\s+of\s+(?:the\s+)?annul\w*", clause
    ):
        hits.append(_Hit(m.start(), m.end(), "annular_fissure", True, "plain"))
    for m in re.finditer(
        r"fractur\w*|subluxation|hemangioma|cyst\w*|lipoma|marrow|operat\w*|surger\w*|granulation|"
        r"enhancing|renal|pelvic|previo?us|reconversion|mottled|contrast\s+study",
        clause,
    ):
        hits.append(_Hit(m.start(), m.end(), "other_finding", True, "plain"))
    for m in re.finditer(r"vacuum\w*|vacume\w*", clause):
        hits.append(_Hit(m.start(), m.end(), "vacuum_phenomenon", True, "plain"))
    for m in re.finditer(
        r"narrowed\s+(?=[lts]\d)|narrow\w*\s+(?:the\s+)?(?:disc|intervertebral)|disc\s+(?:space\s+)?narrow\w*|disc\s+height\s+(?:loss|reduc\w*)",
        clause,
    ):
        hits.append(_Hit(m.start(), m.end(), "disc_narrowing", True, "plain"))
    for m in re.finditer(r"de?hyadr\w*|dehydr\w*|dis+ec\w*|dis+ic\w*|desicc\w*", clause):
        hits.append(_Hit(m.start(), m.end(), "disc_dehydration", True, "plain"))
    for m in re.finditer(r"degenerat\w*|spondylo(?:tic|sis)\b|spondylotic", clause):
        hits.append(_Hit(m.start(), m.end(), "degeneration", True, "plain"))
    for m in re.finditer(r"scoliosis", clause):
        hits.append(_Hit(m.start(), m.end(), "scoliosis", True, "plain"))
    for m in re.finditer(r"muscle\s+spasm|muscular\s+spasm", clause):
        hits.append(_Hit(m.start(), m.end(), "muscle_spasm", True, "plain"))
    hits.sort(key=lambda h: (h.start, h.end))
    return _drop_overlaps(hits)


def _drop_overlaps(hits: list[_Hit]) -> list[_Hit]:
    """Keep the first of two overlapping hits (e.g. ``herniat`` inside ``protrusion or herniation`` stays, but a
    ``narrowed`` lookahead does not duplicate a ``disc`` narrowing match)."""
    out: list[_Hit] = []
    for h in hits:
        if out and h.start < out[-1].end and h.field == out[-1].field:
            continue
        out.append(h)
    return out


def _nearest(clause: str, pos: int, pattern: re.Pattern[str], window: int) -> str | None:
    best: tuple[int, str] | None = None
    for m in pattern.finditer(clause, max(0, pos - window), min(len(clause), pos + window)):
        d = abs(m.start() - pos)
        if best is None or d < best[0]:
            best = (d, m.group(0))
    return best[1] if best else None


def _side(clause: str, hit: _Hit) -> str | None:
    tok = _nearest(clause, hit.start, _SIDE_RE, 40)
    if tok in ("right", "rt"):
        return "right"
    if tok in ("left", "lt"):
        return "left"
    if tok in ("bilateral", "both"):
        return "bilateral"
    return None


def _zone(clause: str, hit: _Hit) -> str | None:
    best: tuple[int, str] | None = None
    for name, pat in _ZONE_RES:
        m = None
        for m in pat.finditer(clause, max(0, hit.start - 40), min(len(clause), hit.end + 40)):
            d = abs(m.start() - hit.start)
            if best is None or d < best[0]:
                best = (d, name)
    return best[1] if best else None


def _negated(clause: str, hit: _Hit, prev: _Hit | None, prev_neg: bool) -> bool:
    if hit.kind == "normal":  # the normal finding is named by the term itself ("adequate spinal canal")
        return True
    start = prev.end if prev is not None else 0
    window = clause[start : hit.start]
    conj = list(_CONJ_RE.finditer(window))
    if conj:
        window = window[conj[-1].end() :]
    if _NEG_RE.search(window):
        return True
    if prev_neg and prev is not None and re.fullmatch(r"[\s/&,]*(?:or|and)?[\s/&,]*", window or " "):
        return True
    # "the spinal canal is still adequate": for a canal term the normal cue comes after it
    after = clause[hit.end : min(len(clause), hit.end + 50)]
    return hit.field == "stenosis" and bool(_NORMAL_AFTER_RE.search(after))


def _value_for_modic(clause: str, hit: _Hit) -> Any:
    m = _MODIC_TYPE_RE.match(clause, hit.end)
    if m is None:
        return "present"
    if m.group(1):
        return _ROMAN.get(m.group(1), "present")
    return int(m.group(2))


def _grade(clause: str, hit: _Hit) -> str | None:
    window = clause[max(0, hit.start - 30) : hit.end + 50]
    g = _GRADE_RE.search(window)
    if g:
        return g.group(1).upper()
    s = _SEVERITY_RE.search(window)
    return s.group(1) if s else None


def _direction(clause: str, hit: _Hit) -> str | None:
    window = clause[max(0, hit.start - 40) : hit.end + 40]
    if "retro" in window:
        return "retrolisthesis"
    if "antero" in window or "anterior" in window:
        return "anterolisthesis"
    return None


# --------------------------------------------------------------------------- clause parsing
def _assign_levels(levels: list[tuple[int, int, str]], hits: list[_Hit]) -> dict[int, list[str]]:
    """Level names for each hit. Primary findings (contours, listhesis, Modic, ...) take the level mentions nearest to
    them; a primary hit left without one takes the nearest mention in the clause. Event terms (nerve root, thecal sac,
    exit canal, lateral recess) take a level only when a mention lies within 30 characters, otherwise they are global."""
    assigned: dict[int, list[str]] = {i: [] for i in range(len(hits))}
    if not levels or not hits:
        return assigned
    primary = [i for i, h in enumerate(hits) if h.kind != "event"]
    for s, e, name in levels:
        if not primary:
            break
        best = min(primary, key=lambda i: max(0, hits[i].start - e, s - hits[i].end))
        assigned[best].append(name)
    for i in primary:
        if not assigned[i]:
            h = hits[i]
            _s, _e, name = min(levels, key=lambda lv: max(0, h.start - lv[1], lv[0] - h.end))
            assigned[i] = [name]
    for i, h in enumerate(hits):
        if h.kind == "event":
            near = [name for s, e, name in levels if max(0, h.start - e, s - h.end) <= 30]
            assigned[i] = near
    return assigned


def _labels_for_clause(clause: str, index: int) -> list[ParsedLabel]:
    levels = level_mentions(clause)
    hits = _hits(clause)
    assigned = _assign_levels(levels, hits)
    out: list[ParsedLabel] = []
    prev: _Hit | None = None
    prev_neg = False
    for i, hit in enumerate(hits):
        neg = _negated(clause, hit, prev, prev_neg)
        if hit.kind == "event" and not neg and hit.field != "lateral_recess":
            window = clause[max(0, hit.start - 40) : min(len(clause), hit.end + 60)]
            if not _EVENT_RE.search(window):
                prev, prev_neg = hit, neg
                continue
        level_names: list[str | None] = list(dict.fromkeys(assigned.get(i, []))) or [None]
        for lvl in level_names:
            value, side, zone, grade, direction = _value_of(clause, hit, neg)
            out.append(
                ParsedLabel(hit.field, value, neg, lvl, side, zone, index, grade=grade, direction=direction)
            )
        prev, prev_neg = hit, neg
    return out


def _value_of(
    clause: str, hit: _Hit, neg: bool
) -> tuple[Any, str | None, str | None, str | None, str | None]:
    """(value, side, zone, grade, direction) of one hit. Disc contours keep their name, for the herniation logic."""
    if hit.field == "contour":
        return hit.value, _side(clause, hit), _zone(clause, hit), None, None
    if hit.field == "modic":
        return (0 if neg else _value_for_modic(clause, hit)), None, None, None, None
    if hit.field == "listhesis":
        return (not neg), None, None, _grade(clause, hit), _direction(clause, hit)
    if hit.field == "stenosis":
        window = clause[max(0, hit.start - 40) : hit.end + 40]
        if re.search(r"foramin\w*|foramen|exit", window):
            return (not neg), _side(clause, hit), "foraminal", None, None
        return (not neg), None, None, _grade(clause, hit), None
    if hit.field in ("facet", "nerve_root"):
        return (not neg), _side(clause, hit), None, None, None
    return (not neg), None, None, None, None


def _is_boilerplate(clause: str, labels_found: bool) -> bool:
    if labels_found:
        return False
    stripped = clause.strip(" -:/*\t")
    if not stripped:
        return True
    if _BOILERPLATE_RE.match(stripped):
        return True
    return len(stripped.split()) <= 2 and not re.search(r"[a-z]{5,}", stripped)


def generalise(text: str) -> str:
    """Template of a clause: levels become ``<LVL>``, numbers ``<N>`` (used for the coverage summary only)."""
    t = _LEVEL_RE.sub("<LVL>", text.lower())
    t = re.sub(r"\d+(?:\.\d+)?", "<N>", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_report(text: str) -> ReportParse:
    """Parse one report. Returns derived labels and coverage counts; the text itself is not kept."""
    out = ReportParse()
    lowered = (text or "").lower().replace("\r", "\n")
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(lowered)]
    clauses = [c for c in clauses if c]
    out.clauses = len(clauses)
    for i, clause in enumerate(clauses):
        labels = _labels_for_clause(clause, i)
        if _is_boilerplate(clause, bool(labels)):
            continue
        out.substantive += 1
        if labels:
            out.parsed += 1
            out.labels.extend(labels)
        else:
            out.unparsed_templates[generalise(clause)] += 1
    return out


# --------------------------------------------------------------------------- per-level aggregation
#: Presence-type fields and the value a positive label gives.
BOOLEAN_FIELDS: tuple[str, ...] = (
    "canal_stenosis",
    "foraminal_narrowing",
    "lateral_recess_stenosis",
    "nerve_root_compression",
    "thecal_sac_compression",
    "listhesis",
    "facet_arthropathy",
    "ligamentum_flavum_thickening",
    "osteophytes",
    "schmorl_node",
    "annular_fissure",
    "vacuum_phenomenon",
    "disc_narrowing",
    "disc_dehydration",
    "degeneration",
    "scoliosis",
    "muscle_spasm",
    "other_finding",
)
_FIELD_NAME: dict[str, str] = {
    "stenosis": "canal_stenosis",
    "lateral_recess": "lateral_recess_stenosis",
    "nerve_root": "nerve_root_compression",
    "thecal_sac": "thecal_sac_compression",
    "facet": "facet_arthropathy",
    "ligamentum_flavum": "ligamentum_flavum_thickening",
    "osteophytes": "osteophytes",
    "schmorl_node": "schmorl_node",
    "annular_fissure": "annular_fissure",
    "vacuum_phenomenon": "vacuum_phenomenon",
    "disc_narrowing": "disc_narrowing",
    "disc_dehydration": "disc_dehydration",
    "degeneration": "degeneration",
    "scoliosis": "scoliosis",
    "muscle_spasm": "muscle_spasm",
    "other_finding": "other_finding",
    "listhesis": "listhesis",
}


def aggregate_levels(labels: Iterable[ParsedLabel]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-level fields from the labels of one report.

    Each field is ``{"value", "conflict", "n"}``. Boolean fields are True if any positive label, False if only
    negated labels, and ``conflict`` is set when both exist (value None). ``contour`` is the strongest positive
    contour (False-type negations are not contours). ``herniation_present`` is True for protrusion, extrusion,
    sequestration or herniation; False when a herniation is negated, or when only a bulge is described (inferred,
    flagged ``inferred``); ``modic`` is the highest positive Modic type (``"present"`` if untyped), 0 if negated.
    Global labels (``level`` None) are not returned here.
    """
    per: dict[str, dict[str, list[ParsedLabel]]] = {}
    for lab in labels:
        if lab.level is None:
            continue
        per.setdefault(lab.level, {}).setdefault(lab.field, []).append(lab)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for level, by_field in per.items():
        rec: dict[str, dict[str, Any]] = {}
        contours = by_field.get("contour", [])
        pos = [c for c in contours if not c.negated]
        neg = [c for c in contours if c.negated]
        if pos:
            best = max(pos, key=lambda c: CONTOUR_ORDER.index(c.value) if c.value in CONTOUR_ORDER else -1)
            rec["contour"] = {
                "value": best.value,
                "conflict": False,
                "n": len(pos),
                "side": best.side,
                "zone": best.zone,
            }
            if best.value in HERNIATION_CONTOURS:
                rec["herniation_present"] = {
                    "value": True,
                    "conflict": False,
                    "n": len(pos),
                    "inferred": False,
                }
            else:
                rec["herniation_present"] = {
                    "value": False,
                    "conflict": False,
                    "n": len(pos),
                    "inferred": True,
                }
        elif neg:
            rec["contour"] = {"value": "normal", "conflict": False, "n": len(neg), "side": None, "zone": None}
        if neg and not pos and any(c.value in HERNIATION_CONTOURS for c in neg):
            rec["herniation_present"] = {"value": False, "conflict": False, "n": len(neg), "inferred": False}
        if "modic" in by_field:
            ms = by_field["modic"]
            types = [m.value for m in ms if not m.negated and isinstance(m.value, int)]
            any_pos = [m for m in ms if not m.negated]
            if types:
                value: Any = max(types)
            elif any_pos:
                value = "present"
            elif ms:
                value = 0
            else:
                value = None
            rec["modic"] = {
                "value": value,
                "conflict": bool(any_pos) and any(m.negated for m in ms),
                "n": len(ms),
            }
        for raw_field, labs in by_field.items():
            if raw_field in ("contour", "modic"):
                continue
            name = _FIELD_NAME.get(raw_field, raw_field)
            p = [x for x in labs if not x.negated]
            n = [x for x in labs if x.negated]
            if p and n:
                rec[name] = {"value": None, "conflict": True, "n": len(labs)}
            elif p:
                sides = sorted({x.side for x in p if x.side})
                rec[name] = {"value": True, "conflict": False, "n": len(p), "sides": sides}
            else:
                rec[name] = {"value": False, "conflict": False, "n": len(n)}
        out[level] = rec
    return out


def parse_reports(texts: dict[int, str]) -> dict[int, ReportParse]:
    """Parse several reports keyed by report number (no identifiers are read or kept)."""
    return {k: parse_report(v) for k, v in sorted(texts.items())}


def coverage_summary(parses: dict[int, ReportParse]) -> dict[str, Any]:
    """Coverage across reports: clause and report level, and the most frequent unparsed templates."""
    subst = sum(p.substantive for p in parses.values())
    parsed = sum(p.parsed for p in parses.values())
    reports_with = sum(1 for p in parses.values() if p.parsed > 0)
    unparsed: Counter[str] = Counter()
    for p in parses.values():
        unparsed.update(p.unparsed_templates)
    return {
        "reports": len(parses),
        "reports_with_any_label": reports_with,
        "substantive_clauses": subst,
        "parsed_clauses": parsed,
        "clause_coverage": round(parsed / subst, 4) if subst else None,
        "unparsed_clauses": subst - parsed,
        "top_unparsed_templates": unparsed.most_common(40),
    }


def load_sudirman_report_labels(root: str | None = None) -> dict[int, ReportParse]:
    """Parse the Sudirman radiologist notes (``Radiologists Report.xlsx``) keyed by the notes' report number."""
    from .sudirman import read_radiologist_notes

    return parse_reports(read_radiologist_notes(root))
