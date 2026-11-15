"""Synthesis: every registered analysis turned into one per-level and per-study picture a reader can use.

Registry-driven. The fields come from :func:`spinescope.analysis.findings.registered_fields` (level fields, one per
``register_field``) and :func:`registered_global_fields` (study-level fields, :func:`register_global_field` here).
An analysis merged later appears in the output as soon as it registers its fields; nothing in this module lists
analyses by name, except the built-in global registration at the bottom.

What :func:`build_synthesis` returns (:class:`Synthesis`):

* ``levels``: per disc level (superior to inferior): the level-name status, the findings (value, confidence, the
  validation status of the field, evidence figure names, notes), and the fields that were **not assessed** with the
  reason (analysis not run, analysis failed, missing sequence, out of the field of view, no image evidence, not
  measured by the analysis).
* ``global_findings``: whole-study fields (alignment, marrow, ...), same shape with ``level == "global"``.
* ``pain_evidence``: levels ordered by the strength of the measured evidence, with the abnormal structures grouped as
  disc / endplate (vertebrogenic) / facet / canal / foramen and nerve root / segment. Descriptive only.
* ``study``: image quality, coverage (sequences, numbering, analyses run), and the study-level "what was not assessed".

Wording: a measured number is "measured"; a graded or categorical call is "suggests". No treatment advice. The
disclaimer block (:data:`DISCLAIMER`) is printed on every output. Research software, not a diagnosis.

:func:`findings_document` builds the findings JSON (schema 1.0) from the same registry, with image paths of the saved
evidence PNGs; a field whose figure was not saved is left out of the file and listed under ``not assessed``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..analysis.base import Result, available
from ..analysis.findings import Candidate, cap_confidence, confidence_word, level_findings, registered_fields
from ..anatomy.anchors import Anchors, split_disc
from ..anatomy.summary import ABSTAIN_POLICY, labelling_summary, quality_summary, status_of
from ..config import DEFAULT, Config
from ..study import Study
from .schema import (
    CONFIDENCE_LEVELS,
    GLOBAL,
    SCHEMA_VERSION,
    field_order,
    global_fields,
    is_disc_label,
    level_fields,
    level_sort_key,
)
from .validation import STATUS_WORDS, STATUSES, Validation, validation_of

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Research software (spinescope), not a medical device and not a diagnosis. Everything below is a measurement or "
    "a suggestion from automatic analyses, each with a confidence and a validation status. Nothing here establishes "
    "that a structure causes pain. Consider correlation with symptoms and examination, and review the images with a "
    "clinician. Level names marked probable or unconfirmed must be checked before use. No treatment advice is given."
)

FORMULA = (
    "strength = sum over the abnormal findings of the confidence weight times the validation weight "
    "(config.findings_confidence_weights; config.synthesis_status_weights in the order validated, weak labels, "
    "experimental, not validated). Levels are ordered by strength, then anatomically. It orders the evidence for a "
    "reader to review; it is not a severity and not a likelihood of causing symptoms."
)

#: Pain-evidence structures, in display order.
PAIN_STRUCTURES: tuple[str, ...] = (
    "disc",
    "endplate_vertebrogenic",
    "facet",
    "canal",
    "foramen_nerve_root",
    "segment",
    "other",
)
PAIN_TITLES: dict[str, str] = {
    "disc": "Disc",
    "endplate_vertebrogenic": "Endplate (vertebrogenic)",
    "facet": "Facet joint",
    "canal": "Canal and lateral recess",
    "foramen_nerve_root": "Foramen and nerve root",
    "segment": "Segment (listhesis)",
    "other": "Other",
}
_PAIN_OF_GROUP: dict[str, str] = {
    "disc": "disc",
    "endplate_above": "endplate_vertebrogenic",
    "endplate_below": "endplate_vertebrogenic",
    "facet_R": "facet",
    "facet_L": "facet",
    "central_canal": "canal",
    "lateral_recess_R": "canal",
    "lateral_recess_L": "canal",
    "ligamentum_flavum": "canal",
    "foramen_R": "foramen_nerve_root",
    "foramen_L": "foramen_nerve_root",
    "nerve_root_R": "foramen_nerve_root",
    "nerve_root_L": "foramen_nerve_root",
    "listhesis": "segment",
}

#: Reasons a field is not assessed, with the wording used in every output.
REASON_TEXT: dict[str, str] = {
    "not_run": "analysis not run for this study",
    "failed": "analysis failed",
    "missing_sequence": "missing sequence",
    "out_of_field": "out of the field of view",
    "no_evidence": "no image evidence saved",
    "not_measured": "not measured by the analysis",
    "no_value": "no value for this level",
    "no_analysis": "no analysis registered for this field",
}

_MISSING_SEQUENCE_RE = re.compile(
    r"\bno\b.*\b(T1|T2|STIR|PD|sagittal|axial|stack|sequence)\b|\bSTIR\b.*\brequired\b", re.IGNORECASE
)
_OUT_OF_FIELD_RE = re.compile(
    r"outside|field of view|out of (the )?(field|fov)|edge of the field|not in the field", re.IGNORECASE
)


# ------------------------------------------------------------------------------------- global field registry
@dataclass(frozen=True)
class GlobalRule:
    """A whole-study field: ``group`` and ``name`` are a global field of the findings schema."""

    group: str
    name: str
    analysis: str
    extract: Callable[[Result], Candidate | None]
    figure: str | None = None  # figure name in the analysis' Result.figures (image evidence is required)
    notes: str = ""


_GLOBAL_RULES: dict[tuple[str, str], GlobalRule] = {}


def register_global_field(rule: GlobalRule) -> GlobalRule:
    """Register a global field. Raises ValueError for an unknown schema field or a duplicate."""
    known = {(f.group, f.name) for f in global_fields()}
    if (rule.group, rule.name) not in known:
        raise ValueError(f"{rule.group}.{rule.name} is not a global field of the findings schema")
    if (rule.group, rule.name) in _GLOBAL_RULES:
        raise ValueError(f"{rule.group}.{rule.name} is already registered")
    _GLOBAL_RULES[(rule.group, rule.name)] = rule
    return rule


def registered_global_fields() -> list[GlobalRule]:
    """Registered global rules in schema order."""
    order = {(f.group, f.name): i for i, f in enumerate(global_fields())}
    return sorted(_GLOBAL_RULES.values(), key=lambda r: order.get((r.group, r.name), 10**6))


def _lordosis(res: Result) -> Candidate | None:
    for r in res.rows:
        if r.get("scope") == "global" and r.get("name") == "lumbar_lordosis_deg":
            value = r.get("value")
            if isinstance(value, (int, float)):
                # Supine, single slice per body: the value is kept at medium at most (see the validation record).
                return Candidate(float(value), 0.4, "supine measurement; underestimates standing lordosis")
    return None


register_global_field(
    GlobalRule("alignment", "lumbar_lordosis_deg", "alignment", _lordosis, figure="lordosis")
)


# ------------------------------------------------------------------------------------------------ data model
@dataclass(frozen=True)
class Finding:
    """One finding: a level (or global) field with a value, its confidence, validation and evidence."""

    level: str  # disc label, or "global"
    side: str  # "R", "L" or "" (not side-specific)
    group: str
    name: str
    title: str  # schema title, with the side, e.g. "Foramen Lee grade (right)"
    value: Any
    confidence: str  # low | medium | high (after the level-name cap)
    analysis: str
    figures: tuple[str, ...]  # evidence figure names, "analysis:figure"
    series: tuple[str, ...]
    notes: str
    validation: Validation
    abnormal: bool
    pain_structure: str | None

    @property
    def experimental(self) -> bool:
        return self.validation.experimental

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "side": self.side,
            "field": f"{self.group}.{self.name}",
            "title": self.title,
            "value": _plain(self.value),
            "confidence": self.confidence,
            "analysis": self.analysis,
            "figures": list(self.figures),
            "series": list(self.series),
            "notes": self.notes,
            "validation": self.validation.to_dict(),
            "experimental": self.experimental,
            "abnormal": self.abnormal,
            "pain_structure": self.pain_structure,
            "text": finding_text(self),
        }


@dataclass(frozen=True)
class NotAssessed:
    """A field that has no value for a level (or study), with the reason and its detail text."""

    field: str
    side: str
    reason: str  # a key of REASON_TEXT
    detail: str
    analysis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "side": self.side,
            "reason": self.reason,
            "reason_text": REASON_TEXT.get(self.reason, self.reason),
            "detail": self.detail,
            "analysis": self.analysis,
        }


@dataclass
class LevelSynthesis:
    """Everything known about one disc level."""

    level: str
    name_status: str
    name_policy: str
    findings: list[Finding] = field(default_factory=list)
    not_assessed: list[NotAssessed] = field(default_factory=list)
    strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "name_status": self.name_status,
            "name_policy": self.name_policy,
            "strength": round(self.strength, 3),
            "findings": [f.to_dict() for f in self.findings],
            "not_assessed": [n.to_dict() for n in self.not_assessed],
        }


@dataclass
class Synthesis:
    """The whole synthesis of one study. Plain data after :meth:`to_dict`; no paths, no free text from the study."""

    numbering: str
    levels: list[LevelSynthesis]
    global_findings: list[Finding]
    study_not_assessed: list[NotAssessed]
    image_quality: dict[str, Any]
    coverage: dict[str, Any]
    pain_evidence: list[dict[str, Any]]
    formula: str = FORMULA
    disclaimer: str = DISCLAIMER

    def level(self, label: str) -> LevelSynthesis:
        for lv in self.levels:
            if lv.level == label:
                return lv
        raise KeyError(label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "numbering": self.numbering,
            "disclaimer": self.disclaimer,
            "formula": self.formula,
            "validation_statuses": list(STATUSES),
            "levels": [lv.to_dict() for lv in self.levels],
            "global": {
                "findings": [f.to_dict() for f in self.global_findings],
                "image_quality": self.image_quality,
                "coverage": self.coverage,
                "not_assessed": [n.to_dict() for n in self.study_not_assessed],
            },
            "pain_evidence": self.pain_evidence,
        }


# ------------------------------------------------------------------------------------------- wording
def _plain(x: Any) -> Any:
    if isinstance(x, float):
        return None if x != x or x in (float("inf"), float("-inf")) else x
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _value_text(value: Any) -> str:
    if value is None:
        return "indeterminate"
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".") if value == value else "not measured"
    return str(value)


def _verb(value: Any) -> str:
    """ "measured" for a continuous number (mm, degrees); "suggests" for a grade, a category or an indeterminate value."""
    return "measured" if isinstance(value, float) else "suggests"


def finding_text(f: Finding) -> str:
    """One line for a reader: "measured" for a number, "suggests" for a grade or call; confidence and status."""
    where = "" if f.level == GLOBAL else f"{f.level}: "
    return (
        f"{_verb(f.value)} {where}{f.title} = {_value_text(f.value)} "
        f"(confidence {f.confidence}; {_status_badge(f.validation)})"
    )


def _status_badge(v: Validation) -> str:
    return STATUS_WORDS.get(v.status, v.status)


def _side_of(group: str) -> str:
    if group.endswith("_R"):
        return "R"
    if group.endswith("_L"):
        return "L"
    return ""


def _side_word(side: str) -> str:
    return {"R": "right", "L": "left"}.get(side, "")


# ------------------------------------------------------------------------------------- abnormal predicate
def is_abnormal(group: str, name: str, value: Any, cfg: Config = DEFAULT) -> bool:
    """True when a field value is outside the normal or none category (presentation rule, see config)."""
    if value is None:
        return False
    if name == "pfirrmann":
        return int(value) >= cfg.synthesis_pfirrmann_abnormal_grade
    if name == "lee_grade":
        return int(value) >= cfg.synthesis_lee_abnormal_grade
    if name == "modic":
        return str(value) != "0"
    if name == "contour":
        return str(value) != "normal"
    if name == "herniation_location":
        return str(value) != "none"
    if name == "herniation_size_mm":
        return float(value) >= cfg.herniation_tolerance_mm
    if name in ("height_loss", "annular_fissure", "migration"):
        return str(value) != "none"
    if name == "hiz":
        return str(value) == "present"
    if name == "effusion":
        return str(value) == "present"
    if name in ("stenosis", "thickening", "arthropathy", "status", "direction", "meyerding_grade"):
        return str(value) != "none"
    return False


# ------------------------------------------------------------------------------------- reasons
def _classify(warning: str) -> str:
    if warning.startswith("analysis failed"):
        return "failed"
    if _MISSING_SEQUENCE_RE.search(warning):
        return "missing_sequence"
    if _OUT_OF_FIELD_RE.search(warning):
        return "out_of_field"
    return "not_measured"


def _short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _failure_detail(res: Result) -> str:
    for w in res.warnings:
        if w.startswith("analysis failed"):
            # run_all keeps "analysis failed: <ExceptionType>: <message>"; the message can quote inputs, so the type only
            return "analysis failed: " + w[len("analysis failed: ") :].split(":", 1)[0]
    return "analysis failed"


def _reason_for_level(analysis: str, level: str, results: Mapping[str, Result]) -> tuple[str, str]:
    """Why an analysis gave no value for one level: (reason key, detail text)."""
    res = results.get(analysis)
    if res is None:
        return "not_run", "the analysis was not run for this study"
    if any(w.startswith("analysis failed") for w in res.warnings):
        return "failed", _failure_detail(res)
    structured = res.meta.get("not_assessed")
    if isinstance(structured, Mapping) and isinstance(structured.get(level), Mapping):
        entry = structured[level]
        if entry.get("reason") in REASON_TEXT:
            return str(entry["reason"]), _short(str(entry.get("detail", "")))
    for row in res.rows:
        if row.get("disc") == level and isinstance(row.get("reason"), str) and row["reason"]:
            return "not_measured", _short(row["reason"])
    for w in res.warnings:
        if level in w:
            return _classify(w), _short(w)
    for w in res.warnings:
        kind = _classify(w)
        if kind in ("missing_sequence", "out_of_field"):
            return kind, _short(w)
    return "no_value", "the analysis gave no value for this level"


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen


# ------------------------------------------------------------------------------------- findings objects
def _titles() -> dict[tuple[str, str], str]:
    return {(f.group, f.name): f.label for f in (*level_fields(), *global_fields())}


def _rules_by_field() -> dict[tuple[str, str], Any]:
    return {(r.group, r.name): r for r in registered_fields()}


def _finding(
    level: str,
    group: str,
    name: str,
    obj: Mapping[str, Any],
    analysis: str,
    cfg: Config,
    titles: Mapping[tuple[str, str], str],
    pain: str | None,
) -> Finding:
    value = obj.get("value")
    ev = list(obj.get("evidence") or [])
    figs = tuple(f"{analysis}:{e['image']}" for e in ev)
    series = tuple(sorted({str(e["series"]) for e in ev}))
    v = validation_of(group, name, value)
    return Finding(
        level=level,
        side=_side_of(group),
        group=group,
        name=name,
        title=titles.get((group, name), name),
        value=value,
        confidence=str(obj.get("confidence", "low")),
        analysis=analysis,
        figures=figs,
        series=series,
        notes=str(obj.get("notes", "")),
        validation=v,
        abnormal=is_abnormal(group, name, value, cfg),
        pain_structure=pain,
    )


def _level_objects(
    results: Mapping[str, Result], level: str, anchors: Anchors, cfg: Config
) -> tuple[list[Finding], dict[str, str]]:
    """Findings for one level, from the findings registry, plus the fields skipped for lack of image evidence."""
    figs = {name: {f: f for f in res.figures} for name, res in results.items()}
    obj = level_findings(results, level, anchors, figures=figs)
    skipped = {str(s["field"]): str(s["reason"]) for s in obj.get("_skipped", [])}
    titles = _titles()
    out: list[Finding] = []
    rules = _rules_by_field()
    for group, fields in obj.items():
        if group.startswith("_"):
            continue
        for name, f in fields.items():
            rule = rules.get((group, name))
            if rule is None:  # pragma: no cover - level_findings only emits registered fields
                continue
            pain = _PAIN_OF_GROUP.get(group, "other")
            out.append(_finding(level, group, name, f, rule.analysis, cfg, titles, pain))
    return out, skipped


def _global_objects(
    results: Mapping[str, Result], status: str, figures: Mapping[str, Mapping[str, str]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Global fields as findings-schema objects (cap by level-name status), and the fields without image evidence."""
    out: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for rule in registered_global_fields():
        res = results.get(rule.analysis)
        if res is None:
            continue
        cand = rule.extract(res)
        if cand is None:
            continue
        key = f"{rule.group}.{rule.name}"
        fig = rule.figure
        series = str(res.meta.get("series_key") or "")
        if not fig or fig not in figures.get(rule.analysis, {}) or not series:
            skipped[key] = "no image evidence saved"
            continue
        word = cap_confidence(confidence_word(cand.confidence), status)
        notes = [n for n in (cand.notes, rule.notes) if n]
        if status != "confirmed":
            notes.append(f"level name {status}: the position is measured, the name is not confirmed")
        ev = [{"image": figures[rule.analysis][fig], "series": series, "slices": list(cand.slices)}]
        out.setdefault(rule.group, {})[rule.name] = {
            "value": cand.value,
            "confidence": word,
            "evidence": ev,
            "notes": "; ".join(notes),
        }
    return out, skipped


def _global_findings(
    results: Mapping[str, Result], status: str, cfg: Config
) -> tuple[list[Finding], dict[str, str]]:
    figs = {name: {f: f for f in res.figures} for name, res in results.items()}
    objs, skipped = _global_objects(results, status, figs)
    titles = _titles()
    out: list[Finding] = []
    for group, fields in objs.items():
        for name, f in fields.items():
            analysis = _analysis_of_global(f"{group}.{name}")
            out.append(_finding(GLOBAL, group, name, f, analysis, cfg, titles, None))
    return out, skipped


# ------------------------------------------------------------------------------------- pain evidence
def _strength(findings: list[Finding], cfg: Config) -> float:
    total = 0.0
    for f in findings:
        if not f.abnormal:
            continue
        cw = cfg.findings_confidence_weights[CONFIDENCE_LEVELS.index(f.confidence)]
        sw = cfg.synthesis_status_weights[STATUSES.index(f.validation.status)]
        total += cw * sw
    return round(total, 3)


def _root_text(level: str, f: Finding, by_key: Mapping[tuple[str, str, str], Any]) -> str | None:
    """Anatomical name of the nerve root a finding involves, from the level label (not from any image)."""
    upper, lower = split_disc(level) if is_disc_label(level) else ("", "")
    if not upper:
        return None
    side = _side_word(f.side)
    if f.group.startswith("foramen_"):
        return f"{side} {upper} root, exiting at {level}".strip()
    if f.group.startswith("nerve_root_"):
        named = by_key.get((level, f.group, "root"))
        if isinstance(named, str) and named and named != "unknown":
            return f"{side} {named} root".strip()
        return f"{side} {upper} root (exiting) or {lower} root (traversing), course not stated".strip()
    if f.group == "lateral_recess_" + f.side:
        return f"{side} traversing {lower} root in the {level} lateral recess".strip()
    if f.group == "central_canal":
        return f"cauda equina and the traversing roots at {level}"
    return None


def _pain_line(f: Finding, root: str | None) -> dict[str, Any]:
    text = finding_text(f) + (f"; root: {root}" if root else "")
    return {"field": f"{f.group}.{f.name}", "side": f.side, "text": text, "experimental": f.experimental}


def _pain_entries(
    level: str, findings: list[Finding], by_key: Mapping[tuple[str, str, str], Any]
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {s: [] for s in PAIN_STRUCTURES}
    for f in findings:
        if f.abnormal and f.pain_structure:
            root = _root_text(level, f, by_key)
            groups[f.pain_structure].append(_pain_line(f, root))
    return groups


# ------------------------------------------------------------------------------------- study sections
def _sequences(study: Study) -> list[dict[str, Any]]:
    out = []
    for key in sorted(study.stacks):
        st = study.stacks[key]
        role = st.meta.get("role", {}) or {}
        if role.get("localizer"):
            continue
        out.append(
            {
                "key": key,
                "plane": str(st.plane),
                "contrast": str(role.get("contrast") or "unknown"),
                "slices": int(st.n),
            }
        )
    return out


def _missing_sequences(seqs: list[dict[str, Any]]) -> list[str]:
    have = {(s["plane"], s["contrast"]) for s in seqs}
    expected = [
        ("sag", "t1", "sagittal T1"),
        ("sag", "t2", "sagittal T2"),
        ("sag", "stir", "sagittal STIR"),
    ]
    missing = [label for plane, contrast, label in expected if (plane, contrast) not in have]
    if not any(p == "ax" for p, _ in have):
        missing.append("axial")
    return missing


def build_synthesis(
    study: Study,
    anchors: Anchors,
    results: Mapping[str, Result],
    config: Config | None = None,
) -> Synthesis:
    """Synthesis of one study from its labelled anchors and the analysis results (any subset of the analyses).

    ``results`` maps analysis name to :class:`Result`. An analysis that is registered but absent from ``results`` is
    reported as not run. Nothing is raised for missing inputs: they become not-assessed entries.
    """
    cfg = config or DEFAULT
    available()  # loads every analysis module, so their registrations are in place
    status = status_of(anchors)
    levels_labels = sorted(anchors.disc_labels(), key=level_sort_key)
    level_schema = level_fields()
    registered = {(f.group, f.name) for f in registered_fields()}
    schema_registered = [(f.group, f.name) for f in level_schema if (f.group, f.name) in registered]
    not_covered = [f"{f.group}.{f.name}" for f in level_schema if (f.group, f.name) not in registered]
    by_key: dict[tuple[str, str, str], Any] = {}
    levels: list[LevelSynthesis] = []
    for lab in levels_labels:
        findings, skipped = _level_objects(results, lab, anchors, cfg)
        lv = LevelSynthesis(
            level=lab,
            name_status=status,
            name_policy=ABSTAIN_POLICY[status],
            findings=sorted(findings, key=lambda f: (_group_order(f), f.side)),
        )
        for f in lv.findings:
            by_key[(lab, f.group, f.name)] = f.value
        present = {(f.group, f.name) for f in findings}
        for group, name in schema_registered:
            if (group, name) in present:
                continue
            rule = _rules_by_field()[(group, name)]
            key = f"{group}.{name}"
            if key in skipped:
                reason, detail = "no_evidence", "no image evidence saved (the field needs an evidence figure)"
            else:
                reason, detail = _reason_for_level(rule.analysis, lab, results)
            lv.not_assessed.append(NotAssessed(key, _side_of(group), reason, detail, rule.analysis))
        lv.strength = _strength(findings, cfg)
        levels.append(lv)

    g_findings, g_skipped = _global_findings(results, status, cfg)
    seqs = _sequences(study)
    study_na: list[NotAssessed] = []
    for name in sorted(results):
        res = results[name]
        if any(w.startswith("analysis failed") for w in res.warnings):
            study_na.append(NotAssessed("analysis", "", "failed", _failure_detail(res), name))
        for w in _unique([str(x) for x in res.warnings if not str(x).startswith("analysis failed")]):
            kind = _classify(w)
            if kind in ("missing_sequence", "out_of_field"):
                study_na.append(NotAssessed("analysis", "", kind, _short(w), name))
    for name in available():
        if name not in results:
            study_na.append(
                NotAssessed("analysis", "", "not_run", "the analysis was not run for this study", name)
            )
    for seq in _missing_sequences(seqs):
        study_na.append(
            NotAssessed("sequence", "", "missing_sequence", f"no {seq} stack in this study", "study")
        )
    for key, reason in sorted(g_skipped.items()):
        study_na.append(NotAssessed(key, "", "no_evidence", reason, _analysis_of_global(key)))
    if not_covered:
        study_na.append(
            NotAssessed(
                "level fields",
                "",
                "no_analysis",
                "no registered analysis yet: " + ", ".join(sorted(set(not_covered))),
                "none",
            )
        )

    covered_levels = {lv.level: {f.analysis for f in lv.findings} for lv in levels}
    analyses_cov: dict[str, Any] = {}
    for name in sorted(set(available()) | set(results)):
        got = results.get(name)
        status_word = "not_run"
        if got is not None:
            failed = any(w.startswith("analysis failed") for w in got.warnings)
            status_word = "failed" if failed else "ok"
        analyses_cov[name] = {
            "status": status_word,
            "levels_with_values": sum(1 for s in covered_levels.values() if name in s),
            "warnings": 0 if got is None else len(got.warnings),
        }
    summary = labelling_summary(anchors, cfg)
    coverage = {
        "levels_labelled": len(levels_labels),
        "numbering": status,
        "numbering_text": summary.get("status_text", ""),
        "sequences": seqs,
        "missing_sequences": _missing_sequences(seqs),
        "analyses": analyses_cov,
        "fields_registered": len(schema_registered),
        "fields_in_schema": len(level_schema),
    }
    pain = _pain_order(levels, findings_by_level={lv.level: lv.findings for lv in levels}, by_key=by_key)
    return Synthesis(
        numbering=status,
        levels=levels,
        global_findings=g_findings,
        study_not_assessed=study_na,
        image_quality=_quality(anchors, cfg),
        coverage=coverage,
        pain_evidence=pain,
    )


def _analysis_of_global(key: str) -> str:
    for r in registered_global_fields():
        if f"{r.group}.{r.name}" == key:
            return r.analysis
    return ""  # pragma: no cover - keys come from the global registry only


def _group_order(f: Finding) -> int:
    order = field_order()
    return order.get(("level", f.group, f.name), 10**6)


def _quality(anchors: Anchors, cfg: Config) -> dict[str, Any]:
    q = quality_summary(anchors, cfg.quality_show_advisory)
    return {
        "assessed": bool(q.get("assessed")),
        "score": q.get("score"),
        "flagged": list(q.get("flagged", [])),
        "line": str(q.get("line", "")),
    }


def _pain_order(
    levels: list[LevelSynthesis],
    *,
    findings_by_level: Mapping[str, list[Finding]],
    by_key: Mapping[tuple[str, str, str], Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lv in levels:
        groups = _pain_entries(lv.level, findings_by_level[lv.level], by_key)
        n_abnormal = sum(len(v) for v in groups.values())
        rows.append(
            {
                "level": lv.level,
                "name_status": lv.name_status,
                "strength": lv.strength,
                "abnormal_count": n_abnormal,
                "groups": groups,
            }
        )
    rows.sort(key=lambda r: (-r["strength"], level_sort_key(r["level"])))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


# ------------------------------------------------------------------------------------- findings document
def findings_document(
    results: Mapping[str, Result],
    anchors: Anchors,
    figures: Mapping[str, Mapping[str, str]],
    *,
    reader: str = "findings",
    scope: str | None = None,
) -> dict[str, Any]:
    """The findings JSON (schema 1.0) for the automatic measurements, with ``figures`` mapping analysis -> figure ->
    saved image path (relative to the output folder). Fields without a saved figure are left out (not assessed)."""
    available()
    status = status_of(anchors)
    levels: dict[str, Any] = {}
    for lab in sorted(anchors.disc_labels(), key=level_sort_key):
        obj = level_findings(results, lab, anchors, figures=figures)
        obj.pop("_skipped", None)
        if obj:
            levels[lab] = obj
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reader": reader,
        "scope": scope
        or (
            "automatic measurements (not a reader): "
            + (", ".join(sorted(results)) or "no analyses")
            + f"; level names {status}"
        ),
    }
    if levels:
        doc["levels"] = levels
    g_objs, _ = _global_objects(results, status, figures)
    if g_objs:
        doc["global"] = g_objs
    return doc


__all__ = [
    "DISCLAIMER",
    "FORMULA",
    "PAIN_STRUCTURES",
    "REASON_TEXT",
    "Finding",
    "GlobalRule",
    "LevelSynthesis",
    "NotAssessed",
    "Synthesis",
    "build_synthesis",
    "finding_text",
    "findings_document",
    "is_abnormal",
    "register_global_field",
    "registered_global_fields",
]
