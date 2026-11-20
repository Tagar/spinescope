"""AI-reader surface for measured findings: validation status, reading sheets, and reader reviews.

Three things an AI reader needs, all computed from the findings registry (``analysis/findings.py``) so that a new
analysis or field appears here without code changes here:

* :func:`measured_findings`: the findings of every disc level, as the schema's level objects, each field with its
  confidence after the numbering cap, its validation status and the doc that states it.
* :func:`sheet_specs` and :func:`render_sheet`: one reading sheet per measured field (see ``render/reading_sheet.py``).
* A **review**: the reader confirms or rejects each measured field on its sheet. :func:`review_errors` validates the
  answer, and :func:`review_to_findings` turns it into an ordinary findings document, so :mod:`spinescope.report.merge`
  can merge it as one reader.

The trust rule (the same as the labelling review, ``anatomy/verify.py``): **an AI review can only lower trust.** A
confirm keeps the measured value and never raises its confidence; a reject or an unclear verdict lowers it. No AI answer
is a human confirmation, and nothing here sets a status to "confirmed by a human".

Research and education only. Nothing here is a diagnosis.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..analysis.base import Result
from ..analysis.findings import level_findings
from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..report.schema import CONFIDENCE_LEVELS, SCHEMA_VERSION, field_path
from ..report.validate import validate
from ..study import Study

log = logging.getLogger(__name__)

#: Verdicts a reviewer may give on one measured field.
REVIEW_VERDICTS: tuple[str, ...] = ("confirm", "reject", "unclear")

#: Words of the confidence scale, lowest first (the same order as the findings schema).
_RANK = {w: i for i, w in enumerate(CONFIDENCE_LEVELS)}

#: Shown for every field whose validation is not recorded here: an unrecorded field is experimental by default.
UNRECORDED = "no validation is recorded for this field: treat it as experimental"


@dataclass(frozen=True)
class FieldStatus:
    """Validation status of one findings field. ``status``: ``validated`` | ``partly_validated`` | ``experimental``.

    ``note`` says what was and was not supported, in words (no numbers: the numbers live in ``doc``, which is a page
    under ``docs/analysis/`` and is the place to check them).
    """

    status: str
    note: str
    doc: str


#: Validation status of the registered fields, keyed by (group, name). A field that is not listed is experimental.
#: Each entry points at the analysis page that holds the numbers and the pre-registered rule. Update it when a page
#: changes its verdict. Fields registered by another module list their status here too, so that the status reaches
#: the reader without editing the analysis module.
FIELD_STATUS: dict[tuple[str, str], FieldStatus] = {
    ("disc", "pfirrmann"): FieldStatus(
        "validated",
        "model suggestion on SPIDER validation (pre-registered acceptance met); the agreement interval is wide",
        "docs/analysis/discs.md",
    ),
    ("disc", "contour"): FieldStatus(
        "partly_validated",
        "bulge supported on SPIDER validation; protrusion and extrusion not supported",
        "docs/analysis/herniation.md",
    ),
    ("disc", "herniation_location"): FieldStatus(
        "experimental",
        "herniation rule not supported on SPIDER validation (too few positives)",
        "docs/analysis/herniation.md",
    ),
    ("disc", "herniation_size_mm"): FieldStatus(
        "experimental",
        "herniation rule not supported on SPIDER validation (too few positives)",
        "docs/analysis/herniation.md",
    ),
    ("endplate_above", "modic"): FieldStatus(
        "experimental",
        "Modic typing: agreement with radiologists on SPIDER validation is low; check T1 and T2",
        "docs/analysis/endplates.md",
    ),
    ("endplate_below", "modic"): FieldStatus(
        "experimental",
        "Modic typing: agreement with radiologists on SPIDER validation is low; check T1 and T2",
        "docs/analysis/endplates.md",
    ),
    ("listhesis", "direction"): FieldStatus(
        "experimental",
        "screening measure; the slip cut-off is a convention and is not validated against outcomes",
        "docs/analysis/alignment.md",
    ),
    ("listhesis", "translation_mm"): FieldStatus(
        "experimental",
        "screening measure; not validated against outcomes",
        "docs/analysis/alignment.md",
    ),
    ("listhesis", "meyerding_grade"): FieldStatus(
        "experimental",
        "screening measure; not validated against outcomes",
        "docs/analysis/alignment.md",
    ),
}

#: Policy shown to every reader next to the review: what an AI verdict can and cannot do.
REVIEW_POLICY = (
    "An AI review can only lower trust. confirm keeps the measured value and never raises its confidence; reject "
    "and unclear lower it. No AI verdict is a human confirmation, and the numbering status is never raised by a review."
)


def field_status(group: str, name: str) -> FieldStatus:
    """Validation status of a findings field (experimental when the field is not recorded)."""
    return FIELD_STATUS.get((group, name), FieldStatus("experimental", UNRECORDED, "docs/analysis/README.md"))


def measured_findings(results: Mapping[str, Result], anchors: Anchors) -> dict[str, dict[str, Any]]:
    """Findings of every disc level that has at least one measured field, as ``{level: {group: {name: field}}}``.

    Each field is the findings-schema object (value, confidence after the numbering cap, evidence, notes) plus
    ``status``, ``status_note`` and ``doc``. Evidence images are figure tokens (``analysis/figure``), not paths: the
    reading sheet of a field is what a reader looks at. Levels without a measured field are left out.
    """
    tokens = {name: {fig: f"{name}/{fig}" for fig in res.figures} for name, res in results.items()}
    out: dict[str, dict[str, Any]] = {}
    for lab in anchors.disc_labels():
        obj = level_findings(results, lab, anchors, figures=tokens)
        for group, body in obj.items():
            if group.startswith("_"):
                continue
            for name, fld in body.items():
                fs = field_status(group, name)
                fld["status"] = fs.status
                fld["status_note"] = fs.note
                fld["doc"] = fs.doc
        if any(not g.startswith("_") for g in obj):
            out[lab] = obj
    return out


def sheet_file_name(level: str, group: str, name: str) -> str:
    """File name of the reading sheet of one field, e.g. ``sheet_L4-L5_disc.pfirrmann.png``."""
    return f"sheet_{level}_{group}.{name}.png"


def measured_fields(measured: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """(level, group, name) of every measured field, superior to inferior, schema order within a level."""
    out: list[tuple[str, str, str]] = []
    for lab, obj in measured.items():
        for group, body in obj.items():
            if group.startswith("_"):
                continue
            for name in body:
                out.append((lab, group, name))
    return out


def render_sheet(
    study: Study,
    anchors: Anchors,
    measured: Mapping[str, Any],
    level: str,
    group: str,
    name: str,
    *,
    show_value: bool = True,
    config: Config = DEFAULT,
) -> NDArray[np.uint8]:
    """The reading sheet (RGB array) of one measured field. ``show_value=False`` gives a blind sheet (no number).

    Raises KeyError when the field is not measured for this level.
    """
    from ..render.reading_sheet import reading_sheet

    fld = measured[level][group][name]
    ev = (fld.get("evidence") or [{}])[0]
    slices = ev.get("slices") or []
    value_text = None
    if show_value:
        value_text = "no value" if fld["value"] is None else _fmt(fld["value"])
    return reading_sheet(
        study,
        anchors,
        level,
        field_label=f"{group}.{name}",
        primary=ev.get("series") or None,
        slice_k=int(slices[0]) if slices else None,
        value_text=value_text,
        confidence=fld.get("confidence") if show_value else None,
        validation=f"{fld['status']}: {fld['status_note']}",
        config=config,
    )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ----------------------------------------------------------------------- reviews


def review_paths(measured: Mapping[str, Any]) -> dict[str, tuple[str, str, str]]:
    """Canonical findings path -> (level, group, name) for every measured field."""
    return {field_path(lv, g, n): (lv, g, n) for lv, g, n in measured_fields(measured)}


def _rank_min(a: str, b: str) -> str:
    return a if _RANK[a] <= _RANK[b] else b


def review_to_findings(doc: Mapping[str, Any], measured: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a review (confirm / reject / unclear per measured field) into one findings document for the reader.

    * confirm: value = the measured value; confidence = the lower of the review's and the measured confidence.
    * reject: value = the reviewer's alternative when one is given (a disagreement that merge counts), otherwise null
      (indeterminate). Confidence as for confirm.
    * unclear: value null, confidence low.

    The result is an ordinary findings document, valid against the schema when the review is valid. It carries the
    reader name from the review. Call :func:`review_errors` first; this function does not check the review itself.
    """
    paths = review_paths(measured)
    levels: dict[str, dict[str, dict[str, Any]]] = {}
    for chk in doc.get("checks", []):
        lv, g, n = paths[chk["path"]]
        base = measured[lv][g][n]
        verdict = chk["verdict"]
        mconf = str(base["confidence"])
        if verdict == "confirm":
            value: Any = base["value"]
            conf = _rank_min(str(chk["confidence"]), mconf)
            note = "AI review: confirms the measured value"
        elif verdict == "reject":
            value = chk.get("value")
            conf = _rank_min(str(chk["confidence"]), mconf)
            note = "AI review: rejects the measured value" + (
                "; reviewer's alternative value given" if value is not None else "; no alternative value"
            )
        else:
            value = None
            conf = "low"
            note = "AI review: unclear"
        if chk.get("notes"):
            note += f"; {chk['notes']}"
        note += "; not a human confirmation"
        levels.setdefault(lv, {}).setdefault(g, {})[n] = {
            "value": value,
            "confidence": conf,
            "evidence": list(chk["evidence"]),
            "notes": note,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "reader": str(doc.get("reader", "")),
        "scope": "AI review of measured findings: confirm or reject only, cannot raise a confidence",
        "levels": levels,
    }


def review_errors(doc: Any, measured: Mapping[str, Any], *, status: str | None = None) -> list[str]:
    """Problems with a review document. Empty when it is valid and can be converted and merged.

    ``status`` is the numbering status of the study (``status_of(anchors)``). A review names its levels, so none is
    accepted while the numbering is unconfirmed (the abstain policy: positions only).

    Checks: the document shape; ``kind`` is ``review``; ``reader`` is a valid name; every check names one measured
    field exactly once; a verdict from :data:`REVIEW_VERDICTS`; a confidence word; at least one evidence entry; a
    ``value`` only with reject. The converted findings document is then validated against the findings schema.
    """
    from .briefs import READER_NAME_RE

    if not isinstance(doc, dict):
        return ["review must be a JSON object"]
    if status == "unconfirmed":
        return [
            "numbering is unconfirmed: positions only; verify the level labels before a review names levels"
        ]
    errs: list[str] = []
    if doc.get("kind") != "review":
        errs.append('"kind" must be "review"')
    reader = doc.get("reader")
    if not isinstance(reader, str) or not READER_NAME_RE.match(reader):
        errs.append("reader must be 1-64 characters from A-Z, a-z, 0-9, '_', '.', '-'")
    checks = doc.get("checks")
    if not isinstance(checks, list) or not checks:
        return [*errs, '"checks" must be a non-empty list']
    known = review_paths(measured)
    seen: set[str] = set()
    for i, chk in enumerate(checks):
        where = f"checks[{i}]"
        if not isinstance(chk, dict):
            errs.append(f"{where} must be an object")
            continue
        path = chk.get("path")
        if not isinstance(path, str) or path not in known:
            errs.append(
                f"{where}.path is not a measured field of this study (use the path from the review brief)"
            )
            continue
        if path in seen:
            errs.append(f"{where}.path {path} is reviewed twice")
        seen.add(path)
        if chk.get("verdict") not in REVIEW_VERDICTS:
            errs.append(f"{where}.verdict must be confirm, reject or unclear")
        if chk.get("confidence") not in CONFIDENCE_LEVELS:
            errs.append(f"{where}.confidence must be low, medium or high")
        ev = chk.get("evidence")
        if not isinstance(ev, list) or not ev:
            errs.append(f"{where}.evidence must be a non-empty list (the sheets you looked at)")
        if "value" in chk and chk.get("verdict") != "reject":
            errs.append(f"{where}.value is only allowed with verdict reject")
    if errs:
        return errs
    converted = review_to_findings(doc, measured)
    res = validate(converted)
    return [f"converted: {m}" for m in res.error_messages()]


def write_review_outputs(doc: Mapping[str, Any], measured: Mapping[str, Any], outdir: str | Path) -> Path:
    """Write the converted reader findings file (``<reader>.json``) into ``outdir`` and return its path."""
    import json

    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    converted = review_to_findings(doc, measured)
    path = d / f"{converted['reader']}.json"
    path.write_text(json.dumps(converted, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
