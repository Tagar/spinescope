"""Paste-ready instructions for an AI reader agent that fills the findings schema.

The field list is generated from the packaged schema, so the instructions cannot drift from it.
"""

from __future__ import annotations

import json
from typing import Any

from .schema import (
    ITEM_SECTIONS,
    SCHEMA_VERSION,
    FieldSpec,
    global_fields,
    item_fields,
    level_fields,
)

_RULES = """Rules
1. A missing key means NOT ASSESSED (you did not look at it). Never leave a field out and mean "normal".
2. value null means INDETERMINATE: you looked and could not decide. Use only the allowed values otherwise.
3. Every field is an object {"value", "confidence", "evidence", "notes"}. confidence is low, medium or high.
   Use low for single-slice, blurred or oblique-only calls.
4. evidence is required (at least one entry) for every field you fill, including null values. Each entry is
   one PNG you actually looked at: {"image": path, "series": stack key, "slices": [indices shown] or [] for
   the whole series}.
5. notes is one line of reasoning, or an empty string.
6. levels: keys are disc labels such as L4-L5, T12-L1 or L5-S1. Include only the levels you examined.
   Anything you did not examine: leave the key out.
7. Array sections (focal_lesions, incidental_findings, pain_generator_hypotheses): one object per finding with
   level (disc label, vertebra such as L4, or "global"), kind (short snake_case slug, reuse the most standard
   slug you can), summary (one sentence), confidence, evidence, notes and optional attributes. Each attribute
   is a field object like the ones above.
8. Pain generator entries are hypotheses, not diagnoses.
9. Do not write patient names, identifiers, dates, institutions or file paths of the input data into the file.
   Evidence image paths are the PNGs you produced for review.
10. Output JSON only, in one file named after your reader name. Validate it before you stop and fix every error:
    python -c "from spinescope.report import validate; r = validate('FILE.json'); print(r.ok); print(*r.error_messages(), sep='\\n')"
"""


def _allowed_text(spec: FieldSpec) -> str:
    if spec.allowed is not None:
        return "one of " + ", ".join(json.dumps(v) for v in spec.allowed) + ", or null"
    if spec.is_continuous:
        bounds = _bounds(spec)
        return "number" + (f" {bounds}" if bounds else "") + ", or null"
    if spec.pattern:
        return f"text matching {spec.pattern}, or null"
    if spec.minimum is not None or spec.maximum is not None:
        bounds = _bounds(spec)
        return "integer" + (f" {bounds}" if bounds else "") + ", or null"
    return "text, or null"


def _bounds(spec: FieldSpec) -> str:
    if spec.minimum is not None and spec.maximum is not None:
        return f"from {spec.minimum:g} to {spec.maximum:g}"
    if spec.minimum is not None:
        return f"at least {spec.minimum:g}"
    if spec.maximum is not None:
        return f"at most {spec.maximum:g}"
    return ""


def _example(reader: str) -> dict[str, Any]:
    """A minimal valid document showing the shape. Values are placeholders, not a read."""
    evidence = [{"image": "out/example/L4-L5_sag_t2.png", "series": "sag_t2", "slices": [12]}]
    level_field = next(f for f in level_fields() if f.group == "disc" and f.name == "pfirrmann")
    example_value: Any = 3 if level_field.kind == "categorical" else None
    body = {
        "disc": {
            "pfirrmann": {
                "value": example_value,
                "confidence": "medium",
                "evidence": evidence,
                "notes": "example only",
            },
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "reader": reader,
        "scope": "L4-L5 disc on sagittal T2, one example field",
        "levels": {"L4-L5": body},
        "focal_lesions": [],
    }


def reader_instructions(reader_name: str = "reader_a") -> str:
    """Return plain-text instructions for an AI reader agent to fill one findings file.

    Paste the whole string into the agent prompt. ``reader_name`` is the name the agent must use
    (lowercase letters, digits, ``_``, ``.`` or ``-``); it becomes the file name and the ``reader`` key.
    """
    lines = [
        "You are filling a structured findings file for one spine MRI study. This is research use: "
        "describe what you see, do not diagnose.",
        "",
        f"Reader name: {reader_name}. Write ONE JSON file named {reader_name}.json. JSON only, no prose outside it.",
        f'Top-level keys: "schema_version": "{SCHEMA_VERSION}", "reader", "scope" (required); "levels", '
        '"global", "focal_lesions", "incidental_findings", "pain_generator_hypotheses", "notes" (optional).',
        "",
        _RULES,
        "Per disc level (under levels, keyed by the disc label), group.field:",
    ]
    for spec in level_fields():
        lines.append(
            f'  levels["<LEVEL>"].{spec.group}.{spec.name}: {spec.title}. {spec.description} '
            f"Value: {_allowed_text(spec)}."
        )
    lines += ["", "Global sections (under global), group.field:"]
    for spec in global_fields():
        lines.append(
            f"  global.{spec.group}.{spec.name}: {spec.title}. {spec.description} "
            f"Value: {_allowed_text(spec)}."
        )
    lines += ["", "Array item attributes (each attribute is a field object):"]
    for section in ITEM_SECTIONS:
        lines.append(f"  {section}[].attributes:")
        for spec in item_fields(section):
            lines.append(f"    {spec.name}: {spec.title}. {spec.description} Value: {_allowed_text(spec)}.")
    lines += [
        "",
        "Example of the shape (placeholder values, not a read):",
        json.dumps(_example(reader_name), indent=2, ensure_ascii=False),
    ]
    return "\n".join(lines) + "\n"


#: The review answer: one verdict per measured field, converted to a findings file by ``ai.reading``.
_REVIEW_RULES = """Rules
1. One check per measured field listed in your scope. Do not add fields that are not listed. A field you do not check
   is left out, and then it is simply not reviewed. Never write a level name you were not given.
2. verdict: confirm = the reading sheet shows what the measured value says; reject = the sheet does not; unclear = you
   cannot tell from the sheet. Use unclear rather than guess.
3. reject: add "value" only when you can read a different value of the same field on the sheet, using the allowed
   values listed below. Without an alternative, leave "value" out.
4. confidence: how sure you are of your verdict, low, medium or high. Use low for single-slice, blurred or
   oblique-only sheets.
5. evidence: at least one entry per check, the sheet PNG you looked at: {"image": path, "series": stack key,
   "slices": [indices shown]}.
6. notes: one line of reasoning, or an empty string.
7. Trust rule: an AI review can only lower trust. A confirm does not raise a confidence, and nothing you write
   makes a finding confirmed by a human. Do not present your verdict as a clinical opinion.
8. Do not write patient names, identifiers, dates, institutions or input file paths into the file.
9. Output JSON only, in one file named after your reader name. Check it with the spinescope review command or the
   submit_reader_answer tool, and fix every error before you stop.
"""


def review_instructions(reader_name: str, level: str, fields: list[tuple[str, str]]) -> str:
    """Plain-text answer format for a reviewer of the measured findings of one disc level.

    ``fields`` are the (group, name) pairs being reviewed. The allowed values of each field come from the schema,
    so a reject alternative is checked the same way a reader's own value is.
    """
    specs = {(f.group, f.name): f for f in level_fields()}
    lines = [
        f"You are reviewing the measured findings of disc level {level} of one spine MRI study against their reading "
        "sheets. Research use: check each measurement on the image; do not diagnose.",
        "",
        f"Reader name: {reader_name}. Write ONE JSON file named {reader_name}.json.",
        f'Top-level keys: "kind": "review", "reader": "{reader_name}", "checks": [one object per field].',
        "Each check: path (the canonical path from the scope), verdict (confirm, reject or unclear), confidence, "
        "evidence (list), notes, and value only for reject.",
        "",
        _REVIEW_RULES,
        "Allowed alternative values for reject (only if you use value):",
    ]
    for group, name in fields:
        spec = specs.get((group, name))
        allowed = _allowed_text(spec) if spec is not None else "text, or null"
        lines.append(f"  {group}.{name}: {allowed}")
    example = {
        "kind": "review",
        "reader": reader_name,
        "checks": [
            {
                "path": f'levels["{level}"].{fields[0][0]}.{fields[0][1]}'
                if fields
                else f'levels["{level}"]',
                "verdict": "unclear",
                "confidence": "low",
                "evidence": [{"image": "out/sheets/example.png", "series": "sag_t2", "slices": [12]}],
                "notes": "example only",
            }
        ],
    }
    lines += ["", "Example of the shape (placeholder values, not a read):", json.dumps(example, indent=2)]
    return "\n".join(lines) + "\n"
