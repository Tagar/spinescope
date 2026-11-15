"""Validation status of every findings field: what the numbers on public data say about it.

Each field of the findings schema (level or global) can carry a :class:`Validation`: one of

* ``validated``: measured against public reference data with the metric quoted (``metric``);
* ``weak_labels``: checked only against weak labels (for example parsed free-text reports), never a radiologist grading;
* ``experimental``: implemented, but the pre-registered acceptance rule was not met, or the metric is too weak to use;
* ``not_validated``: no validation recorded for this field. This is the default for any field not registered here.

A status may depend on the value (a bulge call can be validated while a protrusion call on the same field is not);
``values`` gives the overrides. Statuses are sourced from the analysis documents: every record names the document
(``source``) and a ``quote`` that must appear verbatim in it (a test checks this), so a number cannot drift from its doc.

Analyses register their own fields next to ``register_field`` (``register_validation``); the built-in records below
cover the analyses in this build. Nothing here is a clinical threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .schema import global_fields, level_fields

VALIDATED = "validated"
WEAK_LABELS = "weak_labels"
EXPERIMENTAL = "experimental"
NOT_VALIDATED = "not_validated"

#: Statuses in order of decreasing support. The synthesis weights are in ``Config.synthesis_status_weights``.
STATUSES: tuple[str, ...] = (VALIDATED, WEAK_LABELS, EXPERIMENTAL, NOT_VALIDATED)

#: Short words for reports; the badge text is always shown (colour is never the only cue).
STATUS_WORDS: dict[str, str] = {
    VALIDATED: "validated on public data",
    WEAK_LABELS: "weak labels only",
    EXPERIMENTAL: "EXPERIMENTAL",
    NOT_VALIDATED: "NOT VALIDATED",
}


@dataclass(frozen=True)
class Validation:
    """Validation status of one field (or one value of a field)."""

    status: str
    metric: str = ""  # the number(s) that justify the status, as written in the source document
    source: str = ""  # repository path of the document, e.g. docs/analysis/discs.md
    quote: str = ""  # a substring of ``source`` that carries the metric (checked by the tests)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown validation status {self.status!r}; use one of {', '.join(STATUSES)}")

    @property
    def experimental(self) -> bool:
        """True for experimental and not-validated fields: shown apart from validated findings."""
        return self.status in (EXPERIMENTAL, NOT_VALIDATED)

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "metric": self.metric, "source": self.source}


NOT_REGISTERED = Validation(NOT_VALIDATED, metric="no validation recorded for this field")


@dataclass
class _Entry:
    default: Validation
    values: dict[str, Validation] = field(default_factory=dict)


_RECORDS: dict[tuple[str, str], _Entry] = {}


def register_validation(
    group: str,
    name: str,
    default: Validation,
    *,
    values: Mapping[str, Validation] | None = None,
) -> None:
    """Record the validation status of a findings field (``group.name``), optionally per value.

    The field must be a level or global field of the findings schema. Re-registering a field replaces its record.
    """
    known = {(f.group, f.name) for f in (*level_fields(), *global_fields())}
    if (group, name) not in known:
        raise ValueError(f"{group}.{name} is not a field of the findings schema")
    _RECORDS[(group, name)] = _Entry(default=default, values=dict(values or {}))


def validation_of(group: str, name: str, value: Any = None) -> Validation:
    """The status of ``group.name`` for ``value`` (the value override when one is registered), else not validated."""
    entry = _RECORDS.get((group, name))
    if entry is None:
        return NOT_REGISTERED
    if value is not None and str(value) in entry.values:
        return entry.values[str(value)]
    return entry.default


def registered_validation() -> dict[str, dict[str, Any]]:
    """All records as plain data, keyed by ``group.name`` (value overrides under ``values``). For docs and JSON."""
    out: dict[str, dict[str, Any]] = {}
    for (group, name), entry in sorted(_RECORDS.items()):
        out[f"{group}.{name}"] = {
            **entry.default.to_dict(),
            "values": {v: rec.to_dict() for v, rec in sorted(entry.values.items())},
        }
    return out


# ------------------------------------------------------------------------------ built-in records (this build)
_DISCS = "docs/analysis/discs.md"
_ENDPLATES = "docs/analysis/endplates.md"
_HERNIATION = "docs/analysis/herniation.md"
_ALIGNMENT = "docs/analysis/alignment.md"

_PFIRRMANN_METRIC = (
    "SPIDER validation (39 patients, 257 discs): quadratic kappa 0.58 [0.36, 0.75] against one radiologist; "
    "within one grade 0.89; accuracy 0.46. Confidence is not a calibrated probability."
)

register_validation(
    "disc",
    "pfirrmann",
    Validation(VALIDATED, _PFIRRMANN_METRIC, _DISCS, "0.58 [0.36, 0.75]"),
)

register_validation(
    "disc",
    "contour",
    Validation(
        EXPERIMENTAL,
        "sagittal contour only; the protrusion/extrusion rule was not supported on SPIDER validation",
        _HERNIATION,
        "Herniation: not supported",
    ),
    values={
        "normal": Validation(
            VALIDATED,
            "sagittal bulge rule (excursion 1 mm or more): SPIDER validation AUC 0.82 [0.76, 0.87], "
            "sensitivity 0.70, specificity 0.82",
            _HERNIATION,
            "0.82 [0.76, 0.87]",
        ),
        "bulge": Validation(
            VALIDATED,
            "sagittal bulge rule (excursion 1 mm or more): SPIDER validation AUC 0.82 [0.76, 0.87], "
            "sensitivity 0.70, specificity 0.82",
            _HERNIATION,
            "0.82 [0.76, 0.87]",
        ),
        "protrusion": Validation(
            EXPERIMENTAL,
            "focal call not supported: AUC 0.64 [0.38, 0.86] on 8 positive discs; check the axial images",
            _HERNIATION,
            "0.64 [0.38, 0.86]",
        ),
        "extrusion": Validation(
            EXPERIMENTAL,
            "focal call not supported: AUC 0.64 [0.38, 0.86] on 8 positive discs; check the axial images",
            _HERNIATION,
            "0.64 [0.38, 0.86]",
        ),
    },
)

register_validation(
    "disc",
    "herniation_location",
    Validation(
        EXPERIMENTAL,
        "sagittal only: the zone is central by construction; lateral and foraminal zones are not measured",
        _HERNIATION,
        "lateral zones are **not** measured",
    ),
)

register_validation(
    "disc",
    "herniation_size_mm",
    Validation(
        EXPERIMENTAL,
        "posterior excursion beyond the body-corner line (baseline-corrected); the protrusion rule was not supported",
        _HERNIATION,
        "Herniation: not supported",
    ),
)

_MODIC_METRIC = (
    "SPIDER validation (80 scored discs): Modic 0-3 kappa 0.14, accuracy 0.61; any-Modic specificity 0.74 "
    "(target 0.85); Modic 1 cannot be scored on SPIDER"
)
for _endplate in ("endplate_above", "endplate_below"):
    register_validation(
        _endplate,
        "modic",
        Validation(EXPERIMENTAL, _MODIC_METRIC, _ENDPLATES, "kappa 0.14"),
    )

register_validation(
    "listhesis",
    "translation_mm",
    Validation(
        VALIDATED,
        "geometric reference, not a radiologist: MAE 1.63 mm against SPIDER ground-truth vertebra masks "
        "(220 discs, validation)",
        _ALIGNMENT,
        "1.63 mm",
    ),
)
_LISTHESIS_METRIC = (
    "3 mm rule, SPIDER validation: 4 positive discs (sensitivity 1.00 [1.00, 1.00]), specificity 0.79 "
    "[0.74, 0.84], PPV 0.08; a measurement to review, not a flag"
)
register_validation(
    "listhesis",
    "direction",
    Validation(EXPERIMENTAL, _LISTHESIS_METRIC, _ALIGNMENT, "0.08 [0.02, 0.16]"),
)
register_validation(
    "listhesis",
    "meyerding_grade",
    Validation(EXPERIMENTAL, _LISTHESIS_METRIC, _ALIGNMENT, "0.08 [0.02, 0.16]"),
)

register_validation(
    "alignment",
    "lumbar_lordosis_deg",
    Validation(
        NOT_VALIDATED,
        "supine, single mid-sagittal slice per body; underestimates standing lordosis; no radiologist reference",
        _ALIGNMENT,
        "underestimates standing lordosis",
    ),
)


def all_validations() -> list[tuple[str, Validation]]:
    """Every registered status (defaults and value overrides) as (``group.name[=value]``, record). For tests and docs."""
    out: list[tuple[str, Validation]] = []
    for (group, name), entry in sorted(_RECORDS.items()):
        out.append((f"{group}.{name}", entry.default))
        out += [(f"{group}.{name}={v}", rec) for v, rec in sorted(entry.values.items())]
    return out
