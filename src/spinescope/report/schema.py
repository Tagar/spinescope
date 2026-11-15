"""Findings JSON Schema: loaded from package data and turned into a field catalogue.

The schema file ``spinescope/data/findings.schema.json`` is the single source of truth. Validation,
merging, rendering and the reader instructions all read it through the helpers here.
"""

from __future__ import annotations

import copy
import functools
import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

#: Version written into and required from every findings document.
SCHEMA_VERSION = "1.0"

#: Allowed values of every ``confidence`` key, lowest first.
CONFIDENCE_LEVELS: tuple[str, ...] = ("low", "medium", "high")

#: Array sections of a findings document (one object per finding).
ITEM_SECTIONS: tuple[str, ...] = ("focal_lesions", "incidental_findings", "pain_generator_hypotheses")

#: Pseudo-level used for global sections and for items that are not tied to a level.
GLOBAL = "global"

_DISC_RE = re.compile(r"^(C|T|L)\d{1,2}-(C|T|L|S)\d{1,2}$")
_LABEL_RE = re.compile(r"^(C|T|L|S)(\d{1,2})")
_RANK = {"C": 0, "T": 1, "L": 2, "S": 3}


@functools.lru_cache(maxsize=1)
def _raw_schema() -> dict[str, Any]:
    text = (resources.files("spinescope") / "data" / "findings.schema.json").read_text(encoding="utf-8")
    loaded = json.loads(text)
    if not isinstance(loaded, dict):  # pragma: no cover - guards a corrupted install
        raise ValueError("findings.schema.json must contain a JSON object")
    return loaded


def load_schema() -> dict[str, Any]:
    """Return a fresh copy of the findings JSON Schema (draft 2020-12) from package data."""
    return copy.deepcopy(_raw_schema())


def is_disc_label(label: str) -> bool:
    """True for labels like ``L4-L5``, ``T12-L1`` or ``L5-S1``."""
    return bool(_DISC_RE.match(label))


def resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Follow ``#/$defs/Name`` references (one level at a time) until a concrete node is reached."""
    seen = 0
    while "$ref" in node:
        ref = str(node["$ref"])
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"unsupported $ref {ref!r}")
        node = schema["$defs"][ref[len("#/$defs/") :]]
        seen += 1
        if seen > 20:  # pragma: no cover - guards a cyclic schema
            raise ValueError("cyclic $ref chain")
    return node


def level_sort_key(label: str) -> tuple[int, int, str]:
    """Superior-to-inferior ordering for disc labels and vertebra labels; ``global`` sorts last."""
    if label == GLOBAL:
        return (9, 0, label)
    m = _LABEL_RE.match(label)
    if m is None:
        return (8, 0, label)
    return (_RANK[m.group(1)], int(m.group(2)), label)


def normalise_kind(kind: str) -> str:
    """Lower-case snake_case form of an item kind slug, used for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "_", kind.lower()).strip("_")


@dataclass(frozen=True)
class FieldSpec:
    """One leaf field of the schema (a level field, a global field or an item attribute)."""

    scope: str  # "level", "global" or "item"
    group: str  # e.g. "disc", "nerve_root_R", or the item section for items
    name: str  # e.g. "pfirrmann", or the attribute name for items
    title: str
    description: str
    kind: str  # "categorical" (enum / integer / text), "continuous" (JSON number)
    allowed: tuple[Any, ...] | None  # enum values without null, or None when unconstrained
    minimum: float | None
    maximum: float | None
    pattern: str | None

    @property
    def label(self) -> str:
        """Title with the side for paired groups, e.g. ``Foramen Lee grade (right)``."""
        if self.group.endswith("_R"):
            return f"{self.title} (right)"
        if self.group.endswith("_L"):
            return f"{self.title} (left)"
        return self.title

    @property
    def is_continuous(self) -> bool:
        return self.kind == "continuous"


def _spec_from_field(
    schema: dict[str, Any], field_node: dict[str, Any], scope: str, group: str, name: str
) -> FieldSpec:
    value = resolve(schema, field_node["properties"]["value"])
    types = value.get("type", [])
    types = [types] if isinstance(types, str) else list(types)
    allowed = None
    if "enum" in value:
        allowed = tuple(v for v in value["enum"] if v is not None)
    kind = "categorical"
    if "number" in types and "integer" not in types:
        kind = "continuous"
    return FieldSpec(
        scope=scope,
        group=group,
        name=name,
        title=str(field_node.get("title", name)),
        description=str(field_node.get("description", "")),
        kind=kind,
        allowed=allowed,
        minimum=value.get("minimum"),
        maximum=value.get("maximum"),
        pattern=value.get("pattern"),
    )


def _fields_of_group(
    schema: dict[str, Any], group_node: dict[str, Any], scope: str, group: str
) -> list[FieldSpec]:
    out: list[FieldSpec] = []
    for name, node in group_node["properties"].items():
        out.append(_spec_from_field(schema, resolve(schema, node), scope, group, name))
    return out


@functools.lru_cache(maxsize=1)
def _catalogue() -> tuple[tuple[FieldSpec, ...], dict[str, tuple[FieldSpec, ...]]]:
    schema = _raw_schema()
    level_fields: list[FieldSpec] = []
    for group, node in schema["$defs"]["Level"]["properties"].items():
        level_fields += _fields_of_group(schema, resolve(schema, node), "level", group)
    global_fields: list[FieldSpec] = []
    for group, node in schema["$defs"]["Global"]["properties"].items():
        global_fields += _fields_of_group(schema, resolve(schema, node), "global", group)
    items: dict[str, tuple[FieldSpec, ...]] = {}
    for section in ITEM_SECTIONS:
        item_def = resolve(schema, schema["properties"][section]["items"])
        attrs = item_def["properties"]["attributes"]["properties"]
        items[section] = tuple(
            _spec_from_field(schema, resolve(schema, node), "item", section, name)
            for name, node in attrs.items()
        )
    return tuple(level_fields + global_fields), items


def level_fields() -> tuple[FieldSpec, ...]:
    """Fields of one disc level, in schema order (``scope == "level"``)."""
    return tuple(f for f in _catalogue()[0] if f.scope == "level")


def global_fields() -> tuple[FieldSpec, ...]:
    """Fields of the global sections, in schema order."""
    return tuple(f for f in _catalogue()[0] if f.scope == "global")


def item_fields(section: str) -> tuple[FieldSpec, ...]:
    """Attributes of one array section (focal_lesions, incidental_findings, ...)."""
    return _catalogue()[1][section]


def field_order() -> dict[tuple[str, str, str], int]:
    """Stable position of every (scope, group, name) in the schema, for deterministic output ordering."""
    return {(f.scope, f.group, f.name): i for i, f in enumerate(_catalogue()[0])}


def level_group_names() -> tuple[str, ...]:
    """Group names under one disc level, in schema order (disc, central_canal, ...)."""
    schema = _raw_schema()
    return tuple(schema["$defs"]["Level"]["properties"].keys())


def global_group_names() -> tuple[str, ...]:
    """Group names of the global sections, in schema order."""
    schema = _raw_schema()
    return tuple(schema["$defs"]["Global"]["properties"].keys())


def field_path(level: str, group: str, name: str) -> str:
    """Canonical path of a field in a findings document, e.g. ``levels["L4-L5"].disc.pfirrmann``."""
    if level == GLOBAL:
        return f"global.{group}.{name}"
    return f'levels["{level}"].{group}.{name}'
