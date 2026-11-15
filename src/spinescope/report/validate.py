"""Validate findings documents against the packaged JSON Schema.

A small built-in validator covers the JSON Schema keywords the findings schema uses, so no extra
dependency is needed. If the optional ``jsonschema`` package is installed, its draft 2020-12 errors are
added as well. Errors carry JSON paths such as ``$.levels["L4-L5"].disc.pfirrmann.value``.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, load_schema, resolve

log = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "title",
        "description",
        "type",
        "enum",
        "const",
        "required",
        "properties",
        "patternProperties",
        "additionalProperties",
        "items",
        "minItems",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
    }
)


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a findings document."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Outcome of :func:`validate`. ``ok`` is True when there are no errors (warnings are allowed)."""

    ok: bool
    source: str
    reader: str | None
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    document: dict[str, Any] | None = None

    def error_messages(self) -> list[str]:
        return [str(e) for e in self.errors]


def json_path(parts: list[Any] | tuple[Any, ...]) -> str:
    """Format path parts as a JSON path rooted at ``$``."""
    out = "$"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        elif _IDENT_RE.match(str(p)):
            out += f".{p}"
        else:
            out += f"[{json.dumps(str(p))}]"
    return out


def _is_type(x: Any, t: str) -> bool:
    if t == "null":
        return x is None
    if t == "boolean":
        return isinstance(x, bool)
    if t == "integer":
        return isinstance(x, int) and not isinstance(x, bool)
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if t == "string":
        return isinstance(x, str)
    if t == "object":
        return isinstance(x, dict)
    if t == "array":
        return isinstance(x, list)
    return False


def _describe(x: Any) -> str:
    text = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else repr(x)
    return text if len(text) <= 60 else text[:57] + "..."


class _BuiltinChecker:
    """Validates an instance against the supported JSON Schema subset, collecting (path, message)."""

    def __init__(self, root: dict[str, Any]) -> None:
        self.root = root
        self.errors: list[tuple[list[Any], str]] = []

    def check(self, inst: Any, node: dict[str, Any], parts: list[Any]) -> None:
        node = resolve(self.root, node)
        if "type" in node:
            types = [node["type"]] if isinstance(node["type"], str) else list(node["type"])
            if not any(_is_type(inst, t) for t in types):
                want = " or ".join(types)
                self.errors.append((parts, f"expected {want}, got {_describe(inst)}"))
                return
        if "const" in node and inst != node["const"]:
            self.errors.append((parts, f"must be {_describe(node['const'])}, got {_describe(inst)}"))
        if "enum" in node and inst not in node["enum"]:
            allowed = [v for v in node["enum"] if v is not None]
            hint = ""
            if isinstance(inst, str):
                close = difflib.get_close_matches(inst, [str(a) for a in allowed], n=1)
                if close:
                    hint = f' (did you mean "{close[0]}"?)'
            self.errors.append(
                (parts, f"{_describe(inst)} is not an allowed value; allowed: {allowed} or null{hint}")
            )
        if _is_type(inst, "number"):
            if "minimum" in node and inst < node["minimum"]:
                self.errors.append((parts, f"{inst} is less than the minimum {node['minimum']}"))
            if "maximum" in node and inst > node["maximum"]:
                self.errors.append((parts, f"{inst} is greater than the maximum {node['maximum']}"))
        if isinstance(inst, str):
            if "minLength" in node and len(inst) < node["minLength"]:
                self.errors.append((parts, "must not be empty" if node["minLength"] == 1 else "too short"))
            if "maxLength" in node and len(inst) > node["maxLength"]:
                self.errors.append((parts, f"longer than {node['maxLength']} characters"))
            if "pattern" in node and not re.search(node["pattern"], inst):
                self.errors.append((parts, f"{_describe(inst)} does not match the pattern {node['pattern']}"))
        if isinstance(inst, dict):
            self._check_object(inst, node, parts)
        if isinstance(inst, list):
            if "minItems" in node and len(inst) < node["minItems"]:
                self.errors.append((parts, f"needs at least {node['minItems']} item(s), has {len(inst)}"))
            if "items" in node:
                for i, v in enumerate(inst):
                    self.check(v, node["items"], [*parts, i])

    def _check_object(self, inst: dict[str, Any], node: dict[str, Any], parts: list[Any]) -> None:
        for key in node.get("required", []):
            if key not in inst:
                self.errors.append((parts, f'missing required key "{key}"'))
        props: dict[str, Any] = node.get("properties", {})
        patterns: dict[str, Any] = node.get("patternProperties", {})
        extra = node.get("additionalProperties", True)
        for key, value in inst.items():
            matched = False
            if key in props:
                matched = True
                self.check(value, props[key], [*parts, key])
            for pat, sub in patterns.items():
                if re.search(pat, key):
                    matched = True
                    self.check(value, sub, [*parts, key])
            if matched:
                continue
            if extra is False:
                allowed = sorted(set(props) | {"<" + p + ">" for p in patterns})
                hint = difflib.get_close_matches(key, sorted(props), n=1)
                tail = f' (did you mean "{hint[0]}"?)' if hint else ""
                self.errors.append((parts, f'unexpected key "{key}"; allowed: {", ".join(allowed)}{tail}'))
            elif isinstance(extra, dict):
                self.check(value, extra, [*parts, key])


def _jsonschema_errors(doc: Any, schema: dict[str, Any]) -> list[tuple[str, str]]:
    """Errors from the optional jsonschema package, as (json path, message). Empty if not installed."""
    try:
        import jsonschema
    except ImportError:
        return []
    validator = jsonschema.Draft202012Validator(schema)
    out: list[tuple[str, str]] = []
    for err in validator.iter_errors(doc):
        out.append((json_path(list(err.absolute_path)), str(err.message)))
    return out


def _check_references(doc: dict[str, Any], base: Path | None, warnings: list[str]) -> None:
    """Non-fatal check: evidence images that cannot be found. One warning per image path."""
    cited: dict[str, list[str]] = {}
    for where, ev_list in _iter_evidence(doc):
        for ev in ev_list:
            image = ev.get("image")
            if isinstance(image, str):
                cited.setdefault(image, []).append(where)
    for image, places in sorted(cited.items()):
        candidates = [Path(image)] if Path(image).is_absolute() else [Path.cwd() / image]
        if base is not None and not Path(image).is_absolute():
            candidates.insert(0, base / image)
        if not any(c.exists() for c in candidates):
            warnings.append(
                f"evidence image not found: {image} (cited {len(places)} time(s), first at {places[0]})"
            )


def _iter_evidence(doc: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    out: list[tuple[str, list[dict[str, Any]]]] = []
    levels = doc.get("levels", {})
    if isinstance(levels, dict):
        for lvl, body in levels.items():
            if not isinstance(body, dict):
                continue
            for group, fields in body.items():
                if not isinstance(fields, dict):
                    continue
                for name, f in fields.items():
                    if isinstance(f, dict) and isinstance(f.get("evidence"), list):
                        out.append((json_path(["levels", lvl, group, name]), f["evidence"]))
    glob = doc.get("global", {})
    if isinstance(glob, dict):
        for group, fields in glob.items():
            if isinstance(fields, dict):
                for name, f in fields.items():
                    if isinstance(f, dict) and isinstance(f.get("evidence"), list):
                        out.append((json_path(["global", group, name]), f["evidence"]))
    for section in ("focal_lesions", "incidental_findings", "pain_generator_hypotheses"):
        items = doc.get(section, [])
        if not isinstance(items, list):
            continue
        for i, it in enumerate(items):
            if isinstance(it, dict) and isinstance(it.get("evidence"), list):
                out.append((json_path([section, i]), it["evidence"]))
    return out


def validate(source: str | Path | dict[str, Any], schema: dict[str, Any] | None = None) -> ValidationResult:
    """Validate a findings file (path) or an already loaded findings dict.

    Returns a :class:`ValidationResult`; it never raises for bad content. Missing evidence images are
    warnings, not errors. ``schema`` defaults to the packaged schema.
    """
    root = schema if schema is not None else load_schema()
    if isinstance(source, dict):
        doc: Any = source
        label = "<dict>"
        base: Path | None = None
        res = ValidationResult(ok=False, source=label, reader=None)
    else:
        path = Path(source)
        label = path.name
        base = path.resolve().parent
        res = ValidationResult(ok=False, source=label, reader=None)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            res.errors.append(ValidationIssue("$", f"cannot read JSON: {exc}"))
            return res

    res.document = doc if isinstance(doc, dict) else None
    if not isinstance(doc, dict):
        res.errors.append(ValidationIssue("$", "top level must be a JSON object"))
        return res

    if isinstance(doc.get("reader"), str):
        res.reader = doc["reader"]

    checker = _BuiltinChecker(root)
    checker.check(doc, root, [])
    seen: set[str] = set()
    for parts, msg in checker.errors:
        issue = ValidationIssue(json_path(parts), msg)
        if str(issue) not in seen:
            seen.add(str(issue))
            res.errors.append(issue)
    builtin_paths = {e.path for e in res.errors}
    for path_str, msg in _jsonschema_errors(doc, root):
        if path_str not in builtin_paths:
            res.errors.append(ValidationIssue(path_str, f"(jsonschema) {msg}"))

    if not res.errors:
        if doc.get("schema_version") != SCHEMA_VERSION:  # pragma: no cover - enforced by schema
            res.errors.append(ValidationIssue("$.schema_version", f"must be {SCHEMA_VERSION!r}"))
        if base is not None and res.reader and os.path.splitext(label)[0] != res.reader:
            res.warnings.append(f'file name {label} differs from reader "{res.reader}"')
        _check_references(doc, base, res.warnings)
    res.warnings = sorted(set(res.warnings))
    res.ok = not res.errors
    return res
