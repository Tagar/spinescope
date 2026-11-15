"""Findings panel data: analysis flags and opted-in rows -> the viewer's JSON (generic, PHI-free).

The panel is driven by the analysis output, not by a list of analyses. For every registered analysis ``name``
with a :class:`~spinescope.analysis.base.Result` (or its ``to_dict()`` form, as written by ``spinescope analyze``):

* every **flag** is a finding. Its level is the first of ``level``, ``adjacent_disc``, ``disc``, ``vertebra``;
  its kind is ``kind`` (or ``finding``); its value is ``value``, ``grade``, ``class``, ``pattern``, ``contour`` or
  ``direction``, whichever is present. ``confidence`` is a number in [0, 1] or one of the words high/medium/low.
* a **row** is a finding only when it carries a ``finding`` key (the analysis opts in; rows are otherwise
  measurements, far too many to list).
* ``structure`` (optional, on a flag or row) groups findings for the filter; the default is the analysis name.
* ``position`` (or ``center``): LPS mm of the finding, if the analysis knows it. Otherwise the position is the centre
  of the level from the anchors (a disc or a vertebra centre). A finding with no position is listed but cannot be
  jumped to.
* ``experimental: true`` on a flag, or the analysis' ``meta["status"] == "experimental"`` or
  ``meta["<prefix>_status"] == "experimental"`` for a kind starting with ``<prefix>``, marks it experimental.
* ``validation`` (text) on a flag, or ``meta["validation"]`` (text, or a dict keyed by kind), says how the finding was
  validated. Absent means "not reported".

Only the keys above are copied, scalars are checked and text is cut to a short length, so nothing else from a
results file (a path, an extra key) can reach the page. Free text here is written by the analysis code, not read
from DICOM; the canary test (``tests/test_privacy_canaries.py``) scans this surface as well.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..analysis.base import Result
from ..analysis.findings import confidence_word
from ..anatomy.anchors import Anchors
from ..anatomy.summary import status_of

log = logging.getLogger(__name__)

#: Format tag of the ``findings`` section of the viewer payload.
FINDINGS_FORMAT = "spinescope-findings-view-1"
#: Upper bound on listed findings (a runaway analysis must not make a huge page).
MAX_ITEMS = 2000
_TEXT_MAX = 160
_NOTE_MAX = 400
_LEVEL_KEYS = ("level", "adjacent_disc", "disc", "vertebra")
_VALUE_KEYS = ("value", "grade", "class", "pattern", "contour", "direction")
_WORDS = {"high": 0.85, "medium": 0.55, "low": 0.25}
_CONF_WORDS = frozenset(_WORDS)


def _text(x: Any, limit: int = _TEXT_MAX) -> str:
    """Single-line text of ``limit`` characters at most (control characters dropped)."""
    s = " ".join(str(x).split())
    return s[:limit]


def _scalar(v: Any) -> Any:
    """A JSON scalar for a finding value: bool, finite number or short text; anything else is None."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return round(v, 3) if math.isfinite(v) else None
    if isinstance(v, str):
        return _text(v)
    return None


def _vec3(v: Any) -> list[float] | None:
    """``[x, y, z]`` as finite floats (LPS mm), or None."""
    try:
        seq = [float(c) for c in v]
    except (TypeError, ValueError):
        return None
    if len(seq) != 3 or not all(math.isfinite(c) for c in seq):
        return None
    return [round(c, 3) for c in seq]


def _is_experimental(kind: str, rec: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    if rec.get("experimental") is True:
        return True
    if meta.get("status") == "experimental":
        return True
    for key, val in meta.items():
        if isinstance(key, str) and key.endswith("_status") and val == "experimental":
            prefix = key[: -len("_status")]
            if prefix and kind.startswith(prefix):
                return True
    return False


def _validation(kind: str, rec: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    v = rec.get("validation")
    if v is None:
        mv = meta.get("validation")
        v = mv.get(kind) if isinstance(mv, Mapping) else mv
    return _text(v) if isinstance(v, str) else ""


def _level_of(rec: Mapping[str, Any]) -> str:
    """The level name without a status suffix: analyses may write ``"L2-L3 (unconfirmed)"``; the status is shown
    from the anchors, not from the name."""
    for key in _LEVEL_KEYS:
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return _text(v.split(" (")[0], 24)
    return ""


def _confidence(rec: Mapping[str, Any]) -> tuple[float | None, str]:
    c = rec.get("confidence")
    if isinstance(c, bool):
        return None, ""
    if isinstance(c, (int, float)) and math.isfinite(float(c)):
        f = min(1.0, max(0.0, float(c)))
        return round(f, 3), confidence_word(f)
    if isinstance(c, str) and c.strip().lower() in _CONF_WORDS:
        return None, c.strip().lower()
    return None, ""


def _as_dict(res: Result | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(res, Result):
        return res.to_dict()
    return dict(res)


def _anchor_position(anchors: Anchors | None, level: str) -> list[float] | None:
    if anchors is None or not level:
        return None
    if level in anchors.discs:
        return _vec3(anchors.discs[level].center)
    if level in anchors.vertebrae:
        return _vec3(anchors.vertebrae[level].center)
    return None


def _item(
    name: str,
    meta: Mapping[str, Any],
    rec: Mapping[str, Any],
    anchors: Anchors | None,
    structure_default: str,
) -> dict[str, Any]:
    level = _level_of(rec)
    kind = _text(rec.get("kind") or rec.get("finding") or name, 60)
    value: Any = None
    for key in _VALUE_KEYS:
        if key in rec:
            value = _scalar(rec[key])
            break
    conf, word = _confidence(rec)
    pos = _vec3(rec.get("position")) or _vec3(rec.get("center"))
    pos_source = "analysis" if pos else ""
    if pos is None:
        pos = _anchor_position(anchors, level)
        pos_source = "anchors" if pos else ""
    structure = rec.get("structure")
    side = rec.get("side")
    note = rec.get("note", rec.get("notes", ""))
    return {
        "level": level,
        "side": _text(side, 12) if isinstance(side, str) else "",
        "structure": _text(structure, 40) if isinstance(structure, str) and structure else structure_default,
        "analysis": _text(name, 40),
        "kind": kind,
        "value": value,
        "confidence": conf,
        "confidence_word": word,
        "experimental": _is_experimental(kind, rec, meta),
        "validation": _validation(kind, rec, meta),
        "notes": _text(note, _NOTE_MAX) if isinstance(note, str) else "",
        "pos": pos,
        "pos_source": pos_source,
    }


def collect(
    results: Mapping[str, Result | Mapping[str, Any]],
    anchors: Anchors | None = None,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """The viewer's ``findings`` section for analysis outputs.

    ``results`` maps analysis name to a :class:`Result` or its ``to_dict()`` form. Items are ordered by level
    (superior first, as the anchors order them; unplaced levels last), then analysis, then kind. The section is
    JSON-able and carries no keys beyond those documented in this module.
    """
    order = {lab: i for i, lab in enumerate(anchors.disc_labels())} if anchors is not None else {}
    items: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    seq = {name: i for i, name in enumerate(results)}
    for name, raw in results.items():
        doc = _as_dict(raw)
        raw_meta = doc.get("meta")
        meta: Mapping[str, Any] = raw_meta if isinstance(raw_meta, Mapping) else {}
        n_here = 0
        for flag in doc.get("flags") or []:
            if isinstance(flag, Mapping):
                items.append(_item(name, meta, flag, anchors, structure_default=_text(name, 40)))
                n_here += 1
        for row in doc.get("rows") or []:
            if isinstance(row, Mapping) and "finding" in row:
                items.append(_item(name, meta, row, anchors, structure_default=_text(name, 40)))
                n_here += 1
        summary.append(
            {
                "name": _text(name, 40),
                "findings": n_here,
                "warnings": [_text(w) for w in (doc.get("warnings") or []) if isinstance(w, str)][:20],
            }
        )
    items.sort(
        key=lambda it: (
            order.get(it["level"], len(order) + 1),
            it["level"],
            seq.get(it["analysis"], len(seq)),
            it["kind"],
            it["value"] is None,
            str(it["value"]),
        )
    )
    dropped = max(0, len(items) - MAX_ITEMS)
    items = items[:MAX_ITEMS]
    for i, it in enumerate(items):
        it["id"] = i
    notes = list(warnings or [])
    if dropped:
        notes.append(f"{dropped} findings not shown (limit {MAX_ITEMS})")
    return {
        "format": FINDINGS_FORMAT,
        "numbering": status_of(anchors) if anchors is not None else "",
        "structures": sorted({it["structure"] for it in items}),
        "kinds": sorted({it["kind"] for it in items}),
        "items": items,
        "analyses": summary,
        "warnings": [_text(w, _NOTE_MAX) for w in notes],
    }


def results_from_analyze_json(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The analysis results in the JSON that ``spinescope analyze`` prints (its ``analyses`` object)."""
    analyses = doc.get("analyses")
    if not isinstance(analyses, Mapping):
        raise ValueError(
            "not an analyze output: expected an 'analyses' object (spinescope analyze --out ...)"
        )
    return {str(name): dict(entry) for name, entry in analyses.items() if isinstance(entry, Mapping)}


def load_findings_file(
    path: str | Path, anchors: Anchors | None
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Read an ``analyze`` JSON file. The levels it was made for are compared with ``anchors`` (positions come
    from the current anchors), and a difference is returned as a note instead of failing."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise ValueError("findings file must hold a JSON object")
    results = results_from_analyze_json(doc)
    notes: list[str] = []
    levels = doc.get("levels")
    if anchors is not None and isinstance(levels, list):
        here = anchors.disc_labels()
        if sorted(map(str, levels)) != sorted(here):
            msg = (
                "findings file was made for other levels than these labels; positions use the current labels"
            )
            log.warning("viewer: %s", msg)
            notes.append(msg)
    return results, notes
