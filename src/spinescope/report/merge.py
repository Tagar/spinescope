"""Merge several readers' findings into one consensus, with agreement counts and disagreements.

Rules (all explicit in the output):

* A field a reader did not report is **not assessed** by that reader. It is never counted as normal.
* A field reported with ``value: null`` is **indeterminate** for that reader. Indeterminate votes are
  listed (``indeterminate_by``) but do not count as disagreement against a definite value, unless
  ``indeterminate_is_disagreement=True``.
* Categorical fields (enums, integer grades, text) take the weighted plurality of definite values. A
  value with more than half of the total weight is a **majority**; a unique top value otherwise is a
  **plurality**; a tie at the top is a **split** with no consensus value.
* Continuous fields (JSON numbers such as mm or degrees) take the weighted median. They are flagged as
  disagreeing when their spread exceeds ``findings_numeric_rel_tol`` of the consensus value.
* Array items (focal lesions, incidental findings, pain generator hypotheses) are grouped by disc level
  or vertebra and by a fuzzily matched kind slug, then their attributes are merged like fields.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import DEFAULT, Config
from .schema import (
    CONFIDENCE_LEVELS,
    GLOBAL,
    ITEM_SECTIONS,
    field_path,
    global_fields,
    item_fields,
    level_fields,
    level_sort_key,
    normalise_kind,
)
from .validate import validate

log = logging.getLogger(__name__)

Source = str | Path | dict[str, Any]


@dataclass
class Consensus:
    """Merged view of several findings documents. All content is plain JSON-compatible data.

    ``fields`` maps a canonical path (see :func:`spinescope.report.schema.field_path`) to a field
    consensus record. ``items`` maps each array section to a list of item groups.
    """

    readers: list[dict[str, Any]] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    indeterminate: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def reader_names(self) -> list[str]:
        return [r["reader"] for r in self.readers]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path | None = None, indent: int = 1) -> str:
        """Serialise to JSON; optionally also write it to ``path``."""
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Consensus:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> Consensus:
        return cls.from_dict(json.loads(text))


@dataclass
class _Entry:
    reader: str
    doc: dict[str, Any]
    label: str


def _collect(sources: Source | list[Source] | tuple[Source, ...], consensus: Consensus) -> list[_Entry]:
    """Load and validate sources. Invalid ones go to ``consensus.rejected``, non-findings JSON to skipped."""
    if isinstance(sources, (str, Path)):
        p = Path(sources)
        items: list[Source] = [f for f in sorted(p.glob("*.json"))] if p.is_dir() else [p]
    elif isinstance(sources, dict):
        items = [sources]
    else:
        items = list(sources)

    entries: list[_Entry] = []
    seen: set[str] = set()
    for i, src in enumerate(items):
        if isinstance(src, dict):
            label = f"dict[{i}]"
            if "schema_version" not in src:
                consensus.skipped.append(label)
                continue
            res = validate(src)
        else:
            path = Path(src)
            label = f"file {i + 1}"  # never the file name: it may carry identifying text
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                consensus.rejected.append({"source": label, "errors": [f"cannot read JSON: {exc}"]})
                continue
            if not isinstance(raw, dict) or "schema_version" not in raw:
                consensus.skipped.append(label)
                continue
            res = validate(path)
        consensus.warnings += [f"{label}: {w}" for w in res.warnings]
        if not res.ok or res.document is None:
            consensus.rejected.append({"source": label, "errors": res.error_messages()})
            continue
        reader = str(res.document["reader"])
        if reader in seen:
            consensus.rejected.append({"source": label, "errors": [f'duplicate reader name "{reader}"']})
            continue
        seen.add(reader)
        entries.append(_Entry(reader=reader, doc=res.document, label=label))
    entries.sort(key=lambda e: e.reader)
    return entries


def _weights_for(entries: list[_Entry], weights: dict[str, float] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in entries:
        w = 1.0 if weights is None else float(weights.get(e.reader, 1.0))
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"weight for reader {e.reader!r} must be a finite number >= 0, got {w}")
        out[e.reader] = w
    return out


def _round(x: float) -> float:
    return round(float(x), 4)


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    """Weighted median of (value, weight) pairs. Falls back to an unweighted median if weights sum to 0."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        pairs = [(v, 1.0) for v, _ in pairs]
        total = float(len(pairs))
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc > total / 2 + 1e-12:
            return v
        if abs(acc - total / 2) <= 1e-12:
            return (v + pairs[i + 1][0]) / 2 if i + 1 < len(pairs) else v
    return pairs[-1][0]  # pragma: no cover - loop always returns


def _evidence_union(per_reader: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Union of evidence across readers, keeping which readers cited each image/slice selection."""
    out: dict[tuple[str, str, tuple[int, ...]], dict[str, Any]] = {}
    for reader in sorted(per_reader):
        for ev in per_reader[reader].get("evidence", []):
            key = (ev["image"], ev["series"], tuple(ev["slices"]))
            rec = out.setdefault(
                key,
                {"image": ev["image"], "series": ev["series"], "slices": list(ev["slices"]), "readers": []},
            )
            rec["readers"].append(reader)
    return list(out.values())


def _field_consensus(
    *,
    votes: dict[str, dict[str, Any]],
    is_continuous: bool,
    weights: dict[str, float],
    use_confidence: bool,
    indeterminate_is_disagreement: bool,
    not_assessed_by: list[str],
    config: Config,
) -> dict[str, Any]:
    """Consensus for one field from {reader: field object}. Pure function; see module docstring."""
    if not votes:
        return {
            "status": "not_assessed",
            "value": None,
            "n_reporting": 0,
            "n_definite": 0,
            "agreement": 0,
            "disagreement": False,
            "indeterminate_by": [],
            "not_assessed_by": not_assessed_by,
            "distribution": [],
            "range": None,
            "votes": {},
            "evidence": [],
        }

    rw: dict[str, float] = {}
    for reader, f in votes.items():
        w = weights[reader]
        if use_confidence:
            w *= config.findings_confidence_weights[CONFIDENCE_LEVELS.index(f["confidence"])]
        rw[reader] = w

    definite = {r: f["value"] for r, f in votes.items() if f["value"] is not None}
    indet = sorted(r for r, f in votes.items() if f["value"] is None)
    dist: list[dict[str, Any]] = []
    value: Any = None
    agreement = 0
    rng: list[float] | None = None
    disagree = False

    if not definite:
        status = "indeterminate"
    elif is_continuous:
        pairs = sorted((float(v), rw[r]) for r, v in definite.items())
        med = _weighted_median([(v, w) for v, w in pairs])
        lo, hi = pairs[0][0], pairs[-1][0]
        tol = config.findings_numeric_rel_tol * max(abs(med), 1.0)
        disagree = (hi - lo) > tol
        agreement = sum(1 for v, _ in pairs if abs(v - med) <= tol)
        status = "single" if len(definite) == 1 else "numeric"
        value = med
        rng = [lo, hi]
        dist = [{"value": v, "readers": [r], "weight": _round(rw[r])} for r, v in sorted(definite.items())]
    else:
        groups: dict[str, dict[str, Any]] = {}
        for r in sorted(definite):
            key = json.dumps(definite[r], sort_keys=True)
            g = groups.setdefault(key, {"value": definite[r], "readers": [], "weight": 0.0})
            g["readers"].append(r)
            g["weight"] += rw[r]
        dist = list(groups.values())
        disagree = len(dist) > 1
        winner: dict[str, Any] | None
        if len(dist) == 1:
            winner = dist[0]
            status = "single" if len(definite) == 1 else "unanimous"
        else:
            ranked = sorted(dist, key=lambda g: (-g["weight"], -len(g["readers"]), json.dumps(g["value"])))
            total = sum(g["weight"] for g in dist)
            if abs(ranked[0]["weight"] - ranked[1]["weight"]) <= 1e-12:
                winner = None
                status = "split"
            else:
                winner = ranked[0]
                status = "majority" if winner["weight"] > total / 2 + 1e-12 else "plurality"
        if winner is not None:
            value = winner["value"]
            agreement = len(winner["readers"])
        for g in dist:
            g["weight"] = _round(g["weight"])

    if indeterminate_is_disagreement and indet and definite:
        disagree = True

    return {
        "status": status,
        "value": value,
        "n_reporting": len(votes),
        "n_definite": len(definite),
        "agreement": agreement,
        "disagreement": disagree,
        "indeterminate_by": indet,
        "not_assessed_by": not_assessed_by,
        "distribution": dist,
        "range": rng,
        "votes": {
            r: {
                "value": f["value"],
                "confidence": f["confidence"],
                "notes": f["notes"],
                "weight": _round(rw[r]),
            }
            for r, f in sorted(votes.items())
        },
        "evidence": _evidence_union(votes),
    }


def _merge_fields(
    entries: list[_Entry],
    weights: dict[str, float],
    consensus: Consensus,
    *,
    use_confidence: bool,
    indeterminate_is_disagreement: bool,
    config: Config,
) -> None:
    """Fill ``consensus.fields`` for every level any reader covered, plus the global sections."""
    covering: dict[str, set[str]] = {}
    per_path: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for e in entries:
        for lvl, body in e.doc.get("levels", {}).items():
            covering.setdefault(lvl, set()).add(e.reader)
            for group, fields in body.items():
                for name, f in fields.items():
                    per_path.setdefault((lvl, group, name), {})[e.reader] = f
        if "global" in e.doc:
            covering.setdefault(GLOBAL, set()).add(e.reader)
            for group, fields in e.doc["global"].items():
                for name, f in fields.items():
                    per_path.setdefault((GLOBAL, group, name), {})[e.reader] = f

    consensus.levels = sorted((lvl for lvl in covering if lvl != GLOBAL), key=level_sort_key)
    scopes = [(lvl, level_fields()) for lvl in consensus.levels]
    if GLOBAL in covering:
        scopes.append((GLOBAL, global_fields()))
    for lvl, specs in scopes:
        for spec in specs:
            votes = per_path.get((lvl, spec.group, spec.name), {})
            rec = _field_consensus(
                votes=votes,
                is_continuous=spec.is_continuous,
                weights=weights,
                use_confidence=use_confidence,
                indeterminate_is_disagreement=indeterminate_is_disagreement,
                not_assessed_by=sorted(covering[lvl] - set(votes)),
                config=config,
            )
            path = field_path(lvl, spec.group, spec.name)
            rec.update(
                path=path,
                level=lvl,
                group=spec.group,
                field=spec.name,
                title=spec.label,
                where=f"{lvl}: {spec.label}",
                n_covering=len(covering[lvl]),
            )
            consensus.fields[path] = rec


def _cluster_items(entries: list[_Entry], section: str, min_similarity: float) -> list[dict[str, Any]]:
    """Group one array section by (level, kind). Kinds are matched fuzzily within the same level."""
    clusters: list[dict[str, Any]] = []  # each: {"level", "kinds": set, "members": [(reader, item)]}
    for e in entries:
        for item in e.doc.get(section, []):
            lvl = str(item["level"])
            nk = normalise_kind(str(item["kind"]))
            best: dict[str, Any] | None = None
            best_ratio = 0.0
            for c in clusters:
                if c["level"] != lvl:
                    continue
                ratio = max(difflib.SequenceMatcher(None, k, nk).ratio() for k in c["kinds"])
                if ratio >= min_similarity and ratio > best_ratio:
                    best, best_ratio = c, ratio
            if best is None:
                best = {"level": lvl, "kinds": set(), "members": []}
                clusters.append(best)
            best["kinds"].add(nk)
            best["members"].append((e.reader, item))
    clusters.sort(key=lambda c: (level_sort_key(c["level"]), min(c["kinds"])))
    return clusters


def _merge_items(
    entries: list[_Entry],
    weights: dict[str, float],
    consensus: Consensus,
    *,
    use_confidence: bool,
    indeterminate_is_disagreement: bool,
    config: Config,
) -> None:
    """Fill ``consensus.items`` with one group per (level, kind) cluster per array section."""
    n_readers = len(entries)
    for section in ITEM_SECTIONS:
        groups: list[dict[str, Any]] = []
        specs = item_fields(section)
        for c in _cluster_items(entries, section, config.findings_kind_min_similarity):
            members: list[tuple[str, dict[str, Any]]] = c["members"]
            kinds = Counter(str(it["kind"]) for _, it in members)
            canonical = sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            reporters = sorted({r for r, _ in members})
            attrs: dict[str, dict[str, Any]] = {}
            for spec in specs:
                votes = {
                    r: it["attributes"][spec.name]
                    for r, it in members
                    if spec.name in it.get("attributes", {})
                }
                if not votes:
                    continue
                rec = _field_consensus(
                    votes=votes,
                    is_continuous=spec.is_continuous,
                    weights=weights,
                    use_confidence=use_confidence,
                    indeterminate_is_disagreement=indeterminate_is_disagreement,
                    not_assessed_by=sorted({r for r, _ in members} - set(votes)),
                    config=config,
                )
                rec.update(name=spec.name, title=spec.label)
                attrs[spec.name] = rec
            per_reader_item = {r: it for r, it in members}
            evidence = _evidence_union({r: {"evidence": it["evidence"]} for r, it in per_reader_item.items()})
            groups.append(
                {
                    "section": section,
                    "level": c["level"],
                    "kind": canonical,
                    "kind_variants": sorted(kinds) if len(kinds) > 1 else [],
                    "reported_by": reporters,
                    "n_readers": n_readers,
                    "partial": 0 < len(reporters) < n_readers,
                    "summaries": {r: it["summary"] for r, it in sorted(per_reader_item.items())},
                    "confidence": {r: it["confidence"] for r, it in sorted(per_reader_item.items())},
                    "notes": {r: it["notes"] for r, it in sorted(per_reader_item.items())},
                    "kind_by": {r: it["kind"] for r, it in sorted(per_reader_item.items())},
                    "attributes": attrs,
                    "disagreement": any(a["disagreement"] for a in attrs.values()),
                    "evidence": evidence,
                }
            )
        consensus.items[section] = groups


def _collect_findings(consensus: Consensus) -> None:
    """Derive disagreement and indeterminate tables from the merged fields and items."""
    for rec in consensus.fields.values():
        if rec["disagreement"]:
            consensus.disagreements.append(
                {
                    "kind": "field",
                    "path": rec["path"],
                    "where": rec["where"],
                    "status": rec["status"],
                    "value": rec["value"],
                    "distribution": rec["distribution"],
                    "readers": sorted(rec["votes"]),
                }
            )
        if rec["indeterminate_by"]:
            consensus.indeterminate.append(
                {
                    "kind": "field",
                    "path": rec["path"],
                    "where": rec["where"],
                    "value": rec["value"],
                    "indeterminate_by": rec["indeterminate_by"],
                    "n_definite": rec["n_definite"],
                }
            )
    for section, groups in consensus.items.items():
        for g in groups:
            for name, a in g["attributes"].items():
                path = f"{section}[{g['level']}|{g['kind']}].{name}"
                where = f"{section} {g['level']} / {g['kind']}: {a['title']}"
                if a["disagreement"]:
                    consensus.disagreements.append(
                        {
                            "kind": "item_attribute",
                            "path": path,
                            "where": where,
                            "status": a["status"],
                            "value": a["value"],
                            "distribution": a["distribution"],
                            "readers": sorted(a["votes"]),
                        }
                    )
                if a["indeterminate_by"]:
                    consensus.indeterminate.append(
                        {
                            "kind": "item_attribute",
                            "path": path,
                            "where": where,
                            "value": a["value"],
                            "indeterminate_by": a["indeterminate_by"],
                            "n_definite": a["n_definite"],
                        }
                    )


def merge(
    sources: Source | list[Source] | tuple[Source, ...],
    weights: dict[str, float] | None = None,
    *,
    use_confidence: bool = False,
    indeterminate_is_disagreement: bool = False,
    config: Config | None = None,
) -> Consensus:
    """Merge findings from several readers into a :class:`Consensus`.

    ``sources`` is a directory (every ``*.json`` in it), one file, a findings dict, or a list of
    files and/or dicts. Invalid documents are listed under ``rejected`` and left out; the merge
    continues with the rest. ``weights`` maps reader name to a non-negative weight (default 1 each).
    ``use_confidence=True`` multiplies each vote by its self-reported confidence weight
    (``Config.findings_confidence_weights``). ``indeterminate_is_disagreement=True`` makes a null
    vote count as disagreement against a definite value; by default it does not.
    """
    cfg = config or DEFAULT
    consensus = Consensus()
    entries = _collect(sources, consensus)
    w = _weights_for(entries, weights)
    consensus.readers = [
        {
            "reader": e.reader,
            "source": e.label,
            "scope": str(e.doc.get("scope", "")),
            "weight": _round(w[e.reader]),
        }
        for e in entries
    ]
    _merge_fields(
        entries,
        w,
        consensus,
        use_confidence=use_confidence,
        indeterminate_is_disagreement=indeterminate_is_disagreement,
        config=cfg,
    )
    _merge_items(
        entries,
        w,
        consensus,
        use_confidence=use_confidence,
        indeterminate_is_disagreement=indeterminate_is_disagreement,
        config=cfg,
    )
    _collect_findings(consensus)
    consensus.warnings = sorted(set(consensus.warnings))
    consensus.settings = {
        "weights": {r: _round(v) for r, v in sorted(w.items())},
        "use_confidence": use_confidence,
        "indeterminate_is_disagreement": indeterminate_is_disagreement,
        "numeric_rel_tol": cfg.findings_numeric_rel_tol,
        "kind_min_similarity": cfg.findings_kind_min_similarity,
        "confidence_weights": list(cfg.findings_confidence_weights),
    }
    return consensus
