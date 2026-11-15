"""Built-in ground truth and predictors for :mod:`spinescope.analysis.evaluate` (SPIDER gradings).

Imported lazily by the harness (never import it directly from library code). It declares the SPIDER truth fields,
connects the existing analyses to them, and registers the SPIDER case loader.

Predictor rules (fixed before any validation number was looked at; see ``docs/analysis/README.md``):

* ``pfirrmann`` (discs): the disc's Pfirrmann suggestion (``pfirrmann_suggestion``).
* ``disc_narrowing`` (discs): positive when the disc has the analysis' ``low_disc_height`` flag (height ratio < 0.6,
  the threshold already in ``discs.py``); score is minus the height ratio, for AUC.
* ``modic`` (endplates): the highest Modic type over the two endplates of the disc (type from the endplate pattern:
  ``none`` -> 0, ``Modic 1`` and ``Modic 1-like`` -> 1, ``Modic 2`` -> 2, ``Modic 3`` -> 3). 0 only when both endplates
  are known to be normal; unknown when either is unknown and none is abnormal.
* ``up_endplate`` / ``low_endplate`` (endplates): abnormal (1) when the endplate pattern is any Modic-like pattern,
  normal (0) when ``none``; unknown otherwise. UP is the inferior endplate of the body above the disc, LOW the superior
  endplate of the body below it. These are a proxy: the SPIDER endplate codes are not defined here (see gradings).

SPIDER fields with no analysis (``disc_herniation``, ``disc_bulging``, ``spondylolisthesis``) are reported as not
assessed, with their counts.
"""

from __future__ import annotations

import math
from typing import Any

from ..analysis.base import Result
from ..analysis.endplates import MODIC_PATTERNS
from ..anatomy.anchors import Anchors
from .evaluate import (
    CaseSpec,
    Predictor,
    TruthField,
    TruthRef,
    register_loader,
    register_predictor,
    register_truth_field,
)

_MODIC_TYPE = {"none": 0, "Modic 1": 1, "Modic 1-like": 1, "Modic 2": 2, "Modic 3": 3}

register_truth_field(TruthField("pfirrmann", "ordinal", (1, 2, 3, 4, 5), "Pfirrmann grade 1-5 (SPIDER)"))
register_truth_field(TruthField("modic", "categorical", (0, 1, 2, 3), "Modic type 0-3 (SPIDER, per disc)"))
register_truth_field(
    TruthField("up_endplate", "binary", (), "UP endplate code 0/1 (SPIDER; semantics unverified)")
)
register_truth_field(
    TruthField("low_endplate", "binary", (), "LOW endplate code 0/1 (SPIDER; semantics unverified)")
)
register_truth_field(TruthField("disc_narrowing", "binary", (), "Disc narrowing code 0/1 (SPIDER)"))
register_truth_field(TruthField("disc_herniation", "binary", (), "Disc herniation code 0/1 (SPIDER)"))
register_truth_field(TruthField("disc_bulging", "binary", (), "Disc bulging code 0/1 (SPIDER)"))
register_truth_field(TruthField("spondylolisthesis", "binary", (), "Spondylolisthesis code 0/1 (SPIDER)"))


def _disc_rows(res: Result) -> dict[str, dict[str, Any]]:
    return {str(r.get("disc")): r for r in res.rows if r.get("scope") == "disc"}


def _endplate_rows(res: Result, disc: str) -> list[dict[str, Any]]:
    return [
        r
        for r in res.rows
        if r.get("scope") == "endplate" and r.get("side") == "all" and r.get("adjacent_disc") == disc
    ]


def _pfirrmann(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
    rows = _disc_rows(res)
    out: dict[str, tuple[Any, float | None]] = {}
    for lab in anchors.discs:
        r = rows.get(lab)
        grade = r.get("pfirrmann_suggestion") if r else None
        out[lab] = (int(grade) if grade else None, None)
    return out


def _disc_narrowing(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
    rows = _disc_rows(res)
    flagged = {str(f.get("level")) for f in res.flags if f.get("kind") == "low_disc_height"}
    out: dict[str, tuple[Any, float | None]] = {}
    for lab in anchors.discs:
        r = rows.get(lab)
        raw = r.get("height_ratio") if r else None
        ratio = float(raw) if raw is not None else float("nan")
        if not math.isfinite(ratio):
            out[lab] = (None, None)
            continue
        out[lab] = (1 if lab in flagged else 0, -ratio)
    return out


def _modic(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
    out: dict[str, tuple[Any, float | None]] = {}
    for lab in anchors.discs:
        types: list[int] = []
        unknown = 0
        for r in _endplate_rows(res, lab):
            pat = str(r.get("pattern"))
            if pat in _MODIC_TYPE:
                types.append(_MODIC_TYPE[pat])
            else:
                unknown += 1
        if types and max(types) >= 1:
            out[lab] = (max(types), None)
        elif len(types) == 2 and unknown == 0:
            out[lab] = (0, None)
        else:
            out[lab] = (None, None)
    return out


def _endplate(endplate: str) -> Predictor:
    def fn(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
        out: dict[str, tuple[Any, float | None]] = {}
        for lab in anchors.discs:
            rows = [r for r in _endplate_rows(res, lab) if r.get("endplate") == endplate]
            pat = str(rows[0].get("pattern")) if rows else ""
            if pat in MODIC_PATTERNS:
                out[lab] = (1, None)
            elif pat == "none":
                out[lab] = (0, None)
            else:
                out[lab] = (None, None)
        return out

    return fn


register_predictor("discs", "pfirrmann", _pfirrmann)
register_predictor("discs", "disc_narrowing", _disc_narrowing)
register_predictor("endplates", "modic", _modic)
# "inferior" endplate of the body above a disc is its UP endplate; "superior" of the body below is its LOW endplate.
register_predictor("endplates", "up_endplate", _endplate("inferior"))
register_predictor("endplates", "low_endplate", _endplate("superior"))


def _load_spider(spec: CaseSpec) -> tuple[Any, list[TruthRef], dict[str, str], bool]:
    """One SPIDER patient: a study of the requested sequences and the disc truths from the reference mask."""
    from ..anatomy._validate import truth_from_mask
    from ..datasets.gradings import GRADING_FIELDS, load_spider_file_meta, load_spider_gradings
    from ..datasets.spider import load_spider_studies

    sds = load_spider_studies(spec.root, cases=[spec.case])
    if not sds:
        raise FileNotFoundError("case not found in the SPIDER images folder")
    sd = sds[0]
    seqs = [s for s in (spec.sequences or tuple(sd.sequences)) if s in sd.image_paths]
    if not seqs:
        raise ValueError("no requested sequence for this case")
    ref = "t2" if "t2" in seqs and "t2" in sd.mask_paths else next(s for s in seqs if s in sd.mask_paths)
    gradings = {g.ivd_label: g for g in load_spider_gradings(spec.root).get(spec.case, [])}
    metas = load_spider_file_meta(spec.root)
    meta = metas.get((spec.case, ref)) or next((m for (c, _s), m in metas.items() if c == spec.case), None)
    truth: list[TruthRef] = []
    discs, _canal = truth_from_mask(str(sd.mask_paths[ref]))
    for d in discs:
        ivd = int(d.label) - 200
        g = gradings.get(ivd)
        if g is None or g.level is None:
            continue
        truth.append(
            TruthRef(
                name=g.level,
                center=(float(d.center[0]), float(d.center[1]), float(d.center[2])),
                values={f: g.value(f) for f in GRADING_FIELDS},
            )
        )
    study = sd.to_study(seqs)
    nv = meta.num_vertebrae if meta else None
    groups = {
        "vendor": meta.vendor if meta else "unknown",
        "field_strength": (meta.field_strength or "unknown") if meta else "unknown",
        "sequences": "+".join(seqs),
        "transitional_suspect": "yes" if nv is not None and nv != 5 else "no",
    }
    return study, truth, groups, meta is not None


register_loader("spider", _load_spider)
