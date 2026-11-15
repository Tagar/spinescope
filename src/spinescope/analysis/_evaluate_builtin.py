"""Built-in ground truth and predictors for :mod:`spinescope.analysis.evaluate` (SPIDER gradings).

Imported lazily by the harness (never import it directly from library code). It declares the SPIDER truth fields,
connects the existing analyses to them, and registers the SPIDER case loader.

Predictor rules (fixed before any validation number was looked at; see ``docs/analysis/README.md``):

* ``pfirrmann`` (discs): the disc's Pfirrmann suggestion (``pfirrmann_suggestion``).
* ``disc_narrowing`` (discs): positive when the disc height index ``dhi`` is below the SPIDER-training threshold
  (``disc_narrowing_dhi_threshold``, Youden's J on training; see ``docs/analysis/discs.md``); score is minus ``dhi``.
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
from ..config import DEFAULT
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
    TruthField("modic_any", "binary", (), "Any Modic change: SPIDER Modic >= 1 (derived from the Modic code)")
)
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
    out: dict[str, tuple[Any, float | None]] = {}
    for lab in anchors.discs:
        r = rows.get(lab)
        call = r.get("disc_narrowing_suggestion") if r else None
        raw = r.get("dhi") if r else None
        dhi = float(raw) if raw is not None else float("nan")
        if call is None or not math.isfinite(dhi):
            out[lab] = (None, None)
            continue
        out[lab] = (int(call), -dhi)
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


def _modic_any(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
    """1 when an endplate shows a typed Modic change OR a presence-only change; 0 when both endplates are normal."""
    out: dict[str, tuple[Any, float | None]] = {}
    for lab in anchors.discs:
        pats = [str(r.get("pattern")) for r in _endplate_rows(res, lab)]
        if any(p in MODIC_PATTERNS or p.startswith("presence only") for p in pats):
            out[lab] = (1, None)
        elif len(pats) == 2 and all(p == "none" for p in pats):
            out[lab] = (0, None)
        else:
            out[lab] = (None, None)
    return out


register_predictor("discs", "pfirrmann", _pfirrmann)
register_predictor("discs", "disc_narrowing", _disc_narrowing)
register_predictor("endplates", "modic", _modic)
register_predictor("endplates", "modic_any", _modic_any)


def _endplate_defect(endplate: str) -> Predictor:
    """Endplate contour defect proxy: concavity (mm) against ``config.endplate_defect_mm`` (see endplates.md)."""

    def fn(res: Result, anchors: Anchors) -> dict[str, tuple[Any, float | None]]:
        thr = DEFAULT.endplate_defect_mm
        out: dict[str, tuple[Any, float | None]] = {}
        for lab in anchors.discs:
            rows = [r for r in _endplate_rows(res, lab) if r.get("endplate") == endplate]
            raw = rows[0].get("concavity_mm") if rows else None
            conc = float(raw) if raw is not None else float("nan")
            if not math.isfinite(conc):
                out[lab] = (None, None)
            else:
                out[lab] = (1 if conc >= thr else 0, conc)
        return out

    return fn


# UP / LOW: endplate contour defect (concavity), replacing the Modic-type proxy (that proxy measured marrow signal,
# not the endplate; kept in the baseline report for comparison).
register_predictor("endplates", "up_endplate", _endplate_defect("inferior"))
register_predictor("endplates", "low_endplate", _endplate_defect("superior"))


def _modic_any_truth(modic: Any) -> int | None:
    """Derived truth: 1 for a Modic code >= 1, 0 for code 0, unknown otherwise."""
    if modic is None:
        return None
    return 1 if int(modic) >= 1 else 0


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
                values={
                    **{f: g.value(f) for f in GRADING_FIELDS},
                    "modic_any": _modic_any_truth(g.value("modic")),
                },
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
