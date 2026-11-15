"""Validation of the level correspondence on public data (not used by the labeller or the map).

* :func:`sudirman_slab_agreement`: for each annotated level of a Sudirman patient, does :func:`level_map` pick the
  axial slab that the dataset annotated for that level? Restricted to the annotated contrast (T1) by default, because
  the dataset's slabs are T1 and the same geometry is also present as T2 slabs.
* :func:`spider_graceful`: sagittal-only SPIDER studies: the map must build without error, have no slab, and say so.

Both return aggregate numbers only (no paths, no file names). The labeller is the default one (:func:`label_spine`).
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .correspondence import CorrespondenceConfig, level_map

log = logging.getLogger(__name__)


#: A labeller disc farther than this from the annotated centre is not the annotated disc (the scorer's MATCH_MM).
MATCH_MM = 12.0


@dataclass
class SlabRow:
    """One annotated level of one patient: the map's slab against the annotated one, in both variants."""

    patient: str
    level: str
    annotated: str  # annotated slab (stack key in the matched session)
    predicted: str | None  # the map's slab for the level (default config)
    predicted_strict: str | None  # the same with the angle-only rule (no position fallback)
    centre_err_mm: float | None  # labeller disc centre against the annotated IVD centroid
    reason: str  # agree | disc_mismatch | no_disc | no_slab | other_slab
    reason_strict: str


def _reason(lab_found: bool, err: float | None, predicted: str | None, annotated: str) -> str:
    if predicted == annotated:
        return "agree"
    if not lab_found:
        return "no_disc"
    if err is not None and err > MATCH_MM:
        return "disc_mismatch"
    if predicted is None:
        return "no_slab"
    return "other_slab"


def _slab_patient(args: tuple[str, list[str], str | None, str | None]) -> dict[str, Any]:
    """Worker: select the labelled session, label it once, build the map twice (default and strict), compare."""
    from pathlib import Path

    from ..datasets.sudirman import select_session
    from .pipeline import label_spine

    patient, sessions, labels_root, contrast = args
    out: dict[str, Any] = {"patient": patient, "rows": [], "error": None}
    try:
        sel = select_session(patient, [Path(s) for s in sessions], labels_root)
        if sel is None:
            out["error"] = "no session reproduces the label slices"
            return out
        _sess, study, levels = sel
        anchors = label_spine(study)
        slabs = (contrast,) if contrast else None
        default = level_map(
            study, anchors, CorrespondenceConfig(slab_contrasts=slabs), measure_geometry=False
        )
        strict = level_map(
            study,
            anchors,
            CorrespondenceConfig(slab_contrasts=slabs, slab_fallback_deg=CorrespondenceConfig().tilt_tol_deg),
            measure_geometry=False,
        )
        for lt in levels:
            lab = lt.level
            disc = anchors.discs.get(lab)
            err = None
            if disc is not None:
                err = round(float(np.linalg.norm(np.asarray(disc.center) - np.asarray(lt.center))), 2)
            pred = default.slab_for_level.get(lab) if disc is not None else None
            pred_strict = strict.slab_for_level.get(lab) if disc is not None else None
            out["rows"].append(
                asdict(
                    SlabRow(
                        patient=patient,
                        level=lab,
                        annotated=lt.stack_key,
                        predicted=pred,
                        predicted_strict=pred_strict,
                        centre_err_mm=err,
                        reason=_reason(disc is not None, err, pred, lt.stack_key),
                        reason_strict=_reason(disc is not None, err, pred_strict, lt.stack_key),
                    )
                )
            )
    except Exception as exc:  # one bad patient must not stop the run; the error is counted
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _summary(rows: list[dict[str, Any]], which: str) -> dict[str, Any]:
    """Agreement with the annotated slab; ``which`` is ``""`` (default) or ``"_strict"``."""
    n = len(rows)
    reasons = Counter(r["reason" + which] for r in rows)
    matched = [r for r in rows if r["centre_err_mm"] is not None and r["centre_err_mm"] <= MATCH_MM]
    m_agree = sum(r["reason" + which] == "agree" for r in matched)
    return {
        "levels": n,
        "slab_agreement": round(reasons["agree"] / n, 4) if n else None,
        "reasons": dict(reasons),
        "levels_with_labelled_disc_within_12mm": len(matched),
        "slab_agreement_given_labelled_disc": round(m_agree / len(matched), 4) if matched else None,
    }


def sudirman_slab_agreement(
    patients: list[str] | None = None,
    *,
    root: str | None = None,
    labels_root: str | None = None,
    workers: int | None = None,
    contrast: str | None = "t1",
) -> dict[str, Any]:
    """How often the map's slab for a level is the annotated slab. Two variants from the same labeller run:
    the default (with the position fallback, see :class:`CorrespondenceConfig`) and ``strict`` (angle-only).

    ``contrast`` restricts the candidate slabs (``"t1"``: the annotated contrast; ``None``: every slab). Runs in a
    spawn process pool (``workers`` defaults to the CPU count, capped at 8).
    """
    from multiprocessing import get_context

    from ..datasets.sudirman import labelled_patients, patient_sessions

    sessions = patient_sessions(root)
    ids = patients if patients is not None else labelled_patients(labels_root)
    jobs: list[tuple[str, list[str], str | None, str | None]] = []
    skipped = 0
    for pid in ids:
        if pid not in sessions:
            skipped += 1
            continue
        jobs.append((pid, [str(s) for s in sessions[pid]], labels_root, contrast))
    n = workers or min(8, os.cpu_count() or 1)
    if n <= 1 or len(jobs) <= 1:
        results = [_slab_patient(j) for j in jobs]
    else:
        with get_context("spawn").Pool(n) as pool:
            results = pool.map(_slab_patient, jobs, chunksize=4)
    rows = [r for res in results for r in res["rows"]]
    errors = Counter(res["error"] for res in results if res["error"])
    errs = [r["centre_err_mm"] for r in rows if r["centre_err_mm"] is not None]
    return {
        "contrast": contrast,
        "patients": len(jobs),
        "patients_skipped_no_dicom": skipped,
        "patients_failed": int(sum(errors.values())),
        "failures": dict(errors),
        "default": _summary(rows, ""),
        "strict_angle_only": _summary(rows, "_strict"),
        "labeller_centre_error_median_mm": round(float(np.median(errs)), 2) if errs else None,
        "labeller_centre_error_p90_mm": round(float(np.percentile(errs, 90)), 2) if errs else None,
        "by_level": {
            lab: {
                "levels": sum(1 for r in rows if r["level"] == lab),
                "agree": sum(1 for r in rows if r["level"] == lab and r["reason"] == "agree"),
            }
            for lab in sorted({r["level"] for r in rows})
        },
    }


def spider_graceful(
    *, root: str | None = None, subset: str = "validation", limit: int | None = None
) -> dict[str, Any]:
    """Run the labeller and the map on SPIDER sagittal studies; count failures and coverage. No axial exists here."""
    from ..datasets.spider import load_spider_studies
    from .pipeline import label_spine

    studies = [s for s in load_spider_studies(root, load=False) if not subset or s.subset == subset]
    if limit is not None:
        studies = studies[:limit]
    counts: Counter[str] = Counter()
    levels = 0
    with_sag = 0
    slab_any = 0
    quality: Counter[str] = Counter()
    failed = 0
    for sp in studies:
        try:
            study = sp.to_study(sequences=["t2"]) if "t2" in sp.sequences else sp.to_study()
            anchors = label_spine(study)
            lm = level_map(study, anchors, measure_geometry=False)
        except Exception as exc:  # the graceful run counts failures instead of stopping
            failed += 1
            counts[type(exc).__name__] += 1
            log.warning("SPIDER level map failed: %s", type(exc).__name__)
            continue
        if any(v is not None for v in lm.slab_for_level.values()):
            slab_any += 1
        for e in lm.discs.values():
            levels += 1
            quality[e.quality] += 1
            if any(h.in_slab and h.in_fov for k, h in e.stacks.items() if study[k].plane == "sag"):
                with_sag += 1
    n = len(studies) - failed
    return {
        "subset": subset,
        "studies": len(studies),
        "failed": failed,
        "failure_types": dict(counts),
        "studies_with_any_slab": slab_any,
        "levels": levels,
        "levels_inside_a_sagittal": with_sag,
        "levels_inside_a_sagittal_share": round(with_sag / levels, 4) if levels else None,
        "quality": dict(quality),
        "studies_ok": n,
    }
