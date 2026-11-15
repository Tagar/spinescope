"""Scorer adapter for the Sudirman lumbar DICOM dataset: disc level names from the dataset's own labels.

Not used by the labeller. Ground truth is the three annotated discs of each labelled patient (L3-L4,
L4-L5, L5-S1; see :mod:`spinescope.datasets.sudirman`). Only those discs are annotated, so the
shared scorer's precision is structurally low here (the labeller also reports unannotated discs);
the level-wise scorer :func:`score_levels` is the primary measure: for each annotated level, the
predicted disc nearest in the sagittal plane must carry the right name and lie close to the truth.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets.sudirman import LevelTruth, select_session
from ._validate import TruthDisc, VolumeScore, aggregate, score_anchors
from .anchors import Anchors, Disc

log = logging.getLogger(__name__)

#: In-plane (sagittal y, z) distance within which a predicted disc counts as the annotated one.
MATCH_MM = 12.0
#: SPIDER-style label values for the annotated discs (201 = lowest, as in the SPIDER masks).
_SPIDER_LABEL = {"L5-S1": 201, "L4-L5": 202, "L3-L4": 203}


@dataclass
class LevelRow:
    """Score of one annotated level against the predicted disc nearest to it."""

    patient: str
    dlabel: str
    level: str
    predicted: str | None  # name of the nearest predicted disc (any distance), None if no disc was found
    dist_mm: float | None  # in-plane (y, z) distance to that disc
    dy_mm: float | None
    dz_mm: float | None
    matched: bool  # dist_mm <= MATCH_MM
    name_ok: bool  # predicted == level (regardless of distance)
    normal_err_deg: float | None  # angle between predicted and slab normal, only when matched
    n_axial_slabs: int = 0


@dataclass
class PatientResult:
    patient: str
    rows: list[LevelRow] = field(default_factory=list)
    shared: VolumeScore | None = None
    n_predicted: int = 0
    n_axial_slabs: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def truth_discs(levels: list[LevelTruth]) -> list[TruthDisc]:
    """The annotated discs as :class:`TruthDisc` (for the shared scorer in ``_validate``)."""
    return [
        TruthDisc(
            label=_SPIDER_LABEL.get(t.level, 0),
            name=t.level,
            center=np.asarray(t.center, dtype=np.float64),
            normal=np.asarray(t.normal, dtype=np.float64),
        )
        for t in levels
    ]


def score_levels(anchors: Anchors, levels: list[LevelTruth], *, match_mm: float = MATCH_MM) -> list[LevelRow]:
    """For each annotated level, the nearest predicted disc in the sagittal plane and whether its name is right."""
    rows: list[LevelRow] = []
    preds = list(anchors.discs.items())
    for t in levels:
        tc = np.asarray(t.center, dtype=np.float64)
        best_name: str | None = None
        best_disc: Disc | None = None
        best_d = float("inf")
        for name, d in preds:
            c = np.asarray(d.center, dtype=np.float64)
            dist = float(np.hypot(c[1] - tc[1], c[2] - tc[2]))  # in-plane (y, z)
            if dist < best_d:
                best_name, best_disc, best_d = name, d, dist
        if best_disc is None or best_name is None:
            rows.append(
                LevelRow(
                    patient=t.patient,
                    dlabel=t.dlabel,
                    level=t.level,
                    predicted=None,
                    dist_mm=None,
                    dy_mm=None,
                    dz_mm=None,
                    matched=False,
                    name_ok=False,
                    normal_err_deg=None,
                )
            )
            continue
        c = np.asarray(best_disc.center, dtype=np.float64)
        matched = best_d <= match_mm
        normal_err = None
        if matched:
            n = np.asarray(best_disc.normal, dtype=np.float64)
            normal_err = float(np.degrees(np.arccos(np.clip(abs(n @ t.normal), -1.0, 1.0))))
        rows.append(
            LevelRow(
                patient=t.patient,
                dlabel=t.dlabel,
                level=t.level,
                predicted=best_name,
                dist_mm=best_d,
                dy_mm=float(c[1] - tc[1]),
                dz_mm=float(c[2] - tc[2]),
                matched=matched,
                name_ok=best_name == t.level,
                normal_err_deg=normal_err,
            )
        )
    return rows


def summarise_levels(rows: list[LevelRow]) -> dict[str, Any]:
    """Pooled level-wise metrics, plus per-level and per-slab-count breakdowns.

    ``recall``: annotated levels with a predicted disc within ``MATCH_MM``.
    ``label_accuracy``: among those, the nearest disc has the right name.
    ``label_accuracy_any_distance``: the nearest predicted disc has the right name, distance ignored.
    """
    n = len(rows)
    if n == 0:
        return {"levels": 0}
    m = [r for r in rows if r.matched]
    dist = [float(r.dist_mm) for r in m if r.dist_mm is not None]
    dz = [abs(float(r.dz_mm)) for r in m if r.dz_mm is not None]
    dy = [abs(float(r.dy_mm)) for r in m if r.dy_mm is not None]
    ne = [r.normal_err_deg for r in m if r.normal_err_deg is not None]
    out: dict[str, Any] = {
        "levels": n,
        "patients": len({r.patient for r in rows}),
        "recall": len(m) / n,
        "label_accuracy": sum(r.name_ok for r in m) / max(len(m), 1),
        "label_accuracy_any_distance": sum(r.name_ok for r in rows) / n,
        "centre_err_median_mm": _med(dist),
        "centre_err_p90_mm": _pct(dist, 90),
        "dz_median_mm": _med(dz),
        "dz_p90_mm": _pct(dz, 90),
        "dy_median_mm": _med(dy),
        "normal_err_median_deg": _med(ne),
        "by_level": {},
        "by_axial_slabs": {},
    }
    for lev in sorted({r.level for r in rows}):
        sub = [r for r in rows if r.level == lev]
        sm = [r for r in sub if r.matched]
        out["by_level"][lev] = {
            "levels": len(sub),
            "recall": len(sm) / len(sub),
            "label_accuracy": sum(r.name_ok for r in sm) / max(len(sm), 1),
            "label_accuracy_any_distance": sum(r.name_ok for r in sub) / len(sub),
            "centre_err_median_mm": _med([float(r.dist_mm) for r in sm if r.dist_mm is not None]),
        }
    for k in sorted({r.n_axial_slabs for r in rows}):
        sub = [r for r in rows if r.n_axial_slabs == k]
        sm = [r for r in sub if r.matched]
        out["by_axial_slabs"][str(k)] = {
            "levels": len(sub),
            "recall": len(sm) / len(sub),
            "label_accuracy": sum(r.name_ok for r in sm) / max(len(sm), 1),
            "label_accuracy_any_distance": sum(r.name_ok for r in sub) / len(sub),
        }
    return out


def _med(x: list[float]) -> float:
    return float(np.median(x)) if x else float("nan")


def _pct(x: list[float], q: float) -> float:
    return float(np.percentile(x, q)) if x else float("nan")


def _evaluate_patient(args: tuple[str, list[str], str | None, int | None]) -> PatientResult:
    """Worker: pick the labelled session, run :func:`label_spine`, score it against the annotated discs."""
    from ..anatomy.pipeline import label_spine
    from ..config import Config

    patient, sessions, labels_root, threads = args
    res = PatientResult(patient=patient)
    try:
        sel = select_session(patient, [Path(s) for s in sessions], labels_root)
        if sel is None:
            res.error = "no session reproduces the label slices"
            return res
        _sess, study, levels = sel
        res.n_axial_slabs = sum(1 for k in study.stacks if k.startswith("ax_t1"))
        cfg = Config(ml_intra_op_threads=threads) if threads is not None else None
        anchors = label_spine(study, cfg)
        res.n_predicted = len(anchors.discs)
        res.warnings = list(anchors.warnings)
        rows = score_levels(anchors, levels)
        for r in rows:
            r.n_axial_slabs = res.n_axial_slabs
        res.rows = rows
        res.shared = score_anchors(anchors, truth_discs(levels), None)
    except Exception as exc:  # one bad study must not stop the run; the error is reported per patient
        res.error = f"{type(exc).__name__}: {exc}"
    return res


@dataclass
class Evaluation:
    results: list[PatientResult]

    @property
    def rows(self) -> list[LevelRow]:
        return [r for p in self.results for r in p.rows]

    @property
    def failures(self) -> dict[str, str]:
        return {p.patient: p.error for p in self.results if p.error}

    def summary(self) -> dict[str, Any]:
        """Level-wise summary plus the shared scorer's pooled numbers (comparable with SPIDER)."""
        shared = [p.shared for p in self.results if p.shared is not None]
        out = summarise_levels(self.rows)
        out["shared_scorer"] = aggregate(shared) if shared else {}
        out["failed_patients"] = len(self.failures)
        return out


def evaluate(
    patients: list[str] | None = None,
    *,
    root: str | None = None,
    labels_root: str | None = None,
    workers: int | None = None,
    threads: int | None = None,
) -> Evaluation:
    """Run :func:`label_spine` on each labelled patient (or ``patients``) and score it against the dataset labels.

    Runs in a process pool (``workers`` defaults to the CPU count, capped at 8). Each pool worker runs the
    labeller on one ONNX Runtime thread unless ``threads`` says otherwise (0 = ORT default), so the pool does not
    oversubscribe the machine. The result carries no paths.
    """
    from ..datasets.sudirman import labelled_patients, patient_sessions

    sessions = patient_sessions(root)
    ids = patients if patients is not None else labelled_patients(labels_root)
    jobs: list[tuple[str, list[str], str | None, int | None]] = []
    for pid in ids:
        if pid not in sessions:
            log.warning("labelled patient %s has no DICOM folder; skipped", pid)
            continue
        jobs.append((pid, [str(s) for s in sessions[pid]], labels_root, threads))
    n = workers or min(8, os.cpu_count() or 1)
    if n <= 1 or len(jobs) <= 1:
        results = [_evaluate_patient(j) for j in jobs]
    else:
        with get_context("spawn").Pool(n) as pool:
            results = pool.map(_evaluate_patient, jobs, chunksize=4)
    return Evaluation(results=results)
