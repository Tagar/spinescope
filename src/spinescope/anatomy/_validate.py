"""Validation harness: ground truth from SPIDER-style label masks and scoring of :class:`Anchors`.

Not used by the labeller itself. SPIDER masks label discs 201..2xx (disc 20k lies directly below
vertebra k; the lowest disc, 201, is taken as L5-S1) and the spinal canal as 100. Truth discs are
ranked from the bottom, so the rank-r disc from the bottom is named with the r-th level name from
the bottom (L5-S1, L4-L5, ...), which is the convention :func:`spinescope.anatomy.numbering.disc_names`
uses. Requires the optional ``volumes`` extra (SimpleITK) to read ``.mha`` masks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .anchors import Anchors
from .numbering import disc_names

DISC_LABEL_MIN = 201
DISC_LABEL_MAX = 299
CANAL_LABEL = 100


@dataclass
class TruthDisc:
    label: int  # SPIDER disc label (201 = lowest)
    name: str  # level name by rank from the bottom (assumes 201 = L5-S1)
    center: np.ndarray  # LPS mm, voxel centroid
    normal: np.ndarray  # unit, superior


@dataclass
class VolumeScore:
    n_truth: int
    n_pred: int
    matched: int
    label_correct: int
    centre_err_mm: list[float] = field(default_factory=list)
    normal_err_deg: list[float] = field(default_factory=list)
    midline_err_mm: float | None = None
    numbering_conf: float = 0.0


def _sitk_arrays(path: str) -> tuple[np.ndarray, Any]:
    import SimpleITK as sitk

    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img), img


def truth_from_mask(mask_path: str) -> tuple[list[TruthDisc], float | None]:
    """Disc truths (bottom-up, named by rank) and the canal's median x (LPS) from a SPIDER mask."""

    arr, img = _sitk_arrays(mask_path)
    labels = sorted({int(v) for v in np.unique(arr) if DISC_LABEL_MIN <= v <= DISC_LABEL_MAX})
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    names = disc_names(len(labels))
    discs: list[TruthDisc] = []
    for rank, lab in enumerate(labels):
        zz, yy, xx = np.nonzero(arr == lab)
        if len(zz) < 10:
            continue
        idx = np.stack([xx, yy, zz], axis=1).astype(np.float64)  # sitk index order (x, y, z)
        pts = origin + (idx * spacing) @ direction.T
        centre = pts.mean(axis=0)
        cov = np.cov((pts - centre).T)
        _w, v = np.linalg.eigh(cov)
        normal = v[:, 0]
        if normal[2] < 0:
            normal = -normal
        discs.append(
            TruthDisc(label=lab, name=names[rank], center=centre, normal=normal / np.linalg.norm(normal))
        )
    canal = np.nonzero(arr == CANAL_LABEL)
    canal_x = None
    if len(canal[0]):
        # Same index -> LPS transform as the discs (full direction matrix, not only its x term).
        cidx = np.stack([canal[2], canal[1], canal[0]], axis=1).astype(np.float64)
        canal_x = float(np.median((origin + (cidx * spacing) @ direction.T)[:, 0]))
    return discs, canal_x


def _inplane(p: np.ndarray) -> np.ndarray:
    """Sagittal-plane (y, z) of an LPS point: the in-plane distance used for matching."""
    return np.asarray(p, dtype=np.float64)[1:]


def score_anchors(
    anchors: Anchors, truth: list[TruthDisc], midline_truth: float | None = None
) -> VolumeScore:
    """Match predicted discs to truth discs (nearest in the sagittal plane, within 12 mm) and score.

    ``label_correct`` counts matches whose predicted name equals the truth name.
    """
    preds = list(anchors.discs.items())
    s = VolumeScore(n_truth=len(truth), n_pred=len(preds), matched=0, label_correct=0)
    s.numbering_conf = float(anchors.numbering.get("confidence", 0.0) or 0.0)
    if midline_truth is not None and anchors.midline_x is not None:
        s.midline_err_mm = abs(anchors.midline_x - midline_truth)
    used: set[int] = set()
    for name, d in preds:
        c = np.asarray(d.center)
        best: tuple[float, int] | None = None
        for i, t in enumerate(truth):
            if i in used:
                continue
            dist = float(np.linalg.norm(_inplane(c) - _inplane(t.center)))
            if best is None or dist < best[0]:
                best = (dist, i)
        if best is None or best[0] > 12.0:
            continue
        used.add(best[1])
        t = truth[best[1]]
        s.matched += 1
        s.centre_err_mm.append(float(np.linalg.norm(_inplane(c) - _inplane(t.center))))
        n = np.asarray(d.normal)
        s.normal_err_deg.append(float(np.degrees(np.arccos(np.clip(abs(n @ t.normal), -1, 1)))))
        s.label_correct += int(name == t.name)
    return s


def aggregate(scores: list[VolumeScore]) -> dict[str, float]:
    """Pooled metrics over volumes."""
    n_t = sum(s.n_truth for s in scores)
    n_p = sum(s.n_pred for s in scores)
    m = sum(s.matched for s in scores)
    lc = sum(s.label_correct for s in scores)
    ce = [e for s in scores for e in s.centre_err_mm]
    ne = [e for s in scores for e in s.normal_err_deg]
    out: dict[str, float] = {
        "volumes": float(len(scores)),
        "recall": m / max(n_t, 1),
        "precision": m / max(n_p, 1),
        "label_accuracy": lc / max(m, 1),
        "label_accuracy_all_truth": lc / max(n_t, 1),
        "centre_err_median_mm": float(np.median(ce)) if ce else float("nan"),
        "centre_err_p90_mm": float(np.percentile(ce, 90)) if ce else float("nan"),
        "centre_within_3mm": float(np.mean(np.asarray(ce) <= 3.0)) if ce else float("nan"),
        "normal_err_median_deg": float(np.median(ne)) if ne else float("nan"),
    }
    return out
