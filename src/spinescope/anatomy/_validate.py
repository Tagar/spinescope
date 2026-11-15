"""Validation harness: ground truth from SPIDER-style label masks and scoring of :class:`Anchors`.

Not used by the labeller itself. SPIDER masks label discs 201..2xx (disc 20k lies directly below
vertebra k; the lowest disc, 201, is taken as L5-S1) and the spinal canal as 100. Truth discs are
ranked from the bottom, so the rank-r disc from the bottom is named with the r-th level name from
the bottom (L5-S1, L4-L5, ...), which is the convention :func:`spinescope.anatomy.numbering.disc_names`
uses. Requires the optional ``volumes`` extra (SimpleITK) to read ``.mha`` masks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .anchors import Anchors
from .numbering import VERTEBRA_NAMES, disc_names

DISC_LABEL_MIN = 201
DISC_LABEL_MAX = 299
CANAL_LABEL = 100


@dataclass
class TruthDisc:
    label: int  # SPIDER disc label (201 = lowest)
    name: str  # level name by rank from the bottom (assumes 201 = L5-S1)
    center: np.ndarray  # LPS mm, voxel centroid
    normal: np.ndarray  # unit, superior


@dataclass(frozen=True)
class DiscMatch:
    """One predicted disc and what it was matched to (``truth`` is None for a false positive)."""

    pred: str
    pred_confidence: float
    truth: str | None
    centre_err_mm: float | None
    normal_err_deg: float | None
    label_correct: bool
    level_offset: int | None  # level index(pred) - level index(truth); + = labelled too cranial


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
    matches: list[DiscMatch] = field(default_factory=list)  # one per predicted disc
    missed: list[str] = field(default_factory=list)  # names of truth discs without a prediction


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
    # Canal voxels through the full image-to-LPS transform (the sitk x axis is not LPS x in general:
    # on some SPIDER volumes it runs along y, so using only the x index gave a midline 50 mm off).
    zc, yc, xc = np.nonzero(arr == CANAL_LABEL)
    canal_x = None
    if len(zc):
        cidx = np.stack([xc, yc, zc], axis=1).astype(np.float64)
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
            s.matches.append(DiscMatch(name, float(d.confidence), None, None, None, False, None))
            continue
        used.add(best[1])
        t = truth[best[1]]
        s.matched += 1
        s.centre_err_mm.append(float(np.linalg.norm(_inplane(c) - _inplane(t.center))))
        n = np.asarray(d.normal)
        ne = float(np.degrees(np.arccos(np.clip(abs(n @ t.normal), -1, 1))))
        s.normal_err_deg.append(ne)
        s.label_correct += int(name == t.name)
        s.matches.append(
            DiscMatch(
                pred=name,
                pred_confidence=float(d.confidence),
                truth=t.name,
                centre_err_mm=s.centre_err_mm[-1],
                normal_err_deg=ne,
                label_correct=name == t.name,
                level_offset=level_offset(name, t.name),
            )
        )
    s.missed = [t.name for i, t in enumerate(truth) if i not in used]
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


# ----------------------------------------------------------------------------- extended scoring
#: Level index of a vertebra name: S1 = 0, L5 = 1, ... (increases superiorly, as VERTEBRA_NAMES does).
_LEVEL_INDEX = {name: i for i, name in enumerate(VERTEBRA_NAMES)}


def level_index(disc: str) -> int | None:
    """Index of a disc's lower vertebra (``L4-L5`` -> 1, ``L5-S1`` -> 0); None if not a known disc name."""
    parts = disc.split("-")
    if len(parts) != 2:
        return None
    return _LEVEL_INDEX.get(parts[1])


def level_offset(pred: str, truth: str) -> int | None:
    """Signed level error: index(pred) - index(truth). +1 means labelled one level too cranial."""
    a, b = level_index(pred), level_index(truth)
    if a is None or b is None:
        return None
    return a - b


def truth_from_phantom(truth: dict[str, Any]) -> tuple[list[TruthDisc], float | None]:
    """Truth discs from ``SpinePhantom.truth`` (names are known; ``label`` is 0 for synthetic discs)."""
    discs = [
        TruthDisc(
            label=0,
            name=name,
            center=np.asarray(v["center"], dtype=np.float64),
            normal=np.asarray(v["normal"], dtype=np.float64) / np.linalg.norm(v["normal"]),
        )
        for name, v in truth["discs"].items()
    ]
    mx = truth.get("midline_x")
    return discs, (float(mx) if mx is not None else None)


def _prediction_records(scores: Iterable[VolumeScore]) -> list[DiscMatch]:
    return [m for s in scores for m in s.matches]


def _truth_names(scores: Iterable[VolumeScore]) -> list[str]:
    out: list[str] = []
    for s in scores:
        out.extend(m.truth for m in s.matches if m.truth is not None)
        out.extend(s.missed)
    return out


def per_level_recall(scores: list[VolumeScore]) -> dict[str, dict[str, float]]:
    """For each truth level name: truth discs, matched, recall and exact-label accuracy of the matches."""
    truth_count: dict[str, int] = {}
    for name in _truth_names(scores):
        truth_count[name] = truth_count.get(name, 0) + 1
    out: dict[str, dict[str, float]] = {}
    for name in sorted(truth_count, key=lambda n: (level_index(n) is None, level_index(n) or 0)):
        hits = [m for m in _prediction_records(scores) if m.truth == name]
        correct = sum(m.label_correct for m in hits)
        out[name] = {
            "truth": float(truth_count[name]),
            "matched": float(len(hits)),
            "recall": len(hits) / truth_count[name],
            "label_accuracy": correct / len(hits) if hits else float("nan"),
        }
    return out


def numbering_error_histogram(scores: list[VolumeScore]) -> dict[int, int]:
    """Counts of signed level errors over matched discs (0 = correct level; see :func:`level_offset`)."""
    hist: dict[int, int] = {}
    for m in _prediction_records(scores):
        if m.level_offset is not None:
            hist[m.level_offset] = hist.get(m.level_offset, 0) + 1
    return dict(sorted(hist.items()))


def reliability(scores: list[VolumeScore], n_bins: int = 10) -> dict[str, Any]:
    """Does disc confidence predict a correct label? Every predicted disc counts; false positives are wrong.

    Returns equal-width confidence bins (``n``, ``mean_confidence``, ``accuracy``), the expected
    calibration error (ECE, count-weighted) and the AUROC of confidence for correctness (None when only
    one class is present). Correct means matched to a truth disc with the exact level name.
    """
    recs = _prediction_records(scores)
    conf = np.array([m.pred_confidence for m in recs], dtype=np.float64)
    ok = np.array([m.label_correct for m in recs], dtype=bool)
    bins: list[dict[str, float]] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (conf >= lo) & ((conf < hi) if i < n_bins - 1 else (conf <= hi))
        n = int(sel.sum())
        if n == 0:
            bins.append(
                {"lo": lo, "hi": hi, "n": 0.0, "mean_confidence": float("nan"), "accuracy": float("nan")}
            )
            continue
        mc, acc = float(conf[sel].mean()), float(ok[sel].mean())
        ece += n / max(len(conf), 1) * abs(mc - acc)
        bins.append({"lo": lo, "hi": hi, "n": float(n), "mean_confidence": mc, "accuracy": acc})
    auc: float | None = None
    if ok.any() and (~ok).any():
        from scipy.stats import rankdata

        ranks = rankdata(conf)
        npos, nneg = int(ok.sum()), int((~ok).sum())
        auc = float((ranks[ok].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return {"bins": bins, "ece": float(ece), "auroc": auc, "n": float(len(conf))}


def aggregate_extended(scores: list[VolumeScore]) -> dict[str, float]:
    """Extra pooled metrics on top of :func:`aggregate`: level-error rates and whole-volume correctness."""
    recs = _prediction_records(scores)
    matched = [m for m in recs if m.truth is not None]
    offs = [m.level_offset for m in matched if m.level_offset is not None]
    n_truth = sum(s.n_truth for s in scores)
    off1 = sum(1 for o in offs if abs(o) == 1)
    return {
        "off_by_one_of_matched": off1 / max(len(offs), 1),
        "off_by_one_of_truth": off1 / max(n_truth, 1),
        "level_error_of_matched": sum(1 for o in offs if o != 0) / max(len(offs), 1),
        "false_positives_per_volume": sum(1 for m in recs if m.truth is None) / max(len(scores), 1),
        "volumes_all_truth_labelled": float(
            sum(1 for s in scores if s.n_truth and s.label_correct == s.n_truth)
        )
        / max(len(scores), 1),
    }


def group_scores(
    items: Iterable[tuple[str, VolumeScore]], key: Callable[[str], str]
) -> dict[str, list[VolumeScore]]:
    """Group per-volume scores by ``key(case_name)`` (e.g. vendor, field strength, sequence)."""
    out: dict[str, list[VolumeScore]] = {}
    for name, s in items:
        out.setdefault(key(name), []).append(s)
    return dict(sorted(out.items()))
