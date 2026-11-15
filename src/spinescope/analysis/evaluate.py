"""Validation harness: run registered analyses on labelled cases, match levels to ground truth, and score agreement.

Pipeline for each case (``score_case``, run in a worker process):

1. Load the study and the reference truth: disc centres from the dataset's masks, and the radiologist grade of
   each disc from its gradings (SPIDER: :mod:`spinescope.datasets.gradings`).
2. ``label_spine`` gives the anchors; each requested analysis runs with :func:`spinescope.analysis.base.run`.
3. Field predictors (registered per analysis and field with :func:`register_predictor`) turn the analysis rows into
   one value per predicted disc, with an optional score (for AUC).
4. Predicted discs are matched one-to-one to truth discs by sagittal-plane (y, z) distance, within
   :data:`MATCH_MM` (12 mm, the labelling validation's rule). A match is **scored only when the predicted level name
   equals the truth level name**. Name disagreements (a label shifted by one level, for example) are dropped and
   counted as ``label_mismatch``; predicted discs with no truth disc nearby are ``no_reference``; truth discs with no
   prediction are ``missing_pred``. Nothing is dropped silently: :class:`EvalReport` carries every count.
5. Per scored disc and field the pair is ``scored`` (both values present), ``not_assessed`` (the analysis gave no
   value) or ``no_truth`` (the reference has no grade for the field).

Metrics (:func:`summarise`) are per field kind: binary (sensitivity, specificity, PPV, NPV, AUC if a score exists),
ordinal (quadratic and linear weighted kappa, MAE, accuracy, within one grade, confusion matrix), categorical (accuracy,
unweighted kappa, one-vs-rest sensitivity, specificity and PPV per class, confusion matrix), continuous (MAE, RMSE,
bias). Confidence intervals are percentile bootstrap intervals that resample **cases** (patients), not discs, because
discs of one patient are not independent. Breakdowns by vendor, field strength, sequence set, transitional flag and
reference contrast are point estimates.

Everything is deterministic: studies are processed in the order given, bootstrap draws use a fixed seed, and worker
processes share the ONNX Runtime thread budget ``cores // workers``.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from .base import Result, run

log = logging.getLogger(__name__)

#: Maximum sagittal-plane (y, z) distance for a predicted disc to match a truth disc (mm). Same as the labelling
#: validation (docs/validation.md).
MATCH_MM = 12.0
#: Kinds of field the metrics understand.
FIELD_KINDS: tuple[str, ...] = ("binary", "ordinal", "categorical", "continuous")
#: Breakdown keys reported for every field (point estimates).
BREAKDOWN_KEYS: tuple[str, ...] = (
    "vendor",
    "field_strength",
    "sequences",
    "transitional_suspect",
    "reference",
)
#: Confidence level of the reported intervals.
CI_LEVEL = 0.95


# ============================================================================= field registry
@dataclass(frozen=True)
class TruthField:
    """A field of the ground truth: its kind and, for ordinal / categorical fields, its class labels."""

    name: str
    kind: str
    labels: tuple[int, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in FIELD_KINDS:
            raise ValueError(f"field {self.name}: kind must be one of {FIELD_KINDS}")


#: Predictor: (result, anchors) -> {predicted level name: (value or None, score or None)}.
Predictor = Callable[[Result, Anchors], dict[str, tuple[Any, float | None]]]
_TRUTH_FIELDS: dict[str, TruthField] = {}
_PREDICTORS: dict[tuple[str, str], Predictor] = {}  # (analysis, truth field name) -> predictor


def register_truth_field(spec: TruthField) -> TruthField:
    """Declare a ground-truth field (idempotent for identical specs)."""
    old = _TRUTH_FIELDS.get(spec.name)
    if old is not None and old != spec:
        raise ValueError(f"truth field {spec.name!r} already declared differently")
    _TRUTH_FIELDS[spec.name] = spec
    return spec


def register_predictor(analysis: str, truth: str, fn: Predictor) -> None:
    """Connect an analysis output to a declared truth field: ``fn(result, anchors)`` gives one value per disc.

    One analysis may predict several truth fields; a truth field may be predicted by one predictor only.
    """
    _ensure_builtin()
    if truth not in _TRUTH_FIELDS:
        raise KeyError(f"no truth field {truth!r}; declare it with register_truth_field first")
    for (other, tname), old in _PREDICTORS.items():
        if tname == truth and old is not fn:
            raise ValueError(f"truth field {truth!r} is already predicted by analysis {other!r}")
    _PREDICTORS[(analysis, truth)] = fn


def truth_fields() -> dict[str, TruthField]:
    """Declared ground-truth fields (name -> spec)."""
    _ensure_builtin()
    return dict(_TRUTH_FIELDS)


def predictors() -> dict[str, str]:
    """Registered truth field -> analysis that predicts it."""
    _ensure_builtin()
    return {tname: an for (an, tname) in _PREDICTORS}


# ============================================================================= metrics
def _div(a: Any, b: Any) -> float:
    return float(a) / float(b) if b else float("nan")


def confusion(y: NDArray[Any], p: NDArray[Any], labels: Sequence[int]) -> NDArray[np.int64]:
    """Confusion matrix (rows: truth, columns: prediction) over ``labels``; values outside ``labels`` are ignored."""
    idx = {int(v): i for i, v in enumerate(labels)}
    k = len(labels)
    m = np.zeros((k, k), dtype=np.int64)
    for a, b in zip(y, p, strict=False):
        if int(a) in idx and int(b) in idx:
            m[idx[int(a)], idx[int(b)]] += 1
    return m


def weighted_kappa(conf: NDArray[np.int64], weights: str = "quadratic") -> float:
    """Cohen's kappa with ``quadratic`` or ``linear`` ordinal weights, or ``none`` (unweighted) from a confusion matrix."""
    k = conf.shape[0]
    n = float(conf.sum())
    if n == 0 or k < 2:
        return float("nan")
    i = np.arange(k, dtype=np.float64)
    if weights == "none":
        w = 1.0 - np.eye(k)
    else:
        d = np.abs(i[:, None] - i[None, :])
        w = d**2 / float((k - 1) ** 2) if weights == "quadratic" else d / float(k - 1)
    expected = np.outer(conf.sum(axis=1), conf.sum(axis=0)) / n
    denom = float((w * expected).sum())
    if denom == 0:
        return float("nan")
    return 1.0 - float((w * conf).sum()) / denom


def auc(y: NDArray[Any], s: NDArray[Any]) -> float:
    """ROC AUC by the rank-sum statistic (ties counted half); NaN without both classes."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _binary(y: NDArray[Any], p: NDArray[Any]) -> dict[str, float]:
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    tn = float(np.sum((y == 0) & (p == 0)))
    return {
        "sensitivity": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "ppv": _div(tp, tp + fp),
        "npv": _div(tn, tn + fn),
        "prevalence": _div(tp + fn, tp + fn + fp + tn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _per_class(conf: NDArray[np.int64], labels: Sequence[int]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    total = float(conf.sum())
    for i, lab in enumerate(labels):
        tp = float(conf[i, i])
        fn = float(conf[i, :].sum() - tp)
        fp = float(conf[:, i].sum() - tp)
        tn = total - tp - fn - fp
        out[str(lab)] = {
            "n": float(conf[i, :].sum()),
            "sensitivity": _div(tp, tp + fn),
            "specificity": _div(tn, tn + fp),
            "ppv": _div(tp, tp + fp),
        }
    return out


def summarise(
    kind: str, labels: Sequence[int], y: NDArray[Any], p: NDArray[Any], s: NDArray[Any] | None = None
) -> dict[str, Any]:
    """Metrics of one field from paired arrays (truth ``y``, prediction ``p``, optional score ``s``).

    Arrays must contain only scored pairs (no NaN in ``y`` or ``p``). Returns a flat dict; NaN where undefined.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = int(y.size)
    out: dict[str, Any] = {"n": n}
    if n == 0:
        return out
    if kind == "binary":
        out.update(_binary(y, p))
        out["auc"] = auc(y, s) if s is not None and np.isfinite(s).all() else float("nan")
        return out
    if kind == "continuous":
        err = p - y
        out.update(
            mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err**2))), bias=float(np.mean(err))
        )
        return out
    yi = y.astype(np.int64)
    pi = p.astype(np.int64)
    conf = confusion(yi, pi, labels)
    out["accuracy"] = _div(np.trace(conf), conf.sum())
    out["confusion"] = conf.tolist()
    out["per_class"] = _per_class(conf, labels)
    if kind == "ordinal":
        out["kappa_quadratic"] = weighted_kappa(conf, "quadratic")
        out["kappa_linear"] = weighted_kappa(conf, "linear")
        out["mae"] = float(np.mean(np.abs(pi - yi)))
        out["within_one"] = float(np.mean(np.abs(pi - yi) <= 1))
    elif kind == "categorical":
        out["kappa"] = weighted_kappa(conf, "none")
        any_b = _binary((yi > 0).astype(np.float64), (pi > 0).astype(np.float64))
        out["any_sensitivity"] = any_b["sensitivity"]
        out["any_specificity"] = any_b["specificity"]
        out["any_ppv"] = any_b["ppv"]
    return out


#: Headline metrics per kind: the ones with bootstrap intervals.
HEADLINE: dict[str, tuple[str, ...]] = {
    "binary": ("sensitivity", "specificity", "ppv", "npv", "auc"),
    "ordinal": ("kappa_quadratic", "mae", "accuracy"),
    "categorical": ("accuracy", "kappa", "any_sensitivity", "any_specificity"),
    "continuous": ("mae",),
}


def bootstrap_ci(
    kind: str,
    labels: Sequence[int],
    y: NDArray[Any],
    p: NDArray[Any],
    s: NDArray[Any] | None,
    case_index: NDArray[np.int64],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap intervals of the headline metrics, resampling whole cases (``case_index`` per row)."""
    if n_boot <= 0 or y.size == 0:
        return {}
    cases = np.unique(case_index)
    rows = {c: np.where(case_index == c)[0] for c in cases}
    rng = np.random.default_rng(seed)
    alpha = (1.0 - CI_LEVEL) / 2.0
    draws: dict[str, list[float]] = {m: [] for m in HEADLINE[kind]}
    for _ in range(n_boot):
        pick = rng.integers(0, cases.size, cases.size)
        idx = np.concatenate([rows[cases[i]] for i in pick])
        m = summarise(kind, labels, y[idx], p[idx], None if s is None else s[idx])
        for name in draws:
            draws[name].append(float(m.get(name, float("nan"))))
    out: dict[str, tuple[float, float]] = {}
    for name, vals in draws.items():
        arr = np.asarray(vals, dtype=np.float64)
        if np.isfinite(arr).sum() == 0:
            out[name] = (float("nan"), float("nan"))
        else:
            out[name] = (
                float(np.nanpercentile(arr, 100 * alpha)),
                float(np.nanpercentile(arr, 100 * (1 - alpha))),
            )
    return out


# ============================================================================= matching
@dataclass(frozen=True)
class TruthRef:
    """One reference disc: level name, centre (LPS mm) and the grades of every truth field."""

    name: str
    center: tuple[float, float, float]
    values: dict[str, Any]


def match_discs(
    pred: Mapping[str, Sequence[float]], truth: Sequence[TruthRef], max_mm: float = MATCH_MM
) -> dict[str, Any]:
    """One-to-one matching of predicted disc centres to truth discs by (y, z) distance.

    Returns ``{"pairs": [(pred_name, truth_index)], "no_reference": [pred names], "missing_pred": [truth indices]}``.
    Greedy by distance (closest pairs first), ties broken by names, so the result is deterministic.
    """
    cand: list[tuple[float, str, int]] = []
    for pname, c in sorted(pred.items()):
        pc = np.asarray(c, dtype=np.float64)
        for j, t in enumerate(truth):
            d = float(np.hypot(pc[1] - t.center[1], pc[2] - t.center[2]))
            if d <= max_mm:
                cand.append((d, pname, j))
    cand.sort()
    used_p: set[str] = set()
    used_t: set[int] = set()
    pairs: list[tuple[str, int]] = []
    for _d, pname, j in cand:
        if pname in used_p or j in used_t:
            continue
        used_p.add(pname)
        used_t.add(j)
        pairs.append((pname, j))
    return {
        "pairs": sorted(pairs, key=lambda x: x[0]),
        "no_reference": sorted(p for p in pred if p not in used_p),
        "missing_pred": sorted(j for j in range(len(truth)) if j not in used_t),
    }


# ============================================================================= case scoring
@dataclass(frozen=True)
class CaseSpec:
    """What to score: one case of a dataset. ``sequences`` None means every sequence the case has."""

    source: str  # "spider"
    case: str
    root: str | None = None
    sequences: tuple[str, ...] | None = None
    analyses: tuple[str, ...] = ()
    backend: str = "auto"


@dataclass
class Pair:
    """One disc and one field. ``status``: scored | not_assessed | no_truth."""

    case: str
    level: str
    field: str
    status: str
    truth: float | None
    pred: float | None
    score: float | None
    groups: dict[str, str] = dc_field(default_factory=dict)


@dataclass
class CaseOutcome:
    case: str
    counts: dict[str, int] = dc_field(default_factory=dict)
    pairs: list[Pair] = dc_field(default_factory=list)
    groups: dict[str, str] = dc_field(default_factory=dict)
    warnings: list[str] = dc_field(default_factory=list)


def _predict(
    results: dict[str, Result], anchors: Anchors, analyses: Sequence[str]
) -> dict[str, dict[str, tuple[Any, float | None]]]:
    """Truth-field values per predicted disc, from the predictors of the requested analyses that ran."""
    out: dict[str, dict[str, tuple[Any, float | None]]] = {}
    for (analysis, tname), fn in sorted(_PREDICTORS.items()):
        if analysis not in analyses or analysis not in results:
            continue
        for level, pair in fn(results[analysis], anchors).items():
            out.setdefault(level, {})[tname] = pair
    return out


def score_case(spec: CaseSpec, config: Config | None = None) -> CaseOutcome:
    """Score one case. Never raises: a failure is reported as a warning on the outcome (no paths in the message)."""
    _ensure_builtin()
    cfg = config or DEFAULT
    out = CaseOutcome(case=spec.case)
    try:
        loader = _LOADERS[spec.source]
        study, truth, groups, meta_ok = loader(spec)
    except Exception as exc:  # the message can quote paths: keep the type only
        out.warnings.append(f"case not loaded: {type(exc).__name__}")
        log.info("case %s not loaded (%s)", spec.case, type(exc).__name__)
        return out
    out.groups.update(groups)
    if not meta_ok:
        out.warnings.append("no technical metadata for this case")
    from ..anatomy.pipeline import label_spine

    try:
        anchors = label_spine(study, cfg, backend=spec.backend)
    except Exception as exc:
        out.warnings.append(f"labelling failed: {type(exc).__name__}")
        return out
    results: dict[str, Result] = {}
    for name in spec.analyses:
        try:
            results[name] = run(name, study, anchors, cfg)
        except Exception as exc:
            out.warnings.append(f"analysis {name} failed: {type(exc).__name__}")
    contrast = str(results.get("discs", Result(name="")).meta.get("contrast", "none"))
    out.groups["reference"] = contrast
    preds = _predict(results, anchors, spec.analyses)
    pred_centres = {lab: anchors.discs[lab].center for lab in anchors.discs}
    matched = match_discs(pred_centres, truth)
    counts = {
        "pred_discs": len(pred_centres),
        "truth_discs": len(truth),
        "matched": len(matched["pairs"]),
        "label_mismatch": 0,
        "no_reference": len(matched["no_reference"]),
        "missing_pred": len(matched["missing_pred"]),
        "scored_discs": 0,
    }
    for pname, j in matched["pairs"]:
        t = truth[j]
        if pname != t.name:
            counts["label_mismatch"] += 1
            continue
        counts["scored_discs"] += 1
        for fname in sorted(_TRUTH_FIELDS):
            value, score = preds.get(pname, {}).get(fname, (None, None))
            tval = t.values.get(fname)
            if tval is None:
                status = "no_truth"
            elif value is None:
                status = "not_assessed"
            else:
                status = "scored"
            out.pairs.append(
                Pair(
                    case=spec.case,
                    level=pname,
                    field=fname,
                    status=status,
                    truth=None if tval is None else float(tval),
                    pred=None if value is None else float(value),
                    score=None if score is None or not math.isfinite(float(score)) else float(score),
                    groups=dict(out.groups),
                )
            )
    out.counts = counts
    return out


# ============================================================================= report
@dataclass
class EvalReport:
    """Aggregated evaluation. ``to_dict`` is JSON-ready and contains no paths."""

    dataset: str
    analyses: list[str]
    n_cases: int
    counts: dict[str, int]
    cases_with_warnings: int
    fields: dict[str, dict[str, Any]]
    settings: dict[str, Any]
    #: Per-case outcomes (for pairs and warnings). Not part of :meth:`to_dict`.
    outcomes: list[CaseOutcome] = dc_field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict: NaN (undefined metrics) becomes null."""
        return _json_clean(
            {
                "dataset": self.dataset,
                "analyses": list(self.analyses),
                "n_cases": self.n_cases,
                "counts": dict(self.counts),
                "cases_with_warnings": self.cases_with_warnings,
                "fields": self.fields,
                "settings": dict(self.settings),
            }
        )


def _json_clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _field_block(
    tf: TruthField, pairs: list[Pair], all_case_ids: list[str], n_boot: int, seed: int
) -> dict[str, Any]:
    scored = [p for p in pairs if p.status == "scored"]
    block: dict[str, Any] = {
        "kind": tf.kind,
        "labels": list(tf.labels),
        "pairs": len(pairs),
        "scored": len(scored),
        "not_assessed": sum(1 for p in pairs if p.status == "not_assessed"),
        "no_truth": sum(1 for p in pairs if p.status == "no_truth"),
    }
    if not scored:
        block["metrics"] = {}
        block["ci95"] = {}
        block["breakdown"] = {}
        return block
    cases_present = sorted({p.case for p in scored}, key=lambda c: (len(c), c))
    cidx = {c: i for i, c in enumerate(cases_present)}
    y, pr, sc, ci = _arrays(scored, cidx)
    labels = list(tf.labels)
    block["metrics"] = summarise(tf.kind, labels, y, pr, sc)
    block["ci95"] = {
        k: list(v) for k, v in bootstrap_ci(tf.kind, labels, y, pr, sc, ci, n_boot=n_boot, seed=seed).items()
    }
    breakdown: dict[str, dict[str, Any]] = {}
    for key in BREAKDOWN_KEYS:
        groups: dict[str, list[Pair]] = {}
        for p in scored:
            groups.setdefault(p.groups.get(key, "unknown"), []).append(p)
        if len(groups) < 2 and key != "sequences":
            continue
        sub: dict[str, Any] = {}
        for gval, plist in sorted(groups.items()):
            gy, gp, gs, _ = _arrays(plist, {c: 0 for c in {p.case for p in plist}})
            sub[gval] = {"cases": len({p.case for p in plist}), **summarise(tf.kind, labels, gy, gp, gs)}
        breakdown[key] = sub
    block["breakdown"] = breakdown
    return block


def _arrays(
    pairs: list[Pair], cidx: dict[str, int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.int64]]:
    y = np.array([p.truth for p in pairs], dtype=np.float64)
    pr = np.array([p.pred for p in pairs], dtype=np.float64)
    sc_vals = [np.nan if p.score is None else p.score for p in pairs]
    sc = np.array(sc_vals, dtype=np.float64)
    ci = np.array([cidx[p.case] for p in pairs], dtype=np.int64)
    return y, pr, (sc if np.isfinite(sc).any() else None), ci


def _ensure_builtin() -> None:
    """Import the built-in truth fields, predictors and loaders (idempotent)."""
    from . import _evaluate_builtin  # noqa: F401


_LOADERS: dict[str, Callable[[CaseSpec], tuple[Any, list[TruthRef], dict[str, str], bool]]] = {}


def register_loader(
    source: str, fn: Callable[[CaseSpec], tuple[Any, list[TruthRef], dict[str, str], bool]]
) -> None:
    """Register a dataset loader: ``fn(spec) -> (study, truth discs, group labels, metadata found)``."""
    _LOADERS[source] = fn


def _worker_init(threads: int) -> None:
    from ..anatomy.ml.labeller import set_intra_op_threads

    set_intra_op_threads(max(1, threads))


def default_workers() -> int:
    """Half the cores (at least one): leaves room for the ONNX Runtime threads of each worker."""
    return max(1, (os.cpu_count() or 2) // 2)


def evaluate(
    cases: Sequence[CaseSpec],
    *,
    dataset: str,
    workers: int | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    config: Config | None = None,
) -> EvalReport:
    """Score every case (in order), aggregate per field with bootstrap intervals and breakdowns.

    ``workers`` processes each get ``cores // workers`` ONNX Runtime threads (at least one). ``workers=1`` runs in
    this process.
    """
    _ensure_builtin()
    cores = os.cpu_count() or 2
    n = max(1, min(workers or default_workers(), max(1, len(cases))))
    threads = max(1, cores // n)
    if n == 1:
        _worker_init(threads)
        outcomes = [score_case(c, config) for c in cases]
    else:
        with ProcessPoolExecutor(max_workers=n, initializer=_worker_init, initargs=(threads,)) as ex:
            outcomes = list(ex.map(_score_case_entry, [(c, config) for c in cases], chunksize=1))
    return aggregate(
        outcomes,
        dataset=dataset,
        analyses=sorted({a for c in cases for a in c.analyses}),
        settings={
            "workers": n,
            "onnx_threads_per_worker": threads,
            "match_mm": MATCH_MM,
            "n_boot": n_boot,
            "seed": seed,
            "ci": "percentile bootstrap over cases, 95%",
        },
        n_boot=n_boot,
        seed=seed,
    )


def _score_case_entry(item: tuple[CaseSpec, Config | None]) -> CaseOutcome:
    spec, cfg = item
    return score_case(spec, cfg)


def aggregate(
    outcomes: Sequence[CaseOutcome],
    *,
    dataset: str,
    analyses: Sequence[str],
    settings: dict[str, Any],
    n_boot: int,
    seed: int,
) -> EvalReport:
    """Turn case outcomes into an :class:`EvalReport` (counts, per-field metrics, intervals, breakdowns)."""
    _ensure_builtin()
    totals: dict[str, int] = {}
    for o in outcomes:
        for k, v in o.counts.items():
            totals[k] = totals.get(k, 0) + int(v)
    all_pairs = [p for o in outcomes for p in o.pairs]
    case_ids = [o.case for o in outcomes]
    fields: dict[str, dict[str, Any]] = {}
    owner = {tname: an for (an, tname) in _PREDICTORS if an in analyses}
    for fname, tf in sorted(_TRUTH_FIELDS.items()):
        pairs = [p for p in all_pairs if p.field == fname]
        block = _field_block(tf, pairs, case_ids, n_boot, seed)
        block["analysis"] = owner.get(fname)  # None: no requested analysis predicts this field
        fields[fname] = block
    return EvalReport(
        dataset=dataset,
        analyses=list(analyses),
        n_cases=len(outcomes),
        counts=totals,
        cases_with_warnings=sum(1 for o in outcomes if o.warnings),
        fields=fields,
        settings=settings,
        outcomes=list(outcomes),
    )


def pairs_table(outcomes: Sequence[CaseOutcome]) -> list[dict[str, Any]]:
    """Flat per-disc, per-field rows (case ids and level names only) for CSV export."""
    rows: list[dict[str, Any]] = []
    for o in outcomes:
        for p in o.pairs:
            rows.append(
                {
                    "case": p.case,
                    "level": p.level,
                    "field": p.field,
                    "status": p.status,
                    "truth": p.truth,
                    "pred": p.pred,
                    "score": p.score,
                    **{k: p.groups.get(k, "") for k in BREAKDOWN_KEYS},
                }
            )
    return rows


def case_warnings(outcomes: Sequence[CaseOutcome]) -> list[dict[str, Any]]:
    """Cases with warnings, with their counts (no paths)."""
    return [{"case": o.case, "warnings": list(o.warnings)} for o in outcomes if o.warnings]


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return "-" if math.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def _ci_text(ci: Sequence[float] | None, nd: int = 2) -> str:
    if not ci or len(ci) != 2 or any(math.isnan(v) for v in ci):
        return ""
    return f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def render_markdown(report: EvalReport) -> str:
    """Markdown tables of an :class:`EvalReport`: coverage, per-field metrics with 95% CI, breakdowns."""
    lines: list[str] = [
        f"# Analysis evaluation: {report.dataset}",
        "",
        f"Analyses: {', '.join(report.analyses) or 'none'}. Cases: {report.n_cases} "
        f"({report.cases_with_warnings} with warnings).",
        "",
        "## Coverage (discs matched to truth)",
        "",
        "| predicted discs | truth discs | matched | scored discs | label mismatch (dropped) | no reference (dropped) | "
        "truth not predicted |",
        "|---|---|---|---|---|---|---|",
        "| {pred_discs} | {truth_discs} | {matched} | {scored_discs} | {label_mismatch} | {no_reference} | "
        "{missing_pred} |".format(
            **{
                k: report.counts.get(k, 0)
                for k in (
                    "pred_discs",
                    "truth_discs",
                    "matched",
                    "scored_discs",
                    "label_mismatch",
                    "no_reference",
                    "missing_pred",
                )
            }
        ),
        "",
        "A disc is scored only when its predicted level name equals the reference name at the same position.",
        "",
    ]
    for name, block in report.fields.items():
        head = f"## {name} ({block['kind']})"
        if block.get("analysis"):
            head += f", from {block['analysis']}"
        else:
            head += ", no requested analysis predicts this field"
        lines += [head, ""]
        lines.append(
            f"Pairs {block['pairs']}, scored {block['scored']}, not assessed {block['not_assessed']}, "
            f"no reference grade {block['no_truth']}."
        )
        lines.append("")
        m = block.get("metrics") or {}
        ci = block.get("ci95") or {}
        if not m:
            lines += ["No scored pairs.", ""]
            continue
        if block["kind"] == "binary":
            lines += [
                "| sensitivity | specificity | PPV | NPV | AUC | prevalence | n |",
                "|---|---|---|---|---|---|---|",
            ]
            lines.append(
                f"| {_fmt(m.get('sensitivity'))}{_ci_text(ci.get('sensitivity'))} | "
                f"{_fmt(m.get('specificity'))}{_ci_text(ci.get('specificity'))} | "
                f"{_fmt(m.get('ppv'))}{_ci_text(ci.get('ppv'))} | {_fmt(m.get('npv'))}{_ci_text(ci.get('npv'))} | "
                f"{_fmt(m.get('auc'))}{_ci_text(ci.get('auc'))} | {_fmt(m.get('prevalence'))} | {m.get('n')} |"
            )
        elif block["kind"] == "ordinal":
            lines += [
                "| weighted kappa (quadratic) | kappa (linear) | MAE | accuracy | within 1 | n |",
                "|---|---|---|---|---|---|",
            ]
            lines.append(
                f"| {_fmt(m.get('kappa_quadratic'))}{_ci_text(ci.get('kappa_quadratic'))} | "
                f"{_fmt(m.get('kappa_linear'))} | {_fmt(m.get('mae'))}{_ci_text(ci.get('mae'))} | "
                f"{_fmt(m.get('accuracy'))}{_ci_text(ci.get('accuracy'))} | {_fmt(m.get('within_one'))} | {m.get('n')} |"
            )
        elif block["kind"] == "categorical":
            lines += [
                "| accuracy | kappa (unweighted) | any-positive sensitivity | any-positive specificity | n |",
                "|---|---|---|---|---|",
            ]
            lines.append(
                f"| {_fmt(m.get('accuracy'))}{_ci_text(ci.get('accuracy'))} | {_fmt(m.get('kappa'))} | "
                f"{_fmt(m.get('any_sensitivity'))} | {_fmt(m.get('any_specificity'))} | {m.get('n')} |"
            )
        else:
            lines += ["| MAE | RMSE | bias | n |", "|---|---|---|---|"]
            lines.append(
                f"| {_fmt(m.get('mae'))} | {_fmt(m.get('rmse'))} | {_fmt(m.get('bias'))} | {m.get('n')} |"
            )
        lines.append("")
        per = m.get("per_class")
        if per:
            lines += ["| class | n | sensitivity | specificity | PPV |", "|---|---|---|---|---|"]
            for lab, pc in per.items():
                lines.append(
                    f"| {lab} | {int(pc['n'])} | {_fmt(pc['sensitivity'])} | {_fmt(pc['specificity'])} | {_fmt(pc['ppv'])} |"
                )
            lines.append("")
        if m.get("confusion"):
            labels = block.get("labels") or list(range(len(m["confusion"])))
            lines += [
                "Confusion (rows: reference, columns: prediction):",
                "",
                "| ref \\ pred | " + " | ".join(str(x) for x in labels) + " |",
                "|---|" + "---|" * len(labels),
            ]
            for lab, row in zip(labels, m["confusion"], strict=False):
                lines.append(f"| {lab} | " + " | ".join(str(int(v)) for v in row) + " |")
            lines.append("")
        for key, sub in (block.get("breakdown") or {}).items():
            lines += [f"By {key}:", ""]
            head_metric = (
                "kappa_quadratic"
                if block["kind"] == "ordinal"
                else (
                    "sensitivity"
                    if block["kind"] == "binary"
                    else ("mae" if block["kind"] == "continuous" else "accuracy")
                )
            )
            lines += [f"| {key} | cases | n | {head_metric} |", "|---|---|---|---|"]
            for gval, gm in sub.items():
                lines.append(f"| {gval} | {gm.get('cases')} | {gm.get('n')} | {_fmt(gm.get(head_metric))} |")
            lines.append("")
    lines += ["## Settings", ""]
    lines += [f"- {k}: {v}" for k, v in report.settings.items()]
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "BREAKDOWN_KEYS",
    "MATCH_MM",
    "CaseOutcome",
    "CaseSpec",
    "EvalReport",
    "Pair",
    "TruthField",
    "TruthRef",
    "aggregate",
    "auc",
    "bootstrap_ci",
    "case_warnings",
    "confusion",
    "evaluate",
    "match_discs",
    "pairs_table",
    "predictors",
    "register_loader",
    "register_predictor",
    "register_truth_field",
    "render_markdown",
    "score_case",
    "summarise",
    "truth_fields",
    "weighted_kappa",
]
