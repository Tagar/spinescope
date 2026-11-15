#!/usr/bin/env python3
"""Evaluate a labeller on a dataset split: cached predictions, shared metrics, markdown and an HTML gallery.

Examples::

    # the current classical labeller on SPIDER validation, 8 processes, outputs outside the repo
    python tools/eval_labeling.py --backend classical --dataset spider --split validation \\
        --out /tmp/spinescope-lab01_eval/run --cache /tmp/spinescope-lab01_eval/cache --jobs 8

    # any callable: module:function taking a Study and returning Anchors (or their dict)
    python tools/eval_labeling.py --backend mypkg.label:label_spine --dataset phantom --n 6

Per-case predictions are cached as JSON under ``<cache>/<fingerprint>/pred``; the fingerprint hashes the
labeller's source (``spinescope/anatomy`` and ``config.py``) and the backend spec, so a re-run after no
change reads the cache, and any labeller edit recomputes. Ground truth from the SPIDER masks is cached
under ``<cache>/truth``. Scoring is done from the cache, so scorer changes never need a re-run.

Outputs (in ``--out``, never inside the repository): ``summary.md``, ``metrics.json``, ``cases.csv``,
``gallery.html`` and ``png/`` (one page per worst case). Nothing derived from SPIDER is committed.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import html
import importlib
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spinescope import __file__ as _pkg_init
from spinescope.anatomy._validate import (
    TruthDisc,
    VolumeScore,
    aggregate,
    aggregate_extended,
    group_scores,
    numbering_error_histogram,
    per_level_recall,
    reliability,
    score_anchors,
    truth_from_mask,
    truth_from_phantom,
)
from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.ml.labeller import THREADS_ENV

log = logging.getLogger("eval_labeling")

SPIDER_DIR = Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public")) / "spider"
PKG_DIR = Path(_pkg_init).resolve().parent
TOOL_VERSION = "1"


# --------------------------------------------------------------------------- cases
@dataclass(frozen=True)
class Case:
    name: str  # e.g. "101_t2" (SPIDER file stem) or "phantom_03"
    dataset: str  # "spider" | "phantom"
    meta: dict[str, str]  # vendor, field, sequence, subset (never patient attributes)


def spider_cases(root: Path, split: str) -> list[Case]:
    """SPIDER volumes of one split. Reads only the technical overview columns (no sex, dates or serials)."""
    cases: list[Case] = []
    with open(root / "overview.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("new_file_name") or "").strip()
            if row.get("subset", "").strip() != split or not name:
                continue
            if (
                not (root / "images" / f"{name}.mha").is_file()
                or not (root / "masks" / f"{name}.mha").is_file()
            ):
                continue
            manu = (row.get("Manufacturer") or "").split(" ")[0] or "unknown"
            try:
                field_t = f"{float(row.get('MagneticFieldStrength') or 'nan'):.1f} T"
            except ValueError:
                field_t = "unknown"
            seq = name.split("_", 1)[1] if "_" in name else name
            cases.append(
                Case(
                    name,
                    "spider",
                    {"vendor": manu, "field": field_t, "sequence": seq, "subset": split},
                )
            )
    return sorted(cases, key=lambda c: (int(c.name.split("_")[0]), c.name))


def phantom_cases(n: int) -> list[Case]:
    """Synthetic cases with varied sacral slope, noise and degeneration (for pipeline tests)."""
    return [
        Case(
            f"phantom_{i:02d}",
            "phantom",
            {"vendor": "phantom", "field": "synthetic", "sequence": "t2", "subset": "synthetic"},
        )
        for i in range(n)
    ]


def _subsample(cases: list[Case], k: int | None) -> list[Case]:
    if not k or k >= len(cases):
        return cases
    idx = sorted({round(x) for x in np.linspace(0, len(cases) - 1, k)})
    return [cases[i] for i in idx]


# --------------------------------------------------------------------------- backends
def resolve_backend(spec: str) -> Callable[[Any], Any]:
    """'classical' | 'ml' | 'auto' -> label_spine with that backend, or 'module:function'."""
    if spec in ("classical", "ml", "auto"):
        from spinescope.anatomy import label_spine

        return functools.partial(label_spine, backend=spec)
    mod, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"backend must be 'classical', 'ml' or 'module:function', got {spec!r}")
    return getattr(importlib.import_module(mod), attr)  # type: ignore[no-any-return]


def fingerprint(spec: str) -> str:
    """Hash of the backend spec and the labeller source (anatomy package, config, callable's module)."""
    h = hashlib.sha256(f"{TOOL_VERSION}|{spec}".encode())
    files = [*sorted((PKG_DIR / "anatomy").rglob("*.py")), PKG_DIR / "config.py"]
    if ":" in spec and spec not in ("classical", "ml"):
        mod = spec.partition(":")[0]
        src = getattr(importlib.import_module(mod), "__file__", None)
        if src:
            files.append(Path(src))
    for f in files:
        h.update(f.name.encode())
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- per-case work
def _load_study(case: Case, spider_root: Path) -> Any:
    from spinescope import load_study
    from spinescope.study import from_stacks
    from spinescope.testing.phantom import PhantomSpec, SpinePhantom

    if case.dataset == "spider":
        return load_study(str(spider_root / "images" / f"{case.name}.mha"))
    i = int(case.name.split("_")[1])
    ph = SpinePhantom(
        PhantomSpec(
            sacral_slope=22.0 + 2.0 * (i % 4),
            noise=0.015 + 0.01 * (i % 3),
            degenerated={"L4-L5": 0.1 * (i % 5)},
            seed=i,
        )
    )
    stacks = [
        ph.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2),
        ph.render("sag", "t1", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2),
    ]
    return from_stacks(stacks)


def _truth(case: Case, spider_root: Path) -> tuple[list[TruthDisc], float | None]:
    if case.dataset == "spider":
        return truth_from_mask(str(spider_root / "masks" / f"{case.name}.mha"))
    from spinescope.testing.phantom import PhantomSpec, SpinePhantom

    i = int(case.name.split("_")[1])
    ph = SpinePhantom(
        PhantomSpec(
            sacral_slope=22.0 + 2.0 * (i % 4),
            noise=0.015 + 0.01 * (i % 3),
            degenerated={"L4-L5": 0.1 * (i % 5)},
            seed=i,
        )
    )
    return truth_from_phantom(ph.truth)


def _truth_cached(case: Case, spider_root: Path, cache: Path) -> dict[str, Any]:
    """Truth JSON for a case, cached on disk (SPIDER masks are re-read only when they change)."""
    path = cache / "truth" / f"{case.name}.json"
    stamp: dict[str, Any] = {"version": TOOL_VERSION}
    if case.dataset == "spider":
        m = spider_root / "masks" / f"{case.name}.mha"
        stamp.update(size=m.stat().st_size, mtime=int(m.stat().st_mtime), truth_formula="canal-lps-v2")
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("stamp") == stamp:
            return dict(cached["truth"])
    discs, mx = _truth(case, spider_root)
    truth = {
        "discs": [
            {"label": t.label, "name": t.name, "center": t.center.tolist(), "normal": t.normal.tolist()}
            for t in discs
        ],
        "midline_x": mx,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stamp": stamp, "truth": truth}))
    return truth


def _truth_from_dict(d: dict[str, Any]) -> tuple[list[TruthDisc], float | None]:
    discs = [
        TruthDisc(
            label=int(t["label"]),
            name=str(t["name"]),
            center=np.asarray(t["center"], dtype=np.float64),
            normal=np.asarray(t["normal"], dtype=np.float64),
        )
        for t in d["discs"]
    ]
    return discs, d.get("midline_x")


def sag_x_range(study: Any) -> list[float] | None:
    """LPS x extent (mm) covered by the best sagittal stack, including half a slice at each end."""
    st = study.best("sag")
    if st is None or abs(float(st.normal[0])) < 0.95:
        return None
    pos = np.asarray(st.positions, dtype=np.float64) / float(st.normal[0])
    half = 0.5 * float(st.spacing or st.thickness or 0.0)
    return [float(pos.min() - half), float(pos.max() + half)]


def predict_case(job: dict[str, Any]) -> dict[str, Any]:
    """Worker: label one case (or read it from the cache) and return the JSON record."""
    case = Case(job["name"], job["dataset"], job["meta"])
    cache = Path(job["cache"])
    spider_root = Path(job["spider_root"])
    pred_path = cache / job["fp"] / "pred" / f"{case.name}.json"
    if pred_path.is_file():
        rec = json.loads(pred_path.read_text())
        if "sag_x_range" not in rec:  # records cached before the coverage field existed
            try:
                rec["sag_x_range"] = sag_x_range(_load_study(case, spider_root))
            except Exception:
                rec["sag_x_range"] = None
            pred_path.write_text(json.dumps(rec))
    else:
        rec = {"name": case.name, "dataset": case.dataset, "elapsed_s": None, "anchors": None, "error": None}
        t0 = time.perf_counter()
        try:
            study = _load_study(case, spider_root)
            rec["sag_x_range"] = sag_x_range(study)
            out = resolve_backend(job["backend"])(study)
            anchors = out if isinstance(out, Anchors) else Anchors.from_dict(out)
            rec["anchors"] = anchors.to_dict()
        except Exception as exc:  # a failing case is a result, not a crash of the run
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
        if rec["error"] is None:  # failures are re-tried on the next run, successes are cached
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_text(json.dumps(rec))
    rec["truth"] = (
        _truth_cached(case, spider_root, cache) if case.dataset == "spider" else _truth_dict_phantom(case)
    )
    rec["meta"] = case.meta
    rec["dataset"] = case.dataset
    return rec


def _truth_dict_phantom(case: Case) -> dict[str, Any]:
    discs, mx = _truth(case, Path())
    return {
        "discs": [
            {"label": t.label, "name": t.name, "center": t.center.tolist(), "normal": t.normal.tolist()}
            for t in discs
        ],
        "midline_x": mx,
    }


def render_case(job: dict[str, Any]) -> str:
    """Worker: write the one-page QA PNG of one case; returns the file name."""
    from spinescope.anatomy.qa import case_report

    case = Case(job["name"], job["dataset"], job["meta"])
    rec = job["rec"]
    out = Path(job["out"]) / "png" / f"{case.name}.png"
    try:
        anchors = Anchors.from_dict(rec["anchors"]) if rec.get("anchors") else Anchors()
        discs, mx = _truth_from_dict(rec["truth"])
        study = _load_study(case, Path(job["spider_root"]))
        title = f"{case.name}  {case.meta.get('vendor')} {case.meta.get('field')} {case.meta.get('sequence')}"
        if not anchors.discs:
            anchors.warnings.append("no discs predicted")
        case_report(study, anchors, discs, out_path=str(out), title=title, midline_truth=mx)
    except Exception as exc:  # one unreadable case must not stop the gallery
        log.warning("no QA page for %s: %s", case.name, type(exc).__name__)
        return ""
    return out.name


# --------------------------------------------------------------------------- scoring
@dataclass
class CaseResult:
    case: Case
    score: VolumeScore | None
    anchors: Anchors | None
    elapsed_s: float | None
    error: str | None
    truth: list[TruthDisc]
    sag_x_range: list[float] | None = None
    truth_midline: float | None = None


def score_records(recs: list[dict[str, Any]]) -> list[CaseResult]:
    out: list[CaseResult] = []
    for rec in recs:
        case = Case(rec["name"], rec["dataset"], rec["meta"])
        discs, mx = _truth_from_dict(rec["truth"])
        cov = rec.get("sag_x_range")
        if rec.get("anchors") is None:
            out.append(
                CaseResult(
                    case, None, None, rec.get("elapsed_s"), rec.get("error") or "no output", discs, cov, mx
                )
            )
            continue
        anchors = Anchors.from_dict(rec["anchors"])
        sc = score_anchors(anchors, discs, mx)
        out.append(CaseResult(case, sc, anchors, rec.get("elapsed_s"), rec.get("error"), discs, cov, mx))
    return out


def failure_score(r: CaseResult) -> tuple[float, float]:
    """Severity for ranking the gallery: mean over truth discs of 1 if missed, 0.5 if matched with the
    wrong level name, 0 if correct; ties broken by the midline error. Higher = worse."""
    if r.score is None or r.score.n_truth == 0:
        return (2.0, 0.0)
    s = r.score
    wrong_name = sum(1 for m in s.matches if m.truth is not None and not m.label_correct)
    severity = (len(s.missed) + 0.5 * wrong_name) / s.n_truth
    return (severity, s.midline_err_mm or 0.0)


#: Automatic failure tags, counted per volume (see docs/labeling/lab01_eval.md for the taxonomy).
TAG_NAMES = (
    "no_discs",
    "truth_outside_stack",
    "wrong_midline",
    "midline_at_stack_edge",
    "no_match",
    "partial_column",
    "over_detected",
    "numbering_shift",
    "spurious_lowest_disc",
)


def warning_patterns(r: CaseResult) -> set[str]:
    """The labeller's own warnings for one volume, with numbers replaced so that equal messages group."""
    if r.anchors is None:
        return set()
    return {re.sub(r"(?<![A-Za-z])\d+(\.\d+)?", "N", w)[:90] for w in r.anchors.warnings}


def failure_tags(r: CaseResult) -> list[str]:
    """Heuristic tags for one volume from the scored matches (no image inspection).

    ``truth_outside_stack``: the truth midline is outside the sagittal coverage. ``wrong_midline``: midline
    more than 5 mm from the truth canal; ``midline_at_stack_edge``: a wrong midline in the outer 10% of the
    coverage. ``no_match``: no predicted disc matches
    a truth disc. ``partial_column``: fewer than half the truth discs matched. ``over_detected``: more
    predicted discs than truth discs. ``numbering_shift``: most matched discs carry a non-zero level error.
    ``spurious_lowest_disc``: the most caudal predicted disc matches no truth while other discs did match.
    """
    if r.score is None or r.anchors is None or not r.anchors.discs:
        return ["no_discs"]
    s, tags = r.score, []
    if r.truth_midline is not None and r.sag_x_range is not None:
        lo, hi = r.sag_x_range
        if not lo <= r.truth_midline <= hi:
            tags.append("truth_outside_stack")
    if s.midline_err_mm is not None and s.midline_err_mm > 5.0:
        tags.append("wrong_midline")
        mid = r.anchors.midline_x if r.anchors is not None else None
        if mid is not None and r.sag_x_range is not None and r.sag_x_range[1] > r.sag_x_range[0]:
            lo, hi = r.sag_x_range
            frac = (mid - lo) / (hi - lo)
            if frac < 0.1 or frac > 0.9:
                tags.append("midline_at_stack_edge")
    if s.matched == 0:
        tags.append("no_match")
    elif s.matched < s.n_truth / 2:
        tags.append("partial_column")
    if s.n_pred > s.n_truth:
        tags.append("over_detected")
    offs = [m.level_offset for m in s.matches if m.level_offset is not None]
    if offs and sum(1 for o in offs if o != 0) > len(offs) / 2:
        tags.append("numbering_shift")
    lowest = min(r.anchors.discs, key=lambda k: r.anchors.discs[k].center[2])
    if s.matched and any(m.pred == lowest and m.truth is None for m in s.matches):
        tags.append("spurious_lowest_disc")
    return tags


def _fmt(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _pct(x: float) -> str:
    return "n/a" if not math.isfinite(x) else f"{100 * x:.1f}%"


def _overall(scores: list[VolumeScore]) -> dict[str, float]:
    if not scores:
        return {}
    m = aggregate(scores)
    m.update(aggregate_extended(scores))
    me = [s.midline_err_mm for s in scores if s.midline_err_mm is not None]
    m["midline_err_median_mm"] = float(np.median(me)) if me else float("nan")
    m["midline_err_p90_mm"] = float(np.percentile(me, 90)) if me else float("nan")
    m["numbering_conf_median"] = float(np.median([s.numbering_conf for s in scores]))
    return m


def build_metrics(results: list[CaseResult]) -> dict[str, Any]:
    ok = [r for r in results if r.score is not None]
    scores = [r.score for r in ok if r.score is not None]
    named = [(r.case.name, r.score) for r in ok if r.score is not None]
    meta = {r.case.name: r.case.meta for r in ok}
    breakdowns: dict[str, Any] = {}
    for dim in ("sequence", "field", "vendor"):
        groups = group_scores(named, lambda n, d=dim: meta[n][d])
        breakdowns[dim] = {g: _overall(v) | {"volumes": float(len(v))} for g, v in groups.items()}
    rel = reliability(scores)
    tag_counts = dict.fromkeys(TAG_NAMES, 0)
    for r in results:
        for t in failure_tags(r):
            tag_counts[t] = tag_counts.get(t, 0) + 1
    warn_counts: dict[str, int] = {}
    for r in results:
        for w in warning_patterns(r):
            warn_counts[w] = warn_counts.get(w, 0) + 1
    return {
        "tags": tag_counts,
        "warnings": dict(sorted(warn_counts.items(), key=lambda kv: -kv[1])),
        "volumes": len(results),
        "scored": len(ok),
        "errors": len(results) - len(ok),
        "overall": _overall(scores),
        "breakdown": breakdowns,
        "per_level": per_level_recall(scores),
        "numbering_error_histogram": {str(k): v for k, v in numbering_error_histogram(scores).items()},
        "reliability": rel,
        "seconds_per_case_median": float(np.median([r.elapsed_s for r in results if r.elapsed_s is not None]))
        if results
        else float("nan"),
    }


def _case_row(r: CaseResult) -> dict[str, Any]:
    s = r.score
    row: dict[str, Any] = {
        "case": r.case.name,
        **r.case.meta,
        "error": r.error or "",
        "tags": "|".join(failure_tags(r)),
    }
    if s is None:
        return row
    row.update(
        n_truth=s.n_truth,
        n_pred=s.n_pred,
        matched=s.matched,
        label_correct=s.label_correct,
        centre_median_mm=round(float(np.median(s.centre_err_mm)), 2) if s.centre_err_mm else "",
        midline_err_mm=round(s.midline_err_mm, 2) if s.midline_err_mm is not None else "",
        midline_signed_mm=round(r.anchors.midline_x - r.truth_midline, 2)
        if r.anchors is not None and r.anchors.midline_x is not None and r.truth_midline is not None
        else "",
        numbering_conf=round(s.numbering_conf, 3),
        missed="|".join(s.missed),
        elapsed_s=r.elapsed_s if r.elapsed_s is not None else "",
    )
    return row


# --------------------------------------------------------------------------- markdown and html
def _metrics_table(m: dict[str, float]) -> list[str]:
    rows = [
        ("disc recall", _pct(m.get("recall", float("nan")))),
        ("disc precision", _pct(m.get("precision", float("nan")))),
        ("label accuracy, of matched discs", _pct(m.get("label_accuracy", float("nan")))),
        ("label accuracy, of all truth discs", _pct(m.get("label_accuracy_all_truth", float("nan")))),
        ("off-by-one level, of matched discs", _pct(m.get("off_by_one_of_matched", float("nan")))),
        ("off-by-one level, of all truth discs", _pct(m.get("off_by_one_of_truth", float("nan")))),
        (
            "volumes with every truth disc labelled right",
            _pct(m.get("volumes_all_truth_labelled", float("nan"))),
        ),
        ("false positives per volume", _fmt(m.get("false_positives_per_volume", float("nan")))),
        ("centre error, median (mm)", _fmt(m.get("centre_err_median_mm", float("nan")), 1)),
        ("centre error, 90th percentile (mm)", _fmt(m.get("centre_err_p90_mm", float("nan")), 1)),
        ("share of matched discs within 3 mm", _pct(m.get("centre_within_3mm", float("nan")))),
        ("normal error, median (deg)", _fmt(m.get("normal_err_median_deg", float("nan")), 1)),
        ("midline error, median (mm)", _fmt(m.get("midline_err_median_mm", float("nan")), 1)),
        ("midline error, 90th percentile (mm)", _fmt(m.get("midline_err_p90_mm", float("nan")), 1)),
    ]
    out = ["| metric | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return out


def write_summary(out: Path, meta: dict[str, Any], metrics: dict[str, Any], worst: list[CaseResult]) -> None:
    o = metrics["overall"]
    lines = [
        f"# Labelling evaluation: {meta['label']}",
        "",
        f"- dataset: {meta['dataset']}, split: {meta['split']}; backend: `{meta['backend']}`; "
        f"labeller fingerprint `{meta['fingerprint']}`",
        f"- volumes: {metrics['volumes']} (scored {metrics['scored']}, errors {metrics['errors']})"
        + (f"; subsampled from {meta['available']}" if meta.get("available") else ""),
        f"- median seconds per volume (labeller): {_fmt(metrics['seconds_per_case_median'], 2)}",
        "",
        "## Overall",
        "",
        *_metrics_table(o),
        "",
        "## Breakdown",
        "",
    ]
    for dim, groups in metrics["breakdown"].items():
        lines += [
            f"### By {dim}",
            "",
            "| group | volumes | recall | precision | label acc (truth) | off-by-one (matched) | centre median mm | midline median mm |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for g, v in groups.items():
            lines.append(
                f"| {g} | {int(v['volumes'])} | {_pct(v['recall'])} | {_pct(v['precision'])} | "
                f"{_pct(v['label_accuracy_all_truth'])} | {_pct(v['off_by_one_of_matched'])} | "
                f"{_fmt(v['centre_err_median_mm'], 1)} | {_fmt(v['midline_err_median_mm'], 1)} |"
            )
        lines.append("")
    lines += [
        "## Per-level recall (truth level names)",
        "",
        "| level | truth discs | matched | recall | label acc (matched) |",
        "|---|---|---|---|---|",
    ]
    for lvl, v in metrics["per_level"].items():
        lines.append(
            f"| {lvl} | {int(v['truth'])} | {int(v['matched'])} | {_pct(v['recall'])} | {_pct(v['label_accuracy'])} |"
        )
    lines += [
        "",
        "## Numbering error histogram (matched discs)",
        "",
        "Signed level error = predicted level minus truth level; + = labelled too cranial.",
        "",
        "| level error | discs |",
        "|---|---|",
    ]
    for k, v in metrics["numbering_error_histogram"].items():
        lines.append(f"| {int(k):+d} | {v} |")
    rel = metrics["reliability"]
    lines += [
        "",
        "## Confidence calibration (correct = matched with the exact level name)",
        "",
        f"ECE {_fmt(rel['ece'], 3)}; AUROC of confidence for correctness "
        f"{_fmt(rel['auroc'], 3) if rel['auroc'] is not None else 'n/a'}; predictions {int(rel['n'])}.",
        "",
        "| confidence bin | predictions | mean confidence | accuracy |",
        "|---|---|---|---|",
    ]
    for b in rel["bins"]:
        lines.append(
            f"| {b['lo']:.1f}-{b['hi']:.1f} | {int(b['n'])} | {_fmt(b['mean_confidence'], 2)} | {_fmt(b['accuracy'], 2)} |"
        )
    lines += [
        "",
        "## Automatic failure tags (volumes with the tag)",
        "",
        "| tag | volumes | share |",
        "|---|---|---|",
    ]
    n_vol = max(int(meta["n_volumes"]), 1)
    for tag in TAG_NAMES:
        n = metrics["tags"].get(tag, 0)
        lines.append(f"| {tag} | {n} | {_pct(n / n_vol)} |")
    lines += [
        "",
        "## Labeller warnings (volumes with the message, numbers grouped)",
        "",
        "| warning | volumes | share |",
        "|---|---|---|",
    ]
    for w, n in metrics["warnings"].items():
        lines.append(f"| {html.escape(w)} | {n} | {_pct(n / n_vol)} |")
    lines += [
        "",
        f"## Worst {len(worst)} volumes (severity: missed truth disc 1, wrong level name 0.5, correct 0)",
        "",
        "| volume | severity | truth | matched | centre median mm | midline err mm |",
        "|---|---|---|---|---|---|",
    ]
    for r in worst:
        s = r.score
        if s is None:
            lines.append(f"| {r.case.name} | 2.00 (error: {html.escape(r.error or '')}) | - | - | - | - |")
            continue
        med = _fmt(float(np.median(s.centre_err_mm)), 1) if s.centre_err_mm else "n/a"
        mid = _fmt(s.midline_err_mm, 1) if s.midline_err_mm is not None else "n/a"
        lines.append(
            f"| {r.case.name} | {failure_score(r)[0]:.2f} | {s.n_truth} | {s.matched} | {med} | {mid} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def write_gallery(out: Path, worst: list[CaseResult], label: str) -> None:
    rows = []
    for r in worst:
        if not (out / "png" / f"{r.case.name}.png").is_file():
            continue
        s = r.score
        info = "" if s is None else f"truth {s.n_truth}, matched {s.matched}, label correct {s.label_correct}"
        rows.append(
            f"<figure><figcaption><b>{html.escape(r.case.name)}</b> {html.escape(' '.join(r.case.meta.values()))}"
            f"<br>{html.escape(info)}</figcaption>"
            f'<a href="png/{html.escape(r.case.name)}.png"><img loading="lazy" src="png/{html.escape(r.case.name)}.png"></a></figure>'
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>labelling gallery</title>"
        "<style>body{background:#111;color:#eee;font:14px sans-serif;margin:16px}"
        "figure{display:inline-block;vertical-align:top;margin:0 12px 16px 0;max-width:560px}"
        "img{max-width:100%;border:1px solid #444}</style>"
        f"<h2>Worst cases: {html.escape(label)}</h2>" + "".join(rows)
    )
    (out / "gallery.html").write_text(page)


# --------------------------------------------------------------------------- main
def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out)
    cache = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(args.backend)
    if args.dataset == "spider":
        available = spider_cases(SPIDER_DIR, args.split)
        if not available:
            raise SystemExit(f"no SPIDER volumes with split {args.split!r} under {SPIDER_DIR}")
        cases = _subsample(available, args.subsample)
        if args.case:
            cases = [c for c in available if c.name in set(args.case)]
    else:
        available = phantom_cases(args.n)
        cases = available
    log.info("%d of %d cases, backend %s (fingerprint %s)", len(cases), len(available), args.backend, fp)
    jobs = [
        {
            "name": c.name,
            "dataset": c.dataset,
            "meta": c.meta,
            "backend": args.backend,
            "fp": fp,
            "cache": str(cache),
            "spider_root": str(SPIDER_DIR),
        }
        for c in cases
    ]
    recs = _map(predict_case, jobs, args.jobs)
    results = score_records(recs)
    metrics = build_metrics(results)
    worst = sorted(results, key=failure_score, reverse=True)[: args.gallery_n]
    label = f"{args.backend} on {args.dataset}/{args.split if args.dataset == 'spider' else 'phantom'}"
    meta = {
        "label": label,
        "dataset": args.dataset,
        "split": args.split if args.dataset == "spider" else "synthetic",
        "backend": args.backend,
        "fingerprint": fp,
        "available": len(available) if len(cases) != len(available) else None,
        "n_volumes": len(results),
    }
    (out / "metrics.json").write_text(json.dumps({"meta": meta, **metrics}, indent=1, default=str))
    with open(out / "cases.csv", "w", newline="") as fh:
        rows = [_case_row(r) for r in results]
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_summary(out, meta, metrics, worst)
    if args.gallery_n:
        (out / "png").mkdir(exist_ok=True)
        render_jobs = [
            {
                "name": r.case.name,
                "dataset": r.case.dataset,
                "meta": r.case.meta,
                "rec": next(x for x in recs if x["name"] == r.case.name),
                "out": str(out),
                "spider_root": str(SPIDER_DIR),
            }
            for r in worst
        ]
        names = _map(render_case, render_jobs, args.jobs)
        log.info("rendered %d overlay pages", len(names))
        write_gallery(out, worst, label)
    o = metrics["overall"]
    log.info(
        "recall %s, label accuracy (truth) %s, centre median %s mm",
        _pct(o.get("recall", float("nan"))),
        _pct(o.get("label_accuracy_all_truth", float("nan"))),
        _fmt(o.get("centre_err_median_mm", float("nan")), 1),
    )
    return 0


def _map(fn: Callable[[dict[str, Any]], Any], jobs: list[dict[str, Any]], workers: int) -> list[Any]:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="classical", help="'classical', 'ml' or 'module:function'")
    p.add_argument("--dataset", choices=("spider", "phantom"), default="spider")
    p.add_argument("--split", default="validation", help="SPIDER subset from overview.csv")
    p.add_argument("--n", type=int, default=6, help="number of phantom cases")
    p.add_argument("--case", action="append", default=[], help="restrict to these case names (repeatable)")
    p.add_argument("--subsample", type=int, default=None, help="evaluate an evenly spaced subset of N cases")
    p.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="ONNX Runtime threads per labelling process (default: SPINESCOPE_THREADS, else 1 per pool worker "
        "and ORT's default when --jobs 1; 0 = ORT default)",
    )
    p.add_argument("--gallery-n", type=int, default=30, help="worst cases to render (0 = none)")
    p.add_argument("--out", required=True, help="output folder (keep it outside the repository)")
    p.add_argument("--cache", default="/tmp/spinescope-lab01_eval/cache", help="prediction cache folder")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = PKG_DIR.parents[1]
    if Path(args.out).resolve().is_relative_to(repo):
        raise SystemExit("refusing to write outputs inside the repository; use a folder under /tmp")
    if args.threads is not None:
        # Inherited by the pool workers (spawn copies the environment): one knob for the whole run.
        os.environ[THREADS_ENV] = str(args.threads)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
