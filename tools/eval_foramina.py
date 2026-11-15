#!/usr/bin/env python3
"""Validate the foramina analysis: Sudirman radiologist reports (weak labels) and SPIDER geometry sanity.

Examples::

    .venv/bin/python tools/eval_foramina.py --dataset sudirman --out /tmp/spinescope-dx07_foramina/sudirman
    .venv/bin/python tools/eval_foramina.py --dataset spider --split training --out /tmp/spinescope-dx07_foramina/spider

Decision rule (pre-registered in docs/analysis/foramina.md before the first run; do not edit after seeing numbers):

* Truth for (disc, side) from the report's foraminal labels (zone "foraminal", a side given):
  positive = an explicit positive foraminal label on that side at that level; negative = an explicit negated
  foraminal label at that level on that side (or unsided) and no positive label at that level on that side or
  unsided. Everything else (unsided positives, no mention) is excluded and counted.
* Scored pair = (patient, disc, side) with an assessed foramen and a truth. Disc names are the anchors' own names
  (no shift); a name disagreement is not scored.
* Operating points: P1 positive if >= 2 directions of perineural fat obliteration (proxy); P2 positive if the
  foraminal height is <= 15 mm (Hasegawa 1995 critical height).
* Metrics: sensitivity, specificity, PPV, NPV, AUC (score: -height for P2, n_obliterated for P1), with percentile
  bootstrap 95% CIs resampling patients (seed 0, 1000 draws). Coverage = assessed / all (disc, side) of the labelled
  discs. Nothing is dropped silently; every count is written.

SPIDER has no foraminal grading. Its run reports coverage and the foraminal height distribution per level against the
Hasegawa 1995 normal group (geometry sanity only; SPIDER masks are not used for foramina).

Outputs (``metrics.json``, ``metrics.md``, ``pairs.csv``) carry case numbers and level names only, never paths or
report text. Keep ``--out`` outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from spinescope.analysis import foramina
from spinescope.anatomy import label_spine
from spinescope.anatomy.anchors import Anchors
from spinescope.config import DEFAULT
from spinescope.datasets import sudirman as sd
from spinescope.datasets.report_labels import parse_report
from spinescope.study import load_study

P1_MIN_DIRECTIONS = 2
BOOT = 1000
SEED = 0
SIDE_NONE = None
_SIDE_CODE = {"left": "L", "lt": "L", "right": "R", "rt": "R"}


def _truth(labels: Sequence[Any]) -> dict[tuple[str, str], bool]:
    """(disc, side) -> True / False from the foraminal labels, per the pre-registered rule; others are absent."""
    pos: set[tuple[str, str]] = set()
    unsided_pos: set[str] = set()
    neg: set[tuple[str, str | None]] = set()
    for lab in labels:
        if lab.field != "stenosis" or lab.zone != "foraminal" or lab.level is None:
            continue
        side = _SIDE_CODE.get(
            str(lab.side or "").lower()
        )  # the parser says left/right; the analysis uses L/R
        if lab.value is True and not lab.negated:
            if side is not None:
                pos.add((lab.level, side))
            else:
                unsided_pos.add(lab.level)
        elif lab.negated or lab.value is False:
            neg.add((lab.level, side))
    out: dict[tuple[str, str], bool] = {}
    for lv in {lv for lv, _ in pos | neg}:
        for side in foramina.SIDES:
            if (lv, side) in pos:
                out[(lv, side)] = True
            elif lv in unsided_pos:
                continue  # an unsided positive could mean either side: not scored
            elif (lv, side) in neg or (lv, SIDE_NONE) in neg:
                out[(lv, side)] = False
    return out


def _job(args: tuple[str, list[str], str | None, str]) -> dict[str, Any]:
    """Worker: label one case (anchors cached as JSON), run the foramina analysis, parse its truth. Errors are recorded."""
    patient, paths, text, cache = args
    rec: dict[str, Any] = {"case": patient, "rows": [], "error": None, "session_found": False}
    try:
        study = None
        for sess in paths:
            st = load_study(Path(sess))
            if st.find("sag", "t1", slabs=False):
                study = st
                break
        if study is None:
            rec["error"] = "no sagittal T1 series"
            return rec
        rec["session_found"] = True
        cache_path = Path(cache) / f"{patient}.json"
        if cache_path.is_file():
            anchors = Anchors.from_json(str(cache_path))
        else:
            anchors = label_spine(study)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            anchors.to_json(str(cache_path))
        res = foramina.analyze(study, anchors, DEFAULT)
        rec["numbering"] = anchors.numbering.get("status")
        rec["discs"] = list(anchors.discs)
        rec["warnings"] = list(res.warnings)[:3]
        keep = (
            "disc",
            "side",
            "status",
            "reason",
            "foraminal_height_mm",
            "ap_width_mm",
            "foraminal_fat_area_mm2",
            "n_obliterated",
            "lee_grade",
            "perineural_status",
            "lateral_mm",
            "edge_plane",
            "score_margin",
            "centre_fat_index",
            "confidence",
            "rims_found",
        )
        rec["rows"] = [{k: r.get(k) for k in keep} for r in res.rows]
        if text is not None:
            rec["truth"] = {f"{d}|{s}": v for (d, s), v in _truth(parse_report(text).labels).items()}
    except Exception as exc:  # contain one case: it is counted as an error
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _auc(y: Sequence[int], s: Sequence[float]) -> float:
    pos = [v for yy, v in zip(y, s, strict=False) if yy == 1]
    neg = [v for yy, v in zip(y, s, strict=False) if yy == 0]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def _pred(p: dict[str, Any], point: str) -> int:
    if point == "P1":
        return int((p["n_obliterated"] or 0) >= P1_MIN_DIRECTIONS)
    h = p["foraminal_height_mm"]
    return int(h is not None and h <= DEFAULT.foramen_critical_height_mm)


def _metrics(pairs: list[dict[str, Any]], point: str) -> dict[str, float]:
    y = [p["truth"] for p in pairs]
    yhat = [_pred(p, point) for p in pairs]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, yhat, strict=False))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, yhat, strict=False))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, yhat, strict=False))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, yhat, strict=False))

    def div(a: int, b: int) -> float:
        return a / b if b else float("nan")

    if point == "P1":
        score = [float(p["n_obliterated"] or 0) for p in pairs]
    else:
        score = [-float(p["foraminal_height_mm"] or 0.0) for p in pairs]
    return {
        "n": len(pairs),
        "positives": tp + fn,
        "negatives": tn + fp,
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "auc": _auc(y, score),
    }


def _bootstrap(pairs: list[dict[str, Any]], point: str) -> dict[str, list[float]]:
    """Percentile 95% CIs resampling patients (cases), not discs."""
    rng = random.Random(SEED)
    cases = sorted({p["case"] for p in pairs})
    by_case: dict[str, list[dict[str, Any]]] = {c: [] for c in cases}
    for p in pairs:
        by_case[p["case"]].append(p)
    draws: dict[str, list[float]] = {k: [] for k in ("sensitivity", "specificity", "ppv", "npv", "auc")}
    for _ in range(BOOT if cases else 0):
        sample_pairs = [q for _ in cases for q in by_case[cases[rng.randrange(len(cases))]]]
        m = _metrics(sample_pairs, point)
        for k in draws:
            if math.isfinite(m[k]):
                draws[k].append(m[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
        for k, v in draws.items()
    }


def _flatten(recs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scored pairs and the counts that explain them."""
    counts: dict[str, Any] = {
        "cases": len(recs),
        "case_errors": 0,
        "foramina_rows": 0,
        "assessed": 0,
        "not_assessed": 0,
        "not_assessed_reasons": {},
        "rows_without_truth": 0,
        "scored": 0,
        "truth_pairs": 0,
        "truth_name_not_in_anchors": 0,
    }
    pairs: list[dict[str, Any]] = []
    for rec in recs:
        if rec.get("error"):
            counts["case_errors"] += 1
            kinds = counts.setdefault("case_error_types", {})
            kind = str(rec["error"]).split(":")[0]
            kinds[kind] = kinds.get(kind, 0) + 1
            continue
        truth = rec.get("truth", {})
        discs = set(rec.get("discs", []))
        counts["truth_pairs"] += len(truth)
        counts["truth_name_not_in_anchors"] += sum(1 for k in truth if k.split("|")[0] not in discs)
        for r in rec["rows"]:
            counts["foramina_rows"] += 1
            if r["status"] != "assessed":
                counts["not_assessed"] += 1
                why = str(r["reason"]).split(":")[0]
                counts["not_assessed_reasons"][why] = counts["not_assessed_reasons"].get(why, 0) + 1
                continue
            counts["assessed"] += 1
            key = f"{r['disc']}|{r['side']}"
            if key not in truth or r["disc"] not in discs:
                counts["rows_without_truth"] += 1
                continue
            counts["scored"] += 1
            pairs.append(
                {**r, "case": rec["case"], "truth": int(truth[key]), "numbering": rec.get("numbering")}
            )
    return pairs, counts


def _sudirman_jobs(limit: int | None, cache: str) -> list[tuple[str, list[str], str | None, str]]:
    notes = sd.read_radiologist_notes()
    sess = sd.patient_sessions()
    jobs: list[tuple[str, list[str], str | None, str]] = []
    for patient in sorted(sess):
        text = notes.get(int(patient))
        if text is None:
            continue  # no report for this patient: nothing to score against
        jobs.append((patient, [str(p) for p in sess[patient]], text, cache))
        if limit and len(jobs) >= limit:
            break
    return jobs


def _clean(x: Any) -> Any:
    """NaN and infinity become null (JSON has neither)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _md(report: dict[str, Any]) -> str:
    lines = [
        "# Foramina validation",
        "",
        f"Dataset: {report['dataset']}",
        "",
        "```json",
        json.dumps(report["counts"], indent=1, default=str),
        "```",
        "",
    ]
    for point, m in report.get("operating_points", {}).items():
        lines += [f"## {point}", "", "| metric | value | 95% CI |", "|---|---|---|"]
        for k in ("sensitivity", "specificity", "ppv", "npv", "auc"):
            v = m[k]
            ci = m["ci95"][k]
            lines.append(f"| {k} | {v:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
        lines += ["", f"n = {m['n']}, positives = {m['positives']}, negatives = {m['negatives']}", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("sudirman", "spider"), default="sudirman")
    p.add_argument("--split", default="validation", choices=("training", "validation"))
    p.add_argument("--limit", type=int, help="first N cases in order (reported as subsampled)")
    p.add_argument("--workers", type=int, default=2, help="processes (at most 2 on the shared machine)")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--anchors", default="/tmp/spinescope-dx07_foramina/anchors", help="anchor JSON cache folder"
    )
    args = p.parse_args(argv)
    # shared machine: two threads per labeller process, at most two processes
    os.environ["SPINESCOPE_THREADS"] = "2"
    args.workers = max(1, min(args.workers, 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = args.anchors
    if args.dataset == "sudirman":
        jobs = _sudirman_jobs(args.limit, os.path.join(cache, "sudirman"))
    else:
        from spinescope.datasets.gradings import spider_dir, spider_subset_cases

        root = spider_dir()
        ids = spider_subset_cases(args.split, root)
        if args.limit:
            ids = ids[: args.limit]
        jobs = [
            (
                str(c),
                [str(root / "images" / f"{c}_t1.mha")],
                None,
                os.path.join(cache, f"spider_{args.split}"),
            )
            for c in ids
        ]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(_job, jobs, chunksize=1))
    pairs, counts = _flatten(recs)
    counts["subsampled"] = bool(args.limit)
    counts["jobs_total"] = len(jobs)
    report: dict[str, Any] = {
        "dataset": f"{args.dataset}/{args.split}" if args.dataset == "spider" else "sudirman",
        "counts": counts,
        "operating_points": {},
    }
    if args.dataset == "sudirman":
        for point in ("P1", "P2"):
            m = _metrics(pairs, point)
            m["ci95"] = _bootstrap(pairs, point)
            report["operating_points"][point] = m
    else:
        heights: dict[str, list[float]] = {}
        for rec in recs:
            for r in rec.get("rows", []):
                if r["status"] == "assessed" and r["foraminal_height_mm"] is not None:
                    heights.setdefault(r["disc"], []).append(float(r["foraminal_height_mm"]))
        report["spider_height_by_level"] = {
            lv: {
                "n": len(v),
                "median_mm": float(np.median(v)),
                "mean_mm": float(np.mean(v)),
                "normal_mean_mm": (foramina.normal_height(lv, DEFAULT) or (None, None))[0],
            }
            for lv, v in sorted(heights.items())
        }
    (out / "metrics.json").write_text(json.dumps(_clean(report), indent=1, default=str, allow_nan=False))
    (out / "metrics.md").write_text(_md(report))
    with open(out / "pairs.csv", "w", newline="") as f:
        if pairs:
            keys = [k for k in pairs[0] if k != "warnings"]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(pairs)
    print((out / "metrics.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
