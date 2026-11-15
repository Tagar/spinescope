#!/usr/bin/env python3
"""Geometric reference for the alignment analysis: listhesis measured on SPIDER labeller output vs on ground-truth masks.

For every validation disc whose predicted level name equals the reference name (the same matching rule as the harness),
the translation is measured twice, with the same functions and on the same slices:

* predicted: ``spinescope.analysis.alignment`` on the labeller's body endplates (the production path);
* reference: the same listhesis on the SPIDER ground-truth vertebra masks (``body_plane_geometry`` on the labeller's plane
  grid of the same slice), so any bias of the measurement method cancels out of the comparison.

Reported: MAE (mm) of the predicted against the reference translation, the signed bias, and the agreement at the 3 mm
cut-off. Robustness: the reference translation is also measured on the neighbouring slices (k-1, k+1); the median
change across slices is reported. The SPIDER spondylolisthesis label is compared with the reference translation as an
oracle (what the geometry would say if the labeller were perfect); it is not a score of the labeller.

The reference cannot reveal a bias of the endplate fit itself (the same fit is used on both sides; see
docs/analysis/alignment.md, "Endplate fit, steep bodies").

Example::

    .venv/bin/python tools/eval_alignment.py --split validation --jobs 3 --out /tmp/spinescope-dx04_listhesis/geo

Outputs (in ``--out``, outside the repository): ``cases.json``, ``pairs.csv``, ``summary.json``. Only case numbers, level
names and measurements appear in them; SPIDER paths are never written.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

log = logging.getLogger("eval_alignment")

DISC_BASE = 200
MATCH_MM = 12.0
CUT_MM = 3.0  # alignment_listhesis_min_mm (pre-registered, docs/analysis/alignment.md)


def _root() -> Path:
    return Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public")) / "spider"


def _mean_or_none(arr: np.ndarray, label: int, vol: Any) -> np.ndarray | None:
    m = arr == label
    if not m.any():
        return None
    return vol.lps_of_voxels(m).mean(axis=0)


def _gt_body_endplates(
    vol: Any, label: int, stack: Any, k: int, sup: bool, inf: bool
) -> dict[str, Any] | None:
    """Endplates of the ground-truth vertebra ``label`` on slice ``k`` of the labeller's stack (LPS records)."""
    from spinescope.anatomy.canal_ml import largest_component
    from spinescope.anatomy.ml.grid import plane_grid
    from spinescope.anatomy.vertebrae import body_plane_geometry
    from spinescope.config import DEFAULT

    grid = plane_grid(stack, k, 1.0)
    ii, jj = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing="ij")
    down, post = grid.mm_of_index(ii, jj)
    lab = vol.labels_at(grid.to_lps(down, post))
    mask = largest_component(lab == label)
    if not mask.any():
        return None
    plane = body_plane_geometry(mask, grid, sup=sup, inf=inf, cfg=DEFAULT)
    if plane is None:
        return None
    out: dict[str, Any] = {}
    for key, name in (("sup", "superior_endplate"), ("inf", "inferior_endplate")):
        if key not in plane:
            continue
        e = plane[key]
        rec: dict[str, Any] = {
            "anterior": [float(x) for x in grid.to_lps(*e["anterior"])],
            "posterior": [float(x) for x in grid.to_lps(*e["posterior"])],
        }
        if "anterior_wall" in e and "posterior_wall" in e:
            rec["anterior_wall"] = [float(x) for x in grid.to_lps(*e["anterior_wall"])]
            rec["posterior_wall"] = [float(x) for x in grid.to_lps(*e["posterior_wall"])]
        out[name] = rec
    return out


def _gt_translation(
    vol: Any, stack: Any, lower_label: int, upper_label: int, k_up: int, k_lo: int
) -> float | None:
    """Translation (anterior positive) from the ground-truth masks at the given slices, same listhesis function."""
    from spinescope.analysis.alignment import endplate_from_record, listhesis
    from spinescope.config import DEFAULT

    up = _gt_body_endplates(vol, upper_label, stack, k_up, sup=True, inf=True)
    lo = _gt_body_endplates(vol, lower_label, stack, k_lo, sup=True, inf=True)
    if not up or not lo:
        return None
    up_inf = endplate_from_record(up.get("inferior_endplate"))
    lo_sup = endplate_from_record(lo.get("superior_endplate"))
    if up_inf is None or lo_sup is None:
        return None
    return float(listhesis(up_inf, lo_sup, DEFAULT)["translation_mm"])


def _case(args: tuple[str, str]) -> dict[str, Any]:
    """Predicted and reference translations for one SPIDER validation case."""
    case, root = args
    from eval_vertebrae import TruthVolume  # type: ignore[import-not-found]

    from spinescope.analysis import alignment
    from spinescope.anatomy.pipeline import label_spine
    from spinescope.datasets.gradings import ivd_level_name, load_spider_gradings
    from spinescope.datasets.spider import load_spider_studies

    out: dict[str, Any] = {"case": case, "pairs": [], "warnings": []}
    sds = load_spider_studies(root, cases=[case])
    if not sds:
        out["warnings"].append("case not found")
        return out
    sd = sds[0]
    ref = "t2" if "t2" in sd.mask_paths else next(iter(sd.mask_paths))
    study = sd.to_study(None)
    anchors = label_spine(study)
    res = alignment.analyze(study, anchors)
    stack = alignment._stack_for(study, anchors)
    if stack is None:
        out["warnings"].append("no sagittal stack")
        return out
    vol = TruthVolume(sd.mask_paths[ref])
    gradings = {g.ivd_label: g for g in load_spider_gradings(root).get(case, [])}
    gt_disc: dict[int, np.ndarray] = {}
    for i in range(1, 30):
        c = _mean_or_none(vol.arr, DISC_BASE + i, vol)
        if c is not None:
            gt_disc[i] = c
    rows = {str(r["disc"]): r for r in res.rows if r.get("scope") == "level"}
    used: set[int] = set()
    for lab in sorted(anchors.discs, key=lambda k: -anchors.discs[k].center[2]):
        pc = np.asarray(anchors.discs[lab].center)[1:]
        best: tuple[float, int] | None = None
        for i, c in gt_disc.items():
            if i in used:
                continue
            d = float(np.linalg.norm(pc - c[1:]))
            if d <= MATCH_MM and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            continue
        i = best[1]
        used.add(i)
        if ivd_level_name(i) != lab:
            out["pairs"].append({"level": lab, "ivd": i, "name_aligned": False})
            continue
        row = rows.get(lab)
        if row is None or row.get("translation_mm") is None:
            out["pairs"].append({"level": lab, "ivd": i, "name_aligned": True, "pred_mm": None})
            continue
        k_up, k_lo = int(row["slice_upper"]), int(row["slice_lower"])
        upper, lower = i, i - 1
        gt = _gt_translation(vol, stack, lower, upper, k_up, k_lo)
        neigh = []
        for dk in (-1, 1):
            if 0 <= k_up + dk < stack.n and 0 <= k_lo + dk < stack.n:
                t = _gt_translation(vol, stack, lower, upper, k_up + dk, k_lo + dk)
                if t is not None:
                    neigh.append(t)
        g = gradings.get(i)
        out["pairs"].append(
            {
                "level": lab,
                "ivd": i,
                "name_aligned": True,
                "pred_mm": float(row["translation_mm"]),
                "gt_mm": gt,
                "gt_neighbour_mm": neigh,
                "spondylolisthesis": (g.value("spondylolisthesis") if g is not None else None),
                "pred_status": row.get("name_status"),
            }
        )
    return out


def _metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    both = [p for p in pairs if p.get("pred_mm") is not None and p.get("gt_mm") is not None]
    if not both:
        return {"n": 0}
    pred = np.array([p["pred_mm"] for p in both], dtype=float)
    gt = np.array([p["gt_mm"] for p in both], dtype=float)
    err = pred - gt
    agree = (np.abs(pred) >= CUT_MM) == (np.abs(gt) >= CUT_MM)
    neigh = [abs(p["gt_mm"] - n) for p in both for n in p.get("gt_neighbour_mm", [])]
    return {
        "n": len(both),
        "mae_mm": round(float(np.mean(np.abs(err))), 3),
        "median_abs_mm": round(float(np.median(np.abs(err))), 3),
        "bias_mm": round(float(np.mean(err)), 3),
        "rmse_mm": round(float(np.sqrt(np.mean(err**2))), 3),
        "cut_agreement_3mm": round(float(np.mean(agree)), 3),
        "gt_neighbour_median_change_mm": round(float(np.median(neigh)), 3) if neigh else None,
        "gt_neighbour_pairs": len(neigh),
    }


#: Thresholds on |translation| (mm) for the sweep. Chosen on training; see docs/analysis/alignment.md.
SWEEP_MM: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def sweep_from_pairs(pairs_csv: str, out: str) -> int:
    """Threshold sweep of the SPIDER spondylolisthesis predictor from an ``eval_analysis.py --pairs`` file.

    Reads the scored pairs of field ``spondylolisthesis`` (score = |translation| mm, truth 0/1) and writes ``sweep.json``.
    Intervals resample cases (patients), 1000 resamples, seed 0.
    """
    from spinescope.analysis.alignment import threshold_sweep

    with open(pairs_csv, newline="") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r.get("field") == "spondylolisthesis"
            and r.get("status") == "scored"
            and r.get("score") not in ("", None)
        ]
    scores = [float(r["score"]) for r in rows]
    truth = [int(float(r["truth"])) for r in rows]
    cases = [str(r["case"]) for r in rows]
    table = threshold_sweep(scores, truth, cases, SWEEP_MM, n_boot=1000, seed=0)
    result = {
        "pairs_csv_scored": len(rows),
        "positives": int(sum(truth)),
        "positive_cases": sorted({c for c, t in zip(cases, truth, strict=True) if t == 1}, key=int),
        "thresholds": table,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1, allow_nan=False))
    print(json.dumps(result, indent=1))
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="validation", choices=("training", "validation"))
    p.add_argument("--n", type=int, help="first N cases in numeric order (subsample; the report says so)")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--out", help="output folder (outside the repository)")
    p.add_argument(
        "--sweep-pairs",
        help="instead of running: threshold sweep of an eval_analysis.py pairs.csv; writes --out/sweep.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse(argv)
    if args.sweep_pairs:
        if not args.out:
            raise SystemExit("--out is required with --sweep-pairs")
        return sweep_from_pairs(args.sweep_pairs, os.path.join(args.out, "sweep.json"))
    if not args.out:
        raise SystemExit("--out is required")
    from spinescope.datasets.gradings import spider_subset_cases

    root = str(_root())
    cases = spider_subset_cases(args.split, root)
    total = len(cases)
    if args.n:
        cases = cases[: args.n]
    jobs = max(1, args.jobs)
    work = [(c, root) for c in cases]
    results: list[dict[str, Any]] = []
    if jobs == 1:
        results = [_case(w) for w in work]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_case, work))
    pairs = [dict(case=r["case"], **p) for r in results for p in r["pairs"]]
    summary = {
        "split": args.split,
        "split_cases": total,
        "cases_scored": len(cases),
        "subsampled": len(cases) < total,
        "reference": "SPIDER ground-truth vertebra masks, same listhesis function, same slices",
        "cut_mm": CUT_MM,
        "all": _metrics(pairs),
        "by_name_status": {
            s: _metrics([p for p in pairs if p.get("pred_status") == s])
            for s in ("confirmed", "probable", "unconfirmed")
        },
        "name_mismatch_dropped": sum(1 for p in pairs if not p.get("name_aligned")),
        "not_measured": sum(1 for p in pairs if p.get("name_aligned") and p.get("pred_mm") is None),
    }
    out = Path(os.fspath(args.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(results, indent=1, allow_nan=False, default=str))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False, default=str))
    with open(out / "pairs.csv", "w", newline="") as f:
        fields = [
            "case",
            "level",
            "ivd",
            "name_aligned",
            "pred_mm",
            "gt_mm",
            "gt_neighbour_mm",
            "spondylolisthesis",
            "pred_status",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for p in pairs:
            w.writerow(p)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
