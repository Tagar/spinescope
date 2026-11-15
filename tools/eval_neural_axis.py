"""Validation of the neural-axis cyst call (spinescope.analysis.neural_axis) on public data.

Two checks, both study-level (the disc harness in tools/eval_analysis.py does not fit them):

* ``sudirman``: weak labels from the radiologists' free text. A report that names a Tarlov or perineural cyst is a
  positive; a report without that wording is a weak negative (it may simply not mention it). Pre-registered rule:
  the study is called positive when the analysis gives a ``perineural_cyst`` row of at least
  ``neural_axis_cyst_min_mm``. Reports: sensitivity on the positives, specificity on the sampled negatives, Wilson 95%
  intervals. Needs the packaged labeller (for the anchors) and the sagittal STIR/T2 series.
* ``spider``: false-positive burden. SPIDER has no cyst labels, so every call on the validation split is a candidate
  false positive. Uses the SPIDER canal mask (label 100) as the segmentation and the lowest IVD (label 201 = L5-S1) as
  the anchor, so the burden does not depend on the labeller. Reports calls per study and cyst sizes.

Outputs are written as JSON under ``--out``; no images are saved (the data is not committed).

    .venv/bin/python tools/eval_neural_axis.py sudirman --n-negatives 40 --seed 0 --out /tmp/spinescope-dx14_cord_cysts/val
    .venv/bin/python tools/eval_neural_axis.py spider --split validation --out /tmp/spinescope-dx14_cord_cysts/val
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from spinescope.analysis import neural_axis as na
from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.anatomy.ml.grid import intensity_window, plane_grid
from spinescope.anatomy.ml.labeller import CANAL, SegmentedStack
from spinescope.config import DEFAULT
from spinescope.datasets import gradings, spider, sudirman

log = logging.getLogger("eval_neural_axis")

#: Report wording that names a perineural (Tarlov) cyst. Negation is not parsed: the 10 hits were checked by hand.
TARLOV_RE = re.compile(
    r"tarlov|perineural cyst|sacral (?:nerve )?root (?:sheath )?cyst|nerve root sheath", re.I
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (0, 0 when n is 0)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _called(res: na.Result) -> list[dict[str, Any]]:
    return [
        r
        for r in res.rows
        if r.get("scope") == "perineural_cyst" and r["size_mm"] >= DEFAULT.neural_axis_cyst_min_mm
    ]


def sudirman_eval(n_negatives: int, seed: int, out: Path) -> dict[str, Any]:
    from spinescope.anatomy import label_spine
    from spinescope.study import load_study

    notes = sudirman.read_radiologist_notes()
    sessions = sudirman.patient_sessions()
    positives = sorted(n for n, t in notes.items() if TARLOV_RE.search(str(t or "")))
    negatives = sorted(n for n in notes if n not in positives and f"{n:04d}" in sessions)
    rng = random.Random(seed)
    sampled_neg = sorted(rng.sample(negatives, min(n_negatives, len(negatives))))
    cases = [(n, True) for n in positives if f"{n:04d}" in sessions] + [(n, False) for n in sampled_neg]
    rows: list[dict[str, Any]] = []
    for n, is_pos in cases:
        path = sessions[f"{n:04d}"][0]
        try:
            study = load_study(path)
            anchors = label_spine(study)
            res = na.analyze(study, anchors, DEFAULT)
            calls = _called(res)
            rows.append(
                {
                    "case": n,
                    "reference": is_pos,
                    "called": bool(calls),
                    "max_size_mm": max((r["size_mm"] for r in calls), default=None),
                    "warnings": len(res.warnings),
                }
            )
        except Exception as exc:  # one bad study must not stop the run
            log.exception("case %s failed", n)
            rows.append({"case": n, "reference": is_pos, "error": f"{type(exc).__name__}: {exc}"})
    ok = [r for r in rows if "error" not in r]
    tp = sum(1 for r in ok if r["reference"] and r["called"])
    fn = sum(1 for r in ok if r["reference"] and not r["called"])
    fp = sum(1 for r in ok if not r["reference"] and r["called"])
    tn = sum(1 for r in ok if not r["reference"] and not r["called"])
    summary = {
        "positives_in_reports": len(positives),
        "negatives_sampled": len(sampled_neg),
        "negatives_available": len(negatives),
        "evaluated": len(ok),
        "errors": len(rows) - len(ok),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "sensitivity_ci95": wilson(tp, tp + fn),
        "specificity": tn / (tn + fp) if tn + fp else None,
        "specificity_ci95": wilson(tn, tn + fp),
        "call_rate_negatives": fp / (fp + tn) if fp + tn else None,
        "rule": "positive if a perineural_cyst row with size >= neural_axis_cyst_min_mm",
        "caveat": "negatives are reports without Tarlov wording (weak negatives); only the sampled subset is run",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "sudirman_cysts.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, default=str)
    )
    return summary


def _spider_seg(case: spider.SpiderStudy, seq: str) -> tuple[SegmentedStack | None, Anchors | None, str]:
    stack = case.volume(seq)
    mask = case.masks[seq]
    if mask.shape != stack.vol.shape:
        return None, None, f"mask shape {mask.shape} differs from the image {stack.vol.shape}"
    h, w = stack.vol.shape[1], stack.vol.shape[2]
    grids = [plane_grid(stack, k, 1.0) for k in range(stack.n)]
    gh, gw = grids[0].shape
    probs = np.zeros((stack.n, 4, gh, gw), np.float32)
    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    for k, g in enumerate(grids):
        # nearest image pixel for each grid point (the masks share the image grid)
        rows, cols = g.index_of_mm(*g.mm_of_index(ii, jj))
        r = np.clip(np.rint(rows).astype(int), 0, h - 1)
        c = np.clip(np.rint(cols).astype(int), 0, w - 1)
        lab = mask[k][r, c]
        probs[k, CANAL] = lab == spider.CANAL_LABEL
    disc_lab = spider.DISC_LABEL_OFFSET + 1  # IVD 1 is the lowest disc (L5-S1)
    if not np.any(mask == disc_lab):
        return None, None, "no lowest IVD label in the mask"
    zs, ys = [], []
    for k in range(stack.n):
        sl = np.argwhere(mask[k] == disc_lab)
        if sl.size == 0:
            continue
        pts = stack.pix_to_lps_many(k, sl.astype(float))  # image pixels, not grid pixels
        zs.append(pts[:, 2].mean())
        ys.append(pts[:, 1].mean())
    if not zs:
        return None, None, "lowest IVD not on any slice"
    height = float(np.ptp(zs)) if len(zs) > 1 else 8.0
    anchors = Anchors(
        discs={
            "L5-S1": Disc(
                center=[0.0, float(np.mean(ys)), float(np.mean(zs))],
                normal=[0.0, 0.0, 1.0],
                height_mm=max(height, 8.0),
            )
        }
    )
    seg = SegmentedStack(stack=stack, grids=grids, probs=probs, window=intensity_window(stack.vol))
    return seg, anchors, ""


def spider_eval(split: str, out: Path) -> dict[str, Any]:
    cases = gradings.spider_subset_cases(split)
    rows: list[dict[str, Any]] = []
    for cid in cases:
        try:
            case = spider.load_spider_study(cid)
            seq = next((s for s in ("t2", "t2_SPACE") if s in case.sequences), None)
            if seq is None:
                rows.append({"case": cid, "skipped": "no sagittal T2"})
                continue
            seg, anchors, why = _spider_seg(case, seq)
            if seg is None or anchors is None:
                rows.append({"case": cid, "seq": seq, "skipped": why})
                continue
            rows_c, warns, _ = na.sagittal_cysts(seg, anchors, DEFAULT, contrast="t2")
            calls = [r for r in rows_c if r["size_mm"] >= DEFAULT.neural_axis_cyst_min_mm]
            # a study with no call is only a true negative if the search was assessed (not "not assessed")
            assessed = not any("not assessed" in w for w in warns)
            rows.append(
                {
                    "case": cid,
                    "seq": seq,
                    "assessed": assessed,
                    "calls": len(calls),
                    "max_size_mm": max((r["size_mm"] for r in calls), default=None),
                    "reason": warns[0] if warns else "",
                }
            )
        except Exception as exc:
            log.exception("spider case %s failed", cid)
            rows.append({"case": cid, "error": f"{type(exc).__name__}: {exc}"})
    done = [r for r in rows if "calls" in r and r["assessed"]]
    n_calls_study = sum(1 for r in done if r["calls"] > 0)
    k = n_calls_study
    summary = {
        "split": split,
        "cases": len(cases),
        "evaluated": len(done),
        "not_assessed": sum(1 for r in rows if "calls" in r and not r["assessed"]),
        "studies_with_a_call": k,
        "call_rate": k / len(done) if done else None,
        "call_rate_ci95": wilson(k, len(done)),
        "calls_per_study": sum(r["calls"] for r in done) / len(done) if done else None,
        "sizes_mm": sorted(r["max_size_mm"] for r in done if r["calls"] > 0),
        "skipped": sum(1 for r in rows if "skipped" in r),
        "errors": sum(1 for r in rows if "error" in r),
        "caveat": "no cyst labels in SPIDER: every call is a candidate false positive",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"spider_{split}_cysts.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, default=str)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="which", required=True)
    s = sub.add_parser("sudirman")
    s.add_argument("--n-negatives", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("spider")
    sp.add_argument("--split", default="validation", choices=["training", "validation"])
    sp.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.which == "sudirman":
        summary = sudirman_eval(args.n_negatives, args.seed, args.out)
    else:
        summary = spider_eval(args.split, args.out)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
