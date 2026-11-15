#!/usr/bin/env python
"""Validate the axial ligamentum flavum (LF) measurement on the Sudirman reference slices (dx09).

For every labelled disc (D3-D5 = L3-L4, L4-L5, L5-S1) the axial T1 slice of the label map is measured with the same
pixel-space core as the analysis (``spinescope.analysis.ligamentum.measure_rays``). The rays start at the dural sac
(label 150) and point posterolaterally from the posterior direction (disc centroid, label 50, to the posterior
elements, label 100). Two references are compared:

* ``label_mm``: the thickness of the interval between the sac and the posterior elements (label 200, the area between
  the anterior and posterior elements) along the same rays. Geometric proxy for the LF, not a truth.
* the radiologists' report: ``ligamentum_flavum_thickening`` from ``datasets.report_labels`` (rule-based parse).

Operating point and rules are pre-registered in ``docs/analysis/ligamentum.md`` before this tool was run on the full set.
Patient-level bootstrap (1000 resamples, seed 0). Only derived numbers are written; no report text, no image.

Usage::

    SPINESCOPE_THREADS=2 .venv/bin/python tools/eval_ligamentum.py --workers 2 --out /tmp/spinescope-dx09_lf_epidural/val
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from spinescope.analysis import ligamentum as LF
from spinescope.config import DEFAULT
from spinescope.datasets import sudirman as S
from spinescope.datasets.report_labels import aggregate_levels, parse_report

log = logging.getLogger("eval_ligamentum")

BOOT_N = 1000
SEED = 0
LABEL_SAC = S.THECAL_SAC_VALUE  # 150
LABEL_POST = S.POSTERIOR_ELEMENT_VALUE  # 100
LABEL_IVD = S.IVD_VALUE  # 50
LABEL_AAP = S.AAP_VALUE  # 200
FAT_MIN_MM = DEFAULT.ligamentum_epidural_fat_mm


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    rr, cc = np.nonzero(mask)
    return float(rr.mean()), float(cc.mean())


def _rotate(d: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    t = math.radians(angle_deg)
    dr, dc = d
    return math.cos(t) * dr - math.sin(t) * dc, math.sin(t) * dr + math.cos(t) * dc


def label_band_mm(
    mask: np.ndarray,
    row_mm: float,
    col_mm: float,
    centre_rc: tuple[float, float],
    direction_mm: tuple[float, float],
    radius_mm: float,
    cfg: Any = DEFAULT,
) -> float | None:
    """Length of the first run of label-200 pixels along one ray, from the dural edge outward (mm); None if absent."""
    step = cfg.ligamentum_step_mm
    s = np.arange(0.0, radius_mm + cfg.ligamentum_search_mm + step / 2, step)
    rr = centre_rc[0] + s * direction_mm[0] / row_mm
    cc = centre_rc[1] + s * direction_mm[1] / col_mm
    vals = ndi.map_coordinates((mask == LABEL_AAP).astype(float), [rr, cc], order=0, mode="nearest")
    i0 = round(radius_mm / step)
    seg = vals[i0:]
    if not np.any(seg > 0.5):
        return None
    start = int(np.argmax(seg > 0.5))
    end = start
    while end + 1 < seg.size and seg[end + 1] > 0.5:
        end += 1
    return float((end - start + 1) * step)


def measure_level(
    img: np.ndarray, mask: np.ndarray, row_mm: float, col_mm: float, cfg: Any = DEFAULT
) -> dict[str, Any]:
    """The analysis measured on one reference slice, with the label-200 reference along the same rays."""
    sac = mask == LABEL_SAC
    ivd = mask == LABEL_IVD
    post = mask == LABEL_POST
    if not (sac.any() and ivd.any() and post.any()):
        return {"status": "no_geometry"}
    centre = _centroid(sac)
    radius_mm = math.sqrt(float(sac.sum()) * row_mm * col_mm / math.pi)
    c_ivd, c_post = _centroid(ivd), _centroid(post)
    d = ((c_post[0] - c_ivd[0]) * row_mm, (c_post[1] - c_ivd[1]) * col_mm)
    norm = math.hypot(*d)
    if norm < 1e-9:
        return {"status": "no_direction"}
    d = (d[0] / norm, d[1] / norm)
    rays = [(a, _rotate(d, a)) for a in LF.ray_angles(cfg)]
    meas = LF.measure_rays(img, row_mm, col_mm, centre, rays, radius_mm, cfg, fat_contrast=True)
    thick = [r.thickness_mm for r in meas.rays if r.thickness_mm is not None]
    labels = [label_band_mm(mask, row_mm, col_mm, centre, dd, radius_mm, cfg) for _, dd in rays]
    labels_ok = [x for x in labels if x is not None]
    return {
        "status": "measured" if thick else "not_assessed",
        "measured_mm": float(max(thick)) if thick else None,
        "label_mm": float(max(labels_ok)) if labels_ok else None,
        "fat_mm": meas.fat_mm,
        "fat_reason": meas.fat_reason,
        "radius_mm": radius_mm,
        "rays_measured": len(thick),
        "noise": meas.noise,
    }


_NEG = re.compile(r"\b(no|not|without|negative|absent|nil)\b")


def lipomatosis_mentioned(text: str) -> bool | None:
    """True when a clause mentions lipomatosis without a negation cue before it; None when no mention at all."""
    seen = False
    for clause in re.split(r"[.;\n]", (text or "").lower()):
        i = clause.find("lipomat")
        if i < 0:
            continue
        seen = True
        if not _NEG.search(clause[:i]):
            return True
    return False if seen else None


def case(patient: str, cfg: Any = DEFAULT) -> dict[str, Any]:
    """Measure every labelled disc of one patient and attach the report labels (derived, no text)."""
    out: dict[str, Any] = {"patient": patient, "levels": []}
    try:
        sel = S.select_session(patient, S.patient_sessions()[patient])
    except Exception as exc:  # contained: one patient must not stop the run
        out["status"] = f"error: {type(exc).__name__}"
        return out
    if sel is None:
        out["status"] = "no_matching_session"
        return out
    _, study, truths = sel
    for tr in truths:
        st = study.stacks[tr.stack_key]
        mask = S.read_mask(patient, tr.dlabel)
        img = np.asarray(st.vol[tr.slice_index], dtype=np.float64)
        if mask.shape != img.shape:
            continue
        rec = measure_level(img, mask, st.row_mm, st.col_mm, cfg)
        rec.update({"dlabel": tr.dlabel, "level": tr.level, "ncc": tr.match_ncc, "stack": tr.stack_key})
        out["levels"].append(rec)
    out["status"] = "ok"
    return out


_NOTES: dict[int, str] | None = None


def _notes() -> dict[int, str]:
    global _NOTES
    if _NOTES is None:
        _NOTES = S.read_radiologist_notes()
    return _NOTES


def report_labels(patient: str) -> dict[str, dict[str, Any]]:
    """Per-level derived report labels for one patient (no text kept)."""
    text = _notes().get(int(patient), "")
    return aggregate_levels(parse_report(text).labels)


# --------------------------------------------------------------------------------------- statistics
def _auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def _binary(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(pred & truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        "n": int(truth.size),
        "positives": int(truth.sum()),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
    }


def patient_bootstrap(rows: list[dict[str, Any]], metric: Any) -> tuple[float, float]:
    """95% percentile interval of ``metric(rows)`` resampling patients (not levels), seed 0."""
    by_pat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_pat.setdefault(r["patient"], []).append(r)
    keys = sorted(by_pat)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(BOOT_N):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [r for i in pick for r in by_pat[keys[i]]]
        v = metric(sample)
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _binary_metric(key: str, threshold: float) -> Any:
    def fn(rows: list[dict[str, Any]]) -> float:
        pred = np.array([r["measured_mm"] >= threshold for r in rows], dtype=bool)
        truth = np.array([r["report_lf"] for r in rows], dtype=bool)
        return _binary(pred, truth)[key]

    return fn


def _auc_metric(rows: list[dict[str, Any]]) -> float:
    return _auc(
        np.array([r["measured_mm"] for r in rows], dtype=float),
        np.array([1 if r["report_lf"] else 0 for r in rows], dtype=int),
    )


_LF_TERM = re.compile(r"lig\w*\s+fl\w*")
_LF_POS = re.compile(r"hypertro|thicken|thickn")


def lf_thickening_mentioned(text: str) -> bool | None:
    """Patient level: True when a clause names the ligamentum flavum with hypertrophy or thickening and no negation
    before the term; False when named only without that; None when the ligament is not named. (Exploratory.)"""
    seen = False
    for clause in re.split(r"[.;\n]", (text or "").lower()):
        m = _LF_TERM.search(clause)
        if m is None:
            continue
        seen = True
        pos = _LF_POS.search(clause, m.start())
        if pos is not None and not _NEG.search(clause[: m.start()]):
            return True
    return False if seen else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    pats = S.labelled_patients()
    if args.limit:
        pats = pats[: args.limit]
    cases: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            cases = list(ex.map(case, pats, chunksize=8))
    else:
        cases = [case(p) for p in pats]
    levels: list[dict[str, Any]] = []  # every measured level (agreement with label 200)
    scored: list[dict[str, Any]] = []  # measured, with a per-level report value (primary)
    patients: list[dict[str, Any]] = []  # exploratory: patient-level
    coverage: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for c in cases:
        status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
        rep = report_labels(c["patient"])
        raw = _notes().get(int(c["patient"]), "")
        lipo = lipomatosis_mentioned(raw)
        lf_any = lf_thickening_mentioned(raw)
        maxima: list[float] = []
        for lv in c["levels"]:
            level = lv["level"]
            if lv["status"] != "measured":
                key = f"not_measured: {lv['status']}"
                coverage[key] = coverage.get(key, 0) + 1
                continue
            coverage["measured"] = coverage.get("measured", 0) + 1
            maxima.append(float(lv["measured_mm"]))
            lf = rep.get(level, {}).get("ligamentum_flavum_thickening", {})
            base = {k: lv[k] for k in ("level", "measured_mm", "label_mm", "fat_mm", "rays_measured")}
            base["patient"] = c["patient"]
            base["report_lf_level"] = bool(lf["value"]) if lf.get("value") is not None else None
            base["lipomatosis_mention"] = lipo
            levels.append(base)
            if base["report_lf_level"] is not None and lf.get("conflict") is not True:
                scored.append(base)
        if maxima:
            patients.append({"patient": c["patient"], "measured_mm": max(maxima), "report_lf": lf_any})
    return {
        "levels": levels,
        "scored": scored,
        "patients": patients,
        "coverage": coverage,
        "status": status_counts,
        "n_patients": len(pats),
    }


def _ba(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bland-Altman of the measured band against the label-200 band on the same rays (levels with both)."""
    lab = [r for r in rows if r.get("label_mm") is not None and r.get("measured_mm") is not None]
    if not lab:
        return {"n": 0}
    diff = np.array([r["measured_mm"] - r["label_mm"] for r in lab], dtype=float)
    ms = np.array([r["measured_mm"] for r in lab], dtype=float)
    ls = np.array([r["label_mm"] for r in lab], dtype=float)
    out: dict[str, Any] = {
        "n": int(diff.size),
        "bias_mm": float(diff.mean()),
        "within_1mm": float(np.mean(np.abs(diff) <= 1.0)),
        "measured_median_mm": float(np.median(ms)),
        "label_median_mm": float(np.median(ls)),
    }
    if diff.size > 1:
        sd = float(diff.std(ddof=1))
        out["loa_mm"] = [float(diff.mean() - 1.96 * sd), float(diff.mean() + 1.96 * sd)]
    if diff.size > 2 and ms.std() > 0 and ls.std() > 0:
        out["pearson"] = float(np.corrcoef(ms, ls)[0, 1])
    # patient-level bootstrap of the bias and of the within-1 mm share
    out["bias_ci95"] = patient_bootstrap(
        lab, lambda rr: float(np.mean([r["measured_mm"] - r["label_mm"] for r in rr]))
    )
    out["within_1mm_ci95"] = patient_bootstrap(
        lab, lambda rr: float(np.mean([abs(r["measured_mm"] - r["label_mm"]) <= 1.0 for r in rr]))
    )
    return out


def _primary(rows: list[dict[str, Any]], thr: float) -> dict[str, Any]:
    if not rows:
        return {"scored_levels": 0, "report_positive_levels": 0}
    truth = np.array([bool(r["report_lf_level"]) for r in rows], dtype=bool)
    pred = np.array([r["measured_mm"] >= thr for r in rows], dtype=bool)
    out: dict[str, Any] = {
        "scored_levels": len(rows),
        "report_positive_levels": int(truth.sum()),
        "binary": _binary(pred, truth),
        "ci95": {
            k: patient_bootstrap(rows, _level_metric(k, thr))
            for k in ("sensitivity", "specificity", "ppv", "npv")
        },
    }
    out["auc"] = _auc(np.array([r["measured_mm"] for r in rows], dtype=float), truth.astype(int))
    out["auc_ci95"] = patient_bootstrap(rows, lambda rr: _auc_level(rr))
    return out


def _level_metric(key: str, thr: float) -> Any:
    def fn(rows: list[dict[str, Any]]) -> float:
        pred = np.array([r["measured_mm"] >= thr for r in rows], dtype=bool)
        truth = np.array([bool(r["report_lf_level"]) for r in rows], dtype=bool)
        return _binary(pred, truth)[key] if rows else float("nan")

    return fn


def _auc_level(rows: list[dict[str, Any]]) -> float:
    return _auc(
        np.array([r["measured_mm"] for r in rows], dtype=float),
        np.array([1 if r["report_lf_level"] else 0 for r in rows], dtype=int),
    )


def _exploratory(patients: list[dict[str, Any]], thr: float) -> dict[str, Any]:
    rows = [r for r in patients if r["report_lf"] is not None]
    if not rows:
        return {"patients": 0}
    truth = np.array([bool(r["report_lf"]) for r in rows], dtype=bool)
    pred = np.array([r["measured_mm"] >= thr for r in rows], dtype=bool)
    b = _binary(pred, truth)
    ci = {
        k: patient_bootstrap(rows, _pat_metric(k, thr)) for k in ("sensitivity", "specificity", "ppv", "npv")
    }
    auc = _auc(np.array([r["measured_mm"] for r in rows], dtype=float), truth.astype(int))
    return {
        "patients_named": len(rows),
        "positive_patients": int(truth.sum()),
        "binary": b,
        "ci95": ci,
        "auc": auc,
        "auc_ci95": patient_bootstrap(rows, _pat_auc),
    }


def _pat_metric(key: str, thr: float) -> Any:
    def fn(rows: list[dict[str, Any]]) -> float:
        pred = np.array([r["measured_mm"] >= thr for r in rows], dtype=bool)
        truth = np.array([bool(r["report_lf"]) for r in rows], dtype=bool)
        return _binary(pred, truth)[key] if rows else float("nan")

    return fn


def _pat_auc(rows: list[dict[str, Any]]) -> float:
    return _auc(
        np.array([r["measured_mm"] for r in rows], dtype=float),
        np.array([1 if r["report_lf"] else 0 for r in rows], dtype=int),
    )


def summarise(res: dict[str, Any]) -> dict[str, Any]:
    thr = DEFAULT.ligamentum_thick_mm
    lipo_pts = [r for r in res["levels"] if r["lipomatosis_mention"] is not None]
    return {
        "operating_point_mm": thr,
        "primary_per_level": _primary(res["scored"], thr),
        "label_agreement_all_measured_levels": _ba(res["levels"]),
        "exploratory_patient_level": _exploratory(res["patients"], thr),
        "lipomatosis": {
            "patients_with_mention": int(sum(1 for r in res["levels"] if r["lipomatosis_mention"])),
            "levels_with_fat_measure": int(sum(1 for r in lipo_pts if r["fat_mm"] is not None)),
        },
        "coverage": res["coverage"],
        "patient_status": res["status"],
        "patients": res["n_patients"],
        "measured_levels": len(res["levels"]),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0, help="first N labelled patients (0 = all)")
    p.add_argument("--workers", type=int, default=2, help="processes (max 3 on the shared machine)")
    p.add_argument("--out", type=Path, default=Path("/tmp/spinescope-dx09_lf_epidural/val"))
    args = p.parse_args(argv)
    if args.workers > 3:
        raise SystemExit("--workers must be 3 or fewer (shared machine)")
    logging.basicConfig(level=logging.WARNING)
    res = run(args)
    summary = summarise(res)
    os.makedirs(args.out, exist_ok=True)
    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump(
            summary, f, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else str(x)
        )
    fields = [
        "patient",
        "level",
        "measured_mm",
        "label_mm",
        "fat_mm",
        "rays_measured",
        "report_lf_level",
        "lipomatosis_mention",
    ]
    with open(Path(args.out) / "levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in res["levels"]:
            w.writerow(r)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
