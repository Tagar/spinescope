"""Calibrate the axial disc-likeness cue on the Sudirman axial T1 slabs (writes ``axial_v1.json``).

Not used by the labeller. Data: the Sudirman labelled sessions (:mod:`spinescope.datasets.sudirman`). For each
annotated disc, the labelled axial T1 slice gives:

* positives: the disc-likeness features (see :func:`axial.feature_maps`) at the IVD centroid;
* negatives: the same features at pixels of the same slice at least 25 mm from every labelled region
  (disc, posterior elements, thecal sac, the region between them), seeded and sampled.

Split: patients sorted by id; even positions are the FIT split, odd positions the TEST split. A two-class
Gaussian naive Bayes is fitted on FIT only. The support threshold is the 5th percentile of FIT positive LLRs
(95 % of true discs supported). The missed-disc threshold maximises F1 of disc-like peaks against the annotated
discs on FIT (a peak within ``missed_mm`` of an annotated IVD is a true disc). TEST is only reported.

Limitations (also in docs/labeling/lab20_axial.md): the negatives contain no vertebral bodies (the Sudirman
labels do not cover them), so a body-like region could still score as disc-like; the labels are T1 only; the
level names are the dataset's convention, which is not used here (this tool only needs positions).

Usage::

    .venv/bin/python -m spinescope.anatomy._calibrate_axial --out /tmp/spinescope-lab20_axial/axial_fit.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from ..datasets.sudirman import (
    IVD_VALUE,
    labelled_patients,
    patient_sessions,
    read_mask,
    select_session,
)
from . import axial as ax

log = logging.getLogger(__name__)

NEG_MIN_MM = 25.0  # negatives lie at least this far from every labelled region
NEG_PER_SLICE = 8
LABELLED_VALUES = (IVD_VALUE, 100, 150, 200)


def _split(ids: list[str]) -> dict[str, str]:
    """'fit' for even positions, 'test' for odd, in sorted patient order."""
    return {pid: ("fit" if i % 2 == 0 else "test") for i, pid in enumerate(sorted(ids))}


def _pos_neg_features(args: tuple[str, int]) -> dict[str, Any]:
    """Pass 1 worker: positive and negative feature vectors of one patient's labelled slices."""
    pid, idx = args
    out: dict[str, Any] = {"pid": pid, "pos": [], "neg": [], "pos_labels": [], "error": None}
    try:
        sel = select_session(pid, patient_sessions()[pid])
        if sel is None:
            out["error"] = "no session"
            return out
        _sess, study, levels = sel
        cfg = ax.DEFAULT_AXIAL
        rng = np.random.default_rng(1000 + idx)
        for lv in levels:
            if lv.stack_key not in study.stacks:
                continue
            st = study.stacks[lv.stack_key]
            if "t1" not in str(st.meta.get("role", {}).get("contrast", "")):
                continue
            k = int(lv.slice_index)
            fmaps = ax.feature_maps(st, k, cfg)
            _, r, c = st.lps_to_pix(lv.center)
            rr, cc = round(r), round(c)
            out["pos"].append(fmaps[:, rr, cc].tolist())
            out["pos_labels"].append(lv.level)
            dlabel = {"L3-L4": "D3", "L4-L5": "D4", "L5-S1": "D5"}.get(lv.level)
            if dlabel is None:
                continue
            mask = read_mask(pid, dlabel)
            if mask.shape != st.vol[k].shape:
                continue
            lab = np.isin(mask, LABELLED_VALUES)
            dist = ndimage.distance_transform_edt(~lab, sampling=(st.row_mm, st.col_mm))
            cand = np.argwhere(dist >= NEG_MIN_MM)
            if cand.size == 0:
                continue
            pick = cand[rng.choice(len(cand), size=min(NEG_PER_SLICE, len(cand)), replace=False)]
            for r0, c0 in pick:
                out["neg"].append(fmaps[:, r0, c0].tolist())
    except Exception as exc:  # one bad patient must not stop the run
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _fit_nb(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu_p, var_p = pos.mean(axis=0), pos.var(axis=0) + 1e-9
    mu_n, var_n = neg.mean(axis=0), neg.var(axis=0) + 1e-9
    return mu_p, var_p, mu_n, var_n


def _llr_points(x: np.ndarray, model: ax.AxialModel) -> np.ndarray:
    """LLR of feature rows x (n, 3) under the naive Bayes model."""
    out = np.zeros(len(x))
    for j in range(len(ax.FEATURES)):
        out += ax._log_normal(x[:, j], model.mean_disc[j], model.var_disc[j])
        out -= ax._log_normal(x[:, j], model.mean_non[j], model.var_non[j])
    return out


def _peak_rows(args: tuple[str, int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Pass 2 worker: disc-like peaks of one patient's labelled slices, each labelled true or false disc.

    The model is passed as a plain dict so the worker can rebuild it (spawn-safe).
    """
    pid, _idx, model_d, _ = args
    out: dict[str, Any] = {"pid": pid, "split": None, "peaks": [], "truth": [], "error": None}
    try:
        model = ax.AxialModel(**model_d)
        sel = select_session(pid, patient_sessions()[pid])
        if sel is None:
            out["error"] = "no session"
            return out
        _sess, study, levels = sel
        cfg = ax.DEFAULT_AXIAL
        for lv in levels:
            if lv.stack_key not in study.stacks:
                continue
            st = study.stacks[lv.stack_key]
            if "t1" not in str(st.meta.get("role", {}).get("contrast", "")):
                continue
            k = int(lv.slice_index)
            m = ax.llr_map(st, k, model, cfg)
            size = max(3, round(2 * cfg.missed_mm / max(min(st.row_mm, st.col_mm), 1e-6)) | 1)
            peaks_mask = m == ndimage.maximum_filter(m, size=size)
            rows, cols = np.nonzero(peaks_mask)
            tc = np.asarray(lv.center, dtype=np.float64)
            _, tr, tcc = st.lps_to_pix(tc)
            for r, c in zip(rows, cols, strict=True):
                xyz = st.pix_to_lps(float(k), float(r), float(c))
                in_plane = float(np.linalg.norm((xyz - tc) - float((xyz - tc) @ st.normal) * st.normal))
                out["peaks"].append(
                    {
                        "llr": float(m[r, c]),
                        "true": bool(in_plane <= cfg.missed_mm),
                        "level": lv.level,
                    }
                )
            out["truth"].append({"level": lv.level, "llr_at_centre": float(m[round(tr), round(tcc)])})
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _f1_threshold(peaks: list[tuple[float, bool]], n_true: int, grid: np.ndarray) -> tuple[float, float]:
    """Threshold on peak LLR maximising F1 against n_true annotated discs. Returns (threshold, F1)."""
    best = (float(grid[0]), -1.0)
    llrs = np.array([p[0] for p in peaks]) if peaks else np.array([])
    trues = np.array([p[1] for p in peaks], dtype=bool) if peaks else np.array([], dtype=bool)
    for t in grid:
        sel = llrs >= t
        tp = int(np.sum(trues & sel))
        fp = int(np.sum(~trues & sel))
        fn = max(n_true - tp, 0)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def calibrate(workers: int = 4, limit: int | None = None) -> dict[str, Any]:
    """Run both passes and return the calibration record (the content of ``axial_v1.json`` plus metrics)."""
    ids = labelled_patients()
    if limit is not None:
        ids = ids[:limit]
    split = _split(ids)
    jobs = [(pid, i) for i, pid in enumerate(sorted(ids))]
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        p1 = pool.map(_pos_neg_features, jobs, chunksize=2)
    pos_fit: list[list[float]] = []
    neg_fit: list[list[float]] = []
    pos_test: list[list[float]] = []
    neg_test: list[list[float]] = []
    for r in p1:
        if r["error"]:
            log.warning("pass 1 %s: %s", r["pid"], r["error"])
            continue
        target_pos, target_neg = (pos_fit, neg_fit) if split[r["pid"]] == "fit" else (pos_test, neg_test)
        target_pos.extend(r["pos"])
        target_neg.extend(r["neg"])
    pf, nf = np.asarray(pos_fit, dtype=np.float64), np.asarray(neg_fit, dtype=np.float64)
    if len(pf) < 10 or len(nf) < 10:
        raise RuntimeError(f"too few samples to fit: {len(pf)} positives, {len(nf)} negatives")
    mu_p, var_p, mu_n, var_n = _fit_nb(pf, nf)
    model_d: dict[str, Any] = {
        "mean_disc": tuple(float(x) for x in mu_p),
        "var_disc": tuple(float(x) for x in var_p),
        "mean_non": tuple(float(x) for x in mu_n),
        "var_non": tuple(float(x) for x in var_n),
        "support_llr": 0.0,
        "missed_llr": 0.0,
    }
    model = ax.AxialModel(**model_d)
    llr_pos_fit = _llr_points(pf, model)
    support = float(np.quantile(llr_pos_fit, 0.05))
    model_d["support_llr"] = support
    # Pass 2: peaks on the labelled slices, FIT for the threshold, both splits for the report.
    ctx2 = get_context("spawn")
    with ctx2.Pool(workers) as pool:
        p2 = pool.map(_peak_rows, [(pid, i, model_d, {}) for pid, i in jobs], chunksize=2)
    peaks_fit: list[tuple[float, bool]] = []
    n_true_fit = 0
    peaks_test: list[tuple[float, bool]] = []
    n_true_test = 0
    for r in p2:
        if r["error"]:
            log.warning("pass 2 %s: %s", r["pid"], r["error"])
            continue
        tgt, ntgt = (peaks_fit, "fit") if split[r["pid"]] == "fit" else (peaks_test, "test")
        for pk in r["peaks"]:
            tgt.append((pk["llr"], pk["true"]))
        if ntgt == "fit":
            n_true_fit += len(r["truth"])
        else:
            n_true_test += len(r["truth"])
    grid = (
        np.unique(np.quantile([p[0] for p in peaks_fit], np.linspace(0.5, 0.9999, 400)))
        if peaks_fit
        else np.zeros(1)
    )
    missed_t, f1_fit = _f1_threshold(peaks_fit, n_true_fit, grid)
    model_d["missed_llr"] = missed_t
    final = ax.AxialModel(**model_d)
    # Report: support on TEST positives and negatives; missed-peak P/R on TEST.
    llr_pos_test = _llr_points(np.asarray(pos_test, dtype=np.float64), final) if pos_test else np.zeros(0)
    llr_neg_test = _llr_points(np.asarray(neg_test, dtype=np.float64), final) if neg_test else np.zeros(0)

    def _pr(peaks: list[tuple[float, bool]], n_true: int) -> dict[str, float]:
        sel = [p for p in peaks if p[0] >= missed_t]
        tp = sum(1 for p in sel if p[1])
        fp = len(sel) - tp
        return {
            "peaks_flagged": len(sel),
            "precision": tp / max(len(sel), 1),
            "recall": tp / max(n_true, 1),
            "fp_per_slice_proxy": fp,
        }

    fit_metrics = {
        "positives": len(pf),
        "negatives": len(nf),
        "support_tpr": float(np.mean(llr_pos_fit >= support)),
        "missed": _pr(peaks_fit, n_true_fit),
        "f1_at_threshold": f1_fit,
    }
    test_metrics = {
        "positives": len(pos_test),
        "negatives": len(neg_test),
        "support_tpr": float(np.mean(llr_pos_test >= support)) if len(llr_pos_test) else float("nan"),
        "support_fpr": float(np.mean(llr_neg_test >= support)) if len(llr_neg_test) else float("nan"),
        "missed": _pr(peaks_test, n_true_test),
    }
    return {
        "version": "v1",
        "features": list(ax.FEATURES),
        "disc": {"mean": list(model_d["mean_disc"]), "var": list(model_d["var_disc"])},
        "non": {"mean": list(model_d["mean_non"]), "var": list(model_d["var_non"])},
        "support_llr": support,
        "missed_llr": missed_t,
        "provenance": {
            "dataset": "Sudirman lumbar DICOM (CC BY 4.0), axial T1 slabs, labelled slices",
            "split": "patients sorted by id: even = fit, odd = test",
            "patients": len(ids),
            "fit": fit_metrics,
            "test": test_metrics,
            "config": asdict(ax.DEFAULT_AXIAL),
            "negatives": f">= {NEG_MIN_MM:g} mm from labelled regions, {NEG_PER_SLICE} per labelled slice",
            "limitations": "T1 only; negatives contain no vertebral bodies; level names unused",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--out", required=True, help="output JSON (scratch; ship the fitted axial_v1.json by copying)"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N patients (smoke test)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    rec = calibrate(workers=args.workers, limit=args.limit)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["provenance"], indent=1, default=float))
    print(f"support_llr={rec['support_llr']:.3f} missed_llr={rec['missed_llr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["calibrate", "main"]
