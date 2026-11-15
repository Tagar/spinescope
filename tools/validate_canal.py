"""Validate the central canal analysis on public data (pre-registered rules: docs/analysis/canal.md, section 3).

V1  Sagittal AP: the labeller's AP vs the same AP definition on the SPIDER canal mask (label 100), validation split.
V2  Axial dural sac area: the observed area vs the Sudirman thecal sac label (150) on the annotated axial T1 slice.
V3  Stenosis suggestion vs the parsed Sudirman radiologist notes (weak labels, canal term only).

Usage (from the repository root, project venv)::

    SPINESCOPE_THREADS=2 .venv/bin/python tools/validate_canal.py --out /tmp/spinescope-dx06_canal/val --workers 2

Writes ``<out>/canal_validation.json`` (numbers and per-level rows, no paths, no identifiers beyond the public case
and folder ids). Prints a summary. Patient-level bootstrap intervals: 1000 resamples, seed 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("validate_canal")

BOOT_N = 1000
BOOT_SEED = 0
SAC_VALUE = 150  # Sudirman thecal sac label
CSA_RELATIVE_MM2 = 100.0  # V3 primary prediction threshold (config: canal_csa_relative_mm2)
AP_RELATIVE_MM = 13.0
AP_ABSOLUTE_MM = 10.0


# --------------------------------------------------------------------------- statistics
def _pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _kappa(a: list[int], b: list[int]) -> float:
    """Unweighted Cohen's kappa for two label lists."""
    if not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa = Counter(a)
    pb = Counter(b)
    pe = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def _bland_altman(diff: NDArray[np.float64]) -> dict[str, float]:
    if diff.size == 0:
        return {"bias": float("nan"), "loa_low": float("nan"), "loa_high": float("nan")}
    m, s = float(diff.mean()), float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return {"bias": m, "loa_low": m - 1.96 * s, "loa_high": m + 1.96 * s}


def continuous_summary(pred: NDArray[np.float64], ref: NDArray[np.float64]) -> dict[str, float]:
    diff = pred - ref
    rel = np.abs(diff) / np.where(np.abs(ref) > 0, np.abs(ref), np.nan)
    out: dict[str, float] = {
        "n": float(diff.size),
        "mae": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan"),
        "median_rel_err": float(np.nanmedian(rel)) if diff.size else float("nan"),
        "pearson_r": _pearson(pred, ref),
    }
    out.update(_bland_altman(diff))
    return out


def patient_bootstrap(
    groups: list[str], values: list[tuple[Any, ...]], stat: Callable[[list[tuple[Any, ...]]], float]
) -> dict[str, float]:
    """95% percentile interval of ``stat`` when whole patients are resampled (seed 0, 1000 resamples)."""
    by_patient: dict[str, list[tuple[Any, ...]]] = {}
    for g, v in zip(groups, values, strict=True):
        by_patient.setdefault(g, []).append(v)
    keys = sorted(by_patient)
    if not keys:
        return {"low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(BOOT_N):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [item for i in pick for item in by_patient[keys[i]]]
        val = stat(sample)
        if np.isfinite(val):
            draws.append(val)
    if not draws:
        return {"low": float("nan"), "high": float("nan")}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi)}


def binary_metrics(pred: list[bool], ref: list[bool]) -> dict[str, float]:
    p = np.asarray(pred, dtype=bool)
    r = np.asarray(ref, dtype=bool)
    tp = int(np.sum(p & r))
    tn = int(np.sum(~p & ~r))
    fp = int(np.sum(p & ~r))
    fn = int(np.sum(~p & r))

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    return {
        "n": float(r.size),
        "prevalence": ratio(int(np.sum(r)), int(r.size)),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


# --------------------------------------------------------------------------- V1: SPIDER AP
#: Anchors are computed once per case and reloaded (shared-machine rule of the wave brief).
ANCHOR_CACHE = Path("/tmp/spinescope-dx06_canal/anchors")


def _anchors_for(study: Any, key: str) -> Any:
    """``label_spine(study)``, saved under :data:`ANCHOR_CACHE` as JSON and reloaded on later runs."""
    from spinescope.anatomy import label_spine
    from spinescope.anatomy.anchors import Anchors

    ANCHOR_CACHE.mkdir(parents=True, exist_ok=True)
    path = ANCHOR_CACHE / f"{key}.json"
    if path.is_file():
        return Anchors.from_json(str(path))
    anchors = label_spine(study)
    anchors.to_json(str(path))
    return anchors


def _spider_case(case_id: str) -> dict[str, Any]:
    """Labeller AP and the AP of the SPIDER canal mask on the same slices and levels."""
    from spinescope.anatomy.canal_ml import ap_chord_mm, largest_component
    from spinescope.anatomy.ml.grid import plane_grid, resample_slice
    from spinescope.config import DEFAULT
    from spinescope.datasets.spider import load_spider_studies

    out: dict[str, Any] = {"case": case_id, "rows": [], "error": None}
    try:
        (c,) = load_spider_studies(cases=[case_id], sequences=None)
        seq = "t2" if "t2" in c.sequences else c.sequences[0]
        study = c.to_study(sequences=[seq])
        anchors = _anchors_for(study, f"spider_{case_id}_{seq}")
        canal = anchors.canal or {}
        st = study.best("sag")
        if st is None or not canal:
            out["error"] = "no sagittal stack or canal geometry"
            return out
        mask = c.mask(seq)
        bias = DEFAULT.canal_ap_bias_mm
        band = DEFAULT.canal_level_band_mm
        grids: dict[int, Any] = {}
        ref_masks: dict[int, NDArray[np.bool_]] = {}
        for lab, entry in (canal.get("per_level") or {}).items():
            pred = entry.get("ap_mm")
            k = int(entry["slice"])
            if k not in grids:
                grids[k] = plane_grid(st, k, DEFAULT.ml_resolution_mm)
                resampled = resample_slice(st, grids[k], array=(mask == 100).astype(np.float32), order=0)
                ref_masks[k] = largest_component(resampled > 0.5)
            chord = ap_chord_mm(ref_masks[k], grids[k], float(entry["z_mm"]), band)
            ref = None if chord is None else max(chord - bias, 0.0)
            out["rows"].append(
                {
                    "level": lab,
                    "kind": entry.get("kind"),
                    "pred_ap_mm": None if pred is None else float(pred),
                    "ref_ap_mm": None if ref is None else round(float(ref), 3),
                }
            )
    except Exception as exc:  # one bad case is counted, not fatal
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def run_v1(workers: int) -> dict[str, Any]:
    from spinescope.datasets.spider import load_spider_studies

    cases = [c.case_id for c in load_spider_studies(load=False) if c.subset == "validation"]
    results = _map(_spider_case, cases, workers)
    rows = [dict(r, case=res["case"]) for res in results for r in res["rows"]]
    both = [r for r in rows if r["pred_ap_mm"] is not None and r["ref_ap_mm"] is not None]
    pred = np.array([r["pred_ap_mm"] for r in both], dtype=np.float64)
    ref = np.array([r["ref_ap_mm"] for r in both], dtype=np.float64)
    groups = [r["case"] for r in both]
    pairs = list(zip(pred.tolist(), ref.tolist(), strict=True))
    summary = continuous_summary(pred, ref)
    summary["mae_ci"] = patient_bootstrap(
        groups, pairs, lambda s: float(np.mean([abs(a - b) for a, b in s])) if s else float("nan")
    )
    summary["bias_ci"] = patient_bootstrap(
        groups, pairs, lambda s: float(np.mean([a - b for a, b in s])) if s else float("nan")
    )

    def cls(x: float) -> int:
        return 0 if x < AP_ABSOLUTE_MM else (1 if x < AP_RELATIVE_MM else 2)

    a = [cls(x) for x in pred.tolist()]
    b = [cls(x) for x in ref.tolist()]
    summary["class_accuracy"] = (
        float(np.mean([x == y for x, y in zip(a, b, strict=True)])) if a else float("nan")
    )
    summary["class_kappa"] = _kappa(a, b)
    failures = Counter(res["error"] for res in results if res["error"])
    return {
        "cases": len(cases),
        "cases_failed": int(sum(failures.values())),
        "failures": dict(failures),
        "levels_total": len(rows),
        "levels_compared": len(both),
        "levels_pred_missing": sum(1 for r in rows if r["pred_ap_mm"] is None),
        "summary": summary,
        "rows": rows,
    }


# --------------------------------------------------------------------------- V2 and V3: Sudirman
def canal_reference_labels(parses: dict[int, Any]) -> dict[int, dict[str, bool]]:
    """Per notes number: level -> True/False for the canal term at the three annotated levels (conflicts dropped).

    Only parsed ``stenosis`` labels whose zone is not foraminal or extraforaminal count. A level with both a positive
    and a negated label is dropped.
    """
    from spinescope.datasets.sudirman import DISC_LEVELS

    levels = set(DISC_LEVELS.values())
    out: dict[int, dict[str, bool]] = {}
    for num, parse in parses.items():
        pos: dict[str, bool] = {}
        neg: dict[str, bool] = {}
        for lab in parse.labels:
            if lab.field != "stenosis" or lab.level not in levels:
                continue
            if lab.zone in ("foraminal", "extraforaminal"):
                continue
            (neg if lab.negated else pos)[lab.level] = True
        for lvl in levels:
            if lvl in pos and lvl in neg:
                continue
            if lvl in pos:
                out.setdefault(num, {})[lvl] = True
            elif lvl in neg:
                out.setdefault(num, {})[lvl] = False
    return out


def _sudirman_patient(args: tuple[str, list[str], dict[str, bool]]) -> dict[str, Any]:
    from pathlib import Path as _P

    from spinescope.analysis import canal
    from spinescope.config import DEFAULT
    from spinescope.datasets.sudirman import read_mask, select_session

    pid, sessions, refs = args
    out: dict[str, Any] = {"patient": pid, "rows": [], "error": None}
    try:
        sel = select_session(pid, [_P(s) for s in sessions], None)
        if sel is None:
            out["error"] = "no session reproduces the label slices"
            return out
        _sess, study, levels = sel
        anchors = _anchors_for(study, f"sudirman_{pid}")
        # The annotated labels are T1: the analysis reads the annotated contrast only (protocol amendment, canal.md s.3)
        res = canal.analyze(study, anchors, dataclasses.replace(DEFAULT, canal_axial_contrasts=("t1",)))
        disc_rows = {r["level"]: r for r in res.rows if r.get("scope") == "disc"}
        for lt in levels:
            st = study.stacks[lt.stack_key]
            mask = read_mask(pid, lt.dlabel)
            ref_px = int(np.sum(mask == SAC_VALUE))
            ref_area = ref_px * float(st.row_mm) * float(st.col_mm)
            row = disc_rows.get(lt.level, {})
            match = (
                row.get("axial_slab_chosen") == lt.stack_key
                and row.get("axial_slice_chosen") == lt.slice_index
            )
            out["rows"].append(
                {
                    "patient": pid,
                    "level": lt.level,
                    "slice_matches": bool(match),
                    "slab_chosen": row.get("axial_slab_chosen"),
                    "slice_chosen": row.get("axial_slice_chosen"),
                    "ref_area_mm2": round(ref_area, 2),
                    "pred_observed_mm2": row.get("csa_observed_mm2"),
                    "pred_mm2": row.get("csa_mm2"),
                    "pred_csa_grade": row.get("csa_grade"),
                    "pred_ap_mm": row.get("ap_mm"),
                    "ref_canal_stenosis": refs.get(lt.level),
                    "truncated": row.get("csa_truncated"),
                    "note": row.get("csa_note", ""),
                    "ref_pixels": ref_px,
                }
            )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _map(fn: Callable[[Any], dict[str, Any]], items: list[Any], workers: int) -> list[dict[str, Any]]:
    from multiprocessing import get_context

    if workers <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with get_context("spawn").Pool(workers) as pool:
        return list(pool.map(fn, items, chunksize=2))


def run_sudirman(workers: int) -> dict[str, Any]:
    from spinescope.datasets.report_labels import parse_reports
    from spinescope.datasets.sudirman import (
        DISC_LEVELS,
        labelled_patients,
        patient_sessions,
        read_radiologist_notes,
    )

    sessions = patient_sessions()
    ids = [p for p in labelled_patients() if p in sessions]
    refs_all = canal_reference_labels(parse_reports(read_radiologist_notes()))
    jobs = [(p, [str(s) for s in sessions[p]], refs_all.get(int(p), {})) for p in ids]
    results = _map(_sudirman_patient, jobs, workers)
    rows = [r for res in results for r in res["rows"]]
    failures = Counter(res["error"] for res in results if res["error"])
    levels = sorted(set(DISC_LEVELS.values()))

    matched = [r for r in rows if r["slice_matches"] and r["pred_observed_mm2"] is not None]
    v2_pairs = [(float(r["pred_observed_mm2"]), float(r["ref_area_mm2"])) for r in matched]
    v2_groups = [r["patient"] for r in matched]
    v2: dict[str, Any] = {}
    if v2_pairs:
        pred = np.array([p for p, _ in v2_pairs])
        ref = np.array([q for _, q in v2_pairs])
        v2 = continuous_summary(pred, ref)
        v2["mae_ci"] = patient_bootstrap(
            v2_groups, v2_pairs, lambda s: float(np.mean([abs(a - b) for a, b in s])) if s else float("nan")
        )
        v2["bias_ci"] = patient_bootstrap(
            v2_groups, v2_pairs, lambda s: float(np.mean([a - b for a, b in s])) if s else float("nan")
        )
        v2["by_level"] = {
            lv: {
                "n": sum(1 for r in matched if r["level"] == lv),
                "mae": float(
                    np.mean(
                        [
                            abs(float(r["pred_observed_mm2"]) - r["ref_area_mm2"])
                            for r in matched
                            if r["level"] == lv
                        ]
                    )
                )
                if any(r["level"] == lv for r in matched)
                else float("nan"),
            }
            for lv in levels
        }
    gate = {
        "n_included": len(matched),
        "mae_le_15": bool(v2.get("mae", np.inf) <= 15.0) if v2 else False,
        "abs_bias_le_10": bool(abs(v2.get("bias", np.inf)) <= 10.0) if v2 else False,
        "pearson_ge_0_8": bool(v2.get("pearson_r", -1.0) >= 0.8) if v2 else False,
        "n_ge_100": len(matched) >= 100,
    }
    gate["passes"] = all(gate.values())

    # V3: report labels vs the primary prediction (observed area < 100 mm2), on levels with a label and a measurement
    v3_rows = [r for r in rows if r["ref_canal_stenosis"] is not None]
    v3_meas = [r for r in v3_rows if r["pred_observed_mm2"] is not None]
    v3_primary = binary_metrics(
        [float(r["pred_observed_mm2"]) < CSA_RELATIVE_MM2 for r in v3_meas],
        [bool(r["ref_canal_stenosis"]) for r in v3_meas],
    )
    v3_primary["ci_sensitivity"] = patient_bootstrap(
        [r["patient"] for r in v3_meas],
        [(float(r["pred_observed_mm2"]) < CSA_RELATIVE_MM2, bool(r["ref_canal_stenosis"])) for r in v3_meas],
        lambda s: _rate([p for p, t in s if t], [True] * sum(1 for _, t in s if t)),
    )
    v3_meas_ap = [r for r in v3_meas if r["pred_ap_mm"] is not None]
    v3_secondary = binary_metrics(
        [float(r["pred_ap_mm"]) < AP_RELATIVE_MM for r in v3_meas_ap],
        [bool(r["ref_canal_stenosis"]) for r in v3_meas_ap],
    )
    return {
        "patients": len(jobs),
        "patients_failed": int(sum(failures.values())),
        "failures": dict(failures),
        "levels_total": len(rows),
        "levels_slab_chosen_is_annotated": sum(1 for r in rows if r["slice_matches"]),
        "levels_slab_chosen_other": sum(
            1 for r in rows if r["slab_chosen"] is not None and not r["slice_matches"]
        ),
        "levels_no_slab_chosen": sum(1 for r in rows if r["slab_chosen"] is None),
        "levels_measured": sum(1 for r in rows if r["pred_observed_mm2"] is not None),
        "v2_area_observed_vs_reference": v2,
        "v2_gate": gate,
        "v3_reference_levels": len(v3_rows),
        "v3_reference_positive": sum(1 for r in v3_rows if r["ref_canal_stenosis"]),
        "v3_primary_area_lt_100": v3_primary,
        "v3_secondary_ap_lt_13": v3_secondary,
        "rows": rows,
    }


def _rate(num: list[bool], den: list[bool]) -> float:
    return sum(1 for x in num if x) / len(den) if den else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument(
        "--workers", type=int, default=2, help="worker processes (shared machine: at most 2 or 3)"
    )
    ap.add_argument("--skip-spider", action="store_true")
    ap.add_argument("--skip-sudirman", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    if not args.skip_spider:
        report["V1_spider_ap"] = run_v1(args.workers)
    if not args.skip_sudirman:
        report["V2_V3_sudirman"] = run_sudirman(args.workers)
    (out / "canal_validation.json").write_text(json.dumps(report, indent=1, default=float))
    print(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in report.items()},
            indent=1,
            default=float,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
