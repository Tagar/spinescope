#!/usr/bin/env python3
"""Collect labelling outputs on SPIDER and Sudirman, fit the confidence calibration, report reliability.

Sub-commands::

    # 1. run the default pipeline (learned labeller + fusion) and cache per-patient records (resumable)
    python tools/calibrate_labeling.py collect --split training   --out /tmp/spinescope-lab08_fusion --jobs 3
    python tools/calibrate_labeling.py collect --split validation --out /tmp/spinescope-lab08_fusion --jobs 3
    python tools/calibrate_labeling.py collect --sudirman          --out /tmp/spinescope-lab08_fusion --jobs 3

    # 2. fit on SPIDER training only, report before/after on validation and Sudirman
    python tools/calibrate_labeling.py fit --out /tmp/spinescope-lab08_fusion [--write]

``--write`` replaces ``src/spinescope/anatomy/calibration_v1.json`` with the fitted model. Records hold only
technical outputs (levels, scores, features); no file paths or patient attributes. Nothing derived from SPIDER
or Sudirman is committed: the report goes to ``--out``.

Calibration target (disc): the predicted disc is matched to a true disc within 12 mm in the sagittal plane and
carries the right level name (false positives count as wrong, as in :func:`reliability`).
Calibration target (numbering): every matched disc of the study carries the right name and at least one matched.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from spinescope.anatomy._validate import (
    DiscMatch,
    VolumeScore,
    aggregate,
    reliability,
    score_anchors,
    truth_from_mask,
)
from spinescope.anatomy.calibration import (
    DISC_FEATURES,
    VOLUME_FEATURES,
    Calibration,
    Logistic,
    calibration_path,
    choose_thresholds,
)
from spinescope.anatomy.fusion import fusion_record
from spinescope.anatomy.numbering import STATUS_CONFIRMED, STATUS_PROBABLE, numbering_status
from spinescope.anatomy.pipeline import label_spine_details
from spinescope.config import Config

log = logging.getLogger("calibrate_labeling")
SPIDER_ROOT = Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public")) / "spider"
TOOL_VERSION = "1"


# ---------------------------------------------------------------- collection


def _spider_files(split: str) -> list[tuple[str, str]]:
    """(file stem, manufacturer) of every SPIDER volume of one split, from the overview (one case per file)."""
    out: list[tuple[str, str]] = []
    with open(SPIDER_ROOT / "overview.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("new_file_name") or "").strip()
            if row.get("subset", "").strip() != split or not name:
                continue
            if (
                not (SPIDER_ROOT / "images" / f"{name}.mha").is_file()
                or not (SPIDER_ROOT / "masks" / f"{name}.mha").is_file()
            ):
                continue
            out.append((name, (row.get("Manufacturer") or "").split(" ")[0] or "unknown"))
    return sorted(out, key=lambda t: (int(t[0].split("_")[0]), t[0]))


def _spider_study(name: str) -> Any:
    """Study for one SPIDER file. A T2 file is labelled together with its T1 sibling (second sagittal series)."""
    from spinescope import load_study
    from spinescope.study import from_stacks

    primary = load_study(str(SPIDER_ROOT / "images" / f"{name}.mha"))
    t1_path = SPIDER_ROOT / "images" / f"{name.rsplit('_', 1)[0]}_t1.mha"
    if name.endswith("_t2") and t1_path.is_file():
        sib = load_study(str(t1_path))
        st1, st2 = sib.best("sag"), primary.best("sag")
        if st1 is not None and st2 is not None:
            return from_stacks([st2, st1])
    return primary


def _finite(x: Any) -> Any:
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _match_dict(m: DiscMatch) -> dict[str, Any]:
    d = asdict(m)
    return {k: _finite(v) for k, v in d.items()}


def _spider_record(job: tuple[str, str, str, bool]) -> dict[str, Any]:
    """Worker: label one SPIDER volume (with its T1 sibling for a T2 file) and score it against its mask.

    The last job field turns on the classical second opinion (``Config.classical_second_opinion``), off by default.
    """
    name, manu, split, classical = job
    rec: dict[str, Any] = {"id": name, "split": split, "vendor": manu, "error": None}
    t0 = time.perf_counter()
    try:
        study = _spider_study(name)
        truth, mx = truth_from_mask(str(SPIDER_ROOT / "masks" / f"{name}.mha"))
        rec["second_series"] = len(study.find("sag")) > 1
        det = label_spine_details(
            study, Config(classical_second_opinion=classical), backend="ml", calibrate=False
        )
        rec["primary"] = det.anchors.source_stack
        score = score_anchors(det.anchors, truth, mx)
        rec.update(_record_common(det, score))
        if det.unfused is not None:
            us = score_anchors(det.unfused, truth, mx)
            rec["unfused"] = {"matched": us.matched, "label_correct": us.label_correct, "n_pred": us.n_pred}
    except Exception as exc:  # a failing case is a result, not a crash of the run
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return rec


def _record_common(det: Any, score: VolumeScore) -> dict[str, Any]:
    """Shared record fields for SPIDER and Sudirman: scores, raw confidences, features, fusion outcome."""
    a = det.anchors
    fus = det.fusion
    matched = score.matched
    numbering_ok = matched > 0 and score.label_correct == matched
    out: dict[str, Any] = {
        "score": {
            "n_truth": score.n_truth,
            "n_pred": score.n_pred,
            "matched": score.matched,
            "label_correct": score.label_correct,
            "centre_err_mm": [float(v) for v in score.centre_err_mm],
            "normal_err_deg": [float(v) for v in score.normal_err_deg],
            "midline_err_mm": score.midline_err_mm,
        },
        "matches": [_match_dict(m) for m in score.matches],
        "missed": list(score.missed),
        "raw_disc_confidence": det.raw_disc_confidence,
        "raw_numbering_confidence": det.raw_numbering_confidence,
        "disc_features": {k: [float(x) for x in v] for k, v in det.disc_features.items()},
        "volume_features": [float(x) for x in det.volume_features]
        if det.volume_features is not None
        else None,
        "numbering_correct": bool(numbering_ok),
        "numbering_status_uncal": a.numbering.get("status"),
        "sacrum_below": bool(a.numbering.get("sacrum_below", False)),
        "offset": a.numbering.get("offset"),
        "applied_offset": a.numbering.get("applied_offset"),
        "relabelled": bool(a.numbering.get("relabelled", False)),
        "transitional_candidate": bool(a.numbering.get("transitional_candidate", False)),
        "p_applied": float(fus.p_applied) if fus is not None else None,
        "fusion": fusion_record(fus) if fus is not None else None,
    }
    return out


def _sudirman_record(job: tuple[str, str | None]) -> dict[str, Any]:
    """Worker: one labelled Sudirman patient, scored level-wise against its annotated discs."""
    from spinescope.anatomy._validate_sudirman import score_levels, truth_discs
    from spinescope.datasets.sudirman import patient_sessions, select_session

    pid, labels_root = job
    rec: dict[str, Any] = {"id": pid, "error": None}
    t0 = time.perf_counter()
    try:
        sessions = patient_sessions()
        if pid not in sessions:
            rec["error"] = "no DICOM folder"
            return rec
        sel = select_session(pid, [Path(s) for s in sessions[pid]], labels_root)
        if sel is None:
            rec["error"] = "no session reproduces the label slices"
            return rec
        _sess, study, levels = sel
        det = label_spine_details(study, backend="ml", calibrate=False)
        rows = score_levels(det.anchors, levels)
        score = score_anchors(det.anchors, truth_discs(levels), None)
        rec.update(_record_common(det, score))
        if det.unfused is not None:
            ur = score_levels(det.unfused, levels)
            rec["unfused"] = {
                "matched": sum(1 for r in ur if r.matched),
                "label_correct": sum(1 for r in ur if r.matched and r.name_ok),
                "n_pred": len(det.unfused.discs),
            }
        rec["levels"] = [
            {
                "level": r.level,
                "predicted": r.predicted,
                "matched": r.matched,
                "name_ok": r.name_ok,
                "dist_mm": _finite(r.dist_mm),
            }
            for r in rows
        ]
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return rec


def _run_jobs(fn: Any, jobs: list[Any], workers: int) -> Iterator[dict[str, Any]]:
    """Yield the records in job order as they complete (so an interrupted run keeps what it finished)."""
    if workers <= 1:
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


def cmd_collect(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if args.sudirman:
        cache = out / "sudirman"
        from spinescope.datasets.sudirman import labelled_patients

        ids = labelled_patients(None)
        if args.limit and args.limit < len(ids):  # evenly spaced subset (documented in the report)
            ids = [ids[round(i)] for i in np.linspace(0, len(ids) - 1, args.limit)]
        todo = [(p, None) for p in ids if not (cache / f"{p}.json").is_file()]
        log.info("sudirman: %d labelled, %d to run", len(ids), len(todo))
        for rec in _run_jobs(_sudirman_record, todo, args.jobs):
            (cache).mkdir(parents=True, exist_ok=True)
            (cache / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        return 0
    # A separate cache folder for the classical second opinion, so the default records are never mixed with it.
    cache = out / (args.split + ("_classical" if args.classical_second_opinion else ""))
    cache.mkdir(parents=True, exist_ok=True)
    files = _spider_files(args.split)
    if args.limit:  # evenly spaced subset, recorded so that the fit uses exactly this set
        files = [files[round(i)] for i in np.linspace(0, len(files) - 1, args.limit)]
        (out / f"{args.split}_subset.json").write_text(json.dumps([n for n, _ in files]), encoding="utf-8")
    todo = [
        (name, manu, args.split, bool(args.classical_second_opinion))
        for name, manu in files
        if not (cache / f"{name}.json").is_file()
    ]
    log.info("%s: %d volumes, %d to run", args.split, len(files), len(todo))
    for rec in _run_jobs(_spider_record, todo, args.jobs):
        (cache / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        log.info("%s %s %s", rec["id"], rec.get("error") or "ok", rec.get("elapsed_s"))
    return 0


# ---------------------------------------------------------------- fit and report


def _load(dir_: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(dir_.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if not rec.get("error"):
            out.append(rec)
    return out


def _sudirman_as_matches(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotated levels that were matched (within MATCH_MM) as scored matches: only annotated discs count, so
    unannotated predicted discs are not treated as wrong (the Sudirman truth has three discs per patient)."""
    out = []
    for r in recs:
        matches = []
        for lv in r.get("levels", []):
            if not lv["matched"] or lv["predicted"] is None:
                continue
            raw = r["raw_disc_confidence"].get(lv["predicted"])
            if raw is None:
                continue
            matches.append(
                {
                    "pred": lv["predicted"],
                    "pred_confidence": float(raw),
                    "truth": lv["level"],
                    "centre_err_mm": lv["dist_mm"],
                    "normal_err_deg": None,
                    "label_correct": bool(lv["name_ok"]),
                    "level_offset": None,
                }
            )
        r["matches"] = matches
        r["score"] = {
            "n_truth": len(r.get("levels", [])),
            "n_pred": len(r["raw_disc_confidence"]),
            "matched": len(matches),
            "label_correct": sum(1 for m in matches if m["label_correct"]),
            "centre_err_mm": [float(m["centre_err_mm"]) for m in matches if m["centre_err_mm"] is not None],
            "normal_err_deg": [],
            "midline_err_mm": None,
        }
        r["numbering_correct"] = bool(matches) and all(m["label_correct"] for m in matches)
        out.append(r)
    return out


def _disc_xy(recs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for r in recs:
        for m in r["matches"]:
            feats = r["disc_features"].get(m["pred"])
            if feats is None:
                continue
            X.append(feats)
            y.append(1.0 if m["label_correct"] else 0.0)
    return np.asarray(X, dtype=np.float64).reshape(-1, len(DISC_FEATURES)), np.asarray(y, dtype=np.float64)


def _volume_xy(recs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    X = [r["volume_features"] for r in recs if r.get("volume_features") is not None]
    y = [1.0 if r["numbering_correct"] else 0.0 for r in recs if r.get("volume_features") is not None]
    return np.asarray(X, dtype=np.float64).reshape(-1, len(VOLUME_FEATURES)), np.asarray(y, dtype=np.float64)


def fit_calibration(train: list[dict[str, Any]]) -> Calibration:
    """Disc and volume logistic models and status thresholds, fitted on SPIDER training records only."""
    Xd, yd = _disc_xy(train)
    disc = Logistic.fit(Xd, yd, DISC_FEATURES)
    Xv, yv = _volume_xy(train)
    vol = Logistic.fit(Xv, yv, VOLUME_FEATURES)
    p_vol = vol.predict(Xv)
    th = choose_thresholds(p_vol, yv)
    return Calibration(
        disc=disc,
        volume=vol,
        thresholds=th,
        source=f"SPIDER training: {len(train)} patients, {len(yd)} predicted discs, {len(yv)} studies",
    )


def _scores_from_matches(recs: list[dict[str, Any]], conf_of: Any) -> list[VolumeScore]:
    """One VolumeScore per record with each match's confidence replaced by ``conf_of(record, match)``."""
    out: list[VolumeScore] = []
    for r in recs:
        s = r["score"]
        matches = []
        for m in r["matches"]:
            mm = DiscMatch(**{**m, "pred_confidence": float(conf_of(r, m))})
            matches.append(mm)
        out.append(
            VolumeScore(
                n_truth=int(s["n_truth"]),
                n_pred=int(s["n_pred"]),
                matched=int(s["matched"]),
                label_correct=int(s["label_correct"]),
                centre_err_mm=[float(v) for v in s["centre_err_mm"]],
                normal_err_deg=[float(v) for v in s["normal_err_deg"]],
                matches=matches,
            )
        )
    return out


def _disc_probs(cal: Calibration, r: dict[str, Any], m: dict[str, Any]) -> float:
    feats = r["disc_features"].get(m["pred"])
    if feats is None:
        return 0.0
    return float(cal.disc.predict(np.asarray(feats))[0])


def _rel_block(scores: list[VolumeScore]) -> dict[str, float | None]:
    rel = reliability(scores, n_bins=10)
    wrong = [m.pred_confidence for s in scores for m in s.matches if not m.label_correct]
    right = [m.pred_confidence for s in scores for m in s.matches if m.label_correct]
    return {
        "ece": round(rel["ece"], 4),
        "auroc": None if rel["auroc"] is None else round(rel["auroc"], 4),
        "n": int(rel["n"]),
        "mean_conf_wrong": round(float(np.mean(wrong)), 4) if wrong else None,
        "mean_conf_right": round(float(np.mean(right)), 4) if right else None,
        "wrong_below_0.5": round(float(np.mean(np.asarray(wrong) < 0.5)), 4) if wrong else None,
    }


def _volume_block(recs: list[dict[str, Any]], conf: list[float]) -> dict[str, Any]:
    """Numbering reliability over studies: one pseudo-match per study so the shared ``reliability`` applies."""
    scores = [
        VolumeScore(
            n_truth=1,
            n_pred=1,
            matched=1,
            label_correct=int(bool(r["numbering_correct"])),
            matches=[
                DiscMatch(
                    pred="numbering",
                    pred_confidence=float(c),
                    truth="numbering",
                    centre_err_mm=None,
                    normal_err_deg=None,
                    label_correct=bool(r["numbering_correct"]),
                    level_offset=None,
                )
            ],
        )
        for r, c in zip(recs, conf, strict=True)
    ]
    rel = _rel_block(scores)
    ok = np.asarray([bool(r["numbering_correct"]) for r in recs])
    c = np.asarray(conf)
    wrong = c[~ok]
    out: dict[str, Any] = {"studies": len(recs), "numbering_correct_rate": round(float(ok.mean()), 4), **rel}
    out["mean_conf_wrong"] = round(float(wrong.mean()), 4) if wrong.size else None
    out["mean_conf_right"] = round(float(c[ok].mean()), 4) if ok.any() else None
    out["wrong_below_0.5"] = round(float((wrong < 0.5).mean()), 4) if wrong.size else None
    return out


def _capped_status(rec: dict[str, Any], conf: float, thr: dict[str, float]) -> str:
    """The pipeline's status: thresholds on the calibrated value, then the same hard caps as ``_finalise``."""
    from spinescope.anatomy.pipeline import _apply_status_caps

    second_extra = bool(
        rec.get("fusion")
        and rec["fusion"].get("second_series")
        and rec["fusion"]["second_series"]["extra_below"]
    )
    return _apply_status_caps(
        numbering_status(conf, thr),
        sacrum_below=bool(rec.get("sacrum_below")),
        applied=bool(rec.get("relabelled")),
        second_extra=second_extra,
    )


def _status_table(recs: list[dict[str, Any]], conf: list[float], thr: dict[str, float]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    statuses = [_capped_status(r, c, thr) for r, c in zip(recs, conf, strict=True)]
    for st in (STATUS_CONFIRMED, STATUS_PROBABLE, "unconfirmed"):
        idx = [i for i, s in enumerate(statuses) if s == st]
        ok = [bool(recs[i]["numbering_correct"]) for i in idx]
        table[st] = {"studies": len(idx), "precision": round(float(np.mean(ok)), 4) if ok else None}
    return table


def _label_accuracy(recs: list[dict[str, Any]], which: str) -> dict[str, float | None]:
    """Pooled label accuracy (exact name) of matched discs, before ('unfused') or after ('fused') numbering fusion."""
    if which == "fused":
        scores = [r["score"] for r in recs]
        m = sum(s["matched"] for s in scores)
        c = sum(s["label_correct"] for s in scores)
    else:
        us = [r["unfused"] for r in recs if r.get("unfused")]
        m = sum(u["matched"] for u in us)
        c = sum(u["label_correct"] for u in us)
    return {"matched": m, "label_accuracy": round(c / m, 4) if m else None}


def cmd_fit(args: argparse.Namespace) -> int:
    out = Path(args.out)
    train = _load(out / "training")
    subset_file = out / "training_subset.json"
    if subset_file.is_file():  # only the recorded evenly spaced subset is used for the fit
        keep = set(json.loads(subset_file.read_text(encoding="utf-8")))
        train = [r for r in train if r["id"] in keep]
    val = _load(out / "validation")
    sud = _sudirman_as_matches(_load(out / "sudirman"))
    if not train:
        log.error("no training records under %s/training", out)
        return 2
    cal = fit_calibration(train)
    report: dict[str, Any] = {
        "calibration_source": cal.source,
        "thresholds": cal.thresholds,
        "disc_weights": dict(zip(DISC_FEATURES, [round(w, 4) for w in cal.disc.weights], strict=True)),
        "volume_weights": dict(zip(VOLUME_FEATURES, [round(w, 4) for w in cal.volume.weights], strict=True)),
        "disc_bias": round(cal.disc.bias, 4),
        "volume_bias": round(cal.volume.bias, 4),
    }
    for name, recs in (("training", train), ("validation", val), ("sudirman", sud)):
        if not recs:
            continue
        block: dict[str, Any] = {"patients": len(recs), "errors": 0}
        if name != "sudirman":
            block["metrics_fused"] = {
                k: _round(v)
                for k, v in aggregate(_scores_from_matches(recs, lambda r, m: m["pred_confidence"])).items()
            }
            block["relabelled_studies"] = sum(1 for r in recs if r.get("relabelled"))
        block["label_accuracy_unfused"] = _label_accuracy(recs, "unfused")
        block["label_accuracy_fused"] = _label_accuracy(recs, "fused")
        block["relabelled_studies"] = sum(1 for r in recs if r.get("relabelled"))
        before = _rel_block(_scores_from_matches(recs, lambda r, m: m["pred_confidence"]))
        after = _rel_block(_scores_from_matches(recs, lambda r, m: _disc_probs(cal, r, m)))
        block["disc_reliability"] = {"before_raw": before, "after_calibrated": after}
        raw_num = [float(r["raw_numbering_confidence"] or 0.0) for r in recs]
        uncal = [float(r.get("p_applied") or 0.0) * (1.0 if r.get("sacrum_below") else 0.9) for r in recs]
        cal_num = [
            float(cal.volume.predict(np.asarray(r["volume_features"]))[0])
            for r in recs
            if r.get("volume_features") is not None
        ]
        recs_v = [r for r in recs if r.get("volume_features") is not None]
        block["numbering_reliability"] = {
            "before_heuristic": _volume_block(recs, raw_num),
            "fallback_uncalibrated": _volume_block(recs, uncal),
            "after_calibrated": _volume_block(recs_v, cal_num),
        }
        block["status_after"] = _status_table(recs_v, cal_num, cal.thresholds)
        report[name] = block
    report["scored_by"] = "spinescope.anatomy._validate.reliability / aggregate"
    (out / "calibration_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    if args.write:
        cal.save(calibration_path())
        log.info("wrote %s", calibration_path())
    return 0


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="run the pipeline and cache per-patient records")
    c.add_argument("--split", default="training", choices=("training", "validation"))
    c.add_argument("--sudirman", action="store_true")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--out", required=True)
    c.add_argument("--jobs", type=int, default=3)
    c.add_argument(
        "--classical-second-opinion",
        action="store_true",
        help="label with the classical second opinion on (Config.classical_second_opinion; SPIDER split only)",
    )
    c.set_defaults(func=cmd_collect)
    f = sub.add_parser("fit", help="fit on training records and report")
    f.add_argument("--out", required=True)
    f.add_argument("--write", action="store_true", help="write the packaged calibration_v1.json")
    f.set_defaults(func=cmd_fit)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
