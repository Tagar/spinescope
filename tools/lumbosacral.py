#!/usr/bin/env python3
"""Lumbosacral junction (lab13): train the patch classifier, select its thresholds, evaluate before and after.

Sub-commands, in order (all outputs under ``--out``, nothing derived from SPIDER or Sudirman is committed)::

    # 1. SPIDER training patches: L5-S1 (positive) against L4-L5, L3-L4, the S1-S2 remnant region and the S2 region
    python tools/lumbosacral.py patches --out /tmp/spinescope-lab13_lumbosacral --jobs 2

    # 2. train the 2-D CNN on 85 % of the training volumes (torch, CPU, 4 threads); export ONNX
    python tools/lumbosacral.py train --out /tmp/spinescope-lab13_lumbosacral

    # 3. select the drop threshold on the held-out 15 % of training volumes, run on the labeller's own candidates
    python tools/lumbosacral.py select --out /tmp/spinescope-lab13_lumbosacral [--write]
       (--write copies the ONNX file and thresholds into src/spinescope/anatomy/)

    # 4. full default pipeline: before = labeller v1 (unfused), after = with the module (needs --write first)
    python tools/lumbosacral.py evaluate --split validation --out /tmp/spinescope-lab13_lumbosacral --jobs 2
    python tools/lumbosacral.py evaluate --sudirman --limit 0 --out /tmp/spinescope-lab13_lumbosacral --jobs 2

    # 5. aggregate (stdout)
    python tools/lumbosacral.py report --out /tmp/spinescope-lab13_lumbosacral

Truth. SPIDER: the mask label 201 (named L5-S1 by rank from the bottom, as in the scorer) is the positive. Negatives
are the L4-L5 and L3-L4 discs (masks 202, 203), and the region 33 to 38 mm below L5-S1 (an S1-S2 remnant) and 55 to 65 mm
below it (the sacrum), where no true disc lies. A candidate is correct when it matches the L5-S1 truth within 12 mm.
Sudirman: the annotated D5 slab is L5-S1 (dataset convention, see docs/labeling/lab02_sudirman.md).

The held-out training volumes (every 7th of the sorted training ids) are never used to fit the CNN.
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
from pathlib import Path
from typing import Any

import numpy as np

from spinescope.anatomy import lumbosacral as L
from spinescope.anatomy._validate import score_anchors, truth_from_mask
from spinescope.anatomy.ml import set_intra_op_threads
from spinescope.anatomy.ml.grid import intensity_window, plane_grid, resample_slice
from spinescope.config import DEFAULT

log = logging.getLogger("lumbosacral")
SPIDER_ROOT = Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public")) / "spider"
SEED = 20260
INTRA_OP = 2  # ORT threads per worker process; two workers keep the total at four
TORCH_THREADS = 4
HOLD_OUT_EVERY = 7  # every 7th sorted training id is held out from the CNN fit (threshold selection only)
DROP_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MIN_IN_FOV = 0.7  # a patch with less than this share of acquired field of view is not used


# ---------------------------------------------------------------- SPIDER files and held-out split


def _spider_files(split: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(SPIDER_ROOT / "overview.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("new_file_name") or "").strip()
            if row.get("subset", "").strip() != split or not name:
                continue
            if not (SPIDER_ROOT / "images" / f"{name}.mha").is_file():
                continue
            if not (SPIDER_ROOT / "masks" / f"{name}.mha").is_file():
                continue
            out.append((name, (row.get("Manufacturer") or "").split(" ")[0] or "unknown"))
    return sorted(out, key=lambda t: (int(t[0].split("_")[0]), t[0]))


def _held_out_ids() -> list[str]:
    ids = [n for n, _ in _spider_files("training")]
    return [n for i, n in enumerate(ids) if i % HOLD_OUT_EVERY == 0]


# ---------------------------------------------------------------- patches (truth-based, training volumes)


def _volume_patches(name: str) -> list[tuple[np.ndarray, int, str]]:
    """Patches of one SPIDER training volume: (patch, label, kind). Empty when L5-S1 is not in the mask."""
    from spinescope.study import load_study

    rng = np.random.default_rng(SEED + sum(map(ord, name)))
    study = load_study(str(SPIDER_ROOT / "images" / f"{name}.mha"))
    stack = study.best("sag")
    if stack is None:
        return []
    truth, _mx = truth_from_mask(str(SPIDER_ROOT / "masks" / f"{name}.mha"))
    by = {t.name: t for t in truth}
    if "L5-S1" not in by:
        return []
    normal = np.asarray(stack.normal, dtype=np.float64)
    t_plane = float(np.mean([np.asarray(t.center) @ normal for t in truth]))
    k0 = int(np.argmin(np.abs(np.asarray(stack.positions, dtype=np.float64) - t_plane)))
    window = intensity_window(stack.vol, DEFAULT.ml_intensity_pct, DEFAULT.background_frac)
    ones = np.ones(stack.vol.shape, np.float32)
    out: list[tuple[np.ndarray, int, str]] = []

    def add(center: np.ndarray, dk: int, dd: float, dp: float, y: int, kind: str) -> None:
        kk = int(np.clip(k0 + dk, 0, stack.n - 1))
        g = plane_grid(stack, kk, 1.0)
        down, post = (float(v) for v in g.mm_of(center)[:2])
        down, post = down + dd, post + dp
        cov = resample_slice(stack, L.patch_grid(g, down, post), array=ones, order=1) > 0.5
        if float(cov.mean()) < MIN_IN_FOV:
            return
        out.append((L.disc_patch(stack, g, down, post, window), y, kind))

    l5 = np.asarray(by["L5-S1"].center, dtype=np.float64)
    for dk in (-1, 0, 1):  # positives: the L5-S1 disc on the mid plane and its neighbours, jittered
        add(l5, dk, *rng.uniform(-2.0, 2.0, size=2), 1, "L5-S1")
    for name_, kind in (("L4-L5", "L4-L5"), ("L3-L4", "L3-L4")):  # the discs above: not L5-S1
        if name_ in by:
            add(np.asarray(by[name_].center, dtype=np.float64), 0, *rng.uniform(-2.0, 2.0, size=2), 0, kind)
    for _ in range(2):  # S1-S2 remnant region: 33 to 38 mm below L5-S1 (negatives)
        off = float(rng.uniform(33.0, 38.0))
        add(l5, int(rng.integers(-1, 2)), off, float(rng.uniform(-3.0, 3.0)), 0, "S1-S2")
    add(l5, 0, float(rng.uniform(55.0, 65.0)), float(rng.uniform(-3.0, 3.0)), 0, "S2")
    return out


def _patch_job(name: str) -> tuple[str, list[tuple[np.ndarray, int, str]]]:
    return name, _volume_patches(name)


def cmd_patches(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [n for n, _ in _spider_files("training")]
    held = set(_held_out_ids())
    (out / "heldout.json").write_text(json.dumps(sorted(held)), encoding="utf-8")
    X, y, vol, kind = [], [], [], []
    for name, samples in _run_jobs(_patch_job, files, args.jobs):
        for patch, lab, kd in samples:
            X.append(patch.astype(np.float16))
            y.append(lab)
            vol.append(name)
            kind.append(kd)
    np.savez_compressed(
        out / "patches_train.npz",
        X=np.stack(X),
        y=np.asarray(y, np.int64),
        vol=np.asarray(vol),
        kind=np.asarray(kind),
    )
    counts: dict[str, int] = {}
    for kd in kind:
        counts[kd] = counts.get(kd, 0) + 1
    log.info("patches: %d volumes, %d patches %s, %d held out", len(files), len(y), counts, len(held))
    return 0


def _init_worker() -> None:
    set_intra_op_threads(INTRA_OP)


def _run_jobs(fn: Any, jobs: list[Any], workers: int) -> Iterator[Any]:
    if workers <= 1:
        _init_worker()
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


# ---------------------------------------------------------------- CNN (dev-time: torch)


def _net() -> Any:
    import torch.nn as nn

    def block(ci: int, co: int) -> list[Any]:
        return [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]

    class LumbosacralNet(nn.Module):  # type: ignore[misc]
        """Four conv stages (96 -> 6 px), flattened so the position of the disc in the patch is kept."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                *block(1, 16),
                *block(16, 16),
                nn.MaxPool2d(2),
                *block(16, 32),
                *block(32, 32),
                nn.MaxPool2d(2),
                *block(32, 64),
                *block(64, 64),
                nn.MaxPool2d(2),
                *block(64, 64),
                nn.MaxPool2d(2),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64 * 6 * 6, 1))

        def forward(self, x: Any) -> Any:
            return self.head(self.features(x))

    return LumbosacralNet()


def _augment(x: Any, gen: Any) -> Any:
    """Random gamma, contrast, brightness and noise on a batch in [0, 1] (seeded generator)."""
    import torch

    n = x.shape[0]
    gam = torch.empty(n, 1, 1, 1).uniform_(0.8, 1.25, generator=gen)
    con = torch.empty(n, 1, 1, 1).uniform_(0.85, 1.15, generator=gen)
    bri = torch.empty(n, 1, 1, 1).uniform_(-0.05, 0.05, generator=gen)
    x = x.clamp(1e-6, 1.0) ** gam
    m = x.mean(dim=(2, 3), keepdim=True)
    x = (x - m) * con + m + bri
    x = x + 0.02 * torch.randn(x.shape, generator=gen)
    return x.clamp(0.0, 1.0)


def cmd_train(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(TORCH_THREADS)
    torch.manual_seed(SEED)
    out = Path(args.out)
    data = np.load(out / "patches_train.npz")
    held = set(json.loads((out / "heldout.json").read_text(encoding="utf-8")))
    keep = np.array([v not in held for v in data["vol"]])
    X = torch.from_numpy(data["X"][keep].astype(np.float32))[:, None]
    y = torch.from_numpy(data["y"][keep].astype(np.float32))
    net = _net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs, batch = args.epochs, 64
    steps = epochs * int(np.ceil(len(y) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED)
    hist: list[float] = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y), generator=gen)
        tot = 0.0
        for lo in range(0, len(y), batch):
            idx = perm[lo : lo + batch]
            xb = _augment(X[idx], gen)
            opt.zero_grad()
            loss = loss_fn(net(xb).squeeze(1), y[idx])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        hist.append(round(tot / len(y), 5))
        log.info("epoch %d loss %.4f", ep + 1, hist[-1])
    net.eval()
    dummy = torch.zeros(1, 1, L.PATCH_MM, L.PATCH_MM)
    onnx_path = out / L.MODEL_ONNX
    torch.onnx.export(
        net,
        dummy,
        str(onnx_path),
        input_names=["patch"],
        output_names=["logit"],
        dynamic_axes={"patch": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    meta = {
        "epochs": epochs,
        "batch": batch,
        "seed": SEED,
        "train_patches": int(len(y)),
        "train_positive": int(y.sum().item()),
        "train_volumes_excluded_held_out": len(held),
        "loss_history": hist,
        "torch": torch.__version__,
        "onnx_bytes": onnx_path.stat().st_size,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log.info("wrote %s (%d bytes)", onnx_path, onnx_path.stat().st_size)
    return 0


# ---------------------------------------------------------------- threshold selection (held-out training volumes)


def _candidate_job(job: tuple[str, str]) -> dict[str, Any]:
    """Worker: the labeller's lowest candidates of one training volume, with the CNN probabilities."""
    from spinescope.anatomy.ml.labeller import segment_primary
    from spinescope.anatomy.ml.postprocess import anchors_from_segmentation
    from spinescope.study import load_study

    name, onnx = job
    model = L.LumbosacralModel(onnx_path=Path(onnx))
    rec: dict[str, Any] = {"id": name, "error": None}
    try:
        study = load_study(str(SPIDER_ROOT / "images" / f"{name}.mha"))
        seg = segment_primary(study, DEFAULT)
        if seg is None:
            raise RuntimeError("no sagittal series")
        an = anchors_from_segmentation(seg, DEFAULT)
        cands, patches = L.candidate_patches(seg, an, DEFAULT)
        truth, mx = truth_from_mask(str(SPIDER_ROOT / "masks" / f"{name}.mha"))
        matched = {m.pred: m.truth for m in score_anchors(an, truth, mx).matches}
        rec["drop_truth"] = next((c.rank for c in cands if matched.get(c.label) == "L5-S1"), None)
        rec["probs"] = [float(p) for p in model.predict(patches)] if len(cands) else []
        rec["n_cands"] = len(cands)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


class _Fixed:
    """A scorer with fixed probabilities (threshold sweeps re-use one CNN pass)."""

    def __init__(self, probs: list[float], cfg: L.LumbosacralConfig) -> None:
        self._p = np.asarray(probs, dtype=np.float64)
        self.config = cfg

    def predict(self, patches: Any) -> np.ndarray:
        return self._p


def _sweep(recs: list[dict[str, Any]], drop_min_p: float) -> dict[str, int]:
    cfg = L.LumbosacralConfig(drop_min_p=drop_min_p)
    c = {"n": 0, "correct": 0, "wrong_drop": 0, "missed_drop": 0, "unsure": 0, "drops": 0}
    for r in recs:
        if r.get("error") or r.get("drop_truth") is None or not r["probs"]:
            continue
        cands = [L.Candidate(label=str(i), rank=i) for i in range(len(r["probs"]))]
        dec = L.decide(cands, np.zeros((len(cands), L.PATCH_MM, L.PATCH_MM), np.float32), _Fixed(r["probs"], cfg))
        truth = int(r["drop_truth"])
        c["n"] += 1
        if dec.status == "drop":
            c["drops"] += 1
            c["correct" if dec.drop == truth else "wrong_drop"] += 1
        elif dec.status == "keep":
            c["correct" if truth == 0 else "missed_drop"] += 1
        else:
            c["unsure"] += 1  # not wrong, but not a correct decision either
    return c


def cmd_select(args: argparse.Namespace) -> int:
    out = Path(args.out)
    onnx = out / L.MODEL_ONNX
    if not onnx.is_file():
        log.error("no model at %s: run train first", onnx)
        return 1
    model = L.LumbosacralModel(onnx_path=onnx)
    held = json.loads((out / "heldout.json").read_text(encoding="utf-8"))
    cache = out / "select"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [n for n in held if not (cache / f"{n}.json").is_file()]
    for rec in _run_jobs(_candidate_job, [(n, str(onnx)) for n in todo], args.jobs):
        (cache / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache.glob("*.json"))]
    sweep = {str(g): _sweep(recs, g) for g in DROP_GRID}

    def score(c: dict[str, int]) -> float:
        return c["correct"] - 2.0 * c["wrong_drop"] - 1.0 * c["missed_drop"]

    best = max(DROP_GRID, key=lambda g: (round(score(sweep[str(g)]), 9), -abs(g - 0.8)))
    cfg = L.LumbosacralConfig(drop_min_p=best)
    meta = json.loads((out / "train_meta.json").read_text(encoding="utf-8")) if (out / "train_meta.json").is_file() else {}
    doc = {
        "version": 1,
        "onnx": L.MODEL_ONNX,
        "patch_mm": L.PATCH_MM,
        "input": "float32 (n, 1, 96, 96), intensity normalised to [0, 1] per volume (ml.grid.normalise)",
        "output": "logit of P(L5-S1); sigmoid applied by the caller",
        "thresholds": {"drop_min_p": cfg.drop_min_p, "keep_min_p": cfg.keep_min_p, "unsure_min_p": cfg.unsure_min_p},
        "training": {
            **meta,
            "held_out_volumes": len(held),
            "selection": "held-out SPIDER training volumes, labeller candidates; objective correct - 2 wrong drops - missed",
            "sweep": sweep,
            "selected_drop_min_p": best,
            "selection_n": sweep[str(best)]["n"],
        },
    }
    (out / L.MODEL_JSON).write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"selected_drop_min_p": best, "sweep": sweep}, indent=1, sort_keys=True))
    if args.write:
        import shutil

        pkg = L.model_dir()
        shutil.copyfile(onnx, pkg / L.MODEL_ONNX)
        shutil.copyfile(out / L.MODEL_JSON, pkg / L.MODEL_JSON)
        log.info("packaged %s and %s in %s", L.MODEL_ONNX, L.MODEL_JSON, pkg)
    return 0


# ---------------------------------------------------------------- evaluate (full pipeline before / after)


def _summ(sc: Any, anchors: Any) -> dict[str, Any]:
    fully = sc.matched == sc.n_truth and sc.label_correct == sc.n_truth and sc.n_pred == sc.n_truth
    below = any(m.truth is None for m in sc.matches)
    return {
        "n_truth": sc.n_truth,
        "n_pred": sc.n_pred,
        "matched": sc.matched,
        "label_correct": sc.label_correct,
        "centre_err_mm": [float(v) for v in sc.centre_err_mm],
        "fully_correct": bool(fully),
        "pred_unmatched": below,
        "status": anchors.numbering.get("status"),
        "confidence": float(anchors.numbering.get("confidence", 0.0) or 0.0),
        "labels": {m.pred: m.truth for m in sc.matches},
    }


def _eval_spider(job: tuple[str, str, str]) -> dict[str, Any]:
    from spinescope.anatomy.pipeline import label_spine_details
    from spinescope.study import load_study

    name, manu, split = job
    rec: dict[str, Any] = {"id": name, "split": split, "vendor": manu, "error": None}
    t0 = time.perf_counter()
    try:
        study = load_study(str(SPIDER_ROOT / "images" / f"{name}.mha"))
        truth, mx = truth_from_mask(str(SPIDER_ROOT / "masks" / f"{name}.mha"))
        det = label_spine_details(study, backend="ml", calibrate=False)
        rec["before"] = _summ(score_anchors(det.unfused, truth, mx), det.unfused)
        rec["after"] = _summ(score_anchors(det.anchors, truth, mx), det.anchors)
        rec["lumbosacral"] = det.anchors.numbering.get("lumbosacral")
        rec["drop_truth"] = _drop_truth(truth, det.unfused)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return rec


def _drop_truth(truth: list[Any], unfused: Any) -> int | None:
    """Rank from the bottom of the unfused disc matched to the L5-S1 truth, or None."""
    matched = {m.pred: m.truth for m in score_anchors(unfused, truth, None).matches}
    order = sorted(unfused.discs, key=lambda lab: unfused.discs[lab].center[2])
    for rank, lab in enumerate(order[::-1]):
        if matched.get(lab) == "L5-S1":
            return rank
    return None


def _eval_sudirman(pid: str) -> dict[str, Any]:
    from spinescope.anatomy._validate_sudirman import score_levels, truth_discs
    from spinescope.anatomy.pipeline import label_spine_details
    from spinescope.datasets.sudirman import patient_sessions, select_session

    rec: dict[str, Any] = {"id": pid, "split": "sudirman", "error": None}
    t0 = time.perf_counter()
    try:
        sessions = patient_sessions()
        if pid not in sessions:
            rec["error"] = "no DICOM folder"
            return rec
        sel = select_session(pid, [Path(s) for s in sessions[pid]], None)
        if sel is None:
            rec["error"] = "no session reproduces the label slices"
            return rec
        _sess, study, levels = sel
        det = label_spine_details(study, backend="ml", calibrate=False)
        after = score_levels(det.anchors, levels)
        before = score_levels(det.unfused, levels)
        rec["after"] = [{"level": r.level, "matched": r.matched, "name_ok": r.name_ok} for r in after]
        rec["before"] = [{"level": r.level, "matched": r.matched, "name_ok": r.name_ok} for r in before]
        rec["status"] = det.anchors.numbering.get("status")
        rec["confidence"] = float(det.anchors.numbering.get("confidence", 0.0) or 0.0)
        rec["lumbosacral"] = det.anchors.numbering.get("lumbosacral")
        rec["drop_truth"] = _drop_truth(truth_discs(levels), det.unfused)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return rec


def cmd_evaluate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if args.sudirman:
        from spinescope.datasets.sudirman import labelled_patients

        cache = out / "eval_sudirman"
        cache.mkdir(parents=True, exist_ok=True)
        ids = labelled_patients(None)
        if args.limit and args.limit < len(ids):
            ids = [ids[round(i)] for i in np.linspace(0, len(ids) - 1, args.limit)]
        (out / "eval_sudirman_subset.json").write_text(json.dumps(ids), encoding="utf-8")
        todo = [p for p in ids if not (cache / f"{p}.json").is_file()]
        log.info("sudirman: %d patients, %d to run", len(ids), len(todo))
        for rec in _run_jobs(_eval_sudirman, todo, args.jobs):
            (cache / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        return 0
    cache = out / f"eval_{args.split}"
    cache.mkdir(parents=True, exist_ok=True)
    files = _spider_files(args.split)
    if args.limit:
        files = [files[round(i)] for i in np.linspace(0, len(files) - 1, args.limit)]
    todo = [(n, m, args.split) for n, m in files if not (cache / f"{n}.json").is_file()]
    log.info("%s: %d volumes, %d to run", args.split, len(files), len(todo))
    for rec in _run_jobs(_eval_spider, todo, args.jobs):
        (cache / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        log.info("%s %s %s", rec["id"], rec.get("error") or "ok", rec.get("elapsed_s"))
    return 0


# ---------------------------------------------------------------- report


def _load_dir(d: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(d.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if not rec.get("error"):
            out.append(rec)
    return out


def _agg_spider(recs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n_truth = sum(r[key]["n_truth"] for r in recs)
    matched = sum(r[key]["matched"] for r in recs)
    correct = sum(r[key]["label_correct"] for r in recs)
    cerr = [e for r in recs for e in r[key]["centre_err_mm"]]
    npred = sum(r[key]["n_pred"] for r in recs)
    fully = sum(1 for r in recs if r[key]["fully_correct"])
    wrong = [r for r in recs if not r[key]["fully_correct"]]
    return {
        "volumes": len(recs),
        "disc_recall": round(matched / max(n_truth, 1), 3),
        "disc_precision": round(matched / max(npred, 1), 3),
        "label_accuracy_all_truth": round(correct / max(n_truth, 1), 3),
        "centre_median_mm": round(float(np.median(cerr)), 2) if cerr else None,
        "centre_p90_mm": round(float(np.percentile(cerr, 90)), 2) if cerr else None,
        "fully_correct_volumes": fully,
        "fully_correct_share": round(fully / max(len(recs), 1), 3),
        "volumes_with_pred_below_lowest_truth": sum(1 for r in recs if r[key]["pred_unmatched"]),
        "wrong_volumes": len(wrong),
        "wrong_but_probable_or_confirmed": sum(1 for r in wrong if r[key]["status"] in ("confirmed", "probable")),
        "status_counts": _count([r[key]["status"] for r in recs]),
    }


def _agg_sudirman(recs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    lv = [x for r in recs for x in r[key]]
    matched = [x for x in lv if x["matched"]]
    ok = [x for x in matched if x["name_ok"]]
    full = sum(1 for r in recs if all(x["matched"] and x["name_ok"] for x in r[key]))
    l5 = [x for r in recs for x in r[key] if x["level"] == "L5-S1"]
    wrong = [r for r in recs if not all(x["matched"] and x["name_ok"] for x in r[key])]
    return {
        "patients": len(recs),
        "levels": len(lv),
        "level_recall": round(len(matched) / max(len(lv), 1), 3),
        "label_accuracy_matched": round(len(ok) / max(len(matched), 1), 3),
        "label_accuracy_all_levels": round(len(ok) / max(len(lv), 1), 3),
        "L5-S1_accuracy": round(sum(1 for x in l5 if x["matched"] and x["name_ok"]) / max(len(l5), 1), 3),
        "fully_correct_patients": full,
        "fully_correct_share": round(full / max(len(recs), 1), 3),
        "wrong_patients": len(wrong),
        "wrong_but_probable_or_confirmed": sum(1 for r in wrong if r.get("status") in ("confirmed", "probable")),
        "status_counts": _count([r.get("status") for r in recs]),
    }


def _count(xs: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def _module_summary(recs: list[dict[str, Any]]) -> dict[str, Any]:
    st = [r["lumbosacral"]["status"] for r in recs if r.get("lumbosacral")]
    applied = [r["lumbosacral"]["applied_drop"] for r in recs if r.get("lumbosacral")]
    known = [(r.get("drop_truth"), r["lumbosacral"]) for r in recs if r.get("lumbosacral") and r.get("drop_truth") is not None]
    return {
        "status_counts": _count(st),
        "drops_applied": sum(1 for a in applied if a),
        "drop_truth_known": len(known),
        "drop_correct": sum(1 for t, lsr in known if lsr["applied_drop"] == t),
        "drop_wrong": sum(1 for t, lsr in known if lsr["applied_drop"] != t and lsr["applied_drop"] > 0),
        "truth_needs_drop": sum(1 for t, _ in known if t > 0),
    }


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    report: dict[str, Any] = {}
    for split in ("validation",):
        d = out / f"eval_{split}"
        if d.is_dir():
            recs = _load_dir(d)
            report[split] = {
                "before": _agg_spider(recs, "before"),
                "after": _agg_spider(recs, "after"),
                "module": _module_summary(recs),
            }
    sd = out / "eval_sudirman"
    if sd.is_dir():
        recs = _load_dir(sd)
        report["sudirman"] = {
            "before": _agg_sudirman(recs, "before"),
            "after": _agg_sudirman(recs, "after"),
            "module": _module_summary(recs),
        }
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("patches")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=2)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=40)
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--write", action="store_true", help="copy the ONNX model and thresholds into the package")
    e = sub.add_parser("evaluate")
    e.add_argument("--split", choices=("training", "validation"), default="validation")
    e.add_argument("--sudirman", action="store_true")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--out", required=True)
    e.add_argument("--jobs", type=int, default=2)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    handlers = {
        "patches": cmd_patches,
        "train": cmd_train,
        "select": cmd_select,
        "evaluate": cmd_evaluate,
        "report": cmd_report,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
