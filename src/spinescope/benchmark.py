"""One command for every labelling benchmark, with caching, thread control and a regression check.

:func:`run_benchmark` labels every case of the benchmark suites with the shipped default labeller
(:func:`spinescope.anatomy.label_spine`, backend ``auto``), scores the results with the shared scorers
(:mod:`spinescope.anatomy._validate`, ``_validate_sudirman``, ``_validate_qx5rt``, :mod:`spinescope.testing.robustness`)
and returns one JSON-able dict of numbers. :func:`compare` checks such a dict against the golden file
(``benchmarks/golden.json``) and lists every regression with its tolerance. A suite whose data is not installed is
skipped with a note, so the same command runs in CI on the phantom suites alone. See ``docs/benchmarks.md``.

Per-case labeller outputs are cached under ``<cache>/<labeller fingerprint>/<suite>/``; the fingerprint hashes the
labeller's source and model files, so any change to the labeller recomputes and nothing else does. Scoring always
runs from the cached records, so a change to a scorer never needs a re-run. Outputs never contain paths, file
names or patient or case identifiers: only aggregate numbers, status and robustness outcomes per phantom scenario.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .anatomy._validate import (
    DiscMatch,
    TruthDisc,
    VolumeScore,
    aggregate,
    aggregate_extended,
    reliability,
    score_anchors,
    truth_from_mask,
)
from .anatomy.anchors import Anchors

log = logging.getLogger("spinescope.benchmark")

SCHEMA = 1
#: Benchmark suites in report order. ``phantom`` (below) expands to the two phantom suites.
SUITES: tuple[str, ...] = ("spider", "sudirman", "qx5rt", "phantom_cartoon", "phantom_realistic")
SUITE_GROUPS: dict[str, tuple[str, ...]] = {"phantom": ("phantom_cartoon", "phantom_realistic")}
TITLES: dict[str, str] = {
    "spider": "SPIDER validation (sagittal T1/T2 volumes, mask ground truth)",
    "sudirman": "Sudirman lumbar DICOM (level-wise, annotated L3-L4 to L5-S1)",
    "qx5rt": "OSF qx5rt multi-scanner (level-wise, annotated L1-L2 to L4-L5)",
    "phantom_cartoon": "Robustness suite, cartoon phantom",
    "phantom_realistic": "Robustness suite, realistic phantom",
}
LEVEL_SUITES = ("sudirman", "qx5rt")
PHANTOM_SUITES = ("phantom_cartoon", "phantom_realistic")
#: Labelling backend per suite. Real-data suites use the shipped default (``label_spine``, backend ``auto``). The
#: phantom suites follow the protocol of their robustness tests: the cartoon is graded with the classical chain
#: (tests/robustness/test_robustness.py), the realistic phantom with the learned labeller (test_robustness_realistic.py).
SUITE_BACKEND: dict[str, str] = {
    "spider": "auto",
    "sudirman": "auto",
    "qx5rt": "auto",
    "phantom_cartoon": "classical",
    "phantom_realistic": "ml",
}
STATUSES = ("confirmed", "probable", "unconfirmed")
ANNOUNCED = ("confirmed", "probable")
OUTCOME_RANK = {"silent_error": 0, "unsure": 1, "pass": 2}
DEFAULT_CACHE = Path.home() / ".cache" / "spinescope" / "benchmark"
DEFAULT_OUT = Path("/tmp/spinescope-benchmark")
THREADS_ENV = "SPINESCOPE_THREADS"  # same knob as spinescope.anatomy.ml.labeller.THREADS_ENV
EPS = 1e-9

#: Metrics that are gated by ``--check``: (direction, default tolerance, unit). ``higher``: a drop beyond the tolerance
#: is a regression; ``lower``: a rise beyond it is; ``count``: any increase is; ``outcome`` (per phantom scenario): a
#: lower rank of pass > unsure > silent_error is. Tolerances live in the golden file too, so maintainers can tune them.
RULES: dict[str, tuple[str, float, str]] = {
    "recall": ("higher", 0.005, "rate"),
    "precision": ("higher", 0.005, "rate"),
    "label_accuracy": ("higher", 0.005, "rate"),
    "label_accuracy_all_truth": ("higher", 0.005, "rate"),
    "fully_correct": ("higher", 0.005, "rate"),
    "numbering_correct_rate": ("higher", 0.005, "rate"),
    "confirmed_precision": ("higher", 0.005, "rate"),
    "probable_precision": ("higher", 0.005, "rate"),
    "announced_precision": ("higher", 0.005, "rate"),
    "announced_coverage": ("higher", 0.02, "rate"),
    "centre_err_median_mm": ("lower", 0.5, "mm"),
    "centre_err_p90_mm": ("lower", 1.0, "mm"),
    "midline_err_median_mm": ("lower", 1.0, "mm"),
    "disc_ece": ("lower", 0.01, "decimal"),
    "disc_auroc": ("higher", 0.01, "decimal"),
    "numbering_ece": ("lower", 0.02, "decimal"),
    "numbering_auroc": ("higher", 0.02, "decimal"),
    "wrong_announced": ("count", 0.0, "count"),
    "silent_error": ("count", 0.0, "count"),
    "errors": ("count", 0.0, "count"),
}
#: Metrics printed as percentages in the markdown summary.
RATE_KEYS = frozenset(k for k, v in RULES.items() if v[2] == "rate") | {
    "centre_within_3mm",
    "off_by_one_of_truth",
    "announced_precision",
}


class Unavailable(Exception):
    """A suite whose data is not installed. Reported as skipped with a note, never as an error."""


@dataclass(frozen=True)
class Options:
    """What to run and how. Output folders are handled by :func:`main`, not here."""

    suites: tuple[str, ...] = SUITES
    cache: Path = DEFAULT_CACHE
    jobs: int = 1
    threads: int | None = None  # ONNX Runtime intra-op threads per labelling process (0 = ORT default)
    refresh: bool = False  # ignore cached labeller outputs and recompute them
    spider_split: str = "validation"
    limit: int | None = None  # evenly spaced subsample of the SPIDER, Sudirman and qx5rt cases (None = all)
    scenarios: tuple[str, ...] | None = None  # phantom scenario subset (None = all)


@dataclass
class _Scored:
    """One scored case: shared-scorer match, numbering status and (for level suites) the level rows."""

    meta: dict[str, str]
    vs: VolumeScore
    conf: float
    status: str
    correct: bool  # shared scorer: at least one match and every match carries the right name
    elapsed: float | None
    rows: list[Any] | None = None  # LevelRow list, level suites only
    outcome: str | None = None  # phantom suites only: pass | unsure | silent_error


# ----------------------------------------------------------------------------- suites and fingerprint
def expand_suites(names: Iterable[str] | None) -> tuple[str, ...]:
    """Suite names with groups expanded, in report order; raises ``ValueError`` on an unknown name."""
    wanted: set[str] = set()
    for n in names or SUITES:
        if n in SUITE_GROUPS:
            wanted.update(SUITE_GROUPS[n])
        elif n in SUITES:
            wanted.add(n)
        else:
            raise ValueError(f"unknown suite {n!r}; choose from {', '.join(SUITES)} or 'phantom'")
    return tuple(s for s in SUITES if s in wanted)


def labeller_fingerprint() -> str:
    """Hash of the labeller's inputs: every ``.py``, ``.json`` and ``.onnx`` file of the package the labeller uses.

    Excludes this module, the CLI and the rendering, reporting, viewer and AI packages, which do not change a label.
    """
    pkg = Path(__file__).resolve().parent
    skip_dirs = {"ai", "viewer", "report", "render", "__pycache__"}
    h = hashlib.sha256(f"benchmark-schema-{SCHEMA}".encode())
    for f in sorted(pkg.rglob("*")):
        rel = f.relative_to(pkg)
        if not f.is_file() or f.suffix not in {".py", ".json", ".onnx"}:
            continue
        if rel.parts[0] in skip_dirs or rel.name in {"benchmark.py", "cli.py"}:
            continue
        h.update(str(rel).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- planning (what to run)
def _subsample(items: list[Any], k: int) -> list[Any]:
    idx = sorted({round(x) for x in np.linspace(0, len(items) - 1, k)})
    return [items[i] for i in idx]


def _plan_spider(opts: Options) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .datasets.registry import data_dir

    root = data_dir() / "spider"
    overview = root / "overview.csv"
    if not overview.is_file():
        raise Unavailable("SPIDER is not installed (see docs/datasets.md)")
    jobs: list[dict[str, Any]] = []
    with open(overview, newline="", encoding="utf-8") as fh:
        # Only the technical columns are read: no sex, birth date or serial numbers.
        for row in csv.DictReader(fh):
            name = (row.get("new_file_name") or "").strip()
            if row.get("subset", "").strip() != opts.spider_split or not name:
                continue
            if (
                not (root / "images" / f"{name}.mha").is_file()
                or not (root / "masks" / f"{name}.mha").is_file()
            ):
                continue
            try:
                field_t = f"{float(row.get('MagneticFieldStrength') or 'nan'):.1f} T"
            except ValueError:
                field_t = "unknown"
            meta = {
                "vendor": (row.get("Manufacturer") or "").split(" ")[0] or "unknown",
                "field": field_t,
                "sequence": name.split("_", 1)[1] if "_" in name else name,
            }
            jobs.append({"suite": "spider", "id": name, "root": str(root), "meta": meta, "backend": "auto"})
    if not jobs:
        raise Unavailable(f"no SPIDER volumes in split {opts.spider_split!r}")
    jobs.sort(key=lambda j: (int(j["id"].split("_")[0]), j["id"]))
    available = len(jobs)
    method = f"all {available} volumes of split {opts.spider_split}"
    if opts.limit and opts.limit < available:
        jobs = _subsample(jobs, opts.limit)
        method = f"evenly spaced subsample of {len(jobs)} of {available} volumes of split {opts.spider_split}"
    return jobs, {"method": method, "available": available}


def _plan_sudirman(opts: Options) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .datasets.sudirman import labelled_patients, patient_sessions

    try:
        sessions = patient_sessions()
        ids = [p for p in labelled_patients() if p in sessions]
    except FileNotFoundError:
        raise Unavailable("Sudirman DICOM or label maps are not installed (see docs/datasets.md)") from None
    if not ids:
        raise Unavailable("no labelled Sudirman patients with DICOM folders")
    available = len(ids)
    method = f"all {available} labelled patients"
    if opts.limit and opts.limit < available:
        ids = _subsample(ids, opts.limit)
        method = f"evenly spaced subsample of {len(ids)} of {available} labelled patients"
    jobs = [
        {
            "suite": "sudirman",
            "id": pid,
            "sessions": [str(s) for s in sessions[pid]],
            "meta": {},
            "backend": "auto",
        }
        for pid in ids
    ]
    return jobs, {"method": method, "available": available}


def _plan_qx5rt(opts: Options) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .anatomy._validate_qx5rt import discover

    try:
        sessions = discover(None)
    except FileNotFoundError:
        raise Unavailable("the qx5rt multi-scanner set is not installed (see docs/datasets.md)") from None
    if not sessions:
        raise Unavailable("no qx5rt sessions with a T1 and disc masks")
    available = len(sessions)
    method = f"all {available} sessions"
    if opts.limit and opts.limit < available:
        sessions = _subsample(sessions, opts.limit)
        method = f"evenly spaced subsample of {len(sessions)} of {available} sessions"
    jobs = [
        {"suite": "qx5rt", "id": s.key, "session": s, "meta": {"vendor": s.vendor}, "backend": "auto"}
        for s in sessions
    ]
    return jobs, {"method": method, "available": available}


def _plan_phantom(suite: str, opts: Options) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .testing.robustness import SCENARIOS

    realism = suite.split("_", 1)[1]
    chosen = [s for s in SCENARIOS if not opts.scenarios or s.name in opts.scenarios]
    if not chosen:
        raise Unavailable("no phantom scenario selected")
    jobs = [
        {
            "suite": suite,
            "id": s.name,
            "scenario": s.name,
            "realism": realism,
            "meta": {"group": s.group},
            "backend": SUITE_BACKEND[suite],
        }
        for s in chosen
    ]
    return jobs, {"method": f"all {len(jobs)} scenarios, {realism} phantom", "available": len(SCENARIOS)}


def _plan(suite: str, opts: Options) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if suite == "spider":
        return _plan_spider(opts)
    if suite == "sudirman":
        return _plan_sudirman(opts)
    if suite == "qx5rt":
        return _plan_qx5rt(opts)
    return _plan_phantom(suite, opts)


# ----------------------------------------------------------------------------- per-case work (workers)
def _json_vec(v: Any) -> list[float]:
    return [float(x) for x in np.asarray(v, dtype=np.float64).ravel()]


def _truth_json(discs: Sequence[TruthDisc], midline: float | None) -> dict[str, Any]:
    return {
        "discs": [
            {
                "label": int(t.label),
                "name": t.name,
                "center": _json_vec(t.center),
                "normal": _json_vec(t.normal),
            }
            for t in discs
        ],
        "midline_x": None if midline is None else float(midline),
    }


def _truth_from_json(d: dict[str, Any]) -> tuple[list[TruthDisc], float | None]:
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


def _label(study: Any, backend: str) -> tuple[dict[str, Any], float]:
    """The labeller on one study: anchors as a dict and the labelling time (no loading)."""
    from .anatomy import label_spine

    t0 = time.perf_counter()
    anchors = label_spine(study, backend=backend)
    return anchors.to_dict(), round(time.perf_counter() - t0, 3)


def _spider_study(root: Path, name: str) -> Any:
    """Study of one SPIDER file. A T2 volume is labelled with its T1 sibling, the protocol of the lab18/lab22 numbers."""
    from .study import from_stacks, load_study

    primary = load_study(str(root / "images" / f"{name}.mha"))
    t1 = root / "images" / f"{name.rsplit('_', 1)[0]}_t1.mha"
    if name.endswith("_t2") and t1.is_file():
        sib = load_study(str(t1))
        st1, st2 = sib.best("sag"), primary.best("sag")
        if st1 is not None and st2 is not None:
            return from_stacks([st2, st1])
    return primary


def _level_json(t: Any) -> dict[str, Any]:
    d = {k: getattr(t, k) for k in t.__dataclass_fields__}
    d["center"] = _json_vec(d["center"])
    if "normal" in d:
        d["normal"] = _json_vec(d["normal"])
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in d.items()}


def _compute(job: dict[str, Any]) -> dict[str, Any]:
    """Label one case from its inputs (no cache). Failures are recorded, not raised."""
    suite = job["suite"]
    backend = job["backend"]
    rec: dict[str, Any] = {"id": job["id"], "meta": job.get("meta", {}), "anchors": None, "error": None}
    try:
        if suite == "spider":
            root = Path(job["root"])
            study = _spider_study(root, job["id"])
            discs, mx = truth_from_mask(str(root / "masks" / f"{job['id']}.mha"))
            rec["truth"] = _truth_json(discs, mx)
            rec["anchors"], rec["elapsed_s"] = _label(study, backend)
        elif suite == "sudirman":
            from .datasets.sudirman import select_session

            sel = select_session(job["id"], [Path(s) for s in job["sessions"]], None)
            if sel is None:
                raise ValueError("no session reproduces the label slices")
            _sess, study, levels = sel
            rec["levels"] = [_level_json(t) for t in levels]
            rec["anchors"], rec["elapsed_s"] = _label(study, backend)
        elif suite == "qx5rt":
            from .anatomy._validate_qx5rt import truth_levels
            from .io.volumes import read_volume
            from .study import from_stacks

            session = job["session"]
            stack = read_volume(str(session.t1))
            qx_levels = truth_levels(session, stack)
            rec["levels"] = [
                {"level": t.level, "center": _json_vec(t.center), "voxels": int(t.voxels)} for t in qx_levels
            ]
            rec["anchors"], rec["elapsed_s"] = _label(from_stacks([stack]), backend)
        else:
            from .testing.robustness import build_case, scenario

            case = build_case(scenario(job["scenario"]), job["realism"])
            rec["truth"] = _truth_json(case.truth, case.phantom.truth.get("midline_x"))
            rec["anchors"], rec["elapsed_s"] = _label(case.study, backend)
    except Exception as exc:  # one failing case is a counted result, not a crash of the run
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        log.debug("case failed: %s", rec["error"])
    return rec


def _cache_file(job: dict[str, Any], cache: Path, fp: str) -> Path:
    # Hashed names: case identifiers never appear as file names.
    digest = hashlib.sha256(f"{job['suite']}|{job['id']}|{job['backend']}".encode()).hexdigest()[:20]
    return cache / fp / job["suite"] / f"{digest}.json"


def _case_job(job: dict[str, Any]) -> dict[str, Any]:
    """Worker: the cached record of one case, or a fresh labelling of it (successes only are cached)."""
    path = Path(job["cache_file"])
    if not job["refresh"] and path.is_file():
        try:
            rec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            rec["cached"] = True
            return rec
        except ValueError:
            log.warning("unreadable cache entry, recomputing")
    rec = _compute(job)
    if rec.get("error") is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, path)
    rec["cached"] = False
    return rec


def _map(
    fn: Callable[[dict[str, Any]], dict[str, Any]], jobs: list[dict[str, Any]], workers: int
) -> list[Any]:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


# ----------------------------------------------------------------------------- scoring (main process)
def _level_scores(suite: str, anchors: Anchors, rec: dict[str, Any]) -> tuple[list[Any], VolumeScore]:
    """Level-wise rows and the shared-scorer match of one level-suite case (its annotated discs as truth)."""
    if suite == "sudirman":
        from .anatomy._validate_sudirman import LevelTruth as SudTruth
        from .anatomy._validate_sudirman import score_levels, truth_discs

        levels: list[Any] = [
            SudTruth(
                **{
                    **d,
                    "center": np.asarray(d["center"], dtype=np.float64),
                    "normal": np.asarray(d["normal"], dtype=np.float64),
                }
            )
            for d in rec["levels"]
        ]
        return score_levels(anchors, levels), score_anchors(anchors, truth_discs(levels), None)
    from .anatomy._validate_qx5rt import LevelTruth as QxTruth
    from .anatomy._validate_qx5rt import score_levels as qx_score_levels
    from .anatomy._validate_qx5rt import truth_discs as qx_truth_discs

    qx_levels = [
        QxTruth(level=d["level"], center=np.asarray(d["center"], dtype=np.float64), voxels=int(d["voxels"]))
        for d in rec["levels"]
    ]
    return qx_score_levels(anchors, qx_levels), score_anchors(anchors, qx_truth_discs(qx_levels), None)


def _score_record(suite: str, rec: dict[str, Any]) -> _Scored:
    anchors = Anchors.from_dict(rec["anchors"])
    rows: list[Any] | None = None
    if suite in LEVEL_SUITES:
        rows, vs = _level_scores(suite, anchors, rec)
    else:
        discs, mx = _truth_from_json(rec["truth"])
        vs = score_anchors(anchors, discs, mx)
    conf = float(anchors.numbering.get("confidence") or 0.0)
    status = str(anchors.numbering.get("status") or "unconfirmed")
    if status not in STATUSES:
        status = "unconfirmed"
    out = _Scored(
        meta=dict(rec.get("meta") or {}),
        vs=vs,
        conf=conf,
        status=status,
        correct=vs.matched > 0 and vs.label_correct == vs.matched,
        elapsed=rec.get("elapsed_s"),
        rows=rows,
    )
    if suite in PHANTOM_SUITES:
        from .testing.robustness import classify

        out.outcome = classify(aggregate([vs]), conf)
    return out


def _pool(items: list[_Scored], level: bool) -> dict[str, Any]:
    """Pooled accuracy and localisation metrics over scored cases (shared scorer for volumes, level-wise otherwise)."""
    if not items:
        return {}
    if not level:
        vs = [i.vs for i in items]
        m = aggregate(vs)
        ext = aggregate_extended(vs)
        mids = [s.midline_err_mm for s in vs if s.midline_err_mm is not None]
        return {
            "recall": m["recall"],
            "precision": m["precision"],
            "label_accuracy": m["label_accuracy"],
            "label_accuracy_all_truth": m["label_accuracy_all_truth"],
            "fully_correct": ext["volumes_all_truth_labelled"],
            "centre_err_median_mm": m["centre_err_median_mm"],
            "centre_err_p90_mm": m["centre_err_p90_mm"],
            "centre_within_3mm": m["centre_within_3mm"],
            "normal_err_median_deg": m["normal_err_median_deg"],
            "midline_err_median_mm": float(np.median(mids)) if mids else None,
            "false_positives_per_volume": ext["false_positives_per_volume"],
            "off_by_one_of_truth": ext["off_by_one_of_truth"],
        }
    rows = [r for i in items for r in (i.rows or [])]
    matched = [r for r in rows if r.matched]
    dists = [float(r.dist_mm) for r in matched if r.dist_mm is not None]
    per_case_full = [bool(i.rows) and all(r.matched and r.name_ok for r in i.rows or []) for i in items]
    return {
        "recall": len(matched) / len(rows) if rows else None,
        "precision": None,  # only the annotated discs are known: precision is not defined here
        "label_accuracy": sum(r.name_ok for r in matched) / len(matched) if matched else None,
        "label_accuracy_all_truth": sum(r.name_ok for r in rows) / len(rows) if rows else None,
        "fully_correct": sum(per_case_full) / len(items),
        "centre_err_median_mm": float(np.median(dists)) if dists else None,
        "centre_err_p90_mm": float(np.percentile(dists, 90)) if dists else None,
        "centre_within_3mm": float(np.mean(np.asarray(dists) <= 3.0)) if dists else None,
        "normal_err_median_deg": None,
        "midline_err_median_mm": None,
        "false_positives_per_volume": None,
        "off_by_one_of_truth": None,
    }


def _status_block(items: list[_Scored]) -> dict[str, Any]:
    """Per numbering status: studies, coverage, precision (share with the right numbering) and wrong counts."""
    n = len(items)
    out: dict[str, Any] = {}

    def block(sel: list[_Scored]) -> dict[str, Any]:
        k = len(sel)
        ok = sum(1 for i in sel if i.correct)
        return {
            "studies": k,
            "coverage": k / n if n else None,
            "precision": ok / k if k else None,
            "wrong": k - ok,
        }

    for st in STATUSES:
        out[st] = block([i for i in items if i.status == st])
    out["announced"] = block([i for i in items if i.status in ANNOUNCED])
    return out


def _numbering_reliability(items: list[_Scored]) -> dict[str, Any]:
    """Calibration of the numbering confidence against a fully correct numbering, one pseudo-match per study."""
    scores = [
        VolumeScore(
            n_truth=1,
            n_pred=1,
            matched=1,
            label_correct=int(i.correct),
            matches=[
                DiscMatch(
                    pred="numbering",
                    pred_confidence=i.conf,
                    truth="numbering",
                    centre_err_mm=None,
                    normal_err_deg=None,
                    label_correct=i.correct,
                    level_offset=None,
                )
            ],
        )
        for i in items
    ]
    rel = reliability(scores)
    return {"numbering_ece": rel["ece"], "numbering_auroc": rel["auroc"]}


def _status_metrics(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "confirmed_precision": status["confirmed"]["precision"],
        "probable_precision": status["probable"]["precision"],
        "unconfirmed_precision": status["unconfirmed"]["precision"],
        "announced_coverage": status["announced"]["coverage"],
        "announced_precision": status["announced"]["precision"],
        "wrong_announced": status["announced"]["wrong"],
        "wrong_total": sum(status[s]["wrong"] for s in STATUSES),
    }


def _pct_of(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def _groups(items: list[_Scored], level: bool) -> dict[str, Any]:
    keys = sorted({k for i in items for k in i.meta})
    out: dict[str, Any] = {}
    for key in keys:
        values = sorted({str(i.meta.get(key, "")) for i in items})
        out[key] = {}
        for v in values:
            sub = [i for i in items if str(i.meta.get(key, "")) == v]
            st = _status_block(sub)
            pooled = _pool(sub, level)
            out[key][v] = {
                "cases": len(sub),
                "recall": pooled.get("recall"),
                "label_accuracy_all_truth": pooled.get("label_accuracy_all_truth"),
                "fully_correct": pooled.get("fully_correct"),
                "announced_precision": st["announced"]["precision"],
                "wrong_announced": st["announced"]["wrong"],
            }
    return out


def _clean(x: Any) -> Any:
    """JSON-safe numbers: NaN and infinities become None, numpy scalars become Python numbers, floats are rounded."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else None
    return x


def _section(suite: str, recs: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any]:
    scored: list[_Scored] = []
    errors = 0
    for rec in recs:
        if rec.get("error") or not rec.get("anchors"):
            errors += 1
            continue
        scored.append(_score_record(suite, rec))
    level = suite in LEVEL_SUITES
    metrics: dict[str, Any] = {"errors": errors}
    metrics.update(_pool(scored, level))
    status = _status_block(scored)
    metrics.update(_status_metrics(status))
    if scored:
        metrics.update(_numbering_reliability(scored))
    if not level and scored:
        rel = reliability([i.vs for i in scored])
        metrics["disc_ece"] = rel["ece"]
        metrics["disc_auroc"] = rel["auroc"]
    lat = [i.elapsed for i in scored if i.elapsed is not None]
    metrics["latency_s_median"] = _pct_of(lat, 50)
    metrics["latency_s_p90"] = _pct_of(lat, 90)
    section: dict[str, Any] = {
        "title": TITLES[suite],
        "backend": SUITE_BACKEND[suite],
        "cases": len(recs),
        "sample": {**sample, "from_cache": sum(1 for r in recs if r.get("cached"))},
        "status": status,
    }
    if suite in PHANTOM_SUITES:
        names = [r["id"] for r in recs if not (r.get("error") or not r.get("anchors"))]
        section["scenarios"] = {n: i.outcome for n, i in zip(names, scored, strict=True)}
        metrics["silent_error"] = sum(1 for i in scored if i.outcome == "silent_error")
        metrics["scenarios_passed"] = sum(1 for i in scored if i.outcome == "pass")
    section["metrics"] = _clean(metrics)
    groups = _groups(scored, level)
    if groups:
        section["groups"] = _clean(groups)
    return section


def run_benchmark(opts: Options) -> dict[str, Any]:
    """Label, score and summarise every selected suite. Returns the JSON-able result (no files are written)."""
    from .anatomy.ml.labeller import THREADS_ENV as LABELLER_THREADS_ENV

    if opts.threads is not None:
        # Inherited by the pool workers: one knob for the whole run. Unset means one thread per worker.
        os.environ[LABELLER_THREADS_ENV] = str(opts.threads)
    fp = labeller_fingerprint()
    suites = expand_suites(opts.suites)
    t_start = time.perf_counter()
    sections: dict[str, Any] = {}
    for suite in suites:
        try:
            jobs, sample = _plan(suite, opts)
        except Unavailable as exc:
            log.info("%s: skipped (%s)", suite, exc)
            sections[suite] = {"skipped": str(exc)}
            continue
        for j in jobs:
            j["cache_file"] = str(_cache_file(j, opts.cache, fp))
            j["refresh"] = opts.refresh
        log.info("%s: %d cases on %d worker(s)", suite, len(jobs), opts.jobs)
        t0 = time.perf_counter()
        try:
            recs = _map(_case_job, jobs, opts.jobs)
            failed = [r for r in recs if r.get("error")]
            for r in failed[:3]:
                log.warning("%s: case failed (%s)", suite, r["error"])
            sections[suite] = _section(suite, recs, sample)
        except Exception as exc:  # a broken suite is reported; the other suites still run
            log.error("%s: suite failed (%s)", suite, type(exc).__name__)
            sections[suite] = {"error": type(exc).__name__}
            continue
        log.info("%s: done in %.0f s", suite, time.perf_counter() - t0)
    return {
        "schema": SCHEMA,
        "spinescope": __version__,
        "labeller": {
            "fingerprint": fp,
            "threads": "default" if opts.threads is None else opts.threads,
        },
        "run": {"jobs": opts.jobs, "wall_s": round(time.perf_counter() - t_start, 1)},
        "sections": sections,
    }


# ----------------------------------------------------------------------------- regression check
@dataclass(frozen=True)
class Finding:
    """One comparison of a gated metric (or scenario outcome, or a section) against the golden value."""

    section: str
    metric: str
    verdict: str  # ok | regression | improved | skipped | missing | new | sample_changed | error
    golden: Any = None
    current: Any = None
    tol: float = 0.0
    direction: str = ""
    unit: str = ""
    note: str = ""


def _judge(direction: str, golden: Any, current: Any, tol: float) -> str:
    if direction == "outcome":
        if OUTCOME_RANK.get(str(current), -1) < OUTCOME_RANK.get(str(golden), -1):
            return "regression"
        return "improved" if OUTCOME_RANK.get(str(current), -1) > OUTCOME_RANK.get(str(golden), -1) else "ok"
    d = float(current) - float(golden)
    if direction == "count":
        return "regression" if d > EPS else ("improved" if d < -EPS else "ok")
    if direction == "higher":
        return "regression" if d < -tol - EPS else ("improved" if d > tol + EPS else "ok")
    if direction == "lower":
        return "regression" if d > tol + EPS else ("improved" if d < -tol - EPS else "ok")
    raise ValueError(f"unknown direction {direction!r}")


def _fmt_value(unit: str, v: Any) -> str:
    if v is None:
        return "n/a"
    if unit == "rate":
        return f"{100 * float(v):.2f}%"
    if unit == "count":
        return str(int(v))
    if unit == "outcome":
        return str(v)
    if unit == "decimal":
        return f"{float(v):.3f}"
    return f"{float(v):.2f} {unit}"


def _fmt_change(unit: str, golden: Any, current: Any) -> str:
    if golden is None or current is None or unit == "outcome":
        return ""
    d = float(current) - float(golden)
    if unit == "rate":
        return f"{100 * d:+.2f} pp"
    if unit == "count":
        return f"{round(d):+d}"
    if unit == "decimal":
        return f"{d:+.3f}"
    return f"{d:+.2f} {unit}"


def compare(current: dict[str, Any], golden: dict[str, Any]) -> list[Finding]:
    """Compare a result with the golden file. Only gated metrics in the golden file are judged.

    A section that was skipped (data not installed) or not run is reported, not failed. A section whose sample size
    changed is a failure: its numbers are not comparable, so the golden file must be regenerated after review.
    """
    findings: list[Finding] = []
    cur_secs: dict[str, Any] = current.get("sections", {})
    for name, gold in golden.get("sections", {}).items():
        cur = cur_secs.get(name)
        if cur is None:
            findings.append(Finding(name, "-", "skipped", note="not run"))
            continue
        if "skipped" in cur:
            findings.append(Finding(name, "-", "skipped", note=str(cur["skipped"])))
            continue
        if "error" in cur:
            findings.append(Finding(name, "-", "error", note=f"suite failed: {cur['error']}"))
            continue
        if cur.get("cases") != gold.get("cases"):
            findings.append(
                Finding(
                    name,
                    "cases",
                    "sample_changed",
                    golden=gold.get("cases"),
                    current=cur.get("cases"),
                    note="sample size differs, numbers are not comparable; review, then --update-golden",
                )
            )
            continue
        cur_m: dict[str, Any] = cur.get("metrics", {})
        for metric, g in gold.get("metrics", {}).items():
            direction, tol, unit = g["direction"], float(g.get("tol", 0.0)), g.get("unit", "")
            gv = g.get("value")
            if gv is None:
                continue
            cv = cur_m.get(metric)
            if cv is None:
                findings.append(Finding(name, metric, "missing", gv, None, tol, direction, unit))
                continue
            verdict = _judge(direction, gv, cv, tol)
            findings.append(Finding(name, metric, verdict, gv, cv, tol, direction, unit))
        cur_sc: dict[str, Any] = cur.get("scenarios", {})
        for scen, g in gold.get("scenarios", {}).items():
            gv = g["outcome"]
            cv = cur_sc.get(scen)
            if cv is None:
                findings.append(
                    Finding(name, f"scenario {scen}", "missing", gv, None, 0.0, "outcome", "outcome")
                )
                continue
            verdict = _judge("outcome", gv, cv, 0.0)
            findings.append(Finding(name, f"scenario {scen}", verdict, gv, cv, 0.0, "outcome", "outcome"))
        for metric in sorted(set(cur.get("metrics", {})) - set(gold.get("metrics", {}))):
            if metric in RULES and cur_m.get(metric) is not None:
                findings.append(Finding(name, metric, "new", current=cur_m[metric], note="not in golden"))
        for scen in sorted(set(cur_sc) - set(gold.get("scenarios", {}))):
            findings.append(
                Finding(name, f"scenario {scen}", "new", current=cur_sc[scen], note="not in golden")
            )
    for name in sorted(set(cur_secs) - set(golden.get("sections", {}))):
        if "skipped" not in cur_secs[name]:
            findings.append(Finding(name, "-", "new", note="section not in golden; --update-golden adds it"))
    return findings


def regressions(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.verdict in {"regression", "missing", "sample_changed", "error"}]


def _fmt_tol(unit: str, tol: float) -> str:
    if unit == "rate":
        return f"{100 * tol:.2f} pp"
    if unit == "count":
        return "none"
    return _fmt_value(unit, tol) if unit == "decimal" else f"{tol:.2f} {unit}"


def _describe(f: Finding) -> str:
    """One sentence per regression: what moved, from what to what, and the tolerance it broke."""
    if f.verdict == "sample_changed":
        return f"{f.section}: sample size {f.golden} -> {f.current}; {f.note}"
    if f.verdict == "error":
        return f"{f.section}: {f.note}"
    if f.verdict == "missing":
        return f"{f.section}.{f.metric}: no longer reported (golden {_fmt_value(f.unit, f.golden)})"
    if f.direction == "outcome":
        return f"{f.section}.{f.metric}: {f.golden} -> {f.current}"
    delta = abs(float(f.current) - float(f.golden))
    word = {"higher": "dropped", "lower": "rose", "count": "increased"}[f.direction]
    amount = {"rate": f"{100 * delta:.2f} pp", "count": f"{round(delta)}"}.get(
        f.unit, f"{delta:.3f}" if f.unit == "decimal" else f"{delta:.2f} {f.unit}"
    )
    return (
        f"{f.section}.{f.metric} {word} by {amount} ({_fmt_value(f.unit, f.golden)} -> "
        f"{_fmt_value(f.unit, f.current)}), tolerance {_fmt_tol(f.unit, f.tol)}"
    )


def format_report(findings: list[Finding], golden: dict[str, Any], current: dict[str, Any]) -> str:
    """Readable check output: every regression with its numbers and tolerance, then the changes and the notes."""
    lines = ["Labelling benchmark check against the golden file", ""]
    g_fp = golden.get("labeller_fingerprint")
    c_fp = current.get("labeller", {}).get("fingerprint")
    if g_fp and c_fp and g_fp != c_fp:
        lines.append(
            f"note: labeller source differs from the golden run ({g_fp} -> {c_fp}); numbers may move"
        )
    ok = sum(1 for f in findings if f.verdict == "ok")
    bad = regressions(findings)
    shown = [f for f in findings if f.verdict not in {"ok"}]
    if shown:
        lines.append(
            f"{'section':<18}{'metric':<30}{'golden':>12}{'current':>12}{'change':>14}{'tol':>12}  verdict"
        )
        for f in shown:
            gv = _fmt_value(f.unit, f.golden) if f.verdict != "skipped" else ""
            cv = _fmt_value(f.unit, f.current) if f.verdict != "skipped" else ""
            tol = "" if f.direction in {"", "outcome"} else _fmt_tol(f.unit, f.tol)
            chg = _fmt_change(f.unit, f.golden, f.current)
            verdict = (
                f.verdict.upper()
                if f.verdict in {"regression", "missing", "sample_changed", "error"}
                else f.verdict
            )
            detail = f"  ({f.note})" if f.note and f.verdict in {"skipped", "new"} else ""
            lines.append(
                f"{f.section:<18}{f.metric:<30}{gv:>12}{cv:>12}{chg:>14}{tol:>12}  {verdict}{detail}"
            )
        lines.append("")
    lines.append(f"{ok} metric(s) within tolerance.")
    if bad:
        lines.append(f"RESULT: FAIL, {len(bad)} regression(s):")
        lines += [f"  - {_describe(f)}" for f in bad]
    else:
        lines.append("RESULT: PASS, no regressions.")
    return "\n".join(lines) + "\n"


def golden_from(current: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """The golden file for ``current``. Sections not run stay as they were; tolerances already in ``previous`` are kept."""
    sections: dict[str, Any] = dict((previous or {}).get("sections", {}))
    for name, sec in current.get("sections", {}).items():
        if "skipped" in sec or "error" in sec:
            continue
        old = sections.get(name, {}).get("metrics", {})
        metrics: dict[str, Any] = {}
        for key, value in sec.get("metrics", {}).items():
            if key not in RULES or value is None:
                continue
            direction, tol, unit = RULES[key]
            prior = old.get(key, {})
            metrics[key] = {
                "value": value,
                "direction": direction,
                "tol": prior.get("tol", tol),
                "unit": unit,
            }
        entry: dict[str, Any] = {"cases": sec["cases"], "metrics": dict(sorted(metrics.items()))}
        if sec.get("scenarios"):
            entry["scenarios"] = {k: {"outcome": v} for k, v in sorted(sec["scenarios"].items())}
        sections[name] = entry
    return {
        "schema": SCHEMA,
        "description": (
            "Golden numbers of the labelling benchmark (docs/benchmarks.md). Regenerate with "
            "tools/benchmark.py --update-golden benchmarks/golden.json only after reviewing the change."
        ),
        "spinescope": current.get("spinescope"),
        "labeller_fingerprint": current.get("labeller", {}).get("fingerprint"),
        "sections": dict(sorted(sections.items())),
    }


# ----------------------------------------------------------------------------- markdown summary
#: Counts, printed as integers.
COUNT_KEYS = frozenset({"errors", "wrong_announced", "wrong_total", "silent_error", "scenarios_passed"})


def _md_pct(k: str, v: Any) -> str:
    if v is None:
        return "n/a"
    if k in COUNT_KEYS:
        return str(int(v))
    if k in RATE_KEYS:
        return f"{100 * float(v):.1f}%"
    return f"{float(v):.2f}"


def to_markdown(result: dict[str, Any]) -> str:
    """Markdown summary of a result: one table per section and the status and robustness detail. Numbers only."""
    lab = result["labeller"]
    lines = [
        "# Labelling benchmark",
        "",
        f"- spinescope {result['spinescope']}, labeller fingerprint `{lab['fingerprint']}`",
        f"- ONNX threads per process: {lab['threads']}; workers: {result['run']['jobs']}; "
        f"wall time {result['run']['wall_s'] / 60:.1f} min",
        "",
    ]
    skipped = [(n, s["skipped"]) for n, s in result["sections"].items() if "skipped" in s]
    failed = [(n, s["error"]) for n, s in result["sections"].items() if "error" in s]
    ran = {n: s for n, s in result["sections"].items() if "metrics" in s}
    if ran:
        lines += [
            "## Summary",
            "",
            "| section | cases | errors | recall | label acc (matched) | label acc (all truth) | fully correct "
            "| centre median mm | centre p90 mm | announced | announced precision | wrong announced |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for n, s in ran.items():
            m = s["metrics"]
            lines.append(
                f"| {n} | {s['cases']} | {m['errors']} | {_md_pct('recall', m['recall'])} | "
                f"{_md_pct('label_accuracy', m['label_accuracy'])} | "
                f"{_md_pct('label_accuracy_all_truth', m['label_accuracy_all_truth'])} | "
                f"{_md_pct('fully_correct', m['fully_correct'])} | {_md_pct('', m['centre_err_median_mm'])} | "
                f"{_md_pct('', m['centre_err_p90_mm'])} | {_md_pct('announced_coverage', m['announced_coverage'])} | "
                f"{_md_pct('announced_precision', m['announced_precision'])} | {m['wrong_announced']} |"
            )
        lines.append("")
    for s in ran.values():
        m = s["metrics"]
        lines += [
            f"## {s['title']}",
            "",
            f"Labeller backend `{s['backend']}`. Sample: {s['sample']['method']}; cases {s['cases']}; "
            f"errors {m['errors']}; {s['sample']['from_cache']} from cache.",
            "",
        ]
        lines += ["| metric | value |", "|---|---|"]
        for k, v in m.items():
            if k in {"errors", "latency_s_median", "latency_s_p90"} or k.startswith("status"):
                continue
            lines.append(f"| {k} | {_md_pct(k, v)} |")
        lines += ["", "| status | studies | coverage | precision | wrong |", "|---|---|---|---|---|"]
        for st in (*STATUSES, "announced"):
            b = s["status"][st]
            lines.append(
                f"| {st} | {b['studies']} | {_md_pct('', b['coverage'])} | {_md_pct('', b['precision'])} | {b['wrong']} |"
            )
        lines += [
            "",
            f"Labelling time per study: median {m['latency_s_median']} s, 90th percentile "
            f"{m['latency_s_p90']} s (as recorded when the case was labelled).",
            "",
        ]
        if s.get("scenarios"):
            lines += ["| scenario | outcome |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in s["scenarios"].items()]
            lines.append("")
        for key, groups in s.get("groups", {}).items():
            lines += [
                f"By {key}:",
                "",
                "| group | cases | recall | label acc (all truth) | fully correct | "
                "announced precision | wrong announced |",
                "|---|---|---|---|---|---|---|",
            ]
            for g, b in groups.items():
                lines.append(
                    f"| {g} | {b['cases']} | {_md_pct('recall', b['recall'])} | "
                    f"{_md_pct('label_accuracy_all_truth', b['label_accuracy_all_truth'])} | "
                    f"{_md_pct('fully_correct', b['fully_correct'])} | "
                    f"{_md_pct('announced_precision', b['announced_precision'])} | {b['wrong_announced']} |"
                )
            lines.append("")
    if skipped:
        lines += ["## Skipped", ""] + [f"- {n}: {why}" for n, why in skipped] + [""]
    if failed:
        lines += ["## Failed", ""] + [f"- {n}: {why}" for n, why in failed] + [""]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- command line
def _inside_repo(path: Path) -> bool:
    repo = Path(__file__).resolve().parents[2]
    return (repo / "pyproject.toml").is_file() and path.resolve().is_relative_to(repo)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tools/benchmark.py",
        description="Run every labelling benchmark (or the ones whose data is installed) and summarise them. "
        "--check compares the numbers with the golden file and exits non-zero on a regression.",
    )
    p.add_argument(
        "--suites",
        nargs="+",
        default=list(SUITES),
        metavar="SUITE",
        help=f"{', '.join(SUITES)}; 'phantom' = both phantom suites (default: all)",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output folder, outside the repository")
    p.add_argument(
        "--cache", default=str(DEFAULT_CACHE), help="labeller output cache, outside the repository"
    )
    p.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)), help="parallel cases")
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="ONNX Runtime threads per labelling process (default: one per worker when --jobs > 1)",
    )
    p.add_argument("--refresh", action="store_true", help="ignore the cache and label every case again")
    p.add_argument("--spider-split", default="validation", help="SPIDER subset from overview.csv")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evenly spaced subsample of N cases in SPIDER, Sudirman and qx5rt, for development runs "
        "(golden numbers always use the full sets)",
    )
    p.add_argument("--check", metavar="GOLDEN", help="compare with this golden file; exit 1 on regression")
    p.add_argument(
        "--update-golden", metavar="GOLDEN", help="write this golden file from the run (review it!)"
    )
    p.add_argument("--from-json", metavar="RESULT", help="use an earlier benchmark JSON instead of running")
    p.add_argument("--quiet", action="store_true", help="log warnings only")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``tools/benchmark.py`` and ``spinescope benchmark``. Returns the process exit code."""
    ap = build_parser()
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    if args.check and args.update_golden:
        ap.error("--check and --update-golden are different steps; run them separately")
    try:
        suites = expand_suites(args.suites)
    except ValueError as exc:
        ap.error(str(exc))
    out, cache = Path(args.out).expanduser(), Path(args.cache).expanduser()
    if _inside_repo(out) or _inside_repo(cache):
        ap.error("refusing to write outputs or cache inside the repository; use a folder outside it")

    if args.from_json:
        result = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    else:
        opts = Options(
            suites=suites,
            cache=cache,
            jobs=max(1, args.jobs),
            threads=args.threads,
            refresh=args.refresh,
            spider_split=args.spider_split,
            limit=args.limit,
        )
        result = run_benchmark(opts)
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        (out / "benchmark.md").write_text(to_markdown(result), encoding="utf-8")
        log.info("wrote %s and %s", out / "benchmark.json", out / "benchmark.md")

    if args.update_golden:
        path = Path(args.update_golden)
        prev = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        golden = golden_from(result, prev)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(golden['sections'])} section(s)); review the diff before committing")
        return 0
    if args.check:
        golden = json.loads(Path(args.check).read_text(encoding="utf-8"))
        findings = compare(result, golden)
        text = format_report(findings, golden, result)
        print(text, end="")
        if not args.from_json:
            (out / "check.txt").write_text(text, encoding="utf-8")
        return 1 if regressions(findings) else 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
