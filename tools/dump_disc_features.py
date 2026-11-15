"""Dump per-slice and per-disc discs-analysis features with SPIDER gradings, for calibration.

Runs the same pipeline as the validation harness (labeller, ``discs`` analysis, level matching) and writes one CSV
row per mid-portion slice (``kind=slice``) and per disc (``kind=disc``) of every scored disc. Derived from SPIDER: keep
the output outside the repository (e.g. /tmp/spinescope-<name>/).

    .venv/bin/python tools/dump_disc_features.py --split training --out /tmp/spinescope-<name>/train_disc_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from spinescope.analysis import evaluate as ev
from spinescope.analysis.base import run
from spinescope.anatomy.pipeline import label_spine
from spinescope.config import DEFAULT
from spinescope.datasets.gradings import spider_dir, spider_subset_cases

_CASE_KEYS = (
    "case",
    "level",
    "pfirrmann",
    "disc_narrowing",
    "vendor",
    "field_strength",
    "sequences",
    "contrast",
)


def _work(item: tuple[str, str]) -> list[dict[str, Any]]:
    case, root = item
    ev._ensure_builtin()
    spec = ev.CaseSpec(source="spider", case=case, root=root, analyses=("discs",))
    try:
        study, truth, groups, _ok = ev._LOADERS["spider"](spec)
        anchors = label_spine(study, DEFAULT, backend="auto")
        res = run("discs", study, anchors, DEFAULT)
    except Exception as exc:
        return [{"case": case, "kind": "error", "error": type(exc).__name__}]
    pred = {lab: anchors.discs[lab].center for lab in anchors.discs}
    matched = ev.match_discs(pred, truth)
    slices: dict[str, list[dict[str, Any]]] = {}
    for r in res.rows:
        if r.get("scope") == "slice":
            slices.setdefault(str(r["disc"]), []).append(r)
    summary = {str(r["disc"]): r for r in res.rows if r.get("scope") == "disc"}
    out: list[dict[str, Any]] = []
    for pname, j in matched["pairs"]:
        t = truth[j]
        if pname != t.name:
            continue  # label mismatch: not scored, as in the harness
        common = {
            "case": case,
            "level": pname,
            "pfirrmann": t.values.get("pfirrmann"),
            "disc_narrowing": t.values.get("disc_narrowing"),
            "vendor": groups.get("vendor"),
            "field_strength": groups.get("field_strength"),
            "sequences": groups.get("sequences"),
            "contrast": res.meta.get("contrast"),
        }
        for r in slices.get(pname, []):
            out.append({**r, **common, "kind": "slice"})
        if pname in summary:
            out.append({**summary[pname], **common, "kind": "disc"})
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="training", choices=("training", "validation"))
    ap.add_argument("--root", help="dataset folder (default: $SPINESCOPE_DATA/spider)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    root = args.root or str(spider_dir())
    cases = spider_subset_cases(args.split, root)
    if args.limit:
        cases = cases[: args.limit]
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=ev._worker_init, initargs=(2,)) as ex:
        for out in ex.map(_work, [(c, root) for c in cases], chunksize=1):
            rows.extend(out)
    keys: list[str] = [*_CASE_KEYS, "kind"]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows for {len(cases)} cases in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
