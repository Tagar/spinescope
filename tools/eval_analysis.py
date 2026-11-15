#!/usr/bin/env python3
"""Run registered analyses on a labelled dataset split and score them against the radiologist gradings.

Example::

    .venv/bin/python tools/eval_analysis.py --dataset spider --split validation \\
        --analyses discs,endplates --workers 4 --out /tmp/spinescope-dx01_harness/baseline

Writes ``metrics.json``, ``metrics.md`` and (with ``--pairs``) ``pairs.csv`` into ``--out``. Only case ids
(public SPIDER numbers) and level names appear in the outputs; the dataset path is never written. Keep ``--out``
outside the repository: outputs derived from SPIDER are not committed.

``--limit N`` takes the first N cases of the split in numeric order, and the report says so.
``--sequences`` picks the sequences each study is built from (default: every sequence of the case).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

from spinescope.analysis import evaluate as ev
from spinescope.analysis.base import available
from spinescope.datasets.gradings import spider_dir, spider_subset_cases


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="spider", choices=("spider",))
    p.add_argument("--split", default="validation", choices=("training", "validation"))
    p.add_argument("--root", help="dataset folder (default: $SPINESCOPE_DATA/spider)")
    p.add_argument("--analyses", default="all", help="comma-separated analysis names, or 'all'")
    p.add_argument("--sequences", help="comma-separated sequences, e.g. t1,t2 (default: all of each case)")
    p.add_argument("--limit", type=int, help="score only the first N cases in numeric order")
    p.add_argument("--workers", type=int, help="processes (default: half the cores)")
    p.add_argument("--boot", type=int, default=1000, help="bootstrap resamples over cases (0 disables CIs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="auto", choices=("auto", "ml", "classical"))
    p.add_argument("--out", required=True, help="output folder (keep it outside the repository)")
    p.add_argument("--pairs", action="store_true", help="also write pairs.csv (one row per disc and field)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _parse(argv)
    names = (
        available() if args.analyses == "all" else [a.strip() for a in args.analyses.split(",") if a.strip()]
    )
    unknown = sorted(set(names) - set(available()))
    if unknown:
        print(f"unknown analyses: {', '.join(unknown)}; available: {', '.join(available())}", file=sys.stderr)
        return 2
    root = args.root or str(spider_dir())
    cases = spider_subset_cases(args.split, root)
    total = len(cases)
    if args.limit:
        cases = cases[: args.limit]
    seqs = tuple(s.strip() for s in args.sequences.split(",")) if args.sequences else None
    specs = [
        ev.CaseSpec(
            source=args.dataset,
            case=c,
            root=root,
            sequences=seqs,
            analyses=tuple(names),
            backend=args.backend,
        )
        for c in cases
    ]
    report = ev.evaluate(
        specs,
        dataset=f"{args.dataset}/{args.split}",
        workers=args.workers,
        n_boot=args.boot,
        seed=args.seed,
    )
    report.settings["split_cases"] = total
    report.settings["cases_scored"] = len(cases)
    report.settings["subsampled"] = len(cases) < total
    report.settings["sequences"] = list(seqs) if seqs else "all of each case"
    report.settings["backend"] = args.backend
    out = Path(os.fspath(args.out))
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report.to_dict(), indent=1, allow_nan=False, default=str))
    (out / "metrics.md").write_text(ev.render_markdown(report))
    if args.pairs:
        with open(out / "pairs.csv", "w", newline="") as f:
            rows = ev.pairs_table(report.outcomes)
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["case"])
            w.writeheader()
            w.writerows(rows)
    (out / "warnings.json").write_text(json.dumps(ev.case_warnings(report.outcomes), indent=1))
    print(ev.render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
