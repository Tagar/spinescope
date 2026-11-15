"""Pass/fail table of the robustness scenarios for every labelling backend.

Renders each scenario in :data:`spinescope.testing.robustness.SCENARIOS` from the synthetic phantom, labels it
with every backend in :data:`spinescope.anatomy.pipeline.BACKENDS` (``classical`` today; a learned backend
appears here automatically once it is registered there) and prints a Markdown table. Nothing is read from
disk except the phantom itself, and the output carries no file paths.

Usage::

    .venv/bin/python tools/robustness_report.py            # Markdown table on stdout (cartoon phantom)
    .venv/bin/python tools/robustness_report.py --realism realistic --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from spinescope.anatomy.pipeline import BACKENDS
from spinescope.testing.phantom import REALISMS
from spinescope.testing.robustness import SCENARIOS, build_case, evaluate

log = logging.getLogger("robustness_report")

_LABEL = {"pass": "PASS", "unsure": "FAIL (unsure)", "silent_error": "FAIL (silent)"}


def run(
    names: list[str] | None = None, backends: tuple[str, ...] = BACKENDS, realism: str = "cartoon"
) -> list[dict[str, Any]]:
    """Evaluate the scenarios (optionally a subset) for each backend. One dict per scenario and backend."""
    rows: list[dict[str, Any]] = []
    for sc in SCENARIOS:
        if names and sc.name not in names:
            continue
        case = build_case(sc, realism)
        for backend in backends:
            try:
                r = evaluate(case, backend)
            except Exception as exc:  # a backend failing must not hide the other rows
                log.warning("%s on %s raised %s", sc.name, backend, exc)
                rows.append(
                    {
                        "scenario": sc.name,
                        "group": sc.group,
                        "backend": backend,
                        "realism": realism,
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "scenario": sc.name,
                    "group": sc.group,
                    "backend": backend,
                    "realism": realism,
                    "outcome": r.outcome,
                    "passed": r.passed,
                    "recall": r.metrics["recall"],
                    "precision": r.metrics["precision"],
                    "label_accuracy": r.metrics["label_accuracy_all_truth"],
                    "centre_p90_mm": r.metrics["centre_err_p90_mm"],
                    "numbering_confidence": r.numbering_confidence,
                    "truth": r.truth,
                    "predicted": r.predicted,
                }
            )
    return rows


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.2f}"  # NaN check without importing math


def markdown(rows: list[dict[str, Any]]) -> str:
    """Scenario x backend table: outcome, recall, label accuracy, p90 centre error and numbering confidence."""
    lines = [
        "| scenario | group | backend | status | recall | precision | label acc | centre p90 (mm) | conf |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['scenario']} | {r['group']} | {r['backend']} | ERROR | | | | | |")
            continue
        status = _LABEL.get(r["outcome"], r["outcome"])
        lines.append(
            f"| {r['scenario']} | {r['group']} | {r['backend']} | {status} | {_fmt(r['recall'])} | "
            f"{_fmt(r['precision'])} | {_fmt(r['label_accuracy'])} | {_fmt(r['centre_p90_mm'])} | "
            f"{_fmt(r['numbering_confidence'])} |"
        )
    passed = sum(1 for r in rows if r.get("passed"))
    lines.append("")
    lines.append(f"{passed} of {len(rows)} scenario-backend pairs pass.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="PATH", help="also write the rows as JSON")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="run only these scenarios")
    ap.add_argument("--realism", default="cartoon", choices=REALISMS, help="phantom look (default: cartoon)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    rows = run(args.only, realism=args.realism)
    print(markdown(rows))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
    return 0  # a report, not a gate: the test suite is the gate


if __name__ == "__main__":
    sys.exit(main())
