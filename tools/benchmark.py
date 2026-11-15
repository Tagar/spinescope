#!/usr/bin/env python3
"""Run every labelling benchmark, write one JSON and one markdown summary, and check them against the golden file.

Examples::

    # everything whose data is installed (SPIDER validation, Sudirman, qx5rt, both phantom suites)
    .venv/bin/python tools/benchmark.py --out /tmp/spinescope-benchmark

    # regression check against the committed golden numbers (exit 1 on a regression)
    .venv/bin/python tools/benchmark.py --check benchmarks/golden.json

    # phantom suites only (what CI runs; needs no data)
    .venv/bin/python tools/benchmark.py --suites phantom --check benchmarks/golden.json

    # after reviewing a deliberate change: rewrite the golden numbers (keeps sections that were not run)
    .venv/bin/python tools/benchmark.py --update-golden benchmarks/golden.json

The same commands are available as ``spinescope benchmark``. Full documentation: docs/benchmarks.md.
"""

from __future__ import annotations

import sys

from spinescope.benchmark import main

if __name__ == "__main__":
    sys.exit(main())
