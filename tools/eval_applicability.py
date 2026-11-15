#!/usr/bin/env python3
"""Applicability (when not to label) on SPIDER: verdicts, cues and the false-rejection rate per split.

Runs the shipped labeller (``label_spine_details``, gated by :mod:`spinescope.anatomy.ood`) on every paired SPIDER
study (sagittal T2 with T1 when present), with the technical DICOM attributes of the public overview injected as
the metadata a DICOM reader would supply (SeriesDescription, BodyPartExamined, MagneticFieldStrength, ...).

Example::

    python tools/eval_applicability.py --split validation --out /tmp/spinescope-lab12_ood/val.json --jobs 1

Output is JSON: one record per case (case id, subset, verdict, labelled, region, factor, cues, reasons, the
applicability warnings) and a summary per split. Case ids are dataset ids; no patient field, file path or free
text from the series is written. Nothing derived from SPIDER is committed to the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))  # the SPIDER helpers used by the validation tests

from _spider import load_t2_t1, paired_study_ids, spider_root  # noqa: E402
from spinescope.anatomy.ood import LABELLED_VERDICTS  # noqa: E402
from spinescope.anatomy.pipeline import label_spine_details  # noqa: E402
from spinescope.io.dicom import KEEP_TAGS  # noqa: E402

log = logging.getLogger("eval_applicability")

#: Overview columns that are technical and in the DICOM allowlist (never sex, birth date or device serial).
_TECH = tuple(k for k in KEEP_TAGS if k not in ("PatientPosition",))


def overview_metadata(root: Path) -> dict[str, dict[str, str]]:
    """Technical attributes per SPIDER file name (``<id>_<seq>``), from overview.csv."""
    out: dict[str, dict[str, str]] = {}
    with open(root / "overview.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("new_file_name") or "").strip()
            meta = {k: row[k] for k in _TECH if row.get(k) not in (None, "")}
            out[name] = meta
    return out


def _inject(study: Any, case: str, overview: dict[str, dict[str, str]]) -> None:
    for seq, key in (("t2", "sag_t2"), ("t1", "sag_t1")):
        if key in study.stacks:
            meta = overview.get(f"{case}_{seq}", {})
            study.stacks[key].meta.setdefault("dicom", {}).update(meta)


def t2_study_ids(root: Path) -> list[str]:
    """Every case with a sagittal T2 volume and its mask (T1 optional), numerically sorted."""
    ids = [p.name[: -len("_t2.mha")] for p in (root / "images").glob("*_t2.mha")]
    ids = [i for i in ids if (root / "masks" / f"{i}_t2.mha").exists()]
    return sorted(ids, key=lambda s: int(s) if s.isdigit() else 10**9)


def run(split: str, limit: int | None, root: Path | None, all_t2: bool = False) -> dict[str, Any]:
    base = root or spider_root()
    if base is None:
        raise SystemExit("SPIDER not found under $SPINESCOPE_DATA/spider")
    overview = overview_metadata(base)
    ids = paired_study_ids(base) if not all_t2 else t2_study_ids(base)
    subset = {}
    with open(base / "overview.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("new_file_name") or "").strip()
            if name.endswith("_t2"):
                subset[name[: -len("_t2")]] = row.get("subset", "")
    ids = [i for i in ids if subset.get(i, "") == split]
    if limit:
        ids = ids[:limit]
    records: list[dict[str, Any]] = []
    t0 = time.time()
    for n, case in enumerate(ids, 1):
        loaded = load_t2_t1(case, base)
        if loaded is None:
            log.warning("case %s: could not load", case)
            continue
        study, _t2, _mask = loaded
        _inject(study, case, overview)
        det = label_spine_details(study, backend="ml")
        a = det.anchors
        app = a.applicability
        rec = {
            "case": case,
            "subset": split,
            "verdict": app.get("verdict"),
            "labelled": bool(app.get("labelled")),
            "region": app.get("region"),
            "factor": app.get("factor"),
            "discs_after_gate": a.disc_labels(),
            "numbering_status": a.numbering.get("status"),
            "numbering_confidence": a.numbering.get("confidence"),
            "sacrum_covered": app.get("sacrum_covered"),
            "capped": app.get("capped", []),
            "unconfirmed_reasons": app.get("unconfirmed_reasons", []),
            "reasons": app.get("reasons", []),
            "cues": app.get("cues", {}),
            "coverage": {k: v for k, v in app.get("coverage", {}).items() if k != "levels_in_fov"},
            "applicability_warnings": [w for w in a.warnings if w.startswith("applicability")],
        }
        records.append(rec)
        if n % 10 == 0:
            log.info("%d/%d cases, %.0f s", n, len(ids), time.time() - t0)
    return {"split": split, "n": len(records), "seconds": round(time.time() - t0, 1), "records": records}


def summarise(result: dict[str, Any]) -> dict[str, Any]:
    recs = result["records"]
    verdicts = Counter(r["verdict"] for r in recs)
    rejected = [r["case"] for r in recs if not r["labelled"]]
    return {
        "split": result["split"],
        "n": len(recs),
        "verdicts": dict(verdicts),
        "false_rejection_rate": round(len(rejected) / len(recs), 4) if recs else None,
        "rejected_cases": rejected,
        "labelled_verdicts": list(LABELLED_VERDICTS),
        "numbering_unconfirmed": sum(1 for r in recs if r["numbering_status"] == "unconfirmed"),
        "sacrum_not_covered": sum(1 for r in recs if r["sacrum_covered"] is False),
        "cases_with_capped_thoracic": sum(1 for r in recs if r["capped"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["training", "validation"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N cases of the split (0 = all)")
    ap.add_argument("--root", type=Path, default=None, help="SPIDER folder (default $SPINESCOPE_DATA/spider)")
    ap.add_argument("--out", type=Path, required=True, help="JSON output path (outside the repository)")
    ap.add_argument(
        "--all-t2", action="store_true", help="every case with a sagittal T2 (not only paired T1+T2)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("spinescope").setLevel(logging.ERROR)
    result = run(args.split, args.limit or None, args.root, all_t2=args.all_t2)
    result["summary"] = summarise(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
