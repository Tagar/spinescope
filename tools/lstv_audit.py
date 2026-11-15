"""Run the transitional-vertebra check (lab28) over a dataset and write one JSON record per case, plus PNG renders.

Usage (scratch outputs go outside the repository; nothing here reads data other than the public sets):

    python tools/lstv_audit.py run --dataset spider --out /tmp/spinescope-lab28_lstv/spider.json [--limit N]
    python tools/lstv_audit.py run --dataset phantom --out /tmp/spinescope-lab28_lstv/phantom.json
    python tools/lstv_audit.py render --records /tmp/.../spider.json --ids 1 2 3 --out /tmp/spinescope-lab28_lstv/png

Datasets: ``spider`` (T2 sagittal per case, subsets from overview.csv), ``sudirman`` (evenly spaced labelled patients,
axial plus sagittal), ``qx5rt`` (one session per scanner vendor sample), ``phantom`` (cartoon and realistic variants).
Labels: none of these datasets carries an LSTV label; the records are the check's output, not a score.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("lstv_audit")


def _record(case_id: str, anchors: Any, extra: dict[str, Any]) -> dict[str, Any]:
    rec = dict(anchors.lstv or {})
    labels = anchors.disc_labels()
    heights = {lab: anchors.discs[lab].height_mm for lab in labels}
    rec.update(
        {
            "case": case_id,
            "n_discs": len(labels),
            "count_names": labels,
            "disc_heights_mm": {k: (None if v is None else round(float(v), 1)) for k, v in heights.items()},
            "numbering_status": anchors.numbering.get("status"),
            "lumbosacral": (anchors.numbering.get("lumbosacral") or {}).get("status"),
            "applied_offset": anchors.numbering.get("applied_offset", 0),
            **extra,
        }
    )
    return rec


def _spider(limit: int | None) -> list[dict[str, Any]]:
    """Every SPIDER case with a T2 image and mask: the same study build as tests/_spider.load_t2_t1."""
    import csv

    from spinescope.anatomy.pipeline import label_spine
    from spinescope.datasets.spider import SpiderStudy, spider_root

    root = spider_root()
    if root is None:
        raise SystemExit("SPIDER not found (set SPINESCOPE_DATA)")
    subsets: dict[str, str] = {}
    with open(root / "overview.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["new_file_name"]  # e.g. "12_t2"
            case, _, seq = name.rpartition("_")
            if seq == "t2":
                subsets[case] = row.get("subset", "")
    out: list[dict[str, Any]] = []
    for n, case_id in enumerate(sorted(subsets, key=lambda c: int(c) if c.isdigit() else 0)):
        if limit is not None and n >= limit:
            break
        t2p = root / "images" / f"{case_id}_t2.mha"
        t2m = root / "masks" / f"{case_id}_t2.mha"
        if not (t2p.exists() and t2m.exists()):
            out.append({"case": case_id, "subset": subsets[case_id], "error": "T2 image or mask missing"})
            continue
        images = {"t2": t2p}
        t1p = root / "images" / f"{case_id}_t1.mha"
        if t1p.exists():
            images["t1"] = t1p
        study = SpiderStudy(case_id=case_id, image_paths=images, mask_paths={"t2": t2m}).to_study()
        study.info.update({"Modality": "MR", "source": "SPIDER"})
        anchors = label_spine(study)
        out.append(_record(case_id, anchors, {"subset": subsets[case_id]}))
        log.info("%s %s", case_id, (anchors.lstv or {}).get("status"))
    return out


def _sudirman(limit: int | None, sample: int) -> list[dict[str, Any]]:
    """Evenly spaced labelled Sudirman patients (as lab02), the session with the labels, labelled as usual."""
    from spinescope.anatomy.pipeline import label_spine
    from spinescope.datasets.sudirman import (
        labelled_patients,
        patient_sessions,
        select_session,
        sudirman_labels_root,
        sudirman_root,
    )

    labels = sudirman_labels_root()
    ids = labelled_patients(labels)
    step = max(1, len(ids) // sample)
    ids = ids[::step][:sample]
    sessions = patient_sessions(sudirman_root())
    out: list[dict[str, Any]] = []
    for n, pid in enumerate(ids):
        if limit is not None and n >= limit:
            break
        sel = select_session(pid, sessions.get(pid, []), labels)
        if sel is None:
            out.append({"case": pid, "error": "no session reproduces a label slice"})
            continue
        _session, study, _truths = sel
        anchors = label_spine(study)
        out.append(_record(pid, anchors, {"sampled_every": step}))
        log.info("%s %s", pid, (anchors.lstv or {}).get("status"))
    return out


def _qx5rt() -> list[dict[str, Any]]:
    """Every qx5rt session that the scorer can read (T1 plus disc masks), labelled as usual."""
    from spinescope.anatomy import _validate_qx5rt as q

    out: list[dict[str, Any]] = []
    for sess in q.discover(None):
        _res, anchors, _stack = q.evaluate_session(sess)
        if anchors is None:
            out.append({"case": sess.key, "error": "not labelled"})
            continue
        out.append(_record(sess.key, anchors, {"vendor": sess.vendor, "model": sess.model}))
        log.info("%s %s", sess.key, (anchors.lstv or {}).get("status"))
    return out


def _phantom() -> list[dict[str, Any]]:
    from spinescope.anatomy.pipeline import label_spine
    from spinescope.testing.phantom import PhantomSpec, SpinePhantom

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from _studies import phantom_study  # type: ignore[import-not-found]

    out: list[dict[str, Any]] = []
    for realism in ("cartoon", "realistic"):
        for tr in (None, "lumbarised_s1", "sacralised_l5"):
            for seed in (0, 1, 2):
                ph = SpinePhantom(PhantomSpec(transitional=tr, realism=realism, seed=seed))
                anchors = label_spine(phantom_study(ph))
                cid = f"{realism}:{tr or 'none'}:{seed}"
                out.append(_record(cid, anchors, {"truth_transitional": tr, "realism": realism}))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=["spider", "sudirman", "qx5rt", "phantom"])
    r.add_argument("--sample", type=int, default=120, help="sudirman: evenly spaced labelled patients")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.cmd == "run":
        if args.dataset == "phantom":
            rows = _phantom()
        elif args.dataset == "qx5rt":
            rows = _qx5rt()
        elif args.dataset == "sudirman":
            rows = _sudirman(args.limit, args.sample)
        else:
            rows = _spider(args.limit)
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        log.info("wrote %d records to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    raise SystemExit(main())
