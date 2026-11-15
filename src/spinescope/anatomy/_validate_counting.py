"""Measure the top-chain check (:mod:`spinescope.anatomy.counting`) on public data. Numbers only, no identifiers.

Two measurements, both with the production labeller path (``label_spine`` primary, :func:`count_evidence` for the
wider series):

* SPIDER validation volumes (ground truth from the masks). For each validation volume used as the primary, the other
  validation volumes of the same case are the candidates. The top-chain error of the primary is read from the mask:
  a disc of the truth, or a predicted disc, that is unmatched and lies above the fourth-lowest truth disc. The flag is
  the candidate-based ``top_disagreement``. Counts: flagged and wrong (catch), flagged and right (false alarm), wrong
  and not flagged (miss).
* Sudirman sessions (no disc truth; only consistency). Per session: whether a wider sagittal series exists, how far
  it reaches above the primary's top, the in-plane agreement of its discs with the primary's in the overlap, and
  whether the top of the chain disagrees.

Run: ``python -m spinescope.anatomy._validate_counting spider|sudirman [--max N]``. Outputs JSON on stdout.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets.spider import SpiderStudy, load_spider_studies, spider_root
from ._validate import score_anchors, truth_from_mask
from .anchors import Anchors
from .counting import DEFAULT_COUNTING, CountingResult, count_evidence
from .ml.labeller import label_spine_ml_segmented

log = logging.getLogger(__name__)

#: Index (from the bottom, 0 = lowest disc) at which the chain counts as "top": errors above the fourth disc from the
#: bottom are top-chain errors. Judgement: the lowest four levels (L5-S1 to L2-L3) are the ones the bottom cues see.
TOP_FROM_BOTTOM = 3


def _spider_validation_sequences(sp: SpiderStudy, overview: dict[str, dict[str, str]]) -> list[str]:
    return [q for q in sp.sequences if overview.get(f"{sp.case_id}_{q}", {}).get("subset") == "validation"]


def _overview(root: str | Path | None = None) -> dict[str, dict[str, str]]:
    with open(spider_root(root) / "overview.csv", newline="", encoding="utf-8") as fh:
        return {row["new_file_name"]: row for row in csv.DictReader(fh)}


def top_chain_error(
    anchors: Anchors, truth: list[Any], n_truth_above: int = TOP_FROM_BOTTOM
) -> tuple[bool, list[str]]:
    """True when the predicted chain differs from the mask truth above the ``n_truth_above``-th truth disc.

    ``truth`` is a list of ``TruthDisc`` (bottom-up). Matching uses :func:`score_anchors`. Returns the flag and the
    names of the unmatched items (for the report).
    """
    if len(truth) <= n_truth_above:
        return False, []
    z_cut = float(sorted(t.center[2] for t in truth)[n_truth_above])
    score = score_anchors(anchors, truth)
    bad: list[str] = []
    for m in score.matches:
        if m.truth is None and float(anchors.discs[m.pred].center[2]) > z_cut:
            bad.append(f"extra {m.pred}")
    for name in score.missed:
        t = next((t for t in truth if t.name == name), None)
        if t is not None and float(t.center[2]) > z_cut:
            bad.append(f"missed {name}")
    return bool(bad), bad


def spider_measure(max_volumes: int | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Catches, false alarms and misses of the top-chain flag on SPIDER validation volumes."""
    overview = _overview(root)
    studies = [s for s in load_spider_studies(root, load=True) if _spider_validation_sequences(s, overview)]
    rows: list[dict[str, Any]] = []
    for sp in studies:
        seqs = _spider_validation_sequences(sp, overview)
        if len(seqs) < 2:
            continue  # the check needs a second series of the same case
        study = sp.to_study(seqs)
        prim_stack = study.best("sag")
        if prim_stack is None:
            continue
        seq = next((q for q in seqs if np.array_equal(sp.volume(q).ipp, prim_stack.ipp)), None)
        if seq is None:
            continue
        prim_anchors, seg = label_spine_ml_segmented(study)
        truth, _ = truth_from_mask(str(sp.mask_paths[seq]))
        wrong, bad = top_chain_error(prim_anchors, truth)
        res: CountingResult = count_evidence(
            study, prim_anchors, primary_stack=prim_stack, segs={} if seg is None else {prim_stack.key: seg}
        )
        rows.append(
            {"wrong": wrong, "flag": res.top_disagreement, "candidates": len(res.candidates), "bad": bad}
        )
        if max_volumes is not None and len(rows) >= max_volumes:
            break
    cand_rows = [r for r in rows if r["candidates"]]
    catches = sum(1 for r in cand_rows if r["flag"] and r["wrong"])
    false_alarms = sum(1 for r in cand_rows if r["flag"] and not r["wrong"])
    misses = sum(1 for r in cand_rows if not r["flag"] and r["wrong"])
    return {
        "primary_volumes": len(rows),
        "with_wider_series": len(cand_rows),
        "primary_top_chain_errors_all": sum(1 for r in rows if r["wrong"]),
        "flagged": sum(1 for r in cand_rows if r["flag"]),
        "catches": catches,
        "false_alarms": false_alarms,
        "misses": misses,
    }


def sudirman_measure(max_sessions: int | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Coverage and consistency of the wider sagittal series of Sudirman sessions (no disc truth)."""
    from spinescope import load_study

    from ..datasets.sudirman import patient_sessions

    # The first session of each patient: one session per patient, short run.
    sessions = [sess[0] for sess in patient_sessions(root).values()]
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    summary: dict[str, Any] = {
        "sessions": 0,
        "labelled": 0,
        "with_wider_series": 0,
        "wider_reaches_above": 0,
        "wider_reaches_below_only": 0,
        "top_disagreement": 0,
        "landmark_found": 0,
    }
    agreements: list[float] = []
    for sess in sessions:
        summary["sessions"] += 1
        try:
            study = load_study(str(sess))
        except Exception as exc:  # unreadable session: counted, not fatal
            log.warning("session not loaded: %s", type(exc).__name__)
            continue
        prim_stack = study.best("sag")
        if prim_stack is None:
            continue
        prim, seg = label_spine_ml_segmented(study)
        if not prim.discs:
            continue
        summary["labelled"] += 1
        res = count_evidence(
            study, prim, primary_stack=prim_stack, segs={} if seg is None else {seg.stack.key: seg}
        )
        if res.candidates:
            summary["with_wider_series"] += 1
        if any(c.reach_above_mm >= DEFAULT_COUNTING.reach_min_mm for c in res.candidates):
            summary["wider_reaches_above"] += 1
        elif res.candidates:
            summary["wider_reaches_below_only"] += 1
        for c in res.candidates:
            if c.overlap_discs:
                agreements.append(c.overlap_matched / c.overlap_discs)
        if res.top_disagreement:
            summary["top_disagreement"] += 1
        if res.landmark is not None:
            summary["landmark_found"] += 1
    if agreements:
        arr = np.asarray(agreements)
        summary["overlap_candidates"] = int(arr.size)
        summary["overlap_agreement_median"] = round(float(np.median(arr)), 3)
        summary["overlap_agreement_ge_0.7"] = round(float(np.mean(arr >= 0.7)), 3)
        summary["overlap_agreement_zero"] = round(float(np.mean(arr == 0.0)), 3)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("spider", "sudirman"):
        print(
            "usage: python -m spinescope.anatomy._validate_counting spider|sudirman [--max N]",
            file=sys.stderr,
        )
        return 2
    limit = int(args[args.index("--max") + 1]) if "--max" in args else None
    out = spider_measure(limit) if args[0] == "spider" else sudirman_measure(limit)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "spider_measure", "sudirman_measure", "top_chain_error"]
