"""Scorer adapter for the OSF multi-scanner lumbar database (project qx5rt): vendor-diverse ground truth.

Not used by the labeller. Ground truth is the binary disc masks of each session (``Labels/BS_L<a>_<b>``,
drawn on the non-contrast T1, levels L1-L2 .. L4-L5; the L5-S1 disc is not labelled). Level names come
from the mask file names, so no rank assumption is needed. Each annotated disc's centroid (LPS mm, on the
T1 grid) is matched to the nearest predicted disc in the sagittal (y, z) plane within :data:`MATCH_MM`,
as in :mod:`spinescope.anatomy._validate_sudirman`.

Only four discs per session are annotated, so predicted discs without a truth disc are not counted and
precision is not reported. Outputs carry pseudonymous session keys (``ID04:Siemens_Espree``), never paths.

Run ``python -m spinescope.anatomy._validate_qx5rt --out summary.json`` after
``spinescope.datasets.fetch("lumbar_multiscanner_qx5rt")``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..datasets.registry import REGISTRY
from ..geometry import Stack
from ..io.volumes import read_label_volume, read_volume
from ..study import from_stacks
from ._validate import TruthDisc, level_offset
from .anchors import Anchors
from .pipeline import label_spine

log = logging.getLogger(__name__)

DATASET_KEY = "lumbar_multiscanner_qx5rt"
#: In-plane (sagittal y, z) distance within which a predicted disc counts as the annotated one (as in the other scorers).
MATCH_MM = 12.0
#: Masks are stored as float with ~1e-14 noise; anything above this is inside the mask.
MASK_THRESHOLD = 0.5
_T1_NAMES = ("T1_non_contrast.nii.gz", "T1.nii.gz")
_MASK_RE = re.compile(r"^BS_L(\d+)_(\d+)\.nii(?:\.gz)?$")


@dataclass(frozen=True)
class Session:
    """One scanning session: the T1 the masks were drawn on, and its disc masks by level name."""

    key: str  # pseudonymous, e.g. "ID04:Siemens_Espree" (no paths)
    subject: str
    vendor: str  # "GE", "Philips" or "Siemens" (folder prefix)
    model: str  # scanner model from the folder name, e.g. "Espree"
    t1: Path
    disc_masks: dict[str, Path] = field(default_factory=dict)  # "L4-L5" -> mask file


@dataclass(frozen=True)
class LevelTruth:
    """Annotated disc: level name and the centroid of its mask in LPS mm."""

    level: str
    center: np.ndarray
    voxels: int


@dataclass
class LevelRow:
    """Score of one annotated disc against the nearest predicted disc in the sagittal plane."""

    session: str
    vendor: str
    model: str
    level: str
    predicted: str | None  # name of the nearest predicted disc, any distance; None if none was predicted
    dist_mm: float | None  # in-plane (y, z) distance to that disc
    dist3d_mm: float | None
    matched: bool  # dist_mm <= MATCH_MM
    name_ok: bool  # matched and the predicted name equals the annotated level
    offset: int | None  # level index(predicted) - level index(annotated), when matched


@dataclass
class SessionResult:
    key: str
    vendor: str
    model: str
    n_levels: int = 0
    rows: list[LevelRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None  # exception class name only


def default_root() -> Path:
    """The ``raw/`` folder of the registered dataset (``$SPINESCOPE_DATA/lumbar_multiscanner_qx5rt/raw``)."""
    return REGISTRY[DATASET_KEY].path() / "raw"


def discover(root: str | Path | None = None) -> list[Session]:
    """Sessions with a non-contrast T1 and at least one disc mask, sorted by key."""
    base = Path(root).expanduser() if root is not None else default_root()
    if not base.is_dir():
        raise FileNotFoundError(
            f"no {DATASET_KEY} raw folder at {base}; fetch it with spinescope.datasets.fetch"
        )
    out: list[Session] = []
    for subj in sorted(base.glob("ID*")):
        if not subj.is_dir():
            continue
        for vend in sorted(p for p in subj.iterdir() if p.is_dir()):
            img_dir, lab_dir = vend / "Images", vend / "Labels"
            t1 = next((img_dir / n for n in _T1_NAMES if (img_dir / n).is_file()), None)
            if t1 is None or not lab_dir.is_dir():
                continue
            masks: dict[str, Path] = {}
            for p in sorted(lab_dir.iterdir()):
                m = _MASK_RE.match(p.name)
                if m:
                    masks[f"L{m.group(1)}-L{m.group(2)}"] = p
            if not masks:
                continue
            vendor, _, model = vend.name.partition("_")
            out.append(
                Session(
                    key=f"{subj.name}:{vend.name}",
                    subject=subj.name,
                    vendor=vendor,
                    model=model,
                    t1=t1,
                    disc_masks=masks,
                )
            )
    return out


def truth_levels(session: Session, stack: Stack) -> list[LevelTruth]:
    """Mask centroids in LPS mm, using the grid of ``stack`` (the T1 read with :func:`read_volume`)."""
    out: list[LevelTruth] = []
    for level, path in sorted(session.disc_masks.items()):
        arr = read_label_volume(str(path))
        if arr.shape != stack.vol.shape:
            raise ValueError(
                f"mask grid {arr.shape} differs from image grid {stack.vol.shape} for {session.key}"
            )
        idx = np.argwhere(arr > MASK_THRESHOLD)
        if idx.size == 0:
            continue
        k, r, c = idx.mean(axis=0)
        out.append(LevelTruth(level, np.asarray(stack.pix_to_lps(k, r, c), dtype=np.float64), len(idx)))
    return out


def truth_discs(levels: Sequence[LevelTruth]) -> list[TruthDisc]:
    """The annotated discs as :class:`TruthDisc` (for :func:`spinescope.anatomy.qa.overlay`)."""
    return [
        TruthDisc(label=0, name=t.level, center=t.center, normal=np.array([0.0, 0.0, 1.0])) for t in levels
    ]


def score_levels(
    anchors: Anchors, truths: Sequence[LevelTruth], session: Session | None = None
) -> list[LevelRow]:
    """For each annotated disc, the nearest predicted disc in the sagittal plane and whether its name is right."""
    key = session.key if session else ""
    vendor = session.vendor if session else ""
    model = session.model if session else ""
    preds = [(name, np.asarray(d.center, dtype=np.float64)) for name, d in anchors.discs.items()]
    rows: list[LevelRow] = []
    for t in truths:
        best: tuple[float, float, str] | None = None  # (in-plane dist, 3-D dist, name)
        for name, c in preds:
            d2 = float(np.linalg.norm(c[1:] - t.center[1:]))
            if best is None or d2 < best[0]:
                best = (d2, float(np.linalg.norm(c - t.center)), name)
        matched = best is not None and best[0] <= MATCH_MM
        name_ok = bool(matched and best is not None and best[2] == t.level)
        rows.append(
            LevelRow(
                session=key,
                vendor=vendor,
                model=model,
                level=t.level,
                predicted=best[2] if best else None,
                dist_mm=best[0] if best else None,
                dist3d_mm=best[1] if best else None,
                matched=matched,
                name_ok=name_ok,
                offset=level_offset(best[2], t.level) if (matched and best) else None,
            )
        )
    return rows


def evaluate_session(
    session: Session, *, backend: str = "auto", threads: int | None = None
) -> tuple[SessionResult, Anchors | None, Stack | None]:
    """Run the labeller on the session's T1 and score it. Returns the result plus anchors and stack (for renders)."""
    res = SessionResult(key=session.key, vendor=session.vendor, model=session.model)
    try:
        stack = read_volume(str(session.t1))
        truths = truth_levels(session, stack)
        res.n_levels = len(truths)
        study = from_stacks([stack])
        cfg = Config(ml_intra_op_threads=threads) if threads is not None else None
        anchors = label_spine(study, cfg, backend=backend)
        res.warnings = list(anchors.warnings)
        res.rows = score_levels(anchors, truths, session)
        return res, anchors, stack
    except Exception as e:  # one bad session must not stop the run
        res.error = type(e).__name__
        log.warning("session %s failed: %s", session.key, type(e).__name__)
        return res, None, None


def _pct(xs: list[float], q: float) -> float | None:
    return round(float(np.percentile(xs, q)), 2) if xs else None


def summarise(rows: Sequence[LevelRow]) -> dict[str, Any]:
    """Level-wise metrics over ``rows``: recall within MATCH_MM, correct-name recall, error and offsets."""
    n = len(rows)
    matched = [r for r in rows if r.matched]
    dists = [r.dist_mm for r in matched if r.dist_mm is not None]
    d3 = [r.dist3d_mm for r in matched if r.dist3d_mm is not None]
    offsets: dict[str, int] = {}
    for r in matched:
        if r.offset is not None:
            offsets[str(r.offset)] = offsets.get(str(r.offset), 0) + 1
    return {
        "n_levels": n,
        "recall_12mm": round(len(matched) / n, 3) if n else None,
        "label_accuracy_matched": round(sum(r.name_ok for r in matched) / len(matched), 3)
        if matched
        else None,
        "correct_name_recall": round(sum(r.name_ok for r in rows) / n, 3) if n else None,
        "inplane_err_median_mm": _pct(dists, 50),
        "inplane_err_p90_mm": _pct(dists, 90),
        "err3d_median_mm": _pct(d3, 50),
        "offset_counts": dict(sorted(offsets.items(), key=lambda kv: int(kv[0]))),
    }


def summarise_by(results: Sequence[SessionResult]) -> dict[str, Any]:
    """Pooled metrics overall, per vendor and per (vendor, model), plus session counts and failures."""
    rows = [r for s in results for r in s.rows]
    groups: dict[str, list[SessionResult]] = {"all": list(results)}
    for s in results:
        groups.setdefault(f"vendor={s.vendor}", []).append(s)
        groups.setdefault(f"model={s.vendor}/{s.model}", []).append(s)
    out: dict[str, Any] = {"overall": summarise(rows)}
    for name, members in sorted(groups.items()):
        if name == "all":
            continue
        out[name] = {
            "sessions": len(members),
            "failed": sum(1 for s in members if s.error),
            **summarise([r for s in members for r in s.rows]),
        }
    out["sessions"] = len(results)
    out["failed"] = sum(1 for s in results if s.error)
    out["field_strength"] = "not stated in the release"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="raw/ folder (default: the registered dataset)")
    ap.add_argument("--out", required=True, help="write the summary JSON here (no paths inside)")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--threads", type=int, default=None, help="ONNX Runtime threads (0 = ORT default)")
    args = ap.parse_args(argv)
    sessions = discover(args.root)
    results = [evaluate_session(s, backend=args.backend, threads=args.threads)[0] for s in sessions]
    summary = summarise_by(results)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["overall"], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
