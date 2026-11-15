"""Lumbosacral junction: which of the lowest disc candidates is L5-S1, and is a lower one a sacral remnant?

The learned labeller (SPIDER-trained, no sacrum class) sometimes finds an extra disc below the true L5-S1: an S1-S2
remnant or a sacral segment. Every level name then shifts up by one (lab02, lab08). This module decides it outside
the segmentation network.

For the lowest ``K`` disc candidates (bottom-up) it cuts a sagittal patch of the image around each disc centre
(:func:`candidate_patches`, 96 x 96 mm at 1 mm, the disc at the upper-middle of the patch so the sacrum below is in
view) and scores it with a small 2-D CNN (packaged ONNX file ``lumbosacral_v1.onnx``): P(this disc is L5-S1).

Decision (:func:`decide`), with the thresholds of the packaged ``lumbosacral_v1.json``:

* ``keep``: the lowest candidate is L5-S1 (no change);
* ``drop``: the candidate with the highest P is the k-th from the bottom (k >= 1) with P at least ``drop_min_p``:
  the k lowest candidates are removed (:func:`drop_lowest`) and the names count up from the new lowest disc;
* ``unsure``: a lower candidate may be a sacral disc but the evidence does not reach the drop threshold. The numbering
  status is capped at ``unconfirmed`` (``pipeline._apply_status_caps``);
* ``not_assessed``: no packaged model, no sagittal segmentation or no disc.

The module only removes candidates. It never adds one and never shifts the remaining discs; the transitional offset
votes of :mod:`fusion` still run on the reduced set. Training, threshold selection and evaluation are in
``tools/lumbosacral.py``; the training data are SPIDER training volumes (labels from the masks) only.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..config import DEFAULT, Config
from ..geometry import Stack
from .anchors import Anchors, Vertebra
from .ml.grid import PlaneGrid, normalise, resample_slice
from .ml.labeller import SegmentedStack, session_options
from .ml.postprocess import mid_sagittal
from .numbering import disc_names, vertebra_name

if TYPE_CHECKING:  # pragma: no cover
    import onnxruntime as ort

log = logging.getLogger(__name__)

MODEL_JSON = "lumbosacral_v1.json"
MODEL_ONNX = "lumbosacral_v1.onnx"
#: Candidates considered: the lowest K discs. A sacral remnant is at most the second or third disc from the bottom.
K_CANDIDATES = 3
#: Patch geometry (mm, 1 mm grid): the disc centre sits PATCH_ABOVE_MM below the patch top and PATCH_ANTERIOR_MM
#: anterior to the patch's anterior-posterior... see :func:`patch_grid`. 96 mm covers about three levels.
PATCH_MM = 96
PATCH_ABOVE_MM = 40.0  # disc centre this far below the patch's superior edge (room for the sacrum below)
PATCH_ANTERIOR_MM = 40.0  # disc centre this far posterior to the patch's anterior edge (canal in view)


@dataclass(frozen=True)
class LumbosacralConfig:
    """Decision thresholds on P(L5-S1). Selected on SPIDER training volumes held out from the CNN fit (lab13 report)."""

    #: A lower candidate is dropped when its P(L5-S1) is the best and reaches this.
    drop_min_p: float = 0.8
    #: The lowest candidate is kept as L5-S1 when it is the best and its P reaches this.
    keep_min_p: float = 0.5
    #: A lower candidate that is best but below ``drop_min_p`` (and at least this) makes the numbering unsure.
    unsure_min_p: float = 0.5


DEFAULT_LUMBOSACRAL = LumbosacralConfig()


@dataclass
class Candidate:
    """One of the lowest discs: its provisional label (count from the bottom, before any drop) and P(L5-S1)."""

    label: str
    rank: int  # 0 = lowest
    p_l5s1: float = float("nan")


class Scorer(Protocol):
    """Anything that scores patches (the packaged model, or a stand-in in tests)."""

    config: LumbosacralConfig

    def predict(self, patches: NDArray[np.float32]) -> NDArray[np.float64]: ...


@dataclass
class LumbosacralDecision:
    """Outcome of :func:`decide`. ``drop`` is the number of lowest discs to remove."""

    status: str  # keep | drop | unsure | not_assessed
    drop: int
    candidates: list[Candidate] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def probs(self) -> list[float]:
        return [c.p_l5s1 for c in self.candidates]

    def to_dict(self) -> dict[str, Any]:
        """Plain record for ``anchors.numbering['lumbosacral']`` (no file paths, no patient data)."""
        return {
            "status": self.status,
            "drop": self.drop,
            "p_l5s1": [round(float(p), 4) for p in self.probs],
            "candidates": [c.label for c in self.candidates],
            "evidence": list(self.evidence),
        }


@dataclass
class LumbosacralModel:
    """The packaged patch CNN (ONNX file) with its thresholds and training record."""

    onnx_path: Path
    config: LumbosacralConfig = DEFAULT_LUMBOSACRAL
    training: dict[str, Any] = field(default_factory=dict)

    def predict(self, patches: NDArray[np.float32]) -> NDArray[np.float64]:
        """P(L5-S1) for patches ``(n, PATCH_MM, PATCH_MM)`` normalised to [0, 1]."""
        if patches.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        sess = _session(str(self.onnx_path))
        name = sess.get_inputs()[0].name
        x = np.ascontiguousarray(patches[:, None], dtype=np.float32)
        logits = np.asarray(sess.run(None, {name: x})[0], dtype=np.float64).reshape(-1)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


@lru_cache(maxsize=2)
def _session(path: str) -> ort.InferenceSession:
    import onnxruntime as ort

    return ort.InferenceSession(path, sess_options=session_options(), providers=["CPUExecutionProvider"])


def model_dir() -> Path:
    return Path(str(resources.files("spinescope.anatomy")))


@lru_cache(maxsize=2)
def load_model(directory: str | None = None) -> LumbosacralModel | None:
    """The packaged lumbosacral model (or the one in ``directory``); None when either file is missing."""
    root = Path(directory) if directory else model_dir()
    meta, onnx = root / MODEL_JSON, root / MODEL_ONNX
    if not (meta.is_file() and onnx.is_file()):
        return None
    d = json.loads(meta.read_text(encoding="utf-8"))
    th = d.get("thresholds", {})
    cfg = LumbosacralConfig(
        drop_min_p=float(th.get("drop_min_p", DEFAULT_LUMBOSACRAL.drop_min_p)),
        keep_min_p=float(th.get("keep_min_p", DEFAULT_LUMBOSACRAL.keep_min_p)),
        unsure_min_p=float(th.get("unsure_min_p", DEFAULT_LUMBOSACRAL.unsure_min_p)),
    )
    return LumbosacralModel(onnx_path=onnx, config=cfg, training=dict(d.get("training", {})))


# ----------------------------------------------------------------- patches


def patch_grid(grid: PlaneGrid, cdown: float, cpost: float) -> PlaneGrid:
    """The PATCH_MM x PATCH_MM grid around the plane point (cdown, cpost) of the slice ``grid`` lies on."""
    return PlaneGrid(
        k=grid.k,
        ipp=grid.ipp,
        e_down=grid.e_down,
        e_post=grid.e_post,
        down0=cdown - PATCH_ABOVE_MM,
        post0=cpost - PATCH_ANTERIOR_MM,
        res_mm=1.0,
        shape=(PATCH_MM, PATCH_MM),
    )


def disc_patch(stack: Stack, grid: PlaneGrid, cdown: float, cpost: float, window: tuple[float, float]) -> NDArray[np.float32]:
    """Normalised image patch around (cdown, cpost) on slice ``grid.k`` of ``stack``."""
    return normalise(resample_slice(stack, patch_grid(grid, cdown, cpost)), window)


def candidate_patches(
    seg: SegmentedStack, anchors: Anchors, cfg: Config = DEFAULT
) -> tuple[list[Candidate], NDArray[np.float32]]:
    """The lowest :data:`K_CANDIDATES` discs (bottom-up) and their patches on the mid-sagittal slice of ``seg``."""
    empty = np.zeros((0, PATCH_MM, PATCH_MM), dtype=np.float32)
    if not anchors.discs:
        return [], empty
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        return [], empty
    grid = seg.grids[mid.k]
    order = sorted(anchors.discs, key=lambda lab: anchors.discs[lab].center[2])  # bottom-up
    cands: list[Candidate] = []
    patches: list[NDArray[np.float32]] = []
    for i, lab in enumerate(order[:K_CANDIDATES]):
        cdown, cpost = (float(v) for v in grid.mm_of(np.asarray(anchors.discs[lab].center))[:2])
        patches.append(disc_patch(seg.stack, grid, cdown, cpost, seg.window))
        cands.append(Candidate(label=lab, rank=i))
    return cands, np.stack(patches).astype(np.float32)


# ----------------------------------------------------------------- decision


def decide(cands: list[Candidate], patches: NDArray[np.float32], model: Scorer | None) -> LumbosacralDecision:
    """Score the candidates and decide whether the lowest discs are sacral (see the module docstring)."""
    if model is None:
        return LumbosacralDecision("not_assessed", 0, warnings=["lumbosacral model not packaged"])
    if not cands:
        return LumbosacralDecision("not_assessed", 0, warnings=["no disc candidates for the lumbosacral check"])
    probs = model.predict(patches)
    for c, p in zip(cands, probs, strict=True):
        c.p_l5s1 = float(p)
    th = model.config
    best = max(range(len(cands)), key=lambda i: (round(cands[i].p_l5s1, 12), -i))
    pb = cands[best].p_l5s1
    ev = [f"P(L5-S1) rank {c.rank} ({c.label}) = {c.p_l5s1:.2f}" for c in cands]
    if best == 0 and pb >= th.keep_min_p:
        return LumbosacralDecision("keep", 0, cands, ev + ["lowest disc is L5-S1"])
    if best >= 1 and pb >= th.drop_min_p:
        ev.append(f"{best} lowest disc(s) are sacral (S1-S2 remnant or sacral segment): removed")
        return LumbosacralDecision("drop", best, cands, ev)
    warns = [
        f"lumbosacral check unsure: best candidate rank {best} ({cands[best].label}) has P(L5-S1) {pb:.2f}; "
        "a lower disc may be a sacral remnant, or the lowest disc may be L4-L5"
    ]
    return LumbosacralDecision("unsure", 0, cands, ev, warns)


# ----------------------------------------------------------------- applying the drop


def drop_lowest(anchors: Anchors, k: int) -> dict[str, str]:
    """Remove the ``k`` lowest discs and count the names up from the new lowest disc. Returns old -> new disc names.

    Vertebral bodies are renamed by the number of remaining discs below them; when two bodies fall below the new
    lowest disc, the one nearest to it keeps the name S1. The conus level follows its disc (or is cleared).
    """
    if k <= 0 or not anchors.discs:
        return {lab: lab for lab in anchors.discs}
    order = sorted(anchors.discs, key=lambda lab: anchors.discs[lab].center[2])  # bottom-up
    k = min(k, len(order))
    dropped = order[:k]
    kept = order[k:]
    for lab in dropped:
        anchors.discs.pop(lab)
    new_names = disc_names(len(kept))  # bottom-up
    mapping = dict(zip(kept, new_names, strict=True))
    anchors.discs = {mapping[lab]: anchors.discs[lab] for lab in kept}
    for d in anchors.discs.values():
        d.evidence.append(f"lumbosacral: {k} sacral disc(s) below removed; numbered from the new lowest disc")
    z_kept = [float(anchors.discs[name].center[2]) for name in new_names]
    verts: dict[str, Vertebra] = {}
    for v in anchors.vertebrae.values():
        z = float(v.center[2])
        new = vertebra_name(sum(1 for zk in z_kept if zk < z))
        if new in verts and float(verts[new].center[2]) >= z:
            continue  # collision below the lowest disc: keep the body nearest to it
        verts[new] = v
    anchors.vertebrae = verts
    if anchors.conus and anchors.conus.get("level"):
        lvl = str(anchors.conus["level"])
        anchors.conus["level"] = mapping.get(lvl, None if lvl in dropped else lvl)
    return mapping


def lumbosacral_record(dec: LumbosacralDecision, applied: int) -> dict[str, Any]:
    """The ``numbering['lumbosacral']`` record: the decision and how many discs were removed."""
    rec = dec.to_dict()
    rec["applied_drop"] = int(applied)
    rec["warnings"] = list(dec.warnings)
    return rec


__all__ = [
    "DEFAULT_LUMBOSACRAL",
    "K_CANDIDATES",
    "PATCH_MM",
    "Candidate",
    "LumbosacralConfig",
    "LumbosacralDecision",
    "LumbosacralModel",
    "candidate_patches",
    "decide",
    "disc_patch",
    "drop_lowest",
    "load_model",
    "lumbosacral_record",
    "patch_grid",
]
