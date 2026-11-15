"""Learned labeller: ``label_spine_ml(study) -> Anchors`` with a small 2-D U-Net run by ONNX Runtime.

Pipeline (all geometry in LPS mm, from :mod:`.grid`):

1. Pick the sagittal stack (``study.best('sag')``, the same choice as the classical labeller).
2. Resample every slice onto a fixed 1 mm grid (rows superior->inferior, columns anterior->posterior)
   and normalise intensities per volume.
3. Run the network: background / vertebral body / disc / spinal canal probabilities per slice.
4. Mid-sagittal plane: the slice where the cumulative canal area reaches half its total (the canal median).
5. On that plane, disc instances are 8-connected components of the disc class; each gives a height,
   anterior/posterior margins at mid-height, and a centre and normal from the 3-D disc pooled over nearby
   slices (the normal stays on the mid-plane unless ``Config.ml_disc_normal_3d``).
6. Numbering from the bottom: the lowest disc is L5-S1; vertebral bodies are named by the number of
   discs below them. Confidence is lower when no body (or sacrum) is visible below the lowest disc, and
   the field of view is checked for coverage below the lowest disc.

The network is optional: this module needs the ``ml`` extra (onnxruntime) and never imports torch.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ...config import DEFAULT, Config
from ...geometry import Stack
from ...study import Study
from ..anchors import Anchors
from .grid import PlaneGrid, intensity_window, normalise, plane_grid, resample_slice

if TYPE_CHECKING:  # pragma: no cover
    import onnxruntime as ort

log = logging.getLogger(__name__)

METHOD = "label_spine_ml/1 learned 2-D U-Net (SPIDER-trained, ONNX Runtime), mid-sagittal discs, sacrum-up numbering"
MODEL_ASSET = "labeller_v1.onnx"
THREADS_ENV = "SPINESCOPE_THREADS"  # ONNX Runtime intra-op threads for the labeller (overrides the default)
# Classes of the shipped model. A 5-class candidate (sacrum channel, index SACRUM) is supported when its file is named via
# Config.ml_model; see docs/model_card.md.
CLASSES = ("background", "vertebral body", "disc", "spinal canal")
BODY, DISC, CANAL, SACRUM = 1, 2, 3, 4
N_CLASSES = len(CLASSES)


@dataclass
class SegmentedStack:
    """Network output for every slice of a sagittal stack, on the fixed-mm grids."""

    stack: Stack
    grids: list[PlaneGrid]
    probs: NDArray[np.float32]  # (n, classes, rows, cols)
    window: tuple[float, float]
    evidence: list[str] = field(default_factory=list)


def model_path(name: str = MODEL_ASSET) -> Path:
    """Path of a packaged ONNX weights file (default: the shipped labeller, ``labeller_v1.onnx``)."""
    return Path(str(resources.files("spinescope.anatomy.ml") / "assets" / name))


_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[tuple[str, int, int], ort.InferenceSession] = {}
_SESSIONS_MAX = 4  # distinct (weights, thread) combinations kept alive per process


def in_worker_process() -> bool:
    """True inside a ``multiprocessing`` child (a pool worker, or a ``spawn``ed child of one)."""
    return multiprocessing.parent_process() is not None


def resolve_threads(config: Config | None = None) -> tuple[int, int]:
    """ONNX Runtime ``(intra_op, inter_op)`` thread counts; 0 means ORT's default (all cores).

    Precedence: an explicit non-zero ``Config.ml_intra_op_threads`` / ``ml_inter_op_threads`` wins; then the
    environment variable ``SPINESCOPE_THREADS`` (intra-op; ``0`` means ORT's default); then, inside a
    multiprocessing child, one thread, so N pool workers do not oversubscribe the machine. Otherwise 0.
    """
    cfg = config or DEFAULT
    intra, inter = cfg.ml_intra_op_threads, cfg.ml_inter_op_threads
    if intra <= 0:
        env = os.environ.get(THREADS_ENV, "").strip()
        if env.isdigit():  # "0" is an explicit ORT default: no pool cap either
            intra = int(env)
        elif in_worker_process():
            intra = 1
    if inter <= 0 and in_worker_process():
        inter = 1
    return max(intra, 0), max(inter, 0)


def _session_for(path: str, intra: int, inter: int) -> ort.InferenceSession:
    key = (path, intra, inter)
    with _SESSION_LOCK:
        sess = _SESSIONS.get(key)
        if sess is not None:
            return sess
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised when the extra is missing
            raise ImportError("label_spine_ml needs onnxruntime: pip install 'spinescope[ml]'") from exc
        if not Path(path).is_file():
            raise FileNotFoundError(f"labeller weights not found: {path}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if intra > 0:
            opts.intra_op_num_threads = intra
        if inter > 0:
            opts.inter_op_num_threads = inter
        sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        if len(_SESSIONS) >= _SESSIONS_MAX:
            _SESSIONS.pop(next(iter(_SESSIONS)))
        _SESSIONS[key] = sess
        return sess


def load_session(
    path: str | Path | None = None,
    *,
    config: Config | None = None,
    intra_op_threads: int | None = None,
    inter_op_threads: int | None = None,
) -> ort.InferenceSession:
    """ONNX Runtime session for the labeller (the packaged weights unless ``path`` is given).

    Cached per process and thread setting, and safe to call from several threads. Thread counts default to
    :func:`resolve_threads`; pass explicit counts to override (0 = ORT default).
    """
    intra, inter = resolve_threads(config)
    if intra_op_threads is not None:
        intra = max(intra_op_threads, 0)
    if inter_op_threads is not None:
        inter = max(inter_op_threads, 0)
    return _session_for(str(path) if path is not None else str(model_path()), intra, inter)


def run_network(
    session: ort.InferenceSession, images: NDArray[np.float32], batch: int = 1
) -> NDArray[np.float32]:
    """Softmax class probabilities for normalised slices ``(n, rows, cols)`` -> ``(n, classes, rows, cols)``.

    The class count is taken from the network output, so 4-class (v1) and 5-class (v2) weights both work.
    Slices run ``batch`` at a time (default 1: on the CPU a batch of 8 full-resolution slices was ~1.9x slower and
    needed ~250 MB more peak memory for the same output; measured on a 120-slice SPACE stack, lab14_perf).
    The output is written in place (no list of batches to concatenate).
    """
    name = session.get_inputs()[0].name
    if images.shape[0] == 0:
        n_out = int(session.get_outputs()[0].shape[1] or N_CLASSES)
        return np.zeros((0, n_out, *images.shape[1:]), dtype=np.float32)
    probs: NDArray[np.float32] | None = None
    step = max(1, int(batch))
    for lo in range(0, images.shape[0], step):
        x = np.ascontiguousarray(images[lo : lo + step, None], dtype=np.float32)
        logits = np.asarray(session.run(None, {name: x})[0], dtype=np.float32)
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True)
        if probs is None:
            probs = np.empty((images.shape[0], logits.shape[1], *logits.shape[2:]), dtype=np.float32)
        probs[lo : lo + logits.shape[0]] = logits
    assert probs is not None
    return probs


def segment_stack(
    stack: Stack, cfg: Config = DEFAULT, session: ort.InferenceSession | None = None
) -> SegmentedStack:
    """Resample every slice of ``stack`` on the fixed-mm grid and run the network on all of them."""
    sess = session if session is not None else load_session(config=cfg)
    res = cfg.ml_resolution_mm
    grids = [plane_grid(stack, k, res) for k in range(stack.n)]
    window = intensity_window(stack.vol, cfg.ml_intensity_pct, cfg.background_frac)
    imgs = np.stack([normalise(resample_slice(stack, g), window) for g in grids])
    probs = run_network(sess, imgs)
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=window)


def label_spine_ml(study: Study, config: Config | None = None, *, session: Any = None) -> Anchors:
    """Detect discs, vertebral bodies, sacrum and the mid-sagittal plane with the learned labeller.

    Returns :class:`Anchors` in LPS mm with confidence and evidence. Missing sequences give warnings.
    ``session`` is an optional ONNX Runtime session; by default the packaged weights named by
    ``Config.ml_model`` (``labeller_v1.onnx``). A file must exist in ``ml/assets`` for any other name.
    """
    from .postprocess import anchors_from_segmentation

    cfg = config or DEFAULT
    if session is None and cfg.ml_model != MODEL_ASSET:
        session = load_session(model_path(cfg.ml_model), config=cfg)
    stack = study.best("sag")
    if stack is None:
        return Anchors(
            method=METHOD,
            warnings=["no sagittal series: spine anatomy cannot be labelled"],
            numbering={"method": "none", "confidence": 0.0, "notes": "no sagittal"},
        )
    seg = segment_stack(stack, cfg, session=session)
    return anchors_from_segmentation(seg, cfg)


__all__ = [
    "CLASSES",
    "METHOD",
    "MODEL_ASSET",
    "SegmentedStack",
    "in_worker_process",
    "label_spine_ml",
    "load_session",
    "model_path",
    "resolve_threads",
    "run_network",
    "segment_stack",
]
