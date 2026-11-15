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
from .grid import PlaneGrid, intensity_window, network_input_volume, normalise, plane_grid, resample_slice

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
    """Path of a packaged ONNX weights file (default: the shipped labeller, ``labeller_v1.onnx``).

    An absolute path to an existing ONNX file is returned as it is (candidate weights in evaluation; see
    ``Config.ml_model``).
    """
    if Path(name).is_absolute():
        return Path(name)
    return Path(str(resources.files("spinescope.anatomy.ml") / "assets" / name))


_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[tuple[str, int, int], ort.InferenceSession] = {}
_SESSIONS_MAX = 4  # distinct (weights, thread) combinations kept alive per process
#: Process-wide intra-op thread override set by :func:`set_intra_op_threads` (None = not set). It sits between an
#: explicit ``Config`` value and the ``SPINESCOPE_THREADS`` environment variable in :func:`resolve_threads`.
_PROCESS_INTRA_OP: int | None = None


def set_intra_op_threads(n: int) -> None:
    """Process-wide ONNX Runtime intra-op threads for labeller sessions created from now on (0 = runtime default).

    Same knob as ``SPINESCOPE_THREADS``, for code that sets it in-process. An explicit ``Config`` value wins over it.
    """
    global _PROCESS_INTRA_OP
    if n < 0:
        raise ValueError("intra-op thread count must be >= 0")
    _PROCESS_INTRA_OP = n


def in_worker_process() -> bool:
    """True inside a ``multiprocessing`` child (a pool worker, or a ``spawn``ed child of one)."""
    return multiprocessing.parent_process() is not None


def _process_or_env_threads() -> int | None:
    """Intra-op count from :func:`set_intra_op_threads`, else ``SPINESCOPE_THREADS``; None when neither is set."""
    if _PROCESS_INTRA_OP is not None:
        return _PROCESS_INTRA_OP
    env = os.environ.get(THREADS_ENV, "").strip()
    return int(env) if env.isdigit() else None  # "0" is an explicit ORT default: no pool cap either


def resolve_threads(config: Config | None = None) -> tuple[int, int]:
    """ONNX Runtime ``(intra_op, inter_op)`` thread counts; 0 means ORT's default (all cores).

    The single thread-control mechanism. Precedence: an explicit non-zero ``Config.ml_intra_op_threads`` /
    ``ml_inter_op_threads`` wins; then :func:`set_intra_op_threads`; then ``SPINESCOPE_THREADS`` (intra-op); then,
    inside a multiprocessing child, one thread, so N pool workers do not oversubscribe the machine. Otherwise 0.
    """
    cfg = config or DEFAULT
    intra, inter = cfg.ml_intra_op_threads, cfg.ml_inter_op_threads
    if intra <= 0:
        from_process = _process_or_env_threads()
        intra = from_process if from_process is not None else (1 if in_worker_process() else 0)
    if inter <= 0 and in_worker_process():
        inter = 1
    return max(intra, 0), max(inter, 0)


def session_options(
    intra_op_threads: int | None = None, inter_op_threads: int | None = None
) -> ort.SessionOptions:
    """Quiet CPU ONNX Runtime options for the packaged models. Thread counts default to :func:`resolve_threads`."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised when the extra is missing
        raise ImportError("the learned labellers need onnxruntime: pip install 'spinescope[ml]'") from exc
    default_intra, default_inter = resolve_threads()
    intra = default_intra if intra_op_threads is None else max(intra_op_threads, 0)
    inter = default_inter if inter_op_threads is None else max(inter_op_threads, 0)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if intra > 0:
        opts.intra_op_num_threads = intra
    if inter > 0:
        opts.inter_op_num_threads = inter
    return opts


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
        sess = ort.InferenceSession(
            path, sess_options=session_options(intra, inter), providers=["CPUExecutionProvider"]
        )
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
        return np.zeros((0, n_out, *images.shape[-2:]), dtype=np.float32)
    probs: NDArray[np.float32] | None = None
    step = max(1, int(batch))
    for lo in range(0, images.shape[0], step):
        chunk = images[lo : lo + step]
        # (n, rows, cols) for one-slice models; (n, channels, rows, cols) for adjacent-slice (2.5-D) models.
        x = np.ascontiguousarray(chunk[:, None] if chunk.ndim == 3 else chunk, dtype=np.float32)
        logits = np.asarray(session.run(None, {name: x})[0], dtype=np.float32)
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        logits /= logits.sum(axis=1, keepdims=True)
        if probs is None:
            probs = np.empty((images.shape[0], logits.shape[1], *logits.shape[2:]), dtype=np.float32)
        probs[lo : lo + logits.shape[0]] = logits
    assert probs is not None
    return probs


def input_channels(session: ort.InferenceSession) -> int:
    """Number of input channels of the network: 1 for one-slice models, 3 for adjacent-slice (2.5-D) models."""
    shape = session.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[1], int) and shape[1] > 0:
        return int(shape[1])
    return 1


def adjacent_channels(images: NDArray[np.float32], channels: int) -> NDArray[np.float32]:
    """Stack each slice with its neighbours along the stack as channels (k-1, k, k+1; edges repeat).

    ``images`` is (n, rows, cols) on one grid per slice (the grids share their shape and in-plane offsets, so
    the same pixel index means the same in-plane position on each neighbour). One channel returns the input.
    """
    if channels == 1:
        return images
    if channels != 3:
        raise ValueError(f"unsupported number of input channels: {channels} (1 or 3)")
    n = images.shape[0]
    prev_idx = np.clip(np.arange(n) - 1, 0, max(n - 1, 0))
    next_idx = np.clip(np.arange(n) + 1, 0, max(n - 1, 0))
    return np.stack([images[prev_idx], images, images[next_idx]], axis=1)


def segment_stack(
    stack: Stack, cfg: Config = DEFAULT, session: ort.InferenceSession | None = None
) -> SegmentedStack:
    """Resample every slice of ``stack`` on the fixed-mm grid and run the network on all of them."""
    sess = session if session is not None else load_session(config=cfg)
    res = cfg.ml_resolution_mm
    grids = [plane_grid(stack, k, res) for k in range(stack.n)]
    src = network_input_volume(stack)  # signed DICOM: background shifted to zero; others unchanged
    window = intensity_window(src, cfg.ml_intensity_pct, cfg.background_frac)
    imgs = np.stack([normalise(resample_slice(stack, g, array=src), window) for g in grids])
    probs = run_network(sess, adjacent_channels(imgs, input_channels(sess)))
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=window)


def segment_primary(
    study: Study, config: Config | None = None, *, session: Any = None
) -> SegmentedStack | None:
    """Network output for the primary sagittal stack of ``study`` (``study.best('sag')``); None without sagittal."""
    cfg = config or DEFAULT
    if session is None and cfg.ml_model != MODEL_ASSET:
        session = load_session(model_path(cfg.ml_model))
    stack = study.best("sag")
    if stack is None:
        return None
    return segment_stack(stack, cfg, session=session)


def label_spine_ml(study: Study, config: Config | None = None, *, session: Any = None) -> Anchors:
    """Detect discs, vertebral bodies, sacrum and the mid-sagittal plane with the learned labeller.

    Returns :class:`Anchors` in LPS mm with confidence and evidence. Missing sequences give warnings.
    ``session`` is an optional ONNX Runtime session; by default the packaged weights named by
    ``Config.ml_model`` (``labeller_v1.onnx``). A file must exist in ``ml/assets`` for any other name.
    """

    anchors, _seg = label_spine_ml_segmented(study, config, session=session)
    return anchors


def label_spine_ml_segmented(
    study: Study, config: Config | None = None, *, session: Any = None
) -> tuple[Anchors, SegmentedStack | None]:
    """As :func:`label_spine_ml`, also returning the network output used (None when there is no sagittal).

    The applicability check (:mod:`spinescope.anatomy.ood`) needs the probabilities for its entropy and canal cues.
    """
    from .postprocess import anchors_from_segmentation

    cfg = config or DEFAULT
    if session is None and cfg.ml_model != MODEL_ASSET:
        session = load_session(model_path(cfg.ml_model), config=cfg)
    stack = study.best("sag")
    if stack is None:
        return (
            Anchors(
                method=METHOD,
                warnings=["no sagittal series: spine anatomy cannot be labelled"],
                numbering={"method": "none", "confidence": 0.0, "notes": "no sagittal"},
            ),
            None,
        )
    seg = segment_stack(stack, cfg, session=session)
    return anchors_from_segmentation(seg, cfg), seg


__all__ = [
    "CLASSES",
    "METHOD",
    "MODEL_ASSET",
    "SegmentedStack",
    "adjacent_channels",
    "in_worker_process",
    "input_channels",
    "label_spine_ml",
    "label_spine_ml_segmented",
    "load_session",
    "model_path",
    "resolve_threads",
    "run_network",
    "segment_primary",
    "segment_stack",
    "session_options",
    "set_intra_op_threads",
]
