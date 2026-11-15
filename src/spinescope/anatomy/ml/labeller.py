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
from dataclasses import dataclass, field
from functools import lru_cache
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


#: ONNX Runtime intra-op threads for new sessions (0 = runtime default). Set with :func:`set_intra_op_threads`
#: when several labelling jobs share a machine (each process should use a few threads, not every core).
_INTRA_OP_THREADS = 0


def set_intra_op_threads(n: int) -> None:
    """Limit ONNX Runtime intra-op threads for labeller sessions created from now on (0 = runtime default)."""
    global _INTRA_OP_THREADS
    if n < 0:
        raise ValueError("intra-op thread count must be >= 0")
    if n != _INTRA_OP_THREADS:
        _INTRA_OP_THREADS = n
        _session_for.cache_clear()


def session_options() -> ort.SessionOptions:
    """ONNX Runtime options shared by the packaged models (quiet, CPU, thread limit from :func:`set_intra_op_threads`)."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised when the extra is missing
        raise ImportError("the learned labellers need onnxruntime: pip install 'spinescope[ml]'") from exc
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if _INTRA_OP_THREADS > 0:
        opts.intra_op_num_threads = _INTRA_OP_THREADS
    return opts


@lru_cache(maxsize=4)
def _session_for(path: str) -> ort.InferenceSession:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised when the extra is missing
        raise ImportError("label_spine_ml needs onnxruntime: pip install 'spinescope[ml]'") from exc
    if not Path(path).is_file():
        raise FileNotFoundError(f"labeller weights not found: {path}")
    return ort.InferenceSession(path, sess_options=session_options(), providers=["CPUExecutionProvider"])


def load_session(path: str | Path | None = None) -> ort.InferenceSession:
    """ONNX Runtime session for the labeller (the packaged weights unless ``path`` is given). Cached."""
    return _session_for(str(path) if path is not None else str(model_path()))


def run_network(
    session: ort.InferenceSession, images: NDArray[np.float32], batch: int = 8
) -> NDArray[np.float32]:
    """Softmax class probabilities for normalised slices ``(n, rows, cols)`` -> ``(n, classes, rows, cols)``.

    The class count is taken from the network output, so 4-class (v1) and 5-class (v2) weights both work.
    """
    name = session.get_inputs()[0].name
    out: list[NDArray[np.float32]] = []
    for lo in range(0, images.shape[0], batch):
        x = np.ascontiguousarray(images[lo : lo + batch, None], dtype=np.float32)
        logits = np.asarray(session.run(None, {name: x})[0], dtype=np.float32)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        out.append((e / e.sum(axis=1, keepdims=True)).astype(np.float32))
    if not out:
        n_out = int(session.get_outputs()[0].shape[1] or N_CLASSES)
        return np.zeros((0, n_out, *images.shape[1:]), dtype=np.float32)
    return np.concatenate(out, axis=0)


def segment_stack(
    stack: Stack, cfg: Config = DEFAULT, session: ort.InferenceSession | None = None
) -> SegmentedStack:
    """Resample every slice of ``stack`` on the fixed-mm grid and run the network on all of them."""
    sess = session if session is not None else load_session()
    res = cfg.ml_resolution_mm
    grids = [plane_grid(stack, k, res) for k in range(stack.n)]
    window = intensity_window(stack.vol, cfg.ml_intensity_pct, cfg.background_frac)
    imgs = np.stack([normalise(resample_slice(stack, g), window) for g in grids])
    probs = run_network(sess, imgs)
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=window)


def segment_primary(study: Study, config: Config | None = None, *, session: Any = None) -> SegmentedStack | None:
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
    from .postprocess import anchors_from_segmentation

    cfg = config or DEFAULT
    if session is None and cfg.ml_model != MODEL_ASSET:
        session = load_session(model_path(cfg.ml_model))
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
    "label_spine_ml",
    "load_session",
    "model_path",
    "run_network",
    "segment_primary",
    "segment_stack",
    "session_options",
    "set_intra_op_threads",
]
