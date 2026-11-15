"""Optional learned labeller: a small 2-D U-Net trained on SPIDER, run with ONNX Runtime.

Install the runtime with ``pip install 'spinescope[ml]'``. Training lives in ``tools/train_labeller.py``
(dev-time only, needs the ``train`` extra). See ``docs/model_card.md``.
"""

from .grid import PlaneGrid, intensity_window, normalise, plane_grid, resample_slice
from .labeller import (
    CLASSES,
    label_spine_ml,
    load_session,
    model_path,
    segment_primary,
    segment_stack,
    set_intra_op_threads,
)
from .postprocess import anchors_from_segmentation

__all__ = [
    "CLASSES",
    "PlaneGrid",
    "anchors_from_segmentation",
    "intensity_window",
    "label_spine_ml",
    "load_session",
    "model_path",
    "normalise",
    "plane_grid",
    "resample_slice",
    "segment_primary",
    "segment_stack",
    "set_intra_op_threads",
]
