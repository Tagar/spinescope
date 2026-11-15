"""Automatic spinal anatomy: levels, midline, canal and conus, returned as :class:`Anchors`.

The labelling pipeline (scipy, the ONNX labeller) is imported on first use of ``label_spine``,
``label_spine_cli`` or ``label_spine_ml``, so ``import spinescope.anatomy`` stays light.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .anchors import Anchors, Disc, Vertebra

_LAZY: dict[str, str] = {
    "label_spine": ".pipeline",
    "label_spine_cli": ".pipeline",
    "label_spine_ml": ".ml",
}

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .ml import label_spine_ml
    from .pipeline import label_spine, label_spine_cli


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'spinescope.anatomy' has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = ["Anchors", "Disc", "Vertebra", "label_spine", "label_spine_cli", "label_spine_ml"]
