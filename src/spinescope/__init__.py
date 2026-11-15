"""spinescope: geometry-aware spine MRI analysis for humans and AI agents.

Research and education only. Not a medical device and not for diagnosis.

Public names are imported lazily (PEP 562), so ``import spinescope`` only pays for numpy and the
package itself; DICOM (pydicom) and the rest load on first use of the name.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0.dev0"

_LAZY: dict[str, str] = {
    "Stack": ".geometry",
    "intersection_segment": ".geometry",
    "sample_plane": ".geometry",
    "Study": ".study",
    "list_studies": ".study",
    "load_study": ".study",
}

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .geometry import Stack, intersection_segment, sample_plane
    from .study import Study, list_studies, load_study


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'spinescope' has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "Stack",
    "Study",
    "__version__",
    "intersection_segment",
    "list_studies",
    "load_study",
    "sample_plane",
]
