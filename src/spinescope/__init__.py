"""spinescope: geometry-aware spine MRI analysis for humans and AI agents.

Research and education only. Not a medical device and not for diagnosis.
"""

from .geometry import Stack, intersection_segment, sample_plane
from .study import Study, list_studies, load_study

__version__ = "0.1.0.dev0"

__all__ = [
    "Stack",
    "Study",
    "__version__",
    "intersection_segment",
    "list_studies",
    "load_study",
    "sample_plane",
]
