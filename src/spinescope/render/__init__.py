"""Image toolkit: windowing, contact sheets, zoom crops, annotations, cross-reference, MPR, enhancement.

Every public function returns RGB ``(H, W, 3)`` uint8 arrays (``save_png`` writes them). Geometry is in
patient LPS mm; pixels are only used where a reader needs native coordinates.
"""

from __future__ import annotations

from .annotate import (
    add_rulers,
    arrow,
    circle,
    crosshair,
    label,
    line,
    orientation_letters,
    pixel_labels,
    scale_bar,
    text_at,
)
from .crossref import linked_views, reference_lines
from .enhance import (
    ENHANCEMENT_WARNING,
    compare_panel,
    denoise,
    fusion,
    fusion_slice,
    ratio_image,
    t2_t1_ratio,
)
from .frame import Frame, direction_letter, nice_step, slice_frame
from .layout import grid, hstack, title_band, vstack, with_title
from .mpr import (
    Reformat,
    curved_coronal,
    disc_plane,
    effective_resolution,
    reformat,
    render_reformat,
    resample_onto,
    sample_frame,
    sample_points,
)
from .sheets import contact_sheet, slice_label_lines, triptych, zoom_crop
from .window import (
    contrast_of,
    foreground_values,
    grey_rgb,
    presets,
    resolve_window,
    save_png,
    series_window,
    to_uint8,
)

__all__ = [
    "ENHANCEMENT_WARNING",
    "Frame",
    "Reformat",
    "add_rulers",
    "arrow",
    "circle",
    "compare_panel",
    "contact_sheet",
    "contrast_of",
    "crosshair",
    "curved_coronal",
    "denoise",
    "direction_letter",
    "disc_plane",
    "effective_resolution",
    "foreground_values",
    "fusion",
    "fusion_slice",
    "grey_rgb",
    "grid",
    "hstack",
    "label",
    "line",
    "linked_views",
    "nice_step",
    "orientation_letters",
    "pixel_labels",
    "presets",
    "ratio_image",
    "reference_lines",
    "reformat",
    "render_reformat",
    "resample_onto",
    "resolve_window",
    "sample_frame",
    "sample_points",
    "save_png",
    "scale_bar",
    "series_window",
    "slice_frame",
    "slice_label_lines",
    "t2_t1_ratio",
    "text_at",
    "title_band",
    "to_uint8",
    "triptych",
    "vstack",
    "with_title",
    "zoom_crop",
]
