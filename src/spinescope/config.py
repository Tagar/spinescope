"""Tunable defaults. Every value is in mm, degrees or relative units, with its source noted.

Modules take an optional ``Config``; nothing here is tuned to a single study.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Anatomy priors (adult lumbar spine; ranges from standard anatomy texts, validated on SPIDER).
    disc_height_mm: tuple[float, float] = (4.0, 18.0)
    body_height_mm: tuple[float, float] = (18.0, 36.0)
    level_spacing_mm: tuple[float, float] = (25.0, 50.0)  # disc centre to disc centre
    canal_ap_mm: tuple[float, float] = (8.0, 25.0)
    # Stenosis reference thresholds (dural sac CSA; Hamanishi 1994 / Schizas 2010, approximate).
    sac_area_relative_mm2: float = 100.0
    sac_area_absolute_mm2: float = 75.0
    # Automatic anatomy (spinescope.anatomy). Mid-sagittal 2-D analysis on a regular mm grid.
    sag_resolution_mm: float = 1.0  # resampling grid of the mid-sagittal image (isotropic, fast)
    canal_half_width_mm: float = 5.0  # half-width of the core in the mid-sagittal plane (sac AP ~10-15 mm)
    canal_flank_offset_mm: float = 5.0  # flank windows sit this far beyond the core edge (outside the sac)
    canal_max_slope: float = 1.5  # max |dA/dS| of the canal centreline (mm/mm, ~56 deg from vertical)
    canal_min_contrast: float = 0.05  # ridge contrast (0..1 units) below which tissue is not CSF-like
    canal_min_length_mm: float = 25.0  # shortest canal track accepted as the spinal canal
    body_axis_offset_mm: tuple[float, float] = (18.0, 34.0)  # canal centre -> body centre, searched
    endplate_sigma_mm: float = 1.0  # smoothing of the dark endplate lines along the spine axis
    line_penalty: float = 0.2  # DP cost per endplate line (keeps spurious lines out of the chain)
    gap_penalty: float = 1.0  # DP cost of an interval that is neither a disc nor a vertebral body
    gap_mm: tuple[float, float] = (36.0, 72.0)  # interval length treated as a missed endplate
    endplate_search_mm: float = (
        5.0  # half-width of the first search for an endplate around its chain position
    )
    endplate_min_contrast: float = 0.12  # endplate darkness (fraction of the image range) per AP sample
    disc_margin_min_mm: float = 12.0  # AP extent of followed endplates needed to measure disc margins
    sacrum_min_visible_mm: float = 9.0  # body visible below the lowest disc needed to call it S1
    midline_axial_tol_mm: float = 4.0  # max disagreement between sagittal and axial midline estimates
    midline_search_mm: float = 10.0  # slices within this distance of the canal-ridge peak set the midline
    axial_canal_window_mm: float = 24.0  # axial window around the canal for the CSF-blob midline check
    conus_min_contrast: float = 0.08  # cord-vs-CSF contrast (0..1) on the canal centreline
    disc_match_mm: float = 6.0  # tolerance when matching discs between two sagittal series
    # Rendering
    max_png_side_px: int = 2000


DEFAULT = Config()
