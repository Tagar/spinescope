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
    # Rendering (spinescope.render). Display choices only; no measurement depends on them.
    max_png_side_px: int = 2000  # PNG side limit; sheets are split to stay below it
    # Series-wide display window: percentiles of foreground voxels (legacy prototype default, 0.5-99.7).
    window_pct: tuple[float, float] = (0.5, 99.7)
    # Background = exact zeros and voxels below this fraction of the 99.5th percentile (zero-filled air,
    # noise floor). Dropped before percentiles so zero-filled backgrounds do not set the window.
    background_frac: float = 0.02
    # Series with fewer foreground voxels than this fraction fall back to all non-zero voxels.
    min_foreground_frac: float = 0.01
    # Per-contrast presets: (contrast, name, lo_pct, hi_pct) on foreground voxels. Heuristic percentiles
    # chosen by eye on a 1.5 T spine (disc/CSF on T2, marrow on T1, oedema on STIR). Not calibrated.
    window_presets: tuple[tuple[str, str, float, float], ...] = (
        ("t2", "disc/CSF", 5.0, 98.0),
        ("t2", "soft tissue", 1.0, 99.5),
        ("t1", "marrow", 30.0, 99.0),
        ("t1", "global", 1.0, 99.5),
        ("stir", "edema", 60.0, 99.7),
        ("stir", "global", 1.0, 99.5),
    )
    # Label font height as a fraction of the shorter side of the image it labels.
    font_frac: float = 0.06
    title_frac: float = 0.045  # title band font, as a fraction of the shorter side (shrinks/wraps to fit)
    # Smallest shorter side of a rendered crop or reformat; coarser native grids are upscaled to this.
    display_min_frac: float = 0.125
    # Spacing of sparse native pixel-coordinate labels and of ruler ticks, mm. ~ a vertebral body
    # height divided in two: readable without covering the anatomy.
    grid_mm: float = 10.0
    zoom_mm: float = 60.0  # default side of a zoom crop / linked view (paraspinal to canal at lumbar scale)
    disc_plane_mm: float = 60.0  # default side of a disc-plane reformat
    # Canal centre behind the posterior disc margin for curved coronal reformats. Measured on one scan
    # in the legacy prototype (CSF half-max centre); to be re-validated on public data (SPIDER).
    thecal_sac_offset_mm: float = 3.5
    sacral_canal_offset_mm: float = 10.0  # below the thecal sac the sacral canal lies further posterior
    # Normalised T2/T1 ratio: reference ROI radius around a vertebral body centre and the lower limit on
    # the normalised T1 below which the ratio is not shown (amplified noise). Legacy prototype values.
    ratio_ref_radius_mm: float = 6.0
    ratio_min_t1: float = 0.25
    ratio_display_max: float = 2.0  # colour scale 0..this for the ratio image (1 = reference; display only)
    # STIR-on-T1 fusion: tint where the STIR, normalised to its window, is at least this fraction.
    fusion_threshold: float = 0.5
    # Denoising: scikit-image non-local means recommends h >= 3 sigma for Gaussian noise. Patch sizes
    # are algorithm settings (scikit-image defaults), not tuned to a study.
    nlm_h_per_sigma: float = 3.0
    nlm_patch_size: int = 5
    nlm_patch_distance: int = 6
    bilateral_sigma_color: float = 2.0  # in units of the noise sigma (scikit-image bilateral filter)
    # Air (background) for denoising and bias correction: smoothed voxels below this fraction of the 99.5th
    # percentile, connected to the image border. Air is Rician noise (~1.25 sigma), so 5% is well above it.
    air_frac: float = 0.05
    bias_clip: tuple[float, float] = (0.25, 4.0)  # clip the estimated surface-coil field to this range
    # Smoothing scales as fractions of the image side (resolution-independent), e.g. bias field.
    smooth_frac: float = 0.006
    bias_sigma_frac: float = 0.2  # surface-coil field is smooth over about a fifth of the field of view


DEFAULT = Config()
