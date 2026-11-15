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
    # Reader consensus (report/merge.py). Judgement defaults, not literature: relative weight of a
    # reader's vote at low / medium / high self-reported confidence (used only with use_confidence=True).
    findings_confidence_weights: tuple[float, float, float] = (0.25, 0.5, 1.0)
    # Continuous findings (mm, deg, mm2) agree when their spread is within this fraction of the consensus
    # value (and at least 1 unit). Judgement default for inter-reader variability.
    findings_numeric_rel_tol: float = 0.2
    # Item kinds from different readers are grouped when their normalised slugs have a difflib ratio >= this.
    findings_kind_min_similarity: float = 0.85
    # Viewer reformats (planes computed in the browser from the best volume when a plane is missing).
    # Largest matrix side of a reformat; keeps browser resampling interactive (typical sources: 128-256).
    viewer_mpr_max_dim: int = 320
    # Fraction of the study's body box a source must cover to be used for a reformat (measured on the data).
    viewer_mpr_min_coverage: float = 0.05
    # Sources within this coverage of the best are tie-broken by contrast preference, then resolution.
    viewer_mpr_coverage_tie: float = 0.05
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
    # Automatic anatomy (spinescope.anatomy). Mid-sagittal 2-D analysis on a regular mm grid.
    sag_resolution_mm: float = 1.0  # resampling grid of the mid-sagittal image (isotropic, fast)
    canal_half_width_mm: float = 5.0  # half-width of the core in the mid-sagittal plane (sac AP ~10-15 mm)
    canal_flank_offset_mm: float = 5.0  # flank windows sit this far beyond the core edge (outside the sac)
    canal_max_slope: float = 1.5  # max |dA/dS| of the canal centreline (mm/mm, ~56 deg from vertical)
    canal_min_contrast: float = 0.05  # ridge contrast (0..1 units) below which tissue is not CSF-like
    canal_ap_fraction: tuple[float, float] = (0.45, 0.9)  # canal search band as a fraction of the AP field
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
    # Learned labeller (spinescope.anatomy.ml). The network was trained on a 1 mm grid (tools/train_labeller.py).
    ml_resolution_mm: float = 1.0  # training grid of the network input; mm per pixel
    ml_intensity_pct: tuple[float, float] = (0.5, 99.5)  # per-volume window percentiles (foreground voxels)
    ml_prob_threshold: float = 0.5  # class probability above which a pixel belongs to the class
    ml_disc_min_area_mm2: float = 30.0  # smallest disc component: disc_height_mm[0] (4) x canal_ap_mm[0] (8)
    ml_disc_min_aspect: float = (
        1.2  # discs are wider (AP) than tall: AP length / height, from anatomy (~35 x 9 mm)
    )
    ml_body_min_area_mm2: float = 150.0  # smallest midline vertebral body: about a fifth of a 25 x 30 mm body
    ml_disc_slice_window: int = (
        1  # discs also searched this many slices either side of the midline (~3-4 mm each)
    )


DEFAULT = Config()
