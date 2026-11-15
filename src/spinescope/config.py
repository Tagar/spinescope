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

    # ---- analysis_core: discs (sagittal, disc frame; mm unless noted) ----
    # Lumbar disc transverse diameter ~40-46 mm (anatomy texts) -> half width used to pick mid-portion slices.
    disc_half_width_mm: float = 22.0
    # Slices within this fraction of the half width of the disc centre count as "mid-portion".
    disc_mid_fraction: float = 0.25
    # Fallback AP length when anchors lack anterior/posterior margins (adult lumbar disc ~30-40 mm).
    disc_ap_default_mm: float = 34.0
    # Columns excluded at the AP margins of the disc (margins are curved and partial-volumed).
    disc_inset_mm: float = 2.0
    # Sampling step of the straightened disc grid.
    grid_mm: float = 0.5
    # Endplate search along the normal from the disc centre: from half the minimum disc height (4 mm / 2)
    # to half the maximum disc height (18 mm / 2) plus margin for the cortex.
    endplate_min_h_mm: float = 2.0
    endplate_search_mm: float = 13.0
    # Dark endplate line must be this much darker than the disc next to it (relative contrast).
    # Conservative; cortical endplates are near signal-void against disc/marrow on T1, T2 and STIR.
    endplate_min_contrast: float = 0.15
    # Reach (mm) used to find the bright plateau on each side of the dark endplate line.
    endplate_plateau_mm: float = 3.0
    # Per-mm penalty for endplate steps in the dynamic-programming trace (normalised intensity units).
    endplate_smooth: float = 0.2
    # Nucleus = central box with half sizes 0.30 of the disc AP length and 0.35 of its local height.
    # Anatomy: the nucleus occupies the middle of the disc; exact fractions are a convention.
    nucleus_ap_fraction: float = 0.30
    nucleus_h_fraction: float = 0.35
    # Annulus voxels within this distance of an endplate are excluded (partial volume with cortex).
    endplate_rim_mm: float = 1.0
    # Posterior annulus starts at this fraction of the AP length (HIZ search zone).
    posterior_annulus_fraction: float = 0.6
    # Thecal sac (CSF) reference: search window posterior to the posterior disc margin, mm.
    csf_search_mm: tuple[float, float] = (3.0, 25.0)
    csf_half_height_mm: float = 6.0
    # HIZ: posterior annulus brighter than this multiple of the mean nucleus signal.
    hiz_nucleus_ratio: float = 1.0
    # HIZ candidates within this distance of CSF-bright voxels are partial volume with the sac; dropped.
    hiz_csf_exclusion_mm: float = 1.5
    hiz_area_mm2: tuple[float, float] = (0.5, 20.0)
    # Pfirrmann suggestion ladders (grade 1..4 thresholds; grade 5 below the last). UNCALIBRATED until
    # validated on public gradings (see docs/validation.md). Brightness: nucleus/CSF; homogeneity: nucleus CV
    # (upper limits); contrast: nucleus-annulus contrast (lower limits); height: mean height / study median.
    pfirrmann_brightness: tuple[float, float, float, float] = (0.85, 0.70, 0.55, 0.40)
    pfirrmann_cv: tuple[float, float, float, float] = (0.10, 0.16, 0.22, 0.30)
    pfirrmann_contrast: tuple[float, float, float, float] = (0.50, 0.40, 0.30, 0.20)
    pfirrmann_height: tuple[float, float, float, float] = (0.90, 0.75, 0.60, 0.45)

    # ---- analysis_core: endplates / Modic / marrow ----
    # Marrow band next to an endplate: 3-8 mm inside the body, clear of cortex and partial volume.
    # Source: prototype (lspine endplate_marrow.py) convention; subchondral marrow is typically <10 mm.
    endplate_band_mm: tuple[float, float] = (3.0, 8.0)
    # AP inset from the body corners (cortex, ALL/PLL, vessels) and posterior inset (canal).
    corner_inset_mm: float = 3.0
    posterior_inset_mm: float = 5.0
    # Mid-body reference lies at least this far from both endplates.
    midbody_min_mm: float = 8.0
    # Lesion search keeps this distance from both endplates (beyond the Modic band, plus margin).
    lesion_endplate_mm: float = 10.0
    min_region_mm2: float = 6.0
    # Ratios to the own mid-body (relative units). Modic typing thresholds, approximate and
    # uncalibrated; to be validated on public Modic gradings.
    t1_low: float = 0.85
    t1_high: float = 1.15
    t2_low: float = 0.90
    t2_high: float = 1.10
    stir_high: float = 1.20
    # Cortical bone is near signal-void on T1: ratio to marrow below this marks cortex (excluded from lesions).
    cortex_t1_ratio: float = 0.30
    # Pixel smoothing (mm) applied before pixel-level classification (noise control).
    ratio_smooth_mm: float = 1.0
    # Healthiest body: bodies with normal endplates on both sides form the study reference.
    # Side classification: a slice is "central" when within half a slice spacing of the midline.
    side_central_spacing_fraction: float = 0.5
    # Rendering
    max_png_side_px: int = 2000


DEFAULT = Config()
