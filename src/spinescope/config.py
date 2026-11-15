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
    # ---- conus locator (anatomy/conus.py), on the learned canal segmentation, 1 mm grid.
    # Cord core: samples within this half-width of the canal centreline (cord ~ 8 mm AP; core 4 mm wide).
    conus_core_half_mm: float = 2.0  # judgement: inside any lumbar cord
    # CSF flank: samples from this fraction of the canal half-width outwards (CSF rim, clear of the cord).
    conus_flank_inner_frac: float = 0.6  # judgement
    # Cord-likeness per row, window-relative: 1 - core/CSF (T2, STIR: CSF bright) or 1 - CSF/core (T1). Judgement:
    # cord is about half as bright as CSF on T2 and brighter than CSF on T1; cauda equina is thin and fragmented.
    conus_cord_contrast_min: float = 0.25
    # CSF must be in the bright half of the intensity window on T2/STIR (dark half on T1): the sac is the
    # brightest (T2) or darkest (T1) fluid. Plausibility gate against bone or body signal in the canal mask.
    conus_csf_min_rel: float = 0.5
    conus_smooth_mm: float = 3.0  # median filter along the canal (removes single-row flicker)
    conus_min_run_mm: float = 6.0  # shortest cord-like run that counts as cord
    conus_full_run_mm: float = 20.0  # run length that gives full length credit to the confidence
    conus_slice_halfwidth_mm: float = (
        4.0  # sagittal slices within this distance of the midline are analysed (~1 either side)
    )
    conus_agree_mm: float = (
        6.0  # slices whose tip lies within this of the median tip agree (about half a body)
    )
    conus_single_slice_factor: float = 0.8  # confidence factor when only one slice gives a tip
    # Confidence factor by numbering status (confirmed, probable, unconfirmed): a conus named on unconfirmed
    # level names is less certain than the same tip on a confirmed numbering.
    conus_status_factor: tuple[float, float, float] = (1.0, 0.9, 0.7)  # judgement
    # Low-lying conus: tip below the L2-L3 disc (Wilson & Prince, AJNR 1989; L3 is indeterminate there).
    conus_low_lying_disc: str = "L2-L3"
    conus_low_lying_min_confidence: float = 0.6  # below this the flag is worded as unconfirmed
    disc_match_mm: float = 6.0  # tolerance when matching discs between two sagittal series
    # Mid-sagittal plane by mirror symmetry (spinescope.anatomy.midline_robust). Mirror pairs of sagittal
    # slices at x0 +- d are compared; the symmetry axis is the midline. Lengths in mm.
    # Calibrated on SPIDER training only (see docs/labeling/lab05_midline.md).
    midline_sym_d_mm: tuple[float, float, float] = (3.0, 20.0, 2.0)  # mirror offsets d: min, max, step
    midline_grid_mm: float = 1.0  # candidate-plane grid and in-plane resampling step
    midline_hp_sigma_mm: float = 6.0  # scale of the high-pass (structure-only) similarity
    midline_patch_mm: float = 40.0  # side of the square patches that each vote for a local symmetry axis
    midline_patch_step_mm: float = 20.0  # spacing of patch centres (50 % overlap)
    midline_patch_min_prominence: float = 0.5  # patch votes below this prominence are ignored
    midline_fit_tukey_mm: float = 3.0  # Tukey scale of the robust x(z) fit
    midline_slice_weight_mm: float = 10.0  # Gaussian width of the per-slice midline weight
    midline_method: str = "symmetry"  # "symmetry" (midline_robust) or "ridge" (earlier canal-ridge method)
    # Slice geometry (io). Spacing is judged against the median slice spacing: deviations above the
    # relative tolerance make a series non-uniform, and a spacing of >= gap_factor x median is a gap
    # (missing slice). Judgement defaults: well above DICOM position rounding (1e-3 mm), well below a
    # missing slice (2 x spacing).
    slice_spacing_rel_tol: float = 0.1
    slice_gap_factor: float = 1.5

    # ---- analysis_core: discs (sagittal, disc frame; mm unless noted) ----
    # Lumbar disc transverse diameter ~40-46 mm (anatomy texts) -> half width used to pick mid-portion slices.
    disc_half_width_mm: float = 22.0
    # Slices within this fraction of the half width of the disc centre count as "mid-portion".
    disc_mid_fraction: float = 0.25
    # Fallback AP length when anchors lack anterior/posterior margins (adult lumbar disc ~30-40 mm).
    disc_ap_default_mm: float = 34.0
    # Columns excluded at the AP margins of the disc (margins are curved and partial-volumed).
    disc_inset_mm: float = 2.0
    # Sampling step of the straightened disc grid (discs analysis).
    disc_grid_mm: float = 0.5
    # Discs analysis endplate search along the normal from the disc centre: up to half the maximum disc
    # height (18 mm / 2) plus margin for the cortex.
    disc_endplate_search_mm: float = 13.0
    # Discs analysis: dark endplate line must be this much darker than the disc next to it (relative).
    disc_endplate_min_contrast: float = 0.15
    # Endplate search along the normal from the disc centre: from half the minimum disc height (4 mm / 2)
    # to half the maximum disc height (18 mm / 2) plus margin for the cortex.
    endplate_min_h_mm: float = 2.0
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
    # Packaged weights used by label_spine_ml. labeller_v1.onnx (4 classes) is the shipped model. A 5-class
    # candidate with a sacrum channel (labeller_v2, lab07) was not shipped: it scored worse on SPIDER validation
    # (docs/model_card.md). Without a sacrum channel, numbering confirmation is skipped.
    ml_model: str = "labeller_v1.onnx"
    # Sacrum class area below the lowest disc that confirms an S1 body there (about 15 x 10 mm of bone).
    ml_sacrum_min_area_mm2: float = 150.0
    # Field of view needed below the lowest disc to show an S1 body (the smallest pseudo-label box in
    # tools/train_labeller.py is 8 mm; less than this and the sacrum is not in view).
    ml_fov_below_disc_min_mm: float = 8.0

    # ---- anatomy: label verification (verify.py, Anchors corrections) ----
    # Confidence written to numbering['confidence'] after a reviewed correction. Judgement defaults, not
    # calibrated: a human confirmation is trusted more than an AI reader's answer on a 2-D packet.
    review_confidence_human: float = 0.95
    review_confidence_ai: float = 0.85
    # Numbering confidence when the reader answers 'unsure' (no structural change is made).
    review_unsure_confidence: float = 0.3
    # Penalty per reader-named disc that could not be placed (no neighbours to place it between).
    review_unplaced_penalty: float = 0.1
    # Confidence of a disc the reader asked for and that was placed by interpolation or extrapolation
    # from its neighbours (its position is a guess, not a measurement).
    placed_disc_confidence: float = 0.3
    # Sacrum zoom of the verification packet: centre this far inferior to the lowest disc, crop side (mm).
    verify_sacrum_offset_mm: float = 30.0
    verify_sacrum_fov_mm: float = 100.0

    # ---- viewer: label review (viewer/assets/labels.js) ----
    # Disc confidence colour bands: at or above ``high`` green, at or above ``low`` amber, else red.
    # Numbering confidence below ``low`` raises the "labels unconfirmed" banner (the same 0.5 that
    # anatomy.pipeline uses for "numbering confidence is low"). Judgement defaults, not calibrated.
    viewer_conf_high: float = 0.8
    viewer_conf_low: float = 0.5
    # Disc plausibility. SPIDER training truth: normals tilt at most ~60 deg from the superior-inferior axis
    # (99.9th percentile 57 deg). A component whose normal tilts more than this is a sagittal strip, not a disc.
    ml_disc_max_tilt_deg: float = 70.0
    # 3-D centroid and normal: discs are averaged over the sagittal slices within this distance of the
    # mid-sagittal plane. SPIDER truth is the 3-D voxel centroid, and the mid-plane centroid sits about 1.3 mm
    # anterior of it (lab09 training split). 27 mm = eight 3.34 mm SPIDER slices; the error plateaus there.
    ml_disc_3d_half_width_mm: float = 27.0
    ml_disc_3d_ap_half_mm: float = 22.0  # AP half window around the mid-plane centre (lumbar AP ~35 mm)
    # Use the 3-D normal too. Off: on the independent Sudirman axial-slab normals it is worse (3.3 vs 2.5 deg
    # median), while SPIDER's own normals are 3-D masks, so the SPIDER gain is partly definitional.
    ml_disc_normal_3d: bool = False
    # ---- anatomy: applicability (spinescope.anatomy.ood: when NOT to label). Judgement defaults from anatomy
    # texts, checked on SPIDER training only (docs/labeling/lab12_ood.md). Lengths in mm, unitless otherwise.
    # Lumbar discs are ~8-12 mm high; thoracic and cervical discs ~4-6 mm.
    ood_lumbar_disc_min_mm: float = 7.0
    # Disc-to-disc spacing (centre to centre): cervical ~18-21 mm (body ~14 + disc ~5), thoracic ~26-30 mm,
    # lumbar ~34-40 mm (body ~26-30 + disc ~8-12).
    ood_cervical_spacing_max_mm: float = 22.0
    ood_lumbar_spacing_min_mm: float = 30.0
    # A sagittal stack needs at least this many slices and this z extent to be usable for labelling.
    # Bone below the lowest disc (mm, along z) needed to call the sacrum in view: one lumbar body (~25-30 mm) plus the
    # first sacral segment. A 25-30 mm body is the L5 body, not S1, so less than this does not confirm the numbering.
    ood_sacrum_min_mm: float = 45.0
    ood_min_sag_slices: int = 3
    ood_min_sag_extent_mm: float = 40.0
    # Field of view must extend this far beyond the outermost disc for another body to be in view
    # (the shortest vertebral body, config.body_height_mm[0]).
    ood_fov_margin_mm: float = 18.0
    # Off-distribution cue (contrast, field strength, voxel shape, output entropy): multiplies the
    # confidences by this factor per cue, never below the floor. Judgement default, not calibrated.
    ood_offdist_factor: float = 0.6
    ood_factor_floor: float = 0.3
    # Thoracic discs above T10 (T10-T11 and higher) without a landmark named in the series descriptions
    # are counted from the sacrum only: a missed disc anywhere below shifts them all. Confidence cap.
    ood_thoracic_cap: float = 0.3
    # Field strengths below this (tesla) have no training data: SPIDER is 1.5 T and 3 T.
    ood_low_field_t: float = 1.0
    # Voxels with all three sides at or below this size, and slice thickness within a factor 1.25 of the
    # in-plane pixel size, are treated as 3-D isotropic (SPIDER T2 SPACE is 0.47 x 0.47 x 0.9 mm, not isotropic).
    ood_isotropic_mm: float = 1.2
    # Foreground pixels: network probability of any anatomy class above this. Normalised entropy of the
    # class distribution over those pixels above ood_entropy_max means the network is unsure (0 = certain).
    # Calibrated on SPIDER training: the 99th percentile of real studies is 0.17 (see docs/labeling/lab12_ood.md).
    ood_fg_prob: float = 0.3
    ood_entropy_max: float = 0.2
    # A vertebral body lying between two adjacent discs longer than this (mm) means a disc between them was missed.
    # Calibrated on SPIDER training (mid-sagittal network bodies: median 30 mm, 99th percentile 41 mm, 2-body
    # merges ~55-65 mm): see docs/labeling/lab12_ood.md.
    ood_long_body_mm: float = 45.0
    # Labelling needs at least this many discs in view (one disc is not a chain to count).
    ood_min_discs: int = 2
    # Canal is searched within this distance (mm) of each disc level on the mid-sagittal slice; a spine has canal at
    # every level, so the fraction of levels with canal must reach ood_canal_fraction_min.
    ood_canal_tol_mm: float = 10.0
    ood_canal_fraction_min: float = 0.5

    # ---- vertebral geometry (anatomy/vertebrae.py, anatomy/canal_ml.py; lab15) ----
    # Endplate lines are fitted on the central part of each body: this fraction of its AP depth is cut at each end,
    # so the rounded corners do not bias the line. Judgement default, checked on SPIDER training.
    vert_core_fraction: float = 0.2
    # A body narrower than this in AP on the mid-sagittal plane is not measured (a fragment, not a vertebral body).
    # Lumbar bodies are about 25-35 mm AP (anatomy; the disc AP length is ~35 mm); 15 mm rejects fragments.
    vert_min_ap_mm: float = 15.0
    # 3-D body width needs slices spanning at least this far (lateral extent cannot be seen on fewer slices).
    vert_width_min_span_mm: float = 20.0
    # 3-D width uses the slices holding at least this fraction of the body's largest slice count. Calibrated on SPIDER
    # training (60 volumes, lab15): 0.0 (all slices) gave the lowest median absolute width error, 6.7 mm, against 9.6 mm
    # at 0.25 and 13.2 mm at 0.5. Width stays low-confidence (p90 about 28 mm), see docs/labeling/lab15_vertebrae.md.
    vert_width_slice_fraction: float = 0.0
    # Canal centreline: z bins (mm), each needs canal pixels from at least two slices.
    canal_z_bin_mm: float = 5.0
    canal_min_bin_slices: int = 2
    # A slice joins the bin's centre only if it holds this fraction of the bin's largest slice count (lateral strays).
    canal_bin_slice_fraction: float = 0.3
    # AP diameter at a level: mid-sagittal canal rows within this distance (mm) of the level are used (median chord).
    canal_level_band_mm: float = 2.0
    # Calibrated on SPIDER training (lab15): the mid-sagittal AP chord of the labeller's canal is a median 0.5 mm wider
    # than the truth chord (one-pixel boundary effect); subtracting this removes it (median abs error 1.0 -> 0.5 mm).
    canal_ap_bias_mm: float = 0.5
    # The centreline is reported this far (mm) beyond the lowest and highest disc, within the canal's extent.
    canal_zone_margin_mm: float = 10.0
    # Posterior elements (approximations, no training labels exist for them; see docs/labeling/lab15_vertebrae.md).
    # Spinous tip: search from this gap behind the posterior canal edge out to the search depth, for the posterior-most
    # intensity edge at least spinous_edge_per_mm (window-normalised intensity per mm) strong.
    spinous_gap_mm: float = 3.0
    spinous_search_mm: float = 45.0
    spinous_smooth_mm: float = 1.0
    spinous_edge_per_mm: float = 0.03

    # Conus locator is experimental (no ground truth; a 60-case AI visual audit could not verify most tips):
    # cap its reported confidence until it is calibrated.
    conus_max_confidence: float = 0.5


DEFAULT = Config()
