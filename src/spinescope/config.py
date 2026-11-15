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
    # Slice geometry (io). Spacing is judged against the median slice spacing: deviations above the
    # relative tolerance make a series non-uniform, and a spacing of >= gap_factor x median is a gap
    # (missing slice). Judgement defaults: well above DICOM position rounding (1e-3 mm), well below a
    # missing slice (2 x spacing).
    slice_spacing_rel_tol: float = 0.1
    slice_gap_factor: float = 1.5
    # Rendering
    max_png_side_px: int = 2000


DEFAULT = Config()
