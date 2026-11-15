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
    # Rendering
    max_png_side_px: int = 2000
    # Viewer reformats (planes computed in the browser from the best volume when a plane is missing).
    # Largest matrix side of a reformat; keeps browser resampling interactive (typical sources: 128-256).
    viewer_mpr_max_dim: int = 320
    # Fraction of the study's body box a source must cover to be used for a reformat (measured on the data).
    viewer_mpr_min_coverage: float = 0.05
    # Sources within this coverage of the best are tie-broken by contrast preference, then resolution.
    viewer_mpr_coverage_tie: float = 0.05


DEFAULT = Config()
