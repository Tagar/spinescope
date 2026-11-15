"""Test-time augmentation (TTA) of the learned labeller: uncertainty from the agreement of label variants (lab35).

The labeller is run again on variants of the same study, and the chain of each variant is compared with the base chain.
A disc that disappears, moves or changes its level name under a variant that a sagittal scan could plausibly have
produced is a weak point of the labelling. Variants (:data:`TTA_ORDER`; the first ``Config.ml_tta_variants`` are used):

* ``rot_pos`` / ``rot_neg``: in-plane rotation by +/- ``ml_tta_rotation_deg`` about the centre of every slice. The network
  runs on the rotated slices; its probability maps are rotated back to the base grid before any anatomy step, so every
  anchor stays in the study frame.
* ``gamma``: intensity contrast (power ``ml_tta_gamma`` of the normalised slices). No geometric change.
* ``plane_up`` / ``plane_down``: the analysed mid-sagittal plane moved one slice either way (re-centring). The network
  output is reused, so these cost no extra network pass.
* ``resolution``: the network on a grid of ``ml_tta_resolution_mm`` (the network was trained at 1 mm).
* ``lr_reverse``: the slice order reversed (the mid-sagittal symmetry; no anterior-posterior or superior-inferior flip).
  The labeller is per slice and the output is reversed back, so this variant equals the base by construction. It is
  kept to measure that (docs/labeling/lab35_uncertainty.md) and is outside the default count.

The aggregate (:func:`aggregate`) matches every variant disc to a base disc within half the smallest level spacing
(sagittal plane, as the duplicate-disc rule) and reports, per base disc: ``presence`` (share of variants that find it),
``name_agree`` (share that name it the same, the labeller's count from the bottom), ``spread_mm`` (largest 3-D centre
displacement) and ``mean_prob``; and per study the features of :data:`spinescope.anatomy.calibration.TTA_STUDY_FEATURES`.

Nothing here is calibrated by itself. The study features enter the numbering calibration only when a TTA calibration is
shipped and ``Config.ml_tta`` is on (:func:`spinescope.anatomy.pipeline.label_spine_details`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ...config import DEFAULT, Config
from ..anchors import Anchors, Disc
from ..calibration import TTA_STUDY_FEATURES
from .grid import network_input_volume, normalise, resample_slice
from .labeller import (
    MODEL_ASSET,
    SegmentedStack,
    adjacent_channels,
    input_channels,
    load_session,
    model_path,
    run_network,
    segment_stack,
)
from .postprocess import anchors_from_segmentation

#: Variants in the order they are used; ``Config.ml_tta_variants`` takes the first N.
TTA_ORDER: tuple[str, ...] = (
    "rot_pos",
    "rot_neg",
    "gamma",
    "plane_up",
    "plane_down",
    "resolution",
    "lr_reverse",
)
#: Variants that need no extra network pass.
NO_NETWORK_VARIANTS: frozenset[str] = frozenset({"plane_up", "plane_down"})


def variant_names(n: int) -> tuple[str, ...]:
    """The first ``n`` variants of :data:`TTA_ORDER` (0 disables the augmentation)."""
    if not 0 <= n <= len(TTA_ORDER):
        raise ValueError(f"ml_tta_variants must be between 0 and {len(TTA_ORDER)}, got {n}")
    return TTA_ORDER[:n]


@dataclass
class DiscTTA:
    """Agreement of one base disc (the labeller's disc before the numbering steps) across the variants."""

    label: str  # the base label (count from the bottom, before the numbering steps)
    center: list[float]  # base centre, LPS mm
    presence: float  # share of variants with a disc matched to this one
    name_agree: float  # share of variants where the matched disc carries the same label
    spread_mm: float  # largest 3-D displacement of the matched centre over the variants (0 when none matched)
    mean_prob: float  # mean disc probability over the base and the matched variant discs

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "center": [round(float(v), 3) for v in self.center],
            "presence": round(self.presence, 4),
            "name_agree": round(self.name_agree, 4),
            "spread_mm": round(self.spread_mm, 3),
            "mean_prob": round(self.mean_prob, 4),
        }


@dataclass
class TTAResult:
    """Variant agreement of one study: the study features and the per-base-disc records."""

    variants: list[str]
    study: dict[str, float]
    discs: list[DiscTTA] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variants": list(self.variants),
            "study": {k: round(float(v), 4) for k, v in self.study.items()},
            "discs": [d.to_dict() for d in self.discs],
        }


# ------------------------------------------------------------------ variant runs


def _session_for(cfg: Config) -> Any:
    """The labeller session the base run uses for ``cfg`` (packaged weights unless ``Config.ml_model`` names others)."""
    if cfg.ml_model != MODEL_ASSET:
        return load_session(model_path(cfg.ml_model), config=cfg)
    return load_session(config=cfg)


def _base_images(seg: SegmentedStack) -> NDArray[np.float32]:
    """The normalised network input of every slice on the base grids (as :func:`.labeller.segment_stack` builds it)."""
    src = network_input_volume(seg.stack)
    return np.stack([normalise(resample_slice(seg.stack, g, array=src), seg.window) for g in seg.grids])


def _warp(arr: NDArray[np.floating], coords: NDArray[np.float64]) -> NDArray[np.float32]:
    """Bilinear sampling of a 2-D array at fractional (row, col) coordinates; outside the array is 0."""
    out = ndimage.map_coordinates(
        np.asarray(arr, dtype=np.float32), coords, order=1, mode="constant", cval=0.0
    )
    return np.asarray(out, dtype=np.float32)


def _rotation_maps(shape: tuple[int, int], deg: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coordinates for an in-plane rotation of a slice about its centre (pixels; the grid is isotropic).

    Returns ``(inv, fwd)``. ``inv[:, p]`` is the base pixel shown at variant pixel ``p`` (to build the rotated input);
    ``fwd[:, q]`` is the variant pixel that shows base pixel ``q`` (to bring the probability maps back).
    """
    h, w = shape
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ii, jj = np.meshgrid(
        np.arange(h, dtype=np.float64) - cy, np.arange(w, dtype=np.float64) - cx, indexing="ij"
    )
    inv = np.stack([c * ii + s * jj + cy, -s * ii + c * jj + cx])
    fwd = np.stack([c * ii - s * jj + cy, s * ii + c * jj + cx])
    return inv, fwd


def _probs_of(seg: SegmentedStack, probs: NDArray[np.float32]) -> SegmentedStack:
    """A segmentation of the base stack and grids with other probability maps (no evidence carried over)."""
    return SegmentedStack(stack=seg.stack, grids=seg.grids, probs=probs.astype(np.float32), window=seg.window)


def run_variant(
    name: str, seg: SegmentedStack, cfg: Config, session: Any, base_imgs: NDArray[np.float32]
) -> Anchors:
    """Anchors of one variant, in the base study frame (LPS mm). ``name`` is one of :data:`TTA_ORDER`."""
    if name in ("rot_pos", "rot_neg"):
        deg = cfg.ml_tta_rotation_deg if name == "rot_pos" else -cfg.ml_tta_rotation_deg
        inv, fwd = _rotation_maps(base_imgs.shape[-2:], deg)
        rotated = np.stack([_warp(im, inv) for im in base_imgs])
        probs = run_network(session, adjacent_channels(rotated, input_channels(session)))
        back = np.stack([np.stack([_warp(p[c], fwd) for c in range(p.shape[0])]) for p in probs])
        return anchors_from_segmentation(_probs_of(seg, back), cfg)
    if name == "gamma":
        contrast = np.power(np.clip(base_imgs, 0.0, 1.0), cfg.ml_tta_gamma).astype(np.float32)
        probs = run_network(session, adjacent_channels(contrast, input_channels(session)))
        return anchors_from_segmentation(_probs_of(seg, probs), cfg)
    if name in NO_NETWORK_VARIANTS:
        return anchors_from_segmentation(seg, cfg, plane_shift=1 if name == "plane_up" else -1)
    if name == "resolution":
        cfg_r = dataclasses.replace(cfg, ml_resolution_mm=cfg.ml_tta_resolution_mm)
        seg_r = segment_stack(seg.stack, cfg_r, session=session)
        return anchors_from_segmentation(seg_r, cfg_r)
    if name == "lr_reverse":
        probs = run_network(session, adjacent_channels(base_imgs[::-1].copy(), input_channels(session)))
        return anchors_from_segmentation(_probs_of(seg, probs[::-1]), cfg)
    raise ValueError(f"unknown test-time variant {name!r}; choose from {', '.join(TTA_ORDER)}")


def run_variants(
    seg: SegmentedStack,
    cfg: Config = DEFAULT,
    *,
    names: Sequence[str] | None = None,
    session: Any = None,
) -> list[tuple[str, Anchors]]:
    """Anchors of every variant in ``names`` (default: the first ``cfg.ml_tta_variants`` of :data:`TTA_ORDER`)."""
    chosen = tuple(names) if names is not None else variant_names(cfg.ml_tta_variants)
    if not chosen:
        return []
    sess = session if session is not None else _session_for(cfg)
    base_imgs = _base_images(seg)
    return [(n, run_variant(n, seg, cfg, sess, base_imgs)) for n in chosen]


# ------------------------------------------------------------------ aggregation


def _nearest(base: Disc, var: Anchors, radius_mm: float) -> tuple[str, Disc] | None:
    """The variant disc nearest to ``base`` in the sagittal plane, if closer than ``radius_mm``."""
    best: tuple[str, Disc] | None = None
    best_d = radius_mm
    c = np.asarray(base.center, dtype=np.float64)[[1, 2]]
    for lab, d in var.discs.items():
        dist = float(np.linalg.norm(np.asarray(d.center, dtype=np.float64)[[1, 2]] - c))
        if dist < best_d:
            best, best_d = (lab, d), dist
    return best


def _neutral_study() -> dict[str, float]:
    """Study features when there is nothing to compare (no base disc or no variant): the values that never flag."""
    return {
        "tta_count_agree": 1.0,
        "tta_chain_agree": 1.0,
        "tta_presence_min": 1.0,
        "tta_name_min": 1.0,
        "tta_spread_max": 0.0,
        "tta_bottom_name": 1.0,
    }


def aggregate(base: Anchors, variants: Sequence[tuple[str, Anchors]], cfg: Config = DEFAULT) -> TTAResult:
    """Agreement of the variants with the base chain (features as in :data:`TTA_STUDY_FEATURES`).

    ``base`` is the labeller's anchors before any numbering step, so its names are the labeller's count from the bottom.
    """
    names = [n for n, _ in variants]
    if not base.discs or not variants:
        return TTAResult(names, _neutral_study(), [])
    # A variant disc is the same disc within half the smallest level spacing in the sagittal plane (the duplicate rule).
    radius = cfg.level_spacing_mm[0] / 2.0
    n_base = len(base.discs)
    items = sorted(base.discs.items(), key=lambda kv: float(kv[1].center[2]))  # inferior first
    presence: dict[str, list[float]] = {lab: [] for lab, _ in items}
    named: dict[str, list[float]] = {lab: [] for lab, _ in items}
    disp: dict[str, list[float]] = {lab: [] for lab, _ in items}
    probs: dict[str, list[float]] = {lab: [float(d.confidence)] for lab, d in items}
    count_ok: list[float] = []
    chain_ok: list[float] = []
    for _name, var in variants:
        count_ok.append(1.0 if len(var.discs) == n_base else 0.0)
        all_named = True
        for lab, d in items:
            hit = _nearest(d, var, radius)
            presence[lab].append(1.0 if hit else 0.0)
            same = hit is not None and hit[0] == lab
            named[lab].append(1.0 if same else 0.0)
            all_named = all_named and same
            if hit is not None:
                disp[lab].append(float(np.linalg.norm(np.asarray(hit[1].center) - np.asarray(d.center))))
                probs[lab].append(float(hit[1].confidence))
        chain_ok.append(1.0 if (all_named and count_ok[-1] == 1.0) else 0.0)
    discs = [
        DiscTTA(
            label=lab,
            center=[float(v) for v in d.center],
            presence=float(np.mean(presence[lab])),
            name_agree=float(np.mean(named[lab])),
            spread_mm=float(max(disp[lab])) if disp[lab] else 0.0,
            mean_prob=float(np.mean(probs[lab])),
        )
        for lab, d in items
    ]
    study = {
        "tta_count_agree": float(np.mean(count_ok)),
        "tta_chain_agree": float(np.mean(chain_ok)),
        "tta_presence_min": min(x.presence for x in discs),
        "tta_name_min": min(x.name_agree for x in discs),
        "tta_spread_max": max(x.spread_mm for x in discs),
        "tta_bottom_name": discs[0].name_agree,
    }
    assert set(study) == set(TTA_STUDY_FEATURES)
    return TTAResult(names, study, discs)


def analyse(
    base: Anchors, seg: SegmentedStack, cfg: Config = DEFAULT, *, session: Any = None
) -> TTAResult | None:
    """Run the configured variants on ``seg`` and aggregate them against ``base``; None when there is no disc."""
    if not base.discs:
        return None
    variants = run_variants(seg, cfg, session=session)
    return aggregate(base, variants, cfg)


def disc_records(result: TTAResult, final: Anchors) -> dict[str, dict[str, float]]:
    """Per-disc records keyed by the final labels of ``final`` (the nearest base disc to each final centre)."""
    out: dict[str, dict[str, float]] = {}
    if not result.discs:
        return out
    for lab, d in final.discs.items():
        c = np.asarray(d.center, dtype=np.float64)
        best = min(result.discs, key=lambda x: float(np.linalg.norm(np.asarray(x.center) - c)))
        if float(np.linalg.norm(np.asarray(best.center) - c)) > 0.5:
            continue  # the disc was not in the base chain (cannot happen after the numbering steps; kept safe)
        out[lab] = {
            "presence": round(best.presence, 4),
            "name_agree": round(best.name_agree, 4),
            "spread_mm": round(best.spread_mm, 3),
            "mean_prob": round(best.mean_prob, 4),
        }
    return out


__all__ = [
    "TTA_ORDER",
    "DiscTTA",
    "TTAResult",
    "aggregate",
    "analyse",
    "disc_records",
    "run_variant",
    "run_variants",
    "variant_names",
]
