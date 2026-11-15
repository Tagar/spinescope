"""Axial evidence for the sagittal labeller: disc position support, missed-disc candidates and the slab vote input.

Most clinical lumbar protocols add axial slabs angled to each disc (usually T1 or T2, 3 to 6 slabs, one per level).
Each slab is a thin stack whose normal is close to the disc normal, so a disc that is really there appears as a
round, intermediate-signal region in the slab, while the labeller's disc position is an independent measurement.
This module uses that in three ways:

* **Disc support** (:class:`DiscAxial`): a predicted disc covered by a slab (normal within
  ``AxialConfig.slab_match_angle_deg`` and centre inside the slab block) is scored by a disc-likeness log
  likelihood ratio (LLR) at its centre. A covered disc with an LLR below the calibrated threshold is flagged
  ``unsupported`` (possible false disc). A disc with no covering slab is not scored: no evidence either way.
* **Missed-disc candidates** (:class:`MissedCandidate`): disc-like peaks in a slab slice with no predicted disc
  within ``AxialConfig.missed_mm`` in that plane. Each one is a warning (a possible missed disc).
* **Slab matches** (:class:`SlabMatch`): which predicted disc each slab is angled to, and the slab's rank from the
  bottom. :func:`spinescope.anatomy.numbering_checks.axial_slab_vote` turns the lowest supported match into a
  numbering vote (the lowest slab is usually L5-S1).

Disc likeness is a two-class Gaussian naive Bayes on three ellipse features (mean, coefficient of variation and
ring contrast of the normalised slice intensity inside a disc-sized ellipse), fitted on the axial T1 slices of
the Sudirman dataset by :mod:`spinescope.anatomy._calibrate_axial` and shipped as ``axial_v1.json`` next to this
file. Only T1 slabs are calibrated (``AxialConfig.contrasts``); T2 slabs and full axial stacks are not used.

What is NOT here (no data to calibrate or test it on): iliac crest level, L5 transverse process against sacral
ala (Castellvi), iliolumbar ligament and sacral promontory. Sudirman slabs cover only the three annotated discs,
and the phantom has none of that anatomy. Those cues remain "not evaluated" in ``numbering_checks``.

Every function abstains when the study has no usable axial T1 slab: the report has ``available=False`` and a
warning, and no vote is produced.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage, signal

from ..geometry import Stack
from ..study import Study
from .anchors import Anchors, Disc

log = logging.getLogger(__name__)

#: Shipped calibration (Sudirman axial T1, see _calibrate_axial.py). Missing file: the cue abstains.
MODEL_PATH = Path(__file__).with_name("axial_v1.json")
#: Feature order of the naive Bayes model. Stable: stored in the JSON, so do not reorder.
FEATURES: tuple[str, ...] = ("mean", "cv", "ring")


@dataclass(frozen=True)
class AxialConfig:
    """Axial cue settings. Ellipse sizes are judgement defaults; thresholds come from the calibration file."""

    #: Sequences of the axial slabs that are used. T1 is the calibrated one (Sudirman labels are on T1).
    contrasts: tuple[str, ...] = ("t1",)
    #: A slab covers a disc when its normal is within this angle of the disc normal (same as numbering_checks).
    slab_match_angle_deg: float = 12.0
    #: Disc-sized ellipse semi-axes in the disc plane, mm. Judgement: about half an adult lumbar disc's width
    #: (about 40 mm) and AP depth (about 32 mm). The calibration adjusts the thresholds, not these sizes.
    ellipse_lr_mm: float = 20.0
    ellipse_ap_mm: float = 16.0
    #: Ring used for the contrast feature, as a multiple of the ellipse semi-axes (inner ellipse excluded).
    ring_outer: float = 1.5
    #: A disc-like peak within this in-plane distance of a predicted disc is the same disc (the scorer's 12 mm).
    missed_mm: float = 12.0


DEFAULT_AXIAL = AxialConfig()


@dataclass(frozen=True)
class AxialModel:
    """Calibrated naive Bayes model and thresholds (loaded from ``axial_v1.json``)."""

    mean_disc: tuple[float, ...]
    var_disc: tuple[float, ...]
    mean_non: tuple[float, ...]
    var_non: tuple[float, ...]
    support_llr: float  # a covered disc with LLR below this is unsupported
    missed_llr: float  # a disc-like peak with LLR at or above this is a missed-disc candidate
    version: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscAxial:
    """Axial support of one predicted disc."""

    label: str
    covered: bool  # a T1 slab covers the disc (normal and block)
    slab: str | None = None
    llr: float | None = None  # disc-likeness log LR at the centre (None when not covered)
    supported: bool | None = None  # None when not covered (no evidence either way)
    evidence: str = ""


@dataclass
class SlabMatch:
    """Which predicted disc a slab is angled to. ``rank`` 0 is the lowest slab of the study."""

    slab: str
    rank: int
    disc: str | None  # predicted disc label, None when no predicted disc is covered
    llr: float | None = None
    supported: bool | None = None  # None when no disc is covered


@dataclass
class MissedCandidate:
    """A disc-like peak in an axial slab with no predicted disc in that plane."""

    slab: str
    lps: tuple[float, float, float]  # rounded, LPS mm
    llr: float
    nearest_disc_mm: float | None  # in-plane distance to the nearest predicted disc of the slab (None: none)


@dataclass
class AxialReport:
    """Everything the axial cue computed for one study. Empty and ``available=False`` when it abstained."""

    available: bool
    contrast: str | None = None
    discs: dict[str, DiscAxial] = field(default_factory=dict)
    slabs: list[SlabMatch] = field(default_factory=list)
    missed: list[MissedCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def unsupported(self) -> list[str]:
        """Predicted discs covered by a slab with no disc-like appearance there (possible false discs)."""
        return [lab for lab, d in self.discs.items() if d.supported is False]

    @property
    def covered(self) -> list[str]:
        return [lab for lab, d in self.discs.items() if d.covered]

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary (no paths, no image data)."""
        return {
            "available": self.available,
            "contrast": self.contrast,
            "discs": {
                lab: {
                    "covered": d.covered,
                    "slab": d.slab,
                    "llr": None if d.llr is None else round(float(d.llr), 3),
                    "supported": d.supported,
                }
                for lab, d in self.discs.items()
            },
            "slabs": [
                {
                    "slab": s.slab,
                    "rank": s.rank,
                    "disc": s.disc,
                    "supported": s.supported,
                }
                for s in self.slabs
            ],
            "missed": [
                {"lps": [round(float(x), 1) for x in m.lps], "llr": round(float(m.llr), 3)}
                for m in self.missed
            ],
            "warnings": list(self.warnings),
        }


# ------------------------------------------------------------------ model


@lru_cache(maxsize=4)
def _load_model(path: str) -> AxialModel | None:
    p = Path(path)
    if not p.is_file():
        log.warning("axial calibration %s not found: axial cue abstains", p)
        return None
    raw = json.loads(p.read_text())
    if list(raw["features"]) != list(FEATURES):
        raise ValueError(f"axial model features {raw['features']} do not match {list(FEATURES)}")
    return AxialModel(
        mean_disc=tuple(float(x) for x in raw["disc"]["mean"]),
        var_disc=tuple(float(x) for x in raw["disc"]["var"]),
        mean_non=tuple(float(x) for x in raw["non"]["mean"]),
        var_non=tuple(float(x) for x in raw["non"]["var"]),
        support_llr=float(raw["support_llr"]),
        missed_llr=float(raw["missed_llr"]),
        version=str(raw.get("version", "")),
        provenance=dict(raw.get("provenance", {})),
    )


def load_axial_model(path: str | Path | None = None) -> AxialModel | None:
    """The calibrated axial model, or None when the calibration file is missing (the cue then abstains)."""
    return _load_model(str(Path(path) if path is not None else MODEL_PATH))


# --------------------------------------------------------------- features


def _in_plane_basis(stack: Stack) -> tuple[np.ndarray, np.ndarray]:
    """Left-right (LPS x) and anterior-posterior (LPS y) unit vectors projected into the slab plane."""
    n = np.asarray(stack.normal, dtype=np.float64)
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def _kernels(stack: Stack, cfg: AxialConfig) -> tuple[np.ndarray, np.ndarray]:
    """Inner-ellipse and ring kernels (each normalised to sum 1) on the slab's pixel grid, mm-accurate."""
    e1, e2 = _in_plane_basis(stack)
    # Displacement of one pixel step in the row and column direction, projected on the in-plane basis (mm).
    dr_vec = stack.row_mm * np.asarray(stack.col_dir, dtype=np.float64)
    dc_vec = stack.col_mm * np.asarray(stack.row_dir, dtype=np.float64)
    ring_mm = cfg.ring_outer * max(cfg.ellipse_lr_mm, cfg.ellipse_ap_mm)
    half_r = math.ceil(ring_mm / max(stack.row_mm, 1e-6))
    half_c = math.ceil(ring_mm / max(stack.col_mm, 1e-6))
    rr, cc = np.meshgrid(np.arange(-half_r, half_r + 1), np.arange(-half_c, half_c + 1), indexing="ij")
    lr = rr * float(dr_vec @ e1) + cc * float(dc_vec @ e1)
    ap = rr * float(dr_vec @ e2) + cc * float(dc_vec @ e2)
    rad2 = (lr / cfg.ellipse_lr_mm) ** 2 + (ap / cfg.ellipse_ap_mm) ** 2
    inner = (rad2 <= 1.0).astype(np.float64)
    ring = ((rad2 > 1.0) & (rad2 <= cfg.ring_outer**2)).astype(np.float64)
    return inner / max(inner.sum(), 1.0), ring / max(ring.sum(), 1.0)


def _normalised(stack: Stack, k: int) -> np.ndarray:
    img = np.asarray(stack.vol[k], dtype=np.float64)
    pos = img[img > 0]
    scale = float(np.percentile(pos, 95)) if pos.size else 1.0
    return img / max(scale, 1e-9)


def feature_maps(stack: Stack, k: int, cfg: AxialConfig = DEFAULT_AXIAL) -> np.ndarray:
    """Per-pixel ellipse features of slice ``k``, shape (3, rows, cols): mean, cv, ring contrast (see FEATURES)."""
    img = _normalised(stack, k)
    inner, ring = _kernels(stack, cfg)
    m = signal.fftconvolve(img, inner, mode="same")
    m2 = signal.fftconvolve(img * img, inner, mode="same")
    var = np.clip(m2 - m * m, 0.0, None)
    cv = np.sqrt(var) / np.maximum(np.abs(m), 1e-6)
    ring_mean = signal.fftconvolve(img, ring, mode="same")
    return np.stack([m, cv, m - ring_mean])


def llr_map(stack: Stack, k: int, model: AxialModel, cfg: AxialConfig = DEFAULT_AXIAL) -> np.ndarray:
    """Disc-likeness log likelihood ratio per pixel of slice ``k`` (naive Bayes, equal priors)."""
    f = feature_maps(stack, k, cfg)
    out = np.zeros(f.shape[1:], dtype=np.float64)
    for j in range(len(FEATURES)):
        out += _log_normal(f[j], model.mean_disc[j], model.var_disc[j])
        out -= _log_normal(f[j], model.mean_non[j], model.var_non[j])
    return out


def _log_normal(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    v = max(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * v) + (x - mu) ** 2 / v)


# ------------------------------------------------------------- coverage


def _contrast_ok(stack: Stack, cfg: AxialConfig) -> bool:
    role = stack.meta.get("role", {}) if isinstance(stack.meta, dict) else {}
    return str(role.get("contrast", "")) in cfg.contrasts


def axial_slabs(study: Study, cfg: AxialConfig = DEFAULT_AXIAL) -> list[Stack]:
    """Axial slabs of the configured contrasts, lowest first (rank 0 = lowest)."""
    slabs = [s for s in study.slabs("ax") if _contrast_ok(s, cfg)]
    return sorted(slabs, key=lambda s: float(s.ipp[:, 2].mean()))


def _covering_slab(disc: Disc, slabs: list[Stack], cfg: AxialConfig) -> tuple[Stack, int] | None:
    """The slab (and slice) covering ``disc``: normal within the angle and centre inside the block."""
    n = np.asarray(disc.normal, dtype=np.float64)
    c = np.asarray(disc.center, dtype=np.float64)
    margin = (disc.height_mm or 8.0) / 2.0
    best: tuple[float, Stack] | None = None
    for s in slabs:
        ang = math.degrees(math.acos(min(1.0, abs(float(n @ s.normal)))))
        if ang > cfg.slab_match_angle_deg or not s.contains(c, slab_margin_mm=margin):
            continue
        off = abs(float(c @ s.normal) - float(s.positions.mean()))
        if best is None or off < best[0]:
            best = (off, s)
    if best is None:
        return None
    s = best[1]
    k, _ = s.nearest_slice(c)
    return s, k


def _sample(stack: Stack, k: int, xyz: np.ndarray, field_map: np.ndarray) -> float:
    _, r, c = stack.lps_to_pix(xyz)
    rr = int(np.clip(round(r), 0, field_map.shape[0] - 1))
    cc = int(np.clip(round(c), 0, field_map.shape[1] - 1))
    return float(field_map[rr, cc])


# ------------------------------------------------------------ the report


def assess_axial(
    study: Study,
    anchors: Anchors,
    cfg: AxialConfig = DEFAULT_AXIAL,
    model: AxialModel | None = None,
) -> AxialReport:
    """Axial support of every predicted disc, slab matches and missed-disc candidates for ``study``.

    Abstains (``available=False``, one warning) when there is no axial slab of the configured contrast or no
    calibration file. Never raises for missing inputs.
    """
    mdl = model if model is not None else load_axial_model()
    if mdl is None:
        return AxialReport(available=False, warnings=["axial calibration missing: axial cue not computed"])
    slabs = axial_slabs(study, cfg)
    if not slabs:
        return AxialReport(
            available=False,
            warnings=[f"no axial slab of contrast {'/'.join(cfg.contrasts)}: axial cue not available"],
        )
    rep = AxialReport(available=True, contrast="/".join(cfg.contrasts))
    if not anchors.discs:
        rep.warnings.append("no predicted discs: axial support not computed")
        return rep

    maps: dict[tuple[str, int], np.ndarray] = {}

    def _map(s: Stack, k: int) -> np.ndarray:
        key = (s.key, k)
        if key not in maps:
            maps[key] = llr_map(s, k, mdl, cfg)
        return maps[key]

    # Predicted discs: coverage, LLR at the centre, support flag.
    for lab, d in anchors.discs.items():
        hit = _covering_slab(d, slabs, cfg)
        if hit is None:
            rep.discs[lab] = DiscAxial(label=lab, covered=False, evidence="no axial slab covers this disc")
            continue
        s, k = hit
        llr = _sample(s, k, np.asarray(d.center, dtype=np.float64), _map(s, k))
        ok = llr >= mdl.support_llr
        rep.discs[lab] = DiscAxial(
            label=lab,
            covered=True,
            slab=s.key,
            llr=llr,
            supported=bool(ok),
            evidence=(
                f"slab {s.key} slice {k}: disc-like LLR {llr:.2f} "
                f"({'supported' if ok else 'no disc-like appearance'}, threshold {mdl.support_llr:.2f})"
            ),
        )
    if rep.unsupported:
        rep.warnings.append(
            "predicted disc(s) without disc-like axial appearance in their slab: "
            + ", ".join(rep.unsupported)
            + " (possible false disc)"
        )

    # Slab matches, lowest first (the lowest slab usually sits on L5-S1).
    for rank, s in enumerate(slabs):
        best: tuple[float, str, float] | None = None
        for lab, d in anchors.discs.items():
            n = np.asarray(d.normal, dtype=np.float64)
            ang = math.degrees(math.acos(min(1.0, abs(float(n @ s.normal)))))
            margin = (d.height_mm or 8.0) / 2.0
            if ang > cfg.slab_match_angle_deg or not s.contains(np.asarray(d.center), slab_margin_mm=margin):
                continue
            k, _ = s.nearest_slice(np.asarray(d.center))
            llr = _sample(s, k, np.asarray(d.center, dtype=np.float64), _map(s, k))
            off = abs(float(np.asarray(d.center) @ s.normal) - float(s.positions.mean()))
            if best is None or off < best[0]:
                best = (off, lab, llr)
        if best is None:
            rep.slabs.append(SlabMatch(slab=s.key, rank=rank, disc=None))
        else:
            _, lab, llr = best
            rep.slabs.append(
                SlabMatch(slab=s.key, rank=rank, disc=lab, llr=llr, supported=bool(llr >= mdl.support_llr))
            )

    rep.missed = _missed_candidates(slabs, anchors, mdl, cfg, _map)
    if rep.missed:
        rep.warnings.append(
            f"{len(rep.missed)} disc-like axial region(s) with no predicted disc in their plane: "
            "a disc may be missed"
        )
    return rep


def _missed_candidates(
    slabs: list[Stack],
    anchors: Anchors,
    mdl: AxialModel,
    cfg: AxialConfig,
    mapper: Any,
) -> list[MissedCandidate]:
    out: list[MissedCandidate] = []
    for s in slabs:
        # Predicted discs whose centre lies in this slab's block (any angle): they explain peaks in its slices.
        near = [
            np.asarray(d.center, dtype=np.float64)
            for d in anchors.discs.values()
            if s.contains(np.asarray(d.center), slab_margin_mm=(d.height_mm or 8.0) / 2.0)
        ]
        best_per_peak: list[MissedCandidate] = []
        for k in range(s.n):
            m = mapper(s, k)
            size = max(3, round(2 * cfg.missed_mm / max(min(s.row_mm, s.col_mm), 1e-6)) | 1)
            peaks = (m == ndimage.maximum_filter(m, size=size)) & (m >= mdl.missed_llr)
            for r, c in zip(*np.nonzero(peaks), strict=True):
                xyz = s.pix_to_lps(float(k), float(r), float(c))
                dists = [
                    float(np.linalg.norm((xyz - p) - float((xyz - p) @ s.normal) * s.normal)) for p in near
                ]
                nd = min(dists) if dists else None
                if nd is not None and nd <= cfg.missed_mm:
                    continue
                best_per_peak.append(
                    MissedCandidate(
                        slab=s.key,
                        lps=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                        llr=float(m[r, c]),
                        nearest_disc_mm=nd,
                    )
                )
        # One candidate per place: keep the strongest of peaks within the in-plane distance of each other.
        for cand in sorted(best_per_peak, key=lambda x: -x.llr):
            if all(
                np.linalg.norm(np.asarray(cand.lps) - np.asarray(o.lps)) > cfg.missed_mm
                for o in out
                if o.slab == cand.slab
            ):
                out.append(cand)
    return out


__all__ = [
    "DEFAULT_AXIAL",
    "FEATURES",
    "MODEL_PATH",
    "AxialConfig",
    "AxialModel",
    "AxialReport",
    "DiscAxial",
    "MissedCandidate",
    "SlabMatch",
    "assess_axial",
    "axial_slabs",
    "feature_maps",
    "llr_map",
    "load_axial_model",
]
