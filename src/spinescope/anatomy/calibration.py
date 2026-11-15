"""Calibrated confidence for disc labels and for the numbering of a study.

The raw disc confidence is the mean disc-class probability of the learned labeller, and the raw numbering
confidence is a heuristic (bone below the lowest disc, tilt, second-series agreement). Neither is a probability
of being right. This module maps a few features to one: a logistic (Platt-style) model fitted on SPIDER
*training* cases only (``tools/calibrate_labeling.py``), shipped as ``calibration_v1.json`` next to this file.

Disc features (one row per predicted disc; the target is "matched to a true disc with the right level name"):

* ``logit_raw``: logit of the model's disc probability;
* ``height_dev``: |log(height / study median height)|, disc-size plausibility;
* ``spacing_dev``: mean relative deviation of the centre-to-centre spacing to the neighbours from the study
  median spacing, spacing regularity;
* ``second_present`` / ``t1t2_agree``: a second sagittal series exists / the same disc with the same label is
  found in it (T1 against T2);
* ``sacrum_below``: vertebral bone is visible below the lowest disc (sacrum confirmation).

Volume features (the target is "every matched disc carries the right name"):

* ``logit_offset``: logit of the posterior of the offset that the final names carry (see fusion.py);
* ``sacrum_below``, ``second_present``, ``t1t2_frac`` (share of discs confirmed by the second series),
  ``classical_agree`` (the classical labeller agrees geometrically), ``min_logit_raw`` (the weakest disc).

Status thresholds (see :mod:`numbering`) are chosen on the training cases: the lowest calibrated value with at
least the target precision for each status.

Version 2 (``calibration_v2.json``, lab18) adds *numbering signals* to the volume model (:func:`numbering_signals`):
the lumbosacral CNN's probabilities for the lowest three discs (margin, second disc), the size and spacing of the
lowest discs, the field-of-view margin below the lowest disc, the vote pattern of the fusion, the S1 body confidence
and the discs below the conus. They are computed from the labeller's outputs only (no second network). v2 is fitted
on SPIDER training plus half of the Sudirman patients (``tools/calibrate_labeling.py``, protocol in
docs/labeling/lab18_reliability.md); v1 stays loadable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from ..config import DEFAULT
from .anchors import Anchors
from .numbering import DEFAULT_STATUS_THRESHOLDS, STATUS_CONFIRMED, STATUS_PROBABLE

DISC_FEATURES: tuple[str, ...] = (
    "logit_raw",
    "height_dev",
    "spacing_dev",
    "second_present",
    "t1t2_agree",
    "sacrum_below",
)
VOLUME_FEATURES: tuple[str, ...] = (
    "logit_offset",
    "sacrum_below",
    "second_present",
    "t1t2_frac",
    "classical_agree",
    "min_logit_raw",
)
#: Precision targets for the status thresholds (policy, chosen before the validation numbers were
#: computed; the training base rate of correct numbering is about 0.96, so these are set above it).
#: Version 1 used 0.90 for probable, below the base rate of correct numbering on the external sets (0.85 to 0.92), so
#: "probable" admitted almost every study. Version 2 uses 0.97 (lab18, fixed before the held-out sets were scored).
TARGET_PRECISION_V1: dict[str, float] = {STATUS_CONFIRMED: 0.98, STATUS_PROBABLE: 0.9}
TARGET_PRECISION: dict[str, float] = {STATUS_CONFIRMED: 0.98, STATUS_PROBABLE: 0.97}
#: Version 3 (lab22): ``confirmed`` needs out-of-fold precision 0.99 (a confirmed numbering is a claim about bone
#: below the lowest disc, so it is held to a stricter target than ``probable``). Fixed before any held-out set was scored.
TARGET_PRECISION_V3: dict[str, float] = {STATUS_CONFIRMED: 0.99, STATUS_PROBABLE: 0.97}
#: Threshold of a status whose target precision cannot be reached on the fit set: no study reaches it.
UNREACHABLE_THRESHOLD = 1.01
MIN_VOLUMES_ABOVE = 20
CALIBRATION_FILE = "calibration_v2.json"  # packaged calibration (lab18); v1 below
CALIBRATION_FILE_V1 = (
    "calibration_v1.json"  # lab08 calibration, still loadable (``load_calibration(version=1)``)
)
CALIBRATION_FILE_V3 = "calibration_v3.json"  # lab22: numbering after the geometry (S1 body, conus vote)
#: Calibration version loaded when none is asked for. Stays 2 unless a version 3 file is shipped (lab22).
DEFAULT_CALIBRATION_VERSION = 2
#: Volume features of version 1 (names of the calibration's volume model in v1 files).
VOLUME_FEATURES_V1: tuple[str, ...] = VOLUME_FEATURES
#: Numbering signals of version 2 (see :func:`numbering_signals`), in the order of the returned dict.
SIGNAL_FEATURES: tuple[str, ...] = (
    "lumbo_p0",
    "lumbo_margin",
    "lumbo_p1",
    "lumbo_available",
    "h_bottom",
    "h_second",
    "gap_bottom_dev",
    "fov_below_mm",
    "vote_shift_frac",
    "p_shift",
    "n_discs",
)
#: Volume features of version 2: the version 1 set plus the two signals selected on the fit set (lab18).
#: ``lumbo_margin`` (the lowest candidate's P(L5-S1) minus the best of the others) and ``gap_bottom_dev`` (spacing of
#: the lowest two discs against the study median). Selected by out-of-fold log-loss on SPIDER training plus
#: Sudirman half A only (docs/labeling/lab18_reliability.md); the other signals are computed and recorded but not used.
VOLUME_FEATURES_V2: tuple[str, ...] = (*VOLUME_FEATURES, "lumbo_margin", "gap_bottom_dev")
#: Numbering signals added in version 3 (lab22). They need the vertebral geometry, which is measured before the
#: numbering is calibrated: ``s1_below_mm`` the bone below the lowest disc on the mid-sagittal slice (the S1 body),
#: ``s1_present`` that it reaches ``sacrum_min_visible_mm``, ``s1_ap_ratio`` the AP depth of that body over the
#: body above the lowest disc (sacral morphology: sacral segments are narrower), ``t1t2_bottom`` the lowest disc is
#: confirmed by the second sagittal series.
SIGNAL_FEATURES_V3: tuple[str, ...] = (
    *SIGNAL_FEATURES,
    "s1_present",
    "s1_below_mm",
    "s1_ap_ratio",
    "t1t2_bottom",
)
#: Neutral values of the version 3 signals when the geometry (or a second series) is missing.
S1_AP_RATIO_NEUTRAL = 1.0


def logit(p: float) -> float:
    """Logit with the probability clipped to [1e-3, 1 - 1e-3]."""
    q = min(max(float(p), 1e-3), 1 - 1e-3)
    return math.log(q / (1 - q))


# ------------------------------------------------------------------ features


def spacing_deviation(anchors: Anchors) -> dict[str, float]:
    """Per disc: mean relative deviation of its spacing to the adjacent discs from the study median spacing."""
    labels = sorted(anchors.discs, key=lambda lab: anchors.discs[lab].center[2])
    z = [float(anchors.discs[lab].center[2]) for lab in labels]
    gaps = [abs(z[i + 1] - z[i]) for i in range(len(z) - 1)]
    out = {lab: 0.0 for lab in labels}
    if not gaps:
        return out
    med = float(np.median(gaps))
    if med <= 0:
        return out
    for i, lab in enumerate(labels):
        devs = [abs(gaps[j] - med) / med for j in (i - 1, i) if 0 <= j < len(gaps)]
        out[lab] = float(np.mean(devs)) if devs else 0.0
    return out


def disc_feature_rows(
    anchors: Anchors,
    raw_confidence: dict[str, float],
    t1t2_by_label: dict[str, bool],
    *,
    second_present: bool,
    sacrum_below: bool,
) -> dict[str, NDArray[np.float64]]:
    """Feature vector (order :data:`DISC_FEATURES`) for every disc of ``anchors``, keyed by its label."""
    heights = [float(d.height_mm) for d in anchors.discs.values() if d.height_mm]
    med_h = float(np.median(heights)) if heights else 0.0
    spac = spacing_deviation(anchors)
    rows: dict[str, NDArray[np.float64]] = {}
    for lab, d in anchors.discs.items():
        h = float(d.height_mm) if d.height_mm else med_h
        height_dev = abs(math.log(h / med_h)) if med_h > 0 and h > 0 else 0.0
        rows[lab] = np.array(
            [
                logit(raw_confidence.get(lab, d.confidence)),
                height_dev,
                spac.get(lab, 0.0),
                1.0 if second_present else 0.0,
                1.0 if second_present and t1t2_by_label.get(lab, False) else 0.0,
                1.0 if sacrum_below else 0.0,
            ],
            dtype=np.float64,
        )
    return rows


def volume_feature_vector(
    p_applied: float,
    raw_confidence: dict[str, float],
    *,
    sacrum_below: bool,
    second_present: bool,
    t1t2_fraction: float,
    classical_agree: bool,
) -> NDArray[np.float64]:
    """Volume feature vector (order :data:`VOLUME_FEATURES`)."""
    f = volume_feature_map(
        p_applied,
        raw_confidence,
        sacrum_below=sacrum_below,
        second_present=second_present,
        t1t2_fraction=t1t2_fraction,
        classical_agree=classical_agree,
    )
    return np.array([f[k] for k in VOLUME_FEATURES], dtype=np.float64)


def volume_feature_map(
    p_applied: float,
    raw_confidence: dict[str, float],
    *,
    sacrum_below: bool,
    second_present: bool,
    t1t2_fraction: float,
    classical_agree: bool,
) -> dict[str, float]:
    """The version 1 volume features by name (keys :data:`VOLUME_FEATURES`)."""
    raws = [logit(v) for v in raw_confidence.values()] or [0.0]
    return {
        "logit_offset": logit(p_applied),
        "sacrum_below": 1.0 if sacrum_below else 0.0,
        "second_present": 1.0 if second_present else 0.0,
        "t1t2_frac": float(t1t2_fraction) if second_present else 0.0,
        "classical_agree": 1.0 if classical_agree else 0.0,
        "min_logit_raw": min(raws),
    }


def numbering_signals(
    anchors: Anchors,
    *,
    lumbo_p: list[float] | None,
    votes: list[dict[str, Any]],
    p_shift: float,
    coverage_low_z: float | None,
    s1_below_mm: float | None = None,
    s1_ap_ratio: float | None = None,
    t1t2_bottom: bool = False,
    min_visible_mm: float = DEFAULT.sacrum_min_visible_mm,
) -> dict[str, float]:
    """Version 2 numbering signals of one study, from the labeller's outputs (keys :data:`SIGNAL_FEATURES`).

    Computed on the anchors after the sacral-disc check and before relabelling (the geometry is name-free). Every
    value is finite; when a source is missing the neutral default is used and ``lumbo_available`` says so.

    * ``lumbo_p0`` / ``lumbo_p1`` / ``lumbo_margin``: P(L5-S1) of the lowest candidate and of the second-lowest, and
      the margin of the lowest over the best of the others (lumbosacral CNN, lab13). Default 0.5 / 0.0 / 0.0.
    * ``h_bottom`` / ``h_second``: height of the two lowest discs over the study median height (default 1).
    * ``gap_bottom_dev``: |log(gap / median gap)| between the two lowest discs (spacing near the bottom; default 0).
    * ``fov_below_mm``: lowest disc to the lower edge of the sagittal coverage, in mm (0 when unknown).
    * ``vote_shift_frac``: share of the fusion's cue votes that favour a non-zero offset; ``p_shift``: posterior of
      any non-zero offset.
    * ``n_discs``: number of discs after the sacral check.
    * Version 3 (lab22; the vertebral geometry is measured before numbering): ``s1_below_mm`` bone below the lowest
      disc on the mid-sagittal slice (0 when not measured), ``s1_present`` 1 when it reaches ``min_visible_mm``,
      ``s1_ap_ratio`` AP depth of that body over the body above the lowest disc (1 when unknown), ``t1t2_bottom``
      1 when the lowest disc is confirmed by the second series.
    """
    discs = sorted(anchors.discs.values(), key=lambda d: float(d.center[2]))  # bottom-up
    n = len(discs)
    heights = [float(d.height_mm) for d in discs if d.height_mm]
    med_h = float(np.median(heights)) if heights else 0.0
    zs = [float(d.center[2]) for d in discs]
    gaps = [zs[i + 1] - zs[i] for i in range(n - 1)]
    med_gap = float(np.median(gaps)) if gaps else 0.0

    def h_ratio(d: Any) -> float:
        return float(d.height_mm) / med_h if (d.height_mm and med_h > 0) else 1.0

    lp = [float(p) for p in (lumbo_p or [])]
    lumbo_available = 1.0 if lp else 0.0
    p0 = lp[0] if lp else 0.5
    p1 = lp[1] if len(lp) > 1 else 0.0
    upper = max(lp[1:]) if len(lp) > 1 else 0.0
    margin = (p0 - upper) if lp else 0.0

    gap_dev = abs(math.log(gaps[0] / med_gap)) if n >= 2 and med_gap > 0 and gaps[0] > 0 else 0.0
    fov = zs[0] - float(coverage_low_z) if (n and coverage_low_z is not None) else 0.0
    shift_votes = [v for v in votes if int(v.get("best", 0)) != 0]
    vote_frac = len(shift_votes) / len(votes) if votes else 0.0
    below = float(s1_below_mm) if s1_below_mm is not None else 0.0
    ratio = float(s1_ap_ratio) if s1_ap_ratio is not None else S1_AP_RATIO_NEUTRAL
    return {
        "lumbo_p0": p0,
        "lumbo_margin": margin,
        "lumbo_p1": p1,
        "lumbo_available": lumbo_available,
        "h_bottom": h_ratio(discs[0]) if discs else 1.0,
        "h_second": h_ratio(discs[1]) if n > 1 else 1.0,
        "gap_bottom_dev": gap_dev,
        "fov_below_mm": float(fov),
        "vote_shift_frac": float(vote_frac),
        "p_shift": float(p_shift),
        "n_discs": float(n),
        "s1_present": 1.0 if below >= min_visible_mm else 0.0,
        "s1_below_mm": below,
        "s1_ap_ratio": ratio,
        "t1t2_bottom": 1.0 if t1t2_bottom else 0.0,
    }


# ------------------------------------------------------------------ model


@dataclass
class Logistic:
    """Standardised logistic regression with an L2 penalty on the weights (not the bias)."""

    names: tuple[str, ...]
    mean: list[float]
    scale: list[float]
    weights: list[float]
    bias: float
    l2: float = 1.0
    n_train: int = 0

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calibrated probabilities for rows of ``x`` (shape (n, len(names)) or (len(names),))."""
        X = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = (X - np.asarray(self.mean)) / np.asarray(self.scale)
        p = expit(z @ np.asarray(self.weights) + self.bias)
        return np.asarray(p, dtype=np.float64)

    @classmethod
    def fit(
        cls, X: NDArray[np.float64], y: NDArray[np.float64], names: tuple[str, ...], l2: float = 1.0
    ) -> Logistic:
        """Maximum a posteriori logistic fit (L-BFGS on the penalised log-loss). Deterministic."""
        X = np.asarray(X, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z = np.hstack([(X - mu) / sd, np.ones((X.shape[0], 1))])
        t = 2.0 * yy - 1.0
        dim = Z.shape[1]

        def obj(w: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            s = Z @ w
            loss = float(np.sum(np.logaddexp(0.0, -t * s)))
            reg = 0.5 * l2 * float(np.sum(w[:-1] ** 2))
            p = expit(s)
            grad = Z.T @ (p - yy)
            grad[:-1] += l2 * w[:-1]
            return loss + reg, grad

        res = minimize(obj, np.zeros(dim), jac=True, method="L-BFGS-B")
        w = res.x
        return cls(
            names=tuple(names),
            mean=[float(v) for v in mu],
            scale=[float(v) for v in sd],
            weights=[float(v) for v in w[:-1]],
            bias=float(w[-1]),
            l2=l2,
            n_train=int(X.shape[0]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": self.mean,
            "scale": self.scale,
            "weights": self.weights,
            "bias": self.bias,
            "l2": self.l2,
            "n_train": self.n_train,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Logistic:
        return cls(
            names=tuple(d["names"]),
            mean=[float(v) for v in d["mean"]],
            scale=[float(v) for v in d["scale"]],
            weights=[float(v) for v in d["weights"]],
            bias=float(d["bias"]),
            l2=float(d.get("l2", 1.0)),
            n_train=int(d.get("n_train", 0)),
        )


def choose_thresholds(
    p_volume: NDArray[np.float64],
    correct: NDArray[np.float64],
    *,
    targets: dict[str, float] | None = None,
    fallback: dict[str, float] | None = None,
) -> dict[str, float]:
    """Lowest threshold per status whose training precision reaches ``targets`` (default :data:`TARGET_PRECISION`).

    Searched on a 0.005 grid; a threshold needs at least :data:`MIN_VOLUMES_ABOVE` volumes above it.
    When no threshold qualifies the status gets ``fallback`` (default :data:`DEFAULT_STATUS_THRESHOLDS`; version 3
    passes :data:`UNREACHABLE_THRESHOLD`, so an unreachable status is never announced).
    """
    tg = targets or TARGET_PRECISION
    fb = fallback or DEFAULT_STATUS_THRESHOLDS
    out: dict[str, float] = {}
    grid = np.round(np.arange(0.5, 1.0001, 0.005), 3)
    for status in (STATUS_CONFIRMED, STATUS_PROBABLE):
        chosen: float | None = None
        for thr in grid:
            sel = p_volume >= thr
            if int(sel.sum()) < MIN_VOLUMES_ABOVE:
                continue
            if float(correct[sel].mean()) >= tg[status]:
                chosen = float(thr)
                break
        out[status] = chosen if chosen is not None else fb[status]
    if out[STATUS_PROBABLE] > out[STATUS_CONFIRMED]:
        out[STATUS_PROBABLE] = out[STATUS_CONFIRMED]
    return out


# ------------------------------------------------------------------ packaged calibration


@dataclass
class Calibration:
    """Disc and volume calibrators with the status thresholds. ``source`` names the training data.

    ``version`` 1: the volume model uses :data:`VOLUME_FEATURES_V1`; version 2 (lab18): the volume model may use
    the numbering signals too. The model's own ``names`` say which features it takes.
    """

    disc: Logistic
    volume: Logistic
    thresholds: dict[str, float]
    source: str = ""
    version: int = 1

    def volume_vector(self, feats: dict[str, float]) -> NDArray[np.float64]:
        """The volume model's input for one study, from the named features (see :func:`volume_feature_map`)."""
        return np.array([float(feats[k]) for k in self.volume.names], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "disc": self.disc.to_dict(),
            "volume": self.volume.to_dict(),
            "thresholds": self.thresholds,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibration:
        return cls(
            disc=Logistic.from_dict(d["disc"]),
            volume=Logistic.from_dict(d["volume"]),
            thresholds={k: float(v) for k, v in d["thresholds"].items()},
            source=str(d.get("source", "")),
            version=int(d.get("version", 1)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")


def calibration_path(version: int = 2) -> Path:
    """Packaged calibration file of ``version``: 1 (lab08), 2 (lab18, default) or 3 (lab22)."""
    if version >= 3:
        return Path(__file__).with_name(CALIBRATION_FILE_V3)
    return Path(__file__).with_name(CALIBRATION_FILE if version >= 2 else CALIBRATION_FILE_V1)


@lru_cache(maxsize=4)
def load_calibration(path: str | None = None, version: int | None = None) -> Calibration | None:
    """The packaged calibration (:data:`DEFAULT_CALIBRATION_VERSION` by default; ``version=1`` or ``3`` for the others),
    or the file at ``path``.

    None when the file is missing (the caller then uses the uncalibrated fallback).
    """
    p = Path(path) if path else calibration_path(DEFAULT_CALIBRATION_VERSION if version is None else version)
    if not p.is_file():
        return None
    return Calibration.from_dict(json.loads(p.read_text(encoding="utf-8")))


__all__ = [
    "CALIBRATION_FILE",
    "CALIBRATION_FILE_V3",
    "DEFAULT_CALIBRATION_VERSION",
    "DISC_FEATURES",
    "SIGNAL_FEATURES",
    "SIGNAL_FEATURES_V3",
    "TARGET_PRECISION",
    "TARGET_PRECISION_V1",
    "TARGET_PRECISION_V3",
    "UNREACHABLE_THRESHOLD",
    "VOLUME_FEATURES",
    "VOLUME_FEATURES_V1",
    "VOLUME_FEATURES_V2",
    "Calibration",
    "Logistic",
    "calibration_path",
    "choose_thresholds",
    "disc_feature_rows",
    "load_calibration",
    "logit",
    "numbering_signals",
    "spacing_deviation",
    "volume_feature_map",
    "volume_feature_vector",
]
