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
TARGET_PRECISION: dict[str, float] = {STATUS_CONFIRMED: 0.98, STATUS_PROBABLE: 0.9}
MIN_VOLUMES_ABOVE = 20
CALIBRATION_FILE = "calibration_v1.json"


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
    raws = [logit(v) for v in raw_confidence.values()] or [0.0]
    return np.array(
        [
            logit(p_applied),
            1.0 if sacrum_below else 0.0,
            1.0 if second_present else 0.0,
            float(t1t2_fraction) if second_present else 0.0,
            1.0 if classical_agree else 0.0,
            min(raws),
        ],
        dtype=np.float64,
    )


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


def choose_thresholds(p_volume: NDArray[np.float64], correct: NDArray[np.float64]) -> dict[str, float]:
    """Lowest threshold per status whose training precision reaches :data:`TARGET_PRECISION`.

    Searched on a 0.005 grid; a threshold needs at least :data:`MIN_VOLUMES_ABOVE` volumes above it.
    Falls back to :data:`DEFAULT_STATUS_THRESHOLDS` when no threshold qualifies.
    """
    out: dict[str, float] = {}
    grid = np.round(np.arange(0.5, 1.0001, 0.005), 3)
    for status in (STATUS_CONFIRMED, STATUS_PROBABLE):
        chosen: float | None = None
        for thr in grid:
            sel = p_volume >= thr
            if int(sel.sum()) < MIN_VOLUMES_ABOVE:
                continue
            if float(correct[sel].mean()) >= TARGET_PRECISION[status]:
                chosen = float(thr)
                break
        out[status] = chosen if chosen is not None else DEFAULT_STATUS_THRESHOLDS[status]
    if out[STATUS_PROBABLE] > out[STATUS_CONFIRMED]:
        out[STATUS_PROBABLE] = out[STATUS_CONFIRMED]
    return out


# ------------------------------------------------------------------ packaged calibration


@dataclass
class Calibration:
    """Disc and volume calibrators with the status thresholds. ``source`` names the training data."""

    disc: Logistic
    volume: Logistic
    thresholds: dict[str, float]
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
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
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")


def calibration_path() -> Path:
    return Path(__file__).with_name(CALIBRATION_FILE)


@lru_cache(maxsize=2)
def load_calibration(path: str | None = None) -> Calibration | None:
    """The packaged calibration (or the one at ``path``); None when the file is missing."""
    p = Path(path) if path else calibration_path()
    if not p.is_file():
        return None
    return Calibration.from_dict(json.loads(p.read_text(encoding="utf-8")))


__all__ = [
    "CALIBRATION_FILE",
    "DISC_FEATURES",
    "VOLUME_FEATURES",
    "Calibration",
    "Logistic",
    "calibration_path",
    "choose_thresholds",
    "disc_feature_rows",
    "load_calibration",
    "logit",
    "spacing_deviation",
    "volume_feature_vector",
]
