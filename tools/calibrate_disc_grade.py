"""Calibrate the Pfirrmann model and the disc narrowing threshold on SPIDER TRAINING discs.

Input: a feature table written by ``tools/dump_disc_features.py --split training`` (one row per disc, disc-level
summary rows only). Output: the constants to paste into ``spinescope/config.py`` (``pfirrmann_*``,
``disc_narrowing_dhi_threshold``, ``pfirrmann_snr_reference``) and training-set scores. Never run it on the
validation split: the validation numbers are produced once, by ``tools/eval_analysis.py``, after the rules are fixed
in ``docs/analysis/discs.md``.

    .venv/bin/python tools/calibrate_disc_grade.py /tmp/spinescope-<name>/train_disc_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def _f(row: dict[str, str], key: str) -> float:
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _kappa_quadratic(truth: np.ndarray, pred: np.ndarray, k: int = 5) -> float:
    conf = np.zeros((k, k))
    for a, b in zip(truth, pred, strict=True):
        conf[a - 1, b - 1] += 1
    n = conf.sum()
    expected = np.outer(conf.sum(1), conf.sum(0)) / n
    i = np.arange(k)
    w = (i[:, None] - i[None, :]) ** 2 / float((k - 1) ** 2)
    return 1.0 - float((w * conf).sum()) / float((w * expected).sum())


def fit_proportional_odds(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a proportional-odds model on standardised ``x`` (no NaN). Returns (mean, sd, weights, cutpoints)."""
    mean, sd = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / sd
    p = z.shape[1]
    idx = np.arange(len(y))

    def cumulative(th: np.ndarray) -> np.ndarray:
        c = np.sort(th[p : p + 4])
        eta = z @ th[:p]
        cum = 1.0 / (1.0 + np.exp(-(c[None, :] - eta[:, None])))
        return np.hstack([np.zeros((len(y), 1)), cum, np.ones((len(y), 1))])

    def nll(th: np.ndarray) -> float:
        cum = cumulative(th)
        prob = cum[idx, y] - cum[idx, y - 1]
        return float(-np.sum(np.log(np.clip(prob, 1e-12, None))))

    res = minimize(nll, np.r_[np.zeros(p), [-2.0, -1.0, 0.0, 1.0]], method="L-BFGS-B")
    cum = cumulative(res.x)
    grade = np.argmax(cum[:, 1:] >= 0.5, axis=1) + 1
    print(f"fit converged={res.success} nll={res.fun:.1f} in-sample QWK={_kappa_quadratic(y, grade):.3f}")
    return mean, sd, res.x[:p], np.sort(res.x[p : p + 4])


def youden_threshold(dhi: np.ndarray, narrowed: np.ndarray) -> tuple[float, float]:
    """Threshold t (narrowing = dhi < t) maximising sensitivity + specificity - 1, over the observed range."""
    best = (float("nan"), -1.0)
    for t in np.arange(float(dhi.min()), float(dhi.max()) + 1e-9, 0.005):
        pred = dhi < t
        tp = np.sum(pred & (narrowed == 1))
        fn = np.sum(~pred & (narrowed == 1))
        tn = np.sum(~pred & (narrowed == 0))
        fp = np.sum(pred & (narrowed == 0))
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best[1]:
            best = (float(t), float(j))
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", type=Path, help="training disc feature table (CSV)")
    args = ap.parse_args(argv)
    with args.rows.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("kind") == "disc"]
    graded = [
        r
        for r in rows
        if r.get("pfirrmann")
        and r.get("contrast") == "t2"
        and math.isfinite(_f(r, "nuc_over_csf"))
        and _f(r, "nuc_over_csf") > 0
    ]
    x = np.array(
        [[math.log(_f(r, "nuc_over_csf")), _f(r, "nuc_cv"), _f(r, "nuc_ann_contrast")] for r in graded]
    )
    y = np.array([int(r["pfirrmann"]) for r in graded])
    ok = np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    print(f"graded T2 discs: {len(y)} (grades 1-5: {np.bincount(y, minlength=6)[1:].tolist()})")
    mean, sd, weights, cuts = fit_proportional_odds(x, y)
    fmt = lambda a: "(" + ", ".join(f"{v:.3f}" for v in a) + ")"  # noqa: E731
    print(f"pfirrmann_feature_mean = {fmt(mean)}")
    print(f"pfirrmann_feature_sd = {fmt(sd)}")
    print(f"pfirrmann_weights = {fmt(weights)}")
    print(f"pfirrmann_cutpoints = {fmt(cuts)}")

    snr = [_f(r, "nuc_snr") for r in rows if r.get("contrast") == "t2"]
    snr = [s for s in snr if math.isfinite(s)]
    print(
        f"pfirrmann_snr_reference = {statistics.median(snr):.1f}  (median nucleus SNR over {len(snr)} T2 discs)"
    )

    dhi_rows = [r for r in rows if r.get("disc_narrowing") not in (None, "") and math.isfinite(_f(r, "dhi"))]
    dhi = np.array([_f(r, "dhi") for r in dhi_rows])
    narrowed = np.array([int(r["disc_narrowing"]) for r in dhi_rows])
    t, j = youden_threshold(dhi, narrowed)
    print(
        f"disc_narrowing_dhi_threshold = {t:.3f}  (Youden J {j:.3f} over {len(dhi)} discs, prevalence {narrowed.mean():.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
