"""Dataset validation on the public SPIDER lumbar MRI set (CC BY 4.0). Skipped unless it is on disk.

Anchors come from the SPIDER masks (not the auto-labeller). Expert Pfirrmann grades and Modic / endplate
flags are compared with the suggestions. Numbers from a 30-study run are reported in the change notes;
the thresholds below are sanity bounds, not performance claims.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from _spider import (
    _lps_of,
    anchors_from_masks,
    disc_name,
    gradings,
    load_t2_t1,
    paired_study_ids,
    spider_root,
)
from spinescope.analysis import discs, endplates
from spinescope.analysis.endplates import MODIC_PATTERNS
from spinescope.config import DEFAULT

N_STUDIES = 10

pytestmark = [
    pytest.mark.dataset,
    pytest.mark.slow,
    pytest.mark.skipif(spider_root() is None, reason="SPIDER not found under $SPINESCOPE_DATA/spider"),
]


def _kappa(pairs: list[tuple[int, int]]) -> float:
    a = np.asarray(pairs, dtype=int)
    tp = int(np.sum((a[:, 0] == 1) & (a[:, 1] == 1)))
    tn = int(np.sum((a[:, 0] == 0) & (a[:, 1] == 0)))
    fp = int(np.sum((a[:, 0] == 1) & (a[:, 1] == 0)))
    fn = int(np.sum((a[:, 0] == 0) & (a[:, 1] == 1)))
    n = tp + tn + fp + fn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def _flagged(rows: list[dict], body: str, endplate: str) -> int | None:
    """1 if any side of this endplate carries a Modic-type suggestion, 0 if assessed and normal, else None."""
    hits = [
        r
        for r in rows
        if r["scope"] == "endplate"
        and r["vertebra"] == body
        and r["endplate"] == endplate
        and r["side"] != "all"
    ]
    return int(any(r["pattern"] in MODIC_PATTERNS for r in hits)) if hits else None


def _height_bias(sid: str, study, t2, mask, an, row_k: dict[str, int], out: list[float]) -> None:
    """Our mid-third disc height minus the mask height on the same slice (mm)."""
    for lab, k in row_k.items():
        i = next(ii for ii in range(1, 10) if disc_name(ii) == lab)
        tr = discs.trace_disc(t2, k, lab, an.discs[lab], DEFAULT)
        m_k = mask[k] == 200 + i
        if not m_k.any():
            continue
        r_, c_ = np.nonzero(m_k)
        pts = _lps_of(t2, np.full(r_.size, k), r_, c_)
        s_ = (pts - tr.origin) @ tr.u
        h_ = (pts - tr.origin) @ tr.n
        edges = np.arange(np.floor(s_.min()), s_.max() + 0.5, 0.5)
        ms, mh = [], []
        for a, b in itertools.pairwise(edges):
            sel = (s_ >= a) & (s_ < b)
            if sel.any():
                ms.append(0.5 * (a + b))
                mh.append(h_[sel].max() - h_[sel].min() + t2.row_mm)
        ms, mh = np.asarray(ms), np.asarray(mh)
        L = tr.s_post - tr.s_ant
        lo, hi = tr.s_ant + L / 3, tr.s_ant + 2 * L / 3
        sel3 = (ms >= lo) & (ms < hi)
        s3 = (tr.s >= lo) & (tr.s < hi) & tr.ok
        if sel3.any() and s3.any():
            out.append(float(np.mean((tr.top - tr.bot)[s3]) - np.mean(mh[sel3])))


def test_spider_agreement_sanity():
    g = gradings()
    ids = paired_study_ids()[:N_STUDIES]
    assert ids, "no paired SPIDER studies"
    pf, nuc, up, low, any_disc, bias = [], [], [], [], [], []
    for sid in ids:
        loaded = load_t2_t1(sid)
        if loaded is None:
            continue
        study, t2, mask = loaded
        an = anchors_from_masks(t2, mask)
        dres = discs.analyze(study, an)
        summ = {r["disc"]: r for r in dres.rows if r["scope"] == "disc"}
        primary = {r["disc"]: int(r["slice_k"]) for r in dres.rows if r["scope"] == "slice" and r["primary"]}
        _height_bias(sid, study, t2, mask, an, primary, bias)
        eres = endplates.analyze(study, an) if "sag_t1" in study else None
        for lab in an.disc_labels():
            i = next((ii for ii in range(1, 10) if disc_name(ii) == lab), None)
            if i is None or (sid, i) not in g or lab not in summ:
                continue
            ex = g[(sid, i)]
            s = summ[lab]
            if s["pfirrmann_suggestion"] is not None:
                pf.append((int(s["pfirrmann_suggestion"]), ex.pfirrmann))
            if np.isfinite(s["nuc_over_csf"]):
                nuc.append((float(s["nuc_over_csf"]), ex.pfirrmann))
            if eres is None:
                continue
            up_body, low_body = lab.split("-")

            fu = _flagged(eres.rows, up_body, "inferior")
            fl = _flagged(eres.rows, low_body, "superior")
            if fu is not None:
                up.append((fu, ex.up_endplate))
            if fl is not None:
                low.append((fl, ex.low_endplate))
            if fu is not None and fl is not None:
                any_disc.append((int(bool(fu or fl)), int(ex.modic > 0)))

    assert len(pf) >= 20 and len(nuc) >= 20
    # nucleus brightness should fall as the expert grade rises
    rho_nuc = stats.spearmanr(np.array(nuc)[:, 0], np.array(nuc)[:, 1]).correlation
    assert rho_nuc < 0.0
    # Pfirrmann suggestion is positively associated with the expert grade (uncalibrated, so only a sign check)
    rho_pf = stats.spearmanr(np.array(pf)[:, 0], np.array(pf)[:, 1]).correlation
    assert rho_pf > 0.0
    # Modic presence agrees beyond chance at the disc level
    assert len(any_disc) >= 10 and _kappa(any_disc) > 0.0
    # heights: the method reads higher than the masks (see docs notes); bound the bias
    assert len(bias) >= 10 and abs(float(np.median(bias))) < 3.0
    assert len(up) >= 10 and len(low) >= 10
