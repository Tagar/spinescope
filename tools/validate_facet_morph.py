"""Validate facet joint localisation, facet angles and morphology proxies on Sudirman axial T1 (dx21).

Pre-registered rules: docs/analysis/facets.md, section 6 (written before the validation run).

Split (Sudirman labelled patients, sorted by folder id): the first half calibrates (development: the first
``--dev`` patients only, plumbing checks), the second half validates. Nothing is tuned on the validation half.

Reference (the Sudirman label map, label 100 = posterior elements, on the annotated axial T1 slice, which is the
DICOM grid):

- Mask joint centre: centroid of label-100 pixels on the side, lateral to the canal wall (canal half width), within
  ``facets_loc_search_mm`` of the geometric site. This is a proxy: the mask has no joint, only bone.
- Mask joint angle: the same angle method (principal axis, angle to the mid-sagittal direction) applied to the
  boundary pixels of that label-100 region in the same window.

Predictions (image only): joint centre (dark-ridge localisation), joint angle, joint-line length and width.

Morphology weak labels: notes parsed by ``spinescope.datasets.report_labels``: ``facet_arthropathy`` at the level
(positive = any positive label; negative = only negated labels; conflicts and absence dropped). Primary rule:
mean joint-line length ratio (level / study median over annotated levels) >= ``facets_morph_length_ratio``.
Secondary rule: mean joint-line width ratio <= 0.8. Recall (sensitivity) and specificity, patient bootstrap.

Usage (repository root, project venv)::

    SPINESCOPE_THREADS=2 .venv/bin/python tools/validate_facet_morph.py --split calibrate --dev 20 --workers 2 \\
        --out /tmp/spinescope-dx21_facet_morph/dev
    SPINESCOPE_THREADS=2 .venv/bin/python tools/validate_facet_morph.py --split validate --workers 2 \\
        --out /tmp/spinescope-dx21_facet_morph/val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))

from validate_canal import binary_metrics, continuous_summary, patient_bootstrap

log = logging.getLogger("validate_facet_morph")

ANCHOR_CACHE = Path("/tmp/spinescope-dx21_facet_morph/anchors")
POSTERIOR_ELEMENT_VALUE = 100
LEVELS = ("L3-L4", "L4-L5", "L5-S1")
PRIMARY_RATIO_KEY = "length_ratio"


def _anchors_for(study: Any, key: str) -> Any:
    """``label_spine(study)`` cached as JSON under :data:`ANCHOR_CACHE` (computed once per study)."""
    from spinescope.anatomy import label_spine
    from spinescope.anatomy.anchors import Anchors

    ANCHOR_CACHE.mkdir(parents=True, exist_ok=True)
    path = ANCHOR_CACHE / f"{key}.json"
    if path.is_file():
        return Anchors.from_json(str(path))
    anchors = label_spine(study)
    anchors.to_json(str(path))
    return anchors


def _pca_direction(pts: NDArray[np.float64]) -> tuple[float, float] | None:
    if pts.shape[0] < 3:
        return None
    c = pts - pts.mean(axis=0)
    w, vecs = np.linalg.eigh(np.cov(c.T))
    major = vecs[:, int(np.argmax(w))]
    return float(major[0]), float(major[1])


def _mask_reference(
    st: Any, k: int, mask: NDArray[np.integer], origin: NDArray[np.float64], site: Any, cfg: Any
) -> dict[str, Any]:
    """Mask-derived joint centre (u, v mm) and joint angle for one side, same ROI as the image method."""
    from spinescope.analysis.facets import joint_angle_deg, sample_label_plane

    px = cfg.facets_px_mm
    half = cfg.facets_roi_half_mm
    n = round(2 * half / px)
    lab = sample_label_plane(st, k, mask, origin, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), (n, n), px)
    tt = (np.arange(n) - (n - 1) / 2) * px
    ss = (np.arange(n) - (n - 1) / 2) * px
    ggs, ggt = np.meshgrid(ss, tt)
    uc = float((site.canal_xyz - origin) @ np.array([1.0, 0, 0]))
    lateral = (ggs >= uc + site.canal_half_mm) if site.side == "L" else (ggs <= uc - site.canal_half_mm)
    region = lateral & (np.hypot(ggs, ggt) <= cfg.facets_loc_search_mm)
    bone = (lab == POSTERIOR_ELEMENT_VALUE) & region
    out: dict[str, Any] = {"n_bone": int(bone.sum()), "centre_uv": None, "angle_deg": None}
    if bone.sum() >= 3:
        rr, cc = np.nonzero(bone)
        pts = np.stack([ss[cc], tt[rr]], axis=1)
        out["centre_uv"] = [float(pts[:, 0].mean()), float(pts[:, 1].mean())]
        # boundary of the bone region inside the window: the same principal-axis angle as the image method
        edge = bone & ~ndi.binary_erosion(bone)
        er, ec = np.nonzero(edge)
        epts = np.stack([ss[ec], tt[er]], axis=1)
        d = _pca_direction(epts)
        if d is not None:
            out["angle_deg"] = joint_angle_deg(d)
    return out


def _case(patient: str, sessions: list[Path]) -> dict[str, Any]:
    """Image joint geometry and mask reference for every annotated level of one Sudirman patient."""
    from spinescope.analysis.facets import geometry_at, joint_sites
    from spinescope.anatomy.correspondence import level_map
    from spinescope.config import DEFAULT
    from spinescope.datasets.sudirman import read_mask, select_session

    out: dict[str, Any] = {"patient": patient, "rows": [], "error": None}
    try:
        sel = select_session(patient, sessions)
        if sel is None:
            out["error"] = "no session reproduces the label slices"
            return out
        sess, study, truths = sel
        anchors = _anchors_for(study, f"{patient}_{sess.name}")
        lm = level_map(study, anchors)
        sites = joint_sites(lm, anchors, DEFAULT)
        for t in truths:
            base = {"patient": patient, "level": t.level, "dlabel": t.dlabel, "status": "ok"}
            if t.level not in sites:
                out["rows"].append(dict(base, status="no level-map site"))
                continue
            st = study.stacks[t.stack_key]
            mask = read_mask(patient, t.dlabel)
            for side in ("L", "R"):
                site = sites[t.level][side]
                geo = geometry_at(st, site, DEFAULT, slice_k=int(t.slice_index))
                if geo is None:
                    out["rows"].append(dict(base, side=side, status="not axial"))
                    continue
                k = int(t.slice_index)  # the label map lives on the annotated slice (the DICOM grid)
                n_ = np.asarray(st.normal, dtype=np.float64)
                o_k = st.pix_to_lps(float(k), 0.0, 0.0)
                origin = site.xyz - float((site.xyz - o_k) @ n_) * n_
                ref = _mask_reference(st, k, mask, origin, site, DEFAULT)
                row = dict(base, side=side, status="ok")
                row.update(
                    {
                        "loc_ok": geo["loc_ok"],
                        "loc_reason": geo["loc_reason"],
                        "img_uv": geo["joint_uv_mm"],
                        "img_angle_deg": geo["joint_angle_deg"],
                        "line_length_mm": geo["joint_line_length_mm"],
                        "line_width_mm": geo["joint_line_width_mm"],
                        "mask_uv": ref["centre_uv"],
                        "mask_angle_deg": ref["angle_deg"],
                        "mask_bone_px": ref["n_bone"],
                        "site_uv_mm": [0.0, 0.0],
                    }
                )
                out["rows"].append(row)
    except Exception as exc:  # one bad patient is counted, not fatal
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _worker(args: tuple[str, list[str]]) -> dict[str, Any]:
    patient, sess_names = args
    return _case(patient, [Path(s) for s in sess_names])


def _notes_levels() -> dict[int, dict[str, Any]]:
    """Per notes number: level -> aggregated facet_arthropathy (value True / False / None)."""
    from spinescope.datasets.report_labels import aggregate_levels, load_sudirman_report_labels

    out: dict[int, dict[str, Any]] = {}
    for num, parse in load_sudirman_report_labels().items():
        agg = aggregate_levels(parse.labels)
        out[num] = {lev: (agg.get(lev, {}).get("facet_arthropathy") or {}).get("value") for lev in LEVELS}
    return out


def _ratio_rows(rows: list[dict[str, Any]]) -> None:
    """Add length_ratio and width_ratio per row: level value over the study median over annotated levels."""
    by_pat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_pat.setdefault(r["patient"], []).append(r)
    for rs in by_pat.values():
        for src, dst in (("line_length_mm", PRIMARY_RATIO_KEY), ("line_width_mm", "width_ratio")):
            vals: dict[tuple[str, str], float] = {}
            for r in rs:
                if r.get(src) is not None:
                    vals[(r["level"], r["side"])] = float(r[src])
            if not vals:
                continue
            med = float(np.median(list(vals.values())))
            for r in rs:
                v = vals.get((r["level"], r["side"]))
                r[dst] = None if v is None or med <= 0 else round(v / med, 4)


def run(split: str, dev: int, workers: int) -> dict[str, Any]:
    from spinescope.config import DEFAULT
    from spinescope.datasets.sudirman import labelled_patients, patient_sessions

    pats = labelled_patients()
    half = len(pats) // 2
    chosen = pats[:half] if split == "calibrate" else pats[half:]
    if split == "calibrate" and dev:
        chosen = chosen[:dev]
    sessions = patient_sessions()
    jobs = [(p, [str(s) for s in sessions[p]]) for p in chosen if p in sessions]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_worker, jobs):
            results.append(res)
    rows = [r for res in results for r in res["rows"]]
    _ratio_rows(rows)
    failures = [res["error"] for res in results if res["error"]]
    scored = [r for r in rows if r["status"] == "ok" and r.get("loc_ok")]
    loc_pairs = [r for r in scored if r.get("mask_uv") is not None and r.get("img_uv") is not None]
    geo_ref = [r for r in rows if r.get("mask_uv") is not None and r["status"] == "ok"]

    out: dict[str, Any] = {
        "split": split,
        "patients": len(chosen),
        "patients_failed": len(failures),
        "failures": sorted(set(failures))[:10],
        "levels_total": len(rows),
        "status_counts": _count([r["status"] for r in rows]),
        "config": {
            "facets_loc_sigma_mm": DEFAULT.facets_loc_sigma_mm,
            "facets_loc_search_mm": DEFAULT.facets_loc_search_mm,
            "facets_loc_component_frac": DEFAULT.facets_loc_component_frac,
            "facets_loc_max_radius_mm": DEFAULT.facets_loc_max_radius_mm,
            "facets_morph_length_ratio": DEFAULT.facets_morph_length_ratio,
        },
    }
    out["localisation"] = _localisation(loc_pairs, scored)
    out["angles"] = _angles(scored)
    out["morphology"] = _morphology(rows, results, geo_ref)
    out["rows"] = rows
    return out


def _count(xs: list[str]) -> dict[str, int]:
    d: dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def _localisation(pairs: list[dict[str, Any]], scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Joint-centre error (mm) of the image joint and of the geometric site, against the mask reference."""
    res: dict[str, Any] = {"n_scored": len(pairs), "n_loc_ok": len(scored)}
    for side in ("L", "R", "all"):
        sel = [r for r in pairs if side == "all" or r["side"] == side]
        if not sel:
            continue
        img = np.array([np.hypot(*(np.array(r["img_uv"]) - np.array(r["mask_uv"]))) for r in sel])
        geom = np.array([np.hypot(*np.array(r["mask_uv"])) for r in sel])
        groups = [r["patient"] for r in sel]
        vals = list(zip(img.tolist(), geom.tolist(), strict=True))
        res[side] = {
            "n": len(sel),
            "image_error_mm": continuous_summary(img, np.zeros_like(img)) | {"median": float(np.median(img))},
            "geometric_site_error_mm_median": float(np.median(geom)),
            "image_mae_ci": patient_bootstrap(
                groups, vals, lambda s: float(np.mean([a for a, _ in s])) if s else float("nan")
            ),
            "image_median_ci": patient_bootstrap(
                groups, vals, lambda s: float(np.median([a for a, _ in s])) if s else float("nan")
            ),
            "geometric_median_ci": patient_bootstrap(
                groups, vals, lambda s: float(np.median([b for _, b in s])) if s else float("nan")
            ),
        }
    return res


def _angles(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Image angle vs mask angle (same method): MAE, bias, Bland-Altman, by level and side."""
    out: dict[str, Any] = {}
    pairs = [r for r in scored if r.get("img_angle_deg") is not None and r.get("mask_angle_deg") is not None]
    for lev in (*LEVELS, "all"):
        for side in ("L", "R", "all"):
            sel = [
                r
                for r in pairs
                if (lev == "all" or r["level"] == lev) and (side == "all" or r["side"] == side)
            ]
            if not sel:
                continue
            a = np.array([r["img_angle_deg"] for r in sel], dtype=np.float64)
            b = np.array([r["mask_angle_deg"] for r in sel], dtype=np.float64)
            groups = [r["patient"] for r in sel]
            vals = list(zip(a.tolist(), b.tolist(), strict=True))
            out[f"{lev}|{side}"] = {
                **continuous_summary(a, b),
                "mae_ci": patient_bootstrap(
                    groups, vals, lambda s: float(np.mean([abs(x - y) for x, y in s])) if s else float("nan")
                ),
                "img_mean_deg": float(a.mean()),
                "mask_mean_deg": float(b.mean()),
            }
    # tropism agreement per level: per patient |L-R| against 7 and 10 degrees, image vs mask (descriptive)
    trop: dict[str, Any] = {}
    by_pl: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for r in pairs:
        by_pl.setdefault((r["patient"], r["level"]), {})[r["side"]] = r
    for lev in LEVELS:
        diffs = [
            (
                abs(v["L"]["img_angle_deg"] - v["R"]["img_angle_deg"]),
                abs(v["L"]["mask_angle_deg"] - v["R"]["mask_angle_deg"]),
            )
            for (_, lv), v in by_pl.items()
            if lv == lev and "L" in v and "R" in v
        ]
        if not diffs:
            continue
        di = np.array([a for a, _ in diffs])
        dm = np.array([b for _, b in diffs])
        trop[lev] = {
            "n_patients": len(diffs),
            "img_diff_median_deg": round(float(np.median(di)), 2),
            "mask_diff_median_deg": round(float(np.median(dm)), 2),
            "agree_7deg": round(float(np.mean((di > 7) == (dm > 7))), 3),
            "agree_10deg": round(float(np.mean((di > 10) == (dm > 10))), 3),
            "img_positive_7deg": round(float(np.mean(di > 7)), 3),
            "mask_positive_7deg": round(float(np.mean(dm > 7)), 3),
            "img_positive_10deg": round(float(np.mean(di > 10)), 3),
            "mask_positive_10deg": round(float(np.mean(dm > 10)), 3),
        }
    out["tropism_by_level"] = trop
    return out


def _morphology(
    rows: list[dict[str, Any]], results: list[dict[str, Any]], geo_ref: list[dict[str, Any]]
) -> dict[str, Any]:
    """Weak labels (report facet arthropathy) vs the joint-line proxies, pre-registered rules."""
    from spinescope.config import DEFAULT

    notes = _notes_levels()
    # level-level records: mean over sides of the ratios (both sides measured)
    lv: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if r["status"] != "ok" or r.get(PRIMARY_RATIO_KEY) is None:
            continue
        lv.setdefault((r["patient"], r["level"]), {"len": [], "wid": []})
        lv[(r["patient"], r["level"])]["len"].append(r[PRIMARY_RATIO_KEY])
        if r.get("width_ratio") is not None:
            lv[(r["patient"], r["level"])]["wid"].append(r["width_ratio"])
    records: list[dict[str, Any]] = []
    for (pat, lev), v in lv.items():
        try:
            num = int(pat)
        except ValueError:
            continue
        ref = notes.get(num, {}).get(lev)
        if ref is None or len(v["len"]) < 2:
            continue
        lr = float(np.mean(v["len"]))
        wr = float(np.mean(v["wid"])) if v["wid"] else None
        records.append(
            {
                "patient": pat,
                "level": lev,
                "ref": bool(ref),
                "length_ratio": round(lr, 4),
                "width_ratio": None if wr is None else round(wr, 4),
                "pred_primary": lr >= DEFAULT.facets_morph_length_ratio,
                "pred_secondary": None if wr is None else wr <= 0.8,
            }
        )
    out: dict[str, Any] = {
        "reference_levels_with_label": len(records),
        "reference_positive": sum(1 for r in records if r["ref"]),
        "reference_negative": sum(1 for r in records if not r["ref"]),
        "note": "weak labels: report text, not the images",
    }
    for name, key in (("primary_length_ratio", "pred_primary"), ("secondary_width_ratio", "pred_secondary")):
        sel = [r for r in records if r[key] is not None]
        if not sel:
            out[name] = {"n": 0}
            continue
        pred = [bool(r[key]) for r in sel]
        ref_ = [r["ref"] for r in sel]
        metrics = binary_metrics(pred, ref_)
        vals = list(zip(pred, ref_, strict=True))

        def sens(s: list[tuple[Any, ...]]) -> float:
            tp = sum(1 for p, r in s if p and r)
            fn = sum(1 for p, r in s if not p and r)
            return tp / (tp + fn) if tp + fn else float("nan")

        def spec(s: list[tuple[Any, ...]]) -> float:
            tn = sum(1 for p, r in s if not p and not r)
            fp = sum(1 for p, r in s if p and not r)
            return tn / (tn + fp) if tn + fp else float("nan")

        out[name] = {
            **metrics,
            "sensitivity_ci": patient_bootstrap([r["patient"] for r in sel], vals, sens),
            "specificity_ci": patient_bootstrap([r["patient"] for r in sel], vals, spec),
        }
    out["records"] = records
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("calibrate", "validate"), required=True)
    ap.add_argument("--dev", type=int, default=0, help="calibrate only: first N patients (plumbing checks)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    res = run(args.split, args.dev, args.workers)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"facet_morph_{args.split}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=float)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
