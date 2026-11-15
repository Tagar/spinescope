#!/usr/bin/env python3
"""Validate vertebral geometry (bodies, endplates, heights, canal, disc margins) against SPIDER masks.

Truth is measured from the SPIDER label maps with the SAME functions the labeller uses (``body_plane_geometry``,
``centreline_from_points``, ``ap_chord_mm``, ``disc_margins_plane``), on the same plane grid as the prediction.
That compares like with like: any systematic effect of the measurement method cancels out, so the numbers
measure the labeller's segmentation and its geometry, not the method. Truth 3-D centres are the voxel centroids.

Only bodies whose two bounding discs carry the correct level names are scored (a shifted level would compare
different vertebrae). Missing or extra discs are reported, not scored.

Examples::

    # calibration (training split, subsample 60 volumes) and held-out report (validation, all 87)
    python tools/eval_vertebrae.py --split training --n 60 --jobs 3 --out /tmp/spinescope-lab15_vertebrae/train
    python tools/eval_vertebrae.py --split validation --jobs 3 --out /tmp/spinescope-lab15_vertebrae/val

Outputs (in ``--out``, outside the repository): ``cases.json`` (per-case errors) and ``summary.json``. Nothing
derived from SPIDER is committed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

SPIDER_DIR = Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public")) / "spider"
CANAL_LABEL = 100
DISC_BASE = 200  # disc 200+i lies between vertebra i (above) and i-1 (below)
MATCH_MM = 12.0

log = logging.getLogger("eval_vertebrae")


# --------------------------------------------------------------------------- truth volume (LPS)
class TruthVolume:
    """A SPIDER label map with its own LPS geometry (SimpleITK header), sampled at LPS points."""

    def __init__(self, path: Path) -> None:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        self.arr = sitk.GetArrayFromImage(img)  # [z, y, x]
        self.origin = np.array(img.GetOrigin(), dtype=np.float64)
        spacing = np.array(img.GetSpacing(), dtype=np.float64)
        direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
        self.m = direction * spacing[None, :]  # column j = axis j direction times its spacing
        self.minv = np.linalg.inv(self.m)
        self.shape = self.arr.shape

    def lps_of_voxels(self, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        zz, yy, xx = np.nonzero(mask)
        idx = np.stack([xx, yy, zz], axis=1).astype(np.float64)
        return np.asarray(self.origin + idx @ self.m.T, dtype=np.float64)

    def labels_at(self, lps: NDArray[np.float64]) -> NDArray[np.int64]:
        """Label at each LPS point (nearest voxel; -1 outside the volume)."""
        p = lps.reshape(-1, 3) - self.origin
        idx = np.rint(p @ self.minv.T).astype(np.int64)  # (x, y, z)
        x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]
        ok = (z >= 0) & (z < self.shape[0]) & (y >= 0) & (y < self.shape[1]) & (x >= 0) & (x < self.shape[2])
        out = np.full(p.shape[0], -1, dtype=np.int64)
        out[ok] = self.arr[z[ok], y[ok], x[ok]]
        return out.reshape(lps.shape[:-1])


def _sample_labels_on_grid(vol: TruthVolume, grid: Any, labels: set[int]) -> NDArray[np.bool_]:
    h, w = grid.shape
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    down, post = grid.mm_of_index(ii, jj)
    lab = vol.labels_at(grid.to_lps(down, post))
    return np.isin(lab, np.array(sorted(labels)))


# --------------------------------------------------------------------------- one case
def _case_truth(vol: TruthVolume) -> dict[str, Any]:
    """Body and disc truths in 3-D (LPS voxel centroids), canal points."""
    labels = np.unique(vol.arr)
    bodies = {int(v): vol.lps_of_voxels(vol.arr == v) for v in labels if 1 <= v <= 25}
    discs = {int(v): vol.lps_of_voxels(vol.arr == v) for v in labels if DISC_BASE + 1 <= v <= DISC_BASE + 25}
    if CANAL_LABEL in labels:
        cm = vol.arr == CANAL_LABEL
        canal = vol.lps_of_voxels(cm)
        canal_ids = np.nonzero(cm)[0].astype(np.intp)  # slice index of each canal voxel (z index)
    else:
        canal, canal_ids = np.zeros((0, 3)), np.zeros(0, dtype=np.intp)
    return {"bodies": bodies, "discs": discs, "canal": canal, "canal_ids": canal_ids}


def _score_case(name: str, meta: dict[str, str], pred: dict[str, Any], truth_path: Path) -> dict[str, Any]:
    """Compare one prediction (``Anchors.to_dict()`` after ``attach_geometry``) with the SPIDER mask."""
    from spinescope.anatomy.canal_ml import largest_component
    from spinescope.anatomy.vertebrae import body_plane_geometry
    from spinescope.config import DEFAULT

    vol = TruthVolume(truth_path)
    tr = _case_truth(vol)
    out: dict[str, Any] = {"case": name, **meta, "bodies": [], "canal": {}, "discs": [], "notes": []}
    dv = pred.get("discs", {})
    # Disc matching: nearest truth disc centre in the sagittal (y, z) plane, within MATCH_MM, one-to-one.
    order = sorted(dv, key=lambda lab: dv[lab]["center"][2])  # bottom-up
    tlabs = sorted(tr["discs"])
    tmatch: dict[str, int] = {}
    used: set[int] = set()
    for lab in order:
        c = np.asarray(dv[lab]["center"])[1:]
        best = None
        for tl in tlabs:
            if tl in used:
                continue
            t_c = tr["discs"][tl].mean(axis=0)[1:]
            d = float(np.linalg.norm(c - t_c))
            if d <= MATCH_MM and (best is None or d < best[0]):
                best = (d, tl)
        if best is not None:
            tmatch[lab] = best[1]
            used.add(best[1])
    out["n_pred_discs"] = len(order)
    out["n_truth_discs"] = len(tlabs)
    # Name-aligned: disc at bottom-up index i matches truth label 201+i.
    aligned = {lab: tmatch.get(lab) == DISC_BASE + 1 + i for i, lab in enumerate(order)}
    out["discs_name_aligned"] = int(sum(aligned.values()))

    # Pick the plane grid of each body / disc from the prediction's own slice.
    n = len(order)
    for i, lab in enumerate(order):
        d = dv[lab]
        if "anterior" not in d or d.get("anterior") is None or d.get("posterior") is None:
            continue
        # mid-sagittal plane grid: the slice whose x is nearest the disc's margins (all margins lie on it)
        mt = tmatch.get(lab)
        if mt is None:
            continue
        grid = _grid_nearest_x(vol, np.asarray(d["anterior"]), pred)
        if grid is None:
            continue
        msk = largest_component(_sample_labels_on_grid(vol, grid, {mt}))
        mg = _disc_margins(msk, grid)
        if mg is None:
            continue
        ant_t, post_t = mg
        ant_p = np.asarray(d["anterior"])
        post_p = np.asarray(d["posterior"])
        out["discs"].append(
            {
                "label_rank": i,
                "name_aligned": bool(aligned[lab]),
                "ant_err_mm": float(np.linalg.norm(ant_p - ant_t)),
                "post_err_mm": float(np.linalg.norm(post_p - post_t)),
            }
        )

    bodies = pred.get("vertebrae", {})
    stack_normal = np.asarray(pred.get("_normal", [1.0, 0.0, 0.0]), dtype=np.float64)
    for name, v in bodies.items():
        b = v.get("body")
        if not b:
            continue
        j = int(b["discs_below"])
        ok = (j == 0 or aligned_below(aligned, order, j)) and (j == n or aligned_above(aligned, order, j))
        rec: dict[str, Any] = {"level": name, "j": j, "name_aligned": bool(ok)}
        if not ok:
            out["bodies"].append(rec)
            continue
        tb = tr["bodies"].get(j)
        if tb is None or tb.shape[0] == 0:
            rec["note"] = "no truth body"
            out["bodies"].append(rec)
            continue
        t_centre = tb.mean(axis=0)
        rec["centre3d_err_mm"] = float(np.linalg.norm(np.asarray(b["centre"]) - t_centre))
        rec["centre_inplane_err_mm"] = float(np.linalg.norm(np.asarray(b["centre"])[1:] - t_centre[1:]))
        rec["centre_sup_inf_err_mm"] = float(
            np.asarray(b["centre"])[2] - t_centre[2]
        )  # signed, +: too superior
        t_lat = tb @ stack_normal
        if b.get("width_mm") is not None:
            rec["width_err_mm"] = float(
                b["width_mm"] - (np.percentile(t_lat, 99.5) - np.percentile(t_lat, 0.5))
            )
        # Mid-sagittal (plane) geometry on the same slice grid as the prediction.
        kb = int(b["slice"])
        grid = plane_grid_for(vol, pred, kb)
        if grid is not None and b.get("centre_plane") is not None:
            mt = largest_component(_sample_labels_on_grid(vol, grid, {j}))
            sup, inf = j < n, j >= 1
            tg = body_plane_geometry(mt, grid, sup, inf, DEFAULT) if mt.any() else None
            if tg is not None:
                rec.update(_plane_errors(b, tg, grid))
        out["bodies"].append(rec)

    # Canal: centreline at predicted heights, AP diameters on the predicted slices.
    canal = pred.get("canal") or {}
    if canal and canal.get("points"):
        tpts = tr["canal"]
        out["canal"] = _canal_errors(canal, tpts, tr["canal_ids"], vol, pred)
    return out


def aligned_below(aligned: dict[str, bool], order: list[str], j: int) -> bool:
    return bool(aligned[order[j - 1]])


def aligned_above(aligned: dict[str, bool], order: list[str], j: int) -> bool:
    return bool(aligned[order[j]])


def _plane_errors(b: dict[str, Any], tg: dict[str, Any], grid: Any) -> dict[str, Any]:
    """Errors of the mid-sagittal body geometry (prediction minus truth; ``*_err_*`` absolute, ``signed`` keys signed)."""
    out: dict[str, Any] = {}
    for key, label in (("superior_endplate", "sup"), ("inferior_endplate", "inf")):
        if b.get(key) and label in tg:
            d = float(b[key]["angle_deg"] - tg[label]["angle_deg"])
            out[f"{label}_angle_err_deg"] = abs(d)
            out[f"{label}_angle_signed_deg"] = d
    if b.get("height_mm") and "height_mm" in tg:
        for third in ("anterior", "middle", "posterior"):
            d = float(b["height_mm"][third] - tg["height_mm"][third])
            out[f"height_{third}_err_mm"] = abs(d)
            out[f"height_{third}_signed_mm"] = d
        d = float(b["height_mm"]["mean"] - np.mean(list(tg["height_mm"].values())))
        out["height_mean_err_mm"] = abs(d)
        out["height_mean_signed_mm"] = d
    if b.get("ap_depth_mm") is not None:
        d = float(b["ap_depth_mm"] - tg["ap_mm"])
        out["ap_err_mm"] = abs(d)
        out["ap_signed_mm"] = d
    return out


def _canal_errors(
    canal: dict[str, Any],
    tpts: NDArray[np.float64],
    tids: NDArray[np.intp],
    vol: TruthVolume,
    pred: dict[str, Any],
) -> dict[str, Any]:
    from spinescope.anatomy.canal_ml import ap_chord_mm, centreline_from_points, largest_component
    from spinescope.config import DEFAULT

    res: dict[str, Any] = {"centreline_err_mm": [], "ap_err_mm": [], "ap_level": {}}
    pts = np.asarray(canal["points"], dtype=np.float64)
    if tpts.size and pts.size:
        tc = np.asarray(
            centreline_from_points(tpts, float(pts[:, 2].min()), float(pts[:, 2].max()), DEFAULT, tids)
        )
        if tc.size:
            tz = tc[:, 2][::-1]
            tx = tc[:, 0][::-1]
            ty = tc[:, 1][::-1]
            for p in pts:
                if p[2] < tz.min() or p[2] > tz.max():
                    continue
                x_t = float(np.interp(p[2], tz, tx))
                y_t = float(np.interp(p[2], tz, ty))
                res["centreline_err_mm"].append(float(np.hypot(p[0] - x_t, p[1] - y_t)))
    for lab, v in (canal.get("per_level") or {}).items():
        if v.get("ap_mm") is None or v.get("slice") is None:
            continue
        grid = plane_grid_for(vol, pred, int(v["slice"]))
        if grid is None:
            continue
        tmask = largest_component(_sample_labels_on_grid(vol, grid, {CANAL_LABEL}))
        t_ap = ap_chord_mm(tmask, grid, float(v["z_mm"]), DEFAULT.canal_level_band_mm)
        if t_ap is None:
            continue
        err = float(v["ap_mm"] - t_ap)  # signed: + means the prediction is wider
        res["ap_err_mm"].append(err)
        res["ap_level"][lab] = {
            "kind": v["kind"],
            "pred_ap_mm": v["ap_mm"],
            "truth_ap_mm": t_ap,
            "err_mm": err,
        }
    return res


# --------------------------------------------------------------------------- plane grids
def plane_grid_for(vol: TruthVolume, pred: dict[str, Any], k: int) -> Any:
    """The prediction's grid for slice ``k``: rebuilt from the stored geometry of the sagittal stack."""
    geo = pred.get("_stack")
    if geo is None:
        return None
    return _grid_from_geo(geo, k)


def _grid_from_geo(geo: Any, k: int) -> Any:
    from spinescope.anatomy.ml.grid import plane_grid

    return plane_grid(geo, k, 1.0)


def _grid_nearest_x(vol: TruthVolume, anterior_lps: NDArray[np.float64], pred: dict[str, Any]) -> Any:
    geo = pred.get("_stack")
    if geo is None:
        return None
    x = float(anterior_lps @ geo.normal)
    k = int(np.argmin(np.abs(geo.positions - x)))
    return _grid_from_geo(geo, k)


def _disc_margins(
    mask: NDArray[np.bool_], grid: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Anterior and posterior margins (LPS) at mid-height of one disc on a plane, same rule as the labeller."""
    ii, jj = np.nonzero(mask)
    if ii.size < 3:
        return None
    down, post = grid.mm_of_index(ii, jj)
    pts = np.stack([down, post], axis=1)
    centre = pts.mean(axis=0)
    rel = pts - centre
    _w, vecs = np.linalg.eigh(np.cov(rel.T))
    minor, major = vecs[:, 0], vecs[:, 1]
    if minor[0] > 0:
        minor = -minor
    if major[1] < 0:
        major = -major
    s = rel @ minor
    a = rel @ major
    height = float(s.max() - s.min()) + grid.res_mm
    band = np.abs(s) <= max(height / 4.0, grid.res_mm)
    a_band = a[band] if band.any() else a
    ant = centre + float(a_band.min()) * major
    post_pt = centre + float(a_band.max()) * major
    return grid.to_lps(ant[0], ant[1]), grid.to_lps(post_pt[0], post_pt[1])


# --------------------------------------------------------------------------- driver
def _predict(path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Labeller output for one volume: the learned labeller, then the geometry hook (as label_spine does).

    ``overrides`` replace :class:`~spinescope.config.Config` fields (calibration sweeps only).
    """
    from dataclasses import replace

    from spinescope import load_study
    from spinescope.anatomy.ml import label_spine_ml
    from spinescope.anatomy.vertebrae import attach_geometry
    from spinescope.config import DEFAULT

    cfg = replace(DEFAULT, **overrides) if overrides else DEFAULT
    study = load_study(path)
    anchors = label_spine_ml(study, cfg)
    attach_geometry(study, anchors, cfg)
    d = anchors.to_dict()
    stack = study.best("sag")
    d["_stack"] = stack  # in-process only; never written to JSON
    d["_normal"] = [float(v) for v in stack.normal]
    return d


def _work(args: tuple[str, dict[str, str], str, dict[str, Any]]) -> dict[str, Any]:
    name, meta, path, overrides = args
    try:
        pred = _predict(path, overrides)
        res = _score_case(name, meta, pred, SPIDER_DIR / "masks" / f"{name}.mha")
        res.pop("_stack", None)
        return res
    except Exception as exc:  # one bad case must not stop the run
        return {"case": name, "error": f"{type(exc).__name__}: {exc}", **meta}


def _cases(split: str, n: int | None) -> list[tuple[str, dict[str, str], str]]:
    rows = []
    with open(SPIDER_DIR / "overview.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("new_file_name") or "").strip()
            if row.get("subset", "").strip() != split or not name:
                continue
            img = SPIDER_DIR / "images" / f"{name}.mha"
            if not img.is_file() or not (SPIDER_DIR / "masks" / f"{name}.mha").is_file():
                continue
            try:
                field_t = f"{float(row.get('MagneticFieldStrength') or 'nan'):.1f} T"
            except ValueError:
                field_t = "unknown"
            meta = {
                "vendor": (row.get("Manufacturer") or "").split(" ")[0] or "unknown",
                "field": field_t,
                "sequence": name.split("_", 1)[1] if "_" in name else name,
            }
            rows.append((name, meta, str(img)))
    rows.sort(key=lambda r: (int(r[0].split("_")[0]), r[0]))
    if n and n < len(rows):
        idx = sorted({round(x) for x in np.linspace(0, len(rows) - 1, n)})
        rows = [rows[i] for i in idx]
    return rows


def _summ(vals: list[float]) -> dict[str, float | int | None]:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "median": None, "p90": None, "mean": None}
    return {
        "n": int(a.size),
        "median": round(float(np.median(a)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "mean": round(float(a.mean()), 3),
    }


def summarise(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [c for c in cases if "error" not in c]
    bodies = [b for c in ok for b in c["bodies"] if b.get("name_aligned") and "centre3d_err_mm" in b]
    keys_b = [
        "centre3d_err_mm",
        "centre_inplane_err_mm",
        "sup_angle_err_deg",
        "inf_angle_err_deg",
        "height_anterior_err_mm",
        "height_middle_err_mm",
        "height_posterior_err_mm",
        "height_mean_err_mm",
        "ap_err_mm",
        "width_err_mm",
    ]
    signed_b = [
        "height_mean_signed_mm",
        "height_anterior_signed_mm",
        "height_middle_signed_mm",
        "height_posterior_signed_mm",
        "ap_signed_mm",
        "centre_sup_inf_err_mm",
    ]
    discs = [d for c in ok for d in c["discs"] if d.get("name_aligned")]
    canal_c = [e for c in ok for e in c["canal"].get("centreline_err_mm", [])]
    canal_ap = [e for c in ok for e in c["canal"].get("ap_err_mm", [])]
    canal_levels = [v for c in ok for v in (c["canal"].get("ap_level") or {}).values()]
    out: dict[str, Any] = {
        "volumes": len(cases),
        "volumes_scored": len(ok),
        "error_count": sum(1 for c in cases if "error" in c),
        "errors": [c for c in cases if "error" in c][:5],
        "bodies_scored": len(bodies),
        "bodies": {k: _summ([b.get(k) for b in bodies]) for k in keys_b},
        "bodies_signed": {k: _summ([b.get(k) for b in bodies]) for k in signed_b},
        "discs_scored": len(discs),
        "disc_margin_anterior": _summ([d["ant_err_mm"] for d in discs]),
        "disc_margin_posterior": _summ([d["post_err_mm"] for d in discs]),
        "canal_centreline": _summ(canal_c),
        "canal_ap_signed": _summ(canal_ap),
        "canal_ap_abs": _summ([abs(e) for e in canal_ap]),
        "canal_ap_by_kind": {
            kind: _summ([abs(v["err_mm"]) for v in canal_levels if v["kind"] == kind])
            for kind in ("disc", "mid_body")
        },
        "groups": {},
    }
    # By acquisition. SPIDER's num_vertebrae counts all labelled vertebrae (6-9), so it does not mark transitional
    # anatomy and is not used for grouping.
    for key in ("vendor", "field", "sequence"):
        for g in sorted({str(c.get(key, "?")) for c in ok}):
            sub = [c for c in ok if str(c.get(key, "?")) == g]
            out["groups"][f"{key}={g}"] = {
                "volumes": len(sub),
                "body_centre3d_err_mm": _summ(
                    [
                        b["centre3d_err_mm"]
                        for c in sub
                        for b in c["bodies"]
                        if b.get("name_aligned") and "centre3d_err_mm" in b
                    ]
                ),
                "sup_angle_err_deg": _summ(
                    [
                        b["sup_angle_err_deg"]
                        for c in sub
                        for b in c["bodies"]
                        if b.get("name_aligned") and "sup_angle_err_deg" in b
                    ]
                ),
                "canal_centreline_mm": _summ(
                    [e for c in sub for e in c["canal"].get("centreline_err_mm", [])]
                ),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["training", "validation"])
    ap.add_argument("--n", type=int, default=None, help="subsample this many volumes (evenly spaced)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--set", action="append", default=[], metavar="FIELD=VALUE", help="Config override (sweeps)"
    )
    args = ap.parse_args(argv)
    overrides = {}
    for item in args.set:
        key, _, val = item.partition("=")
        overrides[key] = json.loads(val)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("spinescope").setLevel(logging.ERROR)
    cases = [(n, m, p, overrides) for n, m, p in _cases(args.split, args.n)]
    print(f"{len(cases)} {args.split} volumes", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(ex.map(_work, cases)):
            results.append(res)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(cases)}", file=sys.stderr)
    (args.out / "cases.json").write_text(json.dumps(results, indent=1, default=float))
    if overrides:
        (args.out / "overrides.json").write_text(json.dumps(overrides, indent=1))
    summ = summarise(results)
    (args.out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
