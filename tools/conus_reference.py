#!/usr/bin/env python3
"""Reference set for the conus locator: blinded crops for an eye mark, then the comparison with the locator.

Two steps, in this order (docs/labeling/lab26_conus_v2.md):

1. ``render``: writes one canal crop per case (sagittal slice nearest the midline, the learned disc lines with
   their level names, a height ruler). No locator output is drawn. The marks are typed into
   ``docs/labeling/lab26_reference_marks.csv`` by hand before the locator is run on these cases.
2. ``score``: runs ``label_spine_details`` on the same cases and compares its conus with the marks.

Crops and manifests go to the scratch folder given by ``--out``; no image is committed.

Examples::

    python tools/conus_reference.py render --out /tmp/spinescope-lab26_conus_v2/ref
    python tools/conus_reference.py score --out /tmp/spinescope-lab26_conus_v2/ref
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from spinescope.anatomy.conus import pick_conus_stack
from spinescope.anatomy.ml.grid import plane_grid, resample_slice
from spinescope.anatomy.ml.labeller import CANAL, label_spine_ml, segment_stack
from spinescope.config import DEFAULT

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conus_audit as audit

MARKS = Path(__file__).resolve().parent.parent / "docs" / "labeling" / "lab26_reference_marks.csv"
PX_PER_MM = 4
CROP_AP_MM = 60.0  # field width centred on the canal
RULER_STEP_MM = 10.0


def _cases() -> list[tuple[str, str]]:
    return audit.spider_cases("validation", 40, 16) + audit.sudirman_cases(20, 16)


def _canal_centre_ap(seg: Any, k: int, grid: Any) -> float | None:
    """Posterior mm of the canal centroid on slice ``k`` (None if no canal)."""
    m = seg.probs[k, CANAL] >= DEFAULT.ml_prob_threshold
    if not m.any():
        return None
    _ii, jj = np.nonzero(m)
    return float(grid.mm_of_index(np.array([0.0]), np.array([jj.mean()]))[1][0])


def render_case(name: str, ds: str, out: Path) -> dict[str, Any]:
    """Blinded crop for one case: midline slice, disc lines (learned labeller, no conus), height ruler."""
    study = audit.load_case(name, ds)
    stack = pick_conus_stack(study)
    if stack is None:
        return {"case": name, "dataset": ds, "error": "no sagittal series"}
    anchors = label_spine_ml(study)  # discs and numbering only: the conus hook is not run
    seg = segment_stack(stack, DEFAULT)
    areas = [float((p[CANAL] >= DEFAULT.ml_prob_threshold).sum()) for p in seg.probs]
    k = int(np.argmax(areas)) if areas and max(areas) > 0 else stack.n // 2
    grid = seg.grids[k]
    img = resample_slice(stack, grid)
    lo, hi = np.percentile(img[img > 0], [1, 99.5]) if np.any(img > 0) else (0.0, 1.0)
    img8 = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(img8).convert("RGB")
    discs = anchors.discs
    if not discs:
        return {"case": name, "dataset": ds, "error": "no discs"}
    zc = [float(d.center[2]) for d in discs.values()]
    z_top, z_bot = max(zc) + 30.0, min(zc) - 25.0
    r0 = int(np.clip(_row_of_z(grid, z_top), 0, grid.shape[0] - 1))
    r1 = int(np.clip(_row_of_z(grid, z_bot), 0, grid.shape[0] - 1))
    lo_r, hi_r = min(r0, r1), max(r0, r1)
    ap = _canal_centre_ap(seg, k, grid)
    if ap is None:
        ap = float(np.mean([grid.mm_of(np.asarray(d.center)[None, :])[0][1] for d in discs.values()])) + 25.0
    j_c = round(float(grid.index_of_mm(np.array([0.0]), np.array([ap]))[1][0]))
    half = int(CROP_AP_MM / grid.res_mm / 2)
    c0, c1 = max(0, j_c - half), min(grid.shape[1], j_c + half)
    crop = pil.crop((c0, lo_r, c1, hi_r + 1))
    crop = crop.resize((crop.width * PX_PER_MM, crop.height * PX_PER_MM), Image.Resampling.BILINEAR)
    ruler_w = 70
    canvas = Image.new("RGB", (crop.width + ruler_w, crop.height), (0, 0, 0))
    canvas.paste(crop, (ruler_w, 0))
    draw = ImageDraw.Draw(canvas)
    z_of_row = lambda r: _z_of_row(grid, r)  # noqa: E731

    def y_of(z: float) -> float:
        return (_row_of_z(grid, z) - lo_r) * PX_PER_MM

    zmin = z_of_row(hi_r)
    zmax = z_of_row(lo_r)
    t = np.ceil(zmin / RULER_STEP_MM) * RULER_STEP_MM
    while t <= zmax:
        y = y_of(t)
        draw.line([(ruler_w - 10, y), (ruler_w, y)], fill=(255, 255, 255))
        draw.text((4, y - 6), f"{t:.0f} mm", fill=(255, 255, 255))
        t += RULER_STEP_MM
    for lab, d in discs.items():
        y = y_of(float(d.center[2]))
        draw.line([(ruler_w, y), (canvas.width, y)], fill=(90, 200, 255), width=1)
        draw.text((ruler_w + 4, y - 12), lab, fill=(90, 200, 255))
    draw.text((ruler_w + 4, 4), f"{ds} {name}  z in mm (LPS)", fill=(255, 255, 255))
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{ds}_{name}.png"
    canvas.save(out / fname)
    return {
        "case": name,
        "dataset": ds,
        "png": fname,
        "contrast": (stack.meta.get("role", {}) or {}).get("contrast"),
        "slice_k": k,
        "discs": anchors.disc_labels(),
        "disc_z": {lab: round(float(d.center[2]), 2) for lab, d in discs.items()},
        "z_top_mm": round(zmax, 2),
        "z_bottom_mm": round(zmin, 2),
    }


def _row_of_z(grid: Any, z: float) -> float:
    down, post = grid.mm_of(np.array([0.0, 0.0, z]))
    i, _j = grid.index_of_mm(down, post)
    return float(i)


def _z_of_row(grid: Any, r: float) -> float:
    xyz = grid.lps_of_index(float(r), 0.0)
    return float(xyz[2])


def render(out: Path) -> int:
    """Blinded crops for all reference cases plus the manifest (no locator output)."""
    rows = []
    for name, ds in _cases():
        t0 = time.perf_counter()
        try:
            rec = render_case(name, ds, out)
        except Exception as exc:  # a failing case stays in the manifest
            rec = {"case": name, "dataset": ds, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
        rows.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("case", "dataset", "elapsed_s", "error")}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(rows, indent=1))
    return 0


def run_case(name: str, ds: str) -> dict[str, Any]:
    """The full labeller (with the conus hook) on one reference case: the conus output under test."""
    from spinescope.anatomy.pipeline import label_spine_details

    study = audit.load_case(name, ds)
    det = label_spine_details(study)
    anchors = det.anchors
    contrast = (pick_conus_stack(study).meta.get("role", {}) or {}).get("contrast")
    return {
        "case": name,
        "dataset": ds,
        "contrast": contrast,
        "numbering_status": anchors.numbering.get("status"),
        "discs": anchors.disc_labels(),
        "disc_z": {lab: round(float(d.center[2]), 2) for lab, d in anchors.discs.items()},
        "conus": anchors.conus,
        "conus_warnings": [w for w in anchors.warnings if "conus" in w.lower()],
    }


def run(out: Path) -> int:
    """Run the locator on every reference case; writes ``results.json`` (no images)."""
    rows = []
    for name, ds in _cases():
        t0 = time.perf_counter()
        try:
            rec = run_case(name, ds)
        except Exception as exc:
            rec = {"case": name, "dataset": ds, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
        rows.append(rec)
        print(
            json.dumps(
                {"case": rec["case"], "conus_z": (rec.get("conus") or {}).get("z"), "error": rec.get("error")}
            ),
            flush=True,
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    return 0


HALF_VERTEBRA_MM = 13.0  # half a vertebral body (body height ~26 mm, phantom default; anatomy, not tuned)


def read_marks(path: Path) -> dict[str, tuple[float | None, str]]:
    """Case key (``spider_2_t2`` / ``sudirman_0006``) -> (tip z or None, quality)."""
    out: dict[str, tuple[float | None, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            z = (row.get("tip_z_mm") or "").strip()
            out[row["case"].strip()] = (float(z) if z else None, (row.get("quality") or "").strip())
    return out


def score(out: Path, marks: Path, results: Path | None = None) -> int:
    """Compare the locator's conus with the blinded marks: located, error (mm), level agreement, calibration."""
    from spinescope.anatomy.anchors import Disc
    from spinescope.anatomy.conus import level_at

    rows = json.loads((results or (out / "results.json")).read_text())
    mk = read_marks(marks)
    per = []
    for r in rows:
        key = f"{r['dataset']}_{r['case']}"
        if key not in mk:
            continue
        z_mark, quality = mk[key]
        c = r.get("conus") or {}
        z_loc = c.get("z")
        rec = {
            "key": key,
            "mark_z": z_mark,
            "quality": quality,
            "located": z_loc is not None,
            "loc_z": z_loc,
            "loc_level": c.get("level"),
            "conf": c.get("confidence"),
            "error_mm": None,
            "level_agree": None,
            "mark_level": None,
            "within_half_vertebra": None,
        }
        if z_mark is not None and z_loc is not None:
            rec["error_mm"] = round(abs(float(z_loc) - z_mark), 2)
            rec["within_half_vertebra"] = rec["error_mm"] <= HALF_VERTEBRA_MM
        if z_mark is not None and r.get("disc_z"):
            from spinescope.anatomy.anchors import Anchors

            anc = Anchors(
                discs={
                    lab: Disc(center=[0.0, 0.0, zc], normal=[0.0, 0.0, 1.0], height_mm=10.0)
                    for lab, zc in r["disc_z"].items()
                }
            )
            rec["mark_level"] = level_at(z_mark, anc)[0]
            if z_loc is not None:
                rec["level_agree"] = rec["mark_level"] == c.get("level")
        per.append(rec)
    summary = {
        "n_cases": len(per),
        "n_with_mark_tip": sum(1 for x in per if x["mark_z"] is not None),
        "n_not_visible": sum(1 for x in per if x["mark_z"] is None),
        "located_total": sum(1 for x in per if x["located"]),
        "located_on_tip_marks": sum(1 for x in per if x["located"] and x["mark_z"] is not None),
        "located_on_not_visible": sum(1 for x in per if x["located"] and x["mark_z"] is None),
        "abstained_on_tip_marks": sum(1 for x in per if not x["located"] and x["mark_z"] is not None),
        "within_half_vertebra_of_located_tip_marks": sum(1 for x in per if x["within_half_vertebra"]),
        "level_agree_of_located_tip_marks": sum(1 for x in per if x["level_agree"]),
    }
    located_tip = [x for x in per if x["located"] and x["mark_z"] is not None]
    if located_tip:
        summary["accuracy_within_half_vertebra_on_located"] = round(
            sum(1 for x in located_tip if x["within_half_vertebra"]) / len(located_tip), 3
        )
        summary["level_agreement_on_located"] = round(
            sum(1 for x in located_tip if x["level_agree"]) / len(located_tip), 3
        )
        errs = sorted(x["error_mm"] for x in located_tip)
        summary["median_error_mm"] = errs[len(errs) // 2]
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps({"summary": summary, "per_case": per}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", type=Path, required=True)
    ph = sub.add_parser("phantom", help="locate on phantoms with known conus height")
    ph.add_argument("--out", type=Path, required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument("--marks", type=Path, default=MARKS)
    sc.add_argument(
        "--results", type=Path, default=None, help="results.json to score (default: OUT/results.json)"
    )
    a = ap.parse_args(argv)
    if a.cmd == "render":
        return render(a.out)
    if a.cmd == "run":
        return run(a.out)
    if a.cmd == "phantom":
        return phantom_eval(a.out)
    return score(a.out, a.marks, a.results)


# ------------------------------------------------------------------------------ phantom ground truth
def phantom_cases() -> list[tuple[str, dict[str, Any]]]:
    """Phantom specs with a known conus height: cartoon and realistic, T2 and T1, geometry and noise variants."""
    from spinescope.testing.phantom import PhantomSpec

    return [
        ("cartoon_default_t2", {"spec": PhantomSpec(), "contrast": "t2"}),
        ("cartoon_default_t1", {"spec": PhantomSpec(), "contrast": "t1"}),
        ("cartoon_noise06_t2", {"spec": PhantomSpec(noise=0.06, seed=3), "contrast": "t2"}),
        ("cartoon_conus_T12_t2", {"spec": PhantomSpec(conus_body="T12"), "contrast": "t2"}),
        ("cartoon_conus_L3_t2", {"spec": PhantomSpec(conus_body="L3"), "contrast": "t2"}),
        ("cartoon_conus_L2_t2", {"spec": PhantomSpec(conus_body="L2"), "contrast": "t2"}),
        ("realistic_default_t2", {"spec": PhantomSpec(realism="realistic"), "contrast": "t2"}),
        ("realistic_default_t1", {"spec": PhantomSpec(realism="realistic"), "contrast": "t1"}),
        (
            "realistic_conus_L3_t2",
            {"spec": PhantomSpec(realism="realistic", conus_body="L3"), "contrast": "t2"},
        ),
    ]


def phantom_eval(out: Path) -> int:
    """Locate the conus on phantoms with an oracle canal segmentation (the phantom labels), against ``conus_z``."""
    from spinescope.anatomy.anchors import Anchors
    from spinescope.anatomy.conus import conus_from_segmentation
    from spinescope.testing.phantom import CORD, CSF, NERVE, SpinePhantom

    rows = []
    for name, case in phantom_cases():
        ph = SpinePhantom(case["spec"])
        stack = ph.render("sag", case["contrast"])
        seg = _oracle_segmentation(ph, stack, CORD, CSF, NERVE)
        anchors = Anchors.from_truth(ph.truth)
        res = conus_from_segmentation(seg, anchors, DEFAULT, contrast=case["contrast"])
        truth = float(ph.conus_z)
        z = res.z
        rows.append(
            {
                "phantom": name,
                "truth_z": round(truth, 2),
                "found": res.found,
                "z": None if z is None else round(float(z), 2),
                "error_mm": None if z is None else round(abs(float(z) - truth), 2),
                "level": res.level,
                "confidence": round(res.confidence, 3),
                "warnings": res.warnings,
            }
        )
        print(
            json.dumps(
                {k: rows[-1][k] for k in ("phantom", "truth_z", "z", "error_mm", "level", "confidence")}
            ),
            flush=True,
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "phantom.json").write_text(json.dumps(rows, indent=1))
    return 0


def _oracle_segmentation(ph: Any, stack: Any, cord: int, csf: int, nerve: int) -> Any:
    """Oracle canal (thecal sac, cord and roots) and body/disc classes from the phantom labels, on the 1 mm grid."""
    from spinescope.anatomy.ml.grid import intensity_window
    from spinescope.anatomy.ml.labeller import BODY, CANAL, DISC, SegmentedStack
    from spinescope.testing.phantom import ANNULUS, CORTEX, MARROW, MODIC1, NUCLEUS

    to_class = {
        MARROW: BODY,
        CORTEX: BODY,
        MODIC1: BODY,
        ANNULUS: DISC,
        NUCLEUS: DISC,
        csf: CANAL,
        cord: CANAL,
        nerve: CANAL,
    }
    grids = [plane_grid(stack, k, 1.0) for k in range(stack.n)]
    h, w = grids[0].shape
    probs = np.zeros((stack.n, 4, h, w), np.float32)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for k, g in enumerate(grids):
        tissue = ph.labels(g.lps_of_index(ii, jj).reshape(-1, 3)).reshape(h, w)
        cls = np.zeros((h, w), np.int64)
        for t, c in to_class.items():
            cls[tissue == t] = c
        for c in range(4):
            probs[k, c] = cls == c
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=intensity_window(stack.vol))


if __name__ == "__main__":
    sys.exit(main())
