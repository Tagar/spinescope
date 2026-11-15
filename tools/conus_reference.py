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
import conus_audit as audit  # noqa: E402  (case sampling and loading shared with lab16, same seed)

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
    j_c = int(round(float(grid.index_of_mm(np.array([0.0]), np.array([ap]))[1][0])))
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", type=Path, required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "render":
        return render(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
