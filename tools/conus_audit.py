#!/usr/bin/env python3
"""Visual audit of the conus locator on public data: run ``label_spine`` and render the canal with the tip.

There is no conus ground truth in SPIDER, Sudirman or qx5rt, so the audit is a visual read by an AI reader:
each PNG shows the sagittal slice the conus was located on, the disc lines, and the detected conus tip.
The grades are written by hand afterwards (docs/labeling/lab16_conus.md). Nothing here is committed as an
image; outputs go to the scratch folder given by ``--out``.

Examples::

    python tools/conus_audit.py spider --split validation --n 40 --seed 16 --out /tmp/spinescope-lab16_conus/spider
    python tools/conus_audit.py sudirman --n 20 --seed 16 --out /tmp/spinescope-lab16_conus/sudirman
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from spinescope import load_study
from spinescope.anatomy.conus import pick_conus_stack
from spinescope.anatomy.ml.grid import plane_grid, resample_slice
from spinescope.anatomy.pipeline import label_spine_details
from spinescope.config import DEFAULT
from spinescope.datasets.spider import spider_root
from spinescope.datasets.sudirman import patient_sessions, sudirman_root

SPIDER_ROOT = spider_root()  # $SPINESCOPE_DATA/spider
SUDIRMAN_ROOT = sudirman_root()  # $SPINESCOPE_DATA/lumbar_spine_mri_sudirman
PX_PER_MM = 4
CROP_HALF_AP_MM = 35.0


def spider_cases(split: str, n: int, seed: int) -> list[tuple[str, str]]:
    """Seeded sample of SPIDER sagittal T2 volumes of one split (plain T2 and T2 SPACE): (file key, 'spider').

    The file key is the overview name (e.g. ``101_t2`` or ``107_t2_SPACE``); the image is ``images/<key>.mha``.
    """
    keys: list[str] = []
    with open(SPIDER_ROOT / "overview.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("new_file_name") or "").strip()
            if row.get("subset", "").strip() != split or "_t2" not in name:
                continue
            if (SPIDER_ROOT / "images" / f"{name}.mha").is_file():
                keys.append(name)
    keys = sorted(set(keys), key=lambda s: (int(s.split("_")[0]), s))
    rng = random.Random(seed)
    return [
        (k, "spider")
        for k in sorted(rng.sample(keys, min(n, len(keys))), key=lambda s: (int(s.split("_")[0]), s))
    ]


def sudirman_cases(n: int, seed: int) -> list[tuple[str, str]]:
    """Seeded sample of Sudirman patients (folder names)."""
    pats = sorted(patient_sessions(SUDIRMAN_ROOT).keys())
    rng = random.Random(seed)
    return [(p, "sudirman") for p in sorted(rng.sample(pats, min(n, len(pats))))]


def load_case(name: str, dataset: str) -> Any:
    if dataset == "spider":
        return load_study(str(SPIDER_ROOT / "images" / f"{name}.mha"))
    sess = patient_sessions(SUDIRMAN_ROOT)[name]
    for s in sess:
        st = load_study(str(s))
        if pick_conus_stack(st) is not None and any(
            (x.meta.get("role", {}) or {}).get("contrast") == "t2" for x in st.find("sag")
        ):
            return st
    return load_study(str(sess[0]))


def row_of_z(g: Any, z: float, x_ref: float | None) -> float:
    """Fractional row index on grid ``g`` of the point at height ``z`` on the slice plane."""
    xyz = np.array([x_ref or 0.0, 0.0, z])
    down, post = g.mm_of(xyz)
    i, _j = g.index_of_mm(down, post)
    return float(i)


def render(study: Any, anchors: Any, title: str) -> Image.Image:
    """Canal crop of the conus stack's slice nearest the conus (or the midline), with discs and tip."""
    stack = pick_conus_stack(study)
    if stack is None:
        return Image.new("RGB", (320, 320), (0, 0, 0))
    z_tip = anchors.conus.get("z") if anchors.conus else None
    x_ref = float(anchors.conus["xyz"][0]) if anchors.conus and anchors.conus.get("xyz") else None
    if x_ref is None and anchors.midline_x is not None:
        x_ref = float(anchors.midline_x)
    k = int(np.argmin(np.abs(stack.positions - x_ref))) if x_ref is not None else stack.n // 2
    g = plane_grid(stack, k, DEFAULT.ml_resolution_mm)
    img = resample_slice(stack, g)
    lo, hi = np.percentile(img[img > 0], [1, 99.5]) if np.any(img > 0) else (0.0, 1.0)
    img8 = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(img8).convert("RGB")
    # Crop around the discs, or the tip when no discs are known.
    zs = [float(d.center[2]) for d in anchors.discs.values()] + ([float(z_tip)] if z_tip is not None else [])
    if not zs:
        zs = [float(np.mean(stack.positions))]
    z_hi, z_lo = max(zs) + 25.0, min(zs) - 25.0
    rows = np.array([row_of_z(g, z_hi, x_ref), row_of_z(g, z_lo, x_ref)])
    rows = np.clip(rows, 0, g.shape[0] - 1)
    r0, r1 = int(rows.min()), int(rows.max())
    # Columns: centred on the canal, about 25 mm behind the disc centres (canal offset, Phantom/anatomy).
    if anchors.discs:
        posts = [g.mm_of(np.asarray(d.center)[None, :])[0][1] for d in anchors.discs.values()]
        post_c = float(np.mean(posts)) + 25.0
    else:
        post_c = float(g.post0 + 0.5 * g.shape[1] * g.res_mm)
    j_c = round(g.index_of_mm(np.array([0.0]), np.array([post_c]))[1][0])
    half = int(CROP_HALF_AP_MM / g.res_mm)
    c0, c1 = max(0, j_c - half), min(g.shape[1], j_c + half)
    crop = pil.crop((c0, max(0, r0), c1, min(g.shape[0], r1 + 1)))
    crop = crop.resize((crop.width * PX_PER_MM, crop.height * PX_PER_MM), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (crop.width, crop.height + 36), (0, 0, 0))
    canvas.paste(crop, (0, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title[:120], fill=(255, 255, 255))

    def y_of(z: float) -> float:
        return 36 + (row_of_z(g, z, x_ref) - r0) * PX_PER_MM

    for lab, d in anchors.discs.items():
        y = y_of(float(d.center[2]))
        draw.line([(0, y), (canvas.width, y)], fill=(90, 200, 255), width=1)
        draw.text((4, y - 12), lab, fill=(90, 200, 255))
    if z_tip is not None:
        y = y_of(float(z_tip))
        draw.line([(0, y), (canvas.width, y)], fill=(255, 60, 60), width=2)
        lvl = anchors.conus.get("level") or "?"
        draw.text(
            (canvas.width - 150, y - 14),
            f"conus {lvl} {anchors.conus.get('confidence', 0):.2f}",
            fill=(255, 60, 60),
        )
    return canvas


def run(cases: list[tuple[str, str]], out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(exist_ok=True)
    records = []
    for name, ds in cases:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"case": name, "dataset": ds}
        try:
            study = load_case(name, ds)
            det = label_spine_details(study)
            anchors = det.anchors
            rec["conus"] = anchors.conus
            rec["discs"] = anchors.disc_labels()
            rec["numbering_status"] = anchors.numbering.get("status")
            rec["warnings"] = [w for w in anchors.warnings if "conus" in w.lower()]
            rec["contrast"] = (pick_conus_stack(study).meta.get("role", {}) or {}).get("contrast")
            img = render(study, anchors, f"{ds} {name}  disc labels: {', '.join(anchors.disc_labels())}")
            img.save(out / "png" / f"{ds}_{name}.png")
        except Exception as exc:  # a failing case is a row in the audit, not a crash
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        rec["elapsed_s"] = round(time.perf_counter() - t0, 2)
        records.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("case", "elapsed_s", "error")}), flush=True)
    (out / "records.json").write_text(json.dumps(records, indent=1))
    _sheets(out)
    return 0


def _sheets(out: Path, per_sheet: int = 3) -> None:
    """Side-by-side sheets of ``per_sheet`` case renders, for the visual grading pass."""
    files = sorted((out / "png").glob("*.png"))
    (out / "sheets").mkdir(exist_ok=True)
    for i in range(0, len(files), per_sheet):
        group = [Image.open(f) for f in files[i : i + per_sheet]]
        h = max(g.height for g in group)
        w = sum(g.width for g in group) + 12 * (len(group) - 1)
        sheet = Image.new("RGB", (w, h), (40, 40, 40))
        x = 0
        for g in group:
            sheet.paste(g, (x, 0))
            x += g.width + 12
        sheet.save(out / "sheets" / f"sheet_{i // per_sheet + 1:02d}.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="which", required=True)
    s = sub.add_parser("spider")
    s.add_argument("--split", default="validation")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=16)
    s.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("sudirman")
    u.add_argument("--n", type=int, default=20)
    u.add_argument("--seed", type=int, default=16)
    u.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cases = spider_cases(a.split, a.n, a.seed) if a.which == "spider" else sudirman_cases(a.n, a.seed)
    return run(cases, a.out)


if __name__ == "__main__":
    sys.exit(main())
