"""Render the findings-panel screenshots for docs/assets/viewer/ from the synthetic phantom with planted pathology.

Phantom only: the DICOM series are written by ``spinescope.testing.phantom`` into the output folder (fake PHI in the
headers, never shipped), the labeller and every registered analysis run on them, and the page is driven in headless
Chromium (Playwright, optional). Run from the repository root::

    .venv/bin/python tools/viewer_screenshots.py /tmp/spinescope-viewer-shots

Needs ``playwright`` (``pip install playwright``) and a Chromium it can start; set ``SPINESCOPE_CHROMIUM`` to a
browser binary if the default one is not installed. Without them the viewer HTML is still written.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from spinescope import load_study
from spinescope.analysis.base import run_all
from spinescope.anatomy import label_spine
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study
from spinescope.viewer import build_viewer

log = logging.getLogger("viewer_screenshots")

TITLE = "Synthetic lumbar phantom (no patient data)"
#: Planted pathology: a degenerated L4-L5 disc (nucleus darkened to 30 %), a Modic-1-like marrow change under the
#: L4-L5 inferior endplate on the left, and a 5 mm central herniation at L5-S1. No listhesis is planted: the
#: alignment flags on this phantom do not match a planted slip, so the screenshots leave listhesis out.
SPEC = PhantomSpec(
    degenerated={"L4-L5": 0.3},
    modic1=[("L4-L5", "inf", "L")],
    herniation={"L5-S1": ("central", 5.0)},
)
#: Panel rows to select: (name, text the row must contain, file name).
SHOTS = (
    ("findings_panel", "disc_contour", "05_findings_panel.jpg"),
    ("findings_experimental", "modic_suggestion", "06_findings_experimental.jpg"),
)


def _capture(html: Path, outdir: Path) -> list[Path]:
    """Screenshots of the page with the findings panel open. Raises ImportError when Playwright is missing."""
    from PIL import Image
    from playwright.sync_api import sync_playwright

    paths: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("SPINESCOPE_CHROMIUM") or None)
        page = browser.new_page(viewport={"width": 1600, "height": 960})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(html.as_uri())
        page.wait_for_timeout(2500)
        page.click("#findbtn")
        page.wait_for_timeout(500)
        for name, text, fname in SHOTS:
            row = page.locator("#fx-body tr[data-id]", has_text=text).first
            if row.count() == 0:
                log.warning("no row for %s; skipped", name)
                continue
            row.click()
            page.wait_for_timeout(700)
            png = outdir / f"_{name}.png"
            page.screenshot(path=str(png))
            out = outdir / fname
            Image.open(png).convert("RGB").save(out, quality=88)
            png.unlink()
            paths.append(out)
        browser.close()
        if errors:
            raise RuntimeError(f"page errors: {errors[:3]}")
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", help="scratch folder: DICOM phantom, viewer HTML and screenshots")
    ap.add_argument("--no-shots", action="store_true", help="write the HTML only (no browser)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dicom = out / "phantom_dicom"
    write_standard_study(SpinePhantom(SPEC), dicom)
    study = load_study(dicom)
    anchors = label_spine(study)
    results = run_all(study, anchors)
    html = build_viewer(study, out / "viewer", anchors=anchors, title=TITLE, findings=results)
    print(html)
    if args.no_shots:
        return 0
    try:
        shots = _capture(html, out)
    except ImportError:
        print("playwright is not installed: no screenshots (pip install playwright)", file=sys.stderr)
        return 0
    for s in shots:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
