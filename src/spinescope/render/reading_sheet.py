"""Reading sheets: one compact PNG per level and finding field, so that a reader can check a number against the image.

A sheet shows up to ``config.reading_sheet_panels`` series side by side, each a mm-ruled zoom crop centred on the
labelled disc at the slice that shows it. The anchored disc line (anterior to posterior margin at mid-height) is drawn
on every panel, with the mm rulers and a scale bar. A banner gives the level label, its numbering status (a position
only while the name is unconfirmed), the field, and optionally the measured value and its validation status.

A sheet is evidence, not a measurement: nothing is computed here beyond the crops. The field's own series (the one the
measurement came from) comes first when it covers the disc; the other sagittal series follow, T2 before T1 before STIR.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..anatomy.anchors import Anchors
from ..anatomy.summary import status_of
from ..config import DEFAULT, Config
from ..geometry import Stack, sample_plane
from ..study import Study
from .annotate import YELLOW, add_rulers, crosshair, line, orientation_letters, scale_bar
from .frame import Frame, auto_upscale, min_display_px
from .layout import fit_max_side, title_band, vstack
from .text import label_px
from .window import grey_rgb, resolve_window

log = logging.getLogger(__name__)

RGB = NDArray[np.uint8]

#: Contrast order for the panels after the field's own series.
SEQUENCE_ORDER: tuple[str, ...] = ("t2", "t1", "stir")


def sheet_series(
    study: Study, anchors: Anchors, level: str, primary: str | None, cfg: Config = DEFAULT
) -> list[str]:
    """Series keys for the panels of one level: ``primary`` first (when it covers the disc), then the other
    non-localizer sagittal series that cover it, T2 before T1 before STIR. At most ``reading_sheet_panels``."""
    centre = np.asarray(anchors.discs[level].center, dtype=np.float64)
    keys: list[str] = []
    if primary is not None and primary in study and study[primary].contains(centre):
        keys.append(primary)

    def rank(st: Stack) -> tuple[int, int, str]:
        contrast = str(st.meta.get("role", {}).get("contrast", "unknown"))
        pos = SEQUENCE_ORDER.index(contrast) if contrast in SEQUENCE_ORDER else len(SEQUENCE_ORDER)
        return (pos, -st.n, st.key)

    others = [
        st
        for st in study.stacks.values()
        if st.plane == "sag"
        and not st.meta.get("role", {}).get("localizer")
        and st.key not in keys
        and st.contains(centre)
    ]
    keys += [st.key for st in sorted(others, key=rank)]
    return keys[: max(1, cfg.reading_sheet_panels)]


def _crop(
    st: Stack, centre: NDArray[np.float64], k: int, anchors: Anchors, level: str, cfg: Config
) -> tuple[RGB, Frame]:
    """A mm-ruled crop of slice ``k`` centred on ``centre``, with the disc line drawn (before the title band)."""
    nrm = st.normal
    on_plane = centre - ((centre - st.ipp[k]) @ nrm) * nrm
    side = cfg.zoom_mm
    native_px = min(st.row_mm, st.col_mm)
    up = auto_upscale(native_px, (side, side), min_display_px(cfg))
    px = native_px / up
    win = resolve_window(st, None, cfg)
    samp = sample_plane(st, on_plane, st.row_dir, st.col_dir, size_mm=(side, side), px_mm=px, order=1)
    frame = Frame.centred(on_plane, st.row_dir, st.col_dir, samp.shape, px, px)
    img = grey_rgb(samp, win)
    disc = anchors.discs[level]
    if disc.anterior is not None and disc.posterior is not None:
        img = line(img, frame, disc.anterior, disc.posterior, YELLOW, show_length=False, cfg=cfg)
    img = crosshair(img, frame, on_plane, YELLOW, cfg=cfg)
    img = scale_bar(img, frame, cfg=cfg)
    img = orientation_letters(img, frame, cfg=cfg)
    img = add_rulers(img, frame, cfg=cfg)
    return img, frame


def _panel(st: Stack, anchors: Anchors, level: str, k: int, cfg: Config) -> RGB:
    centre = np.asarray(anchors.discs[level].center, dtype=np.float64)
    img, frame = _crop(st, centre, k, anchors, level, cfg)
    off = float(frame.to_px(centre)[2])
    plane = {"sag": "sagittal", "cor": "coronal", "ax": "axial"}.get(st.plane, st.plane)
    lines = [
        f"{st.key}  k={k:02d}  {plane}  {off:+.1f} mm from the disc centre plane",
        f"crop {cfg.zoom_mm:g} mm square; yellow line = anchored disc, anterior to posterior margin",
    ]
    return vstack([title_band(lines, img.shape[1], font_px=label_px(img.shape, cfg), cfg=cfg), img])


def banner_lines(
    level: str,
    anchors: Anchors,
    *,
    field_label: str,
    value_text: str | None = None,
    confidence: str | None = None,
    validation: str | None = None,
) -> list[str]:
    """Banner text: level label and numbering status, the field, the measured value, the validation status.

    While the numbering is unconfirmed the level is shown as its position (top disc = position 1) and the automatic
    name in brackets, so that the sheet cannot be read as a confirmed name.
    """
    status = status_of(anchors)
    labels = anchors.disc_labels()
    if status == "unconfirmed":
        pos = labels.index(level) + 1 if level in labels else 0
        name = f"position {pos} (automatic name {level}, unconfirmed)"
    else:
        name = level
    head = f"{name}  |  {field_label}  |  numbering {status.upper()}"
    lines = [head]
    if value_text is not None:
        conf = f", confidence {confidence}" if confidence else ""
        lines.append(f"measured: {value_text}{conf}")
    if validation:
        lines.append(f"validation: {validation}")
    lines.append("Research and education only. Evidence for a reader to check, not a diagnosis.")
    return lines


def reading_sheet(
    study: Study,
    anchors: Anchors,
    level: str,
    *,
    field_label: str,
    primary: str | None = None,
    slice_k: int | None = None,
    value_text: str | None = None,
    confidence: str | None = None,
    validation: str | None = None,
    config: Config | None = None,
) -> RGB:
    """One reading sheet for ``level``: the panels (see the module docstring) under a banner.

    ``primary`` is the series the measurement came from and ``slice_k`` its measured slice (both optional). Without
    ``value_text`` the banner shows no number (a blind read). Raises KeyError for a level without an anchor.
    """
    cfg = config or DEFAULT
    if level not in anchors.discs:
        raise KeyError(f"no disc {level!r} in the anchors")
    keys = sheet_series(study, anchors, level, primary, cfg)
    if not keys:
        raise ValueError(f"no series covers disc {level!r}")
    panels: list[RGB] = []
    for key in keys:
        st = study[key]
        centre = np.asarray(anchors.discs[level].center, dtype=np.float64)
        k = slice_k if key == primary and slice_k is not None and 0 <= slice_k < st.n else None
        if k is None:
            k, _dist = st.nearest_slice(centre)
        panels.append(_panel(st, anchors, level, int(k), cfg))
    target_h = max(p.shape[0] for p in panels)
    row = _hstack_same_height(panels, target_h, cfg)
    lines = banner_lines(
        level,
        anchors,
        field_label=field_label,
        value_text=value_text,
        confidence=confidence,
        validation=validation,
    )
    banner = title_band(lines, row.shape[1], font_px=label_px(row.shape, cfg), cfg=cfg)
    return fit_max_side(vstack([banner, row]), cfg.max_png_side_px)


def _hstack_same_height(panels: Sequence[RGB], height: int, cfg: Config) -> RGB:
    """Panels side by side at one height (each resized, aspect kept), with a thin grey gap."""
    gap = max(2, round(cfg.max_png_side_px / 400))
    scaled: list[RGB] = []
    for p in panels:
        w = max(1, round(p.shape[1] * height / p.shape[0]))
        scaled.append(
            np.asarray(Image.fromarray(p).resize((w, height), Image.Resampling.LANCZOS), dtype=np.uint8)
        )
    width = sum(p.shape[1] for p in scaled) + gap * (len(scaled) - 1)
    out = np.full((height, width, 3), 90, dtype=np.uint8)
    x = 0
    for p in scaled:
        out[:, x : x + p.shape[1]] = p
        x += p.shape[1] + gap
    return out


def save_sheet(img: RGB, path: str | Path) -> Path:
    """Write a sheet as PNG (creates the parent folder)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(target, format="PNG")
    return target
