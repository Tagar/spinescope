"""Cross-referencing between stacks: where other stacks' slices cut a slice, and the same LPS point in every
series.

* :func:`reference_lines` draws, on slice ``k`` of one stack, the line where each slice of other stacks
  intersects it (:func:`spinescope.geometry.intersection_segment`), labelled with the source slice and
  position, and tags discs whose centre lies within the slab of that line.
* :func:`linked_views` shows one LPS point on the nearest slice of every stack, cropped and centred on the
  point with a crosshair. Panels are flagged when the point is off the slab or outside the in-plane FOV,
  and each record reports the off-plane distance and the native pixel of the point.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..geometry import Stack, intersection_segment, sample_plane, unit
from .annotate import PALETTE, YELLOW, Canvas, crosshair, scale_bar
from .frame import Frame, auto_upscale, display_scale, min_display_px, slice_frame
from .layout import fit_max_side, grid, title_band, vstack, with_title
from .sheets import slice_label_lines
from .text import label_px
from .window import Window, grey_rgb, resolve_window

log = logging.getLogger(__name__)

RGB = NDArray[np.uint8]


def _seg_dist(p: ArrayLike, a: ArrayLike, b: ArrayLike) -> float:
    """3-D distance from point ``p`` to segment ``a``-``b``."""
    pa, aa, bb = (np.asarray(x, dtype=np.float64) for x in (p, a, b))
    ab = bb - aa
    t = float(np.clip((pa - aa) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(pa - (aa + t * ab)))


def _discs_on_line(anchors: Anchors | None, a: ArrayLike, b: ArrayLike, slab_mm: float) -> list[str]:
    """Labels of discs whose centre lies within the slab of a line (half slab + half disc height)."""
    if anchors is None:
        return []
    return [
        lab
        for lab, d in anchors.discs.items()
        if _seg_dist(d.center, a, b) <= 0.5 * slab_mm + 0.5 * (d.height_mm or 0.0)
    ]


def reference_lines(
    stack_a: Stack,
    k_a: int,
    refs: Sequence[tuple[Stack, Sequence[int]]],
    *,
    anchors: Anchors | None = None,
    window: Window | str | None = None,
    config: Config | None = None,
) -> RGB:
    """Slice ``k_a`` of ``stack_a`` with the lines where slices of other stacks cut it.

    ``refs`` is a list of (stack, slice indices); each stack gets its own colour (see the header). Lines
    are labelled with the source key, slice and position, plus any disc within their slab. Labels that
    would overlap an earlier label are left off (the line is still drawn). With ``anchors``, discs lying in
    the slice slab are marked with a cross and their anterior-posterior line.
    """
    cfg = config or DEFAULT
    win = resolve_window(stack_a, window, cfg)
    native = slice_frame(stack_a, k_a)
    base, frame = display_scale(grey_rgb(stack_a.vol[k_a], win), native, min_display_px(cfg))
    c = Canvas(base)
    size = label_px(c.shape, cfg)
    stroke = max(1, round(size / 10))
    slab_a = stack_a.thickness or stack_a.spacing or 0.0
    if anchors is not None:
        for lab, d in anchors.discs.items():
            off = float((np.asarray(d.center) - native.origin) @ native.normal)
            if abs(off) > 0.5 * slab_a:
                continue
            x, y, _ = frame.to_px(d.center)
            if d.anterior is not None and d.posterior is not None:
                ax_, ay_, _ = frame.to_px(d.anterior)
                px_, py_, _ = frame.to_px(d.posterior)
                c.line((ax_, ay_), (px_, py_), YELLOW, stroke)
            c.draw.line([(x - 2 * size, y), (x + 2 * size, y)], fill=YELLOW, width=stroke)
            c.draw.line([(x, y - 2 * size), (x, y + 2 * size)], fill=YELLOW, width=stroke)
            c.text((x + size / 2, y - size / 2), lab, YELLOW, size, anchor="lb")
    segments: list[tuple[int, str, tuple[float, float], tuple[float, float]]] = []
    for i, (st, ks) in enumerate(refs):
        for kb in ks:
            seg = intersection_segment(stack_a, k_a, st, kb)
            if seg is None:
                continue
            p0 = stack_a.pix_to_lps(k_a, seg[0][0], seg[0][1])
            p1 = stack_a.pix_to_lps(k_a, seg[1][0], seg[1][1])
            discs = _discs_on_line(anchors, p0, p1, st.thickness or st.spacing or 0.0)
            text = f"{st.key} {slice_label_lines(st, kb)[0]}" + (f" [{', '.join(discs)}]" if discs else "")
            a = frame.to_px(p0)[:2]
            b = frame.to_px(p1)[:2]
            segments.append((i, text, a, b))
    # greedy label placement: at the left end of each line, skipping labels that would overlap a placed one
    placed_y: list[float] = []
    for i, text, a, b in sorted(segments, key=lambda s: s[2][1]):
        colour = PALETTE[i % len(PALETTE)]
        left = a if a[0] <= b[0] else b
        c.line(a, b, colour, stroke)
        if all(abs(left[1] - y) >= size * 1.1 for y in placed_y):
            placed_y.append(left[1])
            c.text(
                (left[0] + size / 2, left[1] - size / 3), text, colour, max(6, round(size * 0.8)), anchor="lb"
            )
    header = [
        f"{stack_a.key} {slice_label_lines(stack_a, k_a, anchors)[0]}  (lines: slices of other stacks)",
        "  ".join(st.key for st, _ in refs) if refs else "no reference stacks",
    ]
    if not segments:
        header.append("no slice of the reference stacks cuts this slice inside both fields of view")
    return with_title(c.array(), header, cfg=cfg)


def linked_views(
    xyz: ArrayLike,
    stacks: Mapping[str, Stack] | Sequence[Stack],
    *,
    anchors: Anchors | None = None,
    crop_mm: float | None = None,
    window: Window | str | None = None,
    columns: int | None = None,
    config: Config | None = None,
) -> tuple[RGB, list[dict[str, Any]]]:
    """One LPS point on the nearest slice of every stack: crops centred on the point, with a crosshair.

    Returns the grid image and one record per stack: ``key``, ``k`` (nearest slice), ``signed_mm`` (point
    relative to that slice plane, + along the stack normal), ``dist_mm`` (its absolute value),
    ``in_slab`` (within half a slice thickness), ``inside_fov`` (in-plane), ``native_row`` / ``native_col``
    (fractional pixel of the point on that slice) and ``flags`` ('OFF SLAB', 'OUTSIDE FOV'). The flags also
    appear in the panel titles. Crops are upscaled for display so that each is legible.
    """
    cfg = config or DEFAULT
    p_world = np.asarray(xyz, dtype=np.float64).reshape(3)
    if isinstance(stacks, Mapping):
        items = [(key, st) for key, st in stacks.items() if st is not None]
    else:
        items = [(st.key, st) for st in stacks if st is not None]
    if not items:
        raise ValueError("linked_views needs at least one stack")
    size = float(crop_mm or cfg.zoom_mm)
    panels: list[RGB] = []
    records: list[dict[str, Any]] = []
    for key, st in items:
        k, _ = st.nearest_slice(p_world)
        n = unit(st.normal)
        signed = float((p_world - st.ipp[k]) @ n)
        on_plane = p_world - signed * n
        native = slice_frame(st, k)
        x, y, _ = native.to_px(on_plane)
        inside = native.inside(x, y)
        in_slab = abs(signed) <= 0.5 * (st.thickness or st.spacing or 0.0) + 1e-6
        flags = []
        if not in_slab:
            flags.append("OFF SLAB")
        if not inside:
            flags.append("OUTSIDE FOV")
        native_px = min(st.row_mm, st.col_mm)
        up = auto_upscale(native_px, (size, size), min_display_px(cfg))
        px = native_px / up
        samp = sample_plane(st, on_plane, st.row_dir, st.col_dir, size_mm=(size, size), px_mm=px)
        frame = Frame.centred(on_plane, st.row_dir, st.col_dir, samp.shape, px, px)
        img = grey_rgb(samp, resolve_window(st, window, cfg))
        img = crosshair(img, frame, on_plane, YELLOW, cfg=cfg)
        img = scale_bar(img, frame, cfg=cfg)
        title = [
            f"{key} k={k:02d}  off-plane {signed:+.1f} mm"
            + ("  [" + ", ".join(flags) + "]" if flags else ""),
            f"native row {y:.1f}, col {x:.1f}  (crop {size:g} mm)",
        ]
        panels.append(with_title(img, title, cfg=cfg))
        records.append(
            {
                "key": key,
                "k": int(k),
                "signed_mm": round(signed, 2),
                "dist_mm": round(abs(signed), 2),
                "in_slab": bool(in_slab),
                "inside_fov": bool(inside),
                "native_row": round(float(y), 4),
                "native_col": round(float(x), 4),
                "flags": flags,
            }
        )
    cols = columns or min(3, len(panels))
    body = grid(panels, cols, gap=max(2, round(cfg.max_png_side_px / 400)))
    head = title_band(
        [f"linked views  LPS ({p_world[0]:+.1f}, {p_world[1]:+.1f}, {p_world[2]:+.1f}) mm"],
        body.shape[1],
        font_px=label_px(body.shape, cfg),
        cfg=cfg,
    )
    out = fit_max_side(vstack([head, body]), cfg.max_png_side_px)
    if any(r["flags"] for r in records):
        log.info("linked views: %s", [(r["key"], r["flags"]) for r in records if r["flags"]])
    return out, records
