"""Visual QA for the labeller: overlays of predicted and ground-truth discs, and one-page case reports.

Nothing here feeds back into the labeller. :func:`overlay` draws the mid-sagittal slice nearest the
midline used by :func:`spinescope.anatomy.label_spine`, with each predicted disc (centre, plane line,
label, confidence colour), the ground-truth discs when given (cyan, with a connector to the matched
prediction; red connector when the level name is wrong), and the canal centreline used for the
midline. :func:`case_report` puts the overlay next to a disc-plane reformat per disc and a text table of
matches, on one PNG page. Outputs are RGB ``uint8`` arrays; write them to a scratch folder, never into
the repository when they are derived from imaging data.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ..config import DEFAULT, Config
from ..geometry import Stack
from ..render.frame import display_scale, slice_frame
from ..render.layout import fit_max_side, grid, hstack, title_band, vstack, with_title
from ..render.mpr import disc_plane
from ..render.text import font, label_px
from ..render.window import contrast_of, grey_rgb, save_png, series_window
from ..study import Study
from ._validate import TruthDisc, VolumeScore, score_anchors
from .anchors import Anchors

log = logging.getLogger(__name__)

RGB = NDArray[np.uint8]
#: Side of a disc-plane tile in the case report, px (display only).
TILE_MIN_PX = 240

CANAL_COLOUR = (255, 90, 255)
TRUTH_COLOUR = (60, 225, 255)
CORRECT_COLOUR = (150, 150, 150)
WRONG_COLOUR = (255, 70, 70)
#: Disc axis drawn through the centre when the margins are unknown (mm, half length).
PLANE_HALF_MM = 10.0
#: Label font height in the overlay, as a fraction of the shorter side (labels must not hide the spine).
OVERLAY_FONT_FRAC = 0.028
#: A truth disc further than this from the displayed slice is labelled "off-plane".
OFF_PLANE_MM = 3.0


def confidence_colour(conf: float) -> tuple[int, int, int]:
    """Red (0) -> yellow (0.5) -> green (1) for a confidence in [0, 1]."""
    c = float(np.clip(conf, 0.0, 1.0))
    if c < 0.5:
        t = c / 0.5
        return int(230 + 20 * t), int(40 + 160 * t), 40
    t = (c - 0.5) / 0.5
    return int(250 - 190 * t), int(200 + 20 * t), int(40 + 50 * t)


def _pick_stack(study: Study, anchors: Anchors) -> Stack | None:
    if anchors.source_stack and anchors.source_stack in study.stacks:
        return study.stacks[anchors.source_stack]
    return study.best("sag")


def midline_slice(stack: Stack, anchors: Anchors) -> int:
    """Index of the slice nearest the midline plane used by the labeller (or the disc centres)."""
    if anchors.midline_x is not None and abs(float(stack.normal[0])) >= 0.95:
        plane = float(anchors.midline_x) * float(stack.normal[0])
    elif anchors.discs:
        plane = float(np.mean([np.asarray(d.center) @ stack.normal for d in anchors.discs.values()]))
    else:
        plane = float(np.mean(stack.positions))
    return int(np.argmin(np.abs(stack.positions - plane)))


def canal_centreline(stack: Stack, k: int, cfg: Config = DEFAULT) -> NDArray[np.float64] | None:
    """Thecal-sac centreline (LPS, ordered superior -> inferior) on slice ``k``, as the sagittal tracker sees it.

    Best-effort: the tracker lives in :mod:`spinescope.anatomy.sagittal`; if its interface changes or
    no CSF ridge is found, returns None and the overlay simply omits the canal.
    """
    try:
        from .sagittal import csf_polarities, sagittal_image, track_canal

        sag = sagittal_image(stack, k, cfg.sag_resolution_mm)
        best: Any = None
        for pol in csf_polarities(contrast_of(stack)):
            tr = track_canal(sag, pol, cfg)
            if tr is not None and (best is None or tr.score > best.score):
                best = tr
        if best is None:
            return None
        s = np.linspace(float(best.s[0]), float(best.s[-1]), 60)
        pts: NDArray[np.float64] = np.asarray(sag.to_lps(best.a_at(s), s), dtype=np.float64)
        return pts[::-1]  # superior -> inferior
    except (ValueError, IndexError, AttributeError, TypeError) as exc:
        log.warning("canal centreline not available for QA overlay: %s", exc)
        return None


def _text(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], s: str, colour: Any, size: int, anchor: str = "lb"
) -> None:
    f = font(size)
    x0, y0, x1, y1 = draw.textbbox(xy, s, font=f, anchor=anchor)
    pad = max(1, size // 5)
    draw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=(0, 0, 0))
    draw.text(xy, s, font=f, fill=colour, anchor=anchor)


def _upscale(img: RGB, min_side: int) -> RGB:
    """Bilinear upscale so the shorter side is at least ``min_side`` px (display only)."""
    h, w = img.shape[:2]
    f = max(1.0, min_side / max(1, min(h, w)))
    if f <= 1.0:
        return img
    return np.asarray(Image.fromarray(img).resize((round(w * f), round(h * f)), Image.Resampling.BILINEAR))


def _placeholder(text: str, shape: tuple[int, int] = (320, 320)) -> RGB:
    img = np.zeros((*shape, 3), dtype=np.uint8)
    im = Image.fromarray(img)
    _text(ImageDraw.Draw(im), (8, shape[0] // 2), text, (255, 255, 255), label_px(shape))
    return np.asarray(im, dtype=np.uint8)


def overlay(
    study: Study,
    anchors: Anchors,
    truth: Sequence[TruthDisc] | None = None,
    *,
    canal: NDArray[np.float64] | None = None,
    midline_truth: float | None = None,
    stack: Stack | None = None,
    min_side: int = 640,
    cfg: Config = DEFAULT,
) -> RGB:
    """Mid-sagittal slice nearest the midline with predicted (and optionally truth) discs drawn on it.

    ``truth`` is a list of :class:`TruthDisc` (e.g. from ``truth_from_mask``). ``canal`` is an (n, 3) LPS
    centreline to draw; when omitted it is derived from the image with the labeller's canal tracker.
    Returns an (H, W, 3) uint8 image with a two-line legend band on top.
    """
    st = stack or _pick_stack(study, anchors)
    if st is None:
        return _placeholder("no sagittal series")
    k = midline_slice(st, anchors)
    base = grey_rgb(st.vol[k], series_window(st, cfg))
    frame = slice_frame(st, k)
    base, frame = display_scale(base, frame, min_side)
    im = Image.fromarray(base)
    draw = ImageDraw.Draw(im)
    size = max(9, round(OVERLAY_FONT_FRAC * min(base.shape[:2])))
    width = max(2, round(0.004 * min(base.shape[:2])))

    pts = canal if canal is not None else canal_centreline(st, k, cfg)
    if pts is not None and len(pts) > 1:
        xy = [frame.to_px(p)[:2] for p in pts]
        draw.line(xy, fill=CANAL_COLOUR, width=width)

    by_name = {t.name: t for t in truth or []}
    if truth:
        for t in truth:
            x, y, off = frame.to_px(t.center)
            r = max(4, round(0.008 * min(base.shape[:2])))
            draw.ellipse((x - r, y - r, x + r, y + r), outline=TRUTH_COLOUR, width=width)
            tag = t.name + (f" (off-plane {off:+.0f} mm)" if abs(off) > OFF_PLANE_MM else "")
            _text(draw, (x - r - 3, y + r), tag, TRUTH_COLOUR, size, anchor="rb")

    if truth:
        scored = score_anchors(anchors, list(truth), midline_truth)
        for m in scored.matches:
            if m.truth is None:
                continue
            p = np.asarray(anchors.discs[m.pred].center)
            q = np.asarray(by_name[m.truth].center)
            colour = CORRECT_COLOUR if m.label_correct else WRONG_COLOUR
            draw.line([frame.to_px(p)[:2], frame.to_px(q)[:2]], fill=colour, width=max(1, width - 1))

    for lab in anchors.disc_labels():
        d = anchors.discs[lab]
        colour = confidence_colour(d.confidence)
        if d.anterior is not None and d.posterior is not None:
            a, b = np.asarray(d.anterior), np.asarray(d.posterior)
        else:
            ap = np.asarray(d.ap_dir)
            c = np.asarray(d.center)
            a, b = c - PLANE_HALF_MM * ap, c + PLANE_HALF_MM * ap
        draw.line([frame.to_px(a)[:2], frame.to_px(b)[:2]], fill=colour, width=width + 1)
        x, y, _off = frame.to_px(d.center)
        r = max(3, round(0.006 * min(base.shape[:2])))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)
        _text(draw, (x + r + 3, y - r), f"{lab} {d.confidence:.2f}", colour, size)

    x_mid = anchors.midline_x
    mid_txt = f"midline x {x_mid:+.1f} mm" if x_mid is not None else "midline n/a"
    if midline_truth is not None and x_mid is not None:
        mid_txt += f" (truth {midline_truth:+.1f}, err {abs(x_mid - midline_truth):.1f} mm)"
    line1 = f"{st.key} slice {k + 1}/{st.n}  {mid_txt}  canal {'drawn' if pts is not None else 'n/a'}"
    line2 = "red->green = predicted disc confidence; cyan = truth; magenta = canal; grey/red = match"
    lines = [line1, line2]
    return with_title(np.asarray(im, dtype=np.uint8), lines, cfg=cfg)


def _match_lines(anchors: Anchors, truth: Sequence[TruthDisc], midline_truth: float | None) -> list[str]:
    sc: VolumeScore = score_anchors(anchors, list(truth), midline_truth)
    lines = []
    for m in sc.matches:
        conf = f"conf {m.pred_confidence:.2f}"
        if m.truth is None:
            lines.append(f"pred {m.pred:<9} false positive  {conf}")
        else:
            off = "" if m.level_offset is None else f" (level {m.level_offset:+d})"
            status = "OK" if m.label_correct else f"WRONG{off}"
            err = f"{m.centre_err_mm:.1f} mm" if m.centre_err_mm is not None else "-"
            lines.append(f"pred {m.pred:<9} truth {m.truth:<9} centre {err:<8} {conf}  {status}")
    if sc.missed:
        lines.append("missed truth discs: " + ", ".join(sc.missed))
    if not lines:
        lines.append("no predicted discs")
    return lines


def case_report(
    study: Study,
    anchors: Anchors,
    truth: Sequence[TruthDisc] | None = None,
    *,
    out_path: str,
    title: str = "",
    canal: NDArray[np.float64] | None = None,
    midline_truth: float | None = None,
    max_discs: int = 10,
    cfg: Config = DEFAULT,
) -> str:
    """One-page PNG for a case: header, mid-sagittal overlay, a disc-plane reformat per disc, match table.

    Returns ``out_path``. Works without truth (predictions only) and with no discs (placeholders).
    """
    stack = _pick_stack(study, anchors)
    head = [title or (stack.key if stack is not None else "case")]
    head.append(
        f"method: {anchors.method or 'n/a'}; numbering conf {float(anchors.numbering.get('confidence', 0.0) or 0.0):.2f}"
        f"; {len(anchors.discs)} discs"
    )
    if anchors.warnings:
        head.append("warnings: " + " | ".join(anchors.warnings[:3]))
    if truth:
        sc = score_anchors(anchors, list(truth), midline_truth)
        errs = sc.centre_err_mm
        med = f"{float(np.median(errs)):.1f} mm" if errs else "n/a"
        head.append(
            f"truth discs {sc.n_truth}, matched {sc.matched}, label correct {sc.label_correct}, "
            f"centre median {med}"
            + (f", midline err {sc.midline_err_mm:.1f} mm" if sc.midline_err_mm is not None else "")
        )

    body_overlay = overlay(study, anchors, truth, canal=canal, midline_truth=midline_truth, cfg=cfg)
    tiles: list[RGB] = []
    if stack is not None:
        for lab in anchors.disc_labels()[:max_discs]:
            d = anchors.discs[lab]
            ref = disc_plane(stack, anchors, lab, cfg=cfg)
            if ref is None:
                continue
            img = grey_rgb(ref.img, series_window(stack, cfg))
            tiles.append(with_title(_upscale(img, TILE_MIN_PX), [f"{lab} conf {d.confidence:.2f}"], cfg=cfg))
    if tiles:
        cell_h = max(t.shape[0] for t in tiles)
        cell_w = max(t.shape[1] for t in tiles)
        side = grid(tiles, cols=2, gap=6, cell=(cell_h, cell_w))
    else:
        side = _placeholder("no disc-plane reformats")
    width = max(body_overlay.shape[1] + side.shape[1] + 8, 320)
    head_px = max(12, round(0.017 * width))
    header = title_band(head, width, font_px=head_px, cfg=cfg)
    body = hstack([body_overlay, side], gap=8, align="top")
    if truth:
        footer_lines = _match_lines(anchors, truth, midline_truth)
    else:
        footer_lines = [f"{k} conf {anchors.discs[k].confidence:.2f}" for k in anchors.disc_labels()]
    footer_lines = footer_lines or ["no predicted discs"]
    footer = title_band(footer_lines, body.shape[1], font_px=head_px, cfg=cfg)
    page = fit_max_side(vstack([header, body, footer], gap=0), cfg.max_png_side_px)
    return save_png(page, out_path)
