"""Multiplanar reformats (MPR): sagittal, coronal, axial, disc-plane and curved coronal views of any stack.

Every reformat carries its true effective resolution. The in-plane footprint of the source pixels and
the slab thickness blurred along the reformat axes are combined (in quadrature), and the sampling pitch
is the floor. A reformat can never look sharper than its source. Sampling uses
:func:`spinescope.geometry.sample_plane` for planar views and a vectorised trilinear sampler for curved
surfaces.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import map_coordinates

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..geometry import Stack, sample_plane, unit
from .annotate import WHITE, add_rulers, crosshair, orientation_letters
from .frame import Frame, display_scale, min_display_px, slice_frame
from .layout import with_title
from .window import Window, grey_rgb

log = logging.getLogger(__name__)

#: Plane name -> (axis index of the fixed coordinate, image right, image down) in LPS.
PLANES: dict[str, tuple[int, tuple[float, float, float], tuple[float, float, float]]] = {
    "sag": (0, (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),  # anterior left, head at top
    "cor": (1, (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),  # patient left on the right, head at top
    "ax": (2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # patient left on the right, anterior at top
}


@dataclass
class Reformat:
    """A reformatted image in a known plane, with its geometry and effective resolution."""

    img: NDArray[np.float32]  # NaN outside the source
    frame: Frame  # exact for planar reformats; nominal (centre column) for curved ones
    label: str
    key: str
    eff_res_mm: tuple[float, float]  # (horizontal, vertical) effective resolution, mm
    px_mm: float  # sampling pitch
    note: str

    @property
    def fraction_inside(self) -> float:
        """Share of the image covered by the source (not NaN)."""
        return float(np.isfinite(self.img).mean()) if self.img.size else 0.0


def sample_points(stack: Stack, pts: ArrayLike, order: int = 1) -> NDArray[np.float32]:
    """Sample ``stack`` at LPS points of any shape (..., 3). NaN outside the slab or in-plane FOV.

    Linear interpolation between slices (fractional slice index along the stack normal) and bilinear
    in-plane, using each slice's own origin, so slightly sheared stacks are handled.
    """
    p = np.asarray(pts, dtype=np.float64)
    shape = p.shape[:-1]
    flat = p.reshape(-1, 3)
    n = stack.n
    rows, cols = stack.vol.shape[1:]
    d = flat @ stack.normal
    if n == 1:
        half = max(stack.thickness, 1.0) / 2
        ok = np.abs(d - stack.positions[0]) <= half
        kf = np.zeros(len(flat))
        origin = np.repeat(stack.ipp[:1], len(flat), axis=0)
    else:
        idx = np.arange(n, dtype=np.float64)
        half = 0.5 * (stack.spacing or stack.thickness or 1.0)
        ok = (d >= stack.positions[0] - half) & (d <= stack.positions[-1] + half)
        kf = np.interp(d, stack.positions, idx)
        origin = np.stack([np.interp(kf, idx, stack.ipp[:, i]) for i in range(3)], axis=1)
    rel = flat - origin
    r = rel @ stack.col_dir / stack.row_mm
    c = rel @ stack.row_dir / stack.col_mm
    ok &= (r >= -0.5) & (r <= rows - 0.5) & (c >= -0.5) & (c <= cols - 0.5)
    out = np.full(len(flat), np.nan, dtype=np.float32)
    if ok.any():
        coords = np.vstack([kf[ok], np.clip(r[ok], 0, rows - 1), np.clip(c[ok], 0, cols - 1)])
        out[ok] = map_coordinates(stack.vol, coords, order=order, mode="nearest")
    return out.reshape(shape)


def sample_frame(stack: Stack, frame: Frame, order: int = 1) -> NDArray[np.float32]:
    """Sample ``stack`` on the pixel grid of ``frame`` (shape ``frame.shape``), NaN outside."""
    h, w = frame.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = frame.origin + (x * frame.px_x)[..., None] * frame.u + (y * frame.px_y)[..., None] * frame.v
    return sample_points(stack, pts, order=order)


def resample_onto(src: Stack, dst: Stack, k: int, order: int = 1) -> NDArray[np.float32]:
    """Values of ``src`` on the native pixel grid of slice ``k`` of ``dst`` (any orientation)."""
    return sample_frame(src, slice_frame(dst, k), order=order)


def effective_resolution(stack: Stack, u: Any, v: Any, px_mm: float) -> tuple[float, float]:
    """(horizontal, vertical) effective resolution in mm of ``stack`` seen on the image axes u, v.

    Along each image axis the blur is the quadrature of the native in-plane pixel footprint and the
    slab thickness projected onto that axis (through-plane component of the slice normal). The sampling
    pitch is a floor, because a finer grid does not add detail.
    """
    uu, vv = unit(u), unit(v)
    n_plane = unit(np.cross(uu, vv))
    e = stack.normal - (stack.normal @ n_plane) * n_plane
    t = stack.thickness or stack.spacing

    def in_plane(a: NDArray[np.float64]) -> float:
        return float(abs(a @ stack.row_dir) * stack.col_mm + abs(a @ stack.col_dir) * stack.row_mm)

    h = max(px_mm, math.hypot(in_plane(uu), t * abs(e @ uu)))
    vt = max(px_mm, math.hypot(in_plane(vv), t * abs(e @ vv)))
    return h, vt


def _eff_note(stack: Stack, h: float, vt: float, px_mm: float) -> str:
    return (
        f"{stack.key}: eff. res {h:.2f} x {vt:.2f} mm (sampling {px_mm:.2f} mm; source in-plane "
        f"{stack.row_mm:.2f} x {stack.col_mm:.2f} mm, slices {stack.thickness:g} mm thick, "
        f"{stack.spacing:.2f} mm apart)"
    )


def _bbox(stack: Stack) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """LPS bounding box of the stack's voxel corners, including half a slice thickness each side."""
    pts = np.concatenate([stack.corners(0), stack.corners(stack.n - 1)])
    half = 0.5 * stack.thickness * stack.normal
    pts = np.concatenate([pts + half, pts - half])
    return pts.min(0), pts.max(0)


def reformat(
    stack: Stack,
    plane: str,
    at_mm: float,
    *,
    px_mm: float | None = None,
    order: int = 1,
) -> Reformat:
    """Sagittal ('sag', at x = at_mm), coronal ('cor', y) or axial ('ax', z) reformat covering the stack.

    Display conventions: sagittal anterior left and head up; coronal and axial patient left on screen right.
    """
    if plane not in PLANES:
        raise ValueError(f"plane must be one of {tuple(PLANES)}, got {plane!r}")
    axis, u_t, v_t = PLANES[plane]
    u, v = unit(u_t), unit(v_t)
    lo, hi = _bbox(stack)
    centre = (lo + hi) / 2
    centre[axis] = at_mm
    ext_u = float(hi[int(np.argmax(np.abs(u)))] - lo[int(np.argmax(np.abs(u)))])
    ext_v = float(hi[int(np.argmax(np.abs(v)))] - lo[int(np.argmax(np.abs(v)))])
    px = px_mm or min(stack.row_mm, stack.col_mm)

    img = sample_plane(stack, centre, u, v, size_mm=(ext_v, ext_u), px_mm=px, order=order)
    frame = Frame.centred(centre, u, v, img.shape, px, px)
    h, vt = effective_resolution(stack, u, v, px)
    if not np.isfinite(img).any():
        log.warning("%s: %s reformat at %s=%.1f lies outside the stack", stack.key, plane, "xyz"[axis], at_mm)
    label = f"{stack.key} {plane} at {'xyz'[axis]}={at_mm:+.1f} mm"
    return Reformat(img, frame, label, stack.key, (h, vt), px, _eff_note(stack, h, vt, px))


def disc_plane(
    stack: Stack,
    anchors: Anchors,
    label: str,
    *,
    size_mm: float | None = None,
    px_mm: float | None = None,
    order: int = 1,
    cfg: Config = DEFAULT,
) -> Reformat | None:
    """Reformat ``stack`` onto the plane of disc ``label`` (centre and orientation from ``anchors``).

    Image right is patient left, image down is posterior (the axial convention, tilted with the disc).
    Returns None with a warning if the disc is not in the anchors.
    """
    if label not in anchors.discs:
        log.warning("disc %s not in anchors; no disc-plane reformat", label)
        return None
    d = anchors.discs[label]
    n = unit(d.normal)
    v = unit(d.ap_dir)
    u = unit(np.cross(v, n))
    side = float(size_mm or cfg.disc_plane_mm)
    px = px_mm or min(stack.row_mm, stack.col_mm)
    centre = np.asarray(d.center, dtype=np.float64)
    img = sample_plane(stack, centre, u, v, size_mm=(side, side), px_mm=px, order=order)
    frame = Frame.centred(centre, u, v, img.shape, px, px)
    h, vt = effective_resolution(stack, u, v, px)
    name = f"disc {label}"
    return Reformat(
        img, frame, f"{stack.key} {name} plane", stack.key, (h, vt), px, _eff_note(stack, h, vt, px)
    )


def curved_centreline(anchors: Anchors, cfg: Config = DEFAULT) -> NDArray[np.float64] | None:
    """Thecal sac centre points (N, 3), ordered inferior to superior, from the disc posterior margins.

    Each centre is the posterior disc margin moved posteriorly by ``cfg.thecal_sac_offset_mm``, or by
    ``cfg.sacral_canal_offset_mm`` at the inferior-most disc when it ends on the sacrum.
    """
    labels = anchors.disc_labels()
    pts = []
    for lab in labels:
        d = anchors.discs[lab]
        if d.posterior is None:
            log.warning("disc %s has no posterior margin; left out of the curved centreline", lab)
            continue
        lower = lab.split("-")[-1]
        off = (
            cfg.sacral_canal_offset_mm
            if (lab == labels[-1] and lower.startswith("S"))
            else cfg.thecal_sac_offset_mm
        )
        pts.append(np.asarray(d.posterior, dtype=np.float64) + d.ap_dir * off)
    if len(pts) < 2:
        return None
    arr = np.asarray(pts)
    return arr[np.argsort(arr[:, 2])]


def curved_coronal(
    stack: Stack,
    anchors: Anchors,
    *,
    half_width_mm: float | None = None,
    px_mm: float | None = None,
    order: int = 1,
    cfg: Config = DEFAULT,
) -> Reformat | None:
    """Coronal reformat that follows the canal: each row is a horizontal line through the canal centre.

    Columns are lateral offsets from the canal centre (image right = patient left). The frame is exact
    only along the centre column. Returns None with a warning if fewer than two discs have posterior
    margins.
    """
    zc_pts = curved_centreline(anchors, cfg)
    if zc_pts is None:
        log.warning("%s: curved coronal needs at least two discs with posterior margins", stack.key)
        return None
    lo, hi = _bbox(stack)
    z_top = min(float(hi[2]), float(zc_pts[-1, 2]))
    z_bot = max(float(lo[2]), float(zc_pts[0, 2]))
    if z_top <= z_bot:
        log.warning("%s: canal centreline does not overlap the stack in z", stack.key)
        return None
    px = px_mm or min(stack.row_mm, stack.col_mm)
    nz = max(1, round((z_top - z_bot) / px))
    z = z_top - (np.arange(nz) + 0.5) * px
    xs = np.interp(z, zc_pts[:, 2], zc_pts[:, 0])
    ys = np.interp(z, zc_pts[:, 2], zc_pts[:, 1])
    half = float(half_width_mm or cfg.zoom_mm / 2)
    nt = max(1, round(2 * half / px))
    tt = (np.arange(nt) - (nt - 1) / 2) * px
    pts = np.zeros((nz, nt, 3))
    pts[..., 0] = xs[:, None] + tt[None, :]
    pts[..., 1] = ys[:, None]
    pts[..., 2] = z[:, None]
    img = sample_points(stack, pts, order=order)
    frame = Frame(pts[0, 0], np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), px, px, (nz, nt))
    h, vt = effective_resolution(stack, (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), px)
    label = f"{stack.key} curved coronal (canal)"
    return Reformat(img, frame, label, stack.key, (h, vt), px, _eff_note(stack, h, vt, px))


def render_reformat(
    ref: Reformat,
    window: Window,
    *,
    cross: ArrayLike | None = None,
    cfg: Config = DEFAULT,
) -> NDArray[np.uint8]:
    """Greyscale RGB of a reformat with orientation letters, a mm ruler, the effective resolution in
    the title and an optional crosshair at an LPS point (projected onto the frame)."""
    img, frame = display_scale(grey_rgb(ref.img, window), ref.frame, min_display_px(cfg))
    img = orientation_letters(img, frame, cfg=cfg)
    if cross is not None:
        img = crosshair(img, frame, cross, WHITE, cfg=cfg)
    img = add_rulers(img, frame, cfg=cfg)
    lines = [
        ref.label,
        f"effective resolution {ref.eff_res_mm[0]:.1f} mm horizontal x {ref.eff_res_mm[1]:.1f} mm vertical",
    ]
    if ref.fraction_inside < 1.0:
        lines.append(f"{100 * (1 - ref.fraction_inside):.0f}% of the image is outside the source (black)")
    return with_title(img, lines, cfg=cfg)
