"""Mid-sagittal plane by mirror symmetry of the sagittal slices, as a curve x(z).

Idea. The spine, its paraspinal muscles and the body outline are left-right symmetric about the
mid-sagittal plane. For a candidate plane ``x0`` the sagittal slices at ``x0 - d`` and ``x0 + d`` are
compared pixel by pixel (normalised cross-correlation) over several offsets ``d``. The symmetry axis is
the candidate with the highest mean similarity. No segmentation is needed, and the same test works on
T1, T2 and STIR stacks.

Steps (every number lives in :class:`spinescope.config.Config`, ``midline_*``):

1. Global peak: the symmetry curve over the whole stack, averaged over two preprocessings (raw
   intensities, and a high-pass that keeps structures and drops slowly varying coil and fat signal).
   The peak is refined to sub-grid precision by a parabola and gets a prominence (0..1).
2. Patch votes: the same curve in overlapping square patches (``midline_patch_mm``). Each patch has its
   own symmetry axis, so bowel or fat asymmetry outside the spine is outvoted by the spine. Patches with
   prominence below ``midline_patch_min_prominence`` are ignored.
3. Curve: per vertical position, the weighted median of its votes gives a knot; the curve is the
   piecewise-linear interpolation of the knots. A curved spine (scoliosis, rotation about the sagittal
   stack normal) is followed, not forced into a straight plane.
4. Scalar midline: the median of the curve over the spine's vertical extent. This is the value the
   labeller and the scorer use. Per-slice weights (Gaussian in x) and a confidence (peak prominence,
   vote agreement and residual) are returned with it.

Known limit: in-plane (axial) rotation of the whole anatomy tilts the mid-sagittal plane about the
vertical axis. A sagittal stack cannot represent that tilt, so the estimate is biased toward the average
over the cross-section (about 4 mm at 8 degrees on the phantom). See docs/labeling/lab05_midline.md.

Coordinates are LPS mm; ``x`` is the patient left-right coordinate. Missing inputs (no sagittal stack, a
non-sagittal normal, too few slices to span the midline) give a result with ``x_mm=None`` and a warning,
never an exception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .sagittal import CanalTrack, sagittal_image, track_canal

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: Half-width (mm) around a candidate's peak that is excluded when looking for a second peak.
_SECOND_PEAK_EXCLUDE_MM = 4.0


@dataclass
class MidlineEstimate:
    """Mid-sagittal estimate for one sagittal stack.

    ``x_mm`` is the scalar midline (LPS x). ``knots_z`` and ``knots_x`` (superior mm, x mm) define the
    curve; :meth:`x_at` interpolates it and holds the end values beyond the knots. ``slice_weights`` has one
    entry per slice of the stack (stack order), 1 on the midline and falling off with distance. ``bands``
    lists the knots as ``(z, x, mean vote prominence, used)``.
    """

    stack_key: str
    x_mm: float | None
    knots_z: tuple[float, ...] = ()
    knots_x: tuple[float, ...] = ()
    z_range_mm: tuple[float, float] | None = None
    global_x_mm: float | None = None
    global_prominence: float = 0.0
    slice_weights: FloatArray = field(default_factory=lambda: np.zeros(0))
    bands: list[tuple[float, float, float, bool]] = field(default_factory=list)
    axial_x_mm: float | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def x_at(self, z: Any) -> FloatArray:
        """Midline x (mm) at superior position(s) ``z`` (mm); NaN when there is no estimate."""
        zz = np.asarray(z, dtype=np.float64)
        if self.x_mm is None:
            return np.full(zz.shape, np.nan)
        if not self.knots_z:
            return np.full(zz.shape, float(self.x_mm))
        return np.interp(zz, np.asarray(self.knots_z), np.asarray(self.knots_x))


# ------------------------------------------------------------------ slice geometry
def _vertical_volume(stack: Stack) -> tuple[NDArray[Any], FloatArray, float, float]:
    """Stack volume with the superior axis on axis 1, its superior positions (mm), and the two steps.

    Returns ``(vol, z, step_vertical_mm, step_horizontal_mm)`` for the slices in stack order.
    """
    if abs(float(stack.col_dir[2])) >= abs(float(stack.row_dir[2])):
        vol = stack.vol
        step_v, step_h = stack.row_mm, stack.col_mm
        dz = float(stack.col_dir[2])
    else:
        vol = np.transpose(stack.vol, (0, 2, 1))
        step_v, step_h = stack.col_mm, stack.row_mm
        dz = float(stack.row_dir[2])
    z = float(stack.ipp[0, 2]) + np.arange(vol.shape[1]) * step_v * dz
    return np.asarray(vol), np.asarray(z, dtype=np.float64), float(step_v), float(step_h)


def _sample_x(x_k: FloatArray, vol: NDArray[np.float32], xq: FloatArray) -> NDArray[np.float32]:
    """Slices interpolated linearly in x at positions ``xq`` (NaN outside the stack)."""
    f = np.interp(xq, x_k, np.arange(len(x_k)))
    out_of = (xq < x_k[0]) | (xq > x_k[-1])
    i0 = np.clip(np.floor(f).astype(int), 0, len(x_k) - 1)
    i1 = np.clip(i0 + 1, 0, len(x_k) - 1)
    w = (f - i0).astype(np.float32)[:, None, None]
    out = (1 - w) * vol[i0] + w * vol[i1]
    out[out_of] = np.nan
    return out


def _ncc(a: NDArray[np.float32], b: NDArray[np.float32]) -> FloatArray:
    """Normalised cross-correlation per candidate; ``a`` and ``b`` are (m, R, C) with NaN outside."""
    a2 = a.reshape(a.shape[0], -1).astype(np.float64)
    b2 = b.reshape(b.shape[0], -1).astype(np.float64)
    ok = np.isfinite(a2) & np.isfinite(b2)
    a2 = np.where(ok, a2, 0.0)
    b2 = np.where(ok, b2, 0.0)
    n = np.maximum(ok.sum(axis=1), 1)
    ma = a2.sum(axis=1) / n
    mb = b2.sum(axis=1) / n
    a0 = np.where(ok, a2 - ma[:, None], 0.0)
    b0 = np.where(ok, b2 - mb[:, None], 0.0)
    num = (a0 * b0).sum(axis=1)
    den = np.sqrt((a0 * a0).sum(axis=1) * (b0 * b0).sum(axis=1)) + 1e-12
    return np.asarray(num / den, dtype=np.float64)


def _symmetry_curve(
    x_k: FloatArray, vol: NDArray[np.float32], cand: FloatArray, dvals: FloatArray
) -> FloatArray:
    """Mean mirror similarity for each candidate plane, over offsets ``dvals``."""
    s = np.zeros(len(cand))
    for d in dvals:
        s += _ncc(_sample_x(x_k, vol, cand + d), _sample_x(x_k, vol, cand - d))
    return s / len(dvals)


def _highpass(vol: NDArray[np.float32], sigma_v: float, sigma_h: float) -> NDArray[np.float32]:
    """Structure-only copy of the stack: the image minus a Gaussian blur, slice by slice."""
    blur = ndi.gaussian_filter(vol.astype(np.float64), sigma=(0.0, sigma_v, sigma_h))
    return (vol - blur).astype(np.float32)


def _peak(cand: FloatArray, s: FloatArray) -> tuple[float, float]:
    """Sub-grid position of the highest symmetry value and its prominence (0..1).

    Prominence is the peak height above the best value outside ``_SECOND_PEAK_EXCLUDE_MM``, relative to
    the peak height above the median. A broad or double-humped curve scores low.
    """
    i = int(np.argmax(s))
    x = float(cand[i])
    if 0 < i < len(s) - 1:
        den = s[i - 1] - 2 * s[i] + s[i + 1]
        if den < -1e-12:
            off = 0.5 * (s[i - 1] - s[i + 1]) / den
            x += float(np.clip(off, -0.5, 0.5)) * float(cand[1] - cand[0])
    away = np.abs(cand - cand[i]) > _SECOND_PEAK_EXCLUDE_MM
    second = float(np.max(s[away])) if away.any() else float(np.min(s))
    height = float(s[i] - np.median(s))
    prom = (float(s[i]) - second) / height if height > 1e-9 else 0.0
    return x, float(np.clip(prom, 0.0, 1.0))


def _stack_arrays(stack: Stack, cfg: Config) -> tuple[FloatArray, NDArray[np.float32], FloatArray, float]:
    """Slices sorted by x, downsampled to about ``midline_grid_mm`` and normalised to 0..1.

    Returns ``(x_sorted, vol_sorted, z_vertical, pixel_mm)`` where ``pixel_mm`` is the in-plane step of the
    downsampled grid (used to convert mm scales to pixels).
    """
    x_k = np.asarray(stack.ipp[:, 0], dtype=np.float64)
    order = np.argsort(x_k, kind="stable")
    vol, z, step_v, step_h = _vertical_volume(stack)
    stride = max(1, round(cfg.midline_grid_mm / min(step_v, step_h)))
    v = vol[order][:, ::stride, ::stride].astype(np.float32)
    z_v = z[::stride]
    lo, hi = np.percentile(v, [1.0, 99.5])
    v = np.clip((v - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0).astype(np.float32)
    return x_k[order], v, z_v, float(stride * min(step_v, step_h))


def _weighted_median(x: FloatArray, w: FloatArray) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, 0.5 * cw[-1])])


@dataclass
class _Patch:
    zc: float  # superior centre (mm)
    x: float  # local symmetry axis (mm)
    prom: float  # prominence of the local peak (0..1)


def _patch_votes(
    v: NDArray[np.float32],
    hp: NDArray[np.float32],
    x_k: FloatArray,
    z_v: FloatArray,
    cand: FloatArray,
    dvals: FloatArray,
    pixel_mm: float,
    cfg: Config,
) -> list[_Patch]:
    """Local symmetry axis of each overlapping square patch (the patch's own mirror peak)."""
    _, nr, nc = v.shape
    size = max(4, round(cfg.midline_patch_mm / pixel_mm))
    step = max(1, round(cfg.midline_patch_step_mm / pixel_mm))
    rows_starts = list(range(0, max(nr - size, 0) + 1, step)) or [0]
    cols_starts = list(range(0, max(nc - size, 0) + 1, step)) or [0]
    out: list[_Patch] = []
    for r0 in rows_starts:
        r1 = min(nr, r0 + size)
        for c0 in cols_starts:
            c1 = min(nc, c0 + size)
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue
            raw = _symmetry_curve(x_k, v[:, r0:r1, c0:c1], cand, dvals)
            hpc = _symmetry_curve(x_k, hp[:, r0:r1, c0:c1], cand, dvals)
            x_p, prom = _peak(cand, 0.5 * (raw + hpc))
            out.append(_Patch(zc=float(np.mean(z_v[r0:r1])), x=x_p, prom=prom))
    return out


def estimate_stack_midline(stack: Stack, config: Config | None = None) -> MidlineEstimate:
    """Mid-sagittal curve x(z) for one sagittal stack, from mirror symmetry (no axial data used).

    ``x_mm`` is None when the stack is not sagittal (normal far from x), has fewer than three slices, or
    spans too little for a symmetry axis; the reason is in ``warnings``.
    """
    cfg = config or DEFAULT
    key = stack.key
    empty = MidlineEstimate(stack_key=key, x_mm=None, slice_weights=np.zeros(stack.n))
    if abs(float(stack.normal[0])) < 0.95:
        empty.warnings.append(f"{key}: not a sagittal stack; symmetry midline not estimated")
        return empty
    if stack.n < 3:
        empty.warnings.append(f"{key}: fewer than three sagittal slices; symmetry midline not estimated")
        return empty
    dmin, dmax, dstep = cfg.midline_sym_d_mm
    x_k, v, z_v, pixel_mm = _stack_arrays(stack, cfg)
    span = float(x_k[-1] - x_k[0])
    if span < 2 * dmin + 2 * cfg.midline_grid_mm:
        empty.warnings.append(f"{key}: sagittal stack spans {span:.0f} mm, too little to show the midline")
        return empty
    dmax_eff = min(dmax, span / 2 - cfg.midline_grid_mm)
    dvals = np.arange(dmin, dmax_eff + 1e-9, dstep)
    d_hi = float(dvals[-1])
    cand = np.arange(x_k[0] + d_hi, x_k[-1] - d_hi + 1e-9, cfg.midline_grid_mm)
    if len(cand) < 3:
        empty.warnings.append(f"{key}: too few candidate planes for a symmetry axis")
        return empty

    sig_px = cfg.midline_hp_sigma_mm / pixel_mm
    hp = _highpass(v, sig_px, sig_px)
    raw_g = _symmetry_curve(x_k, v, cand, dvals)
    hp_g = _symmetry_curve(x_k, hp, cand, dvals)
    x_glob, prom_glob = _peak(cand, 0.5 * (raw_g + hp_g))

    votes = _patch_votes(v, hp, x_k, z_v, cand, dvals, pixel_mm, cfg)
    z_lo, z_hi = float(np.min(z_v)), float(np.max(z_v))
    ev = [
        f"global mirror-symmetry peak at x={x_glob:.1f} mm (prominence {prom_glob:.2f}, "
        f"{len(dvals)} offsets {float(dvals[0]):.0f}-{d_hi:.0f} mm)"
    ]
    warns: list[str] = []
    strong = [p for p in votes if p.prom >= cfg.midline_patch_min_prominence]

    # One knot per vertical position: the weighted median of the strong votes at that position.
    knots: list[tuple[float, float, float, int]] = []
    for zc in sorted({p.zc for p in strong}):
        grp = [p for p in strong if p.zc == zc]
        xs = np.array([p.x for p in grp])
        ws = np.array([p.prom for p in grp])
        knots.append((zc, _weighted_median(xs, ws), float(ws.mean()), len(grp)))

    if len(knots) >= 2:
        kz = np.array([k[0] for k in knots])
        kx = np.array([k[1] for k in knots])
        z_grid = np.linspace(kz.min(), kz.max(), 41)
        x_mid = float(np.median(np.interp(z_grid, kz, kx)))
        xs_v = np.array([p.x for p in strong])
        ws_v = np.array([p.prom for p in strong])
        zs_v = np.array([p.zc for p in strong])
        resid = xs_v - np.interp(zs_v, kz, kx)
        agree = float(np.average(np.abs(resid) <= cfg.midline_fit_tukey_mm, weights=ws_v))
        scatter = float(np.median(np.abs(resid)))
        resid_score = float(np.exp(-scatter / max(cfg.midline_fit_tukey_mm, 1e-6)))
        ev.append(
            f"{len(strong)} of {len(votes)} local symmetry patches (prominence >= "
            f"{cfg.midline_patch_min_prominence:.2f}) vote in {len(knots)} vertical positions: "
            f"curve from x={kx[0]:.1f} to {kx[-1]:.1f} mm, median x {x_mid:.1f} mm, "
            f"median vote residual {scatter:.1f} mm"
        )
        knots_z, knots_x = tuple(float(z) for z in kz), tuple(float(x) for x in kx)
        z_span: tuple[float, float] = (float(kz.min()), float(kz.max()))
        band_rows = [(k[0], k[1], k[2], True) for k in knots]
    else:
        x_mid = x_glob
        knots_z, knots_x = (), ()
        agree = 0.5
        resid_score = 0.5
        z_span = (z_lo, z_hi)
        band_rows = []
        warns.append(f"{key}: too few clear local symmetry votes; using the global plane only")
        ev.append("local votes unclear: global plane used")

    conf = float(np.clip(0.4 * prom_glob + 0.4 * agree + 0.2 * resid_score, 0.0, 1.0))
    if len(knots) < 2:
        conf = float(min(conf, 0.5))
    if prom_glob < 0.2:
        warns.append(f"{key}: symmetry peak is weak (prominence {prom_glob:.2f}); midline is uncertain")
    xs_stack = np.asarray(stack.ipp[:, 0], dtype=np.float64)
    weights = np.exp(-0.5 * ((xs_stack - x_mid) / cfg.midline_slice_weight_mm) ** 2)
    return MidlineEstimate(
        stack_key=key,
        x_mm=float(x_mid),
        knots_z=knots_z,
        knots_x=knots_x,
        z_range_mm=z_span,
        global_x_mm=float(x_glob),
        global_prominence=float(prom_glob),
        slice_weights=weights,
        bands=band_rows,
        confidence=conf,
        evidence=ev,
        warnings=warns,
    )


def estimate_midsagittal(study: Study, config: Config | None = None) -> MidlineEstimate:
    """Mid-sagittal curve for a study, from its best sagittal stack (``study.best("sag")``).

    Axial slices that pass through the canal (``study.find('ax')``), when present, refine the scalar
    midline through :func:`spinescope.anatomy.midline.refine_midline_axial`: their CSF-blob centres
    replace the scalar when they agree with the sagittal estimate within ``midline_axial_tol_mm``.
    """
    cfg = config or DEFAULT
    stack = study.best("sag")
    if stack is None:
        return MidlineEstimate(
            stack_key="", x_mm=None, warnings=["no sagittal series: midline not estimated"]
        )
    est = estimate_stack_midline(stack, cfg)
    if est.x_mm is None or not any(s.plane == "ax" for s in study.stacks.values()):
        return est
    # Lazy import: midline.py imports this module.
    from .midline import MidlineChoice, pols_for, refine_midline_axial

    k_near = int(np.argmin(np.abs(stack.ipp[:, 0] - est.x_mm)))
    found: list[tuple[int, CanalTrack]] = []
    for p in pols_for(stack):
        tr = track_canal(sagittal_image(stack, k_near, cfg.sag_resolution_mm), p, cfg)
        if tr is not None:
            found.append((p, tr))
    if not found:
        est.evidence.append("no canal track on the nearest slice: axial refinement skipped")
        return est
    pol, track = max(found, key=lambda pt: pt[1].score)
    choice = MidlineChoice(
        slice_index=k_near,
        polarity=pol,
        plane_coord=est.x_mm * float(stack.normal[0]),
        track=track,
    )
    ax_x, ax_ev, ax_w = refine_midline_axial(study, stack, choice, cfg)
    est.evidence.extend(ax_ev)
    est.warnings.extend(ax_w)
    if ax_x is not None:
        shift = ax_x - est.x_mm
        est.axial_x_mm = float(ax_x)
        est.x_mm = float(ax_x)
        est.knots_x = tuple(float(x) + shift for x in est.knots_x)
        est.confidence = float(min(1.0, est.confidence + 0.1))
    return est
