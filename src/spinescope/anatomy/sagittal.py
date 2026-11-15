"""Mid-sagittal 2-D analysis: mm-grid resampling, spinal canal tracking, body-centre axis, straightening.

Coordinates inside this module are *sagittal-plane* millimetres relative to the origin of one
slice: ``a`` = posterior (mm, increasing posteriorly) and ``s`` = superior (mm, increasing
superiorly). A point is ``origin + a * ap + s * sup`` in patient LPS, with ``ap`` and ``sup``
the unit posterior and superior directions projected into the slice plane.

Spine frame: the *canal track* is the centreline of the thecal sac found as a continuous ridge of
CSF-like signal (bright on T2/STIR, dark on T1). The *body axis* is the canal centreline shifted
anteriorly by an offset (vertebral bodies lie anterior to the canal). The *straightened image*
samples the slice along that axis: arc length ``arc`` along the axis and offset ``t`` along the
anterior normal (``t > 0`` anterior, body centre at ``t = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from ..config import DEFAULT, Config
from ..geometry import Stack

FloatArray = NDArray[np.float64]

#: CSF polarity per contrast: +1 when CSF is bright, -1 when dark. Unknown contrasts try both.
CSF_POLARITY: dict[str, tuple[int, ...]] = {
    "t2": (1,),
    "stir": (1,),
    "pd": (1,),
    "flair": (1,),
    "t1": (-1,),
    "gre": (-1,),
}


def csf_polarities(contrast: str) -> tuple[int, ...]:
    """Candidate CSF polarities for a contrast name (see :data:`CSF_POLARITY`)."""
    return CSF_POLARITY.get(contrast, (1, -1))


def _in_plane(v: FloatArray, normal: FloatArray) -> FloatArray:
    w = v - (v @ normal) * normal
    n = float(np.linalg.norm(w))
    if n < 1e-9:
        raise ValueError("direction is parallel to the slice normal")
    return w / n


@dataclass(eq=False)
class SagImage:
    """One sagittal slice resampled to a regular mm grid (NaN outside the field of view).

    ``img[i, j]`` is the intensity (normalised to 0..1) at superior ``s[i]`` and posterior ``a[j]``.
    """

    img: FloatArray
    a: FloatArray
    s: FloatArray
    origin: FloatArray
    ap: FloatArray
    sup: FloatArray
    normal: FloatArray
    stack_key: str
    slice_index: int
    res: float

    def to_lps(self, a: object, s: object) -> FloatArray:
        """Sagittal-plane mm (broadcastable arrays) -> LPS points, shape ``(..., 3)``."""
        aa = np.asarray(a, dtype=np.float64)[..., None]
        ss = np.asarray(s, dtype=np.float64)[..., None]
        return self.origin + aa * self.ap + ss * self.sup

    def sample(self, a: object, s: object) -> FloatArray:
        """Bilinear intensity at sagittal-plane mm points (NaN outside the image)."""
        aa = np.asarray(a, dtype=np.float64)
        ss = np.asarray(s, dtype=np.float64)
        ia = (aa - self.a[0]) / self.res
        is_ = (ss - self.s[0]) / self.res
        return ndi.map_coordinates(self.img, [is_, ia], order=1, mode="constant", cval=np.nan)


def sagittal_image(stack: Stack, k: int, res: float = DEFAULT.sag_resolution_mm) -> SagImage:
    """Resample slice ``k`` of a sagittal ``stack`` onto a regular ``res`` mm grid."""
    n = stack.normal
    ap = _in_plane(np.array([0.0, 1.0, 0.0]), n)
    sup = _in_plane(np.array([0.0, 0.0, 1.0]), n)
    origin = stack.ipp[k].copy()
    rel = stack.corners(k) - origin
    a_c, s_c = rel @ ap, rel @ sup
    na = int(np.ceil((a_c.max() - a_c.min()) / res)) + 1
    ns = int(np.ceil((s_c.max() - s_c.min()) / res)) + 1
    a = a_c.min() + res * np.arange(na)
    s = s_c.min() + res * np.arange(ns)
    aa, ss = np.meshgrid(a, s)
    pts = origin + aa[..., None] * ap + ss[..., None] * sup
    rel_p = pts - stack.ipp[k]
    c = rel_p @ stack.row_dir / stack.col_mm
    r = rel_p @ stack.col_dir / stack.row_mm
    vals = ndi.map_coordinates(stack.vol[k].astype(np.float64), [r, c], order=1, mode="constant", cval=np.nan)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        raise ValueError("sagittal slice has no in-plane samples")
    lo, hi = np.percentile(finite, [1.0, 99.5])
    img = np.clip((vals - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return SagImage(
        img=img,
        a=a,
        s=s,
        origin=origin,
        ap=ap,
        sup=sup,
        normal=n,
        stack_key=stack.key,
        slice_index=k,
        res=res,
    )


# --------------------------------------------------------------------- canal track
@dataclass
class CanalTrack:
    """Centreline of the thecal sac (CSF ridge) in one sagittal slice."""

    s: FloatArray  # superior mm of the track samples (ascending)
    a: FloatArray  # posterior mm of the canal centre (polynomial fit)
    coef: FloatArray  # polynomial a(s)
    score: float  # summed ridge contrast along the track (higher = clearer canal)
    polarity: int

    def a_at(self, s: object) -> FloatArray:
        """Canal centre a(s); linearly extrapolated beyond the track ends."""
        sv = np.asarray(s, dtype=np.float64)
        s0, s1 = float(self.s[0]), float(self.s[-1])
        sc = np.clip(sv, s0, s1)
        base = np.polyval(self.coef, sc)
        d = np.polyder(self.coef)
        slope_lo = float(np.polyval(d, s0))
        slope_hi = float(np.polyval(d, s1))
        out = (
            base + np.where(sv < s0, slope_lo * (sv - s0), 0.0) + np.where(sv > s1, slope_hi * (sv - s1), 0.0)
        )
        return np.asarray(out, dtype=np.float64)


def ridge_response(sag: SagImage, polarity: int, cfg: Config = DEFAULT) -> FloatArray:
    """CSF-likeness per pixel along the AP direction: canal core minus the brighter flank.

    The core is the mean over ``±canal_half_width_mm``. Each flank is a narrow (3 mm) window centred
    ``canal_flank_offset_mm`` beyond the core edge, so flanks sit just outside the thecal sac. The
    core must beat *both* flanks: a body-to-air or fat-to-air edge is bright on one side only.
    """
    img = sag.img if polarity > 0 else 1.0 - sag.img
    x = np.where(np.isfinite(img), img, 0.5)
    na = x.shape[1]
    w = max(1, round(cfg.canal_half_width_mm / sag.res))
    f = w + max(1, round(cfg.canal_flank_offset_mm / sag.res))
    core = ndi.uniform_filter1d(x, size=2 * w + 1, axis=1, mode="nearest")
    flank_avg = ndi.uniform_filter1d(x, size=3, axis=1, mode="nearest")
    idx = np.arange(na)
    left = flank_avg[:, np.clip(idx - f, 0, na - 1)]
    right = flank_avg[:, np.clip(idx + f, 0, na - 1)]
    return np.asarray(core - np.maximum(left, right) - cfg.canal_min_contrast, dtype=np.float64)


def track_canal(sag: SagImage, polarity: int, cfg: Config = DEFAULT) -> CanalTrack | None:
    """Best continuous CSF ridge in the slice (max-sum path with free start and end).

    Returns ``None`` when no ridge of at least ``canal_min_length_mm`` exists.
    """
    r = ridge_response(sag, polarity, cfg)
    ns, na = r.shape
    # Prior: a spine-centred field puts the thecal sac in the central AP part of the image (keeps
    # anterior fat and bowel edges, which can be as ridge-like as the sac, out of the search).
    frac_lo, frac_hi = cfg.canal_ap_fraction
    keep = (np.arange(na) >= frac_lo * na) & (np.arange(na) <= frac_hi * na)
    r = np.where(keep[None, :], r, -1.0)
    m = max(1, int(np.ceil(cfg.canal_max_slope)))
    d_acc = np.empty_like(r)
    back = np.full(r.shape, -1, dtype=np.int64)
    d_acc[0] = r[0]
    idx = np.arange(na)
    for i in range(1, ns):
        prev = d_acc[i - 1]
        best = np.full(na, -np.inf)
        arg = np.full(na, -1, dtype=np.int64)
        for dj in range(-m, m + 1):
            cand = np.full(na, -np.inf)
            if dj >= 0:
                cand[: na - dj] = prev[dj:]
            else:
                cand[-dj:] = prev[: na + dj]
            better = cand > best
            best = np.where(better, cand, best)
            arg = np.where(better, idx + dj, arg)
        take = best > 0.0
        d_acc[i] = r[i] + np.where(take, best, 0.0)
        back[i] = np.where(take, arg, -1)
    flat = int(np.argmax(d_acc))
    i, j = divmod(flat, na)
    score = float(d_acc[i, j])
    path: list[tuple[int, int]] = []
    while True:
        path.append((i, j))
        pj = int(back[i, j])
        if pj < 0 or i == 0:
            break
        i -= 1
        j = pj
    path.reverse()
    rows = np.array([p[0] for p in path])
    cols = np.array([p[1] for p in path])
    s_pts = sag.s[rows]
    if s_pts[-1] - s_pts[0] < cfg.canal_min_length_mm:
        return None
    a_pts = sag.a[cols]
    deg = 3 if len(path) >= 20 else 1
    coef = np.polyfit(s_pts, a_pts, deg)
    track = CanalTrack(
        s=s_pts,
        a=np.polyval(coef, s_pts),
        coef=coef,
        score=score,
        polarity=polarity,
    )
    return track


# ------------------------------------------------------------------- body axis
@dataclass
class BodyAxis:
    """Vertebral-body centre line in sagittal mm, sampled on a uniform arc-length grid."""

    arc: FloatArray  # arc length (mm) from the lowest sample, ascending superiorly
    a: FloatArray  # posterior mm of the body centre line
    s: FloatArray  # superior mm of the body centre line
    ta: FloatArray  # unit tangent (superior) components
    ts: FloatArray
    na: FloatArray  # unit anterior normal components
    ns: FloatArray
    offset_mm: float

    def point(self, arc: object, t: object) -> tuple[FloatArray, FloatArray]:
        """(a, s) of the straightened coordinate (arc, t); ``t > 0`` is anterior."""
        ar = np.asarray(arc, dtype=np.float64)
        tv = np.asarray(t, dtype=np.float64)
        a0 = np.interp(ar, self.arc, self.a)
        s0 = np.interp(ar, self.arc, self.s)
        nna = np.interp(ar, self.arc, self.na)
        nns = np.interp(ar, self.arc, self.ns)
        return a0 + tv * nna, s0 + tv * nns

    def tangent(self, arc: object) -> tuple[FloatArray, FloatArray]:
        return np.interp(np.asarray(arc, float), self.arc, self.ta), np.interp(
            np.asarray(arc, float), self.arc, self.ts
        )

    def normal(self, arc: object) -> tuple[FloatArray, FloatArray]:
        return np.interp(np.asarray(arc, float), self.arc, self.na), np.interp(
            np.asarray(arc, float), self.arc, self.ns
        )


def build_body_axis(sag: SagImage, track: CanalTrack, offset_mm: float, step_mm: float = 0.5) -> BodyAxis:
    """Body centre line = canal centreline shifted anteriorly by ``offset_mm`` along its normal."""
    s_dense = np.linspace(sag.s[0], sag.s[-1], max(int((sag.s[-1] - sag.s[0]) / 0.25), 2))
    a_c = track.a_at(s_dense)
    da = np.gradient(a_c, s_dense)
    norm = np.sqrt(1.0 + da * da)
    ta, ts = da / norm, 1.0 / norm
    na_, ns_ = -1.0 / norm, da / norm  # anterior normal (negative a is anterior)
    a_b = a_c + offset_mm * na_
    s_b = s_dense + offset_mm * ns_
    seg = np.hypot(np.diff(a_b), np.diff(s_b))
    arc_dense = np.concatenate([[0.0], np.cumsum(seg)])
    arc = np.arange(0.0, arc_dense[-1] + 1e-9, step_mm)
    return BodyAxis(
        arc=arc,
        a=np.interp(arc, arc_dense, a_b),
        s=np.interp(arc, arc_dense, s_b),
        ta=np.interp(arc, arc_dense, ta),
        ts=np.interp(arc, arc_dense, ts),
        na=np.interp(arc, arc_dense, na_),
        ns=np.interp(arc, arc_dense, ns_),
        offset_mm=offset_mm,
    )


def straighten(sag: SagImage, axis: BodyAxis, t_vals: FloatArray) -> FloatArray:
    """Image sampled on the (arc, t) grid, shape ``(len(axis.arc), len(t_vals))``; NaN outside."""
    arc = axis.arc[:, None]
    tt = np.asarray(t_vals, dtype=np.float64)[None, :]
    a_pts, s_pts = axis.point(arc, tt)
    return sag.sample(a_pts, s_pts)
