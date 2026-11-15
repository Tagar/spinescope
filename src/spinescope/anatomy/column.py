"""Vertebral column in the straightened mid-sagittal image: endplates, disc/body chain, disc geometry, conus.

Endplates are thin dark cortical lines, so along the body axis they are local *minima* of the
AP-averaged profile. A chain DP picks endplate lines whose spacing alternates between disc-height
intervals and vertebral-body-height intervals (ranges in mm from :class:`Config`), so spurious
lines inside a body or between two discs cannot form a plausible column on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from ..config import DEFAULT, Config
from .sagittal import CanalTrack, SagImage

FloatArray = NDArray[np.float64]

_KIND = {1: "disc", 2: "body", 3: "gap"}


@dataclass(frozen=True)
class Line:
    arc: float  # arc length (mm) of the dark endplate line at the body-centre plane
    strength: float  # normalised darkness of the line relative to its surroundings (~1 = typical)


@dataclass(frozen=True)
class Segment:
    lo: float  # arc of the lower endplate (mm)
    hi: float  # arc of the upper endplate (mm)
    kind: str  # 'disc' | 'body' | 'gap'
    lo_strength: float
    hi_strength: float


@dataclass
class DiscFit:
    """Geometry of one disc on the straightened image."""

    arc: float  # arc length of the disc centre
    t: float  # AP offset of the disc centre (mm, anterior > 0)
    t_ant: float | None  # anterior margin offset (None when endplates could not be followed)
    t_post: float | None
    height: float  # mm, along the axis
    slope: float  # d(arc)/d(t) of the disc mid-plane (tilt in the sagittal plane)
    lo_strength: float
    hi_strength: float
    n_followed: int  # number of AP samples where both endplates were followed
    evidence: list[str] = field(default_factory=list)


def interp_nans(v: FloatArray) -> FloatArray:
    """Replace NaNs by linear interpolation (edges by nearest finite value)."""
    out = np.asarray(v, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    if bad.all():
        return np.full_like(out, 0.5)
    if bad.any():
        idx = np.arange(len(out))
        out[bad] = np.interp(idx[bad], idx[~bad], out[~bad])
    return out


def endplate_profile(
    straight: FloatArray, t: FloatArray, t_half: float
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """AP-averaged intensity along the axis (samples with |t| <= t_half), and a mask of arc rows that
    lie inside the image (rows mostly outside the field of view are not used for lines)."""
    cols = np.abs(t) <= t_half + 1e-9
    sub = straight[:, cols]
    valid = np.isfinite(sub).mean(axis=1) >= 0.9
    with np.errstate(all="ignore"):
        prof = np.nanmean(sub, axis=1)
    return interp_nans(prof), valid


def find_lines(
    profile: FloatArray,
    arc: FloatArray,
    cfg: Config = DEFAULT,
    valid: NDArray[np.bool_] | None = None,
) -> list[Line]:
    """Dark endplate lines: local maxima of the negated second derivative of the profile.

    Lines within about two smoothing widths of a row outside the field of view are discarded, since
    the derivative there is set by the image edge rather than by anatomy.
    """
    step = float(arc[1] - arc[0]) if len(arc) > 1 else 1.0
    sigma = max(cfg.endplate_sigma_mm / step, 0.5)
    lsig = -ndi.gaussian_filter1d(profile, sigma=sigma, order=2, mode="nearest")
    scale = float(np.percentile(np.abs(lsig), 95)) + 1e-9
    ok = np.ones(len(profile), dtype=bool) if valid is None else valid.copy()
    if valid is not None:
        bad = ~valid
        reach = int(np.ceil(3 * sigma))
        ok = ~ndi.binary_dilation(bad, iterations=reach) if bad.any() else ok
    lines: list[Line] = []
    for i in range(1, len(lsig) - 1):
        if not ok[i]:
            continue
        if lsig[i] >= lsig[i - 1] and lsig[i] > lsig[i + 1] and lsig[i] > 0:
            y0, y1, y2 = lsig[i - 1], lsig[i], lsig[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            off = float(np.clip(off, -0.5, 0.5))
            lines.append(Line(arc=float(arc[i] + off * step), strength=float(y1 / scale)))
    return lines


def chain_lines(lines: list[Line], cfg: Config = DEFAULT) -> tuple[float, list[Segment]]:
    """Best chain of endplate lines whose intervals alternate disc, body (gaps allowed at a cost)."""
    n = len(lines)
    if n < 2:
        return 0.0, []
    lo_d, hi_d = cfg.disc_height_mm
    lo_b, hi_b = cfg.body_height_mm
    lo_g, hi_g = cfg.gap_mm
    pen = cfg.line_penalty
    arcs = np.array([ln.arc for ln in lines])
    strength = np.array([ln.strength for ln in lines])
    neg = -1e18
    best = np.full((n, 4), neg)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    for j in range(n):
        best[j, 0] = strength[j] - pen
        for i in range(j):
            d = arcs[j] - arcs[i]
            if d <= 0:
                continue
            options: list[tuple[int, tuple[int, ...], float]] = []
            if lo_d <= d <= hi_d:
                options.append((1, (0, 2, 3), 0.0))
            if lo_b <= d <= hi_b:
                options.append((2, (0, 1, 3), 0.0))
            if lo_g <= d <= hi_g:
                options.append((3, (0, 1, 2, 3), cfg.gap_penalty))
            for kind, prevs, extra in options:
                for pk in prevs:
                    if best[i, pk] <= neg / 2:
                        continue
                    val = best[i, pk] + strength[j] - pen - extra
                    if val > best[j, kind]:
                        best[j, kind] = val
                        back[(j, kind)] = (i, pk)
    jk = np.unravel_index(int(np.argmax(best[:, 1:])), (n, 3))
    j, k = int(jk[0]), int(jk[1]) + 1
    total = float(best[j, k])
    if total <= neg / 2:
        return 0.0, []
    segs: list[Segment] = []
    while k != 0 and (j, k) in back:
        i, pk = back[(j, k)]
        segs.append(
            Segment(
                lo=float(arcs[i]),
                hi=float(arcs[j]),
                kind=_KIND[k],
                lo_strength=float(strength[i]),
                hi_strength=float(strength[j]),
            )
        )
        j, k = i, pk
    segs.reverse()
    return total, segs


def _robust_line(t: FloatArray, y: FloatArray) -> tuple[float, float]:
    keep = np.ones(len(t), dtype=bool)
    a, b = 0.0, 0.0
    for _ in range(3):
        if keep.sum() < 2:
            break
        b, a = np.polyfit(t[keep], y[keep], 1)
        res = y - (a + b * t)
        sd = 1.4826 * float(np.median(np.abs(res[keep]))) + 1e-6
        keep = np.abs(res) <= max(1.0, 2.5 * sd)
    return float(a), float(b)


def _min_near(
    straight: FloatArray,
    arc: FloatArray,
    j: int,
    ref: float,
    half_mm: float,
    step: float,
    bounds: tuple[float, float] = (-np.inf, np.inf),
) -> tuple[float, float]:
    """Darkest point within ``half_mm`` of ``ref`` (and inside ``bounds``) on AP column ``j``.

    Returns (arc, contrast against the 8 mm surroundings).
    """
    centre = int(np.argmin(np.abs(arc - ref)))
    half = max(1, round(half_mm / step))
    b_lo = int(np.searchsorted(arc, bounds[0]))
    b_hi = int(np.searchsorted(arc, bounds[1], side="right")) - 1
    lo_i = max(centre - half, 0, b_lo)
    hi_i = min(centre + half, len(arc) - 1, b_hi)
    if hi_i < lo_i:
        return float("nan"), 0.0
    seg = straight[lo_i : hi_i + 1, j]
    if not np.isfinite(seg).any():
        return float("nan"), 0.0
    seg = np.where(np.isfinite(seg), seg, np.nanmax(seg))
    k = int(np.argmin(seg))
    p = float(arc[lo_i + k])
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            p += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) * step
    wide = round(8.0 / step)
    base = np.nanmedian(straight[max(centre - wide, 0) : min(centre + wide, len(arc) - 1) + 1, j])
    return p, float(base - seg[k]) if np.isfinite(base) else 0.0


def _follow_line(
    straight: FloatArray,
    arc: FloatArray,
    t: FloatArray,
    target: float,
    cfg: Config,
    bounds: tuple[float, float] = (-np.inf, np.inf),
) -> tuple[FloatArray, FloatArray]:
    """Track one endplate across the AP extent, outward from the sample nearest t = 0.

    The line may be tilted, so each AP sample searches only near the previously tracked position. A
    sample updates the track when its darkness contrast exceeds ``endplate_min_contrast``; after
    several misses in a row the track in that direction stops.
    """
    step = float(arc[1] - arc[0])
    n = len(t)
    pos = np.full(n, np.nan)
    con = np.zeros(n)
    j0 = int(np.argmin(np.abs(t)))
    thr = cfg.endplate_min_contrast
    for direction in (1, -1):
        ref = target
        misses = 0
        first = True
        j = j0 if direction == 1 else j0 - 1
        while 0 <= j < n:
            half_mm = cfg.endplate_search_mm if first else 1.5
            p, c = _min_near(straight, arc, j, ref, half_mm, step, bounds)
            pos[j], con[j] = p, c
            if np.isfinite(p) and c > thr:
                ref = p
                first = False
                misses = 0
            else:
                misses += 1
                if misses > 8:
                    break
            j += direction
    return pos, con


def fit_disc(
    straight: FloatArray,
    arc: FloatArray,
    t: FloatArray,
    seg: Segment,
    cfg: Config = DEFAULT,
) -> DiscFit:
    """Follow both endplates across the AP extent, then fit the disc mid-plane and its margins.

    Each endplate is confined to its own half of the interval so the two tracks cannot lock onto
    the same line. The disc centre is the midpoint of the followed AP extent when that is at least
    ``disc_margin_min_mm`` long; otherwise it sits on the body axis (t = 0), which is the body centre.
    """
    mid = (seg.lo + seg.hi) / 2
    pos_lo, con_lo = _follow_line(straight, arc, t, seg.lo, cfg, (-np.inf, mid))
    pos_hi, con_hi = _follow_line(straight, arc, t, seg.hi, cfg, (mid, np.inf))
    thr = cfg.endplate_min_contrast
    ok = (con_lo > thr) & (con_hi > thr) & np.isfinite(pos_lo) & np.isfinite(pos_hi)
    runs = _runs(ok, max_gap=2)
    ev: list[str] = []
    if not runs or max(b - a for a, b in runs) < 4:
        ev.append("endplates not followed across the disc: centre from the chain only")
        return DiscFit(
            arc=mid,
            t=0.0,
            t_ant=None,
            t_post=None,
            height=seg.hi - seg.lo,
            slope=0.0,
            lo_strength=seg.lo_strength,
            hi_strength=seg.hi_strength,
            n_followed=0,
            evidence=ev,
        )
    run = min(runs, key=lambda r: (0 if r[0] <= _nearest(t, 0.0) <= r[1] else 1, -(r[1] - r[0])))
    j0, j1 = run
    sel = slice(j0, j1 + 1)
    a0, b0 = _robust_line(t[sel], pos_lo[sel])
    a1, b1 = _robust_line(t[sel], pos_hi[sel])
    t_lo, t_hi = float(t[j0]), float(t[j1])
    ap_mm = t_hi - t_lo
    slope = 0.5 * (b0 + b1)
    if ap_mm >= cfg.disc_margin_min_mm:
        t_c = 0.5 * (t_lo + t_hi)
        ev.append(f"endplates followed over {ap_mm:.0f} mm AP ({j1 - j0 + 1} samples)")
        t_ant: float | None = t_hi
        t_post: float | None = t_lo
    else:
        t_c = 0.0
        t_ant = t_post = None
        ev.append(f"endplates followed over only {ap_mm:.0f} mm: centre on the body axis, margins unknown")
    s_lo_c = a0 + b0 * t_c
    s_hi_c = a1 + b1 * t_c
    return DiscFit(
        arc=float(0.5 * (s_lo_c + s_hi_c)),
        t=float(t_c),
        t_ant=t_ant,
        t_post=t_post,
        height=float(s_hi_c - s_lo_c),
        slope=float(slope),
        lo_strength=float(np.median(con_lo[sel])),
        hi_strength=float(np.median(con_hi[sel])),
        n_followed=int(j1 - j0 + 1),
        evidence=ev,
    )


def _nearest(t: FloatArray, v: float) -> int:
    return int(np.argmin(np.abs(t - v)))


def _runs(mask: NDArray[np.bool_], max_gap: int) -> list[tuple[int, int]]:
    """(start, stop) index pairs of True runs, bridging gaps of up to ``max_gap`` samples."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k - prev - 1 <= max_gap:
            prev = k
        else:
            out.append((start, prev))
            start = prev = k
    out.append((start, prev))
    return out


def conus_tip(sag: SagImage, track: CanalTrack, cfg: Config = DEFAULT) -> tuple[float, float] | None:
    """Lower end of the cord signal inside the thecal sac: returns (superior mm, confidence) or None.

    The cord is an intermediate-signal core inside bright CSF on T2 (dark core on T1). Along the
    canal centreline, a core that is less CSF-like than its flanks marks cord; the longest such run
    gives the conus at its lower end.
    """
    s = np.arange(track.s[0], track.s[-1] + 1e-9, sag.res)
    a_c = track.a_at(s)
    pol = track.polarity
    core_offs = (-2.0, -1.0, 0.0, 1.0, 2.0)
    flank_offs = (-7.0, -6.0, -5.0, 5.0, 6.0, 7.0)
    core = np.nanmean(np.stack([sag.sample(a_c + o, s) for o in core_offs]), axis=0)
    flank = np.nanmean(np.stack([sag.sample(a_c + o, s) for o in flank_offs]), axis=0)
    csf_contrast = (flank - core) * (1.0 if pol > 0 else -1.0)
    present = np.nan_to_num(csf_contrast, nan=0.0) > cfg.conus_min_contrast
    runs = _runs(present, max_gap=0)
    min_len = round(10.0 / sag.res)
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_len]
    if not runs:
        return None
    start, stop = max(runs, key=lambda r: r[1] - r[0])
    frac = float(np.mean(present[start : stop + 1]))
    return float(s[start]), float(min(1.0, frac))
