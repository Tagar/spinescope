"""Mid-sagittal plane: pick the sagittal slice where the thecal sac is clearest, refine with axials.

Sagittal evidence: the canal ridge is longest and widest in the mid-sagittal slice, so the canal
track score peaks there (lateral slices show only part of the sac, or foramina). The peak is
refined by a parabola through neighbouring slices. When axial slices pass through the canal, each
gives a CSF-blob centre whose median x replaces the sagittal estimate (if they agree within
``midline_axial_tol_mm``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from ..config import DEFAULT, Config
from ..geometry import Stack, sample_plane
from ..study import Study
from .sagittal import CanalTrack, csf_polarities, sagittal_image, track_canal

#: Upper bound on the number of sagittal slices scanned for the canal (evenly spaced when more).
MAX_SCAN_SLICES = 40


@dataclass
class MidlineChoice:
    slice_index: int
    polarity: int
    plane_coord: float  # position of the mid-sagittal plane along the stack normal (mm)
    track: CanalTrack | None
    scores: dict[int, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


def _scan_indices(n: int) -> list[int]:
    if n <= MAX_SCAN_SLICES:
        return list(range(n))
    return sorted({round(x) for x in np.linspace(0, n - 1, MAX_SCAN_SLICES)})


def select_midline(stack: Stack, pols: tuple[int, ...], cfg: Config = DEFAULT) -> MidlineChoice | None:
    """Mid-sagittal plane and the nearest analysed slice from the canal ridge score across slices.

    The score (canal ridge contrast per slice) peaks on the two slices either side of the midline,
    not on it: the cord sits inside the sac there and lowers the core contrast. The plane is therefore
    the score-weighted centroid of the peak neighbourhood (within ``midline_search_mm`` of the best
    slice), which is symmetric about the canal for a symmetric sac.
    """
    scores: dict[tuple[int, int], float] = {}
    tracks: dict[tuple[int, int], CanalTrack] = {}
    for k in _scan_indices(stack.n):
        sag = sagittal_image(stack, k, cfg.sag_resolution_mm)
        for pol in pols:
            tr = track_canal(sag, pol, cfg)
            scores[(k, pol)] = tr.score if tr is not None else 0.0
            if tr is not None:
                tracks[(k, pol)] = tr
    if not scores or max(scores.values()) <= 0:
        return None
    k_best, pol_best = max(scores, key=lambda kp: scores[kp])
    ks = sorted(k for (k, p) in scores if p == pol_best)
    per_slice = {k: scores[(k, pol_best)] for k in ks}
    ev = [f"canal ridge score {per_slice[k_best]:.1f} at slice {k_best} (polarity {pol_best:+d})"]
    if stack.n == 1:
        plane = float(stack.positions[0])
        ev.append("single sagittal slice: midline assumed at the slice position")
    else:
        pos = stack.positions
        near = [k for k in ks if abs(pos[k] - pos[k_best]) <= cfg.midline_search_mm]
        smax = per_slice[k_best]
        w = np.array([max(per_slice[k] - 0.6 * smax, 0.0) for k in near])
        if w.sum() > 0:
            plane = float(np.sum(w * np.array([pos[k] for k in near])) / w.sum())
        else:
            plane = float(pos[k_best])
        ev.append(f"midline plane from centroid of {len(near)} slices near the canal ridge peak")
    k_near = min(ks, key=lambda k: abs(stack.positions[k] - plane))
    return MidlineChoice(
        slice_index=k_near,
        polarity=pol_best,
        plane_coord=plane,
        track=tracks.get((k_near, pol_best)),
        scores=per_slice,
        evidence=ev,
    )


def refine_midline_axial(
    study: Study,
    sag_stack: Stack,
    choice: MidlineChoice,
    cfg: Config = DEFAULT,
) -> tuple[float | None, list[str], list[str]]:
    """Midline x from the CSF blob centre on axial slices through the canal.

    Returns ``(x_mm or None, evidence, warnings)``. Only applies to sagittal stacks whose normal is
    close to the x axis (the standard acquisition) and needs at least three axial slices that show
    a CSF blob inside the canal window.
    """
    ev: list[str] = []
    warns: list[str] = []
    if choice.track is None or abs(float(sag_stack.normal[0])) < 0.95:
        return None, ev, warns
    axials = [
        s
        for s in study.stacks.values()
        if s.plane == "ax" and s.n >= 3 and not s.meta.get("role", {}).get("localizer")
    ]
    if not axials:
        return None, ev, warns
    sag = sagittal_image(sag_stack, choice.slice_index, cfg.sag_resolution_mm)
    x_mid = choice.plane_coord / float(sag_stack.normal[0])
    xs: list[float] = []
    for ax in axials:
        for k in range(ax.n):
            z = float(ax.ipp[k][2])
            s_coord = (z - sag.origin[2]) / sag.sup[2]
            if not (choice.track.s[0] <= s_coord <= choice.track.s[-1]):
                continue
            a_c = float(choice.track.a_at(s_coord))
            y = float(sag.to_lps(a_c, s_coord)[1])
            x_hit = _csf_blob_x(ax, k, x_mid, y, choice.polarity, cfg)
            if x_hit is not None:
                xs.append(x_hit)
    if len(xs) < 3:
        return None, ev, warns
    med = float(np.median(xs))
    sd = float(np.std(xs))
    if abs(med - x_mid) > cfg.midline_axial_tol_mm:
        warns.append(
            f"axial CSF midline x={med:.1f} mm disagrees with sagittal estimate {x_mid:.1f} mm; sagittal kept"
        )
        return None, ev, warns
    ev.append(f"midline x from {len(xs)} axial CSF blobs (median {med:.1f} mm, sd {sd:.1f} mm)")
    return med, ev, warns


def _csf_blob_x(ax: Stack, k: int, x_mid: float, y: float, polarity: int, cfg: Config) -> float | None:
    """x centre of the CSF blob in axial slice ``k`` near (x_mid, y), or None."""
    w = cfg.axial_canal_window_mm
    px = 0.5
    origin = np.array([x_mid, y, float(ax.ipp[k][2])])
    img = sample_plane(ax, origin, (1, 0, 0), (0, 1, 0), (w, w), px)
    if not np.isfinite(img).any():
        return None
    v = np.where(np.isfinite(img), img, np.nanmedian(img))
    lo, hi = np.percentile(v, [5, 95])
    if hi - lo < 1e-9:
        return None
    norm = np.clip((v - lo) / (hi - lo), 0, 1)
    csf = norm if polarity > 0 else 1 - norm
    p50, p95 = np.percentile(csf, [50, 95])
    thr = p50 + 0.6 * (p95 - p50)
    lab, n = ndi.label(csf > thr)
    if n == 0:
        return None
    centre = np.array(csf.shape) / 2.0
    best: tuple[float, float] | None = None
    for i in range(1, n + 1):
        rr, cc = np.nonzero(lab == i)
        area = len(rr) * px * px
        if not (4.0 <= area <= 250.0):
            continue
        d = float(np.hypot(rr.mean() - centre[0], cc.mean() - centre[1])) * px
        if d > 8.0:
            continue
        if best is None or d < best[0]:
            best = (d, float(cc.mean()))
    if best is None:
        return None
    col = best[1]
    return float(x_mid + (col - (csf.shape[1] - 1) / 2.0) * px)


def pols_for(stack: Stack) -> tuple[int, ...]:
    """Candidate CSF polarities for a stack from its classified contrast."""
    return csf_polarities(str(stack.meta.get("role", {}).get("contrast", "unknown")))
