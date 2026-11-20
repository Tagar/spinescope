"""Realistic lumbar phantom: morphometry, relaxometry and spin-echo physics for ``PhantomSpec(realism="realistic")``.

The phantom is still analytic (ground truth in LPS mm), but its anatomy and its image contrast are modelled
from published numbers rather than drawn by eye:

* **Morphometry** (:data:`BODY_HEIGHT_MM`, :data:`BODY_HALF_WIDTH_MM`, :data:`BODY_HALF_DEPTH_MM`,
  :data:`DISC_HEIGHT_MM`, :data:`SEGMENT_LORDOSIS_DEG`, :data:`SACRAL_SLOPE_DEG`): lumbar body dimensions rise
  from L1 to L5 and disc heights from L1-L2 to L5-S1 (Panjabi 1992; Bogduk 2005; Frobin et al., late 1990s), lordosis and
  sacral slope from the normal-alignment study of Roussouly 2005. The values were typed in from those sources
  as approximate central values; they are inputs to a model, not measurements of any patient.
* **Relaxometry** (:data:`RELAXATION`): proton density and T1/T2 at 1.5 T and 3 T for each tissue, from
  Stanisz 2005 (3 T tissue relaxation), Bottomley 1987 (1.5 T review) and Pfirrmann 2001 (disc T2 grades),
  rounded and treated as ranges. Several entries (intervertebral disc, marrow, dura, ligament) are
  approximate literature ranges, not one measured data set.
* **Physics**: spin-echo ``S = rho (1 - exp(-TR/T1)) exp(-TE/T2)`` and inversion recovery (STIR)
  ``S = rho |1 - 2 exp(-TI/T1) + exp(-TR/T1)| exp(-TE/T2)``, evaluated per tissue. Partial volume comes from
  averaging sub-samples across the slice thickness and in plane (done in ``SpinePhantom``).

Anatomy follows the description in the phantom docs: kidney-bean vertebral bodies with concave endplates and
a cortical rim, marrow texture (smooth random fields), discs with annulus and nucleus, posterior elements
(pedicles, laminae, facets, spinous processes, ligamentum flavum), thecal sac, cord and cauda equina, epidural
fat, a sacrum of five fused segments with an S1-S2 remnant, iliac wings, psoas and paraspinal muscles with fat
streaks, subcutaneous fat and abdominal contents.

Nothing here is fitted to an image. Parameters are fixed from the sources above; see ``docs/labeling/lab21_phantom.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .phantom import (
    AIR,
    ANNULUS,
    BASIVERTEBRAL,
    BLOOM,
    BOWEL,
    CORD,
    CORTEX,
    CSF,
    DURA,
    FACET,
    FAT,
    HARDWARE,
    KIDNEY,
    LESION_ATYPICAL,
    LESION_FATTY,
    LESION_HAEM,
    LESION_ISLAND,
    LIGAMENT,
    LIVER,
    MARROW,
    MARROW_DIM,
    MODIC1,
    MUSCLE,
    NERVE,
    NUCLEUS,
    OEDEMA,
    PSOAS,
    REMNANT,
    SCHMORL,
    SOFT,
    VESSEL,
    _segment_distance,
)

if TYPE_CHECKING:  # pragma: no cover
    from .phantom import SpinePhantom

# ----------------------------------------------------------------------------------------- morphometry
# Lumbar and lower thoracic body dimensions (mm). Sources: Panjabi MM et al., Spine 1992 (quantitative 3-D
# lumbar vertebral dimensions: body height and width grow from L1 to L5); Bogduk N, Clinical Anatomy of the
# Lumbar Spine and Sacrum, 3rd ed. 2005 (body and canal dimensions). Half values are used for the phantom.
BODY_HEIGHT_MM: dict[str, float] = {
    "T9": 22.0, "T10": 22.5, "T11": 23.0, "T12": 24.0, "L1": 25.0, "L2": 26.0,
    "L3": 27.0, "L4": 28.0, "L5": 28.0, "L6": 28.0, "S1": 30.0,
}  # fmt: skip
BODY_HALF_WIDTH_MM: dict[str, float] = {
    "T9": 17.0, "T10": 18.0, "T11": 19.0, "T12": 20.0, "L1": 21.0, "L2": 22.0,
    "L3": 23.5, "L4": 25.0, "L5": 26.5, "L6": 27.0, "S1": 22.0,
}  # fmt: skip
BODY_HALF_DEPTH_MM: dict[str, float] = {
    "T9": 14.0, "T10": 14.5, "T11": 15.0, "T12": 15.5, "L1": 16.0, "L2": 16.5,
    "L3": 17.0, "L4": 17.5, "L5": 18.0, "L6": 18.5, "S1": 17.0,
}  # fmt: skip
#: Sacral segments below S1 (mm): height, half width, half depth. S1 is in the tables above.
SACRAL_SEGMENTS_MM: dict[str, tuple[float, float, float]] = {
    "S2": (24.0, 18.0, 13.0),
    "S3": (19.0, 14.0, 10.0),
    "S4": (15.0, 11.0, 8.0),
    "S5": (12.0, 9.0, 6.5),
}
#: Intervertebral disc heights, central (mm). Frobin et al. (late 1990s) and Bogduk 2005 give anterior disc heights of
#: roughly 7 mm at T12-L1 rising to 11-13 mm at L4-L5 and L5-S1; a central value is about 1 mm less than anterior.
DISC_HEIGHT_MM: dict[str, float] = {
    "T9-T10": 4.5, "T10-T11": 5.0, "T11-T12": 5.5, "T12-L1": 7.0, "L1-L2": 8.0, "L2-L3": 9.0,
    "L3-L4": 10.0, "L4-L5": 11.0, "L5-S1": 11.0, "L5-L6": 9.0, "L6-S1": 11.0,
}  # fmt: skip
#: Relative turn between adjacent disc planes, superior to inferior (T11-T12 .. L4-L5), degrees. Approximate
#: segmental values in the range reported for normal lordosis (Roussouly 2005; Bogduk 2005); the sum gives a
#: lumbar lordosis of about 46 degrees from L1 to S1 (computed from the geometry), inside the normal range of about 40-60 degrees.
SEGMENT_LORDOSIS_DEG: tuple[float, ...] = (5.0, 6.0, 8.0, 9.0, 11.0, 12.0)
#: Sacral slope (degrees), normal range about 35-45 degrees (Roussouly 2005).
SACRAL_SLOPE_DEG = 40.0
#: A lumbar spinous process lies at the level of the disc below its own body (standard anatomy, e.g. Bogduk 2005):
#: its centre is this fraction of the body half height below the body centre (1.0 = at the inferior endplate).
SPINOUS_CAUDAL_SHIFT = 1.0
#: Each sacral segment below S1 is tilted this many degrees further towards vertical (sacral curvature).
SACRAL_TILT_STEP_DEG = 12.0

# Body shape (kidney bean). Posterior wall concave in the axial plane by this fraction of the half depth at the
# midline; endplates dished by DISH_MM at the centre (concave endplates, Bogduk 2005).
KIDNEY_CONCAVITY = 0.06
DISH_MM = 1.2

# Body outline (axial, mm relative to the L3 body centre). Skin: lateral half width, anterior and posterior
# half extents; the subcutaneous fat ring is the gap to the inner (muscle) boundary.
SKIN_HALF_LAT = 150.0
SKIN_CY = -10.0
SKIN_HALF_AP = 105.0
SOFT_HALF_LAT = 132.0
SOFT_CY = -4.5
SOFT_HALF_AP = 85.5

# ----------------------------------------------------------------------------------------- relaxometry


@dataclass(frozen=True)
class Relaxation:
    """Relative proton density (CSF = 1) and T1, T2 in ms as ``(1.5 T, 3 T)``."""

    rho: float
    t1: tuple[float, float]
    t2: tuple[float, float]


#: Tissue relaxation. Values are rounded literature ranges (see the module docstring); fat, marrow, disc,
#: dura and ligament are the least certain. Cortical bone is almost proton-free with T2 near zero: dark on
#: every spin-echo image, as in real cortex. Vessels are a flow void (dark on spin echo).
RELAXATION: dict[int, Relaxation] = {
    AIR: Relaxation(0.0, (1000.0, 1000.0), (1.0, 1.0)),
    SOFT: Relaxation(0.75, (650.0, 900.0), (60.0, 50.0)),  # generic soft tissue (skin, viscera)
    FAT: Relaxation(0.9, (260.0, 370.0), (110.0, 100.0)),  # subcutaneous and epidural fat, fat streaks
    MUSCLE: Relaxation(0.75, (870.0, 1410.0), (47.0, 30.0)),  # paraspinal muscle (Stanisz 2005 at 3 T)
    PSOAS: Relaxation(0.75, (870.0, 1410.0), (47.0, 30.0)),  # psoas: same muscle relaxation as paraspinal
    MARROW: Relaxation(0.8, (380.0, 450.0), (80.0, 70.0)),  # fatty lumbar marrow
    CORTEX: Relaxation(0.15, (300.0, 400.0), (0.5, 0.4)),  # cortical bone, T2 below 1 ms
    ANNULUS: Relaxation(0.7, (700.0, 1000.0), (60.0, 40.0)),  # annulus fibrosus
    NUCLEUS: Relaxation(0.85, (1000.0, 1300.0), (110.0, 90.0)),  # healthy nucleus pulposus
    CSF: Relaxation(1.0, (2500.0, 3500.0), (2200.0, 2000.0)),  # thecal sac fluid
    CORD: Relaxation(0.7, (800.0, 1000.0), (80.0, 70.0)),  # spinal cord and nerve roots
    VESSEL: Relaxation(0.05, (1200.0, 1500.0), (100.0, 80.0)),  # flowing blood: a signal void on spin echo
    MODIC1: Relaxation(0.85, (700.0, 900.0), (120.0, 100.0)),  # Modic 1 marrow oedema-like change
    OEDEMA: Relaxation(0.9, (800.0, 1000.0), (150.0, 120.0)),  # fracture oedema (bright on STIR and T2)
    HARDWARE: Relaxation(0.0, (1.0, 1.0), (1.0, 1.0)),  # metal: no signal
    SCHMORL: Relaxation(0.85, (1000.0, 1300.0), (110.0, 90.0)),  # disc material in the marrow (nucleus-like)
    BLOOM: Relaxation(0.05, (300.0, 400.0), (1.0, 1.0)),  # susceptibility halo: no signal
    DURA: Relaxation(0.5, (800.0, 1000.0), (8.0, 6.0)),  # dura: a dark rim on T2
    LIGAMENT: Relaxation(0.6, (500.0, 600.0), (25.0, 20.0)),  # ligamentum flavum: dark
    FACET: Relaxation(0.9, (900.0, 1200.0), (80.0, 60.0)),  # facet joint fluid and cartilage
    BOWEL: Relaxation(0.8, (600.0, 800.0), (150.0, 120.0)),  # fluid-filled bowel
    LIVER: Relaxation(0.8, (550.0, 800.0), (40.0, 30.0)),  # liver parenchyma
    KIDNEY: Relaxation(0.8, (700.0, 1000.0), (80.0, 60.0)),  # renal parenchyma
    NERVE: Relaxation(0.7, (800.0, 1000.0), (80.0, 70.0)),  # cauda equina roots
    REMNANT: Relaxation(0.6, (800.0, 1000.0), (60.0, 50.0)),  # rudimentary S1-S2 disc
    # Marrow lesion tissues of the cartoon option (PhantomSpec.marrow_lesions); the realistic path never draws them.
    # Approximate entries, not literature values: they keep the table complete.
    LESION_HAEM: Relaxation(0.85, (400.0, 450.0), (120.0, 100.0)),  # fat-rich, vascular focus
    LESION_FATTY: Relaxation(0.9, (260.0, 370.0), (110.0, 100.0)),  # fatty focus (as subcutaneous fat)
    LESION_ISLAND: Relaxation(0.15, (300.0, 400.0), (0.5, 0.4)),  # dense bone (as cortex)
    LESION_ATYPICAL: Relaxation(0.85, (400.0, 500.0), (150.0, 120.0)),  # marrow replacement with water
    BASIVERTEBRAL: Relaxation(0.5, (500.0, 600.0), (120.0, 100.0)),  # venous channel
    MARROW_DIM: Relaxation(0.8, (300.0, 350.0), (80.0, 70.0)),  # diffusely T1-dark marrow
}


def spin_echo_signal(rho: float, t1: float, t2: float, tr: float, te: float) -> float:
    """Spin-echo signal ``rho (1 - exp(-TR/T1)) exp(-TE/T2)`` (times in ms)."""
    return rho * (1.0 - np.exp(-tr / t1)) * np.exp(-te / max(t2, 1e-3))


def inversion_recovery_signal(rho: float, t1: float, t2: float, tr: float, te: float, ti: float) -> float:
    """Inversion-recovery (STIR) signal ``rho |1 - 2 exp(-TI/T1) + exp(-TR/T1)| exp(-TE/T2)`` (times in ms)."""
    return rho * abs(1.0 - 2.0 * np.exp(-ti / t1) + np.exp(-tr / t1)) * np.exp(-te / max(t2, 1e-3))


def signal_table(acq: dict[str, Any], field_strength: float) -> dict[int, float]:
    """Expected signal of every tissue for one acquisition (``acq`` as in ``phantom.ACQ``) at 1.5 or 3 T."""
    k = 0 if field_strength == 1.5 else 1
    out: dict[int, float] = {}
    for tissue, rel in RELAXATION.items():
        t1, t2 = rel.t1[k], rel.t2[k]
        if acq["seq"] == "IR":
            s = inversion_recovery_signal(rel.rho, t1, t2, acq["TR"], acq["TE"], acq["TI"])
        else:
            s = spin_echo_signal(rel.rho, t1, t2, acq["TR"], acq["TE"])
        out[tissue] = float(s)
    return out


# ----------------------------------------------------------------------------------------- texture


def smooth_field(pts: NDArray[np.float64], seed: int, n_waves: int = 10) -> NDArray[np.float64]:
    """Smooth, zero-mean random field in about [-1, 1] at LPS points (sum of plane waves, 8-25 mm wavelengths)."""
    rng = np.random.default_rng(seed + 7919)
    dirs = rng.normal(size=(3, n_waves))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    wavelength = rng.uniform(8.0, 25.0, size=n_waves)
    phase = rng.uniform(0.0, 2 * np.pi, size=n_waves)
    k = (2 * np.pi / wavelength) * dirs  # (3, K)
    return np.cos(pts @ k + phase).sum(axis=1) / np.sqrt(n_waves / 2.0)


TEXTURED = (MARROW, MUSCLE, PSOAS)  # heterogeneous tissues: +-12 % relative signal from the random field
TEXTURE_AMPLITUDE = 0.12


def apply_texture(
    img: NDArray[np.float64], labels: NDArray[np.int8], pts: NDArray[np.float64], seed: int
) -> NDArray[np.float64]:
    """Multiply the signal of heterogeneous tissues by a smooth random modulation (marrow and muscle texture)."""
    m = np.isin(labels, TEXTURED)
    if m.any():
        img = img.copy()
        img[m] *= 1.0 + TEXTURE_AMPLITUDE * smooth_field(pts[m], seed)
    return img


def gibbs_ringing(img: NDArray[np.float64], keep: float = 0.7) -> NDArray[np.float64]:
    """Truncate the 2-D k-space to the central ``keep`` fraction per axis: Gibbs ringing at sharp edges."""
    f = np.fft.fft2(img)
    fy = np.abs(np.fft.fftfreq(img.shape[0]))[:, None]
    fx = np.abs(np.fft.fftfreq(img.shape[1]))[None, :]
    mask = (fy <= 0.5 * keep) & (fx <= 0.5 * keep)
    return np.clip(np.fft.ifft2(f * mask).real, 0.0, None)


# ----------------------------------------------------------------------------------------- geometry helpers


def sacral_segments(
    bottom: NDArray[np.float64],
    s1_tilt: float,
    normal_of: Callable[[float], NDArray[np.float64]],
    ap_of: Callable[[float], NDArray[np.float64]],
) -> list[dict[str, Any]]:
    """Frames of S2..S5 chained below the S1 bottom plane, each tilted further towards vertical.

    The sacrum runs inferiorly, i.e. against the superior normal of each segment.
    """
    segs: list[dict[str, Any]] = []
    p = np.asarray(bottom, dtype=np.float64)
    for i, name in enumerate(("S2", "S3", "S4", "S5"), start=1):
        h, hw, hd = SACRAL_SEGMENTS_MM[name]
        tilt = s1_tilt - SACRAL_TILT_STEP_DEG * i
        n = normal_of(tilt)
        center = p - n * h / 2
        segs.append(
            {
                "name": name,
                "center": center,
                "normal": n,
                "ap": ap_of(tilt),
                "lat": np.array([1.0, 0.0, 0.0]),
                "tilt_deg": tilt,
                "half_height": h / 2,
                "hw": hw,
                "hd": hd,
            }
        )
        p = p - n * h
    return segs


def _body_masks(
    lat: NDArray[np.float64],
    ap: NDArray[np.float64],
    up: NDArray[np.float64],
    hw: float,
    hd: float,
    hh: float,
    fracture: float = 0.0,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(inside, cortical shell) of a kidney-bean body with dished endplates, in the body frame."""
    hd_eff = np.where(ap > 0, hd * (1.0 - KIDNEY_CONCAVITY * (1.0 - np.clip((lat / hw) ** 2, 0.0, 1.0))), hd)
    rr = (np.abs(lat) / hw) ** 2.4 + (np.abs(ap) / hd_eff) ** 2.4
    hh_loc = hh - DISH_MM * (1.0 - np.clip(rr, 0.0, 1.0))
    if fracture:  # anterior wedge: the anterior height is lost, the posterior wall is kept
        hh_loc = hh_loc * (1.0 - fracture * np.clip(-ap / hd, 0.0, 1.0))
    inside = (rr <= 1.0) & (np.abs(up) <= hh_loc)
    shell = inside & ((rr > 0.84) | (np.abs(up) > hh_loc - 1.0))
    return inside, shell


Near = Callable[[NDArray[np.float64], float], NDArray[np.intp]]


def _tree_near(tree: cKDTree, c: NDArray[np.float64], radius: float) -> NDArray[np.intp]:
    """Indices of the points within ``radius`` mm of ``c``: structures are only evaluated where they can be."""
    return np.asarray(tree.query_ball_point(np.asarray(c, dtype=np.float64), radius), dtype=np.intp)


def _frame(
    pts: NDArray[np.float64], v: dict[str, Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """(lateral, AP, superior) coordinates of ``pts`` in the frame of a body or disc."""
    q = pts - v["center"]
    return q @ v["lat"], q @ v["ap"], q @ v["normal"]


def _paint_posterior(
    lab: NDArray[np.int8], pts: NDArray[np.float64], v: dict[str, Any], canal_off: float, r: float, near: Near
) -> None:
    """Laminae, spinous process, pedicles and facets of one body frame (canal radius ``r``, canal ``canal_off`` mm)."""
    idx = near(v["center"], canal_off + r + 40.0)
    if not idx.size:
        return
    p = pts[idx]
    lat, ap, up = _frame(p, v)
    hh, hd = v["half_height"], v["hd"]
    lam = (np.abs(lat) <= 16.0) & (ap >= canal_off + r) & (ap <= canal_off + r + 4.5) & (np.abs(up) <= hh)
    lab[idx[lam]] = CORTEX
    up_s = up + SPINOUS_CAUDAL_SHIFT * hh  # the spinous process sits at the level of the disc below
    spin = (
        (np.abs(lat) <= 4.5)
        & (ap > canal_off + r + 4.5)
        & (ap <= canal_off + r + 26.0)
        & (np.abs(up_s) <= 9.0)
    )
    lab[idx[spin]] = MARROW
    lab[idx[spin & ((np.abs(lat) > 3.0) | (np.abs(up_s) > 7.0))]] = CORTEX
    rho2 = (np.abs(lat) - 11.0) ** 2 + up**2
    ped = (rho2 <= 25.0) & (ap >= hd - 3.0) & (ap <= canal_off + r + 1.0)
    lab[idx[ped]] = MARROW
    lab[idx[ped & (rho2 > 12.25)]] = CORTEX
    fac = ((np.abs(lat) - 19.0) / 5.0) ** 2 + ((ap - (canal_off + 1.0)) / 6.0) ** 2 + (up / 9.0) ** 2 <= 1.0
    lab[idx[fac]] = CORTEX


def _paint_discs(lab: NDArray[np.int8], pts: NDArray[np.float64], ph: SpinePhantom, near: Near) -> None:
    """Annulus, nucleus, herniations, ligamentum flavum and facet joint space for every disc."""
    sp = ph.spec
    r, ca = sp.canal_radius, sp.canal_offset
    for d, v in ph.discs.items():
        idx = near(v["center"], 48.0)
        if not idx.size:
            continue
        p = pts[idx]
        lat, ap, up = _frame(p, v)
        dh, hw, hd = v["height_mm"], v["hw"], v["hd"]
        r2 = (lat / hw) ** 2 + (ap / hd) ** 2
        half = dh / 2 * (1.0 - 0.25 * r2)
        disc = (r2 <= 1.0) & (np.abs(up) <= half)
        if sp.transitional == "sacralised_l5" and d == "L5-S1":
            keep = lat > -4.0
            lab[idx[disc & ~keep]] = MARROW
            disc &= keep
        h = sp.herniation.get(d)
        if h is not None:
            hside, depth = h
            cx = {"central": 0.0, "L": 8.0, "R": -8.0}[hside]
            hern = ((lat - cx) / 7.0) ** 2 + ((ap - hd) / depth) ** 2 <= 1.0
            disc |= hern & (ap > 0) & (np.abs(up) <= half.clip(min=dh * 0.3))
        lab[idx[disc]] = ANNULUS
        lab[idx[disc & (r2 <= 0.35) & (np.abs(up) <= half * 0.7)]] = NUCLEUS
        # ligamentum flavum in the interlaminar space, and facet joint space at disc level
        lf = (
            (np.abs(lat) <= 14.0) & (ap >= ca + r - 0.5) & (ap <= ca + r + 3.0) & (np.abs(up) <= dh / 2 + 3.0)
        )
        lab[idx[lf]] = LIGAMENT
        fj = ((np.abs(lat) - sp.facet_lateral_mm) / 6.0) ** 2 + ((ap - (ca + 1.0)) / 7.0) ** 2 + (
            up / (dh / 2 + 1.5)
        ) ** 2 <= 1.0
        lab[idx[fj]] = FACET
        eff = sp.facet_effusion.get(d)
        if eff is not None:  # CSF-like fluid lens inside the joint: `width` mm along AP, 3 mm across
            side, width = eff
            sgn = 1.0 if side == "L" else -1.0
            fl = ((lat - sgn * sp.facet_lateral_mm) / 1.5) ** 2 + ((ap - (ca + 1.0)) / (width / 2.0)) ** 2 + (
                up / (dh / 2 + 1.5)
            ) ** 2 <= 1.0
            lab[idx[fl]] = CSF


def labels(ph: SpinePhantom, pts: NDArray[np.float64]) -> NDArray[np.int8]:
    """Tissue label of every LPS point for the realistic phantom."""
    sp = ph.spec
    lab = np.full(len(pts), AIR, dtype=np.int8)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ref = ph.bodies["L3"]["center"]
    yc = float(ref[1])
    ca, r = sp.canal_offset, sp.canal_radius
    near = partial(_tree_near, cKDTree(pts))
    body_names, sacrum = ph._body_names, ph._sacrum
    s1 = ph.bodies[sacrum]
    zt12 = float(ph.bodies["T12"]["center"][2])
    zl1 = float(ph.bodies["L1"]["center"][2])
    zl4 = float(ph.bodies["L4"]["center"][2])
    zl5_bottom = float(ph.bodies["L5"]["center"][2]) - float(ph.bodies["L5"]["half_height"])
    zt_top = float(ph.bodies[body_names[0]]["center"][2]) + float(ph.bodies[body_names[0]]["half_height"])

    # body outline: subcutaneous fat ring, soft tissue inside
    lab[(x / SKIN_HALF_LAT) ** 2 + ((y - (yc + SKIN_CY)) / SKIN_HALF_AP) ** 2 <= 1.0] = FAT
    lab[(x / SOFT_HALF_LAT) ** 2 + ((y - (yc + SOFT_CY)) / SOFT_HALF_AP) ** 2 <= 1.0] = SOFT

    # abdominal contents (anterior and lateral to the spine), each evaluated near its centre only
    def ellipsoid(c: NDArray[np.float64], radii: tuple[float, float, float], value: int) -> NDArray[np.intp]:
        idx = near(c, max(radii) + 1.0)
        if idx.size:
            q = (pts[idx] - c) / np.asarray(radii)
            idx = idx[np.einsum("ij,ij->i", q, q) <= 1.0]
            lab[idx] = value
        return idx

    ellipsoid(np.array([-70.0, yc - 35.0, zt12]), (48.0, 36.0, 42.0), LIVER)
    for s in (-1.0, 1.0):
        ellipsoid(np.array([s * 52.0, yc + 12.0, zl1]), (24.0, 22.0, 30.0), KIDNEY)
    bowel = [(-42.0, -70.0, zl4 - 10.0), (0.0, -80.0, zl4 + 10.0), (40.0, -70.0, zl4 - 10.0),
             (-30.0, -60.0, zl1 + 10.0), (30.0, -62.0, zl1 + 15.0)]  # fmt: skip
    for bx, by, bz in bowel:
        c = np.array([bx, yc + by, bz])
        ellipsoid(c, (18.0, 18.0, 22.0), BOWEL)
        ellipsoid(c + np.array([4.0, -4.0, 4.0]), (6.0, 6.0, 6.0), AIR)  # gas

    # psoas (lateral to the bodies, T12 to L5) and paraspinal muscles with fat streaks
    hw3 = float(ph.bodies["L3"]["hw"])
    z_psoas = np.flatnonzero((z >= zl5_bottom) & (z <= zt12 + 10.0))
    for s in (-1.0, 1.0):
        q = pts[z_psoas]
        m = ((q[:, 0] - s * (hw3 + 6.0)) / 11.0) ** 2 + ((q[:, 1] - yc) / 13.0) ** 2 <= 1.0
        lab[z_psoas[m]] = PSOAS
    z_para = np.flatnonzero((z >= zl5_bottom - 10.0) & (z <= zt_top))
    for s in (-1.0, 1.0):
        q = pts[z_para]
        m = ((q[:, 0] - s * 18.0) / 14.0) ** 2 + ((q[:, 1] - (yc + ca + r + 16.0)) / 17.0) ** 2 <= 1.0
        lab[z_para[m]] = MUSCLE
    muscle_idx = np.flatnonzero(lab == MUSCLE)
    streak_idx = muscle_idx[smooth_field(pts[muscle_idx], sp.seed, 6) > 0.72]
    lab[streak_idx] = FAT  # fatty infiltration streaks inside the paraspinal muscles

    # great vessels: aorta and IVC anterior to the bodies (flow voids), above the L4 level
    above_l4 = np.flatnonzero(z > zl4)
    q = pts[above_l4]
    aorta = ((q[:, 0] - 14.0) / 9.0) ** 2 + ((q[:, 1] - (yc - 25.0)) / 9.0) ** 2 <= 1.0
    ivc = ((q[:, 0] + 14.0) / 10.0) ** 2 + ((q[:, 1] - (yc - 25.0)) / 10.0) ** 2 <= 1.0
    lab[above_l4[aorta | ivc]] = VESSEL

    # iliac wings (axial slices at the sacral level): cortical shell and marrow core
    for s in (-1.0, 1.0):
        c = np.asarray(s1["center"], dtype=np.float64) + np.array([s * 78.0, 0.0, 20.0])
        idx = near(c, 45.0)
        q = (pts[idx] - c) / np.array([22.0, 38.0, 28.0])
        rr = np.einsum("ij,ij->i", q, q)
        lab[idx[rr <= 1.0]] = CORTEX
        lab[idx[rr <= 0.7]] = MARROW

    # vertebral bodies, sacrum and sacral segments: bone with cortical rim, marrow inside
    frames: list[tuple[str, dict[str, Any]]] = [(b, ph.bodies[b]) for b in body_names]
    frames += [(sacrum, s1), *((seg["name"], seg) for seg in ph._sacral_segments)]
    for name, v in frames:
        hw, hd, hh = v["hw"], v["hd"], v["half_height"]
        idx = near(v["center"], float(np.sqrt(hw * hw + hd * hd + hh * hh)) + 6.0)
        if not idx.size:
            continue
        lat, ap, up = _frame(pts[idx], v)
        fracture = sp.fracture.get(name, 0.0)
        inside, shell = _body_masks(lat, ap, up, hw, hd, hh, fracture=fracture)
        lab[idx[inside]] = MARROW
        lab[idx[shell]] = CORTEX
        if fracture:
            lab[idx[inside & ~shell]] = OEDEMA
    # S1-S2 remnant: a thin disc-like plate across the S1 bottom plane
    idx = near(np.asarray(s1["center"]), 30.0)
    lat, ap, up = _frame(pts[idx], s1)
    rem = (np.abs(up + s1["half_height"]) <= 1.5) & ((lat / 14.0) ** 2 + (ap / 10.0) ** 2 <= 1.0)
    lab[idx[rem]] = REMNANT
    # posterior elements of every level and of the sacrum
    for name in body_names:
        _paint_posterior(lab, pts, ph.bodies[name], ca, r, near)
    _paint_posterior(lab, pts, s1, ca, r, near)

    # discs: painted after the bodies (the disc space lies between bodies)
    _paint_discs(lab, pts, ph, near)

    # Modic 1 bands and Schmorl nodes (only inside marrow, as in the cartoon)
    for mdisc, which, side in sp.modic1:
        v = ph.discs[mdisc]
        idx = near(v["center"], 48.0)
        lat, ap, up = _frame(pts[idx], v)
        dh = v["height_mm"]
        lo, hi = dh / 2 + 1.0, dh / 2 + 9.0
        r2 = (lat / v["hw"]) ** 2 + (ap / v["hd"]) ** 2
        band = (r2 <= 0.8) & ((up > lo) & (up < hi) if which == "inf" else (up < -lo) & (up > -hi))
        if side == "L":
            band &= lat > 2.0
        elif side == "R":
            band &= lat < -2.0
        sel = idx[band]
        lab[sel[lab[sel] == MARROW]] = MODIC1
    for body, which in sp.schmorl:
        v = ph.bodies[body]
        sgn = -1.0 if which == "inf" else 1.0
        ep = v["center"] + v["normal"] * sgn * v["half_height"]
        idx = near(ep, 12.0)
        q = pts[idx] - ep
        node = ((q @ v["lat"]) / 4.5) ** 2 + ((q @ v["ap"]) / 4.5) ** 2 + (
            (q @ v["normal"]) / 3.0
        ) ** 2 <= 1.0
        sel = idx[node]
        lab[sel[lab[sel] == MARROW]] = SCHMORL

    # thoracic ribs (optional)
    if sp.ribs:
        for name in body_names:
            if not name.startswith("T"):
                continue
            v = ph.bodies[name]
            for s in (-1.0, 1.0):
                p0 = v["center"] + v["lat"] * s * v["hw"] * 0.8
                p1 = p0 + v["lat"] * s * 60.0 - v["ap"] * 40.0
                idx = near(p0, 70.0)
                rib = _segment_distance(pts[idx], p0, p1) <= 3.0
                sel = idx[rib]
                lab[sel[np.isin(lab[sel], (SOFT, FAT, MUSCLE, PSOAS))]] = CORTEX

    # thecal sac, dura, epidural fat, cord and cauda equina. The dural sac ends at the S1 bottom (S2 level);
    # the sacral canal below it is narrow. Only points near the canal can be in these structures.
    near_canal = np.flatnonzero((np.abs(x) <= r + 20.0) & (np.abs(y - (yc + ca)) <= 70.0))
    dist = np.full(len(pts), np.inf)
    dist[near_canal] = ph._canal_distance(pts[near_canal])[0]
    z_end = float(ph.thecal_end_z)
    r_eff = np.where(z < z_end, 3.5, r)
    epidural = (dist > r_eff) & (dist <= r_eff + 3.0)
    keep_epi = np.isin(
        lab, (AIR, SOFT, FAT, MUSCLE, PSOAS, VESSEL, KIDNEY, LIVER, BOWEL)
    )  # not bone or discs
    lab[epidural & keep_epi] = FAT
    lab[(dist > r_eff - 1.0) & (dist <= r_eff)] = DURA
    lab[dist <= r_eff - 1.0] = CSF
    conus = float(ph.conus_z)
    strands = np.flatnonzero((z <= conus) & (dist <= r_eff - 1.0))
    if strands.size:
        for dx, dy in ((2.5, 1.5), (-2.5, 1.5), (2.5, -1.5), (-2.5, -1.5)):
            d_strand = ph._canal_distance(pts[strands] - np.array([dx, dy, 0.0]))[0]
            lab[strands[d_strand <= 1.4]] = NERVE
    cord = (dist <= sp.cord_radius) & (z > conus)
    lab[cord] = CORD

    # hardware and blooming (signal voids), as in the cartoon
    core = np.zeros(len(pts), dtype=bool)
    halo = np.zeros(len(pts), dtype=bool)
    for item in sp.hardware:
        c_, h_ = ph._hardware_masks(item, pts)
        core |= c_
        halo |= h_
    lab[halo & ~core & (lab != AIR)] = BLOOM
    lab[core] = HARDWARE
    return lab
