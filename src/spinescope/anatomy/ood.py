"""Applicability: when the labeller must NOT label (wrong region, unusable field of view, unseen input).

``label_spine`` names levels by counting discs up from the lowest one. That is only meaningful for a lumbar
spine study in which the lowest discs are in view. This module decides, from cheap and explainable cues,
whether a study is one:

* metadata: ``BodyPartExamined`` (strong) and series descriptions (weak), technical attributes only;
* anatomy: disc count, disc heights and disc-to-disc spacings (region by size), the length of the spinal
  canal on the mid-sagittal slice of the network output, and the output entropy;
* field of view: z extent of the sagittal stack, sacrum in view, where the superior edge cuts;
* acquisition: contrasts and sequences the network was not trained on (STIR, PD, Dixon, post-contrast,
  unknown), low field strength, near-isotropic 3-D voxels.

The result is an :class:`Applicability` with a verdict, the region, whether levels are labelled, a confidence
factor, the reasons and the coverage. :func:`apply_applicability` writes it into the anchors (the ``applicability``
field, warnings, and the numbering status). A study that is not lumbar loses its disc labels; cervical discs of a
whole-spine view are dropped; thoracic discs above T10 with no named landmark are capped; confidences are scaled.

Only technical attributes and numbers are used or reported. No patient attribute, file path or free text
from the series is copied into the output (matched keywords only).
"""

from __future__ import annotations

import itertools
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study
from .anchors import Anchors
from .fusion import stack_z_range
from .ml.grid import network_input_volume, normalise, resample_slice
from .ml.labeller import BODY, CANAL, SegmentedStack
from .ml.postprocess import mid_sagittal
from .numbering import DEFAULT_STATUS_THRESHOLDS, STATUS_UNCONFIRMED, VERTEBRA_NAMES, numbering_status

log = logging.getLogger(__name__)

# Verdicts. Labelled verdicts keep level names; the others clear the disc labels.
LUMBAR = "lumbar"
THORACOLUMBAR = "thoracolumbar"
WHOLE_SPINE = "whole_spine"
UNCERTAIN = "uncertain"
NOT_LUMBAR = "not_lumbar"  # a spine of another region, or a body part that is not spine
NOT_SPINE = "not_spine"  # no spinal canal or no disc chain found
NO_SAGITTAL = "no_sagittal"
NOT_USABLE = "not_usable"
LABELLED_VERDICTS = (LUMBAR, THORACOLUMBAR, WHOLE_SPINE, UNCERTAIN)

#: Region names of spine studies (``BodyPartExamined`` or description).
SPINE_REGIONS = ("lumbar", "thoracolumbar", "whole_spine", "thoracic", "cervical", "spine")

_SEP = re.compile(r"[_\-/.,;:()\[\]*#]+")
_NON_SPINE: tuple[tuple[str, str], ...] = (
    ("brain", r"BRAIN|HEAD|KOPF|CRANIUM|CRANIAL"),
    ("knee", r"KNEE|KNIE"),
    ("shoulder", r"SHOULDER|SCHULTER"),
    ("pelvis", r"PELVIS|PELVIC|BECKEN"),
    ("hip", r"HIP|HUFT"),
    ("abdomen", r"ABDOMEN|ABDOMINAL|LIVER|PROSTATE"),
    ("chest", r"CHEST|THORAX"),
    ("ankle", r"ANKLE|FOOT"),
    ("wrist", r"WRIST|HAND|ELBOW"),
)
_SPINE: tuple[tuple[str, str], ...] = (
    ("whole_spine", r"WHOLE ?SPINE|SPINE WHOLE"),
    ("thoracolumbar", r"TL ?SPINE|TLS ?SPINE|THORACOLUMBAR"),
    ("lumbar", r"L ?SPINE|LS ?SPINE|LUMBAR\w*|LUMBOSACRAL|LWK|LENDEN\w*"),
    ("thoracic", r"T ?SPINE|THORACIC|BWK"),
    ("cervical", r"C ?SPINE|CS ?SPINE|CERVIC\w*|HWK|NECK"),
    ("spine", r"SPINE|WIRBELS\w*"),
)
_NON_SPINE_RE = [(name, re.compile(rf"\b(?:{rx})\b")) for name, rx in _NON_SPINE]
_SPINE_RE = [(name, re.compile(rf"\b(?:{rx})\b")) for name, rx in _SPINE]
#: Index of T10 in the bottom-up vertebra names (S1 = 0): discs whose upper vertebra is T10 or higher.
_T10_INDEX = VERTEBRA_NAMES.index("T10")


def _norm(text: str) -> str:
    return " ".join(_SEP.sub(" ", text.upper()).split())


def region_from_text(text: str) -> tuple[str | None, list[str]]:
    """Body region named in ``text``: a spine region (``lumbar``...) or a non-spine body part, and the matched words.

    Non-spine words win (a brain protocol that mentions a spine is not a spine study). Returns ``(None, [])``
    when nothing is named.
    """
    norm = _norm(text)
    if not norm:
        return None, []
    for name, rx in _NON_SPINE_RE:
        hits = [m.group(0) for m in rx.finditer(norm)]
        if hits:
            return name, hits
    found: dict[str, list[str]] = {}
    for name, rx in _SPINE_RE:
        hits = [m.group(0) for m in rx.finditer(norm)]
        if hits:
            found[name] = hits
    s = set(found)
    region: str | None
    if "whole_spine" in s or ("cervical" in s and s & {"lumbar", "thoracic", "thoracolumbar"}):
        region = "whole_spine"
    elif "thoracolumbar" in s or {"lumbar", "thoracic"} <= s:
        region = "thoracolumbar"
    elif "lumbar" in s:
        region = "lumbar"
    elif "thoracic" in s:
        region = "thoracic"
    elif "cervical" in s:
        region = "cervical"
    elif "spine" in s:
        region = "spine"
    else:
        return None, []
    tokens = [t for name in found for t in found[name]]
    return region, tokens


@dataclass(frozen=True)
class RegionCue:
    """Region named by the sagittal series' metadata. ``strong``: from BodyPartExamined, else free text."""

    region: str | None
    strong: bool
    tokens: tuple[str, ...] = ()


def metadata_region(sag: Stack) -> RegionCue:
    """Region from the sagittal series' technical attributes (BodyPartExamined first, then descriptions)."""
    d = sag.meta.get("dicom", {}) or {}
    bpe = str(d.get("BodyPartExamined") or "")
    if bpe.strip():
        region, tokens = region_from_text(bpe)
        if region is not None:
            return RegionCue(region, True, tuple(tokens))
    text = " ".join(str(d.get(k) or "") for k in ("SeriesDescription", "ProtocolName", "SequenceName"))
    region, tokens = region_from_text(text)
    if region is None:
        return RegionCue(None, False, ())
    return RegionCue(region, False, tuple(tokens))


def described_disc_labels(study: Study) -> set[str]:
    """Disc labels named in series descriptions of the study (e.g. a slab called 'T10-T11' is a landmark)."""
    from .numbering import level_names_in

    out: set[str] = set()
    for st in study.stacks.values():
        d = st.meta.get("dicom", {}) or {}
        text = " ".join(
            str(x or "") for x in (st.description, d.get("SeriesDescription"), d.get("ProtocolName"))
        )
        out.update(level_names_in(text))
    return out


def acquisition_cues(sag: Stack, cfg: Config = DEFAULT) -> list[str]:
    """Reasons the sagittal series is not an input the labeller was trained on (empty when it is)."""
    role = sag.meta.get("role", {}) or {}
    d = sag.meta.get("dicom", {}) or {}
    contrast = str(role.get("contrast", "unknown"))
    out: list[str] = []
    if contrast not in ("t1", "t2"):
        out.append(f"{contrast.upper()} sagittal series: the labeller was trained on T1 and T2 only")
    if role.get("dixon"):
        out.append("Dixon component image: not in the training data")
    if role.get("post_contrast"):
        out.append("post-contrast image: not in the training data")
    if contrast == "t1" and role.get("fat_suppressed") and role.get("post_contrast"):
        out.append("post-contrast fat-suppressed T1: not in the training data")
    field_t = _float(d.get("MagneticFieldStrength"))
    if field_t is not None and 0 < field_t < cfg.ood_low_field_t:
        out.append(f"low field strength ({field_t:g} T): the training data are 1.5 T and 3 T")
    thick = float(sag.thickness) or float(sag.spacing)
    inplane = max(float(sag.row_mm), float(sag.col_mm))
    if inplane > 0 and thick > 0:
        iso = max(thick, inplane) <= cfg.ood_isotropic_mm and 0.8 <= thick / inplane <= 1.25
        if iso:
            out.append("3-D isotropic voxels: the training volumes are 2-D or anisotropic 3-D (T2 SPACE)")
    return out


def _float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def disc_region(
    height: float | None,
    spacings: list[float],
    cfg: Config = DEFAULT,
    *,
    neighbour_cervical: bool = False,
) -> str:
    """Region of one disc from its height and its spacings to the discs above and below (centre to centre, mm).

    Spacing is the primary cue (cervical ~18-21 mm, thoracic ~26-30 mm, lumbar ~34-40 mm). A disc is cervical
    when all its spacings are short, it is thin (cervical discs are ~3-6 mm high) and it has two spacings, or it is
    an end disc with one short spacing and a cervical neighbour. Any long spacing makes it lumbar, so one spurious neighbour cannot turn a
    real lumbar disc into a cervical one. Height separates the rest (lumbar discs are at least ~7 mm high).
    """
    all_short = bool(spacings) and all(v < cfg.ood_cervical_spacing_max_mm for v in spacings)
    thin = height is None or height < cfg.ood_lumbar_disc_min_mm
    if all_short and thin and (len(spacings) >= 2 or neighbour_cervical):
        return "cervical"
    if any(v >= cfg.ood_lumbar_spacing_min_mm for v in spacings):
        return "lumbar"
    if height is None:
        return "unknown"
    tall = height >= cfg.ood_lumbar_disc_min_mm
    if not spacings:
        return "lumbar" if tall else "unknown"
    if tall or any(v < cfg.ood_cervical_spacing_max_mm for v in spacings):
        return "unknown"  # thin, but with a cervical-sized spacing: not a thoracic column
    return "thoracic"


def disc_regions(anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, str]:
    """Region of every disc, keyed by label (superior to inferior order)."""
    labs = anchors.disc_labels()
    centres = {lab: np.asarray(anchors.discs[lab].center, dtype=np.float64) for lab in labs}
    gaps = [float(np.linalg.norm(centres[a] - centres[b])) for a, b in itertools.pairwise(labs)]
    heights = {lab: anchors.discs[lab].height_mm for lab in labs}

    def spacings_of(i: int) -> list[float]:
        return [gaps[j] for j in (i - 1, i) if 0 <= j < len(gaps)]

    out: dict[str, str] = {lab: disc_region(heights[lab], spacings_of(i), cfg) for i, lab in enumerate(labs)}
    # End discs: cervical only with a cervical neighbour (second pass, so the neighbour's region is known).
    for i in (0, len(labs) - 1):
        if not labs or len(labs) < 2:
            break
        j = 1 if i == 0 else len(labs) - 2
        out[labs[i]] = disc_region(
            heights[labs[i]], spacings_of(i), cfg, neighbour_cervical=out[labs[j]] == "cervical"
        )
    return out


def _close_pairs(labs: list[str], gaps: list[float], cfg: Config) -> list[tuple[str, str, float]]:
    """Adjacent discs closer than cervical spacing. A lumbar column never has that spacing (a lumbar body is >20 mm)."""
    return [
        (a, b, gap)
        for (a, b), gap in zip(itertools.pairwise(labs), gaps, strict=True)
        if gap < cfg.ood_cervical_spacing_max_mm
    ]


def _disc_half(anchors: Anchors, lab: str, cfg: Config) -> float:
    h = anchors.discs[lab].height_mm
    return float(h) / 2 if h else cfg.disc_height_mm[0] / 2


def fov_coverage(sag: Stack, anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Field of view along z (mm, LPS) and which levels it holds: sacrum in view, superior cut, levels listed.

    Levels above the superior cut are outside the field of view (not missed). The sacrum is in view when at
    least ``sacrum_min_visible_mm`` of bone lies below the lowest disc.
    """
    zmin, zmax = stack_z_range(sag)
    out: dict[str, Any] = {
        "z_min_mm": round(float(zmin), 1),
        "z_max_mm": round(float(zmax), 1),
        "extent_mm": round(float(zmax - zmin), 1),
        "levels_in_fov": [],
        "sacrum_covered": None,
        "bone_below_lowest_disc_mm": None,
        "superior_cut": None,
        "superior_margin_mm": None,
        "top_level_in_view": None,
    }
    labs = anchors.disc_labels()  # superior -> inferior
    if not labs:
        return out
    low = anchors.discs[labs[-1]]
    vis = float(low.center[2]) - _disc_half(anchors, labs[-1], cfg) - float(zmin)
    out["bone_below_lowest_disc_mm"] = round(vis, 1)
    out["sacrum_covered"] = bool(vis >= cfg.ood_sacrum_min_mm)
    top = anchors.discs[labs[0]]
    margin = float(zmax) - (float(top.center[2]) + _disc_half(anchors, labs[0], cfg))
    out["superior_margin_mm"] = round(margin, 1)
    point = [float(top.center[0]), float(top.center[1]), float(zmax)]
    out["top_level_in_view"] = anchors.level_of(point)
    if margin < cfg.ood_fov_margin_mm:
        out["superior_cut"] = out["top_level_in_view"]
    names = [(float(anchors.discs[lab].center[2]), lab) for lab in labs]
    names += [(float(v.center[2]), name) for name, v in anchors.vertebrae.items()]
    inside = [(z, name) for z, name in names if zmin <= z <= zmax]
    out["levels_in_fov"] = [name for _z, name in sorted(inside, key=lambda t: -t[0])]
    return out


def body_intervals(
    seg: SegmentedStack, anchors: Anchors, cfg: Config = DEFAULT
) -> list[tuple[str, str, float]]:
    """Length (mm, along the spine) of the vertebral body between each pair of adjacent discs (mid-sagittal slice).

    Returns ``(upper disc, lower disc, length mm)``; the body is the largest network body component whose centre
    lies between the two discs. Intervals with no body component are omitted.
    """
    labs = anchors.disc_labels()
    mid = mid_sagittal(seg, cfg)
    if mid is None or len(labs) < 2:
        return []
    g = seg.grids[mid.k]
    lab_img, n = ndimage.label(seg.probs[mid.k, BODY] > cfg.ml_prob_threshold)
    bodies: list[tuple[float, float]] = []  # (mean z, length mm)
    for i in range(1, n + 1):
        rr, cc = np.nonzero(lab_img == i)
        if rr.size * g.res_mm**2 < cfg.ml_body_min_area_mm2:
            continue
        down, post = g.mm_of_index(rr, cc)
        z = g.to_lps(down, post)[:, 2]
        bodies.append((float(z.mean()), float((rr.max() - rr.min() + 1) * g.res_mm)))
    out: list[tuple[str, str, float]] = []
    for upper, lower in itertools.pairwise(labs):
        zu, zl = float(anchors.discs[upper].center[2]), float(anchors.discs[lower].center[2])
        inside = [ln for zc, ln in bodies if zl < zc < zu]
        if inside:
            out.append((upper, lower, max(inside)))
    return out


def long_body_intervals(
    seg: SegmentedStack, anchors: Anchors, cfg: Config = DEFAULT
) -> list[tuple[str, str, float]]:
    """Body intervals longer than ``ood_long_body_mm``: two bodies, so a disc between them was missed."""
    return [t for t in body_intervals(seg, anchors, cfg) if t[2] > cfg.ood_long_body_mm]


def canal_at_discs(seg: SegmentedStack, anchors: Anchors, cfg: Config = DEFAULT) -> float | None:
    """Fraction of discs with spinal canal in the mid-sagittal slice at their level (within ``ood_canal_tol_mm``).

    A spine has canal at every disc level; blobs and other anatomy do not. None when there is no disc or no slice.
    """
    labs = anchors.disc_labels()
    mid = mid_sagittal(seg, cfg)
    if mid is None or not labs:
        return None
    g = seg.grids[mid.k]
    canal = (seg.probs[mid.k, CANAL] > cfg.ml_prob_threshold).any(axis=1)
    tol = round(cfg.ood_canal_tol_mm / g.res_mm)
    hits = 0
    for lab in labs:
        down, post = g.mm_of(np.asarray(anchors.discs[lab].center, dtype=np.float64))
        row_f, _col = g.index_of_mm(down, post)
        row = int(np.clip(round(float(row_f)), 0, canal.shape[0] - 1))
        hits += int(canal[max(0, row - tol) : row + tol + 1].any())
    return hits / len(labs)


def canal_run_mm(seg: SegmentedStack, cfg: Config = DEFAULT) -> float:
    """Longest run (mm) of spinal canal along the mid-sagittal slice of the network output."""
    mid = mid_sagittal(seg, cfg)
    if mid is None:
        return 0.0
    rows = (seg.probs[mid.k, CANAL] > cfg.ml_prob_threshold).any(axis=1)
    best = run = 0
    for hit in rows:
        run = run + 1 if hit else 0
        best = max(best, run)
    return float(best) * seg.grids[mid.k].res_mm


def contrast_polarity(seg: SegmentedStack) -> float | None:
    """Canal intensity minus body intensity, in the window-normalised image (about +0.2 on T2, -0.2 on T1).

    Uses the network's own canal and body probabilities on the same grids the labeller sampled, so it reads
    the pixels, not the header. None when the network found no canal or no body.
    """
    stack = seg.stack
    src = network_input_volume(stack)
    imgs = np.stack([normalise(resample_slice(stack, g, array=src), seg.window) for g in seg.grids])
    p_canal = np.asarray(seg.probs[:, CANAL], dtype=np.float64)
    p_body = np.asarray(seg.probs[:, BODY], dtype=np.float64)
    if p_canal.sum() <= 0.0 or p_body.sum() <= 0.0:
        return None
    return float((p_canal * imgs).sum() / p_canal.sum() - (p_body * imgs).sum() / p_body.sum())


def _contradicts_contrast(declared: str, polarity: float, cfg: Config) -> bool:
    """True when the pixel polarity is on the wrong side of the declared T1/T2 contrast by the margin."""
    if declared == "t2":
        return polarity < -cfg.ood_contrast_polarity_min
    if declared == "t1":
        return polarity > cfg.ood_contrast_polarity_min
    return False


def output_entropy(seg: SegmentedStack, cfg: Config = DEFAULT) -> tuple[float, float]:
    """(mean normalised entropy over anatomy pixels, fraction of pixels that are anatomy).

    Anatomy pixels are those where a body, disc or canal probability exceeds ``ood_fg_prob``. Entropy is
    normalised to [0, 1] (0 = one class certain, 1 = uniform). No anatomy gives (1.0, 0.0).
    """
    p = np.asarray(seg.probs, dtype=np.float64)
    if p.size == 0:
        return 1.0, 0.0
    fg: NDArray[np.bool_] = p[:, 1:].max(axis=1) > cfg.ood_fg_prob
    if not fg.any():
        return 1.0, 0.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1) / np.log(p.shape[1])
    return float(ent[fg].mean()), float(fg.mean())


@dataclass
class Applicability:
    """Outcome of :func:`assess_applicability`. JSON-serialisable through :meth:`to_dict`."""

    verdict: str
    region: str
    labelled: bool
    summary: str
    factor: float = 1.0  # multiplies disc and numbering confidences (1.0 = no cue)
    cap: float | None = None  # confidence cap for ``capped`` discs
    capped: list[str] = field(default_factory=list)  # thoracic discs above T10 without a named landmark
    dropped: list[str] = field(default_factory=list)  # cervical discs removed from a whole-spine view
    sacrum_covered: bool | None = None
    unconfirmed_reasons: list[str] = field(default_factory=list)  # any entry forces numbering 'unconfirmed'
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    disc_regions: dict[str, str] = field(default_factory=dict)
    cues: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _not_labelled(verdict: str, region: str, why: str, cues: dict[str, Any]) -> Applicability:
    summary = f"not a lumbar spine study (region: {region}); levels not labelled"
    if verdict == NO_SAGITTAL:
        summary = "not a lumbar spine study: no usable sagittal series; levels not labelled"
    elif verdict == NOT_USABLE:
        summary = "not a lumbar spine study: sagittal series too short to label; levels not labelled"
    elif verdict == NOT_SPINE:
        summary = f"not a lumbar spine study ({region}); no spine anatomy found; levels not labelled"
    return Applicability(
        verdict=verdict,
        region=region,
        labelled=False,
        summary=summary,
        reasons=[why],
        warnings=[f"applicability: {summary[0].upper() + summary[1:]}: {why}"],
        cues=cues,
    )


def _summary(verdict: str, region: str) -> str:
    return {
        LUMBAR: "lumbar spine study: levels labelled",
        THORACOLUMBAR: "thoracolumbar field of view: lumbar levels labelled; thoracic levels above T10 low confidence",
        WHOLE_SPINE: "whole-spine field of view: lumbar levels labelled; cervical discs not labelled",
        UNCERTAIN: "lumbar spine not confirmed (region unclear): levels labelled with reduced confidence",
    }.get(verdict, region)


def assess_applicability(
    study: Study,
    anchors: Anchors,
    *,
    seg: SegmentedStack | None = None,
    cfg: Config = DEFAULT,
) -> Applicability:
    """Decide whether ``anchors`` (from ``study``) may carry level names, and with what confidence.

    ``seg`` is the learned labeller's output (enables the canal and entropy cues); without it only metadata,
    acquisition, disc geometry and field of view are used (the classical backend). Never raises on a study
    without a sagittal series or without discs: the verdict says so.
    """
    sag = study.best("sag")
    if sag is None:
        return _not_labelled(NO_SAGITTAL, "none", "no sagittal series", {})
    zmin, zmax = stack_z_range(sag)
    cues: dict[str, Any] = {"slices": int(sag.n), "extent_mm": round(float(zmax - zmin), 1)}
    if sag.n < cfg.ood_min_sag_slices or zmax - zmin < cfg.ood_min_sag_extent_mm:
        return _not_labelled(
            NOT_USABLE, "none", "sagittal series has too few slices or too short a z extent", cues
        )

    meta = metadata_region(sag)
    acq = acquisition_cues(sag, cfg)
    regions = disc_regions(anchors, cfg)
    if sum(r == "cervical" for r in regions.values()) == 1:
        # A single cervical-sized disc is not a cervical column: it is noise or a fragment, not grounds to drop it.
        regions = {lab: ("unknown" if r == "cervical" else r) for lab, r in regions.items()}
    counts = Counter(regions.values())
    n_lum, n_thor, n_cerv = counts["lumbar"], counts["thoracic"], counts["cervical"]
    cues.update(
        {"discs": len(regions), "lumbar_discs": n_lum, "thoracic_discs": n_thor, "cervical_discs": n_cerv}
    )
    cues["metadata_region"] = meta.region
    cues["metadata_strong"] = meta.strong
    canal: float | None = None
    canal_fraction: float | None = None
    if seg is not None:
        canal = canal_run_mm(seg, cfg)
        canal_fraction = canal_at_discs(seg, anchors, cfg)
        ent, fg_frac = output_entropy(seg, cfg)
        cues.update(
            {
                "canal_run_mm": round(canal, 1),
                "canal_at_discs": None if canal_fraction is None else round(canal_fraction, 3),
                "entropy": round(ent, 3),
                "anatomy_fraction": round(fg_frac, 3),
            }
        )
    cues["disc_heights_mm"] = [round(anchors.discs[lab].height_mm or 0.0, 1) for lab in anchors.disc_labels()]

    if meta.region is not None and meta.strong and meta.region not in SPINE_REGIONS:
        why = f"BodyPartExamined names {meta.region}"
        return _not_labelled(NOT_LUMBAR, meta.region, why, cues)
    region_if_none = meta.region if (meta.region and meta.region not in SPINE_REGIONS) else "not spine"
    if len(regions) < cfg.ood_min_discs:
        why = f"{len(regions)} disc(s) in view: fewer than {cfg.ood_min_discs}, so there is no chain to count"
        return _not_labelled(NOT_SPINE, region_if_none, why, cues)
    if canal is not None and canal < cfg.canal_min_length_mm and n_lum == 0:
        why = f"spinal canal {canal:.0f} mm long on the mid-sagittal slice (no disc chain in a canal)"
        return _not_labelled(NOT_SPINE, region_if_none, why, cues)
    if canal_fraction is not None and canal_fraction < cfg.ood_canal_fraction_min:
        why = f"spinal canal present at {canal_fraction:.0%} of the disc levels (a spine has canal at every disc)"
        return _not_labelled(NOT_SPINE, region_if_none, why, cues)
    if not regions:
        return _not_labelled(NOT_SPINE, "not spine", "no disc chain found", cues)

    reasons: list[str] = []
    warnings: list[str] = []
    dropped: list[str] = []
    factor = 1.0
    labs = anchors.disc_labels()
    heights = [h for h in (anchors.discs[lab].height_mm for lab in labs) if h]
    gaps = [
        float(np.linalg.norm(np.asarray(anchors.discs[a].center) - np.asarray(anchors.discs[b].center)))
        for a, b in itertools.pairwise(labs)
    ]
    med_h = float(np.median(heights)) if heights else None
    med_sp = float(np.median(gaps)) if gaps else None
    # The two lowest discs decide the lumbar question: a thoracolumbar view has thin thoracic discs above and lumbar
    # discs at the bottom, and only a view whose bottom discs are thin too is a thoracic or cervical column.
    low_thin = len(labs) >= 2 and all(
        (anchors.discs[lab].height_mm or 0.0) < cfg.ood_lumbar_disc_min_mm for lab in labs[-2:]
    )
    # Every disc in view thin: a single lumbar-sized disc in the column keeps the study lumbar.
    all_thin = bool(heights) and max(heights) < cfg.ood_lumbar_disc_min_mm
    small_fov = (
        med_h is not None
        and med_h < cfg.ood_lumbar_disc_min_mm
        and (med_sp is None or med_sp < cfg.ood_lumbar_spacing_min_mm)
        and low_thin
        and all_thin
    )
    if small_fov and n_lum > 0:
        # Most discs are short and close together: a thoracic or cervical column, whatever a few discs say.
        verdict = NOT_LUMBAR
        region = "cervical" if med_sp is not None and med_sp < cfg.ood_cervical_spacing_max_mm else "thoracic"
        why = f"median disc height {med_h:.1f} mm and spacing {med_sp or 0:.1f} mm: a {region} column"
        return _not_labelled(verdict, region, why, cues)
    if n_lum > 0:
        if n_cerv > 0:
            verdict, region = WHOLE_SPINE, "whole spine"
            dropped = [lab for lab, r in regions.items() if r == "cervical"]
            reasons.append(f"{n_cerv} cervical-sized disc(s) above the lumbar discs: dropped, not labelled")
        elif n_thor > 0:
            verdict, region = THORACOLUMBAR, "thoracolumbar"
            reasons.append(f"{n_thor} thoracic-sized disc(s) above the lumbar discs")
        else:
            verdict, region = LUMBAR, "lumbar"
        reasons.append(f"{n_lum} lumbar-sized disc(s) (height and spacing)")
        labelled = True
    elif n_thor > 0 or n_cerv > 0:
        region = "cervicothoracic" if n_thor and n_cerv else "thoracic" if n_thor else "cervical"
        verdict, labelled = NOT_LUMBAR, False
        reasons.append(f"{n_thor} thoracic and {n_cerv} cervical-sized disc(s); no lumbar disc in view")
    else:
        verdict, region, labelled = UNCERTAIN, meta.region or "unclear", True
        reasons.append("disc sizes and spacings do not identify a region")
        factor *= cfg.ood_offdist_factor
    if not labelled:
        return _not_labelled(verdict, region, "; ".join(reasons), cues)

    if meta.region in ("thoracic", "cervical") and not meta.strong:
        warnings.append(f"series description names {meta.region}, but the discs look {region}: labels kept")
        factor *= cfg.ood_offdist_factor
    if meta.region in ("thoracic", "cervical") and meta.strong:
        warnings.append(f"BodyPartExamined names {meta.region}; the anatomy shows lumbar discs: labels kept")
        factor *= cfg.ood_offdist_factor
    for why in acq:
        factor *= cfg.ood_offdist_factor
        warnings.append(f"applicability: {why}: confidence lowered")
    if seg is not None and cues["entropy"] > cfg.ood_entropy_max:
        factor *= cfg.ood_offdist_factor
        warnings.append(
            f"applicability: network output uncertain (entropy {cues['entropy']:.2f} > {cfg.ood_entropy_max:.2f})"
        )
    if seg is not None:
        # The pixels, not the header, say whether this is T1 or T2: a contradiction lowers confidence.
        pol = contrast_polarity(seg)
        declared = str(sag.meta.get("role", {}).get("contrast", "unknown"))
        cues["contrast_polarity"] = None if pol is None else round(pol, 3)
        if pol is not None and _contradicts_contrast(declared, pol, cfg):
            factor *= cfg.ood_offdist_factor
            side = "brighter" if pol > 0 else "darker"
            warnings.append(
                f"applicability: intensities contradict the declared {declared.upper()} contrast (canal is "
                f"{side} than bone): the contrast metadata may be wrong; confidence lowered"
            )
    if verdict == UNCERTAIN:
        warnings.append(
            "applicability: region not confirmed by the anatomy: levels labelled with lower confidence"
        )
    factor = max(cfg.ood_factor_floor, factor)

    coverage = fov_coverage(sag, anchors, cfg)
    sacrum = coverage.get("sacrum_covered")
    unconfirmed: list[str] = []
    long_bodies = [] if seg is None else long_body_intervals(seg, anchors, cfg)
    cues["long_bodies_mm"] = [round(ln, 1) for _u, _l, ln in long_bodies]
    if seg is not None:
        cues["body_intervals_mm"] = [round(ln, 1) for _u, _l, ln in body_intervals(seg, anchors, cfg)]
    labs = anchors.disc_labels()
    cues["disc_spacings_mm"] = [
        round(
            float(np.linalg.norm(np.asarray(anchors.discs[a].center) - np.asarray(anchors.discs[b].center))),
            1,
        )
        for a, b in itertools.pairwise(labs)
    ]
    for upper, lower, ln in long_bodies:
        unconfirmed.append(f"body between {upper} and {lower} is {ln:.0f} mm long: a disc may be missed")
        warnings.append(
            f"applicability: body between {upper} and {lower} is {ln:.0f} mm long (one body is at most "
            f"{cfg.ood_long_body_mm:.0f} mm): a disc may be missed, so levels below it are not confirmed"
        )
    if sacrum is False:
        unconfirmed.append("sacrum not in the field of view")
        warnings.append(
            "applicability: the sacrum is not in the field of view (bone below the lowest disc "
            f"{coverage.get('bone_below_lowest_disc_mm')} mm): numbering cannot be confirmed"
        )
    for a, b, gap in _close_pairs(labs, gaps, cfg) if n_lum > 0 else []:
        unconfirmed.append(f"discs {a} and {b} are only {gap:.0f} mm apart: a disc may be spurious or missed")
        warnings.append(
            f"applicability: discs {a} and {b} are only {gap:.0f} mm apart (a lumbar column is at least "
            f"{cfg.ood_cervical_spacing_max_mm:.0f} mm): the chain between them is not confirmed"
        )
    if coverage.get("superior_cut"):
        reasons.append(
            f"field of view ends at {coverage['superior_cut']}: levels above it are outside the view"
        )

    described = described_disc_labels(study)
    capped = []
    if verdict in (THORACOLUMBAR, WHOLE_SPINE, LUMBAR, UNCERTAIN):
        capped = [
            lab for lab in anchors.discs if lab not in dropped and lab not in described and _above_t10(lab)
        ]
    if capped:
        warnings.append(
            f"applicability: {len(capped)} thoracic disc(s) above T10 counted from the sacrum only: "
            f"confidence capped at {cfg.ood_thoracic_cap:.2f}"
        )

    app = Applicability(
        verdict=verdict,
        region=region,
        labelled=True,
        summary=_summary(verdict, region),
        factor=round(factor, 4),
        cap=cfg.ood_thoracic_cap if capped else None,
        capped=capped,
        dropped=dropped,
        sacrum_covered=sacrum,
        unconfirmed_reasons=unconfirmed,
        reasons=reasons,
        warnings=warnings,
        coverage=coverage,
        disc_regions=regions,
        cues=cues,
    )
    app.reasons.extend(acq)
    return app


def _above_t10(label: str) -> bool:
    """True for a disc whose upper vertebra is T10 or higher (``T10-T11`` and above)."""
    upper = label.split("-")[0]
    return upper in VERTEBRA_NAMES and VERTEBRA_NAMES.index(upper) >= _T10_INDEX


def apply_applicability(
    anchors: Anchors, app: Applicability, *, thresholds: dict[str, float] | None = None
) -> None:
    """Write ``app`` into ``anchors`` in place: the ``applicability`` record, warnings, labels and status.

    Not labelled: discs, vertebrae and conus are cleared and the numbering is ``unconfirmed`` with confidence 0.
    Labelled: cervical discs are dropped, capped discs get the cap, confidences are scaled by ``app.factor``, and
    the numbering status is recomputed with ``thresholds`` (default: the uncalibrated thresholds). A missing
    sacrum forces ``unconfirmed``.
    """
    th = thresholds or DEFAULT_STATUS_THRESHOLDS
    anchors.applicability = app.to_dict()
    for w in app.warnings:
        if w not in anchors.warnings:
            anchors.warnings.append(w)
    num = dict(anchors.numbering)
    if not app.labelled:
        anchors.discs = {}
        anchors.vertebrae = {}
        anchors.conus = None
        num.update({"status": STATUS_UNCONFIRMED, "confidence": 0.0, "labelled": False, "notes": app.summary})
        anchors.numbering = num
        return
    for lab in app.dropped:
        anchors.discs.pop(lab, None)
    for lab, d in anchors.discs.items():
        c = d.confidence * app.factor
        if lab in app.capped and app.cap is not None:
            c = min(c, app.cap)
        if c != d.confidence:
            d.evidence.append(
                f"confidence {c:.2f} after applicability (factor {app.factor:.2f}): {app.summary}"
            )
        d.confidence = float(np.clip(c, 0.0, 1.0))
    if anchors.discs and "confidence" in num:
        before = float(num["confidence"])
        new = before * app.factor
        num["confidence_before_applicability"] = round(before, 4)
        num["confidence"] = round(new, 4)
        num["status"] = numbering_status(new, th)
    if app.unconfirmed_reasons:
        num["status"] = STATUS_UNCONFIRMED
    num["labelled"] = True
    num["applicability_factor"] = app.factor
    anchors.numbering = num


__all__ = [
    "LABELLED_VERDICTS",
    "Applicability",
    "RegionCue",
    "acquisition_cues",
    "apply_applicability",
    "assess_applicability",
    "body_intervals",
    "canal_at_discs",
    "canal_run_mm",
    "disc_region",
    "disc_regions",
    "fov_coverage",
    "long_body_intervals",
    "metadata_region",
    "output_entropy",
    "region_from_text",
]
