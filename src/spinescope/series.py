"""Classify stacks (plane, contrast, fat suppression, localizer...) and give them stable keys.

Keys look like ``sag_t2``, ``sag_t1``, ``sag_stir``, ``ax_t2``, ``ax_t1``, ``cor_t2_fs``,
``sag_t1_post``, ``ax_t2_slab1`` (one per angled slab of a multi-slab series),
``loc_sag`` (localizers), with ``_2``, ``_3`` suffixes when a key repeats.

Classification uses acquisition parameters first (TR, TE, TI, scanning sequence, image type)
and the vendor's series description as supporting evidence; every decision is recorded in
``stack.meta['role']['evidence']`` so it can be audited.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .geometry import Stack, unit

_LOCALIZER_RE = re.compile(r"\b(LOC|LOCALI[SZ]ER|SCOUT|SURVEY|TRI-?PILOT|3-?PL(ANE)?|AAHSCOUT|PLANNING)\b")
_STIR_RE = re.compile(r"\b(STIR|TIRM|IR|T2 ?IR|SPAIR-?IR)\b")
_FLAIR_RE = re.compile(r"\bFLAIR\b")
_FS_RE = re.compile(r"(\bFS\b|FAT ?SAT|FATSAT|\bSPAIR\b|\bSPIR\b|\bCHESS\b|\bFSAT\b)")
_POST_RE = re.compile(r"(\+C\b|\bC\+|\bPOST\b|\bGAD|\bGD\b|\bCONTRAST\b|\bENHANC)")
_T1_RE = re.compile(r"\bT1")
_T2_RE = re.compile(r"\bT2")
_PD_RE = re.compile(r"\bPD\b|PROTON")
_DWI_RE = re.compile(r"\b(DWI|DIFF|ADC|TRACE|EP2D_DIFF|RESOLVE)\b")
_DIXON = {
    "w": ("WATER", "W"),
    "f": ("FAT", "F"),
    "ip": ("IN_PHASE", "IN", "IP"),
    "op": ("OUT_PHASE", "OPP", "OP"),
}
_DIXON_DESC = {
    "w": r"(\bW\b|\bWATER\b)",
    "f": r"(\bF\b|\bFAT\b(?! ?SAT))",
    "ip": r"(\bIN\b|\bIN ?PHASE\b)",
    "op": r"(\bOPP?\b|\bOUT ?PHASE\b)",
}

#: Target in-plane display directions per plane: (row_dir, col_dir), i.e. the patient-space
#: directions of increasing column (left->right on screen) and increasing row (top->bottom).
#: Sagittal: anterior on the left, head at the top. Axial/coronal: patient's left on screen right.
DISPLAY_AXES = {
    "sag": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "ax": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    "cor": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
}


@dataclass
class Role:
    plane: str
    contrast: str  # t1 | t2 | pd | stir | flair | gre | dwi | adc | unknown
    fat_suppressed: bool = False
    post_contrast: bool = False
    localizer: bool = False
    dixon: str | None = None  # w | f | ip | op
    slab_index: int | None = None
    slab_count: int = 1
    evidence: list[str] = field(default_factory=list)

    def base_key(self) -> str:
        if self.localizer:
            return f"loc_{self.plane}"
        k = f"{self.plane}_{self.contrast}"
        if self.dixon:
            k += f"_dixon_{self.dixon}"
        elif self.fat_suppressed and self.contrast not in ("stir",):
            k += "_fs"
        if self.post_contrast:
            k += "_post"
        return k


def _text(meta: dict[str, Any]) -> str:
    """Upper-cased description/protocol/sequence names with separators turned into spaces."""
    d = meta.get("dicom", {})
    raw = " ".join(str(d.get(k, "")) for k in ("SeriesDescription", "ProtocolName", "SequenceName")).upper()
    raw = raw.replace("F/S", " FS ")
    return " ".join(re.sub(r"[_\-/.,;:()\[\]*#]+", " ", raw).split())


def _num(d: dict[str, Any], k: str) -> float | None:
    v = d.get(k)
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).upper() for x in v]
    return [str(v).upper()]


def classify(stack: Stack, *, n_orientations_in_series: int = 1) -> Role:
    """Infer the stack's role from its DICOM attributes and geometry."""
    d = stack.meta.get("dicom", {})
    text = _text(stack.meta)
    itype = _as_list(d.get("ImageType"))
    seq = _as_list(d.get("ScanningSequence")) + _as_list(d.get("SequenceVariant"))
    opts = _as_list(d.get("ScanOptions"))
    tr, te, ti = _num(d, "RepetitionTime"), _num(d, "EchoTime"), _num(d, "InversionTime")
    flip = _num(d, "FlipAngle")
    ev: list[str] = []
    role = Role(plane=stack.plane, contrast="unknown")

    # Localizer
    if "LOCALIZER" in itype or _LOCALIZER_RE.search(text):
        role.localizer = True
        ev.append("localizer: ImageType/description")
    elif n_orientations_in_series >= 3 and stack.n <= 12 and stack.thickness >= 6:
        role.localizer = True
        ev.append("localizer: >=3 orientations, few thick slices")

    # Diffusion
    b = _num(d, "DiffusionBValue")
    if "DIFFUSION" in itype or (b is not None and b > 0) or _DWI_RE.search(text):
        role.contrast = "adc" if ("ADC" in itype or "ADC" in text) else "dwi"
        ev.append("diffusion: ImageType/b-value/description")

    # Dixon component
    for comp, tags in _DIXON.items():
        if any(t in itype for t in tags) and ("DIXON" in text or "DIXON" in itype or comp in ("w", "f")):
            if comp in ("w", "f") and not ("WATER" in itype or "FAT" in itype or "DIXON" in text):
                continue
            role.dixon = comp
            ev.append(f"dixon {comp}: ImageType")
            break
    if role.dixon is None and "DIXON" in text:
        for comp, rx in _DIXON_DESC.items():
            if re.search(rx, text):
                role.dixon = comp
                ev.append(f"dixon {comp}: description")
                break

    if role.contrast == "unknown":
        is_ir = "IR" in seq or (ti is not None and ti > 0)
        if (ti is not None and ti > 1200) or _FLAIR_RE.search(text):
            role.contrast = "flair"
            ev.append(f"flair: TI={ti}")
        elif is_ir and ti is not None and 60 <= ti <= 300:
            role.contrast = "stir"
            ev.append(f"stir: TI={ti} ms")
        elif _STIR_RE.search(text) and not (_T1_RE.search(text) and ti is not None and ti > 300):
            role.contrast = "stir"
            ev.append("stir: description")
        elif te is not None and tr is not None:
            gre = "GR" in seq and "SE" not in seq
            if gre and flip is not None and flip < 40 and te > 8:
                role.contrast = "gre"
                ev.append(f"gre: GR, flip={flip}, TE={te}")
            elif te < 30 and tr < 1300:
                role.contrast = "t1"
                ev.append(f"t1: TR={tr} TE={te}")
            elif te >= 60:
                role.contrast = "t2"
                ev.append(f"t2: TR={tr} TE={te}")
            elif te < 45 and tr >= 1300:
                role.contrast = "pd"
                ev.append(f"pd: TR={tr} TE={te}")
        if role.contrast in ("unknown", "pd", "gre"):
            # description as tie-breaker
            if _T1_RE.search(text):
                role.contrast = "t1"
                ev.append("t1: description")
            elif _T2_RE.search(text):
                role.contrast = "t2"
                ev.append("t2: description")
            elif _PD_RE.search(text):
                role.contrast = "pd"
                ev.append("pd: description")

    if role.contrast != "stir" and (
        any(o in ("FS", "FAT_SAT", "FATSAT", "SFS", "SPECTRAL_SAT") for o in opts) or _FS_RE.search(text)
    ):
        role.fat_suppressed = True
        ev.append("fat-suppressed: ScanOptions/description")
    if d.get("ContrastBolusAgent") or _POST_RE.search(text):
        role.post_contrast = True
        ev.append("post-contrast: ContrastBolusAgent/description")
    role.evidence = ev
    return role


def to_display_orientation(stack: Stack) -> Stack:
    """Return ``stack`` with its in-plane axes permuted/flipped to the standard display.

    Sagittal: anterior on screen left, head up. Axial: anterior up, patient left on screen right.
    Coronal: head up, patient left on screen right. Geometry stays exact (``ipp`` and direction
    vectors are updated), so patient coordinates of every voxel are unchanged.
    """
    tr, tc = DISPLAY_AXES[stack.plane]
    vol, ipp = stack.vol, stack.ipp.copy()
    rd, cd, rmm, cmm = stack.row_dir.copy(), stack.col_dir.copy(), stack.row_mm, stack.col_mm
    ops = []
    if abs(rd @ tc) + abs(cd @ tr) > abs(rd @ tr) + abs(cd @ tc):
        vol = vol.transpose(0, 2, 1)
        rd, cd = cd, rd
        rmm, cmm = cmm, rmm
        ops.append("transpose")
    rows, cols = vol.shape[1:]
    if rd @ tr < 0:
        ipp = ipp + (cols - 1) * cmm * rd
        rd = -rd
        vol = vol[:, :, ::-1]
        ops.append("flip_columns")
    if cd @ tc < 0:
        ipp = ipp + (rows - 1) * rmm * cd
        cd = -cd
        vol = vol[:, ::-1, :]
        ops.append("flip_rows")
    if not ops:
        return stack
    meta = dict(stack.meta)
    meta["display_transform"] = ops
    return Stack(
        key=stack.key,
        vol=np.ascontiguousarray(vol),
        ipp=ipp,
        row_dir=rd,
        col_dir=cd,
        row_mm=rmm,
        col_mm=cmm,
        thickness=stack.thickness,
        description=stack.description,
        series_number=stack.series_number,
        meta=meta,
    )


def assign_keys(stacks: list[Stack]) -> dict[str, Stack]:
    """Classify every stack and return them keyed by role (stable, human-readable keys)."""
    by_series: dict[str, list[Stack]] = defaultdict(list)
    for st in stacks:
        by_series[st.meta.get("series_uid_hash", id(st))].append(st)
    roles: list[tuple[Stack, Role]] = []
    for group in by_series.values():
        orients = {tuple(np.round(unit(s.normal), 2)) for s in group}
        group_roles = [(st, classify(st, n_orientations_in_series=len(orients))) for st in group]
        # multi-slab series: several non-localizer orientations (e.g. axial slabs angled per disc)
        nonloc = [(s, r) for s, r in group_roles if not r.localizer]
        if len({tuple(np.round(s.normal, 2)) for s, _ in nonloc}) > 1:
            nonloc.sort(key=lambda sr: -float(sr[0].ipp[:, 2].mean()))
            for i, (_, r) in enumerate(nonloc, 1):
                r.slab_index, r.slab_count = i, len(nonloc)
        roles.extend(group_roles)
    keyed: dict[str, Stack] = {}
    counts: dict[str, int] = defaultdict(int)
    # deterministic order: series number, then superior first
    roles.sort(key=lambda sr: (sr[0].series_number or 0, -float(sr[0].ipp[:, 2].mean())))
    slab_series_base: dict[str, str] = {}
    for st, role in roles:
        base = role.base_key()
        sid = st.meta.get("series_uid_hash", "")
        if role.slab_index is not None:
            if sid not in slab_series_base:
                counts[base + "_slabs"] += 1
                n = counts[base + "_slabs"]
                slab_series_base[sid] = base if n == 1 else f"{base}_{n}"
            key = f"{slab_series_base[sid]}_slab{role.slab_index}"
        elif role.localizer:
            counts[base] += 1
            key = base if counts[base] == 1 else f"{base}_{counts[base]}"
        else:
            counts[base] += 1
            key = base if counts[base] == 1 else f"{base}_{counts[base]}"
        st.key = key
        st.meta["role"] = asdict(role)
        keyed[key] = st
    return keyed
