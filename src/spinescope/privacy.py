"""Privacy helpers for everything spinescope writes or returns.

Policy (see ``docs/privacy.md`` for the full threat model):

* Attributes are copied only from the technical allowlist :data:`spinescope.io.KEEP_TAGS`. Names, IDs, dates,
  institutions, physicians, UIDs, serial numbers and file paths never enter an output.
* The classifier and the level-name readers need the free text of ``SeriesDescription``, ``ProtocolName`` and
  ``SequenceName`` (contrast and plane words, and disc level names such as ``L4-5 OBL AX``). That text stays in
  memory, under the same keys, and is never written as is: every output goes through :func:`public_text`.
* :func:`public_text` keeps only tokens of a protocol vocabulary (contrast, plane, sequence and coil words,
  level names, small numbers with units). Any other token, which may be a name, an ID, a date or a site code,
  becomes ``*``. Cost: a free-text description that is not protocol wording loses that wording in outputs.
  Benefit: the one field a technologist can fill with a name cannot carry one into a report, a viewer or an
  AI prompt.
"""

from __future__ import annotations

import re
from typing import Any

#: Words a protocol name may use. Upper case. Covers the words the series classifier and the level-name readers look
#: for, plus common sequence, plane and coil vocabulary. Extend it when a new classifier keyword is added
#: (``tests/test_privacy_canaries.py`` checks that the classifier's own examples survive scrubbing).
_VOCAB: frozenset[str] = frozenset(
    [
        "SAG",
        "SAGITTAL",
        "AX",
        "AXIAL",
        "COR",
        "CORONAL",
        "OBL",
        "OBLIQUE",
        "OBLIQ",
        "OBLIQUE",
        "TRANS",
        "AXL",
        "SAGL",
        "CORL",
        "LOC",
        "LOCALIZER",
        "LOCALISER",
        "SCOUT",
        "SURVEY",
        "PL",
        "SPINE",
        "SPINAL",
        "LUMBAR",
        "LSPINE",
        "TSPINE",
        "CSPINE",
        "LUMBOSACRAL",
        "CERVICAL",
        "THORACIC",
        "SACRUM",
        "SACRAL",
        "DISC",
        "DISCS",
        "T1",
        "T2",
        "T1W",
        "T2W",
        "T1W1",
        "T2W1",
        "T2STAR",
        "T1STAR",
        "PD",
        "PDW",
        "STIR",
        "FLAIR",
        "DWI",
        "DIFFUSION",
        "ADC",
        "IVIM",
        "TSE",
        "FSE",
        "FFE",
        "GRE",
        "SE",
        "IR",
        "SPGR",
        "MPRAGE",
        "SPACE",
        "VIBE",
        "CUBE",
        "SPC",
        "EPI",
        "HASTE",
        "TRUFI",
        "FIESTA",
        "SSFSE",
        "SPIR",
        "SPAIR",
        "FS",
        "FAT",
        "SAT",
        "SATURATION",
        "FATSAT",
        "SFS",
        "DIXON",
        "IDEAL",
        "WATER",
        "IP",
        "OP",
        "IN",
        "OUT",
        "PHASE",
        "DIXONW",
        "DIXONF",
        "FLOW",
        "MRA",
        "MIP",
        "MPR",
        "CE",
        "GD",
        "GAD",
        "POST",
        "PRE",
        "PLUS",
        "CONTRAST",
        "ENHANCED",
        "UNENHANCED",
        "NON",
        "WITHOUT",
        "WITH",
        "TSE3D",
        "THIN",
        "THICK",
        "SLAB",
        "SLICE",
        "CUT",
        "MM",
        "CM",
        "MR",
        "ORIGINAL",
        "PRIMARY",
        "DERIVED",
        "SECONDARY",
        "NORM",
        "M",
        "OTHER",
        "LOCALIZER",
        "SPIN",
        "ECHO",
        "GRADIENT",
        "INVERSION",
        "RECOVERY",
        "NONCONTRAST",
        "COIL",
        "SPINE_COIL",
        "BODY",
        "HEAD",
        "NECK",
        "MULTI",
        "SINGLE",
        "FOV",
        "LOW",
        "HIGH",
        "RES",
        "HR",
        "LR",
        "REFORMAT",
        "RECON",
        "STACK",
        "CONTROL",
        "SHIM",
        "SE2D",
        "GE3D",
        # words the series classifier matches (series.py): keep them readable in outputs
        "BRAVO",
        "PROTON",
        "TIRM",
        "CHESS",
        "FSAT",
        "SPAIR-IR",
        "SPAIR",
        "SPIR",
        "MEDIC",
        "SWI",
        "T2STAR",
        "FFE",
        "SPGR",
        "FSPGR",
        "LAVA",
        "THRIVE",
        "FLASH",
        "EP2D_DIFF",
        "TRACE",
        "RESOLVE",
        "DIFF",
        "GAD",
        "GADO",
        "NON-CONTRAST",
        "NON-ENHANCED",
        "NONCONTRAST",
        "UNENHANCED",
        "PRECONTRAST",
        "POSTENHANCED",
        "+C",
        "C+",
        "CE",
        "PREC",
        "IDEAL",
        "OPP",
    ]
)

#: Level names (T10-T11, L4-5, L4-L5, L5-S1, L1, S1) and numbers with protocol units (3PL, 3-PL, 4MM, 3.5MM, 3D).
_LEVEL_RE = re.compile(r"^[TLS]\d{1,2}(-[TLS]?\d{1,2})?$")
_UNIT_RE = re.compile(r"^\d{1,3}(\.\d{1,2})?(MM|MM2|PL|D|PL)?$|^\d-?PL$|^\d{1,2}D$")
#: Contrast-weighted modality words with fat saturation or contrast suffixes: T1FS, T2WFS, T1+C, T2*, T1W+C.
_MODALITY_RE = re.compile(r"^(T[12]W?\*?|PD)(FS|FSE|TSE|FSAT|SE)?(\+C\+?|C\+)?$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9+][A-Za-z0-9._*+\-]*")

#: Placeholder for a dropped token (ASCII, so it survives every report format).
REDACTED = "*"


def _keep(token: str) -> bool:
    t = token.upper().rstrip(".")
    if t in _VOCAB:
        return True
    if _LEVEL_RE.match(t) or _UNIT_RE.match(t) or _MODALITY_RE.match(t):
        return True
    return t.replace("*", "") in _VOCAB  # T2* and similar


def public_text(value: Any, *, max_len: int = 80) -> str:
    """Protocol wording only: tokens outside the vocabulary become ``*`` (runs collapse to one ``*``).

    ``None`` and empty values give ``""``. Output is ASCII and at most ``max_len`` characters.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(public_text(v, max_len=max_len) for v in value).strip()
    raw = str(value)
    out: list[str] = []
    for tok in _TOKEN_RE.findall(raw):
        piece = tok.upper() if _keep(tok) else REDACTED
        if piece == REDACTED and out and out[-1] == REDACTED:
            continue
        out.append(piece)
    text = " ".join(out)
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "~"
    return text
