"""Ground-truth gradings for the analysis harness: SPIDER per-disc radiological gradings, mapped to level names.

SPIDER (van der Graaf et al., Scientific Data 2024; CC BY 4.0) grades every intervertebral disc of every patient
in ``radiological_gradings.csv`` (one row per patient and ``IVD label``). The columns and the values observed in
the public release (1 520 rows, 218 patients) are:

=====================  ==============================  ==========================================
column                 field here                      observed values (count of rows)
=====================  ==============================  ==========================================
Modic                  ``modic``                       0 (1006), 1 (4), 2 (503), 3 (7)
UP endplate            ``up_endplate``                 0 (906), 1 (614)
LOW endplate           ``low_endplate``                0 (897), 1 (623)
Spondylolisthesis      ``spondylolisthesis``           0 (1478), 1 (42)
Disc herniation        ``disc_herniation``             0 (1448), 1 (72)
Disc narrowing         ``disc_narrowing``              0 (976), 1 (544)
Disc bulging           ``disc_bulging``                0 (774), 1 (746)
Pfirrman grade         ``pfirrmann``                   1 (286), 2 (341), 3 (418), 4 (291), 5 (184)
=====================  ==============================  ==========================================

Semantics. **Unverified**: the local dataset folder has no codebook (its README describes files only), and the
publisher page could not be read. The meaning of each code is defined in the SPIDER paper (Modic type; endplate defect present or
absent; disc herniation, narrowing and bulging as present or absent; Pfirrmann 1-5). **The exact paper definitions
could not be read while writing this module** (the publisher page redirects to a login), so this module reports
the codes as they appear and treats UP / LOW endplate as "endplate abnormal (1) or not (0)". Anything scored
against these fields must say so (see ``docs/analysis/README.md``).

Level names. SPIDER numbers IVDs from the bottom: IVD 1 is the lowest disc, L5-S1 in normal anatomy, IVD 2 is
L4-L5, and so on. The mask convention (disc label 200 + IVD label; vertebra 1 is the lowest lumbar body) is
checked in :mod:`spinescope.anatomy._validate`. The sacrum is not labelled, so the name of IVD 1 depends on
that convention; a transitional spine can shift every name by one.

Transitional flag. :func:`load_spider_file_meta` reads ``num_vertebrae`` from ``overview.csv``. The release
labels vertebrae down from the lowest lumbar body and up into the thoracic spine, so the count is 6 to 9 in
almost every volume (only one volume has 5). ``transitional_suspect`` is therefore set whenever the count is
not 5, as specified for this wave, but it fires for about 99% of volumes and does not identify transitional
anatomy. Breakdowns by it are reported for completeness and should not be read as a transitional-spine result.

Patient-identifying columns of ``overview.csv`` (sex, birth date, device serial) are never read.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import SPIDER, data_dir

log = logging.getLogger(__name__)

#: Vertebra names from the bottom up, as SPIDER numbers them: index 0 is the sacrum (S1), 1 is L5, ...
SPIDER_BOTTOM_UP: tuple[str, ...] = ("S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10", "T9", "T8")

#: CSV column of ``radiological_gradings.csv`` -> field name used everywhere in spinescope.
GRADING_COLUMNS: dict[str, str] = {
    "Modic": "modic",
    "UP endplate": "up_endplate",
    "LOW endplate": "low_endplate",
    "Spondylolisthesis": "spondylolisthesis",
    "Disc herniation": "disc_herniation",
    "Disc narrowing": "disc_narrowing",
    "Disc bulging": "disc_bulging",
    "Pfirrman grade": "pfirrmann",  # sic: the column name in the release is misspelt
}
#: Fields that are graded in the release (in CSV order).
GRADING_FIELDS: tuple[str, ...] = tuple(GRADING_COLUMNS.values())
#: Lumbar vertebra count that the transitional flag compares with (see module docstring).
STANDARD_VERTEBRA_COUNT = 5


def ivd_level_name(ivd: int) -> str | None:
    """Level name of a SPIDER IVD label: 1 -> ``L5-S1``, 2 -> ``L4-L5``, ...; None outside the naming table."""
    if not 1 <= ivd < len(SPIDER_BOTTOM_UP):
        return None
    return f"{SPIDER_BOTTOM_UP[ivd]}-{SPIDER_BOTTOM_UP[ivd - 1]}"


@dataclass(frozen=True)
class DiscGrading:
    """Radiologist grading of one disc of one SPIDER patient (ground truth; values are the release codes)."""

    case: str
    ivd_label: int
    level: str | None  # e.g. "L4-L5"; None if the IVD label is outside the naming table
    transitional_suspect: bool
    modic: int | None
    up_endplate: int | None
    low_endplate: int | None
    spondylolisthesis: int | None
    disc_herniation: int | None
    disc_narrowing: int | None
    disc_bulging: int | None
    pfirrmann: int | None

    def value(self, field_name: str) -> int | None:
        """Grade of one field (see :data:`GRADING_FIELDS`)."""
        if field_name not in GRADING_FIELDS:
            raise KeyError(f"unknown grading field {field_name!r}")
        out: int | None = getattr(self, field_name)
        return out


@dataclass(frozen=True)
class FileMeta:
    """Technical metadata of one SPIDER volume (no identifying columns)."""

    case: str
    sequence: str  # e.g. "t1", "t2", "t2_SPACE"
    subset: str  # "training" | "validation"
    num_vertebrae: int | None
    vendor: str  # "Siemens" | "Philips" | "other"
    field_strength: str | None  # "1.5" | "3.0"


def vendor_of(manufacturer: str) -> str:
    """Normalise the DICOM manufacturer string of ``overview.csv``."""
    m = manufacturer.upper()
    if "SIEMENS" in m:
        return "Siemens"
    if "PHILIPS" in m:
        return "Philips"
    return "other"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _int_or_none(text: str | None) -> int | None:
    try:
        return int(float(str(text).strip()))
    except (TypeError, ValueError):
        return None


def spider_dir(root: str | Path | None = None) -> Path:
    """SPIDER folder: ``root`` if given, else ``$SPINESCOPE_DATA/spider``."""
    return Path(root).expanduser() if root is not None else data_dir() / SPIDER.subdir


def load_spider_file_meta(root: str | Path | None = None) -> dict[tuple[str, str], FileMeta]:
    """Technical metadata per (case, sequence) from ``overview.csv``; keys are e.g. ``("1", "t2")``."""
    path = spider_dir(root) / "overview.csv"
    if not path.is_file():
        raise FileNotFoundError(f"SPIDER overview.csv not found at {path}")
    out: dict[tuple[str, str], FileMeta] = {}
    for row in _read_csv(path):
        name = (row.get("new_file_name") or "").strip()
        case, _, seq = name.partition("_")
        if not case or not seq:
            continue
        strength = (row.get("MagneticFieldStrength") or "").strip() or None
        out[(case, seq)] = FileMeta(
            case=case,
            sequence=seq,
            subset=(row.get("subset") or "").strip(),
            num_vertebrae=_int_or_none(row.get("num_vertebrae")),
            vendor=vendor_of(row.get("Manufacturer") or ""),
            field_strength=strength,
        )
    return out


def load_spider_gradings(root: str | Path | None = None) -> dict[str, list[DiscGrading]]:
    """Per-disc gradings keyed by case id, sorted by IVD label (superior discs last).

    IVD label 0 (two rows in the release) has no level name and is skipped with a log message.
    """
    path = spider_dir(root) / "radiological_gradings.csv"
    if not path.is_file():
        raise FileNotFoundError(f"SPIDER radiological_gradings.csv not found at {path}")
    meta = load_spider_file_meta(root)
    vert_by_case: dict[str, int | None] = {}
    for (case, _seq), m in meta.items():
        vert_by_case.setdefault(case, m.num_vertebrae)
    out: dict[str, list[DiscGrading]] = {}
    for row in _read_csv(path):
        case = str(row.get("Patient") or "").strip()
        ivd = _int_or_none(row.get("IVD label"))
        if not case or ivd is None:
            continue
        if ivd < 1:
            log.info("case %s: IVD label %s has no level name; skipped", case, ivd)
            continue
        nv = vert_by_case.get(case)
        values = {field_name: _int_or_none(row.get(col)) for col, field_name in GRADING_COLUMNS.items()}
        out.setdefault(case, []).append(
            DiscGrading(
                case=case,
                ivd_label=ivd,
                level=ivd_level_name(ivd),
                transitional_suspect=nv is not None and nv != STANDARD_VERTEBRA_COUNT,
                **values,
            )
        )
    for case in out:
        out[case].sort(key=lambda g: g.ivd_label)
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def spider_subset_cases(subset: str, root: str | Path | None = None) -> list[str]:
    """Case ids of one subset (``training`` or ``validation``), numerically sorted. A case is in one subset only."""
    meta = load_spider_file_meta(root)
    cases = sorted({m.case for m in meta.values() if m.subset == subset}, key=int)
    return cases


def grading_summary(gradings: dict[str, list[DiscGrading]]) -> dict[str, Any]:
    """Counts of cases, discs and values per field (no identifiers), for reports."""
    rows = [g for lst in gradings.values() for g in lst]
    values: dict[str, dict[str, int]] = {}
    for f in GRADING_FIELDS:
        counts: dict[str, int] = {}
        for g in rows:
            v = g.value(f)
            key = "missing" if v is None else str(v)
            counts[key] = counts.get(key, 0) + 1
        values[f] = dict(sorted(counts.items()))
    return {
        "cases": len(gradings),
        "discs": len(rows),
        "transitional_suspect_discs": sum(1 for g in rows if g.transitional_suspect),
        "values": values,
    }
