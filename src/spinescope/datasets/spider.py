"""SPIDER lumbar spine MR dataset (CC-BY-4.0): per-study volumes, masks and gradings.

Layout under the dataset folder (``$SPINESCOPE_DATA/spider`` by default)::

    images/<case>_<seq>.mha   sagittal volumes, seq in {t1, t2, t2_SPACE}
    masks/<case>_<seq>.mha    label maps on the same grid as the image of the same name
    overview.csv              per-file acquisition metadata (only technical columns are used here)
    radiological_gradings.csv one row per intervertebral disc (IVD label), Pfirrman / Modic / ...

Label values observed in the masks: vertebrae 1..N, spinal canal 100, intervertebral discs 201..200+N.
Patient-identifying columns of overview.csv (sex, birth date, device serial) are never read.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import Stack
from ..io.volumes import read_label_volume, read_volume
from ..series import apply_display_ops, to_display_orientation
from ..study import Study, from_stacks
from .registry import SPIDER, data_dir

log = logging.getLogger(__name__)

#: Mask label values (observed on the public release; see the SPIDER paper for the definition).
CANAL_LABEL = 100
VERTEBRA_LABELS = range(1, 26)
DISC_LABEL_OFFSET = 200

_FILE_RE = re.compile(r"^(\d+)_(.+)\.mha$")
_GRADING_KEYS = {
    "IVD label": "ivd_label",
    "Modic": "modic",
    "UP endplate": "up_endplate",
    "LOW endplate": "low_endplate",
    "Spondylolisthesis": "spondylolisthesis",
    "Disc herniation": "disc_herniation",
    "Disc narrowing": "disc_narrowing",
    "Disc bulging": "disc_bulging",
    "Pfirrman grade": "pfirrman_grade",
}


@dataclass
class SpiderStudy:
    """One SPIDER patient: image and mask files per sequence, the disc gradings, and loaded data on demand."""

    case_id: str
    image_paths: dict[str, Path]
    mask_paths: dict[str, Path]
    gradings: list[dict[str, int | None]] = field(default_factory=list)
    subset: str = ""
    num_vertebrae: int | None = None
    num_discs: int | None = None
    volumes: dict[str, Stack] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def sequences(self) -> list[str]:
        return sorted(self.image_paths)

    def volume(self, sequence: str) -> Stack:
        """The image volume of ``sequence`` as a :class:`Stack` in display orientation (loaded once, cached).

        Same orientation as :func:`spinescope.load_study` gives for the file, so masks and studies built
        from it line up with the single-file path.
        """
        if sequence not in self.volumes:
            raw = read_volume(str(self._path(self.image_paths, sequence)))
            self.volumes[sequence] = to_display_orientation(raw)
        return self.volumes[sequence]

    def mask(self, sequence: str) -> np.ndarray:
        """Label map of ``sequence`` as ``[slice, row, col]``, on the same grid as :meth:`volume`."""
        if sequence not in self.masks:
            vol = self.volume(sequence)
            arr = read_label_volume(str(self._path(self.mask_paths, sequence)))
            arr = apply_display_ops(arr, list(vol.meta.get("display_transform", [])))
            if arr.shape != vol.shape:
                raise ValueError(f"mask and image grids differ for case {self.case_id} {sequence}")
            self.masks[sequence] = np.ascontiguousarray(arr)
        return self.masks[sequence]

    def to_study(self, sequences: Sequence[str] | None = None) -> Study:
        """Study of the given sequences (default: all), built the way :func:`spinescope.load_study` builds one.

        Stacks are classified and keyed (``sag_t2``, ``sag_t1``, ...) from their geometry and file-name
        contrast hint. The stacks are copies, so keys written here never leak into the cached volumes.
        """
        seqs = list(sequences) if sequences is not None else self.sequences
        stacks = []
        for s in seqs:
            v = self.volume(s)
            stacks.append(replace(v, key="", meta=dict(v.meta)))
        return from_stacks(stacks, reorient=False)

    def _path(self, table: dict[str, Path], sequence: str) -> Path:
        if sequence not in table:
            raise KeyError(f"case {self.case_id} has no sequence {sequence!r}; available: {self.sequences}")
        return table[sequence]


def spider_root(root: str | Path | None = None) -> Path:
    """Folder of the SPIDER release: ``root`` if given, else ``$SPINESCOPE_DATA/spider``."""
    return Path(root).expanduser() if root is not None else data_dir() / SPIDER.subdir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scan_files(folder: Path) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for p in sorted(folder.glob("*.mha")):
        m = _FILE_RE.match(p.name)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = p
    return out


def load_spider_studies(
    root: str | Path | None = None,
    *,
    cases: list[str] | None = None,
    sequences: list[str] | None = None,
    load: bool = False,
) -> list[SpiderStudy]:
    """Per-patient SPIDER studies, sorted by numeric case id.

    With ``load=False`` (default) only file paths and gradings are read, which is cheap. With
    ``load=True`` each study's volumes and masks for ``sequences`` (default: all) are read into memory.
    ``cases`` restricts the result to the given case ids.
    """
    base = spider_root(root)
    images_dir, masks_dir = base / "images", base / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"no SPIDER images/ and masks/ under {base}; download with "
            "spinescope.datasets.fetch('spider') or extract raw/images.zip and raw/masks.zip there"
        )
    images = _scan_files(images_dir)
    masks = _scan_files(masks_dir)
    gradings: dict[str, list[dict[str, int | None]]] = {}
    grading_csv = base / "radiological_gradings.csv"
    if grading_csv.is_file():
        for row in _read_csv(grading_csv):
            rec = {v: _as_int(row.get(k)) for k, v in _GRADING_KEYS.items() if k in row}
            gradings.setdefault(str(row["Patient"]).strip(), []).append(rec)
    overview: dict[str, dict[str, Any]] = {}
    overview_csv = base / "overview.csv"
    if overview_csv.is_file():
        for row in _read_csv(overview_csv):
            name = (row.get("new_file_name") or "").strip()
            m = _FILE_RE.match(name + ".mha")
            if m:  # technical fields only: never sex, birth date or device serial
                overview.setdefault(
                    m.group(1),
                    {
                        "subset": row.get("subset", ""),
                        "num_vertebrae": _as_int(row.get("num_vertebrae")),
                        "num_discs": _as_int(row.get("num_discs")),
                    },
                )
    wanted = sorted(images, key=lambda c: int(c))
    if cases is not None:
        wanted = [c for c in wanted if c in set(cases)]
    studies = []
    for case in wanted:
        img = images[case]
        msk = masks.get(case, {})
        missing = sorted(set(img) - set(msk))
        if missing:
            log.warning("case %s: no mask for sequence(s) %s", case, missing)
        info = overview.get(case, {})
        study = SpiderStudy(
            case_id=case,
            image_paths=img,
            mask_paths={k: v for k, v in msk.items() if k in img},
            gradings=sorted(gradings.get(case, []), key=lambda r: r.get("ivd_label") or 0),
            subset=str(info.get("subset", "")),
            num_vertebrae=info.get("num_vertebrae"),
            num_discs=info.get("num_discs"),
        )
        if load:
            for seq in sequences or study.sequences:
                study.volume(seq)
                study.mask(seq)
        studies.append(study)
    return studies


def load_spider_study(case_id: str, root: str | Path | None = None, *, load: bool = True) -> SpiderStudy:
    """One SPIDER study by case id (e.g. ``"1"``), loaded by default."""
    found = load_spider_studies(root, cases=[str(case_id)], load=load)
    if not found:
        raise KeyError(f"no SPIDER case {case_id!r} under {spider_root(root)}")
    return found[0]


def _as_int(v: str | None) -> int | None:
    try:
        return int(float(v)) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
