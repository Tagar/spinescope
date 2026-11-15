"""A study: the stacks of one MRI examination, keyed by role, plus a PHI-free summary."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DEFAULT
from .geometry import Stack
from .io.dicom import (
    StudyInfo,
    dicom_scan_summary,
    load_stack_pixels,
    read_dicom,
    slice_geometry,
)
from .io.dicom import list_studies as _list_studies
from .series import assign_keys, to_display_orientation

log = logging.getLogger(__name__)

#: Study-level attributes safe to report (technical, not identifying).
STUDY_INFO_TAGS = (
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "BodyPartExamined",
    "PatientPosition",
)


@dataclass(eq=False)
class Study:
    """Stacks keyed by role, plus technical info and a PHI-free report of what loading found.

    ``report`` holds counts of ignored files, the studies found in the folder, decoder problems and
    warnings (never paths or identifiers).
    """

    stacks: dict[str, Stack]
    info: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Stack:
        return self.stacks[key]

    def __contains__(self, key: object) -> bool:
        return key in self.stacks

    def __iter__(self) -> Iterator[str]:
        return iter(self.stacks)

    def __len__(self) -> int:
        return len(self.stacks)

    def keys(self) -> list[str]:
        return list(self.stacks)

    def find(
        self,
        plane: str | None = None,
        contrast: str | None = None,
        *,
        localizer: bool = False,
        slabs: bool | None = None,
        fat_suppressed: bool | None = None,
        post_contrast: bool | None = False,
    ) -> list[Stack]:
        """Stacks matching the filters, most slices first."""
        out = []
        for st in self.stacks.values():
            role = st.meta.get("role", {})
            if plane and st.plane != plane:
                continue
            if contrast and role.get("contrast") != contrast:
                continue
            if bool(role.get("localizer")) != localizer:
                continue
            if slabs is not None and (role.get("slab_index") is not None) != slabs:
                continue
            if (
                fat_suppressed is not None
                and role.get("contrast") != "stir"
                and bool(role.get("fat_suppressed")) != fat_suppressed
            ):
                continue
            if post_contrast is not None and bool(role.get("post_contrast")) != post_contrast:
                continue
            out.append(st)
        return sorted(out, key=lambda s: -s.n)

    def best(self, plane: str, prefer: tuple[str, ...] = ("t2", "t1", "stir", "pd")) -> Stack | None:
        """The most useful non-localizer stack in ``plane``, by contrast preference."""
        for c in prefer:
            hits = self.find(plane, c, slabs=False) or self.find(plane, c)
            if hits:
                return hits[0]
        hits = self.find(plane)
        return hits[0] if hits else None

    def slabs(self, plane: str = "ax") -> list[Stack]:
        """Angled slabs (each from a multi-slab series), superior first."""
        st = [
            s for s in self.stacks.values() if s.plane == plane and s.meta.get("role", {}).get("slab_index")
        ]
        return sorted(st, key=lambda s: -float(s.ipp[:, 2].mean()))

    def summary(self) -> str:
        lines = [f"Study: {', '.join(f'{k}={v}' for k, v in self.info.items())}"]
        lines += ["  " + s.summary() for s in self.stacks.values()]
        return "\n".join(lines)


def _study_info(stacks: list[Stack]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for st in stacks:
        d = st.meta.get("dicom", {})
        for k in STUDY_INFO_TAGS:
            if k in d and k not in info:
                info[k] = d[k]
    return info


def _finish(
    stacks: list[Stack],
    *,
    reorient: bool,
    keys: Iterable[str] | None,
    load_pixels: bool,
    report: dict[str, Any],
) -> Study:
    """Display-orient, key, check geometry, optionally load only ``keys``, and wrap into a Study."""
    if reorient:
        stacks = [to_display_orientation(s) for s in stacks]
    keyed = assign_keys(stacks)
    if keys is not None:
        wanted = list(keys)
        unknown = [k for k in wanted if k not in keyed]
        if unknown:
            raise KeyError(f"unknown stack key(s) {unknown}; available: {sorted(keyed)}")
        keyed = {k: keyed[k] for k in wanted}
    for key, st in keyed.items():
        st.meta.setdefault("geometry", slice_geometry(st.positions))
        geo = st.meta["geometry"]
        if not geo.get("uniform", True):
            msg = f"stack {key}: slice spacing is not uniform"
            log.warning(msg)
            report.setdefault("warnings", []).append(msg)
        if geo.get("gap_count"):
            msg = f"stack {key}: {geo['gap_count']} gap(s) in slice positions (~{geo['estimated_missing_slices']} missing)"
            log.warning(msg)
            report.setdefault("warnings", []).append(msg)
        if load_pixels:
            load_stack_pixels(st)
            _pixel_qa(key, st, report)
    return Study(stacks=keyed, info=_study_info(list(keyed.values())), report=report)


def _pixel_qa(key: str, st: Stack, report: dict[str, Any]) -> None:
    """Warnings about pixel content that degrades labelling: non-finite values, constant and clipped slices.

    Warnings only (the data are kept): the labeller's applicability check decides whether to label.
    """
    msgs: list[str] = []
    n_bad = int(st.meta.get("nonfinite_voxels", 0) or 0)
    if n_bad:
        msgs.append(
            f"stack {key}: {n_bad} non-finite pixel value(s) (NaN or inf) replaced by the nearest finite extreme"
        )
    vol = st.vol
    if vol.size:
        const = [k for k in range(st.n) if float(np.ptp(vol[k])) == 0.0]
        if const:
            msgs.append(
                f"stack {key}: {len(const)} of {st.n} slice(s) are constant (every voxel equal: empty, "
                "zero-filled or saturated)"
            )
        hi = float(np.max(vol))
        clipped = [
            k
            for k in range(st.n)
            if float(np.ptp(vol[k])) > 0.0
            and np.count_nonzero(vol[k] >= hi) > DEFAULT.saturated_slice_frac * vol[k].size
        ]
        if clipped:
            msgs.append(f"stack {key}: {len(clipped)} slice(s) look saturated (most voxels at the maximum)")
    for msg in msgs:
        log.warning(msg)
        report.setdefault("warnings", []).append(msg)


def from_stacks(stacks: list[Stack], *, reorient: bool = True) -> Study:
    """Classify, key and (optionally) display-orient raw stacks into a :class:`Study`."""
    if reorient:
        stacks = [to_display_orientation(s) for s in stacks]
    keyed = assign_keys(stacks)
    return Study(stacks=keyed, info=_study_info(stacks))


def list_studies(path: str | os.PathLike[str]) -> list[StudyInfo]:
    """Studies in a DICOM folder (by StudyInstanceUID), largest first; header-only and PHI-free.

    Returns :class:`spinescope.io.dicom.StudyInfo` items; pass ``info.index`` or ``info.study_id``
    as ``study=`` to :func:`load_study`.
    """
    return _list_studies(path)


def load_study(
    path: str | os.PathLike[str],
    *,
    study: int | str | None = None,
    keys: Iterable[str] | None = None,
    reorient: bool = True,
    load_pixels: bool = True,
) -> Study:
    """Load a study from a DICOM directory (any layout) or a NIfTI/MetaImage/NRRD volume file.

    ``study`` picks one study when a folder holds several (index or ``study_id`` from
    :func:`list_studies`); the default is the study with the most images, with a warning.
    ``keys`` loads only the named stacks (pixels are read only for those); the other series are
    still classified from headers but dropped. ``load_pixels=False`` returns header-only stacks with
    zero placeholder voxels.
    """
    p = os.fspath(path)
    lower = p.lower()
    if os.path.isfile(p) and lower.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
        from .io.volumes import read_volume

        stack = read_volume(p)
        return _finish([stack], reorient=reorient, keys=keys, load_pixels=load_pixels, report={})

    read = read_dicom(p, study=study)
    report: dict[str, Any] = {
        "scan": dicom_scan_summary(read.scan),
        "studies": [s.as_dict() for s in read.studies],
        "selected_study": read.selected.as_dict(),
        "warnings": list(read.warnings),
    }
    return _finish(read.stacks, reorient=reorient, keys=keys, load_pixels=load_pixels, report=report)
