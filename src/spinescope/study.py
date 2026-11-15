"""A study: the stacks of one MRI examination, keyed by role, plus a PHI-free summary."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .geometry import Stack
from .series import assign_keys, to_display_orientation

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
    stacks: dict[str, Stack]
    info: dict[str, Any] = field(default_factory=dict)

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


def from_stacks(stacks: list[Stack], *, reorient: bool = True) -> Study:
    """Classify, key and (optionally) display-orient raw stacks into a :class:`Study`."""
    if reorient:
        stacks = [to_display_orientation(s) for s in stacks]
    keyed = assign_keys(stacks)
    return Study(stacks=keyed, info=_study_info(stacks))


def load_study(path: str | os.PathLike[str], *, reorient: bool = True, load_pixels: bool = True) -> Study:
    """Load a study from a DICOM directory (any layout) or a NIfTI/MetaImage volume file."""
    p = os.fspath(path)
    lower = p.lower()
    if os.path.isfile(p) and lower.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
        from .io.volumes import read_volume

        return from_stacks([read_volume(p)], reorient=reorient)
    from .io.dicom import read_stacks

    return from_stacks(read_stacks(p, load_pixels=load_pixels), reorient=reorient)
