"""Anchors: detected spinal anatomy (levels, midline, conus) that analysis modules build on.

Coordinates are LPS mm. Disc labels look like ``L4-L5``/``L5-S1``; vertebra labels ``L4``/``S1``.
Every anchor has a ``confidence`` in [0, 1] and free-text ``evidence``. Anchors are produced by
:func:`spinescope.anatomy.label_spine` (automatic) and never typed in by hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"

#: Canonical order of the labels we may emit, superior -> inferior.
VERTEBRA_ORDER = (
    "T8",
    "T9",
    "T10",
    "T11",
    "T12",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "S1",
    "S2",
)


def disc_label(upper: str, lower: str) -> str:
    return f"{upper}-{lower}"


def _v(x: Any) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=np.float64).ravel()]


@dataclass
class Disc:
    center: list[float]
    normal: list[float]  # unit, pointing superior
    anterior: list[float] | None = None  # anterior margin at mid-height
    posterior: list[float] | None = None  # posterior margin at mid-height
    height_mm: float | None = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    @property
    def ap_dir(self) -> np.ndarray:
        """Unit anterior->posterior direction in the disc plane."""
        if self.anterior is not None and self.posterior is not None:
            d = np.asarray(self.posterior) - np.asarray(self.anterior)
            n = np.asarray(self.normal)
            d = d - (d @ n) * n
            return d / np.linalg.norm(d)
        n = np.asarray(self.normal)
        d = np.array([0.0, 1.0, 0.0]) - n[1] * n
        return d / np.linalg.norm(d)


@dataclass
class Vertebra:
    center: list[float]
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class Anchors:
    discs: dict[str, Disc] = field(default_factory=dict)
    vertebrae: dict[str, Vertebra] = field(default_factory=dict)
    midline_x: float | None = None
    conus: dict[str, Any] | None = None  # {"z": float, "level": "L1", "confidence": float}
    numbering: dict[str, Any] = field(
        default_factory=dict
    )  # {"method": ..., "confidence": ..., "notes": ...}
    method: str = ""
    source_stack: str = ""
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    # ---------------------------------------------------------------- queries
    def disc_labels(self) -> list[str]:
        """Disc labels ordered superior -> inferior."""
        return sorted(self.discs, key=lambda d: -self.discs[d].center[2])

    def disc(self, label: str) -> Disc:
        return self.discs[label]

    def nearest_disc(self, xyz: Any) -> tuple[str, float]:
        """(label, signed distance mm along that disc's normal; + = superior to the disc)."""
        p = np.asarray(xyz, dtype=np.float64)
        best: tuple[str, float] | None = None
        for lab, d in self.discs.items():
            s = float((p - np.asarray(d.center)) @ np.asarray(d.normal))
            if best is None or abs(s) < abs(best[1]):
                best = (lab, s)
        if best is None:
            raise ValueError("no discs in anchors")
        return best

    def level_of(self, xyz: Any) -> str:
        """Disc label if within half a disc height of one, else the vertebra containing the point."""
        lab, s = self.nearest_disc(xyz)
        h = self.discs[lab].height_mm or 8.0
        if abs(s) <= h / 2:
            return lab
        upper, lower = lab.split("-")
        return upper if s > 0 else lower

    # ------------------------------------------------------------- serialize
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | None = None, indent: int = 1) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Anchors:
        d = dict(d)
        discs = {k: Disc(**v) for k, v in d.pop("discs", {}).items()}
        verts = {k: Vertebra(**v) for k, v in d.pop("vertebrae", {}).items()}
        return cls(discs=discs, vertebrae=verts, **d)

    @classmethod
    def from_json(cls, path_or_text: str) -> Anchors:
        text = path_or_text
        if not path_or_text.lstrip().startswith("{"):
            with open(path_or_text) as f:
                text = f.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_truth(cls, truth: dict[str, Any], method: str = "ground-truth") -> Anchors:
        """Build anchors from a ground-truth dict (e.g. ``SpinePhantom.truth``) for tests."""
        discs = {
            k: Disc(
                center=_v(v["center"]),
                normal=_v(v["normal"]),
                anterior=_v(v["anterior"]) if "anterior" in v else None,
                posterior=_v(v["posterior"]) if "posterior" in v else None,
                height_mm=v.get("height_mm"),
                confidence=1.0,
                evidence=[method],
            )
            for k, v in truth["discs"].items()
        }
        verts = {
            k: Vertebra(center=_v(c), confidence=1.0, evidence=[method])
            for k, c in truth["vertebrae"].items()
        }
        conus = None
        if "conus_z" in truth:
            conus = {"z": float(truth["conus_z"]), "level": truth.get("conus_level"), "confidence": 1.0}
        return cls(
            discs=discs,
            vertebrae=verts,
            midline_x=truth.get("midline_x"),
            conus=conus,
            numbering={"method": method, "confidence": 1.0},
            method=method,
        )
