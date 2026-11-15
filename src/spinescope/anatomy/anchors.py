"""Anchors: detected spinal anatomy (levels, midline, conus) that analysis modules build on.

Coordinates are LPS mm. Disc labels look like ``L4-L5``/``L5-S1``; vertebra labels ``L4``/``S1``.
Every anchor has a ``confidence`` in [0, 1] and free-text ``evidence``. Anchors are produced by
:func:`spinescope.anatomy.label_spine` (automatic) and never typed in by hand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..config import DEFAULT, Config
from .numbering import VERTEBRA_NAMES

#: 1.1 adds the optional geometry fields ``Vertebra.body``, ``Anchors.canal`` and ``Anchors.posterior_elements``
#: (lab15). Readers of 1.0 documents still load: the new fields default to None.
SCHEMA_VERSION = "1.1"

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


#: Provenance sources accepted by the correction methods. ``programmatic`` is code, not a review.
CORRECTION_SOURCES: tuple[str, ...] = ("programmatic", "ai_reader", "human")
#: Maximum length of the free-text ``who`` / ``why`` fields kept in the provenance record.
_TEXT_MAX = 200
_DISC_RE = re.compile(r"^([TLS]\d{1,2})-([TLS]\d{1,2})$")
#: Vertebra index in the counting convention: S1 = 0, L5 = 1, ... (L6 is not named by the counter).
_BOTTOM_UP = {name: i for i, name in enumerate(VERTEBRA_NAMES)}


def split_disc(label: str) -> tuple[str, str]:
    """(upper, lower) vertebra names of a disc label such as ``L4-L5``; ValueError if malformed."""
    m = _DISC_RE.match(label)
    known = m is not None and all(v in _BOTTOM_UP or v == "L6" for v in m.groups())
    if m is None or not known:
        raise ValueError(f"not a disc label: {label!r} (expected like L4-L5, levels T1..S1)")
    return m.group(1), m.group(2)


def _step(name: str, n: int) -> str:
    """Vertebra ``n`` levels cranial (n > 0) or caudal (n < 0) of ``name`` in the counting convention."""
    if name not in _BOTTOM_UP:
        raise ValueError(f"cannot shift vertebra {name!r}: it is outside the counted range S1..T1")
    j = _BOTTOM_UP[name] + n
    if not 0 <= j < len(VERTEBRA_NAMES):
        raise ValueError(f"shifting {name} by {n} levels leaves the counted range S1..T1")
    return VERTEBRA_NAMES[j]


def _shift_level(level: str, n: int) -> str:
    """Shift a disc label or a vertebra name."""
    if "-" in level:
        up, low = split_disc(level)
        return f"{_step(up, n)}-{_step(low, n)}"
    return _step(level, n)


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
    #: Optional body geometry (see :mod:`spinescope.anatomy.vertebrae`): 3-D centre, endplate lines, heights, AP
    #: depth, width. None until measured (learned labeller only).
    body: dict[str, Any] | None = None


@dataclass
class Anchors:
    discs: dict[str, Disc] = field(default_factory=dict)
    vertebrae: dict[str, Vertebra] = field(default_factory=dict)
    midline_x: float | None = None
    conus: dict[str, Any] | None = None  # {"z": float, "level": "L1", "confidence": float}
    #: Optional canal geometry: ``{"points": [[x, y, z], ...] superior first, "per_level": {label: {"z_mm",
    #: "ap_mm", "kind"}}, "confidence", "method", "evidence"}``. None until measured (learned labeller only).
    canal: dict[str, Any] | None = None
    #: Optional posterior-element approximations (lamina proxy, spinous tip) with their ``limits``. Unvalidated.
    posterior_elements: dict[str, Any] | None = None
    numbering: dict[str, Any] = field(
        default_factory=dict
    )  # {"method": ..., "confidence": ..., "notes": ...}
    method: str = ""
    source_stack: str = ""
    warnings: list[str] = field(default_factory=list)
    # Applicability record (spinescope.anatomy.ood): verdict, region, labelled, factor, coverage, cues. Additive.
    applicability: dict[str, Any] = field(default_factory=dict)
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

    # ------------------------------------------------------------ corrections
    # Every correction records provenance in ``numbering['corrections']`` (who, source, why, detail and
    # the disc labels after the change). Corrections never raise on missing evidence elsewhere; they do
    # raise ValueError/KeyError on a request that cannot be applied, before changing anything.

    def record_correction(
        self,
        action: str,
        *,
        who: str = "programmatic",
        source: str = "programmatic",
        why: str = "",
        detail: dict[str, Any] | None = None,
        confidence: float | None = None,
        warning: str | None = None,
    ) -> Anchors:
        """Append one provenance record; optionally set ``numbering['confidence']`` and add a warning."""
        if source not in CORRECTION_SOURCES:
            raise ValueError(f"correction source must be one of {', '.join(CORRECTION_SOURCES)}")
        entry: dict[str, Any] = {
            "action": action,
            "who": str(who).strip()[:_TEXT_MAX],
            "source": source,
            "why": str(why).strip()[:_TEXT_MAX],
            "detail": dict(detail or {}),
            "labels_after": self.disc_labels(),
        }
        self.numbering.setdefault("corrections", []).append(entry)
        if confidence is not None:
            self.numbering["confidence"] = round(float(np.clip(confidence, 0.0, 1.0)), 3)
        if warning:
            self.warnings.append(warning)
        return self

    def _rename(
        self,
        disc_map: Mapping[str, str],
        vert_map: Mapping[str, str],
        conus_map: Mapping[str, str] | None = None,
    ) -> None:
        """Rename discs and vertebrae atomically (all targets are checked first) and the conus level.

        ``conus_map`` gives the new conus level for a given old one; by default the disc and vertebra maps
        are used when the conus level is one of their keys.
        """
        for new in disc_map.values():
            split_disc(new)
        new_discs = [disc_map.get(k, k) for k in self.discs]
        new_verts = [vert_map.get(k, k) for k in self.vertebrae]
        if len(set(new_discs)) != len(new_discs):
            raise ValueError("relabelling would give two discs the same label")
        if len(set(new_verts)) != len(new_verts):
            raise ValueError("relabelling would give two vertebrae the same name")
        self.discs = {disc_map.get(k, k): v for k, v in self.discs.items()}
        self.vertebrae = {vert_map.get(k, k): v for k, v in self.vertebrae.items()}
        self._sync_geometry(disc_map, vert_map)
        if self.conus and self.conus.get("level"):
            level = str(self.conus["level"])
            if conus_map is not None and level in conus_map:
                self.conus["level"] = conus_map[level]
            elif level in disc_map:
                self.conus["level"] = disc_map[level]
            elif level in vert_map:
                self.conus["level"] = vert_map[level]

    def _sync_geometry(self, disc_map: Mapping[str, str], vert_map: Mapping[str, str]) -> None:
        """Carry the geometry fields (canal levels, posterior elements, body level names) across a relabelling.

        Entries whose level no longer exists are dropped, as the vertebra entries are by :meth:`shift_labels`.
        """
        names = set(self.discs) | set(self.vertebrae)

        def mapped(k: str) -> str:
            return disc_map.get(k, vert_map.get(k, k))

        if self.canal and isinstance(self.canal.get("per_level"), dict):
            self.canal["per_level"] = {
                mapped(k): v for k, v in self.canal["per_level"].items() if mapped(k) in names
            }
        if self.posterior_elements:
            for key in ("lamina_proxy", "spinous_tip"):
                if isinstance(self.posterior_elements.get(key), dict):
                    self.posterior_elements[key] = {
                        mapped(k): v for k, v in self.posterior_elements[key].items() if mapped(k) in names
                    }
        for name, v in self.vertebrae.items():
            if v.body is not None:
                v.body["level"] = name

    def shift_labels(
        self,
        offset: int,
        *,
        who: str = "programmatic",
        source: str = "programmatic",
        why: str = "",
        confidence: float | None = None,
    ) -> Anchors:
        """Rename every level by ``offset`` levels; positive is cranial (``+1``: L5-S1 becomes L4-L5).

        This is the sacralized-L5 correction (the reader says the lowest disc is one level higher). Disc
        names must stay in the counted range S1..T1 (ValueError otherwise, before any change). A vertebra
        entry or the conus level that would leave the range is dropped (with a warning) rather than
        renamed, since those centres are approximate and are not used to name discs.
        """
        n = int(offset)
        if n == 0:
            return self.record_correction(
                "shift", who=who, source=source, why=why, detail={"offset": 0}, confidence=confidence
            )
        disc_map = {lab: _shift_level(lab, n) for lab in self.discs}  # raises before any change
        vert_map: dict[str, str] = {}
        dropped: list[str] = []
        for name in self.vertebrae:
            try:
                vert_map[name] = _shift_level(name, n)
            except ValueError:
                dropped.append(name)
        conus_map: dict[str, str] = {}
        conus_lost = False
        if self.conus and self.conus.get("level"):
            level = str(self.conus["level"])
            try:
                conus_map[level] = _shift_level(level, n)
            except ValueError:
                conus_lost = True
        for name in dropped:
            del self.vertebrae[name]
        self._rename(disc_map, vert_map, conus_map)
        if conus_lost and self.conus is not None:
            self.conus["level"] = None
            self.warnings.append("conus level dropped by a shift outside the counted range")
        detail: dict[str, Any] = {"offset": n, "discs": disc_map}
        if dropped:
            detail["dropped_vertebrae"] = dropped
        return self.record_correction(
            "shift", who=who, source=source, why=why, detail=detail, confidence=confidence
        )

    def relabel(
        self,
        mapping: Mapping[str, str],
        *,
        who: str = "programmatic",
        source: str = "programmatic",
        why: str = "",
        confidence: float | None = None,
    ) -> Anchors:
        """Rename chosen discs (``L5-S1`` -> ``L5-L6``) or vertebrae (``S1`` -> ``L6``) by name.

        Keys that are neither a disc nor a vertebra raise KeyError; the whole mapping is validated first.
        """
        disc_map: dict[str, str] = {}
        vert_map: dict[str, str] = {}
        for old, new in mapping.items():
            if old in self.discs:
                disc_map[old] = str(new)
            elif old in self.vertebrae:
                vert_map[old] = str(new)
            else:
                raise KeyError(f"no disc or vertebra {old!r} to relabel")
        self._rename(disc_map, vert_map)
        return self.record_correction(
            "relabel",
            who=who,
            source=source,
            why=why,
            detail={"mapping": {str(k): str(v) for k, v in mapping.items()}},
            confidence=confidence,
        )

    def drop_disc(
        self,
        label: str,
        *,
        who: str = "programmatic",
        source: str = "programmatic",
        why: str = "",
        confidence: float | None = None,
    ) -> Anchors:
        """Remove a disc that is not a disc (false positive, or a thin interval). KeyError if absent."""
        if label not in self.discs:
            raise KeyError(f"no disc {label!r} to drop")
        del self.discs[label]
        self._sync_geometry({}, {})  # canal and posterior-element entries of the dropped disc go too
        return self.record_correction(
            "drop", who=who, source=source, why=why, detail={"label": label}, confidence=confidence
        )

    def add_disc(
        self,
        label: str,
        center: Any,
        *,
        normal: Any = None,
        anterior: Any = None,
        posterior: Any = None,
        height_mm: float | None = None,
        confidence: float | None = None,
        who: str = "programmatic",
        source: str = "programmatic",
        why: str = "",
        config: Config = DEFAULT,
    ) -> Anchors:
        """Add a disc the labeller missed. ``center`` is LPS mm; ``normal`` defaults to the nearest disc's.

        Height defaults to the median of the other discs (8 mm without any). Confidence defaults to the
        review confidence for ``ai_reader`` / ``human`` sources and 0.5 for programmatic additions.
        """
        split_disc(label)
        if label in self.discs:
            raise ValueError(f"disc {label!r} already exists")
        c = np.asarray(center, dtype=np.float64).reshape(-1)
        if c.size != 3 or not np.all(np.isfinite(c)):
            raise ValueError("center must be three finite LPS coordinates in mm")
        if normal is None:
            if not self.discs:
                raise ValueError("normal is required when there are no discs to copy it from")
            near = min(self.discs.values(), key=lambda d: float(np.linalg.norm(np.asarray(d.center) - c)))
            n = np.asarray(near.normal, dtype=np.float64)
        else:
            n = np.asarray(normal, dtype=np.float64).reshape(-1)
        if n.size != 3 or float(np.linalg.norm(n)) == 0.0:
            raise ValueError("normal must be a non-zero 3-vector")
        n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n  # superior-pointing, as in every other disc
        heights = [d.height_mm for d in self.discs.values() if d.height_mm]
        height = (
            float(height_mm) if height_mm is not None else (float(np.median(heights)) if heights else 8.0)
        )
        if confidence is None:
            confidence = {
                "ai_reader": config.review_confidence_ai,
                "human": config.review_confidence_human,
            }.get(source, 0.5)
        self.discs[label] = Disc(
            center=_v(c),
            normal=_v(n),
            anterior=None if anterior is None else _v(anterior),
            posterior=None if posterior is None else _v(posterior),
            height_mm=height,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            evidence=[f"added {label}: {why or 'no reason given'} (source {source})"],
        )
        return self.record_correction(
            "add",
            who=who,
            source=source,
            why=why,
            detail={"label": label, "center": _v(c)},
        )

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
