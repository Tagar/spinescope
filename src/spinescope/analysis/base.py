"""Analysis contract: every analysis returns a :class:`Result`, and a registry runs them together.

A :class:`Result` holds ``rows`` (one dict per measured item), ``flags`` (notable items, each with a
``confidence``), ``warnings`` (missing inputs, failed checks), ``figures`` (name -> RGB uint8 array)
and ``meta`` (method and configuration, PHI-free). :meth:`Result.save` writes CSV, JSON and PNG files.

Analyses take ``(study, anchors, config=None)``. They must return warnings rather than raise when a
sequence or anatomy is missing; :func:`run_all` additionally logs and contains unexpected exceptions.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..geometry import Stack
from ..study import Study

log = logging.getLogger(__name__)

AnalysisFn = Callable[[Study, Anchors, Config | None], "Result"]


@dataclass
class Result:
    """Output of one analysis."""

    name: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    figures: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (figures are listed by name; pixels are written by :meth:`save`)."""
        return {
            "name": self.name,
            "meta": _jsonable(self.meta),
            "rows": _jsonable(self.rows),
            "flags": _jsonable(self.flags),
            "warnings": list(self.warnings),
            "figures": sorted(self.figures),
        }

    def save(self, outdir: str | os.PathLike[str]) -> list[str]:
        """Write ``<name>.csv`` (rows), ``<name>.json`` (everything but pixels) and one PNG per figure
        (``<name>_<figure>.png``) into ``outdir``. Returns the written paths."""
        from PIL import Image

        out = os.fspath(outdir)
        os.makedirs(out, exist_ok=True)
        paths: list[str] = []
        csv_path = os.path.join(out, f"{self.name}.csv")
        fields: list[str] = []
        for row in self.rows:
            fields += [k for k in row if k not in fields]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in self.rows:
                w.writerow({k: _csv_cell(v) for k, v in row.items()})
        paths.append(csv_path)
        json_path = os.path.join(out, f"{self.name}.json")
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, allow_nan=False)
        paths.append(json_path)
        for fig, arr in self.figures.items():
            p = os.path.join(out, f"{self.name}_{fig}.png")
            Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(p)
            paths.append(p)
        return paths


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _csv_cell(v: Any) -> Any:
    v = _jsonable(v)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return v


def sagittal_stacks(study: Study) -> dict[str, Stack]:
    """Non-localizer, non-slab sagittal stacks by contrast (first match, most slices)."""
    out: dict[str, Stack] = {}
    for contrast in ("t2", "stir", "pd", "t1"):
        hits = study.find("sag", contrast, slabs=False)
        if hits:
            out[contrast] = hits[0]
    return out


# ------------------------------------------------------------------- registry
_REGISTRY: dict[str, AnalysisFn] = {}
_BUILTIN_MODULES = ("discs", "endplates", "alignment", "canal")


def register(name: str) -> Callable[[AnalysisFn], AnalysisFn]:
    """Decorator: register an analysis ``fn(study, anchors, config) -> Result`` under ``name``."""

    def deco(fn: AnalysisFn) -> AnalysisFn:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"analysis {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def _load_builtin() -> None:
    import importlib

    for mod in _BUILTIN_MODULES:
        importlib.import_module(f"{__package__}.{mod}")


def available() -> list[str]:
    """Names of all registered analyses (built-in ones included)."""
    _load_builtin()
    return sorted(_REGISTRY)


def run(name: str, study: Study, anchors: Anchors, config: Config | None = None) -> Result:
    """Run one registered analysis."""
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown analysis {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](study, anchors, config or DEFAULT)


def run_all(
    study: Study,
    anchors: Anchors,
    config: Config | None = None,
    names: list[str] | None = None,
) -> dict[str, Result]:
    """Run several (default: all) registered analyses. A crashing analysis yields a Result with a warning."""
    _load_builtin()
    out: dict[str, Result] = {}
    for name in names or sorted(_REGISTRY):
        try:
            out[name] = run(name, study, anchors, config)
        except Exception as exc:  # contain one analysis so the others still run
            log.exception("analysis %s failed", name)
            out[name] = Result(name=name, warnings=[f"analysis failed: {type(exc).__name__}: {exc}"])
    return out
