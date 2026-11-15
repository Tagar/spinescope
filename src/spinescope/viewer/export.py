"""Turn a :class:`~spinescope.study.Study` (plus optional anchors) into the viewer's JSON payload.

Only technical content leaves this module: series geometry and acquisition numbers, voxel data
(quantised and deflate-compressed), disc labels with their centres from the anchors, and the
planned reformats. DICOM attributes never pass through except those on ``STUDY_INFO_TAGS`` and
the numeric acquisition parameters listed in ``ACQ_KEYS``; ``meta`` keys starting with ``_``
(source file paths) are never read.
"""

from __future__ import annotations

import base64
import json
import logging
import zlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..geometry import Stack, unit
from ..study import Study
from .reformat import CONTRAST_RANK, Reformat, plan_reformats

log = logging.getLogger(__name__)

FORMAT = "spinescope-viewer-1"
DISCLAIMER = "Research and education only. Not for diagnosis."
#: Scanner-level fields shown in the viewer (technical, not identifying).
INFO_KEYS = ("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength")
#: Numeric acquisition parameters per series.
ACQ_KEYS = ("RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "SliceThickness")
PLANE_ORDER = {"sag": 0, "cor": 1, "ax": 2}
#: Who the page records as the author of reviewer corrections (provenance ``who``).
REVIEWER = "viewer user"


def _r(x: Any, nd: int = 9) -> list[float]:
    return [round(float(v), nd) for v in np.ravel(np.asarray(x, dtype=np.float64))]


def encode_volume(vol: NDArray[np.float32]) -> dict[str, Any]:
    """Voxels -> ``{vmin, scale, enc, data}``: uint16 codes, byte-shuffled, zlib (deflate) and base64.

    Exact when the voxels are integers in a 16-bit range (scale 1); otherwise quantised to 16 bits
    over the stack's range. The browser decodes with ``DecompressionStream('deflate')``.
    """
    v = np.asarray(vol, dtype=np.float64)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    exact = bool(np.all(v == np.round(v))) and vmax - vmin <= 65535
    scale = 1.0 if exact else (vmax - vmin) / 65535.0 or 1.0
    q = np.clip(np.round((v - vmin) / scale), 0, 65535).astype("<u2")
    b = q.reshape(-1).view(np.uint8).reshape(-1, 2)
    shuffled = np.concatenate([b[:, 0], b[:, 1]]).tobytes()  # all low bytes, then all high bytes
    comp = zlib.compress(shuffled, 9)
    return {
        "n": int(v.shape[0]),
        "rows": int(v.shape[1]),
        "cols": int(v.shape[2]),
        "vmin": vmin,
        "scale": scale,
        "enc": "deflate;shuffle-u16le",
        "data": base64.b64encode(comp).decode("ascii"),
    }


def _acq(st: Stack) -> dict[str, float]:
    d = st.meta.get("dicom", {})
    out: dict[str, float] = {}
    for k in ACQ_KEYS:
        if k in d:
            try:
                out[k] = float(d[k])
            except (TypeError, ValueError):
                continue
    return out


def _geometry(
    *,
    n: int,
    rows: int,
    cols: int,
    row_mm: float,
    col_mm: float,
    thickness: float,
    row_dir: Any,
    col_dir: Any,
    normal: Any,
    ipp: Any,
    positions: Any,
) -> dict[str, Any]:
    return {
        "n": int(n),
        "rows": int(rows),
        "cols": int(cols),
        "row_mm": float(row_mm),
        "col_mm": float(col_mm),
        "thickness": float(thickness),
        "row_dir": _r(row_dir),
        "col_dir": _r(col_dir),
        "normal": _r(normal),
        "ipp": [_r(p) for p in np.asarray(ipp, dtype=np.float64)],
        "positions": _r(positions),
    }


def stack_record(key: str, st: Stack) -> dict[str, Any]:
    role = st.meta.get("role", {})
    rec: dict[str, Any] = {
        "key": key,
        "label": st.description or key,
        "plane": st.plane,
        "contrast": str(role.get("contrast", "")),
        "localizer": bool(role.get("localizer", False)),
        "reformat": None,
        "acq": _acq(st),
        "enc": encode_volume(st.vol),
    }
    rec.update(
        _geometry(
            n=st.n,
            rows=st.vol.shape[1],
            cols=st.vol.shape[2],
            row_mm=st.row_mm,
            col_mm=st.col_mm,
            thickness=st.thickness,
            row_dir=st.row_dir,
            col_dir=st.col_dir,
            normal=st.normal,
            ipp=st.ipp,
            positions=st.positions,
        )
    )
    return rec


def reformat_record(ref: Reformat) -> dict[str, Any]:
    """Geometry only: the browser resamples the source's voxels (see ``reformat.py``)."""
    rec: dict[str, Any] = {
        "key": ref.key,
        "label": ref.description,
        "plane": ref.plane,
        "contrast": ref.contrast,
        "localizer": False,
        "reformat": {
            "source": ref.source_key,
            "method": "trilinear",
            "px_mm": ref.px_mm,
            "coverage": round(ref.coverage, 4),
        },
        "acq": {},
        "enc": None,
    }
    rec.update(
        _geometry(
            n=ref.n,
            rows=ref.rows,
            cols=ref.cols,
            row_mm=ref.px_mm,
            col_mm=ref.px_mm,
            thickness=ref.px_mm,
            row_dir=ref.row_dir,
            col_dir=ref.col_dir,
            normal=ref.normal,
            ipp=ref.ipp,
            positions=ref.positions,
        )
    )
    return rec


def label_policy(config: Config = DEFAULT) -> dict[str, Any]:
    """Thresholds and provenance settings the page uses for labels (mirrors Config, names in camelCase)."""
    return {
        "who": REVIEWER,
        "confHigh": float(config.viewer_conf_high),
        "confLow": float(config.viewer_conf_low),
        "confHuman": float(config.review_confidence_human),
        "confAi": float(config.review_confidence_ai),
    }


def anchors_record(anchors: Anchors | None) -> dict[str, Any] | None:
    """``Anchors.to_dict()`` as plain JSON types (numpy scalars converted), or None without anchors."""
    if anchors is None:
        return None
    return json.loads(json.dumps(anchors.to_dict(), default=float))


def disc_records(anchors: Anchors | None) -> list[dict[str, Any]]:
    """Disc labels, centres (LPS mm), superior-pointing unit normals and confidences, superior first."""
    if anchors is None:
        return []
    out = []
    for lab in anchors.disc_labels():
        d = anchors.discs[lab]
        out.append(
            {
                "label": lab,
                "center": _r(d.center),
                "normal": _r(unit(d.normal)),
                "confidence": round(float(d.confidence), 3),
            }
        )
    return out


def _pick_default_panes(study: Study, refs: list[Reformat], keys: list[str]) -> list[str]:
    """Four pane series: sagittal, axial, coronal (real or reformat) first, then the rest in order."""
    ref_by_plane = {r.plane: r.key for r in refs}
    chosen: list[str] = []
    for plane in ("sag", "ax", "cor"):
        if plane in ref_by_plane:
            chosen.append(ref_by_plane[plane])
            continue
        best = study.best(plane)
        if best is not None:
            chosen.append(best.key)
    for k in keys:
        if len(chosen) >= 4:
            break
        if k not in chosen:
            chosen.append(k)
    i = 0
    while len(chosen) < 4:  # fewer than four series: repeat, a pane always shows something
        chosen.append(keys[i % len(keys)])
        i += 1
    return chosen[:4]


def build_payload(
    study: Study,
    anchors: Anchors | None = None,
    *,
    title: str,
    config: Config = DEFAULT,
) -> dict[str, Any]:
    """The complete viewer payload (JSON-able). Raises ``ValueError`` for a study with no stacks."""
    if len(study) == 0:
        raise ValueError("study has no stacks to show")
    refs = plan_reformats(study, config=config)
    recs: dict[str, dict[str, Any]] = {}
    for key, st in study.stacks.items():
        recs[key] = stack_record(key, st)
    for ref in refs:
        if ref.key in recs:
            log.warning("viewer: reformat key %s clashes with a study series; skipped", ref.key)
            continue
        recs[ref.key] = reformat_record(ref)

    def order(rec: dict[str, Any]) -> tuple[Any, ...]:
        return (
            rec["localizer"],
            rec["reformat"] is not None,
            PLANE_ORDER.get(rec["plane"], 9),
            CONTRAST_RANK.get(rec["contrast"], 9),
            rec["key"],
        )

    series = sorted(recs.values(), key=order)
    keys = [s["key"] for s in series]
    info = {k: str(study.info[k]) for k in INFO_KEYS if k in study.info}
    return {
        "format": FORMAT,
        "title": title,
        "disclaimer": DISCLAIMER,
        "info": info,
        "layout": 3,
        "panes": _pick_default_panes(study, refs, keys),
        "series": series,
        "discs": disc_records(anchors),
        "anchors": anchors_record(anchors),
        "policy": label_policy(config),
    }
