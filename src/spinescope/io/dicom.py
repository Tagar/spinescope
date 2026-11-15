"""Load MR DICOM studies from any directory layout into geometry-aware stacks.

Handles: arbitrary folder structures (DICOMDIR is not required), several series per study,
series whose slices have different orientations (e.g. oblique slabs angled to each disc, or
3-plane localizers), multi-echo / Dixon series (split by echo time and image type), and
multi-frame Enhanced MR objects (one stack per orientation group of frames).

Privacy: only an allowlist of technical acquisition attributes is kept (see ``KEEP_TAGS``).
Patient name, IDs, dates, institution and physician names never enter the returned objects.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

from ..geometry import Stack

log = logging.getLogger(__name__)

#: Attributes copied into ``Stack.meta['dicom']``. Technical only, never identifying.
KEEP_TAGS = (
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "FlipAngle",
    "EchoTrainLength",
    "ImageType",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelBandwidth",
    "ReceiveCoilName",
    "BodyPartExamined",
    "PatientPosition",
    "ContrastBolusAgent",
    "SeriesNumber",
    "AcquisitionNumber",
    "PhotometricInterpretation",
    "BitsStored",
    "Rows",
    "Columns",
    "DiffusionBValue",
)

#: SOP classes that are never images we can place in space.
_NON_IMAGE_MODALITIES = {"SR", "PR", "KO", "REG", "SEG", "RTSTRUCT", "DOC", "OT"}


@dataclass
class SliceRef:
    """One image plane: a file (and frame index for multi-frame objects) plus geometry."""

    path: str
    frame: int | None
    ipp: np.ndarray
    iop: np.ndarray
    pixel_spacing: tuple[float, float]
    thickness: float
    slope: float
    intercept: float
    group_key: tuple[Any, ...]
    header: dict[str, Any] = field(default_factory=dict)


def _is_dicom(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
    except OSError:
        return False
    # Some exports omit the preamble; let pydicom decide (cheaply) for small files.
    try:
        pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SOPClassUID"])
        return True
    except (InvalidDicomError, OSError, ValueError, TypeError, AttributeError, KeyError):
        return False


def iter_dicom_files(root: str | os.PathLike[str]) -> Iterator[str]:
    """Yield DICOM file paths under ``root`` (a file or directory), skipping DICOMDIR."""
    root = Path(root)
    if root.is_file():
        if _is_dicom(str(root)):
            yield str(root)
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and not d.endswith(".app"))
        for name in sorted(filenames):
            if name.startswith(".") or name.upper() == "DICOMDIR":
                continue
            p = os.path.join(dirpath, name)
            if _is_dicom(p):
                yield p


def _get(ds: Dataset, name: str, default: Any = None) -> Any:
    v = getattr(ds, name, default)
    return default if v is None or v == "" else v


def _jsonable(v: Any) -> Any:
    if isinstance(v, (list, tuple)) or (hasattr(v, "__iter__") and not isinstance(v, (str, bytes))):
        return [_jsonable(x) for x in v]
    if isinstance(v, (int, float, str)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _header(ds: Dataset) -> dict[str, Any]:
    out = {}
    for name in KEEP_TAGS:
        v = _get(ds, name)
        if v is not None:
            out[name] = _jsonable(v)
    return out


def _frame_geometry(ds: Dataset, i: int) -> tuple[Any, Any, Any, Any]:
    """(ipp, iop, pixel spacing, thickness) for frame ``i`` of an Enhanced (multi-frame) object."""

    def find(seq_name: str, item: str) -> Any:
        for groups in (
            ds.PerFrameFunctionalGroupsSequence[i] if "PerFrameFunctionalGroupsSequence" in ds else None,
            ds.SharedFunctionalGroupsSequence[0] if "SharedFunctionalGroupsSequence" in ds else None,
        ):
            if groups is not None and seq_name in groups:
                node = groups[seq_name].value[0]
                if item in node:
                    return node[item].value
        return None

    ipp = find("PlanePositionSequence", "ImagePositionPatient")
    iop = find("PlaneOrientationSequence", "ImageOrientationPatient")
    ps = find("PixelMeasuresSequence", "PixelSpacing")
    thk = find("PixelMeasuresSequence", "SliceThickness")
    return ipp, iop, ps, thk


def _refs_from_file(path: str) -> list[SliceRef]:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except (InvalidDicomError, OSError, ValueError, TypeError, AttributeError, KeyError) as e:
        log.debug("skip %s: %s", path, e)
        return []
    if str(_get(ds, "Modality", "")) in _NON_IMAGE_MODALITIES:
        return []
    header = _header(ds)
    series_uid = str(_get(ds, "SeriesInstanceUID", path))
    image_type = tuple(str(x).upper() for x in (_get(ds, "ImageType", []) or []))
    echo = _get(ds, "EchoTime")
    echo = round(float(echo), 2) if echo is not None else None
    slope = float(_get(ds, "RescaleSlope", 1) or 1)
    intercept = float(_get(ds, "RescaleIntercept", 0) or 0)
    nframes = int(_get(ds, "NumberOfFrames", 1) or 1)
    refs = []
    frames = range(nframes) if nframes > 1 else [None]
    for fr in frames:
        if fr is None:
            ipp, iop, ps, thk = (
                _get(ds, "ImagePositionPatient"),
                _get(ds, "ImageOrientationPatient"),
                _get(ds, "PixelSpacing"),
                _get(ds, "SliceThickness"),
            )
        else:
            ipp, iop, ps, thk = _frame_geometry(ds, fr)
        if ipp is None or iop is None or ps is None:
            if fr in (None, 0):
                log.debug("skip %s: no image plane geometry", path)
            continue
        iop_a = np.array([float(x) for x in iop], dtype=np.float64)
        okey = tuple(np.round(iop_a, 2))
        refs.append(
            SliceRef(
                path=path,
                frame=fr,
                ipp=np.array([float(x) for x in ipp], dtype=np.float64),
                iop=iop_a,
                pixel_spacing=(float(ps[0]), float(ps[1])),
                thickness=float(thk or 0.0),
                slope=slope,
                intercept=intercept,
                group_key=(series_uid, okey, echo, image_type),
                header=header,
            )
        )
    return refs


def _read_pixels(ref: SliceRef, cache: dict[str, np.ndarray]) -> np.ndarray:
    if ref.path not in cache:
        ds = pydicom.dcmread(ref.path)
        arr = ds.pixel_array
        if str(_get(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        cache.clear()  # keep at most one multi-frame file in memory
        cache[ref.path] = np.asarray(arr)
    arr = cache[ref.path]
    img = arr[ref.frame] if ref.frame is not None else arr
    if img.ndim == 3:  # colour image: keep luminance
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32) * ref.slope + ref.intercept


def _split_duplicates(refs: list[SliceRef]) -> list[list[SliceRef]]:
    """Split a group whose slices share positions (e.g. repeated acquisitions) into runs."""
    normal = np.cross(refs[0].iop[:3], refs[0].iop[3:])
    pos = np.array([r.ipp @ normal for r in refs])
    rounded = np.round(pos, 2)
    if len(set(rounded)) == len(rounded):
        return [refs]
    runs: dict[int, list[SliceRef]] = defaultdict(list)
    seen: dict[float, int] = defaultdict(int)
    for r, p in zip(refs, rounded, strict=False):
        runs[seen[p]].append(r)
        seen[p] += 1
    return [runs[i] for i in sorted(runs)]


def build_stacks(refs: Iterable[SliceRef], *, load_pixels: bool = True) -> list[Stack]:
    """Group slice references into single-orientation stacks (keys are provisional)."""
    groups: dict[tuple[Any, ...], list[SliceRef]] = defaultdict(list)
    for r in refs:
        groups[r.group_key].append(r)
    stacks: list[Stack] = []
    cache: dict[str, np.ndarray] = {}
    for gkey, items in groups.items():
        for run in _split_duplicates(items):
            shapes = {(r.pixel_spacing,) for r in run}
            if len(shapes) > 1:
                log.warning("series %s has mixed pixel spacing; using the first", gkey[0])
            ref0 = run[0]
            if load_pixels:
                imgs = [_read_pixels(r, cache) for r in run]
                hw = {i.shape for i in imgs}
                if len(hw) > 1:
                    log.warning("series %s has mixed image sizes; skipping", gkey[0])
                    continue
                vol = np.stack(imgs)
            else:
                rows = int(ref0.header.get("Rows", 1))
                cols = int(ref0.header.get("Columns", 1))
                vol = np.zeros((len(run), rows, cols), dtype=np.float32)
            meta = {
                "dicom": dict(ref0.header),
                "series_uid_hash": _short_hash(str(gkey[0])),
                "_sources": [(r.path, r.frame) for r in run],
            }
            stacks.append(
                Stack(
                    key="",
                    vol=vol,
                    ipp=np.stack([r.ipp for r in run]),
                    row_dir=ref0.iop[:3],
                    col_dir=ref0.iop[3:],
                    row_mm=ref0.pixel_spacing[0],
                    col_mm=ref0.pixel_spacing[1],
                    thickness=ref0.thickness,
                    description=str(ref0.header.get("SeriesDescription", "")),
                    series_number=int(ref0.header["SeriesNumber"]) if "SeriesNumber" in ref0.header else None,
                    meta=meta,
                )
            )
    return stacks


def _short_hash(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()[:12]


def read_stacks(root: str | os.PathLike[str], *, load_pixels: bool = True) -> list[Stack]:
    """Scan ``root`` and return one :class:`Stack` per (series, orientation, echo, image type)."""
    refs: list[SliceRef] = []
    for path in iter_dicom_files(root):
        refs.extend(_refs_from_file(path))
    if not refs:
        raise FileNotFoundError(f"no DICOM images with plane geometry found under {root}")
    return build_stacks(refs, load_pixels=load_pixels)
