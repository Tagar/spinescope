"""Load MR DICOM studies from any directory layout into geometry-aware stacks.

Handles: arbitrary folder structures (DICOMDIR is not required), several studies in one folder
(see :func:`list_studies`), several series per study, series whose slices have different
orientations (oblique slabs, 3-plane localizers), multi-echo / Dixon series (split by echo time and
image type), repeated acquisitions and temporal (4-D) series (split into runs), varying pixel
spacing (split into separate stacks), and multi-frame Enhanced MR objects (one stack per orientation
group of frames, with per-frame functional-group attributes).

Reading is two-phase. :func:`scan_dicom` reads headers only (``stop_before_pixels``) and classifies
every file (image, SR/PR/KO, secondary capture, colour screenshot, duplicate...). Stacks are then
built lazily: their pixels are read from the files only when :func:`load_stack_pixels` is called.

Privacy: only an allowlist of technical acquisition attributes is kept (see ``KEEP_TAGS``).
Patient name, IDs, dates, institution and physician names never enter the returned objects.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import UID

from ..config import DEFAULT, Config
from ..geometry import Stack, unit
from ..series import apply_display_ops

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

#: Enhanced (multi-frame) objects store some attributes under other keywords in functional groups.
_ALIASES: dict[str, tuple[str, ...]] = {"EchoTime": ("EchoTime", "EffectiveEchoTime")}

#: Non-image objects: structured reports, presentation states, key objects, RT, DICOMDIR, ...
_NON_IMAGE_SOP_PREFIXES = (
    "1.2.840.10008.5.1.4.1.1.88.",  # SR and KO
    "1.2.840.10008.5.1.4.1.1.11.",  # PR (presentation states)
    "1.2.840.10008.5.1.4.1.1.481.",  # RT objects
    "1.2.840.10008.1.3.10",  # DICOMDIR
)
_NON_IMAGE_MODALITIES = {"SR", "PR", "KO", "REG", "SEG", "RTSTRUCT", "RTPLAN", "RTDOSE", "DOC", "PLAN"}
#: Secondary captures are screenshots / exports (often with burned-in text): never anatomy.
_SECONDARY_CAPTURE_PREFIX = "1.2.840.10008.5.1.4.1.1.7"
_SECONDARY_CAPTURE_MODALITIES = {"SC", "OT"}

#: Attributes used to order repeated slices of a 4-D (temporal) series, first match wins.
_TIME_KEYWORDS = ("TemporalPositionIdentifier", "TriggerTime", "AcquisitionTime", "ContentTime")

_INSTALL_HINT = (
    "install the optional decoders with  pip install 'spinescope[jpeg]'  "
    "(pylibjpeg + pylibjpeg-libjpeg + pylibjpeg-openjpeg); RLE needs no extra package"
)


class UnsupportedPixelDataError(RuntimeError):
    """Pixel data cannot be decoded (usually a compressed transfer syntax without its decoder)."""


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
    study_uid: str = ""
    series_uid: str = ""
    sop_uid: str = ""
    time_key: str | None = None
    rows: int = 0
    cols: int = 0
    photometric: str = "MONOCHROME2"
    bits_stored: int = 12
    transfer_syntax: str = ""


@dataclass
class StudyInfo:
    """Technical, PHI-free summary of one study found in a folder."""

    index: int
    study_id: str  # short hash of StudyInstanceUID; the UID itself is never exposed
    n_images: int
    n_series: int
    modalities: tuple[str, ...]
    series_descriptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "study_id": self.study_id,
            "n_images": self.n_images,
            "n_series": self.n_series,
            "modalities": list(self.modalities),
            "series_descriptions": list(self.series_descriptions),
        }


@dataclass
class Scan:
    """Header-only result of scanning a folder."""

    refs: list[SliceRef]
    files: int
    ignored: Counter[str]
    missing_decoders: set[str]


@dataclass
class DicomRead:
    """Lazy stacks of the selected study plus everything the scan learned."""

    stacks: list[Stack]
    studies: list[StudyInfo]
    selected: StudyInfo
    scan: Scan
    warnings: list[str]


# ----------------------------------------------------------------------------- file discovery
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
    except Exception:
        return False


def iter_dicom_files(root: str | os.PathLike[str]) -> Iterator[str]:
    """Yield DICOM file paths under ``root`` (a file or directory), skipping DICOMDIR and dotfiles."""
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


# ----------------------------------------------------------------------------- attributes
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
    out: dict[str, Any] = {}
    for name in KEEP_TAGS:
        v = _get(ds, name)
        if v is not None:
            out[name] = _jsonable(v)
    return out


def _lookup(ds: Dataset, frame: int | None, keyword: str) -> Any:
    """Value of ``keyword`` for ``frame`` of an Enhanced object (per-frame group first, then shared).

    For classic single-frame objects (``frame is None``) this is the top-level attribute.
    """
    if frame is None:
        return _get(ds, keyword)
    for seq_name, idx in (("PerFrameFunctionalGroupsSequence", frame), ("SharedFunctionalGroupsSequence", 0)):
        seq = ds.get(seq_name)
        if not seq or idx >= len(seq):
            continue
        for el in seq[idx].iterall():
            if el.keyword == keyword and el.value not in (None, ""):
                return el.value
    return _get(ds, keyword)


def _frame_header(ds: Dataset, base: dict[str, Any], frame: int) -> dict[str, Any]:
    out = dict(base)
    for name in KEEP_TAGS:
        if name in out:
            continue
        for alt in _ALIASES.get(name, (name,)):
            v = _lookup(ds, frame, alt)
            if v is not None:
                out[name] = _jsonable(v)
                break
    return out


def _sop_class(ds: Dataset) -> str:
    return str(ds.get("SOPClassUID") or ds.file_meta.get("MediaStorageSOPClassUID") or "")


def _time_key(ds: Dataset, frame: int | None) -> str | None:
    for kw in _TIME_KEYWORDS:
        v = _lookup(ds, frame, kw)
        if v is not None and str(v) != "":
            return str(v)
    return None


def _float(v: Any, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------- scanning
def _refs_from_file(path: str) -> tuple[list[SliceRef], str | None]:
    """Header-only read of one file -> (slice references, reason it was ignored or None)."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:
        log.debug("skip unreadable file: %s", e)
        return [], "unreadable"
    modality = str(_get(ds, "Modality", "") or "").upper()
    sop = _sop_class(ds)
    if sop.startswith(_NON_IMAGE_SOP_PREFIXES) or modality in _NON_IMAGE_MODALITIES:
        return [], "non_image_object"
    if sop.startswith(_SECONDARY_CAPTURE_PREFIX) or modality in _SECONDARY_CAPTURE_MODALITIES:
        return [], "secondary_capture"
    if modality and modality != "MR":
        return [], "other_modality"
    photometric = str(_get(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if int(_get(ds, "SamplesPerPixel", 1) or 1) != 1 or photometric.startswith(("RGB", "YBR", "PALETTE")):
        return [], "colour_or_screenshot"
    try:
        refs = _build_refs(ds, path, photometric)
    except Exception as e:  # malformed header: ignore the file, never crash the scan
        log.debug("skip malformed file: %s", e)
        return [], "malformed"
    return refs, (None if refs else "no_geometry")


def _build_refs(ds: Dataset, path: str, photometric: str) -> list[SliceRef]:
    base = _header(ds)
    study_uid = str(_get(ds, "StudyInstanceUID", ""))
    series_uid = str(_get(ds, "SeriesInstanceUID", path))
    sop_uid = str(_get(ds, "SOPInstanceUID", ""))
    image_type = tuple(str(x).upper() for x in (_get(ds, "ImageType", []) or []))
    ts = str(ds.file_meta.get("TransferSyntaxUID", "") or "")
    nframes = int(_get(ds, "NumberOfFrames", 1) or 1)
    rows = int(_get(ds, "Rows", 0) or 0)
    cols = int(_get(ds, "Columns", 0) or 0)
    bits = int(_get(ds, "BitsStored", 12) or 12)
    refs: list[SliceRef] = []
    frames: list[int | None] = list(range(nframes)) if nframes > 1 else [None]
    for fr in frames:
        ipp = _lookup(ds, fr, "ImagePositionPatient")
        iop = _lookup(ds, fr, "ImageOrientationPatient")
        ps = _lookup(ds, fr, "PixelSpacing")
        thk = _lookup(ds, fr, "SliceThickness")
        if ipp is None or iop is None or ps is None:
            continue
        header = base if fr is None else _frame_header(ds, base, fr)
        echo = header.get("EchoTime")
        echo = round(float(echo), 2) if echo is not None else None
        iop_a = np.array([float(x) for x in iop], dtype=np.float64)
        okey = tuple(np.round(iop_a, 2))
        slope = _float(_lookup(ds, fr, "RescaleSlope"), 1.0) or 1.0
        intercept = _float(_lookup(ds, fr, "RescaleIntercept"), 0.0)
        refs.append(
            SliceRef(
                path=path,
                frame=fr,
                ipp=np.array([float(x) for x in ipp], dtype=np.float64),
                iop=iop_a,
                pixel_spacing=(float(ps[0]), float(ps[1])),
                thickness=_float(thk, 0.0),
                slope=slope,
                intercept=intercept,
                group_key=(study_uid, series_uid, okey, echo, image_type),
                header=header,
                study_uid=study_uid,
                series_uid=series_uid,
                sop_uid=sop_uid,
                time_key=_time_key(ds, fr),
                rows=rows,
                cols=cols,
                photometric=photometric,
                bits_stored=bits,
                transfer_syntax=ts,
            )
        )
    return refs


def scan_dicom(root: str | os.PathLike[str]) -> Scan:
    """Header-only scan of ``root``: every file is classified, duplicates (same SOP instance) dropped."""
    refs: list[SliceRef] = []
    ignored: Counter[str] = Counter()
    seen: set[tuple[str, int | None]] = set()
    missing: set[str] = set()
    files = 0
    for path in iter_dicom_files(root):
        files += 1
        rs, reason = _refs_from_file(path)
        if reason is not None:
            ignored[reason] += 1
            continue
        if (
            rs[0].transfer_syntax
            and _is_compressed(rs[0].transfer_syntax)
            and not decoder_available(rs[0].transfer_syntax)
        ):
            missing.add(UID(rs[0].transfer_syntax).name)
        for r in rs:
            if r.sop_uid:
                k = (r.sop_uid, r.frame)
                if k in seen:
                    ignored["duplicate"] += 1
                    continue
                seen.add(k)
            refs.append(r)
    return Scan(refs=refs, files=files, ignored=ignored, missing_decoders=missing)


# ----------------------------------------------------------------------------- pixel decoding
def _is_compressed(uid: str) -> bool:
    try:
        return bool(UID(uid).is_compressed)
    except Exception:
        return False


def decoder_available(uid: str) -> bool:
    """True if pydicom can decode the transfer syntax ``uid`` with the decoders installed here."""
    try:
        from pydicom.pixels import get_decoder

        return bool(get_decoder(UID(uid)).is_available)
    except Exception:
        return False


def _pixel_array(ds: Dataset, ref: SliceRef) -> np.ndarray:
    ts = ds.file_meta.get("TransferSyntaxUID")
    name = ts.name if ts is not None and hasattr(ts, "name") else "unknown transfer syntax"
    if ts is not None and _is_compressed(str(ts)) and not decoder_available(str(ts)):
        raise UnsupportedPixelDataError(
            f"pixel data is {name}, which needs a decoder that is not installed; {_INSTALL_HINT}"
        )
    try:
        arr = np.asarray(ds.pixel_array)
    except Exception as e:
        raise UnsupportedPixelDataError(f"cannot decode pixel data ({name}): {e}") from e
    if ref.photometric == "MONOCHROME1":  # inverted display: 0 is white
        if np.issubdtype(arr.dtype, np.integer):
            arr = (2**ref.bits_stored - 1) - arr.astype(np.int64)
        else:
            arr = arr.max() - arr
    return arr


def _read_pixels(ref: SliceRef, cache: dict[str, np.ndarray]) -> np.ndarray:
    if ref.path not in cache:
        ds = pydicom.dcmread(ref.path)
        arr = _pixel_array(ds, ref)
        cache.clear()  # keep at most one (multi-frame) file in memory
        cache[ref.path] = arr
    arr = cache[ref.path]
    img = arr[ref.frame] if ref.frame is not None else arr
    if img.ndim == 3:  # colour image: keep luminance (colour objects are filtered out in the scan)
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32) * ref.slope + ref.intercept


# ----------------------------------------------------------------------------- stacks
def _split_duplicates(refs: list[SliceRef]) -> list[list[SliceRef]]:
    """Split slices that share positions into runs (repeated acquisitions or 4-D time points)."""
    if len(refs) < 2:
        return [refs]
    normal = unit(np.cross(refs[0].iop[:3], refs[0].iop[3:]))
    rounded = np.round([r.ipp @ normal for r in refs], 2)
    if len(set(rounded.tolist())) == len(rounded):
        return [refs]
    tkeys = [r.time_key for r in refs]
    if all(t is not None for t in tkeys):
        runs: dict[str, list[SliceRef]] = defaultdict(list)
        for r in refs:
            runs[str(r.time_key)].append(r)
        unique_per_run = all(
            len({round(float(r.ipp @ normal), 2) for r in run}) == len(run) for run in runs.values()
        )
        if len(runs) > 1 and unique_per_run:
            return [runs[k] for k in sorted(runs, key=_time_sort_key)]
    seen: dict[float, int] = defaultdict(int)
    by_run: dict[int, list[SliceRef]] = defaultdict(list)
    for r, p in zip(refs, rounded, strict=False):
        by_run[seen[float(p)]].append(r)
        seen[float(p)] += 1
    return [by_run[i] for i in sorted(by_run)]


def _time_sort_key(k: str) -> tuple[float, str]:
    try:
        return (float(k), k)
    except ValueError:
        return (float("inf"), k)


def _split_geometry(run: list[SliceRef]) -> list[list[SliceRef]]:
    """Split a run whose slices have different pixel spacing or matrix size into uniform runs."""
    groups: dict[tuple[float, float, int, int], list[SliceRef]] = {}
    for r in run:
        key = (round(r.pixel_spacing[0], 4), round(r.pixel_spacing[1], 4), r.rows, r.cols)
        groups.setdefault(key, []).append(r)
    return list(groups.values())


def _stack_from_refs(run: list[SliceRef], series_key: str) -> Stack:
    """Lazy stack (no pixels read yet) for slices with one orientation and one pixel grid."""
    ref0 = run[0]
    normal = unit(np.cross(ref0.iop[:3], ref0.iop[3:]))
    order = np.argsort(np.array([r.ipp @ normal for r in run]), kind="stable")
    run = [run[i] for i in order]
    n = len(run)
    rows, cols = ref0.rows, ref0.cols
    if rows <= 0 or cols <= 0:
        rows, cols = (int(ref0.header.get("Rows", 1)), int(ref0.header.get("Columns", 1)))
    vol = np.broadcast_to(np.zeros((), dtype=np.float32), (n, rows, cols))  # no memory until loaded
    header = dict(ref0.header)
    meta: dict[str, Any] = {
        "dicom": header,
        "series_uid_hash": series_key,
        "study_uid_hash": _short_hash(ref0.study_uid),
        "_lazy": True,
        "_refs": run,
        "_sources": [(r.path, r.frame) for r in run],
    }
    return Stack(
        key="",
        vol=vol,
        ipp=np.stack([r.ipp for r in run]),
        row_dir=ref0.iop[:3],
        col_dir=ref0.iop[3:],
        row_mm=ref0.pixel_spacing[0],
        col_mm=ref0.pixel_spacing[1],
        thickness=ref0.thickness,
        description=str(header.get("SeriesDescription", "")),
        series_number=int(header["SeriesNumber"]) if "SeriesNumber" in header else None,
        meta=meta,
    )


def build_stacks(
    refs: Iterable[SliceRef], *, load_pixels: bool = True, config: Config = DEFAULT
) -> list[Stack]:
    """Group slice references into single-orientation stacks (keys are provisional).

    With ``load_pixels=False`` the stacks are lazy: their ``vol`` is a zero-cost placeholder with the
    right shape until :func:`load_stack_pixels` is called.
    """
    groups: dict[tuple[Any, ...], list[SliceRef]] = defaultdict(list)
    for r in refs:
        groups[r.group_key].append(r)
    stacks: list[Stack] = []
    for gkey, items in groups.items():
        series_key = _short_hash(str(gkey[1]))
        for ri, run in enumerate(_split_duplicates(items)):
            subs = _split_geometry(run)
            if len(subs) > 1:
                log.warning(
                    "series %s has varying pixel spacing or matrix size; split into %d stacks",
                    series_key,
                    len(subs),
                )
            for pj, sub in enumerate(subs):
                st = _stack_from_refs(sub, series_key)
                st.meta["repeat_index"] = ri
                st.meta["pixel_group"] = pj
                st.meta["geometry"] = slice_geometry(st.positions, config)
                if st.meta["geometry"]["gap_count"]:
                    log.warning(
                        "series %s: %d gap(s) in slice positions (~%d slice(s) missing)",
                        series_key,
                        st.meta["geometry"]["gap_count"],
                        st.meta["geometry"]["estimated_missing_slices"],
                    )
                stacks.append(st)
    if load_pixels:
        for st in stacks:
            load_stack_pixels(st)
    return stacks


def load_stack_pixels(stack: Stack) -> Stack:
    """Read the pixels of a lazy stack (in place) and return it. Loaded stacks are returned unchanged."""
    if not stack.meta.get("_lazy"):
        return stack
    refs: list[SliceRef] = stack.meta["_refs"]
    # Stack.__post_init__ may have re-sorted the slices (e.g. after a display flip): follow its order.
    order = np.argsort(np.array([r.ipp @ stack.normal for r in refs]), kind="stable")
    refs = [refs[i] for i in order]
    cache: dict[str, np.ndarray] = {}
    vol = np.stack([_read_pixels(r, cache) for r in refs])
    vol = apply_display_ops(vol, list(stack.meta.get("display_transform", [])))
    stack.vol = np.ascontiguousarray(vol, dtype=np.float32)
    stack.meta["_refs"] = refs
    del stack.meta["_lazy"]
    return stack


def slice_geometry(positions: np.ndarray, config: Config = DEFAULT) -> dict[str, Any]:
    """Describe slice spacing: uniform or not, and how many slices look missing (gaps)."""
    pos = np.sort(np.asarray(positions, dtype=np.float64))
    n = len(pos)
    out: dict[str, Any] = {"n_slices": n, "uniform": True, "gap_count": 0, "estimated_missing_slices": 0}
    if n < 2:
        return out
    d = np.abs(np.diff(pos))
    med = float(np.median(d))
    if med <= 0:
        return {**out, "uniform": False, "spacing_mm": 0.0}
    gaps = d >= config.slice_gap_factor * med
    dev = float(np.max(np.abs(d - med)) / med)
    missing = int(sum(round(float(x / med)) - 1 for x in d[gaps]))
    uniform = bool(dev <= config.slice_spacing_rel_tol and not gaps.any())
    out.update(
        spacing_mm=round(med, 4),
        spacing_min_mm=round(float(d.min()), 4),
        spacing_max_mm=round(float(d.max()), 4),
        uniform=uniform,
        gap_count=int(gaps.sum()),
        estimated_missing_slices=missing,
    )
    return out


def _short_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- studies
def _group_studies(refs: list[SliceRef]) -> dict[str, list[SliceRef]]:
    groups: dict[str, list[SliceRef]] = {}
    for r in refs:
        groups.setdefault(r.study_uid, []).append(r)
    return groups


def _study_table(groups: dict[str, list[SliceRef]]) -> list[tuple[StudyInfo, str]]:
    """Studies ordered by image count (largest first, ties by id), with their StudyInstanceUID."""
    rows: list[tuple[StudyInfo, str]] = []
    for uid, items in groups.items():
        series = sorted({r.series_uid for r in items})
        descs = sorted({str(r.header.get("SeriesDescription", "")) for r in items} - {""})
        info = StudyInfo(
            index=0,
            study_id=_short_hash(uid),
            n_images=len(items),
            n_series=len(series),
            modalities=tuple(sorted({str(r.header.get("Modality", "")) for r in items} - {""})),
            series_descriptions=tuple(descs),
        )
        rows.append((info, uid))
    rows.sort(key=lambda t: (-t[0].n_images, t[0].study_id))
    out: list[tuple[StudyInfo, str]] = []
    for k, (info, uid) in enumerate(rows):
        out.append(
            (
                StudyInfo(
                    k, info.study_id, info.n_images, info.n_series, info.modalities, info.series_descriptions
                ),
                uid,
            )
        )
    return out


def list_studies(root: str | os.PathLike[str]) -> list[StudyInfo]:
    """Studies (by StudyInstanceUID) found under ``root``, largest first. Header-only, PHI-free."""
    scan = scan_dicom(root)
    return [info for info, _uid in _study_table(_group_studies(scan.refs))]


def read_dicom(
    root: str | os.PathLike[str],
    *,
    study: int | str | None = None,
    config: Config = DEFAULT,
) -> DicomRead:
    """Scan ``root`` and build lazy stacks for one study.

    ``study`` selects by index into :func:`list_studies` order, by ``study_id`` (or the full
    StudyInstanceUID). The default is the study with the most images; a warning is logged when the
    folder holds several studies and no explicit choice was made.
    """
    scan = scan_dicom(root)
    if not scan.refs:
        raise FileNotFoundError(f"no DICOM images with plane geometry found under {root}")
    groups = _group_studies(scan.refs)
    table = _study_table(groups)
    warnings: list[str] = []
    if study is None:
        idx = 0
        if len(table) > 1:
            msg = (
                f"folder holds {len(table)} studies; loading the largest "
                f"(study_id={table[0][0].study_id}, {table[0][0].n_images} images). "
                "Pass study=<index or study_id> to choose another."
            )
            log.warning(msg)
            warnings.append(msg)
    elif isinstance(study, int):
        if not 0 <= study < len(table):
            raise KeyError(f"study index {study} out of range (folder has {len(table)} studies)")
        idx = study
    else:
        matches = [i for i, (info, uid) in enumerate(table) if study in (info.study_id, uid)]
        if not matches:
            ids = [info.study_id for info, _ in table]
            raise KeyError(f"no study {study!r} in folder; available study_ids: {ids}")
        idx = matches[0]
    selected, uid = table[idx]
    if scan.missing_decoders:
        msg = (
            f"pixel data uses {', '.join(sorted(scan.missing_decoders))} but its decoder is not installed; "
            f"header-only loading still works; {_INSTALL_HINT}"
        )
        log.warning(msg)
        warnings.append(msg)
    stacks = build_stacks(groups[uid], load_pixels=False, config=config)
    return DicomRead(
        stacks=stacks,
        studies=[info for info, _ in table],
        selected=selected,
        scan=scan,
        warnings=warnings,
    )


def read_stacks(
    root: str | os.PathLike[str],
    *,
    load_pixels: bool = True,
    study: int | str | None = None,
) -> list[Stack]:
    """Scan ``root`` and return one :class:`Stack` per (study, series, orientation, echo, image type)."""
    read = read_dicom(root, study=study)
    if load_pixels:
        for st in read.stacks:
            load_stack_pixels(st)
    return read.stacks


def dicom_scan_summary(scan: Scan) -> dict[str, Any]:
    """PHI-free counts of what the scan kept and ignored (for ``Study.report``)."""
    return {
        "files": scan.files,
        "images": len(scan.refs),
        "ignored": dict(sorted(scan.ignored.items())),
        "missing_decoders": sorted(scan.missing_decoders),
    }
