"""Read 3-D volume files (NIfTI, MetaImage .mha/.mhd, NRRD) as a :class:`~spinescope.geometry.Stack`.

Research datasets (e.g. SPIDER) ship volumes instead of DICOM. A volume is converted to a stack
by slicing along its through-plane axis (the axis with the largest voxel spacing, i.e. the
acquisition's slice direction). Geometry is kept in DICOM LPS.

Backends: SimpleITK (MetaImage/NRRD/NIfTI, already LPS) if installed, else nibabel for NIfTI
(RAS converted to LPS), else a built-in MetaImage reader.
"""

from __future__ import annotations

import os
import zlib
from typing import Any

import numpy as np

from ..geometry import Stack

_MET_TYPES = {
    "MET_UCHAR": np.uint8,
    "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16,
    "MET_SHORT": np.int16,
    "MET_UINT": np.uint32,
    "MET_INT": np.int32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
    "MET_ULONG_LONG": np.uint64,
    "MET_LONG_LONG": np.int64,
}


def _read_metaimage(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimal MetaImage reader -> (array[z, y, x], spacing[x,y,z], origin[x,y,z], direction 3x3).

    ``direction`` columns are the LPS directions of the i, j, k (x, y, z index) axes, as in ITK.
    """
    with open(path, "rb") as f:
        raw = f.read()
    header: dict[str, str] = {}
    pos = 0
    while True:
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("latin-1").strip()
        pos = end + 1
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            header[k] = v
            if k == "ElementDataFile":
                break
    dims = [int(x) for x in header["DimSize"].split()]
    nd = int(header.get("NDims", len(dims)))
    spacing = np.array(
        [float(x) for x in header.get("ElementSpacing", header.get("ElementSize", "1 1 1")).split()]
    )
    origin = np.array([float(x) for x in header.get("Offset", header.get("Origin", "0 0 0")).split()])
    tm = header.get("TransformMatrix", header.get("Rotation", header.get("Orientation")))
    direction = np.eye(nd) if tm is None else np.array([float(x) for x in tm.split()]).reshape(nd, nd).T
    dtype = np.dtype(_MET_TYPES[header["ElementType"]])
    if header.get("BinaryDataByteOrderMSB", header.get("ElementByteOrderMSB", "False")).lower() == "true":
        dtype = dtype.newbyteorder(">")
    datafile = header["ElementDataFile"]
    if datafile == "LOCAL":
        data = raw[pos:]
    else:
        with open(os.path.join(os.path.dirname(path), datafile), "rb") as f:
            data = f.read()
    if header.get("CompressedData", "False").lower() == "true":
        data = zlib.decompress(data)
    n_comp = int(header.get("ElementNumberOfChannels", 1))
    arr = np.frombuffer(data, dtype=dtype, count=int(np.prod(dims)) * n_comp)
    arr = arr.reshape(dims[::-1] + ([n_comp] if n_comp > 1 else []))
    if n_comp > 1:
        arr = arr.mean(axis=-1)
    return arr, spacing[:3], origin[:3], direction[:3, :3]


def _read_any(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (array indexed [k, j, i], spacing (i, j, k), origin LPS, direction LPS columns i, j, k)."""
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 4:
            arr = arr.mean(axis=-1)
        return (
            arr,
            np.array(img.GetSpacing())[:3],
            np.array(img.GetOrigin())[:3],
            np.array(img.GetDirection()).reshape(img.GetDimension(), img.GetDimension())[:3, :3],
        )
    except ImportError:
        pass
    lower = path.lower()
    if lower.endswith((".mha", ".mhd")):
        return _read_metaimage(path)
    if lower.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        nimg: Any = nib.load(path)
        data = np.asarray(nimg.dataobj)
        if data.ndim == 4:
            data = data[..., 0]
        aff = nimg.affine.copy()
        aff[:2, :] *= -1  # RAS -> LPS
        spacing = np.linalg.norm(aff[:3, :3], axis=0)
        direction = aff[:3, :3] / spacing
        arr = np.transpose(data, (2, 1, 0))  # [i, j, k] -> [k, j, i]
        return arr, spacing, aff[:3, 3], direction
    raise ImportError(f"reading {path} needs SimpleITK (pip install 'spinescope[volumes]')")


def _stack_layout(spacing: np.ndarray) -> tuple[int, int, int, list[int]]:
    """(slice axis, row axis, column axis, transpose permutation) for array indices [k, j, i]."""
    # index axes: i (x), j (y), k (z) -> array axes 2, 1, 0
    slice_axis = int(np.argmax(spacing))  # 0=i, 1=j, 2=k
    inplane = [a for a in (0, 1, 2) if a != slice_axis]
    a_r, a_c = inplane[1], inplane[0]  # rows follow the slower in-plane axis
    return slice_axis, a_r, a_c, [2 - slice_axis, 2 - a_r, 2 - a_c]


def read_label_volume(path: str) -> np.ndarray:
    """Read a volume as a label/intensity array ``[slice, row, col]`` in the same order as :func:`read_volume`.

    Integer dtypes are kept (e.g. segmentation masks), so the array can be indexed with the same
    voxel coordinates as the stack read from the matching image file.
    """
    arr, spacing, _origin, _direction = _read_any(path)
    _sa, _ar, _ac, perm = _stack_layout(spacing)
    return np.ascontiguousarray(np.transpose(np.asarray(arr), perm))


def read_volume(path: str, *, key: str = "", meta: dict[str, Any] | None = None) -> Stack:
    """Read a volume file as a stack sliced along its through-plane (largest-spacing) axis."""
    arr, spacing, origin, direction = _read_any(path)
    arr = np.asarray(arr, dtype=np.float32)  # [k, j, i]
    slice_axis, a_r, a_c, perm = _stack_layout(spacing)
    vol = np.transpose(arr, perm)
    d = np.asarray(direction, dtype=np.float64)
    row_dir, col_dir, slice_dir = d[:, a_c], d[:, a_r], d[:, slice_axis]
    ipp = np.array([origin + k * spacing[slice_axis] * slice_dir for k in range(vol.shape[0])])
    # Volume files carry no DICOM header; the file name is the only hint about the contrast
    # (e.g. SPIDER's "12_t2.mha"), so expose it as a description for classification.
    hint = os.path.basename(path).split(".")[0].replace("_", " ").replace("-", " ")
    m: dict[str, Any] = {"source": "volume", "dicom": {"SeriesDescription": hint}}
    m.update(meta or {})
    return Stack(
        key=key,
        vol=vol,
        ipp=ipp,
        row_dir=row_dir,
        col_dir=col_dir,
        row_mm=float(spacing[a_r]),
        col_mm=float(spacing[a_c]),
        thickness=float(spacing[slice_axis]),
        description=hint,
        meta=m,
    )
