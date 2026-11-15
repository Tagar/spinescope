"""Read-only helpers for the public SPIDER lumbar MRI dataset (CC BY 4.0) used in dataset validation.

Layout (under ``$SPINESCOPE_DATA``, default ``~/data/public/spider``): ``images/<id>_t1.mha|_t2.mha`` sagittal
volumes, ``masks/<id>_t1.mha|_t2.mha`` segmentations, ``radiological_gradings.csv`` and ``overview.csv``.

Conventions used here (checked against the masks): vertebra labels 1..N run from the bottom (1 = L5),
disc label 200+i lies between vertebra i (above) and vertebra i-1 (below; i = 1 is L5-S1), and the
canal is label 100. Gradings ``IVD label`` i corresponds to disc label 200+i. Nothing is written to disk.
"""

from __future__ import annotations

import contextlib
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from spinescope.anatomy.anchors import Anchors, Disc, Vertebra
from spinescope.geometry import Stack
from spinescope.study import Study

VERT_NAMES = ("L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10", "T9", "T8")
CANAL_LABEL = 100
DISC_BASE = 200


def spider_root() -> Path | None:
    base = Path(os.environ.get("SPINESCOPE_DATA", "~/data/public")).expanduser()
    root = base / "spider"
    return root if (root / "images").is_dir() else None


def disc_name(i: int) -> str:
    """Disc name for SPIDER disc index i (1 = L5-S1, 2 = L4-L5, ...)."""
    upper = VERT_NAMES[i - 1]
    lower = VERT_NAMES[i - 2] if i >= 2 else "S1"
    return f"{upper}-{lower}"


def _sitk():  # SimpleITK is part of the 'volumes' extra
    import SimpleITK as sitk

    return sitk


def _stack_from_sitk(path: Path, key: str, contrast: str) -> Stack:
    """Stack in physical (LPS) space: slices along the first image axis, rows along z, columns along y.

    Raises ValueError when the slice axis is not aligned with the in-plane cross product (the stack
    would be reordered and the mask would no longer line up).
    """
    sitk = _sitk()
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    sx, sy, sz = img.GetSpacing()
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    d = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)  # columns = index axes x, y, z
    d0, d1, d2 = d[:, 0], d[:, 1], d[:, 2]
    if float(np.dot(np.cross(d1, d2), d0)) <= 0:
        raise ValueError(f"{path.name}: slice axis not aligned with the in-plane normal")
    v = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # (slice = x, row = z, col = y)
    n = v.shape[0]
    ipp = np.array([origin + k * sx * d0 for k in range(n)], dtype=np.float64)
    return Stack(
        key=key,
        vol=v,
        ipp=ipp,
        row_dir=d1 / np.linalg.norm(d1),  # increasing column (y) direction
        col_dir=d2 / np.linalg.norm(d2),  # increasing row (z) direction
        row_mm=float(sz),
        col_mm=float(sy),
        thickness=float(sx),
        description=path.stem,
        meta={
            "role": {
                "plane": "sag",
                "contrast": contrast,
                "localizer": False,
                "slab_index": None,
                "fat_suppressed": False,
                "post_contrast": False,
                "dixon": None,
                "evidence": [],
            }
        },
    )


def load_t2_t1(study_id: str, root: Path | None = None) -> tuple[Study, Stack, NDArray[np.int16]] | None:
    """Study with sagittal T2 (and T1 when present), plus the T2 mask array aligned to the T2 stack."""
    root = root or spider_root()
    if root is None:
        return None
    t2p = root / "images" / f"{study_id}_t2.mha"
    t2m = root / "masks" / f"{study_id}_t2.mha"
    if not (t2p.exists() and t2m.exists()):
        return None
    try:
        t2 = _stack_from_sitk(t2p, "sag_t2", "t2")
    except ValueError:
        return None
    mask_img = _sitk().ReadImage(str(t2m))
    mask = np.transpose(_sitk().GetArrayFromImage(mask_img), (2, 0, 1)).astype(np.int16)
    if mask.shape != t2.vol.shape:
        return None
    stacks = {"sag_t2": t2}
    t1p = root / "images" / f"{study_id}_t1.mha"
    if t1p.exists():
        with contextlib.suppress(ValueError):
            stacks["sag_t1"] = _stack_from_sitk(t1p, "sag_t1", "t1")
    return Study(stacks=stacks, info={"Modality": "MR", "source": "SPIDER"}), t2, mask


def _lps_of(st: Stack, k: NDArray[np.intp], r: NDArray[np.intp], c: NDArray[np.intp]) -> NDArray[np.float64]:
    return st.ipp[k] + (c * st.col_mm)[:, None] * st.row_dir + (r * st.row_mm)[:, None] * st.col_dir


def anchors_from_masks(st: Stack, mask: NDArray[np.int16]) -> Anchors:
    """Disc centre, superior normal and AP margins from the mid-sagittal mask of each disc, plus vertebral
    centres and the canal-based midline. Derived only from the segmentation (the auto-labeller is separate)."""
    discs: dict[str, Disc] = {}
    vertebrae: dict[str, Vertebra] = {}
    for lab in sorted(int(x) for x in np.unique(mask) if x > 0):
        k_all, r_all, c_all = np.nonzero(mask == lab)
        if k_all.size < 20:
            continue
        pts = _lps_of(st, k_all, r_all, c_all)
        if DISC_BASE < lab < DISC_BASE + 30:
            i = lab - DISC_BASE
            name = disc_name(i)
            ks, counts = np.unique(k_all, return_counts=True)
            kmid = int(ks[np.argmax(counts)])
            sel = k_all == kmid
            pk = pts[sel]
            c3 = pk.mean(axis=0)
            yz = pk[:, 1:] - c3[1:]
            _w, v = np.linalg.eigh(np.cov(yz.T))
            n_yz = v[:, 0]  # smallest variance: disc normal
            u_yz = v[:, 1]  # largest: AP
            if n_yz[1] < 0:
                n_yz = -n_yz  # superior = +z
            if u_yz[0] < 0:
                u_yz = -u_yz  # posterior = +y
            normal = np.array([0.0, n_yz[0], n_yz[1]])
            u = np.array([0.0, u_yz[0], u_yz[1]])
            s = (pk - c3) @ u
            ant = c3 + float(s.min()) * u
            post = c3 + float(s.max()) * u
            center = pts.mean(axis=0)
            discs[name] = Disc(
                center=center.tolist(),
                normal=normal.tolist(),
                anterior=ant.tolist(),
                posterior=post.tolist(),
                confidence=1.0,
                evidence=["spider-mask"],
            )
        elif 0 < lab <= len(VERT_NAMES):
            vertebrae[VERT_NAMES[lab - 1]] = Vertebra(
                center=pts.mean(axis=0).tolist(), confidence=1.0, evidence=["spider-mask"]
            )
    canal = mask == CANAL_LABEL
    midline = None
    if canal.any():
        _, _, _c_can = np.nonzero(canal)
        k_can, r_can, c_can2 = np.nonzero(canal)
        midline = float(_lps_of(st, k_can, r_can, c_can2)[:, 0].mean())
    elif discs:
        midline = float(np.mean([d.center[0] for d in discs.values()]))
    return Anchors(
        discs=discs,
        vertebrae=vertebrae,
        midline_x=midline,
        numbering={"method": "spider-mask", "confidence": 1.0},
        method="spider-mask",
        source_stack=st.key,
    )


@dataclass(frozen=True)
class Grading:
    pfirrmann: int
    modic: int
    up_endplate: int
    low_endplate: int


def gradings(root: Path | None = None) -> dict[tuple[str, int], Grading]:
    """Expert grades keyed by (patient, IVD label)."""
    root = root or spider_root()
    if root is None:
        return {}
    out: dict[tuple[str, int], Grading] = {}
    with open(root / "radiological_gradings.csv", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["Patient"].strip(), int(row["IVD label"]))
            out[key] = Grading(
                pfirrmann=int(row["Pfirrman grade"]),
                modic=int(row["Modic"]),
                up_endplate=int(row["UP endplate"]),
                low_endplate=int(row["LOW endplate"]),
            )
    return out


def paired_study_ids(root: Path | None = None) -> list[str]:
    """Study ids with sagittal T1 and T2 images and T2 masks, numerically sorted."""
    root = root or spider_root()
    if root is None:
        return []
    ids = []
    for p in sorted((root / "images").glob("*_t2.mha")):
        sid = p.name[: -len("_t2.mha")]
        if (root / "images" / f"{sid}_t1.mha").exists() and (root / "masks" / f"{sid}_t2.mha").exists():
            ids.append(sid)
    return sorted(ids, key=lambda s: int(s) if s.isdigit() else 10**9)
