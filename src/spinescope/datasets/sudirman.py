"""Sudirman lumbar spine MRI (CC BY 4.0): clinical DICOM studies and disc ground truth from the dataset.

Layout under ``$SPINESCOPE_DATA`` (default ``~/data/public``)::

    lumbar_spine_mri_sudirman/01_MRI_Data/<patient>/<session>/<series>/*.ima   DICOM (Siemens, 1.5 T)
    lumbar_spine_mri_sudirman_labels/
        04_Intermediary_Ground_Truth_Data/T1_Output/T1_<patient>_D<n>.png       320 x 320 axial T1 slice
        05_Final_Ground_Truth_Data/Label_Images/L1_<patient>_D<n>.png           320 x 320 label map
    lumbar_spine_mri_sudirman_notes/raw/Radiologists Report.xlsx               free-text notes

Each labelled patient has one axial T1 slice per annotated disc ``D3``, ``D4`` and ``D5``. The T1 slice
is the exact DICOM pixel grid of one axial slab (verified: the 320 x 320 reference equals the DICOM slice
with normalised cross-correlation 1.0), so the label map addresses the patient geometry directly. Label
values: 50 intervertebral disc, 100 posterior elements, 150 thecal sac, 200 area between anterior and
posterior elements. The disc names come from the dataset's own convention ("the last three IVDs") and
the radiologist notes, which mention L3-L4, L4-L5 and L5-S1 almost exclusively.

Patients with several sessions have the labelled slices in one session only: :func:`select_session`
picks it by image agreement, never by folder order.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..study import Study
from .registry import SUDIRMAN_LUMBAR_LABELS, SUDIRMAN_LUMBAR_MRI, SUDIRMAN_RADIOLOGIST_NOTES, data_dir

log = logging.getLogger(__name__)

#: Annotated disc label -> level name. D5 is the lowest of the three (L5-S1) in every labelled session.
DISC_LEVELS: dict[str, str] = {"D3": "L3-L4", "D4": "L4-L5", "D5": "L5-S1"}
#: Label-map values of the four annotated regions (verified by overlay on the T1 slice).
IVD_VALUE = 50
POSTERIOR_ELEMENT_VALUE = 100
THECAL_SAC_VALUE = 150
AAP_VALUE = 200
#: Minimum normalised cross-correlation between a reference slice and a DICOM slice to accept the match.
MIN_MATCH_NCC = 0.98

_TRUTH_DIR_T1 = "04_Intermediary_Ground_Truth_Data/T1_Output"
_TRUTH_DIR_MASK = "05_Final_Ground_Truth_Data/Label_Images"
# Unenhanced axial T1 stacks: ax_t1, ax_t1_2 (repeated series) and their slabs (ax_t1_slab1, ...).
_SLAB_RE = re.compile(r"^ax_t1(_\d+)?(_slab\d+)?$")


@dataclass(frozen=True)
class LevelTruth:
    """One annotated disc of one study, located in the patient's LPS space."""

    patient: str
    dlabel: str  # D3 | D4 | D5 (dataset label)
    level: str  # L3-L4 | L4-L5 | L5-S1
    stack_key: str  # axial T1 stack that contains the labelled slice
    slice_index: int
    center: np.ndarray  # LPS mm: centroid of the IVD label on the matched slice
    normal: np.ndarray  # unit, superior: normal of the axial slab (the disc axis)
    ivd_pixels: int
    match_ncc: float


def sudirman_root(root: str | Path | None = None) -> Path:
    """Folder that holds ``01_MRI_Data``: ``root`` if given, else ``$SPINESCOPE_DATA/lumbar_spine_mri_sudirman``."""
    base = Path(root).expanduser() if root is not None else data_dir() / SUDIRMAN_LUMBAR_MRI.subdir
    return base / "01_MRI_Data" if (base / "01_MRI_Data").is_dir() else base


def sudirman_labels_root(root: str | Path | None = None) -> Path:
    """Folder of the label companion set (``zbf6b4pttk``): ``root`` or ``$SPINESCOPE_DATA/<labels subdir>``."""
    return Path(root).expanduser() if root is not None else data_dir() / SUDIRMAN_LUMBAR_LABELS.subdir


def patient_sessions(root: str | Path | None = None) -> dict[str, list[Path]]:
    """Patient id (folder name, e.g. ``0389``) -> its session folders, sorted."""
    base = sudirman_root(root)
    if not base.is_dir():
        raise FileNotFoundError(f"no Sudirman 01_MRI_Data under {base}; see docs/datasets.md")
    out: dict[str, list[Path]] = {}
    for pdir in sorted(p for p in base.iterdir() if p.is_dir()):
        sessions = sorted(s for s in pdir.iterdir() if s.is_dir())
        if sessions:
            out[pdir.name] = sessions
    return out


def labelled_patients(labels_root: str | Path | None = None) -> list[str]:
    """Patient ids that have label maps, sorted."""
    mask_dir = sudirman_labels_root(labels_root) / _TRUTH_DIR_MASK
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"no label maps under {mask_dir}; fetch 'lumbar_spine_mri_sudirman_labels'")
    ids = {p.name.split("_")[1] for p in mask_dir.glob("L1_*_D*.png")}
    return sorted(ids)


def read_mask(patient: str, dlabel: str, labels_root: str | Path | None = None) -> np.ndarray:
    """Label map (``uint8``, 320 x 320) of disc ``dlabel`` of ``patient``."""
    return _read_png(sudirman_labels_root(labels_root) / _TRUTH_DIR_MASK / f"L1_{patient}_{dlabel}.png")


def read_reference(patient: str, dlabel: str, labels_root: str | Path | None = None) -> np.ndarray:
    """The axial T1 slice (320 x 320) that the label map of ``dlabel`` was drawn on."""
    return _read_png(sudirman_labels_root(labels_root) / _TRUTH_DIR_T1 / f"T1_{patient}_{dlabel}.png")


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two equally shaped arrays (1.0 = identical up to gain and offset)."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(x @ y / x.size)


def match_levels_arrays(
    study: Study,
    references: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    *,
    patient: str = "",
    min_ncc: float = MIN_MATCH_NCC,
) -> list[LevelTruth]:
    """Locate each annotated disc in ``study``: the axial T1 slice that matches its reference image.

    ``references`` and ``masks`` map a disc label (``D3``...) to its 2-D arrays. Every ``ax_t1`` stack of
    the study is searched; a disc is returned only when its best match reaches ``min_ncc``.
    """
    stacks = [s for k, s in study.stacks.items() if _SLAB_RE.match(k)]
    out: list[LevelTruth] = []
    for dlabel in sorted(references):
        ref = references[dlabel]
        score, found, k = -2.0, None, -1
        for cand in stacks:
            if cand.vol.shape[1:] != ref.shape:
                continue
            for i in range(cand.vol.shape[0]):
                c = ncc(cand.vol[i], ref)
                if c > score:
                    score, found, k = c, cand, i
        st = found
        if st is None or score < min_ncc:
            log.info(
                "disc %s of %s: no axial T1 slice matches its reference (best NCC %.3f)",
                dlabel,
                patient,
                score,
            )
            continue
        mask = masks.get(dlabel)
        if mask is None or mask.shape != ref.shape:
            raise ValueError(f"label map for {dlabel} does not match its reference slice")
        rr, cc = np.nonzero(mask == IVD_VALUE)
        if rr.size == 0:
            log.info("disc %s of %s: empty IVD label", dlabel, patient)
            continue
        centre = st.pix_to_lps(k, float(rr.mean()), float(cc.mean()))
        normal = np.asarray(st.normal, dtype=np.float64)
        if normal[2] < 0:
            normal = -normal
        out.append(
            LevelTruth(
                patient=patient,
                dlabel=dlabel,
                level=DISC_LEVELS.get(dlabel, dlabel),
                stack_key=st.key,
                slice_index=k,
                center=np.asarray(centre, dtype=np.float64),
                normal=normal / np.linalg.norm(normal),
                ivd_pixels=int(rr.size),
                match_ncc=float(score),
            )
        )
    return out


def match_levels(
    study: Study,
    patient: str,
    labels_root: str | Path | None = None,
    *,
    min_ncc: float = MIN_MATCH_NCC,
) -> list[LevelTruth]:
    """Ground-truth discs of ``patient`` located in ``study`` (see :func:`match_levels_arrays`)."""
    refs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for d in DISC_LEVELS:
        if (sudirman_labels_root(labels_root) / _TRUTH_DIR_MASK / f"L1_{patient}_{d}.png").is_file():
            refs[d] = read_reference(patient, d, labels_root)
            masks[d] = read_mask(patient, d, labels_root)
    return match_levels_arrays(study, refs, masks, patient=patient, min_ncc=min_ncc)


def select_session(
    patient: str,
    sessions: list[Path],
    labels_root: str | Path | None = None,
    *,
    min_ncc: float = MIN_MATCH_NCC,
) -> tuple[Path, Study, list[LevelTruth]] | None:
    """The session of ``patient`` whose axial T1 slices carry the labels (most annotated discs, then best NCC).

    Returns ``(session, loaded study, truths)`` or ``None`` when no session reproduces a label slice.
    Sessions are loaded with pixels; only the chosen study is returned.
    """
    from ..study import load_study

    best: tuple[tuple[int, float], Path, Study, list[LevelTruth]] | None = None
    for sess in sessions:
        study = load_study(sess)
        truths = match_levels(study, patient, labels_root, min_ncc=min_ncc)
        if not truths:
            continue
        score = (len(truths), float(np.mean([t.match_ncc for t in truths])))
        if best is None or score > best[0]:
            best = (score, sess, study, truths)
    if best is None:
        return None
    return best[1], best[2], best[3]


def read_radiologist_notes(root: str | Path | None = None) -> dict[int, str]:
    """Free-text radiologist notes keyed by the notes' patient number (``Radiologists Report.xlsx``).

    Read with the standard library (xlsx is a zip of XML). Text may name levels; it is not used as ground truth.
    """
    base = Path(root).expanduser() if root is not None else data_dir() / SUDIRMAN_RADIOLOGIST_NOTES.subdir
    path = base / "raw" / SUDIRMAN_RADIOLOGIST_NOTES.files[0].name
    if not path.is_file():
        raise FileNotFoundError(f"radiologist notes not found at {path}")
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        shared = [
            "".join(t.text or "" for t in si.iter(f"{ns}t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{ns}si")
        ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out: dict[int, str] = {}
    for row in sheet.iter(f"{ns}row"):
        cells: dict[str, str] = {}
        for c in row.findall(f"{ns}c"):
            col = re.match(r"[A-Z]+", c.get("r", "")).group(0)  # type: ignore[union-attr]
            v = c.find(f"{ns}v")
            if v is not None and v.text is not None:
                cells[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
        if cells.get("A", "").strip().isdigit():
            out[int(cells["A"])] = cells.get("B", "")
    return out
