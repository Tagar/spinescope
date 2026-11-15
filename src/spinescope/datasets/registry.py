"""Registry of public spine datasets: where they come from, their licence, citation and checksums.

Only datasets whose licence was checked at the source are listed. The licence and size fields were
read from the hosting record (Zenodo API, Mendeley Data page) on 2026-11-14; re-check before a release.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Default cache root, overridable with the SPINESCOPE_DATA environment variable.
DEFAULT_DATA_DIR = "~/data/public"


def data_dir() -> Path:
    """Local dataset cache root: ``$SPINESCOPE_DATA`` or ``~/data/public``."""
    return Path(os.environ.get("SPINESCOPE_DATA") or DEFAULT_DATA_DIR).expanduser()


@dataclass(frozen=True)
class FileSpec:
    """One downloadable file. ``name`` is its file name in the dataset's ``raw/`` folder."""

    name: str
    url: str
    size: int
    md5: str | None = None
    sha256: str | None = None
    #: How the URL is obtained: "direct" (fixed URL) or "mendeley-api" (resolved from the files API).
    resolver: str = "direct"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    modality: str
    licence: str  # SPDX-style identifier, e.g. "CC-BY-4.0"
    licence_url: str
    doi: str
    source_url: str
    citation: str
    description: str
    #: Folder under the data root that holds the dataset (``raw/`` downloads, extracted files beside it).
    subdir: str
    files: tuple[FileSpec, ...] = ()
    notes: str = ""
    #: Licence text was verified at the source (False = placeholder, do not use without checking).
    verified: bool = True
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    def path(self, root: str | os.PathLike[str] | None = None) -> Path:
        return Path(root).expanduser() if root is not None else data_dir() / self.subdir


SPIDER = DatasetSpec(
    key="spider",
    title="SPIDER: lumbar spine segmentation in MR images",
    modality="MRI, sagittal T1 / T2 (+ T2 SPACE), vertebra / disc / canal masks, radiological gradings",
    licence="CC-BY-4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    doi="10.5281/zenodo.10159290",
    source_url="https://zenodo.org/records/10159290",
    citation=(
        "van der Graaf JW, et al. Lumbar spine segmentation in MR images: a dataset and a public "
        "benchmark. Scientific Data (2024). doi:10.1038/s41597-024-03090-w. Dataset: "
        "doi:10.5281/zenodo.10159290"
    ),
    description=(
        "218 patients, 447 sagittal volumes (MetaImage .mha) with vertebra, intervertebral disc and "
        "spinal canal masks, plus per-disc Pfirrman, Modic, endplate, herniation, narrowing, bulging "
        "and spondylolisthesis gradings."
    ),
    subdir="spider",
    files=(
        FileSpec(
            "images.zip",
            "https://zenodo.org/api/records/10159290/files/images.zip/content",
            3700562886,
            md5="7a9fa44aac0c72e2937bdef62cff88db",
        ),
        FileSpec(
            "masks.zip",
            "https://zenodo.org/api/records/10159290/files/masks.zip/content",
            58222897,
            md5="49861ff25a0108b3b69d0ce808ae51dc",
        ),
        FileSpec(
            "overview.csv",
            "https://zenodo.org/api/records/10159290/files/overview.csv/content",
            119856,
            md5="f9a808d2cf67844baa32114a1a769fde",
        ),
        FileSpec(
            "radiological_gradings.csv",
            "https://zenodo.org/api/records/10159290/files/radiological_gradings.csv/content",
            34417,
            md5="a5b7a6ad7fb4daa0bfa650b716f026f6",
        ),
    ),
    notes=(
        "images.zip is 3.7 GB, above the default 2 GiB download cap: pass allow_large=True to fetch. "
        "Extracted layout: images/<case>_<seq>.mha, masks/<case>_<seq>.mha, overview.csv, "
        "radiological_gradings.csv."
    ),
)

SUDIRMAN_LUMBAR_MRI = DatasetSpec(
    key="lumbar_spine_mri_sudirman",
    title="Lumbar Spine MRI Dataset (Sudirman et al.)",
    modality="MRI, DICOM, axial and sagittal lumbar spine studies (515 patients, anonymised)",
    licence="CC-BY-4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    doi="10.17632/k57fr854j2.2",
    source_url="https://data.mendeley.com/datasets/k57fr854j2/2",
    citation=(
        "Sudirman S, Al Kafri A, Natalia F, Meidia H, Afriliana N, Al-Rashdan W, Bashtawi M, "
        "Al-Jumaily M. Lumbar Spine MRI Dataset. Mendeley Data, V2 (2019). doi:10.17632/k57fr854j2.2"
    ),
    description=(
        "Anonymised DICOM studies of symptomatic back pain: sagittal slices of the last seven "
        "vertebrae and axial slices of the last three discs, per study."
    ),
    subdir="lumbar_spine_mri_sudirman",
    extra={"mendeley_id": "k57fr854j2"},
    files=(
        FileSpec(
            "MRI_Data.zip",
            "https://data.mendeley.com/public-files/datasets/k57fr854j2/files/"
            "eab74360-db27-4ec5-ade3-7b1b8d88e2db/file_downloaded",
            6270535053,
            sha256="ac6495c243c3e95a820b16bc751063e75f7d90e1d3ef22170f925a2448eadc1a",
            resolver="mendeley-api",
        ),
    ),
    notes=(
        "6.27 GB single zip: not fetched by default (above the 2 GiB cap). The download URL is resolved "
        "from https://data.mendeley.com/api/datasets/k57fr854j2/files at fetch time. Needs the "
        "~6.3 GB of free space. Format is DICOM (no labels): suitable for loader tests and "
        "classification, not for segmentation validation."
    ),
)

SUDIRMAN_LUMBAR_LABELS = DatasetSpec(
    key="lumbar_spine_mri_sudirman_labels",
    title="Label Image Ground Truth Data for Lumbar Spine MRI Dataset (Sudirman et al.)",
    modality="MRI axial T1-weighted label images (IVD, posterior element, thecal sac, AAP) of the last three discs",
    licence="CC-BY-4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    doi="10.17632/zbf6b4pttk.2",
    source_url="https://data.mendeley.com/datasets/zbf6b4pttk/2",
    citation=(
        "Sudirman S, Al Kafri A, Natalia F, Meidia H, Afriliana N, Al-Rashdan W, Bashtawi M, "
        "Al-Jumaily M. Label Image Ground Truth Data for Lumbar Spine MRI Dataset. Mendeley Data, V2 "
        "(2019). doi:10.17632/zbf6b4pttk.2"
    ),
    description=(
        "Raw and ground-truth label images of axial T1 slices of the last three intervertebral discs, "
        "drawn by five labellers under an expert radiologist (companion of the Sudirman MRI dataset)."
    ),
    subdir="lumbar_spine_mri_sudirman_labels",
    extra={"mendeley_id": "zbf6b4pttk"},
    files=(
        FileSpec(
            "Ground_Truth_Label.zip",
            "https://data.mendeley.com/public-files/datasets/zbf6b4pttk/files/"
            "0a216a09-5349-4f4b-9a36-df28d436ae38/file_downloaded",
            1039678878,
            sha256="ba59da9855499265818e360b0bec7c47ec488f26fe14560845417c58b445a217",
            resolver="mendeley-api",
        ),
        FileSpec(
            "Manual_Label_Data.zip",
            "https://data.mendeley.com/public-files/datasets/zbf6b4pttk/files/"
            "78e10a83-df38-44ef-ab2f-82abf129e65f/file_downloaded",
            637438849,
            sha256="d8277b7f46264eb1d5f71b2f08da65f03812237bb1398c8d5235118f51a72b00",
            resolver="mendeley-api",
        ),
    ),
    notes="Both zips are below the 2 GiB cap (1.68 GB together). Resolved from the Mendeley files API.",
)

SUDIRMAN_RADIOLOGIST_NOTES = DatasetSpec(
    key="lumbar_spine_mri_sudirman_notes",
    title="Radiologists Notes for Lumbar Spine MRI Dataset (Sudirman et al.)",
    modality="Expert radiologist annotations (xlsx) for the Sudirman lumbar MRI studies",
    licence="CC-BY-4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    doi="10.17632/s6bgczr8s2.2",
    source_url="https://data.mendeley.com/datasets/s6bgczr8s2/2",
    citation=(
        "Sudirman S, Al Kafri A, Natalia F, Meidia H, Afriliana N, Al-Rashdan W, Bashtawi M, "
        "Al-Jumaily M. Radiologists Notes for Lumbar Spine MRI Dataset. Mendeley Data, V2 (2019). "
        "doi:10.17632/s6bgczr8s2.2"
    ),
    description="Radiologist notes per study: bone marrow disease, end plate degeneration, disc bulges, thecal sac compression.",
    subdir="lumbar_spine_mri_sudirman_notes",
    extra={"mendeley_id": "s6bgczr8s2"},
    files=(
        FileSpec(
            "Radiologists Report.xlsx",
            "https://data.mendeley.com/public-files/datasets/s6bgczr8s2/files/"
            "ed86d033-5078-4658-948f-94c13e8b2291/file_downloaded",
            38292,
            sha256="3ec6d554fe277cb02554e783a36ad89ebfd58f121206ce3c9cf9158f22d954a7",
            resolver="mendeley-api",
        ),
    ),
)

VCF_MRI_HCFMRP = DatasetSpec(
    key="lumbar_vcf_mri_hcfmrp",
    title="Lumbar vertebral compression fractures MRI (T1, benign vs malignant)",
    modality="MRI, sagittal T1-weighted lumbar spine, vertebral compression fractures",
    licence="CC-BY-4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    doi="10.5281/zenodo.13274445",
    source_url="https://zenodo.org/records/13274445",
    citation=(
        "Chiari-Correia N, Nogueira-Barbosa MH, Azevedo-Marques PM de, Chiari-Correia R, Cordeiro SS, "
        "Lima LL de, et al. Lumbar Spine Vertebral Compression Fractures (VCFs) Dataset: MRI T1-Weighted "
        "Images for Benign and Malignant Classification. Zenodo (2024). doi:10.5281/zenodo.13274445"
    ),
    description="Anonymised sagittal T1 lumbar MRI of benign and malignant vertebral compression fractures.",
    subdir="lumbar_vcf_mri_hcfmrp",
    files=(
        FileSpec(
            "HCFMRP_Complete_Anon_VCF_Database.rar",
            "https://zenodo.org/api/records/13274445/files/HCFMRP_Complete_Anon_VCF_Database.rar/content",
            1304730896,
            md5="90679fbac00f0f99acc38bc1c6c4051e",
        ),
    ),
    notes=(
        "RAR archive: download with fetch(), then extract with unrar or 7z (the built-in extractor only "
        "reads zip). Related article: doi:10.1007/s10278-023-00847-4."
    ),
)


def _manifest_files(name: str) -> tuple[FileSpec, ...]:
    """File list (path, size, md5, sha256, URL) shipped as package data under ``datasets/manifests``."""
    import json
    from importlib import resources

    raw = json.loads((resources.files(__package__) / "manifests" / name).read_text(encoding="utf-8"))
    return tuple(
        FileSpec(name=f["path"], url=f["url"], size=int(f["size"]), md5=f["md5"], sha256=f["sha256"])
        for f in raw["files"]
    )


LUMBAR_MULTISCANNER_QX5RT = DatasetSpec(
    key="lumbar_multiscanner_qx5rt",
    title="Multi-scanner and multi-modal lumbar vertebral body and intervertebral disc segmentation database",
    modality=(
        "MRI NIfTI (.nii.gz), lumbar T1 / T2 / Dixon / STIR, manual binary masks of vertebral bodies "
        "L1-L5 and discs L1/2-L4/5 per session; 34 patients, 55 sessions"
    ),
    licence="CC-BY-NC-4.0",
    licence_url="https://creativecommons.org/licenses/by-nc/4.0/",
    doi="10.17605/OSF.IO/QX5RT",
    source_url="https://osf.io/qx5rt/",
    citation=(
        "Al Khalil Y, et al. Multi-scanner and multi-modal lumbar vertebral body and intervertebral disc "
        "segmentation database. Scientific Data 9, 97 (2022). doi:10.1038/s41597-022-01222-8. Dataset: "
        "doi:10.17605/OSF.IO/QX5RT"
    ),
    description=(
        "Manually segmented lumbar MRI from several scanners: Siemens (Espree, Verio, Avanto, Aera, "
        "Magnetom, Symphony, Amira), Philips (Achieva, Ingenia, Elition) and one GE Signa session "
        "(ID38, first scan). Masks are drawn on the T1 image; other sequences are registered to it."
    ),
    subdir="lumbar_multiscanner_qx5rt",
    extra={"osf_node": "bj69m", "field_strength": "not stated in the release"},
    files=_manifest_files("osf_qx5rt_data.json"),
    notes=(
        "Licence read from the project's license.txt (Attribution-NonCommercial 4.0); the OSF licence field "
        "is empty and some aggregators list CC BY 4.0, so NC is used. Research use only; do not "
        "redistribute the data. 700 files, 4.27 GB, each below the 2 GiB cap. Files keep their OSF folder "
        "layout under raw/: ID<nn>/<Vendor_Model>/{Images,Labels}/<file>.nii.gz. Disc label names are "
        "BS_L<a>_<b> (disc between L<a> and L<b>). The L5-S1 disc is not labelled."
    ),
)

#: Public spine MRI datasets, keyed by ``DatasetSpec.key``. Licences were checked at the source.
REGISTRY: dict[str, DatasetSpec] = {
    d.key: d
    for d in (
        SPIDER,
        SUDIRMAN_LUMBAR_MRI,
        SUDIRMAN_LUMBAR_LABELS,
        SUDIRMAN_RADIOLOGIST_NOTES,
        VCF_MRI_HCFMRP,
        LUMBAR_MULTISCANNER_QX5RT,
    )
}

#: Considered and not registered: LumASe (Zenodo 7181338, CC-BY-4.0) is CT, not MRI.


def get_spec(key: str) -> DatasetSpec:
    """The registry entry for ``key``; raises KeyError listing the known keys."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(REGISTRY)}") from None
