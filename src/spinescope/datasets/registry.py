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

#: Public spine MRI datasets, keyed by ``DatasetSpec.key``. Licences were checked at the source.
REGISTRY: dict[str, DatasetSpec] = {d.key: d for d in (SPIDER, SUDIRMAN_LUMBAR_MRI, VCF_MRI_HCFMRP)}

#: Considered and not registered: LumASe (Zenodo 7181338, CC-BY-4.0) is CT, not MRI.


def get_spec(key: str) -> DatasetSpec:
    """The registry entry for ``key``; raises KeyError listing the known keys."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(REGISTRY)}") from None
