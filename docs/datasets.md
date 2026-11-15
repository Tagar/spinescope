# Public spine MRI datasets

`spinescope.datasets` keeps a registry of public spine MRI datasets, downloads them with checksums,
and loads the ones with labels. Only datasets whose licence was checked at the source are registered.
All data lives under `$SPINESCOPE_DATA` (default `~/data/public`), one folder per dataset, and is
never committed to the repository.

```python
from spinescope import datasets

datasets.REGISTRY["spider"].licence  # "CC-BY-4.0"
datasets.fetch("spider", allow_large=True)  # images.zip is 3.7 GB
for study in datasets.load_spider_studies(load=True):
    vol = study.volume("t2")  # Stack, sagittal, display orientation
    mask = study.mask("t2")  # [slice, row, col] labels on the same grid
    grades = study.gradings  # one dict per disc (Pfirrman, Modic, ...)
```

Command-line equivalent: none yet. Downloads use `urllib`, so `http_proxy`, `https_proxy` and
`no_proxy` from the environment are honoured. Interrupted downloads resume from the `.part` file with
an HTTP `Range` request, and a file is only renamed into place after its size and checksum match.
A single download above 2 GiB is refused unless you pass `allow_large=True`.

## Registered datasets

| Key | Content | Licence | Source | Size | Status |
|---|---|---|---|---|---|
| `spider` | 218 patients, 447 sagittal T1 / T2 / T2 SPACE volumes (`.mha`), vertebra, disc and canal masks, per-disc gradings | CC BY 4.0 | Zenodo, doi:10.5281/zenodo.10159290 | 3.7 GB images, 58 MB masks, 0.15 MB CSV | On disk at `~/data/public/spider` (md5 verified by the release) |
| `lumbar_spine_mri_sudirman` | 515 anonymised DICOM studies (sagittal and axial), no labels | CC BY 4.0 | Mendeley Data, doi:10.17632/k57fr854j2.2 | 6.27 GB (one zip) | Not downloaded (above the 2 GiB cap) |
| `lumbar_vcf_mri_hcfmrp` | sagittal T1 lumbar MRI of benign and malignant vertebral compression fractures | CC BY 4.0 | Zenodo, doi:10.5281/zenodo.13274445 | 1.3 GB (RAR) | Not downloaded; RAR needs `unrar` or `7z` |

Licences were read from the hosting records on 2026-11-14 (Zenodo API `license` field, Mendeley
Data dataset page "Licence CC BY 4.0"). Re-check before a public release.

### SPIDER (lumbar spine segmentation in MR images)

- Licence: Creative Commons Attribution 4.0 (<https://creativecommons.org/licenses/by/4.0/>).
- Citation: van der Graaf JW, et al. Lumbar spine segmentation in MR images: a dataset and a public
  benchmark. *Scientific Data* (2024). doi:10.1038/s41597-024-03090-w. Dataset:
  doi:10.5281/zenodo.10159290.
- Layout after extraction: `images/<case>_<seq>.mha` (`seq` is `t1`, `t2` or `t2_SPACE`),
  `masks/<case>_<seq>.mha` on the same grid, `overview.csv`, `radiological_gradings.csv`.
- Mask labels observed on the release: vertebrae 1..N, spinal canal 100, intervertebral discs
  201..200+N. The loader exposes these as `CANAL_LABEL`, `VERTEBRA_LABELS` and `DISC_LABEL_OFFSET`.
- Patient-identifying columns of `overview.csv` (sex, birth date, device serial number) are never read.
- Checked on 40 cases: image and mask headers agree (size, spacing, origin, direction) to within 1e-4 mm
  (float32 storage), so the same voxel indices address both. A mid-sagittal overlay of vertebrae, discs
  and canal lined up with the anatomy after loading.
- Tests: `tests/test_datasets_spider.py` runs on a synthetic tree with the same layout. A test marked
  `dataset` runs on the real release when it is on disk and is skipped otherwise.

### Lumbar Spine MRI Dataset (Sudirman et al.)

- Licence: CC BY 4.0. Mendeley Data, version 2 (published 2019-04-03).
- Citation: Sudirman S, Al Kafri A, Natalia F, Meidia H, Afriliana N, Al-Rashdan W, Bashtawi M,
  Al-Jumaily M. Lumbar Spine MRI Dataset. Mendeley Data, V2 (2019). doi:10.17632/k57fr854j2.2.
- One 6,270,535,053-byte zip (`MRI_Data.zip`, SHA-256 `ac6495c2...eadc1a`). The download URL is
  resolved at fetch time from the Mendeley files API. The dataset is refused by default, so run
  `datasets.fetch("lumbar_spine_mri_sudirman", allow_large=True)` only when you have 6.3 GB free.
- The page lists two related IEEE articles (documents 8622977 and 8678637). Their citations are not
  verified here.
- No labels are included, so it is useful for loader and classification tests, not segmentation
  validation.

### Lumbar vertebral compression fractures MRI (HCFMRP)

- Licence: CC BY 4.0. Zenodo record 13274445 (2024-08-08).
- Citation: Chiari-Correia N, Nogueira-Barbosa MH, Azevedo-Marques PM de, Chiari-Correia R, Cordeiro SS,
  Lima LL de, et al. Lumbar Spine Vertebral Compression Fractures (VCFs) Dataset: MRI T1-Weighted Images
  for Benign and Malignant Classification. Zenodo (2024). doi:10.5281/zenodo.13274445. Related article:
  doi:10.1007/s10278-023-00847-4.
- Archive is RAR, which the built-in extractor does not read. Extract it with `unrar` or `7z` into
  `$SPINESCOPE_DATA/lumbar_vcf_mri_hcfmrp`.

### Considered and not registered

- LumASe (Zenodo 7181338, CC BY 4.0): lumbar vertebra segmentation from CT, so it is out of scope for
  an MRI toolkit.

## Safety

- Archives are checked member by member before anything is written. Absolute paths, `..` components,
  drive letters and symbolic links are rejected, and a rejected archive leaves no files behind
  (`spinescope.datasets.safe_extract_zip`).
- The registry stores each file's size and checksum, and a download that does not match is deleted.
- Nothing from the datasets is written into the repository. Outputs carry no file paths or identifiers.
