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
| `lumbar_spine_mri_sudirman` | 515 anonymised clinical DICOM studies (Siemens 1.5 T, sagittal and axial), no labels in the zip | CC BY 4.0 | Mendeley Data, doi:10.17632/k57fr854j2.2 | 6.27 GB (one zip) | On disk, extracted at `~/data/public/lumbar_spine_mri_sudirman/01_MRI_Data` (fetch with `allow_large=True`) |
| `lumbar_spine_mri_sudirman_labels` | Axial T1 label maps (IVD, posterior elements, thecal sac, AAP) for discs D3 to D5 of 515 patients, with the reference slices | CC BY 4.0 | Mendeley Data, doi:10.17632/zbf6b4pttk.2 | 1.04 GB + 0.64 GB (two zips) | On disk at `~/data/public/lumbar_spine_mri_sudirman_labels` |
| `lumbar_spine_mri_sudirman_notes` | Radiologist notes per patient (xlsx) | CC BY 4.0 | Mendeley Data, doi:10.17632/s6bgczr8s2.2 | 38 kB | On disk at `~/data/public/lumbar_spine_mri_sudirman_notes` |
| `lumbar_vcf_mri_hcfmrp` | sagittal T1 lumbar MRI of benign and malignant vertebral compression fractures | CC BY 4.0 | Zenodo, doi:10.5281/zenodo.13274445 | 1.3 GB (RAR) | Not downloaded; RAR needs `unrar` or `7z` |
| `lumbar_multiscanner_qx5rt` | 34 patients, 55 lumbar MRI sessions from Siemens (15), Philips (39) and one GE Signa; NIfTI T1 / T2 / Dixon / STIR; binary masks of vertebral bodies L1-L5 and discs L1/2-L4/5 | **CC BY-NC 4.0** (non-commercial) | OSF, doi:10.17605/OSF.IO/QX5RT | 4.27 GB, 700 files | Downloaded to `~/data/public/lumbar_multiscanner_qx5rt/raw` (md5 per file, from the OSF API) |

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
- Layout after extraction: `01_MRI_Data/<patient>/<session>/<series>/*.ima`. 515 patients, 558 sessions
  (33 patients with two sessions, 5 with three), 3,761 series folders, 48,345 DICOM files.
- Technical content, all sessions: one vendor and field strength, **Siemens MAGNETOM Essenza, 1.5 T**.
  Main series: sagittal T1 TSE (320), sagittal T2 TSE (384), axial T1 TSE and axial T2 TSE (384), each
  split into slabs by tilt (2 to 12 slabs per session, 3 in most). Also localizers, Siemens "PosDisp"
  position-display copies, and a few contrast, Dixon, DWI, STIR and MIP series.
- The two related IEEE articles (documents 8622977 and 8678637) are listed on the page; their citations
  are not verified here.
- The zip has no labels. Disc labels come from the companion sets below.

### Sudirman label set and radiologist notes

- `lumbar_spine_mri_sudirman_labels` (doi:10.17632/zbf6b4pttk.2, CC BY 4.0): "Label Image Ground Truth
  Data for Lumbar Spine MRI Dataset", 1.04 GB and 0.64 GB zips. Per patient and disc `D3`, `D4`, `D5`:
  a 320 x 320 label map in `05_Final_Ground_Truth_Data/Label_Images` (50 intervertebral disc, 100
  posterior elements, 150 thecal sac, 200 area between anterior and posterior elements) and the axial T1
  slice it was drawn on in `04_Intermediary_Ground_Truth_Data/T1_Output` (the exact DICOM pixels).
  The disc names (D3 = L3-L4, D4 = L4-L5, D5 = L5-S1) are not in the files; see
  `spinescope.datasets.sudirman` and `docs/labeling/lab02_sudirman.md`.
- `lumbar_spine_mri_sudirman_notes` (doi:10.17632/s6bgczr8s2.2, CC BY 4.0): `Radiologists Report.xlsx`,
  free-text notes per patient. Read with `spinescope.datasets.sudirman.read_radiologist_notes`.
- Both are registered and fetched with `datasets.fetch(key)`; each file is below the 2 GiB cap.
- Used for: disc level-name ground truth and `label_spine` scoring (`docs/labeling/lab02_sudirman.md`).

### Lumbar vertebral compression fractures MRI (HCFMRP)

- Licence: CC BY 4.0. Zenodo record 13274445 (2024-08-08).
- Citation: Chiari-Correia N, Nogueira-Barbosa MH, Azevedo-Marques PM de, Chiari-Correia R, Cordeiro SS,
  Lima LL de, et al. Lumbar Spine Vertebral Compression Fractures (VCFs) Dataset: MRI T1-Weighted Images
  for Benign and Malignant Classification. Zenodo (2024). doi:10.5281/zenodo.13274445. Related article:
  doi:10.1007/s10278-023-00847-4.
- Archive is RAR, which the built-in extractor does not read. Extract it with `unrar` or `7z` into
  `$SPINESCOPE_DATA/lumbar_vcf_mri_hcfmrp`.

### Multi-scanner lumbar vertebral body and disc database (OSF qx5rt)

- Licence: **Attribution-NonCommercial 4.0** (CC BY-NC 4.0). Read from the project's `license.txt`
  (OSF component `jd2sv` and `bj69m`). The project's licence field on OSF is empty, and some
  aggregators list CC BY 4.0. The file is the authority, so research use only, no commercial use.
- Citation: Al Khalil Y, et al. Multi-scanner and multi-modal lumbar vertebral body and intervertebral
  disc segmentation database. *Scientific Data* 9, 97 (2022). doi:10.1038/s41597-022-01222-8. Dataset:
  doi:10.17605/OSF.IO/QX5RT.
- Layout (component `bj69m`, "Data"): `ID<nn>/<Vendor_Model>/Images/*.nii.gz` and `.../Labels/*.nii.gz`.
  Vendors per session: Philips 39 (Achieva, Ingenia, Elition, one "Achieva (external)"), Siemens 15
  (Espree, Verio, Avanto, Aera, Magnetom, Symphony, Amira), GE 1 (`ID38/GE_Signa`, scan 1 of 2).
  Field strength is not in the release (the scan table gives vendor and model only).
- Labels: `Labels/L1..L5` are vertebral bodies (posterior elements excluded). `Labels/BS_L<a>_<b>` are
  discs (`BS_L1_2` = L1-L2 ... `BS_L4_5` = L4-L5). The L5-S1 disc is not labelled. Masks are binary
  (stored as float with ~1e-14 noise, so threshold at 0.5). The paper says the masks were drawn on the
  non-contrast T1 and the other sequences registered to it; this is not re-checked here. For the T1 used here, the NIfTI grid is sagittal:
  about 15 slices of 3.3 mm, in-plane 0.35 to 0.78 mm.
- The scan characteristics table (`Image_scan_characteristics.pdf`, component `jd2sv`) is the only
  vendor source. The patient characteristics files are not downloaded.
- Scorer: `spinescope.anatomy._validate_qx5rt` (level-wise, see `docs/labeling/lab10_vendors.md`).

### Considered and not registered

- LumASe (Zenodo 7181338, CC BY 4.0): lumbar vertebra segmentation from CT, so it is out of scope for
  an MRI toolkit.
- SPIDER vendor mix, checked on the release's technical columns (`Manufacturer`,
  `ManufacturerModelName`, `MagneticFieldStrength`; patient columns are not read): 447 series,
  Siemens 236, Philips 211, no GE. Field 1.5 T for 253 series and 3.0 T for 194.

The survey below was done on 2026-11-14. Each row was checked against its hosting page or paper, and
none of them is in the registry unless it is listed above.

| Candidate | Vendors and field | Annotations | Licence | Access | Decision |
|---|---|---|---|---|---|
| AISSLab LSS MRI (Mendeley `rgb77xm3jf` v4, doi:10.17632/rgb77xm3jf.4), 500 patients | GE Signa HDxt 1.5 T and GE Signa Excite 1.5 T, plus Philips (per the paper) | Mid-sagittal vertebra and disc masks, sacrum; full series DICOM | Record says CC BY 4.0, but the text says scientific / non-commercial use. The two conflict. | "Download All" needs a login; the anonymous file list was empty | **Not downloaded.** Strongest GE cohort, but the licence is unclear and access needs a login. Ask the authors for written terms. |
| LSMA-PQR (Mendeley `p3r4xd2488` v2, doi:10.17632/p3r4xd2488.2), 515 patients | Siemens MAGNETOM Essenza 1.5 T, inherited from the Sudirman source dataset | 2D PNG, vertebra and disc polygons, Pfirrmann grades | CC BY 4.0 | Mendeley | **Not used.** Same single vendor and scanner as the Sudirman set, and 2D slices only. |
| Spine Generic (Zenodo 15028452, `spine-generic/data-multi-subject` r20250314; OpenNeuro ds001919) | Siemens, Philips and GE 3 T, harmonised protocol, 267 participants, 42 centres | Cervical cord segmentations and labels; not lumbar discs or vertebrae | CC BY 4.0 in `LICENSE`, but `dataset_description.json` says CC0. The two conflict. | git-annex (the Zenodo record is 13 MB of metadata) | **Not used.** Cervical, so no fit for the lumbar labeller. Useful later for cross-vendor intensity checks of the cord code. |
| RSNA LumbarDISC (Kaggle, RSNA 2024 competition; doi:10.1148/ryai.250480) | Per the paper's supplementary Table S1, reported in a search summary and not opened here: Siemens 66.4 %, GE 23.1 %, Toshiba 142, Hitachi 18, Canon 3 | Disc-level localisers and stenosis grades; no vertebra or disc masks | Competition rules: non-commercial, no redistribution. Kaggle account needed. | Kaggle | **Not used** (Kaggle needs an account and the competition rules apply). The Kaggle headers omit the manufacturer, so the vendor of each study cannot be read from them. |
| Toshiba-containing lumbar cohort (Frontiers Radiology 2026, doi:10.3389/fradi.2026.1722312) | Siemens, Philips, GE and Toshiba (27 of 165 studies) | Vertebra and disc masks | Not stated for public use | Authors will share raw data on request; no public repository found | **Not available** (author contact only). |
| Hitachi Airis Vento 0.3 T cohort (SpineNetV2 external validation, PMC11617751) | Hitachi 0.3 T, 353 examinations | Radiologist disc-pathology grades; no masks | Not open | No open download found | **Not available.** |
| Figshare lumbosacral nerve-root MRI (doi:10.6084/m9.figshare.c.7372564) | Siemens MAGNETOM Prisma 3 T, 14 adults | Nerve roots and cord, not vertebra or disc masks | Open access | Figshare | **Not used.** Wrong target structures and one vendor. |
| MRSpineSeg (challenge data) | Not documented | Vertebra and disc masks | Original site down; the preprocessed copy is on a Baidu link | Corresponding author on request | **Not used.** No verified open download. |
| TCIA | Not reachable from this host (the services endpoint did not respond) | No lumbar MRI with vertebra and disc masks found in search | n/a | n/a | **Not used.** Spine-Mets-CT-SEG is CT. |
| OpenNeuro ds005143 (spinal cord fMRI) | GE, EPI | Cord masks on EPI; not vertebral | Not checked | OpenNeuro | **Not used.** Cord function, not lumbar bone or disc. |
| PhysioNet | Reachable; no open lumbar MRI with vertebra or disc labels found | n/a | n/a | n/a | **Not used.** |
| Kaggle other spine MRI competitions | not searched in depth | n/a | Competition rules | Kaggle account | **Not used** (account and rules). |

Still missing: a 1.5 T or 3 T GE lumbar cohort with disc labels that is openly downloadable, and any
Canon, Hitachi, Fujifilm, United Imaging, Esaote or low-field (Hyperfine, 0.064 T) lumbar set with labels.
The searches found none of these with a licence that allows download.

## Safety

- Archives are checked member by member before anything is written. Absolute paths, `..` components,
  drive letters and symbolic links are rejected, and a rejected archive leaves no files behind
  (`spinescope.datasets.safe_extract_zip`).
- The registry stores each file's size and checksum, and a download that does not match is deleted.
- Nothing from the datasets is written into the repository. Outputs carry no file paths or identifiers.
