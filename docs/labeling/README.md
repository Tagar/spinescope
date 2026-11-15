# Labelling lab reports: index

Start with [pipeline.md](pipeline.md): the stage order as it runs, and every place that can change the numbering status.

Each lab report records one piece of work on the level labeller: what was built or measured, and what was decided.
The reports are kept as they were written. Their numbers are from the commit each one was run on, and some predate the
learned labeller (the default since `1c9f553`). For the current numbers read `docs/validation.md`. For the rejected
candidates in one place read `docs/validation.md`, section 6.

Outcome key:

- **shipped**: on the default path, or a default tool or test.
- **opt-in**: code on `main`, off by default.
- **experimental**: on, but marked as not validated.
- **rejected**: evaluated and not shipped, or shown to be worse.
- **negative**: a result that says the idea does not help as tried.
- **baseline**: a measurement of an earlier labeller, kept for comparison.
- **superseded**: the idea or number is replaced by a later report; the report is kept as history.

| report | what it did | outcome |
|---|---|---|
| [lab01_eval](lab01_eval.md) | QA overlays, eval CLI and a failure taxonomy for the classical labeller on SPIDER validation | baseline; superseded |
| [lab02_sudirman](lab02_sudirman.md) | ran the classical labeller on the 515-patient Sudirman DICOM set to test the loader and the naming | baseline; superseded |
| [lab03_robustness](lab03_robustness.md) | 23-scenario synthetic phantom suite that any labelling backend must pass | shipped (tests and `tools/robustness_report.py`) |
| [lab04_numbering](lab04_numbering.md) | numbering cross-check evidence: slab, tilt, conus and LSTV cues (`numbering_checks.py`) | shipped as evidence used by fusion and LSTV; slab cue off by default |
| [lab05_midline](lab05_midline.md) | mid-sagittal plane by mirror symmetry; SPIDER validation midline median 11.7 to 1.4 mm | shipped in the classical path only |
| [lab06_verify](lab06_verify.md) | label verification packets, a correction API with provenance, and a reader test on 20 volumes | shipped (`verify-labels`); the reader test did not improve label accuracy then |
| [lab08_fusion](lab08_fusion.md) | numbering fusion votes, calibrated confidence, and the `confirmed` / `probable` / `unconfirmed` policy | shipped (policy and calibration v1, later replaced by v2) |
| [lab09_errors](lab09_errors.md) | error analysis of the learned labeller; post-processing fixes (SPIDER validation label accuracy 95.5 % to 97.6 %) | shipped |
| [lab10_vendors](lab10_vendors.md) | OSF qx5rt multi-vendor set (CC BY-NC, numbers only) as an external check, with a vendor breakdown | shipped as a benchmark suite |
| [lab12_ood](lab12_ood.md) | applicability check: when not to label (wrong region, short field of view, unseen input) | shipped |
| [lab13_lumbosacral](lab13_lumbosacral.md) | lowest-disc CNN (L5-S1 versus sacral discs), applied before numbering; a feature logistic was tried | shipped (CNN); feature logistic negative |
| [lab14_perf](lab14_perf.md) | one network pass per study, thread control, memory and import time; classical second opinion made opt-in | shipped; second opinion opt-in |
| [lab15_vertebrae](lab15_vertebrae.md) | vertebral body, canal and posterior-element geometry from the network segmentation | shipped (learned backend only; schema 1.1) |
| [lab16_conus](lab16_conus.md) | conus medullaris locator, first version | experimental |
| [lab17_orientation](lab17_orientation.md) | rotated volumes, slice-axis choice, plane ambiguity and the paired T1+T2 path (SPIDER paired "no sagittal" verdicts 7 to 0) | shipped (I/O and SPIDER loader) |
| [lab18_reliability](lab18_reliability.md) | calibration v2 for the numbering status; a refit of the disc model was rejected | shipped (calibration v2); disc refit rejected |
| [lab19_surface](lab19_surface.md) | one abstain policy and one wording on every surface (CLI, MCP, briefs, viewer, skills) | shipped |
| [lab20_axial](lab20_axial.md) | disc support and missed-disc candidates from oblique axial T1 slabs | opt-in (off); not recommended as default until refit |
| [lab21_phantom](lab21_phantom.md) | realistic synthetic lumbar phantom (analytic, no real images) and its robustness suite | shipped (test asset); the learned labeller is unconfirmed in most scenarios |
| [lab22_confirmed](lab22_confirmed.md) | makes `confirmed` reachable (bone below the lowest disc); calibration v3 | shipped (evidence order); calibration v3 rejected, v2 kept |
| [lab23_redteam](lab23_redteam.md) | 40 adversarial inputs (corrupted DICOM, geometry, contrast mismatch); regression tests | shipped (fixes and tests) |
| [lab24_ai_verify](lab24_ai_verify.md) | blinded test of an AI reader checking level names on 196 cases, with a committed hash before the key | negative for correction (0 of 49 fixed); it found a policy defect, fixed in `2f86036` |
| [lab26_conus_v2](lab26_conus_v2.md) | conus locator v2 and a reference set marked by an AI reader | stays experimental |
| [lab28_lstv](lab28_lstv.md) | transitional vertebra (lumbosacral) suggestion, gated on numbering status | shipped as a suggestion; not calibrated |
| [lab29_correspondence](lab29_correspondence.md) | which slices show each labelled level in every stack (`level_map`) | shipped (briefs and MCP); changes no names |
| [lab31_artifacts](lab31_artifacts.md) | hardware and image-artefact flags per level and per study | shipped; advisory flags hidden by default |
| [lab34_counting](lab34_counting.md) | top-chain count from wide sagittal coverage | opt-in; no landmark, so no vote |
| [lab35_uncertainty](lab35_uncertainty.md) | test-time augmentation as a numbering-uncertainty signal | rejected (not shipped; off in code) |

Reports without a file in this folder:

| work | where it is documented |
|---|---|
| lab07 `labeller_v2` (sacrum class) | rejected; `docs/model_card.md`, section 9 |
| lab11 viewer label status banner and correction tools | shipped; `docs/guide/correcting-labels.md` |
| lab25 `labeller_v3` (2.5-D input and ablations) | rejected; `docs/model_card.md`, section 9 |
| lab27 one-command benchmark and golden check | shipped; `docs/benchmarks.md` |
| lab32 human correction end to end | shipped; `docs/guide/correcting-labels.md` |
| lab33 canary audit of outputs | shipped; `docs/privacy.md` |

Reports cite datasets by name and licence only. No report contains images from the datasets. Scratch outputs were under
`/tmp` and are not committed.
