# lab12_ood: when not to label (applicability)

Branch `lab/lab12_ood`. Owns `anatomy/ood.py` (new), a small hook in `anatomy/pipeline.py`
(`label_spine_details`), `anatomy/ml/labeller.py` (`label_spine_ml_segmented`, the same labeller
exposing the network output; `label_spine_ml` unchanged), the additive `Anchors.applicability` field,
and the `Config` block `ood_*`. Tests: `tests/test_anatomy_ood.py`. Dev tool:
`tools/eval_applicability.py`. Raw outputs stay in `/tmp/spinescope-lab12_ood`; nothing derived from
SPIDER is committed.

## 1. What it does

`label_spine` names levels by counting discs up from the lowest one. That is only right for a lumbar
study whose lowest discs are in view. `assess_applicability` decides from cheap, explainable cues
whether a study is one, and `apply_applicability` writes the decision into the anchors.

Verdicts (`anchors.applicability['verdict']`):

| Verdict | Labelled | When |
|---|---|---|
| `lumbar` | yes | lumbar-sized discs only |
| `thoracolumbar` | yes, thoracic discs above T10 capped | lumbar discs plus thin thoracic discs |
| `whole_spine` | yes, cervical discs dropped | two or more cervical-sized discs above lumbar discs |
| `uncertain` | yes, confidences lowered | no region identified by disc size or spacing |
| `not_lumbar` | **no** | cervical or thoracic column, or a non-spine body part in `BodyPartExamined` |
| `not_spine` | **no** | fewer than two discs, no canal, canal absent at the disc levels |
| `no_sagittal` | **no** | no usable sagittal series |
| `not_usable` | **no** | sagittal series with fewer than 3 slices or a z extent under 40 mm |

Not-labelled results carry `numbering.status = 'unconfirmed'`, confidence 0, no discs, vertebrae or
conus, and a summary such as `not a lumbar spine study (region: brain); levels not labelled`.

Cues, all technical or numeric:

- **Metadata.** `BodyPartExamined` (strong) and `SeriesDescription`/`ProtocolName` (weak), matched
  against words such as `LSPINE`, `LWK`, `TSPINE`, `CSPINE`, `BRAIN`, `KNEE`. Only the matched words
  are kept. A strong non-spine name rejects. A strong thoracic or cervical name rejects only when the
  anatomy has no lumbar disc; a lumbar disc in the anatomy overrides it (warning written).
- **Disc size and spacing.** Disc heights and centre-to-centre spacings on the mid-sagittal chain.
  A disc is cervical only when it is thin, all its spacings are short, and it has two spacings or a
  cervical neighbour. Lumbar when any spacing is long. A FOV is small (non-lumbar) only when the median
  height and spacing are small, the two lowest discs are thin, and no disc is lumbar-tall.
- **Canal.** Canal run on the mid-sagittal slice, and the fraction of disc levels with canal within
  10 mm (`canal_at_discs`).
- **Entropy.** Mean normalised entropy of the network's class distribution over anatomy pixels.
- **Long body.** A body between two adjacent discs longer than `ood_long_body_mm` means a disc was
  missed; numbering is forced to `unconfirmed`.
- **Close discs.** Adjacent discs closer than cervical spacing (22 mm) inside a labelled chain force
  `unconfirmed` (a lumbar column has no such spacing).
- **Field of view.** z extent of the sagittal stack (LPS), bone visible below the lowest disc
  (`bone_below_lowest_disc_mm`), `sacrum_covered`, `superior_margin_mm`, `top_level_in_view`, and
  `levels_in_fov`. A FOV whose top sits inside a body reports that level; levels above it are
  outside the view, not missed.
- **Acquisition.** Contrast other than T1 or T2 (STIR, PD, unknown), Dixon, post-contrast, post-contrast
  fat-suppressed T1, field strength below 1.0 T, and near-isotropic voxels (SPIDER T2 SPACE is
  0.47 x 0.47 x 0.9 mm, so it is not flagged). Each flagged cue multiplies confidences by 0.6, floor 0.3.
- **Thoracic landmark.** Discs above T10 (upper vertebra T10 or higher) are capped at 0.3 unless a
  description names that disc (`T10-T11` in a slab name counts as a landmark).

Confidence effects: disc and numbering confidences are multiplied by `applicability['factor']`; the
status is recomputed with the calibrated thresholds. `numbering.confidence_before_applicability` keeps
the value before the factor.

## 2. Thresholds and where they come from

Anatomy defaults (judgement, from anatomy texts, in `Config.ood_*`): lumbar disc height 7 mm, cervical
spacing 22 mm, lumbar spacing 30 mm, FOV margin 18 mm, canal fraction 0.5, canal tolerance 10 mm, sacrum
45 mm (one lumbar body plus the first sacral segment), minimum discs 2.

Calibrated on SPIDER training (153 studies with T1 and T2; mid-sagittal network output):

| Cue | Training distribution | Value set |
|---|---|---|
| Body interval between discs (mm) | median 30, 95th 37, 99th 41, max 65 | `ood_long_body_mm` 45 (above the 99th percentile; a two-body merge is ~55-65) |
| Entropy over anatomy pixels | 99th 0.17, max 0.175 | `ood_entropy_max` 0.2 (just above the training maximum) |
| Disc spacing (mm) | 1st 23, 5th 27, 25th 32, median 34, 99th 41 | cervical 22, lumbar 30 (kept as anatomy defaults; they sit inside the observed range) |
| Canal at disc levels | 5th 0.71, min 0.57 | `ood_canal_fraction_min` 0.5 (no training study rejected) |
| Bone below lowest disc (mm) | 5th 38, 25th 54, median 62 | `ood_sacrum_min_mm` 45 (about 9-10 % of training FOVs fall below it) |

The first calibration run used a 32 mm body limit (the classical labeller's rule). It flagged roughly a
quarter of real intervals and forced 129 of 153 numberings to `unconfirmed`, so it was replaced by the
training-distribution value above. This is the only threshold changed after seeing the data, and it was
set on training only.

## 3. Results

Calibration and held-out check on SPIDER (ML labeller, gated output of `label_spine_details`):

| Split | Studies | Verdicts | Real false rejections |
|---|---|---|---|
| training | 153 (paired T1+T2) | 102 lumbar, 38 thoracolumbar, 13 no_sagittal | 0 of 140 evaluable |
| validation | 38 with sagittal T2 | 21 lumbar, 10 thoracolumbar, 7 no_sagittal | 0 of 31 evaluable |

Notes:

- The 13 training and 7 validation `no_sagittal` studies have T1 and T2 volumes whose slices run along
  y, so the geometry classifies them as coronal and the labeller has no sagittal series. They fail
  without applicability too. They are not counted as rejections, and are reported separately in
  `tools/eval_applicability.py` output.
- The SPIDER validation folder has 87 volumes (38 sagittal T2, 11 T2 SPACE, the rest T1). The run uses all
  38 T2 cases; T1-only cases are not labelled by the labeller (T2 is primary).
- The training run used the thresholds above; the final small-FOV tightening (all discs thin, the "one
  lumbar-sized disc keeps the study lumbar" rule) can only reduce rejections, and training was not rerun
  for it. Validation was run after it.
- Thoracolumbar verdicts: 38 of 153 training and 10 of 38 validation studies have thin thoracic discs
  above lumbar discs; their lumbar labels are kept and the thoracic discs above T10 are capped.
- Numbering on validation: of the 31 labelled studies, 21 `confirmed`, 2 `probable`, 8 `unconfirmed`;
  the sacrum is not in view for 7 of the labelled studies. The 7 no-sagittal studies are not labelled.

Synthetic tests (`tests/test_anatomy_ood.py`, 63 tests; 43 pure-logic, 20 with the learned labeller):

- Non-spine: brain-like ellipses rejected (`not_spine`); phantom crop anterior to the spine rejected;
  strong `BRAIN` metadata rejected; random structures (seeded) never reach a confirmed numbering.
- Non-lumbar phantom variants (cervical-like: body 14 mm, disc 5 mm; thoracic-like: body 22 mm, disc
  5 mm, full and crop): not labelled.
- FOV: FOV ending at L5 with truth anchors has less bone below L5-S1 than the full protocol; a
  superior cut at L2 names the top level.
- Untrained inputs: STIR-only, PD-only and 3-D isotropic are warned and lowered; low field, Dixon and
  post-contrast are warned (`acquisition_cues`) and tested directly.
- Thoracolumbar FOV: discs above T10 capped, lumbar labels kept.
- Privacy: the applicability record contains no fake PHI and no file path.

## 4. Failures the module does not catch

- **Missed bottom disc in a cropped FOV** (`fov_L1_L5` phantom): the network places its lowest disc
  about 2 levels too high (discs at spacing 17-20 mm). With the close-disc rule the numbering is
  `unconfirmed`, but the sacrum check passes because the bone below that wrong disc is long. The labels
  (L5-S1 for the true L3-L4) would be wrong if confirmed. Cheap cues cannot separate a missed bottom disc
  from a sacrum, because the labeller has no sacrum class.
- **`fov_L2_S1` phantom**: the network misses L5-S1 and labels L4-L5 as L5-S1, with status `confirmed`
  (0.98). No cue here fires. This is a labeller error to be reported to the coordinator.
- **Thin bottom discs in a degenerated lumbar FOV**: a FOV where every disc is thin and the median
  spacing is below 30 mm would be read as thoracic. A single lumbar-sized disc prevents this.
- Random blobs can produce a canal-like track and discs; the module lowers their confidence and
  (through the close-disc and long-body cues) usually leaves them `unconfirmed`, but a random input can
  still be labelled `uncertain` or `lumbar` with factor 1.0 if no cue fires. This is not a confident
  lumbar result, but it is not a rejection either.
- The phantom is out of distribution for the network (see WAVE_LABELING); phantom cervical and thoracic
  variants test the rules, not realistic anatomy.

## 5. Data check and what was not done

- Local public data under `~/data/public`: `spider` and `lumbar_spine_mri_sudirman` only. There is no
  spine-generic, brain, knee or other non-lumbar set, so the non-spine cases are synthetic (brain-like
  ellipses, random structures, phantom crops) and the real-data check is lumbar only. Nothing was
  downloaded.
- Sudirman lumbar DICOM was not run through applicability (time).
- Thoracic-only and cervical real studies were not available; the discs and spacings for those regions
  are judgement defaults.

## 6. Running it

    .venv/bin/python tools/eval_applicability.py --split validation --all-t2 --out /tmp/spinescope-lab12_ood/val.json
    .venv/bin/python -m pytest -q tests/test_anatomy_ood.py
