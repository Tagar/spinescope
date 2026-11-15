# lab28: lumbosacral transitional vertebrae (LSTV)

Branch `lab/lab28_lstv`. Owns `anatomy/lstv.py`, `tests/test_anatomy_lstv.py`, `tools/lstv_audit.py`, and the
additive wiring: `Anchors.lstv` (anchors.py), `_lstv_stage` in `pipeline.py`, `labelling_summary` / `summary_lines`
in `summary.py`, and the `lstv` property in `data/anchors.schema.json`. No labeller, calibration or fusion weight was
changed.

## 1. What the check answers

For a labelled study it reports, in `anchors.lstv`:

- `status`: `suspected` (P >= 0.5), `possible` (P >= 0.2), `not_suspected`, or `not_assessed` (no cue measured, no discs,
  or the numbering is `unconfirmed`, see 4).
- `kind`: `sacralised_l5`, `lumbarised_s1`, `unresolved` (both about equally likely), or null.
- `p_transitional`, `posterior` over `none` / `sacralised_l5` / `lumbarised_s1`.
- `castellvi`: grade (I, IIa/IIb, IIIa/IIIb, IV, normal) **only** when a left/right transverse-process record is supplied.
  Otherwise `not_assessed`. Nothing in the pipeline measures transverse processes.
- `evidence`: one entry per cue, with its likelihoods, detail text, and the sequence it needs.
- `naming`: count names and a suggestion (`suggested_names`), always `applied: false`.
- `not_evaluated`: what this check does not look at.

It never renames discs and never changes the numbering status. A single warning is added when `suspected`.

## 2. Naming convention (stated in every output)

Disc names count up from the lowest disc, which is called L5-S1. This is the SPIDER label convention (label 1 = lowest
lumbar vertebra) and this pipeline's numbering. The anatomical name of a disc is the lumbar-type vertebra above it.
Under transitional anatomy the two differ by one level:

- Sacralised L5: the lowest disc is the true L4-L5. Every count name moves cranially by one (count L5-S1 -> L4-L5,
  count L4-L5 -> L3-L4, ...). Offset +1 (numbering_checks convention).
- Lumbarised S1: the lowest disc is the true L5-L6. Only that disc is renamed (count L5-S1 -> L5-L6). Offset -1.

The suggestion uses `numbering_checks.relabel_discs`. When the fusion has already shifted the names
(`numbering.applied_offset != 0`) no second suggestion is made.

## 3. Evidence, and which sequence each cue needs

| cue | what it measures | needs | status in this release |
|---|---|---|---|
| `disc_height` | lowest disc height / median of discs above; <= 0.6 hypoplastic (LR 3 to sacralised), <= 0.75 thin (LR 1.5) | sagittal T1 or T2, lumbosacral junction in view | measured on 384 of 385 real-data studies; thin or hypoplastic in 19 |
| `s1_block_height` | body below lowest disc / median lumbar body; >= 1.5 is a fused block (LR 2) | sagittal, bone below the lowest disc in view | **not measurable with the current labeller** (no sacrum class; 0 of 385 measured). Code kept |
| `l5_wedge` | anterior / posterior height of the body above the lowest disc | sagittal | reported with weight 1.0: wedged in 226 of 385, so it does not separate the variants |
| `numbering_posterior` | fusion offset posterior, counted only when a shift is a candidate (best shift >= 0.5, below the fusion apply threshold); LR 3 | fusion inputs: slabs, second sagittal series | counted only as a candidate; below 0.5 it is neutral (see 5) |
| `lumbosacral_check` | a lower disc was removed as a sacral remnant (LR 2 to lumbarised) | sagittal | 3 of 385 |
| `castellvi` | transverse process craniocaudal size (>= 19 mm = type I) and contact with the sacral ala | axial T1 or T2 through L5 and S1 (3-5 mm), or coronal T1/T2 covering both ala | not measured (no segmentation); only a supplied record is graded |

Not evaluated: iliolumbar ligament origin on axial (Hughes & Saifuddin 2006), iliac crest and aortic bifurcation levels,
rib or vertebra counts on the coronal localizer (none implemented), and the complete S1-S2 disc as a separate cue
(Carrino 2011; the lumbosacral removal is the only proxy used).

Prior and likelihood ratios are judgement defaults in `LstvConfig`; nothing is calibrated, because no public dataset in
this project has LSTV labels.

## 4. Gate: unconfirmed numbering withholds the verdict

If `numbering.status` is `unconfirmed`, the verdict is `not_assessed`, and the cues stay in the record. The check
depends on which disc is the lowest and on the names. The audit found the labeller's `unconfirmed` chains include a
chain running the wrong way (qx5rt ID05: L5-S1 drawn above T8-T9) and missing bone below the lowest disc. Those
flags were labeller artefacts, so they are withheld.

## 5. What the audit found, and what was changed

Version 1 of the check flagged 10 real cases (SPIDER 8, Sudirman 1, qx5rt 1), every one `possible` with the kind
unresolved. Inspection of the cue record showed all of them were driven by the fusion posterior: offsets +-1 sat at
about 0.15 each, which after dividing out the fusion prior gave a likelihood ratio of about 2 against "none". The fusion
does this at its floors, so it is vote noise. The fix: the posterior counts only when the fusion itself calls a shift
a candidate. The wedge cue was also weighted 1.3 although it fires in 59 percent of spines; it now has weight 1.0.
Version 2 removed all but one flag. The remaining flags were driven by thin lowest discs on chains that the gate now
withholds.

On the phantom, the realistic lumbarised variant was flagged possible (sacralised) in 2 of 3 seeds, from a 4.2 mm
sliver the labeller put at the bottom of the chain. The sagittal cues do not reliably see either variant (section 6).

## 6. Results

All runs: default pipeline, `label_spine` (learned labeller, `backend="auto"`), check version 3 (final). Nothing was
tuned on any dataset. Outputs in `/tmp/spinescope-lab28_lstv/v3_*.json` (not committed; regenerate with
`tools/lstv_audit.py run --dataset <name>`).

### 6.1 Real data (no LSTV labels exist in these sets)

| dataset | cases | `suspected` | `possible` | `not_suspected` | `not_assessed` (numbering unconfirmed) |
|---|---|---|---|---|---|
| SPIDER training | 172 | 0 | 1 | 121 | 50 |
| SPIDER validation | 38 | 0 | 0 | 23 | 15 |
| Sudirman (evenly spaced labelled patients) | 120 | 0 | 0 | 67 | 53 |
| qx5rt (one session per sample, all vendors) | 55 | 0 | 0 | 22 | 33 |

SPIDER validation (38 cases here) was not used for any choice. The only `possible` is SPIDER case 109 (training), kind
sacralised, P 0.26, numbering probable. Two more cases (Sudirman 0053 and 0243) were flagged in version 2 and are now
withheld as unconfirmed.

Prevalence comparison. The literature range is 4 to 30 percent (Konin & Walz 2010, citing the series they review).
Among the 234 assessed real cases (SPIDER 145, Sudirman 67, qx5rt 22) the check flags 1 (0.4 percent). Because no labels exist, this is a low flag rate, not a
measured sensitivity. The check is conservative: it says little, and it cannot say how many LSTVs it misses.

SPIDER `num_vertebrae` in `overview.csv` is not a usable LSTV label: most scans have 6 to 9 vertebra masks, which counts
thoracic bodies too. The Sudirman radiologist notes (575 reports) contain no mention of transitional anatomy (only
Tarlov cysts and an S2 bone island); absence in free text is not a label.

### 6.2 Phantom (ground truth; the labeller is out of distribution on it)

Version 3 gates every phantom verdict, because the phantom's numbering is `unconfirmed` in all 18 runs. Version 2
(ungated, sagittal and numbering cues only):

| realism | truth | flagged | kind reported |
|---|---|---|---|
| cartoon | none (3 seeds) | 0 of 3 | |
| cartoon | sacralised L5 (3) | 0 of 3 | |
| cartoon | lumbarised S1 (3) | 0 of 3 | |
| realistic | none (3) | 0 of 3 | |
| realistic | sacralised L5 (3) | 0 of 3 | |
| realistic | lumbarised S1 (3) | 2 of 3 `possible` | sacralised (wrong direction; a 4.2 mm labeller sliver) |

The labeller places the cartoon lowest disc at the wrong level (for example the disc at z = -114 mm against truth -80 mm),
so the phantom cannot test naming end to end. It can test the logic, which is done in the unit tests with synthetic
anchors.

## 7. Visual audit (AI reader)

This is a read by the AI assistant of rendered mid-sagittal panels (labelled disc centres overlaid, 2 by 2 contact
sheets, 110 dpi). It is not a radiological reading, and it has no axial or coronal input.

Viewed: SPIDER 29, 38, 48, 80, 136, 172, 233, 245 (version 1 flags), 123, 250, 137, 11 (random unflagged), Sudirman
0378 (version 1 flag), 0526, 0445, 0471 (random unflagged), 0053 and 0243 (version 2 flags), qx5rt ID07 (version 1
flag), ID27 and ID21 (random unflagged), ID05 (version 2 flag), SPIDER 109 (version 3 flag).

- Version 1 flags showed no LSTV pattern. Several were labeller artefacts: a stray disc label well outside the spine
  (SPIDER 233, Sudirman 0378), and a chain running the wrong way (qx5rt ID05). These motivated the gate.
- SPIDER 109, Sudirman 0053 and 0243: the lowest disc looks thin and dark, with a sacrum below. This matches the thin-disc
  cue. It is equally consistent with degeneration, and no axial or coronal image was available to settle it.
- The unflagged random sample showed ordinary lumbosacral anatomy, with no visible transitional segment at this
  resolution. SPIDER 85, 168, 161, 130 were rendered but not reviewed; Sudirman 0168 was not rendered.

## 8. Limits

- No public LSTV labels in SPIDER, Sudirman or qx5rt. No sensitivity or specificity can be claimed.
- Castellvi grades need axial or coronal measurements, which are not implemented. A reader-supplied record is graded,
  and that path is unit-tested only.
- The sagittal cues are weak and partly unmeasurable with the current labeller (no sacrum class). In particular,
  `s1_block_height` never fires.
- Confounders: degeneration (collapsed lowest disc) mimics the hypoplastic sign. The check is therefore a suggestion.
- The check depends on the numbering. Its verdict is withheld whenever the numbering is `unconfirmed`, which is 39 percent of the
  real-data cases here (SPIDER 31, Sudirman 44, qx5rt 60 percent).

## 9. Literature used (checked)

- Castellvi AE, Goldstein LA, Chan DP. Lumbosacral transitional vertebrae and their relationship with lumbar extradural
  defects. Spine 1984;9(5):493-5. Type I: transverse process >= 19 mm craniocaudally; II: pseudoarticulation with the
  sacral ala; III: fusion; IV: II on one side and III on the other; a/b for unilateral/bilateral (verified via secondary
  summaries, the primary article was not fetched).
- Konin GP, Walz DM. Lumbosacral transitional vertebrae: classification, imaging findings, and clinical relevance. AJNR
  2010;31(10):1778-86. doi 10.3174/ajnr.A2036. Prevalence 4-30 percent (verified quote); Castellvi definitions verified;
  the Castellvi type alone does not establish vertebral numbering.
- Hughes RJ, Saifuddin A. Imaging of lumbosacral transitional vertebrae. Clin Radiol 2004;59(11):984-91. doi
  10.1016/j.crad.2004.02.019.
- Hughes RJ, Saifuddin A. Numbering of lumbosacral transitional vertebrae on MRI: role of the iliolumbar ligaments. AJR
  2006;187(1):W59-65. doi 10.2214/AJR.05.0415. Axial iliolumbar ligament origin for numbering (not implemented).
- Nardo L, et al. Lumbosacral transitional vertebrae: association with low back pain. Radiology 2012;265(2):497-503.
  doi 10.1148/radiol.12112747. Radiographs: interobserver kappa 0.75 (presence), 0.65 (Castellvi). Not MRI.
- Carrino JA, et al. Effect of spinal segment variants on numbering vertebral levels at lumbar MR imaging. Radiology
  2011;259(1):196-202. doi 10.1148/radiol.11081511. Complete S1-S2 disc associated with LSTV (P = .004); iliolumbar
  ligament identified the lowest lumbar vertebra, not invariably L5.
- Farshad-Amacker NA, et al. Interreader and intermodality reliability of radiograph and MRI in detection and
  classification of lumbosacral transitional vertebra. Spine J 2014;14(8):1470-5. doi 10.1016/j.spinee.2013.08.048.
  MRI (coronal) kappa 0.93 detection, 0.83 classification.
- O'Driscoll CM, Irwin A, Saifuddin A. Variations in morphology of the lumbosacral junction on sagittal MRI: correlation
  with plain radiography. Skeletal Radiol 1996;25(3):225-30. doi 10.1007/s002560050069. Sagittal MRI patterns are not
  one-to-one with Castellvi types.

## 10. Reproduce

    .venv/bin/python tools/lstv_audit.py run --dataset phantom --out /tmp/spinescope-lab28_lstv/phantom.json
    .venv/bin/python tools/lstv_audit.py run --dataset spider --out /tmp/spinescope-lab28_lstv/spider.json
    .venv/bin/python tools/lstv_audit.py run --dataset sudirman --sample 120 --out /tmp/spinescope-lab28_lstv/sudirman.json
    .venv/bin/python tools/lstv_audit.py run --dataset qx5rt --out /tmp/spinescope-lab28_lstv/qx5rt.json

SPIDER takes about 11 minutes on this machine; Sudirman and qx5rt about 5 each.

## 11. Next steps

1. Measure transverse processes and sacral-ala contact on coronal T1 or T2 (the Farshad-Amacker protocol), then grade
   Castellvi automatically. Needs a labelled set first.
2. Give the labeller a sacrum class, so `s1_block_height` can fire and the bone below the lowest disc is confirmed.
3. Collect reader-labelled LSTV cases (Castellvi grade, lumbarised or sacralised) to calibrate the likelihood ratios
   and measure sensitivity. Until then the status is a suggestion.
