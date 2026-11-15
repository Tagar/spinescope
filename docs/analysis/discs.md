# Disc grading: Pfirrmann grade and disc narrowing (discs analysis)

Status: validated on SPIDER validation (dx02_pfirrmann). This page records the pre-registered decision rules first, then the
calibration on SPIDER training, then the validation numbers. Rules were written before any validation number was
produced for them.

## What is measured

Per disc and mid-portion sagittal slice (see `analysis/discs.py`):

- Disc frame from the anchors (centre, normal, AP margins). Endplates are traced by the dark-line trace. When the
  trace fails (for example a dark degenerated disc with no visible endplate contrast) and the disc is in the image,
  the disc is measured on its anchored height (`geometry = anchor`): signal only, no heights.
- Nucleus = central box of the disc frame; annulus = rest of the disc minus an endplate rim. Signal is relative to
  the thecal sac (CSF) posterior to the disc.
- Relative features:
  - `log_nuc_over_csf`: log of nucleus mean over CSF mean (brightness, Pfirrmann nucleus signal relative to CSF).
  - `nuc_cv`: nucleus coefficient of variation (homogeneity).
  - `nuc_ann_contrast`: (nucleus - annulus) / (nucleus + annulus) (distinction of nucleus from annulus).
  - `dhi`: disc height index = mean disc height / mean of the adjacent vertebral body heights (traced endplates
    only). Missing when an adjacent body is not traced. Used for narrowing; not a grader input (see calibration).
- Features are medians over the mid-portion slices of a disc.

## Pfirrmann grade

Reference: Pfirrmann CW, Metzdorf A, Zanetti M, Hodler J, Boos N. Magnetic resonance classification of lumbar
intervertebral disc degeneration. Spine 2001;26(17):1873-8 (doi 10.1097/00007632-200109010-00011 per PubMed record
11568697). Their five grades use nucleus brightness relative to CSF, nucleus/annulus distinction, homogeneity and
disc height. Reported agreement in that study: interobserver kappa 0.69-0.81, intraobserver kappa 0.84-0.90 (300
discs, 60 patients, three observers). These are the published reader benchmarks quoted here.

Model: proportional-odds (cumulative logit) model on the three nucleus features (brightness, homogeneity,
distinction), standardised. Latent score
`eta = w . z`; `P(grade <= g) = logistic(c_g - eta)` for g = 1..4. The grade is the median of the modelled
distribution (smallest g with P(grade <= g) >= 0.5). Coefficients, standardisation constants and cutpoints live in
`config.py` (`pfirrmann_*`) and were fitted by maximum likelihood on SPIDER **training** only.

Confidence = P(grade within one of the suggestion) x quality, where quality multiplies: traced fraction (anchored
geometry counts 0.6), nucleus SNR relative to the training median (floor 0.5), and two or more slices (0.9 for one
slice). Not assessed when the study has no sagittal T2 (T1-only or
STIR-only), or when no nucleus/CSF ratio is measured.

## Disc narrowing

SPIDER `Disc narrowing` is a separate binary annotation (not a threshold of the Pfirrmann grade; see the
dataset notes). Rule: narrowing = 1 when `dhi < disc_narrowing_dhi_threshold`. Threshold chosen on SPIDER training
by maximising Youden's J over disc-level `dhi` against the training narrowing labels. Score for AUC = minus `dhi`.
Not assessed when `dhi` is missing.

## Pre-registered decision rules (written before validation numbers)

1. **Transparent model is the shipped grader** (three-feature proportional-odds model, fitted as above). A learned grader (gradient boosting or small CNN) was not trained in
   this wave. If one is trained later, it ships only if its validation quadratic kappa exceeds the transparent
   model's by at least 0.05, and the patient-bootstrap 95% interval of the difference excludes zero.
2. **Pfirrmann acceptance.** The transparent model replaces the earlier uncalibrated ladder if its validation
   quadratic kappa exceeds the ladder's baseline (0.17, `docs/analysis/baseline.md`). Secondary: within-one-grade
   accuracy reported with its interval; no threshold.
3. **Narrowing acceptance.** Validation AUC of minus `dhi` compared with the earlier height-ratio predictor (AUC
   0.55). Sensitivity and specificity at the training threshold reported, not used to pick it.
4. **No tuning on validation.** If a rule fails, the failure is reported; the rule is not changed to pass.
5. **Confidence check.** Accuracy within one grade by confidence band is reported on validation as a descriptive
   check. Confidence is not recalibrated on validation.

## Calibration (SPIDER training, 172 patients)

Training discs were measured with the same pipeline (`discs.analyze`, labeller backend `auto`), keeping T2 discs
with a measured nucleus/CSF ratio: 1181 discs; grades 1-5 = 195, 300, 342, 203, 141.

- **Height index dropped from the grader.** With four features, its coefficient was -0.02 on the standardised scale
  and the log-likelihood did not change (1353.2 with or without it). It was also missing for 27% of discs. It stays
  as a narrowing feature. This is a training-only decision made before validation.
- Fitted three-feature model (`config.py`, `pfirrmann_*`): weights (brightness, homogeneity, distinction) =
  (-1.107, 1.188, -0.346) on standardised features; means (-1.362, 0.327, 0.100); SDs (0.596, 0.181, 0.124);
  cutpoints (-2.735, -0.776, 1.460, 3.482). In-sample on training: quadratic kappa 0.70, accuracy 0.48, within one
  grade 0.93. These are optimistic (fitted and scored on the same discs); validation is the test.
- Narrowing threshold on training: 898 discs with a height index and the SPIDER narrowing label (all sequences,
  prevalence 0.22). Youden's J is maximised at `dhi` < 0.262 (J 0.30). On the T2 discs alone the optimum is 0.27
  (869 discs, AUC of minus `dhi` 0.68), so the threshold is not sensitive to the sequence mix.
- Nucleus SNR median over 981 T2 discs: 27.5 (`pfirrmann_snr_reference`).
- The training constants are reproduced by `tools/calibrate_disc_grade.py` from a training feature table written by
  `tools/dump_disc_features.py`. The table is derived from SPIDER and is not committed.

## Validation (SPIDER validation, 39 patients, 264 discs)

Run once, after the rules above were fixed: `tools/eval_analysis.py --dataset spider --split validation
--analyses discs --workers 6 --boot 1000 --seed 0 --pairs`. Intervals: 95% percentile bootstrap over patients.
Baseline: `docs/analysis/baseline.md` (earlier ladder and height-ratio rules, same split and labeller).

Coverage: 270 matched, 264 scored, 6 label mismatches dropped, 10 no reference, 2 reference discs not predicted
(same labeller coverage as the baseline).

| field | scored (not assessed) | this model [95% CI] | baseline |
|---|---|---|---|
| Pfirrmann, quadratic kappa | 257 (7) | **0.58 [0.36, 0.75]** | 0.17 [0.06, 0.32] |
| Pfirrmann, linear kappa | 257 | 0.45 | n/a |
| Pfirrmann, accuracy | 257 | 0.46 [0.36, 0.55] | 0.10 [0.05, 0.16] |
| Pfirrmann, MAE (grades) | 257 | 0.69 [0.52, 0.90] | 1.78 [1.52, 2.04] |
| Pfirrmann, within one grade | 257 | 0.89 | 0.40 |
| disc narrowing, AUC (minus DHI) | 183 (81) | 0.63 [0.53, 0.75] | 0.55 [0.46, 0.63] |
| disc narrowing, sensitivity | 183 | 0.45 [0.32, 0.62] | 0.02 [0.00, 0.07] |
| disc narrowing, specificity | 183 | 0.64 [0.56, 0.73] | 0.98 [0.96, 1.00] |
| disc narrowing, PPV / NPV | 183 | 0.26 [0.17, 0.37] / 0.81 [0.72, 0.89] | 0.25 / 0.76 |

Pfirrmann confusion matrix (rows = SPIDER grade, columns = suggestion; 257 discs):

| truth \ suggestion | 1 | 2 | 3 | 4 | 5 | n | sensitivity |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 18 | 5 | 6 | 2 | 36 | 0.14 |
| 2 | 0 | 18 | 15 | 1 | 0 | 34 | 0.53 |
| 3 | 0 | 11 | 44 | 17 | 2 | 74 | 0.59 |
| 4 | 0 | 4 | 24 | 32 | 13 | 73 | 0.44 |
| 5 | 0 | 1 | 6 | 15 | 18 | 40 | 0.45 |

Grade 1 discs are mostly called 2 (18 of 36) and grade 5 discs mostly 4 (15 of 40), so the extremes regress toward the middle. Within-one accuracy stays high (0.89) because most errors are adjacent.

Breakdowns (point estimates, small groups; not effects):

| group | cases | n | quadratic kappa | accuracy | MAE |
|---|---|---|---|---|---|
| Philips (3.0 T) | 16 | 99 | 0.59 | 0.31 | 0.83 |
| Siemens (1.5 T) | 22 | 158 | 0.49 | 0.54 | 0.61 |
| sequences t1+t2 | 26 | 169 | 0.67 | 0.47 | 0.63 |
| sequences t1+t2+t2_SPACE | 11 | 81 | 0.37 | 0.46 | 0.82 |
| sequences t2 only | 1 | 7 | 0.19 | 0.14 | 0.86 |

Vendor and field strength are the same split here (Philips = 3.0 T, Siemens = 1.5 T in this subset).

Grader geometry: 173 discs graded from traced endplates and 84 from anchored geometry (within one grade 0.91 and
0.86). Narrowing is not assessed for 81 discs, most without a traced adjacent body (no height index).

**Confidence check (descriptive; not used to recalibrate, per rule 5).** Exact-grade accuracy by confidence band:
below 0.3, 0.67 (n 30); 0.3-0.5, 0.49 (n 125); 0.5-0.7, 0.24 (n 41); 0.7 and above, 0.43 (n 61). Within-one accuracy
is 0.83-0.93 across bands. Confidence therefore reflects image quality and the model's within-one spread, not
calibrated exact-grade probability. Read it as a ranking of how much to trust a suggestion, not as a probability.

**Pre-registered rules, applied as written.**
1. Learned grader: not trained in this wave, so nothing to compare. Shipped model is the transparent one.
2. Pfirrmann: validation quadratic kappa 0.58, above the ladder baseline 0.17, and its interval lower bound (0.36)
   is above the baseline point. Accepted: the transparent model replaces the ladder.
3. Narrowing: validation AUC 0.63 against 0.55. The point estimate is higher; the intervals overlap ([0.53, 0.75]
   against [0.46, 0.63]). Sensitivity 0.45 and specificity 0.64 at the training threshold are reported, not tuned.

Comparison with published reader agreement: Pfirrmann et al. (Spine 2001) report interobserver kappa 0.69-0.81
and intraobserver kappa 0.84-0.90 on their own 300-disc set with three observers. The SPIDER grades come from one
expert radiologist and this is a single-reader comparison, so the kappa here is not directly comparable to
inter-reader kappa. Against one reader, 0.58 is below the published reader-to-reader agreement but well above the baseline ladder. SpineNet inter-reader numbers were not checked in this wave.

**Other data (coverage only).**
- Sudirman free-text reports (575 reports): 514 have at least one parsed label; 25 reports carry a degeneration label
  (29 labels) and 77 labels are disc dehydration. Weak labels for coverage, not scored.
- qx5rt (OSF, CC BY-NC 4.0): the local folder holds images and label masks only; there are no gradings, so nothing
  was scored on it.

**Learned grader.** Not trained. The transparent model reached 0.58 on validation with three features and no
missing-data penalty. A learned model would need an extra runtime dependency or a torch export for a gain that is
not yet shown. If trained later, it ships only under rule 1 above.

## Limitations

- Grading is one expert musculoskeletal radiologist per disc in SPIDER; no inter-reader data here. The published
  reader kappas above are from a different dataset and protocol.
- Pfirrmann grades are a suggestion only (DESIGN principle 4). A reader confirms.
- The constants were fitted on SPIDER training at 1.5 T and 3 T, Siemens and Philips. Other vendors, coils and
  protocols are unmeasured.
- Discs measured on the anchored height (no traced endplates, 84 of 257 validation grades) carry lower confidence,
  and their nucleus signal is not checked against an endplate trace.
- The disc height index needs both adjacent bodies traced. It is missing for about 30% of discs, so narrowing is not
  assessed there (81 of 264 on validation).
- The confidence value is not a calibrated probability (see the check above).
- Phantom test: the cartoon phantom's homogeneous dark nucleus reads as less degenerate under the homogeneity feature,
  so the phantom test checks brightness ordering only.
- Level names follow the labeller; a name that is not confirmed caps confidence in the findings layer.
