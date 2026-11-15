# Endplates and Modic typing (sagittal T1 / T2 / STIR)

Status: Modic typing is a suggestion for a reader. Every number below is measured on SPIDER and is a suggestion to be
confirmed by a clinician. Nothing here is a diagnosis.

## Pre-registration (fixed before any validation number was produced)

Written before the validation split was scored with the new rule. Calibration uses **training only**.

1. Unit of decision: one endplate (superior or inferior endplate of one vertebral body), per side (left, central,
   right of the anchored midline) and for all slices together ("all").
2. Measurement: per endplate, the median over slices of (subchondral band median) / (the same body's mid-body marrow
   median), per sequence (T1, T2, STIR when present). Healthy-reference ratios (against the study's healthiest
   bodies) are a second reference mode.
3. Rule: the thresholds `t1_low`, `t1_high`, `t2_low`, `t2_high` (and `stir_high`, fixed at 1.20 because SPIDER has
   no STIR) define the type by Modic 1988 / Rahme and Moussa 2008 (see the table below).
4. Per-disc prediction: the highest Modic type over the two endplates of the disc; 0 only when both endplates are
   known normal (same as the baseline predictor).
5. **Objective (training):** maximise the unweighted kappa of the four-class Modic type (0-3), subject to
   any-Modic (type >= 1 against 0) specificity >= 0.85 on scored training discs.
6. **Grid (fixed):** `t1_low` in {0.70, 0.75, 0.80, 0.85, 0.90}; `t1_high` in {1.10, 1.15, 1.20, 1.25, 1.30};
   `t2_low` in {0.80, 0.85, 0.90, 0.95}; `t2_high` in {1.05, 1.10, 1.15, 1.20, 1.25}; reference mode in {own
   mid-body, healthy bodies}. Ties go to the larger specificity, then the smaller `t1_low`.
7. If no grid point meets specificity >= 0.85, the default in `config.py` (the uncalibrated values) is kept and this is
   reported as a failed pre-registration.
8. Validation (split `validation`) is scored once with the selected point. No further change after that.

### Modic type definitions used

| type | T1 | T2 | STIR | rule in code (ratio to own mid-body) |
|---|---|---|---|---|
| none | normal | normal | normal | no endplate flag on T1 or T2 |
| Modic 1 | low | high | high or absent | T1 < t1_low and T2 > t2_high |
| Modic 3 | low | low | any | T1 < t1_low and T2 < t2_low |
| Modic 2 | high | iso/high | not high | T1 > t1_high and T2 >= t2_low |
| not assessed | - | - | - | T1 or T2 missing, or the signature matches no rule |

Missing sequences: with T1 missing, no type is assigned. With T2 missing, a low T1 gives "presence only" and no type.

## Calibration on SPIDER training (Modic rule)

Run on the 179 training cases (7,604 endplate rows, dumped with the pre-registered measurement). The offline search
(`calib`, grid of 1,000 points, own and healthy reference modes) gave 225 points with any-Modic specificity >= 0.85.
The highest kappa among them, with the pre-registered tie-breaks, is the selected point:

| reference | t1_low | t1_high | t2_low | t2_high | kappa (4-class) | any-Modic sens | any-Modic spec | scored discs |
|---|---|---|---|---|---|---|---|---|
| own mid-body (selected) | 0.75 | 1.25 | 0.85 | 1.05 | 0.10 | 0.26 | 0.855 | 333 |
| previous defaults (0.85, 1.15, 0.90, 1.10) | | | | | 0.07 | 0.41 | 0.697 | 618 |

The selected point sits at the edge of the constraint (specificity 0.855 against 0.85). Its kappa is low. Training
agreement of this rule with the SPIDER gradings is poor, so the result should be read as "low sensitivity at a
specificity bar" rather than as a usable reader aid. The previous defaults failed the constraint on training.

## Endplate contour defect rule (pre-registered before validation)

Measure: per traced endplate and slice, the endplate height (bot/top) along AP is compared with its running median over
10 mm (`endplate_defect_window_mm`). **Concavity** (mm) is the largest intrusion into the body relative to that smooth
contour; **irregularity** is the RMS deviation. Both are medians over slices (rows with side "all").

Proxy target: SPIDER UP endplate (endplate of the body above, `inferior` row) and LOW endplate (`superior` row of
the body below), binary 0/1. These codes are not Modic types (see interpretation below), so the Modic proxy is retired
for these fields.

Rule (fixed here): an endplate is positive when concavity >= threshold. The threshold is the value that maximises
Youden's J (sensitivity + specificity - 1) on the training endplate pairs, searched over 0.25 to 4.00 mm in 0.25 mm
steps. AUC of concavity on training and validation is reported. Validation is scored once with that threshold.

Interpretation of the SPIDER endplate codes (unverified): UP/LOW code 1 rises steadily with Pfirrmann grade (0.18 at
grade 1, 0.88 at grade 5) and with Modic 2 (0.73 against 0.24 for Modic 0). The codes look like a degeneration-related
endplate abnormality (irregularity or defect), not a Schmorl-specific code. The publisher codebook could not be read
(redirect to a login page), so this stays an interpretation.

## Method

Per endplate, per side and for all slices together: subchondral band (3-8 mm inside the endplate, clear of cortex)
against the same body's mid-body marrow (smoothed, 1 mm). Each band's median ratio over slices gives T1, T2 and STIR
ratios. The type comes from the pre-registered rule above. Presence-only changes (one sequence missing) are a separate
flag, `marrow_change_presence`, with confidence capped at 0.3, and map to findings with a null Modic value.

Extent is the percentage of band pixels that are abnormal (`extent_pct`). Width is the percentage of the endplate AP
width (1 mm bins) with abnormal band marrow (`width_pct`). Height is the 95th percentile depth of abnormal marrow
(`depth_mm`, mm from the endplate).

Side: left, central or right of the anchored midline, decided by slice position (`_side_of`). The "all" row is the
worst side when a Modic pattern is confined to one or two sides.

Literature on inter-reader agreement (verified by search):
- Jensen TS, Sorensen JS, Kjaer P. Acta Radiologica 2007;48:748-754 (DOI 10.1080/02841850701422112). Nordic Modic
  Consensus Group classification, 50 examinations: intra- and interobserver agreement substantial to almost perfect
  (type-specific kappa values not in the accessible abstract).
- Fayad F, et al. Joint Bone Spine 2009;76:286-289 (DOI 10.1016/j.jbspin.2008.09.012). Modified classification with mixed
  type 1/2, 94 patients, three readers: interobserver weighted kappa 0.56-0.74, intraobserver 0.85.
- Jensen TS, et al. Eur Spine J 2008;17:1407-1422 (DOI 10.1007/s00586-008-0770-2), systematic review, reproducibility
  kappa 0.64-0.91 across the studies it cites.

## Validation on SPIDER validation (39 cases, 264 discs; scored once with the pre-registered rules)

Run: `tools/eval_analysis.py --dataset spider --split validation --analyses endplates --workers 4 --boot 1000 --seed 0`.
Modic thresholds and the defect threshold are the training values above. Nothing was changed after this run.

| field (truth) | scored / pairs | headline [95% CI] | baseline (dx01, 2-run) |
|---|---|---|---|
| modic 0-3 | 80 / 264 | accuracy 0.61 [0.42, 0.77]; kappa 0.14; any-Modic sens 0.50, spec 0.74 | acc 0.52, kappa 0.17, sens 0.88, spec 0.41 (n 190) |
| modic_any (type or presence) | 85 / 264 | sens 0.50 [0.29, 0.69]; spec 0.68 [0.45, 0.85]; PPV 0.35; NPV 0.80 | not in baseline |
| up_endplate (defect proxy) | 227 / 264 | sens 0.74 [0.60, 0.85]; spec 0.44 [0.32, 0.55]; AUC 0.60 [0.51, 0.69] | sens 0.60, spec 0.46 (Modic proxy, n 187) |
| low_endplate (defect proxy) | 224 / 264 | sens 0.73 [0.60, 0.84]; spec 0.35 [0.24, 0.46]; AUC 0.59 [0.50, 0.67] | sens 0.44, spec 0.59 (Modic proxy, n 186) |

Per class (Modic): truth 0 n 58, sens 0.74, spec 0.50; truth 2 n 22, sens 0.27, spec 0.79. There are no truth type 1 or
type 3 discs in validation, so type 1 cannot be scored here (SPIDER has 4 type-1 discs in 1,520).

Coverage fell. Of 264 discs, 184 are not assessed for Modic, because the rule needs T1 and T2 to agree on a type and
229 endplates came out "indeterminate". The baseline scored 190. Read the 0.61 accuracy on 80 discs only.

Specificity bar: any-Modic specificity on validation is 0.74, below the 0.85 that the training objective targeted
(0.855 on training). The operating point does not transfer. Report it as a failed transfer, not a validated reader aid.

## False positives (Modic typed on discs graded Modic 0, validation)

- 24 of 136 truth-normal discs were called typed Modic (any type). They are mostly Modic 2 (12 of 136 called 2; plus
  3 called 3); no truth-normal disc was called Modic 1 on the SPIDER validation split.
- By disc degeneration (SPIDER Pfirrmann of the disc): Pfirrmann 1: 9/31; 2: 2/31; 3: 6/47; 4: 6/21; 5: 1/6. The FPs
  are not concentrated in degenerated discs.
- Next to a degenerated disc (an adjacent disc with Pfirrmann >= 4): 13/44 (30%) against 11/92 (12%) otherwise.
- Partial volume (endplate traced on few slices): 4+ slices 22/124, 2-3 slices 2/9, one slice 0/3. Too few cases at the
  edge to say more than "no excess".

## Modic 1 false-positive burden (per study and per disc)

- SPIDER validation: 0 Modic 1 calls among 234 discs in 36 cases. Modic 1 is not measurable on SPIDER (4 of 1,520).
- Sudirman sagittal (515 patients, first session each; 508 analysed, 7 with no sagittal series): 9 of 508 studies
  (1.8%) had at least one Modic 1 call; 10 calls over 6,159 endplates (0.16% of endplates, 0.02 per study). No presence
  flags on the Sudirman studies. No truth is available, so every Sudirman call is unverified.
- Phantom (synthetic, ground truth): Modic 1 is called only at the true endplate (L4 inferior, left), and the call is
  the same with and without STIR. A normal phantom and a degenerated-only phantom give no calls.
- STIR veto: when STIR is present and not high, T1 low and T2 high give "indeterminate", not Modic 1. STIR is not
  present in SPIDER, so this has no SPIDER effect.

## Endplate defects (UP/LOW): interpretation and result

Interpretation (unverified): UP/LOW code 1 rises with Pfirrmann grade (0.18 at grade 1, 0.88 at grade 5). It
looks like a degeneration-related endplate abnormality (irregularity or defect), not a Schmorl-specific code. The codebook is
unavailable.

Result: concavity above 0.5 mm (training optimum, Youden J 0.227, AUC 0.667) gives UP sens 0.74 and spec 0.44; LOW sens
0.73 and spec 0.35 on validation (AUC 0.60 and 0.59). The defect measure carries some signal, but it is not specific.
Its threshold is at the level of sub-millimetre contour noise, so it should not be used as a defect call.

## Limitations

- SPIDER has no STIR and no fat-sat T2; those paths are tested on the phantom only.
- Type 1 cannot be measured on SPIDER; its FP burden is estimated from Sudirman (no truth).
- The calibration point sits at the specificity bar on training and fails it on validation.
- One rater per disc in SPIDER; no inter-reader kappa is measured here.
- Literature (verified by search): Jensen TS, Sorensen JS, Kjaer P. Acta Radiol 2007;48:748-754 (Nordic Modic
  classification, substantial to almost perfect agreement, 50 MRIs); Fayad F et al. Joint Bone Spine 2009;76:286-289
  (weighted kappa 0.56-0.74 interobserver for a modified classification); Jensen TS et al. Eur Spine J 2008;17:1407-1422
  (review, reproducibility kappa 0.64-0.91). Our kappa 0.14 on validation is well below these reader values, and the
  scoring uses a single SPIDER rater, so the comparison is only indicative.

