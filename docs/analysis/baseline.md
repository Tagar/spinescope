# Baseline: discs and endplates on SPIDER validation

Run: `tools/eval_analysis.py --dataset spider --split validation --analyses discs,endplates --workers 4 --boot 1000 --seed 0`.
39 patients (87 volumes), every sequence of each patient (T1, T2, and T2_SPACE where present). Labeller backend
`auto` (learned labeller). Raw per-disc pairs and the full JSON are kept outside the repository
(`/tmp/spinescope-dx01_harness/baseline`), because they are derived from SPIDER.

Rules were fixed before this run and not changed afterwards (see `docs/analysis/README.md`). A 2-patient smoke run
was used only to check the pipeline. The results below are a baseline to beat, not a tuned result. The next agents
should **not** read `disc_narrowing` or `pfirrmann` as usable: both are poor.

## Coverage (discs matched to reference discs)

| predicted discs | reference discs | matched | scored | label mismatch (dropped) | no reference (dropped) | reference not predicted |
|---|---|---|---|---|---|---|
| 280 | 272 | 270 | 264 | 6 | 10 | 2 |

A disc is scored only if its predicted level name equals the reference name at the same position. Six mismatches
are dropped and visible here, not scored as errors.

## Metrics (95% CI: percentile bootstrap over patients, 1000 resamples)

| field (truth) | kind | scored pairs | not assessed | headline metrics [95% CI] |
|---|---|---|---|---|
| pfirrmann (Pfirrmann 1-5) | ordinal | 198 | 66 | quadratic kappa 0.17 [0.06, 0.32]; MAE 1.78 [1.52, 2.04]; accuracy 0.10 [0.05, 0.16]; within one grade 0.40 |
| disc_narrowing (0/1) | binary | 198 | 66 | sensitivity 0.02 [0.00, 0.07]; specificity 0.98 [0.96, 1.00]; PPV 0.25 [0.00, 1.00]; NPV 0.76 [0.69, 0.82]; AUC 0.55 [0.46, 0.63] |
| modic (0-3) | categorical | 190 | 74 | accuracy 0.52 [0.42, 0.61]; kappa 0.17 [0.03, 0.32]; any-Modic sensitivity 0.88 [0.80, 0.95]; any-Modic specificity 0.41 [0.25, 0.56] |
| up_endplate (0/1, proxy) | binary | 187 | 77 | sensitivity 0.60 [0.45, 0.75]; specificity 0.46 [0.33, 0.59]; PPV 0.47 [0.37, 0.59]; NPV 0.58 [0.43, 0.72] |
| low_endplate (0/1, proxy) | binary | 186 | 78 | sensitivity 0.44 [0.30, 0.58]; specificity 0.59 [0.46, 0.71]; PPV 0.44 [0.30, 0.59]; NPV 0.58 [0.45, 0.72] |
| disc_herniation, disc_bulging, spondylolisthesis | binary | 0 | 264 | no analysis predicts these fields; all pairs not assessed |

Not-assessed counts are per scored disc: the analysis gave no value for that disc (for example, no mid-body slice or
no endplate trace).

## Breakdowns (point estimates; cases are patients)

| group | levels | n scored |
|---|---|---|
| vendor: Siemens / Philips | | 131 / 67 (disc_narrowing; others differ slightly) |
| field strength: 1.5 T / 3.0 T | | 131 / 67 |
| sequences: t1 / t1+t2 / t1+t2+t2_SPACE / t2 | | 5 / 120 / 66 / 7 (disc_narrowing) |

Full breakdown tables for each field are in `metrics.json`. Subgroup sizes are small (5 to 7 discs for T1-only and
T2-only), so do not read sequence differences as effects.

## Caveats

- **Grading semantics are unverified.** The local SPIDER folder ships only `README.md`, `overview.csv`,
  `radiological_gradings.csv` and the zipped images and masks: no codebook. The publisher page could not be read.
  The Modic and endplate codes, and the endplate proxies built on them, are therefore not confirmed against the paper.
- **`num_vertebrae != 5` flags about 99% of volumes** (only one volume has 5), so the transitional breakdown is
  uninformative.
- **Endplate predictors are proxies.** `up_endplate` and `low_endplate` use "any Modic-like pattern" as abnormal.
- **Dropped discs are not scored.** The 6 label mismatches and 10 no-reference discs are excluded from every metric.
- **Single rater.** SPIDER gives one radiologist grading per disc. No inter-reader agreement is quoted here.
- The `discs` analysis reports the sagittal T2 signal, and the endplate analysis reports on T1, T2 and STIR where present.
  SPIDER has no STIR, so STIR-based checks are not exercised.

## Reading the baseline

- `pfirrmann` (MAE 1.78, quadratic kappa 0.17) and `disc_narrowing` (AUC 0.55) are poor. The Pfirrmann suggestion
  and the height flag should not be used as they stand. Beating these numbers is the first target for the next agents.
- `modic`: the any-Modic sensitivity is high (0.88) but specificity is low (0.41), so the endplate analysis flags most
  discs. Calibration on `training` is required before any claim.
