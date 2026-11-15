# Labelling benchmarks

One command runs every labelling benchmark, writes one JSON and one markdown summary of numbers, and checks the
numbers against the golden file so that a regression is caught before it is merged. New models, calibrations and
refactors are judged by the same yardstick.

```
tools/benchmark.py                                     # run what is installed, write the summary
tools/benchmark.py --check benchmarks/golden.json      # ... and fail (exit 1) on a regression
tools/benchmark.py --update-golden benchmarks/golden.json   # after a reviewed, deliberate change
spinescope benchmark --check benchmarks/golden.json    # the same, from the CLI
```

Implementation: `src/spinescope/benchmark.py` (runner, scoring, checker). The golden numbers are in
`benchmarks/golden.json`; the regression-checker tests are in `tests/test_benchmark.py`.

## 1. How to run

Setup: `python -m pip install -e ".[dev]"` (or `uv pip install -e ".[dev]"` in a venv), then from the repository root.

| Who | Command | Data | Time (this Mac, 14 cores, cold cache) |
|---|---|---|---|
| anyone, CI | `tools/benchmark.py --suites phantom --check benchmarks/golden.json` | none (phantom only) | about 40 s |
| maintainer with the datasets | `tools/benchmark.py --check benchmarks/golden.json` | SPIDER, Sudirman, qx5rt, phantom | about 9 min (Sudirman is most of it) |
| development | `tools/benchmark.py --limit 10 --suites spider` | any | minutes; scales with the number of cases |

A suite whose data is not installed is **skipped with a note**, never failed. The run on a machine without datasets
still produces a full summary for the phantom suites, and `--check` only judges the sections that ran.

Datasets are found under `$SPINESCOPE_DATA` (default `~/data/public`); see `docs/datasets.md` for how to fetch them.
SPIDER is `spider/`, Sudirman is `lumbar_spine_mri_sudirman/` plus its label companion, and qx5rt is
`lumbar_multiscanner_qx5rt/`.

Options (all optional):

- `--suites SUITE ...`: `spider`, `sudirman`, `qx5rt`, `phantom_cartoon`, `phantom_realistic`, or `phantom` for both
  phantom suites. Default: all.
- `--jobs N`: parallel cases (default: the number of cores, at most 8). Each worker runs the labeller on one ONNX Runtime thread.
- `--threads N`: ONNX Runtime threads per labelling process, through `SPINESCOPE_THREADS`. `0` means the runtime's default
  (all cores, which oversubscribes when `--jobs > 1`).
- `--limit N`: evenly spaced subsample of N cases in SPIDER, Sudirman and qx5rt, for development only. The subsample is
  recorded in the output, and a golden check fails on it (the sample size differs), by design.
- `--spider-split validation`: SPIDER subset from `overview.csv`. Golden numbers use `validation` only.
- `--refresh`: ignore the cache and label every case again.
- `--out DIR` (default `/tmp/spinescope-benchmark`): writes `benchmark.json`, `benchmark.md` and, with `--check`, `check.txt`.
- `--cache DIR` (default `~/.cache/spinescope/benchmark`): labeller outputs, keyed by the labeller fingerprint.
- `--from-json RESULT`: use an earlier `benchmark.json` instead of running (for checking a saved run).

Both `--out` and `--cache` must be outside the repository; the tool refuses otherwise.

### Caching

Each case's labeller output is cached under `<cache>/<labeller fingerprint>/<suite>/`. The fingerprint hashes every
`.py`, `.json` and `.onnx` file the labeller depends on (the package minus the CLI, rendering, viewer, report and AI
modules), so editing the labeller recomputes everything and editing anything else does not. Scoring always runs from
the cached records, so changing a scorer or the summary costs seconds. Cache file names are hashes; case identifiers
never appear in file names. Only successful labelling is cached; failures are retried on the next run.

The cache is a local copy of labeller outputs (anchors, and for ground truth the centroids of the annotated discs).
It contains no images. Delete it with `rm -rf ~/.cache/spinescope/benchmark` if the datasets change.

### Thread control

Inside a benchmark worker the labeller uses one thread, so `--jobs` processes do not oversubscribe the machine.
`--threads` overrides that for every worker. The labeller's timing numbers depend on it: quote them with the setting used.

## 2. Suites and protocols

| Suite | Data | Cases | Labeller backend | Truth | Scored with |
|---|---|---|---|---|---|
| `spider` | SPIDER sagittal T1/T2 volumes, `validation` split | 87 | `auto` (the shipped default, learned labeller) | vertebra, disc and canal masks | shared volume scorer (`anatomy/_validate.py`) |
| `sudirman` | Sudirman lumbar DICOM, all 515 labelled patients | 515 | `auto` | three annotated discs per patient (L3-L4, L4-L5, L5-S1) from label maps | level-wise scorer (`_validate_sudirman.py`), plus the shared scorer for numbering status |
| `qx5rt` | OSF multi-scanner set, all sessions | 55 | `auto` | annotated disc masks (L1-L2 to L4-L5) on the T1 | level-wise scorer (`_validate_qx5rt.py`), plus the shared scorer for numbering status |
| `phantom_cartoon` | synthetic phantom, 23 scenarios (`spinescope.testing.robustness`) | 23 | `classical` | phantom ground truth | shared scorer and the robustness outcome rule |
| `phantom_realistic` | same scenarios, realistic phantom look | 23 | `ml` | phantom ground truth | shared scorer and the robustness outcome rule |

Protocol decisions:

- **Default labeller on real data.** SPIDER, Sudirman and qx5rt use `label_spine(study)` with backend `auto`, which
  is what a user gets. Since `label_spine` returns the numbering status, the status numbers are the shipped ones.
- **Phantom backends follow the robustness tests.** `tests/robustness/test_robustness.py` grades the cartoon with the
  classical chain and `test_robustness_realistic.py` grades the realistic phantom with the learned labeller. The
  learned labeller is out of distribution on the cartoon (see `docs/model_card.md`), so the cartoon is not run with it.
- **SPIDER T2 volumes are labelled together with their T1 sibling**, as in `tools/calibrate_labeling.py`, the protocol
  behind the status numbers in `docs/labeling/lab18_reliability.md` and `lab22_confirmed.md`.
- **Sudirman and qx5rt** annotate only a few discs per study, so precision is not defined there (shown as n/a); the
  level-wise recall, label accuracy and centre error are the primary measures.

## 3. What each metric means

Volume suites (SPIDER, phantom). Predicted discs are matched to truth discs by the shared scorer: nearest in the sagittal
plane, within 12 mm.

- **recall**: truth discs with a matched prediction, out of all truth discs.
- **precision**: matched predictions, out of all predictions (false positives count against it).
- **label accuracy** (`label_accuracy`): of the matched discs, the share with the exact level name.
- **label accuracy, all truth** (`label_accuracy_all_truth`): correctly named discs, out of all truth discs. This is the
  headline label number: a missed disc counts as wrong.
- **fully correct** (`fully_correct`): share of studies in which every truth disc is named correctly.
- **centre error** (median and 90th percentile, mm): distance from the predicted to the true disc centre. `centre_within_3mm`
  is the share of matched discs within 3 mm.
- **midline error** (median, mm): distance of the labelled midline from the truth canal (where the truth has one).
- **disc ECE and AUROC**: calibration and ranking of the per-disc confidence against a correct name (ECE: expected
  calibration error, lower is better; AUROC: higher is better, 0.5 is chance).

Level suites (Sudirman, qx5rt). The annotated discs are the truth.

- **recall**: annotated discs with a predicted disc within 12 mm. **label accuracy** is then the share of those with the
  right name; **label accuracy, all truth** counts the unmatched ones as wrong. **fully correct**: every annotated disc named right.
- **centre error**: as above, over matched annotated discs.

Numbering status (all suites). The labeller reports one of three statuses per study, with a calibrated confidence
(`anchors.numbering['status']`, policy in `docs/labeling/lab08_fusion.md` and `lab22_confirmed.md`):

- `confirmed`, `probable` (together: **announced**), `unconfirmed` (abstains).
- A study is **correct** when at least one disc is matched and every matched disc has the right name (the calibration
  definition in `tools/calibrate_labeling.py`).
- **coverage**: share of studies at that status. **precision**: share of those that are correct.
- **wrong announced**: announced studies with a wrong numbering. This is the count that must not grow: "unsure" is
  acceptable, "wrong and announced" is the failure that matters most.
- **numbering ECE and AUROC**: calibration and ranking of the study's numbering confidence against a correct numbering.

Phantom outcome per scenario (`spinescope.testing.robustness.classify`): `pass` (every in-view disc found, named and
centred within 3 mm at p90, nothing spurious), `unsure` (not a pass, with numbering confidence below 0.5), and
`silent_error` (not a pass, with confidence at or above 0.5). `silent_error` is the count that matters most.

Latency: `latency_s_median` and `latency_s_p90` are the labelling time of one study (loading excluded). They are
recorded when each case was labelled; a cached run repeats the numbers of the run that filled the cache. Latency is
never gated, because it depends on the machine.

Every section also has breakdowns by vendor, field strength and sequence (SPIDER), by vendor (qx5rt) and by scenario
group (phantom). They are informational and not gated, since most groups are small.

## 4. Current numbers

Labeller fingerprint `f37cf61ff4276b23` (spinescope `0.1.0.dev0`, main at `dccbf9a`). Full run, 8 workers, cold cache,
8.4 minutes wall time. Every case ran; no errors. These are exactly the numbers in `benchmarks/golden.json`.

| Section | Cases | Recall | Label acc (matched) | Label acc (all truth) | Fully correct | Centre median / p90 (mm) | Announced cov. | Announced precision | Wrong announced |
|---|---|---|---|---|---|---|---|---|---|
| SPIDER validation | 87 | 99.5% | 98.1% | 97.6% | 94.3% | 0.60 / 1.98 | 67.8% | 100.0% | 0 |
| Sudirman (all labelled) | 515 | 99.2% | 92.4% | 91.9% | 89.1% | 2.13 / 3.83 | 60.2% | 96.5% | 11 |
| qx5rt (all sessions) | 55 | 100.0% | 86.2% | 86.2% | 85.5% | 0.78 / 2.60 | 40.0% | 100.0% | 0 |
| Phantom, cartoon (classical) | 23 | 79.7% | 72.7% | 58.0% | 30.4% | 0.92 / 5.38 | 8.7% | 50.0% | 1 |
| Phantom, realistic (learned) | 23 | 97.8% | 63.0% | 61.6% | 4.3% | 1.52 / 2.76 | 0.0% | n/a | 0 |

Disc precision (volume suites): SPIDER 99.2%, cartoon phantom 89.4%, realistic phantom 73.4%; not defined for
Sudirman and qx5rt. The table is the headline set. Other figures from the same run:

- SPIDER: disc precision 99.2%; numbering status precision confirmed 100% (38 studies), probable 100% (21), unconfirmed 93%
  (28, 2 wrong). Numbering ECE 0.064. Labelling time median 5.4 s per study under 8 parallel workers.
- Sudirman: numbering status precision confirmed 100% (94), probable 95% (216, 11 wrong), unconfirmed 78% (205, 45 wrong).
  Labelling time median 5.1 s.
- qx5rt: confirmed 100% (11), probable 100% (11), unconfirmed 76% (33). Labelling time median 2.2 s.
- Phantom, cartoon: outcomes 3 pass, 1 unsure, 19 silent error. Confidence is poorly calibrated there (numbering ECE 0.45).
- Phantom, realistic: outcomes 1 pass, 22 unsure, no silent error. All 23 scenarios are unconfirmed; numbering AUROC 0.38.

### SPIDER protocol note

The model card (`docs/model_card.md`, held-out table) reports SPIDER validation with label accuracy of matched discs 0.962,
all truth 0.955 and median centre error 1.5 mm. This run gives 0.981, 0.976 and 0.60 mm. Recall (0.994 there, 0.995 here)
agrees. The two runs use different code paths: the model card comes from `tools/train_labeller.py evaluate`, while the
benchmark uses the shipped `label_spine` with T2 paired with its T1 sibling. The cause of the centre-error gap has not been
established here; it is recorded as an open question for whoever maintains the model card. The golden numbers are the
benchmark's own, and the regression check does not compare them with the model card.

## 5. Golden numbers and the regression check

`benchmarks/golden.json` holds, for every section that ran, the cases count and each gated metric with its value,
direction, tolerance and unit. Scenario outcomes are stored per phantom scenario. Only the gated metrics are stored
(see table below); information-only metrics are not.

| Metric kind | Gated metrics | Rule | Default tolerance |
|---|---|---|---|
| Rates, higher is better | recall, precision, label accuracy (both), fully correct, numbering correct rate, confirmed/probable/announced precision | fail if it drops by more than the tolerance | 0.5 pp |
| Announced coverage | announced coverage | fail if it drops by more than the tolerance | 2 pp |
| Centre and midline errors | centre median / p90, midline median | fail if it rises by more than the tolerance | 0.5 mm / 1.0 mm / 1.0 mm |
| Calibration | disc and numbering ECE (lower), AUROC (higher) | as above | ECE 0.01 / 0.02; AUROC 0.01 / 0.02 |
| Counts | wrong announced, silent errors, errors | fail on any increase | 0 |
| Scenario outcomes | each phantom scenario | fail if it moves down: pass > unsure > silent error | none |
| Sample | number of cases per section | fail if it differs (the numbers are then not comparable) | none |

Tolerances live in the golden file, so a maintainer can change one with a reviewed edit. A section that was skipped
(no data) or not run is reported and not failed. The labeller fingerprint is compared too: a change of labeller source
is noted at the top of the report, but is not a failure by itself.

What a failing check looks like (illustrative):

```
section           metric                              golden     current        change         tol  verdict
spider            label_accuracy_all_truth            97.62%      97.00%      -0.62 pp     0.50 pp  REGRESSION
phantom_realistic scenario baseline                     pass      unsure                            REGRESSION

RESULT: FAIL, 2 regression(s):
  - spider.label_accuracy_all_truth dropped by 0.62 pp (97.62% -> 97.00%), tolerance 0.50 pp
  - phantom_realistic.scenario baseline: pass -> unsure
```

Exit codes: `0` pass, `1` regression, `2` usage error (for example `--check` with `--update-golden`).

### Updating the golden numbers

Run the full benchmark (or at least the sections the change touches), review the diff of `benchmark.md` against the
previous golden, then:

```
tools/benchmark.py --update-golden benchmarks/golden.json
```

This rewrites the sections that ran and keeps the others and any tolerances already in the file. Only update after
review: a regression that is accepted on purpose should be explained in the commit message, and an update must never
be used to hide a failure. Check the diff in `git diff benchmarks/golden.json` before committing.

## 6. Continuous integration

`.github/workflows/benchmark.yml` is manual (`workflow_dispatch`). It installs the package, runs
`tools/benchmark.py --suites phantom --check benchmarks/golden.json` and uploads the summary as an artifact. The
datasets are not available in CI, so the data sections are not run there. The ordinary CI workflow is unchanged.

Full runs happen on a maintainer's machine with the datasets installed. Record the labeller fingerprint and the golden
change in the pull request description.

## 7. Datasets and licences

The benchmark reads the public datasets, computes numbers, and writes numbers only. It never copies, redistributes or
commits images, masks, label files or patient attributes. Outputs contain no paths, file names or case identifiers,
and the cache holds labeller outputs only.

| Dataset | Licence | Use here |
|---|---|---|
| SPIDER (Lumbar spine MRI, sagittal T1/T2, masks) | CC BY 4.0 | numbers only; attribute in any publication |
| Sudirman lumbar DICOM and label maps | CC BY 4.0 | numbers only; attribute in any publication |
| OSF qx5rt multi-scanner lumbar set | **CC BY-NC 4.0** (non-commercial) | numbers only; never redistribute the data; non-commercial use only |
| Synthetic phantom (`spinescope.testing`) | Apache-2.0 (this repository) | no restriction |

Attribution follows `docs/datasets.md`. Redistribution of any of the three datasets is out of scope; the benchmark
is a reader of data the user already holds.

## 8. Limitations

- The golden numbers are one machine's run. The phantom suite is deterministic, and a recomputation (`--refresh`)
  reproduced it exactly on the same machine. Results on other platforms (CI on Linux) may differ slightly at the
  ONNX Runtime level; tolerances absorb small differences, and a flip of a single borderline case is a visible change.
  The first CI run should be checked against this file before it is trusted.
- Sudirman and qx5rt annotate only a few discs per study, so precision and any false-positive measure is not available
  there.
- The phantom suites are synthetic. They test mechanics and robustness, not clinical accuracy.
- SPIDER validation has 87 volumes and qx5rt 55, so one study moves their rates by about 1 to 2 pp; small changes in
  these sections are noise until more data is added.
- Latency depends on the machine and on the number of parallel workers; it is recorded and never gated.
