# lab14_perf: speed, memory, import time and the test budget

Branch `lab/lab14_perf` (merged with main 6c5e20f). Owner: thread control and loading in
`anatomy/ml/labeller.py`, `anatomy/pipeline.py`, `anatomy/ml/grid.py`; conus fix in `anatomy/conus.py`; lazy
imports in `spinescope/__init__.py`, `anatomy/__init__.py`, `measure.py`, `render/enhance.py`; test budget in
`pyproject.toml`, `tests/_labels.py`, `.github/workflows/ci.yml`; calibration tool `tools/calibrate_labeling.py`.

## Summary

| Metric (SPIDER validation volume, 4 threads) | main 6c5e20f | this branch, default | this branch, classical opinion on |
|---|---|---|---|
| `label_spine` latency, volume A / B | 2.88 s / 3.36 s | 0.51 s / 0.51 s | 1.82 s / 2.33 s |
| Peak RSS, `label_spine` in a fresh process | 1105 MB / 1078 MB | 423-438 MB | 649-668 MB |
| `label_spine_cli` (keyed load), one volume, one run | 3.5 s, 1089 MB (ORT default threads) | 0.73 s, 439 MB | - |
| `import spinescope` | 146 ms | 5 ms | - |
| `import spinescope.anatomy` | 595 ms | 40 ms | - |
| `import spinescope.anatomy.pipeline` | 605 ms | 360 ms | - |
| `import spinescope.anatomy.ml` (loads onnxruntime) | 605 ms | 265 ms | - |
| Wheel (`uv build --wheel`) | 5.96 MB | 5.93 MB | - |

Latency is the median of three fresh runs (main: `set_intra_op_threads(4)`; the CLI row is a single run with main at
ORT's default threads) on a 14-core laptop with the load average at 10-25. The machine was
shared with other jobs; read absolute times with that in mind. Volume A and B are the first two validation
volumes listed in `overview.csv` (sagittal T2, 17 slices, 272 x 272 network grid, 6 discs).

Test suite (`SPINESCOPE_THREADS=4`):

| Run | Result | Wall time |
|---|---|---|
| Default `pytest` (`-m "not slow"`) | 829 passed, 1 skipped, 36 deselected, 20 xfailed | 2:37 (pytest 157 s) |
| Full `pytest -m ""` | 841 passed, 1 skipped, 43 xfailed, 1 xpassed | 6:58 (shared machine) |
| Default run at the start of this wave (uncommitted WIP) | 832 passed | 6:08, under load average 60-80 |

The default run is under the 3-minute target at a load average of about 20. The full run passes (the xfails are the documented classical-backend
misses and cartoon-phantom limits, unchanged).

## What changed

### 1. One thread-control mechanism

main had `set_intra_op_threads(n)` and `session_options()` (lab13, used by the lumbosacral model). The WIP branch had
`Config.ml_intra_op_threads`, `ml_inter_op_threads`, `resolve_threads` and `SPINESCOPE_THREADS`. They are now one:

`resolve_threads(config)` returns `(intra_op, inter_op)`. Precedence:

1. an explicit non-zero `Config.ml_intra_op_threads` / `ml_inter_op_threads`;
2. `set_intra_op_threads(n)` (process-wide, kept for main's callers; `0` means ORT's default);
3. the environment variable `SPINESCOPE_THREADS` (`0` means ORT's default);
4. inside a `multiprocessing` child, one thread, so a pool of N workers does not oversubscribe the machine;
5. otherwise `0` (ORT's default).

`load_session(path, config=..., intra_op_threads=..., inter_op_threads=...)` caches sessions per
(path, intra, inter) under a lock. The lumbosacral model now uses `load_session` too, so there is one session cache.
`session_options(intra, inter)` is the only place that builds ORT options.

Intra-op threads on one volume (labeller only, single runs, shared machine): 1 thread 1.28 s, 2 threads 0.87 s,
4 threads 1.27 s, ORT default (14 cores) 1.02 s. The differences are within the noise at this load, so the main
process keeps ORT's default and pool workers use one thread each. The tools accept `--threads` (`tools/eval_labeling.py`,
`spinescope.anatomy._validate_qx5rt`, `tools/train_labeller.py evaluate`, the Sudirman evaluator through `threads=`).

### 2. Batch size 1, in-place output

`run_network` runs one slice at a time by default (`batch=1`). The WIP measurement on a 120-slice SPACE stack (not
re-run here) found batch 8 about 1.9x slower on CPU and about 250 MB larger at peak. Softmax is done in place and the
output array is allocated once.

### 3. Memory

- `grid.intensity_window` percentiles on the volume's own dtype (no float64 copy of the whole stack). The WIP note
  estimated about 1 GB saved on a 120-slice SPACE stack (not re-run here); percentiles differ by about 1e-7 relative.
- The network output of the primary stack is reused. Before, `label_spine` ran the network three times on the same
  stack (labeller, `vertebrae.attach_geometry`, `conus.attach_conus`). Both now take the existing `SegmentedStack`
  (`seg=` argument) when it belongs to the same stack, and recompute otherwise. This is the main latency and RSS win
  on top of the classical change.
- Keyed pixel loading: `pipeline.labelling_keys(study)` names the stacks `label_spine` reads (primary sagittal, the
  second sagittal series, every axial stack). `load_labelling_study(path)` reads the headers first, then the pixels
  of those stacks only. `label_spine_cli` uses it. Anchors are identical to a full load (tests in
  `tests/test_labeling_loading.py`).
- Viewer export: zlib level 6 instead of 9. The WIP note measured it about 9x faster on a real volume with output about
  1% larger (not re-measured here).

### 4. Import time

- `spinescope` and `spinescope.anatomy` load their public names on first use (PEP 562 `__getattr__`), so
  `import spinescope` pulls numpy and the package only.
- scikit-image is imported inside the two functions that need it (`measure.poly_stats`, `render.enhance.denoise`).
- The labelling pipeline (scipy, the ONNX labeller) loads on first use of `label_spine`.
- CI asserts that `import spinescope` does not load onnxruntime, skimage, matplotlib, scipy or pydicom.

### 5. Classical second opinion: opt-in

`Config.classical_second_opinion` is `False` by default. Inside the learned numbering it used to run the whole
classical labeller on the primary stack. It cost about 1.3-1.9 s per study (its midline symmetry search is the bulk)
and its only effect is a weak tilt vote and the `classical_agree` calibration feature.

Evidence, SPIDER validation (87 volumes, current pipeline; records in `/tmp`, nothing committed):

| | classical opinion off (default) | on |
|---|---|---|
| Studies where the classical discs agree geometrically | - | 2 of 87 |
| Label accuracy, fused (626 matched discs) | 0.9808 | 0.9808 |
| Numbering correct | 85 of 87 | 85 of 87 |
| Studies with a different applied shift, relabel or raw labels | - | 0 |
| Calibrated numbering confidence, largest per-study change | - | 0.014 (median 0) |

lab08 found the same: 5 of 267 studies agreed geometrically. With the flag off, `classical_agree` is 0 for every study.
To turn it on, pass `Config(classical_second_opinion=True)`, or use `calibrate_labeling.py collect
--classical-second-opinion` (a separate cache folder).

### 6. Calibration after the change

The features changed (`classical_agree` is now 0 by default), so the calibration was re-collected with the current
pipeline and refit, as `tools/calibrate_labeling.py` does: `collect` on SPIDER training (360), validation (87) and
Sudirman (515), then `fit` on training only. The refit was **not written** to `calibration_v1.json`. Comparison
(packaged calibration vs refit, same records):

| Set | Quantity | Packaged (shipped) | Refit |
|---|---|---|---|
| Validation, disc | ECE / AUROC / wrong below 0.5 | 0.011 / 0.821 / 0.118 | 0.016 / 0.753 / 0.059 |
| Validation, numbering | ECE / AUROC / correct rate | 0.041 / 1.00 / 0.977 | 0.030 / 1.00 / 0.977 |
| Sudirman, disc | ECE / AUROC / wrong below 0.5 | 0.053 / 0.691 / 0.069 | 0.063 / 0.675 / 0.060 |
| Sudirman, numbering | ECE / AUROC / wrong below 0.5 | 0.100 / 0.804 / 0.055 | 0.099 / 0.817 / 0.018 |

The refit trades disc ranking on validation (AUROC down 0.07) for fewer confident wrong discs, and improves numbering
on all three measures. That is not a clear win, and choosing between them on validation would be tuning on it, so
the shipped `calibration_v1.json` is unchanged. Keeping it is valid for the default pipeline: `classical_agree` is
a legal input (0 = no classical agreement), and the calibrated confidences move by at most 0.014 when the opinion
is switched on. Decision for the owner: write the refit (`calibrate_labeling.py fit --write`) if the numbering gain
is preferred over the disc ranking.

### 7. Test budget

Measured with `--durations`: the default run spent most of its time in classical-backend phantom runs (about 2.5-3 s
each), the robustness suite (23 scenarios, one classical run each) and CLI tests that label the phantom.

- `slow` marker, excluded by default (`addopts = "-m 'not slow'"`). Marked: SPIDER dataset runs (validation samples,
  calibration, qx5rt, Sudirman, the ood SPIDER guard) and 14 of the 23 robustness scenarios. The other nine
  (`baseline`, `t2_only`, `field_3T`, `fov_T10_sacrum`, `lumbarised_S1`, `dark_discs`, `scoliosis`,
  `motion_ghosting`, `few_slices_7`) cover each family in the default run. Nothing is dropped from CI: the full job
  runs everything.
- `tests/_labels.py`: `label_spine` and `label_spine_ml` memoised per study content (SHA-256 of voxels, geometry and
  public metadata) and config, with a deep copy on return. Used by most labelling tests.
- New tests: `test_labeling_threads.py` (precedence, pool-child cap, cache, thread-safety), `test_labeling_loading.py`
  (keyed loading), a conus no-consensus regression test, and a test that the classical opinion is opt-in.

CI runs `pytest -m "not slow"` on every OS and Python, and a separate `test-full` job runs `pytest -m ""` once.

## Bugs found on the way

- `conus.conus_from_segmentation`: when the slice tips had no consensus (two tips more than `conus_agree_mm` apart)
  `agree` was empty, `np.mean` returned NaN and the function raised `TypeError`. `attach_conus` caught it and
  reported "locator failed" for that study. Found in the calibration collect on SPIDER validation volume `257_t2`.
  Fix: a non-finite tip is dropped, and no consensus gives "not located: cord tips disagree in z".

## Things tried that did not help

- Vectorising the midline mirror-similarity over offsets (`midline_robust._symmetry_curve`): the output was identical
  but the run was slower (3.8 s vs 2.7 s on one volume). Reverted.
- float32 correlation in `_ncc`: about 10% faster, output moved by 4e-7 mm. Not worth the precision change.

## Known limitations

- `backend="classical"` still costs about 2-3 s per volume, mostly the midline symmetry search (a second series
  doubles it). The learned default no longer runs it unless the flag is on.
- The phantom is slow to render (about 0.6 s per stack, per-voxel geometry). Not changed: its labels must not move.
- Timings were taken on a shared machine; re-run `bench` numbers on an idle machine before quoting them externally.
