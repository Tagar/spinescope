# lab10: vendor coverage for `label_spine` (OSF qx5rt multi-scanner lumbar set)

Branch `lab/lab10_vendors`, labeller = `label_spine(backend="auto")` (the shipped SPIDER-trained ONNX
model, not changed or tuned here). Dataset: Al Khalil et al., multi-scanner lumbar vertebral body and disc
segmentation database (OSF `qx5rt`, doi:10.17605/OSF.IO/QX5RT, Scientific Data 2022), **CC BY-NC 4.0**:
research use only, no commercial use. Data at `~/data/public/lumbar_multiscanner_qx5rt/raw`, never committed.

## Why this set

The validation so far covered Siemens and Philips only (SPIDER, Sudirman). This is the only openly
downloadable lumbar MRI set found with disc level labels that includes a GE scanner. The survey of other
candidates (AISSLab GE cohort, spine-generic, LSMA-PQR, RSNA LumbarDISC, Toshiba and Hitachi cohorts,
TCIA, MRSpineSeg, OpenNeuro, PhysioNet, Figshare) is in `docs/datasets.md`.

## Coverage in this set (what the numbers can and cannot say)

- 55 sessions from 34 patients: Philips 39, Siemens 15, **GE 1** (one GE Signa session, ID38 scan 1).
- **Field strength is not in the release.** The scan table gives vendor and model only, so every
  field-strength split is "not stated". The GE result is a single session and says nothing about GE in
  general.
- No Canon, Toshiba, Hitachi, Fujifilm, United Imaging, Esaote or low-field scanner is present.
- The sequence graded is the non-contrast T1 (the sequence the masks were drawn on, per the paper). The
  set is sagittal; the T1 stacks have about 15 slices of 3.3 mm.

## Ground truth and scorer

Adapter: `spinescope.anatomy._validate_qx5rt` (`discover`, `truth_levels`, `score_levels`, `summarise_by`).

- Truth: the disc masks `Labels/BS_L<a>_<b>` (L1-L2 ... L4-L5). Each mask is thresholded at 0.5 and its
  centroid is taken in LPS mm on the T1 grid (`Stack.pix_to_lps`, same grid as the labeller's input).
- Level names come from the file names, so no rank assumption is needed.
- Match: the nearest predicted disc in the sagittal (y, z) plane, within 12 mm (the threshold of the
  other scorers). A level counts as correctly labelled only if that disc carries the same name.
- Reported: recall within 12 mm, label accuracy among matched levels, correct-name recall over all
  annotated levels, in-plane error (median, p90), level offset histogram.
- Not reported: disc precision. Only four discs per session are annotated, so predicted discs outside
  them are not false positives in any useful sense. The L5-S1 disc is not annotated at all.

## Headline

- All 55 sessions scored, none failed, 217 annotated discs. Most sessions have four (L1-L2 to L4-L5); two
  sessions have three and two because the release lacks some disc masks (220 - 3 = 217). Run sequentially
  in one process; no timing was recorded.
- No matched level is ever named too caudal: every naming error is one level too cranial (offset +1).
- **Position is good on every vendor.** Every annotated disc had a predicted disc within 12 mm in the
  sagittal plane (recall 1.00). Median in-plane error is 1.7 to 1.9 mm per vendor, p90 3.6 mm overall.
- **Naming is the problem, and it depends on the session.** Overall, 81.1 % of matched levels carry the
  right name. Philips is 88.9 % (153 levels), Siemens 60.0 % (60 levels), GE 100 % (4 levels, one session).
- The errors are whole-session shifts, not scattered misses. In 10 of 55 sessions (6 Siemens, 4 Philips)
  all four discs are named one level too cranial (offset +1, 4 of 4). In the other 45 sessions, the names
  are right or the session has one stray error. The 10 shifted sessions are 40 of the 41 level errors.
- Cause, as far as these labels can show: in the shifted sessions the labeller predicts two discs
  caudal to the annotated L4-L5 (the other sessions predict one). The lowest disc is then named L5-S1 and
  the whole column moves up one. By eye, on one shifted Siemens session (ID05 Siemens Verio, overlay in
  the renders folder, not committed), the extra caudal disc sits at the sacral level. The masks do not
  label L5-S1, so the labels cannot say whether the caudal disc is a real L5-S1 with a short L5 or a
  false disc. Same root cause as the known weakness: the model has no sacrum class and takes the lowest
  disc as L5-S1 (`docs/model_card.md`, lines 21 and 69-70).
- The labeller's own warning, "numbering unconfirmed: no vertebral body below the lowest disc", was raised
  on **all 55 sessions** with numbering confidence 0.36 to 0.39, including the correct ones. The warning
  is always on here, so it does not separate right from wrong on this set.

## Results by vendor, field and sequence

Pooled over annotated levels (`summary.json` in scratch, numbers below copied from it).

| Group | Sessions | Levels | Recall 12 mm | Label accuracy (matched) | In-plane median / p90 (mm) | Offsets 0 / +1 |
|---|---|---|---|---|---|---|
| All | 55 | 217 | 1.00 | 0.811 | 1.74 / 3.61 | 176 / 41 |
| GE | 1 | 4 | 1.00 | 1.00 | 1.89 / 3.99 | 4 / 0 |
| Philips | 39 | 153 | 1.00 | 0.889 | 1.79 / 3.79 | 136 / 17 |
| Siemens | 15 | 60 | 1.00 | 0.600 | 1.73 / 3.32 | 36 / 24 |

By model (sessions in brackets):
Philips Achieva 28 (0.89), Elition 5 (0.80), Ingenia 5 (0.95), Achieva external 1 (1.00);
Siemens Avanto 5 (0.80), Verio 4 (0.50), Espree 2 (0.50), Aera 1 (0.00), Symphony 1 (0.00), Amira 1 (1.00),
Magnetom 1 (1.00); GE Signa 1 (1.00).

- **Field strength:** not stated in the release, so there is no field split. Every number is "not stated".
- **Sequence:** the non-contrast T1 only (the sequence the masks were drawn on). T2, Dixon, STIR and
  contrast T1 were not scored: their masks are the T1 masks registered, not drawn on them.
- **The GE result is one session (four discs).** It says the labeller handled one GE Signa T1 correctly,
  and nothing more. It is not a GE accuracy estimate.
- No Canon, Toshiba, Hitachi, Fujifilm, United Imaging, Esaote or low-field data was available. Those
  vendors remain untested.

## Caveats

- This is an out-of-distribution test. The labeller was trained on SPIDER (Siemens and Philips at 1.5 T and
  3 T, sagittal T1, T2 and T2 SPACE). It was not tuned on this set, and no threshold or parameter was
  changed. Nothing here is a training or calibration input.
- The stacks are short: about 15 sagittal slices of 3.3 mm, 50 mm of left-right coverage. SPIDER stacks are
  thinner and longer. Part of the gap may be coverage, not vendor.
- The masks are vertebral bodies (posterior elements excluded) and discs. Labels are per session, and
  the L5-S1 disc is not labelled, so it cannot be scored.
- Scores are against manual masks from one research group, drawn in 2D-thick slices. Their accuracy is
  not an expert-consensus ground truth.
- Only one licence-compatible GE session is available here. A real GE estimate needs the AISSLab GE cohort
  (licence to be clarified with the authors) or a GE set with level labels.

## Visual check

Two overlays were read by eye (renders stay in `/tmp/spinescope-lab10/renders`, not committed): the GE
session, where the names and positions match the masks; and one shifted Siemens Verio session, where
the caudal extra disc is visible. No other renders were graded by eye, so the per-session shift
pattern above comes from the level scorer, not from a visual audit.

## Reproduce

```
spinescope.datasets.fetch("lumbar_multiscanner_qx5rt")          # 4.27 GB, md5 per file
python -m spinescope.anatomy._validate_qx5rt --out /tmp/qx5rt.json
pytest -m dataset tests/test_labeling_qx5rt.py                   # skips when the release is absent
```

## Next step (not done here)

The labels cannot separate a real L5-S1 from a false caudal disc. A fix belongs in the numbering stage:
for stacks that end on the sacrum side, do not assume the lowest disc is L5-S1 unless a vertebra below it
is seen. That needs a sacrum or vertebra check on the stack, and it should be tested on this set and on
SPIDER before anything is tuned. Do not tune on this set.
