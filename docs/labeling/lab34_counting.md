# lab34: top-chain check from wide sagittal coverage (counting)

Branch `lab/lab34_counting`. Owns `anatomy/counting.py` (the check), `anatomy/_validate_counting.py` (the
measurement, numbers only), the `counting` stage in `anatomy/pipeline.py`, the `counting` property in
`data/anchors.schema.json`, the keyed-load change in `labelling_keys`, and `tests/test_labeling_counting.py`.

## 1. The problem and the idea

The bottom cues (lumbosacral CNN, bottom gap, sacrum) cannot see an extra or missed disc higher up the chain. Clinical
practice counts from a landmark instead: C2 on a whole-spine localizer, or the thoracic count from the rib-bearing
T12 on a coronal image. The idea was an independent count from the widest sagittal coverage, compared with the
lumbar chain by geometry, and returned to the fusion layer as an offset vote (-1/0/+1) or as an abstention.

## 2. What was built

`anatomy/counting.py`, run after the labeller on every ML-backend study (`label_spine_details(..., counting=True)`):

1. **Candidates**: sagittal stacks other than the primary that reach at least `CountingConfig.reach_min_mm` (25 mm,
   `config.level_spacing_mm[0]`) beyond the primary's coverage, above or below. Localizers are included.
2. **Discs**: each candidate goes through the same learned labeller (`segment_stack`, `anchors_from_segmentation`).
3. **Geometry against the primary** (sagittal y, z, tolerance `config.disc_match_mm` = 6 mm):
   overlap agreement, discs the candidate shows above the primary's top disc inside the primary's coverage and not
   matched, whether the primary's top disc is seen by the candidate, and the reach above and below.
4. **Plausibility gate**: a candidate chain counts only if its median disc spacing is at least 25 mm and no single
   spacing exceeds 50 mm (`config.level_spacing_mm`). Otherwise it is recorded and cannot raise the flag.
5. **Landmark**: None. C2 and the T12 rib-head count need detectors that do not exist here. The learned labeller
   is trained on lumbar and thoracic sagittal slices, and no public set in use has cervical or rib labels.

The result is stored in `anchors.numbering["counting"]` (`available`, `reason`, `landmark`, `vote`,
`top_disagreement`, and per-candidate records). Names, status and warnings are not changed by it.

### Why no offset vote for top-chain errors

The fusion votes move the lowest disc's name by an offset. A missed or extra disc at the top changes how many discs
lie above the lowest one, not the lowest name. A count from C2 would fix the lowest name only through the presacral
count prior (23 discs from C2-3 down to L5-S1 in the usual 7 + 12 + 5 anatomy; the presacral count varies), and it
would not say which discs above are wrong. Top names would need a top-down relabel, which is not implemented.

## 3. Measurements

Measured with `python -m spinescope.anatomy._validate_counting spider|sudirman` on the shipped labeller (no
training or tuning in this lab). Numbers only; no identifiers are printed.

### 3.1 SPIDER validation (ground truth from the masks)

The primary is the best sagittal volume of each case, and the other validation volumes of the same case are the
candidates. The top-chain error of the primary is read from its mask: an unmatched truth disc or predicted disc above
the fourth-lowest truth disc (`TOP_FROM_BOTTOM = 3`).

| Measure | Value |
|---|---|
| Primary volumes measured (validation, at least two validation series in the case) | 37 |
| Primary volumes with a wider or other sagittal candidate | 1 |
| Primary top-chain errors (mask-based) | 2 of 37 |
| Flagged (top disagreement) | 0 |
| Catches / false alarms | 0 / 0 |
| Top-chain errors that had a candidate to check | 0 of 2 |

Most SPIDER cases share one field of view across their series, so almost no second series reaches beyond the
primary. The two top-chain errors fell in volumes with no candidate to check.

### 3.2 Sudirman sessions (no disc truth; consistency only)

One session per patient (first session by name), 515 patients. The Sudirman localizers are sagittal (3-slice and
1-slice fl2d series, 0.68 to 0.88 mm in plane) and reach 350 mm in z; most reach below the primary (down to the sacrum), and most of the sessions with a candidate also reach
at least 25 mm above the primary's top.

| Measure | Value |
|---|---|
| Sessions read / labelled | 515 / 504 |
| Sessions with a wider or other sagittal candidate | 452 |
| Candidates reaching at least 25 mm above the primary's top | 272 sessions |
| Candidates reaching below only | 180 sessions |
| Overlap candidates compared (discs in the overlap) | 817 |
| Overlap agreement at least 0.7 | 5.8 % |
| Overlap agreement exactly 0 | 71 % |
| Sessions with a top-chain flag | 193 of 504 (38 %) |
| Landmark found | 0 |

Reading: the candidate chains do not reproduce the primary's disc positions. The primary is validated on Sudirman
(lab20: level accuracy 0.91 against the axial truth). The localizer chains are not: examples are a 3-slice GRE
localizer with about 20 mm pitch between lumbar discs (the anatomy is 35 to 40 mm), and a single-slice localizer
that misses lumbar discs and has a 148 mm gap. A flag on 38 % of sessions is therefore dominated by localizer
detection failures. It cannot be scored against truth here, and it is not an error count.

### 3.3 Phantom

The standard phantom study has a 3-plane localizer with a 346 mm field of view. The check runs and records
evidence (the flag is raised on the cartoon, because the localizer chain is also out of distribution). No flag value
is asserted in the tests.

## 4. Decision

- **The vote is off, and no vote is produced.** There is no landmark, so the check abstains on every study. The
  fusion layer (`anatomy/fusion.py`) is unchanged: no `FusionConfig` switch was added, because there is nothing to
  switch on.
- **The evidence is recorded** (`anchors.numbering["counting"]`) and does not change names, status or warnings.
  Default on: it costs one labeller pass per wider series (about 0.5 s per localizer). It runs on most Sudirman
  sessions and on few SPIDER cases. `label_spine(..., counting=False)` skips it. Keyed loads now read the sagittal
  localizers too (`labelling_keys`; coronal series are still skipped), so keyed and full loads give the same anchors.
- **Counts that would decide a vote** (C2 or the T12 rib count) need a cervical or rib detector, plus labelled data to
  calibrate it. Neither exists in the public sets in use. This is the blocker, not the fusion code.

## 5. Limits

- No validated catch or false-alarm number exists for a top-chain vote: SPIDER has two top-chain errors and one
  candidate, and Sudirman has no disc truth above L3-L4.
- The labeller is not reliable on Sudirman localizers (see 3.2). The gate in section 2 stops implausible chains from
  raising the flag, but it does not make a plausible wrong chain right.
- qx5rt (CC BY-NC 4.0) was not used: it has no thoracic or localizer data for this question.
- The bottom coverage of the localizer (it reaches the sacrum on most Sudirman sessions) is recorded as `reach_below_mm`.
  It is a possible sacrum-confirmation source for a later lab. It is not used here.

## 6. Reproduce

```
.venv/bin/python -m spinescope.anatomy._validate_counting spider      # about 2.5 min, needs $SPINESCOPE_DATA/spider
.venv/bin/python -m spinescope.anatomy._validate_counting sudirman    # about 13 min for 515 sessions
.venv/bin/python -m pytest -q tests/test_labeling_counting.py
```
