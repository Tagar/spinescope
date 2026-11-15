# lab04: vertebral numbering evidence

Module: `spinescope.anatomy.numbering_checks` (new). It does not modify `numbering.py` or `pipeline.py`;
the integrator wires `numbering_evidence(study, anchors)` in.

## What it answers

1. Which disc is L5-S1 (the lowest disc's true level versus its provisional count-from-the-sacrum name).
2. Is there a transitional vertebra (sacralized L5 or lumbarized S1).

Offset convention. `offset` is the true level of the lowest visible disc minus its provisional level, in lumbar steps:

- 0: lowest disc is L5-S1 (normal).
- +1: lowest disc is L4-L5. The L5 body is sacralized and the count is shifted, so every provisional name moves up one level (`relabel_discs`).
- -1: lowest disc is L5-L6. The S1 body is lumbarized, so only the lowest disc is renamed.

`NumberingEvidence` carries `offset`, `offset_confidence`, the posterior over {-1, 0, +1}, `transitional`
(offset not 0 with posterior >= 0.5), `transitional_kind`, the per-cue `votes` (each with probabilities and
evidence), `warnings` and `not_evaluated`.

## Cues

| Cue | Implemented | Source of prior or weight | Calibration |
|---|---|---|---|
| Oblique slab (lowest slab matched to a labelled disc by normal within 12 deg and disc centre inside the slab block) | yes | Lowest slab aimed at L5-S1 with p 0.7, L4-L5 with p 0.2 (judgement). ACR MR adult spine practice parameter: L5-S1 stack parallel to the L5-S1 disc, separate stacks when the angle differs (wording as summarised in search results, not checked against the full text). | Not calibrated: no labels for slabs in any public set here. Only cue-firing checked (below). |
| Sacral slope (lowest disc most tilted by >= 3 deg) | yes | Konin & Walz 2010 and the general sacral-slope literature; direction only. | SPIDER training oracle: 167/172 (0.971). SPIDER validation: 35/38 (0.921). Used 0.8 for offset 0 to allow anchor error. Shifted offsets have no labelled cases, so 0.5. |
| Conus tip level (weak prior, mean lower third of L1) | yes | Saifuddin et al., Spine 1998;23(13):1452-6, doi 10.1097/00007632-199807010-00005: 504 adults, mean lower third of L1, range mid T12 to upper L3. | Not calibrated; SD of about one level and a 75% uniform mixture are judgement. |
| Transitional prior | yes | Konin & Walz, AJNR 2010;31(10):1778-86, doi 10.3174/ajnr.A2036: LSTV prevalence 4-30% in the general population (range from the prior literature). Prior: 15% total, split evenly (0.075 / 0.85 / 0.075). | Not measured here. |
| Axial transverse process vs sacral ala (Castellvi typing) | no | Konin & Walz: sagittal alone can miss type II and some type III. | Needs image segmentation of the sacrum and transverse processes. Listed in `not_evaluated`. |
| Iliac crest level, aortic bifurcation (L4), psoas | no | Usually cited as weak priors. | Need crest or vessel localisation. Listed in `not_evaluated`. |
| Localizer rib count from T12 | no | - | Needs rib localisation. Listed in `not_evaluated`. |
| S1-S2 remnant disc, sacral canal end (sagittal) | no | - | Would need the thin-interval rejections from `analysis` and the S1 body. Listed in `not_evaluated`. |

Combination: posterior is proportional to prior times the product of each cue's floored likelihood (floor 0.05),
treating cues as independent. Ties resolve to offset 0.

## Calibration and validation numbers

Data: SPIDER (`~/data/public/spider`, `_t2` masks). Oracle disc normals from the masks (via
`spinescope.anatomy._validate.truth_from_mask`). Assumption: the lowest disc (label 201) is L5-S1, which
is SPIDER's convention.

- Training: lowest disc most tilted by >= 3 deg in 167/172 cases (0.971).
  - By `num_vertebrae`: 6 vertebrae 56/58, 7 vertebrae 74/77, 8 vertebrae 30/30.
  - This suggests `num_vertebrae == 6` is not a transitional marker. SPIDER's vertebra count includes thoracic labels, so `num_vertebrae != 5` does not mean transitional anatomy.
- Validation (reported, not used for tuning): 35/38 (0.921).
- Caveat: the proxy assumes SPIDER's lowest disc is L5-S1, so it cannot detect transitional cases. It measures
  how often the tilt cue agrees with offset 0 when offset 0 is most likely. SPIDER has no shifted-offset labels, so
  its offset cannot be measured.

Sudirman (`~/data/public/lumbar_spine_mri_sudirman`, no level labels): cue-firing check only.

- 15-study subsample (first 15 folders by name), classical `label_spine` anchors, `numbering_evidence` with slabs.
- Every study has oblique axial slabs (three per T1/T2 series).
- The lowest slab matched a labelled disc in 6/15 studies; in 9/15 it passed through no labelled disc.
- Final offset 0 in 15/15 studies, offset confidence 0.54 to 0.93.
- This says nothing about accuracy. It shows the cues do not fire false transitional flags on this subsample,
  given the classical chain. Before calibration changes, the same subsample flagged 6/15 studies as transitional, which was driven by noisy normals (tilt, not most tilted) and a strong conus prior. Both were weakened.

## Limitations

- Sacralized L5 cannot be established without slab or axial evidence. In a synthetic sacralized phantom (lowest true disc L4-L5, provisional names shifted down) the tilt and conus cues do not overturn the prior (test `test_sacralized_spine_without_slab_is_not_established`).
- Lumbarization is modelled as renaming only the lowest disc, which follows the count-from-sacrum scheme. This is an assumption, not measured.
- The tilt cue uses disc normals from the classical chain, which are noisy (the chain's SPIDER recall is about 0.2; see `docs/validation.md`).
- Slab matching uses the slab's stack geometry; slab descriptions carry no level names in Sudirman, so the description cross-check in `numbering.py` does not apply there.
- Phantom tests use synthetic slabs placed on the phantom's true discs. The phantom has no transitional variant; the tests build shifted label sets by hand.

## Tests

`tests/test_numbering_checks.py`: 28 tests (relabelling, slab cue including angle and out-of-range
rejection, tilt, conus, combination, end-to-end evidence and JSON serialisation). All phantom-based.
