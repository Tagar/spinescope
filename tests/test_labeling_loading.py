"""Only the stacks label_spine reads are loaded with pixels; labels do not change."""

from __future__ import annotations

import pytest

from spinescope import load_study
from spinescope.anatomy import pipeline
from spinescope.anatomy.pipeline import labelling_keys, load_labelling_study
from spinescope.study import Study


def test_labelling_keys_drop_coronal_and_localizer_sagittal(phantom_study: Study) -> None:
    keys = labelling_keys(phantom_study)
    assert keys[0] == phantom_study.best("sag").key
    assert "loc_cor" not in keys
    assert "loc_sag" not in keys
    assert all(phantom_study.stacks[k].plane in ("sag", "ax") for k in keys)
    assert len(keys) == len(set(keys))


def test_labelling_keys_empty_without_sagittal(phantom_study: Study) -> None:
    coronal_only = Study(stacks={k: v for k, v in phantom_study.stacks.items() if v.plane == "cor"}, info={})
    assert labelling_keys(coronal_only) == []


def test_keyed_load_reads_pixels_only_for_labelled_stacks(phantom_dir: dict[str, str]) -> None:
    study = load_labelling_study(phantom_dir["root"])
    loaded = {k for k, st in study.stacks.items() if st.vol.size}
    assert loaded == set(labelling_keys(load_study(phantom_dir["root"], load_pixels=False)))
    assert "loc_cor" not in loaded


@pytest.mark.parametrize("backend", ["classical", "ml"])
def test_keyed_load_gives_identical_anchors(phantom_dir: dict[str, str], backend: str) -> None:
    full = pipeline.label_spine(load_study(phantom_dir["root"]), backend=backend).to_dict()
    keyed = pipeline.label_spine(load_labelling_study(phantom_dir["root"]), backend=backend).to_dict()
    assert keyed == full
