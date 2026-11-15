"""Phantom tests for spinescope.anatomy.label_spine.

Tests marked ``xfail`` document limits of the current chain detector on the phantom: they are
expected to fail until the endplate chain handles the phantom's L2-L3 disc (see docs/validation.md).
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from spinescope import load_study
from spinescope.anatomy import label_spine, label_spine_cli
from spinescope.study import from_stacks
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study


@pytest.fixture(scope="module")
def ph() -> SpinePhantom:
    return SpinePhantom(
        PhantomSpec(
            degenerated={"L4-L5": 0.2}, modic1=[("L4-L5", "inf", "L")], herniation={"L5-S1": ("central", 5.0)}
        )
    )


@pytest.fixture(scope="module")
def study(ph: SpinePhantom):
    return load_study(write_standard_study(ph, tempfile.mkdtemp())["root"])


def _dist(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def test_midline_and_conus_on_phantom(study, ph):
    anc = label_spine(study)
    assert anc.midline_x is not None
    assert abs(anc.midline_x - ph.truth["midline_x"]) <= 1.5
    assert anc.conus is not None
    assert abs(anc.conus["z"] - ph.truth["conus_z"]) <= 6.0


@pytest.mark.xfail(strict=False, reason="conus level inherits the shifted labels on the phantom")
def test_conus_level_on_phantom(study, ph):
    assert label_spine(study).conus["level"] == ph.truth["conus_level"]


def test_bottom_disc_is_l5_s1_and_discs_are_sorted(study, ph):
    anc = label_spine(study)
    labels = anc.disc_labels()
    assert labels[-1] == "L5-S1"
    assert anc.numbering["confidence"] > 0.5
    for lab in labels:
        assert 0.0 <= anc.discs[lab].confidence <= 1.0
        assert anc.discs[lab].evidence


@pytest.mark.parametrize("lab", ["L5-S1", "L4-L5", "L3-L4"])
def test_disc_centres_within_3mm_for_lower_discs(study, ph, lab):
    anc = label_spine(study)
    assert lab in anc.discs
    assert _dist(anc.discs[lab].center, ph.truth["discs"][lab]["center"]) <= 3.0


@pytest.mark.xfail(strict=False, reason="chain misses the L2-L3 endplates on the phantom; labels above shift")
def test_all_discs_in_fov_labelled_and_within_3mm(study, ph):
    anc = label_spine(study)
    truth = ph.truth["discs"]
    for lab in ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"):
        assert lab in anc.discs, lab
        assert _dist(anc.discs[lab].center, truth[lab]["center"]) <= 3.0


def test_t1_only_study_runs(ph):
    st = ph.render("sag", "t1", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    study = from_stacks([st])
    anc = label_spine(study)
    assert anc.source_stack == "sag_t1"
    assert anc.midline_x is not None


def test_no_sagittal_gives_warning_not_exception(ph):
    st = ph.render("ax", "t2", n=20, spacing=5.0, thickness=4.0, rows=96, cols=96, px=1.4)
    anc = label_spine(from_stacks([st]))
    assert anc.discs == {}
    assert any("no sagittal" in w for w in anc.warnings)


def test_single_sagittal_slice_does_not_crash(ph):
    st = ph.render("sag", "t2", n=1, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    anc = label_spine(from_stacks([st]))
    assert isinstance(anc.warnings, list)


def test_cli_hook_returns_dict_without_paths(ph, tmp_path):
    root = write_standard_study(ph, tmp_path / "study")["root"]
    d = label_spine_cli(root)
    assert "discs" in d and "numbering" in d
    assert str(tmp_path) not in str(d)


def test_unknown_backend_is_rejected(study):
    with pytest.raises(ValueError, match="labelling backend"):
        label_spine(study, backend="ml")
    assert label_spine(study, backend="classical").method
