"""Reading briefs and reader scopes (spinescope.ai.briefs): phantom-based, with varied phantoms."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from _labels import label_spine
from report_factory import minimal_valid
from spinescope import Study, load_study
from spinescope.ai.briefs import (
    CALIBRATION_GUIDANCE,
    STRUCTURE_FAMILIES,
    brief,
    geometry_facts,
    reading_tasks,
    series_table,
    verification_errors,
)
from spinescope.ai.briefs import READER_NAME_RE as NAME_RE
from spinescope.geometry import Stack
from spinescope.io.dicom import KEEP_TAGS
from spinescope.report.schema import ITEM_SECTIONS, global_group_names, level_group_names
from spinescope.study import from_stacks
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study

PHI = ("PHANTOM^SPINE", "PHANTOM-0001", "DOE^JANE", "Phantom General Hospital")


def _no_phi(text: str, root: str) -> None:
    for s in PHI:
        assert s not in text
    assert root not in text


@pytest.fixture(scope="session")
def anchors(phantom_study):
    return label_spine(phantom_study, backend="classical")


def test_brief_has_sections_and_is_phi_free(phantom_dir, phantom_study, anchors):
    text = brief(phantom_study, anchors, reader="reader_a")
    for head in (
        "1. Study",
        "2. Geometry and display conventions",
        "3. Levels and anchors",
        "4. Tools and images",
        "5. Calibration",
        "7. Findings format",
    ):
        assert head in text
    assert "Reader name: reader_a." in text
    assert "sag_t2" in text and "L4-L5" in text
    assert CALIBRATION_GUIDANCE.splitlines()[0] in text
    _no_phi(text, str(phantom_dir["root"]))


def test_brief_has_no_disallowed_dicom_attributes(phantom_study, anchors):
    text = brief(phantom_study, anchors)
    for tag in ("PatientName", "PatientID", "InstitutionName", "ReferringPhysicianName", "StudyDate"):
        assert tag not in KEEP_TAGS or tag not in text
        assert tag not in text


def test_brief_rejects_bad_reader_name(phantom_study):
    with pytest.raises(ValueError):
        brief(phantom_study, reader="reader a/../x")


def test_geometry_facts_match_the_stacks(phantom_study):
    facts = geometry_facts(phantom_study, None)
    sag = facts["stacks"]["sag_t2"]
    ax = facts["stacks"]["ax_t2"]
    st = phantom_study["sag_t2"]
    # the facts are derived from the same direction vectors the stack uses
    assert sag["screen_right"] == "posterior" and sag["screen_down"] == "inferior (feet)"
    assert ax["screen_right"] == "patient left" and ax["screen_down"] == "posterior"
    assert sag["slices"] == st.n
    # the phantom's mid-sagittal plane is the centre slice of an even stack (x = 0)
    assert abs(sag["midline_slice"] - (st.n - 1) / 2) <= 1.0
    assert sag["midline_source"] == "canal ridge (select_midline)"


def test_midline_from_anchors_is_used_when_present(phantom_study, anchors):
    facts = geometry_facts(phantom_study, anchors)
    assert facts["stacks"]["sag_t2"]["midline_source"] == "anchors.midline_x"


def test_series_table_is_technical_only(phantom_study):
    rows = series_table(phantom_study)
    assert {r["key"] for r in rows} == set(phantom_study.keys())
    for r in rows:
        assert set(r) <= {
            "key",
            "description",
            "plane",
            "contrast",
            "fat_suppressed",
            "post_contrast",
            "localizer",
            "slab_index",
            "slices",
            "rows",
            "cols",
            "pixel_mm",
            "thickness_mm",
            "spacing_mm",
            "fov_mm",
            "obliquity_deg",
            "screen_right",
            "screen_down",
            "slice_index_toward",
            "z_range_mm",
        }


def test_level_groups_are_each_in_exactly_one_structure_family():
    families = [g for _, _, groups in STRUCTURE_FAMILIES for g in groups]
    assert sorted(families) == sorted(level_group_names())  # every group once, none missing or repeated


def test_reading_tasks_per_level_structure_holistic_verify(phantom_study, anchors):
    tasks = reading_tasks(phantom_study, anchors)
    levels = [t for t in tasks if t.kind == "level"]
    assert [t.levels for t in levels] == [(lab,) for lab in anchors.disc_labels()]
    assert {t.kind for t in tasks} == {"level", "structure", "holistic", "verify"}
    assert len([t for t in tasks if t.kind == "structure"]) == len(STRUCTURE_FAMILIES)
    holistic = next(t for t in tasks if t.kind == "holistic")
    assert set(holistic.groups) == set(global_group_names()) | set(ITEM_SECTIONS)
    # readers are independent unless they depend on the merge
    assert all(t.depends_on == () for t in tasks if t.kind != "verify")
    assert next(t for t in tasks if t.kind == "verify").depends_on == ("merge",)
    ids = [t.id for t in tasks]
    readers = [t.reader for t in tasks]
    assert len(set(ids)) == len(ids) and len(set(readers)) == len(readers)
    for t in tasks:
        assert NAME_RE.match(t.reader)
        assert f"Reader name: {t.reader}." in t.prompt
        assert "7. Findings format" in t.prompt
        assert t.scope in t.prompt


def test_level_task_prompt_names_its_level_and_groups(phantom_study, anchors):
    t = next(t for t in reading_tasks(phantom_study, anchors) if t.id == "level-L4-L5")
    assert 'levels["L4-L5"]' in t.scope
    assert set(t.groups) == set(level_group_names())
    assert "sag_t2" in t.series


def test_reading_tasks_without_anchors_warn_and_skip_levels(phantom_study, caplog):
    with caplog.at_level(logging.WARNING, logger="spinescope.ai.briefs"):
        tasks = reading_tasks(phantom_study, None)
    assert "no disc anchors" in caplog.text
    assert not [t for t in tasks if t.kind == "level"]
    assert {t.kind for t in tasks} == {"structure", "holistic", "verify"}


def test_verification_document_check():
    ok = {
        "checks": [
            {
                "path": 'levels["L4-L5"].disc.pfirrmann',
                "verdict": "refuted",
                "confidence": "medium",
                "evidence": [{"image": "a.png", "series": "sag_t2", "slices": [6]}],
            }
        ]
    }
    assert verification_errors(ok) == []
    bad = {"checks": [{"path": "", "verdict": "maybe", "confidence": "x", "evidence": []}]}
    assert len(verification_errors(bad)) == 4
    assert verification_errors([]) and verification_errors({"checks": "x"})


def test_findings_format_in_brief_is_the_schema_instructions(phantom_study):
    text = brief(phantom_study, None, reader="reader_b")
    assert '"schema_version": "1.0"' in text
    assert "levels" in text and "pfirrmann" in text
    assert minimal_valid("reader_b")["reader"] == "reader_b"  # the factory agrees on the reader key


# ------------------------------------------------------------------ varied phantoms


def _study_from(phantom: SpinePhantom, root) -> tuple[Study, dict]:
    info = write_standard_study(phantom, root, rows=128, cols=128, px=1.5)
    return load_study(info["root"]), info


def test_lordosis_and_noise_variant(tmp_path):
    ph = SpinePhantom(PhantomSpec(segment_lordosis=(2.0, 2.0, 3.0, 4.0, 6.0, 10.0), noise=0.06, seed=3))
    st, info = _study_from(ph, tmp_path / "lord")
    an = label_spine(st, backend="classical")
    text = brief(st, an)
    _no_phi(text, str(info["root"]))
    tasks = reading_tasks(st, an)
    assert any(t.kind == "level" for t in tasks)


def test_missing_sequences_single_sagittal(phantom_study, anchors):
    only_sag = Study(stacks={"sag_t2": phantom_study["sag_t2"]}, info=dict(phantom_study.info))
    an = label_spine(only_sag, backend="classical")
    text = brief(only_sag, an)
    assert not [line for line in text.splitlines() if line.startswith("  ax_")]
    tasks = reading_tasks(only_sag, an)
    assert all(t.series == ("sag_t2",) for t in tasks if t.kind in ("structure", "holistic"))


def test_cropped_field_of_view_variant(tmp_path):
    ph = SpinePhantom(PhantomSpec(degenerated={"L5-S1": 0.5}))
    info = write_standard_study(ph, tmp_path / "crop", rows=96, cols=96, px=1.2)
    st = load_study(info["root"])
    text = brief(st, label_spine(st, backend="classical"))
    assert "96x96 px" in text
    _no_phi(text, str(info["root"]))


def test_flipped_acquisition_gives_the_same_conventions(phantom_study):
    """A sagittal stack stored with its in-plane axes flipped describes the same anatomy; the display
    conventions must come out identical after load_study's reorientation."""
    st = phantom_study["sag_t2"]
    cols = st.shape[2]
    flipped_vol = st.vol[:, :, ::-1].copy()
    flipped = Stack(
        key="flipped",
        vol=flipped_vol,
        ipp=st.ipp + (cols - 1) * st.col_mm * st.row_dir,
        row_dir=-st.row_dir,
        col_dir=st.col_dir,
        row_mm=st.row_mm,
        col_mm=st.col_mm,
        thickness=st.thickness,
        meta={"dicom": dict(st.meta.get("dicom", {})), "role": dict(st.meta.get("role", {}))},
    )
    study = from_stacks([flipped])
    (key,) = study.keys()  # classified by role, so it is keyed as the original sagittal T2
    assert key == "sag_t2"
    orig = geometry_facts(phantom_study, None)["stacks"]["sag_t2"]
    assert geometry_facts(study, None)["stacks"][key]["screen_right"] == orig["screen_right"]
    assert geometry_facts(study, None)["stacks"][key]["screen_down"] == orig["screen_down"]
    assert series_table(study)[0]["slices"] == st.n


def test_brief_is_deterministic(phantom_study, anchors):
    a = brief(phantom_study, anchors, reader="reader_a")
    b = brief(phantom_study, anchors, reader="reader_a")
    assert a == b
    assert np.isfinite(anchors.discs[anchors.disc_labels()[0]].confidence)
