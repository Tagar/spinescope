"""Messy folders: several studies, junk objects, duplicates, gaps, varying geometry, lazy keys."""

from __future__ import annotations

import json
import logging
import os
import shutil

import numpy as np
import pydicom
import pytest
from pydicom.uid import MRImageStorage, generate_uid

from spinescope import list_studies, load_study
from spinescope.geometry import Stack
from spinescope.io.dicom import UnsupportedPixelDataError
from spinescope.testing.phantom import (
    FAKE_PHI,
    write_dicom_series,
    write_non_image_object,
)


def _sag_series(phantom, root, *, n=9, study=None, series=1, px=3.0, offset=0.0, **kw):
    st = phantom.render("sag", "t2", n=n, rows=48, cols=48, px=px, spacing=4.0)
    if offset:
        st = Stack(
            key="",
            vol=st.vol,
            ipp=st.ipp + offset * st.normal,
            row_dir=st.row_dir,
            col_dir=st.col_dir,
            row_mm=st.row_mm,
            col_mm=st.col_mm,
            thickness=st.thickness,
            meta=st.meta,
        )
    paths = write_dicom_series(
        st,
        root,
        series_number=series,
        contrast="t2",
        study_uid=study or generate_uid(),
        frame_uid=generate_uid(),
        **kw,
    )
    return st, paths


@pytest.fixture
def two_studies(phantom, tmp_path):
    """Study A: 9 sagittal T2 slices. Study B: 3 slices of another sagittal T2. Same folder."""
    root = tmp_path / "mixed"
    a_uid, b_uid = generate_uid(), generate_uid()
    st_a, _ = _sag_series(phantom, root / "a", study=a_uid, series=1)
    _sag_series(phantom, root / "b", n=3, study=b_uid, series=2)
    return root, a_uid, b_uid, st_a


def test_list_studies_orders_by_size_and_hides_uids(two_studies):
    root, a_uid, b_uid, _ = two_studies
    infos = list_studies(root)
    assert [i.index for i in infos] == [0, 1]
    assert [i.n_images for i in infos] == [9, 3]
    blob = json.dumps([i.as_dict() for i in infos])
    assert a_uid not in blob and b_uid not in blob


def test_default_picks_largest_study_and_warns(two_studies, caplog):
    root, *_ = two_studies
    with caplog.at_level(logging.WARNING, logger="spinescope.io.dicom"):
        st = load_study(root)
    assert st.keys() == ["sag_t2"] and st["sag_t2"].n == 9
    assert any("2 studies" in r.getMessage() for r in caplog.records)
    assert len(st.report["studies"]) == 2


def test_select_study_by_index_or_id(two_studies):
    root, *_ = two_studies
    infos = list_studies(root)
    assert load_study(root, study=1)["sag_t2"].n == 3
    assert load_study(root, study=infos[1].study_id)["sag_t2"].n == 3
    assert load_study(root, study=infos[0].index)["sag_t2"].n == 9
    with pytest.raises(KeyError):
        load_study(root, study=5)
    with pytest.raises(KeyError):
        load_study(root, study="0123456789ab")


def test_single_study_does_not_warn(phantom_dir, caplog):
    with caplog.at_level(logging.WARNING, logger="spinescope.io.dicom"):
        load_study(phantom_dir["root"])
    assert not [r for r in caplog.records if "studies" in r.getMessage()]


def test_junk_objects_are_ignored(phantom, tmp_path):
    root = tmp_path / "junk"
    study = generate_uid()
    st, paths = _sag_series(phantom, root / "s", study=study, series=3)
    jdir = root / "junk"
    jdir.mkdir(parents=True, exist_ok=True)
    write_non_image_object(jdir / "sr.dcm", "SR", study_uid=study)
    write_non_image_object(jdir / "ko.dcm", "KO", study_uid=study)
    write_non_image_object(jdir / "pr.dcm", "PR", study_uid=study)
    write_non_image_object(jdir / "DICOMDIR", "DICOMDIR", study_uid=study)
    sc = write_non_image_object(jdir / "screenshot.dcm", "SC", study_uid=study)  # RGB, burned-in text
    # the same RGB screenshot relabelled as MR: still a colour object, must not be anatomy
    ds = pydicom.dcmread(sc)
    ds.Modality = "MR"
    ds.SOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.save_as(jdir / "screenshot_mr.dcm", enforce_file_format=True)
    shutil.copy(paths[2], jdir / "duplicate_of_slice3.dcm")  # same SOP instance UID as slice 3
    (jdir / "notes.txt").write_text("not dicom")
    st_loaded = load_study(root)
    assert st_loaded.keys() == ["sag_t2"]
    assert st_loaded["sag_t2"].n == st.n
    ign = st_loaded.report["scan"]["ignored"]
    assert ign["non_image_object"] == 3
    assert ign["secondary_capture"] == 1
    assert ign["colour_or_screenshot"] == 1
    assert ign["duplicate"] == 1
    assert str(root) not in json.dumps(st_loaded.report)


def test_missing_slices_are_detected_and_recorded(phantom, tmp_path, caplog):
    root = tmp_path / "gap"
    _st, paths = _sag_series(phantom, root, n=9)
    for p in paths[3:5]:  # drop two consecutive slices -> one gap that spans three spacings
        os.remove(p)
    with caplog.at_level(logging.WARNING, logger="spinescope.io.dicom"):
        st = load_study(root)
    geo = st["sag_t2"].meta["geometry"]
    assert geo["gap_count"] == 1
    assert geo["estimated_missing_slices"] == 2
    assert geo["uniform"] is False
    assert any("gap" in r.getMessage() for r in caplog.records)
    assert any("gap" in w for w in st.report["warnings"])


def test_irregular_spacing_is_flagged(phantom, tmp_path):
    base = phantom.render("sag", "t2", n=5, rows=32, cols=32, px=3.0, spacing=4.0)
    pos = np.array([0.0, 4.0, 8.0, 9.0, 13.0])  # jittered spacing, no gap
    irregular = Stack(
        key="",
        vol=base.vol,
        ipp=base.ipp[0] + pos[:, None] * base.normal,
        row_dir=base.row_dir,
        col_dir=base.col_dir,
        row_mm=base.row_mm,
        col_mm=base.col_mm,
        thickness=base.thickness,
        meta=base.meta,
    )
    write_dicom_series(
        irregular,
        tmp_path,
        series_number=1,
        contrast="t2",
        study_uid=generate_uid(),
        frame_uid=generate_uid(),
    )
    geo = load_study(tmp_path)["sag_t2"].meta["geometry"]
    assert geo["uniform"] is False and geo["gap_count"] == 0
    assert geo["spacing_min_mm"] == 1.0 and geo["spacing_max_mm"] == 4.0


def test_uniform_series_has_clean_geometry(phantom_study):
    for st in phantom_study.stacks.values():
        assert st.meta["geometry"]["uniform"] is True
        assert st.meta["geometry"]["gap_count"] == 0


def test_varying_pixel_spacing_splits_into_stacks(phantom, tmp_path, caplog):
    study = generate_uid()
    series = generate_uid()
    fine = phantom.render("sag", "t2", n=4, rows=48, cols=48, px=1.0, spacing=4.0)
    coarse_base = phantom.render("sag", "t2", n=4, rows=48, cols=48, px=2.0, spacing=4.0)
    coarse = Stack(
        key="",
        vol=coarse_base.vol,
        ipp=coarse_base.ipp + 40.0 * coarse_base.normal,  # further along the stack, same series
        row_dir=coarse_base.row_dir,
        col_dir=coarse_base.col_dir,
        row_mm=coarse_base.row_mm,
        col_mm=coarse_base.col_mm,
        thickness=coarse_base.thickness,
        meta=coarse_base.meta,
    )
    for i, st in enumerate((fine, coarse)):
        write_dicom_series(
            st,
            tmp_path / f"p{i}",
            series_number=7,
            contrast="t2",
            study_uid=study,
            frame_uid=generate_uid(),
            series_uid=series,
        )
    with caplog.at_level(logging.WARNING, logger="spinescope.io.dicom"):
        st = load_study(tmp_path)
    assert sorted(st.keys()) == ["sag_t2", "sag_t2_2"]
    spacings = sorted(s.row_mm for s in st.stacks.values())
    assert spacings == [1.0, 2.0]
    assert all(s.n == 4 for s in st.stacks.values())
    assert any("varying pixel spacing" in r.getMessage() for r in caplog.records)


def test_four_d_series_is_split_by_time(phantom, tmp_path):
    study, series = generate_uid(), generate_uid()
    st = phantom.render("sag", "t2", n=4, rows=48, cols=48, px=3.0, spacing=4.0)
    flipped = Stack(
        key="",
        vol=st.vol[:, ::-1, :].copy(),  # different content at each time point
        ipp=st.ipp,
        row_dir=st.row_dir,
        col_dir=st.col_dir,
        row_mm=st.row_mm,
        col_mm=st.col_mm,
        thickness=st.thickness,
        meta=st.meta,
    )
    # write in reverse time order to prove the reader sorts by TemporalPositionIdentifier
    for t, vol_st in ((1, flipped), (0, st)):
        write_dicom_series(
            vol_st,
            tmp_path / f"t{t}",
            series_number=4,
            contrast="t2",
            study_uid=study,
            frame_uid=generate_uid(),
            series_uid=series,
            temporal_index=t,
        )
    got = load_study(tmp_path)
    assert sorted(got.keys()) == ["sag_t2", "sag_t2_2"]
    first, second = got["sag_t2"], got["sag_t2_2"]
    assert first.meta["repeat_index"] == 0 and second.meta["repeat_index"] == 1
    # time 0 is the unflipped content, time 1 the flipped one
    assert not np.allclose(first.vol, second.vol)
    assert np.allclose(second.vol, np.flip(first.vol, axis=1), atol=1.0)


def test_lazy_keys_loads_only_requested_stacks(phantom_dir):
    st = load_study(phantom_dir["root"], keys=["ax_t2"])
    assert st.keys() == ["ax_t2"]
    assert st["ax_t2"].vol.max() > 0
    with pytest.raises(KeyError):
        load_study(phantom_dir["root"], keys=["nope"])


def test_header_only_load_returns_placeholders(phantom_dir):
    st = load_study(phantom_dir["root"], load_pixels=False)
    assert set(st.keys()) >= {"sag_t2", "sag_t1", "sag_stir", "ax_t2", "loc_sag"}
    assert float(st["sag_t2"].vol.max()) == 0.0 and st["sag_t2"].vol.shape[0] == 13
    assert st["sag_t2"].meta["role"]["contrast"] == "t2"


def test_header_only_load_is_fast_and_pixel_free(phantom_dir, monkeypatch):
    import spinescope.io.dicom as dicom

    calls = []
    real = dicom._read_pixels

    def spy(ref, cache):
        calls.append(ref.path)
        return real(ref, cache)

    monkeypatch.setattr(dicom, "_read_pixels", spy)
    load_study(phantom_dir["root"], load_pixels=False)
    assert calls == []
    load_study(phantom_dir["root"], keys=["sag_t1"])
    assert calls and set(calls) <= set(phantom_dir["series"]["sag_t1"])


def test_report_and_info_are_phi_free(phantom_dir):
    st = load_study(phantom_dir["root"])
    blob = json.dumps({"info": st.info, "report": st.report}, default=str)
    for tag, value in FAKE_PHI.items():
        if tag == "PatientSex":  # a one-letter value, a substring test would match anything
            continue
        assert value not in blob, f"{tag} leaked into the report"
    assert str(phantom_dir["root"]) not in blob


def test_unsupported_pixel_error_is_not_raised_for_header_only(phantom, tmp_path, monkeypatch):
    """A missing decoder must not break header-only loading, but pixel loading must explain itself."""
    import spinescope.io.dicom as dicom

    write_dicom_series(
        phantom.render("sag", "t2", n=3, rows=32, cols=32, px=3.0),
        tmp_path,
        series_number=1,
        contrast="t2",
        study_uid=generate_uid(),
        frame_uid=generate_uid(),
        transfer_syntax="j2k",
    )
    monkeypatch.setattr(dicom, "decoder_available", lambda uid: False)
    header_only = load_study(tmp_path, load_pixels=False)
    assert header_only.keys() == ["sag_t2"]
    assert any("decoder" in w for w in header_only.report["warnings"])
    with pytest.raises(UnsupportedPixelDataError, match=r"spinescope\[jpeg\]"):
        load_study(tmp_path)
