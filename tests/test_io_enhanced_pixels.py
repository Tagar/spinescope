"""Enhanced multi-frame MR objects, compressed transfer syntaxes and pixel encodings."""

from __future__ import annotations

import numpy as np
import pydicom
import pytest
from pydicom.uid import generate_uid

from spinescope import load_study
from spinescope.testing.phantom import write_dicom_series, write_enhanced_mr


@pytest.fixture(scope="module")
def views(phantom):
    sag = phantom.render("sag", "t2", n=7, rows=64, cols=64, px=3.0, spacing=4.0)
    ax = phantom.render("ax", "t2", n=4, rows=64, cols=64, px=3.0, spacing=4.0)
    cor = phantom.render("cor", "t2", n=5, rows=64, cols=64, px=3.0, spacing=4.0)
    return sag, ax, cor


def test_enhanced_object_with_three_orientation_groups(views, tmp_path):
    sag, ax, cor = views
    write_enhanced_mr([sag, ax, cor], tmp_path / "enh.dcm", study_uid=generate_uid())
    st = load_study(tmp_path)
    assert sorted(st.keys()) == ["ax_t2", "cor_t2", "sag_t2"]
    assert (st["sag_t2"].n, st["ax_t2"].n, st["cor_t2"].n) == (7, 4, 5)
    # geometry of every frame is exact
    assert np.allclose(np.sort(st["sag_t2"].ipp, axis=0), np.sort(sag.ipp, axis=0), atol=1e-3)
    assert np.allclose(st["ax_t2"].row_dir, [1, 0, 0]) and np.allclose(st["ax_t2"].col_dir, [0, 1, 0])


def test_enhanced_pixels_and_timing_come_from_functional_groups(views, tmp_path):
    sag, _ax, _cor = views
    write_enhanced_mr([sag], tmp_path / "enh.dcm", study_uid=generate_uid())
    got = load_study(tmp_path)["sag_t2"]
    # TR/TE/ScanningSequence are only inside functional groups: classification must still see them
    assert got.meta["role"]["contrast"] == "t2"
    assert any("TR=4000" in e for e in got.meta["role"]["evidence"])
    # stored values are a monotone function of the phantom intensities (same matrix, one scale)
    corr = np.corrcoef(got.vol.ravel(), sag.vol.ravel())[0, 1]
    assert corr > 0.999
    assert got.vol.dtype == np.float32 and got.n == sag.n


def test_enhanced_and_classic_series_agree(views, tmp_path):
    sag, _ax, _cor = views
    classic_dir, enh_dir = tmp_path / "classic", tmp_path / "enh"
    study = generate_uid()
    write_dicom_series(
        sag, classic_dir, series_number=1, contrast="t2", study_uid=study, frame_uid=generate_uid()
    )
    enh_dir.mkdir()
    write_enhanced_mr([sag], enh_dir / "e.dcm", study_uid=study)
    a = load_study(classic_dir)["sag_t2"]
    b = load_study(enh_dir)["sag_t2"]
    assert np.allclose(a.ipp, b.ipp, atol=1e-3)
    assert np.allclose(a.vol, b.vol, atol=1.0)


@pytest.mark.parametrize("transfer_syntax", ["rle", "j2k"])
def test_compressed_series_load_identically(views, tmp_path, transfer_syntax):
    sag, _ax, _cor = views
    study, frame = generate_uid(), generate_uid()
    write_dicom_series(
        sag, tmp_path / "plain", series_number=1, contrast="t2", study_uid=study, frame_uid=frame
    )
    write_dicom_series(
        sag,
        tmp_path / "packed",
        series_number=2,
        contrast="t2",
        study_uid=study,
        frame_uid=frame,
        transfer_syntax=transfer_syntax,
    )
    f = next((tmp_path / "packed").iterdir())
    assert pydicom.dcmread(f, stop_before_pixels=True).file_meta.TransferSyntaxUID.is_compressed
    ref = load_study(tmp_path / "plain")["sag_t2"]
    got = load_study(tmp_path / "packed")["sag_t2"]
    assert got.n == ref.n
    assert np.array_equal(got.vol, ref.vol)


def test_compressed_header_only_does_not_decode(views, tmp_path, monkeypatch):
    import spinescope.io.dicom as dicom

    sag, _ax, _cor = views
    write_dicom_series(
        sag,
        tmp_path,
        series_number=1,
        contrast="t2",
        study_uid=generate_uid(),
        frame_uid=generate_uid(),
        transfer_syntax="rle",
    )

    def boom(*_a, **_k):
        raise AssertionError("pixels decoded")

    monkeypatch.setattr(dicom, "_read_pixels", boom)
    st = load_study(tmp_path, load_pixels=False)
    assert st["sag_t2"].n == sag.n


def test_signed_pixels_are_read_as_signed(views, tmp_path):
    sag, _ax, _cor = views
    study, frame = generate_uid(), generate_uid()
    write_dicom_series(sag, tmp_path / "u", series_number=1, contrast="t2", study_uid=study, frame_uid=frame)
    write_dicom_series(
        sag, tmp_path / "s", series_number=2, contrast="t2", study_uid=study, frame_uid=frame, signed=True
    )
    unsigned = load_study(tmp_path / "u")["sag_t2"].vol
    signed = load_study(tmp_path / "s")["sag_t2"].vol
    assert signed.min() < 0
    assert np.array_equal(signed, unsigned - 2048)


def test_rescale_slope_and_intercept_are_applied(views, tmp_path):
    sag, _ax, _cor = views
    study, frame = generate_uid(), generate_uid()
    write_dicom_series(sag, tmp_path / "u", series_number=1, contrast="t2", study_uid=study, frame_uid=frame)
    write_dicom_series(
        sag,
        tmp_path / "r",
        series_number=2,
        contrast="t2",
        study_uid=study,
        frame_uid=frame,
        rescale=(2.0, -100.0),
    )
    plain = load_study(tmp_path / "u")["sag_t2"].vol
    rescaled = load_study(tmp_path / "r")["sag_t2"].vol
    # stored integers s map to 2*s - 100: values are on that lattice, and equal the plain image up to one step
    assert np.all(np.mod(rescaled + 100.0, 2.0) == 0)
    assert np.max(np.abs(rescaled - plain)) <= 2.0


def test_monochrome1_is_inverted_on_load(views, tmp_path):
    sag, _ax, _cor = views
    study, frame = generate_uid(), generate_uid()
    write_dicom_series(sag, tmp_path / "m2", series_number=1, contrast="t2", study_uid=study, frame_uid=frame)
    write_dicom_series(
        sag,
        tmp_path / "m1",
        series_number=2,
        contrast="t2",
        study_uid=study,
        frame_uid=frame,
        monochrome1=True,
    )
    raw = pydicom.dcmread(next((tmp_path / "m1").iterdir()))
    assert raw.PhotometricInterpretation == "MONOCHROME1"
    a = load_study(tmp_path / "m2")["sag_t2"].vol
    b = load_study(tmp_path / "m1")["sag_t2"].vol
    assert np.array_equal(a, b)


def test_colour_objects_never_become_stacks(views, tmp_path):
    from spinescope.testing.phantom import write_non_image_object

    tmp_path.mkdir(exist_ok=True)
    sag, _ax, _cor = views
    write_dicom_series(
        sag, tmp_path, series_number=1, contrast="t2", study_uid=generate_uid(), frame_uid=generate_uid()
    )
    write_non_image_object(tmp_path / "shot.dcm", "SC", study_uid=generate_uid())
    st = load_study(tmp_path)
    assert st.keys() == ["sag_t2"]
    assert all(s.vol.max() < 4096 for s in st.stacks.values())
