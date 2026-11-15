from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from spinescope import load_study
from spinescope.testing.phantom import FAKE_PHI, write_dicom_series


def test_standard_study_keys(phantom_study):
    keys = set(phantom_study.keys())
    assert {"sag_t2", "sag_t1", "sag_stir", "ax_t2", "ax_t1", "loc_sag", "loc_cor", "loc_ax"} <= keys
    slabs = phantom_study.slabs("ax")
    assert [s.key for s in slabs] == [f"ax_t2_slab{i}" for i in range(1, 6)]
    assert phantom_study.best("sag").key == "sag_t2"


def test_geometry_matches_phantom_truth(phantom_study, phantom):
    ax = phantom_study["ax_t2"]
    for disc in ("L2-L3", "L4-L5"):
        c = phantom.discs[disc]["center"]
        _k, dist = ax.nearest_slice(c)
        assert dist <= ax.spacing / 2 + 1e-6
    # each oblique slab passes through its disc
    for st, disc in zip(phantom_study.slabs(), ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"), strict=False):
        _, dist = st.nearest_slice(phantom.discs[disc]["center"])
        assert dist < 1.0


def test_display_orientation_conventions(phantom_study):
    sag = phantom_study["sag_t2"]
    assert np.allclose(sag.row_dir, [0, 1, 0]) and np.allclose(sag.col_dir, [0, 0, -1])
    assert sag.pix_to_lps(0, 0, 0)[0] > sag.pix_to_lps(sag.n - 1, 0, 0)[0]  # k=0 is patient left
    ax = phantom_study["ax_t2"]
    assert np.allclose(ax.row_dir, [1, 0, 0]) and np.allclose(ax.col_dir, [0, 1, 0])


def _all_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("_"):
                continue
            yield str(k)
            yield from _all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _all_strings(v)
    else:
        yield str(obj)


def test_no_phi_in_metadata(phantom_study):
    blob = json.dumps(
        list(
            _all_strings({"info": phantom_study.info, **{k: s.meta for k, s in phantom_study.stacks.items()}})
        )
    )
    for tag, value in FAKE_PHI.items():
        if tag == "PatientSex":
            continue
        assert value not in blob, f"{tag} leaked"


def test_odd_layout_and_junk_files(phantom_dir, tmp_path):
    root = tmp_path / "weird layout" / "deeper"
    src = phantom_dir["series"]["sag_t2"]
    for i, p in enumerate(src):
        d = root / f"part{i % 3}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, d / f"img{i}")  # no extension
    (root / "DICOMDIR").write_bytes(b"not really")
    (root / "notes.txt").write_text("hello")
    (root / ".DS_Store").write_bytes(b"\0\1")
    st = load_study(root)
    assert st.keys() == ["sag_t2"] and st["sag_t2"].n == len(src)


def test_flipped_acquisition_is_reoriented(phantom, tmp_path):
    from pydicom.uid import generate_uid

    raw = phantom.render("sag", "t2", n=5, rows=64, cols=64, px=3.0, row_dir=(0, -1, 0))
    write_dicom_series(
        raw,
        tmp_path / "s",
        series_number=3,
        contrast="t2",
        study_uid=generate_uid(),
        frame_uid=generate_uid(),
    )
    st = load_study(tmp_path)
    sag = st["sag_t2"]
    assert np.allclose(sag.row_dir, [0, 1, 0])
    # the writer rescales intensities, so compare voxel-by-voxel up to one global factor
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(200):
        k0, r0, c0 = rng.integers(0, 5), rng.integers(0, 64), rng.integers(0, 64)
        if raw.vol[k0, r0, c0] < 200:
            continue
        k, r, c = (round(v) for v in sag.lps_to_pix(raw.pix_to_lps(k0, r0, c0)))
        ratios.append(sag.vol[k, r, c] / raw.vol[k0, r0, c0])
    ratios = np.array(ratios)
    assert len(ratios) > 20
    assert np.std(ratios) / np.mean(ratios) < 0.01


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_study(tmp_path)


def test_volume_reader_matches_simpleitk(phantom, tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from spinescope.io import volumes

    st = phantom.render("sag", "t2", n=6, rows=50, cols=40, px=1.5, spacing=4.0)
    arr = st.vol  # [k, r, c]; write as an ITK image with x=c, y=r, z=k
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((st.col_mm, st.row_mm, st.spacing))
    img.SetOrigin(tuple(st.ipp[0]))
    img.SetDirection(tuple(np.column_stack([st.row_dir, st.col_dir, st.normal]).ravel()))
    path = os.fspath(tmp_path / "case_t2.mha")
    sitk.WriteImage(img, path, useCompression=True)
    a = volumes.read_volume(path)
    _arr2, _sp2, _org2, dir2 = volumes._read_metaimage(path)
    assert np.allclose(dir2, np.column_stack([st.row_dir, st.col_dir, st.normal]), atol=1e-6)
    for got in (a,):
        xyz = st.pix_to_lps(3, 20, 15)
        k, r, c = (round(v) for v in got.lps_to_pix(xyz))
        assert got.vol[k, r, c] == pytest.approx(st.vol[3, 20, 15], rel=1e-5)
    study = load_study(path)
    assert study.keys() == ["sag_t2"]
