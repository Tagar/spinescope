"""SPIDER loader on a synthetic tree with the same layout (phantom volumes), plus a dataset-marked test."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from spinescope import datasets
from spinescope.datasets import load_spider_studies, load_spider_study, spider_root
from spinescope.io.volumes import read_label_volume

sitk = pytest.importorskip("SimpleITK")


@pytest.fixture
def spider_tree(phantom, tmp_path):
    """Images and masks as .mha with the SPIDER naming; the mask labels mirror the vertebra/disc/canal scheme."""
    root = tmp_path / "spider"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    st = phantom.render("sag", "t2", n=7, rows=40, cols=36, px=1.2, spacing=4.0)
    st1 = phantom.render("sag", "t1", n=7, rows=40, cols=36, px=1.2, spacing=4.0)
    for case_id, seq, stack in (("1", "t2", st), ("1", "t1", st1), ("2", "t2", st)):
        vol = np.asarray(stack.vol, dtype=np.int16)  # [k, r, c]
        mask = np.zeros_like(vol)
        mask[:, 10:30, 12:20] = 1  # a fake vertebra
        mask[2:4, 30:34, 4:9] = 201  # a fake disc
        mask[:, 20:24, 22:26] = 100  # a fake canal
        for folder, arr in (("images", vol), ("masks", mask)):
            # SPIDER volumes are sagittal: slices run along x (largest spacing). SimpleITK arrays are [z, y, x].
            img = sitk.GetImageFromArray(np.transpose(arr, (2, 1, 0)))
            img.SetSpacing((stack.spacing, stack.row_mm, stack.col_mm))
            sitk.WriteImage(img, os.fspath(root / folder / f"{case_id}_{seq}.mha"), False)
    with open(root / "overview.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "new_file_name",
                "num_vertebrae",
                "num_discs",
                "sex",
                "birth_date",
                "subset",
                "DeviceSerialNumber",
            ]
        )
        w.writerow(["1_t2", 5, 4, "F", "19700101", "training", "SECRET-DEVICE-1234"])
        w.writerow(["2_t2", 6, 5, "M", "19650202", "validation", "SECRET-DEVICE-5678"])
    with open(root / "radiological_gradings.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "Patient",
            "IVD label",
            "Modic",
            "UP endplate",
            "LOW endplate",
            "Spondylolisthesis",
            "Disc herniation",
            "Disc narrowing",
            "Disc bulging",
            "Pfirrman grade",
        ]
        w.writerow(header)
        w.writerow([1, 1, 0, 0, 0, 0, 0, 1, 1, 3])
        w.writerow([1, 2, 1, 0, 0, 0, 1, 0, 0, 4])
        w.writerow([2, 1, 0, 0, 0, 0, 0, 0, 0, 2])
    return root


def test_spider_root_default_and_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPINESCOPE_DATA", str(tmp_path))
    assert spider_root() == tmp_path / "spider"
    assert spider_root(tmp_path / "x") == tmp_path / "x"


def test_load_studies_paths_and_gradings_without_loading(spider_tree):
    studies = load_spider_studies(spider_tree)
    assert [s.case_id for s in studies] == ["1", "2"]
    s1 = studies[0]
    assert s1.sequences == ["t1", "t2"] and s1.subset == "training"
    assert (s1.num_vertebrae, s1.num_discs) == (5, 4)
    assert s1.volumes == {} and s1.masks == {}  # lazy by default
    assert [g["pfirrman_grade"] for g in s1.gradings] == [3, 4]
    assert s1.gradings[1]["disc_herniation"] == 1


def test_study_volume_and_mask_are_on_one_grid(spider_tree):
    study = load_spider_study("1", spider_tree)
    vol = study.volume("t2")
    mask = study.mask("t2")
    assert mask.shape == vol.shape
    assert set(np.unique(mask)) == {0, 1, 100, 201}
    assert mask.dtype.kind == "i"
    assert vol.key == "" and vol.plane == "sag"
    assert study.volume("t2") is vol  # cached
    with pytest.raises(KeyError):
        study.volume("t2_SPACE")


def test_mask_reader_matches_image_orientation(spider_tree):
    """The mask gets the same display transform as its image, so voxels line up with the volume."""
    from spinescope.series import apply_display_ops

    study = load_spider_study("1", spider_tree)
    raw = read_label_volume(os.fspath(spider_tree / "masks" / "1_t2.mha"))
    ops = study.volume("t2").meta.get("display_transform", [])
    assert np.array_equal(apply_display_ops(raw, ops), study.mask("t2"))
    assert study.mask("t2").shape == study.volume("t2").shape


def test_privacy_no_identifying_columns_leak(spider_tree):
    studies = load_spider_studies(spider_tree, load=True)
    blob = json.dumps(
        [
            {
                "case": s.case_id,
                "subset": s.subset,
                "gradings": s.gradings,
                "n": s.num_vertebrae,
            }
            for s in studies
        ]
    )
    for secret in ("SECRET-DEVICE", "19700101", "19650202"):
        assert secret not in blob
    for st in studies:
        for stack in st.volumes.values():
            assert "SECRET" not in json.dumps(stack.meta, default=str)


def test_cases_filter_and_missing_folder(spider_tree, tmp_path):
    assert [s.case_id for s in load_spider_studies(spider_tree, cases=["2"])] == ["2"]
    with pytest.raises(FileNotFoundError, match="fetch"):
        load_spider_studies(tmp_path / "nothing")
    with pytest.raises(KeyError):
        load_spider_study("99", spider_tree)


def test_fetch_refuses_big_spider_without_allow_large(tmp_path):
    with pytest.raises(datasets.DownloadTooLarge):
        datasets.fetch("spider", tmp_path)


SPIDER_ON_DISK = os.path.isdir(os.path.join(datasets.data_dir(), "spider", "images"))


@pytest.mark.dataset
@pytest.mark.skipif(not SPIDER_ON_DISK, reason="SPIDER not on disk (set SPINESCOPE_DATA)")
def test_real_spider_case_loads_with_aligned_masks():
    studies = load_spider_studies(load=False)
    assert len(studies) == 218
    assert sum(len(s.image_paths) for s in studies) == 447
    study = load_spider_study(studies[0].case_id, load=False)
    seq = "t2" if "t2" in study.sequences else study.sequences[0]
    mask = study.mask(seq)
    assert mask.shape == study.volume(seq).shape
    assert (mask == 100).any()  # spinal canal label present
