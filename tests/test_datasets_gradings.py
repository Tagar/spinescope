"""SPIDER gradings loader: level naming, transitional flag, subsets, vendor normalisation (synthetic CSVs)."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from spinescope.datasets import gradings as gr

OVERVIEW = """new_file_name,num_vertebrae,subset,Manufacturer,MagneticFieldStrength,sex,birth_date
1_t1,6,training,SIEMENS,1.5,F,1950-01-01
1_t2,6,training,SIEMENS,1.5,F,1950-01-01
2_t2,5,validation,Philips Healthcare,3.0,M,1960-02-02
3_t2,7,validation,Philips Medical Systems,3.0,,
"""

GRADINGS = """Patient,IVD label,Modic,UP endplate,LOW endplate,Spondylolisthesis,Disc herniation,Disc narrowing,Disc bulging,Pfirrman grade
1,2,0,1,0,0,0,1,1,3
1,1,2,0,1,0,1,1,1,4
1,0,0,0,0,0,0,0,0,0
2,1,0,0,0,0,0,1,0,2
3,1,3,1,1,1,0,0,1,5
"""


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "overview.csv").write_text(OVERVIEW)
    (tmp_path / "radiological_gradings.csv").write_text(GRADINGS)
    return tmp_path


def test_ivd_level_names_follow_the_spider_convention() -> None:
    assert gr.ivd_level_name(1) == "L5-S1"
    assert gr.ivd_level_name(2) == "L4-L5"
    assert gr.ivd_level_name(3) == "L3-L4"
    assert gr.ivd_level_name(6) == "T12-L1"
    assert gr.ivd_level_name(0) is None
    assert gr.ivd_level_name(99) is None


def test_loader_maps_levels_values_and_transitional_flag(root: Path) -> None:
    g = gr.load_spider_gradings(root)
    assert list(g) == ["1", "2", "3"]  # numeric order
    p1 = g["1"]
    assert [d.ivd_label for d in p1] == [1, 2]  # IVD 0 skipped, sorted bottom-up
    assert p1[0].level == "L5-S1" and p1[1].level == "L4-L5"
    assert p1[0].modic == 2 and p1[0].pfirrmann == 4 and p1[0].disc_herniation == 1
    assert p1[0].value("up_endplate") == 0 and p1[0].value("low_endplate") == 1
    assert p1[0].transitional_suspect is True  # 6 vertebrae, not 5
    assert g["2"][0].transitional_suspect is False  # 5 vertebrae
    assert g["3"][0].value("modic") == 3


def test_value_rejects_unknown_field(root: Path) -> None:
    d = gr.load_spider_gradings(root)["1"][0]
    with pytest.raises(KeyError):
        d.value("not_a_field")


def test_file_meta_has_no_identifying_fields(root: Path) -> None:
    meta = gr.load_spider_file_meta(root)
    assert meta[("1", "t1")].vendor == "Siemens"
    assert meta[("3", "t2")].vendor == "Philips"  # "Philips Medical Systems" normalised
    assert meta[("2", "t2")].field_strength == "3.0"
    fields = {f.name for f in dataclasses.fields(gr.FileMeta)}
    assert not fields & {"sex", "birth_date", "device"}


def test_subsets_are_listed_per_case(root: Path) -> None:
    assert gr.spider_subset_cases("training", root) == ["1"]
    assert gr.spider_subset_cases("validation", root) == ["2", "3"]


def test_summary_counts(root: Path) -> None:
    s = gr.grading_summary(gr.load_spider_gradings(root))
    assert s["cases"] == 3 and s["discs"] == 4  # IVD 0 is skipped
    assert s["transitional_suspect_discs"] == 3  # cases 1 (2 discs) and 3 (1 disc)
    assert s["values"]["modic"]["0"] == 2  # two discs with Modic 0 (the IVD 0 row is skipped)


def test_vendor_of() -> None:
    assert gr.vendor_of("SIEMENS") == "Siemens"
    assert gr.vendor_of("Philips Healthcare") == "Philips"
    assert gr.vendor_of("GE") == "other"


def _spider_available() -> bool:
    base = Path(os.environ.get("SPINESCOPE_DATA", "~/data/public")).expanduser() / "spider"
    return (base / "radiological_gradings.csv").is_file()


@pytest.mark.dataset
@pytest.mark.skipif(not _spider_available(), reason="SPIDER gradings not on disk")
def test_real_spider_gradings_shape() -> None:
    g = gr.load_spider_gradings()
    s = gr.grading_summary(g)
    assert s["cases"] == 218
    assert s["discs"] > 1400
    assert set(s["values"]["modic"]) <= {"0", "1", "2", "3", "missing"}
