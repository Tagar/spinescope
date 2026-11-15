"""Dataset registry: licences, citations, checksums, and the size gate (no network access needed)."""

from __future__ import annotations

import pytest

from spinescope import datasets
from spinescope.datasets import REGISTRY, DownloadTooLarge, data_dir, get_spec


def test_registry_entries_have_licence_citation_and_checksums():
    assert {"spider", "lumbar_spine_mri_sudirman", "lumbar_vcf_mri_hcfmrp"} <= set(REGISTRY)
    for key, spec in REGISTRY.items():
        assert spec.licence.startswith("CC-BY"), key
        assert spec.licence_url.startswith("https://creativecommons.org/"), key
        assert spec.doi and spec.citation and spec.source_url, key
        assert spec.files, key
        for f in spec.files:
            assert f.size > 0, (key, f.name)
            assert f.md5 or f.sha256, (key, f.name)
            assert f.url.startswith("https://"), (key, f.name)


def test_spider_registry_matches_zenodo_record():
    spider = get_spec("spider")
    assert spider.doi == "10.5281/zenodo.10159290"
    assert spider.licence == "CC-BY-4.0"
    sizes = {f.name: f.size for f in spider.files}
    assert sizes["images.zip"] == 3700562886
    assert {f.name: f.md5 for f in spider.files}["masks.zip"] == "49861ff25a0108b3b69d0ce808ae51dc"


def test_mendeley_dataset_is_refused_by_default_without_network():
    spec = get_spec("lumbar_spine_mri_sudirman")
    assert spec.doi == "10.17632/k57fr854j2.2" and spec.licence == "CC-BY-4.0"
    assert spec.total_size > 2 * 1024**3  # above the default cap
    with pytest.raises(DownloadTooLarge):
        datasets.fetch("lumbar_spine_mri_sudirman", root="/nonexistent-root-for-test")


def test_unknown_key_lists_known_keys():
    with pytest.raises(KeyError, match="spider"):
        get_spec("nope")


def test_data_dir_honours_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SPINESCOPE_DATA", str(tmp_path / "cache"))
    assert data_dir() == tmp_path / "cache"
    monkeypatch.delenv("SPINESCOPE_DATA")
    assert str(data_dir()).endswith("data/public")
