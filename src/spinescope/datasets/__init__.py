"""Public spine MRI datasets: a registry with licences and checksums, resumable downloads, and loaders.

Example::

    from spinescope import datasets
    datasets.REGISTRY["spider"].licence              # 'CC-BY-4.0'
    datasets.fetch("spider", allow_large=True)       # images.zip is 3.7 GB
    for study in datasets.load_spider_studies(load=True):
        study.volume("t2"), study.mask("t2"), study.gradings

Local cache: ``$SPINESCOPE_DATA`` (default ``~/data/public``); each dataset lives in its own folder.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .download import (
    DEFAULT_MAX_BYTES,
    ChecksumError,
    DownloadError,
    DownloadTooLarge,
    download_file,
    fetch_dataset,
)
from .extract import UnsafeArchiveError, safe_extract_zip
from .gradings import DiscGrading, load_spider_gradings, spider_subset_cases
from .registry import REGISTRY, DatasetSpec, FileSpec, data_dir, get_spec
from .report_labels import load_sudirman_report_labels, parse_report
from .spider import SpiderStudy, load_spider_studies, load_spider_study, spider_root

__all__ = [
    "DEFAULT_MAX_BYTES",
    "REGISTRY",
    "ChecksumError",
    "DatasetSpec",
    "DiscGrading",
    "DownloadError",
    "DownloadTooLarge",
    "FileSpec",
    "SpiderStudy",
    "UnsafeArchiveError",
    "data_dir",
    "download_file",
    "fetch",
    "fetch_dataset",
    "get_spec",
    "load_spider_gradings",
    "load_spider_studies",
    "load_spider_study",
    "load_sudirman_report_labels",
    "parse_report",
    "safe_extract_zip",
    "spider_root",
    "spider_subset_cases",
]


def fetch(
    key: str,
    root: str | Path | None = None,
    *,
    allow_large: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
    extract: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download and unpack the registered dataset ``key`` into ``root`` (default ``$SPINESCOPE_DATA``)."""
    return fetch_dataset(
        get_spec(key), root, max_bytes=max_bytes, allow_large=allow_large, extract=extract, progress=progress
    )
