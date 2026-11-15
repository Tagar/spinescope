"""Resumable, checksum-verified downloads with urllib (honours http(s)_proxy / no_proxy).

Downloads go to ``<name>.part`` and are renamed only after the size and checksum match. An interrupted
download resumes with an HTTP ``Range`` request. A file that already matches its checksum is not fetched
again. Downloads above ``max_bytes`` (default 2 GiB) are refused unless the caller raises the cap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .registry import DatasetSpec, FileSpec

log = logging.getLogger(__name__)

#: Default refusal threshold for a single download.
DEFAULT_MAX_BYTES = 2 * 1024**3
USER_AGENT = "spinescope-datasets/0.1 (+https://github.com/Tagar/spinescope)"
_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """A download failed (network, HTTP status or incomplete transfer)."""


class DownloadTooLarge(DownloadError):
    """The file is above the size cap; pass a larger ``max_bytes`` / ``allow_large=True`` to fetch it."""


class ChecksumError(DownloadError):
    """The downloaded file does not match the registry checksum."""


def build_opener() -> urllib.request.OpenerDirector:
    """An opener that uses the proxies from the environment (``http_proxy``, ``https_proxy``, ``no_proxy``)."""
    return urllib.request.build_opener(urllib.request.ProxyHandler())


def hash_file(path: str | Path, algo: str) -> str:
    """Hex digest of ``path`` with ``algo`` (``md5`` or ``sha256``), streamed."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def verify(
    path: str | Path, *, size: int | None = None, md5: str | None = None, sha256: str | None = None
) -> bool:
    """True if ``path`` exists and matches every given attribute (size, md5, sha256)."""
    p = Path(path)
    if not p.is_file():
        return False
    checks: list[bool] = []
    if size is not None:
        checks.append(p.stat().st_size == size)
    if md5 is not None:
        checks.append(hash_file(p, "md5") == md5.lower())
    if sha256 is not None:
        checks.append(hash_file(p, "sha256") == sha256.lower())
    return all(checks)


def download_file(
    url: str,
    dest: str | Path,
    *,
    size: int | None = None,
    md5: str | None = None,
    sha256: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = 60.0,
    opener: urllib.request.OpenerDirector | None = None,
) -> Path:
    """Download ``url`` to ``dest`` (resuming a ``.part`` file), verify it, and return ``dest``."""
    dest = Path(dest).expanduser()
    if size is not None and size > max_bytes:
        raise DownloadTooLarge(
            f"{dest.name} is {size} bytes, above the {max_bytes}-byte cap; pass max_bytes=... to fetch it"
        )
    if verify(dest, size=size, md5=md5, sha256=sha256):
        log.info("already present and verified: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have > size:
        part.unlink()
        have = 0
    opener = opener or build_opener()
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and size is not None and have == size:  # the .part file is already complete
            resp = None
        else:
            raise DownloadError(f"HTTP {e.code} for {dest.name}") from e
    except urllib.error.URLError as e:
        raise DownloadError(f"cannot reach the download host for {dest.name}: {e.reason}") from e
    if resp is not None:
        with resp:
            status = getattr(resp, "status", 200)
            resumed = bool(have) and status == 206
            if have and not resumed:  # server ignored Range: start over
                have = 0
            length = resp.headers.get("Content-Length")
            total = (int(length) + have) if length else size
            if total is not None and total > max_bytes:
                raise DownloadTooLarge(f"{dest.name} is {total} bytes, above the {max_bytes}-byte cap")
            log.info("downloading %s (%s bytes)", dest.name, total or "unknown")
            with open(part, "ab" if resumed else "wb") as out:
                for block in iter(lambda: resp.read(_CHUNK), b""):
                    out.write(block)
    if size is not None and part.stat().st_size != size:
        raise DownloadError(
            f"{dest.name} is incomplete ({part.stat().st_size} of {size} bytes); re-run to resume"
        )
    if not verify(part, md5=md5, sha256=sha256):
        part.unlink(missing_ok=True)
        raise ChecksumError(f"{dest.name} failed its checksum; the partial file was removed")
    part.replace(dest)
    return dest


def resolve_mendeley_url(file_spec: FileSpec, dataset_id: str) -> str:
    """Download URL of ``file_spec`` from the Mendeley Data files API (``content_details.download_url``)."""
    opener = build_opener()
    api = f"https://data.mendeley.com/api/datasets/{dataset_id}/files"
    with opener.open(urllib.request.Request(api, headers={"User-Agent": USER_AGENT}), timeout=60) as r:
        listing: list[dict[str, Any]] = json.loads(r.read().decode("utf-8"))
    for entry in listing:
        if entry.get("filename") == file_spec.name:
            return str(entry["content_details"]["download_url"])
    raise DownloadError(f"{file_spec.name} is not listed by the Mendeley files API for {dataset_id}")


def fetch_dataset(
    spec: DatasetSpec,
    root: str | Path | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allow_large: bool = False,
    extract: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download every file of ``spec`` into ``<root>/<subdir>/raw`` and unpack the zips into the dataset folder.

    Files above ``max_bytes`` are refused unless ``allow_large=True``. Returns the dataset folder.
    """
    from .extract import safe_extract_zip

    base = spec.path(root)
    raw = base / "raw"
    cap = sys.maxsize if allow_large else max_bytes
    for f in spec.files:
        if f.size > cap:
            raise DownloadTooLarge(
                f"{f.name} ({f.size / 1e9:.2f} GB) is above the {cap / 1e9:.2f} GB cap; "
                "pass allow_large=True to fetch it"
            )
    for f in spec.files:
        url = f.url
        if f.resolver == "mendeley-api":
            url = resolve_mendeley_url(f, spec.extra["mendeley_id"])
        if progress:
            progress(f"{spec.key}: {f.name}")
        path = download_file(url, raw / f.name, size=f.size, md5=f.md5, sha256=f.sha256, max_bytes=cap)
        if not extract:
            continue
        if path.suffix.lower() == ".zip":
            safe_extract_zip(path, base)
        else:
            shutil.copyfile(path, base / f.name)
    return base
