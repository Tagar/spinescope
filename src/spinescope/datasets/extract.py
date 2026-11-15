"""Safe zip extraction: every member is checked before anything is written."""

from __future__ import annotations

import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class UnsafeArchiveError(ValueError):
    """The archive contains a member that would be written outside the destination (or a symlink)."""


def _member_target(dest: Path, name: str) -> Path:
    norm = name.replace("\\", "/")
    if norm.startswith("/") or _DRIVE_RE.match(norm):
        raise UnsafeArchiveError(f"absolute path in archive: {name!r}")
    parts = PurePosixPath(norm).parts
    if any(p == ".." for p in parts):
        raise UnsafeArchiveError(f"path traversal in archive: {name!r}")
    target = (dest / Path(*[p for p in parts if p not in ("", ".")])).resolve()
    if target != dest and dest not in target.parents:
        raise UnsafeArchiveError(f"member escapes the destination: {name!r}")
    return target


def safe_extract_zip(
    zip_path: str | Path,
    dest: str | Path,
    *,
    max_total_bytes: int | None = None,
) -> list[Path]:
    """Extract ``zip_path`` into ``dest``, rejecting path traversal, absolute paths, symlinks and
    (optionally) archives that expand beyond ``max_total_bytes``. Returns the extracted file paths.

    All members are validated before the first byte is written, so a rejected archive leaves no files.
    """
    dest_path = Path(dest).expanduser().resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        total = 0
        plan: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in infos:
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise UnsafeArchiveError(f"symbolic link in archive: {info.filename!r}")
            total += info.file_size
            plan.append((info, _member_target(dest_path, info.filename)))
        if max_total_bytes is not None and total > max_total_bytes:
            raise UnsafeArchiveError(f"archive expands to {total} bytes, above the {max_total_bytes} limit")
        written: list[Path] = []
        for info, target in plan:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            written.append(target)
    return written
