"""Downloads: resume with Range, checksum verification, proxy environment variables, extraction."""

from __future__ import annotations

import hashlib
import http.server
import threading
import urllib.request
from pathlib import Path
from typing import ClassVar

import pytest

from spinescope.datasets import (
    ChecksumError,
    DownloadError,
    DownloadTooLarge,
    download_file,
    safe_extract_zip,
)
from spinescope.datasets.download import build_opener

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB, deterministic


class _Server(http.server.BaseHTTPRequestHandler):
    """Serves PAYLOAD with Range support; records the Range headers it received."""

    seen: ClassVar[list[str | None]] = []
    body: ClassVar[bytes] = PAYLOAD

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def do_GET(self) -> None:
        rng = self.headers.get("Range")
        self.seen.append(rng)
        start = 0
        status = 200
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            status = 206
        data = self.body[start:]
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(self.body) - 1}/{len(self.body)}")
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    handler = type("H", (_Server,), {"seen": [], "body": PAYLOAD})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/payload.bin", handler
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def no_env_proxy(monkeypatch):
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(k, raising=False)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_download_verifies_checksum(server, tmp_path, no_env_proxy):
    url, _h = server
    dest = download_file(url, tmp_path / "raw" / "p.bin", size=len(PAYLOAD), sha256=_sha(PAYLOAD))
    assert dest.read_bytes() == PAYLOAD
    assert not (tmp_path / "raw" / "p.bin.part").exists()
    # a second call finds the verified file and does not touch the network
    before = len(_h.seen)
    download_file(url, dest, size=len(PAYLOAD), sha256=_sha(PAYLOAD))
    assert len(_h.seen) == before


def test_checksum_mismatch_removes_partial_file(server, tmp_path, no_env_proxy):
    url, _h = server
    with pytest.raises(ChecksumError):
        download_file(url, tmp_path / "p.bin", size=len(PAYLOAD), sha256="0" * 64)
    assert not (tmp_path / "p.bin").exists() and not (tmp_path / "p.bin.part").exists()


def test_interrupted_download_resumes_with_range(server, tmp_path, no_env_proxy):
    url, handler = server
    dest = tmp_path / "p.bin"
    (tmp_path / "p.bin.part").write_bytes(PAYLOAD[:300_000])  # what an interrupted run left behind
    download_file(url, dest, size=len(PAYLOAD), sha256=_sha(PAYLOAD))
    assert handler.seen[-1] == "bytes=300000-"
    assert dest.read_bytes() == PAYLOAD


def test_download_refuses_files_above_cap(tmp_path):
    with pytest.raises(DownloadTooLarge):
        download_file("http://127.0.0.1:9/never", tmp_path / "big.zip", size=3 * 1024**3)


def test_unreachable_host_is_a_download_error(tmp_path, no_env_proxy):
    with pytest.raises(DownloadError):
        download_file("http://127.0.0.1:9/x", tmp_path / "x.bin", size=10, timeout=2)


def test_environment_proxy_is_used(monkeypatch, tmp_path, no_env_proxy):
    """Requests go through the proxy named in http_proxy (absolute-URI request line, as proxies receive)."""
    seen: list[str] = []

    class Proxy(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            seen.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)

    proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_address[1]}")
        proxy_handler = next(h for h in build_opener().handlers if isinstance(h, urllib.request.ProxyHandler))
        assert proxy_handler.proxies["http"] == f"http://127.0.0.1:{proxy.server_address[1]}"
        dest = download_file("http://spinescope-test.invalid/file.bin", tmp_path / "f.bin", size=len(PAYLOAD))
        assert dest.read_bytes() == PAYLOAD
        assert seen == ["http://spinescope-test.invalid/file.bin"]
    finally:
        proxy.shutdown()
        proxy.server_close()


def _zip_with(path: Path, members: dict[str, bytes], symlink: str | None = None) -> Path:
    import stat
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        if symlink is not None:
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, symlink)
    return path


def test_safe_extract_accepts_normal_archive(tmp_path):
    z = _zip_with(tmp_path / "ok.zip", {"images/1_t2.mha": b"abc", "overview.csv": b"x,y\n"})
    out = safe_extract_zip(z, tmp_path / "dst")
    assert sorted(p.name for p in out) == ["1_t2.mha", "overview.csv"]
    assert (tmp_path / "dst" / "images" / "1_t2.mha").read_bytes() == b"abc"


@pytest.mark.parametrize("bad", ["../evil.txt", "a/../../evil.txt", "/etc/evil.txt", "C:/evil.txt"])
def test_safe_extract_rejects_traversal(tmp_path, bad):
    from spinescope.datasets import UnsafeArchiveError

    z = _zip_with(tmp_path / "bad.zip", {"ok.txt": b"1", bad: b"2"})
    dest = tmp_path / "dst"
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(z, dest)
    assert not (dest / "ok.txt").exists(), "nothing may be written from a rejected archive"
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_rejects_symlinks(tmp_path):
    from spinescope.datasets import UnsafeArchiveError

    z = _zip_with(tmp_path / "link.zip", {"ok.txt": b"1"}, symlink="/etc/passwd")
    with pytest.raises(UnsafeArchiveError, match="symbolic link"):
        safe_extract_zip(z, tmp_path / "dst")


def test_safe_extract_size_limit(tmp_path):
    from spinescope.datasets import UnsafeArchiveError

    z = _zip_with(tmp_path / "big.zip", {"a.bin": b"0" * 1000})
    with pytest.raises(UnsafeArchiveError, match="limit"):
        safe_extract_zip(z, tmp_path / "dst", max_total_bytes=10)
