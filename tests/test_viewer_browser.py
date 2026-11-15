"""Viewer smoke test in a real browser: no JavaScript errors, no network requests, keyboard use of the findings panel.

Skipped (with the reason) unless Playwright is installed and a Chromium it can start is found. Set
``SPINESCOPE_CHROMIUM`` to a browser binary to use one outside Playwright's cache. CI does not install a browser,
so this runs locally (``pip install playwright && playwright install chromium``) and on any machine that has one.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sync_api = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed (pip install playwright)"
)

from spinescope.viewer import build_viewer  # noqa: E402
from test_viewer_findings import _anchors, _result  # noqa: E402


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(executable_path=os.environ.get("SPINESCOPE_CHROMIUM") or None)
        except Exception as exc:  # no browser binary on this machine
            pytest.skip(f"no Chromium for Playwright: {str(exc).splitlines()[0][:120]}")
        yield b
        b.close()


@pytest.fixture(scope="module")
def page_file(phantom_study, tmp_path_factory: pytest.TempPathFactory) -> Path:
    res = {
        "discs": _result(
            "discs", flags=[{"kind": "bulge", "level": "L4-L5", "confidence": 0.8, "note": "check"}]
        ),
        "endplates": _result(
            "endplates",
            meta={"status": "experimental"},
            flags=[
                {"kind": "modic_suggestion", "level": "L4-L5", "side": "left", "confidence": 0.4},
                {"kind": "retrolisthesis", "level": "L3-L4", "confidence": 0.3},
            ],
        ),
    }
    out = tmp_path_factory.mktemp("viewer_browser")
    return build_viewer(phantom_study, out, anchors=_anchors(), title="Phantom (synthetic)", findings=res)


def _open(browser: Any, path: Path) -> tuple[Any, list[str], list[str]]:
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    problems: list[str] = []
    requests: list[str] = []
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.on(
        "console", lambda m: problems.append(f"console {m.type}: {m.text}") if m.type == "error" else None
    )
    page.on(
        "request",
        lambda r: requests.append(r.url) if not r.url.startswith(("file:", "data:", "blob:")) else None,
    )
    page.goto(path.as_uri())
    page.wait_for_timeout(1500)
    return page, problems, requests


def test_page_loads_without_errors_or_network(browser: Any, page_file: Path) -> None:
    page, problems, requests = _open(browser, page_file)
    assert page.inner_text("#findbtn") == "Findings (3)"
    assert problems == [], problems
    assert requests == [], requests
    # every form field, including the ones the page builds in script, has an id or a name (the a11y warning)
    unnamed = page.evaluate(
        "[...document.querySelectorAll('input,select,textarea')].filter(e => !e.id && !e.name).map(e => e.outerHTML)"
    )
    assert unnamed == [], unnamed
    page.close()


def test_findings_panel_keyboard_and_table(browser: Any, page_file: Path) -> None:
    page, problems, _ = _open(browser, page_file)
    page.focus("body")
    page.keyboard.press("f")  # F opens the panel
    assert page.is_visible("#findings")
    assert page.locator("#fx-body tr[data-id]").count() == 3
    # Up and Down move between rows, Enter selects the row and moves the planes
    page.locator("#fx-body tr[data-id]").first.focus()
    page.keyboard.press("ArrowDown")
    focused = page.evaluate("document.activeElement.dataset.id")
    assert focused == "1", focused
    page.keyboard.press("Enter")
    page.wait_for_timeout(300)
    assert "Finding:" in page.inner_text("#status")
    # the structure filter narrows the table
    page.select_option("#fx-structure", value="endplates")
    page.wait_for_timeout(200)
    assert page.locator("#fx-body tr[data-id]").count() == 2
    page.keyboard.press("Escape")  # no help dialog open: harmless
    assert problems == [], problems
    page.close()


def test_page_without_findings_hides_the_button(browser: Any, phantom_study, tmp_path: Path) -> None:
    path = build_viewer(phantom_study, tmp_path, anchors=None, title="Phantom (synthetic)")
    page, problems, _ = _open(browser, path)
    assert not page.is_visible("#findbtn")
    assert problems == [], problems
    page.close()
