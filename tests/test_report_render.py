"""Rendering: Markdown and self-contained HTML, disagreement highlighting, evidence thumbnails, privacy."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from report_factory import F, findings, item, minimal_valid
from spinescope.report import merge, render_html, render_markdown


def _png(path: Path, shade: int = 128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((64, 96), shade, dtype=np.uint8)
    arr[:, 48:] = 255 - shade
    Image.fromarray(arr).save(path)
    return path


def _consensus(tmp_path: Path, *, with_images: bool = True):  # type: ignore[no-untyped-def]
    img_a = str(_png(tmp_path / "a" / "sag_t2.png", 60)) if with_images else "out/a/sag_t2.png"
    img_b = str(_png(tmp_path / "b" / "sag_t2.png", 200)) if with_images else "out/b/sag_t2.png"
    a = findings(
        "reader_a",
        levels={
            "L4-L5": {
                "disc": {"pfirrmann": F(3, image=img_a), "contour": F("protrusion", image=img_a)},
                "endplate_below": {"modic": F(None, "low", image=img_a)},
            },
            "L5-S1": {"nerve_root_R": {"status": F("compression", image=img_a)}},
        },
        global_={"alignment": {"scoliosis": F("none", image=img_a)}},
        focal_lesions=[
            item("L4", "l4_hemangioma", "A | B pipe <b>x</b>", attributes={"size_mm": F(6.0, image=img_a)})
        ],
        scope="sag T2 | L3-S1",
    )
    b = findings(
        "reader_b",
        levels={
            "L4-L5": {"disc": {"pfirrmann": F(4, image=img_b), "contour": F("bulge", image=img_b)}},
        },
        scope="sagittal T2",
    )
    return merge([a, b])


def test_markdown_has_all_sections_and_legend(tmp_path: Path) -> None:
    md = render_markdown(_consensus(tmp_path))
    for heading in [
        "# Spine findings: consensus across readers",
        "## Readers",
        "## Disagreements",
        "## Indeterminate",
        "## Summary by level",
        "## Levels",
        "### L4-L5",
        "### L5-S1",
        "## Global sections",
        "## Focal lesions",
    ]:
        assert heading in md, heading
    assert "Legend:" in md and "n/a = no reader reported" in md


def test_markdown_marks_disagreement_and_not_assessed(tmp_path: Path) -> None:
    md = render_markdown(_consensus(tmp_path))
    assert "[DISAGREE]" in md
    assert "**L4-L5: Pfirrmann grade**" in md
    assert "n/a" in md  # e.g. canal not assessed by anyone
    assert "Not assessed by any reader" in md


def test_markdown_shows_indeterminate_without_calling_it_a_disagreement(tmp_path: Path) -> None:
    res = _consensus(tmp_path)
    md = render_markdown(res)
    row = next(line for line in md.splitlines() if line.startswith("| L4-L5: Modic type"))
    assert "DISAGREE" not in row
    assert "indeterminate" in md


def test_markdown_escapes_table_pipes(tmp_path: Path) -> None:
    md = render_markdown(_consensus(tmp_path))
    assert "sag T2 \\| L3-S1" in md
    assert "A \\| B pipe" in md
    assert "<b>x</b>" in md  # raw text stays as text in Markdown


def test_markdown_writes_file_when_path_given(tmp_path: Path) -> None:
    out = tmp_path / "report" / "consensus.md"
    text = render_markdown(_consensus(tmp_path), out)
    assert out.read_text(encoding="utf-8") == text


def test_markdown_never_shows_full_image_paths(tmp_path: Path) -> None:
    md = render_markdown(_consensus(tmp_path))
    assert str(tmp_path) not in md
    assert "sag_t2.png [sag_t2 k=12]" in md


def test_html_is_self_contained_and_embeds_thumbnails(tmp_path: Path) -> None:
    html = render_html(_consensus(tmp_path))
    assert html.startswith("<!doctype html>")
    assert html.count("data:image/png;base64,") >= 2  # two distinct images
    assert not re.search(r'src="(?!data:)', html)  # no external or relative image links
    assert str(tmp_path) not in html  # no input paths leak


def test_html_embed_images_false_has_no_thumbnails(tmp_path: Path) -> None:
    html = render_html(_consensus(tmp_path), embed_images=False)
    assert "data:image/png" not in html
    assert "sag_t2.png" in html  # still named


def test_html_missing_images_do_not_fail(tmp_path: Path) -> None:
    html = render_html(_consensus(tmp_path, with_images=False))
    assert "data:image/png" not in html
    assert "sag_t2.png" in html
    assert "<table>" in html


def test_html_highlights_disagreement_rows_and_links_evidence(tmp_path: Path) -> None:
    html = render_html(_consensus(tmp_path))
    assert '<tr class="dis">' in html
    assert 'href="#img-1"' in html
    assert 'id="img-1"' in html


def test_html_escapes_untrusted_text(tmp_path: Path) -> None:
    res = _consensus(tmp_path)
    html = render_html(res)
    assert "<b>x</b>" not in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    evil = findings(
        "evil", levels={"L4-L5": {"disc": {"contour": F("bulge", notes="<script>alert(1)</script>")}}}
    )
    html = render_html(merge([evil]))
    assert "<script>alert(1)" not in html
    assert "Reader notes:" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_html_writes_file_and_uses_title(tmp_path: Path) -> None:
    out = tmp_path / "consensus.html"
    text = render_html(_consensus(tmp_path), out, title="Study X consensus")
    assert out.read_text(encoding="utf-8") == text
    assert "<title>Study X consensus</title>" in text


def test_html_thumbnail_size_is_bounded(tmp_path: Path) -> None:
    import base64
    import io

    big = _png(tmp_path / "big" / "big.png", 90)
    Image.fromarray(np.zeros((900, 700), dtype=np.uint8)).save(big)
    doc = findings("big", levels={"L4-L5": {"disc": {"pfirrmann": F(3, image=str(big))}}})
    html = render_html(merge([doc]), thumb_px=100)
    data = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
    assert data is not None
    im = Image.open(io.BytesIO(base64.b64decode(data.group(1))))
    assert max(im.size) <= 100


def test_summary_table_has_one_row_per_level_and_disagreement_flag(tmp_path: Path) -> None:
    md = render_markdown(_consensus(tmp_path))
    summary = md.split("## Summary by level", 1)[1].split("## Levels", 1)[0]
    rows = [line for line in summary.splitlines() if line.startswith("| ") and not line.startswith("| Level")]
    assert len(rows) == 2
    assert rows[0].startswith("| **L4-L5**")  # disagreement in this row
    assert rows[1].startswith("| L5-S1")


@pytest.mark.parametrize("renderer", ["md", "html"])
def test_empty_consensus_renders(renderer: str) -> None:
    res = merge([{"schema_version": "1.0", "reader": "x"}])  # rejected: no readers
    text = render_markdown(res) if renderer == "md" else render_html(res)
    assert "Readers merged: 0" in text
    assert "No disc levels reported." in text
    assert "None reported by any reader." in text


def test_full_minimal_document_renders_all_array_sections(tmp_path: Path) -> None:
    md = render_markdown(merge([minimal_valid("reader_a"), minimal_valid("reader_b")]))
    assert "### L4 / l4_hemangioma" in md
    assert "Small vertebral hemangioma" in md
    assert "Pain generator hypotheses" in md
    assert "Incidental findings" in md
