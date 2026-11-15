"""Render a :class:`Consensus` as Markdown or as a self-contained HTML page.

Both renderers share one block model, so they show the same content. Only file basenames are shown
for evidence images; the HTML page embeds thumbnails (base64 PNG) so it works without the image files.
"""

from __future__ import annotations

import base64
import html
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .merge import Consensus
from .schema import GLOBAL, level_fields
from .schema import field_path as _field_path

log = logging.getLogger(__name__)

LEGEND = (
    "Legend: value (agreeing readers / readers with a definite value). n/a = no reader reported the field "
    "(not assessed; a missing field is not normal). indeterminate = the reader looked and could not decide; "
    "these are listed but do not count as disagreement. [DISAGREE] = definite values differ. SPLIT = no "
    "consensus value (tie). Continuous values show the median and the range across readers."
)

#: Columns of the summary-by-level table: (header, schema group, schema field).
SUMMARY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("Pfirrmann", "disc", "pfirrmann"),
    ("Contour", "disc", "contour"),
    ("Canal", "central_canal", "stenosis"),
    ("Foramen R (Lee)", "foramen_R", "lee_grade"),
    ("Foramen L (Lee)", "foramen_L", "lee_grade"),
    ("Modic above", "endplate_above", "modic"),
    ("Modic below", "endplate_below", "modic"),
    ("Root R", "nerve_root_R", "status"),
    ("Root L", "nerve_root_L", "status"),
    ("Listhesis", "listhesis", "meyerding_grade"),
)

_SECTION_TITLES: dict[str, str] = {
    "focal_lesions": "Focal lesions",
    "incidental_findings": "Incidental findings",
    "pain_generator_hypotheses": "Pain generator hypotheses (hypotheses, not diagnoses)",
}


@dataclass
class _Ev:
    """Evidence list placed in a table cell (text in Markdown, thumbnails in HTML)."""

    items: list[dict[str, Any]]


Cell = str | _Ev
Block = tuple[Any, ...]


def _fmt(v: Any) -> str:
    if v is None:
        return "indeterminate"
    if isinstance(v, float):
        text = f"{v:.2f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(v)


def _cell_text(rec: dict[str, Any]) -> str:
    status = rec["status"]
    if status == "not_assessed":
        return "n/a"
    if status == "indeterminate":
        base = "indeterminate"
    elif status == "split":
        base = "SPLIT: " + " / ".join(
            f"{_fmt(d['value'])} ({len(d['readers'])})" for d in rec["distribution"]
        )
    else:
        base = _fmt(rec["value"])
        if rec["n_definite"] > 1:
            base += f" ({rec['agreement']}/{rec['n_definite']})"
        if status == "numeric" and rec["range"]:
            base += f" range {_fmt(rec['range'][0])} to {_fmt(rec['range'][1])}"
    if rec["indeterminate_by"] and rec["n_definite"]:
        base += f" [+{len(rec['indeterminate_by'])} indeterminate]"
    if rec["disagreement"]:
        base += " [DISAGREE]"
    return base


def _agree_text(rec: dict[str, Any]) -> str:
    return f"{rec['agreement']}/{rec['n_definite']}" if rec["n_definite"] else "-"


def _votes_text(rec: dict[str, Any]) -> str:
    parts = [f"{r}: {_fmt(v['value'])} ({v['confidence']})" for r, v in rec["votes"].items()]
    if rec["not_assessed_by"]:
        parts.append("not assessed by " + ", ".join(rec["not_assessed_by"]))
    return "; ".join(parts) or "-"


def _dist_text(rec: dict[str, Any]) -> str:
    return "; ".join(f"{_fmt(d['value'])}: {', '.join(d['readers'])}" for d in rec["distribution"]) or "-"


def _ev_text(evidence: list[dict[str, Any]]) -> str:
    parts = []
    for ev in evidence:
        sl = ",".join(str(s) for s in ev["slices"]) or "whole"
        parts.append(f"{Path(ev['image']).name} [{ev['series']} k={sl}] ({', '.join(ev['readers'])})")
    return "; ".join(parts) or "-"


def _write(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _field_rows(recs: list[dict[str, Any]]) -> tuple[list[tuple[list[Cell], str | None]], list[str]]:
    """Table rows for fields a reader reported, plus titles of fields no reader reported."""
    rows: list[tuple[list[Cell], str | None]] = []
    missing: list[str] = []
    for rec in recs:
        if rec["status"] == "not_assessed":
            missing.append(rec["title"])
            continue
        flag = "dis" if rec["disagreement"] else ("ind" if rec["indeterminate_by"] else None)
        rows.append(
            ([rec["title"], _cell_text(rec), _agree_text(rec), _votes_text(rec), _Ev(rec["evidence"])], flag)
        )
    return rows, missing


def _blocks(c: Consensus) -> list[Block]:
    b: list[Block] = []
    names = ", ".join(r["reader"] for r in c.readers) or "none"
    b.append(("h1", "Spine findings: consensus across readers"))
    b.append(
        (
            "p",
            f"Readers merged: {len(c.readers)} ({names}). Rejected files: {len(c.rejected)}. "
            f"Non-findings JSON skipped: {len(c.skipped)}.",
        )
    )
    b.append(("p", LEGEND))
    b.append(("h2", "Readers"))
    b.append(
        (
            "table",
            ["Reader", "Weight", "Scope"],
            [([r["reader"], _fmt(r["weight"]), r["scope"]], None) for r in c.readers],
            None,
        )
    )
    if c.rejected:
        b.append(("h2", f"Rejected files ({len(c.rejected)})"))
        b.append(("ul", [f"{r['source']}: {'; '.join(r['errors'][:3])}" for r in c.rejected]))
    if c.warnings:
        b.append(("h2", f"Warnings ({len(c.warnings)})"))
        b.append(("ul", c.warnings[:50]))

    b.append(("h2", f"Disagreements ({len(c.disagreements)})"))
    if c.disagreements:
        rows = [
            (
                [
                    d["where"],
                    _fmt(d["value"]) if d["value"] is not None else d["status"],
                    "; ".join(f"{_fmt(x['value'])}: {', '.join(x['readers'])}" for x in d["distribution"]),
                ],
                "dis",
            )
            for d in c.disagreements
        ]
        b.append(("table", ["Field", "Consensus", "Readers per value"], rows, None))
    else:
        b.append(("p", "No disagreements between definite values."))

    b.append(("h2", f"Indeterminate ({len(c.indeterminate)})"))
    if c.indeterminate:
        rows = [
            (
                [
                    d["where"],
                    ", ".join(d["indeterminate_by"]),
                    _fmt(d["value"]) if d["n_definite"] else "no definite value",
                ],
                "ind",
            )
            for d in c.indeterminate
        ]
        b.append(("p", "Readers that looked and could not decide. Not counted as disagreement."))
        b.append(("table", ["Field", "Indeterminate by", "Consensus of the definite votes"], rows, None))
    else:
        b.append(("p", "None."))

    b.append(("h2", "Summary by level"))
    summary_rows: list[tuple[list[Cell], str | None]] = []
    for lvl in c.levels:
        cells: list[Cell] = [lvl]
        flag: str | None = None
        for _, group, name in SUMMARY_COLUMNS:
            rec = c.fields.get(_field_path(lvl, group, name))
            cells.append(_cell_text(rec) if rec else "n/a")
            if rec and rec["disagreement"]:
                flag = "dis"
        readers_n = max(
            (
                c.fields[_field_path(lvl, g, n)]["n_covering"]
                for _, g, n in SUMMARY_COLUMNS
                if _field_path(lvl, g, n) in c.fields
            ),
            default=0,
        )
        cells.append(f"{readers_n} of {len(c.readers)}")
        summary_rows.append((cells, flag))
    if summary_rows:
        b.append(
            (
                "table",
                ["Level"] + [h for h, _, _ in SUMMARY_COLUMNS] + ["Readers covering"],
                summary_rows,
                None,
            )
        )
    else:
        b.append(("p", "No disc levels reported."))

    b.append(("h2", "Levels"))
    for lvl in c.levels:
        b.append(("h3", lvl))
        recs = [
            c.fields[_field_path(lvl, f.group, f.name)]
            for f in level_fields()
            if _field_path(lvl, f.group, f.name) in c.fields
        ]
        _field_block(b, recs)
    if GLOBAL in {rec["level"] for rec in c.fields.values()}:
        b.append(("h2", "Global sections"))
        recs = [rec for rec in c.fields.values() if rec["level"] == GLOBAL]
        _field_block(b, recs)

    for section, groups in c.items.items():
        b.append(("h2", _SECTION_TITLES.get(section, section)))
        if not groups:
            b.append(("p", "None reported by any reader."))
        for g in groups:
            title = f"{g['level']} / {g['kind']}"
            b.append(("h3", title + (" [DISAGREE]" if g["disagreement"] else "")))
            b.append(
                (
                    "p",
                    f"Reported by {len(g['reported_by'])} of {g['n_readers']}: "
                    f"{', '.join(g['reported_by'])}." + (" Partial coverage." if g["partial"] else ""),
                )
            )
            if g["kind_variants"]:
                b.append(("p", "Kind variants merged: " + ", ".join(g["kind_variants"])))
            b.append(
                ("ul", [f"{r}: {s} (confidence {g['confidence'][r]})" for r, s in g["summaries"].items()])
            )
            if g["attributes"]:
                attr_rows, missing = _field_rows(list(g["attributes"].values()))
                b.append(
                    ("table", ["Attribute", "Consensus", "Agreement", "Votes", "Evidence"], attr_rows, None)
                )
                notes = _notes_lines(list(g["attributes"].values()))
                if notes:
                    b.append(("ul", notes))
                if missing:
                    b.append(("p", "Attributes no reader reported: " + ", ".join(missing)))
            if g["evidence"]:
                b.append(("p", "Evidence: " + _ev_text(g["evidence"])))
    return b


def _notes_lines(recs: list[dict[str, Any]]) -> list[str]:
    """One line per reader note, so reasoning is kept next to the consensus."""
    out = []
    for rec in recs:
        for reader, vote in rec["votes"].items():
            if vote["notes"]:
                out.append(f"{rec['title']} / {reader}: {vote['notes']}")
    return out


def _field_block(b: list[Block], recs: list[dict[str, Any]]) -> None:
    if not recs:
        b.append(("p", "No reader assessed these fields."))
        return
    rows, missing = _field_rows(recs)
    if rows:
        b.append(
            (
                "table",
                ["Field", "Consensus", "Agreement", "Votes (value, confidence)", "Evidence"],
                rows,
                None,
            )
        )
    notes = _notes_lines(recs)
    if notes:
        b.append(("p", "Reader notes:"))
        b.append(("ul", notes))
    if missing:
        b.append(("p", f"Not assessed by any reader ({len(missing)}): " + ", ".join(missing)))


def _md_cell(cell: Cell) -> str:
    if isinstance(cell, _Ev):
        return _md_escape(_ev_text(cell.items))
    return _md_escape(str(cell))


def _render_md(blocks: list[Block]) -> str:
    out: list[str] = []
    for blk in blocks:
        kind = blk[0]
        if kind in ("h1", "h2", "h3"):
            out.append("#" * int(kind[1]) + " " + blk[1])
        elif kind == "p":
            out.append(blk[1])
        elif kind == "ul":
            out += [f"- {_md_escape(x)}" for x in blk[1]] or ["- none"]
        elif kind == "table":
            headers, rows = blk[1], blk[2]
            out.append("| " + " | ".join(headers) + " |")
            out.append("|" + "|".join("---" for _ in headers) + "|")
            for cells, flag in rows:
                txt = [_md_cell(x) for x in cells]
                if flag == "dis":
                    txt[0] = f"**{txt[0]}**"
                out.append("| " + " | ".join(txt) + " |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _thumbnail_b64(image: str, image_root: Path | None, thumb_px: int) -> str | None:
    """Base64 PNG thumbnail of an evidence image, or None if it cannot be read."""
    p = Path(image)
    if not p.is_absolute():
        p = (image_root or Path.cwd()) / p
    try:
        from PIL import Image

        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_px, thumb_px))
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
    except (OSError, ValueError) as exc:
        log.warning("evidence image not embedded (%s): %s", p.name, exc)
        return None
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _Gallery:
    """Thumbnails embedded once per image; evidence cells link to them."""

    def __init__(self, image_root: Path | None, thumb_px: int) -> None:
        self.image_root = image_root
        self.thumb_px = thumb_px
        self.index: dict[str, int] = {}
        self.entries: list[tuple[int, str, str, str]] = []  # (n, basename, series, base64)

    def add_all(self, blocks: list[Block]) -> None:
        for blk in blocks:
            if blk[0] == "table":
                for cells, _ in blk[2]:
                    for cell in cells:
                        if isinstance(cell, _Ev):
                            for ev in cell.items:
                                self.add(ev)

    def add(self, ev: dict[str, Any]) -> int | None:
        image = str(ev["image"])
        if image in self.index:
            return self.index[image] or None
        data = _thumbnail_b64(image, self.image_root, self.thumb_px)
        if data is None:
            self.index[image] = 0  # remember the failure so it is not retried for every citation
            return None
        n = len(self.entries) + 1
        self.index[image] = n
        self.entries.append((n, Path(image).name, str(ev["series"]), data))
        return n


def _html_cell(cell: Cell, gallery: _Gallery) -> str:
    if not isinstance(cell, _Ev):
        return html.escape(str(cell))
    parts = []
    for ev in cell.items:
        sl = ",".join(str(s) for s in ev["slices"]) or "whole"
        label = f"{html.escape(Path(ev['image']).name)} [{html.escape(str(ev['series']))} k={sl}]"
        n = gallery.index.get(str(ev["image"])) or None
        if n is not None:
            parts.append(f'<a href="#img-{n}">{label}</a>')
        else:
            parts.append(label)
    return "<br>".join(parts) or "-"


_CSS = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222;max-width:1400px}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}
th,td{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}
th{background:#f0f0f0}
tr.dis{background:#fde2e2}
tr.dis td:first-child{font-weight:bold}
tr.ind{background:#e8f0fb}
.gallery{display:flex;flex-wrap:wrap;gap:12px}
.gallery figure{margin:0;width:260px;font-size:12px;color:#444}
.gallery img{max-width:250px;border:1px solid #999}
.legend{font-size:12px;color:#555;max-width:1100px}"""


def _render_html(blocks: list[Block], gallery: _Gallery, title: str) -> str:
    out: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
    ]
    for blk in blocks:
        kind = blk[0]
        if kind in ("h1", "h2", "h3"):
            out.append(f"<{kind}>{html.escape(blk[1])}</{kind}>")
        elif kind == "p":
            cls = ' class="legend"' if blk[1].startswith("Legend:") else ""
            out.append(f"<p{cls}>{html.escape(blk[1])}</p>")
        elif kind == "ul":
            items = blk[1] or ["none"]
            out.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in items) + "</ul>")
        elif kind == "table":
            headers, rows = blk[1], blk[2]
            out.append(
                "<table><thead><tr>"
                + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
                + "</tr></thead><tbody>"
            )
            for cells, flag in rows:
                cls = f' class="{flag}"' if flag in ("dis", "ind") else ""
                tds = "".join(f"<td>{_html_cell(x, gallery)}</td>" for x in cells)
                out.append(f"<tr{cls}>{tds}</tr>")
            out.append("</tbody></table>")
    if gallery.entries:
        out.append('<h2>Evidence images</h2><div class="gallery">')
        for n, name, series, data in gallery.entries:
            out.append(
                f'<figure id="img-{n}"><img alt="evidence {html.escape(series)}" '
                f'src="data:image/png;base64,{data}"><figcaption>{html.escape(name)} '
                f"({html.escape(series)})</figcaption></figure>"
            )
        out.append("</div>")
    out.append("</body></html>")
    return "\n".join(out) + "\n"


def render_markdown(consensus: Consensus, path: str | Path | None = None) -> str:
    """Render the consensus as Markdown. If ``path`` is given the text is also written there."""
    text = _render_md(_blocks(consensus))
    if path is not None:
        _write(path, text)
    return text


def render_html(
    consensus: Consensus,
    path: str | Path | None = None,
    *,
    embed_images: bool = True,
    image_root: str | Path | None = None,
    thumb_px: int = 240,
    title: str = "Spine findings: consensus",
) -> str:
    """Render the consensus as one self-contained HTML page.

    Evidence thumbnails are embedded (base64 PNG, ``thumb_px`` max side) when the image files can be
    read. Relative image paths are resolved against ``image_root`` (default: the working directory).
    Set ``embed_images=False`` for a text-only page. Disagreeing rows are highlighted.
    """
    blocks = _blocks(consensus)
    gallery = _Gallery(Path(image_root) if image_root is not None else None, thumb_px)
    if embed_images:
        gallery.add_all(blocks)
    text = _render_html(blocks, gallery, title)
    if path is not None:
        _write(path, text)
    return text
