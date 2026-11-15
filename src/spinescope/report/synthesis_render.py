"""Render a :class:`~spinescope.report.synthesis.Synthesis` as Markdown, self-contained HTML, findings JSON and AI text.

The three human outputs share one content model (:func:`_sections`), so they say the same thing. The HTML page is a
single file: CSS inline, evidence thumbnails embedded as base64 PNG, no scripts, printable on A4 with the table rows
kept whole. Experimental and not-validated findings sit in their own table with a text badge (colour is never the only
cue). Evidence is named by figure ("discs:disc_strip"), never by file path or input file name.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .synthesis import (
    DISCLAIMER,
    FORMULA,
    PAIN_STRUCTURES,
    PAIN_TITLES,
    Finding,
    NotAssessed,
    Synthesis,
    finding_text,
)
from .validation import STATUS_WORDS, STATUSES

log = logging.getLogger(__name__)

PAIN_FOOTER = (
    "Descriptive only. Consider correlation with symptoms and examination. The order is an evidence-strength "
    "ordering, not a severity or a diagnosis."
)
_SIDE_WORD = {"R": "right", "L": "left", "": "-"}
#: Table headers, shared by the Markdown and HTML pages.
_FINDING_HEADERS = ("Side", "Finding", "Value", "Confidence", "Validation", "Evidence figures", "Notes")
_NA_HEADERS = ("Field", "Side", "Reason", "Detail")
_PAIN_HEADERS = ("Rank", "Level (name)", "Strength", *(PAIN_TITLES[s] for s in PAIN_STRUCTURES))


# ------------------------------------------------------------------------------------------ content model
def _value(v: Any) -> str:
    if v is None:
        return "indeterminate"
    if isinstance(v, float):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return str(v)


def _figs(f: Finding | None, figs: tuple[str, ...]) -> str:
    return ", ".join(figs) if figs else "-"


def _finding_row(f: Finding) -> list[str]:
    badge = STATUS_WORDS[f.validation.status]
    return [
        _SIDE_WORD.get(f.side, "-"),
        f.title,
        _value(f.value),
        f.confidence,
        badge,
        ", ".join(f.figures) if f.figures else "-",
        f.notes or "-",
    ]


def _na_row(n: NotAssessed) -> list[str]:
    from .synthesis import REASON_TEXT

    return [n.field, _SIDE_WORD.get(n.side, "-"), REASON_TEXT.get(n.reason, n.reason), n.detail or "-"]


STATUS_MEANING: dict[str, str] = {
    "validated": "measured against public reference data; the metric is in the findings JSON and the report",
    "weak_labels": "checked only against weak labels (free-text reports), not against a radiologist grading",
    "experimental": "implemented; the acceptance rule was not met, or the metric is too weak to use",
    "not_validated": "no validation recorded for this field",
}


def _status_legend() -> list[list[str]]:
    return [[STATUS_WORDS[s], STATUS_MEANING[s]] for s in STATUSES]


def _study_lines(syn: Synthesis) -> list[str]:
    q = syn.image_quality
    cov = syn.coverage
    seqs = (
        ", ".join(f"{s['key']} ({s['contrast']}, {s['slices']} slices)" for s in cov["sequences"]) or "none"
    )
    miss = ", ".join(cov["missing_sequences"]) or "none"
    analyses = "; ".join(
        f"{name}: {a['status']}, {a['levels_with_values']} levels with values"
        for name, a in sorted(cov["analyses"].items())
    )
    return [
        f"Level naming: {syn.numbering}. {cov.get('numbering_text', '')}".strip(),
        f"Image quality: {q.get('line', 'not assessed')}",
        f"Levels labelled: {cov['levels_labelled']}. Sequences: {seqs}. Missing: {miss}.",
        f"Coverage: {cov['fields_registered']} of {cov['fields_in_schema']} level fields have a registered analysis.",
        f"Analyses: {analyses or 'none'}.",
    ]


def _pain_cells(groups: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    out = []
    for s in PAIN_STRUCTURES:
        items = groups.get(s, [])
        out.append("; ".join(x["text"] for x in items) if items else "-")
    return out


# ------------------------------------------------------------------------------------------ markdown
def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _md_table(headers: tuple[str, ...] | list[str], rows: list[list[str]]) -> list[str]:
    out = [
        "| " + " | ".join(_md_escape(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        out.append("| " + " | ".join(_md_escape(str(c)) for c in r) + " |")
    return out


def render_markdown(syn: Synthesis, path: str | Path | None = None) -> str:
    """Markdown report. If ``path`` is given the text is also written there."""
    out: list[str] = ["# spinescope synthesis: automatic measurements", "", DISCLAIMER, ""]
    out += ["## Study", ""] + [f"- {x}" for x in _study_lines(syn)] + [""]
    out += ["## Pain-generator evidence (descriptive)", "", FORMULA, ""]
    rows = []
    for r in syn.pain_evidence:
        rows.append(
            [
                str(r["rank"]),
                f"{r['level']} ({r['name_status']})",
                f"{r['strength']:.2f}",
                *_pain_cells(r["groups"]),
            ]
        )
    out += [*_md_table(_PAIN_HEADERS, rows), "", PAIN_FOOTER, ""]
    out += ["## Global findings", ""]
    if syn.global_findings:
        out += [*_md_table(_FINDING_HEADERS, [_finding_row(f) for f in syn.global_findings]), ""]
    else:
        out += ["No whole-study field has a value with image evidence in this run.", ""]
    out += ["## Levels", ""]
    for lv in syn.levels:
        out += [f"### {lv.level}", "", f"Level name: {lv.name_status}. {lv.name_policy}.", ""]
        validated = [f for f in lv.findings if not f.experimental]
        experimental = [f for f in lv.findings if f.experimental]
        out += ["#### Findings with validation", ""]
        out += _md_table(_FINDING_HEADERS, [_finding_row(f) for f in validated]) if validated else ["None."]
        out += [""]
        if experimental:
            out += [
                "#### Experimental or not validated (shown apart; not for clinical use)",
                "",
                *_md_table(_FINDING_HEADERS, [_finding_row(f) for f in experimental]),
                "",
            ]
        out += ["#### Not assessed", ""]
        out += _md_table(_NA_HEADERS, [_na_row(n) for n in lv.not_assessed]) if lv.not_assessed else ["None."]
        out += [""]
    out += ["## Not assessed for this study", ""]
    out += [
        f"- {n.field}: {_reason(n)}" + (f" ({n.detail})" if n.detail else "") for n in syn.study_not_assessed
    ]
    if not syn.study_not_assessed:
        out += ["None."]
    out += ["", "## Validation status", "", *_md_table(("Status", "Meaning"), _status_legend()), ""]
    out += [DISCLAIMER, ""]
    text = "\n".join(out).rstrip() + "\n"
    if path is not None:
        _write(path, text)
    return text


def _reason(n: NotAssessed) -> str:
    from .synthesis import REASON_TEXT

    return REASON_TEXT.get(n.reason, n.reason)


# ------------------------------------------------------------------------------------------ html
_CSS = """
:root{color-scheme:light}
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px auto;max-width:1200px;padding:0 16px;color:#1a1a1a;
line-height:1.4;background:#fff}
h1{font-size:1.6em}h2{font-size:1.3em;border-bottom:2px solid #333;padding-bottom:2px;margin-top:28px}
h3{font-size:1.1em;margin-top:22px}h4{font-size:1em;margin:14px 0 4px}
.disclaimer{border:2px solid #7a0000;background:#fff4f4;padding:10px 14px;font-size:0.95em}
.skip{position:absolute;left:-999px}.skip:focus{left:8px;top:8px;background:#fff;padding:4px;border:1px solid #000}
table{border-collapse:collapse;margin:6px 0 14px;width:100%;font-size:0.9em}
caption{text-align:left;font-weight:600;padding:2px 0 4px}
th,td{border:1px solid #888;padding:4px 6px;vertical-align:top;text-align:left}
th{background:#eee}
tr{break-inside:avoid}
.badge{display:inline-block;border:1px solid #000;padding:0 5px;border-radius:3px;font-size:0.85em;font-weight:600}
.b-exp{background:#ffd966;color:#000}.b-nv{background:#f4cccc;color:#000}.b-val{background:#d9ead3;color:#000}
.b-weak{background:#cfe2f3;color:#000}
section.experimental{border:2px dashed #7a4b00;padding:6px 12px;margin:10px 0;background:#fffbea}
.note{font-size:0.9em;color:#333}
.gallery{display:flex;flex-wrap:wrap;gap:12px}
.gallery figure{margin:0;width:260px;font-size:0.85em}
.gallery img{max-width:250px;border:1px solid #555}
@page{size:A4;margin:12mm}
@media print{.skip{display:none}a{color:#000;text-decoration:none}body{margin:0;max-width:none}
h2{break-after:avoid}table{font-size:8.5pt}}
"""

_BADGE_CLASS = {
    "validated": "b-val",
    "weak_labels": "b-weak",
    "experimental": "b-exp",
    "not_validated": "b-nv",
}


def _h(text: Any) -> str:
    import html

    return html.escape(str(text), quote=True)


def _html_badge(status: str) -> str:
    return f'<span class="badge {_BADGE_CLASS[status]}">{_h(STATUS_WORDS[status])}</span>'


def _html_table(
    caption: str,
    headers: list[str] | tuple[str, ...],
    rows: list[list[str]],
    raw: frozenset[int] = frozenset(),
) -> str:
    """Table with a caption and column scopes. Cells are escaped unless their column index is in ``raw``."""
    head = "".join(f'<th scope="col">{_h(h)}</th>' for h in headers)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            cells.append(f"<td>{c if i in raw else _h(c)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><caption>{_h(caption)}</caption><thead><tr>{head}</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _finding_rows_html(findings: list[Finding]) -> list[list[str]]:
    """Finding rows for HTML: the validation column holds a badge (already escaped), the rest is escaped text."""
    rows = []
    for f in findings:
        row = [_h(c) for c in _finding_row(f)]
        row[4] = _html_badge(f.validation.status)  # built from constants, not from study text
        rows.append(row)
    return rows


def render_html(
    syn: Synthesis,
    path: str | Path | None = None,
    *,
    figure_files: Mapping[str, Path] | None = None,
    thumb_px: int = 240,
    title: str = "spinescope synthesis: automatic measurements",
) -> str:
    """One self-contained HTML page. ``figure_files`` maps "analysis:figure" to an image file; those used by a finding
    are embedded as thumbnails (base64 PNG). Without it the page has no images."""
    figs_used: list[str] = []
    parts: list[str] = []
    parts.append("<header><h1>" + _h(title) + "</h1></header>")
    parts.append(f'<p class="disclaimer" role="note"><strong>Disclaimer.</strong> {_h(DISCLAIMER)}</p>')
    parts.append('<main id="main">')
    parts.append('<section aria-labelledby="study"><h2 id="study">Study</h2><ul>')
    parts += [f"<li>{_h(x)}</li>" for x in _study_lines(syn)]
    parts.append("</ul></section>")

    parts.append('<section aria-labelledby="pain"><h2 id="pain">Pain-generator evidence (descriptive)</h2>')
    parts.append(f'<p class="note">{_h(FORMULA)}</p>')
    rows = []
    for r in syn.pain_evidence:
        cells = [str(r["rank"]), f"{r['level']} ({r['name_status']})", f"{r['strength']:.2f}"]
        for s in PAIN_STRUCTURES:
            items = r["groups"].get(s, [])
            cell = "<br>".join(_h(x["text"]) for x in items) if items else "-"
            cells.append(cell)
        rows.append([_h(c) for c in cells[:3]] + cells[3:])
    parts.append(
        _html_table(
            "Levels ordered by strength of measured evidence",
            _PAIN_HEADERS,
            rows,
            raw=frozenset(range(3, 3 + len(PAIN_STRUCTURES))),
        )
    )
    parts.append(f'<p class="note">{_h(PAIN_FOOTER)}</p></section>')

    parts.append('<section aria-labelledby="glob"><h2 id="glob">Global findings</h2>')
    if syn.global_findings:
        parts.append(
            _html_table(
                "Whole-study fields",
                _FINDING_HEADERS,
                _finding_rows_html(syn.global_findings),
                raw=frozenset({4}),
            )
        )
        figs_used += [fig for f in syn.global_findings for fig in f.figures]
    else:
        parts.append("<p>No whole-study field has a value with image evidence in this run.</p>")
    parts.append("</section>")

    parts.append('<section aria-labelledby="levels"><h2 id="levels">Levels</h2>')
    for lv in syn.levels:
        parts.append(
            f'<section aria-labelledby="lv-{_h(lv.level)}"><h3 id="lv-{_h(lv.level)}">{_h(lv.level)}</h3>'
            f"<p>Level name: <strong>{_h(lv.name_status)}</strong>. {_h(lv.name_policy)}.</p>"
        )
        validated = [f for f in lv.findings if not f.experimental]
        experimental = [f for f in lv.findings if f.experimental]
        parts.append(
            _html_table(
                "Findings with validation",
                _FINDING_HEADERS,
                _finding_rows_html(validated),
                raw=frozenset({4}),
            )
            if validated
            else "<p>No validated finding.</p>"
        )
        figs_used += [fig for f in validated for fig in f.figures]
        if experimental:
            parts.append(
                '<section class="experimental" aria-label="Experimental findings">'
                "<p><strong>Experimental or not validated.</strong> Shown apart. Not for clinical use.</p>"
                + _html_table(
                    "Experimental or not validated",
                    _FINDING_HEADERS,
                    _finding_rows_html(experimental),
                    raw=frozenset({4}),
                )
                + "</section>"
            )
            figs_used += [fig for f in experimental for fig in f.figures]
        if lv.not_assessed:
            parts.append(_html_table("Not assessed", _NA_HEADERS, [_na_row(n) for n in lv.not_assessed]))
        parts.append("</section>")
    parts.append("</section>")

    parts.append('<section aria-labelledby="na"><h2 id="na">Not assessed for this study</h2><ul>')
    parts += [
        f"<li>{_h(n.field)}: {_h(_reason(n))}" + (f" ({_h(n.detail)})" if n.detail else "") + "</li>"
        for n in syn.study_not_assessed
    ]
    if not syn.study_not_assessed:
        parts.append("<li>None.</li>")
    parts.append("</ul></section>")

    parts.append('<section aria-labelledby="val"><h2 id="val">Validation status</h2>')
    parts.append(
        _html_table(
            "What each status means",
            ("Status", "Meaning"),
            [[_html_badge(s), _h(STATUS_MEANING[s])] for s in STATUSES],
            raw=frozenset({0}),
        )
    )
    parts.append("</section>")

    gallery = _gallery(sorted(set(figs_used)), figure_files, thumb_px)
    if gallery:
        parts.append(
            '<section aria-labelledby="fig"><h2 id="fig">Evidence images</h2>' + gallery + "</section>"
        )
    parts.append("</main>")
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_h(title)}</title><style>{_CSS}</style></head><body>"
        '<a class="skip" href="#main">Skip to content</a>' + "".join(parts) + "</body></html>\n"
    )
    if path is not None:
        _write(path, page)
    return page


def _gallery(names: list[str], figure_files: Mapping[str, Path] | None, thumb_px: int) -> str:
    if not figure_files or not names:
        return ""
    items = []
    for n in names:
        p = figure_files.get(n)
        if p is None:
            continue
        data = _thumbnail_b64(Path(p), thumb_px)
        if data is None:
            continue
        items.append(
            f'<figure><img alt="Evidence figure {_h(n)}" src="data:image/png;base64,{data}">'
            f"<figcaption>{_h(n)}</figcaption></figure>"
        )
    return '<div class="gallery">' + "".join(items) + "</div>" if items else ""


def _thumbnail_b64(path: Path, thumb_px: int) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_px, thumb_px))
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
    except (OSError, ValueError) as exc:
        log.warning("evidence figure not embedded: %s", type(exc).__name__)
        return None
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------------------------------ AI text
def brief_lines(syn: Synthesis, *, max_levels: int = 12) -> list[str]:
    """Plain text for an AI reader: the study lines, the pain-evidence order, each level's findings and not-assessed
    fields, and the disclaimer. Wording is the same as the human outputs."""
    out = [DISCLAIMER, "Study:"]
    out += [f"  {x}" for x in _study_lines(syn)]
    out.append("Pain-generator evidence (descriptive; " + FORMULA.split(";")[0] + "):")
    for r in syn.pain_evidence[:max_levels]:
        kinds = [s for s in PAIN_STRUCTURES if r["groups"].get(s)]
        out.append(
            f"  {r['rank']}. {r['level']} ({r['name_status']}), strength {r['strength']:.2f}: "
            + (", ".join(kinds) or "no abnormal measurement")
        )
    for lv in syn.levels[:max_levels]:
        out.append(f"Level {lv.level} (name {lv.name_status}):")
        for f in lv.findings:
            out.append(
                "  - " + finding_text(f) + (f" [figures: {', '.join(f.figures)}]" if f.figures else "")
            )
        if lv.not_assessed:
            out.append("  not assessed: " + "; ".join(f"{n.field} ({_reason(n)})" for n in lv.not_assessed))
    if syn.global_findings:
        out.append("Global:")
        out += ["  - " + finding_text(f) for f in syn.global_findings]
    if syn.study_not_assessed:
        out.append(
            "Not assessed for the study: "
            + "; ".join(f"{n.field} ({_reason(n)})" for n in syn.study_not_assessed)
        )
    out.append(PAIN_FOOTER)
    return out


# ------------------------------------------------------------------------------------------ files
def _write(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def write_bundle(
    syn: Synthesis,
    findings: Mapping[str, Any],
    outdir: str | Path,
    *,
    figure_files: Mapping[str, Path] | None = None,
) -> list[Path]:
    """Write ``report.md``, ``report.html`` and ``findings.json`` into ``outdir``. Returns the three paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    fj = out / "findings.json"
    fj.write_text(
        json.dumps(findings, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    paths.append(fj)
    md = out / "report.md"
    render_markdown(syn, md)
    paths.append(md)
    html_path = out / "report.html"
    render_html(syn, html_path, figure_files=figure_files)
    paths.append(html_path)
    return paths
