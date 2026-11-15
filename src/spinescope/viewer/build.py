"""Write the offline HTML viewer: one self-contained file, no CDN, no network, works from file://.

The page inlines its CSS, the pure geometry maths, the UI script and the study data (deflate-
compressed voxels, base64). Chrome, Edge, Safari and Firefox open it straight from disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from ..anatomy.anchors import Anchors
from ..config import DEFAULT, Config
from ..study import Study
from .export import build_payload

log = logging.getLogger(__name__)

#: File name written by :func:`build_viewer`.
VIEWER_FILE = "spinescope_viewer.html"
DEFAULT_TITLE = "Spine MRI viewer"


def _asset(name: str) -> str:
    return resources.files("spinescope.viewer").joinpath("assets").joinpath(name).read_text(encoding="utf-8")


def _json_for_script(obj: Any) -> str:
    """JSON safe to embed in a <script>: no '<' '>' '&' (so no '</script>' can appear)."""
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html(payload: dict[str, Any]) -> str:
    """The complete HTML document for a payload from :func:`spinescope.viewer.export.build_payload`."""
    title = str(payload.get("title") or DEFAULT_TITLE)
    subs = {
        "__TITLE__": _html_escape(title),
        "__CSS__": _asset("viewer.css"),
        "__MATH__": _asset("math.js"),
        "__LABELS__": _asset("labels.js"),
        "__APP__": _asset("app.js"),
        "__DATA__": _json_for_script(payload),
    }
    template = _asset("viewer.html")
    # single pass so that substituted text is never scanned for placeholders again
    pattern = re.compile("|".join(re.escape(k) for k in subs))
    return pattern.sub(lambda m: subs[m.group(0)], template)


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_viewer(
    study: Study,
    outdir: str | os.PathLike[str],
    anchors: Anchors | None = None,
    title: str | None = None,
    *,
    config: Config = DEFAULT,
    findings: Mapping[str, Any] | None = None,
    findings_notes: list[str] | None = None,
) -> Path:
    """Write ``spinescope_viewer.html`` into ``outdir`` and return its path.

    Shows every series of ``study``. Planes with no non-localizer series get a reformat computed
    in the browser from the best volume (labelled as such). ``anchors`` (optional) adds disc
    levels: lines across the images, labels on the image edge, and a disc jump list. ``title``
    is shown in the page; keep it free of patient information. The page contains only technical
    series information and the research-only disclaimer.

    ``findings`` (analysis name -> ``Result``, or the dict an ``analyze`` file holds) adds the findings panel:
    a table per level with the value, confidence and validation, and markers on the planes. Only the fields
    :mod:`spinescope.viewer.findings_view` documents are written. ``findings_notes`` are shown with the panel.
    """
    payload = build_payload(
        study,
        anchors,
        title=title or DEFAULT_TITLE,
        config=config,
        findings=findings,
        findings_notes=findings_notes,
    )
    out = Path(os.fspath(outdir))
    out.mkdir(parents=True, exist_ok=True)
    path = out / VIEWER_FILE
    html = render_html(payload)
    path.write_text(html, encoding="utf-8")
    n_ref = sum(1 for s in payload["series"] if s["reformat"])
    log.info(
        "viewer: %d series (%d reformat), %d disc labels, %.1f MB -> %s",
        len(payload["series"]),
        n_ref,
        len(payload["discs"]),
        path.stat().st_size / 1e6,
        path,
    )
    return path
