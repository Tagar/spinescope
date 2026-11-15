"""Model Context Protocol server: lets an AI assistant open a spine MRI study and drive spinescope.

Two layers:

* :class:`SpineTools` holds the tool logic in plain Python (no MCP import), so every tool is unit-tested
  directly. Images are returned as :class:`RenderResult` (PNG bytes plus metadata).
* :func:`build_server` wraps those methods as MCP tools (FastMCP, stdio transport). It needs the optional
  ``mcp`` package (``pip install "spinescope[mcp]"``) and is only imported when the server is built.

Privacy rules for every tool result: no DICOM attribute outside :data:`spinescope.io.KEEP_TAGS`, no ``meta``
keys, no input file path. The only handle on a study is the opaque ``study_id``. Error messages never repeat
the path that was given. Research and education only; not a diagnosis.

Run with the ``spinescope-mcp`` command (stdio), or ``python -m spinescope.ai.mcp_server``.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import secrets
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from ..anatomy.anchors import Anchors
from ..anatomy.pipeline import label_spine
from ..config import DEFAULT, Config
from ..geometry import Stack
from ..io.dicom import KEEP_TAGS
from ..render.annotate import add_rulers, orientation_letters
from ..render.crossref import linked_views, reference_lines
from ..render.frame import display_scale, min_display_px, slice_frame
from ..render.layout import with_title
from ..render.mpr import Reformat, disc_plane, reformat, render_reformat
from ..render.sheets import slice_label_lines, zoom_crop
from ..render.window import grey_rgb, resolve_window, series_window
from ..report.merge import merge
from ..report.render import render_markdown
from ..report.validate import validate
from ..study import Study, load_study
from ..viewer import build_viewer
from .briefs import brief, reading_tasks, series_table

log = logging.getLogger(__name__)

#: Plane names accepted by :meth:`SpineTools.reformat`.
REFORMAT_PLANES = ("sag", "cor", "ax", "disc")


class ToolError(Exception):
    """A tool failed for a reason that is safe to show to the model (never contains a file path)."""


@dataclass
class RenderResult:
    """A rendered image and its metadata. ``saved_to`` is set when the server saves PNGs to an output dir."""

    png: bytes
    meta: dict[str, Any] = field(default_factory=dict)
    saved_to: str | None = None


def png_bytes(img: NDArray[np.uint8]) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array as PNG bytes."""
    buf = BytesIO()
    PILImage.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _window_arg(window: Any) -> tuple[float, float] | str | None:
    """Accept a preset name, a [lo, hi] pair (JSON list) or None."""
    if window is None or isinstance(window, str):
        return window
    if isinstance(window, (list, tuple)) and len(window) == 2:
        return float(window[0]), float(window[1])
    raise ToolError("window must be a preset name, [lo, hi], or omitted")


def _public_info(study: Study) -> dict[str, Any]:
    return {k: v for k, v in study.info.items() if k in KEEP_TAGS}


class SpineTools:
    """Tool logic for the MCP server. Holds loaded studies (LRU) and their anchors, keyed by ``study_id``."""

    def __init__(
        self,
        *,
        max_studies: int = 4,
        output_dir: str | os.PathLike[str] | None = None,
        config: Config = DEFAULT,
    ) -> None:
        self._cfg = config
        self._max = max(1, max_studies)
        self._studies: OrderedDict[str, Study] = OrderedDict()
        self._anchors: dict[str, Anchors] = {}
        self._out = Path(os.fspath(output_dir)) if output_dir is not None else None
        self._counter = 0

    # ----------------------------------------------------------- internals
    def _study(self, study_id: str) -> Study:
        if study_id not in self._studies:
            raise ToolError("unknown study_id: open the study first with open_study")
        self._studies.move_to_end(study_id)
        return self._studies[study_id]

    def _anchors_for(self, study_id: str) -> Anchors:
        if study_id not in self._anchors:
            self._anchors[study_id] = label_spine(self._study(study_id), self._cfg)
        return self._anchors[study_id]

    def _anchors_cached(self, study_id: str) -> Anchors | None:
        return self._anchors.get(study_id)

    def _stack(self, study_id: str, key: str) -> Stack:
        st = self._study(study_id)
        if key not in st:
            raise ToolError(f"unknown series key {key!r}; choose one of: {', '.join(st.keys())}")
        return st[key]

    @staticmethod
    def _slice(stack: Stack, k: int) -> int:
        if not 0 <= int(k) < stack.n:
            raise ToolError(f"slice {k} out of range for {stack.key} (0..{stack.n - 1})")
        return int(k)

    @staticmethod
    def _rc(stack: Stack, rc: Any) -> tuple[float, float]:
        try:
            r, c = (float(v) for v in rc)
        except (TypeError, ValueError) as exc:
            raise ToolError("pixel coordinates must be [row, col]") from exc
        if not (-0.5 <= r <= stack.shape[1] - 0.5 and -0.5 <= c <= stack.shape[2] - 0.5):
            raise ToolError(f"pixel [{r:g}, {c:g}] is outside the {stack.shape[1]}x{stack.shape[2]} grid")
        return r, c

    def _emit(self, study_id: str, name: str, png: bytes, meta: dict[str, Any]) -> RenderResult:
        saved = None
        if self._out is not None:
            self._counter += 1
            d = self._out / study_id
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{self._counter:04d}_{name}.png"
            path.write_bytes(png)
            saved = str(path)
        return RenderResult(png=png, meta=meta, saved_to=saved)

    # ------------------------------------------------------------ study
    def open_study(self, path: str) -> dict[str, Any]:
        """Open a DICOM directory (any layout) or a NIfTI / MetaImage / NRRD volume. Returns a study_id, the
        technical study attributes and the series table. The path is not echoed back."""
        if not isinstance(path, str) or not os.path.exists(path):
            raise ToolError("no study found at the given path")
        try:
            study = load_study(path)
        except Exception as exc:  # the message may carry the path, so report only the type
            raise ToolError(f"could not read the study ({type(exc).__name__})") from None
        if len(study) == 0:
            raise ToolError("no image series could be read from the study")
        study_id = secrets.token_hex(6)
        self._studies[study_id] = study
        while len(self._studies) > self._max:
            old, _ = self._studies.popitem(last=False)
            self._anchors.pop(old, None)
        return {
            "study_id": study_id,
            "info": _public_info(study),
            "n_series": len(study),
            "series": series_table(study),
        }

    def list_series(self, study_id: str) -> list[dict[str, Any]]:
        """Series of an open study: key, role, geometry, display directions and z range."""
        return series_table(self._study(study_id))

    def get_anchors(self, study_id: str) -> dict[str, Any]:
        """Automatic spine anatomy (label_spine): disc labels, centres, heights, midline, conus, numbering,
        each with confidence and evidence. Computed once per study and cached."""
        return self._anchors_for(study_id).to_dict()

    # ---------------------------------------------------------- rendering
    def render_slice(
        self, study_id: str, key: str, k: int, window: str | list[float] | None = None
    ) -> RenderResult:
        """One slice of a series at display size, with orientation letters, rulers and slice labels.
        window: a preset name (for example 't2 disc/CSF') or [lo, hi]; default is the series window."""
        st = self._stack(study_id, key)
        k = self._slice(st, k)
        win = resolve_window(st, _window_arg(window), self._cfg)
        anchors = self._anchors_cached(study_id)
        frame = slice_frame(st, k)
        img, frame = display_scale(grey_rgb(st.vol[k], win), frame, min_display_px(self._cfg))
        img = add_rulers(orientation_letters(img, frame, cfg=self._cfg), frame, cfg=self._cfg)
        lines = [f"{key} k={k:02d} window {win[0]:.0f}-{win[1]:.0f}", *slice_label_lines(st, k, anchors)]
        img = with_title(img, lines, cfg=self._cfg)
        meta = {"key": key, "k": k, "window": [float(win[0]), float(win[1])], "shape": list(img.shape[:2])}
        return self._emit(study_id, f"slice_{key}_{k:02d}", png_bytes(img), meta)

    def zoom(
        self,
        study_id: str,
        key: str,
        k: int,
        center_rc: list[float],
        fov_mm: float | None = None,
        window: str | list[float] | None = None,
    ) -> RenderResult:
        """Zoomed crop around a native pixel [row, col] of slice k, with mm rulers and a scale bar.
        fov_mm is the crop side in mm (default: the configured zoom size)."""
        st = self._stack(study_id, key)
        k = self._slice(st, k)
        r, c = self._rc(st, center_rc)
        if fov_mm is not None and fov_mm <= 0:
            raise ToolError("fov_mm must be positive")
        xyz = st.pix_to_lps(k, r, c)
        win = resolve_window(st, _window_arg(window), self._cfg)
        img, _frame = zoom_crop(
            st,
            xyz,
            size_mm=fov_mm,
            k=k,
            window=win,
            anchors=self._anchors_cached(study_id),
            config=self._cfg,
        )
        meta = {
            "key": key,
            "k": k,
            "centre_rc": [r, c],
            "centre_lps_mm": [round(float(v), 2) for v in xyz],
            "fov_mm": fov_mm or self._cfg.zoom_mm,
        }
        return self._emit(study_id, f"zoom_{key}_{k:02d}", png_bytes(img), meta)

    def linked_views(
        self,
        study_id: str,
        xyz: list[float],
        crop_mm: float | None = None,
        window: str | list[float] | None = None,
    ) -> RenderResult:
        """One LPS point (mm) shown on the nearest slice of every series, with a crosshair. Returns the
        per-series records (nearest slice, signed distance, in-slab and in-FOV flags) in the metadata."""
        st = self._study(study_id)
        if len(xyz) != 3:
            raise ToolError("xyz must be [x, y, z] in LPS mm")
        p = np.asarray([float(v) for v in xyz])
        win = None if window is None else _window_arg(window)
        if isinstance(win, str):
            win = resolve_window(next(iter(st.stacks.values())), win, self._cfg)
        img, records = linked_views(
            p,
            st.stacks,
            anchors=self._anchors_cached(study_id),
            crop_mm=crop_mm,
            window=win,
            config=self._cfg,
        )
        meta = {"xyz_lps_mm": [float(v) for v in p], "views": records}
        return self._emit(study_id, "linked", png_bytes(img), meta)

    def reference_lines(
        self,
        study_id: str,
        key: str,
        k: int,
        ref_keys: list[str],
        window: str | list[float] | None = None,
    ) -> RenderResult:
        """Slice k of series ``key`` with the lines where slices of the series in ref_keys cut it (up to 12
        slices per reference series, evenly spaced)."""
        st = self._stack(study_id, key)
        k = self._slice(st, k)
        if not ref_keys:
            raise ToolError("ref_keys must name at least one other series")
        refs = []
        for rk in ref_keys:
            rs = self._stack(study_id, rk)
            n = min(rs.n, 12)
            idx = sorted({round(float(i)) for i in np.linspace(0, rs.n - 1, n)})
            refs.append((rs, idx))
        win = None if window is None else _window_arg(window)
        img = reference_lines(
            st,
            k,
            refs,
            anchors=self._anchors_cached(study_id),
            window=win,
            config=self._cfg,
        )
        return self._emit(
            study_id, f"refs_{key}_{k:02d}", png_bytes(img), {"key": key, "k": k, "refs": ref_keys}
        )

    def measure_distance(
        self, study_id: str, key: str, k: int, p0_rc: list[float], p1_rc: list[float]
    ) -> dict[str, Any]:
        """Distance in mm between two native pixels [row, col] on slice k of a series (exact LPS geometry)."""
        st = self._stack(study_id, key)
        k = self._slice(st, k)
        a = st.pix_to_lps(k, *self._rc(st, p0_rc))
        b = st.pix_to_lps(k, *self._rc(st, p1_rc))
        d = float(np.linalg.norm(b - a))
        return {
            "key": key,
            "k": k,
            "distance_mm": round(d, 2),
            "pixel_mm": [round(st.row_mm, 4), round(st.col_mm, 4)],
            "note": "distance between two pixel centres on this slice",
        }

    def roi_stats(
        self, study_id: str, key: str, k: int, center_rc: list[float], radius_mm: float
    ) -> dict[str, Any]:
        """Statistics of a circular ROI (radius in mm) around a native pixel on slice k. Raw MR signal, not
        calibrated; compare within one series only."""
        st = self._stack(study_id, key)
        k = self._slice(st, k)
        if radius_mm <= 0:
            raise ToolError("radius_mm must be positive")
        r, c = self._rc(st, center_rc)
        rows, cols = np.mgrid[0 : st.shape[1], 0 : st.shape[2]]
        dist = np.hypot((rows - r) * st.row_mm, (cols - c) * st.col_mm)
        vals = st.vol[k][dist <= radius_mm].astype(np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ToolError("the ROI contains no pixels; move the centre or enlarge the radius")
        return {
            "key": key,
            "k": k,
            "n": int(vals.size),
            "mean": round(float(vals.mean()), 3),
            "std": round(float(vals.std()), 3),
            "median": round(float(np.median(vals)), 3),
            "min": round(float(vals.min()), 3),
            "max": round(float(vals.max()), 3),
            "units": "raw MR signal (arbitrary, not calibrated)",
        }

    def reformat(
        self,
        study_id: str,
        plane: str,
        at_mm: float | None = None,
        label: str | None = None,
        window: str | list[float] | None = None,
    ) -> RenderResult:
        """Reformat onto a plane computed from the study: plane 'sag' (at x = at_mm), 'cor' (y), 'ax' (z),
        or 'disc' (the disc plane of level ``label``, needs anchors). The source is the series that covers the
        plane best. Without at_mm, sagittal uses the mid-sagittal plane and cor/ax the centre of the source."""
        if plane not in REFORMAT_PLANES:
            raise ToolError(f"plane must be one of {', '.join(REFORMAT_PLANES)}")
        st = self._study(study_id)
        candidates = [s for s in st.stacks.values() if not s.meta.get("role", {}).get("localizer")]
        if not candidates:
            raise ToolError("no non-localizer series to reformat from")
        win_arg = _window_arg(window)
        src: Stack
        ref: Reformat
        if plane == "disc":
            if not label:
                raise ToolError("plane 'disc' needs label, for example L4-L5")
            anchors = self._anchors_for(study_id)
            if label not in anchors.discs:
                raise ToolError(
                    f"no disc {label!r} in the anchors; labels: {', '.join(anchors.disc_labels())}"
                )
            centre = np.asarray(anchors.discs[label].center)
            pool = [s for s in candidates if s.contains(centre)] or candidates
            found = [(s, r) for s in pool if (r := disc_plane(s, anchors, label, cfg=self._cfg)) is not None]
            if not found:
                raise ToolError(f"no series covers disc {label!r}")
            src, ref = max(found, key=lambda sr: sr[1].fraction_inside)
            at_value: float | None = None
        else:
            if at_mm is None and plane == "sag":
                mid = self._anchors_for(study_id).midline_x
                if mid is None:
                    raise ToolError("at_mm is required for a sagittal reformat when no midline is known")
                at_mm = float(mid)
            found_r = [
                (s, reformat(s, plane, float(at_mm if at_mm is not None else _centre_along(s, plane))))
                for s in candidates
            ]
            src, ref = max(found_r, key=lambda sr: sr[1].fraction_inside)
            at_value = float(at_mm) if at_mm is not None else None
        win = (
            resolve_window(src, win_arg, self._cfg) if win_arg is not None else series_window(src, self._cfg)
        )
        img = render_reformat(ref, win, cfg=self._cfg)
        meta = {
            "plane": plane,
            "at_mm": at_value,
            "label": label,
            "source_key": src.key,
            "fraction_inside": round(ref.fraction_inside, 3),
            "eff_res_mm": [round(v, 2) for v in ref.eff_res_mm],
            "note": ref.note,
        }
        name = f"reformat_{plane}" + (f"_{label}" if label else "")
        return self._emit(study_id, name, png_bytes(img), meta)

    # ------------------------------------------------------------ reports
    def build_viewer(self, study_id: str, outdir: str) -> dict[str, Any]:
        """Write the offline HTML viewer (every series, reformats, disc lines) into outdir. Returns the file path."""
        study = self._study(study_id)
        path = build_viewer(study, outdir, anchors=self._anchors_for(study_id), config=self._cfg)
        return {"viewer": str(path)}

    def brief(self, study_id: str, reader: str = "reader_a") -> str:
        """Reading brief for an AI reader of this study: conventions, anchors, tools, calibration and the
        findings format. reader is the name the findings file must use."""
        return brief(self._study(study_id), self._anchors_for(study_id), reader=reader, config=self._cfg)

    def tasks(self, study_id: str) -> list[dict[str, Any]]:
        """Independent reader scopes (per level, per structure, holistic, and the adversarial verifier) with
        complete prompts. Run the first three concurrently; run the verifier after merge_findings."""
        tasks = reading_tasks(self._study(study_id), self._anchors_for(study_id), config=self._cfg)
        return [t.to_dict() for t in tasks]

    def validate_findings(self, findings: str) -> dict[str, Any]:
        """Validate one findings document (JSON text) against the schema. Returns ok, errors and warnings."""
        try:
            doc = json.loads(findings)
        except ValueError:
            return {"ok": False, "errors": ["not valid JSON"], "warnings": []}
        res = validate(doc)
        return {
            "ok": res.ok,
            "reader": res.reader,
            "errors": res.error_messages(),
            "warnings": list(res.warnings),
        }

    def merge_findings(
        self, paths: list[str], outdir: str | None = None, use_confidence: bool = False
    ) -> dict[str, Any]:
        """Merge findings JSON files (one per reader) into a consensus with disagreements. Returns the
        consensus and a Markdown report; with outdir, also writes consensus.json and report.md there."""
        if not paths:
            raise ToolError("give at least one findings file")
        real: list[str] = []
        for p in paths:
            if not isinstance(p, str) or not os.path.isfile(p):
                raise ToolError("a findings path is not a file")
            real.append(p)
        consensus = merge(list(real), use_confidence=use_confidence, config=self._cfg)
        names: dict[str, str] = {}
        for p in real:
            names[str(Path(p).resolve())] = Path(p).name
            names[p] = Path(p).name

        def scrub(text: str) -> str:
            for full, short in names.items():
                text = text.replace(full, short)
            return text

        out = json.loads(scrub(json.dumps(consensus.to_dict(), ensure_ascii=False)))
        md = scrub(render_markdown(consensus))
        files: dict[str, str] = {}
        if outdir is not None:
            d = Path(outdir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "consensus.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
            (d / "report.md").write_text(md, encoding="utf-8")
            files = {"consensus": str(d / "consensus.json"), "report": str(d / "report.md")}
        return {"consensus": out, "markdown": md, "files": files}


def _centre_along(stack: Stack, plane: str) -> float:
    """Coordinate of the stack centre along the normal of a canonical plane (x, y or z in LPS mm)."""
    axis = {"sag": 0, "cor": 1, "ax": 2}[plane]
    corners = np.concatenate([stack.corners(k) for k in range(stack.n)])
    return float(0.5 * (corners[:, axis].min() + corners[:, axis].max()))


# ---------------------------------------------------------------- MCP wiring

_TOOL_NAMES = (
    "open_study",
    "list_series",
    "get_anchors",
    "render_slice",
    "zoom",
    "linked_views",
    "reference_lines",
    "measure_distance",
    "roi_stats",
    "reformat",
    "build_viewer",
    "brief",
    "tasks",
    "validate_findings",
    "merge_findings",
)


def _as_content(result: Any) -> Any:
    """Turn a tool result into MCP content: RenderResult -> [image, metadata JSON]; others -> JSON text."""
    from mcp.server.fastmcp import Image

    if isinstance(result, RenderResult):
        meta = dict(result.meta)
        if result.saved_to:
            meta["saved_to"] = result.saved_to
        return [Image(data=result.png, format="png"), json.dumps(meta, ensure_ascii=False)]
    return result


def build_server(tools: SpineTools | None = None) -> Any:
    """A FastMCP server with every spinescope tool registered (needs the ``mcp`` package)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('the MCP server needs the mcp extra: pip install "spinescope[mcp]"') from exc

    impl = tools or SpineTools()
    server = FastMCP(
        "spinescope",
        instructions=(
            "Spine MRI tools for research and education (not diagnosis). Open a study with open_study, then "
            "use brief() before reading, get_anchors() for levels, render and measure tools for images and "
            "numbers, and merge_findings() for reader consensus. Results never contain patient identifiers "
            "or file paths; the study_id is the only handle."
        ),
    )
    for name in _TOOL_NAMES:
        method: Callable[..., Any] = getattr(impl, name)

        def adapter(*args: Any, _m: Callable[..., Any] = method, **kwargs: Any) -> Any:
            return _as_content(_m(*args, **kwargs))

        functools.update_wrapper(adapter, method)
        adapter.__name__ = name
        server.add_tool(
            adapter,
            name=name,
            description=(method.__doc__ or name).strip(),
            structured_output=False,
        )
    return server


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server on stdio. ``--output-dir DIR`` also saves every rendered PNG under DIR."""
    import argparse

    p = argparse.ArgumentParser(prog="spinescope-mcp", description="spinescope MCP server (stdio)")
    p.add_argument("--output-dir", help="also save rendered PNGs here (outputs only, never inputs)")
    p.add_argument("--max-studies", type=int, default=4, help="studies kept open at once")
    args = p.parse_args(argv)
    tools = SpineTools(max_studies=args.max_studies, output_dir=args.output_dir)
    build_server(tools).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
