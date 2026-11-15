"""Command-line interface: ``spinescope <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from . import __version__

if TYPE_CHECKING:
    from .anatomy.anchors import Anchors
    from .study import Study


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .study import load_study

    study = load_study(args.path, load_pixels=False)
    if args.json:
        out = {
            "info": study.info,
            "stacks": {
                k: {
                    "n": s.n,
                    "rows": s.shape[1],
                    "cols": s.shape[2],
                    "pixel_mm": [s.row_mm, s.col_mm],
                    "thickness": s.thickness,
                    "spacing": s.spacing,
                    "plane": s.plane,
                    "obliquity_deg": round(s.obliquity, 2),
                    "description": s.description,
                    "role": s.meta.get("role", {}),
                }
                for k, s in study.stacks.items()
            },
        }
        json.dump(out, sys.stdout, indent=1, default=str)
        print()
    else:
        print(study.summary())
    return 0


def _load(path: str, *, anchors: bool) -> tuple[Study, Anchors | None]:
    from .study import load_study

    study = load_study(path)
    if not anchors:
        return study, None
    from .anatomy import label_spine

    return study, label_spine(study)


def _cmd_brief(args: argparse.Namespace) -> int:
    from .ai.briefs import brief

    study, anchors = _load(args.path, anchors=not args.no_anchors)
    text = brief(study, anchors, reader=args.reader)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .ai.briefs import reading_tasks

    study, anchors = _load(args.path, anchors=not args.no_anchors)
    tasks = reading_tasks(study, anchors)
    if args.outdir:
        out = Path(args.outdir)
        (out / "prompts").mkdir(parents=True, exist_ok=True)
        for t in tasks:
            (out / "prompts" / f"{t.reader}.txt").write_text(t.prompt, encoding="utf-8")
        index = [{k: v for k, v in t.to_dict().items() if k != "prompt"} for t in tasks]
        (out / "tasks.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"{len(tasks)} reader tasks: {out / 'tasks.json'} and {out / 'prompts'}/")
    else:
        json.dump([t.to_dict() for t in tasks], sys.stdout, indent=1)
        print()
    return 0


def _cmd_viewer(args: argparse.Namespace) -> int:
    from .viewer import build_viewer

    study, anchors = _load(args.path, anchors=not args.no_anchors)
    path = build_viewer(study, args.outdir, anchors=anchors, title=args.title)
    print(path)
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from .ai.mcp_server import main as mcp_main
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(f"spinescope mcp needs the mcp extra: {exc}", file=sys.stderr)
        return 1
    argv: list[str] = []
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    return mcp_main(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spinescope", description=__doc__)
    p.add_argument("--version", action="version", version=f"spinescope {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inspect", help="list the series of a study and how they were classified")
    s.add_argument("path", help="DICOM directory (any layout) or a NIfTI/MetaImage file")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=_cmd_inspect)

    s = sub.add_parser("brief", help="reading brief for an AI reader (conventions, anchors, findings format)")
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("--reader", default="reader_a", help="reader name the findings file must use")
    s.add_argument("--no-anchors", action="store_true", help="skip automatic level labelling (faster)")
    s.add_argument("--out", help="write the brief to this file instead of stdout")
    s.set_defaults(func=_cmd_brief)

    s = sub.add_parser("tasks", help="independent reader scopes (per level, structure, holistic, verifier)")
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument(
        "--no-anchors", action="store_true", help="skip automatic level labelling (no level tasks)"
    )
    s.add_argument("--outdir", help="write tasks.json and prompts/<reader>.txt here instead of printing JSON")
    s.set_defaults(func=_cmd_tasks)

    s = sub.add_parser("viewer", help="build the offline HTML viewer")
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("outdir", help="output directory for spinescope_viewer.html")
    s.add_argument("--title", help="page title (keep it free of patient information)")
    s.add_argument("--no-anchors", action="store_true", help="skip automatic level labelling")
    s.set_defaults(func=_cmd_viewer)

    s = sub.add_parser("mcp", help="run the MCP server on stdio (for AI assistants)")
    s.add_argument("--output-dir", help="also save rendered PNGs here")
    s.set_defaults(func=_cmd_mcp)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
