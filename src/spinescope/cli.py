"""Command-line interface: ``spinescope <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spinescope", description=__doc__)
    p.add_argument("--version", action="version", version=f"spinescope {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inspect", help="list the series of a study and how they were classified")
    s.add_argument("path", help="DICOM directory (any layout) or a NIfTI/MetaImage file")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=_cmd_inspect)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
