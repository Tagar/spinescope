"""Command-line interface: ``spinescope <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from . import __version__

if TYPE_CHECKING:
    from .anatomy.anchors import Anchors
    from .study import Study


def _cmd_inspect(args: argparse.Namespace) -> int:
    from .privacy import public_text
    from .study import load_study

    if getattr(args, "label", False):
        return _inspect_with_labels(args)
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
                    "description": public_text(s.description),
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


def _inspect_with_labels(args: argparse.Namespace) -> int:
    """``inspect --label``: the series table plus the one-line labelling summary (labels are computed)."""
    from .anatomy import label_spine
    from .anatomy.summary import labelling_summary, one_line
    from .study import load_study

    study = load_study(args.path)
    anchors = label_spine(study)
    if args.json:
        out = {
            "info": study.info,
            "stacks": {
                k: {"role": s.meta.get("role", {}), "plane": s.plane, "slices": s.n}
                for k, s in study.stacks.items()
            },
            "labelling": labelling_summary(anchors),
        }
        json.dump(out, sys.stdout, indent=1, default=str)
        print()
    else:
        print(study.summary())
        print(one_line(anchors))
    return 0


def _label_table(anchors: Anchors) -> str:
    """Plain-text report of a labelling run: the status banner, the applicability, the levels table, warnings."""
    from .anatomy.summary import labelling_summary, summary_lines

    s = labelling_summary(anchors)
    lines = summary_lines(anchors)
    lines.append("")
    if s["levels"]:
        head = (
            "Disc levels (level names are unconfirmed: positions only)"
            if s["status"] == "unconfirmed"
            else ("Disc levels (level names to be confirmed)" if s["status"] == "probable" else "Disc levels")
        )
        lines.append(head + ", superior to inferior:")
        rows = [("Level", "z (mm)", "Height (mm)", "Confidence", "Tier")]
        for r in s["levels"]:
            h = "n/a" if r["height_mm"] is None else f"{r['height_mm']:g}"
            rows.append((r["level"], f"{r['z_mm']:+.1f}", h, f"{r['confidence']:.2f}", r["tier"]))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        for row in rows:
            lines.append("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    else:
        lines.append("No disc levels were labelled.")
    lines.append("")
    if s["warnings"]:
        lines.append("Warnings:")
        lines += [f"  - {w}" for w in s["warnings"]]
    else:
        lines.append("Warnings: none")
    return "\n".join(lines)


def _cmd_label(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .anatomy import label_spine
    from .anatomy.qa import overlay
    from .render.window import save_png
    from .study import load_study

    study = load_study(args.path)
    if getattr(args, "anchors", None):  # a corrected file: same as apply-labels
        return _review_and_report(args, study, args.anchors)
    anchors = label_spine(study, backend=args.backend)
    doc = anchors.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    if args.overlay:
        save_png(overlay(study, anchors), args.overlay)
    if args.json:
        json.dump(doc, sys.stdout, indent=1, default=str)
        print()
    else:
        print(_label_table(anchors))
        if args.out:
            print(f"Anchors JSON written to {args.out}")
        if args.overlay:
            print(f"QA overlay written to {args.overlay}")
    return 0


def _load(path: str, *, anchors: bool, anchors_file: str | None = None) -> tuple[Study, Anchors | None]:
    from .study import load_study

    study = load_study(path)
    if (
        anchors_file
    ):  # a corrected or verified file: checked against this study (refused if it is another one)
        from .anatomy.review import load_anchors_file

        return study, load_anchors_file(study, anchors_file, path=path)
    if not anchors:
        return study, None
    from .anatomy import label_spine

    return study, label_spine(study)


def _cmd_brief(args: argparse.Namespace) -> int:
    from .ai.briefs import brief

    study, anchors = _load(args.path, anchors=not args.no_anchors, anchors_file=args.anchors)
    synthesis = None
    if args.analyses:
        from .analysis.base import run_all
        from .report.synthesis import build_synthesis

        if anchors is None:
            raise ValueError("--analyses needs labelled anchors (drop --no-anchors)")
        synthesis = build_synthesis(study, anchors, run_all(study, anchors))
    text = brief(study, anchors, reader=args.reader, synthesis=synthesis)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


def _measured(study: Study, anchors: Anchors | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measured findings per level and the analysis results they come from (empty without disc anchors)."""
    from .ai.reading import measured_findings
    from .analysis.base import run_all

    if anchors is None or not anchors.discs:
        return {}, {}
    results = run_all(study, anchors)
    return measured_findings(results, anchors), results


def _cmd_tasks(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .ai.briefs import reading_tasks

    study, anchors = _load(args.path, anchors=not args.no_anchors, anchors_file=args.anchors)
    measured, _ = _measured(study, anchors)
    tasks = reading_tasks(study, anchors, measured=measured)
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


def _cmd_findings(args: argparse.Namespace) -> int:
    """Measured findings with validation status (JSON), and with --outdir a reading sheet per measured field."""
    from pathlib import Path

    from .ai.reading import REVIEW_POLICY, render_sheet, sheet_file_name
    from .anatomy.summary import status_of
    from .render.reading_sheet import save_sheet

    study, anchors = _load(args.path, anchors=not args.no_anchors, anchors_file=args.anchors)
    measured, results = _measured(study, anchors)
    doc = {
        "numbering": status_of(anchors) if anchors is not None else None,
        "review_policy": REVIEW_POLICY,
        "levels": measured,
        "analyses": {n: list(r.warnings) for n, r in sorted(results.items()) if r.warnings},
    }
    if args.outdir:
        out = Path(args.outdir)
        (out / "sheets").mkdir(parents=True, exist_ok=True)
        for lab, obj in measured.items():
            for group, body in obj.items():
                if group.startswith("_") or anchors is None:
                    continue
                for name in body:
                    img = render_sheet(study, anchors, measured, lab, group, name, show_value=not args.blind)
                    save_sheet(img, out / "sheets" / sheet_file_name(lab, group, name))
        (out / "findings.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"findings: {out / 'findings.json'} and {out / 'sheets'}/")
    else:
        print(json.dumps(doc, indent=1, ensure_ascii=False))
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """Validate a reader's review (confirm, reject or unclear per measured field) and convert it to a findings file."""
    import json as _json
    from pathlib import Path

    from .ai.reading import review_errors, review_to_findings, write_review_outputs

    study, anchors = _load(args.path, anchors=True, anchors_file=args.anchors)
    measured, _ = _measured(study, anchors)
    doc = _json.loads(Path(args.answer).read_text(encoding="utf-8"))
    from .anatomy.summary import status_of

    if anchors is None:
        print("error: the review needs the study's level labels (anchors)", file=sys.stderr)
        return 1
    errs = review_errors(doc, measured, status=status_of(anchors))
    if errs:
        for e in errs:
            print(f"error: {e}", file=sys.stderr)
        return 1
    if args.outdir:
        print(write_review_outputs(doc, measured, args.outdir))
    else:
        print(json.dumps(review_to_findings(doc, measured), indent=1, ensure_ascii=False))
    return 0


def _cmd_viewer(args: argparse.Namespace) -> int:
    from .viewer import build_viewer

    study, anchors = _load(args.path, anchors=not args.no_anchors, anchors_file=args.anchors)
    findings: Mapping[str, Any] | None = None
    notes: list[str] = []
    if args.findings is not None:
        if anchors is None:
            print("viewer --findings needs the level labels: drop --no-anchors", file=sys.stderr)
            return 2
        if args.findings:  # a findings file written by `spinescope analyze --out`
            from .viewer.findings_view import load_findings_file

            findings, notes = load_findings_file(args.findings, anchors)
        else:  # no file: run every registered analysis on these labels
            from .analysis.base import run_all

            findings = dict(run_all(study, anchors))
    path = build_viewer(
        study, args.outdir, anchors=anchors, title=args.title, findings=findings, findings_notes=notes
    )
    print(path)
    return 0


def _cmd_verify_labels(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .anatomy import label_spine
    from .anatomy.verify import AnswerError, apply_answer, build_packet
    from .study import load_study

    study = load_study(args.path)
    auto = label_spine(study)
    if args.answer is not None:
        try:
            text = Path(args.answer).read_text(encoding="utf-8")
        except OSError:
            print("cannot read the answer file", file=sys.stderr)
            return 2
        try:
            anchors = apply_answer(auto, text, who=args.who, source=args.source)
        except AnswerError as exc:
            print(f"answer rejected: {exc}", file=sys.stderr)
            return 2
        out = json.dumps(anchors.to_dict(), indent=1)
        if args.out:
            Path(args.out).write_text(out + "\n", encoding="utf-8")
            print(args.out)
        else:
            sys.stdout.write(out + "\n")
        return 0
    if not args.outdir:
        print(
            "give --outdir to write the reader packet, or --answer to apply a reader's answer",
            file=sys.stderr,
        )
        return 2
    paths = build_packet(study, auto).write(args.outdir)
    print(
        f"{len(paths)} files written to {args.outdir}: packet.json (question, answer format) and PNG images"
    )
    return 0


def _review_and_report(args: argparse.Namespace, study: Study, anchors_file: str) -> int:
    """Apply a corrected anchors file (viewer export) to ``study``: checks, provenance, status by policy; print it."""
    from pathlib import Path

    from .anatomy.review import ReviewError, apply_review

    try:
        anchors = apply_review(study, anchors_file, who=args.who, why=args.why, path=args.path)
    except ReviewError as exc:
        print(f"anchors refused: {exc}", file=sys.stderr)
        return 2
    doc = anchors.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    if getattr(args, "json", False):
        json.dump(doc, sys.stdout, indent=1, default=str)
        print()
    else:
        print(_label_table(anchors))
        if args.out:
            print(f"Reviewed anchors JSON written to {args.out}")
    return 0


def _cmd_apply_labels(args: argparse.Namespace) -> int:
    """``apply-labels STUDY CORRECTED --out FILE``: validate a viewer-corrected file against the study and set status."""
    from .study import load_study

    study = load_study(args.path)
    return _review_and_report(args, study, args.corrected)


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


def _cmd_benchmark(argv: Sequence[str]) -> int:
    from .benchmark import main as benchmark_main

    return benchmark_main(argv)


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Run the registered analyses on the labelled levels; JSON to stdout, figures and CSVs to --out.

    With ``--report`` (and ``--out``) also writes report.md, report.html and findings.json (the synthesis) there.
    """
    import os
    from pathlib import Path

    from .analysis.base import Result, available, run
    from .anatomy import label_spine
    from .study import load_study

    names = available() if not args.only else [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = sorted(set(names) - set(available()))
    if unknown:
        raise ValueError(f"unknown analyses: {', '.join(unknown)}; available: {', '.join(available())}")
    if args.report and not args.out:
        raise ValueError("--report writes into the --out folder; give --out DIR")
    study = load_study(args.path)
    anchors = label_spine(study)
    out: dict[str, object] = {
        "numbering": anchors.numbering.get("status", ""),
        "levels": anchors.disc_labels(),
    }
    from .report.synthesis import build_synthesis
    from .report.synthesis_render import write_bundle

    results: dict[str, Result] = {name: run(name, study, anchors) for name in names}
    entries: dict[str, object] = {}
    figure_names: dict[str, dict[str, str]] = {}  # analysis -> figure -> file name (no folder)
    for name, res in results.items():
        entry = res.to_dict()
        if args.out:
            saved = res.save(args.out)
            entry["saved"] = [os.path.basename(p) for p in saved]  # names only: never the output folder
            figure_names[name] = {fig: f"{name}_{fig}.png" for fig in res.figures}
        entries[name] = entry
    out["analyses"] = entries
    syn = build_synthesis(study, anchors, results)
    out["synthesis"] = syn.to_dict()
    if args.report:
        from .report.synthesis import findings_document

        doc = findings_document(results, anchors, figure_names)
        folder = Path(args.out)
        paths = write_bundle(
            syn,
            doc,
            folder,
            figure_files={
                f"{a}:{f}": folder / n for a, figs in figure_names.items() for f, n in figs.items()
            },
        )
        out["report_files"] = [p.name for p in paths]  # names only
    json.dump(out, sys.stdout, indent=1, default=str)  # Result.to_dict() has already mapped NaN to null
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spinescope", description=__doc__)
    p.add_argument("--version", action="version", version=f"spinescope {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inspect", help="list the series of a study and how they were classified")
    s.add_argument("path", help="DICOM directory (any layout) or a NIfTI/MetaImage file")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.add_argument(
        "--label", action="store_true", help="also label the levels and print a one-line labelling summary"
    )
    s.set_defaults(func=_cmd_inspect)

    s = sub.add_parser(
        "label",
        help="label the disc levels: status, applicability, warnings; write anchors JSON and a QA overlay",
    )
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("--json", action="store_true", help="print the anchors JSON instead of the table")
    s.add_argument("--out", help="write the anchors JSON to this file")
    s.add_argument("--overlay", help="write the QA overlay (mid-sagittal discs, confidences) as a PNG here")
    s.add_argument(
        "--backend",
        choices=("auto", "ml", "classical"),
        default="auto",
        help="labeller: auto (learned if available, else classical), ml (learned) or classical",
    )
    s.add_argument(
        "--anchors",
        help="apply a corrected anchors JSON (exported from the viewer) instead of labelling; see apply-labels",
    )
    s.add_argument(
        "--who", default="reviewer", help="with --anchors: reviewer name for the provenance record"
    )
    s.add_argument(
        "--why", help="with --anchors: reason for the review (default: the reason saved in the file)"
    )
    s.set_defaults(func=_cmd_label)

    s = sub.add_parser(
        "apply-labels",
        help="check a corrected anchors JSON against the study (schema, study fingerprint, no paths), add the "
        "review provenance and set the status by policy (docs/guide/correcting-labels.md)",
    )
    s.add_argument("path", help="DICOM directory or volume file (the study the labels were made from)")
    s.add_argument("corrected", help="anchors JSON exported from the viewer, or corrected by hand")
    s.add_argument("--out", help="write the reviewed anchors JSON here")
    s.add_argument("--who", default="reviewer", help="reviewer name or initials, saved in the provenance")
    s.add_argument(
        "--why", help="reason for the review, saved in the provenance (default: the file's own reason)"
    )
    s.add_argument("--json", action="store_true", help="print the reviewed anchors JSON instead of the table")
    s.set_defaults(func=_cmd_apply_labels)

    s = sub.add_parser("brief", help="reading brief for an AI reader (conventions, anchors, findings format)")
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("--reader", default="reader_a", help="reader name the findings file must use")
    s.add_argument("--no-anchors", action="store_true", help="skip automatic level labelling (faster)")
    s.add_argument(
        "--analyses",
        action="store_true",
        help="also run the registered analyses and add their synthesis (validation status, not assessed) to the brief",
    )
    s.add_argument(
        "--anchors", help="use this anchors JSON (e.g. corrected by verify-labels) instead of labelling"
    )
    s.add_argument("--out", help="write the brief to this file instead of stdout")
    s.set_defaults(func=_cmd_brief)

    s = sub.add_parser("tasks", help="independent reader scopes (per level, structure, holistic, verifier)")
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument(
        "--no-anchors", action="store_true", help="skip automatic level labelling (no level tasks)"
    )
    s.add_argument(
        "--anchors", help="use this anchors JSON (e.g. corrected by verify-labels) instead of labelling"
    )
    s.add_argument("--outdir", help="write tasks.json and prompts/<reader>.txt here instead of printing JSON")
    s.set_defaults(func=_cmd_tasks)

    s = sub.add_parser(
        "findings", help="measured findings with validation status; --outdir also writes reading sheets"
    )
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("--no-anchors", action="store_true", help="skip automatic level labelling (no findings)")
    s.add_argument("--anchors", help="use this anchors JSON instead of labelling")
    s.add_argument("--outdir", help="write findings.json and sheets/<level>_<field>.png here")
    s.add_argument(
        "--blind", action="store_true", help="sheets without the measured value (for a blind read)"
    )
    s.set_defaults(func=_cmd_findings)

    s = sub.add_parser(
        "review", help="validate a reader's review (confirm/reject/unclear) and convert it to findings"
    )
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("answer", help="the reader's review JSON")
    s.add_argument("--anchors", help="the anchors JSON the review was made against")
    s.add_argument("--outdir", help="write <reader>.json here instead of printing the converted findings")
    s.set_defaults(func=_cmd_review)

    s = sub.add_parser("viewer", help="build the offline HTML viewer")
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("outdir", help="output directory for spinescope_viewer.html")
    s.add_argument("--title", help="page title (keep it free of patient information)")
    s.add_argument("--no-anchors", action="store_true", help="skip automatic level labelling")
    s.add_argument(
        "--anchors", help="use this anchors JSON (e.g. corrected by verify-labels) instead of labelling"
    )
    s.add_argument(
        "--findings",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="add the findings panel: run the registered analyses on the labels, or with FILE load the JSON that "
        "`spinescope analyze --out` wrote (PHI-free fields only)",
    )
    s.set_defaults(func=_cmd_viewer)

    s = sub.add_parser(
        "verify-labels",
        help="check the level labels with a reader: write the packet, or apply the reader's answer",
    )
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("--outdir", help="write the reader packet (PNGs and packet.json) here")
    s.add_argument(
        "--answer", help="JSON answer file from the reader: apply it and print the corrected anchors"
    )
    s.add_argument("--out", help="with --answer: write the corrected anchors JSON to this file")
    s.add_argument("--who", default="reader", help="who gave the answer, recorded in the provenance")
    s.add_argument(
        "--source", choices=("ai_reader", "human"), default="ai_reader", help="kind of reader, for confidence"
    )
    s.set_defaults(func=_cmd_verify_labels)

    s = sub.add_parser(
        "analyze",
        help="run the registered analyses (discs, endplates, ...) on the labelled levels: JSON to stdout, "
        "figures and CSV/JSON files to --out",
    )
    s.add_argument("path", help="DICOM directory or volume file")
    s.add_argument("--only", help="comma-separated analysis names (default: all registered)")
    s.add_argument("--out", help="write each analysis' CSV, JSON and figure PNGs here")
    s.add_argument(
        "--report",
        action="store_true",
        help="with --out: also write report.md, report.html (self-contained) and findings.json (synthesis) there",
    )
    s.set_defaults(func=_cmd_analyze)

    s = sub.add_parser("mcp", help="run the MCP server on stdio (for AI assistants)")
    s.add_argument("--output-dir", help="also save rendered PNGs here")
    s.set_defaults(func=_cmd_mcp)

    s = sub.add_parser(
        "benchmark",
        add_help=False,
        help="run every labelling benchmark and check them against the golden file (--check, --update-golden; "
        "see docs/benchmarks.md)",
    )
    s.set_defaults(func=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:1] == ["benchmark"]:  # forwarded verbatim: its own options, help and exit codes
        return _cmd_benchmark(raw[1:])
    args = build_parser().parse_args(raw)
    from .anatomy.review import ReviewError

    path = getattr(args, "path", None)

    def _scrub(text: str) -> str:
        return text.replace(str(path), "<input path>") if path else text

    try:
        return int(args.func(args))
    except ReviewError as exc:  # a corrected file that does not belong to this study, or is malformed
        print(f"anchors refused: {_scrub(str(exc))}", file=sys.stderr)
        return 2
    except Exception as exc:  # errors may quote the input path: show the type and the message without it
        print(f"spinescope: {type(exc).__name__}: {_scrub(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
