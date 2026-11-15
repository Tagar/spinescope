"""Reading briefs and reader scopes for AI readers, generated from the loaded study and its anchors.

Everything here is computed: the series table, the display conventions (which way the slice index runs,
which side of the screen is which patient direction, where the mid-sagittal plane sits), the level list
with the automatic confidences, and the reader scopes that an orchestrator fans out. Nothing is tuned to
one study. Outputs carry only technical DICOM attributes (``spinescope.io.KEEP_TAGS``), never file paths.

Research and education only. A brief tells a reader how to look at the images and how to write findings;
it does not make a diagnosis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..anatomy.anchors import Anchors
from ..anatomy.midline import pols_for, select_midline
from ..anatomy.summary import (
    ABSTAIN_POLICY,
    applicability_line,
    conus_line,
    fov_line,
    labelling_summary,
    status_of,
)
from ..config import DEFAULT, Config
from ..geometry import Stack
from ..io.dicom import KEEP_TAGS
from ..render.frame import direction_letter
from ..report.instructions import reader_instructions
from ..report.schema import ITEM_SECTIONS, global_group_names, level_group_names
from ..study import Study

log = logging.getLogger(__name__)

#: Reader names: the findings file name and the ``reader`` key. Same rule as the findings schema.
READER_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

#: Patient direction words for the letters of :func:`spinescope.render.frame.direction_letter`.
_DIRECTION_WORDS = {
    "L": "patient left",
    "R": "patient right",
    "A": "anterior",
    "P": "posterior",
    "S": "superior (head)",
    "I": "inferior (feet)",
}

#: Level findings grouped into the structures a reader can take on separately. Every level group of the
#: findings schema appears in exactly one family (checked in the tests), so a new schema group cannot be
#: silently left unread.
STRUCTURE_FAMILIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "disc",
        "Disc morphology: degeneration (Pfirrmann), height, contour, herniation, annular fissure, HIZ, migration.",
        ("disc",),
    ),
    (
        "central_canal",
        "Central canal and thecal sac: stenosis grade and CSF area at each level.",
        ("central_canal",),
    ),
    (
        "lateral_recess",
        "Lateral recesses, both sides.",
        ("lateral_recess_R", "lateral_recess_L"),
    ),
    (
        "foramina_and_roots",
        "Neural foramina (Lee grade) and the exiting nerve roots, both sides.",
        ("foramen_R", "foramen_L", "nerve_root_R", "nerve_root_L"),
    ),
    (
        "posterior_elements",
        "Facet joints (arthropathy, effusion) and ligamentum flavum, both sides.",
        ("facet_R", "facet_L", "ligamentum_flavum"),
    ),
    (
        "endplates",
        "Vertebral endplates above and below each disc: Modic type.",
        ("endplate_above", "endplate_below"),
    ),
    ("listhesis", "Segmental listhesis: direction, Meyerding grade, translation.", ("listhesis",)),
)

#: Calibration guidance for readers. Literature-based, kept qualitative on purpose (exact prevalences
#: depend on age, sex, population and grading scheme; check the sources before quoting numbers).
CALIBRATION_GUIDANCE = """Calibration: what these findings mean on MRI
- Most degenerative findings are common in people without back pain and rise steeply with age: disc
  desiccation and Pfirrmann grade 3 or more, disc bulges, annular fissures and high-intensity zones, Modic
  changes, facet arthropathy and ligamentum flavum thickening. Meta-analyses of asymptomatic volunteers
  (for example Brinjikji et al., AJNR 2015; Jensen et al., NEJM 1994) show how common they are at each age.
  A finding being present is therefore weak evidence that it explains symptoms. Describe it; do not
  headline it.
- Relatively specific findings: a canal or lateral recess narrowed enough to compress the thecal sac or a
  nerve root, a focal extrusion or sequestration in contact with a root, unstable or progressive listhesis,
  and Modic type 1 signal on a clearly bright STIR or fat-suppressed T2 (Modic type 1 is not specific either;
  describe it with its endplate and level).
- Relatively non-specific findings: a broad bulge without compression, Pfirrmann grade alone, an isolated
  annular high-intensity zone, small facet effusions, mild foraminal narrowing, and small Schmorl nodes.
- Confidence: low for single-slice, blurred, oblique-only or partly covered levels; medium when two
  sequences or two planes agree; high only when the finding is seen on two planes or two sequences with
  a clear boundary. Say "indeterminate" (value null) rather than guess.
- Do not infer a level from a single slice. Count the discs from the sacrum upwards (or check the labels
  against the vertebral bodies) before you call a level.
- Measure before you grade whenever a number is asked for (mm, mm2): use spinescope measurements or the
  distance tool, not the eye.
- Incidental findings (for example a renal cyst at the edge of the field) belong in incidental_findings,
  not in level fields. Red-flag patterns (fracture, infection, tumour, cord signal change) are reported as
  observations with low-to-medium confidence and an explicit note to the reader that a clinician must review.
"""

#: Tools and images a reader may use, by interface. Names match the MCP server and the CLI.
_TOOL_LINES = (
    "Command line (text only; no rendering): spinescope inspect STUDY --json, spinescope label STUDY --json, "
    "spinescope brief STUDY, spinescope tasks STUDY, spinescope verify-labels STUDY --outdir DIR (packet) and "
    "--answer FILE (apply a reader's answer), spinescope viewer STUDY OUTDIR. STUDY is the DICOM directory or volume "
    "file you were given; never copy it into the output.",
    "Images and measurements come from the MCP server spinescope-mcp (when connected): open_study, list_series, "
    "get_anchors (levels, status, applicability), verify_labels() and apply_label_verification(answer_json) "
    "(check the level names before using them), render_slice(key, k, window), zoom(key, k, center_rc, fov_mm), linked_views(xyz), "
    "reference_lines(key, k, ref_keys), measure_distance(key, k, p0_rc, p1_rc), roi_stats(key, k, center_rc, "
    "radius_mm), reformat(plane, at_mm, label), build_viewer(outdir), brief(), tasks(), validate_findings(json), "
    "merge_findings(paths). Without it you cannot look at the images, so do not fill a field from the text alone.",
    "Evidence: write each PNG you looked at to your output directory and cite it in the evidence list (image path, "
    "series key, slice indices). An image you did not look at is not evidence.",
)


@dataclass(frozen=True)
class ReadingTask:
    """One independent reader scope. ``prompt`` is the complete text to hand to a reader agent."""

    id: str
    kind: str  # "level" | "structure" | "holistic" | "verify"
    title: str
    scope: str
    levels: tuple[str, ...]
    groups: tuple[str, ...]
    series: tuple[str, ...]
    reader: str
    depends_on: tuple[str, ...]
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict (tuples as lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


# ------------------------------------------------------------------ computed facts


def series_table(study: Study) -> list[dict[str, Any]]:
    """One technical row per stack: role, geometry and display letters. No identifiers."""
    rows: list[dict[str, Any]] = []
    for key in study:
        st = study[key]
        role = st.meta.get("role", {})
        dicom = st.meta.get("dicom", {})
        desc = dicom.get("SeriesDescription")
        rows.append(
            {
                "key": key,
                "description": "" if desc is None else str(desc),
                "plane": st.plane,
                "contrast": role.get("contrast", "unknown"),
                "fat_suppressed": bool(role.get("fat_suppressed")),
                "post_contrast": bool(role.get("post_contrast")),
                "localizer": bool(role.get("localizer")),
                "slab_index": role.get("slab_index"),
                "slices": st.n,
                "rows": st.shape[1],
                "cols": st.shape[2],
                "pixel_mm": [round(st.row_mm, 3), round(st.col_mm, 3)],
                "thickness_mm": round(st.thickness, 2),
                "spacing_mm": round(st.spacing, 2),
                "fov_mm": [round(v, 1) for v in st.extent_mm()],
                "obliquity_deg": round(st.obliquity, 1),
                "screen_right": _DIRECTION_WORDS[direction_letter(st.row_dir)],
                "screen_down": _DIRECTION_WORDS[direction_letter(st.col_dir)],
                "slice_index_toward": _DIRECTION_WORDS[direction_letter(st.normal)],
                "z_range_mm": _z_range(st),
            }
        )
    return rows


def _z_range(st: Stack) -> list[float]:
    zs = np.concatenate([st.corners(k)[:, 2] for k in range(st.n)])
    return [round(float(zs.min()), 1), round(float(zs.max()), 1)]


def _sagittal_midline(st: Stack, anchors: Anchors | None, cfg: Config) -> tuple[float | None, str]:
    """Fractional slice index of the mid-sagittal plane on a sagittal stack, and where it came from."""
    if st.plane != "sag":
        return None, "not a sagittal stack"
    if anchors is not None and anchors.midline_x is not None and abs(float(st.normal[0])) >= 0.95:
        k = float(st.slice_index(np.array([anchors.midline_x, 0.0, 0.0])))
        if 0.0 <= k <= st.n - 1:
            return k, "anchors.midline_x"
    mid = select_midline(st, pols_for(st), cfg)
    if mid is None:
        return None, "canal not found on this stack"
    return float(mid.slice_index), "canal ridge (select_midline)"


def geometry_facts(study: Study, anchors: Anchors | None = None, cfg: Config = DEFAULT) -> dict[str, Any]:
    """Display conventions and slice-index directions, computed from each stack's geometry."""
    stacks: dict[str, Any] = {}
    for key in study:
        st = study[key]
        entry: dict[str, Any] = {
            "plane": st.plane,
            "screen_right": _DIRECTION_WORDS[direction_letter(st.row_dir)],
            "screen_down": _DIRECTION_WORDS[direction_letter(st.col_dir)],
            "slice_index_toward": _DIRECTION_WORDS[direction_letter(st.normal)],
            "slices": st.n,
        }
        if st.plane == "sag":
            k, source = _sagittal_midline(st, anchors, cfg)
            entry["midline_slice"] = None if k is None else round(k, 2)
            entry["midline_source"] = source
        stacks[key] = entry
    return {
        "display": "sagittal: anterior on screen left, head up; axial and coronal: patient left on screen right",
        "stacks": stacks,
    }


def _level_rows(anchors: Anchors | None) -> list[dict[str, Any]]:
    if anchors is None:
        return []
    rows = []
    for lab in anchors.disc_labels():
        d = anchors.discs[lab]
        rows.append(
            {
                "level": lab,
                "z_mm": round(float(d.center[2]), 1),
                "height_mm": None if d.height_mm is None else round(float(d.height_mm), 1),
                "confidence": round(float(d.confidence), 2),
            }
        )
    return rows


# ---------------------------------------------------------------- brief text


def _fmt_series_line(r: dict[str, Any]) -> str:
    flags = []
    if r["localizer"]:
        flags.append("localizer")
    if r["slab_index"] is not None:
        flags.append(f"slab {r['slab_index']}")
    if r["fat_suppressed"]:
        flags.append("fat-sat")
    if r["post_contrast"]:
        flags.append("post-contrast")
    flag_txt = f" ({', '.join(flags)})" if flags else ""
    desc = f' "{r["description"]}"' if r["description"] else ""
    return (
        f"  {r['key']}{desc}: {r['plane']} {r['contrast']}{flag_txt}, {r['slices']} slices, "
        f"{r['rows']}x{r['cols']} px at {r['pixel_mm'][0]:g}x{r['pixel_mm'][1]:g} mm, "
        f"thickness {r['thickness_mm']:g} mm, spacing {r['spacing_mm']:g} mm, "
        f"FOV {r['fov_mm'][0]:g}x{r['fov_mm'][1]:g} mm, obliquity {r['obliquity_deg']:g} deg, "
        f"z {r['z_range_mm'][0]:g}..{r['z_range_mm'][1]:g} mm"
    )


def _geometry_lines(facts: dict[str, Any]) -> list[str]:
    lines = [f"  {facts['display']}."]
    for key, e in facts["stacks"].items():
        line = (
            f"  {key}: screen right = {e['screen_right']}; screen down = {e['screen_down']}; "
            f"slice index k increases toward {e['slice_index_toward']}"
        )
        if e["plane"] == "sag":
            mid = "not located" if e["midline_slice"] is None else f"k={e['midline_slice']:g}"
            line += f"; mid-sagittal plane at {mid} ({e['midline_source']})"
        lines.append(line)
    return lines


def naming_policy(anchors: Anchors | None) -> str | None:
    """The abstain policy for the level names of this study, as one sentence; None without anchors."""
    if anchors is None or not anchors.discs:
        return None
    status = status_of(anchors)
    return f"Level names are {status.upper()}: {ABSTAIN_POLICY[status]}."


def _status_block(anchors: Anchors) -> list[str]:
    """Numbering status, abstain policy, applicability, field of view and conus: the labelling facts a reader needs."""
    s = labelling_summary(anchors)
    lines = [
        f"  Numbering status: {s['status'].upper()} (probability {s['confidence']:.2f}, {s['tier']} tier"
        + ("" if s["calibrated"] else ", uncalibrated")
        + ").",
        f"  Abstain policy for the level names: {ABSTAIN_POLICY[s['status']]}.",
        f"  {applicability_line(anchors)}.",
    ]
    if s["applicability"]["unconfirmed_reasons"]:
        lines.append(
            "  Why the names are not confirmed: " + "; ".join(s["applicability"]["unconfirmed_reasons"]) + "."
        )
    fov = fov_line(anchors)
    if fov:
        lines.append(f"  Field of view: {fov}. Levels above the top cut are outside the field, not missing.")
    lines.append(f"  {conus_line(anchors)}. Do not report the conus as a finding; it is a locator only.")
    return lines


def _anchor_lines(anchors: Anchors | None) -> list[str]:
    if anchors is None:
        return [
            "  No anchors were supplied. Run label_spine (spinescope anatomy / get_anchors) before reading, "
            "so that level fields can be tied to disc labels. Level tasks are not generated without anchors."
        ]
    lines = [
        f"  Source stack: {anchors.source_stack or 'none'}. Method: {anchors.method or 'unknown'}.",
        *_status_block(anchors),
    ]
    rows = _level_rows(anchors)
    if rows:
        lines.append("  Discs, superior to inferior (confidence is the automatic label's own):")
        unconfirmed = status_of(anchors) == "unconfirmed"
        for pos, r in enumerate(rows, start=1):
            h = "n/a" if r["height_mm"] is None else f"{r['height_mm']:g} mm"
            tail = f" (automatic name {r['level']}, unconfirmed)" if unconfirmed else ""
            label = f"position {pos}" if unconfirmed else r["level"]
            lines.append(
                f"    {label}: z {r['z_mm']:+.1f} mm, height {h}, confidence {r['confidence']:.2f}{tail}"
            )
    else:
        lines.append("  No disc anchors were found. Label the levels yourself from the sagittal images.")
    for w in anchors.warnings:
        lines.append(f"  Warning: {w}")
    lines.append(
        "  Treat these as starting points. Confirm each label on the images (count from the sacrum) and "
        "report a disagreement in notes, not silently."
    )
    if status_of(anchors) == "unconfirmed":
        lines.append(
            "  The names are unconfirmed: write no level name into a finding until verify_labels has been applied "
            "(MCP) or verify-labels --answer has been run (CLI) and the status is confirmed or probable."
        )
    return lines


def _series_lines(study: Study) -> list[str]:
    rows = series_table(study)
    if not rows:
        return ["  No series were loaded."]
    return [_fmt_series_line(r) for r in rows]


def brief(
    study: Study,
    anchors: Anchors | None = None,
    *,
    reader: str = "reader_a",
    scope: str | None = None,
    config: Config = DEFAULT,
) -> str:
    """The reading brief for an AI reader: study technicals, conventions, anchors, tools, calibration,
    the reader's scope (if given) and the findings-format instructions.

    ``reader`` is the reader name the findings file must use. Raises ``ValueError`` for an invalid name.
    Plain text; safe to paste into a prompt.
    """
    return _brief_text(study, anchors, reader, scope, geometry_facts(study, anchors, config))


def _brief_text(
    study: Study, anchors: Anchors | None, reader: str, scope: str | None, facts: dict[str, Any]
) -> str:
    if not READER_NAME_RE.match(reader):
        raise ValueError("reader name must be 1-64 characters from A-Z, a-z, 0-9, '_', '.', '-'")
    info = {k: v for k, v in study.info.items() if k in KEEP_TAGS}
    head = ", ".join(f"{k}={v}" for k, v in info.items()) or "no technical study attributes"
    naming = naming_policy(anchors)
    parts = [
        "SPINE MRI READING BRIEF",
        "Research and education only. Describe what is visible; do not diagnose. Do not identify anyone.",
        *([f"LEVEL NAMING: {naming} Details in section 3."] if naming else []),
        "",
        "1. Study (technical attributes only)",
        f"  {head}",
        "  Series (key, role and geometry; every stack is in display orientation):",
        *_series_lines(study),
        "",
        "2. Geometry and display conventions (computed from the loaded stacks)",
        *_geometry_lines(facts),
        "",
        "3. Levels and anchors",
        *_anchor_lines(anchors),
        "",
        "4. Tools and images",
        *[f"  {line}" for line in _TOOL_LINES],
        "",
        "5. " + CALIBRATION_GUIDANCE.strip(),
    ]
    if scope:
        parts += ["", "6. Your scope", scope.strip()]
    parts += ["", "7. Findings format", reader_instructions(reader).rstrip()]
    return "\n".join(parts) + "\n"


# ------------------------------------------------------------- reader scopes


def _suggested_series(study: Study, anchors: Anchors | None, level: str | None) -> tuple[str, ...]:
    """Stacks worth looking at: the non-localizer sagittals, plus axial slabs that cover ``level``."""
    keys: list[str] = []
    for st in sorted(study.stacks.values(), key=lambda s: (s.plane != "sag", s.key)):
        role = st.meta.get("role", {})
        if role.get("localizer"):
            continue
        if st.plane == "sag":
            keys.append(st.key)
        elif st.plane == "ax" and level is not None and anchors is not None and level in anchors.discs:
            d = anchors.discs[level]
            _k, dist = st.nearest_slice(np.asarray(d.center))
            if dist <= 0.5 * (st.thickness or st.spacing or 1.0) + 0.5 * (d.height_mm or 0.0):
                keys.append(st.key)
    return tuple(keys)


def _level_task(study: Study, anchors: Anchors, lab: str, facts: dict[str, Any]) -> ReadingTask:
    groups = level_group_names()
    scope = (
        f'Level {lab} only. Fill levels["{lab}"] with every group of the disc level '
        f"({', '.join(groups)}). Look at the mid-sagittal slices, the paramedian sagittals for the foramina and "
        "recesses, and the axial slabs that cover this disc. Fill only this level. Do not read the other "
        "levels except to check the endplates that bound this disc."
    )
    return _task(
        study,
        anchors,
        kind="level",
        id=f"level-{lab}",
        title=f"Disc level {lab}",
        scope=scope,
        levels=(lab,),
        groups=groups,
        series=_suggested_series(study, anchors, lab),
        depends_on=(),
        facts=facts,
    )


def _structure_task(
    study: Study,
    anchors: Anchors | None,
    name: str,
    desc: str,
    groups: tuple[str, ...],
    facts: dict[str, Any],
) -> ReadingTask:
    levels = tuple(anchors.disc_labels()) if anchors is not None else ()
    scope = (
        f"Structure: {name}. {desc} Read this structure at every level listed below and fill only the groups "
        f"{', '.join(groups)} under levels. Do not fill other structures. Each level you read gets its own "
        "evidence entries."
    )
    return _task(
        study,
        anchors,
        kind="structure",
        id=f"structure-{name}",
        title=f"Structure: {name}",
        scope=scope,
        levels=levels,
        groups=groups,
        series=tuple(s.key for s in study.stacks.values() if not s.meta.get("role", {}).get("localizer")),
        depends_on=(),
        facts=facts,
    )


def _holistic_task(study: Study, anchors: Anchors | None, facts: dict[str, Any]) -> ReadingTask:
    groups = global_group_names() + ITEM_SECTIONS
    scope = (
        "Holistic read. Fill the global sections (alignment, marrow, conus_cord, posterior_elements, paraspinal, "
        "sacrum_si) and the array sections (focal_lesions, incidental_findings, pain_generator_hypotheses). Look "
        "for anything outside the disc levels: whole-spine alignment, marrow signal, the conus, the paraspinal "
        "muscles, the sacroiliac region, and any focal lesion on any slice you open. Do not fill level fields."
    )
    return _task(
        study,
        anchors,
        kind="holistic",
        id="holistic",
        title="Holistic read",
        scope=scope,
        levels=(),
        groups=groups,
        series=tuple(s.key for s in study.stacks.values() if not s.meta.get("role", {}).get("localizer")),
        depends_on=(),
        facts=facts,
    )


def _verify_task(study: Study, anchors: Anchors | None, facts: dict[str, Any]) -> ReadingTask:
    scope = (
        "Adversarial verifier. You run after the consensus merge. Input: consensus.json (the merged report) "
        "and the PNGs the readers cited. Take the highest-impact findings (largest disagreement first, then "
        "the most confident positives) and try to refute each one on the images. Do not look for support "
        'first. Write verify.json with one check per finding: {"path": canonical field path, "verdict": '
        '"supported" | "refuted" | "unclear", "confidence": low|medium|high, "evidence": [image, series, '
        'slices], "notes": one line}. Output JSON only. Do not write a findings file.'
    )
    return _task(
        study,
        anchors,
        kind="verify",
        id="verify",
        title="Adversarial verification of the consensus",
        scope=scope,
        levels=(),
        groups=(),
        series=tuple(s.key for s in study.stacks.values() if not s.meta.get("role", {}).get("localizer")),
        depends_on=("merge",),
        facts=facts,
        reader="verify_1",
    )


def _task(
    study: Study,
    anchors: Anchors | None,
    *,
    kind: str,
    id: str,
    title: str,
    scope: str,
    levels: tuple[str, ...],
    groups: tuple[str, ...],
    series: tuple[str, ...],
    depends_on: tuple[str, ...],
    facts: dict[str, Any],
    reader: str | None = None,
) -> ReadingTask:
    name = reader or id.replace("-", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:64]
    naming = naming_policy(anchors) if levels else None
    scope_text = scope + (f"\nLevel naming: {naming}" if naming else "")
    scope_text += f"\nSuggested stacks: {', '.join(series)}." if series else ""
    prompt = _brief_text(study, anchors, name, scope_text, facts)
    return ReadingTask(
        id=id,
        kind=kind,
        title=title,
        scope=scope_text,
        levels=levels,
        groups=groups,
        series=series,
        reader=name,
        depends_on=depends_on,
        prompt=prompt,
    )


def reading_tasks(
    study: Study, anchors: Anchors | None = None, *, config: Config = DEFAULT
) -> list[ReadingTask]:
    """Independent reader scopes for one study, in a stable order.

    Per disc level (from ``anchors``), one per structure family, one holistic read, and one adversarial
    verifier that depends on the merged consensus. The first three kinds are independent of each other and
    can run concurrently. Each task's ``prompt`` is a complete brief with its scope and the findings format.
    Without anchors there are no level tasks, and a warning is logged.
    """
    facts = geometry_facts(study, anchors, config)
    tasks: list[ReadingTask] = []
    if anchors is None or not anchors.discs:
        log.warning("reading_tasks: no disc anchors, so no per-level tasks are generated")
    if anchors is not None:
        tasks += [_level_task(study, anchors, lab, facts) for lab in anchors.disc_labels()]
    for name, desc, groups in STRUCTURE_FAMILIES:
        tasks.append(_structure_task(study, anchors, name, desc, groups, facts))
    tasks.append(_holistic_task(study, anchors, facts))
    tasks.append(_verify_task(study, anchors, facts))
    return tasks


# -------------------------------------------------------------- verification


def verification_errors(doc: Any) -> list[str]:
    """Problems with a verifier document (``verify.json``). Empty list when it is well formed."""
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    checks = doc.get("checks")
    if not isinstance(checks, list):
        return ['"checks" must be a list']
    for i, c in enumerate(checks):
        if not isinstance(c, dict):
            errs.append(f"checks[{i}] must be an object")
            continue
        if not isinstance(c.get("path"), str) or not c["path"]:
            errs.append(f"checks[{i}].path must be a non-empty string")
        if c.get("verdict") not in ("supported", "refuted", "unclear"):
            errs.append(f"checks[{i}].verdict must be supported, refuted or unclear")
        if c.get("confidence") not in ("low", "medium", "high"):
            errs.append(f"checks[{i}].confidence must be low, medium or high")
        ev = c.get("evidence")
        if not isinstance(ev, list) or not ev:
            errs.append(f"checks[{i}].evidence must be a non-empty list")
    return errs
