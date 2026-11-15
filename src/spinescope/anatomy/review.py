"""Human review of the level names: bind a corrected anchors file to its study, check it, and set the status by policy.

The viewer (or any editor) can correct the labels: shift, rename, add, drop, each with a reason, then export the
anchors JSON. That file is only used once it passes this module:

1. Schema: it validates against ``spinescope/data/anchors.schema.json``. Unknown keys are refused, so nothing a
   tool adds outside the schema reaches the outputs.
2. Study binding: ``numbering.study_fingerprint`` must equal :func:`study_fingerprint` of the study. The fingerprint
   hashes the sagittal and axial stack geometry (sizes, spacings, orientation, positions); it holds no UIDs, names,
   dates or paths. A file made from another study (or edited against another) is refused.
3. Leaks: no text field may contain the study's file name or path, or something that looks like one.
4. Status: the numbering status is recomputed from the provenance (:func:`reconcile_status`); the status written in
   the file is never trusted. The policy, in order:

   * a bone-blind review (``applicability.unconfirmed_reasons``, e.g. no sacrum in view) is always ``unconfirmed``;
   * ``confirmed`` needs an explicit human ``confirm`` after the last change, and a numbering confidence at or above
     the confirmed threshold (:data:`spinescope.anatomy.numbering.DEFAULT_STATUS_THRESHOLDS`; the human confirm is
     gated by the default on purpose, see section 6 of docs/labeling/pipeline.md);
   * any other human review (a correction, an export or an applied review) is at most ``probable``;
   * an AI reader's relabelling with no human review since is ``unconfirmed``;
   * AI-only review is capped at ``probable`` (the same cap as :func:`spinescope.anatomy.verify.review_status`).

An AI reader can therefore never confirm the names. Only :func:`apply_review` with a human reason, followed by a
human confirm, can.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping
from importlib import resources
from typing import TYPE_CHECKING, Any

from .anchors import Anchors
from .numbering import (
    STATUS_CONFIRMED,
    STATUS_PROBABLE,
    STATUS_UNCONFIRMED,
    numbering_status,
)

if TYPE_CHECKING:
    from ..study import Study

log = logging.getLogger(__name__)

#: Key in ``numbering`` that holds the study fingerprint.
FINGERPRINT_KEY = "study_fingerprint"
#: Correction actions that change level names or positions (a relabelling of the study).
STRUCTURAL_ACTIONS: frozenset[str] = frozenset({"shift", "relabel", "drop", "add"})
#: Labeller warnings about the numbering status and the bone below the lowest disc; a review that sets a higher
#: status supersedes them (the status summary is written by the pipeline, see ``pipeline._status_warnings``).
_SUPERSEDED_WARNINGS: tuple[str, ...] = (
    "numbering unconfirmed",
    "numbering probable (",
    "numbering confidence is low",
    "lowest disc not confirmed by bone:",
)
#: Stack planes the labeller reads; their geometry is what the fingerprint hashes.
_LABELLED_PLANES: tuple[str, ...] = ("sag", "ax")
#: Free-text fields are scanned for these path shapes (POSIX home/volume roots, Windows drive paths).
_PATH_RE = re.compile(r"(?:/Users/|/home/|/Volumes/|/private/|/tmp/|/var/folders/|[A-Za-z]:\\\\)")
#: A bare folder name is only looked for when it looks like an identifier (long, with a digit or separator):
#: words such as "study" appear in ordinary summaries and would give false alarms. The full path is always checked.
_ID_NAME_RE = re.compile(r"^(?=.*[0-9_.\-]).{8,}$")


class ReviewError(ValueError):
    """A corrected anchors file or review request is refused. The message says why and never repeats a path."""


_SCHEMA: dict[str, Any] | None = None


def load_schema() -> dict[str, Any]:
    """The anchors JSON schema shipped with the package (cached)."""
    global _SCHEMA
    if _SCHEMA is None:
        text = (resources.files("spinescope") / "data" / "anchors.schema.json").read_text(encoding="utf-8")
        _SCHEMA = json.loads(text)
    return _SCHEMA


def schema_errors(doc: Mapping[str, Any]) -> list[str]:
    """Schema violations of an anchors document as ``path: rule`` strings (at most 10). Values are never echoed.

    Uses ``jsonschema`` (a core dependency). Messages are reduced to the failing rule name so that a value in the
    file, which could be free text, is never repeated.
    """
    import jsonschema

    validator = jsonschema.Draft202012Validator(load_schema())
    out: list[str] = []
    for err in sorted(validator.iter_errors(dict(doc)), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "(document)"
        out.append(f"{where}: fails the '{err.validator}' rule")
        if len(out) == 10:
            break
    return out


def study_fingerprint(study: Study, source_stack: str | None = None) -> str:
    """``sha256:<16 hex>`` hash of the geometry the labels were measured on: the source sagittal stack
    (``source_stack``; the best sagittal stack when None) and every axial stack that is not a localizer. Sizes,
    pixel spacing, thickness, orientation and slice positions (rounded), sorted so that the series order and the
    series keys do not matter.

    These are the stacks :func:`spinescope.anatomy.pipeline.labelling_keys` loads, so the fingerprint is the same
    whether the study was loaded whole or with only the labelled stacks. No UID, name, date, path or other DICOM
    attribute enters the hash.
    """
    if source_stack is None:
        best = study.best("sag")
        source_stack = best.key if best is not None else None
    recs: list[str] = []
    for st in study.stacks.values():
        if st.plane not in _LABELLED_PLANES or st.meta.get("role", {}).get("localizer"):
            continue
        if st.plane == "sag" and st.key != source_stack:
            continue
        rec = {
            "plane": st.plane,
            "shape": [int(x) for x in st.shape],
            "px_mm": [_r(st.row_mm, 3), _r(st.col_mm, 3)],
            "thickness": _r(st.thickness, 3),
            "row_dir": [_r(v, 6) for v in st.row_dir],
            "col_dir": [_r(v, 6) for v in st.col_dir],
            "normal": [_r(v, 6) for v in st.normal],
            "ipp": [[_r(v, 2) for v in p] for p in st.ipp],
        }
        recs.append(json.dumps(rec, sort_keys=True))
    digest = hashlib.sha256("\n".join(sorted(recs)).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def _r(x: Any, nd: int) -> float:
    return float(round(float(x), nd)) + 0.0  # + 0.0 turns -0.0 into 0.0 so the hash is stable


def stamp_study(anchors: Anchors, study: Study) -> Anchors:
    """Record the study fingerprint in ``anchors.numbering`` (in place; returns ``anchors``)."""
    anchors.numbering[FINGERPRINT_KEY] = study_fingerprint(study, anchors.source_stack or None)
    return anchors


def check_binding(anchors: Anchors, study: Study) -> None:
    """Raise :class:`ReviewError` unless ``anchors`` was made from ``study`` (fingerprint and source series)."""
    fp = anchors.numbering.get(FINGERPRINT_KEY)
    if not isinstance(fp, str):
        raise ReviewError(
            "these anchors are not bound to a study (no study_fingerprint): label the study again with this "
            "spinescope version and correct that file"
        )
    if fp != study_fingerprint(study, anchors.source_stack or None):
        raise ReviewError(
            "these anchors belong to a different study (the stack geometry does not match): correct the labels "
            "of this study, not another one"
        )
    if anchors.source_stack and anchors.source_stack not in study.stacks:
        raise ReviewError("the anchors name a series this study does not have")


def leak_errors(doc: Any, study_path: str | None = None) -> list[str]:
    """Problems of text fields that would leak the input location: the study's file name or path, or a path shape."""
    needles: list[str] = []
    if study_path:
        full = os.path.abspath(os.fspath(study_path)).rstrip(os.sep)
        base = os.path.basename(full)
        needles.append(full)
        if _ID_NAME_RE.match(base):
            needles.append(base)
    errors: list[str] = []
    for text in _strings(doc):
        if any(n in text for n in needles):
            errors.append("a text field contains the study's file name or path")
        elif _PATH_RE.search(text):
            errors.append("a text field looks like a file path")
        if len(errors) >= 3:
            break
    return sorted(set(errors))


def _strings(obj: Any) -> list[str]:
    out: list[str] = []
    stack: list[Any] = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            out.append(cur)
        elif isinstance(cur, Mapping):
            stack.extend(cur.keys())
            stack.extend(cur.values())
        elif isinstance(cur, list | tuple):
            stack.extend(cur)
    return out


def _provenance(anchors: Anchors) -> dict[str, bool]:
    """What the correction history says: human review, AI relabelling not yet reviewed, explicit human confirm."""
    flags = {"human": False, "ai_pending": False, "confirmed": False}
    for rec in anchors.numbering.get("corrections", []) or []:
        if not isinstance(rec, Mapping):
            continue
        action, source = rec.get("action"), rec.get("source")
        if source == "human":
            flags["human"] = True
            flags["ai_pending"] = False  # a human has looked since any AI change
        if action in STRUCTURAL_ACTIONS:
            flags["confirmed"] = False  # a confirm only covers the names it was given with
            if source == "ai_reader":
                flags["ai_pending"] = True
        if action == "confirm" and source == "human":
            flags["confirmed"] = True
    return flags


def reconcile_status(anchors: Anchors) -> str:
    """The numbering status the provenance and the applicability allow (see the module docstring)."""
    if anchors.applicability.get("unconfirmed_reasons") or not anchors.discs:
        return STATUS_UNCONFIRMED
    raw = anchors.numbering.get("confidence", 0.0)
    conf = float(raw) if isinstance(raw, (int, float)) else 0.0
    flags = _provenance(anchors)
    if flags["confirmed"]:
        return STATUS_CONFIRMED if numbering_status(conf) == STATUS_CONFIRMED else STATUS_PROBABLE
    if flags["ai_pending"]:
        return STATUS_UNCONFIRMED
    if flags["human"]:
        return STATUS_PROBABLE
    # AI review or no review at all: the automatic status is never confirmed by this module, so it is capped at
    # probable; a missing or unconfirmed status stays unconfirmed (the file's own word is not trusted).
    prior = anchors.numbering.get("status")
    return STATUS_PROBABLE if prior in (STATUS_CONFIRMED, STATUS_PROBABLE) else STATUS_UNCONFIRMED


def load_anchors_file(
    study: Study, source: str | os.PathLike[str] | Mapping[str, Any], *, path: str | None = None
) -> Anchors:
    """Read a corrected or verified anchors file (a path, JSON text or a dict) for ``study``.

    Validates the schema, checks the study binding and the leak rules, and returns :class:`Anchors` whose status is
    :func:`reconcile_status`. Raises :class:`ReviewError` (never with a path in the message). ``path`` is the study's
    input path, used only for the leak check.
    """
    doc = _as_document(source)
    errors = schema_errors(doc)
    if errors:
        raise ReviewError("the anchors file does not match the anchors schema: " + "; ".join(errors[:5]))
    try:
        anchors = Anchors.from_dict(doc)
    except (TypeError, ValueError, KeyError) as exc:
        raise ReviewError(f"the anchors file could not be read ({type(exc).__name__})") from None
    check_binding(anchors, study)
    leaks = leak_errors(doc, path)
    if leaks:
        raise ReviewError("the anchors file is refused: " + "; ".join(leaks))
    anchors.numbering["status"] = reconcile_status(anchors)
    return anchors


def _as_document(source: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """A dict from a dict, JSON text (a string starting with ``{`` or ``[``) or a file (any path-like)."""
    if isinstance(source, Mapping):
        return dict(source)
    text: str
    if isinstance(source, os.PathLike) or not str(source).lstrip().startswith(("{", "[")):
        try:
            with open(os.fspath(source), encoding="utf-8") as f:
                text = f.read()
        except OSError:
            raise ReviewError("cannot read the anchors file") from None
    else:
        text = str(source)
    try:
        doc = json.loads(text)
    except ValueError:
        raise ReviewError("the anchors file is not valid JSON") from None
    if not isinstance(doc, dict):
        raise ReviewError("the anchors file must hold one JSON object")
    return doc


def default_reason(anchors: Anchors) -> str:
    """The reason saved with the latest human record of ``anchors`` (the reason typed in the viewer), or ''."""
    for rec in reversed(anchors.numbering.get("corrections", []) or []):
        if isinstance(rec, Mapping) and rec.get("source") == "human" and str(rec.get("why", "")).strip():
            return str(rec["why"]).strip()
    return ""


def apply_review(
    study: Study,
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    who: str = "reviewer",
    why: str | None = None,
    path: str | None = None,
) -> Anchors:
    """Load a corrected anchors file, add the human review record and set the status by policy.

    ``why`` defaults to the reason of the file's latest human record; with neither, :class:`ReviewError`. The
    returned anchors carry one ``apply`` provenance entry (source human) and the reconciled status. Nothing is
    written here; the caller writes ``anchors.to_dict()`` after the final leak check (done here).
    """
    anchors = load_anchors_file(study, source, path=path)
    reason = (why or "").strip() or default_reason(anchors)
    if not reason:
        raise ReviewError("a reason is needed for the review: give --why (it is saved with the provenance)")
    if who.strip() == "":
        raise ReviewError("give the reviewer's name or initials for the provenance record")
    before = anchors.numbering.get("status")
    anchors.record_correction("apply", who=who, source="human", why=reason, detail={"status_before": before})
    anchors.numbering["status"] = reconcile_status(anchors)
    if anchors.numbering["status"] != STATUS_UNCONFIRMED:
        # the labeller's "numbering unconfirmed" warnings describe the automatic result; the review supersedes them
        anchors.warnings = [w for w in anchors.warnings if not w.startswith(_SUPERSEDED_WARNINGS)]
        anchors.warnings.append(
            f"review: numbering status {anchors.numbering['status']} after the human review; the automatic "
            "warnings about the status no longer apply"
        )
    final = anchors.to_dict()
    errors = schema_errors(final)
    if errors:  # cannot happen for a valid input; guards the writer
        raise ReviewError("the reviewed anchors do not match the schema: " + "; ".join(errors[:3]))
    leaks = leak_errors(final, path)
    if leaks:
        raise ReviewError("the reviewed anchors are refused: " + "; ".join(leaks))
    log.info("review applied: status %s, %d discs", anchors.numbering["status"], len(anchors.discs))
    return anchors


__all__ = [
    "FINGERPRINT_KEY",
    "ReviewError",
    "apply_review",
    "check_binding",
    "default_reason",
    "leak_errors",
    "load_anchors_file",
    "load_schema",
    "reconcile_status",
    "schema_errors",
    "stamp_study",
    "study_fingerprint",
]
