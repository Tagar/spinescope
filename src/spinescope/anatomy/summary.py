"""One human-readable view of a labelling result, shared by the CLI, the MCP server, the reading briefs and the viewer.

Everything here is derived from :class:`Anchors` (its ``to_dict`` fields) and nothing else: no file path, no DICOM
attribute and no free text from the series is read or returned. The wording (banner, abstain policy, applicability,
field of view, conus) is defined once so that every surface says the same thing about the same numbering status.

Numbering status (``anchors.numbering['status']``) and what a reader may do with the level names (the abstain policy):

* ``confirmed``: use the level names.
* ``probable``: use the level names, each marked "to be confirmed".
* ``unconfirmed``: positions only. Do not use the level names; verify them first (``verify_labels``).

The conus locator is experimental: its record is always marked ``experimental`` and its confidence is capped.
"""

from __future__ import annotations

from typing import Any

from ..config import DEFAULT, Config
from .anchors import Anchors
from .numbering import (
    STATUS_CONFIRMED,
    STATUS_PROBABLE,
    STATUS_UNCONFIRMED,
)

#: What a reader may do with the level names, per numbering status. Briefs, the MCP server and the skills quote this.
ABSTAIN_POLICY: dict[str, str] = {
    STATUS_CONFIRMED: "use the level names",
    STATUS_PROBABLE: 'use the level names, each marked "to be confirmed" in its notes',
    STATUS_UNCONFIRMED: (
        "positions only: do not use the level names; call verify_labels (MCP) or verify-labels (CLI) first "
        "and use the names only after the status is confirmed or probable"
    ),
}

#: Confidence tier of the numbering, by status (the status is the calibrated decision; the number is shown next to it).
STATUS_TIER: dict[str, str] = {STATUS_CONFIRMED: "high", STATUS_PROBABLE: "medium", STATUS_UNCONFIRMED: "low"}

_KNOWN = (STATUS_CONFIRMED, STATUS_PROBABLE, STATUS_UNCONFIRMED)

#: Words for the applicability verdicts (see :mod:`spinescope.anatomy.ood`).
VERDICT_WORDS: dict[str, str] = {
    "lumbar": "lumbar spine study",
    "thoracolumbar": "thoracolumbar field of view",
    "whole_spine": "whole-spine field of view",
    "uncertain": "uncertain region",
    "not_lumbar": "not a lumbar spine study",
    "not_spine": "no spine anatomy found",
    "no_sagittal": "no usable sagittal series",
    "not_usable": "sagittal series too short to label",
}


def status_of(anchors: Anchors) -> str:
    """The numbering status; anything unknown (or no discs) counts as ``unconfirmed``."""
    if not anchors.discs:
        return STATUS_UNCONFIRMED
    s = str(anchors.numbering.get("status", "") or "")
    return s if s in _KNOWN else STATUS_UNCONFIRMED


def disc_tier(confidence: float, cfg: Config = DEFAULT) -> str:
    """'high' / 'medium' / 'low' for a disc confidence, with the viewer's colour bands."""
    if confidence >= cfg.viewer_conf_high:
        return "high"
    if confidence >= cfg.viewer_conf_low:
        return "medium"
    return "low"


def _reviewed(anchors: Anchors) -> bool:
    """True when a reader or a human confirmed or corrected the levels (any provenance record from a reviewer)."""
    for c in anchors.numbering.get("corrections", []) or []:
        if isinstance(c, dict) and c.get("source") in ("ai_reader", "human"):
            return True
    return False


def banner(anchors: Anchors) -> str:
    """One line for the top of any report: the numbering status and what it allows."""
    status = status_of(anchors)
    conf = float(anchors.numbering.get("confidence", 0.0) or 0.0)
    if not anchors.discs:
        return "NO LEVELS: no disc levels were labelled, so there are no level names."
    if status == STATUS_CONFIRMED:
        who = "by a reviewer" if _reviewed(anchors) else "automatically"
        return f"LEVELS CONFIRMED {who} (numbering probability {conf:.2f}): use the level names."
    if status == STATUS_PROBABLE:
        return (
            f"LEVELS PROBABLE (numbering probability {conf:.2f}): the level names are to be confirmed; "
            "mark them so in any output."
        )
    return (
        f"LEVELS UNCONFIRMED (numbering probability {conf:.2f}): use disc positions only; verify the names "
        "(verify_labels) before using them."
    )


def applicability_line(anchors: Anchors) -> str:
    """The applicability verdict in one line ('lumbar spine study: levels labelled')."""
    app = anchors.applicability or {}
    if not app:
        return "applicability: not assessed"
    verdict = str(app.get("verdict", "unknown"))
    words = VERDICT_WORDS.get(verdict, verdict.replace("_", " "))
    labelled = "levels labelled" if app.get("labelled") else "levels NOT labelled"
    return f"applicability: {words}; {labelled} (verdict {verdict})"


def fov_line(anchors: Anchors) -> str | None:
    """Field-of-view coverage in one line, or None when the study has no applicability record."""
    cov = (anchors.applicability or {}).get("coverage")
    if not isinstance(cov, dict) or not cov:
        return None
    parts: list[str] = []
    if cov.get("z_min_mm") is not None and cov.get("z_max_mm") is not None:
        parts.append(f"z {cov['z_min_mm']:g}..{cov['z_max_mm']:g} mm")
    if cov.get("top_level_in_view"):
        parts.append(f"top level in view {cov['top_level_in_view']}")
    if cov.get("superior_cut"):
        parts.append(f"superior edge cuts through {cov['superior_cut']}")
    sacrum = cov.get("sacrum_covered")
    if sacrum is True:
        parts.append("sacrum in view")
    elif sacrum is False:
        parts.append("sacrum NOT in view")
    else:
        parts.append("sacrum coverage unknown")
    return "; ".join(parts)


def conus_line(anchors: Anchors) -> str:
    """Conus status in one line. The conus is always experimental (uncalibrated, not validated)."""
    c = anchors.conus
    if not c or c.get("z") is None:
        return "conus: not located (experimental locator)"
    level = c.get("level") or "level unknown"
    conf = float(c.get("confidence", 0.0) or 0.0)
    low = c.get("low_lying")
    extra = ""
    if low is True:
        extra = ", low-lying"
    return f"conus: near {level}, confidence {conf:.2f}{extra} (EXPERIMENTAL, not validated)"


def lstv_line(anchors: Anchors) -> str:
    """Transitional-vertebra check in one line (see :mod:`spinescope.anatomy.lstv`)."""
    from .lstv import lstv_line as _line

    return _line(anchors)


def numbering_votes(anchors: Anchors) -> tuple[int, int]:
    """(votes that agree with the applied numbering, votes counted), from ``numbering['votes']`` (weight > 0)."""
    votes = anchors.numbering.get("votes") or []
    offset = int(anchors.numbering.get("offset", 0) or 0)
    counted = [v for v in votes if isinstance(v, dict) and float(v.get("weight", 0.0) or 0.0) > 0]
    agree = [v for v in counted if int(v.get("best", 0)) == offset]
    return len(agree), len(counted)


def labelling_summary(anchors: Anchors, cfg: Config = DEFAULT) -> dict[str, Any]:
    """The structured summary (JSON-safe). Keys are stable; ``levels`` is ordered superior to inferior."""
    status = status_of(anchors)
    num = anchors.numbering or {}
    conf = float(num.get("confidence", 0.0) or 0.0) if anchors.discs else 0.0
    agree, counted = numbering_votes(anchors)
    lumbo = num.get("lumbosacral")
    cov = (anchors.applicability or {}).get("coverage") or {}
    conus = anchors.conus or {}
    return {
        "status": status,
        "status_text": banner(anchors),
        "policy": ABSTAIN_POLICY[status],
        "tier": STATUS_TIER[status],
        "confidence": round(conf, 3),
        "calibrated": bool(num.get("calibrated", False)),
        "votes": {"agree": agree, "counted": counted},
        "levels": [
            {
                "level": lab,
                "z_mm": round(float(anchors.discs[lab].center[2]), 1),
                "height_mm": (
                    None
                    if anchors.discs[lab].height_mm is None
                    else round(float(anchors.discs[lab].height_mm or 0.0), 1)
                ),
                "confidence": round(float(anchors.discs[lab].confidence), 2),
                "tier": disc_tier(anchors.discs[lab].confidence, cfg),
            }
            for lab in anchors.disc_labels()
        ],
        "lumbosacral": None
        if not isinstance(lumbo, dict)
        else {"status": lumbo.get("status"), "removed": int(lumbo.get("applied_drop", 0) or 0)},
        "transitional": num.get("transitional_kind") if num.get("transitional") else None,
        "applicability": {
            "verdict": (anchors.applicability or {}).get("verdict"),
            "labelled": bool((anchors.applicability or {}).get("labelled", False)),
            "summary": (anchors.applicability or {}).get("summary", ""),
            "unconfirmed_reasons": list((anchors.applicability or {}).get("unconfirmed_reasons", []) or []),
            "line": applicability_line(anchors),
        },
        "fov": {
            "sacrum_covered": cov.get("sacrum_covered"),
            "top_level_in_view": cov.get("top_level_in_view"),
            "superior_cut": cov.get("superior_cut"),
            "z_range_mm": None if cov.get("z_min_mm") is None else [cov.get("z_min_mm"), cov.get("z_max_mm")],
            "line": fov_line(anchors),
        },
        "conus": {
            "located": bool(conus.get("z") is not None),
            "level": conus.get("level"),
            "confidence": conus.get("confidence"),
            "experimental": True,
            "line": conus_line(anchors),
        },
        "lstv": None
        if not isinstance(anchors.lstv, dict)
        else {
            "status": anchors.lstv.get("status"),
            "kind": anchors.lstv.get("kind"),
            "p_transitional": anchors.lstv.get("p_transitional"),
            "castellvi": (anchors.lstv.get("castellvi") or {}).get("grade"),
            "line": lstv_line(anchors),
        },
        "quality": quality_summary(anchors),
        "warnings": list(anchors.warnings),
    }


def quality_summary(anchors: Anchors, show_advisory: bool | None = None) -> dict[str, Any]:
    """Image quality and hardware in one structure: study score, flagged kinds and levels, and the caps applied.

    Empty-safe: a study without a quality record gets ``assessed`` False and no flags. Advisory detectors (STIR fat
    suppression, bias field) are listed under ``advisory`` and never count as flags.
    """
    if show_advisory is None:
        show_advisory = DEFAULT.quality_show_advisory
    q = anchors.quality or {}
    if not q:
        return {
            "assessed": False,
            "score": None,
            "flagged": [],
            "levels": {},
            "caps": [],
            "line": "image quality: not assessed",
        }
    study = q.get("study", {}) or {}
    flags = study.get("flags", {}) or {}
    present = sorted(k for k, v in flags.items() if isinstance(v, dict) and v.get("present"))
    advisory = sorted(
        k for k, v in flags.items() if isinstance(v, dict) and v.get("advisory") and v.get("would_flag")
    )
    levels = {
        lab: list(v.get("flagged", []))
        for lab, v in (q.get("levels", {}) or {}).items()
        if isinstance(v, dict) and v.get("flagged")
    }
    score = None if study.get("score") is None else float(study["score"])
    if score is None:
        return {
            "assessed": False,
            "score": None,
            "flagged": [],
            "levels": {},
            "caps": [],
            "line": "image quality: not assessed",
        }
    if present:
        line = f"image quality: score {float(score):.2f}; flags " + ", ".join(
            p.replace("_", " ") for p in present
        )
    else:
        line = f"image quality: score {float(score):.2f}; no hardware or artefact flags"
    if advisory and show_advisory:
        line += "; advisory (not capped, unvalidated): " + ", ".join(a.replace("_", " ") for a in advisory)
    return {
        "assessed": True,
        "score": None if score is None else round(float(score), 3),
        "flagged": present,
        "advisory": advisory,
        "levels": levels,
        "caps": [dict(c) for c in (q.get("caps", []) or []) if isinstance(c, dict)],
        "line": line,
    }


def summary_lines(anchors: Anchors, cfg: Config = DEFAULT) -> list[str]:
    """A short, human-readable summary (plain lines, no markup) for the CLI, MCP and briefs."""
    s = labelling_summary(anchors, cfg)
    lines = [
        s["status_text"],
        f"Policy: {s['policy']}.",
        applicability_line(anchors) + ".",
    ]
    if s["fov"]["line"]:
        lines.append(f"Field of view: {s['fov']['line']}.")
    lines.append(conus_line(anchors) + ".")
    if s["lstv"] is not None:
        lines.append(s["lstv"]["line"] + ".")
    if s["lumbosacral"] is not None:
        removed = s["lumbosacral"]["removed"]
        lines.append(
            f"Lumbosacral check: {s['lumbosacral']['status']}"
            + (f" ({removed} sacral disc(s) removed)." if removed else ".")
        )
    lines.append(s["quality"]["line"] + ".")
    if s["votes"]["counted"]:
        lines.append(
            f"Numbering votes: {s['votes']['agree']} of {s['votes']['counted']} agree with the applied count."
        )
    return lines


def one_line(anchors: Anchors, cfg: Config = DEFAULT) -> str:
    """A single line for ``spinescope inspect --label``."""
    s = labelling_summary(anchors, cfg)
    levels = s["levels"]
    span = f"{len(levels)} disc levels" + (
        f" ({levels[0]['level']}..{levels[-1]['level']})" if levels else ""
    )
    conus = "located" if s["conus"]["located"] else "not located"
    return (
        f"labelling: {s['status']} (probability {s['confidence']:.2f}), {span}; "
        f"applicability {s['applicability']['verdict'] or 'n/a'}; conus {conus} (experimental)"
    )


__all__ = [
    "ABSTAIN_POLICY",
    "STATUS_TIER",
    "applicability_line",
    "banner",
    "conus_line",
    "disc_tier",
    "fov_line",
    "labelling_summary",
    "numbering_votes",
    "one_line",
    "quality_summary",
    "status_of",
    "summary_lines",
]
