"""Synthetic findings builders for the report tests. No imaging data: values and paths are invented."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA_VERSION = "1.0"


def evidence(
    image: str = "out/reader/sag_t2.png", series: str = "sag_t2", slices: Iterable[int] = (12,)
) -> list[dict[str, Any]]:
    return [{"image": image, "series": series, "slices": list(slices)}]


def F(
    value: Any,
    confidence: str = "high",
    *,
    notes: str = "",
    image: str = "out/reader/sag_t2.png",
    series: str = "sag_t2",
    slices: Iterable[int] = (12,),
) -> dict[str, Any]:
    """One field object: {value, confidence, evidence, notes}."""
    return {
        "value": value,
        "confidence": confidence,
        "evidence": evidence(image, series, slices),
        "notes": notes,
    }


def item(
    level: str,
    kind: str,
    summary: str = "finding",
    *,
    confidence: str = "medium",
    attributes: Mapping[str, Any] | None = None,
    image: str = "out/reader/sag_t1.png",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "level": level,
        "kind": kind,
        "summary": summary,
        "confidence": confidence,
        "evidence": evidence(image, "sag_t1", (9,)),
        "notes": "",
    }
    if attributes is not None:
        out["attributes"] = dict(attributes)
    return out


def findings(
    reader: str,
    *,
    levels: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    global_: Mapping[str, Mapping[str, Any]] | None = None,
    focal_lesions: list[dict[str, Any]] | None = None,
    incidental_findings: list[dict[str, Any]] | None = None,
    pain_generator_hypotheses: list[dict[str, Any]] | None = None,
    scope: str = "sagittal T2, L3 to S1",
) -> dict[str, Any]:
    """A findings document for one reader. Pass only what the reader examined (missing = not assessed)."""
    doc: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "reader": reader, "scope": scope}
    if levels is not None:
        doc["levels"] = {lvl: dict(body) for lvl, body in levels.items()}
    if global_ is not None:
        doc["global"] = {k: dict(v) for k, v in global_.items()}
    if focal_lesions is not None:
        doc["focal_lesions"] = focal_lesions
    if incidental_findings is not None:
        doc["incidental_findings"] = incidental_findings
    if pain_generator_hypotheses is not None:
        doc["pain_generator_hypotheses"] = pain_generator_hypotheses
    return doc


def l45_disc(pfirrmann: Any, contour: Any = "bulge", confidence: str = "high") -> dict[str, Any]:
    """Disc group for L4-L5 with the two most-compared fields."""
    return {"pfirrmann": F(pfirrmann, confidence), "contour": F(contour, confidence)}


def minimal_valid(reader: str = "reader_a") -> dict[str, Any]:
    """The smallest document that exercises levels, global and every array section."""
    return findings(
        reader,
        levels={
            "L4-L5": {"disc": l45_disc(3), "central_canal": {"stenosis": F("none")}},
            "T12-L1": {"disc": {"pfirrmann": F(2)}},
            "L5-S1": {
                "nerve_root_R": {"status": F("contact", "medium"), "root": F("S1"), "course": F("exiting")}
            },
        },
        global_={"alignment": {"scoliosis": F("none")}, "conus_cord": {"conus_tip_level": F("L1")}},
        focal_lesions=[
            item("L4", "l4_hemangioma", "Small vertebral hemangioma", attributes={"size_mm": F(6.0)})
        ],
        incidental_findings=[
            item(
                "global",
                "renal_cyst",
                "Simple renal cyst",
                attributes={"urgency": F("routine"), "suggested_followup": F("none needed")},
            )
        ],
        pain_generator_hypotheses=[
            item(
                "L4-L5",
                "discogenic_l4_l5",
                "Discogenic pain possible",
                attributes={
                    "explains_chronic_lbp": F("possible"),
                    "evidence_strength": F("weak"),
                    "what_would_confirm": F("discography"),
                },
            )
        ],
    )
