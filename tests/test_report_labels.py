"""Rule-based parser for free-text lumbar reports: synthetic sentences with expected labels, coverage, privacy."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from spinescope.datasets import report_labels as rl

#: (sentence, expected (field, value, negated, level) tuples). Written for this test, not copied from any report.
CASES: list[tuple[str, list[tuple[str, object, bool, str | None]]]] = [
    (
        "L4-L5: mild disc bulge with left paracentral disc protrusion.",
        [("contour", "bulge", False, "L4-L5"), ("contour", "protrusion", False, "L4-L5")],
    ),
    ("No evidence of disc herniation.", [("contour", "herniation", True, None)]),
    (
        "L4-L5 level: no disc protrusion or herniation seen.",
        [("contour", "protrusion", True, "L4-L5"), ("contour", "herniation", True, "L4-L5")],
    ),
    ("Adequate spinal canal.", [("stenosis", False, True, None)]),
    ("The spinal canal is still adequate.", [("stenosis", False, True, None)]),
    ("L4-L5 narrowing of left neural foramen.", [("stenosis", True, False, "L4-L5")]),
    ("Mild canal stenosis at L3-L4.", [("stenosis", True, False, "L3-L4")]),
    ("L5/S1 grade I spondylolisthesis of L5 over S1.", [("listhesis", True, False, "L5-S1")]),
    ("Mild spondylolisthesis noted.", [("listhesis", True, False, None)]),
    ("Modic type II noted at the level of the L4/5 level.", [("modic", 2, False, "L4-L5")]),
    ("Endplate modic typei noted at L3-4 level.", [("modic", 1, False, "L3-L4")]),
    (
        "L3-L4 and L4-L5 narrowed disc space with dehydrated disc.",
        [
            ("disc_narrowing", True, False, "L3-L4"),
            ("disc_narrowing", True, False, "L4-L5"),
            ("disc_dehydration", True, False, "L4-L5"),
        ],
    ),
    (
        "No significant thecal sac or nerve root compression noted.",
        [("thecal_sac", False, True, None), ("nerve_root", False, True, None)],
    ),
    ("Facet joint hypertrophy noted at L4-L5.", [("facet", True, False, "L4-L5")]),
    ("Schmorl's nodes noted at L5-S1.", [("schmorl_node", True, False, "L5-S1")]),
    (
        "No disc protrusion at L4-L5 but mild bulge.",
        [("contour", "protrusion", True, "L4-L5"), ("contour", "bulge", False, "L4-L5")],
    ),
    (
        "L5-S1: broad based right paramedin disc extrusion is noted.",
        [("contour", "extrusion", False, "L5-S1")],
    ),
]


@pytest.mark.parametrize(("sentence", "expected"), CASES)
def test_sentence_labels(sentence: str, expected: list[tuple[str, object, bool, str | None]]) -> None:
    got = [(lab.field, lab.value, lab.negated, lab.level) for lab in rl.parse_report(sentence).labels]
    assert got == expected


def test_side_and_zone_come_from_the_nearest_tokens() -> None:
    labs = rl.parse_report("L5-S1: broad based right paramedin disc extrusion is noted.").labels
    assert labs[0].side == "right" and labs[0].zone == "paracentral"
    labs = rl.parse_report("Foraminal stenosis at L5-S1 on the right side.").labels
    assert labs[0].zone == "foraminal" and labs[0].side == "right"


def test_grade_and_direction_of_listhesis() -> None:
    lab = rl.parse_report("L5/S1 grade I spondylolisthesis of L5 over S1.").labels[0]
    assert lab.grade == "I"
    lab = rl.parse_report("Anterolisthesis of L4 over L5, mild.").labels[0]
    assert lab.direction == "anterolisthesis" and lab.grade == "mild"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("L4-L5", "L4-L5"),
        ("l4/5", "L4-L5"),
        ("L5 s1", "L5-S1"),
        ("T12-L1", "T12-L1"),
        ("L3-L5", None),  # not adjacent
        ("L5-6", None),
    ],
)
def test_level_normalisation(text: str, expected: str | None) -> None:
    mentions = rl.level_mentions(text.lower())
    got = mentions[0][2] if mentions else None
    assert got == expected


def test_clause_coverage_and_unparsed_templates_carry_no_raw_text() -> None:
    text = "L4-L5: mild disc bulge.\nSome unusual wording about a Zorbix clamp 12 days ago.\nLSS MRI:"
    p = rl.parse_report(text)
    assert p.clauses == 3
    assert p.substantive == 2  # the "LSS MRI:" header is boilerplate
    assert p.parsed == 1
    assert p.coverage == pytest.approx(0.5)
    assert list(p.unparsed_templates) == ["some unusual wording about a zorbix clamp <N> days ago"]
    assert not any(re.search(r"\d", t) for t in p.unparsed_templates)


def test_aggregate_levels_keeps_strongest_contour_and_infers_herniation() -> None:
    labs = rl.parse_report("L4-L5 mild disc bulge. L4-L5: right paracentral disc protrusion.").labels
    rec = rl.aggregate_levels(labs)["L4-L5"]
    assert rec["contour"]["value"] == "protrusion"
    assert rec["herniation_present"] == {"value": True, "conflict": False, "n": 2, "inferred": False}
    only_bulge = rl.aggregate_levels(rl.parse_report("L3-L4: disc bulge.").labels)["L3-L4"]
    assert only_bulge["herniation_present"]["value"] is False and only_bulge["herniation_present"]["inferred"]


def test_aggregate_levels_flags_conflicts() -> None:
    labs = rl.parse_report("L4-L5 canal stenosis. L4-L5: adequate spinal canal.").labels
    rec = rl.aggregate_levels(labs)
    assert rec["L4-L5"]["canal_stenosis"]["conflict"] is True
    assert rec["L4-L5"]["canal_stenosis"]["value"] is None


def test_coverage_summary_counts_across_reports() -> None:
    parses = rl.parse_reports(
        {1: "L4-L5: disc bulge.", 2: "Unrecognised phrase about a sandwich.", 3: "LSS MRI:"}
    )
    s = rl.coverage_summary(parses)
    assert s["reports"] == 3
    assert s["reports_with_any_label"] == 1
    assert s["substantive_clauses"] == 2
    assert s["parsed_clauses"] == 1
    assert s["clause_coverage"] == 0.5


def _sudirman_notes_available() -> bool:
    base = Path(os.environ.get("SPINESCOPE_DATA", "~/data/public")).expanduser()
    return (base / "lumbar_spine_mri_sudirman_notes" / "raw" / "Radiologists Report.xlsx").is_file()


@pytest.mark.dataset
@pytest.mark.skipif(not _sudirman_notes_available(), reason="Sudirman radiologist notes not on disk")
def test_sudirman_notes_parse_with_high_clause_coverage() -> None:
    parses = rl.load_sudirman_report_labels()
    s = rl.coverage_summary(parses)
    assert s["reports"] > 500
    assert s["clause_coverage"] is not None and s["clause_coverage"] > 0.9
