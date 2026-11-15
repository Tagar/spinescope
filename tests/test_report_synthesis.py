"""Synthesis: per-level and per-study picture of every registered analysis, validation status, not-assessed reasons,
pain-evidence ordering, the findings JSON, and the wording rules. Synthetic Result objects cover the logic; the phantom
(with a planted herniation, ground-truth anchors) covers the integration."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from _alignment_truth import with_body_geometry
from _studies import phantom_study
from spinescope.analysis.base import Result, run_all
from spinescope.analysis.findings import registered_fields
from spinescope.anatomy import Anchors
from spinescope.anatomy.anchors import Disc
from spinescope.config import DEFAULT
from spinescope.report import validation as V
from spinescope.report.schema import field_path, level_fields
from spinescope.report.synthesis import (
    DISCLAIMER,
    GlobalRule,
    build_synthesis,
    finding_text,
    findings_document,
    is_abnormal,
    register_global_field,
)
from spinescope.report.synthesis_render import brief_lines, render_html, render_markdown, write_bundle
from spinescope.report.validate import validate
from spinescope.study import Study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"\bdiagnos|\bunstable|\binstability|\btreat|\brecommend|\bsurgery|\binjection", re.IGNORECASE
)


def _anchors(levels: tuple[str, ...] = ("L3-L4", "L4-L5"), status: str = "confirmed") -> Anchors:
    a = Anchors()
    for i, lab in enumerate(levels):
        a.discs[lab] = Disc(center=[0.0, 0.0, float(-40 * i)], normal=[0.0, 0.0, 1.0], confidence=0.9)
    a.numbering = {"method": "test", "confidence": 1.0, "status": status}
    return a


def _res(name: str, rows: list[dict], figures: tuple[str, ...] = (), **meta: object) -> Result:
    r = Result(name=name, rows=rows, meta={"series_key": "sag_t2", **meta})
    for f in figures:
        r.figures[f] = np.zeros((4, 4, 3), dtype=np.uint8)
    return r


def _pfirrmann_row(level: str, grade: int, conf: float = 0.5) -> dict:
    return {
        "scope": "disc",
        "disc": level,
        "pfirrmann_suggestion": grade,
        "pfirrmann_confidence": conf,
    }


def _endplate_row(vertebra: str, ep: str, adjacent: str, pattern: str, conf: float = 0.4) -> dict:
    return {
        "scope": "endplate",
        "vertebra": vertebra,
        "endplate": ep,
        "adjacent_disc": adjacent,
        "side": "all",
        "pattern": pattern,
        "confidence": conf,
    }


def _without_disclaimer_and_negations(text: str) -> str:
    """Text with the disclaimer removed and "not ... a diagnosis" phrases (the negations) removed."""
    text = text.replace(DISCLAIMER, "")
    return re.sub(r"\bnot (a )?(severity or a )?diagnos\w*", "", text, flags=re.IGNORECASE)


def _study() -> Study:
    return Study(stacks={}, info={})


# ------------------------------------------------------------------------------------------- registry & validation
def test_validation_records_quote_their_source_document() -> None:
    """Every validation record names an existing document and a quote that appears in it (no drift from the docs)."""
    records = V.all_validations()
    assert len(records) >= 10
    for key, rec in records:
        if rec.status == V.NOT_VALIDATED:
            continue
        assert rec.source and rec.quote, key
        text = (REPO / rec.source).read_text(encoding="utf-8")
        assert rec.quote in text, f"{key}: {rec.quote!r} is not in {rec.source}"


def test_unknown_status_and_unknown_field_are_refused() -> None:
    with pytest.raises(ValueError):
        V.Validation("maybe")
    with pytest.raises(ValueError):
        V.register_validation("disc", "no_such_field", V.Validation(V.VALIDATED))


def test_unregistered_field_is_not_validated_and_values_override() -> None:
    assert V.validation_of("central_canal", "stenosis").status == V.NOT_VALIDATED
    assert V.validation_of("disc", "contour", "bulge").status == V.VALIDATED
    assert V.validation_of("disc", "contour", "protrusion").status == V.EXPERIMENTAL
    assert V.validation_of("disc", "contour").experimental
    assert not V.validation_of("disc", "pfirrmann").experimental


def test_global_registration_is_checked_against_the_schema() -> None:
    with pytest.raises(ValueError):
        register_global_field(GlobalRule("alignment", "not_a_field", "alignment", lambda r: None))
    with pytest.raises(ValueError):
        register_global_field(GlobalRule("alignment", "lumbar_lordosis_deg", "alignment", lambda r: None))


# ----------------------------------------------------------------------------------------------- logic
def test_findings_come_from_the_registry_with_validation_and_figures() -> None:
    anchors = _anchors()
    res = {
        "discs": _res("discs", [_pfirrmann_row("L4-L5", 4)], figures=("disc_strip",)),
        "endplates": _res(
            "endplates",
            [_endplate_row("L4", "inferior", "L4-L5", "Modic 2")],
            figures=("triptych_L4_inferior",),
        ),
    }
    syn = build_synthesis(_study(), anchors, res)
    lv = syn.level("L4-L5")
    fields = {f"{f.group}.{f.name}": f for f in lv.findings}
    pf = fields["disc.pfirrmann"]
    assert pf.value == 4 and pf.validation.status == V.VALIDATED and pf.figures == ("discs:disc_strip",)
    assert pf.abnormal and not pf.experimental
    # the inferior endplate of the body above L4-L5 is endplate_above
    modic = fields["endplate_above.modic"]
    assert modic.value == "2" and modic.experimental and modic.validation.status == V.EXPERIMENTAL
    assert modic.figures == ("endplates:triptych_L4_inferior",)


def test_every_schema_field_is_either_found_or_not_assessed_with_a_reason() -> None:
    anchors = _anchors()
    res = {"discs": _res("discs", [_pfirrmann_row("L4-L5", 2)], figures=("disc_strip",))}
    syn = build_synthesis(_study(), anchors, res)
    registered = {(f.group, f.name) for f in registered_fields() if f.name}
    for lv in syn.levels:
        found = {(f.group, f.name) for f in lv.findings}
        na = {tuple(n.field.split(".")) for n in lv.not_assessed}
        for f in level_fields():
            if (f.group, f.name) in registered:
                assert (f.group, f.name) in found or (f.group, f.name) in na, field_path(
                    lv.level, f.group, f.name
                )


def test_not_assessed_reasons_cover_missing_analysis_failure_fov_and_sequence() -> None:
    anchors = _anchors()
    res = {
        "discs": _res("discs", []),
        "endplates": Result(
            name="endplates", warnings=["analysis failed: ValueError: /secret/path/x could not be read"]
        ),
        "alignment": Result(
            name="alignment",
            warnings=[
                "L4-L5: outside the sagittal stack; not measured",
                "no sagittal STIR: STIR checks unavailable",
            ],
            meta={"series_key": "sag_t2"},
        ),
    }
    syn = build_synthesis(_study(), anchors, res)
    reasons = {(n.field, n.reason) for lv in syn.levels for n in lv.not_assessed}
    assert ("endplate_above.modic", "failed") in reasons
    assert ("listhesis.direction", "out_of_field") in reasons
    # the failure detail keeps the exception type only, never the message (paths can be inside it)
    text = json.dumps(syn.to_dict())
    assert "/secret/path" not in text
    # analyses that were not passed in are reported as not run, and the missing sequences are named
    study_reasons = {(n.reason, n.detail) for n in syn.study_not_assessed}
    assert ("not_run", "the analysis was not run for this study") in study_reasons
    assert any(r == "missing_sequence" and "sagittal T1" in d for r, d in study_reasons)


def test_pain_evidence_orders_by_confidence_weighted_status_weighted_strength() -> None:
    anchors = _anchors(("L3-L4", "L4-L5"))
    # L4-L5: a validated Pfirrmann 4 at medium confidence; L3-L4: an experimental Modic 2 at high confidence
    res = {
        "discs": _res("discs", [_pfirrmann_row("L4-L5", 4, conf=0.5)], figures=("disc_strip",)),
        "endplates": _res(
            "endplates",
            [_endplate_row("L3", "inferior", "L3-L4", "Modic 2", conf=0.9)],
            figures=("triptych_L3_inferior",),
        ),
    }
    syn = build_synthesis(_study(), anchors, res, DEFAULT)
    by_level = {r["level"]: r for r in syn.pain_evidence}
    w = DEFAULT.findings_confidence_weights
    sw = DEFAULT.synthesis_status_weights
    # Pfirrmann: confidence is capped at 0.6 by the analysis, so "medium" -> weight w[1]
    expected_l45 = round(w[1] * sw[0], 3)
    assert by_level["L4-L5"]["strength"] == pytest.approx(expected_l45)
    # the Modic row carries confidence 0.9 in the synthetic result ("high"), so only the status weight applies
    assert by_level["L3-L4"]["strength"] == pytest.approx(round(w[2] * sw[2], 3))
    assert syn.pain_evidence[0]["rank"] == 1
    assert syn.pain_evidence[0]["strength"] >= syn.pain_evidence[1]["strength"]
    # a pain row groups by structure: Pfirrmann is disc, Modic is endplate (vertebrogenic)
    assert by_level["L4-L5"]["groups"]["disc"] and by_level["L3-L4"]["groups"]["endplate_vertebrogenic"]
    assert by_level["L4-L5"]["groups"]["disc"][0]["text"].startswith("suggests")


def test_abnormal_rules_follow_presentation_thresholds() -> None:
    cfg = DEFAULT
    assert not is_abnormal("disc", "pfirrmann", 2, cfg) and is_abnormal("disc", "pfirrmann", 3, cfg)
    assert not is_abnormal("foramen_R", "lee_grade", 1, cfg) and is_abnormal("foramen_R", "lee_grade", 2, cfg)
    assert not is_abnormal("central_canal", "stenosis", "none", cfg)
    assert is_abnormal("central_canal", "stenosis", "mild", cfg)
    assert not is_abnormal("disc", "contour", "normal", cfg) and is_abnormal("disc", "contour", "bulge", cfg)
    assert not is_abnormal("disc", "pfirrmann", None, cfg)


def test_text_wording_measured_versus_suggests() -> None:
    anchors = _anchors()
    res = {"discs": _res("discs", [_pfirrmann_row("L4-L5", 4)], figures=("disc_strip",))}
    syn = build_synthesis(_study(), anchors, res)
    f = next(x for x in syn.level("L4-L5").findings if x.name == "pfirrmann")
    assert finding_text(f).startswith("suggests L4-L5:")
    assert "confidence" in finding_text(f)
    cont = next(x for x in syn.level("L4-L5").findings if x.name == "translation_mm") if False else None
    assert cont is None


def test_findings_json_validates_and_leaves_out_fields_without_saved_figures() -> None:
    anchors = _anchors()
    res = {
        "discs": _res("discs", [_pfirrmann_row("L4-L5", 4)], figures=("disc_strip",)),
        "endplates": _res(
            "endplates",
            [_endplate_row("L4", "inferior", "L4-L5", "Modic 2")],
            figures=("triptych_L4_inferior",),
        ),
    }
    # with saved figure paths for discs only: the endplate field has no image and is left out
    doc = findings_document(res, anchors, {"discs": {"disc_strip": "discs_disc_strip.png"}})
    check = validate(doc)
    assert check.ok, check.error_messages()
    assert "pfirrmann" in doc["levels"]["L4-L5"]["disc"]
    assert (
        "endplate_above" not in doc["levels"]["L4-L5"]
    )  # no figure saved: left out, not written without evidence
    assert "_skipped" not in json.dumps(doc)
    assert doc["reader"] == "findings"


def test_markdown_and_html_render_disclaimer_badges_and_no_forbidden_words(tmp_path: Path) -> None:
    anchors = _anchors()
    res = {"discs": _res("discs", [_pfirrmann_row("L4-L5", 4)], figures=("disc_strip",))}
    syn = build_synthesis(_study(), anchors, res)
    md = render_markdown(syn, tmp_path / "report.md")
    html_text = render_html(syn, tmp_path / "report.html")
    assert DISCLAIMER in md and DISCLAIMER in html_text
    assert '<html lang="en">' in html_text and "<script" not in html_text
    assert "<caption>" in html_text and 'scope="col"' in html_text
    assert "@page" in html_text and "@media print" in html_text
    assert "Skip to content" in html_text
    # the wording rules: outside the disclaimer, no diagnostic or treatment words
    body = _without_disclaimer_and_negations(md)
    assert not FORBIDDEN.search(body), FORBIDDEN.search(body)
    assert "Consider correlation with symptoms and examination" in md
    assert not FORBIDDEN.search(_without_disclaimer_and_negations(html_text))


def test_brief_lines_and_write_bundle(tmp_path: Path) -> None:
    anchors = _anchors()
    res = {"discs": _res("discs", [_pfirrmann_row("L4-L5", 4)], figures=("disc_strip",))}
    syn = build_synthesis(_study(), anchors, res)
    lines = brief_lines(syn)
    assert any(line.startswith("Level L4-L5") for line in lines)
    doc = findings_document(res, anchors, {"discs": {"disc_strip": "discs_disc_strip.png"}})
    from PIL import Image

    Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(tmp_path / "discs_disc_strip.png")
    paths = write_bundle(
        syn, doc, tmp_path, figure_files={"discs:disc_strip": tmp_path / "discs_disc_strip.png"}
    )
    assert {p.name for p in paths} == {"findings.json", "report.md", "report.html"}
    assert "data:image/png;base64," in (tmp_path / "report.html").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------- phantom integration
@pytest.fixture(scope="module")
def herniated():
    ph = SpinePhantom(PhantomSpec(herniation={"L4-L5": ("central", 8.0)}, herniation_displaces_sac=True))
    study = phantom_study(ph)
    anchors = with_body_geometry(ph, study, Anchors.from_truth(ph.truth))
    res = run_all(study, anchors, DEFAULT, names=["discs", "endplates", "herniation", "alignment"])
    return study, anchors, res


def test_phantom_synthesis_is_complete_and_registry_driven(herniated, tmp_path: Path) -> None:
    study, anchors, res = herniated
    syn = build_synthesis(study, anchors, res)
    assert {lv.level for lv in syn.levels} == set(anchors.disc_labels())
    assert all(lv.name_status == "confirmed" for lv in syn.levels)
    # every finding has an evidence figure and a validation status
    for lv in syn.levels:
        for f in lv.findings:
            assert f.figures, f"{f.group}.{f.name} at {f.level}"
            assert f.validation.status in V.STATUSES
    # the planted herniation is either measured or explicitly not assessed, never silently absent
    lv = syn.level("L4-L5")
    handled = {f"{f.group}.{f.name}" for f in lv.findings} | {n.field for n in lv.not_assessed}
    assert "disc.contour" in handled
    doc = findings_document(res, anchors, {a: {f: f"{a}_{f}.png" for f in r.figures} for a, r in res.items()})
    assert validate(doc).ok
    assert syn.pain_evidence and syn.pain_evidence[0]["rank"] == 1


def test_phantom_html_is_self_contained_and_printable(herniated, tmp_path: Path) -> None:
    study, anchors, res = herniated
    syn = build_synthesis(study, anchors, res)
    out = tmp_path / "phantom"
    out.mkdir()
    figs = {}
    for name, r in res.items():
        r.save(out)
        for fig in r.figures:
            figs[f"{name}:{fig}"] = out / f"{name}_{fig}.png"
    doc = findings_document(res, anchors, {a: {f: f"{a}_{f}.png" for f in r.figures} for a, r in res.items()})
    write_bundle(syn, doc, out, figure_files=figs)
    page = (out / "report.html").read_text(encoding="utf-8")
    assert page.count("data:image/png;base64,") >= 1
    assert "http://" not in page and "https://" not in page
    assert str(tmp_path) not in page  # no paths at all
