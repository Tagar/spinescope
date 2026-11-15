"""The labelling benchmark: regression checker logic, golden merging, CLI exit codes and the data-free paths.

The checker tests use hand-made result dicts (no data, no labelling). The end-to-end phantom test is ``slow``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from spinescope import benchmark as bm
from spinescope.benchmark import (
    Options,
    compare,
    expand_suites,
    format_report,
    golden_from,
    labeller_fingerprint,
    main,
    regressions,
    run_benchmark,
)

REPO = Path(__file__).resolve().parents[1]


def _section(cases: int = 10, **metrics: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "label_accuracy": 0.955,
        "label_accuracy_all_truth": 0.950,
        "recall": 0.994,
        "precision": 0.986,
        "centre_err_median_mm": 1.5,
        "centre_err_p90_mm": 3.0,
        "wrong_announced": 0,
        "errors": 0,
        "announced_coverage": 0.6,
        "announced_precision": 1.0,
        "latency_s_median": 2.0,
    }
    base.update(metrics)
    return {"title": "t", "backend": "auto", "cases": cases, "metrics": base, "status": {}}


def _result(**sections: dict[str, Any]) -> dict[str, Any]:
    return {"spinescope": "0.0", "labeller": {"fingerprint": "abc"}, "sections": dict(sections)}


def _golden(result: dict[str, Any]) -> dict[str, Any]:
    return golden_from(result)


def _verdicts(findings: list[bm.Finding]) -> dict[tuple[str, str], str]:
    return {(f.section, f.metric): f.verdict for f in findings}


# ----------------------------------------------------------------------------- suites
def test_expand_suites_expands_phantom_and_keeps_report_order() -> None:
    assert expand_suites(["phantom", "spider"]) == ("spider", "phantom_cartoon", "phantom_realistic")
    assert expand_suites(None) == bm.SUITES


def test_expand_suites_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        expand_suites(["spider", "nope"])


def test_labeller_fingerprint_is_stable_and_short() -> None:
    fp = labeller_fingerprint()
    assert fp == labeller_fingerprint()
    assert len(fp) == 16 and all(c in "0123456789abcdef" for c in fp)


# ----------------------------------------------------------------------------- checker rules
def test_identical_result_passes() -> None:
    res = _result(spider=_section())
    findings = compare(res, _golden(res))
    assert regressions(findings) == []
    assert all(f.verdict == "ok" for f in findings)


def test_label_accuracy_drop_beyond_half_a_point_is_a_regression() -> None:
    golden = _golden(_result(spider=_section(label_accuracy=0.955)))
    current = _result(spider=_section(label_accuracy=0.9499))  # drop of 0.51 pp
    bad = regressions(compare(current, golden))
    assert [(f.section, f.metric) for f in bad] == [("spider", "label_accuracy")]


def test_label_accuracy_drop_within_tolerance_passes() -> None:
    golden = _golden(_result(spider=_section(label_accuracy=0.955)))
    current = _result(spider=_section(label_accuracy=0.9505))  # drop of 0.45 pp
    assert regressions(compare(current, golden)) == []


def test_improvement_is_reported_not_failed() -> None:
    golden = _golden(_result(spider=_section(label_accuracy=0.955)))
    findings = compare(_result(spider=_section(label_accuracy=0.99)), golden)
    assert _verdicts(findings)[("spider", "label_accuracy")] == "improved"
    assert regressions(findings) == []


def test_wrong_announced_must_not_increase() -> None:
    golden = _golden(_result(spider=_section(wrong_announced=1)))
    assert regressions(compare(_result(spider=_section(wrong_announced=2)), golden))
    assert regressions(compare(_result(spider=_section(wrong_announced=0)), golden)) == []


def test_error_count_must_not_increase() -> None:
    golden = _golden(_result(spider=_section(errors=0)))
    assert regressions(compare(_result(spider=_section(errors=1)), golden))


def test_centre_error_rise_beyond_tolerance_is_a_regression() -> None:
    golden = _golden(_result(spider=_section(centre_err_median_mm=1.5)))
    assert regressions(compare(_result(spider=_section(centre_err_median_mm=2.1)), golden))
    assert regressions(compare(_result(spider=_section(centre_err_median_mm=1.9)), golden)) == []


def test_latency_is_not_gated() -> None:
    golden = _golden(_result(spider=_section(latency_s_median=2.0)))
    assert regressions(compare(_result(spider=_section(latency_s_median=20.0)), golden)) == []


def test_metric_disappearing_is_a_regression() -> None:
    golden = _golden(_result(spider=_section(label_accuracy=0.9)))
    current = _result(spider=_section())
    current["sections"]["spider"]["metrics"]["label_accuracy"] = None
    assert [f.verdict for f in regressions(compare(current, golden))] == ["missing"]


def test_none_in_golden_is_not_judged() -> None:
    golden = _golden(_result(spider=_section(precision=None)))
    assert "precision" not in golden["sections"]["spider"]["metrics"]
    assert regressions(compare(_result(spider=_section(precision=0.1)), golden)) == []


def test_new_metric_in_current_is_noted_not_failed() -> None:
    golden = _golden(_result(spider=_section()))
    current = _result(spider=_section(), extra_section=_section())
    findings = compare(current, golden)
    assert regressions(findings) == []
    assert any(f.verdict == "new" and f.section == "extra_section" for f in findings)


# ----------------------------------------------------------------------------- scenarios and sections
def _phantom(outcome: dict[str, str]) -> dict[str, Any]:
    sec = _section()
    sec["scenarios"] = outcome
    return sec


def test_scenario_outcome_worse_is_a_regression_better_is_not() -> None:
    golden = _golden(_result(phantom_cartoon=_phantom({"baseline": "pass", "fov": "unsure"})))
    worse = _result(phantom_cartoon=_phantom({"baseline": "unsure", "fov": "unsure"}))
    better = _result(phantom_cartoon=_phantom({"baseline": "pass", "fov": "pass"}))
    silent = _result(phantom_cartoon=_phantom({"baseline": "pass", "fov": "silent_error"}))
    assert [f.metric for f in regressions(compare(worse, golden))] == ["scenario baseline"]
    assert regressions(compare(better, golden)) == []
    assert [f.metric for f in regressions(compare(silent, golden))] == ["scenario fov"]


def test_missing_scenario_is_a_regression() -> None:
    golden = _golden(_result(phantom_cartoon=_phantom({"baseline": "pass"})))
    assert regressions(compare(_result(phantom_cartoon=_phantom({})), golden))


def test_skipped_section_is_reported_not_failed() -> None:
    golden = _golden(_result(spider=_section(), sudirman=_section(cases=515)))
    current = _result(spider=_section(), sudirman={"skipped": "Sudirman is not installed"})
    findings = compare(current, golden)
    assert regressions(findings) == []
    skipped = [f for f in findings if f.verdict == "skipped"]
    assert [f.section for f in skipped] == ["sudirman"]
    assert "not installed" in skipped[0].note


def test_section_not_run_at_all_is_skipped() -> None:
    golden = _golden(_result(spider=_section(), qx5rt=_section(cases=55)))
    findings = compare(_result(spider=_section()), golden)
    assert regressions(findings) == []
    assert [f.section for f in findings if f.verdict == "skipped"] == ["qx5rt"]


def test_changed_sample_size_fails_because_numbers_are_not_comparable() -> None:
    golden = _golden(_result(spider=_section(cases=87)))
    findings = compare(_result(spider=_section(cases=50)), golden)
    assert [f.verdict for f in regressions(findings)] == ["sample_changed"]


def test_failed_suite_is_a_regression() -> None:
    golden = _golden(_result(spider=_section()))
    findings = compare(_result(spider={"error": "RuntimeError"}), golden)
    assert [f.verdict for f in regressions(findings)] == ["error"]


# ----------------------------------------------------------------------------- golden file and report
def test_golden_keeps_sections_not_run_and_their_tolerances() -> None:
    previous = _golden(_result(spider=_section(), sudirman=_section(cases=515)))
    previous["sections"]["spider"]["metrics"]["label_accuracy"]["tol"] = 0.02  # a maintainer's choice
    new = golden_from(_result(spider=_section(label_accuracy=0.9)), previous)
    assert set(new["sections"]) == {"spider", "sudirman"}
    assert new["sections"]["spider"]["metrics"]["label_accuracy"] == {
        "value": 0.9,
        "direction": "higher",
        "tol": 0.02,
        "unit": "rate",
    }
    assert new["sections"]["sudirman"] == previous["sections"]["sudirman"]


def test_golden_skips_skipped_sections() -> None:
    res = _result(spider={"skipped": "not installed"}, phantom_cartoon=_section())
    assert set(golden_from(res)["sections"]) == {"phantom_cartoon"}


def test_report_names_each_regression_with_numbers_and_tolerance() -> None:
    golden = _golden(_result(spider=_section(label_accuracy=0.955, wrong_announced=0)))
    current = _result(spider=_section(label_accuracy=0.94, wrong_announced=1))
    text = format_report(compare(current, golden), golden, current)
    assert "RESULT: FAIL, 2 regression(s)" in text
    assert "spider.label_accuracy dropped by 1.50 pp (95.50% -> 94.00%), tolerance 0.50 pp" in text
    assert "spider.wrong_announced increased by 1 (0 -> 1)" in text


def test_report_passes_and_mentions_labeller_change() -> None:
    golden = _golden(_result(spider=_section()))
    current = _result(spider=_section())
    current["labeller"]["fingerprint"] = "zzz"
    golden["labeller_fingerprint"] = "abc"
    text = format_report(compare(current, golden), golden, current)
    assert "RESULT: PASS" in text
    assert "labeller source differs" in text


# ----------------------------------------------------------------------------- command line
def _write(path: Path, obj: Any) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_cli_check_exit_codes(tmp_path: Path) -> None:
    res = _result(spider=_section())
    golden = _write(tmp_path / "golden.json", _golden(res))
    ok = _write(tmp_path / "ok.json", res)
    assert main(["--from-json", str(ok), "--check", str(golden), "--quiet"]) == 0
    bad_res = copy.deepcopy(res)
    bad_res["sections"]["spider"]["metrics"]["label_accuracy"] = 0.90
    bad = _write(tmp_path / "bad.json", bad_res)
    assert main(["--from-json", str(bad), "--check", str(golden), "--quiet"]) == 1


def test_cli_update_golden_merges_into_existing_file(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.json"
    _write(golden_path, _golden(_result(sudirman=_section(cases=515))))
    res = _write(tmp_path / "res.json", _result(spider=_section(label_accuracy=0.9)))
    assert main(["--from-json", str(res), "--update-golden", str(golden_path), "--quiet"]) == 0
    written = json.loads(golden_path.read_text(encoding="utf-8"))
    assert set(written["sections"]) == {"spider", "sudirman"}


def test_cli_refuses_outputs_inside_the_repository() -> None:
    with pytest.raises(SystemExit):
        main(["--suites", "phantom", "--out", str(REPO / "benchmark-out"), "--quiet"])


def test_cli_rejects_check_with_update() -> None:
    with pytest.raises(SystemExit):
        main(["--check", "a.json", "--update-golden", "b.json"])


def test_spinescope_cli_forwards_benchmark_arguments(tmp_path: Path) -> None:
    from spinescope.cli import main as cli_main

    res = _write(tmp_path / "res.json", _result(spider=_section()))
    golden = _write(tmp_path / "golden.json", _golden(_result(spider=_section())))
    assert cli_main(["benchmark", "--from-json", str(res), "--check", str(golden), "--quiet"]) == 0


# ----------------------------------------------------------------------------- data-free runs
def test_run_skips_every_data_suite_when_data_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPINESCOPE_DATA", str(tmp_path / "empty"))
    res = run_benchmark(Options(suites=("spider", "sudirman", "qx5rt"), cache=tmp_path / "cache"))
    assert set(res["sections"]) == {"spider", "sudirman", "qx5rt"}
    assert all("skipped" in s for s in res["sections"].values())
    assert not (tmp_path / "cache").exists()


# ----------------------------------------------------------------------------- end to end (phantom)
@pytest.mark.slow
def test_phantom_run_uses_the_cache_and_writes_no_identifiers(tmp_path: Path) -> None:
    opts = Options(
        suites=("phantom_cartoon",),
        scenarios=("baseline", "t2_only"),
        cache=tmp_path / "cache",
        jobs=1,
    )
    first = run_benchmark(opts)
    sec = first["sections"]["phantom_cartoon"]
    assert sec["cases"] == 2 and sec["sample"]["from_cache"] == 0
    assert set(sec["scenarios"]) == {"baseline", "t2_only"}
    assert sec["backend"] == "classical"
    second = run_benchmark(opts)
    assert second["sections"]["phantom_cartoon"]["sample"]["from_cache"] == 2
    assert second["sections"]["phantom_cartoon"]["metrics"] == sec["metrics"]
    text = json.dumps(first)
    assert str(tmp_path) not in text
    assert all("/" not in k for k in sec["metrics"])
    md = bm.to_markdown(first)
    assert "phantom_cartoon" in md and str(tmp_path) not in md
