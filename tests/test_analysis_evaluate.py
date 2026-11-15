"""Validation harness: metrics against hand-computed values, name-matched scoring, phantom end-to-end, rendering."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from spinescope import load_study
from spinescope.analysis import evaluate as ev
from spinescope.analysis.base import Result
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study


# ------------------------------------------------------------------ metrics (hand-computed)
def test_binary_metrics_and_auc() -> None:
    y = np.array([1, 1, 0, 0, 1, 0.0])
    p = np.array([1, 0, 0, 1, 1, 0.0])
    s = np.array([0.9, 0.2, 0.1, 0.8, 0.7, 0.3])
    m = ev.summarise("binary", [0, 1], y, p, s)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["ppv"] == pytest.approx(2 / 3)
    assert m["auc"] == pytest.approx(6 / 9)  # 6 of 9 positive-negative pairs ordered correctly


def test_auc_ties_count_half_and_need_both_classes() -> None:
    assert ev.auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert math.isnan(ev.auc(np.array([1, 1]), np.array([0.1, 0.9])))


def test_quadratic_kappa_equals_unweighted_for_two_classes() -> None:
    conf = np.array([[20, 5], [10, 15]], dtype=np.int64)
    assert ev.weighted_kappa(conf, "quadratic") == pytest.approx(0.4)
    assert ev.weighted_kappa(conf, "none") == pytest.approx(0.4)


def test_perfect_and_linear_kappa() -> None:
    perfect = np.diag([3, 4, 5, 6, 7]).astype(np.int64)
    assert ev.weighted_kappa(perfect, "quadratic") == pytest.approx(1.0)
    conf = np.zeros((3, 3), dtype=np.int64)
    conf[0, 0] = 10
    conf[2, 0] = 10  # class 0 vs 2 is a worse disagreement than 0 vs 1
    assert ev.weighted_kappa(conf, "quadratic") < ev.weighted_kappa(conf, "linear") + 1.0
    assert math.isnan(ev.weighted_kappa(np.zeros((3, 3), dtype=np.int64)))


def test_ordinal_metrics() -> None:
    y = np.array([1, 2, 3, 4, 5.0])
    p = np.array([1, 2, 4, 4, 3.0])
    m = ev.summarise("ordinal", [1, 2, 3, 4, 5], y, p)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["mae"] == pytest.approx(0.6)  # |0+0+1+0+2|/5
    assert m["within_one"] == pytest.approx(0.8)
    assert m["confusion"][2][3] == 1  # truth 3 predicted 4
    assert m["per_class"]["3"]["sensitivity"] == pytest.approx(0.0)


def test_categorical_metrics_and_any_positive() -> None:
    y = np.array([0, 1, 2, 2.0])
    p = np.array([0, 1, 2, 0.0])
    m = ev.summarise("categorical", [0, 1, 2, 3], y, p)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["kappa"] == pytest.approx((0.75 - 5 / 16) / (1 - 5 / 16))
    assert m["any_sensitivity"] == pytest.approx(2 / 3)  # positives 1, 2, 2 -> predicted positive 1, 2
    assert m["any_specificity"] == pytest.approx(1.0)


def test_continuous_metrics() -> None:
    m = ev.summarise("continuous", [], np.array([1.0, 2.0]), np.array([1.5, 1.0]))
    assert m["mae"] == pytest.approx(0.75)
    assert m["bias"] == pytest.approx(-0.25)


def test_bootstrap_is_deterministic_and_resamples_cases() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0.0])
    p = np.array([1, 0, 0, 1, 1, 0, 1, 0.0])
    case = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    a = ev.bootstrap_ci("binary", [0, 1], y, p, None, case, n_boot=200, seed=3)
    b = ev.bootstrap_ci("binary", [0, 1], y, p, None, case, n_boot=200, seed=3)
    assert repr(a) == repr(b)  # NaN intervals compare equal through their repr
    lo, hi = a["sensitivity"]
    assert 0.0 <= lo <= hi <= 1.0
    assert ev.bootstrap_ci("binary", [0, 1], y, p, None, case, n_boot=0, seed=3) == {}


# ------------------------------------------------------------------ matching
def _truth(name: str, z: float, y: float = 0.0) -> ev.TruthRef:
    return ev.TruthRef(name=name, center=(0.0, y, z), values={})


def test_match_is_one_to_one_and_reports_leftovers() -> None:
    truth = [_truth("L4-L5", -10.0), _truth("L5-S1", -40.0)]
    pred = {"L4-L5": (0.0, 0.0, -9.5), "X": (0.0, 0.0, -11.0), "Far": (0.0, 0.0, -200.0)}
    out = ev.match_discs(pred, truth)
    assert len(out["pairs"]) == 1  # "L4-L5" and "X" are both nearest to the same truth disc
    assert out["pairs"][0][0] == "L4-L5"  # the closer one (0.5 mm against 1.0 mm) wins
    assert out["no_reference"] == ["Far", "X"]
    assert out["missing_pred"] == [1]


def test_match_respects_the_distance_limit() -> None:
    out = ev.match_discs({"L4-L5": (0.0, 0.0, 0.0)}, [_truth("L4-L5", -20.0)], max_mm=12.0)
    assert out["pairs"] == []
    assert out["no_reference"] == ["L4-L5"]


# ------------------------------------------------------------------ aggregation with synthetic outcomes
def _pair(
    case: str, level: str, field: str, status: str, truth: float | None, pred: float | None, **groups: str
) -> ev.Pair:
    return ev.Pair(
        case=case,
        level=level,
        field=field,
        status=status,
        truth=truth,
        pred=pred,
        score=None,
        groups={"vendor": "Siemens", **groups},
    )


def test_aggregate_counts_drops_and_not_assessed_visibly() -> None:
    outcome = ev.CaseOutcome(
        case="7",
        counts={
            "pred_discs": 3,
            "truth_discs": 3,
            "matched": 3,
            "label_mismatch": 1,
            "no_reference": 0,
            "missing_pred": 0,
            "scored_discs": 2,
        },
        pairs=[
            _pair("7", "L4-L5", "pfirrmann", "scored", 3, 3),
            _pair("7", "L3-L4", "pfirrmann", "not_assessed", 2, None),
            _pair("7", "L5-S1", "pfirrmann", "scored", 4, 4),
        ],
    )
    rep = ev.aggregate([outcome], dataset="synthetic", analyses=["discs"], settings={}, n_boot=0, seed=0)
    assert rep.counts["label_mismatch"] == 1  # dropped, and visible
    block = rep.fields["pfirrmann"]
    assert (block["scored"], block["not_assessed"]) == (2, 1)
    assert block["metrics"]["accuracy"] == pytest.approx(1.0)
    assert block["analysis"] == "discs"
    # a field no analysis predicts is still listed, with its pairs counted as not assessed
    assert "disc_herniation" in rep.fields and rep.fields["disc_herniation"]["analysis"] is None


def test_render_markdown_and_json_are_clean() -> None:
    outcome = ev.CaseOutcome(
        case="1",
        counts={"matched": 1, "scored_discs": 1},
        pairs=[_pair("1", "L4-L5", "pfirrmann", "scored", 3, 4)],
    )
    rep = ev.aggregate(
        [outcome], dataset="synthetic", analyses=["discs"], settings={"n_boot": 0}, n_boot=0, seed=0
    )
    md = ev.render_markdown(rep)
    assert "## Coverage" in md and "pfirrmann" in md
    import json

    json.dumps(rep.to_dict(), allow_nan=False)  # NaN metrics must come out as null


# ------------------------------------------------------------------ end to end on the phantom
_PH: dict[str, object] = {}


def _phantom_loader(spec: ev.CaseSpec):  # type: ignore[no-untyped-def]
    """Test loader. Truth is the labeller's own disc set (so matching and scoring are tested independently of
    labelling accuracy on the cartoon phantom); ``case == "shift"`` swaps the names of L4-L5 and L3-L4."""
    from spinescope.anatomy import label_spine

    study = _PH["study"]
    anchors = label_spine(study)  # type: ignore[arg-type]
    shift = spec.case == "shift"
    swap = {"L4-L5": "L3-L4", "L3-L4": "L4-L5"} if shift else {}
    truth = [
        ev.TruthRef(
            name=swap.get(name, name),
            center=tuple(float(x) for x in d.center),
            values={"disc_narrowing": 1 if name == "L4-L5" else 0},
        )
        for name, d in anchors.discs.items()
    ]
    groups = {"vendor": "phantom", "field_strength": "1.5", "sequences": "all", "transitional_suspect": "no"}
    return study, truth, groups, True


ev.register_loader("test_phantom", _phantom_loader)


@pytest.fixture(scope="module")
def phantom_case(tmp_path_factory: pytest.TempPathFactory) -> None:
    ph = SpinePhantom(PhantomSpec(degenerated={"L4-L5": 0.2}))
    info = write_standard_study(ph, tmp_path_factory.mktemp("harness_phantom"))
    _PH["study"] = load_study(info["root"])


def test_score_case_on_phantom_scores_name_matched_discs(phantom_case: None) -> None:
    spec = ev.CaseSpec(source="test_phantom", case="ok", analyses=("discs", "endplates"))
    out = ev.score_case(spec)
    assert out.warnings == []
    n = out.counts["pred_discs"]
    assert out.counts["truth_discs"] == n
    assert out.counts["label_mismatch"] == 0
    assert out.counts["scored_discs"] == out.counts["matched"] == n
    statuses = {(p.field, p.status) for p in out.pairs}
    assert ("pfirrmann", "no_truth") in statuses  # the phantom has no gradings
    assert ("disc_narrowing", "scored") in statuses or ("disc_narrowing", "not_assessed") in statuses
    assert all(p.groups["reference"] in ("t2", "t1", "stir") for p in out.pairs)


def test_label_shift_is_dropped_and_counted(phantom_case: None) -> None:
    out = ev.score_case(ev.CaseSpec(source="test_phantom", case="shift", analyses=("discs",)))
    n = out.counts["pred_discs"]
    assert out.counts["label_mismatch"] == 2  # L4-L5 and L3-L4 named the other way round: dropped, not scored
    assert out.counts["scored_discs"] == n - 2
    assert not any(p.level in ("L4-L5", "L3-L4") for p in out.pairs)


def test_unknown_loader_is_reported_not_raised() -> None:
    out = ev.score_case(ev.CaseSpec(source="no_such_source", case="x", analyses=("discs",)))
    assert out.pairs == [] and out.warnings and out.warnings[0].startswith("case not loaded")


def test_evaluate_single_process_end_to_end(phantom_case: None) -> None:
    specs = [ev.CaseSpec(source="test_phantom", case="a", analyses=("discs",))]
    rep = ev.evaluate(specs, dataset="phantom", workers=1, n_boot=20, seed=1)
    assert rep.n_cases == 1 and rep.counts["scored_discs"] == rep.counts["pred_discs"] > 0
    assert rep.fields["disc_narrowing"]["analysis"] == "discs"
    again = ev.evaluate(specs, dataset="phantom", workers=1, n_boot=20, seed=1)
    assert rep.to_dict() == again.to_dict()  # deterministic


def test_registration_guards() -> None:
    with pytest.raises(KeyError):
        ev.register_predictor("discs", "not_declared", lambda r, a: {})
    with pytest.raises(ValueError):
        ev.register_predictor("other", "pfirrmann", lambda r, a: {})  # already predicted by discs
    with pytest.raises(ValueError):
        ev.TruthField("x", "nominal")


def test_default_workers_and_threads_are_sane() -> None:
    assert ev.default_workers() >= 1
    assert isinstance(Result(name="x"), Result)
    assert Path(ev.__file__).name == "evaluate.py"
