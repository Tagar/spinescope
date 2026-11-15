"""Numbering cues: slab, tilt and conus votes, combination, and relabelling (phantom-based)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.numbering_checks import (
    OFFSETS,
    NumberingConfig,
    Vote,
    combine,
    conus_vote,
    numbering_evidence,
    relabel_discs,
    slab_vote,
    tilt_vote,
)
from spinescope.geometry import Stack
from spinescope.testing.phantom import PhantomSpec, SpinePhantom


def _slab(
    centre: np.ndarray, normal: np.ndarray, key: str = "ax_slab", n: int = 4, spacing: float = 4.4
) -> Stack:
    """A small axial-oblique stack whose slices are parallel to ``normal`` and whose block contains ``centre``."""
    nrm = np.asarray(normal, dtype=np.float64)
    nrm = nrm / np.linalg.norm(nrm)
    helper = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    row = np.cross(helper, nrm)
    row /= np.linalg.norm(row)
    col = np.cross(nrm, row)
    origin = np.asarray(centre, dtype=np.float64) - 4.0 * row - 4.0 * col
    ipp = np.array([origin + (k - (n - 1) / 2) * spacing * nrm for k in range(n)])
    return Stack(
        key=key,
        vol=np.zeros((n, 8, 8), dtype=np.float32),
        ipp=ipp,
        row_dir=row,
        col_dir=col,
        row_mm=1.0,
        col_mm=1.0,
        thickness=spacing,
        meta={"role": {"slab_index": 1, "plane": "ax"}},
    )


def _shift_down(label: str) -> str:
    """Provisional-style name of a disc whose true name is one level higher (test helper for sacralization)."""
    order = ["S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10"]
    up, lo = label.split("-")
    return f"{order[order.index(up) - 1]}-{order[order.index(lo) - 1]}"


@pytest.fixture(scope="module")
def truth_anchors() -> Anchors:
    return Anchors.from_truth(SpinePhantom(PhantomSpec()).truth)


@pytest.fixture(scope="module")
def phantom_truth() -> dict:
    return SpinePhantom(PhantomSpec()).truth


# ------------------------------------------------------------ relabelling


def test_relabel_zero_is_identity() -> None:
    names = ["L5-S1", "L4-L5", "L3-L4"]
    assert relabel_discs(names, 0) == names


def test_relabel_plus_one_shifts_every_disc_up() -> None:
    assert relabel_discs(["L5-S1", "L4-L5", "L3-L4"], 1) == ["L4-L5", "L3-L4", "L2-L3"]


def test_relabel_minus_one_renames_only_lowest_disc() -> None:
    assert relabel_discs(["L5-S1", "L4-L5", "L3-L4"], -1) == ["L5-L6", "L4-L5", "L3-L4"]


def test_relabel_rejects_other_offsets() -> None:
    with pytest.raises(ValueError):
        relabel_discs(["L5-S1"], 2)


def test_relabel_empty_list() -> None:
    assert relabel_discs([], 1) == []


# ------------------------------------------------------------------- slab


def test_slab_vote_none_without_slabs(truth_anchors: Anchors) -> None:
    assert slab_vote([], truth_anchors) is None


def test_slab_vote_on_lowest_disc_favours_zero(truth_anchors: Anchors) -> None:
    d = truth_anchors.discs["L5-S1"]
    v = slab_vote([_slab(np.asarray(d.center), np.asarray(d.normal))], truth_anchors)
    assert v is not None
    assert v.best == 0
    assert v.likelihood[0] == pytest.approx(0.7)
    assert v.likelihood[1] == pytest.approx(0.2)  # target L4-L5 on the lowest disc -> offset +1
    assert v.likelihood[-1] == 0.0


def test_slab_vote_on_second_disc_points_to_lumbarized(truth_anchors: Anchors) -> None:
    # Slab angled to the disc above the lowest one: L5-S1 target -> lowest disc is one level lower.
    d = truth_anchors.discs["L4-L5"]
    v = slab_vote([_slab(np.asarray(d.center), np.asarray(d.normal))], truth_anchors)
    assert v is not None
    assert v.best == -1
    assert v.likelihood[-1] == pytest.approx(0.7)
    assert v.likelihood[0] == pytest.approx(0.2)


def test_slab_vote_uses_most_inferior_slab(truth_anchors: Anchors) -> None:
    low = truth_anchors.discs["L5-S1"]
    high = truth_anchors.discs["L2-L3"]
    slabs = [
        _slab(np.asarray(high.center), np.asarray(high.normal), key="sup"),
        _slab(np.asarray(low.center), np.asarray(low.normal), key="inf"),
    ]
    v = slab_vote(slabs, truth_anchors)
    assert v is not None and v.best == 0
    assert "inf" in v.evidence[0]


def test_slab_vote_no_match_is_uninformative(truth_anchors: Anchors) -> None:
    far = np.array([0.0, 0.0, 500.0])
    v = slab_vote([_slab(far, np.array([0.0, 0.0, 1.0]))], truth_anchors)
    assert v is not None
    assert len(set(round(x, 9) for x in v.likelihood.values())) == 1


def test_slab_vote_rejects_wrong_angle(truth_anchors: Anchors) -> None:
    d = truth_anchors.discs["L5-S1"]
    # Slab at the right place but axial (0 deg) instead of the disc's 28 deg: no match.
    v = slab_vote([_slab(np.asarray(d.center), np.array([0.0, 0.0, 1.0]))], truth_anchors)
    assert v is not None
    assert len(set(round(x, 9) for x in v.likelihood.values())) == 1


# ------------------------------------------------------------------- tilt


def test_tilt_vote_lowest_most_tilted_favours_zero(truth_anchors: Anchors) -> None:
    v = tilt_vote(truth_anchors)
    assert v is not None
    assert v.best == 0
    assert "most tilted" in v.evidence[0]


def test_tilt_vote_not_most_tilted_is_weaker(truth_anchors: Anchors) -> None:
    # Make the lowest disc flat so the disc above is the most tilted.
    a = Anchors.from_dict(truth_anchors.to_dict())
    a.discs["L5-S1"].normal = [0.0, 0.0, 1.0]
    v = tilt_vote(a)
    assert v is not None
    assert "not the most tilted" in v.evidence[0]
    assert v.likelihood[0] < 0.5


def test_tilt_vote_needs_two_discs(truth_anchors: Anchors) -> None:
    a = Anchors.from_dict(truth_anchors.to_dict())
    a.discs = {"L5-S1": a.discs["L5-S1"]}
    assert tilt_vote(a) is None


# ------------------------------------------------------------------ conus


def test_conus_vote_prefers_no_shift_for_l1(truth_anchors: Anchors) -> None:
    v = conus_vote(truth_anchors)
    assert v is not None
    # Conus at L1 (index 5) is above the prior mean (4.7): shifting it up by +1 is less likely.
    assert v.likelihood[0] == pytest.approx(v.likelihood[-1])
    assert v.likelihood[1] < v.likelihood[0]


def test_conus_vote_none_without_conus(truth_anchors: Anchors) -> None:
    a = Anchors.from_dict(truth_anchors.to_dict())
    a.conus = None
    assert conus_vote(a) is None


# ---------------------------------------------------------- combination


def test_combine_without_votes_returns_prior() -> None:
    post = combine([])
    cfg = NumberingConfig()
    assert [post[o] for o in OFFSETS] == pytest.approx(list(cfg.prior_offset))


def test_combine_sums_to_one_and_respects_prior() -> None:
    v = Vote(cue="flat", likelihood={-1: 1.0, 0: 1.0, 1: 1.0})
    post = combine([v])
    assert sum(post.values()) == pytest.approx(1.0)
    assert post[0] > post[1]  # prior favours 0


def test_vote_probs_normalise() -> None:
    v = Vote(cue="x", likelihood={-1: 1.0, 0: 3.0, 1: 0.0})
    assert sum(v.probs.values()) == pytest.approx(1.0)
    assert v.best == 0


# ------------------------------------------------------ numbering_evidence


def test_phantom_without_slabs_keeps_offset_zero(truth_anchors: Anchors) -> None:
    ev = numbering_evidence(None, truth_anchors)
    assert ev.offset == 0
    assert not ev.transitional
    assert ev.transitional_kind is None
    assert ev.cues_used == ["tilt", "conus"]
    assert any("no oblique slabs" in w for w in ev.warnings)
    assert ev.not_evaluated  # honest list of cues not implemented


def test_phantom_with_correct_slab_confirms_zero(truth_anchors: Anchors) -> None:
    d = truth_anchors.discs["L5-S1"]
    ev = numbering_evidence(None, truth_anchors, slabs=[_slab(np.asarray(d.center), np.asarray(d.normal))])
    assert ev.offset == 0
    assert ev.cues_used == ["slab", "tilt", "conus"]
    assert ev.offset_confidence > NumberingConfig().prior_offset[1]


def test_lumbarized_slab_raises_minus_one_posterior(truth_anchors: Anchors) -> None:
    d = truth_anchors.discs["L4-L5"]
    slabs = [_slab(np.asarray(d.center), np.asarray(d.normal))]
    base = numbering_evidence(None, truth_anchors)
    ev = numbering_evidence(None, truth_anchors, slabs=slabs)
    assert ev.posterior[-1] > base.posterior[-1]
    assert ev.votes[0].best == -1


def test_sacralized_spine_without_slab_is_not_established(truth_anchors: Anchors) -> None:
    # Sacralized L5: the true lowest disc is L4-L5. Provisional names are one level too low (L5-S1 for the
    # lowest disc). Without a slab the tilt and conus cues cannot overturn the prior, so the module must
    # not claim transitional anatomy from them.
    a = Anchors.from_dict(truth_anchors.to_dict())
    a.discs = {_shift_down(k): v for k, v in a.discs.items() if k != "L5-S1"}
    a.conus = {"z": 0.0, "level": "L2", "confidence": 1.0}
    ev = numbering_evidence(None, a)
    assert "L5-S1" in a.discs
    assert ev.posterior[1] < 0.5
    assert not ev.transitional


def test_uncertain_warning_when_posterior_is_low() -> None:
    cfg = NumberingConfig(prior_offset=(0.34, 0.33, 0.33))
    a = Anchors()
    ev = numbering_evidence(None, a, cfg)
    assert ev.offset_confidence < cfg.uncertain_below
    assert any("uncertain" in w for w in ev.warnings)


def test_no_discs_returns_prior_and_warning() -> None:
    ev = numbering_evidence(None, Anchors())
    assert ev.offset == 0
    assert any("no discs" in w for w in ev.warnings)


def test_to_dict_is_json_serialisable(truth_anchors: Anchors) -> None:
    d = truth_anchors.discs["L5-S1"]
    ev = numbering_evidence(None, truth_anchors, slabs=[_slab(np.asarray(d.center), np.asarray(d.normal))])
    text = json.dumps(ev.to_dict())
    assert '"offset": 0' in text
    assert "not_evaluated" in text


def test_slab_matching_two_discs_up_is_uninformative(truth_anchors: Anchors) -> None:
    d = truth_anchors.discs["L3-L4"]  # third disc from the bottom: index 2
    v = slab_vote([_slab(np.asarray(d.center), np.asarray(d.normal))], truth_anchors)
    assert v is not None
    assert len(set(round(x, 9) for x in v.likelihood.values())) == 1
    assert "not used" in v.evidence[0]


def test_tie_resolves_to_offset_zero() -> None:
    v = Vote(cue="flat", likelihood={-1: 1.0, 0: 1.0, 1: 1.0})
    assert v.best == 0
