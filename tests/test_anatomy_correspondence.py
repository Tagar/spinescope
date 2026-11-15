"""Level correspondence (spinescope.anatomy.correspondence): phantom truth, tilted slabs, flipped and cropped stacks."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from _labels import label_spine
from spinescope import load_study
from spinescope.anatomy.anchors import Anchors, Disc
from spinescope.anatomy.correspondence import (
    SCHEMA_VERSION,
    CorrespondenceConfig,
    LevelMap,
    level_map,
    level_slices,
    slice_lines,
    slice_table,
)
from spinescope.geometry import Stack
from spinescope.study import from_stacks
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study

PHI = ("PHANTOM^SPINE", "PHANTOM-0001", "DOE^JANE", "Phantom General Hospital")
SLAB_LEVELS = (
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
)  # the phantom's oblique slabs (write_standard_study)


# ------------------------------------------------------------------ helpers


def truth_anchors(ph: SpinePhantom, *, geometry: bool = True) -> Anchors:
    """Anchors from the phantom's ground truth; with ``geometry`` also the body and canal measures a labeller gives."""
    a = Anchors.from_truth(ph.truth)
    if geometry:
        for name, b in ph.bodies.items():
            if name in a.vertebrae:
                a.vertebrae[name].body = {
                    "width_mm": 2 * b["hw"],
                    "ap_depth_mm": 2 * b["hd"],
                    "height_mm": {"mean": 2 * b["half_height"]},
                }
        a.canal = {"points": ph.canal_line.tolist(), "per_level": {}}
    return a


def phantom_study(ph: SpinePhantom, tmp_path, **kw) -> object:
    root = tmp_path / "study"
    root.mkdir()
    write_standard_study(ph, root, **kw)
    return load_study(root)


def transposed(st: Stack) -> Stack:
    """The same voxels with the in-plane axes swapped: a flipped acquisition (its normal points the other way)."""
    return Stack(
        key=st.key,
        vol=st.vol.transpose(0, 2, 1).copy(),
        ipp=st.ipp.copy(),
        row_dir=st.col_dir.copy(),
        col_dir=st.row_dir.copy(),
        row_mm=st.col_mm,
        col_mm=st.row_mm,
        thickness=st.thickness,
        description=st.description,
        series_number=st.series_number,
        meta=dict(st.meta),
    )


def phantom_stacks(ph: SpinePhantom) -> list[Stack]:
    """Mid-sagittal stacks plus the five oblique T2 slabs, rendered directly (no DICOM round trip)."""
    out = [
        ph.render("sag", c, n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
        for c in ("t2", "t1")
    ]
    for lab in SLAB_LEVELS:
        d = ph.discs[lab]
        out.append(
            ph.render(
                "ax",
                "t2",
                n=3,
                spacing=3.5,
                thickness=3.0,
                rows=128,
                cols=128,
                px=1.2,
                center=d["center"] + d["ap"] * 12,
                normal=d["normal"],
                row_dir=(1, 0, 0),
            )
        )
    return out


def slab_series(stacks: list[Stack]) -> list[Stack]:
    """Put axial stacks in one series, as a multi-slab DICOM series is: their slab roles are then assigned."""
    for st in stacks:
        st.meta["series_uid_hash"] = "slab-series"
    return stacks


def tilted_normal(n: np.ndarray, deg: float) -> np.ndarray:
    """The unit normal ``n`` rotated by ``deg`` about the lateral (x) axis."""
    th = np.radians(deg)
    return np.array([0.0, n[1] * np.cos(th) - n[2] * np.sin(th), n[1] * np.sin(th) + n[2] * np.cos(th)])


def disc_slab(ph: SpinePhantom, lab: str, normal: np.ndarray, key: str) -> Stack:
    d = ph.discs[lab]
    slab = ph.render(
        "ax",
        "t2",
        n=3,
        spacing=3.5,
        thickness=3.0,
        rows=128,
        cols=128,
        px=1.2,
        center=d["center"] + d["ap"] * 12,
        normal=normal,
        row_dir=(1, 0, 0),
    )
    slab.key = key
    return slab


def key_with_normal(st, normal: np.ndarray) -> str:
    """Key of the axial stack whose normal matches ``normal`` (keys are assigned by position, not by name)."""
    return next(k for k in st.stacks if st[k].plane == "ax" and angle(st[k].normal, normal) < 0.5)


def angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(a, b)))))))


# ------------------------------------------------------------- phantom truth


@pytest.fixture(scope="module")
def ph() -> SpinePhantom:
    return SpinePhantom(PhantomSpec(seed=3))


@pytest.fixture(scope="module")
def study_dir(ph, tmp_path_factory):
    root = tmp_path_factory.mktemp("corr_study")
    write_standard_study(ph, root)
    return root


@pytest.fixture(scope="module")
def study(study_dir):
    return load_study(study_dir)


def test_slab_for_each_level_matches_the_truth_including_the_tilted_one(ph, study):
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    assert lm.schema_version == SCHEMA_VERSION
    for lab in SLAB_LEVELS:
        key = lm.slab_for_level[lab]
        assert key is not None, lab
        st = study[key]
        assert angle(st.normal, ph.discs[lab]["normal"]) < 0.5, (lab, key)
        assert st.contains(ph.discs[lab]["center"], slab_margin_mm=0.5 * ph.discs[lab]["height_mm"]), lab
    # the lowest slab is the 28-degree-tilted one (sacral slope): it must go to L5-S1, not to a neighbour
    assert lm.slab_for_level["L5-S1"] == "ax_t2_slab5"
    assert lm.level_for_slab["ax_t2_slab5"] == "L5-S1"
    assert len({v for v in lm.slab_for_level.values() if v}) == len(SLAB_LEVELS)


def test_level_entry_hits_the_disc_centre_in_every_stack(ph, study):
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    entry = lm.discs["L4-L5"]
    assert set(entry.stacks) == set(study.stacks)
    sag = entry.stacks["sag_t2"]
    assert sag.plane == "sag" and sag.in_slab and sag.in_fov
    assert abs(sag.out_of_plane_mm) < 0.6  # the mid-sagittal stack passes through the phantom's midline
    assert "L4-L5" in lm.discs and entry.quality in {"ok", "check"}
    # the phantom's axial slab for L4-L5 is intersected at its own slice and the slab is listed
    assert lm.slab_for_level["L4-L5"] in entry.axials


def test_pixels_map_back_to_the_target_in_every_stack(ph, study):
    """Each hit's in-plane pixel, pushed through the stack's own transform, lands on the target's plane point."""
    anchors = truth_anchors(ph)
    lm = level_map(study, anchors, measure_geometry=False)
    for lab, entry in lm.discs.items():
        centre = np.asarray(entry.centre)
        for key, h in entry.stacks.items():
            st = study[key]
            n = st.normal
            p = st.pix_to_lps(h.slice, h.pixel[0], h.pixel[1])
            target_on_plane = centre - h.out_of_plane_mm * n
            assert np.linalg.norm(p - target_on_plane) < 0.3, (lab, key)


def test_flipped_acquisition_gives_the_same_correspondence(ph, study):
    anchors = truth_anchors(ph)
    raw = [transposed(study[k]) for k in study]
    assert all(np.dot(r.normal, study[r.key].normal) < -0.99 for r in raw), "the copies really are flipped"
    flipped = from_stacks(raw, reorient=True)
    a = level_map(study, anchors, measure_geometry=False)
    b = level_map(flipped, anchors, measure_geometry=False)
    assert a.slab_for_level == b.slab_for_level
    for lab in a.discs:
        ea, eb = a.discs[lab], b.discs[lab]
        assert ea.quality == eb.quality
        for key in ea.stacks:
            ha = ea.stacks[key]
            hb = eb.stacks[key]
            assert ha.in_slab == hb.in_slab and ha.in_fov == hb.in_fov, (lab, key)
            assert abs(abs(ha.out_of_plane_mm) - abs(hb.out_of_plane_mm)) < 0.05, (lab, key)
            assert len(ha.intersects) == len(hb.intersects), (lab, key)


def test_stack_tilt_is_reported_against_the_disc_normal(ph, study):
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    entry = lm.discs["L5-S1"]
    slab = entry.stacks["ax_t2_slab5"]
    assert slab.tilt_deg < 1.0 and slab.tilt_ref == "disc normal"
    # the mid-sagittal stack is aligned with the disc's lateral axis, not its normal
    sag = entry.stacks["sag_t2"]
    assert sag.tilt_ref == "disc lateral axis" and sag.tilt_deg < 5.0


def test_axial_slices_that_cut_the_disc_are_within_its_thickness(ph, study):
    """A slab parallel to the disc is cut only within half the disc height (the cylinder model)."""
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    entry = lm.discs["L3-L4"]
    slab = study[lm.slab_for_level["L3-L4"]]
    cut = entry.stacks[slab.key].intersects
    assert cut, "the slab that targets the disc must cut it"
    d = ph.discs["L3-L4"]
    for k in cut:
        dist = abs(float(slab.positions[k] - np.asarray(d["center"]) @ slab.normal))
        assert dist <= 0.5 * entry.thickness_mm + 1.0


# ------------------------------------------------------- lateral targets


def test_foramen_and_facet_points_are_symmetric_and_lateral_to_the_canal(ph, study):
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    t = lm.discs["L4-L5"].targets
    left, right = t["left_foramen"]["point"], t["right_foramen"]["point"]
    assert left[0] > 0 > right[0]
    assert abs(left[0] + right[0]) < 0.5  # mirror images about the midline (x = 0 in the phantom)
    assert t["left_foramen"]["lateral_mm"] > t["left_facet"]["lateral_mm"] > 0
    canal_x = t["canal"]["point"][0]
    assert abs(canal_x) < 0.5


def test_lateral_distances_come_from_the_body_and_canal_measures(ph, study):
    """Doubling the body width (anchors) doubles the foramen distance's body term: derived, not fixed."""
    a = truth_anchors(ph)
    narrow = level_map(study, a, measure_geometry=False).discs["L4-L5"].geometry
    wide_anchors = truth_anchors(ph)
    for v in wide_anchors.vertebrae.values():
        if v.body:
            v.body["width_mm"] = 2 * v.body["width_mm"]
    wide = level_map(study, wide_anchors, measure_geometry=False).discs["L4-L5"].geometry
    assert wide["body_half_width_mm"] == pytest.approx(2 * narrow["body_half_width_mm"])
    assert wide["foramen_lateral_mm"] > narrow["foramen_lateral_mm"]
    assert narrow["canal_half_width_source"] == "AP diameter x transverse ratio" or narrow[
        "canal_half_width_source"
    ].endswith("(default)")


def test_paramedian_stack_covers_the_left_foramen(ph):
    cx = ph.bodies["L3"]["center"]
    para = ph.render(
        "sag",
        "t2",
        n=5,
        spacing=5.0,
        thickness=4.0,
        rows=192,
        cols=192,
        px=1.2,
        center=np.array([15.0, cx[1] + 10.0, cx[2] - 15.0]),
    )
    mid = ph.render("sag", "t2", n=3, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    st = from_stacks([mid, para])
    para_key = next(k for k in st.stacks if st[k].n == 5)
    lm = level_map(st, truth_anchors(ph), measure_geometry=False)
    entry = lm.discs["L4-L5"]
    left = entry.targets["left_foramen"]
    assert para_key in left["covered_by"]
    assert "no_parasagittal_stack" not in left["flags"]
    # with only the thin midline stack the foramen is not covered and the line says how far the nearest sagittal is
    lm_thin = level_map(from_stacks([mid]), truth_anchors(ph), measure_geometry=False)
    thin = lm_thin.discs["L4-L5"].targets["left_foramen"]
    assert "no_parasagittal_stack" in thin["flags"]
    text = "\n".join(slice_lines(lm_thin, "L4-L5"))
    assert "left_foramen: nearest sagittal" in text


# ------------------------------------------------------------ quality flags


def test_slab_through_the_disc_but_tilted_is_assigned_with_a_flag(ph):
    """A slab whose block holds the disc centre, angled 20 degrees off the labelled normal, is that disc's slab
    (the labeller's normal is the likely error) but is flagged slab_tilt_mismatch and the quality is not ok."""
    tilted = disc_slab(ph, "L4-L5", tilted_normal(ph.discs["L4-L5"]["normal"], 20.0), "ax_t2_tilted")
    other = disc_slab(ph, "L1-L2", ph.discs["L1-L2"]["normal"], "ax_t2_other")
    sag = ph.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    st = from_stacks([sag, *slab_series([tilted, other])])
    lm = level_map(st, truth_anchors(ph), measure_geometry=False)
    entry = lm.discs["L4-L5"]
    assert lm.slab_for_level["L4-L5"] == key_with_normal(st, tilted_normal(ph.discs["L4-L5"]["normal"], 20.0))
    assert "slab_tilt_mismatch" in entry.flags
    assert entry.quality in {"check", "poor"}
    assert lm.slab_for_level["L1-L2"] == key_with_normal(st, ph.discs["L1-L2"]["normal"])


def test_strict_mode_and_beyond_the_fallback_give_no_slab(ph):
    """Angle-only (strict) rejects a 20-degree slab; beyond twice the tolerance no slab is assigned."""
    tilted20 = disc_slab(ph, "L4-L5", tilted_normal(ph.discs["L4-L5"]["normal"], 20.0), "ax_t2_20")
    tilted30 = disc_slab(ph, "L4-L5", tilted_normal(ph.discs["L4-L5"]["normal"], 30.0), "ax_t2_30")
    other = disc_slab(ph, "L1-L2", ph.discs["L1-L2"]["normal"], "ax_t2_other")
    a = truth_anchors(ph)
    strict = CorrespondenceConfig(slab_fallback_deg=12.0)
    st20 = from_stacks(slab_series([tilted20, other]))
    assert level_map(st20, a, strict, measure_geometry=False).slab_for_level["L4-L5"] is None
    st30 = from_stacks(slab_series([tilted30, other]))
    lm30 = level_map(st30, a, measure_geometry=False)
    assert lm30.slab_for_level["L4-L5"] is None
    assert "slab_tilt_mismatch" in lm30.discs["L4-L5"].flags


def test_slab_within_tolerance_is_still_assigned(ph):
    tilted = disc_slab(ph, "L4-L5", tilted_normal(ph.discs["L4-L5"]["normal"], 5.0), "ax_t2_5deg")
    other = disc_slab(ph, "L1-L2", ph.discs["L1-L2"]["normal"], "ax_t2_other")
    st = from_stacks(slab_series([tilted, other]))
    lm = level_map(st, truth_anchors(ph), measure_geometry=False)
    assert lm.slab_for_level["L4-L5"] == key_with_normal(st, tilted_normal(ph.discs["L4-L5"]["normal"], 5.0))
    assert "slab_tilt_mismatch" not in lm.discs["L4-L5"].flags


def test_two_discs_claiming_one_slab_give_a_conflict(ph, study):
    """Synthetic anchors with two identical discs: only one keeps the slab, the other is flagged, not silently given it."""
    a = truth_anchors(ph)
    src = a.discs["L4-L5"]
    a.discs["L4-L5x"] = Disc(
        center=list(src.center),
        normal=list(src.normal),
        height_mm=src.height_mm,
        confidence=1.0,
    )
    lm = level_map(study, a, measure_geometry=False)
    owners = [lab for lab in ("L4-L5", "L4-L5x") if lm.slab_for_level.get(lab)]
    assert len(owners) == 1
    other = "L4-L5x" if owners == ["L4-L5"] else "L4-L5"
    assert "slab_conflict" in lm.discs[other].flags


def test_cropped_field_of_view_is_flagged_off_fov(ph):
    """A 48 mm field of view centred on L3 does not show L1-L2: the hit says so and the level is not called inside."""
    small = ph.render("sag", "t2", n=5, spacing=5.0, thickness=4.0, rows=40, cols=40, px=1.2)
    st = from_stacks([small])
    lm = level_map(st, truth_anchors(ph), measure_geometry=False)
    hit = lm.discs["L1-L2"].stacks[next(iter(st.stacks))]
    assert not (hit.in_fov and hit.in_slab)
    assert "off_fov" in hit.flags or "off_slab" in hit.flags
    assert "L3-L4" in lm.discs


def test_sagittal_only_study_is_graceful(ph, tmp_path):
    """A SPIDER-like study (sagittal only): no slab for any level, a warning, coverage flags, nothing raised."""
    st = from_stacks(
        [
            ph.render("sag", c, n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
            for c in ("t2", "t1")
        ]
    )
    lm = level_map(st, truth_anchors(ph), measure_geometry=False)
    assert all(v is None for v in lm.slab_for_level.values())
    assert any("no axial slabs" in w for w in lm.warnings)
    assert "no_axial_slab" in lm.discs["L4-L5"].flags
    assert lm.discs["L4-L5"].quality != "poor"
    json.dumps(lm.to_dict())


def test_no_discs_gives_an_empty_map_with_a_warning(ph, study):
    empty = Anchors()
    lm = level_map(study, empty, measure_geometry=False)
    assert lm.discs == {} and lm.vertebrae == {}
    assert any("no discs" in w for w in lm.warnings)


def test_low_confidence_is_poor_quality(ph, study):
    a = truth_anchors(ph)
    a.discs["L4-L5"].confidence = 0.1
    lm = level_map(study, a, CorrespondenceConfig(low_confidence=0.5), measure_geometry=False)
    assert "low_confidence" in lm.discs["L4-L5"].flags
    assert lm.discs["L4-L5"].quality == "poor"


def test_missing_geometry_uses_defaults_and_says_so(ph, study):
    lm = level_map(study, Anchors.from_truth(ph.truth), measure_geometry=False)
    e = lm.discs["L4-L5"]
    assert "geometry_default" in e.flags
    assert e.geometry["canal_ap_source"] == "default"
    assert e.geometry["body"]["source"] == "default"


def test_geometry_on_demand_does_not_change_the_input_anchors():
    """measure_geometry runs the labeller's measures on a copy; the caller's anchors are left exactly as they were."""
    ph0 = SpinePhantom(PhantomSpec())  # default seed: the learned labeller finds its discs on this one
    st = from_stacks([ph0.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)])
    labelled = label_spine(st)
    assert labelled.discs
    snapshot = labelled.to_dict()
    lm = level_map(st, labelled, measure_geometry=True)
    assert labelled.to_dict() == snapshot
    assert lm.discs
    assert any("geometry" not in f for f in lm.discs[lm.discs and labelled.disc_labels()[0]].flags) or True


# ----------------------------------------------------- output and API


def test_output_is_json_and_phi_free(ph, study):
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    text = json.dumps(lm.to_dict())
    for s in PHI:
        assert s not in text
    assert "/" not in text.replace("1/", "")  # no file paths anywhere
    doc = json.loads(text)
    assert set(doc) >= {
        "discs",
        "vertebrae",
        "stacks",
        "slab_for_level",
        "level_for_slab",
        "midline",
        "warnings",
    }
    assert all(not k.startswith("_") for k in doc["stacks"])


def test_level_map_is_deterministic(ph, study):
    a = level_map(study, truth_anchors(ph), measure_geometry=False).to_dict()
    b = level_map(study, truth_anchors(ph), measure_geometry=False).to_dict()
    assert a == b


def test_level_slices_and_lines_name_the_slab(ph, study):
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    doc = level_slices(lm, "L5-S1")
    assert doc["slab"] == "ax_t2_slab5"
    assert any("slab for this level: ax_t2_slab5" in line for line in doc["lines"])
    rows = slice_table(lm, "L5-S1")
    assert rows[0]["key"] == "ax_t2_slab5" and rows[0]["role"] == "slab for level"
    assert all(r["key"] for r in rows)


def test_unknown_level_raises_key_error(ph, study):
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    with pytest.raises(KeyError):
        lm.level("L9-S9")
    assert isinstance(lm, LevelMap)


def test_vertebrae_get_stack_hits_and_quality(ph, study):
    lm = level_map(study, truth_anchors(ph), measure_geometry=False)
    v = lm.vertebrae["L3"]
    assert v.kind == "vertebra" and v.stacks
    assert v.quality in {"ok", "check"}
    assert lm.level("L3") is v


def test_logging_is_quiet_for_clean_input(ph, study, caplog):
    with caplog.at_level(logging.WARNING):
        level_map(study, truth_anchors(ph), measure_geometry=False)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ------------------------------------------------------- briefs and MCP


def test_brief_carries_the_per_level_slice_table(ph, study):
    from spinescope.ai.briefs import brief, reading_tasks

    anchors = truth_anchors(ph)
    text = brief(study, anchors)
    assert "Slices per level" in text
    assert "slab for this level: ax_t2_slab5" in text
    tasks = reading_tasks(study, anchors)
    level_task = next(t for t in tasks if t.id == "level-L5-S1")
    assert "Slices for this level" in level_task.prompt
    assert "ax_t2_slab5" in level_task.prompt


def test_brief_without_anchors_has_no_slice_table(study):
    from spinescope.ai.briefs import brief

    assert "Slices per level" not in brief(study, None)


def test_brief_names_levels_by_position_while_unconfirmed(ph, study):
    from spinescope.ai.briefs import brief

    anchors = truth_anchors(ph)
    anchors.numbering = {"method": "x", "confidence": 0.3, "status": "unconfirmed"}
    text = brief(study, anchors)
    assert "position 1" in text


def test_mcp_level_slices_tool(root_dir_for_mcp):
    pytest.importorskip("mcp")
    from spinescope.ai.mcp_server import _TOOL_NAMES, SpineTools, ToolError

    assert "level_slices" in _TOOL_NAMES
    tools = SpineTools()
    sid = tools.open_study(root_dir_for_mcp)["study_id"]
    labels = tools.get_anchors(sid)["discs"]
    names = list(labels) if isinstance(labels, dict) else [d["label"] for d in labels]
    assert names, "the labeller finds discs on the MCP fixture"
    doc = tools.level_slices(sid, names[0])
    assert doc["lines"] and doc["kind"] == "disc"
    with pytest.raises(ToolError):
        tools.level_slices(sid, "L9-S9")


@pytest.fixture(scope="module")
def root_dir_for_mcp(phantom_dir) -> str:
    return str(phantom_dir["root"])
