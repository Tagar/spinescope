"""Unit tests for the realistic phantom (``PhantomSpec(realism="realistic")``): physics, morphometry and anatomy.

The cartoon phantom is covered by ``test_phantom_options.py``; these tests check the realistic look, which is
built from published relaxation times, spin-echo equations and morphometry (see ``testing/realistic.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spinescope.testing import realistic as R
from spinescope.testing.phantom import (
    ACQ,
    AIR,
    ANNULUS,
    BOWEL,
    CORD,
    CORTEX,
    CSF,
    DURA,
    FACET,
    FAT,
    KIDNEY,
    LIGAMENT,
    LIVER,
    MARROW,
    MUSCLE,
    NERVE,
    NUCLEUS,
    PSOAS,
    REMNANT,
    SOFT,
    TISSUES,
    VESSEL,
    PhantomSpec,
    SpinePhantom,
)

SMALL = {"n": 3, "spacing": 5.0, "thickness": 4.0, "rows": 128, "cols": 128, "px": 1.2}


@pytest.fixture(scope="module")
def real() -> SpinePhantom:
    return SpinePhantom(PhantomSpec(realism="realistic"))


def test_realism_is_validated() -> None:
    with pytest.raises(ValueError, match="realism"):
        SpinePhantom(PhantomSpec(realism="photoreal"))


def test_default_is_cartoon_and_realistic_has_the_same_levels(real: SpinePhantom) -> None:
    cartoon = SpinePhantom(PhantomSpec())
    assert cartoon.spec.realism == "cartoon" and not cartoon.realistic
    assert real.realistic
    assert list(real.truth["discs"]) == list(cartoon.truth["discs"])
    assert set(real.bodies) == set(cartoon.bodies)


def test_every_tissue_has_a_relaxation_entry() -> None:
    assert set(R.RELAXATION) == set(range(len(TISSUES)))


def test_spin_echo_signal_follows_the_equation() -> None:
    rho, t1, t2, tr, te = 0.8, 500.0, 100.0, 2000.0, 20.0
    expected = rho * (1 - math.exp(-tr / t1)) * math.exp(-te / t2)
    assert R.spin_echo_signal(rho, t1, t2, tr, te) == pytest.approx(expected)
    # long TR and TE = 0: full recovery, no decay
    assert R.spin_echo_signal(1.0, 300.0, 50.0, 1e9, 0.0) == pytest.approx(1.0)


def test_inversion_recovery_nulls_at_ln2_t1() -> None:
    t1 = 260.0
    s = R.inversion_recovery_signal(1.0, t1, 100.0, 1e9, 0.0, t1 * math.log(2))
    assert s == pytest.approx(0.0, abs=1e-9)


def test_signal_table_orders_tissues_like_the_literature() -> None:
    t1 = R.signal_table(ACQ["t1"], 1.5)
    t2 = R.signal_table(ACQ["t2"], 1.5)
    stir = R.signal_table(ACQ["stir"], 1.5)
    assert t1[FAT] > t1[MUSCLE] > t1[CORTEX]  # fat is the brightest on T1, cortex the darkest
    assert t2[CSF] == max(t2.values())  # CSF is the brightest on T2
    assert t2[NUCLEUS] > t2[ANNULUS] > t2[CORTEX]  # healthy nucleus brighter than annulus on T2
    assert stir[FAT] < stir[MUSCLE]  # STIR suppresses fat (TI is not exactly the fat null, so not zero)
    assert t2[VESSEL] < 0.05 * t2[CSF]  # flow void


def test_3t_changes_relaxation_times() -> None:
    a = R.signal_table(ACQ["t1"], 1.5)
    b = R.signal_table(ACQ["t1"], 3.0)
    assert a[MUSCLE] != b[MUSCLE]


def test_morphometry_has_lumbar_gradients_and_a_normal_lordosis(real: SpinePhantom) -> None:
    heights = [real.bodies[n]["half_height"] * 2 for n in ("L1", "L2", "L3", "L4", "L5")]
    assert heights == sorted(heights)  # bodies get taller down the lumbar spine
    dh = [real.discs[d]["height_mm"] for d in ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")]
    assert dh == sorted(dh)  # discs get taller too
    assert real.discs["L5-S1"]["tilt_deg"] == pytest.approx(R.SACRAL_SLOPE_DEG)
    lordosis = real.discs["L5-S1"]["tilt_deg"] - real.discs["T12-L1"]["tilt_deg"]
    assert 40.0 <= lordosis <= 60.0  # L1 superior endplate to S1: normal range (Roussouly 2005)


def test_realistic_geometry_has_a_sacrum_of_five_segments(real: SpinePhantom) -> None:
    names = [s["name"] for s in real._sacral_segments]
    assert names == ["S2", "S3", "S4", "S5"]
    # each sacral segment is tilted further towards vertical than the one above it
    tilts = [s["tilt_deg"] for s in real._sacral_segments]
    assert tilts == sorted(tilts, reverse=True)
    assert (
        real.thecal_end_z > real._sacral_segments[-1]["center"][2]
    )  # the dural sac ends above the sacral canal


def test_every_structure_of_the_realistic_anatomy_is_drawn(real: SpinePhantom) -> None:
    # random points in coronal bands through the midline, the canal, the psoas, the kidney/liver and the iliac wing
    rng = np.random.default_rng(0)
    found: set[int] = set()
    for center_x in (-60.0, -12.0, 0.0, 12.0, 20.0, 30.0, 60.0, 80.0):
        q = rng.uniform(-1, 1, size=(30000, 3)) * np.array([10.0, 120.0, 230.0])
        q += np.array([center_x, real.bodies["L3"]["center"][1], real.bodies["L3"]["center"][2]])
        found |= set(np.unique(real.labels(q)).tolist())
    for tissue in (AIR, SOFT, FAT, MUSCLE, PSOAS, MARROW, CORTEX, ANNULUS, NUCLEUS, CSF, CORD, NERVE, DURA,
                   LIGAMENT, FACET, VESSEL, BOWEL, LIVER, KIDNEY, REMNANT):  # fmt: skip
        assert tissue in found, TISSUES[tissue]


def test_discs_are_labelled_where_the_truth_says(real: SpinePhantom) -> None:
    for name, v in real.truth["discs"].items():
        lab = real.labels(np.asarray(v["center"])[None, :])[0]
        assert lab in (ANNULUS, NUCLEUS), name


def test_partial_volume_averages_the_thickness_and_keeps_the_range() -> None:
    ph = SpinePhantom(PhantomSpec(realism="realistic", supersample=(1, 1)))
    sharp = ph.render("sag", "t2", n=3, spacing=5.0, thickness=4.0, rows=96, cols=96, px=1.2).vol
    ph2 = SpinePhantom(PhantomSpec(realism="realistic", supersample=(4, 3)))
    soft = ph2.render("sag", "t2", n=3, spacing=5.0, thickness=4.0, rows=96, cols=96, px=1.2).vol
    assert not np.array_equal(sharp, soft)
    lo, hi = 0.0, max(R.signal_table(ACQ["t2"], 1.5).values()) * 1.6  # noise allowance
    assert float(soft.min()) >= lo and float(soft.max()) <= hi


def test_gibbs_ringing_overshoots_a_sharp_edge() -> None:
    img = np.zeros((64, 64))
    img[:, 32:] = 1.0
    out = R.gibbs_ringing(img, keep=0.5)
    assert out.max() > 1.0  # the ringing overshoot beyond the step
    assert out.min() >= 0.0


def test_render_is_deterministic_and_seeded() -> None:
    a = SpinePhantom(PhantomSpec(realism="realistic")).render("sag", "t2", **SMALL).vol
    b = SpinePhantom(PhantomSpec(realism="realistic")).render("sag", "t2", **SMALL).vol
    c = SpinePhantom(PhantomSpec(realism="realistic", seed=7)).render("sag", "t2", **SMALL).vol
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_smooth_field_is_bounded_and_centred() -> None:
    field = R.smooth_field(np.random.default_rng(1).uniform(-100, 100, size=(5000, 3)), seed=0)
    assert float(np.abs(field).max()) < 4.0  # a sum of plane waves, normalised to about unit variance
    assert 0.5 < float(field.std()) < 1.5
    assert abs(float(field.mean())) < 0.2


def test_gibbs_option_changes_the_render() -> None:
    plain = SpinePhantom(PhantomSpec(realism="realistic")).render("sag", "t1", **SMALL).vol
    ringed = SpinePhantom(PhantomSpec(realism="realistic", gibbs=True)).render("sag", "t1", **SMALL).vol
    assert not np.array_equal(plain, ringed)
    assert math.isfinite(float(ringed.mean()))


def test_realistic_options_keep_working(real: SpinePhantom) -> None:
    spec = PhantomSpec(
        realism="realistic",
        transitional="lumbarised_s1",
        degenerated={"L4-L5": 0.2},
        modic1=[("L4-L5", "inf", "L")],
        herniation={"L5-S1": ("central", 5.0)},
        fracture={"L2": 0.45},
        schmorl=[("L3", "sup")],
        collapsed={"L3-L4": 0.5},
        hardware=("cage:L4-L5", "screws:L4", "rod"),
        ribs=True,
        bias_field=0.5,
        field_strength=3.0,
    )
    ph = SpinePhantom(spec)
    st = ph.render("sag", "t2", **SMALL)
    assert st.vol.shape == (3, 128, 128)
    assert ph.truth["discs"]["L6-S1"]["anatomical"] == "S1-S2"
    assert np.isfinite(st.vol).all()


def test_sacral_slope_and_segments_are_overridable() -> None:
    ph = SpinePhantom(PhantomSpec(realism="realistic", sacral_slope=30.0))
    assert ph.discs["L5-S1"]["tilt_deg"] == pytest.approx(30.0)
    ph_c = SpinePhantom(PhantomSpec(sacral_slope=None))
    assert ph_c.discs["L5-S1"]["tilt_deg"] == pytest.approx(28.0)


def test_cartoon_truth_is_unchanged_by_the_realism_option() -> None:
    a = SpinePhantom(PhantomSpec()).truth
    b = SpinePhantom(PhantomSpec(realism="cartoon")).truth
    assert a["discs"].keys() == b["discs"].keys()
    assert a["vertebrae"] == b["vertebrae"]
    assert (
        "hw" not in a["discs"]["L5-S1"]
    )  # the realistic outline keys are only added for the realistic phantom
