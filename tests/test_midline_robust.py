"""Phantom tests for the mirror-symmetry mid-sagittal estimate (spinescope.anatomy.midline_robust).

The phantom is straight and symmetric, so the variants that matter (a shifted spine, another contrast,
a cropped or flipped field) are made with the phantom's own parameters. Scoliosis and in-plane rotation
are local subclasses here, since the phantom has no such parameter. Their truth is computed from the
phantom's own CSF voxels, the same quantity the SPIDER canal mask measures.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from spinescope import load_study
from spinescope.anatomy import label_spine
from spinescope.anatomy.midline_robust import (
    MidlineEstimate,
    estimate_midsagittal,
    estimate_stack_midline,
)
from spinescope.config import Config
from spinescope.study import from_stacks
from spinescope.testing.phantom import CSF, PhantomSpec, SpinePhantom, write_standard_study


class ScoliosisPhantom(SpinePhantom):
    """Anatomy displaced laterally by ``amp * sin(...)`` along z: a curved (scoliotic) spine."""

    def __init__(self, spec: PhantomSpec, amp: float, wavelength: float = 200.0, z0: float = -60.0) -> None:
        self.amp, self.wavelength, self.z0 = amp, wavelength, z0
        super().__init__(spec)

    def lateral(self, z: np.ndarray) -> np.ndarray:
        return self.amp * np.sin(2 * np.pi * (np.asarray(z) - self.z0) / self.wavelength)

    def labels(self, pts: np.ndarray) -> np.ndarray:
        p = np.array(pts, dtype=np.float64)
        p[:, 0] -= self.lateral(p[:, 2])
        return super().labels(p)


class RotatedPhantom(SpinePhantom):
    """Whole anatomy rotated about a vertical axis through (0, y0): axial rotation of the patient."""

    def __init__(self, spec: PhantomSpec, theta_deg: float, y0: float = 60.0) -> None:
        self.theta = np.radians(theta_deg)
        self.y0 = y0
        super().__init__(spec)

    def labels(self, pts: np.ndarray) -> np.ndarray:
        p = np.array(pts, dtype=np.float64)
        x, y = p[:, 0], p[:, 1] - self.y0
        c, s = np.cos(self.theta), np.sin(self.theta)
        q = p.copy()
        q[:, 0] = c * x + s * y
        q[:, 1] = -s * x + c * y + self.y0
        return super().labels(q)


def _sag(ph: SpinePhantom, contrast: str = "t2", **kw: object) -> object:
    args: dict[str, object] = dict(n=13, spacing=5.0, thickness=4.0, rows=192, cols=192, px=1.2)
    args.update(kw)
    return ph.render("sag", contrast, **args)  # type: ignore[arg-type]


def _csf_median_x(ph: SpinePhantom, zlo: float = -95.0, zhi: float = 130.0) -> float:
    """Median LPS x of the phantom's CSF voxels: the truth the SPIDER canal mask gives."""
    xs = np.arange(-30.0, 31.0, 2.0)
    ys = np.arange(20.0, 120.0, 2.0)
    zs = np.arange(zlo, zhi, 3.0)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    return float(np.median(grid[ph.labels(grid) == CSF, 0]))


def _csf_curve_x(ph: SpinePhantom, z: float, halfwidth: float = 6.0) -> float:
    """Median LPS x of the phantom's CSF voxels within ``halfwidth`` mm of superior position ``z``."""
    xs = np.arange(-30.0, 31.0, 1.0)
    ys = np.arange(20.0, 120.0, 1.0)
    zs = np.arange(z - halfwidth, z + halfwidth + 1e-9, 2.0)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    return float(np.median(grid[ph.labels(grid) == CSF, 0]))


# ----------------------------------------------------------------- straight and shifted spines
def test_straight_phantom_midline_is_at_zero():
    ph = SpinePhantom(PhantomSpec())
    est = estimate_stack_midline(_sag(ph))
    assert est.x_mm is not None
    assert abs(est.x_mm) <= 1.0
    assert est.confidence > 0.5
    assert est.warnings == []


@pytest.mark.parametrize("s1_x", [8.0, -10.0])
def test_shifted_spine_is_found(s1_x: float):
    ph = SpinePhantom(PhantomSpec(s1_top=(s1_x, 45.0, -85.0)))
    est = estimate_stack_midline(_sag(ph))
    assert est.x_mm is not None
    assert abs(est.x_mm - ph.truth["midline_x"]) <= 1.5


@pytest.mark.parametrize("contrast", ["t1", "t2", "stir"])
def test_every_contrast_gives_the_same_midline(contrast: str):
    ph = SpinePhantom(PhantomSpec(s1_top=(8.0, 45.0, -85.0)))
    est = estimate_stack_midline(_sag(ph, contrast))
    assert est.x_mm is not None
    assert abs(est.x_mm - 8.0) <= 1.5


def test_cropped_field_still_finds_midline():
    ph = SpinePhantom(PhantomSpec(s1_top=(8.0, 45.0, -85.0)))
    est = estimate_stack_midline(_sag(ph, rows=160, cols=96))
    assert est.x_mm is not None
    assert abs(est.x_mm - 8.0) <= 1.5


def test_flipped_row_direction_within_3mm():
    # The phantom's body outline is centred on x=0 while the spine is shifted, so a flipped raw
    # acquisition gives a 2 mm error here (load_study normalises display orientation in practice).
    ph = SpinePhantom(PhantomSpec(s1_top=(8.0, 45.0, -85.0)))
    est = estimate_stack_midline(_sag(ph, row_dir=(0.0, -1.0, 0.0)))
    assert est.x_mm is not None
    assert abs(est.x_mm - 8.0) <= 3.0


# ----------------------------------------------------------------- curved and rotated spines
def test_scoliotic_curve_is_followed():
    ph = ScoliosisPhantom(PhantomSpec(), amp=8.0)
    est = estimate_stack_midline(_sag(ph, n=13, spacing=5.0))
    assert est.x_mm is not None
    for z in (-60.0, 0.0, 60.0):
        assert abs(float(est.x_at(z)) - float(ph.lateral(z))) <= 3.0
    assert abs(est.x_mm - _csf_median_x(ph)) <= 3.0


def test_in_plane_rotation_bias_is_bounded():
    # Known limit (docs/labeling/lab05_midline.md): a sagittal stack cannot represent axial tilt of the
    # mid-sagittal plane, so the estimate is biased toward the cross-section average (about 4 mm at 8 deg).
    ph = RotatedPhantom(PhantomSpec(), theta_deg=8.0)
    est = estimate_stack_midline(_sag(ph))
    assert est.x_mm is not None
    assert abs(est.x_mm - _csf_median_x(ph)) <= 6.0


# ----------------------------------------------------------------- study level and inputs
def test_axial_slices_refine_the_midline_when_present():
    ph = SpinePhantom(PhantomSpec())
    study = load_study(write_standard_study(ph, tempfile.mkdtemp())["root"])
    est = estimate_midsagittal(study)
    assert est.x_mm is not None
    assert est.axial_x_mm is not None
    assert abs(est.x_mm - ph.truth["midline_x"]) <= 1.0
    assert any("axial" in e for e in est.evidence)


def test_study_without_sagittal_gives_no_estimate_and_a_warning():
    ph = SpinePhantom(PhantomSpec())
    study = from_stacks([ph.render("ax", "t2", n=20, spacing=5.0, thickness=4.0, rows=96, cols=96, px=1.4)])
    est = estimate_midsagittal(study)
    assert est.x_mm is None
    assert est.warnings


def test_two_sagittal_slices_give_no_estimate():
    ph = SpinePhantom(PhantomSpec())
    est = estimate_stack_midline(_sag(ph, n=2))
    assert est.x_mm is None
    assert est.warnings
    assert est.x_at([0.0])[0] != est.x_at([0.0])[0]  # NaN when there is no estimate


def test_non_sagittal_stack_gives_no_estimate():
    ph = SpinePhantom(PhantomSpec())
    est = estimate_stack_midline(
        ph.render("ax", "t2", n=20, spacing=5.0, thickness=4.0, rows=96, cols=96, px=1.4)
    )
    assert est.x_mm is None
    assert est.warnings


def test_slice_weights_peak_at_the_slice_nearest_the_midline():
    ph = SpinePhantom(PhantomSpec(s1_top=(8.0, 45.0, -85.0)))
    st = _sag(ph)
    est = estimate_stack_midline(st)  # type: ignore[arg-type]
    assert est.x_mm is not None
    w = est.slice_weights
    assert w.shape == (st.n,)  # type: ignore[attr-defined]
    assert np.all((w >= 0) & (w <= 1))
    xs = np.asarray(st.ipp[:, 0])  # type: ignore[attr-defined]
    assert abs(xs[int(np.argmax(w))] - est.x_mm) <= 5.0


def test_estimate_is_deterministic():
    ph = SpinePhantom(PhantomSpec(s1_top=(8.0, 45.0, -85.0)))
    a = estimate_stack_midline(_sag(ph))  # type: ignore[arg-type]
    b = estimate_stack_midline(_sag(ph))  # type: ignore[arg-type]
    assert a.x_mm == b.x_mm
    assert a.knots_x == b.knots_x


def test_x_at_without_knots_is_constant():
    est = MidlineEstimate(stack_key="k", x_mm=3.5)
    assert np.allclose(est.x_at([-50.0, 0.0, 50.0]), 3.5)
    none = MidlineEstimate(stack_key="k", x_mm=None)
    assert np.all(np.isnan(none.x_at([0.0])))


def test_label_spine_midline_uses_symmetry_estimate():
    ph = SpinePhantom(PhantomSpec(s1_top=(8.0, 45.0, -85.0)))
    study = from_stacks([_sag(ph), _sag(ph, "t1")])  # type: ignore[list-item]
    anc = label_spine(study)
    assert anc.midline_x is not None
    assert abs(anc.midline_x - 8.0) <= 1.5


def test_ridge_method_still_runs_for_comparison():
    ph = SpinePhantom(PhantomSpec())
    study = from_stacks([_sag(ph)])  # type: ignore[list-item]
    anc = label_spine(study, Config(midline_method="ridge"))
    assert anc.warnings is not None  # runs; the ridge midline is not asserted (it is the baseline)
