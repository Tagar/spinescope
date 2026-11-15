"""Learned labeller (spinescope.anatomy.ml): geometry, post-processing, runtime contract and SPIDER runs.

Most tests need no network: post-processing is driven by probability maps built from the phantom's
analytic labels (an oracle), so the geometry and numbering are tested independently of the model.
Tests that need the packaged weights and ONNX Runtime are skipped when those are unavailable.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from spinescope import load_study
from spinescope.anatomy import label_spine_ml
from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.ml import model_path
from spinescope.anatomy.ml.grid import intensity_window, normalise, plane_grid, resample_slice
from spinescope.anatomy.ml.labeller import BODY, CANAL, DISC, SegmentedStack
from spinescope.anatomy.ml.postprocess import (
    anchors_from_segmentation,
    discs_near_midline,
    extract_discs,
    mid_sagittal,
)
from spinescope.config import DEFAULT
from spinescope.geometry import Stack
from spinescope.study import from_stacks
from spinescope.testing.phantom import (
    ANNULUS,
    CORD,
    CORTEX,
    CSF,
    MARROW,
    MODIC1,
    NUCLEUS,
    PhantomSpec,
    SpinePhantom,
)

# Phantom tissues -> network classes (the SPIDER convention: canal = thecal sac, discs = annulus+nucleus).
_TO_CLASS = {MARROW: BODY, CORTEX: BODY, MODIC1: BODY, ANNULUS: DISC, NUCLEUS: DISC, CSF: CANAL, CORD: CANAL}


def _oracle_segmentation(stack: Stack, phantom: SpinePhantom, res: float = 1.0) -> SegmentedStack:
    """One-hot probabilities from the phantom's analytic tissue labels on each slice's grid."""
    grids = [plane_grid(stack, k, res) for k in range(stack.n)]
    h, w = grids[0].shape
    probs = np.zeros((stack.n, 4, h, w), np.float32)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for k, g in enumerate(grids):
        pts = g.lps_of_index(ii, jj).reshape(-1, 3)
        tissue = phantom.labels(pts).reshape(h, w)
        cls = np.zeros((h, w), np.int64)
        for t, c in _TO_CLASS.items():
            cls[tissue == t] = c
        for c in range(4):
            probs[k, c] = cls == c
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=(0.0, 1.0))


def _match_error(anchors: Anchors, truth: Anchors) -> dict[str, float]:
    """Per truth disc: distance (mm) from the predicted disc with the same label, or NaN if missing."""
    out: dict[str, float] = {}
    for name, t in truth.discs.items():
        if name in anchors.discs:
            d = anchors.discs[name]
            out[name] = float(np.linalg.norm(np.asarray(d.center) - np.asarray(t.center)))
        else:
            out[name] = float("nan")
    return out


# ------------------------------------------------------------------------------ geometry
def test_plane_grid_shape_and_roundtrip(phantom: SpinePhantom) -> None:
    stack = phantom.render("sag", "t2")
    g = plane_grid(stack, 3, 1.0)
    assert g.shape[0] % 16 == 0 and g.shape[1] % 16 == 0
    ii, jj = np.meshgrid(np.arange(5), np.arange(7), indexing="ij")
    lps = g.lps_of_index(ii, jj)
    down, post = g.mm_of_index(ii, jj)
    back = g.mm_of(lps)
    assert np.allclose(back[..., 0], down, atol=1e-6)
    assert np.allclose(back[..., 1], post, atol=1e-6)
    # Points stay on the slice plane (same x for a sagittal stack).
    assert np.allclose(lps[..., 0], stack.ipp[3][0], atol=1e-6)


def test_grid_is_superior_top_and_anterior_left(phantom: SpinePhantom) -> None:
    stack = phantom.render("sag", "t2")
    g = plane_grid(stack, 0, 1.0)
    top = g.lps_of_index(0, 0)
    bottom = g.lps_of_index(g.shape[0] - 1, 0)
    ant = g.lps_of_index(0, 0)
    post = g.lps_of_index(0, g.shape[1] - 1)
    assert top[2] > bottom[2]  # row 0 is superior
    assert post[1] > ant[1]  # columns run anterior -> posterior (LPS +y is posterior)


def test_resample_constant_inside_zero_outside(phantom: SpinePhantom) -> None:
    """Outside the source slice the resampled grid is zero (no clamping to edge values)."""
    stack = phantom.render("sag", "t2")
    stack.vol[:] = 7.0
    g = plane_grid(stack, 2, 1.0)
    out = resample_slice(stack, g, order=1)
    assert out.shape == g.shape
    assert np.isclose(out.max(), 7.0) and out.min() >= 0.0
    # Shift the grid far outside the slice: all zeros.
    far = plane_grid(stack, 2, 1.0)
    far = type(far)(**{**far.__dict__, "down0": far.down0 + 10_000.0})
    assert np.allclose(resample_slice(stack, far, order=1), 0.0)


def test_intensity_window_and_normalise() -> None:
    vol = np.zeros((4, 50, 50), np.float32)
    vol[:, 10:40, 10:40] = np.linspace(100, 900, 30 * 30, dtype=np.float32).reshape(30, 30)
    lo, hi = intensity_window(vol)
    assert 100 <= lo < hi <= 900
    n = normalise(vol, (lo, hi))
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert n.dtype == np.float32


# ------------------------------------------------------------------- post-processing (oracle)
@pytest.mark.parametrize(
    "spec",
    [
        PhantomSpec(),
        PhantomSpec(segment_lordosis=(10.0, 12.0, 14.0, 16.0, 18.0, 22.0)),  # strong lordosis
        PhantomSpec(degenerated={"L4-L5": 0.2}, modic1=[("L4-L5", "inf", "L")]),
        PhantomSpec(noise=0.06, seed=3),
    ],
    ids=["default", "lordosis", "degenerated", "noisy"],
)
def test_oracle_labels_recover_levels_and_numbering(spec: PhantomSpec) -> None:
    ph = SpinePhantom(spec)
    stack = ph.render("sag", "t2", rows=320)  # all seven discs inside the field of view
    seg = _oracle_segmentation(stack, ph)
    anchors = anchors_from_segmentation(seg)
    truth = Anchors.from_truth(ph.truth)
    assert set(anchors.discs) == set(truth.discs), anchors.warnings
    errs = _match_error(anchors, truth)
    assert max(errs.values()) < 4.0, errs
    assert anchors.numbering["confidence"] > 0.5  # the phantom shows a body below L5-S1
    assert anchors.midline_x is not None


def test_oracle_flipped_acquisition_gives_same_labels(phantom: SpinePhantom) -> None:
    """Row/column directions reversed (a flipped acquisition) do not change the answer."""
    stack = phantom.render("sag", "t2", rows=320)
    flipped = phantom.render(
        "sag", "t2", rows=320, row_dir=-np.asarray(stack.row_dir)
    )  # 180 degrees in plane
    a = anchors_from_segmentation(_oracle_segmentation(stack, phantom))
    b = anchors_from_segmentation(_oracle_segmentation(flipped, phantom))
    assert set(a.discs) == set(b.discs)
    for name in a.discs:
        assert np.linalg.norm(np.asarray(a.discs[name].center) - np.asarray(b.discs[name].center)) < 2.0


def test_no_canal_gives_warning_not_exception(phantom: SpinePhantom) -> None:
    stack = phantom.render("sag", "t2")
    seg = _oracle_segmentation(stack, phantom)
    seg.probs[:, CANAL] = 0.0
    anchors = anchors_from_segmentation(seg)
    assert anchors.discs == {}
    assert anchors.numbering["confidence"] == 0.0
    assert any("canal" in w for w in anchors.warnings)


def test_mid_sagittal_is_the_canal_peak(phantom: SpinePhantom) -> None:
    """The phantom canal is centred on x = 0: the chosen plane must be the slice nearest to it."""
    stack = phantom.render("sag", "t2", n=15, spacing=2.0)
    mid = mid_sagittal(_oracle_segmentation(stack, phantom))
    assert mid is not None
    assert abs(mid.x_lps) <= 1.5, mid.x_lps
    assert mid.confidence > 0.5


def test_extract_discs_rejects_non_disc_shapes(phantom: SpinePhantom) -> None:
    stack = phantom.render("sag", "t2")
    seg = _oracle_segmentation(stack, phantom)
    k = stack.n // 2
    # A square 40 x 40 mm blob is not a disc (aspect 1): it must be rejected.
    seg.probs[k, DISC] = 0.0
    g = seg.grids[k]
    blob = np.zeros(g.shape, bool)
    cy, cx = g.shape[0] // 2, g.shape[1] // 2
    blob[cy - 20 : cy + 20, cx - 20 : cx + 20] = True
    seg.probs[k, DISC][blob] = 1.0
    discs, warns = extract_discs(seg, k)
    assert discs == []
    assert any("not disc-shaped" in w for w in warns)


def test_discs_near_midline_merges_fragments(phantom: SpinePhantom) -> None:
    stack = phantom.render("sag", "t2")
    seg = _oracle_segmentation(stack, phantom)
    k = int(np.argmax([(p[CANAL] > 0.5).sum() for p in seg.probs]))
    discs, _ = discs_near_midline(seg, k)
    centres = [np.asarray(d.center) for d in discs]
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            assert np.linalg.norm(centres[i][[1, 2]] - centres[j][[1, 2]]) >= DEFAULT.level_spacing_mm[0] / 2


# ------------------------------------------------------------------- runtime contract
def test_label_spine_ml_without_sagittal_warns(phantom: SpinePhantom) -> None:
    ax = phantom.render("ax", "t2")
    study = from_stacks([ax])
    anchors = label_spine_ml(study)
    assert anchors.discs == {}
    assert anchors.warnings and "sagittal" in anchors.warnings[0]


def test_runtime_does_not_import_torch() -> None:
    code = (
        "import sys, spinescope.anatomy as a, spinescope.anatomy.ml as m;"
        "assert 'torch' not in sys.modules and 'onnx' not in sys.modules, 'torch leaked into runtime';"
        "assert callable(a.label_spine_ml) and callable(m.label_spine_ml)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_model_asset_is_packaged_and_small() -> None:
    path = model_path()
    if not path.is_file():
        pytest.skip("labeller weights not packaged yet")
    assert path.stat().st_size <= 10 * 1024 * 1024
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert sess.get_inputs()[0].name == "image"
    out = sess.run(None, {"image": np.zeros((1, 1, 64, 48), np.float32)})[0]
    assert out.shape == (1, 4, 64, 48)


def test_label_spine_ml_on_phantom(phantom_study) -> None:  # type: ignore[no-untyped-def]
    """Out-of-distribution check: the phantom is synthetic, so this asserts only the contract."""
    pytest.importorskip("onnxruntime")
    if not model_path().is_file():
        pytest.skip("labeller weights not packaged yet")
    anchors = label_spine_ml(phantom_study)
    assert isinstance(anchors, Anchors)
    assert anchors.method.startswith("label_spine_ml")
    assert 0.0 <= float(anchors.numbering.get("confidence", 0.0)) <= 1.0
    anchors.to_json()  # serialisable, PHI-free


# ----------------------------------------------------------------------- SPIDER (dataset)
def _spider_root() -> Path | None:
    base = Path(os.environ.get("SPINESCOPE_DATA", Path.home() / "data" / "public"))
    root = base / "spider"
    return root if (root / "images").is_dir() and (root / "masks").is_dir() else None


@pytest.mark.dataset
def test_spider_validation_sample_runs_ml_backend() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("SimpleITK")
    root = _spider_root()
    if root is None or not model_path().is_file():
        pytest.skip("SPIDER or labeller weights not available")
    from spinescope.anatomy._validate import aggregate, score_anchors, truth_from_mask

    with open(root / "overview.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    names = [r["new_file_name"] for r in rows if r["subset"] == "validation"][:3]
    scores = []
    for nm in names:
        truth, mx = truth_from_mask(str(root / "masks" / f"{nm}.mha"))
        anchors = label_spine_ml(load_study(str(root / "images" / f"{nm}.mha")))
        scores.append(score_anchors(anchors, truth, mx))
    m = aggregate(scores)
    assert m["volumes"] == len(names)
    assert 0.0 <= m["recall"] <= 1.0
