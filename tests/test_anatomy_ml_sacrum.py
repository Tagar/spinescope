"""Sacrum class and field-of-view checks of the learned labeller (labeller_v2), network-free where possible.

The pseudo-label recipe (tools/train_labeller.py) is tested on small label grids. Numbering confirmation is tested
with probability maps built from the phantom's analytic labels plus a sacrum channel, so no weights are needed.
The augmentation test needs PyTorch (the ``train`` extra) and is skipped without it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spinescope.anatomy.ml import postprocess as pp
from spinescope.anatomy.ml.grid import plane_grid
from spinescope.anatomy.ml.labeller import BODY, CANAL, DISC, SACRUM, SegmentedStack
from spinescope.anatomy.ml.postprocess import anchors_from_segmentation, sacrum_below_lowest_disc
from spinescope.config import DEFAULT
from spinescope.testing.phantom import ANNULUS, CORD, CORTEX, CSF, MARROW, MODIC1, NUCLEUS, SpinePhantom

_TO_CLASS = {MARROW: BODY, CORTEX: BODY, MODIC1: BODY, ANNULUS: DISC, NUCLEUS: DISC, CSF: CANAL, CORD: CANAL}


def _load_tool() -> object:
    path = Path(__file__).resolve().parents[1] / "tools" / "train_labeller.py"
    spec = importlib.util.spec_from_file_location("train_labeller_tool", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------ pseudo-label on label grids
def _grid_with_disc(rows: int = 120, cols: int = 100) -> np.ndarray:
    raw = np.zeros((rows, cols), np.int64)
    raw[30:40, 30:70] = 1  # L5 body above the disc (rows increase inferiorly)
    raw[40:46, 35:65] = 201  # lowest disc, inferior edge at row 45
    raw[46:, 35:65] = 0  # below the disc: sacrum (not labelled in SPIDER)
    raw[:, 80:85] = 100  # canal posterior to the bodies (outside the disc columns)
    return raw


def test_sacrum_box_starts_below_disc_and_spans_its_columns() -> None:
    tool = _load_tool()
    raw = _grid_with_disc()
    covered = np.ones(raw.shape, bool)
    m = tool.sacrum_pseudo_label(raw, covered, 1.0)
    rows = np.nonzero(m.any(axis=1))[0]
    cols = np.nonzero(m.any(axis=0))[0]
    assert rows.min() == 45 + 1 + 2  # 2 mm gap below the disc's inferior edge
    assert rows.max() == rows.min() + 40 - 1  # capped at 40 mm
    assert cols.min() == 35 and cols.max() == 64
    assert not m[raw == 201].any() and not (m & (raw == 100)).any()


def test_sacrum_box_cut_by_field_of_view_and_rejected_when_too_short() -> None:
    tool = _load_tool()
    raw = _grid_with_disc()
    covered = np.zeros(raw.shape, bool)
    covered[:, :] = True
    covered[52:, :] = False  # FOV ends 6 mm below the gap start: shorter than 8 mm, rejected
    assert not tool.sacrum_pseudo_label(raw, covered, 1.0).any()
    covered[:, :] = True
    covered[70:, :] = False  # 70 - 48 = 22 mm visible: kept, cut at the FOV edge
    m = tool.sacrum_pseudo_label(raw, covered, 1.0)
    assert m.any() and np.nonzero(m.any(axis=1))[0].max() == 69


def test_sacrum_rejected_without_lowest_disc() -> None:
    tool = _load_tool()
    raw = _grid_with_disc()
    raw[raw == 201] = 0
    assert not tool.sacrum_pseudo_label(raw, np.ones(raw.shape, bool), 1.0).any()


def test_class_map_grid_adds_sacrum_class_only_below_disc() -> None:
    tool = _load_tool()
    raw = _grid_with_disc()
    cls = tool.class_map_grid(raw, np.ones(raw.shape, bool), 1.0)
    assert set(np.unique(cls)) == {0, 1, 2, 3, 4}
    assert (cls[raw == 1] == 1).all() and (cls[raw == 201] == 2).all() and (cls[raw == 100] == 3).all()
    assert (cls == 4).sum() > 0 and (cls[:46] != 4).all()


def test_truth_fov_below_disc() -> None:
    tool = _load_tool()
    raw = _grid_with_disc()
    covered = np.ones(raw.shape, bool)
    assert tool.truth_fov_below_disc_mm(raw, covered, 1.0) == pytest.approx(120 - 46)
    covered[50:, :] = False
    assert tool.truth_fov_below_disc_mm(raw, covered, 1.0) == pytest.approx(4.0)
    raw[raw == 201] = 0
    assert tool.truth_fov_below_disc_mm(raw, covered, 1.0) is None


# ------------------------------------------------------------------------ post-processing with 5 classes
def _five_class_seg(phantom: SpinePhantom) -> SegmentedStack:
    stack = phantom.render("sag", "t2")
    grids = [plane_grid(stack, k, 1.0) for k in range(stack.n)]
    h, w = grids[0].shape
    probs = np.zeros((stack.n, 5, h, w), np.float32)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for k, g in enumerate(grids):
        tissue = phantom.labels(g.lps_of_index(ii, jj).reshape(-1, 3)).reshape(h, w)
        cls = np.zeros((h, w), np.int64)
        for t, c in _TO_CLASS.items():
            cls[tissue == t] = c
        for c in range(4):
            probs[k, c] = cls == c
    return SegmentedStack(stack=stack, grids=grids, probs=probs, window=(0.0, 1.0))


def _sacrum_under_lowest_disc(seg: SegmentedStack, k: int) -> None:
    """Paint a sacrum box below the lowest disc of slice k (the network's job; here analytic)."""
    disc = pp._lowest_disc_component(seg, k, DEFAULT)
    assert disc is not None
    d_bot = int(disc[0].max())
    c0, c1 = int(disc[1].min()), int(disc[1].max())
    h = seg.grids[k].shape[0]
    seg.probs[k, SACRUM, d_bot + 1 : min(h, d_bot + 40), c0 : c1 + 1] = 0.95


def test_sacrum_evidence_and_numbering_confirmed(phantom: SpinePhantom) -> None:
    seg = _five_class_seg(phantom)
    base = anchors_from_segmentation(seg)
    assert base.numbering["confidence"] > 0
    mid_k = pp.mid_sagittal(seg, DEFAULT).k  # type: ignore[union-attr]
    # Without a sacrum or a body below the lowest disc the numbering is unconfirmed.
    no_sac = sacrum_below_lowest_disc(seg, mid_k, DEFAULT)
    assert no_sac is not None and no_sac.area_mm2 == 0.0
    _sacrum_under_lowest_disc(seg, mid_k)
    ev = sacrum_below_lowest_disc(seg, mid_k, DEFAULT)
    assert ev is not None and ev.area_mm2 >= DEFAULT.ml_sacrum_min_area_mm2
    assert ev.centre_lps is not None
    anchors = anchors_from_segmentation(seg)
    assert anchors.numbering["sacrum_confirmed"] is True
    assert "S1" in anchors.vertebrae
    assert not any("numbering unconfirmed" in w for w in anchors.warnings)


def test_fov_ending_above_sacrum_is_reported(phantom: SpinePhantom, monkeypatch: pytest.MonkeyPatch) -> None:
    seg = _five_class_seg(phantom)
    mid_k = pp.mid_sagittal(seg, DEFAULT).k  # type: ignore[union-attr]
    real = pp.fov_coverage_rows

    def short_fov(s: SegmentedStack, k: int) -> np.ndarray:
        cov = real(s, k).copy()
        disc = pp._lowest_disc_component(s, k, DEFAULT)
        assert disc is not None
        cov[int(disc[0].max()) + 1 + 3 :, :] = False  # field of view ends 3 mm below the disc
        return cov

    monkeypatch.setattr(pp, "fov_coverage_rows", short_fov)
    ev = sacrum_below_lowest_disc(seg, mid_k, DEFAULT)
    assert ev is not None and ev.fov_below_disc_mm is not None
    assert ev.fov_below_disc_mm < DEFAULT.ml_fov_below_disc_min_mm
    anchors = anchors_from_segmentation(seg)
    assert anchors.numbering["sacrum_confirmed"] is False
    assert any("sacrum not in view" in w for w in anchors.warnings)
    assert "above the sacrum" in anchors.numbering["notes"]


def test_four_class_v1_output_has_no_sacrum_evidence(phantom: SpinePhantom) -> None:
    seg = _five_class_seg(phantom)
    seg4 = SegmentedStack(stack=seg.stack, grids=seg.grids, probs=seg.probs[:, :4].copy(), window=seg.window)
    mid_k = pp.mid_sagittal(seg4, DEFAULT).k  # type: ignore[union-attr]
    assert sacrum_below_lowest_disc(seg4, mid_k, DEFAULT) is None
    anchors = anchors_from_segmentation(seg4)
    assert "sacrum_confirmed" not in anchors.numbering or anchors.numbering["sacrum_confirmed"] is False


# ------------------------------------------------------------------------ augmentation (needs torch)
def test_augment_keeps_labels_valid_and_is_seeded() -> None:
    torch = pytest.importorskip("torch")
    tool = _load_tool()
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.random((4, 1, 320, 320), dtype=np.float32))
    y = torch.from_numpy(rng.integers(0, 5, (4, 320, 320)))
    torch.manual_seed(7)
    xa, ya = tool.augment(x, y, 256)  # type: ignore[attr-defined]
    torch.manual_seed(7)
    xb, yb = tool.augment(x, y, 256)  # type: ignore[attr-defined]
    assert xa.shape == (4, 1, 256, 256) and ya.shape == (4, 256, 256)
    assert float(xa.min()) >= 0.0 and float(xa.max()) <= 1.0
    assert int(ya.min()) >= 0 and int(ya.max()) <= 4
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
