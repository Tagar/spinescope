"""Visual QA of the labeller (spinescope.anatomy.qa) and the evaluation CLI, on the phantom."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spinescope import load_study
from spinescope.anatomy import label_spine
from spinescope.anatomy._validate import truth_from_phantom
from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.qa import canal_centreline, case_report, confidence_colour, midline_slice, overlay
from spinescope.study import from_stacks
from spinescope.testing.phantom import SpinePhantom, write_standard_study

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def truth_list(phantom: SpinePhantom):
    return truth_from_phantom(phantom.truth)


def _is_rgb(img: np.ndarray) -> bool:
    return (
        img.dtype == np.uint8
        and img.ndim == 3
        and img.shape[2] == 3
        and img.shape[0] > 0
        and img.shape[1] > 0
    )


def test_confidence_colour_endpoints_and_clipping():
    assert confidence_colour(0.0)[1] < confidence_colour(1.0)[1]  # red at 0, green at 1
    assert confidence_colour(-5.0) == confidence_colour(0.0)
    assert confidence_colour(9.0) == confidence_colour(1.0)


def test_midline_slice_is_nearest_to_predicted_plane(phantom_study, phantom: SpinePhantom):
    st = phantom_study.best("sag")
    anc = Anchors.from_truth(phantom.truth)
    k = midline_slice(st, anc)
    plane = float(anc.midline_x) * float(st.normal[0])
    assert abs(st.positions[k] - plane) <= np.min(np.abs(np.diff(st.positions))) + 1e-6


def test_canal_centreline_is_found_on_phantom(phantom_study):
    st = phantom_study.best("sag")
    pts = canal_centreline(st, midline_slice(st, Anchors()))
    assert pts is not None and pts.ndim == 2 and pts.shape[1] == 3 and len(pts) > 3
    # superior -> inferior: z decreases along the centreline
    assert pts[0, 2] > pts[-1, 2]


def test_overlay_with_truth_and_prediction(phantom_study, phantom: SpinePhantom, truth_list):
    truth, mx = truth_list
    anc = label_spine(phantom_study, backend="classical")
    img = overlay(phantom_study, anc, truth, midline_truth=mx)
    assert _is_rgb(img)
    # the overlay draws colour (confidence, truth and canal colours), not just the grey slice
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 50


def test_overlay_without_truth_and_with_explicit_canal(phantom_study, phantom: SpinePhantom):
    anc = Anchors.from_truth(phantom.truth)
    canal = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 0.0], [0.0, 0.0, -100.0]])
    img = overlay(phantom_study, anc, None, canal=canal)
    assert _is_rgb(img)


def test_overlay_without_sagittal_or_discs_does_not_fail(phantom: SpinePhantom):
    ax = phantom.render("ax", "t2", n=20, spacing=5.0, thickness=4.0, rows=96, cols=96, px=1.4)
    study = from_stacks([ax])
    assert _is_rgb(overlay(study, Anchors(), None))
    st = phantom.render("sag", "t2", n=13, spacing=5.0, thickness=4.0, rows=160, cols=160, px=1.2)
    assert _is_rgb(overlay(from_stacks([st]), Anchors(), None))


def test_case_report_writes_one_png_page(tmp_path, phantom_study, phantom: SpinePhantom, truth_list):
    truth, mx = truth_list
    anc = label_spine(phantom_study, backend="classical")
    out = tmp_path / "report" / "case.png"
    path = case_report(phantom_study, anc, truth, out_path=str(out), title="phantom", midline_truth=mx)
    assert Path(path) == out and out.is_file()
    with Image.open(out) as im:
        assert im.mode == "RGB" and im.width > 100 and im.height > 100
        assert max(im.size) <= 2000


def test_case_report_without_truth_or_discs(tmp_path, phantom_study):
    p1 = case_report(
        phantom_study, label_spine(phantom_study, backend="classical"), None, out_path=str(tmp_path / "a.png")
    )
    p2 = case_report(phantom_study, Anchors(), None, out_path=str(tmp_path / "b.png"))
    assert Path(p1).is_file() and Path(p2).is_file()


def test_qa_overlay_on_a_fresh_study(tmp_path, phantom: SpinePhantom):
    """A study loaded from DICOM (the path the CLI uses) gives the same kind of overlay."""
    study = load_study(write_standard_study(phantom, tmp_path / "dcm")["root"])
    truth, mx = truth_from_phantom(phantom.truth)
    assert _is_rgb(overlay(study, label_spine(study, backend="classical"), truth, midline_truth=mx))


def _load_tool():
    spec = importlib.util.spec_from_file_location("eval_labeling", REPO / "tools" / "eval_labeling.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up by name
    spec.loader.exec_module(mod)
    return mod


def test_eval_cli_runs_on_phantom_and_writes_outputs(tmp_path):
    tool = _load_tool()
    out = tmp_path / "run"
    rc = tool.main(
        [
            "--dataset",
            "phantom",
            "--n",
            "2",
            "--jobs",
            "1",
            "--gallery-n",
            "1",
            "--out",
            str(out),
            "--cache",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 0
    for name in ("summary.md", "metrics.json", "cases.csv", "gallery.html"):
        assert (out / name).is_file(), name
    assert len(list((out / "png").glob("*.png"))) == 1
    text = (out / "summary.md").read_text()
    assert "Overall" in text and "Confidence calibration" in text and "Numbering error histogram" in text
    assert str(tmp_path) not in text  # no input paths in the report
    # second run reads predictions from the cache
    rc2 = tool.main(
        [
            "--dataset",
            "phantom",
            "--n",
            "2",
            "--jobs",
            "1",
            "--gallery-n",
            "0",
            "--out",
            str(tmp_path / "again"),
            "--cache",
            str(tmp_path / "cache"),
        ]
    )
    assert rc2 == 0


def test_eval_cli_refuses_outputs_inside_the_repository():
    tool = _load_tool()
    with pytest.raises(SystemExit):
        tool.main(["--dataset", "phantom", "--n", "1", "--out", str(REPO / "docs" / "should_not_exist")])
