"""OSF multi-scanner lumbar database (qx5rt): registry, discovery, truth and scorer units; dataset-marked run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spinescope import datasets
from spinescope.anatomy import _validate_qx5rt as q
from spinescope.anatomy.anchors import Anchors, Disc

KEY = "lumbar_multiscanner_qx5rt"


def _write_nifti(path: Path, arr: np.ndarray) -> None:
    nib = pytest.importorskip("nibabel")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.diag([0.5, 0.5, 3.0, 1.0])), str(path))


def _tiny_tree(root: Path) -> None:
    """Two complete sessions (GE, Siemens) plus one without masks and one without a T1: only the first two count."""
    base = root / "raw"
    shape = (10, 16, 16)  # [i, j, k] on disk: sagittal in-plane (i, j), slices along k
    t1 = np.ones(shape, dtype=np.int16)
    for sess, vend in (("ID01", "GE_Signa"), ("ID02", "Siemens_Espree")):
        _write_nifti(base / sess / vend / "Images" / "T1_non_contrast.nii.gz", t1)
        lab = base / sess / vend / "Labels"
        m = np.zeros(shape, dtype=np.float32)
        m[2:6, 4:8, 3:9] = 1.0
        _write_nifti(lab / "BS_L3_4.nii.gz", m)
        _write_nifti(lab / "L3.nii.gz", m * 0.0)
    _write_nifti(base / "ID03" / "Philips_Achieva" / "Images" / "T1_non_contrast.nii.gz", t1)  # no labels
    _write_nifti(base / "ID04" / "Philips_Achieva" / "Labels" / "BS_L2_3.nii.gz", np.ones(shape))  # no T1


def test_registry_entry_is_nc_licensed_with_manifest_checksums() -> None:
    spec = datasets.REGISTRY[KEY]
    assert spec.licence == "CC-BY-NC-4.0"
    assert len(spec.files) == 700
    assert spec.total_size == 4268597198
    assert all(f.md5 and f.sha256 and f.size > 0 for f in spec.files)
    assert all(f.url.startswith("https://osf.io/download/") for f in spec.files)
    names = [f.name for f in spec.files]
    assert len(set(names)) == len(names)
    assert all(not n.startswith("/") and ".." not in n.split("/") for n in names)


def test_mask_name_pattern_maps_to_level_names() -> None:
    assert q._MASK_RE.match("BS_L1_2.nii.gz").groups() == ("1", "2")  # type: ignore[union-attr]
    assert q._MASK_RE.match("BS_L4_5.nii") is not None
    assert q._MASK_RE.match("L3.nii.gz") is None  # vertebral bodies are not discs


def test_discover_keeps_only_sessions_with_t1_and_disc_masks(tmp_path: Path) -> None:
    _tiny_tree(tmp_path)
    sessions = q.discover(tmp_path / "raw")
    assert [s.key for s in sessions] == ["ID01:GE_Signa", "ID02:Siemens_Espree"]
    ge = sessions[0]
    assert (ge.vendor, ge.model) == ("GE", "Signa")
    assert set(ge.disc_masks) == {"L3-L4"}  # the L3.nii.gz vertebral mask is not a disc
    assert ge.t1.name == "T1_non_contrast.nii.gz"


def test_discover_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        q.discover(tmp_path / "absent")


def test_truth_centroid_is_the_mask_centroid_on_the_image_grid(tmp_path: Path) -> None:
    _tiny_tree(tmp_path)
    s = q.discover(tmp_path / "raw")[0]
    from spinescope.io.volumes import read_volume

    stack = read_volume(str(s.t1))
    truth = q.truth_levels(s, stack)
    assert [t.level for t in truth] == ["L3-L4"]
    # The NIfTI box is i 2..5, j 4..7, k 3..8; the stack is [slice=k, row=j, col=i], so its centroid is:
    k, r, c = 5.5, 5.5, 3.5
    expected = np.asarray(stack.pix_to_lps(k, r, c))
    np.testing.assert_allclose(truth[0].center, expected, atol=0.05)
    assert truth[0].voxels == 4 * 4 * 6


def _anchors(discs: dict[str, list[float]]) -> Anchors:
    a = Anchors()
    for name, c in discs.items():
        a.discs[name] = Disc(center=c, normal=[0.0, 0.0, 1.0], confidence=0.9)
    return a


def test_score_levels_matches_in_plane_and_checks_names() -> None:
    truths = [
        q.LevelTruth("L3-L4", np.array([0.0, 0.0, -100.0]), 10),
        q.LevelTruth("L4-L5", np.array([0.0, 0.0, -130.0]), 10),
        q.LevelTruth("L2-L3", np.array([0.0, 0.0, -70.0]), 10),
    ]
    anchors = _anchors(
        {
            "L3-L4": [5.0, 1.0, -101.0],  # right name, ~1.4 mm
            "L5-S1": [0.0, 0.0, -131.0],  # the L4-L5 truth gets the disc below's name: off by one
            "L1-L2": [0.0, 0.0, -40.0],  # far from L2-L3 (30 mm): not matched
        }
    )
    rows = {r.level: r for r in q.score_levels(anchors, truths)}
    assert rows["L3-L4"].matched and rows["L3-L4"].name_ok and rows["L3-L4"].offset == 0
    assert rows["L4-L5"].matched and not rows["L4-L5"].name_ok and rows["L4-L5"].offset == -1
    assert rows["L2-L3"].matched is False and rows["L2-L3"].name_ok is False
    assert rows["L2-L3"].offset is None


def test_score_levels_with_no_predictions_is_unmatched() -> None:
    truths = [q.LevelTruth("L3-L4", np.zeros(3), 1)]
    (row,) = q.score_levels(Anchors(), truths)
    assert row.predicted is None and not row.matched and row.dist_mm is None


def test_summary_pools_recall_accuracy_and_offsets() -> None:
    rows = [
        q.LevelRow("a", "GE", "Signa", "L3-L4", "L3-L4", 1.0, 2.0, True, True, 0),
        q.LevelRow("a", "GE", "Signa", "L4-L5", "L5-S1", 2.0, 2.0, True, False, -1),
        q.LevelRow("a", "GE", "Signa", "L2-L3", None, None, None, False, False, None),
    ]
    s = q.summarise(rows)
    assert s["n_levels"] == 3
    assert s["recall_12mm"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["label_accuracy_matched"] == 0.5
    assert s["correct_name_recall"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["offset_counts"] == {"-1": 1, "0": 1}


def test_summarise_by_groups_vendor_and_model() -> None:
    ge = q.SessionResult(
        "a", "GE", "Signa", rows=[q.LevelRow("a", "GE", "Signa", "L3-L4", "L3-L4", 1.0, 1.0, True, True, 0)]
    )
    fail = q.SessionResult("b", "Siemens", "Verio", error="ValueError")
    out = q.summarise_by([ge, fail])
    assert out["sessions"] == 2 and out["failed"] == 1
    assert out["vendor=GE"]["sessions"] == 1
    assert out["model=Siemens/Verio"]["failed"] == 1
    assert out["field_strength"] == "not stated in the release"


def _complete_release() -> Path | None:
    root = datasets.REGISTRY[KEY].path() / "raw"
    if not root.is_dir():
        return None
    expected = datasets.REGISTRY[KEY].files
    complete = all((root / f.name).is_file() for f in expected)
    return root if complete else None


@pytest.mark.dataset
def test_qx5rt_labeller_on_the_ge_session_and_a_sample_runs_and_scores() -> None:
    root = _complete_release()
    if root is None:
        pytest.skip(
            "OSF qx5rt release not complete on disk (fetch lumbar_multiscanner_qx5rt; set SPINESCOPE_DATA)"
        )
    sessions = {s.key: s for s in q.discover(root)}
    picks = [sessions[k] for k in sorted(sessions) if k.endswith(":GE_Signa")]
    picks += [sessions[k] for k in sorted(sessions) if "Siemens" in k][:2]
    results = [q.evaluate_session(s)[0] for s in picks]
    assert all(r.error is None for r in results), [r.error for r in results]
    rows = [row for r in results for row in r.rows]
    assert rows, "no annotated discs were scored"
    s = q.summarise(rows)
    # Regression guard only: the numbers are documented in docs/labeling/lab10_vendors.md.
    assert 0.0 <= s["recall_12mm"] <= 1.0
