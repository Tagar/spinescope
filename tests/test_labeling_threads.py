"""ONNX Runtime thread control and the per-process session cache of the learned labeller (phantom-free)."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest

from spinescope.anatomy.ml import labeller
from spinescope.anatomy.ml.labeller import load_session, resolve_threads
from spinescope.config import Config

pytest.importorskip("onnxruntime")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv(labeller.THREADS_ENV, raising=False)
    monkeypatch.setattr(labeller, "in_worker_process", lambda: False)
    return monkeypatch


def test_defaults_are_ort_defaults_in_the_main_process(clean_env: pytest.MonkeyPatch) -> None:
    assert resolve_threads() == (0, 0)


def test_env_variable_sets_intra_op_threads(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv(labeller.THREADS_ENV, "3")
    assert resolve_threads() == (3, 0)


def test_env_zero_is_an_explicit_ort_default(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv(labeller.THREADS_ENV, "0")
    clean_env.setattr(labeller, "in_worker_process", lambda: True)
    assert resolve_threads() == (0, 1)  # no pool cap when the user asked for ORT's default


def test_explicit_config_beats_env(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv(labeller.THREADS_ENV, "3")
    assert resolve_threads(Config(ml_intra_op_threads=2, ml_inter_op_threads=1)) == (2, 1)


def test_pool_child_is_capped_to_one_thread(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setattr(labeller, "in_worker_process", lambda: True)
    assert resolve_threads() == (1, 1)
    clean_env.setenv(labeller.THREADS_ENV, "4")
    assert resolve_threads() == (4, 1)  # an explicit env value still wins over the cap


def test_in_worker_process_is_false_in_the_test_process() -> None:
    assert labeller.in_worker_process() is (multiprocessing.parent_process() is not None)


def test_real_pool_worker_detects_itself() -> None:
    with get_context("spawn").Pool(1) as pool:
        intra, inter = pool.apply(resolve_threads, ())
    assert (intra, inter) == (1, 1)


def test_session_is_cached_per_path_and_thread_setting() -> None:
    a = load_session()
    assert load_session() is a
    b = load_session(intra_op_threads=1)
    assert b is not a
    assert load_session(intra_op_threads=1) is b


def test_session_cache_is_thread_safe() -> None:
    with ThreadPoolExecutor(max_workers=8) as ex:
        sessions = list(ex.map(lambda _: load_session(intra_op_threads=2), range(16)))
    assert all(s is sessions[0] for s in sessions)


def test_missing_weights_raise_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "missing.onnx")


def test_thread_setting_does_not_change_the_network_output() -> None:
    import numpy as np

    images = np.random.default_rng(0).random((2, 64, 48), dtype=np.float32)
    one = labeller.run_network(load_session(intra_op_threads=1), images)
    default = labeller.run_network(load_session(), images)
    np.testing.assert_allclose(one, default, atol=1e-5)
