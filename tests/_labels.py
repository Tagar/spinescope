"""Memoised ``label_spine`` for the test suite.

Labelling is deterministic, and many tests label the same phantom again: the same module- or session-scoped study,
or a fresh render of the same phantom spec in another module. This wrapper runs the pipeline once per distinct
(study content, backend, config) and returns a deep copy, so a test that edits its anchors cannot leak into
another test. The key is a digest of the voxels, geometry and metadata of every stack, not object identity.
Exceptions are not cached.
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import numpy as np

from spinescope.anatomy import Anchors
from spinescope.anatomy import label_spine as _label_spine
from spinescope.config import Config
from spinescope.study import Study

_MAX_ENTRIES = 64
_MEMO: OrderedDict[tuple[str, str, Config | None], Anchors] = OrderedDict()


def study_digest(study: Study) -> str:
    """SHA-256 over every stack's key, voxels, geometry and public metadata (private '_' keys, UID hashes excluded)."""
    h = hashlib.sha256()
    for key in sorted(study.stacks):
        st = study.stacks[key]
        h.update(key.encode())
        h.update(repr((st.vol.shape, str(st.vol.dtype))).encode())
        h.update(np.ascontiguousarray(st.vol).tobytes())
        for arr in (st.ipp, st.row_dir, st.col_dir):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        h.update(repr((float(st.row_mm), float(st.col_mm), float(st.thickness))).encode())
        meta = {k: v for k, v in st.meta.items() if not k.startswith("_") and not k.endswith("_uid_hash")}
        h.update(repr(sorted(meta.items(), key=lambda kv: kv[0])).encode())  # no paths or random UIDs
    return h.hexdigest()


def label_spine(study: Study, config: Config | None = None, *, backend: str = "auto") -> Anchors:
    """:func:`spinescope.anatomy.label_spine`, memoised per study content, backend and config."""
    key = (study_digest(study), backend, config)
    hit = _MEMO.get(key)
    if hit is None:
        hit = _label_spine(study, config, backend=backend)
        _MEMO[key] = hit
        while len(_MEMO) > _MAX_ENTRIES:
            _MEMO.popitem(last=False)
    else:
        _MEMO.move_to_end(key)
    return copy.deepcopy(hit)


def clear() -> None:
    """Forget every memoised result (for tests that need a cold pipeline)."""
    _MEMO.clear()
