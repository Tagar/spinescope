from __future__ import annotations

import pytest

from spinescope import load_study
from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study


@pytest.fixture(scope="session")
def phantom() -> SpinePhantom:
    return SpinePhantom(
        PhantomSpec(
            degenerated={"L4-L5": 0.2}, modic1=[("L4-L5", "inf", "L")], herniation={"L5-S1": ("central", 5.0)}
        )
    )


@pytest.fixture(scope="session")
def phantom_dir(phantom, tmp_path_factory):
    root = tmp_path_factory.mktemp("phantom_study")
    return write_standard_study(phantom, root)


@pytest.fixture(scope="session")
def phantom_study(phantom_dir):
    return load_study(phantom_dir["root"])
