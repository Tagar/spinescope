from __future__ import annotations

import numpy as np

from spinescope.anatomy import Anchors


def test_from_truth_roundtrip_and_queries(phantom):
    a = Anchors.from_truth(phantom.truth)
    assert a.disc_labels()[0] == "T11-T12" and a.disc_labels()[-1] == "L5-S1"
    b = Anchors.from_json(a.to_json())
    assert b.disc_labels() == a.disc_labels()
    c = np.asarray(a.disc("L4-L5").center)
    lab, s = a.nearest_disc(c + 1.0 * np.asarray(a.disc("L4-L5").normal))
    assert lab == "L4-L5" and abs(s - 1.0) < 1e-9
    assert a.level_of(c) == "L4-L5"
    assert a.level_of(phantom.truth["vertebrae"]["L4"]) == "L4"
    ap = a.disc("L4-L5").ap_dir
    assert ap[1] > 0.5  # points posterior
