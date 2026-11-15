"""Ground-truth body geometry for alignment tests: endplates measured from the phantom's own body shapes.

The phantom knows every body analytically, so the body masks are exact. They are measured with the same
:func:`spinescope.anatomy.vertebrae.body_plane_geometry` the labeller uses, on the labeller's own plane grid, and
attached to the anchors as ``Vertebra.body`` records. That isolates the analysis from the labeller.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spinescope.anatomy.anchors import Anchors
from spinescope.anatomy.canal_ml import largest_component
from spinescope.anatomy.ml.grid import PlaneGrid, plane_grid
from spinescope.anatomy.vertebrae import body_plane_geometry
from spinescope.config import DEFAULT
from spinescope.study import Study
from spinescope.testing.phantom import SpinePhantom


def _body_mask(ph: SpinePhantom, name: str, pts: np.ndarray) -> np.ndarray:
    v = ph.bodies[name]
    sp = ph.spec
    q = pts - v["center"]
    lr, ap, up = q @ v["lat"], q @ v["ap"], q @ v["normal"]
    rr = (np.abs(lr) / sp.body_half_width) ** 2.6 + (np.abs(ap) / sp.body_half_depth) ** 2.6
    hh = v.get("half_height", sp.body_height / 2)
    return (rr <= 1.0) & (np.abs(up) <= hh)


def _endplate(grid: PlaneGrid, e: dict[str, Any]) -> dict[str, Any]:
    """Plane (down, post) corners of one endplate -> LPS record."""
    ant = grid.to_lps(e["anterior"][0], e["anterior"][1])
    post = grid.to_lps(e["posterior"][0], e["posterior"][1])
    out: dict[str, Any] = {
        "anterior": [float(x) for x in ant],
        "posterior": [float(x) for x in post],
        "angle_deg": float(e["angle_deg"]),
    }
    if "anterior_wall" in e and "posterior_wall" in e:
        out["anterior_wall"] = [float(x) for x in grid.to_lps(*e["anterior_wall"])]
        out["posterior_wall"] = [float(x) for x in grid.to_lps(*e["posterior_wall"])]
    return out


def with_body_geometry(ph: SpinePhantom, study: Study, anchors: Anchors | None = None) -> Anchors:
    """Anchors from the phantom truth with endplate geometry measured on the phantom's own body masks."""
    anchors = anchors or Anchors.from_truth(ph.truth)
    anchors.numbering = {"method": "truth", "confidence": 1.0, "status": "confirmed"}
    stack = study.find("sag", "t2", slabs=False)[0]
    labels = list(anchors.disc_labels())  # superior -> inferior
    for name, v in anchors.vertebrae.items():
        if name not in ph.bodies:
            continue
        # the sagittal slice nearest the body's own mid-plane (distance along the stack normal)
        kb = int(np.argmin(np.abs(stack.positions - float(np.dot(ph.bodies[name]["center"], stack.normal)))))
        grid = plane_grid(stack, kb, 1.0)
        ii, jj = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing="ij")
        down, post = grid.mm_of_index(ii, jj)
        pts = grid.to_lps(down, post).reshape(-1, 3)
        mask = largest_component(_body_mask(ph, name, pts).reshape(grid.shape))
        sup = any(lab.split("-")[1] == name for lab in labels)  # a disc lies above this body
        inf = any(lab.split("-")[0] == name for lab in labels)  # a disc lies below this body
        plane = body_plane_geometry(mask, grid, sup=sup, inf=inf, cfg=DEFAULT)
        rec: dict[str, Any] = {
            "level": name,
            "slice": kb,
            "superior_endplate": None,
            "inferior_endplate": None,
            "ap_depth_mm": None,
            "complete": False,
            "confidence": 1.0,
            "below_lowest_disc_mm": None,
        }
        if plane is not None:
            if "sup" in plane:
                rec["superior_endplate"] = _endplate(grid, plane["sup"])
            if "inf" in plane:
                rec["inferior_endplate"] = _endplate(grid, plane["inf"])
            rec["ap_depth_mm"] = round(float(plane["ap_mm"]), 3)
            rec["complete"] = bool("sup" in plane and "inf" in plane)
        if name == "S1" and labels and mask.any():
            lowest = anchors.discs[labels[-1]]
            z_low = float(lowest.center[2]) - 0.5 * float(lowest.height_mm or 8.0)
            z_bone = float(pts[mask.ravel(), 2].min())
            rec["below_lowest_disc_mm"] = round(max(0.0, z_low - z_bone), 2)
        v.body = rec
    return anchors
