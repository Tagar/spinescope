"""Robustness scenarios: phantom protocols and anatomies that any labelling backend must cope with.

A :class:`Scenario` names a phantom (``PhantomSpec`` keyword arguments) and a protocol (which sagittal
contrasts to render, slice count and spacing, field of view, artefacts). :func:`build_case` renders it into a
:class:`~spinescope.study.Study` and keeps the ground-truth discs that lie inside the field of view of the
sagittal stack a labeller would use. :func:`evaluate` runs ``label_spine`` on the case and scores it with the
shared validation scorer (:mod:`spinescope.anatomy._validate`).

Each scenario ends in one of three outcomes:

* ``pass``: every in-view disc is found, named correctly and centred within ``PASS_CENTRE_P90_MM``, and no
  spurious disc is reported.
* ``unsure``: not a pass, but the labeller reported low numbering confidence (below
  ``UNSURE_CONFIDENCE``). This is the acceptable failure mode: "unsure" beats "wrong and confident".
* ``silent_error``: not a pass and the labeller was confident (numbering confidence at or above
  ``UNSURE_CONFIDENCE``). This is the failure that matters most.

The truth for transitional anatomy uses the lumbar-count convention of the SPIDER dataset (the lowest disc
is L5-S1, or L6-S1 when S1 is lumbarised); the anatomical name is kept in the phantom truth.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..anatomy._validate import TruthDisc, aggregate, score_anchors, truth_from_phantom
from ..anatomy.anchors import Anchors
from ..anatomy.pipeline import label_spine
from ..study import Study, from_stacks
from .phantom import PhantomSpec, SpinePhantom

PASS_CENTRE_P90_MM = 3.0
UNSURE_CONFIDENCE = 0.5

PASS = "pass"
UNSURE = "unsure"
SILENT_ERROR = "silent_error"


@dataclass(frozen=True)
class Protocol:
    """Which sagittal stacks to render and how."""

    contrasts: tuple[str, ...] = ("t2", "stir", "t1")
    n: int = 13
    spacing: float = 5.0
    thickness: float = 4.0
    px: float = 1.2
    rows: int = 192
    cols: int = 192
    ghost: float = 0.0  # motion ghost amplitude
    wrap_rows: int = 0  # phase wrap-around (aliasing) rows
    fov: tuple[str, str] | None = None  # (top body, bottom body) to cover; None = default FOV
    fov_pad_top: float = 0.0  # mm of field of view above the top body (so the disc above it is in view)
    fov_extra_below: float = 0.0  # mm of field of view below the bottom body (e.g. air below the sacrum)


@dataclass(frozen=True)
class Scenario:
    """One robustness case: a phantom anatomy, a protocol and a short description."""

    name: str
    group: str  # protocol | fov | transitional | pathology | hardware | artefact | anatomy
    description: str
    spec: dict[str, Any] = field(default_factory=dict)  # PhantomSpec keyword arguments
    protocol: Protocol = field(default_factory=Protocol)


@dataclass
class RobustnessCase:
    """A rendered scenario: the phantom, its study and the truth discs in view."""

    scenario: Scenario
    phantom: SpinePhantom
    study: Study
    truth: list[TruthDisc]
    realism: str = "cartoon"  # phantom look: 'cartoon' or 'realistic' (see testing/realistic.py)


@dataclass
class CaseResult:
    """Outcome of one labelling backend on one scenario."""

    scenario: str
    backend: str
    outcome: str  # pass | unsure | silent_error
    metrics: dict[str, float]
    predicted: list[str]
    truth: list[str]
    numbering_confidence: float
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return self.outcome == PASS


_T = Protocol
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("baseline", "protocol", "standard 3-contrast sagittal protocol, default anatomy"),
    Scenario("t2_only", "protocol", "sagittal T2 only", protocol=_T(contrasts=("t2",))),
    Scenario("t1_only", "protocol", "sagittal T1 only", protocol=_T(contrasts=("t1",))),
    Scenario("stir_only", "protocol", "sagittal STIR only", protocol=_T(contrasts=("stir",))),
    Scenario("pd_only", "protocol", "sagittal proton density only", protocol=_T(contrasts=("pd",))),
    Scenario(
        "iso_1mm_3d",
        "protocol",
        "3D isotropic 1 mm sagittal volume (9 slices, 160 x 112 matrix)",
        protocol=_T(contrasts=("t2",), n=9, spacing=1.0, thickness=1.0, px=1.0, rows=160, cols=112),
    ),
    Scenario(
        "thick_6mm",
        "protocol",
        "thick 6 mm sagittal slices (9 slices)",
        protocol=_T(contrasts=("t2",), n=9, spacing=6.0, thickness=6.0),
    ),
    Scenario(
        "few_slices_7",
        "protocol",
        "few sagittal slices: 7 at 5 mm",
        protocol=_T(contrasts=("t2",), n=7),
    ),
    Scenario(
        "field_3T",
        "protocol",
        "3 T intensity and noise variant (sagittal T2 and T1)",
        spec={"field_strength": 3.0},
        protocol=_T(contrasts=("t2", "t1")),
    ),
    Scenario(
        "fov_T10_sacrum",
        "fov",
        "field of view from T10 to the sacrum, with thoracic ribs; the T9-T10 disc is in view",
        spec={"thoracic_top": 9, "ribs": True},
        protocol=_T(contrasts=("t2",), fov=("T10", "S1"), fov_pad_top=14.0),
    ),
    Scenario(
        "fov_L2_S1",
        "fov",
        "field of view from L2 to S1 only (upper discs out of view)",
        protocol=_T(contrasts=("t2",), fov=("L2", "S1")),
    ),
    Scenario(
        "fov_L1_coccyx",
        "fov",
        "field of view from L1 well below the sacrum (air below S1; the phantom has no coccyx)",
        protocol=_T(contrasts=("t2",), fov=("L1", "S1"), fov_extra_below=40.0),
    ),
    Scenario(
        "lumbarised_S1",
        "transitional",
        "six lumbar vertebrae: S1 lumbarised, full S1-S2 disc",
        spec={"transitional": "lumbarised_s1"},
        protocol=_T(contrasts=("t2",)),
    ),
    Scenario(
        "sacralised_L5",
        "transitional",
        "sacralised L5: partial L5-S1 disc, bone bridge on one side",
        spec={"transitional": "sacralised_l5"},
        protocol=_T(contrasts=("t2",)),
    ),
    Scenario(
        "dark_discs",
        "pathology",
        "degenerated discs with signal loss (dark nucleus) at L2-L3 and L4-L5",
        spec={"degenerated": {"L2-L3": 0.0, "L4-L5": 0.0}},
    ),
    Scenario(
        "collapsed_discs",
        "pathology",
        "collapsed discs (height loss 50% at L3-L4, 60% at L4-L5)",
        spec={"collapsed": {"L3-L4": 0.5, "L4-L5": 0.6}},
    ),
    Scenario(
        "compression_fracture_L2",
        "pathology",
        "L2 compression fracture: anterior wedge with STIR and T2 oedema",
        spec={"fracture": {"L2": 0.45}},
    ),
    Scenario(
        "schmorl_nodes",
        "pathology",
        "Schmorl nodes at the L3 superior and L4 inferior endplates",
        spec={"schmorl": [("L3", "sup"), ("L4", "inf")]},
    ),
    Scenario(
        "scoliosis",
        "anatomy",
        "lumbar scoliosis: 14 mm lateral apex at L2 and 12 degrees of axial rotation",
        spec={"scoliosis_mm": 14.0, "scoliosis_rot_deg": 12.0},
    ),
    Scenario(
        "hardware_cage_screws",
        "hardware",
        "post-operative L4-L5 cage, L4 and L5 pedicle screws and a posterior rod",
        spec={"hardware": ("cage:L4-L5", "screws:L4", "screws:L5", "rod")},
    ),
    Scenario(
        "bias_field",
        "artefact",
        "surface-coil intensity bias (strong posterior-to-anterior gradient)",
        spec={"bias_field": 0.6},
        protocol=_T(contrasts=("t2", "t1")),
    ),
    Scenario(
        "motion_ghosting",
        "artefact",
        "motion ghosting along the phase axis",
        protocol=_T(contrasts=("t2",), ghost=0.3),
    ),
    Scenario(
        "aliasing_wrap",
        "artefact",
        "phase wrap-around: 60 rows of anatomy fold onto the opposite edge",
        protocol=_T(contrasts=("t2",), wrap_rows=60),
    ),
)

_BY_NAME = {s.name: s for s in SCENARIOS}


def scenario(name: str) -> Scenario:
    """The scenario called ``name``. Raises ``KeyError`` for unknown names."""
    return _BY_NAME[name]


def _fov_rows_and_centre(ph: SpinePhantom, proto: Protocol) -> tuple[np.ndarray, int]:
    """Sagittal centre and row count that cover ``proto.fov`` (top body, bottom body)."""
    top, bottom = proto.fov or ("", "")
    half_top = ph.bodies[top].get("half_height", ph.spec.body_height / 2)
    z_top = float(ph.bodies[top]["center"][2]) + half_top + proto.fov_pad_top
    half_bottom = ph.bodies[bottom].get("half_height", ph.spec.body_height / 2)
    z_bot = float(ph.bodies[bottom]["center"][2]) - half_bottom - proto.fov_extra_below
    l3 = ph.bodies["L3"]["center"]
    centre = np.array([0.0, l3[1] + 10.0, (z_top + z_bot) / 2])
    rows = math.ceil((z_top - z_bot) / proto.px) + 4
    return centre, rows


def _truth_in_view(ph: SpinePhantom, stack: Any) -> list[TruthDisc]:
    """Truth discs whose projection onto the sagittal stack lies inside its field of view."""
    out: list[TruthDisc] = []
    for t in truth_from_phantom(ph.truth)[0]:
        proj = t.center - ((t.center - stack.ipp[0]) @ stack.normal) * stack.normal
        if stack.contains(proj):
            out.append(TruthDisc(label=t.label, name=t.name, center=proj, normal=t.normal))
    return out


def build_case(sc: Scenario, realism: str = "cartoon") -> RobustnessCase:
    """Render scenario ``sc`` into a study and collect its in-view truth. Deterministic.

    ``realism`` picks the phantom look: ``'cartoon'`` (the original shapes) or ``'realistic'``.
    """
    ph = SpinePhantom(PhantomSpec(**{**sc.spec, "realism": realism}))
    p = sc.protocol
    centre: np.ndarray | None = None
    rows = p.rows
    if p.fov is not None:
        centre, rows = _fov_rows_and_centre(ph, p)
    stacks = [
        ph.render(
            "sag",
            c,
            n=p.n,
            spacing=p.spacing,
            thickness=p.thickness,
            rows=rows,
            cols=p.cols,
            px=p.px,
            center=centre,
            ghost=p.ghost,
            wrap_rows=p.wrap_rows,
        )
        for c in p.contrasts
    ]
    study = from_stacks(stacks)
    best = study.best("sag")
    if best is None:  # pragma: no cover - every protocol renders at least one sagittal
        raise RuntimeError(f"scenario {sc.name}: no sagittal stack")
    return RobustnessCase(sc, ph, study, _truth_in_view(ph, best), realism)


@functools.cache
def cached_case(name: str, realism: str = "cartoon") -> RobustnessCase:
    """:func:`build_case` for the scenario called ``name``, cached for the process (rendering is the slow part)."""
    return build_case(scenario(name), realism)


def classify(metrics: dict[str, float], numbering_confidence: float) -> str:
    """Outcome from pooled metrics and the labeller's own confidence."""
    perfect = (
        metrics["recall"] >= 1.0
        and metrics["precision"] >= 1.0
        and metrics["label_accuracy_all_truth"] >= 1.0
        and metrics["centre_err_p90_mm"] <= PASS_CENTRE_P90_MM
    )
    if perfect:
        return PASS
    return UNSURE if numbering_confidence < UNSURE_CONFIDENCE else SILENT_ERROR


def score(case: RobustnessCase, anchors: Anchors, backend: str) -> CaseResult:
    """Score ``anchors`` against the case truth with the shared scorer."""
    s = score_anchors(anchors, case.truth, midline_truth=case.phantom.truth["midline_x"])
    metrics = aggregate([s])
    conf = float(anchors.numbering.get("confidence", 0.0) or 0.0)
    return CaseResult(
        scenario=case.scenario.name,
        backend=backend,
        outcome=classify(metrics, conf),
        metrics=metrics,
        predicted=sorted(anchors.disc_labels()),
        truth=[t.name for t in case.truth],
        numbering_confidence=conf,
        warnings=list(anchors.warnings),
    )


def evaluate(case: RobustnessCase, backend: str = "classical") -> CaseResult:
    """Run ``label_spine`` with ``backend`` on the case and score it."""
    anchors = label_spine(case.study, backend=backend)
    return score(case, anchors, backend)
