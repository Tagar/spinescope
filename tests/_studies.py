"""Helpers to build in-memory phantom studies with controlled variations (for analysis tests)."""

from __future__ import annotations

from spinescope.geometry import Stack
from spinescope.study import Study, from_stacks
from spinescope.testing.phantom import SpinePhantom

CONTRASTS = ("t2", "stir", "t1")


def phantom_study(
    ph: SpinePhantom,
    contrasts: tuple[str, ...] = CONTRASTS,
    rows: int = 192,
    px: float = 1.2,
) -> Study:
    """Sagittal stacks (13 slices, 5 mm apart) for the requested contrasts, classified as a real study."""
    stacks = [
        ph.render("sag", c, n=13, spacing=5.0, thickness=4.0, rows=rows, cols=rows, px=px) for c in contrasts
    ]
    return from_stacks(stacks)


def flip_rows(stack: Stack) -> Stack:
    """The same geometry with the row axis reversed (a flipped acquisition, different pixel order)."""
    rows = stack.vol.shape[1]
    return Stack(
        key=stack.key,
        vol=stack.vol[:, ::-1, :].copy(),
        ipp=stack.ipp + (rows - 1) * stack.row_mm * stack.col_dir,
        row_dir=stack.row_dir,
        col_dir=-stack.col_dir,
        row_mm=stack.row_mm,
        col_mm=stack.col_mm,
        thickness=stack.thickness,
        description=stack.description,
        meta=dict(stack.meta),
    )


def keep_contrasts(study: Study, keep: tuple[str, ...]) -> Study:
    """Study restricted to the sagittal stacks whose contrast is in ``keep``."""
    out = {}
    for key, st in study.stacks.items():
        if st.meta.get("role", {}).get("contrast") in keep:
            out[key] = st
    return Study(stacks=out, info=dict(study.info))


def stacks_flipped(study: Study) -> Study:
    """Every stack with its rows reversed; pixel order changes, geometry does not."""
    return Study(stacks={k: flip_rows(st) for k, st in study.stacks.items()}, info=dict(study.info))
