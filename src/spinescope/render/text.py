"""Scalable text for labels: Pillow's bundled font at a size relative to the image it labels."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from PIL import ImageFont

from ..config import DEFAULT, Config


@lru_cache(maxsize=64)
def font(size: int) -> Any:
    """Bundled scalable font at ``size`` pixels (Pillow >= 10.1). Falls back to Pillow's default bitmap."""
    return ImageFont.load_default(size=max(1, int(size)))


def label_px(shape: tuple[int, ...], cfg: Config = DEFAULT) -> int:
    """Label font height for an image of ``shape``: a fraction of its shorter side, never below 6 px."""
    return max(6, round(cfg.font_frac * min(shape[0], shape[1])))
