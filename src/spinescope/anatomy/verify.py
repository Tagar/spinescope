"""Verify the level labels with a reader (human or AI) and apply the correction to :class:`Anchors`.

The automatic labeller counts discs up from the lowest one it finds, and that count is wrong whenever the
lowest visible disc is really L4-L5 (sacralized L5) or L5-L6 (lumbarized S1), or a disc is missed or
invented. :func:`build_packet` makes what a reader needs to check the count: labelled mid-sagittal T2 and
T1 overlays (:func:`spinescope.anatomy.qa.overlay`), the localizer with the predicted levels when there is
one, and a zoom on the sacrum below the lowest disc. :func:`parse_answer` validates the reader's JSON
answer, and :func:`apply_answer` changes the anchors through the Anchors correction API, with provenance.

Answer format (the question in :data:`QUESTION` asks for it)::

    {"lowest_disc_is": "L5-S1" | "L4-L5" | "L5-L6" | "unsure",
     "missing": ["L3-L4"], "extra": ["L2-L3"], "transitional": true | false | null, "notes": "..."}

Conventions. ``extra`` lists labels as they are shown on the packet that are not discs. ``lowest_disc_is`` is the
true name of the lowest disc shown, after the extras are removed (the disc marked L5-S1 may really be L4-L5, or
L5-L6 when the S1 is lumbarized). ``missing`` lists true level names of discs that have no label, placed from
their labelled neighbours. An ``unsure`` answer makes no structural change: it lowers the numbering confidence
and adds a warning. The whole answer is applied to a copy, so an answer that cannot be applied changes nothing.

Pluggable reader. Nothing here calls a network service. :func:`verify_labels` takes any callable
``ask(packet) -> answer`` (text or dict). To use a vision-capable model, wire the Anthropic SDK yourself
(not a dependency of this package)::

    import base64, anthropic

    def ask(packet):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode("ascii")}}
            for png in packet.images.values()
        ]
        content.append({"type": "text", "text": packet.question})
        msg = anthropic.Anthropic().messages.create(
            model="<a vision-capable model>", max_tokens=400,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text

Outputs are PNG bytes and JSON. Write packets to a scratch folder, not into the repository when they come
from imaging data. The packet never contains patient attributes or input paths.
"""

from __future__ import annotations

import io
import itertools
import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ..config import DEFAULT, Config
from ..geometry import Stack
from ..render.frame import display_scale, slice_frame
from ..render.layout import with_title
from ..render.sheets import zoom_crop
from ..render.window import grey_rgb, series_window
from ..study import Study
from . import qa
from .anchors import VERTEBRA_NAMES, Anchors, split_disc
from .numbering import STATUS_CONFIRMED, STATUS_PROBABLE, STATUS_UNCONFIRMED, numbering_status

log = logging.getLogger(__name__)

RGB = NDArray[np.uint8]

#: Allowed values of ``lowest_disc_is``.
LOWEST_CHOICES: tuple[str, ...] = ("L5-S1", "L4-L5", "L5-L6", "unsure")
#: Level names a reader may write in missing/extra (L6 is the name of a lumbarized S1).
_LEVEL_TOKEN = r"(?:T1[0-2]|T[1-9]|L[1-6]|S1)"
#: Position of each vertebra name in the counting convention (S1 = 0, L5 = 1, L6 = -1 for a lumbarized S1).
_COUNT_INDEX: dict[str, int] = {"L6": -1, **{name: i for i, name in enumerate(VERTEBRA_NAMES)}}
_DISC_TOKEN_RE = re.compile(rf"^({_LEVEL_TOKEN})-({_LEVEL_TOKEN})$")
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_ALLOWED_KEYS = ("lowest_disc_is", "missing", "extra", "transitional", "notes")
NOTES_MAX = 500

QUESTION = (
    "Count from the sacrum: is the disc marked L5-S1 the lowest mobile disc? Are any discs missed or extra? "
    "Answer as JSON {lowest_disc_is: 'L5-S1'|'L4-L5'|'L5-L6'|'unsure', missing: [...], extra: [...], "
    "transitional: bool|null, notes}."
)
QUESTION_RULES = (
    "Rules: extra lists labels as shown on the images that are not discs (use the labels as displayed). "
    "lowest_disc_is is the true name of the lowest disc shown after you remove the extras: L5-S1 if the sacrum "
    "is below it, L4-L5 if the disc marked L5-S1 is really L4-L5 (sacralized L5), L5-L6 if the S1 below it is "
    "lumbarized. Use 'unsure' if the sacrum is not visible or the count is doubtful. missing lists the true "
    "names of discs that have no label. transitional is true for a sacralized L5 or lumbarized S1, false if the "
    "anatomy is standard, null if unknown. The lumbosacral CNN probabilities in the packet are one cue, not a count: "
    "it can score a sacral remnant as L5-S1 too, so check the bone. notes is at most a sentence. "
    "Answer with the JSON object only."
)
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lowest_disc_is"],
    "properties": {
        "lowest_disc_is": {"enum": list(LOWEST_CHOICES)},
        "missing": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[TLS][0-9]{1,2}-[TLS][0-9]{1,2}$"},
        },
        "extra": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[TLS][0-9]{1,2}-[TLS][0-9]{1,2}$"},
        },
        "transitional": {"type": ["boolean", "null"]},
        "notes": {"type": "string", "maxLength": NOTES_MAX},
    },
}


class AnswerError(ValueError):
    """A reader's answer is not valid; the message says which field and never repeats input paths."""


@dataclass(frozen=True)
class LabelAnswer:
    """A validated answer. ``extra``/``missing`` are normalised disc labels such as ``L3-L4``."""

    lowest_disc_is: str
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    transitional: bool | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lowest_disc_is": self.lowest_disc_is,
            "missing": list(self.missing),
            "extra": list(self.extra),
            "transitional": self.transitional,
            "notes": self.notes,
        }


@dataclass
class VerificationPacket:
    """What a reader is shown: named PNG images, the question, the predicted levels and build notes."""

    images: dict[str, bytes]
    question: str
    predicted_levels: list[str]  # disc labels superior -> inferior, as drawn on the images
    notes: list[str] = field(default_factory=list)
    answer_schema: dict[str, Any] = field(default_factory=lambda: dict(ANSWER_SCHEMA))
    #: Lumbosacral CNN P(L5-S1) for the lowest candidates, bottom-up (:func:`lumbosacral_context`); None if it did not run.
    lumbosacral: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe description (no image bytes)."""
        return {
            "images": list(self.images),
            "question": self.question,
            "rules": QUESTION_RULES,
            "answer_schema": self.answer_schema,
            "predicted_levels": list(self.predicted_levels),
            "lumbosacral": self.lumbosacral,
            "notes": list(self.notes),
        }

    def write(self, outdir: str | Path) -> list[str]:
        """Write ``NN_<name>.png`` for every image and ``packet.json``. Returns the paths written."""
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, (name, png) in enumerate(self.images.items(), start=1):
            p = d / f"{i:02d}_{name}.png"
            p.write_bytes(png)
            paths.append(str(p))
        p = d / "packet.json"
        p.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        paths.append(str(p))
        return paths


# --------------------------------------------------------------------------- answers
def _normalise_level(token: Any, field_name: str) -> str:
    if not isinstance(token, str):
        raise AnswerError(f"{field_name} entries must be strings")
    s = token.strip().upper().replace("/", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    if not _DISC_TOKEN_RE.match(s):
        raise AnswerError(f"{field_name} entry {token!r} is not a disc label such as L4-L5")
    return s


def parse_answer(answer: str | Mapping[str, Any] | LabelAnswer) -> LabelAnswer:
    """Validate a reader's answer (JSON text, possibly fenced, or a dict) and return a :class:`LabelAnswer`.

    Raises :class:`AnswerError` on malformed JSON, unknown keys, a bad ``lowest_disc_is``, bad disc labels,
    non-boolean ``transitional`` or non-string ``notes``.
    """
    if isinstance(answer, LabelAnswer):
        return answer
    if isinstance(answer, str):
        text = answer.strip()
        fence = _FENCE_RE.match(text)
        if fence:
            text = fence.group(1)
        try:
            obj: Any = json.loads(text)
        except ValueError:
            raise AnswerError("answer is not valid JSON") from None
    else:
        obj = answer
    if not isinstance(obj, Mapping):
        raise AnswerError("answer must be a JSON object")
    unknown = sorted(set(obj) - set(_ALLOWED_KEYS))
    if unknown:
        raise AnswerError(f"unexpected key(s): {', '.join(str(u) for u in unknown)}")
    if "lowest_disc_is" not in obj:
        raise AnswerError("lowest_disc_is is required")
    low_raw = obj["lowest_disc_is"]
    if not isinstance(low_raw, str):
        raise AnswerError(f"lowest_disc_is must be one of {', '.join(LOWEST_CHOICES)}")
    low = "unsure" if low_raw.strip().lower() == "unsure" else low_raw.strip().upper().replace("/", "-")
    if low not in LOWEST_CHOICES:
        raise AnswerError(f"lowest_disc_is must be one of {', '.join(LOWEST_CHOICES)}")
    lists: dict[str, tuple[str, ...]] = {}
    for key in ("missing", "extra"):
        raw = obj.get(key, [])
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise AnswerError(f"{key} must be a list")
        items = tuple(_normalise_level(t, key) for t in raw)
        if len(set(items)) != len(items):
            raise AnswerError(f"{key} lists the same level twice")
        lists[key] = items
    trans = obj.get("transitional", None)
    if trans is not None and not isinstance(trans, bool):
        raise AnswerError("transitional must be true, false or null")
    notes = obj.get("notes", "")
    if notes is None:
        notes = ""
    if not isinstance(notes, str):
        raise AnswerError("notes must be a string")
    return LabelAnswer(low, lists["missing"], lists["extra"], trans, notes.strip()[:NOTES_MAX])


# --------------------------------------------------------------------------- packet
def _png(img: RGB) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _contrast(stack: Stack) -> str:
    return str(stack.meta.get("role", {}).get("contrast") or stack.key)


def _localizer_image(study: Study, anchors: Anchors, cfg: Config) -> tuple[RGB | None, str]:
    """Full-FOV localizer slice nearest the discs, with the predicted labels drawn where they fall in it."""
    locs = [s for s in study.stacks.values() if s.meta.get("role", {}).get("localizer")]
    if not locs or not anchors.discs:
        return None, "no localizer" if not locs else "no discs to draw on the localizer"
    centroid = np.mean([np.asarray(d.center) for d in anchors.discs.values()], axis=0)
    # Prefer a sagittal localizer (its slice at the discs' x is the mid-sagittal plane), then the largest.
    st = sorted(locs, key=lambda s: (s.plane != "sag", -s.n))[0]
    k, _dist = st.nearest_slice(centroid)
    frame = slice_frame(st, k)
    base = grey_rgb(st.vol[k], series_window(st, cfg))
    base, frame = display_scale(base, frame, 640)
    im = Image.fromarray(base)
    draw = ImageDraw.Draw(im)
    tol = max(float(st.thickness), 3.0)
    shown = 0
    for lab in anchors.disc_labels():
        x, y, off = frame.to_px(np.asarray(anchors.discs[lab].center))
        if abs(off) > tol or not (0 <= x < base.shape[1] and 0 <= y < base.shape[0]):
            continue
        shown += 1
        r = max(3, round(0.008 * min(base.shape[:2])))
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 220, 0), width=2)
        draw.text((x + r + 3, y - r), lab, fill=(255, 220, 0))
    arr = np.asarray(im, dtype=np.uint8)
    lines = [
        f"{st.key} slice {k + 1}/{st.n} (full-FOV localizer)",
        f"predicted levels drawn where inside this slice: {shown} of {len(anchors.discs)}",
    ]
    return with_title(arr, lines, cfg=cfg), ""


#: Side of the sacrum zoom in display pixels; the crop is upscaled to about this size (legibility).
SACRUM_ZOOM_PX = 600


def _zoom_upscale(stack: Stack, cfg: Config) -> float:
    """Upscale factor that makes the sacrum crop about SACRUM_ZOOM_PX px across."""
    native_px = min(stack.row_mm, stack.col_mm)
    return max(1.0, SACRUM_ZOOM_PX * native_px / cfg.verify_sacrum_fov_mm)


#: Height in px of each tile of the overview sheet (the individual PNGs keep their own size).
OVERVIEW_TILE_PX = 480


def lumbosacral_context(anchors: Anchors) -> dict[str, Any] | None:
    """The lumbosacral CNN's P(L5-S1) for the lowest disc candidates, bottom-up, from ``anchors.numbering``.

    None when the check did not run or recorded no finite probability. The candidates are listed as the labeller
    named them (before any sacral drop), so the reader can see what the CNN scored.
    """
    rec = anchors.numbering.get("lumbosacral")
    if not isinstance(rec, dict):
        return None
    labels = [str(x) for x in (rec.get("candidates") or [])]
    raw = list(rec.get("p_l5s1") or [])
    if not labels or len(raw) != len(labels):
        return None
    pairs = [
        (lab, float(p))
        for lab, p in zip(labels, raw, strict=True)
        if p is not None and math.isfinite(float(p))
    ]
    if not pairs:
        return None
    return {"candidates": [lab for lab, _ in pairs], "p_l5s1": [round(p, 3) for _, p in pairs]}


def _lumbosacral_note(ctx: dict[str, Any] | None) -> str:
    if ctx is None:
        return "lumbosacral CNN: not run (no probability for the lowest candidates)"
    parts = ", ".join(f"{lab} {p:.3f}" for lab, p in zip(ctx["candidates"], ctx["p_l5s1"], strict=True))
    return f"lumbosacral CNN P(L5-S1) for the lowest candidates, bottom-up: {parts}"


def _overview(pngs: list[bytes]) -> bytes:
    """Contact sheet of the packet panels in two columns, each scaled to :data:`OVERVIEW_TILE_PX` tall."""
    tiles: list[Image.Image] = []
    for png in pngs:
        im = Image.open(io.BytesIO(png)).convert("RGB")
        w, h = im.size
        scale = OVERVIEW_TILE_PX / max(h, 1)
        tiles.append(im.resize((max(1, round(w * scale)), OVERVIEW_TILE_PX)))
    cols = min(2, len(tiles))
    rows = -(-len(tiles) // cols)
    col_w = [
        max(tiles[c + cols * r].width for r in range(rows) if c + cols * r < len(tiles)) for c in range(cols)
    ]
    sheet = Image.new("RGB", (sum(col_w), rows * OVERVIEW_TILE_PX), (0, 0, 0))
    x0 = [sum(col_w[:c]) for c in range(cols)]
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (x0[i % cols], (i // cols) * OVERVIEW_TILE_PX))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def build_packet(study: Study, anchors: Anchors, *, cfg: Config = DEFAULT) -> VerificationPacket:
    """The reader packet for ``anchors`` of ``study``. Never raises for missing series: notes say what is absent."""
    notes: list[str] = []
    images: dict[str, bytes] = {}
    if not anchors.discs:
        notes.append("no discs were labelled: the reader can only confirm that there are none")
    primary = study.stacks.get(anchors.source_stack) if anchors.source_stack else None
    if primary is None:
        primary = study.best("sag")
    if primary is None:
        notes.append("no sagittal series")
        return VerificationPacket({}, QUESTION + " " + QUESTION_RULES, [], notes)
    t2 = primary
    images[f"sag_{_contrast(t2)}_midline"] = _png(qa.overlay(study, anchors, stack=t2, cfg=cfg))
    t1s = [s for s in study.find("sag", "t1", localizer=False) if s.key != t2.key]
    if t1s:
        images[f"sag_{_contrast(t1s[0])}_midline"] = _png(qa.overlay(study, anchors, stack=t1s[0], cfg=cfg))
    else:
        notes.append("no second (T1) sagittal series")
    loc, why = _localizer_image(study, anchors, cfg) if anchors.discs else (None, "")
    if loc is not None:
        images["localizer_predicted_levels"] = _png(loc)
    elif why:
        notes.append(why)
    if anchors.discs:
        lowest = anchors.disc_labels()[-1]
        d = anchors.discs[lowest]
        centre = np.asarray(d.center, dtype=np.float64) - cfg.verify_sacrum_offset_mm * np.asarray(
            d.normal, dtype=np.float64
        )
        k_mid = qa.midline_slice(t2, anchors)  # the same slice as the overlay above
        zoom, _frame = zoom_crop(
            t2,
            centre,
            k=k_mid,
            size_mm=cfg.verify_sacrum_fov_mm,
            anchors=anchors,
            pixel_coords=False,
            upscale=_zoom_upscale(t2, cfg),
            config=cfg,
        )
        images["sacrum_zoom"] = _png(zoom)
        notes.append(
            f"sacrum zoom centred {cfg.verify_sacrum_offset_mm:g} mm below the lowest labelled disc "
            f"({lowest}); it may show no sacrum if the field of view ends above it"
        )
    lumbo = lumbosacral_context(anchors)
    if anchors.discs:
        notes.append(_lumbosacral_note(lumbo))
    if len(images) >= 2:  # one picture of all the panels, so a reader sees the whole packet at once
        images["overview"] = _overview(list(images.values()))
    return VerificationPacket(
        images=images,
        question=QUESTION + " " + QUESTION_RULES,
        predicted_levels=anchors.disc_labels(),
        notes=notes,
        lumbosacral=lumbo,
    )


# --------------------------------------------------------------------------- applying
def _neighbours(anchors: Anchors, label: str) -> tuple[str | None, str | None]:
    """Labels of the labelled disc cranial and the one caudal to ``label``, matched by vertebra name."""
    up, low = split_disc(label)
    cran = next((d for d in anchors.discs if split_disc(d)[1] == up), None)
    caud = next((d for d in anchors.discs if split_disc(d)[0] == low), None)
    return cran, caud


def _mean_spacing(anchors: Anchors) -> np.ndarray | None:
    """Mean LPS vector from each labelled disc to the next one caudal to it (points inferior)."""
    labs = anchors.disc_labels()
    if len(labs) < 2:
        return None
    steps = [
        np.asarray(anchors.discs[b].center, dtype=np.float64)
        - np.asarray(anchors.discs[a].center, dtype=np.float64)
        for a, b in itertools.pairwise(labs)
    ]
    return np.mean(steps, axis=0)


def _place_missing(anchors: Anchors, label: str, cfg: Config) -> str | None:
    """Add a reader-named missing disc at a position interpolated or extrapolated from its neighbours.

    The position is a guess (not a measurement) and gets ``cfg.placed_disc_confidence``. Returns the method
    used, or None when it cannot be placed (no labelled neighbour, or a single disc with no spacing).
    """
    cran, caud = _neighbours(anchors, label)
    if cran is not None and caud is not None:
        a, b = anchors.discs[cran], anchors.discs[caud]
        centre = (np.asarray(a.center, dtype=np.float64) + np.asarray(b.center, dtype=np.float64)) / 2.0
        method = "interpolated between its neighbours"
        normal = np.asarray(a.normal, dtype=np.float64) + np.asarray(b.normal, dtype=np.float64)
        heights = [h for h in (a.height_mm, b.height_mm) if h]
    else:
        spacing = _mean_spacing(anchors)
        if spacing is None:
            return None
        if cran is not None:  # no disc caudal to it: the missing disc is the lowest one
            a = anchors.discs[cran]
            centre = np.asarray(a.center, dtype=np.float64) + spacing
            method = "extrapolated below the lowest labelled disc"
        elif caud is not None:  # no disc cranial to it: the missing disc is the highest one
            a = anchors.discs[caud]
            centre = np.asarray(a.center, dtype=np.float64) - spacing
            method = "extrapolated above the highest labelled disc"
        else:
            return None
        normal = np.asarray(a.normal, dtype=np.float64)
        heights = [a.height_mm] if a.height_mm else []
    height = float(np.mean(heights)) if heights else None
    anchors.add_disc(
        label,
        centre,
        normal=normal,
        height_mm=height,
        confidence=cfg.placed_disc_confidence,
        who="reader",
        source="ai_reader",
        why=f"reader reports a missing disc; position {method}, not measured",
        config=cfg,
    )
    return method


def _lowest_action(work: Anchors, target: str, why: str, who: str, source: str) -> None:
    """Make the lowest displayed disc carry the reader's true name ``target`` (see module docstring)."""
    bottom = work.disc_labels()[-1]
    if bottom == target:
        return
    b_up = split_disc(bottom)[0]
    t_up = split_disc(target)[0]
    if b_up == t_up:
        # only the lower vertebra differs: a lumbarized S1 (L5-S1 -> L5-L6); the discs above keep their names
        work.relabel({bottom: target}, who=who, source=source, why=why + " (lowest disc only; lumbarized S1)")
        return
    offset = _COUNT_INDEX[t_up] - _COUNT_INDEX[b_up]
    work.shift_labels(offset, who=who, source=source, why=why)
    if work.disc_labels()[-1] != target:  # work is a copy: nothing has changed for the caller yet
        raise AnswerError(f"lowest_disc_is {target} does not fit the displayed bottom disc {bottom}")


def review_status(anchors: Anchors, source: str) -> str:
    """The numbering status after a reviewed correction, consistent with the applicability record.

    A review that cannot see the bone (``applicability.unconfirmed_reasons``, e.g. no sacrum in view) stays
    ``unconfirmed``. An AI reader's answer is never enough for ``confirmed``: it is at most ``probable``. A human
    review may reach ``confirmed`` when its confidence does.
    """
    if anchors.applicability.get("unconfirmed_reasons"):
        return STATUS_UNCONFIRMED
    status = numbering_status(float(anchors.numbering.get("confidence", 0.0) or 0.0))
    if source != "human" and status != STATUS_UNCONFIRMED:
        return STATUS_PROBABLE
    return status


def apply_answer(
    anchors: Anchors,
    answer: str | Mapping[str, Any] | LabelAnswer,
    *,
    who: str = "reader",
    source: str = "ai_reader",
    cfg: Config = DEFAULT,
) -> Anchors:
    """Apply a reader's answer to ``anchors`` (changed in place and returned), with provenance.

    Order: remove the ``extra`` labels (as displayed); then correct the lowest displayed disc to
    ``lowest_disc_is`` (L5-S1: nothing; L4-L5 or another upper-vertebra change: every level shifts by that
    many, :meth:`Anchors.shift_labels`; L5-L6: only the lowest disc is renamed, a lumbarized S1); then place the
    ``missing`` discs by their true names. ``unsure`` changes nothing and lowers the confidence. The change is
    made on a copy and copied back only when every step succeeds, so a failing answer leaves ``anchors``
    untouched and raises :class:`AnswerError` (or ValueError for a request the Anchors API refuses).
    """
    ans = parse_answer(answer)
    if source not in ("ai_reader", "human"):
        raise ValueError("a reviewed correction needs source 'ai_reader' or 'human'")
    review_conf = cfg.review_confidence_human if source == "human" else cfg.review_confidence_ai
    prior_labels = anchors.disc_labels()
    _pc = anchors.numbering.get("confidence")
    prior_conf = float(_pc) if isinstance(_pc, (int, float)) else None
    _ps = anchors.numbering.get("status")
    prior_status = str(_ps) if isinstance(_ps, str) else None
    detail: dict[str, Any] = {"answer": ans.to_dict()}
    if not anchors.discs:
        anchors.record_correction(
            "review",
            who=who,
            source=source,
            why="no discs to verify",
            detail=detail,
            warning="label review: no discs were labelled, so there was nothing to verify",
        )
        return anchors
    if ans.lowest_disc_is == "unsure":
        anchors.record_correction(
            "review",
            who=who,
            source=source,
            why="reader unsure which disc is lowest: no structural change",
            detail=detail,
            confidence=cfg.review_unsure_confidence,
            warning="label review: reader unsure of the lowest disc; levels are unverified",
        )
        anchors.numbering["status"] = STATUS_UNCONFIRMED
        return anchors

    work = Anchors.from_dict(anchors.to_dict())
    why = f"reader: lowest disc shown is {ans.lowest_disc_is}"
    for lab in ans.extra:
        if lab not in work.discs:
            work.record_correction(
                "unmatched_extra",
                who=who,
                source=source,
                why=f"reader named extra {lab}, not labelled",
                warning=f"label review: reader named an extra disc {lab} that is not labelled",
            )
            continue
        work.drop_disc(lab, who=who, source=source, why=f"reader: {lab} is not a disc")
        detail.setdefault("dropped", []).append(lab)
    if work.discs:
        _lowest_action(work, ans.lowest_disc_is, why, who, source)
        detail["lowest_before_review"] = anchors.disc_labels()[-1]
    unplaced: list[str] = []
    for lab in ans.missing:
        if lab in work.discs:
            work.record_correction(
                "unmatched_missing",
                who=who,
                source=source,
                why=f"reader named missing {lab}, but it is already labelled",
                warning=f"label review: reader reported {lab} missing but it is labelled",
            )
            continue
        method = _place_missing(work, lab, cfg)
        if method is None:
            unplaced.append(lab)
            work.record_correction(
                "unplaced_missing",
                who=who,
                source=source,
                why=f"reader named missing {lab}; no neighbour",
                warning=f"label review: missing disc {lab} could not be placed (no labelled neighbour)",
            )
        else:
            detail.setdefault("placed", []).append({"label": lab, "method": method})
    if ans.missing or ans.extra:
        detail["missing"] = list(ans.missing)
        detail["extra"] = list(ans.extra)
    if ans.transitional is not None:
        work.numbering["transitional"] = ans.transitional
        kind = None
        moved = work.disc_labels()[-1] != anchors.disc_labels()[-1]
        if ans.transitional and moved:
            kind = "sacralized" if ans.lowest_disc_is == "L4-L5" else "lumbarized"
        work.numbering["transitional_kind"] = kind
        if ans.transitional is False and moved:
            work.warnings.append("label review: reader says not transitional but the lowest disc was moved")
    conf = max(0.0, review_conf - cfg.review_unplaced_penalty * len(unplaced))
    relabelled = work.disc_labels() != prior_labels
    # An AI reader's answer may only LOWER trust, never raise it: in a blinded experiment
    # (docs/labeling/lab24_ai_verify.md) AI confirmations lifted wrong numberings from unconfirmed to
    # probable, while AI-proposed shifts were never right. So keep the prior confidence as a ceiling,
    # never exceed the prior status, and treat any AI relabelling as a suggestion awaiting a human.
    if source != "human" and prior_conf is not None:
        conf = min(conf, prior_conf)
    work.record_correction("review", who=who, source=source, why=why, detail=detail, confidence=conf)
    status = review_status(work, source)
    if source != "human":
        status = _min_status(status, prior_status)
        if relabelled:
            status = STATUS_UNCONFIRMED
            work.warnings.append(
                "label review: an AI reader changed the level names; treat them as unconfirmed until a human "
                "reviewer confirms"
            )
    work.numbering["status"] = status
    anchors.__dict__.update(work.__dict__)
    return anchors


_STATUS_RANK = {STATUS_UNCONFIRMED: 0, STATUS_PROBABLE: 1, STATUS_CONFIRMED: 2}


def _min_status(a: str, b: str | None) -> str:
    """The less trusted of two numbering statuses (``b`` None or unknown: ``a``)."""
    if b not in _STATUS_RANK:
        return a
    return a if _STATUS_RANK.get(a, 0) <= _STATUS_RANK[b] else str(b)


def verify_labels(
    study: Study,
    anchors: Anchors,
    ask: Callable[[VerificationPacket], str | Mapping[str, Any]],
    *,
    who: str = "reader",
    source: str = "ai_reader",
    cfg: Config = DEFAULT,
) -> tuple[Anchors, VerificationPacket, LabelAnswer]:
    """Build a packet, ask ``ask`` for an answer, validate it and apply it. Returns (anchors, packet, answer).

    ``ask`` is any callable (see the module docstring for wiring a vision model). Raises AnswerError if the
    answer is invalid; nothing is changed in that case.
    """
    packet = build_packet(study, anchors, cfg=cfg)
    answer = parse_answer(ask(packet))
    anchors = apply_answer(anchors, answer, who=who, source=source, cfg=cfg)
    return anchors, packet, answer
