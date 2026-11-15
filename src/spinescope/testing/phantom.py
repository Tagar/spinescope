"""Synthetic lumbar spine phantom with known ground truth, rendered to stacks or DICOM files.

The phantom is analytic and defined in patient LPS mm, so any slice geometry (sagittal, axial,
oblique slabs, flipped in-plane axes...) can be rendered exactly. It is meant for unit tests and
examples.

Two looks are available through ``PhantomSpec(realism=...)``: ``"cartoon"`` (the default: hand-drawn shapes and an
intensity table, kept so the existing tests keep their meaning) and ``"realistic"`` (kidney-bean bodies with
morphometry, spin-echo signal from published relaxation times, partial-volume averaging; see
``spinescope.testing.realistic`` and ``docs/labeling/lab21_phantom.md``).

Example::

    ph = SpinePhantom(PhantomSpec(degenerated={"L4-L5": 0.3}, modic1=[("L4-L5", "inf", "L")]))
    sag = ph.render("sag", "t2")
    paths = write_standard_study(ph, tmp_path)   # sag T1/T2/STIR, ax T1/T2, oblique slabs, localizer
"""

from __future__ import annotations

import hashlib
import os
import zlib
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any

import numpy as np

from ..geometry import Stack, unit

DISCS = ("T11-T12", "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")  # superior -> inferior
BODIES = ("T11", "T12", "L1", "L2", "L3", "L4", "L5")  # body above each disc, same order

# Tissue labels
AIR, SOFT, FAT, MUSCLE, MARROW, CORTEX, ANNULUS, NUCLEUS, CSF, CORD, VESSEL, MODIC1, PSOAS = range(13)
# Added for the robustness scenarios (ids appended so the original ids keep their values).
OEDEMA, HARDWARE, SCHMORL, BLOOM = range(13, 17)
# Added for the realistic anatomy (``realism="realistic"``); the cartoon never produces these labels.
DURA, LIGAMENT, FACET, BOWEL, LIVER, KIDNEY, NERVE, REMNANT = range(17, 25)
TISSUES = (
    "air",
    "soft",
    "fat",
    "muscle",
    "marrow",
    "cortex",
    "annulus",
    "nucleus",
    "csf",
    "cord",
    "vessel",
    "modic1",
    "psoas",
    "oedema",
    "hardware",
    "schmorl",
    "bloom",
    "dura",
    "ligament",
    "facet",
    "bowel",
    "liver",
    "kidney",
    "nerve",
    "remnant",
)

#: Mean signal per tissue and contrast (arbitrary units, roughly 1.5 T FSE-like ordering).
INTENSITY = {
    "t1": {
        AIR: 0,
        SOFT: 380,
        FAT: 950,
        MUSCLE: 330,
        MARROW: 720,
        CORTEX: 60,
        ANNULUS: 300,
        NUCLEUS: 360,
        CSF: 160,
        CORD: 460,
        VESSEL: 40,
        MODIC1: 360,
        PSOAS: 330,
        OEDEMA: 200,  # marrow oedema is T1-dark
        HARDWARE: 0,
        SCHMORL: 360,
        BLOOM: 120,
    },
    "t2": {
        AIR: 0,
        SOFT: 420,
        FAT: 760,
        MUSCLE: 240,
        MARROW: 470,
        CORTEX: 50,
        ANNULUS: 160,
        NUCLEUS: 950,
        CSF: 1450,
        CORD: 520,
        VESSEL: 40,
        MODIC1: 900,
        PSOAS: 240,
        OEDEMA: 850,
        HARDWARE: 0,
        SCHMORL: 950,
        BLOOM: 150,
    },
    "stir": {
        AIR: 0,
        SOFT: 380,
        FAT: 50,
        MUSCLE: 210,
        MARROW: 130,
        CORTEX: 40,
        ANNULUS: 140,
        NUCLEUS: 760,
        CSF: 1450,
        CORD: 460,
        VESSEL: 40,
        MODIC1: 900,
        PSOAS: 210,
        OEDEMA: 850,
        HARDWARE: 0,
        SCHMORL: 760,
        BLOOM: 60,
    },
    # Proton density: fat and fluid are not suppressed or boosted much; nucleus is brighter than annulus.
    "pd": {
        AIR: 0,
        SOFT: 380,
        FAT: 800,
        MUSCLE: 300,
        MARROW: 500,
        CORTEX: 40,
        ANNULUS: 260,
        NUCLEUS: 700,
        CSF: 350,
        CORD: 420,
        VESSEL: 40,
        MODIC1: 600,
        PSOAS: 300,
        OEDEMA: 650,
        HARDWARE: 0,
        SCHMORL: 700,
        BLOOM: 150,
    },
}

#: Acquisition parameters written to DICOM so series classification can be tested.
ACQ: dict[str, dict[str, Any]] = {
    "t1": {"TR": 550.0, "TE": 8.0, "TI": None, "seq": "SE", "desc": "{plane} T1"},
    "t2": {"TR": 4000.0, "TE": 95.0, "TI": None, "seq": "SE", "desc": "{plane} T2"},
    "stir": {"TR": 5000.0, "TE": 45.0, "TI": 150.0, "seq": "IR", "desc": "{plane} STIR"},
    "pd": {"TR": 3000.0, "TE": 15.0, "TI": None, "seq": "SE", "desc": "{plane} PD"},
}

LAT0 = np.array([1.0, 0.0, 0.0])  # lateral axis of an unrotated level
SCOLIOSIS_SIGMA_MM = 60.0  # width of the thoracolumbar curve around its apex (L2)
TRANSITIONAL = (None, "lumbarised_s1", "sacralised_l5")
FIELD_STRENGTHS = (1.5, 3.0)
REALISMS = (
    "cartoon",
    "realistic",
)  # cartoon: hand-drawn shapes (default); realistic: see testing/realistic.py
_LABEL_CACHE_SIZE = 512  # realistic label volumes kept per phantom (one stack set is about 160 sample blocks)
CARTOON_SACRAL_SLOPE = 28.0
CARTOON_SEGMENT_LORDOSIS: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0, 14.0)


@dataclass
class PhantomSpec:
    """Phantom parameters. Angles in degrees, lengths in mm."""

    sacral_slope: float | None = None  # degrees; None = 28 (cartoon) or 40 (realistic, Roussouly 2005)
    segment_lordosis: tuple[float, ...] | None = None  # T11-T12..L4-L5 relative turns; None = per realism
    body_height: float = 26.0
    disc_height: float = 10.0
    body_half_width: float = 22.0
    body_half_depth: float = 16.0
    canal_offset: float = 27.0  # canal centre, mm posterior to the body centre
    canal_radius: float = 8.0
    cord_radius: float = 4.0
    conus_body: str = "L1"  # cord ends at the middle of this body
    s1_top: tuple[float, float, float] = (0.0, 45.0, -85.0)  # LPS of the S1 superior endplate centre
    degenerated: dict[str, float] = field(default_factory=dict)  # disc -> nucleus brightness factor 0..1
    modic1: list[tuple[str, str, str | None]] = field(
        default_factory=list
    )  # (disc, 'inf'|'sup', 'L'|'R'|None)
    herniation: dict[str, tuple[str, float]] = field(
        default_factory=dict
    )  # disc -> ('central'|'L'|'R', depth mm)
    # Draw herniated disc tissue over the thecal sac (the sac is pushed back), instead of letting the sac overwrite it.
    # Off by default; the default phantom is unchanged.
    herniation_displaces_sac: bool = False
    noise: float = 0.02  # Rician noise sigma as a fraction of the brightest tissue
    seed: int = 0
    # ---- robustness options. All off by default: the default phantom is unchanged.
    scoliosis_mm: float = 0.0  # lateral apex displacement of the thoracolumbar column (apex at L2)
    scoliosis_rot_deg: float = 0.0  # axial rotation of the levels at the apex, degrees
    thoracic_top: int = 11  # top thoracic body number: 10 adds T10 and one more disc
    transitional: str | None = None  # 'lumbarised_s1' (6 lumbar, full S1-S2 disc) | 'sacralised_l5'
    collapsed: dict[str, float] = field(default_factory=dict)  # disc -> fraction of disc height lost
    fracture: dict[str, float] = field(default_factory=dict)  # body -> anterior wedge fraction; oedema inside
    schmorl: list[tuple[str, str]] = field(default_factory=list)  # (body, 'inf'|'sup') Schmorl node
    hardware: tuple[
        str, ...
    ] = ()  # 'screws:<body>', 'cage:<disc>', 'rod' (posterior rod at the spinous line)
    ribs: bool = False  # rib tubes off the thoracic bodies (visible off the midline)
    bias_field: float = 0.0  # surface-coil intensity bias amplitude (relative; 0 = none)
    field_strength: float = 1.5  # 1.5 or 3.0 T: intensity and noise variant (cartoon: an intensity table)
    # ---- realism. ``realistic`` replaces the drawn shapes and intensities by morphometry and spin-echo physics.
    realism: str = "cartoon"  # 'cartoon' (default, original look) | 'realistic' (testing/realistic.py)
    supersample: tuple[int, int] = (2, 2)  # realistic only: (through-plane, in-plane per axis) sub-samples
    gibbs: bool = False  # realistic only: Gibbs ringing from truncated k-space
    # ---- listhesis (spondylolisthesis) option, added for the alignment analysis. Off by default.
    # disc -> translation in mm of the body ABOVE that disc, with every structure above it (the column moves as one),
    # along that body's anterior-posterior axis. Positive = posterior (retrolisthesis), negative = anterior
    # (anterolisthesis). The disc at the slip level takes half the shift (it shears between the two bodies).
    slip: dict[str, float] = field(default_factory=dict)
    # ---- facet joints (realistic phantom only; the cartoon phantom draws no facets). Off by default: the joint
    # sits at facet_lateral_mm (both sides) and is healthy (no fluid) unless it has an effusion.
    facet_lateral_mm: float = 19.0  # facet joint centre, mm lateral to the midline
    facet_effusion: dict[str, tuple[str, float]] = field(
        default_factory=dict
    )  # disc -> ('L'|'R', fluid extent mm along the AP axis of the joint): CSF-like fluid in the joint


def _segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the segment a-b."""
    ab = b - a
    t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def _polyline_distance(pts: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance from each point to the polyline ``line`` (m, 3)."""
    best = np.full(len(pts), np.inf)
    for i in range(len(line) - 1):
        best = np.minimum(best, _segment_distance(pts, line[i], line[i + 1]))
    return best


class SpinePhantom:
    """Analytic lumbar spine. ``truth`` holds the ground-truth anatomy anchors in LPS mm."""

    def __init__(self, spec: PhantomSpec | None = None) -> None:
        self._label_cache: dict[bytes, np.ndarray] = {}
        spec = spec or PhantomSpec()
        if spec.realism not in REALISMS:
            raise ValueError(f"realism must be one of {REALISMS}, got {spec.realism!r}")
        if spec.realism == "realistic":
            from . import realistic as _re

            defaults: dict[str, Any] = {
                "sacral_slope": _re.SACRAL_SLOPE_DEG,
                "segment_lordosis": _re.SEGMENT_LORDOSIS_DEG,
            }
        else:
            defaults = {"sacral_slope": CARTOON_SACRAL_SLOPE, "segment_lordosis": CARTOON_SEGMENT_LORDOSIS}
        resolved = {k: (getattr(spec, k) if getattr(spec, k) is not None else v) for k, v in defaults.items()}
        self.spec = replace(spec, **resolved)
        self._build()

    @property
    def realistic(self) -> bool:
        """True for ``realism="realistic"``."""
        return self.spec.realism == "realistic"

    # ----------------------------------------------------------------- anatomy
    def _level_names(self) -> tuple[list[str], list[str], str]:
        """Disc names, body names (both superior -> inferior, equal length) and the sacral block name."""
        sp = self.spec
        if sp.transitional not in TRANSITIONAL:
            raise ValueError(f"unknown transitional anatomy {sp.transitional!r}")
        if sp.field_strength not in FIELD_STRENGTHS:
            raise ValueError(f"field_strength must be one of {FIELD_STRENGTHS}")
        n_lum = 6 if sp.transitional == "lumbarised_s1" else 5
        bodies = [f"T{k}" for k in range(sp.thoracic_top, 13)] + [f"L{k}" for k in range(1, n_lum + 1)]
        sacrum = "S2" if sp.transitional == "lumbarised_s1" else "S1"
        # Disc names count up from the lowest lumbar vertebra (the SPIDER convention): the lowest disc is
        # L5-S1, or L6-S1 when S1 is lumbarised (its anatomical name, S1-S2, is in truth["discs"][...]["anatomical"]).
        discs = [f"{a}-{b}" for a, b in pairwise(bodies)] + [f"{bodies[-1]}-S1"]
        return discs, bodies, sacrum

    def _disc_height(self, disc: str) -> float:
        base = self.spec.disc_height
        if self.realistic:
            from . import realistic as _re

            base = _re.DISC_HEIGHT_MM.get(disc, base)
        return base * (1.0 - self.spec.collapsed.get(disc, 0.0))

    def _body_dims(self, name: str) -> tuple[float, float, float]:
        """(half width, half depth, height) of a vertebral body or of the S1 block, in mm."""
        sp = self.spec
        if self.realistic:
            from . import realistic as _re

            return (
                _re.BODY_HALF_WIDTH_MM.get(name, sp.body_half_width),
                _re.BODY_HALF_DEPTH_MM.get(name, sp.body_half_depth),
                _re.BODY_HEIGHT_MM.get(name, sp.body_height),
            )
        return sp.body_half_width, sp.body_half_depth, sp.body_height

    def _build(self) -> None:
        sp = self.spec
        slope = float(sp.sacral_slope if sp.sacral_slope is not None else CARTOON_SACRAL_SLOPE)
        seg_spec = sp.segment_lordosis if sp.segment_lordosis is not None else CARTOON_SEGMENT_LORDOSIS
        disc_names, body_names, sacrum = self._level_names()
        self._body_names, self._sacrum = body_names, sacrum
        # disc tilt from axial, inferior -> superior. segment_lordosis is ordered T11-T12 .. L4-L5 (top first);
        # extra thoracic discs take the top value.
        seg = list(seg_spec)
        while len(seg) < len(disc_names) - 1:
            seg.insert(0, seg[0])
        turns = seg[: len(disc_names) - 1][::-1]  # L4-L5 .. top
        tilts = [slope]
        for t in turns:
            tilts.append(tilts[-1] - t)
        p = np.array(sp.s1_top, dtype=np.float64)
        discs: dict[str, dict[str, Any]] = {}
        bodies: dict[str, dict[str, Any]] = {}
        for i, (disc, body) in enumerate(zip(disc_names[::-1], body_names[::-1], strict=True)):
            tilt = tilts[i]
            nxt = tilts[i + 1] if i + 1 < len(tilts) else tilts[-1]
            h = self._disc_height(disc)
            hw_b, hd_b, hb = self._body_dims(body)  # the body above the disc sets the disc outline
            n = self._normal(tilt)
            a = self._ap(tilt)
            c = p + n * h / 2
            discs[disc] = {
                "center": c,
                "normal": n,
                "ap": a,
                "lat": LAT0.copy(),
                "anterior": c - a * hd_b,
                "posterior": c + a * hd_b,
                "tilt_deg": tilt,
                "height_mm": h,
            }
            if self.realistic:
                discs[disc].update(hw=hw_b, hd=hd_b)
            bottom = p + n * h
            tb = (tilt + nxt) / 2
            nb, ab = self._normal(tb), self._ap(tb)
            bc = bottom + nb * hb / 2
            bodies[body] = {
                "center": bc,
                "normal": nb,
                "ap": ab,
                "lat": LAT0.copy(),
                "tilt_deg": tb,
                "hw": hw_b,
                "hd": hd_b,
                "half_height": hb / 2,
            }
            p = bottom + nb * hb
        # Sacrum: a block below the S1 endplate, tilted by the sacral slope (S1 is 30 mm high in both realisms)
        n0, a0 = self._normal(slope), self._ap(slope)
        s1 = np.array(sp.s1_top) - n0 * 15.0
        hw_s, hd_s, _hb = self._body_dims(sacrum)
        bodies[sacrum] = {
            "center": s1,
            "normal": n0,
            "ap": a0,
            "lat": LAT0.copy(),
            "tilt_deg": slope,
            "half_height": 15.0,
            "hw": hw_s,
            "hd": hd_s,
        }
        self.thecal_end_z = float(
            (np.array(sp.s1_top) - n0 * 30.0)[2]
        )  # S1 bottom: the dural sac ends near S2
        self._sacral_segments: list[dict[str, Any]] = []
        if self.realistic:
            from . import realistic as _re

            self._sacral_segments = _re.sacral_segments(
                np.array(sp.s1_top) - n0 * 30.0, slope, self._normal, self._ap
            )
        self.discs, self.bodies = discs, bodies
        for item in sp.hardware:
            kind, _, name = item.partition(":")
            valid = (
                item == "rod" or (kind == "screws" and name in bodies) or (kind == "cage" and name in discs)
            )
            if not valid:
                raise ValueError(f"unknown hardware {item!r}: use 'rod', 'screws:<body>' or 'cage:<disc>'")
        self._slip_truth: dict[str, dict[str, float]] = {}
        if sp.slip:
            self._apply_slip()
        if sp.scoliosis_mm or sp.scoliosis_rot_deg:
            self._apply_scoliosis()
        canal = [bodies[b]["center"] + bodies[b]["ap"] * sp.canal_offset for b in (*body_names, sacrum)]
        top = canal[0] + bodies[body_names[0]]["normal"] * 40
        bottom = canal[-1] - bodies[sacrum]["normal"] * 30
        self.canal_line = np.array([top, *canal, bottom])
        rod = [
            bodies[b]["center"] + bodies[b]["ap"] * (sp.canal_offset + 22.0) for b in (*body_names, sacrum)
        ]
        self.rod_line = np.array(rod)
        self.conus_z = float(bodies[sp.conus_body]["center"][2])
        findings: dict[str, Any] = {
            "transitional": sp.transitional,
            "lumbar_levels": [b for b in body_names if b.startswith("L")],
            "sacrum": sacrum,
            "collapsed": sorted(sp.collapsed),
            "fracture_bodies": sorted(sp.fracture),
            "schmorl": [list(x) for x in sp.schmorl],
            "hardware": list(sp.hardware),
            "scoliosis_mm": sp.scoliosis_mm,
            "scoliosis_rot_deg": sp.scoliosis_rot_deg,
            "field_strength": sp.field_strength,
        }
        self.truth: dict[str, Any] = {
            "discs": {
                k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                for k, v in discs.items()
            },
            "vertebrae": {k: v["center"].tolist() for k, v in bodies.items()},
            "midline_x": float(sp.s1_top[0]),
            "conus_z": self.conus_z,
            "conus_level": sp.conus_body,
            "transitional": sp.transitional,
            "findings": findings,
        }
        if self.realistic:
            self.truth["facets"] = self._facet_truth()
        if self._slip_truth:
            self.truth["listhesis"] = {k: dict(v) for k, v in self._slip_truth.items()}
        if sp.transitional == "sacralised_l5":
            self.truth["discs"]["L5-S1"]["partial"] = True  # only one side of the L5-S1 disc is preserved
        if sp.transitional == "lumbarised_s1":
            self.truth["discs"][disc_names[-1]]["anatomical"] = (
                "S1-S2"  # the full disc below the lumbarised S1
            )

    def _facet_truth(self) -> dict[str, Any]:
        """Facet joint centres (LPS mm) and the injected effusion extent per disc and side (realistic phantom)."""
        sp = self.spec
        out: dict[str, Any] = {}
        for d, v in self.discs.items():
            eff = sp.facet_effusion.get(d)
            per: dict[str, Any] = {}
            for side, sgn in (("L", 1.0), ("R", -1.0)):
                c = v["center"] + sgn * sp.facet_lateral_mm * v["lat"] + (sp.canal_offset + 1.0) * v["ap"]
                width = float(eff[1]) if eff is not None and eff[0] == side else 0.0
                per[side] = {"center": [float(x) for x in c], "effusion_mm": width}
            out[d] = per
        return out

    def _apply_slip(self) -> None:
        """Translate the column above each slipped disc in AP (listhesis). Rigid for the column above, half shift at the disc."""
        sp = self.spec
        for disc, shift in sp.slip.items():
            if disc not in self.discs:
                raise ValueError(f"unknown slip level {disc!r}; discs are {sorted(self.discs)}")
            upper = self._body_above(disc)
            ap_u = self.bodies[upper]["ap"]
            s_vec = float(shift) * ap_u
            z_disc = float(self.discs[disc]["center"][2])
            items: list[dict[str, Any]] = [v for v in self.discs.values() if v is not self.discs[disc]]
            items += [v for v in self.bodies.values()]
            for v in items:
                if (
                    float(v["center"][2]) > z_disc
                ):  # the column above the disc (bodies and discs), moved as one
                    v["center"] = v["center"] + s_vec
                    for key in ("anterior", "posterior"):
                        if key in v:
                            v[key] = v[key] + s_vec
            d = self.discs[disc]
            d["center"] = d["center"] + 0.5 * s_vec
            for key in ("anterior", "posterior"):
                if key in d:
                    d[key] = d[key] + 0.5 * s_vec
            lower = self._body_below(disc)
            ap_l = self.bodies[lower]["ap"]
            self._slip_truth[disc] = {
                "slip_mm": float(shift),
                # Anterior-positive translation of the upper body on the lower one, along the lower body's AP axis.
                "translation_anterior_mm": float(-shift * float(ap_u @ ap_l)),
            }

    def _body_above(self, disc: str) -> str:
        """Name of the vertebral body directly above a disc (the first name of the disc label)."""
        return disc.split("-")[0]

    def _body_below(self, disc: str) -> str:
        """Name of the vertebral body directly below a disc (the second name of the disc label)."""
        return disc.split("-")[1]

    def _apply_scoliosis(self) -> None:
        """C-shaped thoracolumbar curve: lateral shift and axial rotation, largest at L2, zero at the sacrum."""
        sp = self.spec
        apex = float(self.bodies["L2"]["center"][2])
        rot = np.radians(sp.scoliosis_rot_deg)
        items: list[dict[str, Any]] = [*self.discs.values()]
        items += [v for k, v in self.bodies.items() if k != self._sacrum]
        for v in items:
            z = float(v["center"][2])
            w = float(np.exp(-(((z - apex) / SCOLIOSIS_SIGMA_MM) ** 2)))
            dx = sp.scoliosis_mm * w * np.array([1.0, 0.0, 0.0])
            for key in ("center", "anterior", "posterior"):
                if key in v:
                    v[key] = v[key] + dx
            th = rot * w
            lat, ap = v["lat"], v["ap"]
            v["lat"] = np.cos(th) * lat + np.sin(th) * ap
            v["ap"] = -np.sin(th) * lat + np.cos(th) * ap
            if "anterior" in v:  # disc endplate points follow the rotated AP axis
                c = v["center"]
                v["anterior"] = c - v["ap"] * sp.body_half_depth
                v["posterior"] = c + v["ap"] * sp.body_half_depth

    @staticmethod
    def _normal(tilt_deg: float) -> np.ndarray:
        t = np.radians(tilt_deg)
        return np.array([0.0, -np.sin(t), np.cos(t)])  # superior, tilted anteriorly

    @staticmethod
    def _ap(tilt_deg: float) -> np.ndarray:
        t = np.radians(tilt_deg)
        return np.array([0.0, np.cos(t), np.sin(t)])  # anterior -> posterior within the disc plane

    # ------------------------------------------------------------------ labels
    def labels(self, pts: np.ndarray) -> np.ndarray:
        """Tissue label for each LPS point, shape (m, 3) -> (m,)."""
        if self.realistic:
            return self._labels_cached(np.asarray(pts, dtype=np.float64))
        sp = self.spec
        pts = np.asarray(pts, dtype=np.float64)
        lab = np.full(len(pts), AIR, dtype=np.int8)
        ref = self.bodies["L3"]["center"]
        # body outline (ellipse in the axial plane), subcutaneous fat rim
        ex = (pts[:, 0] / 150.0) ** 2 + ((pts[:, 1] - (ref[1] - 10)) / 115.0) ** 2
        lab[ex <= 1.0] = FAT
        lab[ex <= 0.82] = SOFT
        # paraspinal muscles and psoas follow the canal laterally
        cz = self.canal_line
        y_canal = np.interp(pts[:, 2], cz[::-1, 2], cz[::-1, 1])
        for sx in (-1.0, 1.0):
            m = ((pts[:, 0] - sx * 32) / 20.0) ** 2 + ((pts[:, 1] - (y_canal + 22)) / 24.0) ** 2 <= 1.0
            lab[m & (ex <= 0.82)] = MUSCLE
        below_l1 = pts[:, 2] < self.bodies["L1"]["center"][2]
        y_body = np.interp(pts[:, 2], cz[::-1, 2], cz[::-1, 1]) - sp.canal_offset
        for sx in (-1.0, 1.0):
            ps = ((pts[:, 0] - sx * 34) / 13.0) ** 2 + ((pts[:, 1] - (y_body - 2)) / 13.0) ** 2 <= 1.0
            lab[ps & below_l1] = PSOAS
        # aorta: left-anterior of the bodies, bifurcating at the L4 body
        above_l4 = pts[:, 2] > self.bodies["L4"]["center"][2]
        ao = (pts[:, 0] - 16) ** 2 + (pts[:, 1] - (y_body - 30)) ** 2 <= 9.0**2
        lab[ao & above_l4] = VESSEL
        # posterior elements (laminae ring + spinous process) around the canal
        dist, _seg_t = self._canal_distance(pts)
        ring = (dist > sp.canal_radius + 1.0) & (dist < sp.canal_radius + 6.0) & (pts[:, 1] > y_canal)
        lab[ring] = MARROW
        for b in (*self._body_names, self._sacrum):
            v = self.bodies[b]
            c = v["center"] + v["ap"] * (sp.canal_offset + 22)
            q = pts - c
            sproc = (q @ v["lat"] / 5.0) ** 2 + ((q @ v["ap"]) / 14.0) ** 2 + (
                (q @ v["normal"]) / 9.0
            ) ** 2 <= 1.0
            lab[sproc] = MARROW
        # vertebral bodies (with a thin cortical shell) and discs
        for name, v in self.bodies.items():
            hh = v.get("half_height", sp.body_height / 2)
            q = pts - v["center"]
            lr, ap, up = q @ v["lat"], q @ v["ap"], q @ v["normal"]
            rr = (np.abs(lr) / sp.body_half_width) ** 2.6 + (np.abs(ap) / sp.body_half_depth) ** 2.6
            if name in sp.fracture:  # anterior wedge: the anterior height is lost, the posterior wall is kept
                hh = hh * (1.0 - sp.fracture[name] * np.clip(-ap / sp.body_half_depth, 0.0, 1.0))
            inside = (rr <= 1.0) & (np.abs(up) <= hh)
            shell = inside & ((rr > 0.86) | (np.abs(up) > hh - 1.0))
            lab[inside] = MARROW
            lab[shell] = CORTEX
            if name in sp.fracture:  # STIR/T2 bright oedema fills the fractured body
                lab[inside & ~shell] = OEDEMA
        displaced: list[np.ndarray] = []
        for d, v in self.discs.items():
            q = pts - v["center"]
            lr, ap, up = q @ v["lat"], q @ v["ap"], q @ v["normal"]
            r2 = (lr / sp.body_half_width) ** 2 + (ap / sp.body_half_depth) ** 2
            dh = v["height_mm"]
            half = dh / 2 * (1.0 - 0.25 * r2)
            full = (r2 <= 1.0) & (np.abs(up) <= half)
            disc = full
            if sp.transitional == "sacralised_l5" and d == "L5-S1":
                # partial disc: the right side of the L5-S1 level is bridged by bone
                keep = lr > -4.0
                disc = full & keep
                lab[full & ~keep] = MARROW
            h = sp.herniation.get(d)
            if h is not None:
                hside, depth = h
                cx = {"central": 0.0, "L": 8.0, "R": -8.0}[hside]
                hern = ((lr - cx) / 7.0) ** 2 + ((ap - sp.body_half_depth) / depth) ** 2 <= 1.0
                hern_full = hern & (ap > 0) & (np.abs(up) <= half.clip(min=dh * 0.3))
                disc |= hern_full
                if sp.herniation_displaces_sac:
                    displaced.append(hern_full)
            lab[disc] = ANNULUS
            nuc = disc & (r2 <= 0.35) & (np.abs(up) <= half * 0.7)
            lab[nuc] = NUCLEUS
        # Modic type 1 bands next to chosen endplates
        for mdisc, which, side in sp.modic1:
            v = self.discs[mdisc]
            q = pts - v["center"]
            lr, ap, up = q @ v["lat"], q @ v["ap"], q @ v["normal"]
            r2 = (lr / sp.body_half_width) ** 2 + (ap / sp.body_half_depth) ** 2
            lo, hi = (v["height_mm"] / 2 + 1.0, v["height_mm"] / 2 + 9.0)
            band = (r2 <= 0.8) & ((up > lo) & (up < hi) if which == "inf" else (up < -lo) & (up > -hi))
            if side == "L":
                band &= lr > 2.0
            elif side == "R":
                band &= lr < -2.0
            lab[band & (lab == MARROW)] = MODIC1
        # Schmorl nodes: disc material herniated into the marrow next to an endplate (ellipsoid)
        for body, which in sp.schmorl:
            v = self.bodies[body]
            hh = v.get("half_height", sp.body_height / 2)
            sgn = -1.0 if which == "inf" else 1.0
            ep = v["center"] + v["normal"] * sgn * hh
            q = pts - ep
            node = ((q @ v["lat"]) / 4.5) ** 2 + ((q @ v["ap"]) / 4.5) ** 2 + (
                (q @ v["normal"]) / 3.0
            ) ** 2 <= 1.0
            lab[node & (lab == MARROW)] = SCHMORL
        # ribs: thin tubes leaving the thoracic bodies laterally and posteriorly
        if sp.ribs:
            for name, v in self.bodies.items():
                if not name.startswith("T"):
                    continue
                for s in (-1.0, 1.0):
                    p0 = v["center"] + v["lat"] * s * sp.body_half_width * 0.8
                    p1 = p0 + v["lat"] * s * 60.0 - v["ap"] * 40.0
                    rib = _segment_distance(pts, p0, p1) <= 3.0
                    lab[rib & np.isin(lab, (SOFT, FAT, MUSCLE, PSOAS))] = CORTEX
        # hardware (signal voids) and blooming (a dark halo around the metal)
        core = np.zeros(len(pts), dtype=bool)
        halo = np.zeros(len(pts), dtype=bool)
        for item in sp.hardware:
            c_, h_ = self._hardware_masks(item, pts)
            core |= c_
            halo |= h_
        lab[halo & ~core & (lab != AIR)] = BLOOM
        lab[core] = HARDWARE
        # thecal sac and cord (drawn last: nothing else lives in the canal)
        csf = dist <= sp.canal_radius
        lab[csf] = CSF
        # herniated disc in the canal is drawn over the sac (option, see PhantomSpec)
        for hern_full in displaced:
            lab[hern_full] = ANNULUS
        cord = (dist <= sp.cord_radius) & (pts[:, 2] > self.conus_z)
        lab[cord] = CORD
        return lab

    def _labels_cached(self, pts: np.ndarray) -> np.ndarray:
        """Realistic labels, memoised on the exact sample points: the three contrasts of one geometry share them."""
        from . import realistic as _re

        key = hashlib.blake2b(np.ascontiguousarray(pts).tobytes(), digest_size=16).digest()
        hit = self._label_cache.get(key)
        if hit is not None:
            return hit
        lab = _re.labels(self, pts)
        lab.flags.writeable = False
        if len(self._label_cache) >= _LABEL_CACHE_SIZE:
            self._label_cache.pop(next(iter(self._label_cache)))
        self._label_cache[key] = lab
        return lab

    def _canal_distance(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        line = self.canal_line
        best = np.full(len(pts), np.inf)
        best_t = np.zeros(len(pts))
        for i in range(len(line) - 1):
            a, b = line[i], line[i + 1]
            ab = b - a
            t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
            proj = a + t[:, None] * ab
            dd = np.linalg.norm(pts - proj, axis=1)
            better = dd < best
            best[better] = dd[better]
            best_t[better] = i + t[better]
        return best, best_t

    def _hardware_masks(self, item: str, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(metal core, blooming halo) masks of one hardware item: 'rod', 'screws:<body>' or 'cage:<disc>'."""
        sp = self.spec
        if item == "rod":
            d = _polyline_distance(pts, self.rod_line)
            return d <= 3.0, d <= 7.0
        kind, _, name = item.partition(":")
        if kind == "screws":  # two pedicle screws, 12 mm either side of the midline, through the body
            v = self.bodies[name]
            q = pts - v["center"]
            lr, ap, up = np.abs(q @ v["lat"]), q @ v["ap"], q @ v["normal"]
            hh = v.get("half_height", sp.body_height / 2)
            r2 = (lr - 12.0) ** 2 + (ap - 4.0) ** 2
            return (r2 <= 9.0) & (np.abs(up) <= hh), (r2 <= 49.0) & (np.abs(up) <= hh + 3.0)
        v = self.discs[name]  # cage: a block in the disc space
        q = pts - v["center"]
        lr, ap, up = np.abs(q @ v["lat"]), np.abs(q @ v["ap"]), np.abs(q @ v["normal"])
        hd = v["height_mm"] / 2
        return (lr <= 9.0) & (ap <= 9.0) & (up <= hd), (lr <= 12.0) & (ap <= 12.0) & (up <= hd + 3.0)

    def _bias(self, pts: np.ndarray) -> np.ndarray:
        """Surface-coil sensitivity: brighter near a posterior coil, falling off with depth (multiplicative)."""
        ref = self.bodies["L3"]["center"]
        coil = np.array([0.0, ref[1] + 90.0, ref[2]])
        d = np.linalg.norm(pts - coil, axis=1)
        return 1.0 + self.spec.bias_field * (np.exp(-((d / 140.0) ** 2)) - 0.5)

    def _table(self, contrast: str) -> dict[int, float]:
        if self.realistic:
            from . import realistic as _re

            return _re.signal_table(ACQ[contrast], self.spec.field_strength)
        gain = 0.85 if contrast == "t2" and self.spec.field_strength == 3.0 else 1.0  # cartoon 3 T variant
        return {k: float(v) * gain for k, v in INTENSITY[contrast].items()}

    # ----------------------------------------------------------------- images
    def _realistic_slice(
        self,
        ipp: np.ndarray,
        rd: np.ndarray,
        cd: np.ndarray,
        nrm: np.ndarray,
        rows: int,
        cols: int,
        px: float,
        thickness: float,
        contrast: str,
    ) -> np.ndarray:
        """Flat signal of one slice, averaged over sub-samples across the thickness and in plane (partial volume)."""
        st, sx = self.spec.supersample
        if st < 1 or sx < 1:
            raise ValueError("supersample factors must be >= 1")
        rr, cc = np.mgrid[0:rows, 0:cols]
        base = ipp + np.outer(cc.ravel() * px, rd) + np.outer(rr.ravel() * px, cd)
        acc = np.zeros(base.shape[0], dtype=np.float64)
        n_sub = 0
        for t in (np.arange(st) + 0.5) / st - 0.5:
            for du in (np.arange(sx) + 0.5) / sx - 0.5:
                for dv in (np.arange(sx) + 0.5) / sx - 0.5:
                    pts = base + (t * thickness) * nrm + (du * px) * rd + (dv * px) * cd
                    acc += self.intensity(self.labels(pts), contrast, pts)
                    n_sub += 1
        return acc / n_sub

    def intensity(self, labels: np.ndarray, contrast: str, pts: np.ndarray | None = None) -> np.ndarray:
        table = self._table(contrast)
        lut = np.zeros(max(table) + 1, dtype=np.float64)  # label -> signal lookup (labels are small ints)
        for t, val in table.items():
            lut[t] = val
        img = lut[np.asarray(labels, dtype=np.intp)]
        if pts is not None and self.spec.degenerated:
            for d, f in self.spec.degenerated.items():
                dv = self.discs[d]
                q = pts - dv["center"]
                near = (np.abs(q @ dv["normal"]) < self.spec.disc_height) & (labels == NUCLEUS)
                ann = table[ANNULUS]
                img[near] = ann + (table[NUCLEUS] - ann) * f
        if pts is not None and self.realistic:
            from . import realistic as _re

            img = _re.apply_texture(img, labels, pts, self.spec.seed)
        return img

    def render(
        self,
        plane: str = "sag",
        contrast: str = "t2",
        *,
        n: int = 15,
        spacing: float = 4.0,
        thickness: float = 3.5,
        rows: int = 256,
        cols: int = 256,
        px: float = 1.0,
        center: Any = None,
        normal: Any = None,
        row_dir: Any = None,
        key: str = "",
        ghost: float = 0.0,
        wrap_rows: int = 0,
    ) -> Stack:
        """Render a stack. ``plane`` picks default directions; override with ``normal``/``row_dir``.

        ``ghost`` adds motion ghosts (copies shifted along the row/phase axis, relative amplitude).
        ``wrap_rows`` folds that many rows of anatomy beyond the far edge back onto the near edge
        (phase wrap-around aliasing); the FOV and geometry are unchanged.
        """
        defaults = {
            "sag": ((0, 1, 0), (0, 0, -1)),
            "ax": ((1, 0, 0), (0, 1, 0)),
            "cor": ((1, 0, 0), (0, 0, -1)),
        }
        rd, cd = (np.array(v, dtype=np.float64) for v in defaults[plane])
        if normal is not None:
            nrm = unit(normal)
            rd = unit(row_dir) if row_dir is not None else unit(rd - (rd @ nrm) * nrm)
            cd = unit(np.cross(nrm, rd))
        elif row_dir is not None:
            rd = unit(row_dir)
        nrm = unit(np.cross(rd, cd))
        if center is None:
            l3 = self.bodies["L3"]["center"]
            center = np.array([0.0, l3[1] + 10.0, l3[2] - 15.0])
        center = np.asarray(center, dtype=np.float64)
        rng = np.random.default_rng(self.spec.seed + zlib.crc32(f"{plane}{contrast}".encode()) % 1000)
        offs = (np.arange(n) - (n - 1) / 2) * spacing
        ipps, imgs = [], []
        rr, cc = np.mgrid[0 : rows + wrap_rows, 0:cols]
        table = self._table(contrast)
        noise_scale = 0.7 if self.spec.field_strength == 3.0 else 1.0  # higher SNR at 3 T
        for o in offs:
            ipp = center + o * nrm - ((cols - 1) / 2) * px * rd - ((rows - 1) / 2) * px * cd
            pts = ipp + np.outer(cc.ravel() * px, rd) + np.outer(rr.ravel() * px, cd)
            if self.realistic:  # partial volume: average sub-samples through the slice and in plane
                img = self._realistic_slice(ipp, rd, cd, nrm, rows + wrap_rows, cols, px, thickness, contrast)
            else:
                lab = self.labels(pts)
                img = self.intensity(lab, contrast, pts)
            if self.spec.bias_field:
                img = img * self._bias(pts)
            img2 = img.reshape(rows + wrap_rows, cols)
            if self.realistic and self.spec.gibbs:
                from . import realistic as _re

                img2 = _re.gibbs_ringing(img2)
            if ghost:
                s = max(1, rows // 12)  # ghosts repeat along the phase (row) axis
                img2 = img2 + ghost * (np.roll(img2, s, axis=0) + np.roll(img2, -s, axis=0))
            if wrap_rows:
                folded = img2[:rows].copy()
                folded[:wrap_rows] += img2[rows:]
                img2 = folded
            img = img2.ravel()
            sigma = self.spec.noise * max(table.values()) * noise_scale
            img = np.sqrt((img + rng.normal(0, sigma, img.shape)) ** 2 + rng.normal(0, sigma, img.shape) ** 2)
            ipps.append(ipp)
            imgs.append(img.reshape(rows, cols).astype(np.float32))
        acq = ACQ[contrast]
        meta: dict[str, Any] = {
            "dicom": {
                "SeriesDescription": acq["desc"].format(plane=plane.upper()),
                "Modality": "MR",
                "RepetitionTime": acq["TR"],
                "EchoTime": acq["TE"],
                "ScanningSequence": acq["seq"],
                "SliceThickness": thickness,
                "MagneticFieldStrength": self.spec.field_strength,
            },
            "phantom": True,
        }
        if acq["TI"]:
            meta["dicom"]["InversionTime"] = acq["TI"]
        return Stack(
            key=key,
            vol=np.stack(imgs),
            ipp=np.array(ipps),
            row_dir=rd,
            col_dir=cd,
            row_mm=px,
            col_mm=px,
            thickness=thickness,
            description=meta["dicom"]["SeriesDescription"],
            meta=meta,
        )


# --------------------------------------------------------------------- DICOM output
FAKE_PHI = {
    "PatientName": "PHANTOM^SPINE",
    "PatientID": "PHANTOM-0001",
    "PatientBirthDate": "19700101",
    "PatientSex": "O",
    "StudyDate": "20260101",
    "StudyTime": "120000",
    "AccessionNumber": "ACC-PHANTOM",
    "InstitutionName": "Phantom General Hospital",
    "ReferringPhysicianName": "DOE^JANE",
    "StudyDescription": "PHANTOM LUMBAR SPINE",
}


def write_dicom_series(
    stack: Stack,
    outdir: str | os.PathLike[str],
    *,
    series_number: int,
    contrast: str,
    study_uid: str,
    frame_uid: str,
    description: str | None = None,
    image_type: tuple[str, ...] = ("ORIGINAL", "PRIMARY", "OTHER"),
    series_uid: str | None = None,
    start_instance: int = 1,
    transfer_syntax: str = "explicit",
    rescale: tuple[float, float] | None = None,
    signed: bool = False,
    monochrome1: bool = False,
    temporal_index: int | None = None,
) -> list[str]:
    """Write ``stack`` as an MR Image Storage series (one file per slice, fake PHI included).

    Options (all default to a plain 12-bit unsigned MONOCHROME2 file):

    * ``transfer_syntax``: ``"explicit"``, ``"rle"`` (RLE Lossless) or ``"j2k"`` (JPEG 2000 Lossless).
    * ``rescale=(slope, intercept)``: store integers and write RescaleSlope/RescaleIntercept.
    * ``signed``: PixelRepresentation=1 (int16 pixels, centred on zero).
    * ``monochrome1``: store inverted values with PhotometricInterpretation=MONOCHROME1.
    * ``temporal_index``: write TemporalPositionIdentifier (for 4-D series, one call per time point).
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        JPEG2000Lossless,
        MRImageStorage,
        RLELossless,
        generate_uid,
    )

    if transfer_syntax not in ("explicit", "rle", "j2k"):
        raise ValueError(f"unsupported transfer_syntax {transfer_syntax!r}")
    os.makedirs(outdir, exist_ok=True)
    series_uid = series_uid or generate_uid()
    acq = ACQ[contrast]
    vmax = float(np.percentile(stack.vol, 99.9)) or 1.0
    scale = 4000.0 / vmax
    paths = []
    for k in range(stack.n):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset("phantom", {}, file_meta=meta, preamble=b"\0" * 128)
        for tag, val in FAKE_PHI.items():
            setattr(ds, tag, val)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
        ds.Modality = "MR"
        ds.Manufacturer = "SPINESCOPE"
        ds.ManufacturerModelName = "PHANTOM"
        ds.MagneticFieldStrength = float(stack.meta.get("dicom", {}).get("MagneticFieldStrength", 1.5))
        ds.SeriesNumber = series_number
        ds.InstanceNumber = start_instance + k
        ds.SeriesDescription = description or stack.description
        ds.ProtocolName = "PHANTOM LUMBAR"
        ds.ImageType = list(image_type)
        ds.ScanningSequence = acq["seq"]
        ds.SequenceVariant = "SK"
        ds.RepetitionTime = acq["TR"]
        ds.EchoTime = acq["TE"]
        if acq["TI"]:
            ds.InversionTime = acq["TI"]
        ds.FlipAngle = 90
        ds.PatientPosition = "FFS"
        ds.ImagePositionPatient = [f"{x:.4f}" for x in stack.ipp[k]]
        ds.ImageOrientationPatient = [f"{x:.6f}" for x in (*stack.row_dir, *stack.col_dir)]
        ds.PixelSpacing = [f"{stack.row_mm:.4f}", f"{stack.col_mm:.4f}"]
        ds.SliceThickness = stack.thickness
        ds.SpacingBetweenSlices = stack.spacing or stack.thickness
        ds.SliceLocation = float(stack.positions[k])
        ds.Rows, ds.Columns = stack.vol.shape[1:]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        if temporal_index is not None:
            ds.TemporalPositionIdentifier = temporal_index + 1
        stored = np.clip(stack.vol[k] * scale, 0, 4095)
        if rescale is not None:
            slope, intercept = rescale
            ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
            stored = np.round((stored - intercept) / slope)
        stored_i = np.asarray(stored, dtype=np.int64)
        if signed:
            ds.PixelRepresentation = 1
            ds.PixelData = (stored_i - 2048).astype("<i2").tobytes()
        else:
            if monochrome1:
                stored_i = 4095 - stored_i
            ds.PixelData = stored_i.astype("<u2").tobytes()
        if transfer_syntax == "rle":
            ds.compress(RLELossless, encoding_plugin="pydicom")
        elif transfer_syntax == "j2k":
            ds.compress(JPEG2000Lossless, encoding_plugin="pylibjpeg")
        p = os.path.join(os.fspath(outdir), f"IM{series_number:03d}_{start_instance + k:04d}.dcm")
        pydicom.dcmwrite(p, ds, enforce_file_format=True)
        paths.append(p)
    return paths


def write_enhanced_mr(
    stacks: list[Stack],
    path: str | os.PathLike[str],
    *,
    study_uid: str,
    contrast: str = "t2",
    series_number: int = 1,
    description: str = "ENHANCED MR",
) -> str:
    """Write several stacks (any orientations, same matrix) into ONE Enhanced MR Image Storage object.

    Every frame has its own functional groups (plane position/orientation, pixel measures, echo,
    repetition time). Timing, echo and the scanning sequence are shared-group attributes, and
    ScanningSequence is not repeated at top level, so readers must look inside functional groups.
    """
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import EnhancedMRImageStorage, ExplicitVRLittleEndian, generate_uid

    if not stacks:
        raise ValueError("need at least one stack")
    rows, cols = stacks[0].vol.shape[1:]
    if any(st.vol.shape[1:] != (rows, cols) for st in stacks):
        raise ValueError("all frames of an enhanced object must share one matrix size")
    acq = ACQ[contrast]
    frames = [(st, k) for st in stacks for k in range(st.n)]
    vmax = max(float(np.percentile(st.vol, 99.9)) for st in stacks) or 1.0
    scale = 4000.0 / vmax
    pixels = np.stack([np.clip(st.vol[k] * scale, 0, 4095).astype("<u2") for st, k in frames])

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = EnhancedMRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(os.fspath(path), {}, file_meta=meta, preamble=b"\0" * 128)
    for tag, val in FAKE_PHI.items():
        setattr(ds, tag, val)
    ds.SOPClassUID = EnhancedMRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.Modality = "MR"
    ds.Manufacturer = "SPINESCOPE"
    ds.ManufacturerModelName = "PHANTOM"
    ds.MagneticFieldStrength = 3.0
    ds.SeriesNumber = series_number
    ds.SeriesDescription = description
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "NORM"]
    ds.NumberOfFrames = len(frames)
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0

    shared = Dataset()
    timing = Dataset()
    timing.RepetitionTime = acq["TR"]
    echo = Dataset()
    echo.EffectiveEchoTime = acq["TE"]
    modifier = Dataset()
    modifier.ScanningSequence = acq["seq"]
    modifier.SequenceVariant = "SK"
    shared.MRTimingAndRelatedParametersSequence = Sequence([timing])
    shared.MREchoSequence = Sequence([echo])
    shared.MRImagingModifierSequence = Sequence([modifier])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])

    per_frame = []
    for st, k in frames:
        g = Dataset()
        pos = Dataset()
        pos.ImagePositionPatient = [f"{x:.4f}" for x in st.ipp[k]]
        g.PlanePositionSequence = Sequence([pos])
        ori = Dataset()
        ori.ImageOrientationPatient = [f"{x:.6f}" for x in (*st.row_dir, *st.col_dir)]
        g.PlaneOrientationSequence = Sequence([ori])
        pm = Dataset()
        pm.PixelSpacing = [f"{st.row_mm:.4f}", f"{st.col_mm:.4f}"]
        pm.SliceThickness = st.thickness
        g.PixelMeasuresSequence = Sequence([pm])
        ft = Dataset()
        ft.FrameType = ["ORIGINAL", "PRIMARY", "M", "NORM"]
        g.MRImageFrameTypeSequence = Sequence([ft])
        per_frame.append(g)
    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame)
    ds.PixelData = pixels.tobytes()
    out = os.fspath(path)
    pydicom.dcmwrite(out, ds, enforce_file_format=True)
    return out


def write_non_image_object(path: str | os.PathLike[str], kind: str, *, study_uid: str) -> str:
    """Write a minimal non-image object: ``SR``, ``KO``, ``PR``, ``SC`` (screenshot with burned-in text), or
    ``DICOMDIR``. Used to test that junk is ignored (never treated as anatomy)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

    sop = {
        "SR": "1.2.840.10008.5.1.4.1.1.88.11",
        "KO": "1.2.840.10008.5.1.4.1.1.88.59",
        "PR": "1.2.840.10008.5.1.4.1.1.11.1",
        "SC": "1.2.840.10008.5.1.4.1.1.7",
        "DICOMDIR": "1.2.840.10008.1.3.10",
    }[kind]
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID(sop)
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(os.fspath(path), {}, file_meta=meta, preamble=b"\0" * 128)
    for tag, val in FAKE_PHI.items():
        setattr(ds, tag, val)
    ds.SOPClassUID = UID(sop)
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = {"SC": "OT", "DICOMDIR": "DICOMDIR"}.get(kind, kind)
    ds.SeriesNumber = 99
    if kind == "SC":  # RGB screenshot: a grey background with a text-like block burned in
        ds.Rows, ds.Columns = 64, 96
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        ds.ImageType = ["DERIVED", "SECONDARY", "SCREEN SAVE"]
        img = np.full((64, 96, 3), 40, dtype=np.uint8)
        img[20:30, 10:86, :] = 230  # burned-in text line
        ds.PixelData = img.tobytes()
    if kind == "DICOMDIR":
        ds.DirectoryRecordSequence = []
    pydicom.dcmwrite(os.fspath(path), ds, enforce_file_format=True)
    return os.fspath(path)


def write_standard_study(
    phantom: SpinePhantom,
    root: str | os.PathLike[str],
    *,
    rows: int = 192,
    cols: int = 192,
    px: float = 1.2,
    slabs: bool = True,
    localizer: bool = True,
    slab_contrasts: tuple[str, ...] = ("t2",),
) -> dict[str, Any]:
    """Write a typical lumbar protocol: sag T2/T1/STIR, ax T2/T1, oblique slabs, 3-plane localizer.

    ``slab_contrasts`` sets the contrast of the oblique slabs (default T2, as before; add ``"t1"`` for T1 slabs).

    Returns ``{'root': str, 'series': {name: [paths]}, 'truth': phantom.truth}``.
    """
    from pydicom.uid import generate_uid

    root = os.fspath(root)
    study_uid, frame_uid = generate_uid(), generate_uid()
    out: dict[str, list[str]] = {}
    sn = 1
    if localizer:
        loc_uid = generate_uid()
        k0 = 1
        paths = []
        for plane in ("ax", "sag", "cor"):
            st = phantom.render(plane, "t2", n=3, spacing=15.0, thickness=8.0, rows=96, cols=96, px=3.6)
            paths += write_dicom_series(
                st,
                os.path.join(root, f"S{sn:02d}"),
                series_number=sn,
                contrast="t2",
                study_uid=study_uid,
                frame_uid=frame_uid,
                description="3-PL LOC",
                image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"),
                series_uid=loc_uid,
                start_instance=k0,
            )
            k0 += st.n
        out["loc"] = paths
        sn += 1
    for contrast in ("t2", "stir", "t1"):
        st = phantom.render("sag", contrast, n=13, spacing=5.0, thickness=4.0, rows=rows, cols=cols, px=px)
        out[f"sag_{contrast}"] = write_dicom_series(
            st,
            os.path.join(root, f"S{sn:02d}"),
            series_number=sn,
            contrast=contrast,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )
        sn += 1
    lumbar_mid = (phantom.discs["L1-L2"]["center"] + phantom.discs["L5-S1"]["center"]) / 2
    for contrast in ("t2", "t1"):
        st = phantom.render(
            "ax",
            contrast,
            n=36,
            spacing=5.0,
            thickness=4.0,
            rows=128,
            cols=128,
            px=1.4,
            center=lumbar_mid + np.array([0, 10.0, 0]),
        )
        out[f"ax_{contrast}"] = write_dicom_series(
            st,
            os.path.join(root, f"S{sn:02d}"),
            series_number=sn,
            contrast=contrast,
            study_uid=study_uid,
            frame_uid=frame_uid,
        )
        sn += 1
    for slab_contrast in slab_contrasts if slabs else ():
        slab_uid = generate_uid()
        paths, k0 = [], 1
        for disc in ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"):
            d = phantom.discs[disc]
            st = phantom.render(
                "ax",
                slab_contrast,
                n=3,
                spacing=3.5,
                thickness=3.0,
                rows=128,
                cols=128,
                px=1.2,
                center=d["center"] + d["ap"] * 12,
                normal=d["normal"],
                row_dir=(1, 0, 0),
            )
            paths += write_dicom_series(
                st,
                os.path.join(root, f"S{sn:02d}"),
                series_number=sn,
                contrast=slab_contrast,
                study_uid=study_uid,
                frame_uid=frame_uid,
                description=f"OBL AX {slab_contrast.upper()}",
                series_uid=slab_uid,
                start_instance=k0,
            )
            k0 += st.n
        out[f"obl_ax_{slab_contrast}"] = paths
        sn += 1
    return {"root": root, "series": out, "truth": phantom.truth}
