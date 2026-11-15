"""Synthetic lumbar spine phantom with known ground truth, rendered to stacks or DICOM files.

The phantom is analytic and defined in patient LPS mm, so any slice geometry (sagittal, axial,
oblique slabs, flipped in-plane axes...) can be rendered exactly. It is meant for unit tests and
examples. It is a cartoon, not a realistic anatomy simulator.

Example::

    ph = SpinePhantom(PhantomSpec(degenerated={"L4-L5": 0.3}, modic1=[("L4-L5", "inf", "L")]))
    sag = ph.render("sag", "t2")
    paths = write_standard_study(ph, tmp_path)   # sag T1/T2/STIR, ax T1/T2, oblique slabs, localizer
"""

from __future__ import annotations

import os
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry import Stack, unit

DISCS = ("T11-T12", "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")  # superior -> inferior
BODIES = ("T11", "T12", "L1", "L2", "L3", "L4", "L5")  # body above each disc, same order

# Tissue labels
AIR, SOFT, FAT, MUSCLE, MARROW, CORTEX, ANNULUS, NUCLEUS, CSF, CORD, VESSEL, MODIC1, PSOAS = range(13)
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
    },
}

#: Acquisition parameters written to DICOM so series classification can be tested.
ACQ: dict[str, dict[str, Any]] = {
    "t1": {"TR": 550.0, "TE": 8.0, "TI": None, "seq": "SE", "desc": "{plane} T1"},
    "t2": {"TR": 4000.0, "TE": 95.0, "TI": None, "seq": "SE", "desc": "{plane} T2"},
    "stir": {"TR": 5000.0, "TE": 45.0, "TI": 150.0, "seq": "IR", "desc": "{plane} STIR"},
}


@dataclass
class PhantomSpec:
    """Phantom parameters. Angles in degrees, lengths in mm."""

    sacral_slope: float = 28.0
    segment_lordosis: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0, 14.0)  # T11-T12..L4-L5 relative turns
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
    noise: float = 0.02  # Rician noise sigma as a fraction of the brightest tissue
    seed: int = 0


class SpinePhantom:
    """Analytic lumbar spine. ``truth`` holds the ground-truth anatomy anchors in LPS mm."""

    def __init__(self, spec: PhantomSpec | None = None) -> None:
        self.spec = spec or PhantomSpec()
        self._build()

    # ----------------------------------------------------------------- anatomy
    def _build(self) -> None:
        sp = self.spec
        # disc tilt from axial for L5-S1 .. T11-T12 (inferior -> superior)
        turns = list(sp.segment_lordosis)[::-1]  # L4-L5 .. T11-T12
        tilts = [sp.sacral_slope]
        for t in turns:
            tilts.append(tilts[-1] - t)
        p = np.array(sp.s1_top, dtype=np.float64)
        discs: dict[str, dict[str, Any]] = {}
        bodies: dict[str, dict[str, Any]] = {}
        for disc, body, tilt, nxt in zip(
            DISCS[::-1], BODIES[::-1], tilts, [*tilts[1:], tilts[-1]], strict=False
        ):
            n = self._normal(tilt)
            a = self._ap(tilt)
            c = p + n * sp.disc_height / 2
            discs[disc] = {
                "center": c,
                "normal": n,
                "ap": a,
                "anterior": c - a * sp.body_half_depth,
                "posterior": c + a * sp.body_half_depth,
                "tilt_deg": tilt,
                "height_mm": sp.disc_height,
            }
            bottom = p + n * sp.disc_height
            tb = (tilt + nxt) / 2
            nb, ab = self._normal(tb), self._ap(tb)
            bc = bottom + nb * sp.body_height / 2
            bodies[body] = {"center": bc, "normal": nb, "ap": ab, "tilt_deg": tb}
            p = bottom + nb * sp.body_height
        # S1 / sacrum: a block below the S1 endplate, tilted by the sacral slope
        n0, a0 = self._normal(sp.sacral_slope), self._ap(sp.sacral_slope)
        s1 = np.array(sp.s1_top) - n0 * 15.0
        bodies["S1"] = {
            "center": s1,
            "normal": n0,
            "ap": a0,
            "tilt_deg": sp.sacral_slope,
            "half_height": 15.0,
        }
        self.discs, self.bodies = discs, bodies
        canal = [bodies[b]["center"] + bodies[b]["ap"] * sp.canal_offset for b in ("T11", *BODIES[1:], "S1")]
        top = canal[0] + bodies["T11"]["normal"] * 40
        bottom = canal[-1] - bodies["S1"]["normal"] * 30
        self.canal_line = np.array([top, *canal, bottom])
        self.conus_z = float(bodies[sp.conus_body]["center"][2])
        self.truth = {
            "discs": {
                k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                for k, v in discs.items()
            },
            "vertebrae": {k: v["center"].tolist() for k, v in bodies.items()},
            "midline_x": float(sp.s1_top[0]),
            "conus_z": self.conus_z,
            "conus_level": sp.conus_body,
        }

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
        for b in (*BODIES, "S1"):
            c = self.bodies[b]["center"] + self.bodies[b]["ap"] * (sp.canal_offset + 22)
            q = pts - c
            sproc = (q[:, 0] / 5.0) ** 2 + ((q @ self.bodies[b]["ap"]) / 14.0) ** 2 + (
                (q @ self.bodies[b]["normal"]) / 9.0
            ) ** 2 <= 1.0
            lab[sproc] = MARROW
        # vertebral bodies (with a thin cortical shell) and discs
        for v in self.bodies.values():
            hh = v.get("half_height", sp.body_height / 2)
            q = pts - v["center"]
            lr, ap, up = q[:, 0], q @ v["ap"], q @ v["normal"]
            rr = (np.abs(lr) / sp.body_half_width) ** 2.6 + (np.abs(ap) / sp.body_half_depth) ** 2.6
            inside = (rr <= 1.0) & (np.abs(up) <= hh)
            shell = inside & ((rr > 0.86) | (np.abs(up) > hh - 1.0))
            lab[inside] = MARROW
            lab[shell] = CORTEX
        for d, v in self.discs.items():
            q = pts - v["center"]
            lr, ap, up = q[:, 0], q @ v["ap"], q @ v["normal"]
            r2 = (lr / sp.body_half_width) ** 2 + (ap / sp.body_half_depth) ** 2
            half = sp.disc_height / 2 * (1.0 - 0.25 * r2)
            disc = (r2 <= 1.0) & (np.abs(up) <= half)
            h = sp.herniation.get(d)
            if h is not None:
                hside, depth = h
                cx = {"central": 0.0, "L": 8.0, "R": -8.0}[hside]
                hern = ((lr - cx) / 7.0) ** 2 + ((ap - sp.body_half_depth) / depth) ** 2 <= 1.0
                disc |= hern & (ap > 0) & (np.abs(up) <= half.clip(min=sp.disc_height * 0.3))
            lab[disc] = ANNULUS
            nuc = disc & (r2 <= 0.35) & (np.abs(up) <= half * 0.7)
            lab[nuc] = NUCLEUS
        # Modic type 1 bands next to chosen endplates
        for mdisc, which, side in sp.modic1:
            v = self.discs[mdisc]
            q = pts - v["center"]
            lr, ap, up = q[:, 0], q @ v["ap"], q @ v["normal"]
            r2 = (lr / sp.body_half_width) ** 2 + (ap / sp.body_half_depth) ** 2
            lo, hi = (sp.disc_height / 2 + 1.0, sp.disc_height / 2 + 9.0)
            band = (r2 <= 0.8) & ((up > lo) & (up < hi) if which == "inf" else (up < -lo) & (up > -hi))
            if side == "L":
                band &= lr > 2.0
            elif side == "R":
                band &= lr < -2.0
            lab[band & (lab == MARROW)] = MODIC1
        # thecal sac and cord (drawn last: nothing else lives in the canal)
        csf = dist <= sp.canal_radius
        lab[csf] = CSF
        cord = (dist <= sp.cord_radius) & (pts[:, 2] > self.conus_z)
        lab[cord] = CORD
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

    # ----------------------------------------------------------------- images
    def intensity(self, labels: np.ndarray, contrast: str, pts: np.ndarray | None = None) -> np.ndarray:
        table = INTENSITY[contrast]
        img = np.zeros(labels.shape, dtype=np.float64)
        for t, val in table.items():
            img[labels == t] = val
        if pts is not None and self.spec.degenerated:
            for d, f in self.spec.degenerated.items():
                dv = self.discs[d]
                q = pts - dv["center"]
                near = (np.abs(q @ dv["normal"]) < self.spec.disc_height) & (labels == NUCLEUS)
                ann = table[ANNULUS]
                img[near] = ann + (table[NUCLEUS] - ann) * f
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
    ) -> Stack:
        """Render a stack. ``plane`` picks default directions; override with ``normal``/``row_dir``."""
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
        rr, cc = np.mgrid[0:rows, 0:cols]
        for o in offs:
            ipp = center + o * nrm - ((cols - 1) / 2) * px * rd - ((rows - 1) / 2) * px * cd
            pts = ipp + np.outer(cc.ravel() * px, rd) + np.outer(rr.ravel() * px, cd)
            lab = self.labels(pts)
            img = self.intensity(lab, contrast, pts)
            sigma = self.spec.noise * max(INTENSITY[contrast].values())
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
) -> list[str]:
    """Write ``stack`` as an MR Image Storage series (one file per slice, fake PHI included)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

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
        ds.MagneticFieldStrength = 1.5
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
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        arr = np.clip(stack.vol[k] * scale, 0, 4095).astype(np.uint16)
        ds.PixelData = arr.tobytes()
        p = os.path.join(os.fspath(outdir), f"IM{series_number:03d}_{start_instance + k:04d}.dcm")
        pydicom.dcmwrite(p, ds, enforce_file_format=True)
        paths.append(p)
    return paths


def write_standard_study(
    phantom: SpinePhantom,
    root: str | os.PathLike[str],
    *,
    rows: int = 192,
    cols: int = 192,
    px: float = 1.2,
    slabs: bool = True,
    localizer: bool = True,
) -> dict[str, Any]:
    """Write a typical lumbar protocol: sag T2/T1/STIR, ax T2/T1, oblique T2 slabs, 3-plane localizer.

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
    if slabs:
        slab_uid = generate_uid()
        paths, k0 = [], 1
        for disc in ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"):
            d = phantom.discs[disc]
            st = phantom.render(
                "ax",
                "t2",
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
                contrast="t2",
                study_uid=study_uid,
                frame_uid=frame_uid,
                description="OBL AX T2",
                series_uid=slab_uid,
                start_instance=k0,
            )
            k0 += st.n
        out["obl_ax_t2"] = paths
    return {"root": root, "series": out, "truth": phantom.truth}
