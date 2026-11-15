"""PHI canary phantom: a synthetic study where every text-capable DICOM attribute carries a unique marker.

Used by ``tests/test_privacy_canaries.py`` (and reusable by hand) to prove that no identifying text reaches
any output surface. Every canary contains the marker ``ZQCANARY`` (never a word in a protocol name or base64 text), so a single
substring scan finds any leak; each full canary string is also checked on its own.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pydicom
from pydicom.dataset import Dataset

from spinescope.testing.phantom import PhantomSpec, SpinePhantom, write_standard_study

#: Marker shared by every canary. Outputs must not contain it at all.
MARKER = "ZQCANARY"

#: Folder name of the study: a path that carries identifying text (input paths often do).
PATH_CANARY = "ZQCANARY_PATH_Smithson_Jon_1970"

#: Text attributes given a canary value. Values are valid for each VR (PN, LO, SH, LT, DA, TM, AS...).
TEXT_CANARIES: dict[str, str] = {
    "PatientName": "ZQCANARY^PatientName",
    "PatientID": "ZQCANARY_PatientID",
    "OtherPatientIDs": "ZQCANARY_OtherPatientIDs",
    "PatientAge": "047Y",  # AS: numeric, checked by its own value
    "PatientBirthDate": "19990817",  # DA: checked as 19990817 and 1999-08-17
    "PatientBirthName": "ZQCANARY^BirthName",
    "PatientMotherBirthName": "ZQCANARY^MotherBirthName",
    "PatientAddress": "ZQCANARY_PatientAddress",
    "PatientComments": "ZQCANARY_PatientComments",
    "AdditionalPatientHistory": "ZQCANARY_AdditionalPatientHistory",
    "StudyDate": "20010203",
    "StudyTime": "074512",
    "SeriesDate": "20010204",
    "AcquisitionDate": "20010205",
    "ContentDate": "20010206",
    "AccessionNumber": "ZQCANARY_ACC01",
    "StudyID": "ZQCANARY_SID01",
    "InstitutionName": "ZQCANARY_InstitutionName",
    "InstitutionAddress": "ZQCANARY_InstitutionAddress",
    "InstitutionalDepartmentName": "ZQCANARY_Department",
    "StationName": "ZQCANARY_STN01",
    "DeviceSerialNumber": "ZQCANARY-SERIAL-7731",
    "ReferringPhysicianName": "ZQCANARY^Referring",
    "PerformingPhysicianName": "ZQCANARY^Performing",
    "OperatorsName": "ZQCANARY^Operator",
    "NameOfPhysiciansReadingStudy": "ZQCANARY^Reader",
    "StudyDescription": "ZQCANARY_StudyDescription",
    "SeriesDescription": "ZQCANARY_SeriesDescription",
    "ProtocolName": "ZQCANARY_ProtocolName",
    "ImageComments": "ZQCANARY_ImageComments",
    "RequestedProcedureDescription": "ZQCANARY_RequestedProcedure",
    "PerformedProcedureStepDescription": "ZQCANARY_PerformedProcedure",
    "ReasonForTheRequestedProcedure": "ZQCANARY_ReasonForRequest",
    "ScheduledProcedureStepDescription": "ZQCANARY_ScheduledProcedure",
    "SoftwareVersions": "ZQCANARY_SoftwareVersions",
    "ReceiveCoilName": "ZQCANARY_COILX",
    "SequenceName": "ZQCANARY_NAMEX",
}

#: Private (odd-group) attributes with canary text.
PRIVATE_CANARIES: dict[tuple[int, int], str] = {
    (0x0029, 0x1010): "ZQCANARY_PrivateSiemensText",
    (0x0019, 0x1000): "ZQCANARY_PrivateGE",
    (0x0051, 0x1009): "ZQCANARY_PrivatePhilips",
}

#: Numeric identifiers whose values are checked as text in every output.
NUMERIC_CANARIES: tuple[str, ...] = ("19990817", "1999-08-17", "074512", "20010203", "83.4117")


#: Canary strings checked one by one (in addition to the MARKER scan).
def canary_strings() -> list[str]:
    """Every string that must never appear in any output (text, JSON, PNG metadata, logs, errors)."""
    out = [PATH_CANARY, MARKER]
    out += [v for v in TEXT_CANARIES.values()]
    out += list(PRIVATE_CANARIES.values())
    out += list(NUMERIC_CANARIES)
    return out


def _inject(path: str, uids: set[str]) -> None:
    """Rewrite one DICOM file with canary text in every text attribute, a private tag and a PatientWeight."""
    ds = pydicom.dcmread(path)
    for name, value in TEXT_CANARIES.items():
        if name in ("SeriesDescription", "ProtocolName"):  # keep the protocol words, add the canary to them
            value = f"{ds.get(name, '')} {value}".strip()
        setattr(ds, name, value)
    ds.PatientWeight = 83.4117
    for (group, elem), value in PRIVATE_CANARIES.items():
        ds.add_new((group, elem), "LO", value)
    for name in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID"):
        v = ds.get(name)
        if v:
            uids.add(str(v))
    ds.save_as(path, enforce_file_format=True)


def write_canary_study(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Write the standard phantom protocol into ``root/PATH_CANARY`` with canaries everywhere.

    Returns ``{'root': str, 'uids': [...], 'truth': phantom truth}``. ``uids`` are the raw UIDs written,
    which must not appear in outputs either.
    """
    base = Path(os.fspath(root)) / PATH_CANARY
    base.mkdir(parents=True, exist_ok=True)
    phantom = SpinePhantom(
        PhantomSpec(
            degenerated={"L4-L5": 0.2},
            modic1=[("L4-L5", "inf", "L")],
            herniation={"L5-S1": ("central", 5.0)},
        )
    )
    info = write_standard_study(phantom, base)
    uids: set[str] = set()
    for paths in info["series"].values():
        for p in paths:
            _inject(p, uids)
    return {"root": str(base), "uids": sorted(uids), "truth": info["truth"]}


def dataset_text_values(path: str) -> list[str]:
    """All text values in a DICOM file (for tests that check the canaries are really there)."""
    ds: Dataset = pydicom.dcmread(path, stop_before_pixels=True)
    vals: list[str] = []
    for el in ds.iterall():
        if isinstance(el.value, str):
            vals.append(el.value)
    return vals
