"""Audit one DICOM series folder: file, metadata, geometry, protocol, de-identification. Prints JSON."""
import json
import sys
from pathlib import Path

import numpy as np
import pydicom

IDENT = ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge", "PatientWeight", "PatientSize",
         "PatientAddress", "OtherPatientIDs", "InstitutionName", "InstitutionAddress", "ReferringPhysicianName",
         "PerformingPhysicianName", "OperatorsName", "StationName", "DeviceSerialNumber", "AccessionNumber",
         "StudyID", "InstitutionalDepartmentName"]
PROTO = ["MagneticFieldStrength", "ImagingFrequency", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
         "ScanningSequence", "SequenceVariant", "MRAcquisitionType", "PixelBandwidth", "Manufacturer",
         "ManufacturerModelName", "SoftwareVersions", "SeriesDescription", "ContrastBolusAgent"]


def v(x):
    if x is None:
        return None
    if isinstance(x, pydicom.multival.MultiValue):
        return [v(i) for i in x]
    if isinstance(x, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS)):
        return float(x)
    return str(x)


def audit(folder):
    files = sorted(p for p in Path(folder).rglob("*") if p.is_file())
    dss, bad = [], []
    for p in files:
        try:
            dss.append((p, pydicom.dcmread(p)))
        except Exception as e:
            bad.append(f"{p.name}: {type(e).__name__}")
    r = {"folder": Path(folder).name, "files": len(files), "unreadable": bad}
    ds0 = dss[0][1]
    r["sop_class"] = v(ds0.get("SOPClassUID").name if ds0.get("SOPClassUID") else None)
    r["frames_per_file"] = sorted({int(d.get("NumberOfFrames", 1)) for _, d in dss})
    r["uids"] = {k: len({str(d.get(k)) for _, d in dss}) for k in ("StudyInstanceUID", "SeriesInstanceUID")}
    sops = [str(d.get("SOPInstanceUID")) for _, d in dss]
    r["uids"]["SOPInstanceUID_unique"] = len(set(sops)) == len(sops)
    r["protocol"] = {k: v(ds0.get(k)) for k in PROTO}
    r["protocol_varies_within_series"] = [k for k in PROTO if len({str(d.get(k)) for _, d in dss}) > 1]
    r["dates"] = {k: sorted({str(d.get(k)) for _, d in dss}) for k in ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate")}
    r["identifying_present"] = {k: ("<empty>" if str(ds0.get(k, "")).strip() == "" else "<value present>")
                                for k in IDENT if k in ds0}
    r["deid_markers"] = {k: v(ds0.get(k)) for k in ("PatientIdentityRemoved", "DeidentificationMethod") if k in ds0}
    # geometry (single-frame series only)
    if r["frames_per_file"] == [1]:
        ipp = np.array([[float(x) for x in d.ImagePositionPatient] for _, d in dss if "ImagePositionPatient" in d])
        iops = {tuple(round(float(x), 4) for x in d.ImageOrientationPatient) for _, d in dss if "ImageOrientationPatient" in d}
        r["iop_unique"] = len(iops)
        iop = np.array(list(iops)[0])
        normal = np.cross(iop[:3], iop[3:])
        r["iop_orthonormal"] = bool(abs(np.dot(iop[:3], iop[3:])) < 1e-3 and abs(np.linalg.norm(iop[:3]) - 1) < 1e-3)
        pos = np.sort(ipp @ normal)
        gaps = np.diff(pos)
        r["slices"] = len(pos)
        r["slice_gap_mm"] = {"median": float(np.median(gaps)), "min": float(gaps.min()), "max": float(gaps.max())} if len(gaps) else None
        r["duplicate_positions"] = int((gaps < 1e-3).sum()) if len(gaps) else 0
        r["missing_slices_est"] = int(np.round(gaps / np.median(gaps)).sum() - len(gaps)) if len(gaps) else 0
        r["SliceThickness"] = v(ds0.get("SliceThickness"))
        r["SpacingBetweenSlices"] = v(ds0.get("SpacingBetweenSlices"))
        r["PixelSpacing"] = v(ds0.get("PixelSpacing"))
        inst = sorted(int(d.get("InstanceNumber", -1)) for _, d in dss)
        r["instance_numbers"] = {"min": inst[0], "max": inst[-1], "n": len(inst),
                                 "gaps": int(inst[-1] - inst[0] + 1 - len(set(inst)))}
    try:
        px = ds0.pixel_array
        r["rows_cols_match_pixels"] = list(px.shape[-2:]) == [int(ds0.Rows), int(ds0.Columns)]
        r["pixel"] = {"Rows": int(ds0.Rows), "Columns": int(ds0.Columns), "BitsStored": int(ds0.BitsStored),
                      "RescaleSlope": v(ds0.get("RescaleSlope")), "RescaleIntercept": v(ds0.get("RescaleIntercept"))}
    except Exception as e:
        r["pixel_error"] = f"{type(e).__name__}: {e}"[:200]
    return r


if __name__ == "__main__":
    print(json.dumps([audit(f) for f in sys.argv[1:]], indent=1, default=str))
