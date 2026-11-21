"""DICOM series in: header checks (the "envelope"), then the image goes to the T1w pipeline.

DICOM is a container, not a measurement. The checks here look only at the headers and the slice geometry; image
quality is judged afterwards by twin_evidence.t1w on the reconstructed volume (one T1Scan, same as from NIfTI).

Checks and the catalogue rows they come from:
  F54  required tags present (geometry and pixel data FAIL if missing; acquisition parameters WARN)
  F56  units: DICOM stores TR / TE / TI in ms, BIDS sidecars in s; converted to s here so a DICOM scan and a NIfTI
       scan with the same protocol do not raise a false F35
  F57  nominal field and imaging frequency both present
  F58  StudyDate / SeriesDate / AcquisitionDate agree (within one day)
  F59  identifying tags removed (names only are reported, values are never stored); burned-in annotation FAILs
  F35  acquisition parameters constant within the series (between sessions: t1w.measure with the sidecar)
Series completeness, slice geometry and UID consistency have no catalogue row yet (proposed for v1.2 in
docs/scantwin/INFORME_T1w_DICOM.md §3.3); they are reported with catalogue_id None. Their tolerances are numerical
(rounding in the stored positions), not QC thresholds.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .gates import RANK, row

IDENT_DICOM = ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "PatientAge", "PatientWeight",
               "PatientSize", "PatientAddress", "OtherPatientIDs", "InstitutionName", "InstitutionAddress",
               "InstitutionalDepartmentName", "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
               "StationName", "DeviceSerialNumber", "AccessionNumber", "StudyID")
REQUIRED = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns")
OPTIONAL = ("RepetitionTime", "EchoTime", "FlipAngle", "SliceThickness", "MagneticFieldStrength")
PROTOCOL = ("RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "MagneticFieldStrength", "ImagingFrequency",
            "ScanningSequence", "SequenceVariant", "MRAcquisitionType")
MS_TO_S = ("RepetitionTime", "EchoTime", "InversionTime")
POS_TOL = 0.01          # relative tolerance on slice spacing (stored positions are rounded), not a QC limit


@dataclass
class DicomSeries:
    datasets: list                      # pydicom datasets, sorted along the slice normal
    names: list                         # file names in the same order
    ignored: list = field(default_factory=list)     # files that are not DICOM (e.g. a LICENSE file)
    other_series: dict = field(default_factory=dict)  # SeriesInstanceUID -> file count, not used


def _read(files) -> tuple[list, list]:
    """files: paths, or (name, bytes) pairs (uploads). Returns [(name, dataset)], [ignored names]."""
    import pydicom
    out, ignored = [], []
    for f in files:
        name, src = (f[0], io.BytesIO(f[1])) if isinstance(f, tuple) else (Path(f).name, str(f))
        try:
            out.append((name, pydicom.dcmread(src)))
        except Exception:
            ignored.append(name)
    return out, ignored


def read_series(files) -> DicomSeries:
    """Read every file, keep the largest series, sort its slices along the normal of the image plane."""
    dss, ignored = _read(files)
    if not dss:
        raise ValueError("no readable DICOM file" + (f" (ignored: {', '.join(ignored)})" if ignored else ""))
    by_uid: dict = {}
    for name, d in dss:
        by_uid.setdefault(str(d.get("SeriesInstanceUID", "")), []).append((name, d))
    uid = max(by_uid, key=lambda k: len(by_uid[k]))
    chosen = by_uid[uid]
    try:
        iop = np.array(chosen[0][1].ImageOrientationPatient, float)
        normal = np.cross(iop[:3], iop[3:])
        chosen.sort(key=lambda t: float(np.dot(np.array(t[1].ImagePositionPatient, float), normal)))
    except Exception:                       # missing geometry: keep the order, F54 will FAIL
        chosen.sort(key=lambda t: int(t[1].get("InstanceNumber", 0)))
    return DicomSeries([d for _, d in chosen], [n for n, _ in chosen], ignored,
                       {k: len(v) for k, v in by_uid.items() if k != uid})


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _finding(cid, label, status, value, reason, kind, fix=""):
    r = row(cid) if cid else {}
    return {"id": cid or label, "catalogue_id": cid, "label": label, "status": status, "value": value,
            "reason": reason, "kind": kind,
            "fix": fix or r.get("Fix for the next scan", ""),
            "source": r.get("Threshold source", "no catalogue row yet (proposed v1.2)") if cid else
            "no catalogue row yet (proposed v1.2)"}


def _day(s) -> date | None:
    s = str(s or "").strip()
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8])) if len(s) >= 8 else None
    except ValueError:
        return None


def check(series: DicomSeries) -> dict:
    """Header checks. Returns {"status", "findings", "geometry"}; status is the worst finding (PASS / WARN / FAIL)."""
    dss = series.datasets
    d0 = dss[0]
    F = []

    # ---- F54 required tags
    missing = sorted({k for d in dss for k in REQUIRED if k not in d} | ({"PixelData"} if "PixelData" not in d0 else set()))
    opt = [k for k in OPTIONAL if k not in d0 or str(d0.get(k, "")).strip() == ""]
    F.append(_finding("F54", "Required DICOM tags", "FAIL" if missing else "WARN" if opt else "PASS",
                      ", ".join(missing or opt) or "all present",
                      ("missing geometry / pixel tags: the volume cannot be placed in space" if missing else
                       "optional acquisition tags missing: kept as null, never filled in" if opt else ""), "metadata"))
    geom = {}
    if not missing:
        # ---- slice geometry (no catalogue row yet)
        iops = {tuple(np.round(np.array(d.ImageOrientationPatient, float), 4)) for d in dss}
        iop = np.array(d0.ImageOrientationPatient, float)
        ortho = abs(np.dot(iop[:3], iop[3:])) < 1e-3 and all(abs(np.linalg.norm(v) - 1) < 1e-3 for v in (iop[:3], iop[3:]))
        normal = np.cross(iop[:3], iop[3:])
        pos = np.array([np.dot(np.array(d.ImagePositionPatient, float), normal) for d in dss])
        gaps = np.diff(pos)
        step = float(np.median(gaps)) if len(gaps) else None
        dup = int((gaps < 1e-3).sum()) if len(gaps) else 0
        missing_slices = int(round(float(np.sum(np.round(gaps / step) - 1)))) if step else 0
        uneven = bool(step and np.any(np.abs(gaps - step) > POS_TOL * step + 1e-3) and not missing_slices)
        geom = {"slices": len(dss), "slice_step_mm": step, "orientation_unique": len(iops) == 1, "orthonormal": ortho}
        if len(iops) > 1 or not ortho:
            F.append(_finding(None, "Image orientation", "FAIL", f"{len(iops)} orientation(s), orthonormal={ortho}",
                              "ImageOrientationPatient varies within the series or is not a valid rotation",
                              "geometry", "Re-export the series from the scanner / PACS"))
        else:
            F.append(_finding(None, "Image orientation", "PASS", "constant and orthonormal", "", "geometry"))
        if dup or missing_slices or len(dss) < 2:
            F.append(_finding(None, "Series complete", "FAIL",
                              f"{len(dss)} slices; {missing_slices} missing, {dup} duplicate position(s)",
                              "a slice is missing or repeated: the volume would have a hole or a double layer"
                              if len(dss) >= 2 else "a single slice is not a 3D series",
                              "file", "Re-export the complete series from the PACS"))
        elif uneven:
            F.append(_finding(None, "Series complete", "WARN", f"spacing {gaps.min():.3f}-{gaps.max():.3f} mm",
                              "slice positions are unevenly spaced", "geometry"))
        else:
            F.append(_finding(None, "Series complete", "PASS", f"{len(dss)} slices every {step:.3f} mm", "", "file"))
        sbs = _f(d0.get("SpacingBetweenSlices"))
        if sbs is not None and step and abs(sbs - step) > POS_TOL * step + 1e-3:
            F.append(_finding(None, "Slice spacing tag", "WARN", f"tag {sbs:g} mm vs positions {step:.3f} mm",
                              "SpacingBetweenSlices disagrees with the slice positions; the positions are used",
                              "metadata"))
        # Rows / Columns against the pixel data
        try:
            shp = d0.pixel_array.shape[-2:]
            ok = list(shp) == [int(d0.Rows), int(d0.Columns)]
        except Exception as e:
            ok, shp = False, str(type(e).__name__)
        if not ok:
            F.append(_finding("F54", "Pixel data readable", "FAIL", f"Rows×Columns {d0.Rows}×{d0.Columns} vs {shp}",
                              "pixel data missing, compressed with an unsupported codec, or the wrong size", "file"))
    # ---- UIDs (no catalogue row yet)
    studies = {str(d.get("StudyInstanceUID")) for d in dss}
    sops = [str(d.get("SOPInstanceUID")) for d in dss]
    bad_uid = len(studies) > 1 or len(set(sops)) != len(sops)
    F.append(_finding(None, "UIDs consistent", "FAIL" if bad_uid else "PASS",
                      f"{len(studies)} study UID(s), {len(sops) - len(set(sops))} repeated SOP UID(s)",
                      "one series must belong to one study and every slice must have its own UID" if bad_uid else "",
                      "metadata", "Re-export the series"))
    if series.other_series:
        F.append(_finding(None, "Other series ignored", "PASS", f"{sum(series.other_series.values())} file(s)",
                          f"{len(series.other_series)} other series in the upload; the largest one is checked", "file"))
    # ---- F35 within the series
    varies = [k for k in PROTOCOL if len({str(d.get(k)) for d in dss}) > 1]
    F.append(_finding("F35", "Protocol constant within the series", "WARN" if varies else "PASS",
                      ", ".join(varies) or "constant", "parameters change between slices: mixed acquisitions?"
                      if varies else "", "acquisition"))
    # ---- F57 field and frequency
    b0, f0 = _f(d0.get("MagneticFieldStrength")), _f(d0.get("ImagingFrequency"))
    st57 = "PASS" if b0 and f0 else "WARN" if b0 else "FAIL"
    F.append(_finding("F57", "Field strength and frequency recorded", st57,
                      f"{b0 if b0 else '—'} T, {f0 if f0 else '—'} MHz",
                      {"PASS": "", "WARN": "only the nominal field", "FAIL": "neither field nor frequency: comparability "
                       "between sessions cannot be checked"}[st57], "metadata"))
    # ---- F58 dates
    days = {k: _day(d0.get(k)) for k in ("StudyDate", "SeriesDate", "AcquisitionDate")}
    known = {k: v for k, v in days.items() if v}
    spread = (max(known.values()) - min(known.values())).days if known else None
    st58 = "FAIL" if not known or spread > 1 else "PASS"
    F.append(_finding("F58", "Session dates consistent", st58,
                      ", ".join(f"{k} {v.isoformat()}" for k, v in known.items()) or "no date",
                      "no session date" if not known else f"dates differ by {spread} days within one series"
                      if spread > 1 else "", "metadata"))
    # ---- F59 de-identification (names only)
    present = [k for k in IDENT_DICOM if str(d0.get(k, "")).strip()]
    burned = str(d0.get("BurnedInAnnotation", "")).upper() == "YES"
    src_deid = str(d0.get("PatientIdentityRemoved", "")).upper() == "YES"
    st59 = "FAIL" if burned else "PASS" if src_deid or not present else "WARN"
    F.append(_finding("F59", "Identifying data", st59, f"{len(present)} field(s) removed: {', '.join(present) or 'none'}",
                      "text burned into the pixels cannot be removed from the headers" if burned else
                      "the source was not de-identified; ScanTwin removed the fields (values never stored)"
                      if st59 == "WARN" else "removed here; values never stored", "metadata"))
    # ---- F56 units (always converted)
    F.append(_finding("F56", "Units", "PASS", "TR / TE / TI ms → s; pixel spacing mm",
                      "DICOM times are in ms, BIDS in s; converted before any comparison", "metadata"))
    worst = max((x["status"] for x in F), key=RANK.get, default="PASS")
    return {"status": worst, "findings": F, "geometry": geom, "ignored": series.ignored}


def sidecar(series: DicomSeries) -> dict:
    """BIDS-like sidecar in SI units (seconds), identifying tags left out, so t1w.load/measure treat it as NIfTI's."""
    d0 = series.datasets[0]
    out = {}
    for k in PROTOCOL + ("SliceThickness", "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
                         "SeriesDescription", "ReceiveCoilName", "PatientPosition"):
        v = d0.get(k)
        if v is None or str(v).strip() == "":
            continue
        if k in MS_TO_S:
            out[k] = _f(v) / 1000.0 if _f(v) is not None else None
        elif k in ("FlipAngle", "MagneticFieldStrength", "ImagingFrequency", "SliceThickness"):
            out[k] = _f(v)
        else:
            out[k] = list(map(str, v)) if hasattr(v, "__iter__") and not isinstance(v, str) else str(v)
    return out


def volume(series: DicomSeries) -> tuple[np.ndarray, np.ndarray]:
    """Voxel array (column, row, slice) and RAS affine. DICOM patient space is LPS; NIfTI is RAS."""
    dss = series.datasets
    d0 = dss[0]
    iop = np.array(d0.ImageOrientationPatient, float)
    ps = [float(x) for x in d0.PixelSpacing]                # [between rows, between columns]
    ipp = np.array([d.ImagePositionPatient for d in dss], float)
    step = (ipp[-1] - ipp[0]) / (len(dss) - 1)
    a = np.eye(4)
    a[:3, 0], a[:3, 1], a[:3, 2], a[:3, 3] = iop[:3] * ps[1], iop[3:] * ps[0], step, ipp[0]
    vol = np.stack([d.pixel_array.T.astype(float) * float(d.get("RescaleSlope", 1) or 1)
                    + float(d.get("RescaleIntercept", 0) or 0) for d in dss], axis=-1)
    return vol, np.diag([-1.0, -1.0, 1.0, 1.0]) @ a


def to_nifti_bytes(series: DicomSeries) -> bytes:
    import nibabel as nib
    vol, aff = volume(series)
    return nib.Nifti1Image(vol.astype(np.float32), aff).to_bytes()


def load(files, *, mask=None, roi=None):
    """Read a DICOM series, run the header checks, and return (T1Scan, check result). The T1Scan is None when a FAIL
    makes the volume unusable (missing geometry, missing or repeated slices, broken pixel data)."""
    from . import t1w
    s = read_series(files)
    c = check(s)
    blocking = [x for x in c["findings"] if x["status"] == "FAIL" and x["kind"] in ("file", "geometry")
                or (x["status"] == "FAIL" and x["catalogue_id"] == "F54")]
    if blocking:
        return None, c
    scan = t1w.load(to_nifti_bytes(s), name="dicom_series.nii", mask=mask, roi=roi, sidecar=sidecar(s))
    present = [k for k in IDENT_DICOM if str(s.datasets[0].get(k, "")).strip()]
    scan.removed = sorted(set(scan.removed) | set(present))
    scan.name = f"DICOM series ({len(s.datasets)} slices)"
    scan.steps = [f"read DICOM series: {len(s.datasets)} slices"
                  + (f"; ignored non-DICOM: {', '.join(s.ignored)}" if s.ignored else ""),
                  "DICOM header checks: " + ", ".join(f"{x['label']} {x['status']}" for x in c["findings"]),
                  "converted LPS → RAS; TR / TE / TI ms → s",
                  f"de-identify: removed {len(scan.removed)} identifying field(s): {', '.join(scan.removed) or 'none'}"
                  ] + scan.steps[2:]
    return scan, c
