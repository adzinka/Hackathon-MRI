"""DICOM series: header checks on the StarterPack series and on broken copies of it (built in memory).

Run:  STARTERPACK=/path/to/MRI_Hackathon_DataScience_StarterPack_v0.3 pytest -q
"""
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from twin_evidence import dicom_io as D
from twin_evidence import t1w as T

ROOT = Path(os.environ.get("STARTERPACK", "../MRI_Hackathon_DataScience_StarterPack_v0.3")).resolve()
SERIES = ROOT / "01_DICOM" / "baseline"


@pytest.fixture(scope="module")
def files():
    if not SERIES.exists():
        pytest.skip(f"StarterPack DICOM not found at {SERIES}; set STARTERPACK")
    return sorted(SERIES.glob("*.dcm"))


def edited(files, change, only=None):
    """(name, bytes) pairs; `change(ds)` applied to every file, or only to the file names in `only`."""
    import pydicom
    out = []
    for p in files:
        ds = pydicom.dcmread(p)
        if only is None or p.name in only:
            change(ds)
        b = io.BytesIO()
        ds.save_as(b)
        out.append((p.name, b.getvalue()))
    return out


def status(result, label):
    return next(f["status"] for f in result["findings"] if f["label"] == label or f["id"] == label)


def test_clean_series_passes_and_matches_the_nifti(files):
    import nibabel as nib
    s = D.read_series(files)
    c = D.check(s)
    assert c["status"] == "PASS"
    vol, aff = D.volume(s)
    ref = nib.load(ROOT / "02_NIfTI" / "synthetic_T1w_baseline.nii.gz")
    a = nib.as_closest_canonical(nib.Nifti1Image(vol, aff))
    b = nib.as_closest_canonical(ref)
    assert np.allclose(a.affine, b.affine)                              # LPS -> RAS done right (changelog X11)
    assert np.abs(np.asarray(a.dataobj) - np.asarray(b.dataobj)).max() <= 0.5   # DICOM stores rounded integers


def test_missing_slice_is_an_error_and_nothing_is_measured(files):
    scan, c = D.load([p for p in files if p.name != "IM0008.dcm"])
    assert scan is None and c["status"] == "FAIL"
    assert status(c, "Series complete") == "FAIL"


def test_repeated_slice_is_an_error(files):
    c = D.check(D.read_series(files + [files[7]]))
    assert status(c, "Series complete") == "FAIL" or status(c, "UIDs consistent") == "FAIL"


def test_broken_orientation_is_an_error(files):
    c = D.check(D.read_series(edited(files, lambda d: setattr(d, "ImageOrientationPatient", [1, 0, 0, 0.5, 0.5, 0]),
                                     only={"IM0005.dcm"})))
    assert status(c, "Image orientation") == "FAIL"


def test_missing_geometry_tag_blocks_the_volume(files):
    scan, c = D.load(edited(files, lambda d: delattr(d, "PixelSpacing")))
    assert scan is None and status(c, "F54") == "FAIL"


def test_field_and_frequency(files):
    c = D.check(D.read_series(edited(files, lambda d: delattr(d, "ImagingFrequency"))))
    assert status(c, "F57") == "WARN"                                   # only the nominal field
    c = D.check(D.read_series(edited(files, lambda d: (delattr(d, "ImagingFrequency"),
                                                       delattr(d, "MagneticFieldStrength")))))
    assert status(c, "F57") == "FAIL"


def test_contradictory_dates(files):
    c = D.check(D.read_series(edited(files, lambda d: setattr(d, "SeriesDate", "20070308"))))
    assert status(c, "F58") == "FAIL"


def test_identifying_values_are_never_kept(files):
    s = D.read_series(edited(files, lambda d: setattr(d, "PatientBirthDate", "19800101")))
    c = D.check(s)
    assert "PatientBirthDate" in status_value(c, "F59")
    assert "19800101" not in json.dumps(c, default=str) and "19800101" not in json.dumps(D.sidecar(s))
    scan, _ = D.load(files)
    assert "PatientName" in scan.removed and "PatientName" not in scan.meta


def status_value(result, fid):
    return next(f["value"] for f in result["findings"] if f["id"] == fid)


def test_non_dicom_files_are_ignored(files):
    s = D.read_series(list(files) + [("LICENSE", b"Creative Commons Attribution 4.0")])
    assert s.ignored == ["LICENSE"] and D.check(s)["status"] == "PASS"


def test_ms_and_seconds_do_not_fake_a_protocol_change(files):
    """DICOM keeps TR/TE/TI in ms, BIDS in s: once converted, the DICOM baseline and the BIDS follow-up match (F35)."""
    sp = ROOT / "03_BIDS_MRI" / "sub-01" / "ses-followup" / "anat"
    side = json.loads((sp / "sub-01_ses-followup_T1w.json").read_text())
    mask = ROOT / "02_NIfTI" / "synthetic_brain_mask.nii.gz"
    b, c = D.load(files, mask=mask)
    f = T.load(sp / "sub-01_ses-followup_T1w.nii.gz", name="followup_T1w.nii.gz", mask=mask, sidecar=side)
    m = T.measure(f, b)
    assert c["status"] == "PASS" and m["protocol_diff"] == 0
    assert {g["id"]: g["status"] for g in T.gate(m)["gates"]}["F35"] == "PASS"
