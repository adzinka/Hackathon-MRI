"""Structural T1w: quality gates, volumes, and who explains a change (StarterPack images; follow-ups SIMULATED).

Run:  STARTERPACK=/path/to/MRI_Hackathon_DataScience_StarterPack_v0.3 pytest -q
"""
import json
import os
from pathlib import Path

import pytest

from twin_evidence import t1w as T

ROOT = Path(os.environ.get("STARTERPACK", "../MRI_Hackathon_DataScience_StarterPack_v0.3")).resolve()


def load(s):
    return T.load(ROOT / f"03_BIDS_MRI/sub-01/ses-{s}/anat/sub-01_ses-{s}_T1w.nii.gz", name=f"{s}_T1w.nii.gz",
                  mask=ROOT / "02_NIfTI/synthetic_brain_mask.nii.gz",
                  roi=ROOT / f"03_BIDS_MRI/derivatives/roi/sub-01/ses-{s}/anat/sub-01_ses-{s}_space-T1w_label-ROI_probseg.nii.gz",
                  sidecar=json.loads((ROOT / f"03_BIDS_MRI/sub-01/ses-{s}/anat/sub-01_ses-{s}_T1w.json").read_text()))


@pytest.fixture(scope="module")
def base():
    if not (ROOT / "03_BIDS_MRI").exists():
        pytest.skip(f"StarterPack not found at {ROOT}; set STARTERPACK")
    b = load("baseline")
    return b, T.measure(b)


def test_volumes_and_quality_of_the_baseline(base):
    b, m = base
    assert m["roi_mm3"] == pytest.approx(22140, rel=1e-3)          # the packet value, from the outline
    assert T.gate(m)["status"] == "PASS"
    assert m["wm_cv_pct"] < 1 and m["snr_wm"] > 30


def test_the_starterpack_pair_is_a_processing_artefact(base):
    b, mb = base
    f = load("followup")
    mf = T.measure(f, b, mb)
    assert mf["reg_shift_mm"] < 0.1 and mf["image_change_sd"] < 1.5        # same anatomy, scans line up
    c = T.compare(mb, T.gate(mb), mf, T.gate(mf))
    assert c["kind"] == "processing" and c["pct"] == pytest.approx(25.6, abs=0.5)


@pytest.mark.parametrize("fault, strength, kind, gate_id", [
    ("none", 0.0, "unknown", None),
    ("noise", 1.4, "machine", "F27"),
    ("bias", 0.8, "machine", "F28"),
    ("motion", 0.25, "session", "F26"),
    ("shift", 3.0, "session", "F32"),
    ("scale", 0.03, "machine", None),          # geometry: whole head changes size, quality passes
    ("grow", 3.0, "patient", None),
])
def test_who_explains_the_change(base, fault, strength, kind, gate_id):
    b, mb = base
    for seed in (3, 5):
        x = T.degrade(b, fault, strength, seed=seed)
        mx = T.measure(x, b, mb)
        g = T.gate(mx)
        assert T.compare(mb, T.gate(mb), mx, g)["kind"] == kind
        if gate_id:
            assert {q["id"]: q["status"] for q in g["gates"]}[gate_id] in ("WARN", "FAIL")


def test_bias_does_not_break_the_segmentation(base):
    """Realistic brightness non-uniformity (up to about +-50 %) is corrected before segmentation. Extreme fields
    (0.5x to 1.8x) are only partly corrected, but such scans FAIL F28 and their volumes are not used."""
    b, mb = base
    mx = T.measure(T.degrade(b, "bias", 0.5, seed=3), b, mb)
    assert all(abs(r - 1) < 0.02 for r in mx["volume_ratios"].values())


def test_sidecar_is_deidentified(tmp_path):
    if not (ROOT / "03_BIDS_MRI").exists():
        pytest.skip("StarterPack not found")
    s = T.load(ROOT / "03_BIDS_MRI/sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz", name="x.nii.gz",
               sidecar={"PatientName": "Jane Doe", "PatientBirthDate": "19700101", "RepetitionTime": 2.3})
    assert set(s.removed) == {"PatientName", "PatientBirthDate"} and "Jane Doe" not in json.dumps(s.meta)
