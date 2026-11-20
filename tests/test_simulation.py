"""Known-truth tests of the simulator, the fit and the fault chain (all SIMULATED data).

Run:  STARTERPACK=/path/to/MRI_Hackathon_DataScience_StarterPack_v0.3 pytest -q
"""
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from twin_evidence import mrs_fit as F, mrs_sim as S, scenarios as SC

ROOT = Path(os.environ.get("STARTERPACK", "../MRI_Hackathon_DataScience_StarterPack_v0.3")).resolve()


@pytest.fixture(scope="module")
def root():
    if not (ROOT / "02_NIfTI").exists():
        pytest.skip(f"StarterPack not found at {ROOT}; set STARTERPACK")
    return ROOT


def test_fit_recovers_noiseless_truth():
    amps, truth = S.true_amplitudes(S.Acquisition(), None)
    fid = S._noiseless(amps, 8.7, 0.0, 0.0, F.time_axis())
    r = F.fit(fid)
    assert r.ratio("Cho")[0] == pytest.approx(truth["Cho_Cr_measurable"], abs=0.003)
    assert r.lw_hz == pytest.approx(8.7, abs=0.25)


def test_fit_on_real_starterpack_spectra(root):
    import nibabel as nib
    vals = {}
    for s in ("baseline", "followup"):
        fid = np.asarray(nib.load(root / f"04_MRS_BIDS/sub-01/ses-{s}/mrs/sub-01_ses-{s}_acq-press_nuc-1H_voi-frontal_svs.nii.gz").dataobj).reshape(-1)
        vals[s] = F.fit(fid).ratio("Cho")[0]
        q = F.qc_metrics(fid)
        assert q["snr"] == pytest.approx({"baseline": 31.2, "followup": 24.8}[s], abs=0.3)
    assert vals["baseline"] == pytest.approx(0.69, abs=0.03)     # packet values, within ~1 packet SD
    assert vals["followup"] == pytest.approx(0.85, abs=0.04)


def test_clean_simulation_matches_starterpack_baseline(root):
    m = SC.run("clean", root, seed=11)
    assert m["gate"] == "PASS"
    assert m["snr"] == pytest.approx(31.2, abs=4)
    assert m["fwhm_hz"] == pytest.approx(8.7, abs=0.8)
    assert m["cho_cr"] == pytest.approx(0.689, abs=3 * m["cho_cr_crlb"])


def test_shim_broadening_is_not_fixed_by_alignment_but_drift_is(root):
    shim = SC.run("poor_shim", root, seed=3)
    drift = SC.run("freq_drift", root, seed=3)
    assert shim["fwhm_hz"] > 10 and shim["fwhm_gain_aligned"] < 1.0
    assert drift["fwhm_hz"] > 10 and drift["fwhm_gain_aligned"] > 3.0
    assert drift["drift_abs_hz"] == pytest.approx(12, abs=3)


def test_motion_is_flagged_by_transient_outliers(root):
    """Small jumps hide in the noise, so test the average over seeds against the clean scan, not one seed."""
    motion = np.mean([SC.run("motion", root, seed=s)["outlier_frac"] for s in (5, 6, 7, 8)])
    clean = np.mean([SC.run("clean", root, seed=s)["outlier_frac"] for s in (5, 6)])
    assert motion > 0.08 and clean < 0.03


@pytest.mark.parametrize("key, gate_id", [("low_snr_coil", "F13"), ("poor_shim", "F12"), ("protocol_te", "F35"),
                                          ("freq_drift", "F08"), ("motion", "F23"), ("voxel_misplacement", "F20")])
def test_each_fault_trips_its_catalogue_gate(root, key, gate_id):
    assert SC.run(key, root, seed=7)[f"check_{gate_id}"] in ("WARN", "FAIL")


def test_known_gap_water_suppression_passes_f15_but_biases_cho(root):
    """Documents a finding (docs/scantwin/REVIEW.md): with the catalogue F15 limit (WARN > 5) a residual water of ~2x NAA
    passes, yet Cho/Cr is biased by several SD. If the team lowers the F15 limit, update this test."""
    r = SC.run("water_suppression", root, seed=7)
    assert r["check_F15"] == "PASS" and r["cho_cr"] < 0.689 - 3 * 0.022


@pytest.mark.parametrize("key", sorted(SC.SPECTRALLY_SILENT))
def test_spectrally_silent_scenarios_have_clean_spectra(root, key):
    """Misplacement, gradient error and a true change leave the spectrum clean (F12, F13, F15 pass). Only session
    metadata (F20 voxel position) or the scanner twin (phantom geometry) can reveal the first two."""
    r = SC.run(key, root, seed=9)
    assert r["check_F12"] == r["check_F13"] == r["check_F15"] == "PASS"


def test_scanner_chain_blames_the_machine_for_shim(root, tmp_path):
    from twin_evidence import longitudinal as L
    from twin_evidence.validate import load_json
    tw = root / "05_TwinReady_JSON_CSV"
    th = L.thresholds_v11(load_json(tw / "thresholds-v1.0.json"))
    args = (root, tmp_path, th, load_json(tw / "evidence_packet_v1.0.schema.json"), load_json(tw / "packets" / "E-t1-cho.json"))
    shim = L.run_fault("shim_degrading", *args, seed=1)
    assert shim["diagnosis"]["fault_class"] == "shim" and shim["brain"][1]["status"] == "WARN"
    assert shim["verdict"].startswith("MACHINE")
    grad = L.run_fault("gradient_drift", *args, seed=1)
    assert grad["diagnosis"]["fault_class"] == "gradient" and grad["verdict"].startswith("HIDDEN MACHINE FAULT")
    bio = L.run_fault("stable", *args, seed=1, biology_t1={"cho_factor": 1.2})
    assert bio["verdict"].startswith("CLEAN") and bio["brain"][1]["value"] > 0.78
