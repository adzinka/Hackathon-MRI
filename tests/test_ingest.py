"""Field mapper (raw NIfTI-MRS -> packet) on a synthetic file that has every difficulty of real vendor data:
several coils, individual transients, a mirrored frequency axis, tiny units, a frequency offset and identifying metadata."""
import json

import numpy as np
import pytest

from twin_evidence import ingest as I, mrs_fit as F, mrs_sim as S

IDENT = {"PatientName": "Jane Doe", "PatientDoB": "19700101", "PatientSex": "F", "PatientWeight": 60.0,
         "InstitutionAddress": "1 Example Street", "DeviceSerialNumber": "12345", "PatientPosition": "HFS"}


@pytest.fixture(scope="module")
def raw_file(tmp_path_factory):
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(0)
    sim = S.simulate(S.Acquisition(freq_offset_hz=-7.0), None, seed=3)      # 64 transients, 3 T, offset 7 Hz
    tr = sim.transients.T                                                    # (N, 64)
    gains = np.array([1.0, 0.6, 0.3, 0.8]) * np.exp(1j * np.array([0.3, -1.2, 2.0, 0.7]))
    coils = tr[:, None, :] * gains[None, :, None]                            # (N, 4, 64)
    coils = coils + 0.02 * (rng.normal(size=coils.shape) + 1j * rng.normal(size=coils.shape))
    data = (np.conj(coils) * 1e-6).astype(np.complex64)[None, None, None]    # mirrored axis, vendor units
    img = nib.Nifti1Image(data, np.eye(4))
    img.header["pixdim"][4] = F.DWELL_S
    meta = {"SpectrometerFrequency": [123.25], "ResonantNucleus": ["1H"], "dim_5": "DIM_COIL", "dim_6": "DIM_DYN",
            "EchoTime": 0.03, "RepetitionTime": 2.0, "Manufacturer": "TEST", "ProtocolName": "PRESS_test", **IDENT}
    img.header.extensions.append(nib.nifti1.Nifti1Extension(44, json.dumps(meta).encode()))
    path = tmp_path_factory.mktemp("raw") / "raw_svs.nii.gz"
    nib.save(img, str(path))
    return path


def test_read_undoes_vendor_quirks(raw_file):
    ing = I.read(raw_file)
    assert ing.transients.shape == (64, 1024)
    assert set(ing.removed) == {"PatientName", "PatientDoB", "PatientSex", "PatientWeight", "InstitutionAddress",
                                "DeviceSerialNumber"}
    assert "PatientPosition" in ing.meta and not any(k in ing.meta for k in ing.removed)
    assert any("conjugated" in s for s in ing.steps)
    q = F.qc_metrics(ing.fid, ing.dwell, nu0=ing.nu0)
    assert q["naa_ppm"] == pytest.approx(2.01, abs=0.01)


def test_measure_and_gate(raw_file):
    ing = I.read(raw_file)
    m = I.measure(ing)
    assert m["snr"] > 20 and m["fwhm_hz"] == pytest.approx(8.7, abs=1.0)
    assert m["cho_cr"] == pytest.approx(0.689, abs=0.06)           # windowed fit, ~3 % low by design
    assert m["drift_abs_hz"] < 2 and m["outlier_pct"] < 5
    assert I.gate(m, ing.nu0)["status"] in ("PASS", "WARN")


def test_packet_is_schema_valid_and_deidentified(raw_file, tmp_path):
    from jsonschema import Draft202012Validator
    from pathlib import Path
    import os
    root = Path(os.environ.get("STARTERPACK", "../MRI_Hackathon_DataScience_StarterPack_v0.3")).resolve()
    schema_path = root / "05_TwinReady_JSON_CSV" / "evidence_packet_v1.0.schema.json"
    if not schema_path.exists():
        pytest.skip("StarterPack schema not found")
    ing = I.read(raw_file)
    m = I.measure(ing)
    g = I.gate(m, ing.nu0)
    sha = I.write_derivative(ing, tmp_path / "d.nii.gz")
    p = I.build_packet(ing, m, g, packet_id="E-t0-ext_test", subject_id="sub-test", session_id="ses-baseline",
                       months=0.0, rel_path="d.nii.gz", sha=sha)
    assert not list(Draft202012Validator(json.loads(schema_path.read_text())).iter_errors(p))
    text = json.dumps(p) + (tmp_path / "d.nii.gz").read_bytes().decode("latin-1")
    assert "Jane Doe" not in text and "19700101" not in text and "Example Street" not in text


@pytest.mark.parametrize("fault, gate_id", [("shim", "F12"), ("drift", "F08"), ("noise", "F13")])
def test_simulated_followups_trip_their_gate(raw_file, fault, gate_id):
    ing = I.read(raw_file)
    strength = {"shim": 12.0, "drift": 14.0, "noise": 4.0}[fault]
    g = I.gate(I.measure(I.degrade(ing, fault, strength, seed=1)), ing.nu0)
    assert {x["id"]: x["status"] for x in g["gates"]}[gate_id] in ("WARN", "FAIL")


def test_repeat_scan_shows_no_change(raw_file):
    ing = I.read(raw_file)
    a, b = I.measure(ing), I.measure(I.degrade(ing, "none", 0.0, seed=2))
    assert abs(a["cho_cr"] - b["cho_cr"]) < 2 * a["cho_cr_crlb"]


def test_true_change_is_recovered(raw_file):
    ing = I.read(raw_file)
    a, b = I.measure(ing), I.measure(I.degrade(ing, "cho_up", 0.3, seed=4))
    assert b["cho_cr"] / a["cho_cr"] == pytest.approx(1.3, abs=0.06)
