"""Real 1H-MRS (Morelli et al., sub-01, 3 T sLASER, 750 transients) and its water reference through ScanTwin.

Run:  pytest -q tests/real_samples
The expected gate results are the ones recorded in external_data/real_samples/qc_report.json; if the pipeline changes,
regenerate the report (python scripts/run_real_samples.py) and review the difference.
"""
import json

import pytest

from twin_evidence import ingest as I, mrs_fit as F
from twin_evidence.gates import STATE

nib = pytest.importorskip("nibabel")
SVS, REF = "sub-01_svs.nii.gz", "sub-01_ref.nii.gz"


def _ext(path) -> dict:
    exts = [e for e in nib.load(str(path)).header.extensions if e.get_code() == 44]
    assert len(exts) == 1, "NIfTI-MRS needs exactly one JSON header extension (code 44)"
    return json.loads(exts[0].get_content())


@pytest.fixture(scope="module")
def run(real_file):
    ing = I.read(real_file(SVS))
    m = I.measure(ing)
    return ing, m, I.gate(m, ing.nu0)


@pytest.mark.parametrize("name, shape", [(SVS, (1, 1, 1, 2048, 1, 750)), (REF, (1, 1, 1, 2048))])
def test_nifti_mrs_header_is_valid(real_file, name, shape):
    img = nib.load(str(real_file(name)))
    h = img.header
    assert isinstance(img, nib.Nifti2Image)
    assert h.get_intent()[2].startswith("mrs_v")            # NIfTI-MRS intent name, not a plain NIfTI image
    assert h.get_data_shape() == shape
    assert img.get_data_dtype().kind == "c"                 # complex time-domain data
    assert float(h["pixdim"][4]) == pytest.approx(0.0005)   # dwell time -> 2000 Hz spectral width
    meta = _ext(real_file(name))
    assert meta["SpectrometerFrequency"] == [123.251253] and meta["ResonantNucleus"] == ["1H"]
    assert meta["EchoTime"] == [0.028] and meta["RepetitionTime"] == [2]


def test_transient_count_and_sidecar_agree(real_file):
    meta = _ext(real_file(SVS))
    side = json.loads(real_file("sub-01_svs.json").read_text())
    assert meta["NumberOfTransients"] == [750] == [side["NumberOfTransients"]]
    assert nib.load(str(real_file(SVS))).header.get_data_shape()[5] == 750
    for k, v in side.items():
        assert meta[k] == [v] or meta[k] == v


def test_no_identifying_metadata(real_file):
    for name in (SVS, REF):
        meta = _ext(real_file(name))
        assert I.deidentify(meta)[1] == []
    for name in ("sub-01_svs.json", "sub-01_ref.json"):
        assert I.deidentify(json.loads(real_file(name).read_text()))[1] == []


def test_reader_extracts_the_spectrum(run):
    ing, _, _ = run
    assert ing.transients.shape == (750, 2048)
    assert ing.nu0 == pytest.approx(123.251253) and 1 / ing.dwell == pytest.approx(2000.0)
    assert ing.meta["EchoTime"] == pytest.approx(0.028)     # one-element lists are read as scalars
    assert any(s.startswith("metadata:") for s in ing.steps)
    q = F.qc_metrics(ing.fid, ing.dwell, nu0=ing.nu0)
    assert q["naa_ppm"] == pytest.approx(2.01, abs=0.02)


def test_pipeline_computes_every_single_scan_gate(run):
    _, m, g = run
    st = {x["id"]: x["status"] for x in g["gates"]}
    for gid in ("F12", "F13", "F14", "F15", "F08", "F23"):  # measurable from one scan with transients
        assert st[gid] in ("PASS", "WARN", "FAIL")
    for gid in ("F20", "F21", "F35", "F19", "F61"):         # need a baseline / segmentation / phantom: N/A
        assert st[gid] == "N/A"
    assert m["te_ms"] == pytest.approx(28.0)


def test_result_matches_the_recorded_report(run, report):
    _, m, g = run
    rec = report["mrs"]
    assert {x["id"]: x["status"] for x in g["gates"]} == {x["id"]: x["status"] for x in rec["gates"]}
    assert STATE[g["status"]] == rec["state"]
    assert m["snr"] == pytest.approx(rec["metrics"]["snr"], rel=1e-3)
    assert m["water_ratio"] == pytest.approx(rec["metrics"]["water_ratio"], rel=1e-3)


def test_water_reference_loads(real_file, run, report):
    ing, _, _ = run
    w = I.water_reference(real_file(REF))
    assert w["points"] == 2048 and w["removed"] == []
    assert w["nu0"] == pytest.approx(ing.nu0) and w["dwell"] == pytest.approx(ing.dwell)
    assert w["water_fwhm_hz"] == pytest.approx(report["mrs"]["water_reference"]["water_fwhm_hz"], abs=0.01)
