"""The two twins talk: the scanner twin's state on the scan date gates every subject packet (all scanner data SIMULATED).

Run:  STARTERPACK=/path/to/MRI_Hackathon_DataScience_StarterPack_v0.3 pytest -q
"""
import os
from pathlib import Path

import pytest

from twin_evidence.scanner_link import ScannerRecord, attribution
from twin_evidence.twin import Twin
from twin_evidence.validate import evaluate, load_json, load_manifest

ROOT = Path(os.environ.get("STARTERPACK", "../MRI_Hackathon_DataScience_StarterPack_v0.3")).resolve()
TW = ROOT / "05_TwinReady_JSON_CSV"
ORDER = ["E-t0-roi", "E-t0-naa", "E-t0-cho", "E-t1-roi", "E-t1-naa", "E-t1-cho"]


@pytest.fixture(scope="module")
def pack():
    if not TW.exists():
        pytest.skip(f"StarterPack not found at {ROOT}; set STARTERPACK")
    return (load_json(TW / "thresholds-v1.0.json"), load_json(TW / "evidence_packet_v1.0.schema.json"),
            load_manifest(ROOT / "manifest_sha256.csv"))


@pytest.fixture(scope="module")
def records():
    return {f: ScannerRecord.simulate(f) for f in ("stable", "shim_degrading", "coil_degrading", "b0_drift",
                                                   "b1_degrading", "gradient_drift")}


def run(pack, rec):
    th, sc, mf = pack
    t, out = Twin(), {}
    for n in ORDER:
        p = load_json(TW / "packets" / f"{n}.json")
        ev = evaluate(p, th, sc, mf, ROOT, scanner=rec)
        out[n] = (ev, t.ingest(p, ev))
    return out


def scan_finding(ev):
    return next((f for f in ev.findings if f.rule == "F11_scanner_twin"), None)


def test_stable_scanner_keeps_the_gate1_reference(pack, records):
    out = run(pack, records["stable"])
    ev, e = out["E-t1-cho"]
    assert scan_finding(ev).status == "PASS"
    assert e.K == pytest.approx(0.321, abs=0.001) and e.posterior["x"] == pytest.approx(0.741, abs=0.001)


def test_no_scanner_twin_means_no_finding(pack):
    ev, _ = run(pack, None)["E-t1-cho"]
    assert scan_finding(ev) is None


def test_baseline_week_is_healthy_for_every_fault(pack, records):
    for rec in records.values():
        for n in ("E-t0-cho", "E-t0-roi"):
            assert scan_finding(run(pack, rec)[n][0]).status == "PASS"


@pytest.mark.parametrize("fault, cho_action, roi_status", [
    ("shim_degrading", "reject", "WARN"),       # shim FAIL blocks spectroscopy; volumes only see the SNR loss
    ("b0_drift", "reject", "WARN"),
    ("gradient_drift", "reject", "FAIL"),       # geometry FAIL blocks both the spectrum and the volume
    ("coil_degrading", "update_with_warning", "WARN"),
])
def test_scanner_fault_gates_subject_packets(pack, records, fault, cho_action, roi_status):
    out = run(pack, records[fault])
    assert out["E-t1-cho"][1].action == cho_action
    assert scan_finding(out["E-t1-roi"][0]).status == roi_status


def test_blocked_update_leaves_the_subject_twin_unchanged(pack, records):
    out = run(pack, records["shim_degrading"])
    assert out["E-t1-cho"][1].kind == "reject"


@pytest.mark.parametrize("fault, who", [("stable", "session"), ("shim_degrading", "machine"), ("b1_degrading", "machine")])
def test_attribution(pack, records, fault, who):
    ev, _ = run(pack, records[fault])["E-t1-cho"]
    assert attribution(ev.status, scan_finding(ev), records[fault])["who"] == who
