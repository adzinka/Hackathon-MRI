"""Golden and negative regression tests (Dossier §3.10, §7.5, §8.7).

Run:  STARTERPACK=/path/to/MRI_Hackathon_DataScience_StarterPack_v0.3 pytest -q
"""
import copy
import os
from pathlib import Path

import pytest

from twin_evidence.validate import evaluate, load_json, load_manifest
from twin_evidence.twin import Twin

ROOT = Path(os.environ.get("STARTERPACK", "../MRI_Hackathon_DataScience_StarterPack_v0.3")).resolve()
TW = ROOT / "05_TwinReady_JSON_CSV"


@pytest.fixture(scope="module")
def pack():
    if not TW.exists():
        pytest.skip(f"StarterPack not found at {ROOT}; set STARTERPACK")
    return (load_json(TW / "thresholds-v1.0.json"), load_json(TW / "evidence_packet_v1.0.schema.json"),
            load_manifest(ROOT / "manifest_sha256.csv"))


def pk(name):
    return load_json(TW / "packets" / f"{name}.json")


def run(pack, names):
    th, sc, mf = pack
    t = Twin()
    return t, [t.ingest(p, evaluate(p, th, sc, mf, ROOT)) for p in map(pk, names)]


# ---------- golden path: the worked reference update ----------
def test_cho_reference_update(pack):
    t, ev = run(pack, ["E-t0-cho", "E-t1-cho"])
    e = ev[-1]
    assert e.action == "update_with_warning"
    assert e.z_change == pytest.approx(2.05, abs=0.01)
    assert e.d2 == pytest.approx(3.66, abs=0.01) and not e.d2_flag
    assert e.K == pytest.approx(0.321, abs=0.001)
    assert e.posterior["x"] == pytest.approx(0.741, abs=0.001)
    assert e.posterior["sd"] == pytest.approx(0.048, abs=0.001)


def test_cho_without_warn_policy(pack):
    th, sc, mf = pack
    th2 = copy.deepcopy(th); th2["twin_policy"]["WARN"]["R_factor"] = 1.0
    t = Twin()
    for n in ["E-t0-cho", "E-t1-cho"]:
        e = t.ingest(pk(n), evaluate(pk(n), th2, sc, mf, ROOT))
    assert e.K == pytest.approx(0.486, abs=0.001) and e.posterior["x"] == pytest.approx(0.768, abs=0.001)


def test_naa_reference_update(pack):
    _, ev = run(pack, ["E-t0-naa", "E-t1-naa"])
    assert ev[-1].K == pytest.approx(0.26, abs=0.005) and ev[-1].z_change == pytest.approx(-0.90, abs=0.01)


def test_roi_consistency_trap(pack):
    _, ev = run(pack, ["E-t0-roi", "E-t1-roi"])
    e = ev[-1]
    assert e.status == "WARN"                       # stored PASS is too optimistic
    assert e.u == pytest.approx(0.1894 * 27810, rel=0.01)
    assert e.z_change == pytest.approx(1.07, abs=0.01)
    assert e.K == pytest.approx(0.009, abs=0.001)


def test_bad_packet_is_rejected(pack):
    t, ev = run(pack, ["E-t0-cho", "E-t1-chobad"])
    assert ev[-1].kind == "reject" and t.state["Cho_Cr"].x == pytest.approx(0.69)


def test_next_measurement(pack):
    t, _ = run(pack, ["E-t0-cho", "E-t1-cho"])
    assert t.next_measurement("Cho_Cr", 0.050, 1) == pytest.approx(0.035, abs=0.001)


# ---------- negative tests: every failure path from the Data Science lecture ----------
def judge(pack, mutate):
    th, sc, mf = pack
    p = pk("E-t1-cho"); mutate(p)
    return evaluate(p, th, sc, mf, ROOT)


@pytest.mark.parametrize("label, mutate, rule", [
    ("value as text '0.85'", lambda p: p.__setitem__("value", "0.85"), "schema"),
    ("value 3.4 outside [0, 3]", lambda p: p.__setitem__("value", 3.4), "valid_range"),
    ("missing session_id", lambda p: p.pop("session_id"), "schema"),
    ("value null", lambda p: p.__setitem__("value", None), "missing_values"),
    ("uncertainty null", lambda p: p["uncertainty"].__setitem__("value", None), "uncertainty_missing"),
    ("tampered digest", lambda p: p.__setitem__("source_sha256", "0" * 64), "source_sha256"),
    ("unknown source file", lambda p: p.__setitem__("source_object", "nowhere.nii.gz"), "source_in_manifest"),
    ("SNR 9 (re-evaluated)", lambda p: p["qc_checks"][0].__setitem__("value", 9.0), "mrs_snr"),
    ("no system QC id", lambda p: p.pop("system_qc_id"), "system_qc"),
    ("SIMULATED labelled MEASURED", lambda p: p.__setitem__("value_epistemic", "MEASURED"), "epistemic_label"),
])
def test_fail_paths(pack, label, mutate, rule):
    ev = judge(pack, mutate)
    assert ev.status == "FAIL" and ev.action == "reject", label
    assert any(f.rule == rule and f.status == "FAIL" for f in ev.findings), label


def test_protocol_change_warns(pack):
    ev = judge(pack, lambda p: p.__setitem__("protocol_match", False))
    assert ev.status == "WARN" and any(f.rule == "protocol_match" for f in ev.findings)
