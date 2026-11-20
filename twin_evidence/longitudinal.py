"""Time: a weekly phantom record and two brain sessions on the same (possibly failing) scanner. All SIMULATED.

For each hardware fault a parameter drifts from week 14 to week 26. Every week a phantom spectrum is simulated with
mrs_sim and processed like real data (SNR, linewidth, frequency, residual water) plus a geometry measurement. The
scanner twin (scanner.track) follows these values. The brain is scanned at week 0 and week 26 with the scanner state of
that week; both sessions become evidence packets, pass the gate (thresholds-team-v1.1) and update the brain twin.
Finally the scanner twin's signature names the fault class and says whether the brain WARN is the machine's fault.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import mrs_fit as F
from . import mrs_sim as S
from . import scenarios as SC
from .scanner import track
from .twin import Twin
from .validate import evaluate

WEEKS = 26
ONSET = 14

PHANTOM = replace(S.BASE if hasattr(S, "BASE") else S.Acquisition(), lw_hz=3.0, coil_noise_factor=0.6)

# hardware fault -> parameter values at week 26 (linear ramp from ONSET); "blame" for the brain WARN
FAULTS = {
    "stable": {"title": "Scanner stable", "end": {}},
    "shim_degrading": {"title": "Shim degrading", "end": {"shim_hz": 6.0}},
    "coil_degrading": {"title": "Receive coil degrading", "end": {"coil_noise_factor": 1.45}},
    "b0_drift": {"title": "B0 drift", "end": {"freq_offset_hz": 14.0, "drift_hz": 8.0}},
    "b1_degrading": {"title": "RF transmit (B1) miscalibration", "end": {"b1_error": -0.2}},
    "gradient_drift": {"title": "Gradient scale drift", "end": {"gradient_error": 0.025}},
}

# phantom observables: (label, unit, per-scan SD, drift SD per 6 months, gates). Gates are a team PROPOSAL,
# set from the stable simulated phantom (mean +- several SD); a site physicist must replace them.
PHANTOM_OBS = {
    "ph_fwhm_hz": ("Phantom linewidth", "Hz", 0.15, 0.3, {"warn_above": 4.5, "fail_above": 6.0}),
    "ph_snr": ("Phantom SNR", "1", 5.0, 6.0, {"warn_below": 95.0, "fail_below": 80.0}),
    "ph_freq_hz": ("Centre-frequency offset", "Hz", 0.3, 1.0, {"warn_above": 6.0, "fail_above": 12.0}),
    "ph_water_ratio": ("Residual water", "1", 0.02, 0.05, {"warn_above": 0.4, "fail_above": 1.0}),
    "ph_distortion_pct": ("Geometric distortion", "%", 0.15, 0.2, {"warn_above": 1.0, "fail_above": 2.0}),
}


def hw_state(fault: str, week: int) -> dict:
    end = FAULTS[fault]["end"]
    frac = 0.0 if week <= ONSET else (week - ONSET) / (WEEKS - ONSET)
    base = {"shim_hz": 0.0, "coil_noise_factor": 1.0, "freq_offset_hz": 0.0, "drift_hz": 0.0, "b1_error": 0.0, "gradient_error": 0.0}
    return {k: base[k] + frac * (end.get(k, base[k]) - base[k]) for k in base}


def phantom_week(fault: str, week: int, seed: int) -> dict:
    hw = hw_state(fault, week)
    acq = replace(PHANTOM, shim_hz=hw["shim_hz"], coil_noise_factor=PHANTOM.coil_noise_factor * hw["coil_noise_factor"],
                  freq_offset_hz=hw["freq_offset_hz"], drift_hz=hw["drift_hz"], b1_error=hw["b1_error"],
                  gradient_error=hw["gradient_error"])
    sim = S.simulate(acq, None, seed)
    q = F.qc_metrics(sim.average())
    rng = np.random.default_rng(seed + 7)
    return {"week": week, "ph_fwhm_hz": q["fwhm_hz"], "ph_snr": q["snr"],
            "ph_freq_hz": (q["naa_ppm"] - F.METABOLITES["NAA"]) * F.NU0_MHZ, "ph_water_ratio": q["water_ratio"],
            "ph_distortion_pct": 100 * hw["gradient_error"] + rng.normal(0, 0.15), **{f"hw_{k}": v for k, v in hw.items()}}


def brain_session(fault: str, week: int, seed: int, root: Path, biology: dict | None = None):
    hw = hw_state(fault, week)
    acq = replace(S.Acquisition(), shim_hz=hw["shim_hz"], coil_noise_factor=hw["coil_noise_factor"],
                  freq_offset_hz=hw["freq_offset_hz"], drift_hz=hw["drift_hz"], b1_error=hw["b1_error"],
                  gradient_error=hw["gradient_error"], **(biology or {}))
    sim = S.simulate(acq, root, seed)
    return sim, SC.measure(sim)


# --------------------------------------------------------------------------------------- packets (schema v1.0)
OTHER_U = 0.045   # placement, chemical-shift displacement, lineshape and calibration terms (Dossier §5.7), assumption


def make_packet(template: dict, pid: str, session: str, months: float, m: dict, rel_path: str, sha: str,
                ref_te: float = S.TE_REF_MS) -> dict:
    p = json.loads(json.dumps(template))
    u = math.hypot(m["cho_cr_crlb"], OTHER_U)
    checks = [
        ("mrs_snr", round(m["snr"], 2), "1", "QE"), ("mrs_linewidth_Hz", round(m["fwhm_hz"], 2), "Hz", "PM"),
        ("registration_rms_mm", 0.7, "mm", "PM"), ("missing_values", 0, "1", "DS"),
        ("mrs_water_ratio", round(m["water_ratio"], 3), "1", "QE"),
        ("mrs_drift_Hz", round(m["drift_abs_hz"], 2), "Hz", "PM"),
        ("mrs_transient_outliers", round(m["outlier_frac"], 3), "1", "QE"),
    ]
    p.update({
        "packet_id": pid, "session_id": session, "relative_time": {"value": months, "unit": "month"},
        "origin": "SIMULATED", "value_epistemic": "DERIVED", "thresholds_ref": "thresholds-team-v1.1",
        "value": round(float(m["cho_cr"]), 4), "uncertainty": {"type": "SD", "value": round(u, 4), "budget_ref": "CRLB (+) 0.045 other terms"},
        "source_object": rel_path, "source_sha256": sha, "protocol_match": abs(m["te_ms"] - ref_te) < 1e-6,
        "qc_checks": [{"name": n, "value": v, "unit": un, "threshold": "see thresholds-team-v1.1", "status": "PASS", "owner": o}
                      for n, v, un, o in checks],
        "qc_status": "PASS", "qc_reasons": [],
        "interpretation_limit": "SIMULATED fault scenario with known truth; teaching assumptions; no clinical meaning",
        "provenance": {"parents": [rel_path], "activities": ["mrs_sim.simulate", "average", "mrs_fit.fit", "QC aggregation"]},
        "twin_action": {"recommended": "update", "reason": "placeholder: stored statuses are PASS on purpose; the gate re-evaluates every check"},
    })
    p["acquisition_context"] = {**p.get("acquisition_context", {}), "TE_s": m["te_ms"] / 1000}
    return p


def thresholds_v11(th_v10: dict) -> dict:
    th = json.loads(json.dumps(th_v10))
    th["id"] = "thresholds-team-v1.1"
    th["thresholds_version"] = "1.1-team"
    th["source"] = "thresholds-v1.0 + team PROPOSAL gates for residual water, drift and transient outliers (SIMULATED basis)"
    th["gates"] = [g for g in th["gates"] if g["name"] not in {x["name"] for x in SC.NEW_GATES}] + SC.NEW_GATES
    return th


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------- diagnosis from the twin
def diagnose(trk: dict) -> dict:
    """Fault class from the pattern of scanner-twin changes between week 0 and week 26 (transparent rules)."""
    z = trk["z"]
    sig = {"fwhm_up": z["ph_fwhm_hz"] > 1.96, "snr_down": z["ph_snr"] < -1.96, "freq_moved": abs(z["ph_freq_hz"]) > 1.96,
           "water_up": z["ph_water_ratio"] > 1.96, "geometry_up": z["ph_distortion_pct"] > 1.96}
    if sig["geometry_up"]:
        cls, why = "gradient", "phantom geometry changed; spectra unaffected"
    elif sig["freq_moved"]:
        cls, why = "b0_drift", "centre frequency moved"
    elif sig["water_up"]:
        cls, why = "b1_transmit", "residual water rose (water suppression depends on flip angle)"
    elif sig["fwhm_up"]:
        cls, why = "shim", "phantom lines broadened (SNR falls with peak height)"
    elif sig["snr_down"]:
        cls, why = "receive_coil", "SNR fell with normal linewidth"
    else:
        cls, why = "stable", "no phantom observable changed beyond its uncertainty"
    return {"fault_class": cls, "why": why, "signature": sig, "z": z}


EXPECTED_CLASS = {"stable": "stable", "shim_degrading": "shim", "coil_degrading": "receive_coil", "b0_drift": "b0_drift",
                  "b1_degrading": "b1_transmit", "gradient_drift": "gradient"}


def run_fault(fault: str, root: Path, out: Path, th11: dict, schema: dict, template: dict, seed: int = 0,
              biology_t1: dict | None = None) -> dict:
    """Full chain for one fault: phantom record -> scanner twin -> diagnosis; brain t0/t1 -> packets -> gate -> twin."""
    weeks = [phantom_week(fault, w, seed * 1000 + w) for w in range(WEEKS + 1)]
    trk = track({o: [w[o] for w in weeks] for o in PHANTOM_OBS}, PHANTOM_OBS)
    dx = diagnose(trk)

    tag = fault if not biology_t1 else f"{fault}_biology"
    packets, files = [], {}
    for sess, week, months, bio in (("ses-baseline", 0, 0, None), ("ses-followup", WEEKS, 6, biology_t1)):
        sim, m = brain_session(fault, week, seed * 1000 + 500 + week, root, bio)
        rel = f"longitudinal/{tag}/sub-sim_{sess}_svs.nii.gz"
        path = out / rel
        S.write_nifti_mrs(sim.average(), path, {"origin": "SIMULATED", "fault": fault, "week": week, "truth": sim.truth,
                                               "acquisition": sim.acq.to_dict()})
        files[rel] = sha256(path)
        pid = f"E-t{0 if week == 0 else 1}-sim_{tag}"
        packets.append((make_packet(template, pid, sess, months, m, rel, files[rel]), m, sim.truth))
    manifest = dict(files)
    twin, events = Twin(), []
    for p, m, truth in packets:
        ev = evaluate(p, th11, schema, manifest, out)
        e = twin.ingest(p, ev)
        events.append((p, ev, e, m, truth))
        (out / f"longitudinal/{tag}/{p['packet_id']}.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    p1, ev1, e1, m1, tr1 = events[-1]
    machine = dx["fault_class"] != "stable"
    brain_flagged = ev1.status != "PASS"
    if brain_flagged and machine:
        verdict = f"MACHINE: the brain {ev1.status} matches a scanner fault ({dx['fault_class']}). Fix the scanner, then repeat."
    elif brain_flagged:
        verdict = (f"SESSION: the scanner is stable, so the brain {ev1.status} comes from this session "
                   "(motion, position, local shim).")
    elif machine:
        verdict = (f"HIDDEN MACHINE FAULT: the brain spectrum passes, but the scanner twin sees {dx['fault_class']}. "
                   "Values may be biased without any spectral warning.")
    else:
        verdict = "CLEAN: scanner stable and brain evidence PASS; a change in the value is attributed to the patient."
    return {"fault": tag, "title": FAULTS[fault]["title"], "weeks": weeks, "track": trk, "diagnosis": dx,
            "expected_class": EXPECTED_CLASS[fault], "correct": dx["fault_class"] == EXPECTED_CLASS[fault],
            "brain": [{"packet_id": p["packet_id"], "status": ev.status, "action": e.action,
                       "reasons": [f"{f.rule}: {f.reason}" for f in ev.findings if f.status != "PASS"],
                       "value": p["value"], "u": p["uncertainty"]["value"], "truth_tissue": truth["Cho_Cr_tissue"],
                       "truth_measurable": truth["Cho_Cr_measurable"],
                       "posterior": e.posterior, "K": e.K} for p, ev, e, m, truth in events],
            "verdict": verdict}
