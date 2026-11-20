"""A second twin: the MRI scanner itself, followed through a weekly phantom record (Dossier §5.3, §11.3, App. J.6).

The StarterPack has no phantom series yet (tier 2 is open), so this module SIMULATES one. Everything it produces is
labelled SIMULATED; the scenarios are teaching assumptions, not facts about any real scanner.

The scanner twin reuses the same gate and the same gated Kalman step as the brain twin; only the observables,
their uncertainties and their drift change. Its job in the system: separate "the subject changed" from "the
instrument changed", and recommend service before the images degrade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .twin import Twin, CHI2_95_1DOF
from .validate import Evaluation, Finding, worst

WEEKS = 26  # 2026-05-20 (baseline session) -> 2026-11-18 (week of the follow-up session)
MONTHS_PER_WEEK = 12 / 52.1775

# observable -> (label, unit, per-scan SD, drift SD per 6 months, gates, "bad" direction)
# Gates are our team proposal for a spherical water phantom at 3 T; they are not agreed with the partners.
OBS = {
    "phantom_linewidth_Hz": ("Phantom linewidth (shim quality)", "Hz", 0.35, 1.2, {"warn_above": 8.0, "fail_above": 10.0}),
    "phantom_snr": ("Phantom SNR (receive chain)", "1", 4.0, 6.0, {"warn_below": 180.0, "fail_below": 150.0}),
    "freq_offset_Hz": ("Centre-frequency drift (B0)", "Hz", 2.0, 15.0, {"warn_above": 40.0, "fail_above": 80.0}),
}

SCENARIOS = {
    "stable": "Scanner stable: phantom linewidth, SNR and frequency stay flat for six months.",
    "shim_degrading": "Shim degrading: from mid-September the phantom linewidth creeps up (e.g. a failing shim "
                      "supply or a passive-shim shift) and SNR falls with it.",
    "b0_drift": "B0 drift: the centre frequency walks away (e.g. cryostat or room-temperature change); shim and SNR stay OK.",
}


@dataclass
class ScannerReport:
    series: dict          # obs -> list of dicts (week, t_months, value, status, posterior x/sd)
    state_t0: dict        # obs -> (x, sd) at the baseline session
    state_t1: dict        # obs -> (x, sd) at the follow-up session
    z: dict               # obs -> z of the change between the sessions
    status_now: str       # worst gate status of the latest phantom scan
    weeks_to_fail: dict   # obs -> forecast weeks until the FAIL limit (None if not trending there)
    verdict: str          # one line for the brain twin
    machine_explains_mrs: bool


def simulate(scenario: str = "shim_degrading", seed: int = 3) -> dict:
    """Weekly phantom values (SIMULATED). Returns obs -> array of WEEKS + 1 values."""
    rng = np.random.default_rng(seed)
    w = np.arange(WEEKS + 1)
    lw = 6.8 + rng.normal(0, OBS["phantom_linewidth_Hz"][2], w.size)
    snr = 215 + rng.normal(0, OBS["phantom_snr"][2], w.size)
    fo = 3 + rng.normal(0, OBS["freq_offset_Hz"][2], w.size)
    if scenario == "shim_degrading":
        ramp = np.clip(w - 17, 0, None)          # starts mid-September
        lw += 0.14 * ramp ** 1.25
        snr -= 2.6 * ramp ** 1.25                # broader lines -> lower peak height -> lower SNR
    elif scenario == "b0_drift":
        fo += 1.6 * w
    return {"phantom_linewidth_Hz": lw, "phantom_snr": snr, "freq_offset_Hz": fo}


def gate_value(gates: dict, value: float) -> str:
    g = gates
    if "fail_above" in g and value > g["fail_above"] or "fail_below" in g and value < g["fail_below"]:
        return "FAIL"
    if "warn_above" in g and value > g["warn_above"] or "warn_below" in g and value < g["warn_below"]:
        return "WARN"
    return "PASS"


def _gate(obs: str, value: float) -> str:
    return gate_value(OBS[obs][4], value)


def track(data: dict, obs_spec: dict, warn_factor: float = 2.0, gate_evidence: bool = False) -> dict:
    """Run weekly phantom values through the Kalman step, one scalar twin per observable.

    For the scanner twin a bad phantom value is not bad evidence: it is good evidence of a bad machine. So by default
    every reading updates the twin with its own SD, and the PASS/WARN/FAIL status only describes the machine's health.
    gate_evidence=True applies the brain-twin policy instead (WARN x warn_factor, FAIL rejected).

    data: obs -> sequence of weekly values; obs_spec: obs -> (label, unit, per-scan SD, drift SD per 6 months, gates).
    Returns series, the twin state at the first and last week, z of the change, and a straight-line forecast of the
    weeks until each FAIL limit.
    """
    q6 = {o: (lambda x, s=spec[3]: s) for o, spec in obs_spec.items()}
    twin = Twin(q6=q6)
    n = len(next(iter(data.values())))
    series = {o: [] for o in obs_spec}
    for wk in range(n):
        t = wk * MONTHS_PER_WEEK
        for o, (label, unit, sd, _, gates) in obs_spec.items():
            v = float(data[o][wk])
            st = gate_value(gates, v)
            est = st if gate_evidence else "PASS"
            ev = Evaluation(f"PH-w{wk:02d}-{o}", est, st, {"PASS": "update", "WARN": "update_with_warning", "FAIL": "reject"}[est],
                            {"PASS": 1.0, "WARN": warn_factor, "FAIL": None}[est], sd if est != "FAIL" else None,
                            [Finding(o, st, "PM", o, round(v, 3), str(gates), f"{label} {v:.2f} {unit}")])
            pk = {"packet_id": ev.packet_id, "observable": o, "value": v, "origin": "SIMULATED",
                  "relative_time": {"value": round(t, 4), "unit": "month"}}
            e = twin.ingest(pk, ev)
            s = twin.state[o]
            series[o].append({"week": wk, "t": t, "value": v, "status": st, "x": s.x, "sd": s.sd, "action": e.action})
    t0 = {o: (series[o][0]["x"], series[o][0]["sd"]) for o in obs_spec}
    t1 = {o: (series[o][-1]["x"], series[o][-1]["sd"]) for o in obs_spec}
    z = {o: (t1[o][0] - t0[o][0]) / math.hypot(t0[o][1], t1[o][1]) for o in obs_spec}
    weeks_to_fail = {}
    for o, spec in obs_spec.items():
        xs = np.array([p["week"] for p in series[o][-8:]], float)
        ys = np.array([p["x"] for p in series[o][-8:]], float)
        slope = np.polyfit(xs, ys, 1)[0]
        g = spec[4]
        lim, toward = (g["fail_above"], slope > 0) if "fail_above" in g else (g["fail_below"], slope < 0)
        if toward and abs(slope) > spec[2] / 8:
            weeks_to_fail[o] = max(0.0, (lim - ys[-1]) / slope)
        else:
            weeks_to_fail[o] = None
    status_now = worst(series[o][-1]["status"] for o in obs_spec)
    return {"series": series, "t0": t0, "t1": t1, "z": z, "weeks_to_fail": weeks_to_fail, "status_now": status_now}


def run(scenario: str = "shim_degrading", warn_factor: float = 2.0, seed: int = 3) -> ScannerReport:
    tr = track(simulate(scenario, seed), OBS, warn_factor)
    series, t0, t1, z, weeks_to_fail = tr["series"], tr["t0"], tr["t1"], tr["z"], tr["weeks_to_fail"]
    status_now = worst(series[o][-1]["status"] for o in OBS)
    recent = [p["status"] for p in series["phantom_linewidth_Hz"][-4:]]
    lw_up = z["phantom_linewidth_Hz"] > 1.96 or recent.count("PASS") == 0
    fo_moved = abs(z["freq_offset_Hz"]) > 1.96
    if lw_up:
        verdict = ("Scanner twin: the phantom linewidth rose between the sessions (z = %.1f; last four phantom scans not PASS). "
                   "The broader follow-up spectrum is at least partly the machine. Calibrate the shim, then repeat PRESS."
                   % z["phantom_linewidth_Hz"])
    elif fo_moved:
        verdict = ("Scanner twin: B0 drifted by %.0f Hz but shim and SNR are stable. Drift is removed by frequency "
                   "referencing; it does not explain the broader MRS lines." % (t1["freq_offset_Hz"][0] - t0["freq_offset_Hz"][0]))
    else:
        verdict = ("Scanner twin: the machine is stable (phantom linewidth z = %.1f). The broader follow-up spectrum is "
                   "subject- or placement-specific: check voxel position, motion and the local shim." % z["phantom_linewidth_Hz"])
    return ScannerReport(series, t0, t1, z, status_now, weeks_to_fail, verdict, lw_up)
