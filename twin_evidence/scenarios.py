"""Fault scenarios with known truth: simulate -> process like real data -> QC metrics -> gate -> diagnosis.

All outputs are SIMULATED. The catalogue covers the seven error sources of MRI_MRS_Essential_Errors_Explained.md,
three hardware fault classes and one true biological change. Each scenario says who is to blame
(patient / machine / operator) and whether the error is random or systematic.

The fault classifier is a transparent naive-Bayes model: for every scenario the Monte-Carlo runs give a mean and SD
of each QC metric; a new scan is scored by how well its metrics fit each scenario. It is trained on simulations
only, so it can only recognise faults we modelled. It reports probabilities, never a single "root cause".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import mrs_fit as F
from . import mrs_sim as S

# ------------------------------------------------------------------------------------------------- catalogue
@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    source: str            # where the idea comes from
    blame: str             # patient / machine / operator / none
    kind: str              # random / systematic / none
    acq: S.Acquisition
    expected: str          # expected signature in plain words
    next_step: str         # cheapest check or fix


BASE = S.Acquisition()
CATALOGUE = [
    Scenario("clean", "Clean scan (reference)", "StarterPack baseline", "none", "none", BASE,
             "SNR ~31, linewidth ~8.7 Hz, no drift, no outliers", "none"),
    Scenario("low_snr_coil", "Low SNR: receive coil / RF chain", "errors file §1; hardware", "machine", "random",
             replace(BASE, coil_noise_factor=1.5), "SNR down, linewidth normal",
             "coil element test with the phantom; check coil connection"),
    Scenario("poor_shim", "B0 inhomogeneity / poor shim", "errors file §2; Dossier §5.4", "machine", "systematic",
             replace(BASE, shim_hz=7.0), "linewidth up AND SNR down; alignment does not help",
             "re-shim (2 min) and repeat PRESS"),
    Scenario("motion", "Patient motion", "errors file §3", "patient", "systematic",
             replace(BASE, motion_frac=0.2, motion_hz=10.0), "outlier transients; linewidth up, recovers after alignment",
             "repeat with head padding; reject outlier transients"),
    Scenario("freq_drift", "Frequency drift (B0 drift)", "errors file §4; Dossier §5.4", "machine", "systematic",
             replace(BASE, drift_hz=12.0), "linear drift across transients; linewidth up, recovers after alignment",
             "frequency alignment of transients; check cryostat / room temperature"),
    Scenario("voxel_misplacement", "Voxel misplacement / tissue mixing", "errors file §5; Dossier §5.5", "operator", "systematic",
             replace(BASE, voxel_offset_mm=(0.0, 0.0, -4.0)), "spectral quality normal, ratio shifted",
             "check voxel overlap and tissue fractions; re-place voxel"),
    Scenario("water_suppression", "Incomplete water suppression", "errors file §6", "machine", "systematic",
             replace(BASE, water_residual=3.0), "large residual water at 4.65 ppm",
             "recalibrate CHESS flip angles; repeat"),
    Scenario("protocol_te", "Protocol mismatch (TE 144 ms)", "errors file §7", "operator", "systematic",
             replace(BASE, te_ms=144.0), "metadata differ; SNR down; ratio shifted by T2 weighting",
             "repeat with the baseline protocol (TE 30 ms)"),
    Scenario("b1_transmit", "RF transmit / B1 miscalibration", "hardware class; Dossier §5.2", "machine", "systematic",
             replace(BASE, b1_error=-0.2), "SNR down, more residual water, ratio unchanged",
             "transmitter (flip-angle) calibration"),
    Scenario("gradient", "Gradient scale error (+4 %)", "hardware class; Dossier §5.5", "machine", "systematic",
             replace(BASE, gradient_error=0.04), "spectrum looks normal; only a phantom geometry check sees it",
             "gradient calibration with a grid phantom"),
    Scenario("true_change", "True biological change (Cho +20 %)", "the patient", "patient", "none",
             replace(BASE, cho_factor=1.2), "spectral quality normal, Cho/Cr up", "confirm with a repeat scan"),
]
BY_KEY = {s.key: s for s in CATALOGUE}

# metrics used for diagnosis; `needs_transients` marks those unavailable on averaged-only data (like the StarterPack)
METRICS = {
    "snr": dict(needs_transients=False), "fwhm_hz": dict(needs_transients=False),
    "water_ratio": dict(needs_transients=False), "cho_cr": dict(needs_transients=False),
    "te_ms": dict(needs_transients=False),
    "drift_abs_hz": dict(needs_transients=True), "outlier_frac": dict(needs_transients=True),
    "fwhm_gain_aligned": dict(needs_transients=True),
}

# team-proposal gates for the new metrics, on top of thresholds-v1.0 (mrs_snr, mrs_linewidth_Hz)
NEW_GATES = [
    {"name": "mrs_water_ratio", "unit": "1", "owner": "QE", "warn_above": 1.0, "fail_above": 3.0, "applies_to": ["MRS"],
     "note": "PROPOSAL: residual water peak / NAA peak; clean StarterPack spectra give ~0.15-0.25 (noise floor)"},
    {"name": "mrs_drift_Hz", "unit": "Hz", "owner": "PM", "warn_above": 4.0, "fail_above": 12.0, "applies_to": ["MRS"],
     "note": "PROPOSAL: uncorrected frequency drift over the acquisition; 12 Hz = 0.1 ppm (Dossier §5.4)"},
    {"name": "mrs_transient_outliers", "unit": "1", "owner": "QE", "warn_above": 0.1, "fail_above": 0.3, "applies_to": ["MRS"],
     "note": "PROPOSAL: fraction of transients with a frequency jump or low correlation (motion)"},
]


# ------------------------------------------------------------------------------------------------- one run
def measure(sim: S.Simulated) -> dict:
    """Process a simulated acquisition exactly like real data: average, QC, fit; plus transient QC."""
    fid = sim.average()
    q = F.qc_metrics(fid)
    fit = F.fit(fid)
    cho, cho_sd = fit.ratio("Cho")
    naa, naa_sd = fit.ratio("NAA")
    tq = S.transient_qc(sim.transients)
    fwhm_al = F.qc_metrics(S.align_and_average(sim.transients))["fwhm_hz"]
    return {
        "snr": q["snr"], "fwhm_hz": q["fwhm_hz"], "water_ratio": q["water_ratio"], "te_ms": sim.acq.te_ms,
        "cho_cr": cho, "cho_cr_crlb": cho_sd, "naa_cr": naa, "naa_cr_crlb": naa_sd,
        "fit_lw_hz": fit.lw_hz, "fit_shift_hz": fit.shift_hz,
        "drift_abs_hz": abs(tq["drift_hz"]), "outlier_frac": tq["transient_outlier_frac"],
        "fwhm_aligned_hz": fwhm_al, "fwhm_gain_aligned": q["fwhm_hz"] - fwhm_al,
        "cho_cr_truth_tissue": sim.truth["Cho_Cr_tissue"], "cho_cr_truth_measurable": sim.truth["Cho_Cr_measurable"],
        "gm_fraction": sim.truth["tissue"]["GM"],
    }


def gate(m: dict, ref_te_ms: float = S.TE_REF_MS) -> dict:
    """PASS/WARN/FAIL per check, thresholds-v1.0 for SNR and linewidth plus the team-proposal gates."""
    checks = {
        "mrs_snr": "FAIL" if m["snr"] < 10 else "WARN" if m["snr"] < 25 else "PASS",
        "mrs_linewidth_Hz": "FAIL" if m["fwhm_hz"] > 20 else "WARN" if m["fwhm_hz"] > 10 else "PASS",
        "mrs_water_ratio": "FAIL" if m["water_ratio"] > 3 else "WARN" if m["water_ratio"] > 1 else "PASS",
        "protocol_match": "WARN" if abs(m["te_ms"] - ref_te_ms) > 1e-6 else "PASS",
    }
    if m.get("drift_abs_hz") is not None:
        checks["mrs_drift_Hz"] = "FAIL" if m["drift_abs_hz"] > 12 else "WARN" if m["drift_abs_hz"] > 4 else "PASS"
    if m.get("outlier_frac") is not None:
        checks["mrs_transient_outliers"] = "FAIL" if m["outlier_frac"] > 0.3 else "WARN" if m["outlier_frac"] > 0.1 else "PASS"
    rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
    status = max(checks.values(), key=rank.get)
    return {"status": status, "checks": checks}


def run(key: str, root: Path | None, seed: int = 0) -> dict:
    sc = BY_KEY[key]
    sim = S.simulate(sc.acq, root, seed)
    m = measure(sim)
    g = gate(m)
    return {"scenario": key, "seed": seed, **m, "gate": g["status"], **{f"check_{k}": v for k, v in g["checks"].items()}}


def monte_carlo(root: Path | None, n: int = 30, keys=None, seed0: int = 1000, progress=None) -> list[dict]:
    rows = []
    for key in keys or BY_KEY:
        for i in range(n):
            rows.append(run(key, root, seed0 + i))
            if progress:
                progress(key, i)
    return rows


# ------------------------------------------------------------------------------------------------- classifier
@dataclass
class FaultModel:
    stats: dict            # scenario -> metric -> (mean, sd)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "FaultModel":
        stats = {}
        for key in {r["scenario"] for r in rows}:
            rr = [r for r in rows if r["scenario"] == key]
            stats[key] = {}
            for mname in METRICS:
                v = np.array([r[mname] for r in rr], float)
                sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
                floor = {"te_ms": 1.0, "cho_cr": 0.01, "outlier_frac": 0.02, "drift_abs_hz": 0.5,
                         "fwhm_gain_aligned": 0.3, "water_ratio": 0.03, "snr": 0.5, "fwhm_hz": 0.2}[mname]
                stats[key][mname] = (float(v.mean()), max(sd, floor))
        return cls(stats)

    def posterior(self, m: dict, use=None) -> dict:
        """P(scenario | metrics) with a flat prior; metrics missing from `m` are skipped."""
        use = [k for k in (use or METRICS) if m.get(k) is not None]
        logp = {}
        for key, st in self.stats.items():
            lp = 0.0
            for k in use:
                mu, sd = st[k]
                z = (m[k] - mu) / sd
                lp += -0.5 * min(z * z, 400.0) - math.log(sd)
            logp[key] = lp
        mx = max(logp.values())
        w = {k: math.exp(v - mx) for k, v in logp.items()}
        s = sum(w.values())
        return dict(sorted(((k, v / s) for k, v in w.items()), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict:
        return {k: {m: {"mean": a, "sd": b} for m, (a, b) in v.items()} for k, v in self.stats.items()}


# scenarios that leave the averaged spectrum unchanged: only a phantom check or tissue fractions can tell them apart
SPECTRALLY_SILENT = {"clean", "true_change", "voxel_misplacement", "gradient"}


def group_of(key: str) -> str:
    return "spectrally_silent" if key in SPECTRALLY_SILENT else key


def confusion(model: FaultModel, rows: list[dict], use=None) -> dict:
    keys = list(model.stats)
    mat = {a: {b: 0 for b in keys} for a in keys}
    for r in rows:
        post = model.posterior(r, use)
        mat[r["scenario"]][next(iter(post))] += 1
    return mat


def blame_of(post: dict) -> dict:
    out = {}
    for k, p in post.items():
        b = BY_KEY[k].blame
        out[b] = out.get(b, 0.0) + p
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
