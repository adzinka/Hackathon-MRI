#!/usr/bin/env python3
"""Check the ScanTwin catalogue thresholds against the physics simulator (all SIMULATED).

For each built MRS gate: add its fault at increasing strength, measure like real data, and record how far Cho/Cr
moves from the baseline biology (0.689). The simulation-implied limit is the metric value at which the apparent
change exceeds one baseline SD (WARN level) or two (FAIL level). This is the method described in
docs/scantwin/ScanTwin a digital twin of the MRI scanner.pdf ("add one fault at increasing strength").

usage: python scripts/sweep_faults.py ../MRI_Hackathon_DataScience_StarterPack_v0.3 [--seeds 8]
writes scenario_data/sweeps.csv and scenario_data/threshold_check.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin_evidence import mrs_sim as S, scenarios as SC  # noqa: E402
from twin_evidence.gates import MRS_GATES  # noqa: E402

BASE = S.Acquisition()
TRUTH = 0.689
SWEEPS = {   # gate id: (parameter label, strengths, acquisition builder)
    "F12": ("extra shim broadening / Hz", [0, 3, 6, 9, 12, 16], lambda v: replace(BASE, shim_hz=v)),
    "F13": ("coil noise factor", [1.0, 1.25, 1.5, 2.0, 2.5, 3.2], lambda v: replace(BASE, coil_noise_factor=v)),
    "F08": ("drift over the scan / Hz", [0, 4, 8, 12, 16, 24], lambda v: replace(BASE, drift_hz=v)),
    "F15": ("residual water amplitude / Cr", [0, 0.5, 1, 2, 4, 8], lambda v: replace(BASE, water_residual=v)),
    "F23": ("fraction of transients moved", [0, 0.05, 0.1, 0.2, 0.3, 0.45], lambda v: replace(BASE, motion_frac=v, motion_hz=10.0)),
    "F20": ("voxel offset (inferior) / mm", [0, 1, 2, 3, 4, 6], lambda v: replace(BASE, voxel_offset_mm=(0.0, 0.0, -v))),
}


def crossing(x, y, level):
    """First metric value x where y crosses level (linear interpolation); None if never."""
    for i in range(1, len(x)):
        if y[i - 1] < level <= y[i]:
            return x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("starterpack", type=Path)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "scenario_data")
    a = ap.parse_args()
    root = a.starterpack.resolve()
    a.out.mkdir(exist_ok=True)
    gates = {g.id: g for g in MRS_GATES}

    rows, check = [], []
    clean = [SC.measure(S.simulate(BASE, root, 9000 + s))["cho_cr"] for s in range(2 * a.seeds)]
    sd0 = float(np.std(clean, ddof=1))
    for gid, (plabel, strengths, build) in SWEEPS.items():
        g = gates[gid]
        pts = []
        for v in strengths:
            ms = [SC.measure(S.simulate(build(v), root, 7000 + s)) for s in range(a.seeds)]
            metric = float(np.mean([m[g.metric] for m in ms]))
            cho = np.array([m["cho_cr"] for m in ms])
            err = cho - TRUTH
            rmse = float(np.sqrt(np.mean(err ** 2)))
            bias = float(err.mean())
            status = g.status(metric)
            rows.append({"gate": gid, "parameter": plabel, "strength": v, "metric": g.metric, "metric_mean": round(metric, 3),
                         "cho_cr_mean": round(float(cho.mean()), 4), "bias": round(bias, 4), "rmse": round(rmse, 4),
                         "bias_in_sd": round(bias / sd0, 2), "gate_status_at_mean": status, "origin": "SIMULATED"})
            pts.append((metric, abs(bias) / sd0, rmse / sd0))
            print(f"{gid} {plabel} {v}: {g.metric}={metric:.2f} bias={bias:+.3f} ({bias / sd0:+.1f} SD) gate={status}", flush=True)
        pts.sort(key=lambda p: p[0] if g.warn and g.warn[0] == ">" else -p[0])
        xs = [p[0] for p in pts]
        sim_warn = crossing(xs, [p[1] for p in pts], 1.0)
        sim_fail = crossing(xs, [p[1] for p in pts], 2.0)
        cat_warn = g.warn[1] if g.warn else None
        if sim_warn is None:
            verdict = "OK: no harmful bias within the tested range (the gate protects against noise, not bias)"
        elif cat_warn is None:
            verdict = "no numeric WARN limit to compare"
        else:
            lenient = (sim_warn < cat_warn) if g.warn[0] == ">" else (sim_warn > cat_warn)
            ratio = (cat_warn / sim_warn) if g.warn[0] == ">" else (sim_warn / cat_warn)
            verdict = (f"TOO LENIENT: bias reaches 1 SD at {sim_warn:.2f}, the catalogue warns only at {cat_warn}"
                       if lenient and ratio > 1.5 else "OK: consistent with the simulation" if not lenient or ratio <= 1.5 else "")
        check.append({"gate": gid, "label": g.label, "metric": g.metric, "catalogue_warn": cat_warn,
                      "catalogue_fail": g.fail[1] if g.fail else None,
                      "sim_metric_at_1sd_bias": None if sim_warn is None else round(sim_warn, 2),
                      "sim_metric_at_2sd_bias": None if sim_fail is None else round(sim_fail, 2),
                      "baseline_sd": round(sd0, 4), "verdict": verdict})
    for name, data in (("sweeps.csv", rows), ("threshold_check.csv", check)):
        with open(a.out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, list(data[0]))
            w.writeheader()
            w.writerows(data)
    for c in check:
        print(c["gate"], c["label"], "|", c["verdict"])


if __name__ == "__main__":
    main()
