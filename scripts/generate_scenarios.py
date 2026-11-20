#!/usr/bin/env python3
"""Build scenario_data/: fault scenarios with known truth (all SIMULATED).

usage: python scripts/generate_scenarios.py ../MRI_Hackathon_DataScience_StarterPack_v0.3 [--n 30] [--out scenario_data]

Writes
  catalogue.json                 11 scenarios: parameters, who is to blame, random/systematic, expected signature, next step
  montecarlo.csv                 every simulated scan (n per scenario): QC metrics, fitted Cho/Cr, truth, gate
  summary.csv                    per scenario: mean +- SD of every metric, Cho/Cr bias, CRLB vs real error, gate rates
  fault_model.json               naive-Bayes fault model (means and SDs per scenario and metric)
  confusion_all_metrics.csv      held-out confusion matrix, all metrics (needs individual transients)
  confusion_averaged_only.csv    the same with metrics available from an averaged spectrum (like the StarterPack)
  real_data_diagnosis.json       the model applied to the REAL StarterPack spectra
  thresholds-team-v1.1.json      thresholds-v1.0 + proposal gates (water, drift, transient outliers)
  examples/<scenario>.nii.gz     one averaged spectrum per scenario (+ .json sidecar with the truth)
  longitudinal/                  per hardware fault: weekly phantom CSV, brain packets t0/t1, chain result
  longitudinal_summary.csv       per fault: scanner diagnosis, brain gate status, verdict
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin_evidence import mrs_fit as F, mrs_sim as S, scenarios as SC, longitudinal as L  # noqa: E402
from twin_evidence.validate import load_json  # noqa: E402

AVERAGED_ONLY = ["snr", "fwhm_hz", "water_ratio", "te_ms"]
QUALITY_ALL = ["snr", "fwhm_hz", "water_ratio", "te_ms", "drift_abs_hz", "outlier_frac", "fwhm_gain_aligned"]


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("starterpack", type=Path)
    ap.add_argument("--n", type=int, default=30, help="Monte-Carlo runs per scenario (training); the same again for testing")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "scenario_data")
    a = ap.parse_args()
    root, out = a.starterpack.resolve(), a.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- catalogue
    (out / "catalogue.json").write_text(json.dumps([{
        "key": s.key, "title": s.title, "source": s.source, "blame": s.blame, "kind": s.kind,
        "expected_signature": s.expected, "next_step": s.next_step,
        "parameters": {k: v for k, v in s.acq.to_dict().items() if v != S.Acquisition().to_dict()[k]},
        "origin": "SIMULATED"} for s in SC.CATALOGUE], indent=2), encoding="utf-8")

    # ---- Monte Carlo: training and held-out test sets
    def prog(key, i):
        if i == 0:
            print(f"  {key} …", flush=True)
    print("Monte Carlo (training)")
    train = SC.monte_carlo(root, a.n, seed0=1000, progress=prog)
    print("Monte Carlo (test)")
    test = SC.monte_carlo(root, a.n, seed0=5000, progress=prog)
    write_csv(out / "montecarlo.csv", [{"set": "train", **r} for r in train] + [{"set": "test", **r} for r in test])

    summary = []
    for s in SC.CATALOGUE:
        rr = [r for r in train + test if r["scenario"] == s.key]
        row = {"scenario": s.key, "title": s.title, "blame": s.blame, "kind": s.kind, "n": len(rr)}
        for mname in ("snr", "fwhm_hz", "fwhm_aligned_hz", "water_ratio", "drift_abs_hz", "outlier_frac", "cho_cr", "cho_cr_crlb"):
            v = np.array([r[mname] for r in rr])
            row[f"{mname}_mean"], row[f"{mname}_sd"] = float(v.mean()), float(v.std(ddof=1))
        truth = float(np.mean([r["cho_cr_truth_measurable"] for r in rr]))
        tissue = float(np.mean([r["cho_cr_truth_tissue"] for r in rr]))
        row["cho_cr_truth_tissue"], row["cho_cr_truth_measurable"] = tissue, truth
        row["cho_cr_bias_vs_baseline_biology"] = row["cho_cr_mean"] - 0.689
        row["cho_cr_bias_in_sd"] = row["cho_cr_bias_vs_baseline_biology"] / row["cho_cr_sd"]
        row["crlb_over_real_sd"] = row["cho_cr_crlb_mean"] / row["cho_cr_sd"]
        for st in ("PASS", "WARN", "FAIL"):
            row[f"gate_{st}"] = sum(r["gate"] == st for r in rr) / len(rr)
        summary.append(row)
    write_csv(out / "summary.csv", summary)

    # ---- fault model + confusion matrices (trained on `train`, tested on `test`)
    model = SC.FaultModel.from_rows(train)
    (out / "fault_model.json").write_text(json.dumps({"origin": "SIMULATED (trained on montecarlo.csv set=train)",
                                                      "metrics_all": QUALITY_ALL, "metrics_averaged_only": AVERAGED_ONLY,
                                                      "stats": model.to_dict()}, indent=2), encoding="utf-8")
    acc = {}
    for name, use in (("all_metrics", QUALITY_ALL), ("averaged_only", AVERAGED_ONLY)):
        mat = SC.confusion(model, test, use)
        keys = list(mat)
        with open(out / f"confusion_{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["true \\ predicted"] + keys)
            for k in keys:
                w.writerow([k] + [mat[k][j] for j in keys])
        tot = sum(sum(v.values()) for v in mat.values())
        acc[name] = {"exact": sum(mat[k][k] for k in keys) / tot}
        blame_ok = sum(mat[k][j] for k in keys for j in keys if SC.BY_KEY[k].blame == SC.BY_KEY[j].blame)
        acc[name]["blame"] = blame_ok / tot
        grp_ok = sum(mat[k][j] for k in keys for j in keys if SC.group_of(k) == SC.group_of(j))
        acc[name]["grouped"] = grp_ok / tot

    # ---- the model on the REAL StarterPack spectra
    import nibabel as nib
    real = {}
    for sess in ("baseline", "followup"):
        fid = np.asarray(nib.load(root / f"04_MRS_BIDS/sub-01/ses-{sess}/mrs/sub-01_ses-{sess}_acq-press_nuc-1H_voi-frontal_svs.nii.gz").dataobj).reshape(-1)
        q = F.qc_metrics(fid)
        fit = F.fit(fid)
        m = {"snr": q["snr"], "fwhm_hz": q["fwhm_hz"], "water_ratio": q["water_ratio"], "te_ms": 30.0}
        post = model.posterior(m, AVERAGED_ONLY)
        real[sess] = {"metrics": m, "cho_cr_fit": fit.ratio("Cho"), "naa_cr_fit": fit.ratio("NAA"),
                      "top_scenarios": dict(list(post.items())[:5]), "blame": SC.blame_of(post),
                      "note": "averaged spectrum only: drift, motion and alignment metrics are not available"}
    (out / "real_data_diagnosis.json").write_text(json.dumps(real, indent=2), encoding="utf-8")

    # ---- thresholds v1.1 and one example spectrum per scenario
    tw = root / "05_TwinReady_JSON_CSV"
    th11 = L.thresholds_v11(load_json(tw / "thresholds-v1.0.json"))
    (out / "thresholds-team-v1.1.json").write_text(json.dumps(th11, indent=2), encoding="utf-8")
    for s in SC.CATALOGUE:
        sim = S.simulate(s.acq, root, 42)
        S.write_nifti_mrs(sim.average(), out / "examples" / f"{s.key}.nii.gz",
                          {"origin": "SIMULATED", "scenario": s.key, "title": s.title, "truth": sim.truth,
                           "acquisition": sim.acq.to_dict(), "convention": "StarterPack v0.3 README; averaged over transients"})

    # ---- time: hardware faults over 26 weeks, full chain
    print("Longitudinal chains")
    schema = load_json(tw / "evidence_packet_v1.0.schema.json")
    template = load_json(tw / "packets" / "E-t1-cho.json")
    chains = [L.run_fault(f, root, out, th11, schema, template, seed=1) for f in L.FAULTS]
    chains.append(L.run_fault("stable", root, out, th11, schema, template, seed=1, biology_t1={"cho_factor": 1.2}))
    lsum = []
    for c in chains:
        tag = c["fault"]
        write_csv(out / "longitudinal" / tag / "phantom_weekly.csv",
                  [{"origin": "SIMULATED", **w, **{f"{o}_status": c["track"]["series"][o][i]["status"] for o in L.PHANTOM_OBS},
                    **{f"{o}_twin_x": c["track"]["series"][o][i]["x"] for o in L.PHANTOM_OBS}} for i, w in enumerate(c["weeks"])])
        b0, b1 = c["brain"]
        lsum.append({"fault": tag, "title": c["title"], "scanner_status_week26": c["track"]["status_now"],
                     "diagnosed_class": c["diagnosis"]["fault_class"], "expected_class": c["expected_class"], "correct": c["correct"],
                     "why": c["diagnosis"]["why"],
                     **{f"z_{k}": v for k, v in c["diagnosis"]["z"].items()},
                     **{f"weeks_to_fail_{k}": v for k, v in c["track"]["weeks_to_fail"].items()},
                     "brain_t1_status": b1["status"], "brain_t1_action": b1["action"], "brain_t1_value": b1["value"],
                     "brain_t1_truth_tissue": b1["truth_tissue"], "brain_t1_K": b1["K"],
                     "brain_t1_reasons": " | ".join(b1["reasons"]), "verdict": c["verdict"]})
        (out / "longitudinal" / tag / "chain.json").write_text(json.dumps(
            {k: v for k, v in c.items() if k not in ("weeks", "track")} | {"z": c["track"]["z"], "weeks_to_fail": c["track"]["weeks_to_fail"]},
            indent=2, default=lambda o: o if not isinstance(o, (np.floating, np.bool_)) else o.item()), encoding="utf-8")
    write_csv(out / "longitudinal_summary.csv", lsum)

    (out / "run_info.json").write_text(json.dumps({"n_per_scenario": a.n, "accuracy": acc, "seconds": round(time.time() - t0, 1),
                                                   "origin": "SIMULATED"}, indent=2), encoding="utf-8")
    print(json.dumps(acc, indent=2))
    print(f"wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
