#!/usr/bin/env python3
"""Build the scanner_data/ folder for the scanner / metrology experts of the team.

usage: python scripts/export_scanner_data.py ../MRI_Hackathon_DataScience_StarterPack_v0.3 [--out scanner_data]

What it writes:
  real_session_qc.csv        REAL: the scanner-related QC of the two StarterPack sessions (from the packets and
                             recomputed from the raw NIfTI-MRS files)
  real_acquisition.json      REAL: acquisition context, field vs frequency, calibration and system-QC references
  phantom_weekly_<scenario>.csv   SIMULATED: weekly phantom record per fault scenario, with the scanner-twin state
  scanner_thresholds.json    team PROPOSAL (phantom) + StarterPack thresholds-v1.0 (scanner-related gates)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin_evidence import anatomy, scanner  # noqa: E402
from twin_evidence.validate import load_json  # noqa: E402

BASELINE_DATE = date(2026, 5, 20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("starterpack", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "scanner_data")
    a = ap.parse_args()
    root, out = a.starterpack.resolve(), a.out
    out.mkdir(parents=True, exist_ok=True)
    tw = root / "05_TwinReady_JSON_CSV"
    pk = {n: load_json(tw / "packets" / f"{n}.json") for n in ("E-t0-cho", "E-t1-cho", "E-t1-chobad", "E-t0-roi", "E-t1-roi")}

    # ---- REAL: per-session scanner QC ----
    rows = []
    for sess, cho, roi in (("baseline", "E-t0-cho", "E-t0-roi"), ("followup", "E-t1-cho", "E-t1-roi")):
        p, r = pk[cho], pk[roi]
        qc = {c["name"]: c for c in p["qc_checks"]}
        rq = {c["name"]: c for c in r["qc_checks"]}
        raw = anatomy.file_qc(root, sess)
        rows.append({
            "session": sess, "date": str(BASELINE_DATE if sess == "baseline" else date(2026, 11, 20)),
            "origin": p["origin"],
            "mrs_snr_packet": qc["mrs_snr"]["value"], "mrs_snr_from_raw": round(raw["snr"], 2),
            "mrs_linewidth_Hz_packet": qc["mrs_linewidth_Hz"]["value"], "mrs_linewidth_Hz_from_raw": round(raw["fwhm_hz"], 2),
            "naa_peak_ppm_from_raw": round(raw["naa_ppm"], 3),
            "mrs_registration_rms_mm": qc["registration_rms_mm"]["value"],
            "roi_registration_rms_mm": rq.get("registration_rms_mm", {}).get("value"),
            "system_qc_id": p.get("system_qc_id"), "calibration_id": p.get("calibration_id"),
            "protocol_match": p.get("protocol_match"), "csde_mm_NAA_Cho": p["spatial_reference"].get("csde_mm_NAA_Cho"),
            "qc_status_packet": p["qc_status"],
        })
    bad = pk["E-t1-chobad"]
    rows.append({"session": "followup (rejected packet E-t1-chobad)", "date": "2026-11-20", "origin": bad["origin"],
                 **{f"{c['name']}_packet": c["value"] for c in bad["qc_checks"]}, "system_qc_id": bad.get("system_qc_id"),
                 "qc_status_packet": bad["qc_status"]})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "real_session_qc.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)

    side = load_json(root / "04_MRS_BIDS/sub-01/ses-followup/mrs/sub-01_ses-followup_acq-press_nuc-1H_voi-frontal_svs.json")
    acq = {
        "origin": "REAL StarterPack v0.3 (synthetic teaching data, SIMULATED origin)",
        "mrs_acquisition": pk["E-t1-cho"]["acquisition_context"],
        "mrs_sidecar": {k: side.get(k) for k in ("Manufacturer", "MagneticFieldStrength", "SpectrometerFrequency", "PulseSequenceType",
                                                   "EchoTime", "RepetitionTime", "NumberOfTransients", "WaterSuppression")},
        "field_note": "Nominal field 3.0 T is a label; 123.25 MHz / 42.577 MHz/T = 2.895 T is the measured field (Dossier §5.4).",
        "voi": pk["E-t1-cho"]["spatial_reference"],
        "roi_registration": pk["E-t1-roi"]["spatial_reference"],
        "references": {"system_qc_id": pk["E-t1-cho"].get("system_qc_id"), "calibration_id": pk["E-t1-cho"].get("calibration_id"),
                       "note": "These ids point to a system QC and a phantom calibration that are NOT in the StarterPack."},
        "spectrum_convention": "s(t) = sum a_j exp(-pi LW t) exp(+i 2 pi f_j t); f_j = (delta_j - 4.65 ppm) x 123.25 MHz; "
                               "dwell 0.5 ms, 1024 points; SNR = NAA peak height / SD of real spectrum in -3..0 ppm",
    }
    (out / "real_acquisition.json").write_text(json.dumps(acq, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- SIMULATED: weekly phantom record per scenario, with scanner-twin state ----
    for scen in scanner.SCENARIOS:
        rep = scanner.run(scen)
        with open(out / f"phantom_weekly_{scen}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            hdr = ["week", "date", "origin"]
            for o in scanner.OBS:
                hdr += [f"{o}", f"{o}_status", f"{o}_twin_x", f"{o}_twin_sd"]
            w.writerow(hdr)
            for i in range(scanner.WEEKS + 1):
                row = [i, str(BASELINE_DATE + timedelta(weeks=i)), "SIMULATED"]
                for o in scanner.OBS:
                    s = rep.series[o][i]
                    row += [round(s["value"], 3), s["status"], round(s["x"], 3), round(s["sd"], 3)]
                w.writerow(row)

    th = load_json(tw / "thresholds-v1.0.json")
    scanner_gates = [g for g in th["gates"] if g["name"] in ("mrs_snr", "mrs_linewidth_Hz", "registration_rms_mm", "system_qc",
                                                              "protocol_match", "roi_u_consistency")]
    (out / "scanner_thresholds.json").write_text(json.dumps({
        "phantom_gates_team_proposal": {o: {"label": v[0], "unit": v[1], "per_scan_sd": v[2], "drift_sd_per_6_months": v[3], **v[4]}
                                        for o, v in scanner.OBS.items()},
        "phantom_note": "PROPOSAL for a spherical water phantom at 3 T, chosen by the team. Please replace with site values.",
        "starterpack_thresholds_v1_0_scanner_related": scanner_gates,
        "twin_policy": th.get("twin_policy"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
