#!/usr/bin/env python3
"""Convert a raw single-voxel NIfTI-MRS file into ScanTwin evidence packets, judge it, and (optionally) add
SIMULATED follow-up sessions made from the same real transients, then run the subject twin.

usage:
  python -I scripts/ingest_mrs.py RAW.nii.gz --out external_data/NAME --subject sub-ext01 [--assume-system-qc]
                                 [--followups none,shim,drift,noise,water]

The raw file is read but never copied; only a de-identified, coil-combined, averaged derivative is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin_evidence import ingest as I  # noqa: E402
from twin_evidence.twin import Twin  # noqa: E402
from twin_evidence.validate import evaluate, load_json  # noqa: E402

# strengths are given in Hz at the scan's own field; chosen to cross the 3 T-equivalent WARN limits at 7 T
FOLLOWUPS = {"none": ("none", 0.0, "repeat scan, healthy scanner, unchanged subject"),
             "shim": ("shim", 22.0, "poor shim: extra Gaussian broadening 22 Hz"),
             "drift": ("drift", 25.0, "frequency drift 25 Hz over the scan"),
             "noise": ("noise", 8.0, "weak receive coil: noise x8"),
             "water": ("water", 5.0, "incomplete water suppression: extra residual water 5x NAA")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subject", default="sub-ext01")
    ap.add_argument("--assume-system-qc", action="store_true", help="mark the site QC as ASSUMED (gives a WARN finding)")
    ap.add_argument("--followups", default="none,shim,drift,noise,water")
    ap.add_argument("--starterpack", type=Path, default=Path(__file__).resolve().parents[2] / "MRI_Hackathon_DataScience_StarterPack_v0.3")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tw = a.starterpack.resolve() / "05_TwinReady_JSON_CSV"
    thresholds, schema = load_json(tw / "thresholds-v1.0.json"), load_json(tw / "evidence_packet_v1.0.schema.json")
    sysqc = "ASSUMED:external-dataset" if a.assume_system_qc else None

    ing = I.read(a.raw)
    sessions = [("t0", "ses-baseline", 0.0, "MEASURED", ing, "real scan")]
    for i, key in enumerate([k for k in a.followups.split(",") if k], start=1):
        fault, strength, desc = FOLLOWUPS[key]
        sessions.append((f"t1-{key}", f"ses-sim{key}", 6.0, "SIMULATED", I.degrade(ing, fault, strength, seed=100 + i), desc))

    manifest, rows, packets = {}, [], {}
    for tag, ses, months, origin, x, desc in sessions:
        m = I.measure(x)
        g = I.gate(m, x.nu0)
        rel = f"derivatives/{a.subject}_{ses}_svs.nii.gz"
        sha = I.write_derivative(x, out / rel)
        manifest[rel] = sha
        pid = ("E-t0-" if tag == "t0" else "E-t1-") + ("ext_" + tag.replace("-", "_")).lower()
        parents = [f"{ing.source_name} (sha256 {ing.source_sha256})"]
        if origin == "SIMULATED":
            parents.append(f"derivatives/{a.subject}_ses-baseline_svs.nii.gz (real baseline, fault injected)")
        p = I.build_packet(x, m, g, packet_id=pid, subject_id=a.subject, session_id=ses, months=months, rel_path=rel,
                           sha=sha, origin=origin, system_qc_id=sysqc, parents=parents,
                           limit_note="" if origin == "MEASURED" else f"SIMULATED follow-up: {desc}")
        packets[tag] = (p, m, g, desc)
        (out / f"{pid}.json").write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out / "manifest_sha256.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "sha256"])
        w.writerows(manifest.items())

    base_p = packets["t0"][0]
    for tag, (p, m, g, desc) in packets.items():
        twin = Twin()
        ev0 = evaluate(base_p, thresholds, schema, manifest, out)
        twin.ingest(base_p, ev0)
        ev = evaluate(p, thresholds, schema, manifest, out) if tag != "t0" else ev0
        e = twin.ingest(p, ev) if tag != "t0" else None
        rows.append({"session": tag, "origin": p["origin"], "what": desc, "snr": round(m["snr"], 1),
                     "linewidth_ppm": round(m["fwhm_ppm"], 4), "water_ratio": round(m["water_ratio"], 2),
                     "drift_hz": None if m["drift_abs_hz"] is None else round(m["drift_abs_hz"], 2),
                     "outliers_pct": None if m["outlier_pct"] is None else round(m["outlier_pct"], 1),
                     "cho_cr": round(m["cho_cr"], 3), "u": p["uncertainty"]["value"],
                     "scan_gates": g["status"], "deciding": " ".join(d["id"] for d in g["deciding"]),
                     "packet_gate": ev.status, "twin_action": e.action if e else "init",
                     "z_change": None if (e is None or e.z_change is None) else round(e.z_change, 2),
                     "posterior": None if (e is None or e.posterior is None) else f"{e.posterior['x']:.3f} ± {e.posterior['sd']:.3f}",
                     "reasons": " | ".join(f"{f.rule}: {f.reason}" for f in ev.findings if f.status != "PASS")})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "conversion_report.json").write_text(json.dumps({
        "source_file": ing.source_name, "source_sha256": ing.source_sha256, "source_shape": list(ing.shape),
        "identifying_fields_removed": ing.removed, "steps": ing.steps, "nu0_MHz": ing.nu0, "dwell_s": ing.dwell,
        "system_qc": sysqc or "missing", "sessions": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(f"{r['session']:9s} {r['origin']:9s} SNR {r['snr']:6.1f} LW {r['linewidth_ppm']:.4f} ppm water {r['water_ratio']:5.2f} "
              f"drift {r['drift_hz']} out {r['outliers_pct']} Cho/Cr {r['cho_cr']:.3f}±{r['u']:.3f} | scan {r['scan_gates']:4s} "
              f"[{r['deciding']}] packet {r['packet_gate']:4s} -> {r['twin_action']} z={r['z_change']} {r['posterior']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
