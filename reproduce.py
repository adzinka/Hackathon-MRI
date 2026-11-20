#!/usr/bin/env python3
"""Gate 1: re-run the worked reference update (Dossier §3.10 / §7.5) from the StarterPack v0.3 packets.

usage: python reproduce.py PATH/TO/MRI_Hackathon_DataScience_StarterPack_v0.3 [--out out]
       python reproduce.py STARTERPACK --packet my_packet.json   # judge one unseen packet (Gate 3)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twin_evidence.validate import evaluate, load_json, load_manifest
from twin_evidence.twin import Twin


def load_pack(root: Path):
    tw = root / "05_TwinReady_JSON_CSV"
    return (load_json(tw / "thresholds-v1.0.json"), load_json(tw / "evidence_packet_v1.0.schema.json"),
            load_manifest(root / "manifest_sha256.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("starterpack", type=Path)
    ap.add_argument("--packet", type=Path, nargs="*", help="extra (unseen) follow-up packets to judge after the baseline")
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()
    root = a.starterpack.resolve()
    thresholds, schema, manifest = load_pack(root)
    pdir = root / "05_TwinReady_JSON_CSV" / "packets"
    order = ["E-t0-roi", "E-t0-naa", "E-t0-cho", "E-t1-roi", "E-t1-naa", "E-t1-cho", "E-t1-chobad"]
    packets = [load_json(pdir / f"{n}.json") for n in order]
    if a.packet:
        packets = packets[:3] + [load_json(p) for p in a.packet]

    twin, report = Twin(), []
    for p in packets:
        ev = evaluate(p, thresholds, schema, manifest, root)
        e = twin.ingest(p, ev)
        report.append({"evaluation": ev.to_dict(), "twin_event": e.to_dict()})
        line = f"{str(p.get('packet_id')):12s} {str(p.get('observable')):10s} stored={str(ev.stored_status):4s} -> {ev.status:4s}  {e.action:20s}"
        if e.kind == "update":
            line += (f" z={e.z_change:+.2f}" if e.z_change is not None else "") + \
                    f" d2={e.d2:.2f}{' FLAG' if e.d2_flag else ''} K={e.K:.3f} x+={e.posterior['x']:.4g} ± {e.posterior['sd']:.3g}"
        elif e.kind == "init":
            line += f" init x={e.posterior['x']:.4g} ± {e.posterior['sd']:.3g}"
        print(line)
        for f in ev.findings:
            if f.status != "PASS":
                print(f"      {f.status:4s} [{f.owner}] {f.rule}: {f.reason}")
        for n in ev.notes:
            print(f"      note: {n}")

    if "Cho_Cr" in twin.state:
        print("\nNext measurement for Cho/Cr (Dossier §7.7, Eq. 7.3):")
        print(f"  repeat PRESS after re-shim  (sigma_m 0.050, PASS): sd {twin.state['Cho_Cr'].sd:.3f} -> {twin.next_measurement('Cho_Cr', 0.050, 1):.3f}")
        print(f"  double averages, no re-shim (sigma_m 0.042, WARN): sd {twin.state['Cho_Cr'].sd:.3f} -> {twin.next_measurement('Cho_Cr', 0.042, 2):.3f}")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "reference_update.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {a.out / 'reference_update.json'}")


if __name__ == "__main__":
    main()
