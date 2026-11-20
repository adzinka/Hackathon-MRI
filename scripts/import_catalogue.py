#!/usr/bin/env python3
"""Convert the team's ScanTwin failure catalogue (xlsx) into catalogue/scantwin_catalogue.json.

usage: python scripts/import_catalogue.py [docs/scantwin/ScanTwin failure catalogue and QC thresholds.xlsx]
"""
import json
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/scantwin/ScanTwin failure catalogue and QC thresholds.xlsx"
wb = openpyxl.load_workbook(src, data_only=True)


def sheet(name):
    rows = [r for r in wb[name].iter_rows(values_only=True) if any(c is not None for c in r)]
    head = [str(h).strip() for h in rows[0]]
    return [{h: (None if v is None else str(v).strip()) for h, v in zip(head, r) if h and h != "None"} for r in rows[1:]]


rows = sheet("Failure catalogue")
patch_path = ROOT / "catalogue/changes_v1.1.json"
version = "1.0"
if patch_path.exists():                       # team change list on top of the xlsx (the xlsx itself stays untouched)
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    by_id = {r["ID"]: r for r in rows}
    for fid, upd in patch["updates"].items():
        by_id[fid].update(upd)
    for new in patch["new_rows"]:
        if new["ID"] not in by_id:
            rows.append(new)
    version = patch["version"]
    rules = sheet("Decision rules")
    for r in rules:
        if r.get("Step") == "3":
            r["Notes"] = (r.get("Notes") or "") + " " + patch["decision_rules_step3_addition"]
else:
    rules = sheet("Decision rules")

out = {"source": src.name, "version": version, "failure_catalogue": rows, "decision_rules": rules,
       "scan_types": sheet("Scan types"), "context_factors": sheet("Context factors"), "sources": sheet("Sources")}
(ROOT / "catalogue").mkdir(exist_ok=True)
(ROOT / "catalogue/scantwin_catalogue.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"v{version}: {len(out['failure_catalogue'])} failure modes -> catalogue/scantwin_catalogue.json")
