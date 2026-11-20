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


out = {"source": src.name, "failure_catalogue": sheet("Failure catalogue"), "decision_rules": sheet("Decision rules"),
       "scan_types": sheet("Scan types"), "context_factors": sheet("Context factors"), "sources": sheet("Sources")}
(ROOT / "catalogue").mkdir(exist_ok=True)
(ROOT / "catalogue/scantwin_catalogue.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"{len(out['failure_catalogue'])} failure modes -> catalogue/scantwin_catalogue.json")
