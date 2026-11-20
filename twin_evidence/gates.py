"""ScanTwin quality gates: the numeric "Built" rows of the team's failure catalogue as machine-readable gates.

Source: docs/scantwin/ScanTwin failure catalogue and QC thresholds.xlsx (imported to catalogue/scantwin_catalogue.json).
Each gate keeps the catalogue ID, so every verdict can name the row it came from and show that row's fix.

Two catalogue rows are applied as NOTES, not as gates, by team decision (see docs/scantwin/REVIEW.md):
  F19 chemical-shift displacement: PRESS at 3 T always gives ~12 %, so every scan, baseline included, would WARN;
      it is common to both sessions and mostly cancels in the change.
  F22 Cr reference: the StarterPack has ratios only, so every MRS scan would WARN; we show it as a standing limitation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}
CATALOGUE = Path(__file__).resolve().parents[1] / "catalogue" / "scantwin_catalogue.json"


@dataclass(frozen=True)
class Gate:
    id: str
    metric: str           # key in the measured-metrics dict
    label: str
    unit: str
    pass_if: str
    warn: tuple | None    # (op, value)  op in {">", "<"}
    fail: tuple | None
    applies: bool = True  # False = shown as a note, never changes the verdict

    def status(self, v) -> str:
        if v is None:
            return "N/A"                          # catalogue rule 3: kept as explicit null + reason, never imputed
        def hit(rule):
            return rule is not None and (v > rule[1] if rule[0] == ">" else v < rule[1])
        return "FAIL" if hit(self.fail) else "WARN" if hit(self.warn) else "PASS"


# ¹H MRS gates (catalogue rows marked "Built"); values exactly as in the xlsx
MRS_GATES = [
    Gate("F12", "fwhm_hz", "Linewidth (shim)", "Hz", "≤ 10", (">", 10.0), (">", 20.0)),
    Gate("F13", "snr", "SNR", "1", "≥ 25", ("<", 25.0), ("<", 10.0)),
    Gate("F14", "cho_crlb_pct", "Cho fit CRLB", "%", "≤ 20", (">", 20.0), (">", 50.0)),
    Gate("F15", "water_ratio", "Residual water / NAA", "1", "≤ 5", (">", 5.0), (">", 20.0)),
    Gate("F08", "drift_abs_hz", "Frequency drift", "Hz", "≤ 6", (">", 6.0), (">", 12.0)),
    Gate("F23", "outlier_pct", "Transients rejected (motion)", "%", "≤ 5", (">", 5.0), (">", 20.0)),
    Gate("F20", "voxel_shift_mm", "Voxel centre vs baseline", "mm", "≤ 2", (">", 2.0), (">", 3.0)),
    Gate("F21", "csf_change_pp", "CSF fraction change", "pp", "≤ 5", (">", 5.0), (">", 15.0)),
    Gate("F35", "protocol_diff", "Protocol vs baseline (TE)", "", "match", (">", 0.5), None),
    Gate("F19", "csde_pct", "Chemical-shift displacement", "%", "≤ 10", (">", 10.0), (">", 25.0), applies=False),
]


NA_REASON = {
    "drift_abs_hz": "needs the individual transients; the file stores only their average",
    "outlier_pct": "needs the individual transients; the file stores only their average",
    "voxel_shift_mm": "needs the baseline voxel position",
    "csf_change_pp": "needs a tissue segmentation of this session",
    "cho_crlb_pct": "needs a spectral fit",
}


@lru_cache(maxsize=1)
def catalogue() -> dict:
    if not CATALOGUE.exists():
        return {}
    data = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    return {row["ID"]: row for row in data["failure_catalogue"]}


def row(fid: str) -> dict:
    return catalogue().get(fid, {})


def evaluate_mrs(m: dict) -> dict:
    """Run every MRS gate on a metrics dict. Verdict rule from the catalogue's 'Decision rules' sheet."""
    results = []
    for g in MRS_GATES:
        v = m.get(g.metric)
        st = g.status(v)
        reason = "" if v is not None else NA_REASON.get(g.metric, "metric not available for this scan")
        results.append({"id": g.id, "label": g.label, "value": v, "unit": g.unit, "pass_if": g.pass_if, "status": st,
                        "reason": reason,
                        "applies": g.applies, "fix": row(g.id).get("Fix for the next scan", ""),
                        "source": row(g.id).get("Threshold source", "")})
    active = [r for r in results if r["applies"] and r["status"] in RANK]
    worst = max((r["status"] for r in active), key=RANK.get, default="PASS")
    deciding = [r for r in active if r["status"] == worst and worst != "PASS"]
    verdict = {"PASS": "update", "WARN": "update_with_warning", "FAIL": "reject"}[worst]
    return {"status": worst, "verdict": verdict, "gates": results, "deciding": deciding,
            "notes": [r for r in results if not r["applies"]] + [{"id": "F22", "label": "Cr reference",
                     "status": "WARN", "applies": False, "fix": row("F22").get("Fix for the next scan", ""),
                     "value": "ratios only", "unit": "", "pass_if": "ratios + absolute values agree",
                     "source": row("F22").get("Threshold source", "")}]}


def supported_range(value: float, u: float, status: str) -> tuple[float, float]:
    """'What this result can tell you': value ± 1.96 sqrt(R), R doubled for WARN (catalogue decision rule 5)."""
    r = u * u * (2.0 if status == "WARN" else 1.0)
    h = 1.96 * r ** 0.5
    return value - h, value + h
