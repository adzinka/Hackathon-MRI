"""ScanTwin quality gates: the numeric "Built" rows of the team's failure catalogue as machine-readable gates.

Source: docs/scantwin/ScanTwin failure catalogue and QC thresholds.xlsx (imported to catalogue/scantwin_catalogue.json).
Each gate keeps the catalogue ID, so every verdict can name the row it came from and show that row's fix.

Catalogue v1.1 (team change list, catalogue/changes_v1.1.json, after docs/scantwin/REVIEW.md):
  F15 tightened to WARN > 0.5, FAIL > 1.5 (valid for a fit without residual-water removal).
  F19 is a between-session check (change of the displaced fraction); the ~12 % of PRESS at 3 T is a standing limitation.
  F22 is no longer a gate: it is a limitation on the measurement card until water-referenced data exist.
  F61 (new): transmitter / flip-angle (B1) miscalibration, reference amplitude vs phantom baseline.
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
    Gate("F15", "water_ratio", "Residual water / NAA", "1", "≤ 0.5", (">", 0.5), (">", 1.5)),
    Gate("F08", "drift_abs_hz", "Frequency drift", "Hz", "≤ 6", (">", 6.0), (">", 12.0)),
    Gate("F23", "outlier_pct", "Transients rejected (motion)", "%", "≤ 5", (">", 5.0), (">", 20.0)),
    Gate("F20", "voxel_shift_mm", "Voxel centre vs baseline", "mm", "≤ 2", (">", 2.0), (">", 3.0)),
    Gate("F21", "csf_change_pp", "CSF fraction change", "pp", "≤ 5", (">", 5.0), (">", 15.0)),
    Gate("F35", "protocol_diff", "Protocol vs baseline (TE)", "", "match", (">", 0.5), None),
    Gate("F19", "csde_change_pp", "Chemical-shift displacement, change vs baseline", "pp", "≈ 0 (same protocol)",
         (">", 2.0), None),
    Gate("F61", "ref_amp_dev_pct", "Transmitter reference vs phantom baseline", "%", "≤ 10", (">", 10.0), (">", 20.0)),
]


NA_REASON = {
    "drift_abs_hz": "needs the individual transients; the file stores only their average",
    "outlier_pct": "needs the individual transients; the file stores only their average",
    "voxel_shift_mm": "needs the baseline voxel position",
    "csf_change_pp": "needs a tissue segmentation of this session",
    "cho_crlb_pct": "needs a spectral fit",
    "ref_amp_dev_pct": "needs the transmitter reference / phantom baseline of this scanner",
    "csde_change_pp": "needs the sequence and RF bandwidth of both sessions",
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
            "notes": [r for r in results if not r["applies"]] + STANDING_LIMITATIONS}


# shown on every MRS verdict and on the measurement card; never change the verdict (catalogue v1.1)
STANDING_LIMITATIONS = [
    {"id": "F19", "label": "PRESS at 3 T displaces each metabolite's voxel by ~12 % (2.4 mm per axis); common to both sessions",
     "applies": False},
    {"id": "F22", "label": "Ratios only; creatine assumed stable. A change in X/Cr may come from Cr; absolute "
     "water-referenced values are needed to rule this out", "applies": False},
]


def supported_range(value: float, u: float, status: str) -> tuple[float, float]:
    """'What this result can tell you': value ± 1.96 sqrt(R), R doubled for WARN (catalogue decision rule 5)."""
    r = u * u * (2.0 if status == "WARN" else 1.0)
    h = 1.96 * r ** 0.5
    return value - h, value + h


@lru_cache(maxsize=1)
def context_factors() -> list[dict]:
    """Rows of the catalogue's 'Context factors' sheet, with the failure IDs each one raises."""
    if not CATALOGUE.exists():
        return []
    import re
    rows = json.loads(CATALOGUE.read_text(encoding="utf-8")).get("context_factors", [])
    for r in rows:
        r["ids"] = re.findall(r"F\d{2}", r.get("Raises the risk of") or "")
    return rows


def context_advice(selected: list[str], gates_result: dict) -> list[dict]:
    """Scan context changes only the warnings and the fix advice, never the PASS / WARN / FAIL limits (catalogue v1.1)."""
    status = {g["id"]: g["status"] for g in gates_result["gates"]}
    out = []
    for r in context_factors():
        if r["Context factor"] not in selected:
            continue
        for fid in r["ids"]:
            st = status.get(fid)
            if st is None:                     # the factor raises a failure mode of another scan type
                continue
            out.append({"factor": r["Context factor"], "id": fid, "failure": row(fid).get("Failure mode", ""),
                        "gate_status": st or "not a gate for this scan type",
                        "advice": r["How the twin uses it"],
                        "reading": ("matches the context: fix the cause first" if st in ("WARN", "FAIL")
                                    else "watch this gate on the next scan" if st == "PASS" else "")})
    return out


# Structural T1w gates (catalogue rows marked "Built" for Structural T1w / T2w, plus F27/F28 "Designed" with numbers)
T1W_GATES = [
    Gate("F26", "motion_rating", "Subject motion (rating 1-4)", "", "1 (none)", (">", 1.5), (">", 2.5)),
    Gate("F27", "cnr_pct_baseline", "Grey/white contrast vs baseline", "%", "≥ 80", ("<", 80.0), ("<", 60.0)),
    Gate("F28", "wm_cv_pct", "White-matter brightness variation (bias field)", "%", "≤ 10", (">", 10.0), (">", 20.0)),
    Gate("F32", "reg_shift_mm", "Registration error between sessions", "mm", "≤ 2", (">", 2.0), (">", 3.0)),
    Gate("F35", "protocol_diff", "Protocol vs baseline (TR, TE, TI, flip, field)", "", "all match", (">", 0.5), None),
]
NA_REASON.update({
    "motion_rating": "the background was masked to zero, so motion ghosts cannot be measured",
    "cnr_pct_baseline": "needs a baseline scan to compare with",
    "reg_shift_mm": "needs a baseline scan and the region outline of both scans (the images are aligned automatically)",
    "protocol_diff": "needs the protocol (sidecar) of both scans",
})


# ---------------------------------------------------------------------------- TRUSTY / WARNING / ERROR
# The catalogue's decision rules with the names used on screen: PASS -> TRUSTY (update), WARN -> WARNING
# (update_with_warning), FAIL -> ERROR (reject). A check that cannot run is N/A (catalogue rule 3, never FAIL); but if a
# check of the minimum set is N/A the scan cannot be called TRUSTY ("no information" is not "no error").
STATE = {"PASS": "TRUSTY", "WARN": "WARNING", "FAIL": "ERROR"}
CATEGORIES = [
    ("DICOM integrity", lambda f: f.get("kind") in ("file", "geometry") or f["id"] == "F54"),
    ("Data integrity", lambda f: f["id"] in ("F56", "F57", "F58", "F59")),
    ("Motion", lambda f: f["id"] == "F26"),
    ("Image quality", lambda f: f["id"] in ("F27", "F28")),
    ("Protocol", lambda f: f["id"] == "F35"),
    ("Comparison between scans", lambda f: f["id"] in ("F32", "F33", "F34")),
]
MINIMUM_T1W = ("F26", "F28")      # team decision pending (docs/scantwin/INFORME_T1w_DICOM.md §5.2)


def classify(gates_result: dict | None = None, dicom_check: dict | None = None,
             minimum: tuple = MINIMUM_T1W) -> dict:
    """One state for the scan and one per category. Each item keeps its catalogue ID, value, reason and fix."""
    items = list((dicom_check or {}).get("findings", [])) + list((gates_result or {}).get("gates", []))
    cats = []
    for name, sel in CATEGORIES:
        own = [x for x in items if sel(x)]
        if not own:
            continue
        sts = [x["status"] for x in own if x["status"] in RANK]
        st = max(sts, key=RANK.get) if sts else "N/A"
        cats.append({"category": name, "status": STATE.get(st, "N/A"), "items": own})
    worst = max((x["status"] for x in items if x["status"] in RANK), key=RANK.get, default="PASS")
    present = {x["id"]: x for x in items}
    unchecked = [present[i] for i in minimum if i in present and present[i]["status"] not in RANK]
    state = STATE[worst]
    if state == "TRUSTY" and unchecked:
        state = "WARNING"
    reasons = [f"{x['id']} {x['label']}: {x['status']}" + (f" ({x['reason']})" if x.get("reason") else "")
               for x in items if x["status"] == worst and worst != "PASS"]
    reasons += [f"{x['id']} {x['label']} could not be checked: {x.get('reason') or 'no data'}" for x in unchecked]
    return {"state": state, "categories": cats, "reasons": reasons, "unchecked": [x["id"] for x in unchecked],
            "verdict": {"TRUSTY": "update", "WARNING": "update_with_warning", "ERROR": "reject"}[state]}


def evaluate_t1w(m: dict) -> dict:
    results = []
    for g in T1W_GATES:
        v = m.get(g.metric)
        st = g.status(v)
        why = (m.get("na_reasons") or {}).get(g.metric) or NA_REASON.get(g.metric, "not available for this scan")
        results.append({"id": g.id, "label": g.label, "value": v, "unit": g.unit, "pass_if": g.pass_if, "status": st,
                        "reason": "" if v is not None else why,
                        "applies": True, "fix": row(g.id).get("Fix for the next scan", ""),
                        "source": row(g.id).get("Threshold source", "")})
    active = [r for r in results if r["status"] in RANK]
    worst = max((r["status"] for r in active), key=RANK.get, default="PASS")
    deciding = [r for r in active if r["status"] == worst and worst != "PASS"]
    return {"status": worst, "verdict": {"PASS": "update", "WARN": "update_with_warning", "FAIL": "reject"}[worst],
            "gates": results, "deciding": deciding, "notes": []}
