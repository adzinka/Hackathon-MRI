"""The two twins talk: the scanner twin's state on the scan date becomes a gate finding on every subject packet.

ScanTwin rule (docs/scantwin PDF): "the subject twin may only change when the scanner twin says the evidence is good
enough". Catalogue row F11 asks for a passing system QC; here the check is not "is there a QC id" but "what does the
scanner twin say about this scanner on that day, for the parts of the machine this scan depends on".

  scanner twin FAIL on a relevant part  ->  finding FAIL  ->  subject update blocked (reject)
  scanner twin WARN on a relevant part  ->  finding WARN  ->  update with warning (R doubled)
  scanner twin PASS                     ->  finding PASS
  no scanner record                     ->  no finding (the packet-level F11 check still applies)

The scanner record is built from the weekly phantom series (SIMULATED for the StarterPack, which has none).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import longitudinal as L
from .gates import row
from .scanner import gate_value, track
from .validate import Finding, RANK

WEEKS_PER_MONTH = 52.1775 / 12

# which phantom observables matter for which kind of subject measurement
RELEVANT = {
    "MRS": ["ph_fwhm_hz", "ph_snr_pct", "ph_freq_hz", "ph_water_ratio", "ph_geometry_mm"],
    "MRI": ["ph_snr_pct", "ph_geometry_mm"],
}
OBS_TO_CATALOGUE = {"ph_fwhm_hz": "F12", "ph_snr_pct": "F09", "ph_freq_hz": "F08", "ph_water_ratio": "F15",
                    "ph_geometry_mm": "F01"}
CLASS_TEXT = {"shim": "magnet shim degraded", "receive_coil": "receive coil / RF chain degraded",
              "b0_drift": "B0 field drifting", "b1_transmit": "transmitter (B1) miscalibrated",
              "gradient": "gradient calibration drifted", "stable": "scanner stable"}
CLASS_FIX = {"shim": "F12", "receive_coil": "F09", "b0_drift": "F08", "b1_transmit": "F61", "gradient": "F01"}


@dataclass
class ScannerRecord:
    fault: str                       # simulated hardware state (key of longitudinal.FAULTS)
    series: dict                     # obs -> weekly list (value, status, twin x, sd)
    diagnosis: dict                  # fault class over the record
    weeks: int = L.WEEKS
    label: str = ""
    history: list = field(default_factory=list)
    z: dict = field(default_factory=dict)
    weeks_to_fail: dict = field(default_factory=dict)

    @classmethod
    def simulate(cls, fault: str, seed: int = 1) -> "ScannerRecord":
        weeks = [L.phantom_week(fault, w, seed * 1000 + w) for w in range(L.WEEKS + 1)]
        trk = track({o: [w[o] for w in weeks] for o in L.PHANTOM_OBS}, L.PHANTOM_OBS)
        return cls(fault, trk["series"], L.diagnose(trk), label=L.FAULTS[fault]["title"], history=weeks,
                   z=trk["z"], weeks_to_fail=trk["weeks_to_fail"])

    def weekly_rows(self) -> list[dict]:
        """One row per week with value, status and twin estimate of every phantom observable (for charts and CSV)."""
        out = []
        for i in range(self.weeks + 1):
            r = {"week": i}
            for o in L.PHANTOM_OBS:
                p = self.series[o][i]
                r.update({o: p["value"], f"{o}_status": p["status"], f"{o}_twin_x": p["x"]})
            out.append(r)
        return out

    def week_of(self, packet: dict) -> int | None:
        rt = packet.get("relative_time") or {}
        if not isinstance(rt.get("value"), (int, float)):
            return None
        unit = rt.get("unit")
        months = rt["value"] * {"month": 1, "year": 12, "week": 12 / 52.1775, "day": 12 / 365.25}.get(unit, float("nan"))
        if months != months:
            return None
        return max(0, min(self.weeks, round(months * WEEKS_PER_MONTH)))

    def state(self, week: int, modality: str) -> dict:
        """Health of the relevant parts of the scanner on that week, judged on the twin estimate (not one noisy reading)."""
        parts = []
        for o in RELEVANT.get(modality, RELEVANT["MRS"]):
            p = self.series[o][week]
            st = gate_value(L.PHANTOM_OBS[o][4], p["x"])
            parts.append({"obs": o, "label": L.PHANTOM_OBS[o][0], "unit": L.PHANTOM_OBS[o][1], "x": p["x"], "sd": p["sd"],
                          "status": st})
        worst = max((p["status"] for p in parts), key=RANK.get, default="PASS")
        return {"week": week, "status": worst, "parts": parts, "bad": [p for p in parts if p["status"] != "PASS"]}

    def finding(self, packet: dict) -> Finding | None:
        modality = packet.get("modality")
        if modality not in RELEVANT:
            return None
        wk = self.week_of(packet)
        if wk is None:
            return Finding("F11_scanner_twin", "WARN", "PM", "relative_time", None, "scan date on the scanner record",
                           "cannot place the scan on the scanner record (no relative time)", "scanner_twin")
        s = self.state(wk, modality)
        if s["status"] == "PASS":
            reason = f"scanner twin PASS in week {wk}: all {len(s['parts'])} parts this {modality} scan depends on are healthy"
        else:
            parts = "; ".join(f"{p['label']} {p['x']:.2f} {p['unit']} ({p['status']})" for p in s["bad"])
            cls = self.diagnosis["fault_class"]
            fix = row(CLASS_FIX.get(cls, "F11")).get("Fix for the next scan", "")
            action = "subject update blocked" if s["status"] == "FAIL" else "subject evidence down-weighted"
            reason = (f"scanner twin {s['status']} in week {wk}: {parts}. Likely cause: {CLASS_TEXT.get(cls, cls)}. "
                      f"{action}. Fix the scanner first: {fix}")
        return Finding("F11_scanner_twin", s["status"], "PM", "system_qc_id", packet.get("system_qc_id"),
                       "scanner twin state on the scan date (phantom record)", reason, "scanner_twin")


def attribution(subject_status: str, finding: Finding | None, record: ScannerRecord | None) -> dict:
    """Patient or machine? One sentence for the decision panel."""
    if record is None or finding is None:
        return {"who": "unknown", "text": "No scanner twin attached: the machine cannot be ruled out or blamed."}
    cls = CLASS_TEXT.get(record.diagnosis["fault_class"], record.diagnosis["fault_class"])
    if finding.status == "FAIL":
        return {"who": "machine", "text": f"MACHINE. The scanner twin is FAIL on the scan date ({cls}). The result is not used; "
                                          "fix the scanner and repeat."}
    if finding.status == "WARN":
        return {"who": "machine", "text": f"MACHINE (likely). The scanner twin warns on the scan date ({cls}); the subject "
                                          "evidence is down-weighted. Fix the scanner, then repeat to confirm any change."}
    if subject_status != "PASS":
        return {"who": "session", "text": "NOT THE MACHINE. The scanner is healthy on the scan date, so the poor evidence comes "
                                          "from this session or its processing (head position, motion, local shim, "
                                          "registration). Fix that and repeat."}
    return {"who": "patient", "text": "PATIENT. Scanner healthy and evidence clean: any change that passes the change test "
                                      "can be attributed to the subject."}
