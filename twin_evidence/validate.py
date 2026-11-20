"""Data Science gate: is the evidence packet structurally valid, linked, traceable and semantically plausible?

Mirrors the logic of the MRI Data Explorer (MDX.qc.evaluate, module DS-1) and Dossier §5.8 / §8.2.
Never modifies the packet: every finding carries rule, field, status, owner and a human-readable reason.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}
ACTIONS = {"PASS": "update", "WARN": "update_with_warning", "FAIL": "reject"}
TIME_TO_MONTHS = {"month": 1.0, "year": 12.0, "week": 12.0 / 52.1775, "day": 12.0 / 365.25, "s": 12.0 / (365.25 * 86400)}


@dataclass
class Finding:
    rule: str
    status: str            # PASS / WARN / FAIL
    owner: str             # QE / PM / DS / DT / shared
    field: str | None = None
    value: Any = None
    threshold: str | None = None
    reason: str = ""
    source: str = "validator"   # 'packet' = stored check re-evaluated, 'validator' = added by us


@dataclass
class Evaluation:
    packet_id: str | None
    status: str
    stored_status: str | None
    action: str
    r_factor: float | None
    u_effective: float | None
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = [f"{f.rule}: {f.reason}" for f in self.findings if f.status != "PASS"]
        return d


def worst(statuses) -> str:
    statuses = [s for s in statuses if s in RANK]
    return max(statuses, key=RANK.get) if statuses else "PASS"


def to_months(rt: Any) -> float | None:
    if not isinstance(rt, dict) or not isinstance(rt.get("value"), (int, float)) or rt.get("unit") not in TIME_TO_MONTHS:
        return None
    return float(rt["value"]) * TIME_TO_MONTHS[rt["unit"]]


def r_eq(volume_mm3: float) -> float:
    """Equivalent-sphere radius (mm) of a volume (mm^3)."""
    return (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def implied_rel_u(volume_mm3: float, sigma_b_mm: float) -> float:
    """Dossier §3.10: relative volume uncertainty of a compact region for boundary uncertainty sigma_b = 3 sigma_b / r_eq."""
    return 3.0 * sigma_b_mm / r_eq(volume_mm3)


def gate_status(gate: dict, value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or not isinstance(value, (int, float)):
        return "FAIL"
    if "fail_above" in gate and value > gate["fail_above"]:
        return "FAIL"
    if "fail_below" in gate and value < gate["fail_below"]:
        return "FAIL"
    if "warn_above" in gate and value > gate["warn_above"]:
        return "WARN"
    if "warn_below" in gate and value < gate["warn_below"]:
        return "WARN"
    return "PASS"


def gate_text(gate: dict) -> str:
    u = f" {gate['unit']}" if gate.get("unit") not in (None, "1") else ""
    parts = []
    for key, sym, lab in (("warn_below", "<", "WARN"), ("warn_above", ">", "WARN"), ("fail_below", "<", "FAIL"), ("fail_above", ">", "FAIL")):
        if key in gate:
            parts.append(f"{lab} {sym} {gate[key]}{u}")
    return "; ".join(parts) or gate.get("rule", "")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as fh:
        return {row["path"]: row["sha256"] for row in csv.DictReader(fh)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


IDENTIFYING = {"patientname", "patient_name", "patient_full_name", "first_name", "last_name", "surname", "patientbirthdate", "birth_date", "birthdate", "dob",
               "patientid", "patient_id", "address", "phone", "email", "mrn", "social_security", "insurance_id"}


def _identifying_keys(obj, prefix="") -> list[str]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower().replace(" ", "") in IDENTIFYING:
                found.append(prefix + str(k))
            found += _identifying_keys(v, prefix + str(k) + ".")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += _identifying_keys(v, f"{prefix}{i}.")
    return found


def evaluate(packet: dict, thresholds: dict, schema: dict | None = None, manifest: dict[str, str] | None = None,
             root: Path | None = None) -> Evaluation:
    """Re-evaluate one evidence packet. Returns status, twin action and every finding with its reason."""
    gates = {g["name"]: g for g in thresholds.get("gates", [])}
    F: list[Finding] = []
    notes: list[str] = []

    # 1 · structure: JSON Schema 2020-12 (required keys, types, enums, patterns)
    if schema is not None:
        errs = sorted(Draft202012Validator(schema).iter_errors(packet), key=lambda e: list(e.absolute_path))
        for e in errs:
            path = ".".join(str(p) for p in e.absolute_path) or "(root)"
            F.append(Finding("schema", "FAIL", "DS", path, None, "evidence_packet_v1.0.schema.json", e.message))
        if not errs:
            F.append(Finding("schema", "PASS", "DS", None, None, "evidence_packet_v1.0.schema.json", "all 22 required keys present with valid types"))

    # 2 · stored QC checks re-evaluated against the active thresholds (never trust a stored status blindly)
    stored = packet.get("qc_checks") if isinstance(packet.get("qc_checks"), list) else []
    for c in stored:
        g = gates.get(c.get("name"))
        status, thr, src = c.get("status"), c.get("threshold"), "packet"
        if g and any(k in g for k in ("warn_above", "warn_below", "fail_above", "fail_below")) and isinstance(c.get("value"), (int, float)):
            new = gate_status(g, c["value"])
            thr = gate_text(g)
            if new != status:
                notes.append(f"{c.get('name')}: stored {status}, {new} under {thresholds.get('id')}")
                status, src = new, "thresholds"
        F.append(Finding(c.get("name", "?"), status if status in RANK else "FAIL", c.get("owner") or (g or {}).get("owner", "shared"),
                         c.get("name"), c.get("value"), thr, f"value {c.get('value')} {c.get('unit', '')} vs {thr}".strip(), src))
    if not stored:
        F.append(Finding("qc_checks_missing", "FAIL", "DS", "qc_checks", None, ">= 1 check", "a packet without checks cannot pass a gate"))

    # 3 · semantic rules
    value, uv = packet.get("value"), (packet.get("uncertainty") or {}).get("value")
    if value is None:
        stored_mv = [f for f in F if f.rule == "missing_values"]
        for f in stored_mv:  # a stored 'missing_values = 0 / PASS' cannot explain a null value
            if f.status != "FAIL":
                f.status, f.reason, f.source = "FAIL", "stored check says no missing values, but value is null", "validator"
        if not stored_mv:
            F.append(Finding("missing_values", "FAIL", "DS", "value", None, "FAIL > 0", "required value is null (missing stays missing, never 0)"))
    elif not isinstance(value, (int, float)):
        F.append(Finding("value_type", "FAIL", "DS", "value", value, "number", "value is not a number (e.g. text '0.85')"))
    if isinstance(value, (int, float)) and not (isinstance(uv, (int, float)) and math.isfinite(uv) and uv > 0):
        F.append(Finding("uncertainty_missing", "FAIL", "PM", "uncertainty.value", uv, "numeric SD > 0",
                         "the data contract requires an uncertainty for every value used in a state update"))
    vr = packet.get("valid_range")
    if isinstance(value, (int, float)) and isinstance(vr, list) and len(vr) == 2:
        inside = vr[0] <= value <= vr[1]
        F.append(Finding("valid_range", "PASS" if inside else "FAIL", "QE", "value", value, f"[{vr[0]}, {vr[1]}]",
                         "inside the declared range" if inside else "outside the declared valid range"))
    if to_months(packet.get("relative_time")) is None:
        F.append(Finding("relative_time", "FAIL", "DS", "relative_time", packet.get("relative_time"), "value + unit",
                         "the twin needs the time since baseline"))
    if packet.get("origin") == "SIMULATED" and packet.get("value_epistemic") == "MEASURED":
        F.append(Finding("epistemic_label", "FAIL", "DS", "value_epistemic", "MEASURED", "SIMULATED never MEASURED",
                         "a SIMULATED origin is never labelled MEASURED"))
    if "system_qc" in gates and packet.get("modality") in ("MRI", "MRS") and not packet.get("system_qc_id"):
        F.append(Finding("system_qc", "FAIL", "PM", "system_qc_id", None, "system_qc_id present", "no system QC reference"))
    # F57 (ScanTwin catalogue): nominal field and spectrometer frequency both recorded
    acq = packet.get("acquisition_context") if isinstance(packet.get("acquisition_context"), dict) else {}
    has_b0, has_nu = isinstance(acq.get("field_strength_T"), (int, float)), isinstance(acq.get("spectrometer_frequency_MHz"), (int, float))
    if packet.get("modality") in ("MRI", "MRS"):
        st57 = "PASS" if has_b0 and has_nu else "WARN" if has_b0 else "FAIL"
        F.append(Finding("F57_field_frequency", st57, "QE", "acquisition_context", None, "nominal field and nu0 both present",
                         "both recorded" if st57 == "PASS" else "only the nominal field" if st57 == "WARN" else "neither recorded"))
    # F59 (ScanTwin catalogue): de-identification, pseudonymous ids only and no identifying fields anywhere
    ident = _identifying_keys(packet)
    sid = packet.get("subject_id")
    pseudo = isinstance(sid, str) and sid.startswith("sub-")
    if ident or not pseudo:
        F.append(Finding("F59_deidentification", "FAIL", "DS", ", ".join(ident) or "subject_id", None, "pseudonymous ids only",
                         "identifying field present: " + ", ".join(ident) if ident else "subject_id is not a pseudonym (sub-...)"))
    else:
        F.append(Finding("F59_deidentification", "PASS", "DS", "subject_id", sid, "pseudonymous ids only", "pseudonymous"))
    if "protocol_match" in gates and packet.get("protocol_match") is False:
        F.append(Finding("protocol_match", "WARN", "PM", "protocol_match", False, "same sequence, TE, TR, field", "protocol differs from baseline"))

    # 4 · linkage and integrity: source file exists in the manifest and its bytes still match
    src, sha = packet.get("source_object"), packet.get("source_sha256")
    if manifest is not None:
        if src not in manifest:
            F.append(Finding("source_in_manifest", "FAIL", "DS", "source_object", src, "listed in manifest_sha256.csv", "source file not registered"))
        elif sha and sha != manifest[src]:
            F.append(Finding("source_sha256", "FAIL", "DS", "source_sha256", sha[:12], "manifest digest", "packet digest differs from the manifest"))
        else:
            F.append(Finding("source_in_manifest", "PASS", "DS", "source_object", src, "listed in manifest_sha256.csv", "registered source"))
        if root is not None and src in manifest:
            p = Path(root) / src
            ok = p.exists() and sha256_file(p) == manifest[src]
            F.append(Finding("source_bytes", "PASS" if ok else "FAIL", "DS", "source_object", src, "SHA-256 of file == manifest",
                             "file bytes unchanged" if ok else "integrity mismatch: file missing or bytes changed"))

    # 5 · uncertainty consistency for propagated ROI masks (Dossier §3.10; thresholds rule roi_u_consistency)
    u_eff = float(uv) if isinstance(uv, (int, float)) and math.isfinite(uv) and uv > 0 else None
    sref = packet.get("spatial_reference") or {}
    if "roi_u_consistency" in gates and packet.get("observable") == "roi_volume" and isinstance(value, (int, float)) \
            and u_eff is not None and sref.get("mask_propagated") is True:
        reg = next((c.get("value") for c in stored if c.get("name") == "registration_rms_mm"), sref.get("registration_rms_mm"))
        if isinstance(reg, (int, float)):
            vref = sref.get("reference_roi_volume_mm3", value)
            implied, stated = implied_rel_u(vref, reg), u_eff / value
            ok = stated + 1e-12 >= implied
            F.append(Finding("roi_u_consistency", "PASS" if ok else gates["roi_u_consistency"].get("status_if_violated", "WARN"),
                             gates["roi_u_consistency"].get("owner", "PM"), "uncertainty.value", round(stated, 4),
                             f"stated u_rel >= implied {implied:.3f} (3 x {reg} mm / r_eq {r_eq(vref):.1f} mm)",
                             "consistent" if ok else f"stated {stated:.1%} < implied {implied:.1%}: implied uncertainty carried forward"))
            if not ok:
                u_eff = implied * value
                notes.append(f"roi_u_consistency: u carried forward = {u_eff:.1f} {packet.get('unit')}")

    status = worst(f.status for f in F)
    stored_status = packet.get("qc_status")
    if stored_status in RANK and RANK[stored_status] > RANK[status]:
        F.append(Finding("stored_status", stored_status, "DS", "qc_status", stored_status, "explained by checks",
                         "stored status is worse than its checks explain; kept as a floor"))
        status = stored_status
    if stored_status in RANK and RANK[status] > RANK[stored_status]:
        notes.append(f"stored qc_status {stored_status} is too optimistic; re-evaluated status is {status}")
    pol = (thresholds.get("twin_policy") or {}).get(status, {})
    return Evaluation(packet.get("packet_id"), status, stored_status, pol.get("action", ACTIONS[status]),
                      pol.get("R_factor"), u_eff if status != "FAIL" else None, F, notes)
