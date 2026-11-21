"""Run the app's own T1w and MRS pipelines (unchanged gates) on the real samples; write the QC report.

    python scripts/run_real_samples.py        # -> external_data/real_samples/qc_report.json

Every number comes from the pipeline; a metric the pipeline cannot compute is written as null (N/A) with its reason.
The T1w state uses the same classify() as the "Check a scan" page; the MRS state maps PASS / WARN / FAIL to
TRUSTY / WARNING / ERROR exactly as the page does.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from twin_evidence import ingest as I, t1w as T  # noqa: E402
from twin_evidence.gates import STATE, classify  # noqa: E402

DATA = REPO / "external_data" / "real_samples"
T1W = DATA / "T1w" / "sub-01_T1w.nii.gz"
SVS = DATA / "MRS" / "sub-01_svs.nii.gz"
REF = DATA / "MRS" / "sub-01_ref.nii.gz"


def _r(v, n=4):
    return round(v, n) if isinstance(v, float) else v


def _gates(g: dict) -> list:
    return [{"id": x["id"], "label": x["label"], "status": x["status"], "value": _r(x["value"]), "unit": x["unit"],
             "pass_if": x["pass_if"], "reason": x.get("reason") or None} for x in g["gates"]]


def t1w() -> dict:
    t = time.time()
    s = T.load(T1W, name=T1W.name)
    m = T.measure(s)
    g = T.gate(m)
    c = classify(g)
    keys = ("snr_wm", "cnr", "noise", "wm_cv_pct", "ghost_pct", "ghost_method", "motion_rating", "head_ml",
            "brain_ml", "gm_ml", "wm_ml", "csf_ml", "roi_mm3", "reg_shift_mm", "head_shift_mm", "protocol_diff")
    return {"file": str(T1W.relative_to(REPO)).replace("\\", "/"), "loaded": True, "seconds": round(time.time() - t, 1),
            "shape": list(s.data.shape), "voxel_mm": [float(z) for z in s.zooms], "voxel_mm3": s.voxel_mm3,
            "background_masked": s.bg_masked, "brain_mask_supplied": s.brain_mask,
            "identifying_fields_removed": s.removed, "metrics": {k: _r(m.get(k)) for k in keys},
            "gates": _gates(g), "catalogue_status": g["status"], "state": c["state"], "reasons": c["reasons"],
            "unchecked_minimum_gates": c["unchecked"]}


def mrs() -> dict:
    t = time.time()
    ing = I.read(SVS)
    m = I.measure(ing)
    g = I.gate(m, ing.nu0)
    w = I.water_reference(REF)
    keys = ("snr", "fwhm_hz", "fwhm_ppm", "water_ratio", "drift_abs_hz", "outlier_pct", "shift_sd_hz", "cho_cr",
            "cho_cr_crlb", "naa_cr", "naa_cr_crlb", "cho_crlb_pct", "fit_lw_hz", "te_ms")
    return {"file": str(SVS.relative_to(REPO)).replace("\\", "/"), "loaded": True, "seconds": round(time.time() - t, 1),
            "shape": list(ing.shape), "points": int(ing.transients.shape[1]), "transients": int(ing.transients.shape[0]),
            "nu0_mhz": ing.nu0, "field_T": round(ing.nu0 / I.GAMMA_MHZ_PER_T, 2), "dwell_s": ing.dwell,
            "spectral_width_hz": round(1 / ing.dwell, 3), "identifying_fields_removed": ing.removed,
            "processing_steps": ing.steps, "metrics": {k: _r(m.get(k)) for k in keys},
            "metrics_note": "fwhm_hz and drift_abs_hz are measured Hz; the gates compare 3 T-equivalent Hz "
                            "(x 123.25 / nu0, i.e. the same ppm criterion)",
            "gates": _gates(g), "catalogue_status": g["status"], "state": STATE[g["status"]],
            "deciding": [x["id"] for x in g["deciding"]],
            "water_reference": {"file": str(REF.relative_to(REPO)).replace("\\", "/"), "available": True,
                                "loaded": True, "shape": list(w["shape"]),
                                "water_fwhm_hz": _r(w["water_fwhm_hz"], 2), "water_fwhm_ppm": _r(w["water_fwhm_ppm"]),
                                "water_offset_hz": _r(w["water_offset_hz"], 2),
                                "note": "informational; no catalogue gate uses the water reference yet"}}


if __name__ == "__main__":
    man = json.loads((DATA / "dataset_manifest.json").read_text(encoding="utf-8"))
    rep = {"generated_by": "scripts/run_real_samples.py", "dataset_doi": man["dataset"]["doi"],
           "participant": man["participant"], "t1w": t1w(), "mrs": mrs()}
    out = DATA / "qc_report.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for k in ("t1w", "mrs"):
        r = rep[k]
        print(f"{k}: {r['state']} (catalogue {r['catalogue_status']}) · "
              + ", ".join(f"{x['id']} {x['status']}" for x in r["gates"]))
    print(f"wrote {out.relative_to(REPO)}")
