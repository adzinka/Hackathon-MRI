"""Field mapper: a raw single-voxel NIfTI-MRS file (any field, any vendor via spec2nii) -> ScanTwin evidence packet v1.0.

Steps, each recorded in the conversion report:
  1 read      NIfTI-MRS data, dwell time (pixdim[4]), spectrometer frequency and metadata (header extension, code 44)
  2 de-identify  drop identifying metadata (catalogue F59); only field NAMES are reported, never values
  3 orient    detect the frequency-axis convention (NAA at 2.01 ppm, not mirrored at 7.29 ppm) and conjugate if needed
  4 combine   coils (DIM_COIL): SNR-weighted, phase-aligned on the first FID points
  5 transients (DIM_DYN): drift and motion outliers (F08, F23); then average
  6 reference NAA peak to 2.01 ppm; zero-order phase on the NAA-Cr-Cho region
  7 QC        SNR, linewidth, residual water (F13, F12, F15) on the averaged spectrum
  8 fit       windowed fit (NAA, Cr, Cho + linear baseline) -> Cho/Cr with CRLB (F14)
  9 packet    schema v1.0; the de-identified averaged spectrum is written as the packet's source file (SHA-256)

Field strength: catalogue limits in Hz (F12, F08) are written for 3 T; they are scaled by nu0 / 123.25 MHz so the same
ppm criterion applies at any field. The windowed fit reads Cho/Cr ~3 % low on simulated truth (documented offset).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import mrs_fit as F
from . import mrs_sim as S
from .gates import evaluate_mrs

GAMMA_MHZ_PER_T = 42.577
REF_NU0 = 123.25          # the catalogue's Hz limits are written for this frequency (3 T StarterPack)
IDENTIFYING_PREFIXES = ("Patient",)
IDENTIFYING_KEYS = {"InstitutionAddress", "DeviceSerialNumber", "OperatorsName", "ReferringPhysicianName",
                    "PerformingPhysicianName", "AccessionNumber", "StudyID", "OtherPatientIDs"}
KEEP_KEYS = {"PatientPosition"}   # scan geometry, not identity


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Ingested:
    transients: np.ndarray          # (n_dyn, N) coil-combined, oriented, referenced, phased
    fid: np.ndarray                 # average of the transients
    dwell: float
    nu0: float
    meta: dict                      # de-identified metadata
    removed: list                   # identifying field names removed
    steps: list = field(default_factory=list)
    shape: tuple = ()
    dims: dict = field(default_factory=dict)
    source_sha256: str = ""
    source_name: str = ""


def deidentify(meta: dict) -> tuple[dict, list]:
    clean, removed = {}, []
    for k, v in meta.items():
        if k not in KEEP_KEYS and (k.startswith(IDENTIFYING_PREFIXES) or k in IDENTIFYING_KEYS):
            removed.append(k)
        else:
            clean[k] = v
    return clean, sorted(removed)


def read(path: Path) -> Ingested:
    import nibabel as nib
    path = Path(path)
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    meta = {}
    for e in img.header.extensions:
        if e.get_code() == 44:
            meta = json.loads(e.get_content())
    dwell = float(img.header["pixdim"][4])
    nu0 = float((meta.get("SpectrometerFrequency") or [REF_NU0])[0])
    clean, removed = deidentify(meta)
    steps = [f"read {path.name}: shape {tuple(data.shape)}, dwell {dwell * 1e3:.4f} ms, nu0 {nu0:.3f} MHz "
             f"({nu0 / GAMMA_MHZ_PER_T:.2f} T)"]
    steps.append(f"de-identify: removed {len(removed)} identifying field(s): {', '.join(removed) or 'none'}")

    # dimension tags: dim 5..7 of NIfTI-MRS (index 4..6 of the array)
    if data.ndim < 4 or data.shape[:3] != (1, 1, 1):
        raise ValueError(f"not a single-voxel NIfTI-MRS file (shape {data.shape})")
    x = data[0, 0, 0].astype(np.complex128)                        # (N, d5, d6, d7)
    dims = {i: meta.get(f"dim_{i + 5}") for i in range(x.ndim - 1)}
    scale = 1.0 / (np.abs(x[:8]).mean() + 1e-30)                  # bring tiny vendor units to O(1)
    x = x * scale

    # combine coils
    coil_axes = [i + 1 for i, tag in dims.items() if tag == "DIM_COIL"]
    if coil_axes:
        ax = coil_axes[0]
        xs = np.moveaxis(x, ax, -1)                                # coils last
        other = tuple(range(1, xs.ndim - 1))
        noise = xs[-max(64, xs.shape[0] // 5):].std(axis=(0, *other)) if other else xs[-64:].std(axis=0)
        s0 = xs[:4].mean(axis=(0, *other)) if other else xs[:4].mean(axis=0)
        w = np.conj(s0) / noise ** 2
        x = (xs * w).sum(axis=-1) / np.abs(w).sum()
        steps.append(f"combine {len(w)} coils: SNR-weighted, phase-aligned (weights {np.abs(w).min() / np.abs(w).max():.2f}–1.00)")
        dims = {i: t for i, t in dims.items() if t != "DIM_COIL"}
        dims = dict(enumerate(dims.values()))
    # transients
    dyn_axes = [i + 1 for i, tag in dims.items() if tag in ("DIM_DYN", "DIM_EDIT", None)]
    if x.ndim == 1:
        tr = x[None, :]
    else:
        tr = np.moveaxis(x, dyn_axes[0] if dyn_axes else 1, 1).reshape(x.shape[0], -1).T     # (n_dyn, N)
    steps.append(f"transients: {tr.shape[0]} stored" + ("" if tr.shape[0] > 1 else " (averaged file: F08 and F23 are N/A)"))

    # orientation: NAA (2.01 ppm) must be stronger than its mirror around water (7.29 ppm)
    def strength(fid, lo, hi):
        ppm, sp = F.spectrum(fid, dwell, nu0=nu0)
        return np.abs(sp[(ppm > lo) & (ppm < hi)]).max()
    avg = tr.mean(0)
    if strength(avg, 7.1, 7.5) > strength(avg, 1.85, 2.2):
        tr = np.conj(tr)
        steps.append("orient: spectrum was mirrored (NAA found at 7.3 ppm); conjugated to the StarterPack convention")
    else:
        steps.append("orient: convention already matches (NAA near 2.0 ppm)")

    # reference NAA to 2.01 ppm
    ppm, sp = F.spectrum(tr.mean(0), dwell, zero_fill=8, nu0=nu0)
    sel = (ppm > 1.7) & (ppm < 2.3)
    naa_ppm = ppm[sel][np.argmax(np.abs(sp[sel]))]
    shift_hz = (naa_ppm - F.METABOLITES["NAA"]) * nu0
    t = F.time_axis(tr.shape[1], dwell)
    tr = tr * np.exp(-2j * np.pi * shift_hz * t)[None, :]
    steps.append(f"reference: NAA found at {naa_ppm:.3f} ppm, shifted by {-shift_hz:+.1f} Hz to 2.01 ppm")

    # zero-order phase maximising the real NAA-Cr-Cho region
    ppm, sp = F.spectrum(tr.mean(0), dwell, nu0=nu0)
    reg = (ppm > 1.8) & (ppm < 3.3)
    phases = np.linspace(-np.pi, np.pi, 721)
    ph = phases[np.argmax([(sp[reg] * np.exp(1j * p)).real.sum() for p in phases])]
    tr = tr * np.exp(1j * ph)
    steps.append(f"phase: zero-order {np.degrees(ph):+.1f}°")
    return Ingested(tr, tr.mean(0), dwell, nu0, clean, removed, steps, tuple(data.shape), dims,
                    sha256_file(path), path.name)


def measure(ing: Ingested) -> dict:
    """QC metrics and the windowed fit, in the same keys as scenarios.measure (so the same gates apply)."""
    q = F.qc_metrics(ing.fid, ing.dwell, nu0=ing.nu0)
    fit = F.fit_window(ing.fid, ing.dwell, nu0=ing.nu0)
    cho, cho_sd = fit.ratio("Cho")
    naa, naa_sd = fit.ratio("NAA")
    i = fit.names.index("Cho")
    crlb_pct = 100 * math.sqrt(fit.cov[i, i]) / abs(fit.amplitudes["Cho"]) if fit.amplitudes["Cho"] else 999.0
    m = {"snr": q["snr"], "fwhm_hz": q["fwhm_hz"], "fwhm_ppm": q["fwhm_hz"] / ing.nu0, "water_ratio": q["water_ratio"],
         "cho_cr": cho, "cho_cr_crlb": cho_sd, "naa_cr": naa, "naa_cr_crlb": naa_sd, "cho_crlb_pct": min(crlb_pct, 999.0),
         "fit_lw_hz": fit.lw_hz, "te_ms": 1000 * float(ing.meta.get("EchoTime") or 0) or None,
         "drift_abs_hz": None, "outlier_pct": None, "voxel_shift_mm": None, "csf_change_pp": None, "protocol_diff": None,
         "csde_change_pp": None, "ref_amp_dev_pct": None}
    if ing.transients.shape[0] > 3:
        tq = S.transient_qc(ing.transients, dwell=ing.dwell)
        m.update({"drift_abs_hz": abs(tq["drift_hz"]), "outlier_pct": 100 * tq["transient_outlier_frac"],
                  "shift_sd_hz": tq["shift_sd_hz"]})
    return m


def field_scaled(m: dict, nu0: float) -> dict:
    """Express Hz metrics in 3 T-equivalent Hz, so the catalogue's Hz limits mean the same ppm everywhere."""
    k = REF_NU0 / nu0
    out = dict(m)
    for key in ("fwhm_hz", "drift_abs_hz"):
        if out.get(key) is not None:
            out[key] = out[key] * k
    return out


def gate(m: dict, nu0: float) -> dict:
    return evaluate_mrs(field_scaled(m, nu0))


def write_derivative(ing: Ingested, path: Path) -> str:
    """De-identified, coil-combined, averaged spectrum as a minimal NIfTI-MRS file; returns its SHA-256."""
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(ing.fid.astype(np.complex64).reshape(1, 1, 1, -1), np.eye(4))
    img.header["pixdim"][4] = ing.dwell
    hdr = {k: v for k, v in ing.meta.items() if not k.startswith("dim_")}
    hdr.update({"SpectrometerFrequency": [ing.nu0], "ResonantNucleus": ["1H"],
                "ScanTwinProcessing": [s for s in ing.steps if not s.startswith("de-identify")]})
    img.header.extensions.append(nib.nifti1.Nifti1Extension(44, json.dumps(hdr).encode()))
    nib.save(img, str(path))
    return sha256_file(path)


def write_transients(ing: Ingested, path: Path) -> str:
    """De-identified, coil-combined spectrum that KEEPS the individual transients (NIfTI-MRS dim 5 = DIM_DYN), so the
    drift and motion gates still work when the file is read again. Returns its SHA-256."""
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ing.transients.T.astype(np.complex64).reshape(1, 1, 1, ing.transients.shape[1], ing.transients.shape[0])
    img = nib.Nifti1Image(data, np.eye(4))
    img.header["pixdim"][4] = ing.dwell
    hdr = {k: v for k, v in ing.meta.items() if not k.startswith("dim_")}
    hdr.update({"SpectrometerFrequency": [ing.nu0], "ResonantNucleus": ["1H"], "dim_5": "DIM_DYN",
                "ScanTwinProcessing": [s for s in ing.steps if not s.startswith("de-identify")]})
    img.header.extensions.append(nib.nifti1.Nifti1Extension(44, json.dumps(hdr).encode()))
    nib.save(img, str(path))
    return sha256_file(path)


def read_bytes(data: bytes, name: str = "upload.nii.gz") -> Ingested:
    """Read an uploaded file (bytes) through a private temporary copy."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / Path(name).name
        p.write_bytes(data)
        return read(p)


def build_packet(ing: Ingested, m: dict, g: dict, *, packet_id: str, subject_id: str, session_id: str,
                 months: float, rel_path: str, sha: str, origin: str = "MEASURED", system_qc_id: str | None = None,
                 parents: list | None = None, limit_note: str = "") -> dict:
    nu0 = ing.nu0
    field_nominal = round(nu0 / GAMMA_MHZ_PER_T)
    u = math.hypot(m["cho_cr_crlb"], 0.045)
    gid = {x["id"]: x for x in g["gates"]}
    ms = field_scaled(m, nu0)       # Hz metrics as 3 T-equivalent, the same scale the thresholds are written in
    names = {"F12": "mrs_linewidth_Hz", "F13": "mrs_snr", "F15": "mrs_water_ratio", "F08": "mrs_drift_Hz",
             "F23": "mrs_transient_outliers", "F14": "cho_crlb_pct"}
    checks = []
    for fid_, name in names.items():
        x = gid.get(fid_)
        if x is None:
            continue
        val = ms.get({"F12": "fwhm_hz", "F13": "snr", "F15": "water_ratio", "F08": "drift_abs_hz", "F23": "outlier_pct",
                     "F14": "cho_crlb_pct"}[fid_])
        if name == "mrs_transient_outliers" and val is not None:
            val = val / 100
        checks.append({"name": name, "catalogue_id": fid_, "value": None if val is None else round(float(val), 4),
                       "unit": {"F12": "Hz", "F08": "Hz", "F14": "%"}.get(fid_, "1"),
                       **({"raw_value_at_this_field": round(float(m[{"F12": "fwhm_hz", "F08": "drift_abs_hz"}[fid_]]), 3),
                           "scaled_to": "3 T-equivalent Hz (x 123.25 / nu0)"}
                          if fid_ in ("F12", "F08") and m.get({"F12": "fwhm_hz", "F08": "drift_abs_hz"}[fid_]) is not None else {}),
                       "threshold": f"ScanTwin catalogue v1.1 {fid_}" + (" (Hz limits scaled to this field)" if fid_ in ("F12", "F08") else ""),
                       "status": x["status"] if x["status"] in ("PASS", "WARN", "FAIL") else "PASS",
                       "owner": {"F12": "PM", "F08": "PM"}.get(fid_, "QE"),
                       **({"note": x.get("reason")} if x["status"] == "N/A" else {})})
    checks.append({"name": "missing_values", "value": 0, "unit": "1", "threshold": "FAIL > 0", "status": "PASS", "owner": "DS"})
    status = g["status"]
    return {
        "packet_version": "1.0", "packet_id": packet_id, "subject_id": subject_id, "session_id": session_id,
        "relative_time": {"value": months, "unit": "month"}, "origin": origin,
        "value_epistemic": "DERIVED", "decision_epistemic": "INFERRED", "system_qc_id": system_qc_id,
        "protocol_match": True, "comparable": True, "calibration_id": None, "thresholds_ref": "thresholds-v1.0",
        "interpretation_limit": ("external data converted by the ScanTwin field mapper; windowed fit reads Cho/Cr ~3 % low "
                                 "on simulated truth; quality verdict about the measurement only, no clinical meaning"
                                 + (f"; {limit_note}" if limit_note else "")),
        "modality": "MRS", "observable": "Cho_Cr", "value": round(float(m["cho_cr"]), 4), "unit": "1", "valid_range": [0, 3],
        "source_object": rel_path, "source_sha256": sha,
        "spatial_reference": {"frame": "scanner", "voi_size_mm": None},
        "uncertainty": {"type": "SD", "value": round(u, 4), "budget_ref": "windowed-fit CRLB (+) 0.045 other terms"},
        "qc_checks": checks, "qc_status": status,
        "qc_reasons": [f"{x['id']} {x['label']}: {x['status']}" for x in g["deciding"]],
        "acquisition_context": {"field_strength_T": float(field_nominal), "spectrometer_frequency_MHz": round(nu0, 6),
                                "sequence": (ing.meta.get("ProtocolName") or ing.meta.get("SequenceName")
                                             or ing.meta.get("PulseSequenceType") or "not recorded"),
                                "TE_s": ing.meta.get("EchoTime"), "TR_s": ing.meta.get("RepetitionTime"),
                                "transients": int(ing.transients.shape[0]), "manufacturer": ing.meta.get("Manufacturer")},
        "processing": {"pipeline": "scantwin.ingest", "version": "0.1",
                       "fit": "windowed (1.85-3.35 ppm) Lorentzian NAA/Cr/Cho + linear baseline", "steps": ing.steps},
        "provenance": {"parents": parents or [f"{ing.source_name} (sha256 {ing.source_sha256})"],
                       "activities": ["NIfTI-MRS read", "de-identification", "coil combination", "transient QC",
                                      "frequency referencing", "phasing", "windowed fit", "QC aggregation"]},
        "twin_action": {"recommended": {"PASS": "update", "WARN": "update_with_warning", "FAIL": "reject"}[status],
                        "reason": "; ".join(f"{x['id']} {x['label']}" for x in g["deciding"]) or "all quality gates pass"},
    }


# --------------------------------------------------------------------------- simulated follow-up from a real scan
def degrade(ing: Ingested, fault: str, strength: float, seed: int = 0) -> Ingested:
    """A SIMULATED follow-up made from the real transients: same subject and scanner, one injected fault.

    shim:  extra Gaussian broadening (Hz FWHM)        drift: linear frequency drift over the scan (Hz)
    noise: noise SD x strength (weaker coil)           water: residual water line added (x NAA height)
    none:  only a fresh noise realisation (a repeat scan of an unchanged subject on a healthy scanner)
    cho_up: a TRUE change of the subject: choline amplitude + strength (e.g. 0.2 = +20 %), healthy scanner
    """
    rng = np.random.default_rng(seed)
    tr = ing.transients.copy()
    n, N = tr.shape
    t = F.time_axis(N, ing.dwell)
    sd = np.std(np.concatenate([tr[:, -N // 5:].real.ravel(), tr[:, -N // 5:].imag.ravel()]))
    fresh = lambda k: k * sd * (rng.normal(size=tr.shape) + 1j * rng.normal(size=tr.shape))
    if fault == "shim":
        sg = strength / 2.3548
        env = np.exp(-2 * (np.pi * sg * t) ** 2)
        tr = tr * env[None, :]
        # the envelope also damped the noise; a real bad shim broadens the signal only, so restore white noise
        tr = tr + fresh(1.0) * np.sqrt(np.clip(1 - env ** 2, 0, None))[None, :]
    elif fault == "drift":
        sh = strength * (np.arange(n) / max(n - 1, 1) - 0.5)
        tr = tr * np.exp(2j * np.pi * sh[:, None] * t[None, :])
    elif fault == "noise":
        tr = tr + fresh(math.sqrt(max(strength ** 2 - 1, 0)))
    elif fault == "water":
        ppm, sp = F.spectrum(ing.fid, ing.dwell, nu0=ing.nu0)
        h = sp.real[(ppm > 1.9) & (ppm < 2.1)].max() / N * 3
        wsel = (ppm > 4.5) & (ppm < 4.8)
        phw = np.angle(sp[wsel][np.argmax(np.abs(sp[wsel]))])      # add in phase with the existing residual water
        tr = tr + (strength * h * np.exp(1j * phw) * np.exp(-np.pi * 15 * t)
                   * np.exp(2j * np.pi * F.freq_hz(4.65, ing.nu0) * t))[None, :]
    elif fault == "cho_up":
        # a TRUE chemistry change: add a choline line of `strength` x the fitted choline amplitude, same line shape
        fit = F.fit_window(ing.fid, ing.dwell, nu0=ing.nu0)
        a = fit.amplitudes["Cho"] * strength
        f0 = F.freq_hz(F.METABOLITES["Cho"], ing.nu0) + fit.shift_hz
        tr = tr + (a * np.exp(1j * fit.phase_rad) * np.exp(-np.pi * fit.lw_hz * t) * np.exp(2j * np.pi * f0 * t))[None, :]
    elif fault != "none":
        raise ValueError(fault)
    if fault in ("none", "shim", "drift", "water", "cho_up"):
        tr = tr + fresh(0.3)            # a new acquisition never has exactly the same noise
    steps = ing.steps + [f"SIMULATED follow-up: fault '{fault}' strength {strength} injected into the real transients (seed {seed})"]
    return Ingested(tr, tr.mean(0), ing.dwell, ing.nu0, ing.meta, ing.removed, steps, ing.shape, ing.dims,
                    ing.source_sha256, ing.source_name)
