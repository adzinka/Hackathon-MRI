"""Structural T1-weighted MRI: read, de-identify, segment, measure quality and volumes, simulate follow-ups.

Quality metrics map to the ScanTwin catalogue's T1w rows:
  F26 subject motion        -> ghost signal in the background along the phase-encode axis (rating 1-4, team proposal)
  F27 low SNR / CNR         -> grey/white contrast-to-noise as % of the baseline scan (needs a baseline)
  F28 intensity bias field  -> coefficient of variation of white-matter intensity (%)
  F32 registration error    -> rigid shift between sessions found by phase correlation (mm)
  F33 stated vs implied u   -> volume uncertainty carried from the registration error (3 sigma_b / r_eq)
  F35 protocol mismatch     -> TR / TE / TI / flip angle / field from the sidecar
Noise is measured inside white matter (as MRIQC's snr_wm), so it also works when the background was masked to zero.
Volumes: brain (mask), CSF / GM / WM (k-means on the brain voxels), region of interest (probabilistic mask).
All simulated follow-ups are labelled SIMULATED; "true change" is the only one that changes the anatomy.

Limits: k-means volumes, no skull stripping (on real data the automatic mask is a head mask; supply a brain mask);
the polynomial bias correction handles realistic non-uniformity (about +-50 %), not extreme fields, which FAIL F28.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

IDENT_PREFIX = ("Patient",)
IDENT_KEYS = {"InstitutionAddress", "DeviceSerialNumber", "OperatorsName", "ReferringPhysicianName", "AccessionNumber",
              "StudyID", "PerformingPhysicianName", "InstitutionName"}
KEEP = {"PatientPosition"}
PROTOCOL_KEYS = ("RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "MagneticFieldStrength")


@dataclass
class T1Scan:
    data: np.ndarray            # float image
    affine: np.ndarray
    zooms: tuple
    mask: np.ndarray            # brain mask (bool)
    roi: np.ndarray | None      # probabilistic region mask (0..1) or None
    meta: dict
    removed: list
    steps: list = field(default_factory=list)
    source_sha256: str = ""
    name: str = ""
    labels: np.ndarray | None = None      # 0 bg, 1 CSF, 2 GM, 3 WM
    bias: np.ndarray | None = None        # estimated multiplicative bias field (1 = none)
    bg_masked: bool = False               # background set to exactly zero by the source pipeline

    @property
    def voxel_mm3(self) -> float:
        return float(np.prod(self.zooms))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def deidentify(meta: dict) -> tuple[dict, list]:
    clean, removed = {}, []
    for k, v in meta.items():
        if k not in KEEP and (k.startswith(IDENT_PREFIX) or k in IDENT_KEYS):
            removed.append(k)
        else:
            clean[k] = v
    return clean, sorted(removed)


def _auto_mask(d: np.ndarray) -> np.ndarray:
    """Head/brain mask: exact non-zero support if the background was masked, else Otsu + largest component + fill."""
    if (d == 0).mean() > 0.2:
        m = d > 0
    else:
        from skimage.filters import threshold_otsu
        m = d > threshold_otsu(d)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def load(image: Path | bytes, *, name: str = "t1w.nii.gz", mask=None, roi=None, sidecar: dict | None = None) -> T1Scan:
    """Read a T1w NIfTI (path or bytes), plus optional brain mask, ROI mask (paths/bytes/arrays) and BIDS sidecar dict."""
    import nibabel as nib
    import tempfile

    def _nii(x, nm):
        if x is None or isinstance(x, np.ndarray):
            return x, None
        b = x if isinstance(x, bytes) else Path(x).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / Path(nm).name
            p.write_bytes(b)
            img = nib.load(str(p))
            return np.asarray(img.dataobj, dtype=float), (img, b)

    d, (img, raw) = _nii(image, name)
    if d.ndim == 4 and d.shape[3] == 1:
        d = d[..., 0]
    if d.ndim != 3:
        raise ValueError(f"expected a 3D T1-weighted volume, got shape {d.shape}")
    if np.iscomplexobj(d):
        raise ValueError("this looks like a spectroscopy file (complex data), not a T1-weighted image")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    meta, removed = deidentify(sidecar or {})
    steps = [f"read {Path(name).name}: {d.shape[0]}×{d.shape[1]}×{d.shape[2]} voxels of "
             f"{zooms[0]:.2f}×{zooms[1]:.2f}×{zooms[2]:.2f} mm",
             f"de-identify: removed {len(removed)} identifying field(s): {', '.join(removed) or 'none'}"]
    mk, _ = _nii(mask, "mask.nii.gz")
    if mk is not None:
        mk = mk > 0.5
        steps.append("brain mask: provided")
    else:
        mk = _auto_mask(d)
        steps.append("brain mask: estimated (non-zero support or Otsu threshold + largest component); "
                     "no skull stripping, so on real data it is a head mask")
    rr, _ = _nii(roi, "roi.nii.gz")
    if rr is not None:
        steps.append(f"region of interest: provided ({(rr > 0.5).sum()} voxels above 0.5)")
    scan = T1Scan(d, img.affine, zooms, mk, rr, meta, removed, steps, _sha(raw), Path(name).name,
                  bg_masked=bool((d[~ndimage.binary_dilation(mk, iterations=2)] == 0).mean() > 0.9))
    segment(scan)
    return scan


def _kmeans(v: np.ndarray, k: int, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    c = np.quantile(v, np.linspace(0.15, 0.85, k))
    for _ in range(iters):
        a = np.argmin(np.abs(v[:, None] - c[None, :]), axis=1)
        new = np.array([v[a == j].mean() if np.any(a == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return np.argmin(np.abs(v[:, None] - c[None, :]), axis=1), c


def _poly_basis(shape, mask):
    idx = np.argwhere(mask).astype(float)
    x = (idx - (np.array(shape) - 1) / 2) / (np.array(shape) / 2)
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([np.ones_like(X), X, Y, Z, X * X, Y * Y, Z * Z, X * Y, X * Z, Y * Z], 1)


def segment(scan: T1Scan, k: int = 3, rounds: int = 3) -> np.ndarray:
    """k-means on brain intensities -> 1 CSF, 2 GM, 3 WM, alternating with a smooth (2nd-order polynomial) bias-field
    fit on log white-matter intensity, so a brightness gradient does not turn white matter into 'CSF'."""
    d, mask = scan.data, scan.mask
    B = _poly_basis(d.shape, mask)
    logb = np.zeros(mask.sum())
    for _ in range(rounds):
        v = d[mask] / np.exp(logb)
        a, c = _kmeans(v, k)
        wm = a == int(np.argmax(c))
        if wm.sum() > B.shape[1] * 5:
            coef, *_ = np.linalg.lstsq(B[wm], np.log(np.clip(d[mask][wm], 1e-6, None)), rcond=None)
            logb = B @ coef
            logb -= logb[wm].mean()
    v = d[mask] / np.exp(logb)
    a, c = _kmeans(v, k)
    order = np.argsort(c)
    remap = np.empty(k, np.uint8)
    remap[order] = np.arange(1, k + 1)
    lab = np.zeros(d.shape, np.uint8)
    lab[mask] = remap[a]
    bias = np.ones(d.shape)
    bias[mask] = np.exp(logb)
    scan.labels, scan.bias = lab, bias
    scan.steps = [s_ for s_ in scan.steps if not s_.startswith("segment:")] + [
        f"segment: bias-corrected k-means into CSF / GM / WM (centres {', '.join(f'{x:.0f}' for x in np.sort(c))})"]
    return lab


def _ghost_pct(scan: T1Scan, noise: float, pe_axis: int = 1) -> tuple[float | None, str]:
    """Motion ghosts: signal in the background beyond what noise explains, as % of white matter.
    Preferred: phase-encode band vs frequency-encode band (ghosts run along the phase-encode axis). Fallback when the
    field of view is too tight for bands: whole background vs the expected noise floor (0 if the source masked it)."""
    d, m = scan.data, ndimage.binary_dilation(scan.mask, iterations=2)
    bg = ~m
    wm = d[scan.labels == 3].mean()
    if bg.sum() < 50:
        return None, "no background in the field of view"
    vals = np.abs(d[bg])
    if (vals < 1e-9).mean() > 0.95:
        return 0.0, "background masked to zero by the source and free of ghosts"
    proj = m.any(axis=2)
    xs, ys = np.where(proj.any(axis=1))[0], np.where(proj.any(axis=0))[0]
    pe_band, fe_band = np.zeros_like(proj), np.zeros_like(proj)
    pe_band[xs.min():xs.max() + 1, :] = True
    fe_band[:, ys.min():ys.max() + 1] = True
    if pe_axis != 1:
        pe_band, fe_band = fe_band, pe_band
    pe = (pe_band & ~fe_band)[:, :, None] & bg
    fe = (fe_band & ~pe_band)[:, :, None] & bg
    if pe.sum() >= 200 and fe.sum() >= 200:
        return float(100 * (np.abs(d[pe]).mean() - np.abs(d[fe]).mean()) / wm), "phase- vs frequency-encode background"
    floor = 0.0 if scan.bg_masked else noise * math.sqrt(math.pi / 2)
    return float(100 * max(0.0, vals.mean() - floor) / wm), "background above the noise floor"


def motion_rating(ghost_pct: float | None) -> int | None:
    """Catalogue F26 rating (1 none - 4 severe) from the ghost level; team-proposal mapping 1 % / 3 % / 6 %."""
    if ghost_pct is None:
        return None
    return 1 if ghost_pct < 1 else 2 if ghost_pct < 3 else 3 if ghost_pct < 6 else 4


def phase_shift_mm(a: np.ndarray, b: np.ndarray, zooms) -> np.ndarray:
    """Rigid translation of b relative to a by phase correlation, sub-voxel (parabolic peak), in mm."""
    A, B = np.fft.fftn(a - a.mean()), np.fft.fftn(b - b.mean())
    R = A * np.conj(B)
    r = np.fft.ifftn(R / (np.abs(R) + 1e-12)).real
    peak = np.array(np.unravel_index(np.argmax(r), r.shape))
    shift = peak.astype(float)
    for ax in range(3):
        n = r.shape[ax]
        im, ip = peak.copy(), peak.copy()
        im[ax], ip[ax] = (peak[ax] - 1) % n, (peak[ax] + 1) % n
        y0, ym, yp = r[tuple(peak)], r[tuple(im)], r[tuple(ip)]
        den = ym - 2 * y0 + yp
        shift[ax] += 0.5 * (ym - yp) / den if abs(den) > 1e-12 else 0
        if shift[ax] > n / 2:
            shift[ax] -= n
    return -shift * np.asarray(zooms)


def measure(scan: T1Scan, baseline: "T1Scan | None" = None, baseline_m: dict | None = None) -> dict:
    d, lab, v3 = scan.data, scan.labels, scan.voxel_mm3
    wm, gm = d[lab == 3], d[lab == 2]
    # noise inside white matter, away from tissue borders (partial volume): robust SD of the residual after removing
    # the local 3x3x3 mean; the factor sqrt(27/26) corrects for the voxel's own share of that mean
    core = ndimage.binary_erosion(lab == 3, iterations=1)
    if core.sum() < 50:
        core = lab == 3
    sm = ndimage.uniform_filter(d, size=3)
    resid = (d - sm)[core]
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) * math.sqrt(27 / 26)
    m = {"snr_wm": float(wm.mean() / noise), "cnr": float((wm.mean() - gm.mean()) / noise), "noise": float(noise),
         "wm_cv_pct": float(100 * scan.bias[lab == 3].std() / scan.bias[lab == 3].mean()),
         "brain_ml": float(scan.mask.sum() * v3 / 1000),
         "csf_ml": float((lab == 1).sum() * v3 / 1000), "gm_ml": float((lab == 2).sum() * v3 / 1000),
         "wm_ml": float((lab == 3).sum() * v3 / 1000),
         "roi_mm3": None if scan.roi is None else float(scan.roi.sum() * v3)}
    m["ghost_pct"], m["ghost_method"] = _ghost_pct(scan, noise)
    m["motion_rating"] = motion_rating(m["ghost_pct"])
    # volume uncertainty: single scan -> segmentation repeatability 3 % (StarterPack baseline); a follow-up whose mask
    # was propagated through a registration carries 3 sigma_b / r_eq (Dossier §3.10, F33)
    m.update({"cnr_pct_baseline": None, "reg_shift_mm": None, "protocol_diff": None, "image_change_sd": None})
    if m["roi_mm3"]:
        r_eq = (3 * m["roi_mm3"] / (4 * math.pi)) ** (1 / 3)
        m["roi_r_eq_mm"] = r_eq
        m["roi_u_rel"] = 0.03
    if baseline is not None:
        bm = baseline_m or measure(baseline)
        m["cnr_pct_baseline"] = 100 * m["cnr"] / bm["cnr"]
        sh = phase_shift_mm(baseline.data, d, scan.zooms)
        m["reg_shift_mm"] = float(np.linalg.norm(sh))
        if scan.roi is not None and m["roi_mm3"]:
            m["roi_u_rel"] = max(0.03, 3 * m["reg_shift_mm"] / m["roi_r_eq_mm"])
        diff = (d - baseline.data)[scan.mask & baseline.mask]
        m["image_change_sd"] = float(diff.std() / (math.sqrt(2) * noise))      # 1 = noise only
        bp = {k: baseline.meta.get(k) for k in PROTOCOL_KEYS}
        cp = {k: scan.meta.get(k) for k in PROTOCOL_KEYS}
        known = [k for k in PROTOCOL_KEYS if bp[k] is not None and cp[k] is not None]
        m["protocol_diff"] = float(sum(abs(bp[k] - cp[k]) > 1e-9 for k in known)) if known else None
        m["volume_ratios"] = {k: m[k] / bm[k] for k in ("brain_ml", "gm_ml", "wm_ml", "csf_ml") if bm[k]}
        m["head_size_change_pct"] = 100 * (m["brain_ml"] / bm["brain_ml"] - 1)
    return m


def gate(m: dict) -> dict:
    from .gates import evaluate_t1w
    return evaluate_t1w(m)


# --------------------------------------------------------------------------------- simulated follow-ups
def degrade(scan: T1Scan, fault: str, strength: float, seed: int = 0) -> T1Scan:
    """SIMULATED follow-up from this scan (same anatomy unless fault == 'grow').

    none   fresh noise only (repeat scan)              noise  extra noise, x strength (weaker coil)
    bias   smooth brightness gradient, strength = max change (e.g. 0.3 = 30 %)
    motion k-space phase errors on a fraction `strength` of phase-encode lines (ghosts)
    scale  gradient calibration error: image scaled by (1 + strength) about the centre (all volumes change together)
    shift  head position moved by `strength` mm (a registration problem if the region mask is not re-drawn)
    grow   TRUE change: the region of interest grows by `strength` mm at its border (patient changed)
    """
    rng = np.random.default_rng(seed)
    d = scan.data.copy()
    mask = scan.mask.copy()
    roi = None if scan.roi is None else scan.roi.copy()
    noise0 = measure(scan)["noise"]
    fresh = lambda k: k * noise0 * rng.normal(size=d.shape)
    inside = scan.mask
    if fault == "noise":
        d = d + fresh(math.sqrt(max(strength ** 2 - 1, 0))) * inside
    elif fault == "bias":
        x = np.linspace(-1, 1, d.shape[0])[:, None, None]
        y = np.linspace(-1, 1, d.shape[1])[None, :, None]
        d = d * (1 + strength * (0.6 * x + 0.4 * y ** 2))
    elif fault == "motion":
        k = np.fft.fft2(d, axes=(0, 1))
        lines = rng.random(d.shape[1]) < strength
        ph = np.exp(2j * np.pi * np.fft.fftfreq(d.shape[1])[:, None] * rng.normal(0, 3, size=(d.shape[1], 1)))
        k[:, lines, :] *= ph[lines][None, :, :]
        d = np.abs(np.fft.ifft2(k, axes=(0, 1)))
    elif fault in ("scale", "shift"):
        c = (np.array(d.shape) - 1) / 2
        if fault == "scale":
            mat, off = np.eye(3) / (1 + strength), c - c / (1 + strength)
        else:
            mat, off = np.eye(3), -np.array([strength / scan.zooms[0], 0.0, 0.0])
        d = ndimage.affine_transform(d, mat, off, order=1)
        mask = ndimage.affine_transform(mask.astype(float), mat, off, order=0) > 0.5
        if roi is not None and fault == "scale":
            roi = ndimage.affine_transform(roi, mat, off, order=1)
    elif fault == "grow":
        if roi is None:
            raise ValueError("a true change needs a region of interest")
        it = max(1, int(round(strength / min(scan.zooms))))
        core = roi > 0.5
        grown = ndimage.binary_dilation(core, iterations=it) & scan.mask
        ring = grown & ~core
        lesion_level = d[core].mean()
        d[ring] = lesion_level + rng.normal(0, noise0, ring.sum())
        roi = np.maximum(roi, ndimage.gaussian_filter(grown.astype(float), 0.5) * grown)
    elif fault != "none":
        raise ValueError(fault)
    d = d + fresh(0.3) * inside           # every new acquisition has its own noise
    if (scan.data == 0).mean() > 0.2 and fault not in ("motion",):
        d = d * mask                      # keep the masked-background convention of the source
    out = T1Scan(d, scan.affine, scan.zooms, mask, roi, dict(scan.meta), scan.removed,
                 scan.steps[:2] + [f"SIMULATED follow-up: '{fault}' strength {strength} (seed {seed})"],
                 scan.source_sha256, scan.name, bg_masked=scan.bg_masked)
    segment(out)
    return out


# --------------------------------------------------------------------------------- packet
def build_packet(scan: T1Scan, m: dict, g: dict, *, packet_id: str, subject_id: str, session_id: str, months: float,
                 rel_path: str, sha: str, origin: str = "MEASURED", system_qc_id: str | None = None,
                 limit_note: str = "") -> dict:
    use_roi = m.get("roi_mm3") is not None
    value = m["roi_mm3"] if use_roi else m["brain_ml"] * 1000
    u = value * (m.get("roi_u_rel", 0.03) if use_roi else 0.03)
    checks = [{"name": {"F26": "motion_rating", "F27": "cnr_pct_baseline", "F28": "wm_cv_pct", "F32": "registration_rms_mm",
                        "F35": "protocol_diff"}.get(x["id"], x["id"]),
               "catalogue_id": x["id"], "value": None if x["value"] is None else round(float(x["value"]), 4),
               "unit": x["unit"] or "1", "threshold": f"ScanTwin catalogue v1.1 {x['id']}",
               "status": x["status"] if x["status"] in ("PASS", "WARN", "FAIL") else "PASS", "owner": "PM"}
              for x in g["gates"] if x["applies"]]
    checks.append({"name": "missing_values", "value": 0, "unit": "1", "threshold": "FAIL > 0", "status": "PASS", "owner": "DS"})
    status = g["status"]
    f = scan.meta.get("MagneticFieldStrength")
    return {
        "packet_version": "1.0", "packet_id": packet_id, "subject_id": subject_id, "session_id": session_id,
        "relative_time": {"value": months, "unit": "month"}, "origin": origin, "value_epistemic": "DERIVED",
        "decision_epistemic": "INFERRED", "system_qc_id": system_qc_id, "protocol_match": not m.get("protocol_diff"),
        "comparable": True, "calibration_id": None, "thresholds_ref": "thresholds-v1.0",
        "interpretation_limit": "ScanTwin T1w check; k-means volumes without skull stripping; quality verdict about the "
                                "measurement only, no clinical meaning" + (f"; {limit_note}" if limit_note else ""),
        "modality": "MRI", "observable": "roi_volume" if use_roi else "brain_volume", "value": round(float(value), 1),
        "unit": "mm3", "valid_range": [0, 2000000], "source_object": rel_path, "source_sha256": sha,
        "spatial_reference": {"frame": "scanner", "voxel_size_mm": list(scan.zooms),
                              "registration_rms_mm": m.get("reg_shift_mm")},
        "uncertainty": {"type": "SD", "value": round(float(u), 1), "relative": round(float(u / value), 4),
                        "budget_ref": "segmentation repeatability 3 % or registration-implied 3 sigma_b / r_eq"},
        "qc_checks": checks, "qc_status": status,
        "qc_reasons": [f"{x['id']} {x['label']}: {x['status']}" for x in g["deciding"]],
        "acquisition_context": {"field_strength_T": f, "spectrometer_frequency_MHz": scan.meta.get("ImagingFrequency"),
                                "sequence": scan.meta.get("PulseSequenceType") or scan.meta.get("ScanningSequence") or "not recorded",
                                "TE_s": scan.meta.get("EchoTime"), "TR_s": scan.meta.get("RepetitionTime"),
                                "TI_s": scan.meta.get("InversionTime"), "flip_deg": scan.meta.get("FlipAngle")},
        "processing": {"pipeline": "scantwin.t1w", "version": "0.1", "steps": scan.steps},
        "provenance": {"parents": [f"{scan.name} (sha256 {scan.source_sha256})"],
                       "activities": ["NIfTI read", "de-identification", "brain mask", "k-means segmentation",
                                      "quality metrics", "volumes"]},
        "twin_action": {"recommended": {"PASS": "update", "WARN": "update_with_warning", "FAIL": "reject"}[status],
                        "reason": "; ".join(f"{x['id']} {x['label']}" for x in g["deciding"]) or "all quality gates pass"},
    }


# --------------------------------------------------------------------------------- patient, machine, session or processing?
MACHINE_GATES = {"F27": "the image got noisier (receive coil / RF chain)", "F28": "the brightness became uneven (coil / transmitter)"}
SIDE_EFFECT = {"F27": "made the image noisier", "F28": "made the brightness uneven"}
SESSION_GATES = {"F26": "the patient moved during the scan", "F32": "the head was positioned differently and the scans do not line up",
                 "F35": "the scan settings differ from last time"}


def compare(m0: dict, g0: dict, m1: dict, g1: dict) -> dict:
    """Who explains the change between two T1w scans? Returns (status colour, title, text, kind, z)."""
    use_roi = m0.get("roi_mm3") is not None and m1.get("roi_mm3") is not None
    key = "roi_mm3" if use_roi else "brain_ml"
    v0, v1 = m0[key], m1[key]
    u0 = v0 * (m0.get("roi_u_rel", 0.03) if use_roi else 0.03)
    u1 = v1 * (m1.get("roi_u_rel", 0.03) if use_roi else 0.03)
    z = (v1 - v0) / math.hypot(u0, u1)
    # every check that got worse (WARN or FAIL), not only the ones that decided the verdict: motion may only WARN while
    # the contrast it ruined FAILs, and motion is still the explanation
    before = {y["id"] for y in g0["gates"] if y["status"] in ("WARN", "FAIL")}
    new_bad = [x for x in g1["gates"] if x["status"] in ("WARN", "FAIL") and x["id"] not in before]
    pct = 100 * (v1 / v0 - 1)
    what = "region" if use_roi else "brain"
    if new_bad:
        ids = [x["id"] for x in new_bad]
        sess = [SESSION_GATES[i] for i in ids if i in SESSION_GATES]
        mach = [MACHINE_GATES[i] for i in sorted(ids, key=lambda i: i != "F28") if i in MACHINE_GATES]   # most specific first
        fail = g1["status"] == "FAIL"
        if sess:      # motion or positioning also degrades contrast; they explain the quality loss first
            return {"status": "FAIL" if fail else "WARN", "kind": "session", "z": z, "pct": pct,
                    "title": "🎯 This session went wrong" if fail else "🎯 Probably this session",
                    "text": f"The later scan is not comparable: {' and '.join(sess)}"
                            + (", which also " + " and ".join(SIDE_EFFECT[i] for i in ids if i in SIDE_EFFECT) if mach else "")
                            + ". "
                            + ("The result is not used. " if fail else "It counts at reduced weight. ")
                            + "Repeat the scan with care; this says nothing about the patient."}
        return {"status": "FAIL" if fail else "WARN", "kind": "machine", "z": z, "pct": pct,
                "title": "🛠 It's the machine" if fail else "🛠 Probably the machine",
                "text": f"The scan quality fell: {' and '.join(mach + sess)}. "
                        + ("The result is not used. " if fail else "It counts at reduced weight. ")
                        + "Check the scanner before reading any change."}
    head = m1.get("head_size_change_pct")
    if head is not None and abs(head) > 2.0:
        return {"status": "WARN", "kind": "machine", "z": z, "pct": pct, "title": "🛠 Probably the machine (geometry)",
                "text": f"The whole head measures {head:+.1f} % in size. Real brains do not change size like that in months, "
                        "but a gradient calibration error does (catalogue F01). Check the scanner geometry with a phantom "
                        "before believing any volume change."}
    img_same = (m1.get("image_change_sd") or 0) < 1.5 and all(abs(r - 1) < 0.02 for r in (m1.get("volume_ratios") or {}).values())
    if abs(z) >= 1.96 and img_same:
        return {"status": "WARN", "kind": "processing", "z": z, "pct": pct, "title": "⚙️ The processing, not the patient",
                "text": f"The {what} outline changed by {pct:+.0f} %, but the images themselves did not change beyond noise and "
                        "the scans line up. The difference comes from how the outline was drawn (mask propagation, "
                        "segmentation version; catalogue F33, F34). Re-draw both outlines with the same method."}
    if abs(z) < 1.96:
        return {"status": "N/A", "kind": "unknown", "z": z, "pct": pct, "title": "❔ Can't tell: no real change",
                "text": f"The {what} volume differs by {pct:+.1f} %, which is smaller than these scans can measure. "
                        "Nothing is claimed about the patient."}
    return {"status": "PASS", "kind": "patient", "z": z, "pct": pct, "title": "🧑 The patient changed",
            "text": f"Both scans have good, comparable quality, the scanner geometry is stable and the images themselves "
                    f"changed: the {what} volume is {pct:+.0f} % (beyond measurement error). Confirm with a repeat scan "
                    "before anyone acts on it."}


def write_image(scan: T1Scan, path: Path) -> str:
    """De-identified image (no sidecar) for a packet's source file; returns its SHA-256."""
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(scan.data.astype(np.float32), scan.affine), str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()
