"""Physics-based simulator of a single-voxel PRESS acquisition with known truth (Dossier §4.8: simulation with known truth).

Everything this module produces is SIMULATED. It reproduces the StarterPack v0.3 baseline (LW 8.7 Hz, SNR 31.2,
Cho/Cr 0.69, NAA/Cr 1.25) and then lets one inject the error sources of MRI_MRS_Essential_Errors_Explained.md plus
hardware fault classes:

  error source (file §)          parameter                     physical model
  1 low SNR / receive coil       coil_noise_factor             noise SD x factor
  2 B0 inhomogeneity / shim      shim_hz                       extra Gaussian broadening (Voigt line)
  3 patient motion               motion_frac, motion_hz        random frequency + phase jumps, partial signal loss
  4 frequency drift              drift_hz                      linear frequency drift over the transients
  5 voxel misplacement           voxel_offset_mm               tissue mix from the StarterPack T1 map
  6 incomplete water suppression water_residual                residual water line at 4.65 ppm
  7 protocol mismatch            te_ms                         T2 weighting exp(-(TE-30 ms)/T2)
  hardware: RF transmit (B1)     b1_error                      PRESS flip-angle efficiency; worse water suppression
  hardware: gradient scale       gradient_error                voxel size and position scale with (1 + g)
  hardware: static B0 offset     freq_offset_hz                all lines shift (removed by referencing)
  biology (true change)          cho_factor, naa_factor        concentration change, the only "patient" truth

Teaching assumptions (marked in the output): tissue-specific ratios (GM/WM), T2 values, B1 -> water leakage.
They are chosen to reproduce the StarterPack baseline and point in the direction of the literature; QE must check them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .mrs_fit import METABOLITES, DWELL_S, N_POINTS, WATER_PPM, freq_hz, time_axis

# --- teaching assumptions --------------------------------------------------------------------------------------
# Cr amplitude equal in GM and WM; ratios chosen so the StarterPack voxel (80 % WM, 20 % GM on a 1 mm grid) gives
# Cho/Cr 0.69 and NAA/Cr 1.25. Direction (Cho/Cr higher in WM) follows the literature; values are assumptions.
TISSUE_RATIOS = {"WM": {"Cho": 0.74, "NAA": 1.29}, "GM": {"Cho": 0.48, "NAA": 1.09}}
CR_AMP = 0.83                                   # Cr amplitude of the StarterPack baseline fit
OTHER_AMPS = {"mI": 0.356, "Glx": 0.0, "Cr2": 0.531}
T2_MS = {"NAA": 247.0, "Cr": 162.0, "Cho": 207.0, "mI": 200.0, "Glx": 180.0, "Cr2": 120.0}   # approx. 3 T literature
TE_REF_MS = 30.0
VOI_CENTER = (-16.0, 45.0, 6.0)
VOI_SIZE_MM = 20.0
BASE_SNR = 31.2
BASE_LW = 8.7


@dataclass(frozen=True)
class Acquisition:
    lw_hz: float = BASE_LW            # tissue Lorentzian linewidth (T2*) of a well-shimmed scan
    shim_hz: float = 0.0              # extra Gaussian FWHM from B0 inhomogeneity
    coil_noise_factor: float = 1.0
    n_transients: int = 64
    drift_hz: float = 0.0
    motion_frac: float = 0.0
    motion_hz: float = 0.0
    motion_signal_loss: float = 0.35
    water_residual: float = 0.0       # residual water amplitude relative to Cr
    water_lw_hz: float = 10.0
    te_ms: float = TE_REF_MS
    b1_error: float = 0.0
    gradient_error: float = 0.0
    voxel_offset_mm: tuple = (0.0, 0.0, 0.0)
    freq_offset_hz: float = 0.0
    cho_factor: float = 1.0
    naa_factor: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# --- anatomy: tissue fractions in the voxel ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _t1map(root: str):
    import nibabel as nib
    img = nib.load(Path(root) / "02_NIfTI/synthetic_T1map_ms.nii.gz")
    return np.asarray(img.dataobj, float), img.affine


def tissue_fractions(root: Path, center=VOI_CENTER, size_mm=VOI_SIZE_MM, step_mm=1.0) -> dict:
    """GM / WM / CSF / lesion / outside fractions of a cubic voxel, by sampling the T1 map on a 1 mm grid."""
    t1, A = _t1map(str(Path(root).resolve()))
    h = size_mm / 2
    g = np.arange(-h + step_mm / 2, h, step_mm)
    X, Y, Z = np.meshgrid(g + center[0], g + center[1], g + center[2], indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    ijk = np.linalg.solve(A[:3, :3], pts - A[:3, 3:4])
    ijk = np.rint(ijk).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(t1.shape)[:, None]), axis=0)
    vals = np.zeros(pts.shape[1])
    vals[inside] = t1[ijk[0, inside], ijk[1, inside], ijk[2, inside]]
    lab = {0: "outside", 850: "WM", 1200: "GM", 1450: "lesion", 3500: "CSF"}
    r = np.round(vals, -1)
    return {name: float(np.mean(r == k)) for k, name in lab.items()}


def true_amplitudes(acq: Acquisition, root: Path | None) -> tuple[dict, dict]:
    """Noise-free amplitudes at TE, plus the bookkeeping of where they came from (the known truth)."""
    g = acq.gradient_error
    center = tuple(c * (1 + g) + o for c, o in zip(VOI_CENTER, acq.voxel_offset_mm))
    size = VOI_SIZE_MM * (1 + g)
    if root is not None:
        fr = tissue_fractions(root, center, size)
    else:
        fr = {"WM": 0.76, "GM": 0.24, "CSF": 0.0, "lesion": 0.0, "outside": 0.0}
    brain = fr["WM"] + fr["GM"] + fr["lesion"]
    wm = (fr["WM"] + fr["lesion"]) / brain if brain else 0.76
    gm = fr["GM"] / brain if brain else 0.24
    cho_cr = (wm * TISSUE_RATIOS["WM"]["Cho"] + gm * TISSUE_RATIOS["GM"]["Cho"]) * acq.cho_factor
    naa_cr = (wm * TISSUE_RATIOS["WM"]["NAA"] + gm * TISSUE_RATIOS["GM"]["NAA"]) * acq.naa_factor
    vol_scale = (1 + g) ** 3 * brain              # CSF / outside carry no metabolites
    a = 90 * (1 + acq.b1_error) * math.pi / 180   # PRESS: 90 - 180 - 180, all scaled by the same B1 error
    b1_eff = math.sin(a) * math.sin(a) ** 4
    amps = {"Cr": CR_AMP, "NAA": CR_AMP * naa_cr, "Cho": CR_AMP * cho_cr, **OTHER_AMPS}
    for k in amps:
        amps[k] *= vol_scale * b1_eff * math.exp(-(acq.te_ms - TE_REF_MS) / T2_MS[k])
    truth = {"Cho_Cr_tissue": cho_cr, "NAA_Cr_tissue": naa_cr,
             "Cho_Cr_measurable": amps["Cho"] / amps["Cr"], "NAA_Cr_measurable": amps["NAA"] / amps["Cr"],
             "tissue": fr, "voxel_center_mm": center, "voxel_size_mm": size, "b1_efficiency": b1_eff,
             "signal_scale": vol_scale * b1_eff}
    return amps, truth


# --- signal generation --------------------------------------------------------------------------------------------
def _noiseless(amps: dict, lw: float, shim: float, df: float, t: np.ndarray, water=0.0, water_lw=10.0, phase=0.0):
    sig_g = shim / 2.3548                                       # Gaussian FWHM -> sigma (Hz)
    env = np.exp(-np.pi * lw * t) * np.exp(-2 * (np.pi * sig_g * t) ** 2)
    s = np.zeros_like(t, dtype=complex)
    for k, a in amps.items():
        s += a * np.exp(2j * np.pi * (freq_hz(METABOLITES[k]) + df) * t)
    s *= env
    if water:
        s += water * CR_AMP * np.exp(-np.pi * water_lw * t) * np.exp(2j * np.pi * (freq_hz(WATER_PPM) + df) * t + 0.7j)
    return s * np.exp(1j * phase)


@lru_cache(maxsize=1)
def noise_sd_per_transient(n_transients: int = 64) -> float:
    """Per-transient, per-channel noise SD that gives SNR 31.2 for the StarterPack baseline (README definition)."""
    t = time_axis()
    amps, _ = true_amplitudes(Acquisition(), None)
    s = _noiseless(amps, BASE_LW, 0.0, 0.0, t)
    from .mrs_fit import spectrum
    ppm, sp = spectrum(s)
    peak = sp.real[(ppm > 1.9) & (ppm < 2.1)].max()
    sd_avg = peak / BASE_SNR / math.sqrt(N_POINTS)              # FFT: real-part noise SD = sigma * sqrt(N)
    return sd_avg * math.sqrt(n_transients)


@dataclass
class Simulated:
    transients: np.ndarray            # (n_transients, N) complex
    acq: Acquisition
    truth: dict
    shifts_hz: np.ndarray             # true per-transient frequency offsets
    motion_mask: np.ndarray

    def average(self) -> np.ndarray:
        return self.transients.mean(axis=0)


def simulate(acq: Acquisition, root: Path | None = None, seed: int = 0) -> Simulated:
    rng = np.random.default_rng(seed)
    t = time_axis()
    amps, truth = true_amplitudes(acq, root)
    n = acq.n_transients
    sd = noise_sd_per_transient(64) * acq.coil_noise_factor
    shifts = acq.freq_offset_hz + acq.drift_hz * (np.arange(n) / max(n - 1, 1) - 0.5)
    phases = np.zeros(n)
    gains = np.ones(n)
    motion = rng.random(n) < acq.motion_frac
    shifts = shifts + motion * rng.normal(0, max(acq.motion_hz, 1e-9), n)
    phases = phases + motion * rng.uniform(-math.pi / 2, math.pi / 2, n)
    gains = gains - motion * acq.motion_signal_loss
    water = acq.water_residual + 12.0 * abs(acq.b1_error)       # assumption: CHESS leaks with flip-angle error
    out = np.empty((n, t.size), complex)
    for i in range(n):
        s = _noiseless(amps, acq.lw_hz, acq.shim_hz, shifts[i], t, water, acq.water_lw_hz, phases[i]) * gains[i]
        out[i] = s + sd * (rng.normal(size=t.size) + 1j * rng.normal(size=t.size))
    # F61: transmitter reference amplitude measured on the phantom vs its baseline (1.5 % measurement scatter)
    ref_dev = 100 * abs(1 - truth["b1_efficiency"]) + abs(rng.normal(0, 1.5))
    truth = {**truth, "water_residual_total": water, "noise_sd_per_transient": sd, "ref_amp_dev_pct": ref_dev}
    return Simulated(out, acq, truth, shifts, motion)


# --- transient analysis (needs individual transients, as NIfTI-MRS DIM_DYN) -----------------------------------------
def transient_shifts(transients: np.ndarray, search_hz: float = 30.0, step_hz: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Per-transient frequency offset by maximising |correlation| with the mean (apodised) transient."""
    t = time_axis(transients.shape[1])
    apod = np.exp(-np.pi * 5.0 * t)
    ref = transients.mean(axis=0) * apod
    grid = np.arange(-search_hz, search_hz + 1e-9, step_hz)
    E = np.exp(-2j * np.pi * grid[:, None] * t[None, :])              # (G, N)
    X = transients * apod * np.conj(ref)                               # (n, N)
    C = np.abs(X @ E.T)                                                # (n, G)
    best = grid[np.argmax(C, axis=1)]
    corr = C.max(axis=1) / (np.linalg.norm(transients * apod, axis=1) * np.linalg.norm(ref) + 1e-12)
    return best, corr


def _robust_shifts(transients: np.ndarray):
    """Per-transient shifts split into a smooth linear trend (drift) and flagged outliers (motion)."""
    shifts, corr = transient_shifts(transients)
    n = shifts.size
    x = np.arange(n) / max(n - 1, 1) - 0.5
    keep = np.ones(n, bool)
    for _ in range(3):                                    # iteratively re-fit the trend without outliers
        slope, icpt = np.polyfit(x[keep], shifts[keep], 1)
        resid = shifts - (slope * x + icpt)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) * 1.4826
        out = (np.abs(resid) > max(2.5, 3.5 * mad)) | (corr < 0.6 * np.median(corr))
        keep = ~out
    trend = slope * x + icpt
    return shifts, trend, out, float(slope), resid


def align_and_average(transients: np.ndarray) -> np.ndarray:
    """Remove the drift trend from all transients and the individual jump from outlier transients, then average."""
    shifts, trend, out, _, _ = _robust_shifts(transients)
    use = np.where(out, shifts, trend)
    t = time_axis(transients.shape[1])
    return (transients * np.exp(-2j * np.pi * use[:, None] * t[None, :])).mean(axis=0)


def transient_qc(transients: np.ndarray) -> dict:
    """Drift (Hz over the acquisition) and fraction of outlier transients (motion), from the transients."""
    shifts, trend, out, slope, resid = _robust_shifts(transients)
    return {"drift_hz": slope, "transient_outlier_frac": float(out.mean()), "shift_sd_hz": float(resid[~out].std())}


def write_nifti_mrs(fid: np.ndarray, path: Path, extra: dict | None = None) -> None:
    """Averaged FID as a minimal NIfTI-MRS-like file (shape 1x1x1xN, pixdim[4] = dwell), like the StarterPack."""
    import nibabel as nib
    data = fid.astype(np.complex64).reshape(1, 1, 1, -1)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header["pixdim"][4] = DWELL_S
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    if extra is not None:
        import json
        path.with_suffix("").with_suffix(".json").write_text(json.dumps(extra, indent=2, default=float), encoding="utf-8")
