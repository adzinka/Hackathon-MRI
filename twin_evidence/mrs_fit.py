"""Time-domain fit of a single-voxel 1H spectrum and spectral QC metrics.

Model (StarterPack v0.3 README convention):
    s(t) = exp(i phi0) * sum_j a_j exp(-pi LW t) exp(+i 2 pi (f_j + df) t),   f_j = (delta_j - 4.65 ppm) * nu0
with one common Lorentzian linewidth LW, one common frequency shift df and one zero-order phase phi0.
Amplitudes a_j are real; their covariance (Cramer-Rao lower bound, CRLB) comes from the linear least-squares step.

The fit deliberately uses a pure Lorentzian line: when the true line is not Lorentzian (poor shim, drift, motion),
the fit has a model error the CRLB cannot see. That gap is the reason for the WARN policy (Dossier §4.6, §7.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NU0_MHZ = 123.25
PPM_REF = 4.65
DWELL_S = 0.0005
N_POINTS = 1024
# six singlet-equivalent lines of the StarterPack (CHANGELOG X1)
METABOLITES = {"NAA": 2.01, "Cr": 3.03, "Cho": 3.20, "mI": 3.56, "Glx": 3.75, "Cr2": 3.92}
WATER_PPM = 4.65


def freq_hz(ppm: float, nu0: float = NU0_MHZ) -> float:
    return (ppm - PPM_REF) * nu0


def time_axis(n: int = N_POINTS, dwell: float = DWELL_S) -> np.ndarray:
    return np.arange(n) * dwell


def spectrum(fid: np.ndarray, dwell: float = DWELL_S, zero_fill: int = 1, nu0: float = NU0_MHZ):
    """ppm axis and complex spectrum (fftshift(FFT), ppm = 4.65 + f / nu0)."""
    if zero_fill > 1:
        fid = np.concatenate([fid, np.zeros(fid.size * (zero_fill - 1), fid.dtype)])
    spec = np.fft.fftshift(np.fft.fft(fid))
    f = np.fft.fftshift(np.fft.fftfreq(fid.size, dwell))
    return PPM_REF + f / nu0, spec


def qc_metrics(fid: np.ndarray, dwell: float = DWELL_S, nu0: float = NU0_MHZ) -> dict:
    """SNR, NAA FWHM (Hz) and residual-water ratio, all from the averaged FID.

    SNR       = NAA peak height (1.9-2.1 ppm, real spectrum) / SD of the real spectrum in -3.0..0.0 ppm (README).
    FWHM      = width at half height of the NAA peak, on an 8x zero-filled spectrum.
    water_ratio = max |spectrum| in 4.5-4.8 ppm / NAA peak height (incomplete water suppression; 0 is perfect).
    """
    ppm, sp = spectrum(fid, dwell, nu0=nu0)
    re = sp.real
    sel = (ppm > 1.9) & (ppm < 2.1)
    noise = re[(ppm > -3.0) & (ppm < 0.0)].std()
    peak = re[sel].max()
    ppm8, sp8 = spectrum(fid, dwell, zero_fill=8, nu0=nu0)
    r8 = sp8.real
    s8 = np.where((ppm8 > 1.85) & (ppm8 < 2.17))[0]
    i = s8[np.argmax(r8[s8])]
    half = r8[i] / 2
    lo, hi = i, i
    while lo > 0 and r8[lo] > half:
        lo -= 1
    while hi < r8.size - 1 and r8[hi] > half:
        hi += 1
    left = np.interp(half, [r8[lo], r8[lo + 1]], [ppm8[lo], ppm8[lo + 1]])
    right = np.interp(half, [r8[hi], r8[hi - 1]], [ppm8[hi], ppm8[hi - 1]])
    water = np.abs(sp[(ppm > 4.5) & (ppm < 4.8)]).max()
    return {"snr": float(peak / noise), "fwhm_hz": float(abs(right - left) * nu0),
            "naa_ppm": float(ppm8[i]), "water_ratio": float(water / peak), "noise_sd": float(noise)}


@dataclass
class FitResult:
    amplitudes: dict
    cov: np.ndarray
    names: list
    lw_hz: float
    shift_hz: float
    phase_rad: float
    noise_sd: float
    residual_rms: float
    ratios: dict = field(default_factory=dict)      # name/Cr -> (value, crlb_sd)

    def ratio(self, num: str, den: str = "Cr"):
        a, c = self.amplitudes[num], self.amplitudes[den]
        i, j = self.names.index(num), self.names.index(den)
        r = a / c
        var = r * r * (self.cov[i, i] / a ** 2 + self.cov[j, j] / c ** 2 - 2 * self.cov[i, j] / (a * c))
        return r, math.sqrt(max(var, 0.0))


def _design(t, lw, df, phase, names, nu0=NU0_MHZ):
    cols = [np.exp(1j * phase) * np.exp(-np.pi * lw * t) * np.exp(2j * np.pi * (freq_hz(METABOLITES[n], nu0) + df) * t)
            for n in names]
    B = np.stack(cols, axis=1)
    return np.vstack([B.real, B.imag])          # real-valued design: amplitudes are real


def fit(fid: np.ndarray, dwell: float = DWELL_S, names=None, nu0: float = NU0_MHZ) -> FitResult:
    names = list(names or METABOLITES)
    t = time_axis(fid.size, dwell)
    y = np.concatenate([fid.real, fid.imag])

    # zero-order phase from an unconstrained complex fit at a rough linewidth
    Bc = np.stack([np.exp(-np.pi * 8 * t) * np.exp(2j * np.pi * freq_hz(METABOLITES[n], nu0) * t) for n in names], axis=1)
    ac, *_ = np.linalg.lstsq(Bc, fid, rcond=None)
    phase0 = float(np.angle(np.sum(ac * np.abs(ac))))

    def sse(lw, df, ph):
        B = _design(t, lw, df, ph, names, nu0)
        a, res, *_ = np.linalg.lstsq(B, y, rcond=None)
        r = y - B @ a
        return float(r @ r), a, B

    best = None
    for lw in np.arange(2.0, 40.01, 1.5):                       # coarse grid
        for df in np.arange(-16.0, 16.01, 2.0):
            e = sse(lw, df, phase0)[0]
            if best is None or e < best[0]:
                best = (e, lw, df)
    _, lw0, df0 = best
    for _ in range(2):                                           # refine LW, shift, phase
        cand = [(sse(lw, df, ph)[0], lw, df, ph)
                for lw in np.arange(max(0.5, lw0 - 1.0), lw0 + 1.01, 0.2)
                for df in np.arange(df0 - 1.5, df0 + 1.51, 0.3)
                for ph in (phase0 - 0.05, phase0, phase0 + 0.05)]
        _, lw0, df0, phase0 = min(cand)
    e, a, B = sse(lw0, df0, phase0)
    dof = y.size - len(names) - 3
    sigma2 = e / dof
    cov = sigma2 * np.linalg.inv(B.T @ B)
    res = FitResult(dict(zip(names, map(float, a))), cov, names, float(lw0), float(df0), float(phase0),
                    math.sqrt(sigma2), math.sqrt(e / y.size))
    for n in names:
        if n != "Cr":
            res.ratios[f"{n}_Cr"] = res.ratio(n)
    return res


WINDOW_PEAKS = {"NAA": 2.01, "Cr": 3.03, "Cho": 3.20}


def fit_window(fid: np.ndarray, dwell: float = DWELL_S, nu0: float = NU0_MHZ, lo: float = 1.85, hi: float = 3.35,
               peaks: dict | None = None) -> FitResult:
    """Frequency-domain fit of a ppm window: Lorentzian NAA, Cr, Cho with one linewidth, shift and phase, plus a
    complex linear baseline. The usual way to get ratios from a crowded spectrum (short TE, high field, residual water,
    macromolecules) where a full-spectrum singlet model fails. CRLB from the linear least-squares step."""
    peaks = peaks or WINDOW_PEAKS
    names = list(peaks)
    ppm, sp = spectrum(fid, dwell, zero_fill=2, nu0=nu0)
    sel = (ppm > lo) & (ppm < hi)
    f = (ppm[sel] - PPM_REF) * nu0                     # Hz axis of the window
    y = sp[sel] * dwell                                 # FFT scale ~ integral of the FID
    yy = np.concatenate([y.real, y.imag])
    x = (f - f.mean()) / (f.max() - f.min())

    def design(lw, df, ph):
        cols = []
        for n in names:
            f0 = freq_hz(peaks[n], nu0) + df
            L = 1.0 / (np.pi * lw + 2j * np.pi * (f - f0))  # FFT (exp(-i2pi f t)) of exp(-pi lw t) exp(i 2 pi f0 t)
            cols.append(np.exp(1j * ph) * L)
        cols += [np.ones_like(x) + 0j, x + 0j, 1j * np.ones_like(x), 1j * x]   # complex linear baseline
        B = np.stack(cols, axis=1)
        return np.vstack([B.real, B.imag])

    def sse(lw, df, ph):
        B = design(lw, df, ph)
        a, *_ = np.linalg.lstsq(B, yy, rcond=None)
        r = yy - B @ a
        return float(r @ r), a, B

    best = min(((sse(lw, df, ph)[0], lw, df, ph) for lw in np.arange(2.0, 60.01, 2.0)
                for df in np.arange(-20.0, 20.01, 2.0) for ph in np.linspace(-np.pi, np.pi, 24, endpoint=False)))
    _, lw0, df0, ph0 = best
    for step in (1.0, 0.3, 0.1):
        cand = [(sse(lw, df, ph)[0], lw, df, ph)
                for lw in (lw0 - step * 2, lw0 - step, lw0, lw0 + step, lw0 + step * 2) if lw > 0.5
                for df in (df0 - step, df0, df0 + step)
                for ph in (ph0 - step / 5, ph0, ph0 + step / 5)]
        _, lw0, df0, ph0 = min(cand)
    e, a, B = sse(lw0, df0, ph0)
    dof = yy.size - B.shape[1] - 3
    sigma2 = e / dof
    cov_full = sigma2 * np.linalg.pinv(B.T @ B)
    k = len(names)
    res = FitResult(dict(zip(names, map(float, a[:k]))), cov_full[:k, :k], names, float(lw0), float(df0), float(ph0),
                    float(np.sqrt(sigma2)), float(np.sqrt(e / yy.size)))
    for n in names:
        if n != "Cr":
            res.ratios[f"{n}_Cr"] = res.ratio(n)
    return res
