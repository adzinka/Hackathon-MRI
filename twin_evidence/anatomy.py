"""Evidence views from the StarterPack source files: 3D anatomy, ROI masks, the MRS voxel and the spectra.

Everything here is read from the files the packets point to, so the pictures are evidence, not decoration.
Geometry: NIfTI affine (RAS, mm). MRS convention: StarterPack v0.3 README (fftshift(FFT), ppm = 4.65 + f / nu0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

T1W = {"baseline": "02_NIfTI/synthetic_T1w_baseline.nii.gz", "followup": "02_NIfTI/synthetic_T1w_followup.nii.gz"}
MASK = "02_NIfTI/synthetic_brain_mask.nii.gz"
T1MAP = "02_NIfTI/synthetic_T1map_ms.nii.gz"
ROI = {s: f"03_BIDS_MRI/derivatives/roi/sub-01/ses-{s}/anat/sub-01_ses-{s}_space-T1w_label-ROI_probseg.nii.gz"
       for s in ("baseline", "followup")}
MRS = {s: f"04_MRS_BIDS/sub-01/ses-{s}/mrs/sub-01_ses-{s}_acq-press_nuc-1H_voi-frontal_svs.nii.gz"
       for s in ("baseline", "followup")}
NU0_MHZ = 123.25
PPM_REF = 4.65


def _nib():
    try:
        import nibabel as nib
    except ImportError as e:  # pragma: no cover - only hit when the optional dependency is missing
        raise ImportError("the 3D/spectrum views need nibabel: pip install nibabel") from e
    return nib


def load_volumes(root: Path) -> dict:
    """All 3D volumes on the common 64 x 64 x 16 grid, plus world coordinates (mm, RAS) of every voxel."""
    nib = _nib()
    root = Path(root)
    img = nib.load(root / MASK)
    A = img.affine
    shape = img.shape
    ijk = np.indices(shape).reshape(3, -1)
    xyz = (A[:3, :3] @ ijk + A[:3, 3:4]).reshape(3, *shape)
    get = lambda rel: np.asarray(nib.load(root / rel).dataobj, dtype=np.float32)
    vox_mm3 = float(abs(np.linalg.det(A[:3, :3])))
    return {
        "affine": A, "shape": shape, "x": xyz[0], "y": xyz[1], "z": xyz[2], "voxel_mm3": vox_mm3,
        "mask": get(MASK), "t1map": get(T1MAP),
        "t1w": {s: get(p) for s, p in T1W.items()},
        "roi": {s: get(p) for s, p in ROI.items()},
    }


def roi_volume_mm3(vol: dict, session: str) -> float:
    """ROI volume = sum of partial-volume fractions x voxel volume (dataset_description of the ROI derivative)."""
    return float(vol["roi"][session].sum() * vol["voxel_mm3"])


def world_to_index(vol: dict, xyz_mm) -> np.ndarray:
    A = vol["affine"]
    return np.linalg.solve(A[:3, :3], np.asarray(xyz_mm, float) - A[:3, 3])


def box_edges(center, size, shift=(0.0, 0.0, 0.0)):
    """Line segments (x, y, z lists with None breaks) of an axis-aligned box: for Scatter3d wireframes."""
    c = np.asarray(center, float) + np.asarray(shift, float)
    h = np.asarray(size, float) / 2
    corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    X, Y, Z = [], [], []
    for a, b in edges:
        X += [corners[a, 0], corners[b, 0], None]
        Y += [corners[a, 1], corners[b, 1], None]
        Z += [corners[a, 2], corners[b, 2], None]
    return X, Y, Z


def box_mesh(center, size):
    """Vertices and triangles of a solid box: for Mesh3d."""
    c = np.asarray(center, float)
    h = np.asarray(size, float) / 2
    v = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tri = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
           (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    i, j, k = zip(*tri)
    return v[:, 0], v[:, 1], v[:, 2], i, j, k


def spectrum(root: Path, session: str, zero_fill: int = 1):
    """Real part of the spectrum of one NIfTI-MRS file, with its ppm axis (StarterPack convention).

    zero_fill > 1 pads the FID before the FFT: a finer frequency grid for measuring peak widths
    (it interpolates; it adds no information and must not be used for the SNR)."""
    nib = _nib()
    img = nib.load(Path(root) / MRS[session])
    fid = np.asarray(img.dataobj).reshape(-1)
    if zero_fill > 1:
        fid = np.concatenate([fid, np.zeros(fid.size * (zero_fill - 1), fid.dtype)])
    dwell = float(img.header["pixdim"][4])
    spec = np.fft.fftshift(np.fft.fft(fid))
    f = np.fft.fftshift(np.fft.fftfreq(fid.size, dwell))
    return PPM_REF + f / NU0_MHZ, spec.real


def file_qc(root: Path, session: str) -> dict:
    """SNR from the spectrum as stored; FWHM from an 8x zero-filled copy (finer grid for the width)."""
    snr = spectrum_qc(*spectrum(root, session))["snr"]
    fine = spectrum_qc(*spectrum(root, session, zero_fill=8))
    return {"snr": snr, "fwhm_hz": fine["fwhm_hz"], "naa_ppm": fine["naa_ppm"]}


def spectrum_qc(ppm: np.ndarray, re: np.ndarray) -> dict:
    """Recompute the two spectral QC numbers from the raw data (README definitions).

    SNR  = NAA peak height (1.9-2.1 ppm) / SD of the real spectrum in -3.0..0.0 ppm.
    FWHM = full width at half maximum of the NAA peak, linear interpolation, in Hz.
    """
    sel = (ppm > 1.9) & (ppm < 2.1)
    noise = re[(ppm > -3.0) & (ppm < 0.0)]
    idx = np.where(sel)[0][np.argmax(re[sel])]
    peak = re[idx]
    half = peak / 2
    lo = idx
    while lo > 0 and re[lo] > half:
        lo -= 1
    hi = idx
    while hi < len(re) - 1 and re[hi] > half:
        hi += 1
    left = np.interp(half, [re[lo], re[lo + 1]], [ppm[lo], ppm[lo + 1]])
    right = np.interp(half, [re[hi], re[hi - 1]], [ppm[hi], ppm[hi - 1]])
    return {"snr": float(peak / noise.std()), "fwhm_hz": float(abs(right - left) * NU0_MHZ), "naa_ppm": float(ppm[idx])}
