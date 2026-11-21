# Real test data: one T1w and one ¹H-MRS acquisition (Morelli et al., sub-01)

Genuine in-vivo acquisitions, not simulated, kept **byte-for-byte as published** so ScanTwin can be tested on real
scans. Provenance and checksums: [`dataset_manifest.json`](dataset_manifest.json). Pipeline results:
[`qc_report.json`](qc_report.json), summarised below.

## 1 · Source, licence and citation

*fMRS data of prolonged motor activation acquired at 3 Tesla*, Martin Wilson, Zenodo, 2024.
DOI [10.5281/zenodo.13753316](https://doi.org/10.5281/zenodo.13753316) · licence **CC BY 4.0** (reuse, including
commercial, with attribution; indicate changes, none were made to these files).

Data from: Morelli M, Dudzikowska K, Deelchand DK, Quinn AJ, Mullins PG, Apps MAJ, Wilson M. *Functional magnetic
resonance spectroscopy of prolonged motor activation using conventional and spectral GLM analyses.* Imaging
Neuroscience 2025; 3: imag_a_00452. doi:[10.1162/imag_a_00452](https://doi.org/10.1162/imag_a_00452)

Any slide, report or derived dataset that uses these files must cite both.

## 2 · Participant and acquisition

Participant **sub-01** (the archive has sub-01 … sub-23). The archive was inspected before selection: each participant
has `anat/sub-XX_T1w.nii.gz`, `mrs/sub-XX_svs.nii.gz` + `.json`, `mrs/sub-XX_ref.nii.gz` + `.json` and
`func/sub-XX_bold.nii.gz`. The BOLD file was not taken. Only these five members were read from the 1.68 GB archive.

| File | What it is |
|---|---|
| `T1w/sub-01_T1w.nii.gz` | Structural T1w, NIfTI-1, 208 × 256 × 256, 1.0 mm isotropic, float32, RAS. **Defaced** by the providers. No JSON sidecar in the archive. |
| `MRS/sub-01_svs.nii.gz` | Water-suppressed single-voxel ¹H-MRS, NIfTI-MRS v0.2 (NIfTI-2), 1 × 1 × 1 × 2048 × 1 × 750 complex. Siemens 3 T (123.251 MHz), sLASER (`svs_slaser_dkd`), TE 28 ms, TR 2 s, 25 mm cubic voxel, 2000 Hz spectral width, **750 individual transients** (about 25 min, motor-task fMRS). |
| `MRS/sub-01_svs.json` | Sidecar with the same acquisition fields as the NIfTI-MRS header extension. |
| `MRS/sub-01_ref.nii.gz` | **Unsuppressed water reference**, same voxel and sequence, NIfTI-MRS v0.2, 1 × 1 × 1 × 2048 (already combined to one FID). |
| `MRS/sub-01_ref.json` | Its sidecar. |

## 3 · Loading them in ScanTwin

**App:** `streamlit run app.py` → *🔎 Check a scan* → *Use an example* → pick one of
- `Real test data · Spectroscopy · 3 T sLASER, 750 repeats, sub-01 …`
- `Real test data · Brain anatomy (T1w) · 3 T, defaced, sub-01 …`

The entries appear only when the files are present, and they do not touch the upload path. The files can also be
dropped into *Upload my files* like any other scan.

**Code:**
```python
from twin_evidence import ingest as I, t1w as T
ing = I.read("external_data/real_samples/MRS/sub-01_svs.nii.gz"); m = I.measure(ing); g = I.gate(m, ing.nu0)
w = I.water_reference("external_data/real_samples/MRS/sub-01_ref.nii.gz")
s = T.load("external_data/real_samples/T1w/sub-01_T1w.nii.gz"); mt = T.measure(s); gt = T.gate(mt)
```

**Reproduce / verify:**
```bash
python scripts/fetch_real_samples.py --check   # size + SHA-256 of every file against the manifest
python scripts/fetch_real_samples.py           # re-download anything missing or wrong from Zenodo
python scripts/run_real_samples.py             # re-run both pipelines -> qc_report.json
pytest -q tests/real_samples                      # header, metadata, privacy and pipeline tests
```

## 4 · ScanTwin results on the real data (21 Nov 2026)

Produced by `scripts/run_real_samples.py` with the repository's gates unchanged; the app shows the same verdicts.

### T1w → **WARNING** (catalogue status PASS; one minimum check could not be run)

| Item | Result |
|---|---|
| Loaded | yes, 9.5 s |
| Dimensions / voxel | 208 × 256 × 256 · 1.0 × 1.0 × 1.0 mm |
| F26 motion (background ghosting) | **PASS**: rating 1 (none), ghost 0.0 % |
| F27 SNR / CNR vs baseline | N/A (needs a brain mask, and a baseline scan) |
| F28 intensity uniformity (bias field) | N/A (needs a brain mask; ScanTwin does not strip the skull) |
| F32 alignment / registration | N/A (needs a baseline scan and region outlines) |
| F35 protocol match | N/A (needs sidecars of both scans; the archive has no T1w sidecar) |
| Head volume (automatic head mask) | 2994 mL (brain / GM / WM / CSF volumes N/A without a brain mask) |

WARNING, not TRUSTY, because F28 is in the team's minimum set (`MINIMUM_T1W`, a pending team decision) and could not
be checked. Supplying a brain mask would let F27 / F28 and the tissue volumes run.

### MRS → **ERROR** (catalogue status FAIL, deciding gate F15)

| Gate | Measured | Limit (catalogue v1.1) | Status |
|---|---|---|---|
| F13 SNR | 79.9 | ≥ 25 | PASS |
| F12 linewidth (NAA FWHM, averaged spectrum) | 11.35 Hz (0.092 ppm) | ≤ 10 Hz | **WARN** |
| F14 Cho CRLB | 4.2 % | ≤ 20 % | PASS |
| F15 residual water / NAA | **11.7** | ≤ 0.5 (FAIL > 1.5) | **FAIL** |
| F08 frequency drift | 5.2 Hz | ≤ 6 Hz | PASS |
| F23 transients rejected (motion) | 0.27 % | ≤ 5 % | PASS |
| F20, F21, F35, F19, F61 | N/A | need a baseline / segmentation / phantom | N/A |

Spectral dimensions: 2048 points × 750 transients, 2000 Hz. Fit (indicative): Cho/Cr 0.669 ± 0.032,
NAA/Cr 2.11 ± 0.06. **Water reference:** available and loaded. Water FWHM 6.6 Hz (0.054 ppm) on the phased real
spectrum, 3.7 Hz off resonance. This is informational only; no gate uses it.

### Reading these results honestly

- **F15 FAIL is a real property of the scan, not a reader error.** The spectrum was checked visually (NAA at 2.01 ppm,
  Cr at 3.03, Cho at 3.2) and the residual water peak at 4.65 ppm is about 12 × the NAA height. The F15 limits are a
  **team proposal** (catalogue v1.1, checked with the simulator) and the catalogue states they hold "for a fit
  without residual-water removal". The windowed fit used for external data was found to be robust to residual water
  ([`../README.md`](../README.md)), so F15 may be stricter than needed for this pipeline. **That is an open team
  decision; the limit was not changed to make this scan pass.**
- **F12 WARN is probably caused partly by the processing.** `ingest.read` averages all 750 transients without
  frequency alignment. With 5.2 Hz drift over about 25 minutes this broadens the averaged NAA line (11.35 Hz). The
  unsuppressed water line from the same voxel is 6.6 Hz, which suggests the shim itself was good. Per-transient
  frequency alignment before averaging is not implemented.
- **Threshold status:** F13 and F12 come from the Dossier (§5.8); F14 follows the MRS literature convention; F15, F08,
  F23 and all T1w limits (F26, F27, F28) are team proposals that have not been validated on real data. This is the
  first real 3 T sLASER scan run through them.
- **fMRS context:** the 750 transients span a motor task. ScanTwin treats them as one acquisition and averages them
  all; task-related metabolite changes (a few %) are below what these quality gates look at.

## 5 · Known limitations

- One participant, one session: no follow-up scan, so every between-session gate (F20, F21, F27, F32, F35, F19) is N/A.
- No T1w sidecar (TR / TE / TI / field) and no brain mask in the archive.
- The svs header extension has no `dim_5` / `dim_6` tags. The NIfTI-MRS defaults (DIM_COIL size 1, DIM_DYN size 750)
  give the same result as the reader's fallback. The water reference's extension still declares `dim_5` / `dim_6`
  and 4 transients although the stored array is a single combined FID.
- The header extension stores scalars as one-element lists (`"EchoTime": [0.028]`). The original reader crashed on
  this; `ingest.read` now reads them as scalars in memory. The file is not changed.
- The svs JSON gives `ChemicalShiftReference` 2.4 ppm (water ref: 4.7). ScanTwin references NAA to 2.01 ppm itself,
  so it does not use this field.
- The reader found the spectrum mirrored relative to its convention (NAA at 7.3 ppm) and conjugated it, as it does
  for other vendors' data. This is recorded in the processing steps in `qc_report.json`.

## 6 · Privacy check

Checked before adding to Git (and re-checked by `tests/real_samples`): the NIfTI headers, NIfTI-MRS header extensions
and JSON sidecars contain **no** `Patient*`, institution, operator, physician, accession or device-serial fields. No
DICOM files are included. The T1w is defaced by the providers. The T1w header `descrip` field holds a processing-tool
string with a timestamp (`2203.6-dirty 2022-09-08T15:38:56+01:00`), probably the conversion time. It is not linked to
any identity and was kept unchanged to preserve the original file and its checksum.

## 7 · Storage

About 42 MB in total, the largest file 23.5 MB: below GitHub's 50 MB warning and 100 MB limit, so the files are kept in
plain Git (Git LFS is not configured in this repository and is not needed at this size). If the repository should
ever drop them, `scripts/fetch_real_samples.py` restores the identical files from Zenodo and verifies them.
