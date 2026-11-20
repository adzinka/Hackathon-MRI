# External data: real scans converted by the ScanTwin field mapper

## zenodo_5085449_example_02 (REAL in-vivo brain spectrum + SIMULATED follow-ups)

**Source and licence.** `example_02.nii.gz` from *NIfTI-MRS example data* (Zenodo record 5085449), by William T. Clarke,
University of Oxford, licensed **CC BY 4.0** (reuse, including commercial, with attribution).
https://zenodo.org/records/5085449 · original file SHA-256 `172e6117887ce55aa2677df85801cec38a62dd4fbc50c58b9b650ac5c2afc356`

**The raw file is not in this repository.** Its metadata contained six identifying fields (name code, date of birth,
sex, weight, institution address, device serial). The converter removed them (catalogue F59); this folder holds only
de-identified, coil-combined, averaged derivatives. Do not commit the raw file.

**Scan.** Siemens 7 T (297.22 MHz), single voxel STEAM, TE 11 ms, TR 5 s, 32 coils × 64 transients, 2048 points.

**Reproduce.** Download the file yourself (Zenodo may block automated downloads), then:
```bash
python -I scripts/ingest_mrs.py path/to/example_02.nii.gz --out external_data/zenodo_5085449_example_02 \
       --subject sub-ext01 --assume-system-qc
```
`--assume-system-qc` marks the missing site QC as `ASSUMED:external-dataset`; the validator turns that into a visible
WARN ("system QC assumed, not verified"). Without it every packet FAILs on the missing system QC, which is also correct.

### Results (`summary.csv`, packets `E-t*.json`, `conversion_report.json`)

| Session | Origin | What | SNR | Linewidth | Water / NAA | Drift | Cho/Cr | Gates | Twin |
|---|---|---|---|---|---|---|---|---|---|
| t0 | **MEASURED** | the real scan | 171 | 0.054 ppm | 1.11 | 1.5 Hz | 0.514 ± 0.060 | WARN (F15) | baseline |
| t1-none | SIMULATED | repeat scan, healthy scanner, same subject | 160 | 0.054 ppm | 1.11 | 1.4 Hz | 0.512 | WARN (F15) | update, **z = −0.02: no change** |
| t1-shim | SIMULATED | poor shim (+22 Hz) | 97 | **0.147 ppm** | 1.10 | 1.8 Hz | 0.471 | WARN (F12) | down-weighted |
| t1-drift | SIMULATED | drift 25 Hz over the scan | 101 | 0.123 ppm | 1.18 | **23.3 Hz** | 0.479 | WARN (F08, F12) | down-weighted |
| t1-noise | SIMULATED | weak coil (noise × 8) | **19** | 0.059 ppm | 1.12 | 1.0 Hz | 0.514 ± 0.073 | WARN (F13) | down-weighted |
| t1-water | SIMULATED | water suppression failed | 162 | 0.054 ppm | **1.99** | 1.0 Hz | 0.513 | **FAIL (F15)** | **rejected** |

All SIMULATED follow-ups are the real transients with one injected fault (`twin_evidence/ingest.degrade`), so the
subject's true chemistry is known to be unchanged: every Cho/Cr change above is the machine.

### What to know

- **Field scaling.** Catalogue limits in Hz (F12 linewidth, F08 drift) are written for 3 T. The converter compares
  3 T-equivalent Hz (× 123.25 / ν0), i.e. the same ppm criterion; packets store both values.
- **Fit.** Short-TE 7 T spectra break the full-spectrum 6-singlet model used for the StarterPack (it hit its linewidth
  limit). External data use a windowed fit (1.85–3.35 ppm, NAA/Cr/Cho + linear baseline). On simulated truth it reads
  Cho/Cr about 3 % low; it is robust to residual water (the baseline absorbs the tail), which supports the catalogue
  v1.1 note that the F15 limit depends on the pipeline.
- **F15 at 7 T.** The real scan WARNs on residual water (1.11 × NAA). The limit was set at 3 T with the old fit; with the
  windowed fit the water did not bias Cho/Cr. Whether to relax F15 for this pipeline is a team decision.
- **Quality verdict only.** No clinical meaning. Values are indicative; the quality gates are the robust part.
