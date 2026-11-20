# Scenario data: fault scenarios with known truth (all SIMULATED)

This folder answers point 3 of the team plan: *"get the data for every scenario"*. There are no real fault
recordings, so we **simulate the physics** and keep the truth:

```
fault (we choose it)  →  64 PRESS transients (StarterPack signal model)  →  average  →  fit + QC (same code as for real data)
     →  gate (thresholds-team-v1.1)  →  twin  →  diagnosis      …and we compare every step with the known truth
```

Regenerate everything with
`python scripts/generate_scenarios.py ../MRI_Hackathon_DataScience_StarterPack_v0.3 --n 30` (≈ 7 min).
Code: `twin_evidence/mrs_sim.py` (simulator), `mrs_fit.py` (fit + QC), `scenarios.py` (catalogue, gate, fault model),
`longitudinal.py` (time: weekly phantom + brain sessions). Tests: `tests/test_simulation.py`.

## Validation first: does the pipeline work on known truth?

| Check | Result |
|---|---|
| Fit of a noiseless spectrum | Cho/Cr 0.688 vs truth 0.689 |
| Fit of the **real** StarterPack spectra | Cho/Cr 0.672 / 0.816 (packets: 0.69 / 0.85); NAA/Cr 1.252 / 1.166 (packets 1.25 / 1.16) |
| SNR recomputed from the **real** files | 31.3 / 24.9 (packets 31.2 / 24.8) |
| Clean simulation (60 runs) | SNR 31.8 ± 2.0, FWHM 8.8 Hz, Cho/Cr 0.692 ± 0.022 vs truth 0.689 (unbiased) |

## The 11 scenarios (`catalogue.json`, `summary.csv`, 60 runs each)

Sources: `MRI_MRS_Essential_Errors_Explained.md` (§1–7), hardware classes from Dossier §5.2, and one true change.

| Scenario | Blame | SNR | FWHM → after alignment (Hz) | Other signature | Cho/Cr (truth) | Gate |
|---|---|---|---|---|---|---|
| clean | none | 31.8 | 8.8 → 8.8 | | 0.692 (0.689) | 98 % PASS |
| low SNR: receive coil | machine | **21.3** | 8.7 → 8.7 | | 0.692 ± **0.034** (0.689) | 100 % WARN |
| poor shim (B0) | machine / session | 25.2 | **12.8 → 12.8** | alignment does not help | 0.678 (0.689) | 100 % WARN |
| patient motion | patient | 26.4 | 9.2 → 8.8 | **16 % outlier transients** | 0.694 (0.689) | 92 % WARN |
| frequency drift | machine | 22.8 | **14.8 → 8.9** | **drift 11.3 Hz**; alignment fixes it | 0.668 (0.689) | 98 % WARN |
| voxel misplacement (4 mm) | operator | 32.5 | 8.8 → 8.8 | GM fraction 20 % → 5 % | **0.729 (0.726)** | **98 % PASS** |
| incomplete water suppression | machine | 31.6 | 8.8 → 8.8 | **residual water 2.1× NAA** | **0.568** (0.689) | 100 % WARN |
| protocol TE 144 ms | operator | 20.1 | 8.6 → 8.7 | metadata differ | **0.808** (0.689) | 100 % WARN |
| RF transmit (B1 −20 %) | machine | 24.6 | 8.7 → 8.7 | residual water 2.2× | **0.566** (0.689) | 100 % WARN |
| gradient scale +4 % | machine | 35.2 | 8.8 → 8.8 | voxel larger and moved | **0.668 (0.666)** | **98 % PASS** |
| true change (Cho +20 %) | patient | 31.8 | 8.8 → 8.8 | | **0.830 (0.827)** | **98 % PASS** |

What the numbers teach:

1. **Random vs systematic.** A weak coil only widens the scatter (SD 0.022 → 0.034); the mean stays right. Water
   leakage, B1 errors and a different TE move the *mean* by up to 0.12, i.e. 5–6 SD, far outside what the CRLB reports.
2. **Shim vs drift.** Both broaden the line. Only drift recovers after aligning the transients (14.8 → 8.9 Hz). This
   needs the individual transients, which the StarterPack files do not keep.
3. **Four scenarios are spectrally silent.** Voxel misplacement, a gradient error and a true change all pass the gate
   with a clean spectrum, yet Cho/Cr moves. Only the scanner twin (phantom geometry) and tissue fractions can tell them
   apart. This is the strongest argument for a scanner twin.

## Fault classifier (`fault_model.json`, confusion matrices)

A transparent naive-Bayes model: for each scenario the mean and SD of each quality metric (trained on 30 runs, tested
on 30 other runs). It ranks scenarios by how well a scan's metrics fit, never by Cho/Cr itself.

| Metrics available | Exact scenario | Who is to blame | Scenario group* |
|---|---|---|---|
| all (needs individual transients) | 79 % | 79 % | **100 %** |
| averaged spectrum only (like the StarterPack) | 76 % | 78 % | **96 %** |

\* The four spectrally silent scenarios count as one group; they cannot be separated from the spectrum, by design.
Trained on simulations only, so it recognises only faults we modelled. It is not a replacement for a physicist.

## Applied to the REAL StarterPack spectra (`real_data_diagnosis.json`)

| Session | Most likely | Blame |
|---|---|---|
| May (baseline) | clean-looking group (clean 31 %, true change 31 %, misplacement 28 %) | no fault |
| **November (follow-up)** | **poor shim 92 %**, motion 7 %, weak coil 1 % | **machine 93 %** |

So for the real November WARN, the model trained on simulated physics points to a **shim problem**. Whether it is the
scanner's shim or a local, session-specific shim (head position, sinuses) cannot be decided from one spectrum: that is
exactly what the scanner twin's phantom record is for. Recommended next step: re-shim and repeat PRESS, and store the
individual transients.

## Time: the scanner twin (`longitudinal/`, `longitudinal_summary.csv`)

Six hardware states, ramping from week 14 to week 26, each with a weekly simulated phantom spectrum (linewidth, SNR,
centre frequency, residual water) plus a geometry check. The brain is scanned at week 0 and week 26 **on the same
scanner**; both sessions become schema-v1.0 packets with their own NIfTI file, SHA-256 and manifest check, pass the gate
and update the brain twin.

| Hardware state | Scanner twin diagnosis | Brain follow-up gate | Brain Cho/Cr (truth 0.689) | Verdict |
|---|---|---|---|---|
| stable | stable ✅ | PASS | 0.690 | CLEAN |
| shim degrading | shim ✅ | **WARN** (SNR 22.8, LW 11.6 Hz, like the real November) | 0.680 | **MACHINE** |
| receive coil degrading | receive coil ✅ | WARN | 0.690 | MACHINE |
| B0 drift | B0 drift ✅ | WARN | **0.553** | MACHINE |
| B1 miscalibration | B1 transmit ✅ | WARN | **0.569** | MACHINE |
| gradient drift | gradient ✅ | **PASS** | 0.687 (truth 0.686) | **HIDDEN MACHINE FAULT** |
| stable + true Cho change | stable ✅ | PASS | **0.828** (truth 0.827) | CLEAN: patient changed |

6 of 6 hardware states diagnosed correctly (one seed; the classifier above gives the statistics for single scans).
The gradient case shows the value of the scanner twin: the brain data look perfect and nothing else would notice.

## Files

| File | Content |
|---|---|
| `catalogue.json` | scenario parameters, blame, random/systematic, expected signature, next step |
| `montecarlo.csv` | all 660 simulated scans (set = train / test) |
| `summary.csv` | per scenario: mean ± SD of each metric, bias, CRLB vs real scatter, gate rates |
| `fault_model.json`, `confusion_*.csv`, `run_info.json` | classifier and its held-out accuracy |
| `real_data_diagnosis.json` | the classifier on the real StarterPack spectra |
| `thresholds-team-v1.1.json` | thresholds-v1.0 + team PROPOSAL gates: `mrs_water_ratio`, `mrs_drift_Hz`, `mrs_transient_outliers` |
| `examples/<scenario>.nii.gz` + `.json` | one averaged spectrum per scenario, with its truth (open it in the Explorer) |
| `longitudinal/<fault>/` | `phantom_weekly.csv`, two NIfTI spectra, two evidence packets, `chain.json` |
| `longitudinal_summary.csv` | one line per hardware state: diagnosis, brain gate, verdict |

## Assumptions to check (QE and scanner experts)

- Tissue ratios: WM Cho/Cr 0.74, NAA/Cr 1.29; GM 0.48 and 1.09 (chosen to reproduce the StarterPack voxel; Cr equal in GM and WM).
- T2 at 3 T: NAA 247, Cr 162, Cho 207 ms; mI, Glx, Cr2 rough.
- B1 error leaks water: residual water amplitude 12 × |error|.
- Packet uncertainty: CRLB ⊕ 0.045 for the other budget terms (Dossier §5.7).
- Phantom: brain-like solution, LW 3 Hz, SNR ~120; all phantom gates are our proposal.
- Severity of each scenario (e.g. shim 7 Hz, drift 12 Hz = 0.1 ppm, motion 20 % of transients) is illustrative.

Everything here is SIMULATED. No clinical claim, no device certification.
