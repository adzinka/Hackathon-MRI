# twin-evidence: evidence gate + gated twin update (InnovateU NMR challenge)

Python reproduction of the Data Science gate and the Digital Twins update from Hackathon Dossier v1.5
(§3.10, §5.8, §7.5, §8.2), checked against the numbers in the Dossier and the MRI Data Explorer v1.0-rc5.
Synthetic, educational data. No clinical meaning.

AI use (declare it in the pitch): drafted with Claude and reviewed by the team.

**Goal:** when a follow-up MRI scan looks different, tell whether the patient changed or the machine did, how sure we
are, and what to measure next. Plan and checkpoints: [CHECKPOINTS.md](CHECKPOINTS.md). New here: [CONTRIBUTING.md](CONTRIBUTING.md).
Data pack for the scanner experts: [scanner_data/](scanner_data/README.md).

## Setup
```bash
pip install -r requirements.txt
# put the unzipped StarterPack next to this folder or pass its path
```

## Gate 1: re-run the worked reference update
```bash
python reproduce.py ../MRI_Hackathon_DataScience_StarterPack_v0.3
```
Expected (Dossier Table 1 and §7.5):

| packet | status | action | z | d² | K | posterior |
|---|---|---|---|---|---|---|
| E-t1-roi | stored PASS → **WARN** (roi_u_consistency) | update_with_warning | +1.07 | 1.14 | 0.009 | 22 190 ± 697 mm³ |
| E-t1-naa | WARN (SNR, linewidth) | update_with_warning | −0.90 | 0.74 | 0.260 | 1.227 ± 0.058 |
| E-t1-cho | WARN (SNR, linewidth) | update_with_warning | +2.05 | 3.66 | 0.321 | **0.741 ± 0.048** |
| E-t1-chobad | FAIL | **reject** | – | – | – | state unchanged |

Next measurement for Cho/Cr: repeat PRESS after re-shim → posterior SD 0.048 → 0.035.

## Gate 3: judge an unseen packet live
```bash
python reproduce.py ../MRI_Hackathon_DataScience_StarterPack_v0.3 --packet unseen.json
```
The baseline packets are ingested first, then each unseen packet is evaluated. The script prints the status, the twin
action (update / update_with_warning / reject) and every non-PASS finding with its owner track and reason.
The full trace is written to `out/reference_update.json`.

## Tests (regression tests for Gate 3)
```bash
STARTERPACK=../MRI_Hackathon_DataScience_StarterPack_v0.3 pytest -q
```
- **Real data** (`pytest -q tests/real_samples`, no StarterPack needed): a real 3 T T1w and a real sLASER spectrum with
  its water reference (Morelli et al., Zenodo, CC BY 4.0) in [external_data/real_samples/](external_data/real_samples/README.md):
  checksums, headers, privacy, and the pipeline verdicts recorded in `qc_report.json`.
- **Golden tests:** the Cho/Cr, NAA/Cr and ROI updates, the result without the WARN policy (K = 0.486), the rejection
  of `E-t1-chobad`, and the next-measurement gain.
- **Negative tests** (every failure path from the Data Science lecture):
  - value as text `"0.85"`
  - value 3.4 outside [0, 3]
  - missing `session_id`
  - value null
  - uncertainty null
  - tampered SHA-256
  - unregistered source file
  - SNR 9
  - missing system QC id
  - SIMULATED labelled MEASURED
  - protocol change → WARN

## What the code does
- `twin_evidence/validate.py` (Data Science gate) runs four kinds of check:
  - JSON Schema 2020-12;
  - the stored QC checks re-evaluated against `thresholds-v1.0.json`;
  - semantic rules: missing value, value type, uncertainty, valid_range, relative_time, epistemic labels, system QC, protocol match;
  - linkage and integrity: the source is in the manifest and its SHA-256 matches the file bytes; plus `roi_u_consistency`, which carries the implied uncertainty forward.

  It never edits a packet, and every finding has a rule, field, status, owner and reason.
- `twin_evidence/twin.py` (Digital Twins) does the gated scalar Kalman step:
  - drift Q = q6²·dt/6;
  - the gate uses u²;
  - the gain uses R_factor·u² (PASS 1, WARN 2);
  - FAIL means no update;
  - the next-measurement planner uses Eq. 7.3.

## Streamlit demonstrator

The sidebar has two modes.

**🔎 Check a scan** (the landing page): upload a raw single-voxel spectrum (NIfTI-MRS `.nii.gz`, any field, with coils
and transients), a **structural T1-weighted image** (NIfTI, optionally with a brain mask, a region outline and the JSON
sidecar; `twin_evidence/t1w.py`), **two scans of the same person** to compare real data, or an evidence packet
(`.json`, e.g. an unseen Gate 3 packet), or pick an example (a real 7 T scan from
Zenodo, CC BY 4.0, or the StarterPack scans). The converter (`twin_evidence/ingest.py`) de-identifies, combines coils,
fixes the axis, references, phases, measures and fits. The page shows, as in the ScanTwin concept: the scan, scanner
health (catalogue gates), the verdict and the gate that decided it, the range the result can support, the fix for the
next scan, scan-context advice and the trace to the file, plus a downloadable packet. "Simulate a follow-up" injects a
scanner fault or a true choline change into the real transients and answers *patient or machine?* with the catalogue's
change rule F60 (not distinguishable / inconclusive / real change).

**🧠 Twin explorer**: the worked StarterPack example and everything below.

```bash
streamlit run app.py
```
Pick a follow-up packet in the sidebar (or upload an unseen one). Tabs:
1. **Prior → evidence → posterior**: the three distributions and the same step over time (±95 %), with the WARN-policy
   comparison and the Kalman numbers.
2. **Decision**: the gate status and action, z / d² / K, the non-PASS findings with their owner track, a prediction of
   the next repeat, the next-measurement table (Eq. 7.3) and the competing explanations.
3. **Provenance**: the path from the source file (SHA-256 check) through processing to the packet, gate and twin variable.
4. **Timeline**: every StarterPack packet, judged in order.
5. **3D evidence map**: brain mask, baseline and follow-up ROI masks, the MRS voxel (with NAA/Cho chemical-shift
   displacement) and the scanner bore, all drawn from the StarterPack NIfTI files, coloured by gate status. Next to it, a
   table of possible causes judged by the evidence; a T1w slice with the difference map (the anatomy did not change); and
   both spectra recomputed from the raw NIfTI-MRS files (SNR and FWHM checked against the packets).
6. **Scanner twin** (`twin_evidence/scanner_link.py`, **SIMULATED** weekly phantom record): pick the scanner's state in
   the sidebar (stable, shim, coil, B0 drift, B1, gradient, or none). The scanner twin's state **on the scan date** becomes
   gate finding F11 on every subject packet: FAIL blocks the update, WARN down-weights it, and only the parts of the machine
   a scan depends on count (geometry and SNR for volumes, all parts for spectroscopy). The header answers
   **"Patient or machine?"** for the selected packet. Phantom limits follow the ScanTwin catalogue (F01 geometry, F09 coil,
   F15 water) where it has one. With "no scanner twin" or a stable scanner the Gate 1 reference numbers are unchanged.
7. **ScanTwin: fault scenarios** (`scenario_data/`, **SIMULATED with known truth**): pick one of 11 faults (the seven error sources
   of `scanner_data/MRI_MRS_Essential_Errors_Explained.md` plus coil, B1, gradient and a true change) and see the
   simulated spectrum next to a clean one, every quality check, the measured vs true Cho/Cr and the fault model's
   diagnosis with who is to blame. Further sub-tabs: what each fault does to Cho/Cr (60 scans each), the fault model
   applied to the real November spectrum, and six hardware faults followed for 26 weeks through to the brain decision.

The WARN variance factor in the sidebar is the declared team policy (default 2).

## Next steps
1. ~~Streamlit UI~~ (done: `app.py`).
2. Field mapper for a new dataset (Gate 2): the cohort data layer from the CTU homework (FreeSurfer .stats → packets v1.0).
3. Scenario context fields (for example smoking status), clearly labelled SIMULATED and kept apart from the measurement fields.
