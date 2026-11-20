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
6. **Scanner twin** (`twin_evidence/scanner.py`, **SIMULATED**): a weekly phantom record of linewidth, SNR and centre
   frequency, run through the same gate and Kalman step. Three scenarios (stable, shim degrading, B0 drift). Its verdict
   feeds the competing explanations of the brain twin and the bore colour in the 3D map, and it forecasts when the phantom
   will cross the FAIL limit. The StarterPack has no phantom data yet; the series and its thresholds are our assumptions.
7. **Fault scenarios** (`scenario_data/`, **SIMULATED with known truth**): pick one of 11 faults (the seven error sources
   of `scanner_data/MRI_MRS_Essential_Errors_Explained.md` plus coil, B1, gradient and a true change) and see the
   simulated spectrum next to a clean one, every quality check, the measured vs true Cho/Cr and the fault model's
   diagnosis with who is to blame. Further sub-tabs: what each fault does to Cho/Cr (60 scans each), the fault model
   applied to the real November spectrum, and six hardware faults followed for 26 weeks through to the brain decision.

The WARN variance factor in the sidebar is the declared team policy (default 2).

## Next steps
1. ~~Streamlit UI~~ (done: `app.py`).
2. Field mapper for a new dataset (Gate 2): the cohort data layer from the CTU homework (FreeSurfer .stats → packets v1.0).
3. Scenario context fields (for example smoking status), clearly labelled SIMULATED and kept apart from the measurement fields.
