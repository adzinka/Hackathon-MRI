# ScanTwin / twin-evidence: full context handover

Written at the end of a long working session, so a new chat can continue without losing context.
Everything below comes from that session (files read, code written, results measured, decisions taken).
Where something was **not** done or not verified, it says so.

---

## 0. Quick orientation

| Item | Value |
|---|---|
| Event | InnovateU hackathon, NMR Quantum Engineering challenge, KICU Ulm, Fri 20 Nov 2026 16:00 to Sat 21 Nov 2026 17:00 |
| Challenge question | "What evidence is sufficient to update a digital twin of a system's health, and what remains uncertain?" |
| Product name (team decision) | **ScanTwin** (earlier draft name "Twin Evidence"; ScanTwin is used everywhere now) |
| Repository | https://github.com/adzinka/Hackathon-MRI (branch `main`, last pushed commit `b4293c3`) |
| Local repo | `/home/liza/cvut/1semestr/hackaton/twin-evidence` |
| Hackathon materials | `/home/liza/cvut/1semestr/hackaton/` (StarterPack, dossiers, slides, Explorer, team files) |
| Python env | `twin-evidence/.venv` (Python 3.10). Run `source .venv/bin/activate` |
| Run the app | `streamlit run app.py` (restart after code changes; a browser refresh does NOT reload changed modules) |
| Run tests | `STARTERPACK=../MRI_Hackathon_DataScience_StarterPack_v0.3 pytest -q`: **69 tests pass** |
| Published pages (claude.ai artifacts) | Team briefing: https://claude.ai/artifact/XTm6AU4VtyxGLFLqqdjwqR · Sales showcase "ScanTwin": https://claude.ai/artifact/1LV83n5MSNUDewHhiHeypX (shared "anyone with the link") |
| Team roles mentioned | a bioengineer (wrote the ScanTwin concept PDF, the failure catalogue xlsx and change list v1.1), a second programmer (works from GitHub), scanner/metrology experts, QE/physics, pitch |
| User language | mostly Russian; asked for English explanations while building the converter; once wrote in Spanish |

---

## 1. Hackathon framing (from Dossier v1.5/v1.6 and the slides)

### 1.1 Tracks and the chain
Four tracks: Quantum Engineering (what is measured), Precision Mechanics / Metrology (can it be trusted), Data Science
(evidence packet, QC, provenance), Digital Twins (state, update, next measurement). Chain:
measurement card → uncertainty card → evidence packet → twin update → next measurement.
Non-negotiable rule: no claim of change without evidence, calibration context, uncertainty, provenance and a baseline.

### 1.2 Mentor gates (Dossier §9.2, §9.3, §9.6; Conclusion slides v2.3 slide 16)
| Gate | When | Slides say | Dossier asks |
|---|---|---|---|
| Gate 1 · Reproduce | Fri 17:00 | re-run and explain the worked reference update; **every member traces one twin value to its source file** | aim contract + state variable + data source |
| Gate 2 · Extend | Sat 10:30 | solve one case the prototype does not: cohort data layer, **a second system**, or perturbed follow-up sessions | QC + uncertainty + provenance + alternative explanation |
| Gate 3 · Defend | Sat 13:30 (pitch 15:00) | **decide unseen evidence packets live**: update / update_with_warning / reject, with reasons | regression tests + limitation card + pitch |

### 1.3 Deliverables (Dossier §9.7, Conclusion slide 15)
Challenge canvas + architecture; data dictionary + evidence packet (schema v1.0); measurement card + uncertainty card;
working prototype with ≥1 baseline→follow-up update; provenance view; uncertainty + competing-explanations view;
next-evidence proposal; 5-minute pitch with limitations; individual reflection (course credit).

### 1.4 Rubric (Dossier §9.10)
Scientific correctness 20 % · Digital-twin logic 20 % · Measurement trust 15 % · Data integrity 15 % ·
Cross-track integration 15 % · Prototype and communication 10 % · Ethics and limits 5 %.

### 1.5 Maturity ladder (Dossier §7.11)
L0 data viewer · L1 longitudinal registry · L2 evidence-aware · **L3 dynamic twin (minimum target)** · L4 predictive ·
L5 evidence-steering (recommends next measurement). Our code covers L3–L5.

### 1.6 Dossier v1.6 additions (vs v1.5)
- §6.10: CTU Data Science lecture (Dr. Prokop): "valid JSON is not yet a correct scientific measurement"; "a matching
  SHA-256 confirms the file bytes, not the physics"; a new dataset needs a field mapper.
- §7.9: TU/e lecture (Dr. Barosan) maps the challenge to a Digital Twin System: PT (subject + scanner), DT (state +
  update rule), DD (packets, QC, provenance), CN (data contract), SS (monitoring, decision, uncertainty display),
  CC (next-measurement recommendation, advises not diagnoses), mission + federation; four views (domain, developer,
  application, user). "Named owners per interface are still open."
- Numbers, thresholds and gates unchanged from v1.5.

### 1.7 Other team-provided documents read
- `SystemThinking_FINAL (1).pdf` (TU/e 2IN70): Why → How → What; Cause → Idea/Form → Existence; MBSE; TRIZ tools
  (contradictions); project/system "black box" (inputs: stakeholders, users, domains, analysis, risk, resources,
  constraints, law, standards; outputs beneficial / non-beneficial; external systems / environment); Go / No-Go.
- `MRI_MRS_Essential_Errors_Explained.md` (now in `scanner_data/`): 7 priority error sources (see §6.1).
- `ScanTwin a digital twin of the MRI scanner.pdf` (bioengineer, 4 pages): goal, two twins, six steps, UI panels,
  scope tiers, "never says diagnosis", pitch line.
- `ScanTwin failure catalogue and QC thresholds.xlsx` (bioengineer): 60 failure modes F01–F60, Decision rules,
  Scan types, Context factors, Sources. Change list v1.1 pasted later (see §5.3).
- `MRI_Data_Explorer_v1.0-rc5.html` (organisers): offline teaching workbench with a 9-stop tour and ~40 labs
  (DICOM per slice, NIfTI, BIDS, local spectrum, quality gate, twin running estimate, gated Kalman update, timeline,
  next measurement, registration check, repeatability, GUM budget, phantom QC gradient non-linearity, BIDS validator
  lite, hashes/provenance, DICOM tag diff t0↔t1 and identifying tags, export). Runs locally on embedded StarterPack data.

---

## 2. Goal and product (as agreed in the chat)

**Team goal (refined):** "When a follow-up MRI scan looks different, tell whether the patient changed or the machine
did, how sure we are, and what to measure next." The team's own framing: "Is it working? → thresholds. Why not? →
virtual twin."

**Product: ScanTwin, quality assurance for MR measurements.** Pitch line (from the bioengineer's PDF): *"ScanTwin
doesn't tell you what's wrong with the patient. It tells you whether the scan is good enough to say anything at all,
and how to make the next one better."*

- Customers discussed: hospital MR physics / QA teams, research centres and clinical trials with repeated scans,
  scanner service.
- Value: fewer false alarms, fewer wasted rescans, early warning of scanner faults, audit trail.
- Not sold: diagnosis, image-reading AI, medical decisions.

**Difference vs the MRI Data Explorer:** Explorer = textbook/workbench that shows and explains each step on the
built-in example. ScanTwin = working inspector that takes new files (incl. a real 7 T scan), de-identifies, applies
the team catalogue, links a scanner twin to the subject twin, and answers patient vs machine with tests.

**Difference vs MRIQC (discussed from general knowledge, not verified against MRIQC docs):** MRIQC computes image
quality metrics per scan (T1w/T2w/BOLD, newer versions diffusion) and accept/exclude; no persistent state, no update,
no MRS, no scanner twin. ScanTwin is "the next stage": its gates could consume MRIQC metrics.

---

## 3. Data available

### 3.1 StarterPack v0.3 (synthetic, `MRI_Hackathon_DataScience_StarterPack_v0.3/`)
- `01_DICOM/baseline|followup/IM0001–IM0016.dcm` (T1w slices), `dicom_metadata_extract_example.csv`, `dicom_tag_guide.csv`
- `02_NIfTI/`: `synthetic_T1w_baseline/followup.nii.gz`, `synthetic_T1map_ms.nii.gz`, `synthetic_brain_mask.nii.gz`
  (64×64×16, 3×3×4 mm, RAS affine origin −96, −96, −32)
- `03_BIDS_MRI/`: T1w + JSON sidecars per session; `derivatives/roi/.../label-ROI_probseg.nii.gz` (partial-volume ROI,
  volume = Σ fraction × 36 mm³ = **22 140** baseline, **27 810** follow-up)
- `04_MRS_BIDS/`: NIfTI-MRS single-voxel PRESS, shape (1,1,1,1024) complex64 (averaged, no transients), dwell 0.5 ms,
  ν0 123.25 MHz, TE 30 ms, TR 2 s, 64 averages, VOI 20 mm at (−16, 45, 6) mm RAS, frontal
- `05_TwinReady_JSON_CSV/`: `packets/` E-t0-roi, E-t0-naa, E-t0-cho, E-t1-roi, E-t1-naa, E-t1-cho, E-t1-chobad;
  `thresholds-v1.0.json`; `evidence_packet_v1.0.schema.json`; `data_contract.json`; `features.csv`; `qc_report.json`
- `manifest_sha256.csv` (paths + SHA-256)

Key facts learned:
- T1 map classes: WM 850 ms, GM 1200, lesion 1450, CSF 3500; T1w intensities WM ~900, GM ~650, lesion ~540, CSF ~180,
  noise SD 12. **T1w background is masked to exactly 0.** The ROI sits at the lesion (~(31.6, −15.6, −0.4) mm).
- Follow-up T1w has the same anatomy with new noise (CHANGELOG X15); the follow-up ROI +25.6 % is a processing artefact
  (dossier "trap": stated u 4 % vs implied ~19 % from 1.1 mm registration RMS).
- MRS VOI tissue (T1-map sampling, 1 mm grid): WM 80 %, GM 20 %.
- Spectrum convention: s(t)=Σ a_j exp(−π LW t) exp(+i 2π f_j t), f_j=(δ_j−4.65)·ν0; spectrum = fftshift(FFT);
  ppm = 4.65 + f/ν0. SNR = NAA peak height (1.9–2.1 ppm, real) / SD of real spectrum in −3…0 ppm.
- Field label 3.0 T, but 123.25 MHz / 42.577 = 2.895 T.

### 3.2 Real data
- **Zenodo record 5085449 `example_02.nii.gz`** (NIfTI-MRS example data, W. T. Clarke, Oxford, **CC BY 4.0**).
  Siemens 7 T (297.219948 MHz), STEAM, TE 11 ms, TR 5 s, shape (1,1,1,2048,32,64) = 2048 points × 32 coils (DIM_COIL)
  × 64 transients (DIM_DYN), dwell 0.1666 ms. SHA-256 `172e6117…c356`.
  **Its metadata contained 6 identifying fields** (PatientName, PatientDoB, PatientSex, PatientWeight,
  InstitutionAddress, DeviceSerialNumber), removed by the converter (values never stored).
  Raw file kept OUTSIDE git: `/home/liza/cvut/1semestr/hackaton/real_data/zenodo_5085449_example_02/`.
  Zenodo blocked automated download from the sandbox (403 "unusual traffic"); the user downloaded it in a browser.
- Found but not downloaded: NITRC "Multi-site Frequency Drift MRS" (Big Drift): PRESS before/after a heavy fMRI
  sequence, Siemens TWIX zips (~360 MB per file), **CC BY-NC-SA 4.0 (non-commercial)**. MRSHub list mentions simulated
  3T PRESS (BSD-3), macromolecule aging (102 datasets, citation), Big GABA.
- A ChatGPT search prompt was written for the three data types (NIfTI-MRS, T1w NIfTI/BIDS, DICOM), prioritising
  repeated scans of the same person and commercial-use licences. Results not yet received.

---

## 4. Code: what exists (`twin-evidence/`)

| File | Purpose |
|---|---|
| `twin_evidence/validate.py` | Data Science gate: JSON Schema 2020-12; stored QC checks re-evaluated against thresholds; semantics (missing value, type, uncertainty, valid_range, relative_time, SIMULATED≠MEASURED, system_qc, protocol_match); manifest + SHA-256 + file bytes; `roi_u_consistency` (implied u = 3σ_b/r_eq carried forward); **F57** field+frequency; **F59** de-identification (identifying keys, `sub-` pseudonym); `system_qc_id` starting with `ASSUMED` → WARN "assumed, not verified"; optional `scanner=` (scanner-twin finding F11) |
| `twin_evidence/twin.py` | Gated scalar Kalman twin, `next_measurement` (Eq. 7.3), `q6` override |
| `twin_evidence/mrs_fit.py` | Full-spectrum time-domain Lorentzian fit (6 singlets NAA 2.01, Cr 3.03, Cho 3.20, mI 3.56, Glx 3.75, Cr2 3.92; common LW, shift, phase; CRLB); **windowed fit** `fit_window` (1.85–3.35 ppm, NAA/Cr/Cho + complex linear baseline); `qc_metrics` (SNR, FWHM zero-filled ×8, water_ratio); field frequency `nu0` parameter |
| `twin_evidence/mrs_sim.py` | Physics simulator of PRESS transients with injected faults; tissue fractions from the T1 map; transient shift estimation, robust alignment, `transient_qc` (drift, outliers); `dwell` parameter |
| `twin_evidence/scenarios.py` | 11 MRS fault scenarios, `measure`, `gate` (catalogue IDs + legacy names), naive-Bayes `FaultModel`, confusion |
| `twin_evidence/longitudinal.py` | 26-week phantom record per hardware fault + brain sessions; `diagnose` (fault class from z of phantom twin); `run_fault` full chain |
| `twin_evidence/scanner.py` | Original simple simulated scanner twin (3 scenarios) + generic `track()` |
| `twin_evidence/scanner_link.py` | `ScannerRecord` (physics phantom record), `state(week, modality)`, `finding(packet)` → F11, `attribution` |
| `twin_evidence/gates.py` | ScanTwin catalogue v1.1 gates `MRS_GATES`, `T1W_GATES`, `evaluate_mrs`, `evaluate_t1w`, N/A reasons, standing limitations, context factors/advice, `supported_range` |
| `twin_evidence/ingest.py` | **NIfTI-MRS field mapper** (see §5.1) + `degrade` (simulated follow-ups) + `build_packet`, `write_derivative`, `write_transients`, `read_bytes` |
| `twin_evidence/t1w.py` | **T1w pipeline** (see §5.2) + `compare` + `degrade` + `build_packet` |
| `twin_evidence/anatomy.py`, `views.py`, `scenario_views.py` | StarterPack 3D/slices/spectra figures, scenario figures |
| `twin_evidence/check_page.py` | Streamlit "Check a scan" page |
| `app.py` | Streamlit app: sidebar page switch "🔎 Check a scan" / "🧪 Hackathon lab"; settings expander |
| `reproduce.py` | Gate 1 reproduction; `--packet` judges unseen packets |
| `scripts/` | `generate_scenarios.py`, `sweep_faults.py`, `export_scanner_data.py`, `import_catalogue.py`, `ingest_mrs.py` |
| `tests/` | `test_reference.py`, `test_simulation.py`, `test_scanner_link.py`, `test_ingest.py`, `test_t1w.py` (69 total) |
| Data folders | `scenario_data/` (simulated, with README), `scanner_data/` (for scanner experts, with README + questions), `external_data/zenodo_5085449_example_02/` (de-identified derivatives, packets, summary), `catalogue/` (JSON + changes_v1.1.json) |
| Docs | `README.md`, `CONTRIBUTING.md`, `CHECKPOINTS.md`, `docs/scantwin/` (PDF, xlsx, REVIEW.md), `docs/cards/measurement_card_Cho_Cr.md`, `docs/team_briefing.html`, `docs/showcase_src.html` + `showcase.html` |

Conventions: every invented number is labelled SIMULATED; raw downloads never committed (`.gitignore` has
`real_data/`, `*.dat`, etc.); downloaded files read with `python -I` in their own folder; commits by `user.name=liza`
with "Co-Authored-By: Claude Opus 5.5"; UI in English; the word "diagnosis" is not used for the subject.

---

## 5. Input types: how each is read

### 5.1 MRS: single-voxel NIfTI-MRS (`ingest.py`) ✅ implemented
Steps (recorded in a conversion report):
1. **read** data, dwell (pixdim[4]), ν0 and metadata (header extension code 44 JSON); require shape (1,1,1,…).
2. **de-identify**: drop keys starting with `Patient` (except `PatientPosition`) and InstitutionAddress,
   DeviceSerialNumber, OperatorsName, ReferringPhysicianName, PerformingPhysicianName, AccessionNumber, StudyID,
   OtherPatientIDs; report names only.
3. rescale tiny vendor units.
4. **combine coils** (`DIM_COIL`): weights conj(s0)/σ_noise², phase-aligned on first points.
5. keep **transients** (`DIM_DYN`) for F08 drift and F23 motion; averaged files → N/A.
6. **orient**: if NAA mirror at 7.29 ppm stronger than 2.01 ppm → conjugate (the 7 T file was mirrored).
7. **reference** NAA to 2.01 ppm; **zero-order phase** on 1.8–3.3 ppm.
8. **QC** on the average: SNR, FWHM, water ratio; transient drift/outliers.
9. **fit**: windowed fit (external data). Validation: clean simulation Cho/Cr 0.668 ± 0.019 vs truth 0.689
   (~3 % low); StarterPack 0.655 / 0.778 (packets 0.69 / 0.85). A sign bug in the Lorentzian was caught by validation
   and fixed. The full-spectrum 6-singlet fit **fails on 7 T short-TE** (hit its 40 Hz LW limit, NAA/Cr 2.07), which is
   why the windowed fit exists. The StarterPack/Gate-1 pipeline keeps the original fit/packet values.
10. **field scaling**: Hz limits (F12, F08) compared as 3 T-equivalent (× 123.25/ν0); packets store raw and scaled.
11. **packet** schema v1.0 (MEASURED or SIMULATED), uncertainty = CRLB ⊕ 0.045; source = de-identified derivative.
`scripts/ingest_mrs.py RAW --out DIR --subject sub-ext01 --assume-system-qc` also writes simulated follow-ups.

Real 7 T results: SNR 171 · FWHM 16.3 Hz = 0.054 ppm · water/NAA 1.11 (**WARN F15**) · drift 1.46 Hz · 0 % outliers ·
Cho/Cr 0.514 ± 0.060 (bundled transients file re-read: 0.510) · NAA/Cr 1.83 · fit LW 13.5 Hz.
Simulated follow-ups from the real transients: repeat z = −0.02 (no change); shim +22 Hz → WARN F12; drift 25 Hz →
WARN F08/F12; noise ×8 → SNR 19 WARN F13; water ×5 → 1.99 FAIL F15 (rejected). Adding water did NOT bias Cho/Cr in the
windowed fit (baseline absorbs the tail). True choline change injection (`cho_up`) recovered (+20 % → ×1.21).

### 5.2 Structural T1w: NIfTI (`t1w.py`) ✅ implemented
- `load(image, mask=None, roi=None, sidecar=None)` (paths or bytes); de-identify sidecar (Patient*, InstitutionAddress,
  DeviceSerialNumber, OperatorsName, ReferringPhysicianName, AccessionNumber, StudyID, PerformingPhysicianName,
  InstitutionName); rejects complex/non-3D data.
- brain mask: provided, or automatic (non-zero support if background masked, else Otsu + largest component + fill;
  **no skull stripping**, so on real data it is a head mask).
- **segmentation**: k-means (CSF/GM/WM) alternated with a 2nd-order polynomial bias-field fit on log WM intensity.
- **metrics**: SNR in WM (noise = robust SD of residual after local 3×3×3 mean, in eroded WM, like MRIQC snr_wm),
  CNR, WM bias-field CV %, ghost % (phase- vs frequency-encode background; fallback: background above noise floor;
  0 if source masked background), motion rating 1–4 (1 % / 3 % / 6 %, team proposal), volumes brain/GM/WM/CSF (mL),
  ROI (Σ probseg × voxel). With a baseline: CNR % of baseline, registration shift by phase correlation (mm), image
  change SD (×noise), protocol differences (TR/TE/TI/flip/field), volume ratios, head-size change %.
- ROI uncertainty: 3 % (segmentation repeatability) or 3σ_b/r_eq from the measured registration shift.
- StarterPack numbers: baseline SNR_wm 56, CNR 15.7, noise ≈ 16, bias CV 0.04 %, brain 796.32 mL, GM 447, WM 331,
  CSF 18.3, ROI 22 140 mm³, gates PASS. Follow-up: ROI 27 810 (+25.6 %), shift **0.002 mm**, image change 0.73 ×
  noise, tissue ratios 1.000 → **"⚙️ The processing, not the patient"**.
- Limits: realistic bias (±50 %) corrected; extreme fields (0.5×–1.8×) not fully (but FAIL F28); only synthetic T1w tested.

### 5.3 DICOM ❌ not implemented yet
- Exists in StarterPack: `01_DICOM/baseline|followup/IM0001–IM0016.dcm` (T1w slices, "one file per slice"),
  `dicom_metadata_extract_example.csv`, `dicom_tag_guide.csv`. CHANGELOG notes: X10 extract CSV fixed (TE, IPP),
  X11 DICOM LPS vs NIfTI RAS geometry made consistent (IOP 1\0\0\0\1\0, IPP −93\−93\z, pixels rotated 180°),
  X5 ImagingFrequency (0018,0084) = 123.25 MHz added, X14 StudyDate 2026-05-20 / 2026-11-20.
- Planned approach (agreed, not built): DICOM series → NIfTI (dcm2niix or pydicom) → strip identifying tags (F59)
  → the T1w path. Estimated ~1 h. The Explorer already shows "DICOM tag diff t0↔t1 and identifying tags".
- The user framed DICOM as an "envelope" for T1w: errors are data errors, not physics.

### 5.4 Evidence packets: JSON (schema v1.0) ✅
22 required keys: packet_version, packet_id (`^E-t[0-9]+-[a-z0-9_]+$`), subject_id (`^sub-…`), session_id (`^ses-…`),
relative_time, modality (MRI/qMRI/MRS/MRSI/HP13C/calibration), source_object, spatial_reference, observable, value,
unit, uncertainty, qc_checks, qc_status, origin (MEASURED/SIMULATED), value_epistemic, decision_epistemic,
acquisition_context, processing, provenance, twin_action, interpretation_limit. Upload a JSON on "Check a scan" →
validator + twin against the StarterPack baseline of the same observable (Gate 3 path). E-t1-cho → 0.741 ± 0.048;
E-t1-chobad → reject with reasons.

---

## 6. Error sources by input type

### 6.1 MRS (the 7 essential errors + hardware classes)
| Error (file §) | Who | Detection in our system | Simulated? |
|---|---|---|---|
| 1 Low SNR | machine (coil) / scan | F13 SNR | ✅ `low_snr_coil` |
| 2 B0 inhomogeneity / poor shim | machine / session | F12 linewidth; **not fixed by transient alignment** | ✅ `poor_shim` |
| 3 Patient motion | patient/session | F23 outlier transients (needs transients) | ✅ `motion` |
| 4 Frequency drift | machine | F08 drift (needs transients); **alignment restores the line** (14.8 → 8.9 Hz) | ✅ `freq_drift` |
| 5 Voxel misplacement / tissue mixing | operator | F20 voxel centre vs baseline; F21 CSF change (tissue fractions) | ✅ `voxel_misplacement` |
| 6 Incomplete water suppression | machine | F15 residual water / NAA | ✅ `water_suppression` |
| 7 Protocol mismatch (TE/TR) | operator | F35 | ✅ `protocol_te` |
| RF transmit (B1) | machine | **F61** reference amplitude vs phantom baseline (new row v1.1) | ✅ `b1_transmit` |
| Gradient scale | machine | only via scanner twin geometry (F01) | ✅ `gradient` |
| True change | patient | passes gates; change test F60 | ✅ `true_change` |
Also: F14 CRLB (precision only, never alone), F19 chemical-shift displacement (~12 % PRESS 3 T, standing limitation;
v1.1 makes it a between-session change check), F22 Cr reference (ratios only, measurement-card limitation).
Indistinguishable from an averaged spectrum: shim vs drift vs motion (need transients or phantom); misplacement,
gradient error and true change are **spectrally silent**.

### 6.2 T1w
| Catalogue row | Error | Our detection |
|---|---|---|
| F26 | subject motion | ghost % → rating (team-proposal mapping) |
| F27 | low SNR/CNR | CNR % of baseline (needs baseline) |
| F28 | intensity bias field | bias-field CV % in WM |
| F32 | registration error between sessions | phase-correlation shift (mm) |
| F33 | stated u inconsistent | validator `roi_u_consistency`; T1w u from 3σ_b/r_eq |
| F34 | segmentation version change | detected indirectly: outline changed, images didn't → "processing" |
| F35 | protocol mismatch | sidecar TR/TE/TI/flip/field |
| F01 / F36 | gradient geometry | whole-head size change > 2 % → "machine (geometry)" rule |
Not implemented: F29 susceptibility, F30 aliasing, F31 Gibbs, F37 flow artefacts.

### 6.3 DICOM / data (catalogue data rows)
F54 schema/required fields, F55 valid range, F56 units, F57 field + frequency recorded (implemented in validator),
F58 session time consistency, **F59 de-identification** (implemented for packets, NIfTI-MRS metadata and T1w sidecars;
not yet for DICOM tags), F11 system QC present/passed. Changelog issues (orientation LPS/RAS, extract mismatch, dates,
frequency tag) are the kinds of DICOM problems seen in the StarterPack history.

---

## 7. The digital twins and how they interact

### 7.1 Subject twin (`twin.py`, Dossier §7.5)
- prior P⁻ = P + Q, Q = q6²·dt/6 (q6: ROI 1 % of value, NAA/Cr 0.03, Cho/Cr 0.03, T1 30 ms);
- gate d² = (y − x⁻)²/(P⁻ + u²) (flag > 3.841);
- gain K = P⁻/(P⁻ + R), R = u² (PASS) or 2u² (WARN, declared team policy); FAIL → reject;
- change test z = Δ/√(u0² + u1²).

Gate 1 reference (reproduced exactly):
- E-t1-cho: WARN, z +2.05, d² 3.66, K 0.321 → **0.741 ± 0.048** (without the WARN policy K 0.486 → 0.768);
- E-t1-naa: z −0.90, K 0.26 → 1.227 ± 0.058;
- E-t1-roi: stored PASS → WARN (roi_u_consistency, implied 18.9 %), z +1.07, K 0.009 → 22 190 ± 697;
- E-t1-chobad: reject;
- next measurement: re-shim repeat σ 0.048 → 0.035.

### 7.2 Scanner twin
Weekly phantom spectra simulated with `mrs_sim` (brain-like phantom, LW 3 Hz, SNR ~120); faults ramp weeks 14→26:
- shim 6 Hz;
- coil noise ×1.45;
- B0 offset 14 Hz + drift 8 Hz;
- B1 −20 %;
- gradient +2.5 %.

Phantom observables and limits:
| Observable | Limits | Source |
|---|---|---|
| linewidth | WARN > 4.5, FAIL > 6 Hz | team |
| SNR % of coil baseline | WARN < 90, FAIL < 70 | **F09** |
| centre frequency | WARN > 6, FAIL > 12 Hz | team |
| residual water / NAA | WARN > 0.5, FAIL > 1.5 | **F15 v1.1** |
| geometric error on a 190 mm phantom | WARN > 1, FAIL > 2 mm | **F01** |

Every phantom reading updates the scanner twin at full weight (a bad phantom value is good evidence of a bad machine).
Fault class from the pattern of z changes: geometry → gradient; frequency → B0 drift; water → B1; linewidth → shim;
SNR only → coil. 6/6 classes found. Week-26 state (MRS parts): stable PASS, shim FAIL, coil WARN, B0 FAIL, B1 WARN,
gradient FAIL.

### 7.3 Interaction (`scanner_link.py`, F11)
The scanner twin's state on the packet's scan date (relative_time → week) becomes finding `F11_scanner_twin`.
Only parts relevant to the modality count: MRS = all five; MRI volumes = SNR + geometry. FAIL blocks the subject
update, WARN down-weights. `attribution`: MACHINE / NOT THE MACHINE (session or processing) / PATIENT / unknown.
With "no scanner twin" or a stable scanner the Gate 1 numbers are unchanged. On the StarterPack November packets:
- shim degrading → Cho rejected, ROI WARN;
- B0 drift → Cho rejected;
- coil → WARN;
- B1 → Cho WARN, ROI unaffected;
- gradient → **both Cho and ROI rejected**.

### 7.4 Patient-or-machine on "Check a scan"
- **MRS (catalogue F60):** later scan FAIL → "🛠 It's the machine"; a new non-PASS gate → "🛠 Probably the machine";
  |z| < 1.96 → "❔ Can't tell / not distinguishable"; 1.96 ≤ |z| < 3 with any WARN → "❔ Inconclusive";
  otherwise → "🧑 The patient changed". On the real 7 T scan a choline change needs about +70 % (z > 3) because the
  scan carries a WARN.
- **T1w (`t1w.compare`), five answers:** a session gate (F26/F32/F35) got worse → 🎯 session (motion explains a
  contrast loss first); machine gates (F27/F28) → 🛠 machine; whole-head size change > 2 % → 🛠 machine (geometry);
  ROI change significant but images unchanged and tissue ratios ≈ 1 → ⚙️ processing; |z| < 1.96 → ❔; otherwise 🧑.
  Verified on seeds 3/5/7 for noise, bias, motion, scale, shift, grow, and on the StarterPack pair (processing).

---

## 8. Simulator, scenario data and threshold checks
- `scenario_data/`: 11 MRS scenarios × 60 scans (30 train, 30 test). Clean: SNR 31.8, FWHM 8.8, Cho/Cr 0.692 vs truth
  0.689. Biases: water −0.12, B1 −0.12, TE 144 +0.12; misplacement 0.729 (truth 0.726); gradient 0.668 (0.666);
  true change 0.830 (0.827).
- Naive-Bayes fault model accuracy: all metrics 79 % exact / 100 % group; averaged-only 76 % / 96 %.
  On the real StarterPack November spectrum: **poor shim 92 %, motion 7 %; machine 93 %**.
- Fault-strength sweeps (`sweeps.csv`, `threshold_check.csv`): Cho/Cr shift reaches 1 SD at F12 ≈ 14 Hz, F08 ≈ 10 Hz,
  F20 ≈ 1.9 mm, F15 ≈ 0.32 (2 SD ≈ 0.56). Rule: WARN must fire before 2 SD. v1.0 F15 (WARN > 5) was ~15× too
  lenient; v1.1 (0.5) is OK. All six checked gates OK under v1.1.
- Assumptions to be checked by experts: GM/WM ratios (Cho/Cr 0.48/0.74, NAA/Cr 1.09/1.29), T2 (NAA 247, Cr 162,
  Cho 207 ms), B1 water leak 12×|error|, packet u = CRLB ⊕ 0.045, phantom values and limits.

## 9. ScanTwin catalogue versions
- v1.0 (xlsx, 60 rows): Built/Designed/Future tiers; decision rules (FAIL→reject, WARN→R=2u², z, d², fix ranking).
- **v1.1** (`catalogue/changes_v1.1.json`, applied on import, xlsx untouched):
  - F15 PASS ≤ 0.5, WARN 0.5–1.5, FAIL > 1.5 (pipeline-dependent note);
  - F19 → between-session change of displaced fraction (WARN > 2 pp; FAIL on sequence/bandwidth change);
  - F22 removed from gates → measurement card;
  - notes on F08/F23 (need transients) and F14 (precision not bias);
  - new **F61** transmitter/B1 (≤ 10 / 10–20 / > 20 %);
  - data contract "store transients (DIM_DYN)";
  - decision rule: missing input → N/A, never FAIL;
  - name ScanTwin;
  - a Scan-context input that never changes limits.
- `docs/scantwin/REVIEW.md` documents the review and the v1.1 resolution.

## 10. The app (what users see)
- Sidebar: page switch **🔎 Check a scan** / **🧪 Hackathon lab**; ⚙️ Settings (StarterPack path).
- **Check a scan:** View switch **Simple / Expert**.
  - Step 1: "Upload my files" (multi-file: scan, optional later scan, brain mask, outline with `roi`/`probseg`, JSON
    sidecars; or one JSON packet) or "Use an example". Examples:
    - real 7 T spectrum;
    - StarterPack MRS May + Nov;
    - StarterPack T1w May + Nov with outlines;
    - T1w May only.
  - A green "Now showing …" banner, analysis status, then:
  - Step 2: result card (🟢 Good / 🟡 Use with caution / 🔴 Don't use) with plain reason; quality bars; what to do;
    range picture. Expert tabs: MRS = Quality checks / Spectrum / Scan context / Trace (+ packet download);
    T1w = Quality checks / Images / Volumes / Trace.
  - Step 3: compare with the real later scan or a simulated scenario; result card + two range bands; expert metrics.
- **Hackathon lab:** prior→evidence→posterior, Decision (patient-or-machine banner, scanner on the scan date),
  Provenance, Timeline, 3D evidence map (brain + ROI + VOI + scanner bore), Scanner twin (physics record,
  with/without comparison), ScanTwin fault scenarios (single scan, all 11, real November, scanner over time,
  threshold check, catalogue browser).
- The user found the first version confusing (could not tell file vs example; uploading the same scan as the example
  "did nothing"). This was fixed with the explicit source choice and banner. **Never visually verified by the
  assistant** (headless browser can't render Streamlit); logic verified with Streamlit AppTest.

## 11. Open items / next steps
1. **DICOM** reader (series → NIfTI, tag de-identification F59, then T1w path) using StarterPack `01_DICOM/`.
2. Real external data: run the ChatGPT prompt; prefer repeated scans; check licences (CC BY ok, NC not for sales);
   test T1w on real images (needs brain mask; no skull stripping).
3. Gate 3 robustness: 10–15 tricky packets (missing fields, text values, unknown observable, other TE) must never crash.
4. Team deliverables:
   - finish the measurement card (QE) and write the uncertainty card (metrology);
   - limitation card and architecture slide (PT/DT/DD/CN/SS/CC + black box);
   - name owners per interface;
   - every member traces a value to its file;
   - 5-minute pitch;
   - tag `v1.0` and record a demo video.
5. Decisions pending:
   - relax F15 for the windowed-fit / 7 T pipeline? (real 7 T scan WARNs on water 1.11);
   - expert review of simulator assumptions and phantom limits.
6. Known limitations to state:
   - synthetic StarterPack + simulated scanner record;
   - windowed fit reads ~3 % low;
   - Hz limits scaled from 3 T;
   - T1w without skull stripping;
   - fault classes, not root causes;
   - no clinical meaning, no device certification.
