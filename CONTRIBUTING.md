# Working on twin-evidence

## Setup (once)
```bash
git clone <repo-url> twin-evidence && cd twin-evidence
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# the StarterPack is NOT in the repo: unzip MRI_Hackathon_DataScience_StarterPack_v0.3 next to this folder
STARTERPACK=../MRI_Hackathon_DataScience_StarterPack_v0.3 pytest -q     # 57 tests must pass
streamlit run app.py
```

## Map of the code
| File | What it does | Owner track |
|---|---|---|
| `twin_evidence/validate.py` | evidence gate: schema, QC re-evaluation, semantics, SHA-256, ROI consistency | Data Science |
| `twin_evidence/twin.py` | gated scalar Kalman twin + next-measurement planner | Digital Twins |
| `twin_evidence/scanner.py` | scanner twin from a weekly phantom record (SIMULATED) | Metrology |
| `twin_evidence/anatomy.py` | NIfTI / NIfTI-MRS readers, raw SNR and FWHM | QE / DS |
| `twin_evidence/views.py` | plotly figures (3D map, slices, spectra, scanner trends) | UI |
| `app.py` | Streamlit app (6 tabs) | UI |
| `reproduce.py` | Gate 1: re-run the reference update; judge unseen packets | all |
| `twin_evidence/mrs_sim.py` | physics simulator of PRESS transients with injected faults (SIMULATED, known truth) | QE / Metrology |
| `twin_evidence/mrs_fit.py` | time-domain fit (Cho/Cr with CRLB) + SNR / FWHM / residual-water QC | QE / DS |
| `twin_evidence/scenarios.py` | 11 fault scenarios, gate, naive-Bayes fault model | all |
| `twin_evidence/ingest.py` + `scripts/ingest_mrs.py` | field mapper: raw NIfTI-MRS (any field, coils, transients) → de-identified derivative + packet v1.0 + simulated follow-ups | DS |
| `twin_evidence/scanner_link.py` | the two twins talk: scanner-twin state on the scan date → gate finding F11 on every subject packet; patient-or-machine attribution | Metrology / DT |
| `twin_evidence/gates.py` | ScanTwin catalogue v1.1 gates with IDs, scan-context advice | all |
| `twin_evidence/longitudinal.py` | weekly phantom + brain sessions on a failing scanner: the full chain | Metrology / DT |
| `scripts/generate_scenarios.py` | builds `scenario_data/` (≈ 7 min) | all |
| `scripts/export_scanner_data.py` | builds `scanner_data/` for the scanner experts | Metrology |
| `docs/` | team briefing and pitch showcase pages | pitch |
| `CHECKPOINTS.md` | the plan: what must be true after each working hour | all |

## Data contract (ScanTwin catalogue v1.1)
Store individual MRS transients (NIfTI-MRS dimension DIM_DYN). Needed for gates F08 (drift) and F23 (motion); without them
these gates are N/A. The catalogue lives in `docs/scantwin/` (xlsx from the bioengineer) + `catalogue/changes_v1.1.json`;
run `python scripts/import_catalogue.py` after either changes.

## External data
Raw downloads never go into the repo (they can contain identifying metadata). Keep them outside, e.g.
`../real_data/<source>/`, and run the converter with `python -I`. See `external_data/README.md`.

## Rules
- Work on a branch (`git switch -c p2/rf-fault-scenario`), open a PR or merge after `pytest` is green.
- Never change the reference numbers in `tests/test_reference.py`: they reproduce the dossier.
- Every invented number is labelled `SIMULATED` in data, UI and slides.
- After changing `scanner.py`, rerun `python scripts/export_scanner_data.py ../MRI_Hackathon_DataScience_StarterPack_v0.3`.
- `docs/showcase.html` is built from `docs/showcase_src.html` + brain meshes; edit the source.
