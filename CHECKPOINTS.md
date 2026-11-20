# Checkpoints: the next 10 working hours

## Our goal (aim contract)

> **When a follow-up MRI scan looks different, our twin tells you whether the patient changed or the machine did,
> how sure it is, and what to measure next.**

- **Physical counterpart:** a brain (StarterPack sub-01) *and* the scanner that measures it. Two federated twins.
- **State variables:** brain: Cho/Cr (primary), NAA/Cr, ROI volume. Scanner: phantom linewidth, SNR, centre frequency.
- **Ambiguity we resolve:** the Cho/Cr rise 0.69 → 0.85 arrives with broader, noisier spectra. True chemistry or the machine?
- **Output:** update / update_with_warning / reject, with a reason and an owner; a patient-or-machine attribution;
  the next measurement.
- **Limitation:** the scanner record is SIMULATED; no clinical or device-certification claims.

### Why not "know when the MRI scanner makes an error, and why" as the whole goal?

We considered it and rejected it as the *main* goal, for four reasons:

1. **Gate 1 and the minimum prototype need the brain update.** The dossier requires one baseline → follow-up update on
   the provided data. The scanner has no data in the StarterPack: a scanner-only project would rest entirely on numbers we
   invented, which hurts *Data integrity* (15 %) and *Scientific correctness* (20 %).
2. **"Why" has a limit.** Phantom data can tell fault *classes* apart by their signature (shim vs B0 drift vs receive
   chain vs gradients). It cannot name the broken part. We claim the class and recommend the check, nothing more.
3. **The scanner error only matters through its effect on a measurement.** The value comes from linking the two twins:
   the scanner twin explains or excludes a cause for the brain twin's WARN.
4. **This exact question is a dossier lane.** Lane A, "True change or measurement change?", with the strongest
   metrology focus. The scanner twin is our Gate 2 extension ("a second system") and our transfer case in the pitch.

So the scanner error is half of our answer, not the whole question.

## Scoring we aim at (Dossier §9.10)

| Criterion | Weight | What gets us the points |
|---|---|---|
| Scientific correctness | 20 % | QE + scanner experts sign off the fault-signature table and the measurement card |
| Digital-twin logic | 20 % | two persistent twins, gated update, prediction (forecast), next measurement: already L5 |
| Measurement trust | 15 % | uncertainty card; honest 19 % ROI; scanner twin feeds the gate |
| Data integrity | 15 % | SIMULATED labels everywhere; SHA-256; physics re-checked from raw files |
| Cross-track integration | 15 % | the scanner verdict changes the brain decision, owners named per interface |
| Prototype and communication | 10 % | app + showcase page, live unseen-packet demo |
| Ethics and limits | 5 % | limitation card, "advises, never diagnoses" |

## Status of the team plan (point 3: data for every scenario)

Done: `scenario_data/` (see its README). 11 physics-simulated scenarios from `MRI_MRS_Essential_Errors_Explained.md`
plus hardware classes, 660 scans with known truth, a fault classifier (96 % correct scenario group on held-out data),
the classifier applied to the real November spectrum (poor shim 92 %), and six hardware faults over 26 weeks with the
full chain to the brain decision (6/6 diagnosed). Open: experts to check the assumptions listed in that README.

## Checkpoints

Each checkpoint has a **definition of done (DoD)** that anyone can check in two minutes. Hours count working time from now.
Owners: **P1** = programmer 1, **P2** = programmer 2, **SC** = scanner / metrology experts, **QE** = physics,
**ALL** = everyone, **PITCH** = whoever presents.

### H1: goal locked, owners named
- [ ] Aim-contract slide with the goal above (PITCH)
- [ ] One named owner per interface: QE ↔ SC, SC ↔ DS, DS ↔ DT, plus one integrator (ALL)
- [ ] Every member traced one twin value back to its source file in app tab ③ (ALL; Gate 1 leftover)
- [ ] `scanner_data/` sent to SC; P2 has the repo running (`pytest` green)
- **DoD:** slide exists; owner list in this file; 17 tests pass on both laptops.

### H3: scanner twin is physically credible and wired into the gate
- [ ] SC answer questions 1–6 in `scanner_data/README.md`; team updates `scanner.py` values and thresholds (SC → P2)
- [ ] P2: two more fault scenarios, **RF receive chain** (SNR ↓ only) and **gradient** (distortion ↑), with a
      geometric-distortion observable
- [ ] P2: `attribute()` maps the pattern of scanner-twin changes between two dates to a fault class (the signature
      table), with tests for each scenario
- [ ] P1: the scanner-twin status on a packet's scan date becomes a `system_qc` finding in `validate.evaluate`
      (FAIL blocks the brain update, WARN downgrades it); regression tests
- **DoD:** `pytest` covers 5 scenarios × attribution; switching the scenario in the app changes the brain decision
  for E-t1-cho in a way SC agree with.

### H5: "patient or machine?" is one clear answer in the app
- [ ] P1: a verdict panel in tab ② Decision: *patient / machine / session (position, motion) / undecided*, with the
      evidence for each and the next check (re-shim, coil test, grid phantom, repeat PRESS)
- [ ] QE: measurement card for Cho/Cr (what the ratio means, PRESS voxel, chemical shift)
- [ ] SC: uncertainty card for Cho/Cr (fit, lineshape, placement, chemical-shift displacement, calibration; Dossier §5.7)
- **DoD:** for each of the 5 scenarios the app gives a one-sentence answer that SC and QE accept.

### H6: Gate 2 rehearsal (Sat 10:30: "Does the evidence justify the update?")
- [ ] 3-minute walkthrough: QC → uncertainty → provenance → alternative explanation → scanner twin (PITCH, P1)
- [ ] Architecture slide: PT / DT / DD / CN / SS / CC, four views, the two federated twins (PITCH)
- **DoD:** one dry run in front of a teammate who did not build it; they can repeat the answer.

### H8: ready for unseen packets (Gate 3: decide live)
- [ ] P2: perturbation tests, i.e. packets with voxel shift, protocol change, low SNR, missing fields, unknown observable;
      the app never crashes and always gives a reason
- [ ] ALL: each member writes one tricky packet; the team judges them live in the app sidebar
- [ ] P1: a field mapper stub for a new dataset (CTU homework style), or a documented "not supported" reason
- **DoD:** 10 hand-made packets judged with zero crashes; every reason readable by a non-programmer.

### H9: pitch final
- [ ] 5-minute script following Dossier §9.9; the showcase page as the backdrop, the app for the live part
- [ ] Limitation card: synthetic data, teaching thresholds, WARN factor 2 is a policy, phantom record simulated,
      fault class not root cause, no clinical or certification claims
- [ ] AI use declared on a slide
- **DoD:** two timed rehearsals under 5:00.

### H10: freeze
- [ ] `git tag v1.0`; fresh clone + setup + `pytest` + `streamlit run app.py` works on a second laptop
- [ ] Backup: screen recording of the demo and screenshots, in case the Wi-Fi or a laptop fails
- [ ] Handover package: README, data dictionary, cards, this file
- **DoD:** the demo runs from the tagged commit on a machine that did not build it.

## Risks to watch

| Risk | Sign | Response |
|---|---|---|
| Scope creep into a scanner-only project | the brain update is no longer on screen | keep the brain update as the opening demo |
| Simulated data mistaken for real | a slide without the SIMULATED label | label check before every rehearsal |
| SC disagree with our fault signatures | answers to question 5 contradict the table | change the table, say so in the pitch: it shows real cross-track work |
| Demo breaks live | an exception on an unseen packet | H8 perturbation tests, plus the recording |
