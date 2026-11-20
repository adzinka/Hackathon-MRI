# Measurement card: Cho/Cr (frontal ¹H PRESS)

*Draft for the Quantum Engineering track to complete. Facts below come from the StarterPack v0.3 and the dossier.*

| Field | Value |
|---|---|
| Observable | Cho/Cr: amplitude of the choline singlet (3.20 ppm) divided by the creatine CH3 singlet (3.03 ppm) |
| Physical meaning | relative choline signal in the voxel; choline compounds relate to membrane turnover (no clinical claim) |
| Sequence | single-voxel PRESS, TE 30 ms, TR 2 s, 64 averages, water suppression on |
| Field | 3.0 T nominal; ν0 = 123.25 MHz, i.e. 2.895 T measured (both recorded, catalogue F57) |
| Voxel | 20 × 20 × 20 mm, centre (−16, 45, 6) mm RAS, frontal, ~80 % WM / 20 % GM (from the T1 map) |
| Processing | time-domain Lorentzian fit, common linewidth, shift and phase; ratio to Cr CH3 |
| Uncertainty | packet u 0.05 (t0) / 0.06 (t1); fit CRLB ≈ 0.02; other terms (placement, chemical shift, lineshape, calibration) see the uncertainty card |
| Quality gates | ScanTwin catalogue v1.1: F12 linewidth, F13 SNR, F14 CRLB, F15 residual water, F08 drift, F23 motion, F20 voxel position, F21 CSF change, F35 protocol, F19 displacement change, F61 transmitter |

## Standing limitations (shown with every result; never change the verdict)

- **F22, unstable reference.** Ratios only; creatine is assumed stable. A change in X/Cr may come from Cr, not from X.
  Absolute water-referenced values are needed to rule this out. (Becomes a gate once water-referenced data exist.)
- **F19, chemical-shift displacement.** PRESS at 3 T displaces each metabolite's voxel by about 2.4 mm per axis
  (≈ 12 % of the voxel). This is a property of the method, common to both sessions, not a session failure.
- **F08 / F23 need individual transients.** The StarterPack stores only the averaged spectrum, so drift and motion are N/A.

## Data contract requirement (catalogue v1.1)

Store individual MRS transients (NIfTI-MRS dimension DIM_DYN). They are needed for F08 and F23, and they are the only way
to tell frequency drift (fixable by alignment) from a bad shim (not fixable).

## To complete (QE)

- [ ] T2 values and TE dependence used for F35 (simulator assumption: NAA 247, Cr 162, Cho 207 ms at 3 T)
- [ ] GM / WM ratio assumptions used for F21 (simulator: Cho/Cr 0.48 GM, 0.74 WM)
- [ ] what a Cho/Cr change can and cannot mean physically
