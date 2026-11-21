# T1w / DICOM audit evidence (21 Nov 2026)

Derived tables only; no raw images. Report: `docs/scantwin/INFORME_T1w_DICOM.md`.

- `mrart_mriqc_vs_radiologist_score.csv`: OpenNeuro ds004173 v1.0.2 (MR-ART, CC0). Clinical artefact score
  (scans.tsv; 1 good, 2 medium, 3 unusable) joined with MRIQC 0.16.1 IQMs (derivatives/mriqc-0.16.1/group_T1w.tsv);
  `*_pct_of_standard` = same subject's no-motion scan as the reference.
- `dicom_series_audit.json`: `scripts/audit_dicom_series.py` on TCIA series (CC BY 4.0) and the StarterPack DICOM.
  Identifying tags are reported as present or empty, never with their values.
  vs_1..3 = Vestibular-Schwannoma-MC-RC VS-SEG-010; upenn_4 = UPENN-GBM-00491; remind_5 = ReMIND-051.
- `t1w_pipeline_real.json`: `scripts/run_t1w_real.py`, the repo's unchanged T1w pipeline on 15 real T1w volumes
  (MR-ART sub-000103/000175/003617, Maclaren ds000239 sub-01 run-01..04, Ford ds007806 sub-101 ses-1/2).
- `t1w_pipeline_real_after_fixes.json`: the same 15 volumes after the fixes (head mask on defaced images, F26 N/A
  when the background is removed, F27/F28 N/A without a brain mask, F32 = outline vs head, histogram k-means).
