# ScanTwin · T1w y DICOM: auditoría de datos, matriz de errores y diseño TRUSTY / WARNING / ERROR

Fecha: 21 nov 2026 (madrugada, antes del Gate 2). Referencia: `ScanTwin_traduccion_es.pdf` y el catálogo del equipo
(`catalogue/scantwin_catalogue.json`, 61 filas, con el change list v1.1 aplicado).

Etiquetas usadas en todo el informe:
- **[MEDIDO]**: lo calculé o lo abrí en esta sesión. Los archivos y scripts están en `external_data/t1w_dicom_audit/` y `scripts/`.
- **[INFERENCIA]**: se deduce de lo medido, pero no lo verifiqué directamente.
- **[PENDIENTE]**: no está validado, o no pude acceder a la fuente.

Ningún umbral de este informe es nuevo. Los que aparecen vienen del catálogo; si una métrica no tiene umbral, figura como
*pendiente de validación*. Los puntos de corte de MR-ART (§4.2) son **candidatos medidos** para que el bioingeniero
los revise; el código **no** los aplica.

> **Estado tras la implementación (21 nov, mañana).** Implementado:
> - lector DICOM (`twin_evidence/dicom_io.py`);
> - clasificación TRUSTY / WARNING / ERROR por categorías (`gates.classify`);
> - correcciones de T1w de §4.4: máscara con defacing, F26 N/A sin fondo, F27/F28 N/A sin máscara cerebral, F32
>   = contorno frente a cabeza, sin culpar a la máquina sin evidencia del escáner, k-means ~8× más rápido;
> - página "Check a scan" con DICOM, categorías y tabla de problemas.
>
> 88 tests pasan. Resultados reales tras los cambios: `external_data/t1w_dicom_audit/t1w_pipeline_real_after_fixes.json`.
> Ya no hay falsos ERROR. El % de fantasma sube con el movimiento en los 3 sujetos de MR-ART, pero con los cortes
> del catálogo (1/3/6 %, team proposal) solo 1 de las 3 imágenes inutilizables da WARN: **pendiente de calibrar**.
> El resto del documento describe el estado *antes* de estos cambios.

---

## 1. Resumen ejecutivo

1. **T1w (NIfTI) ya tiene código** (`twin_evidence/t1w.py`): gates F26, F27, F28, F32 y F35, más la comparación
   paciente / máquina / sesión / procesamiento. Hasta hoy solo se había probado con la T1w sintética del StarterPack.
   Hoy lo ejecuté **sin cambios** sobre 15 T1w reales de MR-ART, Maclaren y Ford (§4.3). Resultado:
   - **F26 sale PASS en los 15**, incluidas las tres que el radiólogo puntuó como inutilizables: el detector de
     movimiento está ciego con imágenes con defacing.
   - **F28 da FAIL o WARN en imágenes buenas.**
   - **F32 da FAIL en todas las repeticiones** de Maclaren y Ford, porque mide el reposicionamiento.

   Con datos reales produciría falsos ERROR y falsos TRUSTY (§4.4).
2. **MR-ART es el dataset de validación clave**:
   - 148 sujetos y 436 T1w; cada uno con una **puntuación clínica de neurorradiólogo** (1 buena, 2 media, 3 inutilizable)
     y con los IQMs de MRIQC 0.16.1.
   - Con esto validé F27 tal como está en el catálogo (CNR ≥ 80 % / ≥ 60 % de la referencia). Sobre las 288 adquisiciones
     con movimiento que tienen su propia referencia sin movimiento:
     - marca WARN o FAIL en el **80 %** de las puntuadas 3;
     - solo el **18 %** de ellas llega a FAIL **[MEDIDO]**.
   - Conclusión: el límite de FAIL en 60 % es permisivo frente al criterio clínico.
3. **DICOM no tiene lector todavía.** Audité 5 series DICOM reales de TCIA y las 2 del StarterPack con un script nuevo
   (`scripts/audit_dicom_series.py`). Aparecen problemas reales de datos:
   - **ReMIND:** sin campo ni frecuencia (F57 FAIL); `StudyDate` 1982 frente a `SeriesDate` 2007 dentro de la misma serie
     (F58); `SpacingBetweenSlices` = 1,0 mm cuando las posiciones reales están a 1,05 mm.
   - **VS-SEG-010:** cambio de fabricante y de secuencia entre momentos (F35).
   - **UPenn-GBM:** las series son `DERIVED\SECONDARY` (procesadas con CaPTk), no adquisiciones originales.
4. **Hueco en el catálogo.** No hay fila para *serie DICOM incompleta*, *coherencia geométrica de etiquetas
   (IPP/IOP/espaciado)* ni *coherencia de UIDs*. No creé códigos: propongo tres filas para un change list v1.2 que
   decide el bioingeniero (§3.3).
5. **Tres estados = las decisiones del catálogo con otro nombre:** TRUSTY = PASS / update, WARNING = WARN /
   update_with_warning, ERROR = FAIL / reject; N/A se muestra aparte. Añado una regla: si un gate del **conjunto
   mínimo** queda N/A, el global no puede ser TRUSTY (§5).
6. **Dos correcciones de lógica que propongo antes de la demo:**
   - (a) `t1w.compare` atribuye F27/F28 "a la máquina" sin evidencia del escáner. MR-ART demuestra que el movimiento
     baja el CNR, así que eso viola la regla "no culpar al escáner por una imagen mala".
   - (b) F32 se calcula con el desplazamiento bruto de la cabeza, no con el residuo de un registro. En Maclaren, un simple
     reposicionamiento da FAIL F32 (§4.4).
   - (c) La máscara automática con defacing se traga el fondo. Esto rompe F26, F28 y los volúmenes.

---

## 2. Auditoría de datasets (FASE 1)

| # | Dataset | Acceso real | Licencia | Contenido verificado | Repetidos | Uso en ScanTwin |
|---|---|---|---|---|---|---|
| 1 | **Maclaren** ds000239 | ✅ listado S3 + 4 T1w descargadas | CC0 [MEDIDO] | 3 sujetos × 40 T1w = **120 archivos** (`sub-XX_run-01..40_T1w.nii.gz`, ~7,5 MB). GE MR750 3 T, IR-SPGR 3D, TR 7,3 ms, TI 400 ms, flip 11°. **Un solo sidecar** (`T1w.json`) para todo el dataset. Sin `ses-`: el reparto de los 40 runs entre las 20 visitas está en `sourcedata/scan_number_vs_time_of_day.xlsx` (no lo abrí). El README cita `derivatives/mriqc`, pero **no está** en el bucket. Fondo puesto a 0 en el 58 % del FOV (defacing SPM / enmascarado) | Sí: ~40 por sujeto | Repetibilidad, reposicionamiento, falsos "cambios" |
| 2 | **MR-ART** ds004173 v1.0.2 | ✅ listado, 148 `scans.tsv`, tabla MRIQC, 9 T1w + sidecars | CC0 [MEDIDO] | 148 sujetos, **436 T1w** (standard 148, headmotion1 141, headmotion2 147). Siemens Prisma 3 T, MPRAGE (TR 2,3 s, TI 0,9 s, flip 9°). `scans.tsv` = puntuación clínica 1/2/3. MRIQC 0.16.1 con 68 IQMs por escaneo. Los sidecars incluyen `InstitutionName`, `InstitutionAddress`, `DeviceSerialNumber`, `TxRefAmp`, `ShimSetting` e `ImageType` con **ND** | 3 condiciones por sujeto, misma sesión | **Movimiento (prioridad)**, umbrales F26/F27, F59 |
| 3 | **Ford SPfMRS** ds007806 v2.0.0 | ✅ listado + T1w y sidecars de sub-101 ses-1/2 | CC0 [MEDIDO] | 40 sujetos × 2 sesiones (838 archivos, 45 GB). T1w MPRAGE Prisma_fit (TR 1,9 s, TE 2,34 ms). MRS HERMES (TE 80 ms, 780 tránsitos, ~535 MB por archivo) + referencia de agua. Entre ses-1 y ses-2 de sub-101 el protocolo es **idéntico**; solo cambian hora, orientación del corte y TxRefAmp (245,0 → 243,3) | 2 sesiones | F35 PASS esperado, T1w + MRS del mismo sujeto |
| 4 | **MSLesSeg** (figshare 27919209) | ⚠️ solo metadatos y README | CC BY [MEDIDO] | 115 escaneos / 75 pacientes, T1, T2, FLAIR, máscaras de lesión. Además del zip procesado (1,47 GB) **hay un `MSLesSeg_RAW.zip` (1,3 GB)** y las matrices de transformación, en contra de "solo procesados". No descargado | Longitudinal (115 / 75) | Comparación longitudinal, F28 sobre RAW **[PENDIENTE]** |
| 5/9 | **UPenn-GBM** (TCIA) | ✅ API (3680 series) + 1 serie T1 descargada | CC BY 4.0 [MEDIDO] | 630 pacientes, 3680 series MR, 3301 StudyInstanceUID. **Casi cada serie tiene su propio StudyInstanceUID**, así que no se agrupan sesiones por UID. Las T1 se llaman `... : Processed_CaPTk`; la serie abierta tiene `ImageType = DERIVED\SECONDARY\OTHER`, 192 cortes de 1 mm, TR 1760 ms, TI 950 ms, 3 T. Los "41 pares longitudinales" **no los verifiqué** | Sí (según la colección) | DICOM procesado: integridad y UIDs; no sirve para física de adquisición |
| 8 | **Vestibular-Schwannoma-MC-RC** (TCIA) | ✅ API (487 series) + 3 series T1 de un paciente | CC BY 4.0 [MEDIDO] | 160 pacientes, 427 estudios; 118 pacientes con 3 estudios. Multifabricante (Siemens, GE, Philips, Hitachi). Solo 124 series T1 (93 pacientes), casi todas con contraste y 2D (mediana 24 cortes). VS-SEG-010: Siemens 1,5 T SE 2D 3 mm (16 cortes) → GE 1,5 T FSPGR 3D 1,5 mm (116) → igual (116). Fechas desplazadas (1993/1996/1999) | Hasta 3 momentos | **F35 real**, multifabricante, integridad DICOM nativa |
| 10 | **ReMIND** (TCIA) | ✅ API (1346 series) + 1 serie T1 | CC BY 4.0 [MEDIDO] | 114 pacientes × 2 estudios (pre / intraoperatorio). MR 670, SEG 356, US 320. Fabricante "Unknown" (DICOM regenerado desde NRRD con PixelMed/dcmqi). La serie abierta no tiene campo, frecuencia, TR, TE ni flip; StudyDate ≠ SeriesDate | Pre / intra (cambio anatómico **real**) | Errores de metadatos reales; "cambio real ≠ error técnico" |
| 6 | **Morelli** (Zenodo 13753316) | ❌ Zenodo devuelve 403 desde este entorno | — | No inspeccionado | — | **[PENDIENTE]**: descargar a mano en el navegador |
| 7 | **SynthRAD2023** (Zenodo 7260705) | ❌ 403 | No comercial (según el prompt; no verificado) | No inspeccionado | — | **[PENDIENTE]**; la licencia NC impide usarlo en el producto que se vende |

Notas:
- **Licencias:** MR-ART, Maclaren y Ford (CC0) y las colecciones TCIA (CC BY 4.0, con cita) sirven para un producto
  comercial. Los datos crudos **no** van al repo (`.gitignore` ya excluye `real_data/`); en el repo solo hay derivados
  tabulares.
- **De-identificación en origen [MEDIDO]:**
  - TCIA marca `PatientIdentityRemoved = YES` y `PatientName = PatientID` = código de colección. Aun así conserva
    `PatientSex`, `PatientAge` y `PatientWeight`, y 3 etiquetas privadas.
  - En MR-ART los sidecars publicados mantienen la dirección del centro y el número de serie del escáner. Nuestro
    `t1w.deidentify` los elimina (IDENT_KEYS); conviene confirmarlo con un test (T-10).

---

## 3. Matriz de errores (FASE 2)

### 3.1 Filas existentes del catálogo (definiciones literales en `catalogue/scantwin_catalogue.json`)

| Fuente (PDF) | ID | Métrica del catálogo | PASS / WARN / FAIL (catálogo) | Fuente del umbral | Código hoy | Dataset de validación | Estado esperado |
|---|---|---|---|---|---|---|---|
| 3 Movimiento | **F26** | Puntuación de movimiento 1–4 o nitidez | 1 / 2 / ≥ 3 | Team proposal | ✅ % de fantasma → puntuación (cortes 1/3/6 %, team proposal) | **MR-ART** (puntuación clínica) | standard → TRUSTY; puntuación 3 → WARNING/ERROR |
| 1 SNR bajo | **F27** | CNR GM/WM frente a referencia | ≥ 80 % / 60–80 % / < 60 % | Team proposal | ✅ necesita referencia | MR-ART (medido, §4.2), StarterPack `degrade('noise')` | ruido ×2 → WARN/FAIL |
| (bobina/B1) | **F28** | CV de la intensidad de SB | ≤ 10 / 10–20 / > 20 % | Team proposal | ⚠️ el código mide el CV del **campo de sesgo ajustado**, no de la intensidad de SB (discrepancia con el catálogo) | StarterPack `degrade('bias')`; MSLesSeg RAW | — |
| 2 Shim | **F29** | Solapamiento del artefacto de susceptibilidad con la ROI | Sin solape / borde / solapa | Team proposal | ❌ cualitativo | — | Revisión visual → WARNING "requiere revisión" |
| — | **F30** | Aliasing que solapa el cerebro | Ninguno / fuera / solapa | Team proposal | ❌ | — | idem |
| — | **F31** | Gibbs visible | Ninguno / visible / sobre la ROI | Team proposal | ❌ | — | idem |
| 5 Desplazamiento | **F32** | RMS del registro (mm) | ≤ 2 / 2–3 / > 3 | Dossier §5.8 | ⚠️ mide la **traslación bruta**, no el residuo del registro | **Maclaren** (reposicionamiento) | reposicionamiento + registro correcto → PASS |
| — | **F33** | u declarada frente a u implícita (3σb/r) | declarada ≥ implícita / solo WARN | Dossier §5.8 | ✅ (`validate.roi_u_consistency`) | StarterPack E-t1-roi | WARN (reproducido) |
| — | **F34** | Versión de segmentación igual | igual / distinta en ambas / distinta en una | Team proposal | ✅ indirecto (contorno cambia, imagen no) | StarterPack ROI +25,6 % | "procesamiento" |
| 7 Protocolo | **F35** | Familia de secuencia, TE, TR, TI, flip, campo | todo igual / difiere (mismo campo) / campo distinto sin armonizar | Dossier §5.8 | ✅ TR, TE, TI, flip y campo desde el sidecar (falta la familia de secuencia) | **VS-SEG-010** (SE → FSPGR), Ford (igual) | VS → WARNING atribuido a **protocolo**; Ford → PASS |
| Geometría | **F01** | Longitudes del fantoma (mm) | ≤ 1 / 1–2 / > 2 | ACR + team | Solo gemelo del escáner (simulado) | — (sin fantomas reales) | N/A sin datos del escáner |
| Geometría | **F36** | Corrección de distorsión aplicada; distancia al isocentro | aplicada / no y < 100 mm / no y ≥ 100 mm | Dossier §5.5 + team | ❌ | MR-ART y Ford: `ImageType` con **ND**; VS: **DIS2D** [MEDIDO] | MR-ART → WARNING (no aplicada; cerebro < 100 mm del isocentro **[INFERENCIA]**) |
| Escáner | **F11** | system_qc_id presente y PASS | presente y PASS / — / falta | Dossier §5.8 | ✅ | — | Sin QC del escáner → no se puede atribuir al escáner |
| Datos | **F54** | Paquete válido según schema v1.0 | válido / opcional falta / obligatorio falta | Dossier §1.9 | ✅ paquetes | paquetes trampa del Gate 3 | — |
| Datos | **F55** | Valor dentro de valid_range | dentro / — / fuera | Dossier §5.8 | ✅ | — | — |
| Datos | **F56** | Unidad presente y coherente | presente / — / falta o no coincide | Dossier §6 | ✅ paquetes; ❌ DICOM | **Trampa real:** DICOM guarda TR/TE en **ms** y BIDS en **s** | Comparar DICOM con BIDS sin normalizar → falso F35 |
| Datos | **F57** | Campo nominal y ν0 presentes | ambos / solo campo / ninguno | Dossier §4.2 | ✅ paquetes MRS | **ReMIND** (ninguno) [MEDIDO] | **ERROR** |
| Datos | **F58** | Fecha de sesión coherente | coherente / solo tiempo relativo / falta o contradictoria | changelog X14 + team | ⚠️ parcial | **ReMIND** (StudyDate ≠ SeriesDate); Maclaren (sin fechas) | ReMIND → ERROR; Maclaren → WARNING |
| Datos | **F59** | Comprobación de de-identificación | solo IDs seudónimos / — / campo identificativo | Dossier App. I | ✅ paquetes, NIfTI-MRS, sidecars T1w; ❌ etiquetas DICOM | **MR-ART** sidecars, TCIA (Sex/Age/Weight) | ver política §5.4 |
| Cambio | **F60** | z = Δ / u(Δ) | \|z\| < 1,96 / 1,96–3 con WARN / — | Dossier §3.10 | ✅ | Maclaren (sin cambio real), ReMIND intra (cambio real) | Maclaren → "sin cambio" |

Errores 4 (deriva de frecuencia) y 6 (agua residual): **N/A en T1w** (PDF §2). La deriva solo se notaría como un leve
desplazamiento en la dirección de lectura; no hay evidencia para hacer un gate con ella y no se propone ninguno.

### 3.2 Comprobaciones DICOM → fila del catálogo

| Comprobación | Tipo de error | Fila existente | Cómo se detecta [MEDIDO en el script] |
|---|---|---|---|
| Archivo ilegible o corrupto | Archivo | (ninguna exacta) → ERROR de ingesta | `pydicom.dcmread` falla; en TCIA viene un `LICENSE` en la carpeta: hay que ignorarlo, no tratarlo como error |
| Cortes ausentes / huecos | Archivo | **sin fila** → propuesta A | huecos de IPP proyectada sobre la normal mayores que la mediana; saltos en `InstanceNumber` |
| Posiciones duplicadas | Archivo | propuesta A | separación < 1e-3 mm |
| Etiquetas obligatorias ausentes | Metadatos | **F54** (FAIL obligatoria / WARN opcional) | lista mínima: IPP, IOP, PixelSpacing, Rows, Columns, campo, TR, TE |
| PixelSpacing / SliceThickness / SpacingBetweenSlices frente a IPP | Metadatos / geometría | **sin fila** → propuesta B | ReMIND: Spacing 1,0 frente a IPP 1,05 mm |
| IOP no ortonormal o variable dentro de la serie | Geometría | propuesta B | producto escalar y normas; nº de IOP distintos |
| Rows/Columns ≠ píxeles reales | Archivo | F54 (tipo incorrecto) | `pixel_array.shape` |
| UIDs (Study / Series únicos, SOP sin repetir) | Metadatos | **sin fila** → propuesta C | conteos |
| Parámetros que cambian dentro de la serie | Adquisición | F35 (variante intra-serie) | conjunto de valores por etiqueta |
| Fechas incoherentes | Metadatos | **F58** | Study / Series / Acquisition / Content |
| Datos personales | Metadatos | **F59** | lista de etiquetas, sin guardar valores |
| Unidades (ms ↔ s, RescaleSlope) | Metadatos | **F56** | normalizar a SI antes de comparar |
| Campo / frecuencia | Metadatos | **F57** | etiquetas (0018,0087) y (0018,0084) |
| Series incompatibles en un estudio | Adquisición | F35 | tabla de protocolo entre series |
| Imagen derivada, no original (`ImageType` DERIVED) | Contexto | (ninguna) → **nota de limitación**, no un gate | UPenn-GBM |

### 3.3 Propuestas para el change list v1.2 (NO creadas; decide el bioingeniero)

- **A. Serie DICOM completa:** nº de cortes igual al esperado, sin huecos ni duplicados. Propuesta de estado: FAIL si
  falta un corte. El umbral está *pendiente*, aunque un corte ausente rompe el volumen por definición.
- **B. Coherencia geométrica:** IOP ortonormal y constante; IPP equiespaciadas; `SpacingBetweenSlices` igual a la
  separación medida. Umbral de tolerancia *pendiente de validación*.
- **C. Coherencia de UIDs:** un Study y una Series por serie; SOP únicos.

Alternativa sin filas nuevas: ampliar el texto de F54 a "paquete **o serie DICOM** válido". Es más rápido, pero mezcla
dos cosas distintas.

---

## 4. Métricas QC (perspectiva A)

### 4.1 Tabla de métricas

| Métrica | Qué mide | Cálculo (ScanTwin / referencia) | Necesita | Detecta | Limitaciones | Umbral | ¿Método de validación? |
|---|---|---|---|---|---|---|---|
| SNR (snr_wm) | Señal frente a ruido en SB | media SB / SD robusta del residuo local 3×3×3 (como MRIQC snr_wm) | T1w + segmentación | Señal débil, bobina | Depende del filtrado del vendedor y de GRAPPA; no comparable entre escáneres | **Sin fila en el catálogo**; solo se usa relativo | ROC frente a la puntuación de MR-ART (§4.2) |
| CNR | Separación GM/WM | (SB − SG) / ruido | T1w + segmentación + **referencia** del mismo sujeto | Ruido, movimiento, protocolo | El movimiento también lo baja, así que **no es específico del escáner** | **F27** 80 / 60 % | §4.2 medido |
| Uniformidad | Campo de sesgo | CV del sesgo ajustado (código) / CV de intensidad en SB (catálogo) / MRIQC inu_range | T1w + SB | Bobina, B1 | Ver discrepancia F28 | **F28** 10 / 20 % | **[PENDIENTE]** |
| Movimiento / fantasmas | Señal fuera de la cabeza en la dirección de fase | % de fantasma → puntuación F26 | T1w con **fondo no enmascarado** | Movimiento | **Ciego** si el fondo está a 0 (StarterPack, Maclaren 58 % del FOV) o recortado por defacing → N/A, no PASS | **F26** (puntuación); los cortes 1/3/6 % son team proposal | MR-ART (pendiente con nuestro detector; §4.3) |
| Blur | Nitidez | MRIQC fwhm_avg | T1w | Movimiento, filtrado | En MR-ART **no separa** la puntuación (AUC 0,46) [MEDIDO] | Ninguno | — |
| Distorsión geométrica | Escala / no linealidad | Fantoma (F01) o `ImageType` (F36) | Fantoma o etiquetas | Gradientes | Sin fantoma no se mide en el sujeto | F01, F36 | ACR |
| Calidad de segmentación | Plausibilidad de la máscara | MRIQC tpm_overlap, rpve; en ScanTwin nada | Atlas | Fallos de máscara | Sin skull stripping, la "máscara" es de **cabeza** | Ninguno **[PENDIENTE]** | — |
| Registro | Alineación entre sesiones | Hoy: traslación por correlación de fase. Debería ser: **residuo** tras registro rígido | Dos T1w | Desalineación | Hoy confunde reposicionamiento con error de registro | **F32** 2 / 3 mm | Maclaren |
| Volumen | Consistencia | Volúmenes k-means; z de F60 | Segmentación | Cambio aparente | Con máscara de cabeza, "cerebro" incluye cuello y cráneo | F60 (z) | Maclaren (prueba-reprueba) |
| Calidad global | Agregado | Peor estado de los gates | Todo lo anterior | — | — | Reglas del catálogo, pasos 4–6 | — |

### 4.2 Validación medida con MR-ART (MRIQC frente a neurorradiólogo) [MEDIDO]

Tabla cruzada condición × puntuación (n = 436):

| | puntuación 1 | puntuación 2 | puntuación 3 |
|---|---|---|---|
| standard | 119 | 28 | 1 |
| headmotion1 | 7 | 59 | 75 |
| headmotion2 | 3 | 22 | 122 |

Las "sin movimiento" no siempre son buenas: 29 de 148 tienen puntuación ≥ 2.

AUC de cada IQM absoluto para separar la puntuación (orientado de modo que 1 = perfecto):

| IQM | puntuación ≥ 2 | puntuación 3 |
|---|---|---|
| snr_wm | 0,93 | 0,96 |
| cnr | 0,89 | 0,93 |
| fber | 0,90 | 0,93 |
| cjv | 0,91 | 0,93 |
| inu_range | 0,73 | 0,74 |
| efc | 0,73 | 0,78 |
| fwhm_avg | 0,54 | 0,57 |

F27 **con los límites del catálogo**, sobre las 288 adquisiciones con movimiento comparadas con su propia `standard`:

| F27 | puntuación 1 | puntuación 2 | puntuación 3 |
|---|---|---|---|
| PASS | 9 | 53 | 40 |
| WARN | 1 | 26 | 121 |
| FAIL | 0 | 2 | 36 |

- Detecta puntuación 3 (WARN o FAIL): **80 %**. Llega a FAIL: **18 %**. Falsas alarmas en puntuación 1: 1 de 10
  (n muy pequeño).
- Punto de corte de Youden para puntuación 3 con CNR % de la referencia: < 75 % (sensibilidad 0,68, especificidad
  0,85). Con SNR_wm %: < 70 % (0,74 / 0,84). **Son candidatos, no umbrales del catálogo.** Proceden del CNR de MRIQC, no
  del nuestro, y de un solo escáner.
- Lectura **[INFERENCIA]**: el WARN de F27 en 80 % está bien situado. El FAIL en 60 % deja pasar como "utilizable"
  muchas imágenes que un radiólogo considera inutilizables. Decisión del bioingeniero: mantenerlo o moverlo
  documentando la fuente (MR-ART).

### 4.3 El pipeline T1w actual sobre datos reales [MEDIDO]

Código del repo sin cambios (`t1w.load` + `measure` + `gate` + `compare`). Script: `scripts/run_t1w_real.py`;
salida: `external_data/t1w_dicom_audit/t1w_pipeline_real.json`.

| Escaneo (referencia) | Puntuación clínica | SNR_wm | CV F28 | Fantasma F26 | CNR % ref. F27 | Desplaz. F32 | Cabeza Δ% | Global | `compare` |
|---|---|---|---|---|---|---|---|---|---|
| MR-ART 000103 standard | 1 | 27,9 | 15,9 % WARN | 0 → PASS | — | — | — | WARN | — |
| 000103 headmotion1 | 2 | 25,5 | WARN | 0 → PASS | 91,8 PASS | 1,6 PASS | −0,4 | WARN | ❔ sin cambio |
| 000103 headmotion2 | 3 | 21,8 | WARN | 0 → PASS | 79,7 WARN | 2,4 WARN | −0,7 | WARN | 🎯 sesión |
| 000175 standard | 1 | 23,8 | 23,3 % **FAIL** | 0 → PASS | — | — | — | **FAIL** | — |
| 000175 headmotion2 | 3 | 21,5 | FAIL | 0 → PASS | 91,4 PASS | 0,7 PASS | +0,3 | FAIL | ❔ sin cambio |
| 003617 standard | 1 | 26,2 | 21,5 % **FAIL** | 0 → PASS | — | — | — | **FAIL** | — |
| 003617 headmotion2 | 3 | 22,8 | FAIL | 0 → PASS | 86,9 PASS | 1,1 PASS | −0,3 | FAIL | ❔ sin cambio |
| Maclaren sub-01 run-02 (run-01) | — | 18,6 | 16,1 % WARN | 0 → PASS | 99,5 PASS | **4,2 FAIL** | −6,7 | FAIL | 🎯 "la sesión salió mal" |
| run-03 (run-01) | — | 18,7 | WARN | 0 → PASS | 100,3 | **9,0 FAIL** | −4,6 | FAIL | 🎯 idem |
| run-04 (run-01) | — | 18,8 | WARN | 0 → PASS | 100,8 | **5,7 FAIL** | −4,0 | FAIL | 🎯 idem |
| Ford sub-101 ses-2 (ses-1) | — | 21,3 | 24,0 % FAIL | 0 → PASS | 116 | **11,9 FAIL** | −5,2 | FAIL | 🎯 idem |

(Fila completa de los 15 escaneos en el JSON.) La de-identificación funcionó en los 12 sidecars que traían
identificativos: se eliminaron `InstitutionName`, `InstitutionAddress` y `DeviceSerialNumber` (T-10 ✔). Cada volumen
tarda **55–112 s**.

### 4.4 Fallos de diseño detectados (por orden de gravedad)

1. **F26 está ciego con datos reales y lo reporta como PASS.** El fantasma sale 0,0 en los 15 escaneos, incluidas las
   tres puntuadas 3 por el radiólogo. Causa: en volúmenes con más del 20 % de ceros (defacing), `_auto_mask` toma como
   "cabeza" todo vóxel ≠ 0, ruido de fondo incluido. El "fondo" queda reducido a los ceros exactos, y `_ghost_pct` lo
   interpreta como "fondo limpio" → 0 %. Corrección: máscara por umbral (Otsu) también con defacing, y **N/A** (no PASS)
   cuando el fondo no es evaluable.
2. **F28 da ERROR en imágenes buenas.** Las tres `standard` con puntuación 1 dan CV 15,9–23,3 % → WARN/FAIL, y Ford y
   Maclaren también fallan. La causa más probable **[INFERENCIA]** es la misma máscara (incluye fondo y cuello) más que
   el CV mida el polinomio del sesgo y no la intensidad de SB. Con datos reales, hoy **F28 no se puede usar** para
   decidir.
3. **F32 confunde reposicionamiento con error de registro.** Maclaren repite al mismo sujeto (desplazamientos de
   4–9 mm) y Ford tiene 11,9 mm. En los cuatro casos la salida es FAIL y "🎯 This session went wrong", cuando la imagen
   está bien (CNR 99–116 %). F32 tiene que medir el residuo **después** de un registro rígido.
4. **"Volumen cerebral" = soporte no nulo:** 8–10 L en MR-ART. Por eso el cambio de tamaño de la "cabeza" llega a −6,7 %
   entre dos repeticiones de Maclaren. Si no lo frena el F32 FAIL, la regla "> 2 % → 🛠 máquina (geometría)" dispararía
   falsamente.
5. **F27 con nuestro CNR detecta menos que el de MRIQC.** Dos de las tres puntuadas 3 dan 87–91 % (PASS). Con el CNR de
   MRIQC, F27 marca el 80 % de las puntuadas 3 (§4.2). Nuestra segmentación k-means sin máscara cerebral es la sospechosa
   **[INFERENCIA]**.
6. **Lentitud:** 1–2 min por volumen no sirve para el Gate 3 en directo; hace falta un k-means por histograma.

Conclusión: el módulo T1w funciona con la T1w sintética, pero **con datos reales hoy produciría falsos ERROR y falsos
PASS**. Los puntos 1–3 van antes de cualquier demo con datos reales.

---

## 5. Lógica TRUSTY / WARNING / ERROR (FASE 4)

### 5.1 Correspondencia (sin umbrales nuevos)

| Estado ScanTwin | Catálogo (Decision rules) | Acción del gemelo |
|---|---|---|
| **TRUSTY** | todos los gates aplicables en PASS (paso 6) **y** el conjunto mínimo evaluado | update, R = u² |
| **WARNING** | ≥ 1 WARN, sin FAIL (paso 5), **o** un gate del conjunto mínimo en N/A | update_with_warning, R = 2u² (o sin update si no hay valor) |
| **ERROR** | ≥ 1 FAIL (paso 4) o ingesta imposible | reject: el estado del gemelo no cambia |
| **N/A** (por gate) | paso 3: falta el dato → N/A con motivo, **nunca FAIL** | no cuenta, pero se muestra |

### 5.2 Conjunto mínimo para poder decir TRUSTY (propuesta, a confirmar por el equipo)

- **NIfTI T1w (un escaneo):** volumen legible 3D + F59 + F57 + F26 + F28.
- **DICOM:** lo anterior + propuestas A y B (o F54 ampliada) + F58.
- **Con referencia:** + F27, F32 y F35.

Si un gate de ese conjunto queda N/A, el resultado es *WARNING: "no evaluable: falta X"*. Así se cumple a la vez la regla
v1.1 del catálogo ("falta el dato → N/A, nunca FAIL") y la del prompt ("ausencia de información ≠ ausencia de errores").
Consecuencia visible: la T1w del StarterPack (fondo a 0, F26 N/A) pasaría de 🟢 a 🟡 *"movimiento no evaluable"*.
Es una decisión del equipo.

### 5.3 Flujo

```
archivo(s) ─► 0 INGESTA ── ilegible / no 3D / serie rota ──────────────────► ERROR (motivo)
               │ DICOM: agrupar por SeriesInstanceUID, ignorar no-DICOM (LICENSE)
               ▼
            1 DATOS: F54 (+A, B, C) · F56 normalizar ms→s · F57 · F58 · F59 (eliminar y registrar)
               ▼  (pydicom → volumen + afín LPS→RAS; mismo T1Scan que NIfTI)
            2 IMAGEN (un escaneo): F26 · F28 · [F29–F31 → "revisión visual", WARNING si se marca]
               ▼
            3 LONGITUDINAL (si hay referencia): F35 · F32 (residuo) · F27 · F33 · F34 · F60 (z)
               ▼
            4 ESCÁNER: F11 / F01 / F36 si hay historial o QC; si no → "sin historial: no se compara"
               ▼
            AGREGADO: por categoría = peor estado de sus gates (N/A si todos N/A);
                      global = peor categoría, con la regla del conjunto mínimo  ─► TRUSTY / WARNING / ERROR
            ATRIBUCIÓN (§6) ─► paciente / escáner / protocolo / datos / procesamiento / indeterminada
```

### 5.4 F59: política propuesta para los archivos de entrada

El catálogo define F59 sobre el **paquete**. Un archivo de entrada con datos personales no es una medición mala, y
ScanTwin los elimina antes de crear el paquete (como ya hace con NIfTI-MRS). Propuesta:

| Situación | Estado |
|---|---|
| Entrada con campos identificativos eliminados y registrados por nombre | **WARNING** ("se eliminaron N campos"), paquete F59 PASS |
| No se pueden eliminar (p. ej. `BurnedInAnnotation = YES`, texto en píxeles) | **ERROR** |

Pendiente de decisión del equipo: ¿cuentan como identificativos `PatientSex`, `PatientAge` y `PatientWeight`, que TCIA
conserva? El código actual los elimina (prefijo `Patient`).

---

## 6. Identificación del origen (perspectiva D)

| Origen | Evidencia que lo permite afirmar | Confianza |
|---|---|---|
| **Paciente** (movimiento) | F26 peor que en la referencia; fantasmas en la dirección de fase | media (detector team proposal) |
| **Paciente** (anatomía) | todos los gates PASS, imagen cambiada más allá del ruido, \|z\| ≥ 1,96 (F60), geometría estable | media; confirmar con repetición |
| **Paciente** (reposicionamiento) | desplazamiento bruto grande **con residuo de registro pequeño** | alta |
| **Escáner** | **solo** con evidencia del equipo: F11 FAIL o ausente, fantoma (F01) o el mismo patrón en varios sujetos del mismo escáner (historial). TxRefAmp y ShimSetting de los sidecars son candidatos para ese historial | alta con fantoma; si no, **no se atribuye** |
| **Protocolo** | F35 (diferencia en etiquetas) | alta (es un hecho del encabezado) |
| **Datos** | F54–F59 y propuestas A–C | alta |
| **Procesamiento** | contorno cambiado con imagen sin cambio (F34), F33, residuo de registro alto (F32) | media |
| **Indeterminado** | F27 o F28 empeoran sin F26 ni evidencia del escáner | se dice así: "calidad de imagen baja; causa posible: bobina/escáner o movimiento no detectado" |

**Cambio de código necesario:** hoy `t1w.compare` devuelve "🛠 Probably the machine" cuando F27 o F28 empeoran
(`MACHINE_GATES`). Sin F11 ni historial, eso debe ser **"❔ Calidad baja, causa indeterminada"**. Si el gemelo del
escáner tiene historial **real** del mismo equipo, la atribución "escáner" sí es posible.

Además, la página no debe usar el `ScannerRecord` simulado para escaneos subidos reales.

Tampoco se debe culpar al escáner de un cambio anatómico real, como en el intraoperatorio de ReMIND (resección y
desplazamiento del cerebro): si la calidad pasa y la imagen cambió, la salida es "paciente / cambio real", nunca "error".

---

## 7. Diseño de la interfaz del gemelo (ampliando "Check a scan")

- **Nivel 1:** la tarjeta actual 🟢 / 🟡 / 🔴 con la etiqueta **TRUSTY / WARNING / ERROR** y una frase de motivo.
- **Nivel 2:** tabla de categorías:

  | Categoría | Gates |
  |---|---|
  | Calidad de imagen T1w | F27, F28 |
  | Integridad DICOM | A, C, F54 |
  | Geometría y orientación | B, F36, F01 |
  | Movimiento | F26 |
  | Protocolo | F35 |
  | Integridad de datos | F56–F59 |
  | Longitudinal | F32, F33, F34, F60 |

  Cada categoría muestra su estado, o N/A con el motivo.
- **Nivel 3:** una fila por anomalía con estas columnas:
  - nombre;
  - ID del catálogo;
  - valor medido;
  - valor de referencia (límite del catálogo y fuente);
  - gravedad;
  - evidencia (archivo y etiqueta, o métrica);
  - origen probable;
  - confianza;
  - corrección (columna "Fix for the next scan" del catálogo).
- **Nivel 4, escáner:**
  - Con historial real del mismo equipo, identificado por un hash con sal del `DeviceSerialNumber` calculado **antes**
    de eliminarlo: se muestra la tendencia (TxRefAmp, SNR del fantoma).
  - Sin historial: el texto fijo *"No hay historial de este escáner: no se puede comparar ni atribuir un fallo al
    equipo"*.
  - La simulación solo aparece en el *Hackathon lab*, etiquetada SIMULATED.

---

## 8. Arquitectura técnica (FASE 5): sin tocar MRS

| Módulo | Cambio |
|---|---|
| `twin_evidence/dicom_io.py` (**nuevo**) | Agrupar por serie; comprobaciones A/B/C y F54/F56/F57/F58/F59; volumen + afín (LPS → RAS, como el changelog X11 del StarterPack); sidecar normalizado a SI (s); devuelve el mismo `T1Scan` que `t1w.load` |
| `twin_evidence/gates.py` | Añadir `DATA_GATES` y `CATEGORIES`; función `classify(gates_result, minimum_set)` → TRUSTY/WARNING/ERROR + categorías. `evaluate_mrs` no se toca: la clasificación lo envuelve, así que los números del Gate 1 no cambian |
| `twin_evidence/t1w.py` | F32: registro rígido y RMS del residuo. F26: N/A si el fondo está enmascarado o recortado. F28: alinear la métrica con el catálogo (o cambiar el texto de la fila). Quitar la atribución automática a la máquina. Acelerar el k-means (histograma): hoy tarda **~50 s por volumen 256³** [MEDIDO] |
| `twin_evidence/check_page.py` | Aceptar `.dcm` y `.zip` → `dicom_io`; tabla de categorías; panel del escáner con "sin historial" |
| `scripts/audit_dicom_series.py` (añadido hoy) | Auditoría independiente de una carpeta DICOM (JSON sin valores personales) |
| `catalogue/changes_v1.2.json` | Solo si el bioingeniero aprueba A/B/C o cambia los límites de F27 |

---

## 9. Plan de validación (FASE 6)

| Test | Entrada | Esperado | Fuente de verdad | Limitación |
|---|---|---|---|---|
| T-01 imagen correcta | T1w del StarterPack (baseline) | sin FAIL; WARNING solo si se aplica el conjunto mínimo (F26 N/A) | sintético conocido | No es real |
| T-02 movimiento | `degrade('motion', 0.1 / 0.4)`; MR-ART hm1/hm2 | leve → WARN, fuerte → WARN/FAIL; MR-ART puntuación 3 → no TRUSTY | simulación; neurorradiólogo | Nuestro detector F26 en MR-ART: ver §4.3 |
| T-03 ruido | `degrade('noise', 2)` | F27 WARN/FAIL; origen **indeterminado** (sin escáner) | simulación | — |
| T-04 serie incompleta | copia temporal del DICOM del StarterPack sin `IM0008` | **ERROR** (A) | construcción | — |
| T-05 orientación | IOP alterada / no ortonormal en la copia | ERROR (B) | construcción | — |
| T-06 metadatos | ReMIND-051 (real) | F57 FAIL, F58 FAIL → ERROR; aviso de Spacing (B) | etiquetas medidas | Una sola serie |
| T-07 protocolo | VS-SEG-010 (SE → FSPGR) | F35 WARN, origen = **protocolo**, nunca escáner | etiquetas | Fechas desplazadas por TCIA |
| T-07b unidades | mismo protocolo, DICOM (ms) frente a BIDS (s) | F35 **PASS** tras normalizar (F56) | construcción | — |
| T-08 cambio real | `degrade('grow', 4)` | "paciente", no error | simulación | ReMIND intra necesita segmentación |
| T-09 repetibilidad | Maclaren sub-01 run-01..04 | sin cambio (\|z\| < 1,96); F32 no debe hacer FAIL por reposicionamiento | prueba-reprueba | Agrupación por visita sin verificar |
| T-10 de-identificación | sidecar de MR-ART | `InstitutionAddress` y `DeviceSerialNumber` eliminados y registrados | sidecar real | — |
| T-11 DICOM ↔ NIfTI | DICOM frente a NIfTI del StarterPack | misma geometría y orientación | changelog X11 | — |
| T-12 MRS intacto | suite actual (69 tests) | todo pasa, Gate 1 igual | `reproduce.py` | — |

---

## 10. Limitaciones

- No hay fantomas ni historial real de ningún escáner, así que **ninguna atribución "escáner" es posible con datos
  reales** hoy.
- Los umbrales T1w son team proposal (F26–F31, F34) o vienen del Dossier (F32, F35). Solo F27 se contrastó con un
  criterio externo, y fue con un único escáner (Prisma) y con el CNR de MRIQC.
- Sin skull stripping, los volúmenes de "cerebro" con datos reales son de cabeza. No sirven para F60 con datos reales
  sin una máscara cerebral externa.
- El defacing (SPM, pydeface) y el enmascarado del fondo dejan ciego el detector de fantasmas.
- UPenn-GBM y ReMIND son DICOM procesados o regenerados: valen para pruebas de **datos**, no de física de adquisición.
- No pude inspeccionar Morelli ni SynthRAD (Zenodo 403), ni el contenido de MSLesSeg.
- F29–F31 y F37 son cualitativos: sin un detector, solo admiten revisión visual.

---

## 11. Plan de implementación (orden y motivo)

1. **`dicom_io` + comprobaciones de datos + T-04/T-05/T-11** (~1,5 h). Es el "caso que el prototipo no resuelve" del
   Gate 2 y usa el DICOM del StarterPack, que ya conocemos.
2. **`classify` (TRUSTY/WARNING/ERROR + categorías + conjunto mínimo) y quitar la atribución automática a la máquina**
   (~1 h). Lo pide la regla central del prompt, y el Gate 3 decide paquetes con motivos.
3. **Máscara con defacing, F26 N/A cuando el fondo no es evaluable, y F32 como residuo de registro** (~1–1,5 h).
   Sin esto, los datos reales dan falsos ERROR (F28, F32) y falsos PASS (F26). Test de regresión: las 15 filas de
   `t1w_pipeline_real.json`.
4. **Script de validación MR-ART** (tabla §4.2 reproducible) → decisión del bioingeniero sobre F27 (y v1.2).
5. **UI de categorías y panel del escáner "sin historial"**.
6. Después de la hackatón: optimizar la segmentación, descargar Morelli y MSLesSeg RAW, y crear el historial del escáner
   con TxRefAmp.

---

## Anexo: cómo respondería ScanTwin (ejemplos)

**A. Escaneo correcto, NIfTI.** Ejemplo: MR-ART `standard` con puntuación 1, sin referencia. Esta es la respuesta
**objetivo, tras las correcciones de §4.4**. Hoy el código da WARN o FAIL por F28 en este mismo archivo [MEDIDO].

> 🟢 TRUSTY si el detector de movimiento puede evaluar el fondo; si no, 🟡 WARNING "movimiento no evaluable (el fondo
> fue recortado o enmascarado)".
> - Calidad de imagen: F28 PASS. F27 N/A (necesita referencia).
> - Datos: F57 PASS (3 T, 123,21 MHz). F59: se eliminaron `InstitutionName`, `InstitutionAddress` y `DeviceSerialNumber`.
> - Geometría: F36 WARNING (ImageType "ND": sin corrección de distorsión).
> - Escáner: sin historial, no se compara.

**B. Movimiento.** Ejemplo: `headmotion2` frente a su `standard`.

> 🔴 ERROR o 🟡 WARNING según F26/F27: *"La segunda adquisición no es comparable: el paciente se movió (F26) y eso
> también bajó el contraste a X % de la referencia (F27). Origen: paciente (confianza media). Corrección: acolchado,
> instrucciones, secuencia corregida por movimiento; repetir."*
> Nunca "escáner".

**C. Metadatos.** Ejemplo: ReMIND-051, medido.

> 🔴 ERROR.
> - Integridad de datos: F57 FAIL (no consta ni el campo ni la frecuencia: no se puede comprobar la comparabilidad);
>   F58 FAIL (StudyDate 1982 frente a SeriesDate 2007 en la misma serie).
> - Geometría: aviso, SpacingBetweenSlices 1,0 mm frente a 1,05 mm medidos entre posiciones.
> - Contexto: ImageType MPR, fabricante "Unknown": el DICOM fue regenerado.
> - Origen: datos (confianza alta). La imagen puede ser buena, pero la medición no es trazable.

**D. Serie DICOM incompleta.** Ejemplo: StarterPack sin el corte 8 (prueba construida).

> 🔴 ERROR. Integridad DICOM: falta 1 corte (hueco de 8 mm donde la separación es 4 mm, entre los InstanceNumber 7 y 9).
> No se reconstruye el volumen y no se mide nada. Origen: datos o exportación (confianza alta). Corrección: reexportar la
> serie completa desde el PACS.
