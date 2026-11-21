# ScanTwin – Slide guideline and 3-minute pitch

Pitch: 3 minutes strict, then 2 minutes of questions.
Flow: introduction → problem → why it happened → how it works → live demo → change over time → reliability → limitations and next steps.

Before presenting, check with the data scientist which choline verdict the demo shows:
- **With the scanner twin:** "🛠 It's the machine". The shim failed on the scan date, so the scan is blocked.
- **Without it:** "Use with caution". The scan counts at reduced weight, and the twin moves from 0.69 to 0.74, not 0.85.

Use whichever the demo shows, everywhere.

---

## Slide 1 – Introduction

**Title:** Is it the patient, or the machine?

**Text:**
> ScanTwin is a digital twin of the MRI scanner and of every measurement it makes.
> Before anyone acts on a follow-up result, it tells you whether the scan can be trusted, and if not, why.

**Visual:** product name, team name, "InnovateU NMR challenge · Ulm 2026".

---

## Slide 2 – The problem

**Title:** One follow-up scan, two alarming numbers

**Table:**

| | May | November | Change |
|---|---|---|---|
| Choline / creatine (brain chemistry) | 0.69 | 0.85 | **+23%** |
| Brain region volume (anatomy) | 22,140 mm³ | 27,810 mm³ | **+26%** |

**Text:**
> **Why this matters:**
> - Choline reflects how fast cells rebuild their membranes. Doctors watch for a rising choline because it can signal increased cell activity, for example around a tumour or inflammation.
> - A healthy brain region changes by well under 1% a year. A 26% jump in six months would suggest something is growing.
> - Together, these two numbers would worry any patient and lead to more tests.
>
> **The question:** did the brain change, or did the scan?

**Footnote:** Hackathon StarterPack · synthetic subject

**Visual:** +23% and +26% large, in red.

---

## Slide 3 – What went wrong, and what we offer

**Title:** The scan changed, not the brain

**Left – the chemistry (choline vs creatine):**
> - Choline and creatine sit only 0.17 ppm apart in the spectrum.
> - In November the magnet was poorly tuned:
>   - signal-to-noise **31 → 25**
>   - peak width **8.7 → 11.4 Hz**
> - The blurred peaks bleed into each other, so the ratio can't be trusted.

**Visual (key picture of the pitch):** May and November spectra side by side, with the Cho and Cr peaks labelled. May shows two sharp, separate peaks; November shows two wide, overlapping ones.

**Right – the anatomy (the region):**
> - The May and November images are identical apart from noise.
> - Only the outline drawn around the region changed.
> - So the 26% "growth" came from processing, not the patient.

**Bottom line (our offer):**
> ScanTwin checks the machine and the processing every time it checks the patient, so a bad scan is caught before anyone acts on it.

---

## Slide 4 – How it works

**Title:** Four steps, every scan

**Text (draw as a left-to-right flow):**
1. **Measure:** the scan comes in, personal data is removed, and the file is fingerprinted.
2. **Check:** it's compared against 61 known failure types, each with a limit and a source → **good, caution, or don't use**.
3. **Update:** two twins move.
   - The **scanner twin** records the machine's health.
   - The **patient twin** moves only as far as the scan quality allows. A failing scanner blocks the update.
4. **Advise:** says who changed (patient, machine or processing) and the cheapest fix.

**Bottom line:** *It never diagnoses. It decides how much a scan can be trusted.*

---

## Slide 4.1 – Live demo

**Title:** See it in action

**Big button / text:** **▶ Open the ScanTwin prototype**
- If the app has a public URL, link it here.
- If it runs on a laptop, use this slide only as the cue for a teammate to switch tabs. Don't rely on clicking live.

**Small text:**
> Watch for:
> 1. The choline rise → **🛠 It's the machine:** the shim failed on the scan date
> 2. The region growth → **the processing changed, not the patient**
> 3. The fix → **re-shim and repeat, about 4 minutes**

**Prepare beforehand:** demo tabs already loaded, plus backup screenshots in case the app freezes.

---

## Slide 5 – Tracking change over time

**Title:** The twin remembers

**Text:**
> - ScanTwin keeps a running record, not a one-off comparison.
> - **For the brain region:** the November scan failed its consistency check, so the twin held its estimate at about **22,190 mm³** instead of jumping to 27,810.
> - **For the scanner:** every week it measures a bottle of liquid that never changes. If the readings drift, the twin flags the machine weeks before patient scans suffer.
>
> **Result:** a real change builds up over trustworthy scans; a bad scan can't rewrite the record.

**Visual:** timeline of the region: May 22,140 → November measured 27,810 (grey, crossed out) → twin estimate 22,190.

---

## Slide 6 – Reliability and current status

**Title:** Tested, not assumed

**Text:**
> **What exists today:**
> - Working prototype for two data types: spectroscopy and anatomy
> - 61 failure modes catalogued, each limit with its source (ACR phantom standards, MRS expert consensus, challenge dossier, or labelled team proposal)
> - 660 simulated scans with known truth
> - 69 automated tests; reproduces the challenge's reference result exactly
> - A real public 7 T brain scan processed end to end, with six identifying fields removed
>
> **What we caught:** our own simulator showed one of our limits was **15× too loose**, so we fixed it.
>
> **Compared with current tools:** existing quality-control tools judge one image at a time. ScanTwin adds memory over time, spectroscopy, and a twin of the scanner itself.

---

## Slide 7 – Limitations and future work

**Title:** What's next

**Limitations:**
> - Hackathon data are synthetic; the scanner's history is simulated.
> - Many limits are our own proposals, not yet clinically validated.
> - Tested on brain spectroscopy and anatomy only.
> - No diagnosis, no clinical claims, not a certified device.

**Next steps:**
> 1. Validate on real repeat scans of the same people.
> 2. Connect to real daily scanner quality checks.
> 3. Add more scan types: diffusion, functional MRI, DICOM.
> 4. Have MRI physicists review every limit.

**Closing line:**
> **ScanTwin: trust the number, or know why not.**

**Footnote:** 7 T example data: W. T. Clarke, University of Oxford (Zenodo 5085449, CC BY 4.0).

---

# 3-minute talking points

| Slide | Time | Say |
|---|---|---|
| 1 | 0:00–0:15 | Every follow-up MRI raises one question: if a number changed, was it the patient or the machine? ScanTwin is a digital twin of the scanner that answers that before anyone acts on the result. |
| 2 | 0:15–0:45 | Here's a patient's follow-up after six months. Choline, a marker of how fast cells rebuild their membranes, rose 23%. A brain region grew 26%. A healthy region barely changes in a year, so together these numbers look like something is growing. That's the kind of result that frightens a patient and triggers more tests. So: did the brain change, or the scan? |
| 3 | 0:45–1:15 | Look at the signal. Choline and creatine sit almost on top of each other. In November the magnet was poorly tuned: the peaks got wider and noisier, so they bled together and the ratio became unreliable. And the region? The images are identical; only the outline drawn around it changed. Neither number reflects the patient. ScanTwin catches this before anyone acts on it. |
| 4 | 1:15–1:40 | It works in four steps. Every scan is checked against 61 known failure types and gets a verdict: good, caution or don't use. Then two twins update: one for the scanner, one for the patient. The patient's record only moves as far as the scan quality allows. Finally, it tells you who changed and the cheapest fix. |
| 4.1 | 1:40–2:15 | *[Switch to the app.]* Here's the November scan. On that date, the scanner twin shows the shim had failed, so the verdict is "it's the machine" and the patient's record isn't changed. Here's the region: the outline grew, the images didn't, so "processing, not the patient". And the fix: re-shim and repeat, about four minutes. |
| 5 | 2:15–2:30 | Because it's a twin, it remembers. A bad scan can't rewrite the record: the region estimate stayed at about 22,000 instead of jumping to 27,000. And the scanner's own weekly checks reveal a drifting machine before patients are affected. |
| 6 | 2:30–2:48 | We tested it: 660 simulated scans with known answers, 69 automated tests, and a real 7 Tesla scan. Our simulator even caught one of our own limits being 15 times too loose, so we fixed it. |
| 7 | 2:48–3:00 | Our data is mostly synthetic, so next we validate on real repeat scans and real scanner checks. ScanTwin never diagnoses. It tells you whether to trust the number, or exactly why not. |

**If running over time,** cut these first:
1. The scanner-bottle sentence (slide 5)
2. "Compared with current tools" (slide 6)

**If the demo shows the warning version for choline,** say instead: "The scan only counts at reduced weight. The twin moves from 0.69 to 0.74, not 0.85."
