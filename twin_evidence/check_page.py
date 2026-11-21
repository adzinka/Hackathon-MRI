"""ScanTwin "Check a scan" page (Streamlit).

Flow:   1 Choose a scan (upload files OR an example)  ->  2 Result  ->  3 Compare with a later scan (optional)
Data:   single-voxel MR spectroscopy (NIfTI-MRS) and structural T1-weighted MRI (NIfTI or a DICOM series, .dcm files
        or one .zip), detected automatically. Upload several files at once: the scan, an optional second (later) scan,
        masks and JSON sidecars. DICOM headers are checked first (the "envelope"), then the image goes the T1w way.
Result: TRUSTY / WARNING / ERROR (catalogue PASS / WARN / FAIL), one state per category, every problem with its ID.
Views:  Simple (plain words, no codes or units) and Expert (gates, images, numbers, trace). Both read the SAME result.
The page judges the measurement, never the person; it never says "diagnosis".
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import dicom_io as DI
from . import ingest as I
from . import mrs_fit as F
from . import t1w as T
from .gates import (MRS_GATES, T1W_GATES, classify, context_advice, context_factors, supported_range,
                    STANDING_LIMITATIONS)
from .twin import Twin
from .validate import evaluate

REPO = Path(__file__).resolve().parents[1]
ICON = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴", "N/A": "⚪"}
WORD = {"PASS": "Good", "WARN": "Caution", "FAIL": "Problem", "N/A": "Can't check"}
COLOR = {"PASS": "#1f8a4c", "WARN": "#b77800", "FAIL": "#c23b35", "N/A": "#8a96a3"}
BG = {"PASS": "rgba(31,138,76,.10)", "WARN": "rgba(245,177,61,.14)", "FAIL": "rgba(194,59,53,.10)", "N/A": "rgba(138,150,163,.12)"}

SP = "SP"   # placeholder for the StarterPack root in example specs
EXAMPLES = {
    "Spectroscopy · real brain scan · 7 T (Zenodo, CC BY 4.0)":
        {"kind": "mrs", "files": [REPO / "external_data/zenodo_5085449_example_02/derivatives/sub-ext01_ses-baseline_transients_svs.nii.gz"]},
    "Spectroscopy · hackathon May + November · 3 T (synthetic)":
        {"kind": "mrs", "files": [f"{SP}/04_MRS_BIDS/sub-01/ses-{s}/mrs/sub-01_ses-{s}_acq-press_nuc-1H_voi-frontal_svs.nii.gz"
                                  for s in ("baseline", "followup")]},
    "Brain anatomy (T1w) · hackathon May + November, with outlines (synthetic)":
        {"kind": "t1w", "scans": [{"image": f"{SP}/03_BIDS_MRI/sub-01/ses-{s}/anat/sub-01_ses-{s}_T1w.nii.gz",
                                   "mask": f"{SP}/02_NIfTI/synthetic_brain_mask.nii.gz",
                                   "roi": f"{SP}/03_BIDS_MRI/derivatives/roi/sub-01/ses-{s}/anat/sub-01_ses-{s}_space-T1w_label-ROI_probseg.nii.gz",
                                   "sidecar": f"{SP}/03_BIDS_MRI/sub-01/ses-{s}/anat/sub-01_ses-{s}_T1w.json"}
                                  for s in ("baseline", "followup")]},
    "Brain anatomy (T1w) · DICOM series, hackathon May (synthetic)":
        {"kind": "t1w", "dicom": f"{SP}/01_DICOM/baseline", "scans": [{"mask": f"{SP}/02_NIfTI/synthetic_brain_mask.nii.gz"}]},
    "Brain anatomy (T1w) · DICOM series with one slice missing (built from the hackathon data)":
        {"kind": "t1w", "dicom": f"{SP}/01_DICOM/baseline", "drop": ["IM0008.dcm"], "scans": [{}]},
    "Brain anatomy (T1w) · hackathon May only (synthetic)":
        {"kind": "t1w", "scans": [{"image": f"{SP}/03_BIDS_MRI/sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz",
                                   "mask": f"{SP}/02_NIfTI/synthetic_brain_mask.nii.gz",
                                   "roi": f"{SP}/03_BIDS_MRI/derivatives/roi/sub-01/ses-baseline/anat/sub-01_ses-baseline_space-T1w_label-ROI_probseg.nii.gz",
                                   "sidecar": f"{SP}/03_BIDS_MRI/sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.json"}]},
}
# Real test data (external_data/real_samples; Morelli et al., Zenodo 10.5281/zenodo.13753316, CC BY 4.0), original
# files, listed only when present (python scripts/fetch_real_samples.py)
REAL = REPO / "external_data/real_samples"
for _label, _spec, _file in [
        ("Real test data · Spectroscopy · 3 T sLASER, 750 repeats, sub-01 (Morelli et al., Zenodo, CC BY 4.0)",
         {"kind": "mrs", "files": [REAL / "MRS/sub-01_svs.nii.gz"]}, REAL / "MRS/sub-01_svs.nii.gz"),
        ("Real test data · Brain anatomy (T1w) · 3 T, defaced, sub-01 (Morelli et al., Zenodo, CC BY 4.0)",
         {"kind": "t1w", "scans": [{"image": REAL / "T1w/sub-01_T1w.nii.gz"}]}, REAL / "T1w/sub-01_T1w.nii.gz")]:
    if _file.exists():
        EXAMPLES[_label] = _spec

# plain language for every gate (Simple view): name, reason, action
PLAIN = {
    "F13": ("Signal strength", "the signal is weak compared with the noise", "Repeat with more averages or check the receive coil"),
    "F12": ("Sharpness (magnet tuning)", "the magnet was not tuned well, so the signal is blurred", "Re-tune the magnet (re-shim) and repeat, about 4 minutes"),
    "F08": ("Stability during the scan", "the scanner's frequency drifted during the scan", "Let the scanner settle, switch on frequency correction, repeat"),
    "F23": ("Patient stillness", "the patient moved during the scan", "Help the patient lie still (padding, shorter scan) and repeat"),
    "F15": ("Leftover water signal", "leftover water signal is too high", "Recalibrate the water suppression and repeat"),
    "F14": ("Measurement precision", "the measurement is not precise enough", "Repeat at better quality"),
    "F20": ("Same spot as last time", "the measured spot is not where it was last time", "Re-position using the previous scan as a guide"),
    "F21": ("Same tissue mix", "the tissue inside the measured spot is different", "Re-position and correct for tissue composition"),
    "F19": ("Same sequence as last time", "the sequence or pulse settings changed", "Use the baseline sequence"),
    "F61": ("Transmitter calibration", "the transmitter is miscalibrated", "Recalibrate the transmitter and repeat"),
    "F26": ("Patient stillness", "the patient moved during the scan (ghost images)", "Help the patient lie still (padding, shorter scan) and repeat"),
    "F27": ("Image clarity vs last scan", "the image is noisier than last time", "Check the receive coil and repeat"),
    "F28": ("Even brightness", "the brightness is uneven across the head", "Check coil and transmitter calibration; apply brightness correction"),
    "F32": ("Outline follows the head", "the region outline did not move with the head, so it lands in the wrong place", "Re-draw or re-register the outline on the later scan"),
    "F35": ("Same settings as last time", "the scan settings differ from last time", "Repeat with the same settings as last time"),
}
MRS_BARS = ["F13", "F12", "F08", "F23", "F15", "F14"]
T1W_BARS = ["F26", "F28", "F27", "F32", "F35"]

MRS_SCENARIOS = {   # label, fault, strength (None = default from the scan)
    "repeat": ("Same patient · scanner OK", "none", 0.0),
    "machine_noise": ("Scanner problem · weak receive coil", "noise", None),
    "machine_shim": ("Scanner problem · magnet badly tuned", "shim", None),
    "machine_drift": ("Scanner problem · frequency drift", "drift", None),
    "machine_water": ("Scanner problem · water suppression failed", "water", 5.0),
    "patient": ("Patient changed · more choline", "cho_up", None),
}
T1W_SCENARIOS = {
    "repeat": ("Same patient · scanner OK", "none", 0.0),
    "machine_noise": ("Scanner problem · weak receive coil", "noise", 1.4),
    "machine_bias": ("Scanner problem · uneven brightness", "bias", 0.8),
    "machine_scale": ("Scanner problem · gradient calibration off by 3 %", "scale", 0.03),
    "session_motion": ("This session · patient moved", "motion", 0.25),
    "session_shift": ("This session · head positioned 3 mm differently", "shift", 3.0),
    "patient": ("Patient changed · region grew", "grow", 3.0),
}


# ------------------------------------------------------------------------------------------------ shared helpers
def _quality_score(gid: str, value) -> float | None:
    """0-100 bar: PASS zone 60-100, WARN zone 30-60, FAIL zone 0-30."""
    g = next((x for x in MRS_GATES + T1W_GATES if x.id == gid), None)
    if g is None or value is None or g.warn is None:
        return None
    w = g.warn[1]
    f = g.fail[1] if g.fail else (w * 2 + 1 if g.warn[0] == ">" else w / 2)
    v = float(value)
    if g.warn[0] == ">":
        if v <= w:
            return 60 + 40 * max(0.0, 1 - v / w) if w > 1 else 100.0 if v < 0.5 else 60.0
        if v <= f:
            return 30 + 30 * (f - v) / (f - w)
        return max(0.0, 30 * (1 - (v - f) / max(f, 1e-9)))
    if v >= w:
        return min(100.0, 60 + 40 * (v - w) / w)
    if v >= f:
        return 30 + 30 * (v - f) / (w - f)
    return max(0.0, 30 * v / f)


def _card(st, status: str, title: str, body: str):
    st.markdown(f"""<div style="border-left:6px solid {COLOR[status]};background:{BG[status]};border-radius:10px;
        padding:16px 20px;margin:4px 0 14px 0">
        <div style="font-size:1.6rem;font-weight:700;color:{COLOR[status]};line-height:1.25">{title}</div>
        <div style="font-size:1.05rem;margin-top:6px">{body}</div></div>""", unsafe_allow_html=True)


def _bars(st, g: dict, order: list):
    status = {x["id"]: x for x in g["gates"]}
    for gid in order:
        x = status.get(gid)
        if x is None or not x.get("applies", True):
            continue
        name = PLAIN[gid][0]
        if x["status"] == "N/A":
            r_ = x.get("reason") or ""
            why = ("needs a brain mask" if "brain mask" in r_ else "needs an earlier scan" if "baseline" in r_
                   else "background removed: motion can't be seen" if "background" in r_ else "can't check this for this file")
            st.markdown(f"<div style='display:flex;gap:12px;align-items:center;margin:7px 0'><div style='width:210px'>{name}</div>"
                        f"<div style='flex:1;color:#8a96a3'>⚪ {why}</div></div>", unsafe_allow_html=True)
            continue
        sc = _quality_score(gid, x["value"])
        sc = 100.0 if sc is None else sc
        cause = " ← this is why" if x["status"] != "PASS" else ""
        st.markdown(f"""<div style="display:flex;gap:12px;align-items:center;margin:7px 0">
            <div style="width:210px">{name}</div>
            <div style="flex:1;height:12px;border-radius:6px;background:rgba(138,150,163,.25);overflow:hidden">
              <div style="width:{sc:.0f}%;height:100%;background:{COLOR[x['status']]}"></div></div>
            <div style="width:160px;color:{COLOR[x['status']]};font-weight:600">{WORD[x['status']]}{cause}</div></div>""",
                    unsafe_allow_html=True)


def _range_fig(items: list, xtitle: str, fmt: str = "{:.2f}") -> go.Figure:
    """items: (label, value, lo, hi, status). If two bands overlap a lot, the scans cannot tell a difference."""
    fig = go.Figure()
    for i, (lab, v, lo, hi, st_) in enumerate(items):
        y = len(items) - i
        fig.add_trace(go.Scatter(x=[lo, hi], y=[y, y], mode="lines", line=dict(color=COLOR[st_], width=12),
                                 hoverinfo="skip", showlegend=False, opacity=0.45))
        fig.add_trace(go.Scatter(x=[v], y=[y], mode="markers+text", marker=dict(size=16, color="white",
                                 line=dict(color=COLOR[st_], width=3)), text=[fmt.format(v)], textposition="top center",
                                 showlegend=False, hovertemplate=f"{lab}: %{{x}}<extra></extra>"))
    fig.update_layout(height=80 + 55 * len(items), margin=dict(l=10, r=10, t=10, b=30),
                      yaxis=dict(tickvals=list(range(len(items), 0, -1)), ticktext=[x[0] for x in items],
                                 range=[0.4, len(items) + 0.7]),
                      xaxis=dict(title=xtitle), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _headline(g: dict) -> tuple[str, str]:
    dec = g["deciding"]
    why = " and ".join(PLAIN.get(x["id"], (None, x["label"].lower()))[1] for x in dec)
    if g["status"] == "PASS":
        return "🟢 Good to use", "This scan is reliable. Every quality check that can be run passed."
    if g["status"] == "WARN":
        return "🟡 Use with caution", f"Usable, but less precise than normal: {why}."
    return "🔴 Don't use", f"This scan can't be trusted: {why}. The result must not be used."


def _actions(st, g: dict):
    st.markdown("**What to do**")
    if g["deciding"]:
        for x in g["deciding"]:
            st.markdown(f"✅ {PLAIN.get(x['id'], (None, None, x['fix']))[2]}")
    else:
        st.markdown("✅ Nothing to fix.")


def _gate_table(g: dict, raw: dict | None = None, field_note: str = "") -> list[dict]:
    rows = []
    for x in g["gates"]:
        if not x.get("applies", True):
            continue
        v = x["value"]
        if v is None:
            val = "N/A"
        elif raw and x["id"] in raw and raw[x["id"]] is not None and field_note:
            val = f"{raw[x['id']]:.1f} Hz {field_note} (= {v:.1f} Hz at 3 T)"
        else:
            val = f"{v:.2f} {x['unit']}".strip() if isinstance(v, float) else str(v)
        rows.append({"": ICON.get(x["status"], "⚪"), "ID": x["id"], "check": PLAIN.get(x["id"], (x["label"],))[0],
                     "measured": val, "PASS if": x["pass_if"], "source": x.get("source", ""), "note": x.get("reason") or ""})
    return rows


def _resolve(path, root: Path):
    if path is None:
        return None
    s = str(path)
    return Path(s.replace(SP, str(root))) if s.startswith(SP) else Path(s)


def _kind_of(data: bytes, name: str) -> str:
    """'mrs' for complex / single-voxel NIfTI-MRS, 't1w' for a 3D real image."""
    import nibabel as nib
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / Path(name).name
        p.write_bytes(data)
        img = nib.load(str(p))
        shp, dt = img.shape, img.get_data_dtype()
    if np.issubdtype(dt, np.complexfloating) or (len(shp) >= 4 and tuple(shp[:3]) == (1, 1, 1)):
        return "mrs"
    return "t1w"


def _is_dicom(name: str, data: bytes) -> bool:
    return name.lower().endswith(".dcm") or (len(data) > 132 and data[128:132] == b"DICM")


def _dicom_files(files) -> list:
    """(name, bytes) of every DICOM file in the uploads, unpacking .zip archives in memory."""
    import zipfile
    import io as _io
    out = []
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(_io.BytesIO(data)) as z:
                out += [(Path(n).name, z.read(n)) for n in z.namelist() if not n.endswith("/")]
        elif _is_dicom(f.name, data):
            out.append((f.name, data))
    return out


def _sort_uploads(files) -> dict:
    """Split uploads into main scans, masks, region outlines, JSON (sidecars or a packet) and DICOM files."""
    out = {"scans": [], "masks": [], "rois": [], "jsons": [], "dicom": []}
    for f in sorted(files, key=lambda f: f.name):
        n = f.name.lower()
        if n.endswith(".zip") or _is_dicom(n, f.getvalue()):
            out["dicom"].append(f)
        elif n.endswith(".json"):
            out["jsons"].append(f)
        elif "mask" in n:
            out["masks"].append(f)
        elif any(k in n for k in ("roi", "probseg", "label")):
            out["rois"].append(f)
        else:
            out["scans"].append(f)
    return out


def _stem(name: str) -> str:
    for ext in (".nii.gz", ".nii", ".json"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


# ------------------------------------------------------------------------------------------------ page
def render(st, *, starterpack_root: Path, thresholds: dict, schema: dict, packets: dict, manifest: dict):
    top_l, top_r = st.columns([4, 1])
    top_l.title("ScanTwin · Check a scan")
    top_l.markdown("Find out **if a brain scan can be trusted**, **why**, and **what to do next**: spectroscopy "
                   "(brain chemistry) or T1-weighted MRI (brain anatomy). ScanTwin judges the *measurement*, never the person.")
    view = top_r.radio("View", ["Simple", "Expert"], index=0, horizontal=True, key="chk_view",
                       help="Simple: plain words. Expert: every check, number and source.")
    expert = view == "Expert"

    # ---------------------------------------------------------------- STEP 1 · choose
    st.markdown("### 1 · Choose a scan")
    src = st.radio("Source", ["Upload my files", "Use an example"], index=0, horizontal=True, key="chk_src",
                   label_visibility="collapsed")
    cache = st.session_state.setdefault("chk_cache", {})
    if src == "Upload my files":
        ups = st.file_uploader("NIfTI files (.nii.gz / .nii): the scan, optionally a later scan of the same person, a brain "
                               "mask, a region outline (name contains 'roi' or 'probseg'), and JSON sidecars. "
                               "Or a T1w DICOM series (all its .dcm files, or one .zip). "
                               "Or one ScanTwin evidence packet (.json).", type=["gz", "nii", "json", "dcm", "zip"],
                               accept_multiple_files=True, key="chk_upload")
        if not ups:
            st.info("⬆️ **Drop one or more files above to start.** No files at hand? Choose **Use an example**.")
            return
        groups = _sort_uploads(ups)
        if groups["dicom"]:
            key = tuple((f.name, f.size) for f in ups)
            if key not in cache:
                cache[key] = {"kind": "t1w", "label": f"DICOM series ({len(groups['dicom'])} file(s))", "uploaded": True,
                              "sources": groups, "dicom_files": _dicom_files(groups["dicom"])}
            _page_t1w(st, cache[key], key, cache, starterpack_root, thresholds, schema, expert)
            return
        if not groups["scans"] and groups["jsons"]:
            f = groups["jsons"][0]
            _render_packet(st, json.loads(f.getvalue()), f.name, starterpack_root, thresholds, schema, packets, manifest, expert)
            return
        if not groups["scans"]:
            st.warning("No scan found among the files: masks and outlines need the image they belong to.")
            return
        key = tuple((f.name, f.size) for f in ups)
        if key not in cache:
            try:
                kind = _kind_of(groups["scans"][0].getvalue(), groups["scans"][0].name)
            except Exception as err:
                st.error("This doesn't look like a NIfTI brain scan. Ask your physicist to export it as NIfTI "
                         "(dcm2niix for images, spec2nii for spectroscopy)." + (f"\n\nTechnical detail: `{err}`" if expert else ""))
                return
            cache[key] = {"kind": kind, "label": ", ".join(f.name for f in groups["scans"]), "uploaded": True}
            cache[key]["sources"] = groups
        spec = cache[key]
    else:
        ex = st.selectbox("Example", list(EXAMPLES), index=0, key="chk_example")
        key = ("example", ex)
        spec = {**EXAMPLES[ex], "label": ex, "uploaded": False}

    if spec["kind"] == "mrs":
        _page_mrs(st, spec, key, cache, starterpack_root, thresholds, schema, expert)
    else:
        _page_t1w(st, spec, key, cache, starterpack_root, thresholds, schema, expert)


# ------------------------------------------------------------------------------------------------ spectroscopy
def _spectrum_fig(fids: list, nu0: float, dwell: float) -> go.Figure:
    fig = go.Figure()
    fig.add_vrect(x0=1.85, x1=3.35, fillcolor="#2b6cb0", opacity=0.06, line_width=0, annotation_text="fit window",
                  annotation_position="top left")
    for lab, fid, col in fids:
        ppm, sp = F.spectrum(fid, dwell, nu0=nu0)
        sel = (ppm > 0.5) & (ppm < 5.0)
        y = sp.real[sel]
        fig.add_trace(go.Scatter(x=ppm[sel], y=y / np.abs(y).max(), mode="lines", name=lab, line=dict(color=col, width=1.6)))
    for lab, p in (("NAA", 2.01), ("Cr", 3.03), ("Cho", 3.20), ("water", 4.65)):
        fig.add_vline(x=p, line=dict(color="#c9d1d8", dash="dot"), annotation_text=lab, annotation_position="bottom")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(title="chemical shift / ppm", autorange="reversed"),
                      yaxis=dict(title="real spectrum (normalised)"), legend=dict(orientation="h", y=1.12))
    return fig


def _packetise_mrs(ing, m, g, tmp: Path, *, tag, ses, months, origin, sysqc, desc=""):
    rel = f"{tag}.nii.gz"
    sha = I.write_derivative(ing, tmp / rel)
    pid = ("E-t0-" if months == 0 else "E-t1-") + f"check_{tag}"
    p = I.build_packet(ing, m, g, packet_id=pid, subject_id="sub-check", session_id=ses, months=months, rel_path=rel,
                       sha=sha, origin=origin, system_qc_id=sysqc, limit_note=desc if origin == "SIMULATED" else "")
    return p, {rel: sha}


def _compare_mrs(ing, m, g, fi, fm, thresholds, schema, sysqc, label, later_origin):
    fg = I.gate(fm, fi.nu0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p0, man0 = _packetise_mrs(ing, m, g, tmp, tag="baseline", ses="ses-baseline", months=0.0, origin="MEASURED", sysqc=sysqc)
        p1, man1 = _packetise_mrs(fi, fm, fg, tmp, tag="followup", ses="ses-followup", months=6.0, origin=later_origin,
                                  sysqc=sysqc, desc=f"SIMULATED follow-up: {label}")
        man = {**man0, **man1}
        tw = Twin()
        tw.ingest(p0, evaluate(p0, thresholds, schema, man, tmp))
        ev1 = evaluate(p1, thresholds, schema, man, tmp)
        e1 = tw.ingest(p1, ev1)
    new_bad = [x["id"] for x in fg["deciding"] if x["id"] not in [y["id"] for y in g["deciding"]]]
    names = lambda ids: ", ".join(PLAIN.get(i, (i,))[0].lower() for i in ids)
    z = e1.z_change or 0.0
    if ev1.status == "FAIL":
        who = ("FAIL", "🛠 It's the machine",
               f"The later scan failed a quality check ({names(new_bad or [x['id'] for x in fg['deciding']])}). "
               "The result is thrown away and the patient's record does not change. Fix the scanner and repeat.")
    elif new_bad:
        who = ("WARN", "🛠 Probably the machine",
               f"The number moved because the scan got worse ({names(new_bad)}), not because the patient changed. "
               "ScanTwin counts this scan at reduced weight.")
    elif abs(z) < 1.96:
        who = ("N/A", "❔ Can't tell: no real change",
               "Both scans have the same quality and the difference is smaller than these scans can measure. "
               "Nothing is claimed about the patient.")
    elif abs(z) < 3 and "WARN" in (ev1.status, g["status"]):
        who = ("WARN", "❔ Inconclusive",
               "The difference is beyond the noise, but one of the scans has a quality warning. "
               "Repeat at good quality before claiming a change.")
    else:
        who = ("PASS", "🧑 The patient changed",
               "Both scans have good, comparable quality and the difference is beyond measurement error. "
               "Confirm with a repeat scan before anyone acts on it.")
    return {"fg": fg, "ev1": ev1, "e1": e1, "z": z, "who": who}


def _page_mrs(st, spec, key, cache, root, thresholds, schema, expert):
    ck = ("mrs", key)
    if ck not in cache:
        with st.status("Analysing the spectrum …", expanded=True) as status:
            t0 = time.time()
            try:
                st.write("Reading the file(s) and removing personal details …")
                if spec.get("uploaded"):
                    scans = [I.read_bytes(f.getvalue(), f.name) for f in spec["sources"]["scans"][:2]]
                else:
                    scans = [I.read(_resolve(p, root)) for p in spec["files"][:2]]
                st.write("Combining coils, checking every repeat, measuring quality …")
                ms = [I.measure(s) for s in scans]
            except Exception as err:
                status.update(label="Could not read this file", state="error")
                st.error("This doesn't look like a single-voxel spectroscopy file (NIfTI-MRS). Ask your physicist to "
                         "export it with spec2nii." + (f"\n\nTechnical detail: `{err}`" if expert else ""))
                return
            cache[ck] = (scans, ms, time.time() - t0)
            status.update(label=f"Done in {time.time() - t0:.1f} s", state="complete", expanded=False)
    scans, ms, secs = cache[ck]
    ing, m = scans[0], ms[0]
    g = I.gate(m, ing.nu0)
    field_T = ing.nu0 / I.GAMMA_MHZ_PER_T
    who = "your file" if spec.get("uploaded") else "the example"
    st.success(f"✅ Now showing {who} **{spec['label']}** · spectroscopy · {ing.meta.get('Manufacturer', 'scanner')} "
               f"{field_T:.1f} T · {ing.transients.shape[0]} repeat(s)"
               + (f" · {len(ing.removed)} personal detail(s) removed" if ing.removed else "")
               + (" · a later scan is included" if len(scans) > 1 else ""))

    st.markdown("### 2 · Result" + (" · first scan" if len(scans) > 1 else ""))
    _card(st, g["status"], *_headline(g))
    u = math.hypot(m["cho_cr_crlb"], 0.045)
    lo, hi = supported_range(m["cho_cr"], u, g["status"])
    if not expert:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Scan quality**")
            _bars(st, g, MRS_BARS)
        with c2:
            _actions(st, g)
            st.markdown("**What the result says**")
            if g["status"] == "FAIL":
                st.markdown("Nothing: the value from this scan must not be used.")
            else:
                st.plotly_chart(_range_fig([("this scan", m["cho_cr"], lo, hi, g["status"])], "choline level (Cho/Cr)"),
                                width="stretch", key="rng_main")
                st.caption(f"The true choline level is somewhere between **{lo:.2f} and {hi:.2f}**. "
                           "Differences smaller than this band mean nothing.")
    else:
        t_q, t_s, t_c, t_t = st.tabs(["Quality checks", "Spectrum", "Scan context", "Trace"])
        with t_q:
            note = f"at {field_T:.0f} T" if abs(ing.nu0 - I.REF_NU0) > 1 else ""
            st.dataframe(_gate_table(g, {"F12": m.get("fwhm_hz"), "F08": m.get("drift_abs_hz")}, note), width="stretch", hide_index=True)
            k = st.columns(4)
            k[0].metric("Cho/Cr", f"{m['cho_cr']:.3f} ± {u:.3f}", f"supports {lo:.2f}–{hi:.2f}", delta_color="off")
            k[1].metric("NAA/Cr", f"{m['naa_cr']:.2f}")
            k[2].metric("SNR", f"{m['snr']:.0f}")
            k[3].metric("Linewidth", f"{m['fwhm_ppm']:.3f} ppm", f"{m['fwhm_hz']:.1f} Hz", delta_color="off")
            for x in g["deciding"]:
                st.markdown(f"- **{x['id']}** · {x['fix']}")
        with t_s:
            st.plotly_chart(_spectrum_fig([("this scan", ing.fid, "#2b6cb0")], ing.nu0, ing.dwell), width="stretch", key="spec_main")
            st.caption(f"Windowed fit 1.85–3.35 ppm (NAA, Cr, Cho + linear baseline), fit linewidth {m['fit_lw_hz']:.1f} Hz. "
                       "Reads Cho/Cr about 3 % low on simulated truth.")
        with t_c:
            ctx = st.multiselect("Scan context (changes only warnings and advice, never the limits)",
                                 [r["Context factor"] for r in context_factors()], key="chk_ctx")
            for x in context_advice(ctx, g) if ctx else []:
                st.markdown(f"- {ICON.get(x['gate_status'], '⚪')} **{x['id']}** {x['failure']}: {x['reading']}. {x['advice']}")
        with t_t:
            st.markdown(f"**Source file** SHA-256 `{ing.source_sha256}` · processed in {secs:.1f} s")
            if ing.removed:
                st.warning(f"Removed {len(ing.removed)} identifying field(s): {', '.join(ing.removed)} (catalogue F59). "
                           "Values are never shown or stored.")
            for s_ in ing.steps:
                if not s_.startswith("de-identify"):
                    st.markdown(f"- {s_}")
            st.caption("Standing limitations: " + " · ".join(f"{x['id']}: {x['label']}" for x in STANDING_LIMITATIONS))
            with tempfile.TemporaryDirectory() as tmp:
                p, _ = _packetise_mrs(ing, m, g, Path(tmp), tag="scan", ses="ses-baseline", months=0.0,
                                      origin="SIMULATED" if "synthetic" in spec["label"] else "MEASURED", sysqc=None)
            st.download_button("Download the evidence packet (JSON)", json.dumps(p, indent=2), file_name=f"{p['packet_id']}.json",
                               mime="application/json", key="chk_dl")

    # ---------------------------------------------------------------- STEP 3 · compare
    st.markdown("### 3 · Compare with a later scan *(optional)*")
    opts = (["real"] if len(scans) > 1 else []) + ["—"] + list(MRS_SCENARIOS)
    labels = {"real": "The later scan I provided (real data)", "—": "Don't compare",
              **{k: f"Simulated: {v[0]}" for k, v in MRS_SCENARIOS.items()}}
    st.caption("Use your own later scan, or imagine the same person is scanned again on this scanner: the later scan is then "
               "simulated from this one, so we know what really changed.")
    sc_key = st.radio("What is the later scan?", opts, index=0, format_func=labels.get, key="chk_scn_mrs")
    if sc_key == "—":
        st.caption("A verdict about the measurement only. No diagnosis, no treatment advice.")
        return
    if sc_key == "real":
        fi, fm, label, origin = scans[1], ms[1], "the later scan you provided", "MEASURED"
    else:
        label, fault, strength = MRS_SCENARIOS[sc_key]
        if strength is None:
            k_ = ing.nu0 / I.REF_NU0
            dflt = {"shim": round(9 * k_, 1), "drift": round(10 * k_, 1), "noise": round(max(1.5, m["snr"] / 18), 1),
                    "cho_up": 70.0}[fault]
            if expert or fault == "cho_up":
                unit = {"shim": "Hz extra blur", "drift": "Hz drift", "noise": "× noise", "cho_up": "% more choline"}[fault]
                strength = st.slider(f"How strong? ({unit})", 0.0, 100.0 if fault == "cho_up" else float(3 * dflt), float(dflt),
                                     key=f"chk_str_{sc_key}")
            else:
                strength = dflt
            if fault == "cho_up":
                strength = strength / 100.0
        fk = ("follow", ck, sc_key, strength)
        if fk not in cache:
            with st.spinner("Simulating the later scan …"):
                f_ = I.degrade(ing, fault, strength, seed=11)
                cache[fk] = (f_, I.measure(f_))
        fi, fm = cache[fk]
        origin = "SIMULATED"
    sysqc = "ASSUMED:external-dataset"
    if expert and st.toggle("Strict: reject scans without a site QC record", value=False, key="chk_strict"):
        sysqc = None
    r = _compare_mrs(ing, m, g, fi, fm, thresholds, schema, sysqc, label, origin)
    _card(st, *r["who"])
    fg = r["fg"]
    items = [("first scan", m["cho_cr"], lo, hi, g["status"])]
    if fg["status"] != "FAIL":
        lo1, hi1 = supported_range(fm["cho_cr"], math.hypot(fm["cho_cr_crlb"], 0.045), fg["status"])
        items.append(("later scan", fm["cho_cr"], lo1, hi1, fg["status"]))
    st.plotly_chart(_range_fig(items, "choline level (Cho/Cr)"), width="stretch", key="rng_cmp")
    st.caption("Each band is what that scan can support. If the bands overlap a lot, the scans cannot tell a difference.")
    if expert:
        e1 = r["e1"]
        k = st.columns(4)
        k[0].metric("Later scan gates", f"{ICON[fg['status']]} {fg['status']}", " ".join(x["id"] for x in fg["deciding"]) or "all pass",
                    delta_color="off")
        k[1].metric("Change z (F60)", f"{r['z']:+.2f}")
        k[2].metric("Gain K", "—" if e1.K is None else f"{e1.K:.2f}", e1.action, delta_color="off")
        k[3].metric("Record after", "unchanged" if e1.posterior is None else f"{e1.posterior['x']:.3f} ± {e1.posterior['sd']:.3f}")
        st.plotly_chart(_spectrum_fig([("first scan", ing.fid, "#2b6cb0"), ("later scan", fi.fid, "#c8642a")], ing.nu0, ing.dwell),
                        width="stretch", key="spec_cmp")
    st.caption("A verdict about the measurement only. No diagnosis, no treatment advice.")


# ------------------------------------------------------------------------------------------------ T1-weighted
def _slice_fig(items: list) -> go.Figure:
    """Axial slice through the region (or the brain centre) for each (label, scan); region outline in red."""
    fig = make_subplots(rows=1, cols=len(items), subplot_titles=[x[0] for x in items], horizontal_spacing=0.03)
    for c, (lab, sc) in enumerate(items, start=1):
        ref = sc.roi if sc.roi is not None else sc.mask.astype(float)
        zc = int(round(np.average(np.arange(ref.shape[2]), weights=ref.sum(axis=(0, 1)) + 1e-9)))
        img = sc.data[:, :, zc].T
        fig.add_trace(go.Heatmap(z=img, colorscale="gray", showscale=False, hoverinfo="skip"), row=1, col=c)
        if sc.roi is not None:
            fig.add_trace(go.Contour(z=sc.roi[:, :, zc].T, contours=dict(start=0.5, end=0.5, coloring="none"),
                                     line=dict(color="#e23d2f", width=2), showscale=False, hoverinfo="skip"), row=1, col=c)
        fig.update_xaxes(showticklabels=False, row=1, col=c)
        fig.update_yaxes(showticklabels=False, scaleanchor=f"x{'' if c == 1 else c}", row=1, col=c)
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def _load_t1w(spec, root):
    """Returns (scans, dicom header check or None). A DICOM series whose geometry is broken gives no scan."""
    out = []
    if spec.get("dicom_files") is not None or spec.get("dicom"):
        if spec.get("dicom_files") is not None:
            files = spec["dicom_files"]
            masks = spec["sources"]["masks"]
            mask = masks[0].getvalue() if masks else None
        else:
            drop = set(spec.get("drop", []))
            files = [p for p in sorted(_resolve(spec["dicom"], root).iterdir()) if p.is_file() and p.name not in drop]
            mask = _resolve(spec["scans"][0].get("mask"), root)
        scan, check = DI.load(files, mask=mask)
        return ([scan] if scan is not None else []), check
    if spec.get("uploaded"):
        gr = spec["sources"]
        side = {_stem(f.name): json.loads(f.getvalue()) for f in gr["jsons"]}
        for i, f in enumerate(gr["scans"][:2]):
            sc = side.get(_stem(f.name)) or (next(iter(side.values())) if len(side) == 1 else None)
            mask = gr["masks"][min(i, len(gr["masks"]) - 1)].getvalue() if gr["masks"] else None
            roi = gr["rois"][i].getvalue() if len(gr["rois"]) > i else None
            out.append(T.load(f.getvalue(), name=f.name, mask=mask, roi=roi, sidecar=sc))
    else:
        for s in spec["scans"]:
            sc = json.loads(_resolve(s["sidecar"], root).read_text()) if s.get("sidecar") else None
            out.append(T.load(_resolve(s["image"], root), name=Path(s["image"]).name, mask=_resolve(s.get("mask"), root),
                              roi=_resolve(s.get("roi"), root), sidecar=sc))
    return out, None


STATE_STATUS = {"TRUSTY": "PASS", "WARNING": "WARN", "ERROR": "FAIL", "N/A": "N/A"}
ORIGIN = {"F26": "patient (moved)", "F32": "processing (outline)", "F35": "protocol", "F27": "not proven (image quality)",
          "F28": "not proven (image quality)", "F57": "data", "F58": "data", "F59": "data", "F54": "data / export",
          "F56": "data"}


def _state_card(st, c: dict):
    """Level 1: TRUSTY / WARNING / ERROR with the reasons."""
    s = c["state"]
    title = {"TRUSTY": "🟢 TRUSTY · the scan can be used for measurements",
             "WARNING": "🟡 WARNING · usable only with care; check the points below",
             "ERROR": "🔴 ERROR · do not use this scan for measurements"}[s]
    body = "Every check that applies passed." if not c["reasons"] else "<br>".join(f"• {r}" for r in c["reasons"])
    _card(st, STATE_STATUS[s], title, body)


def _category_table(c: dict) -> list[dict]:
    """Level 2: one state per category."""
    return [{"": ICON[STATE_STATUS[x["status"]]], "category": x["category"], "state": x["status"],
             "checks": ", ".join(i["id"] for i in x["items"])} for x in c["categories"]]


def _problem_table(c: dict) -> list[dict]:
    """Level 3: every check that is not PASS, with its catalogue ID, value, reference, likely origin and fix."""
    rows = []
    for cat in c["categories"]:
        for x in cat["items"]:
            if x["status"] == "PASS":
                continue
            cid = x.get("catalogue_id", x["id"] if str(x["id"]).startswith("F") else None)
            rows.append({"": ICON.get(x["status"], "⚪"), "problem": x["label"], "catalogue": cid or "— (proposed v1.2)",
                         "measured": "N/A" if x.get("value") is None else (f"{x['value']:.2f}" if isinstance(x["value"], float)
                                                                            else str(x["value"])),
                         "reference": x.get("pass_if") or x.get("source", ""), "severity": x["status"],
                         "evidence / reason": x.get("reason") or "",
                         "likely origin": "unknown (input missing)" if x["status"] == "N/A" else
                         ORIGIN.get(cid, "data / export" if x.get("kind") in ("file", "geometry", "metadata") else "not proven"),
                         "fix": x.get("fix", "")})
    return rows


def _page_t1w(st, spec, key, cache, root, thresholds, schema, expert):
    ck = ("t1w", key)
    if ck not in cache:
        with st.status("Analysing the anatomy scan …", expanded=True) as status:
            t0 = time.time()
            try:
                st.write("Reading the image(s), checking the DICOM headers if any, and removing personal details …")
                scans, dcheck = _load_t1w(spec, root)
                ms = []
                if scans:
                    st.write("Separating grey matter, white matter and fluid; measuring quality and volumes …")
                    ms = [T.measure(scans[0])]
                    if len(scans) > 1:
                        ms.append(T.measure(scans[1], scans[0], ms[0]))
            except Exception as err:
                status.update(label="Could not read this file", state="error")
                st.error("This doesn't look like a 3D T1-weighted brain image (NIfTI, or a complete DICOM series). "
                         + (f"\n\nTechnical detail: `{err}`" if expert else ""))
                return
            cache[ck] = (scans, ms, time.time() - t0, dcheck)
            status.update(label=f"Done in {time.time() - t0:.1f} s", state="complete", expanded=False)
    scans, ms, secs, dcheck = cache[ck]
    if not scans:             # the DICOM envelope is broken: nothing is measured
        c = classify(None, dcheck)
        st.markdown("### 2 · Result")
        _state_card(st, c)
        st.markdown("**By category**")
        st.dataframe(_category_table(c), width="stretch", hide_index=True)
        st.markdown("**What was found**")
        st.dataframe(_problem_table(c), width="stretch", hide_index=True)
        st.caption("The image was not reconstructed or measured: a broken series would give a wrong volume.")
        return
    sc0, m0 = scans[0], ms[0]
    g0 = T.gate(m0)
    cls0 = classify(g0, dcheck)
    who = "your file" if spec.get("uploaded") else "the example"
    f = sc0.meta.get("MagneticFieldStrength")
    st.success(f"✅ Now showing {who} **{spec['label']}** · brain anatomy (T1w)" + (f" · {f:g} T" if f else "")
               + f" · {sc0.data.shape[0]}×{sc0.data.shape[1]}×{sc0.data.shape[2]} voxels"
               + (" · region outline included" if sc0.roi is not None else "")
               + (f" · {len(sc0.removed)} personal detail(s) removed" if sc0.removed else "")
               + (" · a later scan is included" if len(scans) > 1 else ""))

    st.markdown("### 2 · Result" + (" · first scan" if len(scans) > 1 else ""))
    _state_card(st, cls0)
    use_roi = m0.get("roi_mm3") is not None
    val = m0["roi_mm3"] if use_roi else m0["brain_ml"]
    unit = "mm³" if use_roi else "mL"
    what = "region volume" if use_roi else "brain volume"
    if val:
        u = val * (m0.get("roi_u_rel", 0.03) if use_roi else 0.03)
        lo, hi = supported_range(val, u, g0["status"])
    fmt = lambda v, f: "N/A" if v is None else f.format(v)
    if not expert:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Scan quality**")
            _bars(st, g0, T1W_BARS)
            st.markdown("**By category**")
            st.dataframe(_category_table(cls0), width="stretch", hide_index=True)
        with c2:
            _actions(st, g0)
            st.markdown("**What the result says**")
            if cls0["state"] == "ERROR":
                st.markdown("Nothing: volumes from this scan must not be used.")
            elif not val:
                st.markdown("No volume: without a brain mask ScanTwin measures only motion and settings "
                            "(it does not strip the skull). Upload a brain mask with the scan to measure volumes.")
            else:
                st.plotly_chart(_range_fig([("this scan", val, lo, hi, g0["status"])], f"{what} ({unit})", "{:,.0f}"),
                                width="stretch", key="t1_rng")
                st.caption(f"The true {what} is somewhere between **{lo:,.0f} and {hi:,.0f} {unit}**. "
                           f"Brain {fmt(m0['brain_ml'], '{:.0f}')} mL · grey matter {fmt(m0['gm_ml'], '{:.0f}')} mL · "
                           f"white matter {fmt(m0['wm_ml'], '{:.0f}')} mL.")
        st.caption("Scanner: no quality-control history of this scanner is loaded, so nothing is compared with it and "
                   "no problem is blamed on the machine.")
    else:
        tabs = st.tabs(["Quality checks", "Problems", "Images", "Volumes", "Trace"] + (["DICOM envelope"] if dcheck else []))
        t_q, t_p, t_i, t_v, t_t = tabs[:5]
        if dcheck:
            with tabs[5]:
                st.dataframe([{"": ICON.get(x["status"], "⚪"), "check": x["label"], "catalogue": x["catalogue_id"] or "— (proposed v1.2)",
                               "kind": x["kind"], "value": x["value"], "reason": x["reason"]} for x in dcheck["findings"]],
                             width="stretch", hide_index=True)
                if dcheck.get("ignored"):
                    st.caption(f"Ignored (not DICOM): {', '.join(dcheck['ignored'])}")
        with t_p:
            st.markdown("**By category**")
            st.dataframe(_category_table(cls0), width="stretch", hide_index=True)
            rows = _problem_table(cls0)
            st.markdown("**Everything that is not PASS**")
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.markdown("Nothing.")
            st.caption("Scanner twin: no quality-control history of this scanner is loaded, so it is not compared and no "
                       "problem is attributed to the machine.")
        with t_q:
            st.dataframe(_gate_table(g0), width="stretch", hide_index=True)
            k = st.columns(4)
            k[0].metric("SNR (white matter)", fmt(m0["snr_wm"], "{:.0f}"))
            k[1].metric("Grey/white contrast-to-noise", fmt(m0["cnr"], "{:.1f}"))
            k[2].metric("Brightness variation", fmt(m0["wm_cv_pct"], "{:.1f} %"))
            k[3].metric("Ghost signal", "N/A" if m0["ghost_pct"] is None else f"{m0['ghost_pct']:.2f} %", m0.get("ghost_method", ""),
                        delta_color="off")
            for x in g0["deciding"]:
                st.markdown(f"- **{x['id']}** · {x['fix']}")
        with t_i:
            st.plotly_chart(_slice_fig([("this scan", sc0)]), width="stretch", key="t1_img")
            st.caption("Red: region outline. Volumes use the provided brain mask or an automatic one (no skull stripping).")
        with t_v:
            st.dataframe([{"structure": n, "volume": fmt(v, "{:,.1f} " + un)} for n, v, un in
                          [("head (automatic mask)" if not sc0.brain_mask else "brain (mask)",
                            m0["head_ml"] if not sc0.brain_mask else m0["brain_ml"], "mL"),
                           ("grey matter", m0["gm_ml"], "mL"), ("white matter", m0["wm_ml"], "mL"),
                           ("fluid (CSF)", m0["csf_ml"], "mL")] + ([("region of interest", m0["roi_mm3"], "mm³")] if use_roi else [])],
                         width="stretch", hide_index=True)
            if val:
                st.caption(f"{what.capitalize()} uncertainty: {100 * (u / val):.0f} % (segmentation repeatability, or the registration-implied 3σ_b/r).")
            else:
                st.caption("Tissue volumes need a brain mask: ScanTwin does not strip the skull.")
        with t_t:
            st.markdown(f"**Source file** SHA-256 `{sc0.source_sha256}` · processed in {secs:.1f} s")
            if sc0.removed:
                st.warning(f"Removed {len(sc0.removed)} identifying field(s): {', '.join(sc0.removed)} (catalogue F59).")
            for s_ in sc0.steps:
                if not s_.startswith("de-identify"):
                    st.markdown(f"- {s_}")
            if val:
                with tempfile.TemporaryDirectory() as tmp:
                    sha = T.write_image(sc0, Path(tmp) / "scan.nii.gz")
                p = T.build_packet(sc0, m0, g0, packet_id="E-t0-check_t1w", subject_id="sub-check", session_id="ses-baseline",
                                   months=0.0, rel_path="scan.nii.gz", sha=sha,
                                   origin="SIMULATED" if "synthetic" in spec["label"] else "MEASURED")
                st.download_button("Download the evidence packet (JSON)", json.dumps(p, indent=2), file_name=f"{p['packet_id']}.json",
                                   mime="application/json", key="t1_dl")
            else:
                st.caption("No evidence packet: a packet carries a volume, and none was measured.")

    # ---------------------------------------------------------------- STEP 3 · compare
    st.markdown("### 3 · Compare with a later scan *(optional)*")
    scen = {k: v for k, v in T1W_SCENARIOS.items() if not (v[1] == "grow" and sc0.roi is None)}
    opts = (["real"] if len(scans) > 1 else []) + ["—"] + list(scen)
    labels = {"real": "The later scan I provided (real data)", "—": "Don't compare",
              **{k: f"Simulated: {v[0]}" for k, v in scen.items()}}
    st.caption("Use your own later scan, or imagine the same person is scanned again on this scanner: the later scan is then "
               "simulated from this one, so we know what really changed.")
    sc_key = st.radio("What is the later scan?", opts, index=0, format_func=labels.get, key="chk_scn_t1w")
    if sc_key == "—":
        st.caption("A verdict about the measurement only. No diagnosis, no treatment advice.")
        return
    if sc_key == "real":
        sc1, m1 = scans[1], ms[1]
    else:
        _, fault, strength = scen[sc_key]
        if expert:
            unit_ = {"none": "", "noise": "× noise", "bias": "max brightness change", "scale": "relative size", "motion":
                     "fraction of lines", "shift": "mm", "grow": "mm at the border"}[fault]
            if fault != "none":
                top = {"noise": 4.0, "bias": 2.0, "scale": 0.1, "motion": 0.6, "shift": 8.0, "grow": 9.0}[fault]
                strength = st.slider(f"How strong? ({unit_})", 0.0, top, float(strength), key=f"t1_str_{sc_key}")
        fk = ("t1follow", ck, sc_key, strength)
        if fk not in cache:
            with st.spinner("Simulating the later scan …"):
                s1 = T.degrade(sc0, fault, strength, seed=5)
                cache[fk] = (s1, T.measure(s1, sc0, m0))
        sc1, m1 = cache[fk]
    g1 = T.gate(m1)
    cls1 = classify(g1)
    r = T.compare(m0, g0, m1, g1)
    _card(st, r["status"], r["title"], r["text"])
    st.markdown(f"**Later scan:** {ICON[STATE_STATUS[cls1['state']]]} {cls1['state']}"
                + (f" · {'; '.join(cls1['reasons'])}" if cls1["reasons"] else ""))
    v1 = m1["roi_mm3"] if use_roi else m1["brain_ml"]
    if val and v1:
        items = [("first scan", val, lo, hi, g0["status"])]
        if g1["status"] != "FAIL":
            u1 = v1 * (m1.get("roi_u_rel", 0.03) if use_roi else 0.03)
            lo1, hi1 = supported_range(v1, u1, g1["status"])
            items.append(("later scan", v1, lo1, hi1, g1["status"]))
        st.plotly_chart(_range_fig(items, f"{what} ({unit})", "{:,.0f}"), width="stretch", key="t1_rng_cmp")
        st.caption("Each band is what that scan can support. If the bands overlap a lot, the scans cannot tell a difference.")
    if not expert:
        st.markdown("**Later scan quality**")
        _bars(st, g1, T1W_BARS)
    else:
        k = st.columns(5)
        k[0].metric("Later scan gates", f"{ICON[g1['status']]} {g1['status']}", " ".join(x["id"] for x in g1["deciding"]) or "all pass",
                    delta_color="off")
        k[1].metric("Change z (F60)", fmt(r["z"], "{:+.2f}"), fmt(r["pct"], "{:+.1f} %"), delta_color="off")
        k[2].metric("Head moved by", fmt(m1.get("head_shift_mm"), "{:.1f} mm"), "not an error: scans are aligned", delta_color="off")
        k[3].metric("Outline follows the head within", fmt(m1.get("reg_shift_mm"), "{:.2f} mm"))
        k[4].metric("Image change", fmt(m1.get("image_change_sd"), "{:.2f} × noise"))
        st.plotly_chart(_slice_fig([("first scan", sc0), ("later scan", sc1)]), width="stretch", key="t1_img_cmp")
        st.dataframe(_gate_table(g1), width="stretch", hide_index=True)
    st.caption("A verdict about the measurement only. No diagnosis, no treatment advice.")


# ------------------------------------------------------------------------------------------------ evidence packet
def _render_packet(st, p: dict, name: str, root: Path, thresholds: dict, schema: dict, packets: dict, manifest: dict, expert: bool):
    """An evidence packet (e.g. an unseen Gate 3 packet): same verdict card, StarterPack baseline as the earlier scan."""
    ev = evaluate(p, thresholds, schema, manifest, root)
    obs = p.get("observable")
    base = next((q for k, q in packets.items() if k.startswith("E-t0") and q.get("observable") == obs), None)
    tw = Twin()
    if base is not None:
        tw.ingest(base, evaluate(base, thresholds, schema, manifest, root))
    e = tw.ingest(p, ev)
    st.success(f"✅ Now showing your evidence packet **{name}** ({p.get('packet_id')}, {obs})")
    st.markdown("### 2 · Result")
    bad = [f for f in ev.findings if f.status != "PASS"]
    plain_rule = {"mrs_snr": "F13", "mrs_linewidth_Hz": "F12", "registration_rms_mm": "F32"}
    reasons = [PLAIN[plain_rule[f.rule]][1] if f.rule in plain_rule else f.reason for f in bad[:4]]
    head = {"PASS": ("🟢 Good to use", "Every check passed."),
            "WARN": ("🟡 Use with caution", "Usable, but less precise than normal: " + "; ".join(reasons) + "."),
            "FAIL": ("🔴 Don't use", "This result can't be trusted: " + "; ".join(reasons) + ".")}[ev.status]
    _card(st, ev.status, *head)
    if e.kind == "update":
        st.markdown(f"The record moves to **{e.posterior['x']:.3f} ± {e.posterior['sd']:.3f}**"
                    + (" (counted at reduced weight)" if ev.status == "WARN" else "") + ".")
    elif e.kind == "reject":
        st.markdown("The record **does not change**; the reasons are logged.")
    if expert and bad:
        st.dataframe([{"": ICON.get(f.status, "⚪"), "rule": f.rule, "owner": f.owner, "reason": f.reason} for f in bad],
                     width="stretch", hide_index=True)
    st.caption("The packet is never modified. A verdict about the measurement only.")
