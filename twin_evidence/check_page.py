"""ScanTwin "Check a scan" page (Streamlit).

Flow:   1 Choose a scan (upload OR example, never both)  ->  2 Result  ->  3 Compare with a later scan (optional)
Views:  Simple (plain words, no codes or units) and Expert (gates, spectrum, twin, trace). Both read the SAME result.
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

from . import ingest as I
from . import mrs_fit as F
from .gates import MRS_GATES, context_advice, context_factors, supported_range, STANDING_LIMITATIONS
from .twin import Twin
from .validate import evaluate

REPO = Path(__file__).resolve().parents[1]
ICON = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴", "N/A": "⚪"}
WORD = {"PASS": "Good", "WARN": "Caution", "FAIL": "Problem", "N/A": "Can't check"}
COLOR = {"PASS": "#1f8a4c", "WARN": "#b77800", "FAIL": "#c23b35", "N/A": "#8a96a3"}
BG = {"PASS": "rgba(31,138,76,.10)", "WARN": "rgba(245,177,61,.14)", "FAIL": "rgba(194,59,53,.10)", "N/A": "rgba(138,150,163,.12)"}

EXAMPLES = {
    "Real brain scan · 7 T · public dataset (Zenodo, CC BY 4.0)":
        REPO / "external_data/zenodo_5085449_example_02/derivatives/sub-ext01_ses-baseline_transients_svs.nii.gz",
    "Hackathon scan · May · 3 T (synthetic)": "SP:baseline",
    "Hackathon scan · November · 3 T (synthetic)": "SP:followup",
}

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
    "F35": ("Same settings as last time", "the scan settings differ from last time", "Repeat with the same settings as last time"),
    "F19": ("Same sequence as last time", "the sequence or pulse settings changed", "Use the baseline sequence"),
    "F61": ("Transmitter calibration", "the transmitter is miscalibrated", "Recalibrate the transmitter and repeat"),
}
SIMPLE_BARS = ["F13", "F12", "F08", "F23", "F15", "F14"]

SCENARIOS = {   # what happens before the next scan: label, fault, strength (None = default from the scan)
    "repeat": ("Same patient · scanner OK", "none", 0.0),
    "machine_noise": ("Scanner problem · weak receive coil", "noise", None),
    "machine_shim": ("Scanner problem · magnet badly tuned", "shim", None),
    "machine_drift": ("Scanner problem · frequency drift", "drift", None),
    "machine_water": ("Scanner problem · water suppression failed", "water", 5.0),
    "patient": ("Patient changed · more choline", "cho_up", None),
}


# ------------------------------------------------------------------------------------------------ helpers
def _default_strength(fault: str, nu0: float, snr: float) -> float:
    k = nu0 / I.REF_NU0
    return {"shim": round(9 * k, 1), "drift": round(10 * k, 1), "noise": round(max(1.5, snr / 18), 1), "cho_up": 70.0}.get(fault, 0.0)


def _quality_score(gid: str, value) -> float | None:
    """0-100 bar for the Simple view: PASS zone 60-100, WARN zone 30-60, FAIL zone 0-30."""
    g = next((x for x in MRS_GATES if x.id == gid), None)
    if g is None or value is None or g.warn is None:
        return None
    w = g.warn[1]
    f = g.fail[1] if g.fail else (w * 2 if g.warn[0] == ">" else w / 2)
    v = float(value)
    if g.warn[0] == ">":                         # lower is better
        if v <= w:
            return 60 + 40 * max(0.0, 1 - v / w)
        if v <= f:
            return 30 + 30 * (f - v) / (f - w)
        return max(0.0, 30 * (1 - (v - f) / f))
    if v >= w:                                   # higher is better
        return min(100.0, 60 + 40 * (v - w) / w)
    if v >= f:
        return 30 + 30 * (v - f) / (w - f)
    return max(0.0, 30 * v / f)


def _card(st, status: str, title: str, body: str):
    st.markdown(f"""<div style="border-left:6px solid {COLOR[status]};background:{BG[status]};border-radius:10px;
        padding:16px 20px;margin:4px 0 14px 0">
        <div style="font-size:1.6rem;font-weight:700;color:{COLOR[status]};line-height:1.25">{title}</div>
        <div style="font-size:1.05rem;margin-top:6px">{body}</div></div>""", unsafe_allow_html=True)


def _bars(st, g: dict):
    status = {x["id"]: x for x in g["gates"]}
    for gid in SIMPLE_BARS:
        x = status.get(gid)
        if x is None:
            continue
        name = PLAIN[gid][0]
        if x["status"] == "N/A":
            st.markdown(f"<div style='display:flex;gap:12px;align-items:center;margin:7px 0'><div style='width:200px'>{name}</div>"
                        f"<div style='flex:1;color:#8a96a3'>⚪ can't check this for this file</div></div>", unsafe_allow_html=True)
            continue
        sc = _quality_score(gid, x["value"]) or 0
        cause = " ← this is why" if x["status"] != "PASS" else ""
        st.markdown(f"""<div style="display:flex;gap:12px;align-items:center;margin:7px 0">
            <div style="width:200px">{name}</div>
            <div style="flex:1;height:12px;border-radius:6px;background:rgba(138,150,163,.25);overflow:hidden">
              <div style="width:{sc:.0f}%;height:100%;background:{COLOR[x['status']]}"></div></div>
            <div style="width:160px;color:{COLOR[x['status']]};font-weight:600">{WORD[x['status']]}{cause}</div></div>""",
                    unsafe_allow_html=True)


def _range_fig(items: list) -> go.Figure:
    """items: (label, value, lo, hi, status). If two bands overlap a lot, the scans cannot tell a difference."""
    fig = go.Figure()
    for i, (lab, v, lo, hi, st_) in enumerate(items):
        y = len(items) - i
        fig.add_trace(go.Scatter(x=[lo, hi], y=[y, y], mode="lines", line=dict(color=COLOR[st_], width=12),
                                 hoverinfo="skip", showlegend=False, opacity=0.45))
        fig.add_trace(go.Scatter(x=[v], y=[y], mode="markers+text", marker=dict(size=16, color="white",
                                 line=dict(color=COLOR[st_], width=3)), text=[f"{v:.2f}"], textposition="top center",
                                 showlegend=False, hovertemplate=f"{lab}: %{{x:.3f}}<extra></extra>"))
    fig.update_layout(height=80 + 55 * len(items), margin=dict(l=10, r=10, t=10, b=30),
                      yaxis=dict(tickvals=list(range(len(items), 0, -1)), ticktext=[x[0] for x in items],
                                 range=[0.4, len(items) + 0.7]),
                      xaxis=dict(title="choline level (Cho/Cr)"), plot_bgcolor="rgba(0,0,0,0)")
    return fig


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


def _headline(g: dict) -> tuple[str, str]:
    dec = g["deciding"]
    why = " and ".join(PLAIN.get(x["id"], (None, x["label"].lower()))[1] for x in dec)
    if g["status"] == "PASS":
        return "🟢 Good to use", "This scan is reliable. Every quality check passed."
    if g["status"] == "WARN":
        return "🟡 Use with caution", f"Usable, but less precise than normal: {why}."
    return "🔴 Don't use", f"This scan can't be trusted: {why}. The result must not be used."


def _packetise(ing, m, g, tmp: Path, *, tag, ses, months, origin, sysqc, desc=""):
    rel = f"{tag}.nii.gz"
    sha = I.write_derivative(ing, tmp / rel)
    pid = ("E-t0-" if months == 0 else "E-t1-") + f"check_{tag}"
    p = I.build_packet(ing, m, g, packet_id=pid, subject_id="sub-check", session_id=ses, months=months, rel_path=rel,
                       sha=sha, origin=origin, system_qc_id=sysqc, limit_note=desc if origin == "SIMULATED" else "")
    return p, {rel: sha}


def _compare(ing, m, g, fi, fm, thresholds, schema, sysqc, label):
    fg = I.gate(fm, fi.nu0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p0, man0 = _packetise(ing, m, g, tmp, tag="baseline", ses="ses-baseline", months=0.0, origin="MEASURED", sysqc=sysqc)
        p1, man1 = _packetise(fi, fm, fg, tmp, tag="followup", ses="ses-followup", months=6.0, origin="SIMULATED",
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


# ------------------------------------------------------------------------------------------------ page
def render(st, *, starterpack_root: Path, thresholds: dict, schema: dict, packets: dict, manifest: dict):
    top_l, top_r = st.columns([4, 1])
    top_l.title("ScanTwin · Check a scan")
    top_l.markdown("Find out **if a spectroscopy scan can be trusted**, **why**, and **what to do next**. "
                   "ScanTwin judges the *measurement*, never the person.")
    view = top_r.radio("View", ["Simple", "Expert"], index=0, horizontal=True, key="chk_view",
                       help="Simple: plain words. Expert: every check, number and source.")
    expert = view == "Expert"

    # ---------------------------------------------------------------- STEP 1 · choose
    st.markdown("### 1 · Choose a scan")
    src = st.radio("Source", ["Upload my file", "Use an example"], index=0, horizontal=True, key="chk_src",
                   label_visibility="collapsed")
    up = ex = None
    if src == "Upload my file":
        up = st.file_uploader("Spectroscopy file: NIfTI-MRS (.nii.gz / .nii) or a ScanTwin evidence packet (.json)",
                              type=["gz", "nii", "json"], key="chk_upload")
        if up is None:
            st.info("⬆️ **Drop a file above to start.** No file at hand? Choose **Use an example**.")
            return
    else:
        ex = st.selectbox("Example", list(EXAMPLES), index=0, key="chk_example")

    if up is not None and up.name.lower().endswith(".json"):
        _render_packet(st, json.loads(up.getvalue()), up.name, starterpack_root, thresholds, schema, packets, manifest, expert)
        return

    cache = st.session_state.setdefault("chk_cache", {})
    key = ("upload", up.name, up.size) if up is not None else ("example", ex)
    if key not in cache:
        with st.status("Analysing the scan …", expanded=True) as status:
            t0 = time.time()
            try:
                st.write("Reading the file and removing personal details …")
                if up is not None:
                    ing = I.read_bytes(up.getvalue(), up.name)
                else:
                    target = EXAMPLES[ex]
                    if isinstance(target, str) and target.startswith("SP:"):
                        s = target[3:]
                        ing = I.read(starterpack_root / f"04_MRS_BIDS/sub-01/ses-{s}/mrs/sub-01_ses-{s}_acq-press_nuc-1H_voi-frontal_svs.nii.gz")
                    else:
                        ing = I.read(target)
                st.write("Combining coils, checking every repeat, measuring quality …")
                m = I.measure(ing)
            except Exception as err:
                status.update(label="Could not read this file", state="error")
                st.error("This doesn't look like a single-voxel spectroscopy file (NIfTI-MRS). "
                         "Ask your physicist to export it as NIfTI-MRS, for example with spec2nii."
                         + (f"\n\nTechnical detail: `{err}`" if expert else ""))
                return
            cache[key] = (ing, m, time.time() - t0)
            status.update(label=f"Done in {time.time() - t0:.1f} s", state="complete", expanded=False)
    ing, m, secs = cache[key]
    g = I.gate(m, ing.nu0)
    meta = ing.meta
    field_T = ing.nu0 / I.GAMMA_MHZ_PER_T
    who = f"your file **{up.name}**" if up is not None else f"the example **{ex}**"
    st.success(f"✅ Now showing {who} · {meta.get('Manufacturer', 'scanner')} {field_T:.1f} T · "
               f"{ing.transients.shape[0]} repeat(s)" + (f" · {len(ing.removed)} personal detail(s) removed" if ing.removed else ""))

    # ---------------------------------------------------------------- STEP 2 · result
    st.markdown("### 2 · Result")
    title, body = _headline(g)
    _card(st, g["status"], title, body)
    u = math.hypot(m["cho_cr_crlb"], 0.045)
    lo, hi = supported_range(m["cho_cr"], u, g["status"])

    if not expert:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Scan quality**")
            _bars(st, g)
        with c2:
            st.markdown("**What to do**")
            if g["deciding"]:
                for x in g["deciding"]:
                    st.markdown(f"✅ {PLAIN.get(x['id'], (None, None, x['fix']))[2]}")
            else:
                st.markdown("✅ Nothing to fix.")
            st.markdown("**What the result says**")
            if g["status"] == "FAIL":
                st.markdown("Nothing: the value from this scan must not be used.")
            else:
                st.plotly_chart(_range_fig([("this scan", m["cho_cr"], lo, hi, g["status"])]), width="stretch", key="rng_main")
                st.caption(f"The true choline level is somewhere between **{lo:.2f} and {hi:.2f}**. "
                           "Differences smaller than this band mean nothing.")
    else:
        t_q, t_s, t_c, t_t = st.tabs(["Quality checks", "Spectrum", "Scan context", "Trace"])
        with t_q:
            raw = {"F12": m.get("fwhm_hz"), "F08": m.get("drift_abs_hz")}
            rows = []
            for x in g["gates"]:
                if not x["applies"]:
                    continue
                v = x["value"]
                if v is None:
                    val = "N/A"
                elif x["id"] in raw and raw[x["id"]] is not None and abs(ing.nu0 - I.REF_NU0) > 1:
                    val = f"{raw[x['id']]:.1f} Hz at {field_T:.0f} T (= {v:.1f} Hz at 3 T)"
                else:
                    val = f"{v:.2f} {x['unit']}".strip() if isinstance(v, float) else str(v)
                rows.append({"": ICON.get(x["status"], "⚪"), "ID": x["id"], "check": PLAIN.get(x["id"], (x["label"],))[0],
                             "measured": val, "PASS if": x["pass_if"], "source": x.get("source", ""),
                             "note": x.get("reason") or ""})
            st.dataframe(rows, width="stretch", hide_index=True)
            k = st.columns(4)
            k[0].metric("Cho/Cr", f"{m['cho_cr']:.3f} ± {u:.3f}", f"supports {lo:.2f}–{hi:.2f}", delta_color="off")
            k[1].metric("NAA/Cr", f"{m['naa_cr']:.2f}")
            k[2].metric("SNR", f"{m['snr']:.0f}")
            k[3].metric("Linewidth", f"{m['fwhm_ppm']:.3f} ppm", f"{m['fwhm_hz']:.1f} Hz", delta_color="off")
            if g["deciding"]:
                st.markdown("**Fix (catalogue text)**")
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
                p, _ = _packetise(ing, m, g, Path(tmp), tag="scan", ses="ses-baseline", months=0.0,
                                  origin="SIMULATED" if "synthetic" in (ex or "") else "MEASURED", sysqc=None)
            st.download_button("Download the evidence packet (JSON)", json.dumps(p, indent=2), file_name=f"{p['packet_id']}.json",
                               mime="application/json", key="chk_dl")

    # ---------------------------------------------------------------- STEP 3 · compare
    st.markdown("### 3 · Compare with a later scan *(optional)*")
    st.caption("Imagine the same person is scanned again in six months on this scanner. "
               "The later scan is simulated from this scan, so we know what really changed.")
    sc_key = st.radio("What happens before the next scan?", ["—"] + list(SCENARIOS), index=0,
                      format_func=lambda k: "Don't compare" if k == "—" else SCENARIOS[k][0], key="chk_scn")
    if sc_key == "—":
        st.caption("A verdict about the measurement only. No diagnosis, no treatment advice.")
        return
    label, fault, strength = SCENARIOS[sc_key]
    if strength is None:
        dflt = _default_strength(fault, ing.nu0, m["snr"])
        if expert or fault == "cho_up":
            unit = {"shim": "Hz extra blur", "drift": "Hz drift", "noise": "× noise", "cho_up": "% more choline"}[fault]
            strength = st.slider(f"How strong? ({unit})", 0.0, 100.0 if fault == "cho_up" else float(3 * dflt), float(dflt),
                                 key=f"chk_str_{sc_key}")
        else:
            strength = dflt
        if fault == "cho_up":
            strength = strength / 100.0
    sysqc = "ASSUMED:external-dataset"
    if expert and st.toggle("Strict: reject scans without a site QC record", value=False, key="chk_strict"):
        sysqc = None
    fk = ("follow", key, sc_key, strength)
    if fk not in cache:
        with st.spinner("Simulating the later scan …"):
            fi = I.degrade(ing, fault, strength, seed=11)
            cache[fk] = (fi, I.measure(fi))
    fi, fm = cache[fk]
    r = _compare(ing, m, g, fi, fm, thresholds, schema, sysqc, label)
    colour_status, title, text = r["who"]
    _card(st, colour_status, title, text)
    fg = r["fg"]
    items = [("first scan", m["cho_cr"], lo, hi, g["status"])]
    if fg["status"] != "FAIL":
        lo1, hi1 = supported_range(fm["cho_cr"], math.hypot(fm["cho_cr_crlb"], 0.045), fg["status"])
        items.append(("later scan", fm["cho_cr"], lo1, hi1, fg["status"]))
    st.plotly_chart(_range_fig(items), width="stretch", key="rng_cmp")
    st.caption("Each band is what that scan can support. If the bands overlap a lot, the scans cannot tell a difference.")
    if expert:
        e1 = r["e1"]
        k = st.columns(4)
        k[0].metric("Later scan gates", f"{ICON[fg['status']]} {fg['status']}", " ".join(x["id"] for x in fg["deciding"]) or "all pass",
                    delta_color="off")
        k[1].metric("Change z (F60)", f"{r['z']:+.2f}")
        k[2].metric("Gain K", "—" if e1.K is None else f"{e1.K:.2f}", e1.action, delta_color="off")
        k[3].metric("Record after", "unchanged" if e1.posterior is None else f"{e1.posterior['x']:.3f} ± {e1.posterior['sd']:.3f}")
        st.plotly_chart(_spectrum_fig([("first scan", ing.fid, "#2b6cb0"), ("later scan (simulated)", fi.fid, "#c8642a")],
                                      ing.nu0, ing.dwell), width="stretch", key="spec_cmp")
        reasons = [f"{f.rule}: {f.reason}" for f in r["ev1"].findings if f.status != "PASS"]
        if reasons:
            st.caption("Later-scan findings: " + " · ".join(reasons))
    st.caption("A verdict about the measurement only. No diagnosis, no treatment advice.")


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
    plain_rule = {"mrs_snr": "F13", "mrs_linewidth_Hz": "F12"}
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
