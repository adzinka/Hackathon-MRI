"""Streamlit demonstrator on top of twin_evidence: prior -> evidence -> posterior, decision panel, provenance.

run:  streamlit run app.py
      (StarterPack path is taken from the sidebar; default ../MRI_Hackathon_DataScience_StarterPack_v0.3)
Synthetic, educational data. No clinical meaning.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from twin_evidence.twin import Twin, CHI2_95_1DOF
from twin_evidence.validate import evaluate, load_json, load_manifest, to_months
from twin_evidence import views

st.set_page_config(page_title="ScanTwin", page_icon="🧠", layout="wide")

BASELINE = ["E-t0-roi", "E-t0-naa", "E-t0-cho"]
FOLLOWUP = ["E-t1-cho", "E-t1-naa", "E-t1-roi", "E-t1-chobad"]
STATUS_COLOR = {"PASS": "#2e9e5b", "WARN": "#e0a100", "FAIL": "#d64545"}
STATUS_ICON = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴"}
OWNER = {"QE": "Quantum Engineering", "PM": "Precision Mechanics / Metrology", "DS": "Data Science",
         "DT": "Digital Twins", "shared": "shared"}
C_PRIOR, C_EVID, C_POST, C_ALT = "#6b7a8f", "#e07b39", "#2b6cb0", "#9aa5b1"

# candidate next measurements (Dossier §7.7); sigma_m in units of the observable, minutes of scanner time
NEXT = {
    "Cho_Cr": [("Repeat PRESS after re-shim", 0.050, "PASS", 4.1),
               ("Double averages, no re-shim", 0.042, "WARN", 4.3)],
    "NAA_Cr": [("Repeat PRESS after re-shim", 0.060, "PASS", 4.1),
               ("Double averages, no re-shim", 0.057, "WARN", 4.3)],
    "roi_volume": [("Registration + segmentation audit, then re-measure", 0.03, "PASS", 10.0),
                   ("Repeat T1w scan, same pipeline", 0.19, "WARN", 6.0)],  # relative SD for the ROI
}


# ---------- data ----------
@st.cache_data
def load_pack(root: str):
    r = Path(root)
    tw = r / "05_TwinReady_JSON_CSV"
    return (load_json(tw / "thresholds-v1.0.json"), load_json(tw / "evidence_packet_v1.0.schema.json"),
            load_manifest(r / "manifest_sha256.csv"),
            {n: load_json(tw / "packets" / f"{n}.json") for n in BASELINE + FOLLOWUP})


def run_twin(packets, thresholds, schema, manifest, root, scanner=None):
    twin, rows = Twin(), []
    for p in packets:
        ev = evaluate(p, thresholds, schema, manifest, Path(root), scanner=scanner)
        rows.append((p, ev, twin.ingest(p, ev)))
    return twin, rows


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def fmt(v, obs):
    if v is None:
        return "–"
    return f"{v:,.0f}".replace(",", " ") if obs == "roi_volume" else f"{v:.3f}"


# ---------- sidebar ----------
st.sidebar.title("ScanTwin")
with st.sidebar.expander("⚙️ Settings"):
    root = st.text_input("StarterPack v0.3 folder", "../MRI_Hackathon_DataScience_StarterPack_v0.3")
if not (Path(root) / "05_TwinReady_JSON_CSV").exists():
    st.error(f"StarterPack not found at `{root}`. Set the path in the sidebar.")
    st.stop()
thresholds, schema, manifest, PK = load_pack(str(Path(root).resolve()))

mode = st.sidebar.radio("Page", ["🔎 Check a scan", "🧪 Hackathon lab"], index=0, key="mode",
                        help="Check a scan: upload a spectrum or packet and get a verdict. "
                             "Hackathon lab: the worked StarterPack example, scanner twin and fault scenarios.")
if mode.startswith("🔎"):
    from twin_evidence import check_page
    st.sidebar.caption("Synthetic and public de-identified data only. Verdicts are about the measurement, never the person.")
    check_page.render(st, starterpack_root=Path(root).resolve(), thresholds=thresholds, schema=schema,
                      packets=PK, manifest=manifest)
    st.stop()

uploaded = st.sidebar.file_uploader("Unseen follow-up packet (JSON)", type="json")
choices = {f"{n}  ·  {PK[n]['observable']}": PK[n] for n in FOLLOWUP}
if uploaded is not None:
    try:
        up = json.load(uploaded)
        choices = {f"⬆ {up.get('packet_id', 'uploaded')}  ·  {up.get('observable')}": up, **choices}
    except json.JSONDecodeError as e:
        st.sidebar.error(f"Not valid JSON: {e}")
label = st.sidebar.radio("Follow-up packet to judge", list(choices))
packet = choices[label]

st.sidebar.markdown("**Declared team policy**")
warn_factor = st.sidebar.slider("WARN variance factor (R = f·u² in the gain)", 1.0, 4.0,
                                float(thresholds["twin_policy"]["WARN"]["R_factor"]), 0.5)
show_alt = st.sidebar.checkbox("Show result without WARN policy (f = 1)", True)
st.sidebar.markdown("**Scanner twin** (SIMULATED weekly phantom record)")
from twin_evidence import longitudinal as LNG
from twin_evidence.scanner_link import ScannerRecord, attribution, CLASS_TEXT
SCANNER_CHOICES = ["none"] + list(LNG.FAULTS)
scanner_choice = st.sidebar.radio("State of the scanner from week 14 to 26", SCANNER_CHOICES, index=1,
                                  format_func=lambda k: "no scanner twin (Gate 1 reference)" if k == "none" else LNG.FAULTS[k]["title"])


@st.cache_resource(show_spinner="Simulating 27 weekly phantom scans …")
def _record(fault: str):
    return ScannerRecord.simulate(fault)


rec = None if scanner_choice == "none" else _record(scanner_choice)
st.sidebar.caption("Synthetic, educational data (SIMULATED origin). No clinical meaning. Drafted with Claude, reviewed by the team.")

th = copy.deepcopy(thresholds)
th["twin_policy"]["WARN"]["R_factor"] = warn_factor
th_alt = copy.deepcopy(thresholds)
th_alt["twin_policy"]["WARN"]["R_factor"] = 1.0

seq = [PK[n] for n in BASELINE] + [packet]
twin, rows = run_twin(seq, th, schema, manifest, root, rec)
p, ev, e = rows[-1]
obs = p.get("observable")
base = next(((bp, bev, be) for bp, bev, be in rows[:-1] if bp.get("observable") == obs), None)
alt = run_twin(seq, th_alt, schema, manifest, root, rec)[1][-1][2] if show_alt else None
scan_f = next((f for f in ev.findings if f.rule == "F11_scanner_twin"), None)
who = attribution(ev.status, scan_f, rec)

# ---------- header ----------
st.title("ScanTwin")
st.markdown("**A digital twin of the MRI scanner and of the subject.** It does not tell you what is wrong with the patient. "
            "It tells you whether the scan is good enough to say anything at all, and how to make the next one better.")
st.caption("Evidence gate + gated Kalman update · StarterPack v0.3 · Hackathon Dossier v1.6 (§3.10, §5.3, §7.5, §7.7, §8.2)")

c = st.columns(4)
c[0].metric("Packet", p.get("packet_id", "?"), obs)
c[1].metric("Gate status", f"{STATUS_ICON[ev.status]} {ev.status}",
            f"stored: {ev.stored_status}" if ev.stored_status != ev.status else "stored status confirmed",
            delta_color="off")
c[2].metric("Twin action", e.action)
if e.kind == "update":
    c[3].metric("Twin variable", f"{fmt(e.posterior['x'], obs)} ± {fmt(e.posterior['sd'], obs)}",
                f"{e.posterior['x'] - e.prior['x']:+.3g} vs prior", delta_color="off")
elif base:
    c[3].metric("Twin variable", f"{fmt(twin.state[obs].x, obs)} ± {fmt(twin.state[obs].sd, obs)}", "unchanged", delta_color="off")

WHO_ICON = {"machine": "🛠", "patient": "🧑", "session": "🎯", "unknown": "❔"}
WHO_BOX = {"machine": st.warning, "patient": st.success, "session": st.info, "unknown": st.info}
sc_line = f"Scanner twin on the scan date: **{STATUS_ICON[scan_f.status]} {scan_f.status}**" if scan_f else "Scanner twin: not attached"
WHO_BOX[who["who"]](f"### {WHO_ICON[who['who']]} Patient or machine?  \n{who['text']}  \n{sc_line}"
                    + (f" · scanner state: {rec.label} (SIMULATED)" if rec else ""))

tab_upd, tab_dec, tab_prov, tab_time, tab_3d, tab_scan, tab_sc = st.tabs(
    ["① Prior → evidence → posterior", "② Decision", "③ Provenance", "④ Timeline (all packets)",
     "⑤ 3D evidence map", "⑥ Scanner twin", "⑦ ScanTwin: fault scenarios"])

# ---------- ① prior -> evidence -> posterior ----------
with tab_upd:
    if base is None:
        st.warning(f"No baseline for observable `{obs}`: this packet would only initialise the state.")
    else:
        b_state = base[2].posterior
        if e.kind == "update":
            prior, post = e.prior, e.posterior
            y, u, u_gain = e.y, e.u, math.sqrt(e.R_gain)
        else:  # rejected: the state stays where it was
            prior = post = b_state
            y = p.get("value") if isinstance(p.get("value"), (int, float)) else None
            u = (p.get("uncertainty") or {}).get("value")
            u = u if isinstance(u, (int, float)) and u > 0 else None
            u_gain = u

        left, right = st.columns([3, 2])
        with left:
            sds = [s for s in (prior["sd"], post["sd"], u_gain) if s]
            mus = [m for m in (prior["x"], post["x"], y) if m is not None]
            xs = np.linspace(min(mus) - 4 * max(sds), max(mus) + 4 * max(sds), 600)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=gauss(xs, prior["x"], prior["sd"]), name="old estimate (prior, incl. drift Q)",
                                     line=dict(color=C_PRIOR, width=2), fill="tozeroy", fillcolor="rgba(107,122,143,.12)"))
            if y is not None and u:
                rejected = e.kind == "reject"
                fig.add_trace(go.Scatter(x=xs, y=gauss(xs, y, u), name="new measurement (packet u)" + (" — REJECTED" if rejected else ""),
                                         line=dict(color=C_ALT if rejected else C_EVID, width=2, dash="dot" if rejected else "solid")))
                if not rejected and abs(u_gain - u) > 1e-12:
                    fig.add_trace(go.Scatter(x=xs, y=gauss(xs, y, u_gain), name=f"measurement as used in the gain (√{e.R_gain / e.R_gate:.1f}·u)",
                                             line=dict(color=C_EVID, width=1.5, dash="dash")))
            if e.kind == "update":
                fig.add_trace(go.Scatter(x=xs, y=gauss(xs, post["x"], post["sd"]), name="new estimate (posterior)",
                                         line=dict(color=C_POST, width=3), fill="tozeroy", fillcolor="rgba(43,108,176,.15)"))
                if alt is not None and alt.kind == "update" and abs(alt.K - e.K) > 1e-9:
                    fig.add_trace(go.Scatter(x=xs, y=gauss(xs, alt.posterior["x"], alt.posterior["sd"]), name="posterior without WARN policy",
                                             line=dict(color=C_POST, width=1.5, dash="dot")))
            for m, col in ((prior["x"], C_PRIOR), (y, C_EVID), (post["x"] if e.kind == "update" else None, C_POST)):
                if m is not None:
                    fig.add_vline(x=m, line=dict(color=col, width=1, dash="dot"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.18),
                              title=f"{obs}: old estimate → new measurement → new estimate",
                              xaxis_title=f"{obs} [{p.get('unit')}]", yaxis_title="probability density", yaxis_showticklabels=False)
            st.plotly_chart(fig, width="stretch")

        with right:
            # the same story over time: t0 state, drift, t1 measurement, posterior
            t0 = to_months(base[0].get("relative_time")) or 0.0
            t1 = to_months(p.get("relative_time")) or 6.0
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=[t0, t1], y=[b_state["x"], prior["x"]], mode="lines", name="prior carried forward",
                                    line=dict(color=C_PRIOR, dash="dot")))
            f2.add_trace(go.Scatter(x=[t0, t1, t1, t0], y=[b_state["x"] + 1.96 * b_state["sd"], prior["x"] + 1.96 * prior["sd"],
                                                            prior["x"] - 1.96 * prior["sd"], b_state["x"] - 1.96 * b_state["sd"]],
                                    fill="toself", fillcolor="rgba(107,122,143,.12)", line=dict(width=0), name="prior 95 % band"))
            f2.add_trace(go.Scatter(x=[t0], y=[b_state["x"]], error_y=dict(array=[1.96 * b_state["sd"]]), mode="markers",
                                    marker=dict(color=C_PRIOR, size=11), name="baseline t0"))
            if y is not None and u:
                f2.add_trace(go.Scatter(x=[t1 - 0.12], y=[y], error_y=dict(array=[1.96 * u]), mode="markers",
                                        marker=dict(color=STATUS_COLOR[ev.status], size=11, symbol="x" if e.kind == "reject" else "diamond"),
                                        name=f"measurement t1 ({ev.status})"))
            if e.kind == "update":
                f2.add_trace(go.Scatter(x=[t1 + 0.12], y=[post["x"]], error_y=dict(array=[1.96 * post["sd"]]), mode="markers",
                                        marker=dict(color=C_POST, size=12), name="posterior t1"))
            f2.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), title="Over time (±95 %)",
                             xaxis_title="months since baseline", yaxis_title=f"{obs} [{p.get('unit')}]",
                             legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(f2, width="stretch")

        if e.kind == "update":
            st.markdown("**The step in numbers** (scalar Kalman, h = 1)")
            st.latex(
                rf"x^- = {fmt(e.prior['x'], obs)},\; P^- = P_0 + Q = {fmt(e.prior['sd'], obs)}^2,\; "
                rf"y = {fmt(e.y, obs)},\; R_{{gate}} = u^2 = {fmt(e.u, obs)}^2,\; R_{{gain}} = {e.R_gain / e.R_gate:.1f}\,u^2")
            st.latex(
                rf"K = \frac{{P^-}}{{P^- + R_{{gain}}}} = {e.K:.3f},\quad x^+ = x^- + K(y - x^-) = {fmt(e.posterior['x'], obs)},"
                rf"\quad \sigma^+ = {fmt(e.posterior['sd'], obs)}")
            if alt is not None and alt.kind == "update" and abs(alt.K - e.K) > 1e-9:
                st.caption(f"Without the WARN policy: K = {alt.K:.3f}, x⁺ = {fmt(alt.posterior['x'], obs)} ± {fmt(alt.posterior['sd'], obs)}. "
                           "The policy factor is a declared team choice, not a physical constant (Dossier §7.5).")
        else:
            st.error("Evidence rejected: the twin state is **unchanged**. The measurement is shown greyed for transparency only.")

# ---------- ② decision ----------
with tab_dec:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        box = {"PASS": st.success, "WARN": st.warning, "FAIL": st.error}[ev.status]
        box(f"### {STATUS_ICON[ev.status]} {ev.status} → `{e.action}`\n"
            + {"update": "Evidence accepted with its packet uncertainty.",
               "update_with_warning": f"Evidence accepted, but its variance is multiplied by {warn_factor:g} in the gain.",
               "reject": "Evidence rejected. The twin state is not changed; the reasons are logged."}[e.action])
        if ev.stored_status != ev.status:
            st.info(f"Stored `qc_status` was **{ev.stored_status}**, re-evaluation gives **{ev.status}**. "
                    "We never trust a stored status blindly.")
        if e.kind == "update":
            m = st.columns(3)
            m[0].metric("z (did the measurement change?)", f"{e.z_change:+.2f}" if e.z_change is not None else "–",
                        "≥ 1.96: beyond noise" if e.z_change is not None and abs(e.z_change) >= 1.96 else "within noise", delta_color="off")
            m[1].metric("d² (surprising for the twin?)", f"{e.d2:.2f}",
                        f"> {CHI2_95_1DOF}: model disagreement" if e.d2_flag else f"< {CHI2_95_1DOF}: consistent", delta_color="off")
            m[2].metric("Gain K", f"{e.K:.3f}", "state held" if e.K < 0.05 else "state moves", delta_color="off")
            if e.z_change is not None and abs(e.z_change) >= 1.96 and not e.d2_flag:
                st.caption("Borderline z with an unsurprising d²: the twin updates with a warning and asks for a better measurement (Dossier §3.10).")
            # level-4: prediction of the next repeat if nothing changes (Dossier §7.6)
            nx = NEXT.get(obs, [])
            if nx:
                px = e.posterior
                s_m = nx[0][1] * (px["x"] if obs == "roi_volume" else 1)
                half = 1.96 * math.sqrt(px["sd"] ** 2 + s_m ** 2)
                st.markdown(f"**Prediction** (no true change): a PASS-quality repeat should land in "
                            f"**{fmt(px['x'] - half, obs)} … {fmt(px['x'] + half, obs)}** (95 %).")

        st.markdown("#### Why? Non-PASS findings")
        bad = [f for f in ev.findings if f.status != "PASS"]
        if bad:
            st.dataframe([{"": STATUS_ICON[f.status], "rule": f.rule, "owner": OWNER.get(f.owner, f.owner),
                           "value": "" if f.value is None else str(f.value), "threshold": f.threshold, "reason": f.reason} for f in bad],
                         width="stretch", hide_index=True)
        else:
            st.write("All checks passed.")
        for n in ev.notes:
            st.caption(f"note: {n}")
        with st.expander(f"All {len(ev.findings)} checks"):
            st.dataframe([{"": STATUS_ICON[f.status], "rule": f.rule, "owner": f.owner, "value": "" if f.value is None else str(f.value),
                           "threshold": f.threshold, "re-evaluated by": f.source} for f in ev.findings],
                         width="stretch", hide_index=True)

    with col_b:
        st.markdown("#### What should we measure next?")
        st.caption("Eq. 7.3: posterior SD after an unbiased repeat with SD σₘ. Evidence selection, not treatment selection.")
        if obs in twin.state and obs in NEXT:
            s = twin.state[obs]
            opts = []
            for name, sig, qc, minutes in NEXT[obs]:
                sm = sig * (s.x if obs == "roi_volume" else 1)
                rf = 1.0 if qc == "PASS" else warn_factor
                new_sd = twin.next_measurement(obs, sm, rf)
                opts.append({"option": name, "σₘ": fmt(sm, obs), "expected QC": f"{STATUS_ICON[qc]} {qc}",
                             "minutes": minutes, "σ now": fmt(s.sd, obs), "σ after": fmt(new_sd, obs),
                             "variance gain / min": f"{(s.P - new_sd ** 2) / minutes:.2e}"})
            st.dataframe(opts, width="stretch", hide_index=True)
            rec_text = {"Cho_Cr": "**Repeat PRESS after re-shim** — not mainly for the variance gain, but because it removes the cause "
                             "of the WARN (broad line, low SNR) and a possible Cho–Cr overlap bias.",
                   "NAA_Cr": "**Repeat PRESS after re-shim** — removes the linewidth/SNR WARN; NAA/Cr shows no significant change.",
                   "roi_volume": "**Registration and segmentation audit** — a +25 % volume change in six months is implausible; "
                                 "the stated 4 % uncertainty is inconsistent with the 1.1 mm registration RMS."}
            st.success(rec_text.get(obs, "Repeat with a PASS-quality protocol."))
        st.markdown("#### Competing explanations")
        expl = {"Cho_Cr": ["true chemistry change", "broad line / low SNR → fit bias (CRLB is a lower bound)",
                           "voxel shift or chemical-shift displacement", "tissue-fraction change in the VOI"],
                "NAA_Cr": ["measurement variation (z = −0.9: within noise)", "linewidth / SNR", "voxel shift"],
                "roi_volume": ["mask propagated through an imperfect registration", "segmentation version",
                               "geometric distortion", "true morphology (implausible: +25 % in 6 months)"]}
        for x in expl.get(obs, ["true change", "measurement variation", "processing variation", "model discrepancy"]):
            st.markdown(f"- {x}")
        st.markdown("#### The scanner on the scan date")
        if rec is None:
            st.caption("No scanner twin attached (sidebar). The machine can be neither blamed nor ruled out.")
        else:
            wk = rec.week_of(p)
            stt = rec.state(wk, p.get("modality", "MRS")) if wk is not None else None
            if stt:
                st.dataframe([{"": STATUS_ICON[q["status"]], "scanner part": q["label"], "twin estimate": f"{q['x']:.2f} ± {q['sd']:.2f} {q['unit']}",
                               "limit source": LNG.PHANTOM_SOURCE[q["obs"]]} for q in stt["parts"]], width="stretch", hide_index=True)
            st.caption(scan_f.reason if scan_f else "")
        st.caption(f"Limit: {p.get('interpretation_limit', 'educational synthetic data')}")

# ---------- ③ provenance ----------
with tab_prov:
    st.markdown("**Why did the twin change?** — from source file to twin variable")
    src = p.get("source_object", "?")
    f_src = {f.rule: f for f in ev.findings if f.rule in ("source_in_manifest", "source_sha256", "source_bytes")}
    integ = "FAIL" if any(f.status == "FAIL" for f in f_src.values()) else "PASS"
    acts = (p.get("provenance") or {}).get("activities", [])
    proc = p.get("processing") or {}
    acq = p.get("acquisition_context") or {}

    def q(s):
        return str(s).replace('"', "'")

    dot = ['digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];']
    dot.append(f'src [label="source file\\n{q(Path(src).name)}\\nSHA-256 {"✓" if integ == "PASS" else "✗"}", fillcolor="{STATUS_COLOR[integ]}33"];')
    dot.append(f'ctx [label="acquisition\\n{q(acq.get("sequence", p.get("modality")))} · {acq.get("field_strength_T", "?")} T\\n'
               f'calibration {q(p.get("calibration_id", "–"))}\\nsystem QC {q(p.get("system_qc_id", "–"))}", fillcolor="#eeeeee"];')
    prev = "src"
    for i, a in enumerate(acts):
        dot.append(f'a{i} [label="{q(a)}", shape=ellipse, fillcolor="#f5f5f5"]; {prev} -> a{i};')
        prev = f"a{i}"
    dot.append(f'pk [label="evidence packet\\n{q(p.get("packet_id"))}\\n{q(obs)} = {p.get("value")} ± {(p.get("uncertainty") or {}).get("value")}\\n'
               f'{q(proc.get("pipeline", ""))} v{q(proc.get("version", ""))}", fillcolor="#fff6e5"]; {prev} -> pk; ctx -> pk;')
    dot.append(f'gate [label="evidence gate\\n{ev.status} → {e.action}\\n{len([f for f in ev.findings if f.status != "PASS"])} non-PASS findings\\n'
               f'{q(p.get("thresholds_ref", ""))}", fillcolor="{STATUS_COLOR[ev.status]}44"]; pk -> gate;')
    if e.kind == "update":
        dot.append(f'tw [label="twin variable {q(obs)}\\n{fmt(e.prior["x"], obs)} → {fmt(e.posterior["x"], obs)}\\n'
                   f'σ {fmt(e.prior["sd"], obs)} → {fmt(e.posterior["sd"], obs)}\\nK = {e.K:.3f}", fillcolor="#dbe8f7"]; gate -> tw;')
    else:
        dot.append(f'tw [label="twin variable {q(obs)}\\nUNCHANGED\\nrejection logged", fillcolor="#eeeeee"]; gate -> tw [style=dashed, label="blocked"];')
    if base:
        dot.append(f'b [label="baseline packet\\n{q(base[0]["packet_id"])}\\n{fmt(base[2].posterior["x"], obs)} ± {fmt(base[2].posterior["sd"], obs)}", '
                   f'fillcolor="#eeeeee"]; b -> tw [label="prior + drift Q"];')
    dot.append("}")
    st.graphviz_chart("\n".join(dot), width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Integrity**")
        for f in f_src.values():
            st.markdown(f"{STATUS_ICON[f.status]} `{f.rule}` — {f.reason}")
        st.markdown(f"`{src}`")
        if p.get("source_sha256"):
            st.code(p["source_sha256"], language=None)
    with b:
        st.markdown("**Epistemic labels**")
        st.markdown(f"- origin: `{p.get('origin')}`\n- value: `{p.get('value_epistemic')}`\n"
                    f"- decision: `{p.get('decision_epistemic')}`\n- protocol match: `{p.get('protocol_match')}`")
        st.markdown(f"**Processing:** {proc.get('fit', proc.get('pipeline', '–'))}")
    with st.expander("Raw evidence packet (never modified by the gate)"):
        st.json(p)
    with st.expander("Machine-readable update event"):
        st.json({"evaluation": ev.to_dict(), "twin_event": e.to_dict()})

# ---------- ④ timeline ----------
with tab_time:
    st.markdown("All StarterPack packets, judged in order (baseline first). The selected packet is shown in the other tabs.")
    full = [PK[n] for n in BASELINE + FOLLOWUP]
    _, frows = run_twin(full, th, schema, manifest, root, rec)
    st.dataframe([{"": STATUS_ICON[fev.status], "packet": fp["packet_id"], "t [months]": to_months(fp.get("relative_time")),
                   "observable": fp["observable"], "value": fp.get("value"), "u": (fp.get("uncertainty") or {}).get("value"),
                   "stored": fp.get("qc_status"), "re-evaluated": fev.status, "action": fe.action,
                   "K": round(fe.K, 3) if fe.K is not None else None,
                   "posterior": f"{fmt(fe.posterior['x'], fp['observable'])} ± {fmt(fe.posterior['sd'], fp['observable'])}" if fe.posterior else "unchanged"}
                  for fp, fev, fe in frows], width="stretch", hide_index=True)
    st.caption("E-t1-chobad is rejected after E-t1-cho was accepted: the Cho/Cr state keeps the accepted value.")


# ---------- ⑤ 3D evidence map ----------
@st.cache_data(show_spinner="Loading the 3D volumes …")
def _volumes(r: str):
    from twin_evidence import anatomy
    return anatomy.load_volumes(Path(r))


with tab_3d:
    try:
        vol = _volumes(str(Path(root).resolve()))
    except ImportError as err:
        st.error(str(err))
        st.stop()
    status_of = {fp["observable"]: fev.status for fp, fev, _ in run_twin([PK[n] for n in BASELINE + FOLLOWUP[:3]], th, schema, manifest, root, rec)[1][3:]}
    from types import SimpleNamespace
    if rec is not None:
        st26 = rec.state(LNG.WEEKS, "MRS")
        bad = "; ".join(f"{q['label']} {q['x']:.1f} {q['unit']}" for q in st26["bad"]) or "all parts healthy"
        bore = SimpleNamespace(status_now=st26["status"], label_text=f"{rec.label} · {bad}")
    else:
        bore = None
    st.markdown("**Where does each number come from, and what could have gone wrong?** Everything below is drawn from the "
                "StarterPack files the packets point to. Colours are the gate status. The bore colour is the scanner twin.")
    o1, o2 = st.columns(2)
    show_bore = o1.toggle("Show the scanner bore", True)
    show_fu = o2.toggle("Show the follow-up ROI mask", True)
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(views.brain_3d(vol, status_of.get("Cho_Cr", "WARN"), status_of.get("roi_volume", "WARN"),
                                       bore, show_bore and bore is not None, show_fu), width="stretch")
    sl_fig, sl = views.slices(vol, 0.0)
    with right:
        st.markdown("#### Possible causes, judged by the evidence")
        mrs_machine = who["who"] == "machine"
        causes = [
            ("MRS", "The scanner itself", "supported" if mrs_machine else "excluded" if rec else "open",
             (f"scanner twin: {CLASS_TEXT.get(rec.diagnosis['fault_class'])} (SIMULATED)" if rec else "no scanner twin attached")),
            ("MRS", "Local shim, voxel position or motion in this session", "open" if mrs_machine else "most likely",
             "linewidth 8.7 → 11.4 Hz in the subject"),
            ("MRS", "Chemical-shift displacement (NAA vs Cho voxel 2.4 mm)", "minor",
             "same in both sessions, mostly cancels in Δ"),
            ("MRS", "True Cho/Cr change", "open", "z = 2.05 borderline, d² = 3.66 not surprising"),
            ("ROI", "Mask propagated through a 1.1 mm registration", "supported", "implies u ≈ 19 %, packet claims 4 %"),
            ("ROI", "True growth of the region (+25 %)", "excluded",
             f"T1w inside the ROI unchanged: mean Δ = {sl['roi_mean_diff']:+.1f} a.u., SD {sl['roi_sd_diff']:.1f} ≈ noise {sl['expected_noise_sd']:.1f}"),
        ]
        badge = {"supported": "🟠 supported", "most likely": "🟠 most likely", "open": "⚪ open", "minor": "⚪ minor",
                 "excluded": "🟢 excluded"}
        st.dataframe([{"": c[0], "cause": c[1], "verdict": badge[c[2]], "evidence": c[3]} for c in causes],
                     width="stretch", hide_index=True)
        st.caption("The 3D view is evidence for the twin, not the twin itself. The twin is the set of numbers with uncertainty "
                   "and the rule that updates them.")
    st.markdown("#### Slice through the ROI and the MRS voxel (z ≈ 0 mm)")
    st.plotly_chart(sl_fig, width="stretch")
    st.caption(f"Inside the baseline ROI ({sl['n_vox']} voxels) the follow-up minus baseline difference has mean "
               f"{sl['roi_mean_diff']:+.1f} a.u. and SD {sl['roi_sd_diff']:.1f} a.u.; pure noise would give SD ≈ "
               f"{sl['expected_noise_sd']:.1f}. The anatomy did not change; the larger follow-up mask is a processing artifact.")
    st.markdown("#### The two spectra, recomputed from the raw NIfTI-MRS files")
    sp_fig, qc = views.spectra(Path(root))
    st.plotly_chart(sp_fig, width="stretch")
    st.dataframe([{"session": s, "SNR from file": round(q["snr"], 1), "SNR in packet": {"baseline": 31.2, "followup": 24.8}[s],
                   "FWHM from file / Hz": round(q["fwhm_hz"], 1), "FWHM in packet / Hz": {"baseline": 8.7, "followup": 11.4}[s]}
                  for s, q in qc.items()], width="stretch", hide_index=True)
    st.caption("A matching SHA-256 confirms the bytes, not the physics. Here we also re-measure the physics from the bytes: "
               "SNR exactly, FWHM within a few percent (zero-filled estimate).")

# ---------- ⑥ scanner twin ----------
with tab_scan:
    from twin_evidence import scenario_views as SVW
    if rec is None:
        st.info("Pick a scanner state in the sidebar to attach the scanner twin. With 'no scanner twin' the app reproduces the "
                "Gate 1 reference numbers exactly.")
    else:
        st.warning("SIMULATED weekly phantom record (the StarterPack has none). Every week a phantom spectrum is simulated "
                   "on this scanner and measured like real data; the same Kalman step follows each part of the machine.")
        st.markdown(f"**Scanner state:** {rec.label} · fault class found by the scanner twin: "
                    f"**{CLASS_TEXT.get(rec.diagnosis['fault_class'])}** ({rec.diagnosis['why']})")
        st.plotly_chart(SVW.phantom_chart(rec.weekly_rows(), {o: v[4] for o, v in LNG.PHANTOM_OBS.items()}), width="stretch")
        cols = st.columns(len(LNG.PHANTOM_OBS))
        for col, o in zip(cols, LNG.PHANTOM_OBS):
            x1 = rec.series[o][-1]
            col.metric(LNG.PHANTOM_OBS[o][0], f"{x1['x']:.2f} {LNG.PHANTOM_OBS[o][1]}", f"z = {rec.z[o]:+.1f} since May",
                       delta_color="off")
            w = rec.weeks_to_fail.get(o)
            col.caption((f"FAIL limit in ≈ {w:.0f} weeks · " if w is not None and w > 0 else "") + LNG.PHANTOM_SOURCE[o])
        st.markdown("#### What the scanner twin does to the subject's evidence")
        full = [PK[n] for n in BASELINE + FOLLOWUP]
        _, without = run_twin(full, th, schema, manifest, root, None)
        _, with_ = run_twin(full, th, schema, manifest, root, rec)
        st.dataframe([{"packet": a[0]["packet_id"], "observable": a[0]["observable"],
                       "without scanner twin": f"{STATUS_ICON[a[1].status]} {a[2].action}",
                       "with scanner twin": f"{STATUS_ICON[b[1].status]} {b[2].action}",
                       "scanner finding": next((f"{f.status}: {f.reason[:90]}…" for f in b[1].findings if f.rule == "F11_scanner_twin"), "")}
                      for a, b in zip(without, with_)], width="stretch", hide_index=True)
        st.markdown("""
- **A phantom never changes**, so a trend in the phantom is the machine. That separates *the subject changed* from *the instrument changed*.
- **The twins talk**: the scanner twin's state on the scan date is a gate finding (F11) on every subject packet. FAIL blocks the
  update, WARN down-weights it. Only the parts the scan depends on count (geometry and SNR for volumes; all parts for spectroscopy).
- **Service before images degrade**: the forecast says when each part will cross its FAIL limit.
""")

# ---------- ⑦ fault scenarios (SIMULATED, known truth) ----------
@st.cache_data(show_spinner=False)
def _scenario_data(folder: str):
    from twin_evidence import scenario_views as sv
    return sv.load(Path(folder))


@st.cache_data(show_spinner="Simulating 64 transients and fitting …", max_entries=64)
def _one_scan(key: str, seed: int, r: str):
    from twin_evidence import mrs_sim as S, scenarios as SC
    sim = S.simulate(SC.BY_KEY[key].acq, Path(r), seed)
    clean = S.simulate(SC.BY_KEY["clean"].acq, Path(r), seed)
    m = SC.measure(sim)
    g = SC.gate(m)
    return {"m": m, "gate": g, "fid": sim.average(), "clean": clean.average(), "aligned": S.align_and_average(sim.transients),
            "truth": {k: v for k, v in sim.truth.items() if k != "tissue"}, "tissue": sim.truth["tissue"]}


with tab_sc:
    from twin_evidence import scenarios as SCN, scenario_views as SV
    sd_dir = Path(__file__).resolve().parent / "scenario_data"
    data = _scenario_data(str(sd_dir))
    st.warning("SIMULATED, with known truth. Each fault is injected into 64 PRESS transients built with the StarterPack "
               "signal model, then processed by the same fit, QC and gate as real data. Assumptions: `scenario_data/README.md`.")
    if data is None:
        st.error("No `scenario_data/` yet. Run `python scripts/generate_scenarios.py ../MRI_Hackathon_DataScience_StarterPack_v0.3`.")
    else:
        s1, s2, s3, s4, s5, s6 = st.tabs(["Single scan", "All 11 scenarios", "Real November scan", "Scanner over time",
                                          "Threshold check", "ScanTwin catalogue"])

        with s1:
            c1, c2 = st.columns([2, 1])
            key = c1.selectbox("Fault scenario", [s.key for s in SCN.CATALOGUE], index=2,
                               format_func=lambda k: f"{SCN.BY_KEY[k].title}  ·  blame: {SCN.BY_KEY[k].blame}")
            seed = c2.number_input("Noise seed (another scan)", 0, 999, 7, 1)
            from twin_evidence.gates import context_factors, context_advice, STANDING_LIMITATIONS
            ctx = st.multiselect("Scan context (changes only the warnings and fix advice, never the limits)",
                                 [r["Context factor"] for r in context_factors()], default=[])
            sc = SCN.BY_KEY[key]
            res = _one_scan(key, int(seed), str(Path(root).resolve()))
            m, g = res["m"], res["gate"]
            k = st.columns(4)
            k[0].metric("Gate", f"{STATUS_ICON[g['status']]} {g['status']}")
            k[1].metric("Cho/Cr measured", f"{m['cho_cr']:.3f} ± {m['cho_cr_crlb']:.3f}")
            k[2].metric("Cho/Cr truth (tissue)", f"{res['truth']['Cho_Cr_tissue']:.3f}",
                        f"error {m['cho_cr'] - res['truth']['Cho_Cr_tissue']:+.3f}", delta_color="off")
            k[3].metric("Who is to blame", sc.blame, sc.kind, delta_color="off")
            left, right = st.columns([3, 2])
            with left:
                show_al = m["fwhm_gain_aligned"] > 1.0
                st.plotly_chart(SV.spectra_compare(res["clean"], res["fid"], res["aligned"] if show_al else None,
                                                   f"{sc.title}: spectrum vs a clean scan (same noise seed)"), width="stretch")
                st.caption(f"Expected signature: {sc.expected}.  Source: {sc.source}.")
            with right:
                from twin_evidence.gates import supported_range
                det = g["detail"]
                st.markdown("**Scanner health: ScanTwin gates**")
                fmtv = lambda v: "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
                st.dataframe([{"": STATUS_ICON.get(x["status"], "⚪"), "ID": x["id"], "gate": x["label"],
                               "value": f"{fmtv(x['value'])} {x['unit']}".strip(), "PASS if": x["pass_if"]}
                              for x in det["gates"] if x["applies"]], width="stretch", hide_index=True, height=360)
                dec = det["deciding"]
                why = ", ".join(f"{d['id']} {d['label']}" for d in dec) if dec else "all gates PASS"
                {"PASS": st.success, "WARN": st.warning, "FAIL": st.error}[g["status"]](
                    f"**Scan verdict: {g['status']} → {g['verdict']}**  \nDecided by: {why}")
                if g["status"] != "FAIL":
                    lo, hi = supported_range(m["cho_cr"], math.hypot(m["cho_cr_crlb"], 0.045), g["status"])
                    st.markdown(f"**What this result can tell you:** Cho/Cr {m['cho_cr']:.2f}, consistent with about "
                                f"**{lo:.2f}–{hi:.2f}** at this scan quality.")
                else:
                    st.markdown("**What this result can tell you:** nothing. The value is not used and the twin does not change.")
                if dec:
                    st.markdown("**Fix for the next scan** (from the catalogue)")
                    for d in dec:
                        st.markdown(f"- **{d['id']}**: {d['fix']}")
                if ctx:
                    adv = context_advice(ctx, det)
                    if adv:
                        st.markdown("**Scan context**")
                        st.dataframe([{"": STATUS_ICON.get(a["gate_status"], "⚪"), "context": a["factor"].split(" (")[0],
                                       "raises": f"{a['id']} {a['failure']}", "reading": a["reading"], "advice": a["advice"]}
                                      for a in adv], width="stretch", hide_index=True)
                if show_al:
                    st.caption(f"Aligning the transients narrows the line {m['fwhm_hz']:.1f} → {m['fwhm_aligned_hz']:.1f} Hz: "
                               "a frequency problem (drift or motion), not the shim.")
                with st.expander("Likely fault (fault model trained on simulations)"):
                    post = data["model"].posterior(m, SV.QUALITY_ALL)
                    st.plotly_chart(SV.posterior_bar(post, 5), width="stretch")
                    blame = SCN.blame_of(post)
                    st.markdown("Most likely cause: " + " · ".join(f"**{b}** {p:.0%}" for b, p in list(blame.items())[:3]))
                    st.caption("A statement about the measurement, not about the person.")
                st.caption("Standing limitations (never change the verdict): " +
                           " · ".join(f"{x['id']}: {x['label']}" for x in STANDING_LIMITATIONS))
            tf = res["tissue"]
            st.caption(f"Voxel tissue (truth): WM {tf['WM']:.0%}, GM {tf['GM']:.0%}, CSF {tf['CSF']:.0%}, outside {tf['outside']:.0%}. "
                       f"Signal scale {res['truth']['signal_scale']:.2f}.")

        with s2:
            st.markdown("**What each fault does to Cho/Cr** (60 simulated scans each). Green = the gate mostly passes it, "
                        "amber = the gate mostly warns. Black bar = the true tissue value.")
            st.plotly_chart(SV.bias_chart(data["summary"]), width="stretch")
            st.dataframe([{"scenario": SCN.BY_KEY[r["scenario"]].title, "blame": r["blame"], "type": r["kind"],
                           "SNR": round(float(r["snr_mean"]), 1), "FWHM / Hz": round(float(r["fwhm_hz_mean"]), 1),
                           "FWHM aligned": round(float(r["fwhm_aligned_hz_mean"]), 1), "water": round(float(r["water_ratio_mean"]), 2),
                           "drift / Hz": round(float(r["drift_abs_hz_mean"]), 1), "outliers": f"{float(r['outlier_frac_mean']):.0%}",
                           "Cho/Cr": f"{float(r['cho_cr_mean']):.3f} ± {float(r['cho_cr_sd']):.3f}",
                           "truth": round(float(r["cho_cr_truth_tissue"]), 3),
                           "PASS": f"{float(r['gate_PASS']):.0%}", "WARN": f"{float(r['gate_WARN']):.0%}"} for r in data["summary"]],
                         width="stretch", hide_index=True)
            acc = data["run_info"]["accuracy"]
            a1, a2 = st.columns(2)
            a1.metric("Fault model, all metrics (held-out)", f"{acc['all_metrics']['grouped']:.0%} group",
                      f"{acc['all_metrics']['exact']:.0%} exact scenario", delta_color="off")
            a2.metric("Averaged spectrum only (like the StarterPack)", f"{acc['averaged_only']['grouped']:.0%} group",
                      f"{acc['averaged_only']['exact']:.0%} exact scenario", delta_color="off")
            st.info("Spectrally silent: voxel misplacement, a gradient error and a true change pass the gate with a clean "
                    "spectrum, yet Cho/Cr moves. Only the scanner twin (phantom geometry) and tissue fractions tell them apart.")

        with s3:
            st.markdown("The fault model was trained **only on simulations**. Here it reads the **real** StarterPack spectra "
                        "(averaged files: drift and motion metrics are not available).")
            r1, r2 = st.columns(2)
            for col, sess in ((r1, "baseline"), (r2, "followup")):
                d = data["real"][sess]
                mm = d["metrics"]
                col.markdown(f"#### {'May (baseline)' if sess == 'baseline' else 'November (follow-up)'}")
                col.caption(f"SNR {mm['snr']:.1f} · linewidth {mm['fwhm_hz']:.1f} Hz · residual water {mm['water_ratio']:.2f} · "
                            f"fitted Cho/Cr {d['cho_cr_fit'][0]:.3f} ± {d['cho_cr_fit'][1]:.3f}")
                col.plotly_chart(SV.posterior_bar(d["top_scenarios"], 5), width="stretch")
                col.markdown("Blame: " + " · ".join(f"**{b}** {p:.0%}" for b, p in list(d["blame"].items())[:3]))
            st.success("November looks like a **shim problem**. Whether it is the scanner's shim or this session's local shim "
                       "cannot be decided from one spectrum: the scanner twin's phantom record decides. Next: re-shim, repeat "
                       "PRESS and keep the individual transients.")

        with s4:
            names = {t: v["summary"]["title"] + (" + true Cho change" if t.endswith("_biology") else "") for t, v in data["long"].items()}
            tag = st.radio("Scanner state from week 14 to 26", list(data["long"]), index=1, horizontal=True, format_func=names.get)
            L = data["long"][tag]
            row, chain = L["summary"], L["chain"]
            from twin_evidence.longitudinal import PHANTOM_OBS
            st.plotly_chart(SV.phantom_chart(L["weekly"], {o: v[4] for o, v in PHANTOM_OBS.items()}), width="stretch")
            b0, b1 = chain["brain"]
            v1, v2, v3 = st.columns(3)
            v1.metric("Scanner twin: fault class", row["diagnosed_class"], "correct" if row["correct"] == "True" else "WRONG",
                      delta_color="off")
            v2.metric("Brain follow-up gate", f"{STATUS_ICON[b1['status']]} {b1['status']}", b1["action"], delta_color="off")
            v3.metric("Brain Cho/Cr (truth)", f"{b1['value']:.3f}", f"truth {b1['truth_tissue']:.3f}", delta_color="off")
            box = st.success if row["verdict"].startswith("CLEAN") else st.warning if row["verdict"].startswith(("MACHINE", "SESSION")) else st.error
            box(row["verdict"])
            st.caption(f"Why: {chain['diagnosis']['why']}. Brain reasons: " + ("; ".join(b1["reasons"]) if b1["reasons"] else "none") +
                       ". The brain sessions are schema-v1.0 packets with their own NIfTI file and SHA-256, judged by the same gate.")

        with s5:
            tc = sd_dir / "threshold_check.csv"
            if not tc.exists():
                st.info("Run `python scripts/sweep_faults.py ../MRI_Hackathon_DataScience_StarterPack_v0.3` to check the thresholds.")
            else:
                import csv as _csv
                checks = list(_csv.DictReader(open(tc, encoding="utf-8")))
                sweeps = list(_csv.DictReader(open(sd_dir / "sweeps.csv", encoding="utf-8")))
                st.markdown("Each built MRS gate of the catalogue, checked with the simulator: the fault is added at increasing "
                            "strength, and we record when Cho/Cr drifts from the truth by more than the normal scatter "
                            f"(1 SD = {float(checks[0]['baseline_sd']):.3f}).")
                st.dataframe([{"": "⚠️" if c["verdict"].startswith("TOO") else "✅", "ID": c["gate"], "gate": c["label"],
                               "catalogue WARN": c["catalogue_warn"], "catalogue FAIL": c["catalogue_fail"],
                               "bias reaches 1 SD at": c["sim_metric_at_1sd_bias"] or "never",
                               "bias reaches 2 SD at": c["sim_metric_at_2sd_bias"] or "never", "verdict": c["verdict"]}
                              for c in checks], width="stretch", hide_index=True)
                gid = st.selectbox("Show the sweep for", [c["gate"] for c in checks],
                                   format_func=lambda i: f"{i} {next(c['label'] for c in checks if c['gate'] == i)}")
                st.plotly_chart(SV.sweep_chart([r for r in sweeps if r["gate"] == gid],
                                               next(c for c in checks if c["gate"] == gid)), width="stretch")

        with s6:
            from twin_evidence.gates import catalogue as _cat
            cat = list(_cat().values())
            if not cat:
                st.info("Run `python scripts/import_catalogue.py`.")
            else:
                f1, f2 = st.columns(2)
                types = sorted({c["Scan type"] for c in cat})
                tsel = f1.multiselect("Scan type", types, default=[t for t in types if "MRS" in t or t.startswith("All")])
                tiers = f2.multiselect("Demo tier", ["Built", "Designed", "Future"], default=["Built", "Designed"])
                rows = [c for c in cat if c["Scan type"] in tsel and c["Demo tier"] in tiers]
                st.caption(f"{len(rows)} of {len(cat)} failure modes · source: docs/scantwin (team bioengineer). "
                           "Built = simulated and gated in this app.")
                st.dataframe([{"ID": c["ID"], "tier": c["Demo tier"], "scan type": c["Scan type"], "failure": c["Failure mode"],
                               "effect": c["Effect on the result"], "metric": c["QC metric"], "PASS": c["PASS if"],
                               "WARN": c["WARN if"], "FAIL": c["FAIL if"], "fix": c["Fix for the next scan"],
                               "source": c["Threshold source"]} for c in rows], width="stretch", hide_index=True, height=480)
