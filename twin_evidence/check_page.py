"""ScanTwin "Check a scan" page (Streamlit): one scan in, a verdict out.

Layout follows docs/scantwin PDF ("What the user sees"): Scan · Scanner health · Verdict · What this result can tell
you · Change since last scan · Fix for the next scan · Trace. The page never says "diagnosis": it judges the
measurement, not the person.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from . import ingest as I
from . import mrs_fit as F
from .gates import context_advice, context_factors, supported_range, STANDING_LIMITATIONS
from .twin import Twin
from .validate import evaluate

ICON = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴", "N/A": "⚪"}
REPO = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "Real 7 T brain scan (Zenodo, CC BY 4.0)": REPO / "external_data/zenodo_5085449_example_02/derivatives/sub-ext01_ses-baseline_transients_svs.nii.gz",
    "StarterPack May scan (3 T, synthetic)": "SP:baseline",
    "StarterPack November scan (3 T, synthetic)": "SP:followup",
}
FOLLOWUPS = {
    "none": ("No follow-up", None, None),
    "repeat": ("Repeat scan · healthy scanner · unchanged subject", "none", 0.0),
    "cho_up": ("Subject changed · choline up · healthy scanner", "cho_up", None),
    "shim": ("Scanner fault · poor shim", "shim", None),
    "drift": ("Scanner fault · frequency drift", "drift", None),
    "noise": ("Scanner fault · weak receive coil", "noise", None),
    "water": ("Scanner fault · water suppression failed", "water", 5.0),
}
GATE_HELP = {"F12": "line width (shim)", "F13": "signal-to-noise", "F14": "fit precision", "F15": "residual water",
             "F08": "frequency drift during the scan", "F23": "motion (rejected transients)", "F20": "voxel position vs last scan",
             "F21": "tissue mix change", "F35": "protocol vs last scan", "F19": "chemical-shift displacement change",
             "F61": "transmitter calibration"}


def _strength_default(fault: str, nu0: float, snr: float = 31.0) -> float:
    """Default fault strengths that cross the WARN limits at this field and this scan's own SNR."""
    k = nu0 / I.REF_NU0
    return {"shim": round(9 * k, 1), "drift": round(10 * k, 1), "noise": round(max(1.5, snr / 18), 1),
            "cho_up": 70.0}.get(fault, 0.0)


def _spectrum_fig(fids: list, nu0: float, dwell: float) -> go.Figure:
    fig = go.Figure()
    fig.add_vrect(x0=1.85, x1=3.35, fillcolor="#2b6cb0", opacity=0.06, line_width=0,
                  annotation_text="fit window", annotation_position="top left")
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


def _gate_rows(g: dict, m: dict, nu0: float) -> list[dict]:
    raw = {"F12": m.get("fwhm_hz"), "F08": m.get("drift_abs_hz")}
    rows = []
    for x in g["gates"]:
        if not x["applies"]:
            continue
        v = x["value"]
        if v is None:
            val = "N/A"
        elif x["id"] in raw and raw[x["id"]] is not None and abs(nu0 - I.REF_NU0) > 1:
            val = f"{raw[x['id']]:.1f} Hz at {nu0 / I.GAMMA_MHZ_PER_T:.0f} T (= {v:.1f} Hz at 3 T)"
        else:
            val = f"{v:.2f} {x['unit']}".strip() if isinstance(v, float) else str(v)
        rows.append({"": ICON.get(x["status"], "⚪"), "gate": f"{x['id']} {GATE_HELP.get(x['id'], x['label'])}",
                     "measured": val, "PASS if": x["pass_if"],
                     "note": x.get("reason") or ""})
    return rows


def _packetise(ing, m, g, tmp: Path, *, tag: str, ses: str, months: float, origin: str, sysqc: str | None, desc: str = ""):
    rel = f"{tag}.nii.gz"
    sha = I.write_derivative(ing, tmp / rel)
    pid = ("E-t0-" if months == 0 else "E-t1-") + f"check_{tag}"
    p = I.build_packet(ing, m, g, packet_id=pid, subject_id="sub-check", session_id=ses, months=months, rel_path=rel,
                       sha=sha, origin=origin, system_qc_id=sysqc,
                       limit_note=desc if origin == "SIMULATED" else "")
    return p, {rel: sha}


def render(st, *, starterpack_root: Path, thresholds: dict, schema: dict, packets: dict, manifest: dict):
    st.title("ScanTwin · Check a scan")
    st.markdown("**Is this scan good enough to say anything at all?** Upload a raw single-voxel spectrum "
                "(NIfTI-MRS `.nii.gz`) or an evidence packet (`.json`), or pick an example. ScanTwin judges the "
                "*measurement*, never the person.")

    c1, c2 = st.columns([3, 2])
    up = c1.file_uploader("Raw spectrum (.nii.gz) or evidence packet (.json)", type=["gz", "nii", "json"], key="chk_upload")
    ex = c2.selectbox("…or an example", list(EXAMPLES), index=0, key="chk_example")
    ctx = st.multiselect("Scan context (changes only warnings and fix advice, never the limits)",
                         [r["Context factor"] for r in context_factors()], key="chk_ctx")

    if up is not None and up.name.endswith(".json"):
        _render_packet(st, json.loads(up.getvalue()), starterpack_root, thresholds, schema, packets, manifest)
        return

    # ---------------------------------------------------------------- load the spectrum
    if up is not None:
        cache_key = ("upload", up.name, up.size)
        src_label = up.name
    else:
        cache_key = ("example", ex)
        src_label = ex
    cache = st.session_state.setdefault("chk_cache", {})
    if cache_key not in cache:
        with st.spinner("Reading, de-identifying, combining coils, fitting …"):
            try:
                if up is not None:
                    ing = I.read_bytes(up.getvalue(), up.name)
                else:
                    target = EXAMPLES[ex]
                    if isinstance(target, str) and target.startswith("SP:"):
                        sess = target[3:]
                        ing = I.read(starterpack_root / f"04_MRS_BIDS/sub-01/ses-{sess}/mrs/sub-01_ses-{sess}_acq-press_nuc-1H_voi-frontal_svs.nii.gz")
                    else:
                        ing = I.read(target)
            except Exception as err:   # never crash on a strange file: explain instead
                st.error(f"This file could not be read as a single-voxel NIfTI-MRS spectrum: {err}")
                return
            cache[cache_key] = (ing, I.measure(ing))
    ing, m = cache[cache_key]
    g = I.gate(m, ing.nu0)

    # ---------------------------------------------------------------- follow-up
    st.markdown("##### Simulate a follow-up on this scanner")
    f1, f2 = st.columns([3, 2])
    fkey = f1.selectbox("What happens before the next scan", list(FOLLOWUPS), index=0, format_func=lambda k: FOLLOWUPS[k][0],
                        key="chk_follow")
    _, fault, strength = FOLLOWUPS[fkey]
    if fault in ("shim", "drift", "noise", "cho_up"):
        unit = {"shim": "Hz extra broadening", "drift": "Hz drift over the scan", "noise": "× noise",
                "cho_up": "% more choline"}[fault]
        dflt = _strength_default(fault, ing.nu0, m["snr"])
        top = 100.0 if fault == "cho_up" else float(3 * dflt)
        strength = f2.slider(f"{'Size of the change' if fault == 'cho_up' else 'Fault strength'} ({unit})", 0.0, top,
                             float(dflt), key=f"chk_str_{fault}")
        if fault == "cho_up":
            strength = strength / 100.0
    site_qc = f2.radio("Site QC record for this scanner", ["missing: assume it, flag it", "missing: strict (reject)"],
                       index=0, key="chk_sysqc", horizontal=False) if fkey != "none" else "missing: assume it, flag it"
    sysqc = "ASSUMED:external-dataset" if site_qc.startswith("missing: assume") else None

    follow = None
    if fault is not None:
        fk = ("follow", cache_key, fkey, strength)
        if fk not in cache:
            with st.spinner("Simulating the follow-up from the real transients …"):
                fi = I.degrade(ing, fault, strength, seed=11)
                cache[fk] = (fi, I.measure(fi))
        follow = cache[fk]

    # ---------------------------------------------------------------- panels
    field_T = ing.nu0 / I.GAMMA_MHZ_PER_T
    meta = ing.meta
    st.markdown("---")
    a, b, c, d = st.columns(4)
    a.metric("Scan", meta.get("ProtocolName") or meta.get("SequenceName") or "single-voxel MRS",
             f"{meta.get('Manufacturer', 'unknown vendor')} · {field_T:.1f} T", delta_color="off")
    b.metric("Echo / repetition time", f"{1000 * (meta.get('EchoTime') or 0):.0f} / {1000 * (meta.get('RepetitionTime') or 0):.0f} ms"
             if meta.get("EchoTime") else "not recorded", f"{ing.transients.shape[0]} transient(s)", delta_color="off")
    b2 = g["status"]
    c.metric("Scan verdict", f"{ICON[b2]} {b2}", {"PASS": "use normally", "WARN": "use with doubled uncertainty",
                                                    "FAIL": "do not use"}[b2], delta_color="off")
    u = math.hypot(m["cho_cr_crlb"], 0.045)
    d.metric("Cho/Cr", f"{m['cho_cr']:.3f} ± {u:.3f}", f"NAA/Cr {m['naa_cr']:.2f}", delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        fids = [("this scan", ing.fid, "#2b6cb0")]
        if follow is not None:
            fids.append(("simulated follow-up", follow[0].fid, "#c8642a"))
        st.plotly_chart(_spectrum_fig(fids, ing.nu0, ing.dwell), width="stretch")
        st.markdown("**Scanner health** (ScanTwin catalogue v1.1)")
        st.dataframe(_gate_rows(g, m, ing.nu0), width="stretch", hide_index=True)
    with right:
        dec = g["deciding"]
        box = {"PASS": st.success, "WARN": st.warning, "FAIL": st.error}[g["status"]]
        box(f"**Verdict: {g['status']}** · " + (
            "decided by " + ", ".join(f"{x['id']} {GATE_HELP.get(x['id'], x['label'])}" for x in dec) if dec
            else "every quality gate passes"))
        if g["status"] != "FAIL":
            lo, hi = supported_range(m["cho_cr"], u, g["status"])
            st.markdown(f"**What this result can tell you:** Cho/Cr {m['cho_cr']:.2f}, consistent with about "
                        f"**{lo:.2f}–{hi:.2f}** at this scan quality.")
        else:
            st.markdown("**What this result can tell you:** nothing. The value is not used.")
        if dec:
            st.markdown("**Fix for the next scan**")
            for x in dec:
                st.markdown(f"- **{x['id']}** · {x['fix']}")
        if ctx:
            adv = context_advice(ctx, g)
            if adv:
                st.markdown("**Scan context**")
                for x in adv:
                    st.markdown(f"- {ICON.get(x['gate_status'], '⚪')} {x['id']} {x['failure']}: {x['reading']}. {x['advice']}")
        if follow is not None:
            _render_change(st, ing, m, g, follow, fkey, strength, thresholds, schema, sysqc)

    with st.expander("Trace: where every number came from"):
        st.markdown(f"**Source:** `{src_label}` · SHA-256 `{ing.source_sha256}`")
        if ing.removed:
            st.warning(f"De-identified: removed {len(ing.removed)} identifying field(s) from the metadata: "
                       f"{', '.join(ing.removed)} (catalogue F59). Their values are never shown or stored.")
        else:
            st.caption("No identifying metadata found (F59).")
        for s_ in ing.steps:
            if not s_.startswith("de-identify"):
                st.markdown(f"- {s_}")
        st.markdown(f"- QC on the averaged spectrum; Hz limits compared at the 3 T-equivalent scale (× {I.REF_NU0} / {ing.nu0:.2f})")
        st.markdown("- windowed fit 1.85–3.35 ppm (NAA, Cr, Cho + linear baseline); reads Cho/Cr ~3 % low on simulated truth")
        st.caption("Standing limitations: " + " · ".join(f"{x['id']}: {x['label']}" for x in STANDING_LIMITATIONS))
        with tempfile.TemporaryDirectory() as tmp:
            p, _ = _packetise(ing, m, g, Path(tmp), tag="scan", ses="ses-baseline", months=0.0,
                              origin="MEASURED" if "synthetic" not in src_label else "SIMULATED", sysqc=sysqc)
        st.download_button("Download the evidence packet (JSON)", json.dumps(p, indent=2), file_name=f"{p['packet_id']}.json",
                           mime="application/json", key="chk_dl")
    st.caption("A verdict about the measurement only. No diagnosis, no treatment advice, no clinical meaning.")


def _render_change(st, ing, m, g, follow, fkey, strength, thresholds, schema, sysqc):
    fi, fm = follow
    fg = I.gate(fm, fi.nu0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p0, man0 = _packetise(ing, m, g, tmp, tag="baseline", ses="ses-baseline", months=0.0, origin="MEASURED", sysqc=sysqc)
        p1, man1 = _packetise(fi, fm, fg, tmp, tag=fkey, ses="ses-followup", months=6.0, origin="SIMULATED", sysqc=sysqc,
                              desc=f"SIMULATED follow-up: {FOLLOWUPS[fkey][0]}")
        man = {**man0, **man1}
        tw = Twin()
        ev0 = evaluate(p0, thresholds, schema, man, tmp)
        tw.ingest(p0, ev0)
        ev1 = evaluate(p1, thresholds, schema, man, tmp)
        e1 = tw.ingest(p1, ev1)
    st.markdown("---")
    st.markdown(f"**Change since last scan** · follow-up: {FOLLOWUPS[fkey][0]}"
                + (f" (+{100 * strength:.0f} %)" if fkey == "cho_up" else
                   f" (strength {strength:g})" if fkey in ("shim", "drift", "noise") else ""))
    k1, k2 = st.columns(2)
    k1.metric("Follow-up verdict", f"{ICON[ev1.status]} {ev1.status}", e1.action, delta_color="off")
    k2.metric("Cho/Cr follow-up", f"{fm['cho_cr']:.3f}", f"{fm['cho_cr'] - m['cho_cr']:+.3f} vs this scan", delta_color="off")
    bad = [x["id"] for x in fg["deciding"] if x["id"] not in [y["id"] for y in g["deciding"]]]
    if ev1.status == "FAIL":
        st.error("**It's the machine.** The follow-up fails a quality gate (" + ", ".join(bad or [x["id"] for x in fg["deciding"]])
                 + "). The result is not used and the subject twin does not change. Fix the scanner and repeat.")
    elif bad:
        st.warning(f"**Probably the machine.** Scan quality fell ({', '.join(f'{b} {GATE_HELP.get(b, b)}' for b in bad)}). "
                   f"The twin uses the value with doubled uncertainty (K = {e1.K:.2f}) and moves only to "
                   f"{e1.posterior['x']:.3f} ± {e1.posterior['sd']:.3f}. Do not read this as a change in the subject.")
    else:
        # catalogue F60: |z| < 1.96 no change claimed; 1.96-3 with any WARN inconclusive; otherwise a real change
        z = e1.z_change or 0.0
        any_warn = "WARN" in (ev1.status, g["status"])
        if abs(z) < 1.96:
            st.info(f"**Not distinguishable from measurement error** (z = {z:+.2f}, F60). Same scan quality; the difference "
                    f"is within what this scan can resolve (±{1.96 * math.hypot(math.hypot(m['cho_cr_crlb'], 0.045), math.hypot(fm['cho_cr_crlb'], 0.045)):.3f}). "
                    "No change is claimed about the subject.")
        elif abs(z) < 3 and any_warn:
            st.warning(f"**Inconclusive** (z = {z:+.2f}, F60). Beyond noise, but a scan carries a WARN. Repeat at PASS quality "
                       "before claiming a change.")
        else:
            st.success(f"**The subject changed.** Scan quality is as good as before and the change is beyond measurement error "
                       f"(z = {z:+.2f}, F60). The twin moves to {e1.posterior['x']:.3f} ± {e1.posterior['sd']:.3f}. "
                       "A statement about the measurement: confirm with a repeat before anyone acts on it.")
    reasons = [f"{f.rule}: {f.reason}" for f in ev1.findings if f.status != "PASS"]
    if reasons:
        st.caption("Follow-up findings: " + " · ".join(reasons))


def _render_packet(st, p: dict, root: Path, thresholds: dict, schema: dict, packets: dict, manifest: dict):
    """An evidence packet (e.g. an unseen Gate 3 packet): the same verdict panel, with the StarterPack baseline as prior."""
    ev = evaluate(p, thresholds, schema, manifest, root)
    obs = p.get("observable")
    base = next((q for k, q in packets.items() if k.startswith("E-t0") and q.get("observable") == obs), None)
    tw = Twin()
    if base is not None:
        tw.ingest(base, evaluate(base, thresholds, schema, manifest, root))
    e = tw.ingest(p, ev)
    a, b, c = st.columns(3)
    a.metric("Packet", str(p.get("packet_id")), str(obs), delta_color="off")
    b.metric("Verdict", f"{ICON[ev.status]} {ev.status}", e.action, delta_color="off")
    if e.kind == "update":
        c.metric("Twin", f"{e.posterior['x']:.3f} ± {e.posterior['sd']:.3f}", f"K = {e.K:.2f}", delta_color="off")
    bad = [f for f in ev.findings if f.status != "PASS"]
    box = {"PASS": st.success, "WARN": st.warning, "FAIL": st.error}[ev.status]
    box(f"**Verdict: {ev.status} → {e.action}**" + (" · " + "; ".join(f"{f.rule}: {f.reason}" for f in bad[:4]) if bad else ""))
    if bad:
        st.dataframe([{"": ICON.get(f.status, "⚪"), "rule": f.rule, "owner": f.owner, "reason": f.reason} for f in bad],
                     width="stretch", hide_index=True)
    st.caption("The packet is never modified; every finding names its rule and owner track.")
