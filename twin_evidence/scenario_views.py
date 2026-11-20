"""Figures and loaders for the fault-scenario tab (all data SIMULATED, built by scripts/generate_scenarios.py)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import mrs_fit as F
from . import scenarios as SC

STATUS_COLOR = {"PASS": "#2e9e5b", "WARN": "#e0a100", "FAIL": "#d64545"}
BLAME_COLOR = {"machine": "#c8642a", "patient": "#2b6cb0", "operator": "#7a4fbf", "none": "#6f7d89"}
AVERAGED_ONLY = ["snr", "fwhm_hz", "water_ratio", "te_ms"]
QUALITY_ALL = ["snr", "fwhm_hz", "water_ratio", "te_ms", "drift_abs_hz", "outlier_frac", "fwhm_gain_aligned"]


def load(folder: Path) -> dict | None:
    folder = Path(folder)
    if not (folder / "summary.csv").exists():
        return None
    rd = lambda n: list(csv.DictReader(open(folder / n, encoding="utf-8")))
    fm = json.load(open(folder / "fault_model.json", encoding="utf-8"))
    model = SC.FaultModel({k: {m: (v["mean"], v["sd"]) for m, v in d.items()} for k, d in fm["stats"].items()})
    long = {}
    for row in rd("longitudinal_summary.csv"):
        tag = row["fault"]
        long[tag] = {"summary": row, "weekly": rd(f"longitudinal/{tag}/phantom_weekly.csv"),
                     "chain": json.load(open(folder / "longitudinal" / tag / "chain.json", encoding="utf-8"))}
    return {"summary": rd("summary.csv"), "model": model, "run_info": json.load(open(folder / "run_info.json")),
            "real": json.load(open(folder / "real_data_diagnosis.json")), "long": long,
            "catalogue": {c["key"]: c for c in json.load(open(folder / "catalogue.json"))}}


def spectra_compare(clean_fid, fault_fid, aligned_fid=None, title="") -> go.Figure:
    fig = go.Figure()
    for lab, fid, col, dash in (("clean scan", clean_fid, "#9aa5b1", "solid"), ("with the fault", fault_fid, "#c8642a", "solid"),
                                ("fault, transients aligned", aligned_fid, "#2b6cb0", "dot")):
        if fid is None:
            continue
        ppm, sp = F.spectrum(fid)
        sel = (ppm > 0.5) & (ppm < 5.0)
        fig.add_trace(go.Scatter(x=ppm[sel], y=sp.real[sel], mode="lines", name=lab, line=dict(color=col, width=1.6, dash=dash)))
    for lab, p in (("NAA", 2.01), ("Cr", 3.03), ("Cho", 3.20), ("water", 4.65)):
        fig.add_vline(x=p, line=dict(color="#c9d1d8", dash="dot"), annotation_text=lab, annotation_position="top")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title=title,
                      xaxis=dict(title="chemical shift / ppm", autorange="reversed"), yaxis=dict(title="real spectrum / a.u."),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def posterior_bar(post: dict, top: int = 6) -> go.Figure:
    items = list(post.items())[:top][::-1]
    fig = go.Figure(go.Bar(x=[v * 100 for _, v in items], y=[SC.BY_KEY[k].title for k, _ in items], orientation="h",
                           marker_color=[BLAME_COLOR[SC.BY_KEY[k].blame] for k, _ in items],
                           text=[f"{v * 100:.0f} %" for _, v in items], textposition="outside",
                           hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
    fig.update_layout(height=60 + 34 * len(items), margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(title="probability / %", range=[0, 110]))
    return fig


def bias_chart(summary: list[dict]) -> go.Figure:
    rows = summary
    names = [SC.BY_KEY[r["scenario"]].title for r in rows]
    mean = [float(r["cho_cr_mean"]) for r in rows]
    sd = [float(r["cho_cr_sd"]) for r in rows]
    truth = [float(r["cho_cr_truth_tissue"]) for r in rows]
    pass_rate = [float(r["gate_PASS"]) for r in rows]
    colors = [STATUS_COLOR["PASS"] if p > 0.5 else STATUS_COLOR["WARN"] for p in pass_rate]
    fig = go.Figure()
    fig.add_hline(y=0.689, line=dict(color="#6f7d89", dash="dash"), annotation_text="biology of the baseline (0.689)",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(x=names, y=truth, mode="markers", name="true tissue Cho/Cr", marker=dict(symbol="line-ew-open", size=26,
                                                                                                   color="#142029", line=dict(width=2))))
    fig.add_trace(go.Scatter(x=names, y=mean, mode="markers", name="measured (mean ± 1.96 SD of 60 scans)",
                             error_y=dict(array=[1.96 * s for s in sd]), marker=dict(size=11, color=colors),
                             hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Cho/Cr"), legend=dict(orientation="h", y=1.1),
                      xaxis=dict(tickangle=-30))
    return fig


PH_OBS = [("ph_fwhm_hz", "Phantom linewidth / Hz"), ("ph_snr", "Phantom SNR"), ("ph_freq_hz", "Centre frequency / Hz"),
          ("ph_water_ratio", "Residual water"), ("ph_distortion_pct", "Geometry / %")]


def phantom_chart(weekly: list[dict], gates: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=5, subplot_titles=[t for _, t in PH_OBS], horizontal_spacing=0.05)
    wk = [int(w["week"]) for w in weekly]
    for c, (o, _) in enumerate(PH_OBS, start=1):
        fig.add_trace(go.Scatter(x=wk, y=[float(w[f"{o}_twin_x"]) for w in weekly], mode="lines", line=dict(color="#2b6cb0", width=2),
                                 name="scanner twin", showlegend=(c == 1)), row=1, col=c)
        fig.add_trace(go.Scatter(x=wk, y=[float(w[o]) for w in weekly], mode="markers", name="weekly phantom (SIMULATED)",
                                 marker=dict(size=6, color=[STATUS_COLOR[w[f"{o}_status"]] for w in weekly]), showlegend=(c == 1)),
                      row=1, col=c)
        for key, val in gates.get(o, {}).items():
            fig.add_hline(y=val, line=dict(color=STATUS_COLOR["FAIL" if key.startswith("fail") else "WARN"], dash="dash", width=1),
                          row=1, col=c)
        fig.add_vrect(x0=14, x1=26, fillcolor="#c8642a", opacity=0.06, line_width=0, row=1, col=c)
        fig.update_xaxes(title_text="week", row=1, col=c)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return fig
