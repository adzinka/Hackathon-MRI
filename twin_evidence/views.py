"""Plotly figures for the evidence map: 3D brain + MRS voxel + scanner bore, slices, spectra, scanner-twin trends."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import anatomy
from .scanner import OBS, ScannerReport

STATUS_COLOR = {"PASS": "#2e9e5b", "WARN": "#e0a100", "FAIL": "#d64545"}
VOI_CENTER = (-16.0, 45.0, 6.0)
VOI_SIZE = (20.0, 20.0, 20.0)
CSDE_MM = 2.4  # NAA vs Cho displacement per axis (packet field csde_mm_NAA_Cho, Dossier §5.5)


def _iso(vol, values, color, opacity, name, isomin=0.5):
    """Surface of a volume at level isomin (marching cubes on the voxel grid, mapped to mm by the NIfTI affine)."""
    from skimage.measure import marching_cubes
    v = np.pad(np.asarray(values, float), 1)                     # close the surface at the grid border
    verts, faces, _, _ = marching_cubes(v, level=isomin)
    verts -= 1
    A = vol["affine"]
    xyz = verts @ A[:3, :3].T + A[:3, 3]
    return go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     color=color, opacity=opacity, flatshading=False, name=name, showlegend=True, hoverinfo="name",
                     lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15, roughness=0.6))


def _bore(radius=120.0, z0=-70.0, z1=70.0, n=48):
    th = np.linspace(0, 2 * np.pi, n)
    zz = np.linspace(z0, z1, 2)
    TH, ZZ = np.meshgrid(th, zz)
    return radius * np.cos(TH), radius * np.sin(TH), ZZ


def brain_3d(vol: dict, mrs_status: str, roi_status: str, scanner: ScannerReport | None, show_scanner: bool = True,
             show_followup_roi: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(_iso(vol, vol["mask"], "#c3ccd4", 0.15, "brain (mask)"))
    fig.add_trace(_iso(vol, vol["roi"]["baseline"], "#2b6cb0", 0.75, "ROI baseline · 22 140 mm³"))
    if show_followup_roi:
        fig.add_trace(_iso(vol, vol["roi"]["followup"], STATUS_COLOR[roi_status], 0.28, "ROI follow-up mask · 27 810 mm³"))

    # MRS voxel: solid cube coloured by the gate, plus the NAA / Cho voxels displaced by chemical shift
    x, y, z, i, j, k = anatomy.box_mesh(VOI_CENTER, VOI_SIZE)
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=STATUS_COLOR[mrs_status], opacity=0.45,
                            name=f"MRS voxel (PRESS, 20 mm) · {mrs_status}", showlegend=True, hoverinfo="name"))
    for lab, sh, dash, col in (("NAA voxel", (-CSDE_MM / 2,) * 3, "dot", "#5a3e99"), ("Cho voxel", (CSDE_MM / 2,) * 3, "dash", "#c8642a")):
        X, Y, Z = anatomy.box_edges(VOI_CENTER, VOI_SIZE, sh)
        fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", line=dict(color=col, width=4, dash=dash),
                                   name=f"{lab} (chemical-shift displaced, ±{CSDE_MM / 2:.1f} mm)", hoverinfo="name"))

    ann = [
        dict(x=VOI_CENTER[0], y=VOI_CENTER[1], z=VOI_CENTER[2] + 13, text=f"<b>MRS {mrs_status}</b><br>SNR 24.8 · linewidth 11.4 Hz",
             showarrow=True, arrowcolor=STATUS_COLOR[mrs_status], ax=-60, ay=-50, bgcolor="rgba(255,255,255,.85)",
             bordercolor=STATUS_COLOR[mrs_status], font=dict(size=11, color="#142029")),
        dict(x=31.6, y=-15.6, z=14, text=f"<b>ROI {roi_status}</b><br>mask via 1.1 mm registration → u ≈ 19 %",
             showarrow=True, arrowcolor=STATUS_COLOR[roi_status], ax=60, ay=-50, bgcolor="rgba(255,255,255,.85)",
             bordercolor=STATUS_COLOR[roi_status], font=dict(size=11, color="#142029")),
    ]
    if show_scanner:
        sc_status = scanner.status_now if scanner else "PASS"
        # the bore as a light wireframe so the head inside stays visible
        R, Z0, Z1 = 120.0, -70.0, 70.0
        th = np.linspace(0, 2 * np.pi, 80)
        first = True
        for zr in np.linspace(Z0, Z1, 5):
            fig.add_trace(go.Scatter3d(x=R * np.cos(th), y=R * np.sin(th), z=np.full(th.size, zr), mode="lines",
                                       line=dict(color=STATUS_COLOR[sc_status], width=6 if zr in (Z0, Z1) else 2),
                                       name=f"scanner bore · scanner twin {sc_status} (SIMULATED)", showlegend=first,
                                       legendgroup="bore", hoverinfo="name"))
            first = False
        for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            fig.add_trace(go.Scatter3d(x=[R * np.cos(a)] * 2, y=[R * np.sin(a)] * 2, z=[Z0, Z1], mode="lines",
                                       line=dict(color=STATUS_COLOR[sc_status], width=1.5), showlegend=False,
                                       legendgroup="bore", hoverinfo="skip"))
        fig.add_trace(go.Cone(x=[0], y=[-120], z=[75], u=[0], v=[0], w=[30], sizemode="absolute", sizeref=14,
                              colorscale=[[0, "#142029"], [1, "#142029"]], showscale=False, name="B0 (along the bore)", hoverinfo="name"))
        if scanner:
            if hasattr(scanner, "label_text"):
                txt = scanner.label_text
            else:
                lw = scanner.state_t1["phantom_linewidth_Hz"]
                txt = f"phantom linewidth {lw[0]:.1f} ± {lw[1]:.1f} Hz"
            ann.append(dict(x=-120, y=0, z=-70, text=f"<b>Scanner twin {sc_status}</b><br>{txt}",
                            showarrow=False, bgcolor="rgba(255,255,255,.85)", bordercolor=STATUS_COLOR[sc_status],
                            font=dict(size=11, color="#142029")))
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02, font=dict(size=11)),
                      scene=dict(aspectmode="data", annotations=ann,
                                 xaxis=dict(title="x / mm (R+)", showbackground=False), yaxis=dict(title="y / mm (A+)", showbackground=False),
                                 zaxis=dict(title="z / mm (S+)", showbackground=False),
                                 camera=dict(eye=dict(x=0.95, y=-1.05, z=0.75))))
    return fig


def slices(vol: dict, z_mm: float = 0.0) -> tuple[go.Figure, dict]:
    """Baseline, follow-up and difference of the T1w slice nearest z_mm, with ROI contours and the MRS voxel."""
    k = int(round(anatomy.world_to_index(vol, (0, 0, z_mm))[2]))
    k = max(0, min(vol["shape"][2] - 1, k))
    xs, ys = vol["x"][:, 0, k], vol["y"][0, :, k]
    b, f = vol["t1w"]["baseline"][:, :, k], vol["t1w"]["followup"][:, :, k]
    d = f - b
    fig = make_subplots(rows=1, cols=3, subplot_titles=("T1w baseline (May)", "T1w follow-up (Nov)", "difference (Nov − May)"),
                        horizontal_spacing=0.04)
    for c, img, cs, zr in ((1, b, "gray", (0, 950)), (2, f, "gray", (0, 950)), (3, d, "RdBu", (-60, 60))):
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=img.T, colorscale=cs, zmin=zr[0], zmax=zr[1], showscale=(c == 3),
                                 colorbar=dict(len=0.8, thickness=10, title="a.u."), hovertemplate="x %{x} y %{y}: %{z:.0f}<extra></extra>"),
                      row=1, col=c)
        for s, col in (("baseline", "#2b6cb0"), ("followup", "#e0a100")):
            fig.add_trace(go.Contour(x=xs, y=ys, z=vol["roi"][s][:, :, k].T, contours=dict(start=0.5, end=0.5, coloring="none"),
                                     line=dict(color=col, width=2), showscale=False, hoverinfo="skip", name=f"ROI {s}",
                                     showlegend=(c == 1)), row=1, col=c)
        if abs(z_mm - VOI_CENTER[2]) <= VOI_SIZE[2] / 2:
            x0, x1 = VOI_CENTER[0] - 10, VOI_CENTER[0] + 10
            y0, y1 = VOI_CENTER[1] - 10, VOI_CENTER[1] + 10
            fig.add_trace(go.Scatter(x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0], mode="lines", line=dict(color="#c8642a", width=2),
                                     name="MRS voxel", showlegend=(c == 1), hoverinfo="name"), row=1, col=c)
    for c in (1, 2, 3):
        fig.update_xaxes(showticklabels=False, row=1, col=c)
        fig.update_yaxes(showticklabels=False, scaleanchor=f"x{'' if c == 1 else c}", row=1, col=c)
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.05))
    m = vol["roi"]["baseline"][:, :, :] > 0.5
    dd = (vol["t1w"]["followup"] - vol["t1w"]["baseline"])[m]
    stats = {"slice_z_mm": float(vol["z"][0, 0, k]), "roi_mean_diff": float(dd.mean()), "roi_sd_diff": float(dd.std()),
             "expected_noise_sd": 12.0 * np.sqrt(2), "n_vox": int(m.sum())}
    return fig, stats


def spectra(root) -> tuple[go.Figure, dict]:
    fig = go.Figure()
    qc = {}
    for s, col, w in (("baseline", "#6f7d89", 1.6), ("followup", "#c8642a", 1.6)):
        ppm, re = anatomy.spectrum(root, s)
        qc[s] = anatomy.file_qc(root, s)
        sel = (ppm > 0.5) & (ppm < 4.2)
        fig.add_trace(go.Scatter(x=ppm[sel], y=re[sel], mode="lines", line=dict(color=col, width=w),
                                 name=f"{s}: SNR {qc[s]['snr']:.1f} · FWHM {qc[s]['fwhm_hz']:.1f} Hz"))
    for lab, p in (("NAA 2.01", 2.01), ("Cr 3.03", 3.03), ("Cho 3.20", 3.20), ("mI 3.56", 3.56), ("Glx 3.75", 3.75)):
        fig.add_vline(x=p, line=dict(color="#c9d1d8", dash="dot"), annotation_text=lab, annotation_position="top")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title="chemical shift / ppm", autorange="reversed"),
                      yaxis=dict(title="real spectrum / a.u."), legend=dict(orientation="h", y=-0.2))
    return fig, qc


def scanner_trends(rep: ScannerReport, warn_factor: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=[OBS[o][0] for o in OBS], horizontal_spacing=0.07)
    for c, o in enumerate(OBS, start=1):
        s = rep.series[o]
        t = [p["week"] for p in s]
        x = np.array([p["x"] for p in s]); sd = np.array([p["sd"] for p in s])
        fig.add_trace(go.Scatter(x=t + t[::-1], y=list(x + 1.96 * sd) + list((x - 1.96 * sd)[::-1]), fill="toself",
                                 fillcolor="rgba(43,108,176,.15)", line=dict(width=0), hoverinfo="skip", showlegend=(c == 1),
                                 name="scanner twin 95 % band"), row=1, col=c)
        fig.add_trace(go.Scatter(x=t, y=x, mode="lines", line=dict(color="#2b6cb0", width=2), name="scanner twin state",
                                 showlegend=(c == 1)), row=1, col=c)
        fig.add_trace(go.Scatter(x=t, y=[p["value"] for p in s], mode="markers", name="weekly phantom scan (SIMULATED)",
                                 marker=dict(color=[STATUS_COLOR[p["status"]] for p in s], size=7,
                                             symbol=["x" if p["status"] == "FAIL" else "circle" for p in s]),
                                 showlegend=(c == 1), hovertemplate="week %{x}: %{y:.1f} " + OBS[o][1] + "<extra></extra>"), row=1, col=c)
        for key, val in OBS[o][4].items():
            fig.add_hline(y=val, line=dict(color=STATUS_COLOR["FAIL" if key.startswith("fail") else "WARN"], dash="dash", width=1), row=1, col=c)
        for wk, lab in ((0, "t0 brain scan"), (len(s) - 1, "t1 brain scan")):
            fig.add_vline(x=wk, line=dict(color="#142029", dash="dot", width=1), row=1, col=c)
        fig.update_xaxes(title_text="weeks since 2026-05-20", row=1, col=c)
        fig.update_yaxes(title_text=OBS[o][1], row=1, col=c)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return fig
