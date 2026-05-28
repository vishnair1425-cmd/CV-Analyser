"""
Cyclic Voltammetry (CV) Charge Integration App
================================================
Upload an Excel file containing CV data with multiple scans. The app:
  - Plots the combined CV (Potential vs WE(1).Current (A)) coloured by scan.
  - Shows each scan as a separate interactive plot (Potential X vs Current Y).
    The default baseline is horizontal at Y = 0; draw a new baseline freehand
    with the mouse (press & drag) or drag either endpoint. Area above the
    baseline is shaded red (positive), below it blue (negative).
  - Integrates charge as  Q = ∫ I dt  over the TIME column, giving coulombs
    directly (matches a "Current over Time" integration in CV software).
    Charge is split into positive (above baseline) and negative (below) parts.
  - Builds peak current / peak voltage tables for the positive and negative
    regions, the Neg/Pos charge ratio, ΔEp, and trend-vs-scan plots.

Expected columns:  Potential , WE(1).Current (A) , scan , and a time column
(Manual column overrides are available in the sidebar if names differ.)

Run with:  streamlit run cv_app.py
"""

import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="CV Charge Integration", layout="wide")

PALETTE = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Set2
    + px.colors.qualitative.Dark24
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def guess_column(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for low, orig in cols_lower.items():
            if low == cand.lower():
                return orig
    for cand in candidates:
        for low, orig in cols_lower.items():
            if cand.lower() in low:
                return orig
    return None


@st.cache_data(show_spinner=False)
def load_excel(file_bytes, sheet_name):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def list_sheets(file_bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def trapz_signed(xvar, current, baseline):
    """Integrate (current - baseline) over xvar (TIME, in seconds); split into
    positive / negative areas. Zero-crossings within a segment are split at the
    crossing for accuracy. With xvar = time, the result is charge in coulombs.
    Mirrors the in-browser JS implementation exactly."""
    X = np.asarray(xvar, dtype=float)
    diff = np.asarray(current, dtype=float) - np.asarray(baseline, dtype=float)
    pos = neg = 0.0
    for i in range(len(X) - 1):
        dt = X[i + 1] - X[i]
        y0, y1 = diff[i], diff[i + 1]
        if y0 >= 0 and y1 >= 0:
            pos += 0.5 * (y0 + y1) * dt
        elif y0 <= 0 and y1 <= 0:
            neg += 0.5 * (y0 + y1) * dt
        else:
            t = y0 / (y0 - y1) if y1 != y0 else 0.5
            a1 = 0.5 * y0 * dt * t
            a2 = 0.5 * y1 * dt * (1 - t)
            for a in (a1, a2):
                if a >= 0:
                    pos += a
                else:
                    neg += a
    return pos, neg


# ----------------------------------------------------------------------------
# Draw/drag-baseline component (HTML + Plotly.js)
# ----------------------------------------------------------------------------

def draggable_cv_component(potential, current, time, color, x_title, y_title,
                           default_x1, default_y1, default_x2, default_y2,
                           key_height=440):
    """Render a Plotly chart (Potential X vs Current Y) with a draw/drag linear
    baseline. Charge is integrated over TIME -> coulombs, live in the browser."""
    P = [float(v) for v in potential]
    I = [float(v) for v in current]
    T = [float(v) for v in time]

    # Fixed axis ranges from the data extent (with padding) so the plot is
    # scaled correctly and dragging an anchor never distorts the view.
    pmin, pmax = min(P), max(P)
    imin, imax = min(I), max(I)
    imin = min(imin, 0.0)   # ensure Y=0 baseline is always within view
    imax = max(imax, 0.0)
    px_pad = (pmax - pmin) * 0.05 or 0.01
    iy_pad = (imax - imin) * 0.10 or abs(imax) * 0.1 or 1e-9
    x_range = [pmin - px_pad, pmax + px_pad]
    y_range = [imin - iy_pad, imax + iy_pad]

    payload = json.dumps({
        "P": P, "I": I, "T": T, "color": color,
        "xTitle": x_title, "yTitle": y_title,
        "x1": default_x1, "y1": default_y1,
        "x2": default_x2, "y2": default_y2,
        "xRange": x_range, "yRange": y_range,
    })

    html = """
<div id="toolbar">
  <button id="btnDraw" class="tbtn">\u270F\uFE0F Draw baseline</button>
  <button id="btnReset" class="tbtn">\u21BA Reset</button>
  <span class="hint">Tip: click "Draw baseline", then press &amp; drag on the plot. Or drag either red endpoint directly.</span>
</div>
<div id="root"></div>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const D = __PAYLOAD__;
const P = D.P, I = D.I, T = D.T;
let A1 = {x: D.x1, y: D.y1};
let A2 = {x: D.x2, y: D.y2};
const unit = "C";

function baselineAt(x){
  if (A2.x === A1.x) return A1.y;
  const m = (A2.y - A1.y)/(A2.x - A1.x);
  const b = A1.y - m*A1.x;
  return m*x + b;
}
function integrate(){
  // Charge = integral of (current - baseline) over TIME -> coulombs.
  // The baseline is drawn on the Potential-Current plot, so it is evaluated
  // at each point's potential, but the integration step is dt (seconds).
  let pos=0, neg=0;
  for(let i=0;i<P.length-1;i++){
    const dt = T[i+1]-T[i];
    const y0 = I[i]   - baselineAt(P[i]);
    const y1 = I[i+1] - baselineAt(P[i+1]);
    if(y0>=0 && y1>=0){ pos += 0.5*(y0+y1)*dt; }
    else if(y0<=0 && y1<=0){ neg += 0.5*(y0+y1)*dt; }
    else {
      const t = (y1!==y0) ? y0/(y0-y1) : 0.5;
      const a1 = 0.5*y0*dt*t, a2 = 0.5*y1*dt*(1-t);
      [a1,a2].forEach(a => { if(a>=0) pos+=a; else neg+=a; });
    }
  }
  return {pos, neg};
}
function baselineYs(){ return P.map(baselineAt); }
function fmt(v){ return v.toExponential(4); }

function buildData(){
  const ys = baselineYs();
  // Clamp the curve to the baseline to shade each region separately:
  //  - posCurve fills only where current is ABOVE the baseline
  //  - negCurve fills only where current is BELOW the baseline
  const posCurve = I.map((v,i) => Math.max(v, ys[i]));
  const negCurve = I.map((v,i) => Math.min(v, ys[i]));
  const curve = {x:P, y:I, mode:"lines", name:"CV",
    line:{color:D.color, width:2},
    hovertemplate:"E=%{x:.4f} V<br>I=%{y:.3e} A<extra></extra>"};
  const base = {x:P, y:ys, mode:"lines", name:"Baseline",
    line:{color:"black", width:2, dash:"dash"}, hoverinfo:"skip"};
  const fillPos = {x: P.concat(P.slice().reverse()),
    y: posCurve.concat(ys.slice().reverse()),
    fill:"toself", fillcolor:"rgba(214,39,40,0.20)",   // red = positive area
    line:{color:"rgba(0,0,0,0)"}, hoverinfo:"skip",
    name:"Positive area"};
  const fillNeg = {x: P.concat(P.slice().reverse()),
    y: negCurve.concat(ys.slice().reverse()),
    fill:"toself", fillcolor:"rgba(31,119,180,0.20)",  // blue = negative area
    line:{color:"rgba(0,0,0,0)"}, hoverinfo:"skip",
    name:"Negative area"};
  const anchors = {x:[A1.x, A2.x], y:[A1.y, A2.y], mode:"markers",
    name:"Drag me", marker:{color:"red", size:14, symbol:"circle",
      line:{color:"white", width:2}},
    hovertemplate:"drag<br>E=%{x:.4f} V<br>I=%{y:.3e} A<extra></extra>"};
  return [fillPos, fillNeg, curve, base, anchors];
}

const layout = {
  height: __HEIGHT__,
  margin:{l:70, r:20, t:10, b:50},
  xaxis:{title:{text:D.xTitle}, zeroline:false, range:D.xRange.slice(), autorange:false},
  yaxis:{title:{text:D.yTitle}, zeroline:true, range:D.yRange.slice(), autorange:false},
  template:"plotly_white",
  legend:{orientation:"h", yanchor:"bottom", y:1.02, xanchor:"right", x:1},
  dragmode:false
};

const DEF1 = {x: D.x1, y: D.y1};   // default baseline endpoints (for reset)
const DEF2 = {x: D.x2, y: D.y2};

const gd = document.getElementById("root");
Plotly.newPlot(gd, buildData(), layout,
  {displayModeBar:true, responsive:true,
   modeBarButtonsToRemove:["lasso2d","select2d"]});

function refresh(){
  // Update only the traces; keep the current axis view (initial fixed range,
  // or whatever the user has zoomed/panned to) instead of re-applying layout.
  Plotly.react(gd, buildData(), gd.layout || layout);
  const r = integrate();
  document.getElementById("out").innerHTML =
    '<span class="pos">\u25A0 Positive charge: '+fmt(r.pos)+' '+unit+'</span>'+
    '<span class="neg">\u25A0 Negative charge: '+fmt(r.neg)+' '+unit+'</span>'+
    '<span class="net">Net: '+fmt(r.pos+r.neg)+' '+unit+
    '  &middot;  |Total|: '+fmt(Math.abs(r.pos)+Math.abs(r.neg))+' '+unit+'</span>';
}

function pixelToData(e){
  const bb = gd.getBoundingClientRect();
  const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
  const px = e.clientX - bb.left - gd._fullLayout.margin.l;
  const py = e.clientY - bb.top  - gd._fullLayout.margin.t;
  return {x: xa.p2d(px), y: ya.p2d(py)};
}
function nearestWithin(e, tol){
  const d = pixelToData(e);
  const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
  function dist(A){
    const dx = xa.d2p(A.x) - xa.d2p(d.x);
    const dy = ya.d2p(A.y) - ya.d2p(d.y);
    return Math.sqrt(dx*dx+dy*dy);
  }
  const d1 = dist(A1), d2 = dist(A2);
  const which = d1 <= d2 ? 1 : 2;
  return (Math.min(d1,d2) < tol) ? which : null;
}

// --- interaction state ---
let drawMode = false;   // when true, mouse draws a fresh baseline
let dragging = null;    // 1 or 2 = adjusting an existing endpoint
let drawingNew = false; // mid-draw in draw mode

const btnDraw  = document.getElementById("btnDraw");
const btnReset = document.getElementById("btnReset");

function setDrawMode(on){
  drawMode = on;
  btnDraw.classList.toggle("active", on);
  btnDraw.textContent = on ? "\u270F\uFE0F Drawing… (click & drag)" : "\u270F\uFE0F Draw baseline";
  gd.style.cursor = on ? "crosshair" : "default";
}
btnDraw.addEventListener("click", () => setDrawMode(!drawMode));
btnReset.addEventListener("click", () => {
  A1 = {x: DEF1.x, y: DEF1.y};
  A2 = {x: DEF2.x, y: DEF2.y};
  setDrawMode(false);
  refresh();
});

gd.addEventListener("mousedown", e => {
  if(drawMode){
    // start a fresh baseline: both ends at the press point
    const d = pixelToData(e);
    A1 = {x: d.x, y: d.y};
    A2 = {x: d.x, y: d.y};
    drawingNew = true;
    e.preventDefault();
    refresh();
  } else {
    // adjust an existing endpoint if the press is near one
    const which = nearestWithin(e, 25);
    if(which){ dragging = which; e.preventDefault(); }
  }
});
window.addEventListener("mousemove", e => {
  if(drawingNew){
    A2 = pixelToData(e);     // second end follows the cursor
    refresh();
  } else if(dragging){
    const d = pixelToData(e);
    if(dragging===1){ A1 = d; } else { A2 = d; }
    refresh();
  }
});
window.addEventListener("mouseup", () => {
  if(drawingNew){
    drawingNew = false;
    setDrawMode(false);   // one draw per click of the button
  }
  dragging = null;
});

refresh();
</script>
<style>
  #toolbar{font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin-bottom:6px;
           display:flex; align-items:center; gap:10px; flex-wrap:wrap;}
  .tbtn{font-size:14px; padding:6px 12px; border:1px solid #c7c7c7; border-radius:6px;
        background:#f6f6f6; cursor:pointer;}
  .tbtn:hover{background:#ececec;}
  .tbtn.active{background:#1f77b4; color:#fff; border-color:#1f77b4;}
  #toolbar .hint{font-size:12px; color:#888;}
  #out{font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin-top:10px;
       display:flex; gap:18px; flex-wrap:wrap; font-size:15px;}
  #out .pos{color:#d62728; font-weight:600;}
  #out .neg{color:#1f77b4; font-weight:600;}
  #out .net{color:#444;}
</style>
<div id="out"></div>
"""
    html = html.replace("__PAYLOAD__", payload).replace("__HEIGHT__", str(key_height))
    components.html(html, height=key_height + 130, scrolling=False)


# ----------------------------------------------------------------------------
# Sidebar - input & configuration
# ----------------------------------------------------------------------------

st.title("Cyclic Voltammetry — Charge Integration")
st.caption(
    "Upload CV data, draw or drag the baseline on each scan, and read off "
    "the positive / negative integrated charge (coulombs) live."
)

with st.sidebar:
    st.header("1 · Data")
    uploaded = st.file_uploader("Excel file (.xlsx / .xls)", type=["xlsx", "xls"])

if uploaded is None:
    st.info("⬅️ Upload an Excel file to begin. Expected columns: "
            "`Potential`, `WE(1).Current (A)`, `scan`, and a time column.")
    st.stop()

file_bytes = uploaded.getvalue()

with st.sidebar:
    sheets = list_sheets(file_bytes)
    sheet = st.selectbox("Sheet", sheets, index=0)

df = load_excel(file_bytes, sheet)
cols = list(df.columns)

with st.sidebar:
    st.header("2 · Columns")
    pot_guess = guess_column(["Potential"], cols) or cols[0]
    cur_guess = guess_column(["WE(1).Current (A)", "Current (A)"], cols) or cols[0]
    scan_guess = guess_column(["scan", "cycle"], cols) or cols[0]
    time_guess = guess_column(["Corrected time", "Time (s)", "Time", "t (s)", "time"], cols) or cols[0]

    pot_col = st.selectbox("Potential (X)", cols, index=cols.index(pot_guess))
    cur_col = st.selectbox("Current (Y)", cols, index=cols.index(cur_guess))
    scan_col = st.selectbox("Scan / Cycle", cols, index=cols.index(scan_guess))
    time_col = st.selectbox(
        "Time (s) — for charge integration", cols,
        index=cols.index(time_guess),
        help="Charge is integrated as ∫ I dt over this time column, giving "
             "coulombs directly (matches a 'Current over Time' integration).",
    )

# Clean / validate
work = df[[scan_col, pot_col, cur_col, time_col]].copy()
work.columns = ["scan", "potential", "current", "time"]
work = work.dropna(subset=["potential", "current", "time"])
work["scan"] = work["scan"].astype(str)

scans = list(work["scan"].unique())
try:
    scans = sorted(scans, key=lambda s: float(s))
except (ValueError, TypeError):
    scans = sorted(scans)

color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(scans)}
unit_label = "C"


# ----------------------------------------------------------------------------
# Combined plot
# ----------------------------------------------------------------------------

st.subheader("Combined CV — all scans")
fig = go.Figure()
for s in scans:
    sub = work[work["scan"] == s]
    fig.add_trace(go.Scatter(
        x=sub["potential"], y=sub["current"], mode="lines",
        name=f"Scan {s}", line=dict(color=color_map[s], width=1.5),
        hovertemplate="Scan " + s + "<br>E=%{x:.4f} V<br>I=%{y:.3e} A<extra></extra>",
    ))
fig.update_layout(
    xaxis_title=pot_col, yaxis_title=cur_col, height=560,
    legend=dict(title="Scan"), margin=dict(l=60, r=20, t=20, b=50),
    template="plotly_white", hovermode="closest",
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Peak tables (positive & negative regions)
# ----------------------------------------------------------------------------

st.subheader("Peak current / voltage per scan")
st.caption(
    "Positive region = maximum (anodic) current; Negative region = minimum "
    "(cathodic) current. Peak Voltage is the potential at that peak current. "
    "ΔEp = anodic peak voltage − cathodic peak voltage."
)

peak_metrics = {}   # scan -> dict of anodic/cathodic peak voltage & current, ΔEp
pos_rows, neg_rows = [], []
for s in scans:
    sub = work[work["scan"] == s]
    if sub.empty:
        continue
    i_max = sub["current"].idxmax()      # anodic (positive) peak
    i_min = sub["current"].idxmin()      # cathodic (negative) peak
    v_anodic = float(sub.loc[i_max, "potential"])
    i_anodic = float(sub.loc[i_max, "current"])
    v_cathodic = float(sub.loc[i_min, "potential"])
    i_cathodic = float(sub.loc[i_min, "current"])
    dEp = v_anodic - v_cathodic
    peak_metrics[s] = {
        "v_anodic": v_anodic, "i_anodic": i_anodic,
        "v_cathodic": v_cathodic, "i_cathodic": i_cathodic,
        "dEp": dEp,
    }
    pos_rows.append({"Scan": s, "Peak Current (A)": i_anodic,
                     "Peak Voltage (V)": v_anodic})
    neg_rows.append({"Scan": s, "Peak Current (A)": i_cathodic,
                     "Peak Voltage (V)": v_cathodic})

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Positive region (anodic peak)**")
    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
with c2:
    st.markdown("**Negative region (cathodic peak)**")
    st.dataframe(pd.DataFrame(neg_rows), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Per-scan plots with DRAW/DRAG baseline
# ----------------------------------------------------------------------------

st.subheader("Per-scan integration — draw or drag the baseline")
st.caption(
    "Default baseline is horizontal at Y = 0 (the current axis). Area above the "
    "baseline is shaded **red** (positive), below it **blue** (negative). "
    "Charge is integrated as ∫ I dt over time (coulombs). "
    "Click **Draw baseline** then press-and-drag on the plot to draw a "
    "new baseline with your mouse, or drag either red endpoint to fine-tune. "
    "**Reset** returns to Y = 0. Positive / negative charge update live underneath."
)

summary_rows = []
for s in scans:
    sub = work[work["scan"] == s].reset_index(drop=True)
    if len(sub) < 2:
        continue

    # Default baseline is horizontal at Y = 0 (the current axis), spanning the
    # full potential range. Drag an endpoint or use "Draw baseline" to change it.
    x1 = float(sub["potential"].min())
    y1 = 0.0
    x2 = float(sub["potential"].max())
    y2 = 0.0

    with st.expander(f"Scan {s}", expanded=(s == scans[0])):
        st.markdown(f"#### Scan {s}")
        draggable_cv_component(
            potential=sub["potential"].tolist(),
            current=sub["current"].tolist(),
            time=sub["time"].tolist(),
            color=color_map[s],
            x_title=pot_col, y_title=cur_col,
            default_x1=x1, default_y1=y1, default_x2=x2, default_y2=y2,
            key_height=440,
        )

    # default-baseline charge for the summary table (server-side, matches JS).
    # Baseline evaluated vs potential; integration is over TIME -> coulombs.
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        bl = m * sub["potential"].values + b
    else:
        bl = np.full(len(sub), y1)
    pos_area, neg_area = trapz_signed(sub["time"].values,
                                      sub["current"].values, bl)
    pm = peak_metrics.get(s, {})
    ratio = (neg_area / pos_area) if pos_area != 0 else float("nan")
    summary_rows.append({
        "Scan": s,
        "Anodic peak V (V)": pm.get("v_anodic", float("nan")),
        "Cathodic peak V (V)": pm.get("v_cathodic", float("nan")),
        "ΔEp (V)": pm.get("dEp", float("nan")),
        "Anodic peak I (A)": pm.get("i_anodic", float("nan")),
        "Cathodic peak I (A)": pm.get("i_cathodic", float("nan")),
        f"Positive charge ({unit_label})": pos_area,
        f"Negative charge ({unit_label})": neg_area,
        f"Net charge ({unit_label})": pos_area + neg_area,
        "Neg/Pos charge ratio": ratio,
    })

# ----------------------------------------------------------------------------
# Summary + download (default-baseline values)
# ----------------------------------------------------------------------------

if summary_rows:
    st.subheader("Charge summary — all scans (default baseline)")
    st.caption(
        "Charge = ∫ I dt over the time column, with the default Y = 0 baseline "
        "(coulombs). The live values above each plot reflect any "
        "drawing/dragging you do in-browser."
    )
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download charge summary (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="cv_charge_summary.csv", mime="text/csv",
    )

    # ------------------------------------------------------------------------
    # Trend plots vs scan number
    # ------------------------------------------------------------------------
    st.subheader("Trends vs scan number")
    st.caption(
        "Each metric plotted against scan number. Charges and the Neg/Pos ratio "
        "use the default Y = 0 baseline integrated over time."
    )

    # numeric x-axis: parse scan labels to floats, else fall back to 1..N
    try:
        x_scan = [float(s) for s in summary["Scan"]]
    except (ValueError, TypeError):
        x_scan = list(range(1, len(summary) + 1))

    pos_q_col = f"Positive charge ({unit_label})"
    neg_q_col = f"Negative charge ({unit_label})"

    def trend_chart(y, y_title, color):
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=x_scan, y=list(y), mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=7, color=color),
            hovertemplate="Scan %{x}<br>%{y:.4g}<extra></extra>",
        ))
        f.update_layout(
            xaxis_title="Scan", yaxis_title=y_title, height=320,
            margin=dict(l=70, r=20, t=30, b=45), template="plotly_white",
            showlegend=False,
        )
        return f

    # (a)-(h) in the requested order, two per row
    trends = [
        ("a) Positive (anodic) peak voltage vs scan", summary["Anodic peak V (V)"],   "Anodic peak V (V)",   "#d62728"),
        ("b) Negative (cathodic) peak voltage vs scan", summary["Cathodic peak V (V)"], "Cathodic peak V (V)", "#1f77b4"),
        ("c) Positive (anodic) peak current vs scan", summary["Anodic peak I (A)"],   "Anodic peak I (A)",   "#d62728"),
        ("d) Negative (cathodic) peak current vs scan", summary["Cathodic peak I (A)"], "Cathodic peak I (A)", "#1f77b4"),
        ("e) Peak voltage difference ΔEp vs scan", summary["ΔEp (V)"],                "ΔEp (V)",             "#2ca02c"),
        ("f) Positive charge vs scan", summary[pos_q_col],                            pos_q_col,             "#d62728"),
        ("g) Negative charge vs scan", summary[neg_q_col],                            neg_q_col,             "#1f77b4"),
        ("h) |Neg/Pos| charge ratio vs scan", summary["Neg/Pos charge ratio"].abs(), "|Neg/Pos| charge ratio","#9467bd"),
    ]

    for i in range(0, len(trends), 2):
        cols = st.columns(2)
        for col, (title, ydata, ytitle, color) in zip(cols, trends[i:i + 2]):
            with col:
                st.markdown(f"**{title}**")
                st.plotly_chart(trend_chart(ydata, ytitle, color),
                                use_container_width=True)
