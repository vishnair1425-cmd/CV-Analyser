"""
Cyclic Voltammetry (CV) Charge Integration App
================================================
Upload an Excel file containing CV data with multiple scans. The app:
  - Plots the combined CV (Potential vs WE(1).Current (A)) coloured by scan.
  - Shows each scan as a separate interactive plot. The default baseline is
    horizontal at Y = 0; draw a new baseline freehand with the mouse
    (press & drag) or drag either endpoint. The area above the baseline is
    shaded red (positive) and below it blue (negative); baseline, shading, and
    integrated charges update live in the browser.
  - Integrates charge separately for the region above the baseline (positive)
    and below it (negative), per scan.
  - Builds peak current / peak voltage tables for the positive and negative
    regions of the combined plot.

Expected columns (exact):  Potential , WE(1).Current (A) , scan
(Manual column overrides are available in the sidebar if needed.)

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


def trapz_signed(potential, current, baseline):
    """Integrate (current - baseline) dV; split into positive / negative areas.
    Zero-crossings within a segment are split at the crossing for accuracy.
    Mirrors the in-browser JS implementation exactly."""
    P = np.asarray(potential, dtype=float)
    diff = np.asarray(current, dtype=float) - np.asarray(baseline, dtype=float)
    pos = neg = 0.0
    for i in range(len(P) - 1):
        dV = P[i + 1] - P[i]
        y0, y1 = diff[i], diff[i + 1]
        if y0 >= 0 and y1 >= 0:
            pos += 0.5 * (y0 + y1) * dV
        elif y0 <= 0 and y1 <= 0:
            neg += 0.5 * (y0 + y1) * dV
        else:
            t = y0 / (y0 - y1) if y1 != y0 else 0.5
            a1 = 0.5 * y0 * dV * t
            a2 = 0.5 * y1 * dV * (1 - t)
            for a in (a1, a2):
                if a >= 0:
                    pos += a
                else:
                    neg += a
    return pos, neg


# ----------------------------------------------------------------------------
# Draw/drag-baseline component (HTML + Plotly.js)
# ----------------------------------------------------------------------------

def draggable_cv_component(potential, current, color, x_title, y_title,
                           default_x1, default_y1, default_x2, default_y2,
                           scan_rate, key_height=440):
    """Render a Plotly chart whose linear baseline can be drawn freehand with
    the mouse (press & drag) or adjusted by dragging either endpoint. The area
    above the baseline is shaded red, below it blue. Integration is live."""
    P = [float(v) for v in potential]
    I = [float(v) for v in current]

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
        "P": P, "I": I, "color": color,
        "xTitle": x_title, "yTitle": y_title,
        "x1": default_x1, "y1": default_y1,
        "x2": default_x2, "y2": default_y2,
        "scanRate": scan_rate,
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
const P = D.P, I = D.I;
let A1 = {x: D.x1, y: D.y1};
let A2 = {x: D.x2, y: D.y2};
const unit = D.scanRate > 0 ? "C" : "A\u00B7V";

function baselineAt(x){
  if (A2.x === A1.x) return A1.y;
  const m = (A2.y - A1.y)/(A2.x - A1.x);
  const b = A1.y - m*A1.x;
  return m*x + b;
}
function integrate(){
  let pos=0, neg=0;
  for(let i=0;i<P.length-1;i++){
    const dV = P[i+1]-P[i];
    const y0 = I[i]   - baselineAt(P[i]);
    const y1 = I[i+1] - baselineAt(P[i+1]);
    if(y0>=0 && y1>=0){ pos += 0.5*(y0+y1)*dV; }
    else if(y0<=0 && y1<=0){ neg += 0.5*(y0+y1)*dV; }
    else {
      const t = (y1!==y0) ? y0/(y0-y1) : 0.5;
      const a1 = 0.5*y0*dV*t, a2 = 0.5*y1*dV*(1-t);
      [a1,a2].forEach(a => { if(a>=0) pos+=a; else neg+=a; });
    }
  }
  if(D.scanRate>0){ pos/=D.scanRate; neg/=D.scanRate; }
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
    "the positive / negative integrated charge live."
)

with st.sidebar:
    st.header("1 · Data")
    uploaded = st.file_uploader("Excel file (.xlsx / .xls)", type=["xlsx", "xls"])

if uploaded is None:
    st.info("⬅️ Upload an Excel file to begin. Expected columns: "
            "`Potential`, `WE(1).Current (A)`, `scan`.")
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

    pot_col = st.selectbox("Potential (X)", cols, index=cols.index(pot_guess))
    cur_col = st.selectbox("Current (Y)", cols, index=cols.index(cur_guess))
    scan_col = st.selectbox("Scan / Cycle", cols, index=cols.index(scan_guess))

    st.header("3 · Optional")
    scan_rate = st.number_input(
        "Scan rate (V/s) — for charge in Coulombs",
        min_value=0.0, value=0.0, step=0.001, format="%.4f",
        help="If > 0, integrated area (A·V) is divided by scan rate to give Coulombs.",
    )

# Clean / validate
work = df[[scan_col, pot_col, cur_col]].copy()
work.columns = ["scan", "potential", "current"]
work = work.dropna(subset=["potential", "current"])
work["scan"] = work["scan"].astype(str)

scans = list(work["scan"].unique())
try:
    scans = sorted(scans, key=lambda s: float(s))
except (ValueError, TypeError):
    scans = sorted(scans)

color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(scans)}
unit_label = "C" if scan_rate > 0 else "A·V"


def to_charge(area):
    return area / scan_rate if scan_rate > 0 else area


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
    "(cathodic) current. Peak Voltage is the potential at that peak current."
)

pos_rows, neg_rows = [], []
for s in scans:
    sub = work[work["scan"] == s]
    if sub.empty:
        continue
    i_max = sub["current"].idxmax()
    i_min = sub["current"].idxmin()
    pos_rows.append({"Scan": s, "Peak Current (A)": sub.loc[i_max, "current"],
                     "Peak Voltage (V)": sub.loc[i_max, "potential"]})
    neg_rows.append({"Scan": s, "Peak Current (A)": sub.loc[i_min, "current"],
                     "Peak Voltage (V)": sub.loc[i_min, "potential"]})

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
            color=color_map[s],
            x_title=pot_col, y_title=cur_col,
            default_x1=x1, default_y1=y1, default_x2=x2, default_y2=y2,
            scan_rate=scan_rate, key_height=440,
        )

    # default-baseline charge for the summary table (server-side, matches JS)
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        bl = m * sub["potential"].values + b
    else:
        bl = np.full(len(sub), y1)
    pos_area, neg_area = trapz_signed(sub["potential"].values,
                                      sub["current"].values, bl)
    summary_rows.append({
        "Scan": s,
        f"Positive charge ({unit_label})": to_charge(pos_area),
        f"Negative charge ({unit_label})": to_charge(neg_area),
        f"Net charge ({unit_label})": to_charge(pos_area + neg_area),
    })

# ----------------------------------------------------------------------------
# Summary + download (default-baseline values)
# ----------------------------------------------------------------------------

if summary_rows:
    st.subheader("Charge summary — all scans (default baseline)")
    st.caption(
        "Computed with the default Y = 0 baseline. The live values above each "
        "plot reflect any drawing/dragging you do in-browser."
    )
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download charge summary (CSV)",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="cv_charge_summary.csv", mime="text/csv",
    )
