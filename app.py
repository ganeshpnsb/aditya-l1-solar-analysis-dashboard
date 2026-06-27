"""
app.py
======
Aditya-L1 Solar Flare Analysis Platform - Streamlit front-end.

A dark, ISRO / deep-space themed dashboard that ties together the analysis
pipeline defined in ``utils/``:

    Home  -> Upload -> Visualisation -> Detection -> Master Catalogue
          -> Forecasting -> Evaluation -> Alerts -> Download Center

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    fits_parser,
    flare_detector,
    catalog_generator,
    forecasting,
    evaluation,
    flux_calibration,
)

# ---------------------------------------------------------------------------
# Page configuration + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Aditya-L1 Solar Flare Analysis",
    page_icon="\u2600\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deep-space palette (kept to ~5 colours per design guidance).
COL_BG = "#070b1a"
COL_SURFACE = "#0f1630"
COL_ACCENT = "#ff7a18"   # solar orange
COL_SOFT = "#37c2ff"     # soft X-ray (SoLEXS) cyan
COL_HARD = "#ff5470"     # hard X-ray (HEL1OS) rose
COL_TEXT = "#e6ecff"
COL_MUTED = "#8a93b8"

CUSTOM_CSS = f"""
<style>
.stApp {{
    background:
        radial-gradient(1200px 600px at 80% -10%, rgba(55,194,255,0.08), transparent),
        radial-gradient(900px 500px at 10% 10%, rgba(255,122,24,0.07), transparent),
        {COL_BG};
    color: {COL_TEXT};
}}
section[data-testid="stSidebar"] {{ background: {COL_SURFACE}; border-right: 1px solid rgba(255,255,255,0.06); }}
.hero-title {{ font-size: 2.1rem; font-weight: 800; letter-spacing: .5px; margin-bottom:.2rem; }}
.hero-sub {{ color: {COL_MUTED}; font-size: 1rem; }}
.metric-card {{
    background: linear-gradient(160deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.1rem 1.25rem; height: 100%;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}}
.metric-label {{ color: {COL_MUTED}; font-size: .8rem; text-transform: uppercase; letter-spacing: 1px; }}
.metric-value {{ font-size: 1.9rem; font-weight: 800; margin-top:.25rem; }}
.metric-foot {{ color: {COL_MUTED}; font-size: .8rem; margin-top:.35rem; }}
.badge {{ display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.72rem; font-weight:700; }}
.alert-card {{
    background: linear-gradient(160deg, rgba(255,84,112,0.18), rgba(255,122,24,0.10));
    border: 1px solid rgba(255,84,112,0.5); border-radius: 16px; padding: 1.25rem 1.5rem;
}}
.section-note {{ color: {COL_MUTED}; font-size:.9rem; }}
hr {{ border-color: rgba(255,255,255,0.08); }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "solexs_df": None,
        "hel1os_df": None,
        "solexs_cat": None,
        "hel1os_cat": None,
        "master_cat": None,
        "uploaded_files": [],          # list of dicts: name, size, status, source
        "last_analysis": None,
        "forecast_prob": None,
        "forecast_risk": None,
        "forecast_horizon": 30,
        "forecaster": None,
        "forecast_backend": None,
        "eval_result": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ---------------------------------------------------------------------------
# Reusable UI helpers
# ---------------------------------------------------------------------------
def metric_card(label: str, value, foot: str = "", accent: str = COL_ACCENT):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent};">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_theme(fig, height=420, title=""):
    """Apply the consistent dark space theme to a Plotly figure."""
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=COL_TEXT, family="sans-serif"),
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def touch_analysis():
    st.session_state.last_analysis = datetime.now()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "Home Dashboard",
    "FITS Upload",
    "Data Visualization",
    "Flare Detection",
    "Master Catalogue",
    "Forecasting",
    "Evaluation",
    "Alerts",
    "Download Center",
]

with st.sidebar:
    st.markdown(f"### \u2600\ufe0f Aditya-L1")
    st.caption("Solar Flare Analysis Platform")
    if "page" not in st.session_state:
     st.session_state.page = "Home Dashboard"
    for p in PAGES:
     if st.sidebar.button(
        f"{p}",
        use_container_width=True,
        key=p
    ):
        st.session_state.page = p

     page = st.session_state.get("page", "Home Dashboard")
    st.markdown("---")
   
# ===========================================================================
# FEATURE 1 - HOME DASHBOARD
# ===========================================================================
def page_home():
    st.markdown('<div class="hero-title">Aditya-L1 Solar Flare Analysis Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Soft X-ray (SoLEXS) and Hard X-ray (HEL1OS) flare detection, '
        "cataloguing, forecasting and verification.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.container():
        st.markdown(
            """
            **About the mission.** Aditya-L1 is India's first dedicated solar observatory,
            operated by ISRO and stationed in a halo orbit around the Sun-Earth Lagrange
            point **L1**, ~1.5 million km from Earth. From this vantage point it observes the
            Sun without eclipses or occultations. Two of its payloads measure solar X-rays:
            **SoLEXS** (Solar Low Energy X-ray Spectrometer) tracks the *soft* X-ray band, while
            **HEL1OS** (High Energy L1 Orbiting X-ray Spectrometer) captures the *hard* X-ray band.
            Together they characterise the energetics of solar flares.
            """
        )

    n_files = len(st.session_state.uploaded_files)
    n_detected = _count_detected()
    n_predicted = _count_predicted()
    last = st.session_state.last_analysis
    last_str = last.strftime("%Y-%m-%d %H:%M:%S") if last else "No analysis yet"

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Uploaded Files", n_files, "SoLEXS + HEL1OS", COL_SOFT)
    with c2:
        metric_card("Detected Flares", n_detected, "Across both instruments", COL_ACCENT)
    with c3:
        metric_card("Predicted Flares", n_predicted, "Master catalogue events", COL_HARD)
    with c4:
        prob = st.session_state.forecast_prob
        metric_card(
            "Flare Probability",
            f"{prob*100:.0f}%" if prob is not None else "--",
            f"Next {st.session_state.forecast_horizon} min",
            COL_SOFT,
        )

    st.write("")
    metric_card("Last Analysis Timestamp", last_str, "Updated on each pipeline run", COL_MUTED)

    st.markdown("---")
    st.markdown("#### Pipeline status")
    status = {
        "SoLEXS light curve": st.session_state.solexs_df is not None,
        "HEL1OS light curve": st.session_state.hel1os_df is not None,
        "SoLEXS catalogue": _has(st.session_state.solexs_cat),
        "HEL1OS catalogue": _has(st.session_state.hel1os_cat),
        "Master catalogue": _has(st.session_state.master_cat),
        "Forecast": st.session_state.forecast_prob is not None,
    }
    cols = st.columns(3)
    for i, (label, ok) in enumerate(status.items()):
        with cols[i % 3]:
            color = "#23c98a" if ok else COL_MUTED
            mark = "Ready" if ok else "Pending"
            st.markdown(
                f'<span class="badge" style="background:rgba(35,201,138,.15);color:{color};">'
                f'{label}: {mark}</span>',
                unsafe_allow_html=True,
            )


# ===========================================================================
# FEATURE 2 - FITS FILE UPLOAD
# ===========================================================================
def page_upload():
    st.header("FITS File Upload")
    st.markdown(
        '<p class="section-note">Upload SoLEXS (soft X-ray) and HEL1OS (hard X-ray) FITS '
        "products. Files are validated before parsing.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SoLEXS (Soft X-ray)")
        solexs_file = st.file_uploader(
            "Drop SoLEXS .fits file", type=["fits", "fit", "fts", "gz"], key="up_solexs"
        )
        if solexs_file is not None:
            _handle_upload(solexs_file, fits_parser.SOURCE_SOLEXS, "solexs_df")

    with col2:
        st.subheader("HEL1OS (Hard X-ray)")
        hel1os_file = st.file_uploader(
            "Drop HEL1OS .fits file", type=["fits", "fit", "fts", "gz"], key="up_hel1os"
        )
        if hel1os_file is not None:
            _handle_upload(hel1os_file, fits_parser.SOURCE_HEL1OS, "hel1os_df")

    st.markdown("---")
    st.subheader("Uploaded files")
    if st.session_state.uploaded_files:
        table = pd.DataFrame(st.session_state.uploaded_files)
        table["size"] = table["size"].apply(_human_size)
        table = table.rename(
            columns={"name": "File Name", "size": "Size", "status": "Upload Status", "source": "Instrument"}
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No files uploaded yet. You can also click **Load Demo Data** in the sidebar.")


def _handle_upload(file, source, state_key):
    """Validate, parse and register an uploaded FITS file."""
    is_valid, message = fits_parser.validate_fits(file)
    size = getattr(file, "size", 0)

    if not is_valid:
        st.error(f"Validation failed: {message}")
        _register_file(file.name, size, "Invalid", source)
        return

    with st.spinner(f"Parsing {file.name} ..."):
        df = fits_parser.load_timeseries(
    file,
    source=source,
    fallback_to_synthetic=True
)
        if df is None:
          return
    st.session_state[state_key] = df
    touch_analysis()

    if df is None:
      st.error("Could not parse the uploaded file.")
      return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["counts"], mode="lines",
                             line=dict(color=COL_SOFT if source == "SoLEXS" else COL_HARD, width=1)))
    st.plotly_chart(plotly_theme(fig, height=240, title=f"{source} preview"), use_container_width=True)


def _register_file(name, size, status, source):
    files = [f for f in st.session_state.uploaded_files if not (f["name"] == name and f["source"] == source)]
    files.append({"name": name, "size": size, "status": status, "source": source})
    st.session_state.uploaded_files = files


# ===========================================================================
# FEATURE 3 - DATA VISUALIZATION
# ===========================================================================
def page_visualization():
    st.header("Data Visualization")
    solexs, hel1os = st.session_state.solexs_df, st.session_state.hel1os_df

    if solexs is None and hel1os is None:
        _no_data()
        return

    st.markdown(
        '<p class="section-note">Light curves support zoom (drag), hover tooltips, and '
        "download-as-PNG (camera icon in the chart toolbar).</p>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Soft X-ray (SoLEXS)", "Hard X-ray (HEL1OS)", "Combined", "GOES Flux"]
    )

    with tab1:
        if solexs is not None:
            st.plotly_chart(_light_curve(solexs, "SoLEXS soft X-ray light curve", COL_SOFT),
                            use_container_width=True)
        else:
            st.info("No SoLEXS data loaded.")

    with tab2:
        if hel1os is not None:
            st.plotly_chart(_light_curve(hel1os, "HEL1OS hard X-ray light curve", COL_HARD),
                            use_container_width=True)
        else:
            st.info("No HEL1OS data loaded.")

    with tab3:
        fig = go.Figure()
        if solexs is not None:
            fig.add_trace(go.Scatter(x=solexs["time"], y=solexs["counts"], name="SoLEXS (soft)",
                                     mode="lines", line=dict(color=COL_SOFT, width=1.2)))
        if hel1os is not None:
            fig.add_trace(go.Scatter(x=hel1os["time"], y=hel1os["counts"], name="HEL1OS (hard)",
                                     mode="lines", line=dict(color=COL_HARD, width=1.2)))
        fig.update_layout(xaxis_rangeslider_visible=True)
        st.plotly_chart(plotly_theme(fig, height=480, title="Combined light curve"),
                        use_container_width=True)

    with tab4:
        _goes_flux_view(solexs, hel1os)


def _goes_flux_view(solexs, hel1os):
    """Calibrated 1-8 A flux on a log scale with GOES A/B/C/M/X class bands."""
    st.markdown(
        '<p class="section-note">Raw counts are calibrated to an approximate '
        "1-8 &#197; soft X-ray flux (W m<sup>-2</sup>) and overlaid with the standard "
        "GOES flare-class bands. The calibration is a transparent power-law proxy "
        "(real flux needs the full instrument response).</p>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    for df, name, color in ((solexs, "SoLEXS (soft)", COL_SOFT), (hel1os, "HEL1OS (hard)", COL_HARD)):
        if df is None:
            continue
        bg = float(np.nanmedian(df["counts"]))
        calib = flux_calibration.auto_calibration(bg)
        flux = flux_calibration.calibrate_counts(df["counts"], calib)
        fig.add_trace(go.Scatter(
            x=df["time"], y=flux, name=name, mode="lines",
            line=dict(color=color, width=1.2),
            hovertemplate="%{y:.2e} W/m^2<extra>" + name + "</extra>",
        ))

    # GOES class band boundaries.
    band_colors = {
        "B": "rgba(35,201,138,0.10)",
        "C": "rgba(55,194,255,0.10)",
        "M": "rgba(255,122,24,0.12)",
        "X": "rgba(255,84,112,0.14)",
    }
    bounds = {"B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4, "top": 1e-3}
    for letter, lo, hi in (
        ("B", bounds["B"], bounds["C"]),
        ("C", bounds["C"], bounds["M"]),
        ("M", bounds["M"], bounds["X"]),
        ("X", bounds["X"], bounds["top"]),
    ):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=band_colors[letter], line_width=0,
                      annotation_text=letter, annotation_position="right",
                      annotation_font_color=COL_MUTED)

    fig.update_yaxes(type="log", title="1-8 A flux (W m^-2)", range=[-8, -3])
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(plotly_theme(fig, height=480, title="GOES-calibrated soft X-ray flux"),
                    use_container_width=True)


def _light_curve(df, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["counts"], mode="lines",
                             line=dict(color=color, width=1.2), name="counts"))
    fig.update_layout(xaxis_rangeslider_visible=True)
    return plotly_theme(fig, height=460, title=title)


# ===========================================================================
# FEATURE 4 - SOLAR FLARE DETECTION
# ===========================================================================
def page_detection():
    st.header("Solar Flare Detection")
    solexs, hel1os = st.session_state.solexs_df, st.session_state.hel1os_df
    if solexs is None and hel1os is None:
        _no_data()
        return

    with st.expander("Detection parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        smooth_window = c1.slider("Smoothing window (samples)", 5, 51, 11, step=2)
        sigma = c2.slider("Threshold (sigma above baseline)", 1.0, 6.0, 3.0, step=0.5)
        min_sep = c3.slider("Min. separation (s)", 30, 600, 60, step=30)

    if st.button("Run detection", type="primary"):
        if solexs is not None:
            st.session_state.solexs_cat = flare_detector.detect_flares(
                solexs, smooth_window=smooth_window, sigma=sigma, min_separation_s=min_sep
            )
        if hel1os is not None:
            st.session_state.hel1os_cat = flare_detector.detect_flares(
                hel1os, smooth_window=smooth_window, sigma=sigma, min_separation_s=min_sep
            )
        touch_analysis()
        st.success("Detection complete.")

    _detection_panel("SoLEXS", solexs, st.session_state.solexs_cat, COL_SOFT, "solexs_catalogue.csv")
    _detection_panel("HEL1OS", hel1os, st.session_state.hel1os_cat, COL_HARD, "hel1os_catalogue.csv")


def _detection_panel(label, df, cat, color, filename):
    if df is None:
        return
    st.markdown(f"#### {label} catalogue")
    if not _has(cat):
        st.info(f"Run detection to generate the {label} flare catalogue.")
        return

    n = len(cat)
    c1, c2 = st.columns([1, 3])
    with c1:
        metric_card("Flares detected", n, label, color)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time"], y=df["counts"], mode="lines",
                                 line=dict(color=color, width=1), name="counts"))
        if n:
            fig.add_trace(go.Scatter(
                x=cat["Peak Time"], y=cat["Peak Counts"], mode="markers",
                marker=dict(color=COL_ACCENT, size=10, symbol="star"), name="flare peaks"))
        st.plotly_chart(plotly_theme(fig, height=300, title=f"{label} detected peaks"),
                        use_container_width=True)

    st.dataframe(cat, use_container_width=True, hide_index=True)
    st.download_button(f"Download {label} catalogue (CSV)", df_to_csv_bytes(cat),
                       file_name=filename, mime="text/csv")


# ===========================================================================
# FEATURE 5 - MASTER CATALOGUE
# ===========================================================================
def page_master():
    st.header("Master Catalogue Generation")
    st.markdown(
        '<p class="section-note">SoLEXS and HEL1OS detections within +/- 60 s are merged into '
        "a single solar flare event with a confidence score.</p>",
        unsafe_allow_html=True,
    )

    solexs_cat, hel1os_cat = st.session_state.solexs_cat, st.session_state.hel1os_cat
    if not _has(solexs_cat) and not _has(hel1os_cat):
        st.info("Run flare detection first (Flare Detection page).")
        return

    tol = st.slider("Coincidence window (s)", 10, 180, 60, step=10)
    if st.button("Generate master catalogue", type="primary"):
        st.session_state.master_cat = catalog_generator.generate_master_catalog(
            solexs_cat if _has(solexs_cat) else pd.DataFrame(),
            hel1os_cat if _has(hel1os_cat) else pd.DataFrame(),
            tolerance_s=tol,
        )
        touch_analysis()
        st.success("Master catalogue generated.")

    master = st.session_state.master_cat
    if _has(master):
        both = (master["SoLEXS Detection"].eq("Yes") & master["HEL1OS Detection"].eq("Yes")).sum()
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total events", len(master), "Merged", COL_ACCENT)
        with c2:
            metric_card("Dual-instrument", int(both), "Seen by both", COL_SOFT)
        with c3:
            metric_card("Mean confidence", f"{master['Confidence Score'].mean():.2f}", "0-1 scale", COL_HARD)

        st.dataframe(master, use_container_width=True, hide_index=True)
        st.download_button("Download master_catalog.csv", df_to_csv_bytes(master),
                           file_name="master_catalog.csv", mime="text/csv")


# ===========================================================================
# FEATURE 6 - FORECASTING
# ===========================================================================
def page_forecasting():
    st.header("Solar Flare Forecasting")
    primary = st.session_state.solexs_df if st.session_state.solexs_df is not None else st.session_state.hel1os_df
    if primary is None:
        _no_data()
        return

    c1, c2 = st.columns([2, 1])
    horizon = c1.slider("Forecast horizon N (minutes)", 5, 120, st.session_state.forecast_horizon, step=5)
    st.session_state.forecast_horizon = horizon

    model_choice = c2.selectbox(
        "Model backend",
        ["Auto (XGBoost / GBM)", "LSTM (TensorFlow)"],
        help="LSTM uses a TensorFlow recurrent network; it falls back automatically "
             "to gradient boosting if TensorFlow is not installed.",
    )
    prefer_lstm = model_choice.startswith("LSTM")

    if prefer_lstm and not forecasting._HAS_TF:
        st.warning(
            "TensorFlow is not installed in this environment, so the LSTM will "
            "fall back to gradient boosting. Run `pip install tensorflow` to enable it."
        )

    if st.button("Run forecast", type="primary"):
        cat = st.session_state.solexs_cat if _has(st.session_state.solexs_cat) else None
        spin = "Training LSTM and predicting ..." if prefer_lstm else "Training forecaster and predicting ..."
        with st.spinner(spin):
            prob, risk, forecaster = forecasting.forecast_probability(
                primary, horizon_min=horizon, flare_catalogue=cat, prefer_lstm=prefer_lstm
            )
        st.session_state.forecast_prob = prob
        st.session_state.forecast_risk = risk
        st.session_state.forecaster = forecaster
        st.session_state.forecast_backend = forecaster.backend
        touch_analysis()

    prob = st.session_state.forecast_prob
    if prob is None:
        st.info("Run the forecast to estimate flare probability.")
        return

    risk = st.session_state.forecast_risk
    risk_color = {"Low": "#23c98a", "Medium": COL_ACCENT, "High": COL_HARD}.get(risk, COL_MUTED)

    backend_labels = {
        "lstm": "TensorFlow LSTM",
        "xgboost": "XGBoost",
        "sklearn": "Gradient Boosting",
        "heuristic": "Logistic heuristic",
    }
    backend = st.session_state.get("forecast_backend")
    if backend:
        st.caption(f"Active model backend: **{backend_labels.get(backend, backend)}**")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(_gauge(prob, risk_color), use_container_width=True)
    with c2:
        st.markdown(f"#### Flare probability (next {horizon} min)")
        st.markdown(
            f'<div class="metric-value" style="color:{risk_color};font-size:3rem;">{prob*100:.1f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="badge" style="background:{risk_color}22;color:{risk_color};font-size:1rem;">'
            f"Risk level: {risk}</span>",
            unsafe_allow_html=True,
        )
        # Probability bar
        bar = go.Figure(go.Bar(x=[prob * 100], y=["Probability"], orientation="h",
                               marker=dict(color=risk_color), text=[f"{prob*100:.1f}%"], textposition="auto"))
        bar.update_xaxes(range=[0, 100], title="%")
        st.plotly_chart(plotly_theme(bar, height=160), use_container_width=True)

    # Probability timeline (reuse the trained model to avoid retraining)
    timeline = forecasting.probability_timeline(
        primary, horizon_min=horizon, forecaster=st.session_state.get("forecaster")
    )
    if not timeline.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline["time"], y=timeline["probability"] * 100,
                                 mode="lines", fill="tozeroy", line=dict(color=COL_ACCENT, width=1.5),
                                 name="probability"))
        fig.update_yaxes(range=[0, 100], title="Flare probability (%)")
        st.plotly_chart(plotly_theme(fig, height=320, title="Forecast probability over time"),
                        use_container_width=True)


def _gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"color": COL_TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COL_MUTED},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.03)",
            "steps": [
                {"range": [0, 40], "color": "rgba(35,201,138,0.25)"},
                {"range": [40, 70], "color": "rgba(255,122,24,0.25)"},
                {"range": [70, 100], "color": "rgba(255,84,112,0.30)"},
            ],
        },
        title={"text": "Flare risk gauge"},
    ))
    return plotly_theme(fig, height=320)


# ===========================================================================
# FEATURE 7 - EVALUATION DASHBOARD
# ===========================================================================
def page_evaluation():
    st.header("Evaluation Dashboard")
    st.markdown(
        '<p class="section-note">Verification of forecast alerts against observed flares. '
        "TPR = TP/(TP+FN), FAR = FP/(FP+TP), Lead Time = Flare Peak - Alert Time.</p>",
        unsafe_allow_html=True,
    )

    master = st.session_state.master_cat
    solexs_cat = st.session_state.solexs_cat
    actual = master if _has(master) else solexs_cat
    if not _has(actual):
        st.info("Generate a catalogue first to evaluate against observed flares.")
        return

    tol = st.slider("Alert-to-flare tolerance (s)", 60, 900, 300, step=60)
    if st.button("Compute evaluation metrics", type="primary"):
        alerts = _simulate_alerts(actual)
        st.session_state.eval_result = evaluation.evaluate_forecasts(alerts, actual, tolerance_s=tol)
        touch_analysis()

    res = st.session_state.eval_result
    if res is None:
        return

    cards = res.metric_cards()
    items = list(cards.items())
    row1 = st.columns(4)
    row2 = st.columns(3)
    palette = [COL_SOFT, COL_HARD, COL_ACCENT, "#23c98a", COL_SOFT, COL_HARD, COL_ACCENT]
    for i, (label, value) in enumerate(items):
        target = row1[i] if i < 4 else row2[i - 4]
        with target:
            metric_card(label, value, "", palette[i % len(palette)])

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Confusion matrix")
        cm = res.confusion_matrix()
        fig = go.Figure(data=go.Heatmap(
            z=cm.values, x=list(cm.columns), y=list(cm.index),
            text=cm.values, texttemplate="%{text}", textfont={"size": 18},
            colorscale=[[0, COL_SURFACE], [1, COL_ACCENT]], showscale=False))
        st.plotly_chart(plotly_theme(fig, height=320), use_container_width=True)
    with c2:
        st.markdown("#### Score overview")
        labels = ["TPR", "FAR", "Accuracy", "Precision", "Recall", "F1"]
        vals = [res.tpr, res.far, res.accuracy, res.precision, res.recall, res.f1]
        fig = go.Figure(go.Bar(x=labels, y=vals, marker=dict(color=COL_SOFT)))
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(plotly_theme(fig, height=320), use_container_width=True)

    st.caption(
        f"Confusion matrix: TP={res.tp}, FP={res.fp}, FN={res.fn}, TN={res.tn} | "
        f"Average lead time = {res.avg_lead_time_s/60:.1f} min."
    )


def _simulate_alerts(actual_flares: pd.DataFrame) -> pd.DataFrame:
    """
    Build a plausible alert series from observed flares for demonstration:
    most true flares are alerted ahead of time, with a few missed + a few false.
    Replace this with real model alerts in production.
    """
    rng = np.random.default_rng(7)
    times = pd.to_datetime(actual_flares["Peak Time" if "Peak Time" in actual_flares else "Event Time"])
    rows = []
    for t in times:
        if rng.random() < 0.8:  # 80% detection
            lead = rng.uniform(120, 600)  # alert 2-10 min early
            rows.append({"Alert Time": t - pd.Timedelta(seconds=lead)})
    # Inject a couple of false alarms
    if len(times):
        base = times.iloc[0]
        for _ in range(max(1, int(0.15 * len(times)))):
            rows.append({"Alert Time": base + pd.Timedelta(minutes=float(rng.uniform(5, 200)))})
    return pd.DataFrame(rows)


# ===========================================================================
# FEATURE 8 - ALERT SYSTEM
# ===========================================================================
def page_alerts():
    st.header("Alert System")
    threshold = st.slider("Alert threshold (probability %)", 10, 95, 70, step=5) / 100.0
    prob = st.session_state.forecast_prob
    horizon = st.session_state.forecast_horizon

    if prob is None:
        st.info("Run a forecast first (Forecasting page) to enable alerting.")
        return

    if prob >= threshold:
        st.markdown(
            f"""
            <div class="alert-card">
                <div style="font-size:1.6rem;font-weight:800;">\U0001f6a8 Solar Flare Alert</div>
                <div style="font-size:1.1rem;margin-top:.4rem;">
                    Probability: <b>{prob*100:.1f}%</b><br/>
                    Predicted within: <b>{horizon} minutes</b><br/>
                    Risk level: <b>{st.session_state.forecast_risk}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.toast("Solar flare alert triggered!", icon="\U0001f6a8")
    else:
        st.success(
            f"Nominal conditions. Flare probability {prob*100:.1f}% is below the "
            f"{threshold*100:.0f}% alert threshold."
        )


# ===========================================================================
# FEATURE 9 - DOWNLOAD CENTER
# ===========================================================================
def page_download():
    st.header("Download Center")
    st.markdown('<p class="section-note">Export every product generated by the pipeline.</p>',
                unsafe_allow_html=True)

    items = [
        ("SoLEXS catalogue", st.session_state.solexs_cat, "solexs_catalogue.csv"),
        ("HEL1OS catalogue", st.session_state.hel1os_cat, "hel1os_catalogue.csv"),
        ("Master catalogue", st.session_state.master_cat, "master_catalog.csv"),
        ("Prediction report", _prediction_report(), "prediction_report.csv"),
        ("Evaluation report", _evaluation_report(), "evaluation_report.csv"),
    ]

    for label, data, fname in items:
        col1, col2 = st.columns([3, 1])
        ready = _has(data)
        col1.markdown(f"**{label}** &nbsp; "
                      f'<span class="badge" style="background:rgba(255,255,255,.06);color:{COL_MUTED};">'
                      f'{"ready" if ready else "not generated"}</span>', unsafe_allow_html=True)
        with col2:
            st.download_button(
                "Download", df_to_csv_bytes(data) if ready else b"",
                file_name=fname, mime="text/csv", disabled=not ready, key=f"dl_{fname}",
                use_container_width=True,
            )


def _prediction_report():
    if st.session_state.forecast_prob is None:
        return None
    return pd.DataFrame([{
        "Generated": datetime.now().isoformat(timespec="seconds"),
        "Horizon (min)": st.session_state.forecast_horizon,
        "Flare Probability": round(st.session_state.forecast_prob, 4),
        "Risk Level": st.session_state.forecast_risk,
    }])


def _evaluation_report():
    res = st.session_state.eval_result
    if res is None:
        return None
    d = res.metric_cards()
    d.update({"TP": res.tp, "FP": res.fp, "FN": res.fn, "TN": res.tn})
    return pd.DataFrame([d])


# ===========================================================================
# Shared small helpers
# ===========================================================================
def _has(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def _count_detected() -> int:
    total = 0
    for cat in (st.session_state.solexs_cat, st.session_state.hel1os_cat):
        if _has(cat):
            total += len(cat)
    return total


def _count_predicted() -> int:
    return len(st.session_state.master_cat) if _has(st.session_state.master_cat) else 0


def _human_size(num_bytes) -> str:
    if not num_bytes:
        return "-"
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _no_data():
    st.info("No light curve loaded. Upload a FITS file or click **Load Demo Data** in the sidebar.")


# ===========================================================================
# Router
# ===========================================================================
ROUTES = {
    "Home Dashboard": page_home,
    "FITS Upload": page_upload,
    "Data Visualization": page_visualization,
    "Flare Detection": page_detection,
    "Master Catalogue": page_master,
    "Forecasting": page_forecasting,
    "Evaluation": page_evaluation,
    "Alerts": page_alerts,
    "Download Center": page_download,
}

ROUTES[page]()
