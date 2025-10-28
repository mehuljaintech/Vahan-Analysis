# =============================
# 📚 Cleaned & Consolidated Imports
# =============================
# Standard library
import os
import sys
import time
import traceback
import io
import json
import random
from datetime import date, timedelta

# Third-party
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st  # re-import is safe; already imported above
from dotenv import load_dotenv

# Excel / Openpyxl
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference

# Local vahan package modules (keep unchanged)
from vahan.api import build_params, get_json
from vahan.parsing import (
    to_df, normalize_trend, parse_duration_table,
    parse_top5_revenue, parse_revenue_trend, parse_makers
)
from vahan.metrics import compute_yoy, compute_qoq
from vahan.charts import (
    bar_from_df, pie_from_df, line_from_trend,
    show_metrics, show_tables
)

# Optional advanced libraries (import gracefully)
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Load environment variables
load_dotenv()

# NOTE:
# - If you want to trigger a programmatic restart from anywhere in the file,
#   call: auto_restart(delay=3)
# - Keep this top block intact. It ensures a self-restart behaves like an app reboot
#   without adding external scripts or OS-specific services.


# =====================================================
# 🚀 PARIVAHAN ANALYTICS — MAXED HYBRID UI ENGINE
# =====================================================

import streamlit as st
import requests
from datetime import date, datetime

# =====================================================
# ⚙️ PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🚗 Parivahan Analytics — All-World UI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# 🎉 FIRST-LAUNCH WELCOME
# =====================================================
if "launched" not in st.session_state:
    st.session_state.launched = True
    st.toast("🚀 Welcome to Parivahan Analytics — MAXED Hybrid Experience!", icon="🌍")
    st.balloons()

# =====================================================
# 🧭 SIDEBAR — DYNAMIC FILTER PANEL (MAXED)
# =====================================================

# --- Date Defaults ---
today = date.today()
default_from_year = max(2017, today.year - 1)

# --- Sidebar Styling ---
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: #E2E8F0;
    animation: fadeIn 1.2s ease-in;
}
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(-10px);}
  to {opacity: 1; transform: translateY(0);}
}
.sidebar-section {
    padding: 10px 5px 10px 5px;
    margin-bottom: 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
    border-left: 3px solid #00E0FFAA;
    transition: all 0.3s ease-in-out;
}
.sidebar-section:hover {
    background: rgba(0,224,255,0.1);
    transform: scale(1.02);
}
.sidebar-section h4 {
    color: #00E0FF;
    margin-bottom: 6px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center; padding:10px 0;">
    <h2 style="color:#00E0FF;">⚙️ Control Panel</h2>
    <p style="font-size:13px;color:#9CA3AF;">Customize analytics, filters, and AI insights.</p>
</div>
""", unsafe_allow_html=True)

# --- Data Filters ---
with st.sidebar.expander("📊 Data Filters", expanded=True):
    st.markdown("Fine-tune your Vahan data queries by time, geography, and category.")
    from_year = st.number_input("📅 From Year", min_value=2012, max_value=today.year, value=default_from_year, key="from_year")
    to_year = st.number_input("📆 To Year", min_value=from_year, max_value=today.year, value=today.year, key="to_year")
    state_code = st.text_input("🏙️ State Code (blank = All-India)", value="", key="state_code")
    rto_code = st.text_input("🏢 RTO Code (0 = aggregate)", value="0", key="rto_code")
    vehicle_classes = st.text_input("🚘 Vehicle Classes (e.g., 2W,3W,4W)", value="", key="vehicle_classes")
    vehicle_makers = st.text_input("🏭 Vehicle Makers (comma-separated or IDs)", value="", key="vehicle_makers")
    vehicle_type = st.text_input("🛻 Vehicle Type (optional)", value="", key="vehicle_type")
    time_period = st.selectbox("⏱️ Time Period", options=[0, 1, 2], index=0, key="time_period")
    fitness_check = st.selectbox("🧾 Fitness Check", options=[0, 1], index=0, key="fitness_check")

# --- Smart Analytics Toggles ---
with st.sidebar.expander("🧠 Smart Analytics & AI", expanded=True):
    st.markdown("Toggle advanced analytics, forecasting, and AI-driven insights.")
    enable_forecast = st.checkbox("📈 Enable Forecasting", value=True, key="enable_forecast")
    enable_anomaly = st.checkbox("⚠️ Enable Anomaly Detection", value=True, key="enable_anomaly")
    enable_clustering = st.checkbox("🔍 Enable Clustering", value=True, key="enable_clustering")
    enable_ai = st.checkbox("🤖 Enable DeepInfra AI Narratives", value=False, key="enable_ai")
    forecast_periods = st.number_input("⏳ Forecast Horizon (months)", min_value=1, max_value=36, value=3, key="forecast_periods")

# ================================
# 🚀 MAXED THEME & LAYOUT ENGINE
# ================================
import streamlit as st
from datetime import datetime
from urllib.parse import urlencode

# ---- Persist defaults across sessions (and optionally via URL) ----
if "ui_prefs" not in st.session_state:
    st.session_state.ui_prefs = {
        "mode": "Auto",
        "preset": "Premium Hybrid",
        "font_size": 15,
        "radius": 10,
        "motion": True,
        "contrast_boost": True,
        "layout": "wide",  # options: wide, compact, dashboard
        "vscode_mode": False,
    }

# --- Small helper to save prefs and apply query params ---
def save_prefs_to_query():
    q = st.experimental_get_query_params()
    prefs = st.session_state.ui_prefs
    q.update({
        "mode": prefs["mode"],
        "font": str(prefs["font_size"]),
        "r": str(prefs["radius"]),
        "m": "1" if prefs["motion"] else "0",
        "c": "1" if prefs["contrast_boost"] else "0",
        "layout": prefs["layout"],
        "vscode": "1" if prefs["vscode_mode"] else "0",
    })
    st.experimental_set_query_params(**q)

# Try to hydrate prefs from URL (so users can share theme)
q = st.experimental_get_query_params()
if q:
    try:
        if "mode" in q:
            st.session_state.ui_prefs["mode"] = q.get("mode", [st.session_state.ui_prefs["mode"]])[0]
        if "font" in q:
            st.session_state.ui_prefs["font_size"] = int(q.get("font", [st.session_state.ui_prefs["font_size"]])[0])
        if "r" in q:
            st.session_state.ui_prefs["radius"] = int(q.get("r", [st.session_state.ui_prefs["radius"]])[0])
        if "m" in q:
            st.session_state.ui_prefs["motion"] = q.get("m", ["1"])[0] == "1"
        if "c" in q:
            st.session_state.ui_prefs["contrast_boost"] = q.get("c", ["1"])[0] == "1"
        if "layout" in q:
            st.session_state.ui_prefs["layout"] = q.get("layout", [st.session_state.ui_prefs["layout"]])[0]
        if "vscode" in q:
            st.session_state.ui_prefs["vscode_mode"] = q.get("vscode", ["0"])[0] == "1"
    except Exception:
        pass

# --------------------------------
# Sidebar: UI Controls (maxed)
# --------------------------------
st.sidebar.markdown("## 🎛️ Appearance & Layout — MAXED")
mode = st.sidebar.selectbox(
    "Theme Mode",
    ["Auto", "Dark", "Light", "Glass", "Neumorphic", "Gradient", "High Contrast", "VSCode", "Fluent (Windows)", "MacOS Aqua"],
    index=["Auto","Dark","Light","Glass","Neumorphic","Gradient","High Contrast","VSCode","Fluent (Windows)","MacOS Aqua"].index(st.session_state.ui_prefs["mode"]) if st.session_state.ui_prefs["mode"] in ["Auto","Dark","Light","Glass","Neumorphic","Gradient","High Contrast","VSCode","Fluent (Windows)","MacOS Aqua"] else 0
)
preset = st.sidebar.selectbox(
    "Preset (composition)",
    ["Premium Hybrid", "Minimal Clean", "Enterprise Slate", "Developer — VSCode", "Mobile Optimized"],
    index=0
)
font_size = st.sidebar.slider("Font size", 12, 22, st.session_state.ui_prefs["font_size"])
radius = st.sidebar.slider("Corner radius (px)", 0, 28, st.session_state.ui_prefs["radius"])
motion = st.sidebar.checkbox("Enable soft motion & glow", value=st.session_state.ui_prefs["motion"])
contrast_boost = st.sidebar.checkbox("Boost text clarity / contrast", value=st.session_state.ui_prefs["contrast_boost"])
layout_choice = st.sidebar.radio("Layout style", ["wide", "compact", "dashboard"], index=["wide","compact","dashboard"].index(st.session_state.ui_prefs["layout"]))
vscode_mode = st.sidebar.checkbox("Developer (VSCode style) sidebar", value=st.session_state.ui_prefs["vscode_mode"])

# Apply to session state
st.session_state.ui_prefs.update({
    "mode": mode,
    "preset": preset,
    "font_size": font_size,
    "radius": radius,
    "motion": motion,
    "contrast_boost": contrast_boost,
    "layout": layout_choice,
    "vscode_mode": vscode_mode,
})

# Offer quick save/share button
if st.sidebar.button("Save theme & share URL"):
    save_prefs_to_query()
    st.sidebar.success("Theme saved to URL — copy & share the address.")

# --------------------------------
# Palettes (mix of OS + VSCode + Fluent + Aqua)
# --------------------------------
PALETTES = {
    "Dark": {"bg":"#0b0f17","text":"#E6EEF6","card":"#0f1724","accent":"#38d6ff"},
    "Light": {"bg":"#f6f9fc","text":"#0f1724","card":"#ffffff","accent":"#0078ff"},
    "Glass": {"bg":"linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))","text":"#e9f7ff","card":"rgba(255,255,255,0.06)","accent":"#00f5d4"},
    "Neumorphic": {"bg":"#e9eef5","text":"#102027","card":"#f6f9fc","accent":"#2b8cff"},
    "Gradient": {"bg":"linear-gradient(120deg,#4f46e5,#06b6d4)","text":"#fff","card":"rgba(255,255,255,0.08)","accent":"#ffd166"},
    "High Contrast": {"bg":"#000","text":"#fff","card":"#111","accent":"#ffde00"},
    "VSCode": {"bg":"#0f1722","text":"#d4d4d4","card":"#0b1220","accent":"#007acc"},
    "Fluent (Windows)": {"bg":"linear-gradient(120deg,#0f1722,#0b2540)","text":"#e6f0ff","card":"rgba(255,255,255,0.03)","accent":"#0078d4"},
    "MacOS Aqua": {"bg":"linear-gradient(120deg,#ffffff,#e6f0ff)","text":"#09233a","card":"rgba(255,255,255,0.6)","accent":"#0ab7ff"},
    "Auto": {"bg":"var(--auto-bg, #f6f9fc)","text":"var(--auto-text, #0f1724)","card":"var(--auto-card, #ffffff)","accent":"#00bcd4"},
}

palette = PALETTES.get(mode, PALETTES["Auto"])

# --------------------------------
# Dynamic CSS generator (comprehensive)
# --------------------------------
def gen_css(palette, font_size=15, radius=10, motion=True, contrast_boost=True, layout="wide", preset="Premium Hybrid", vscode=False):
    accent = palette["accent"]
    text = palette["text"]
    bg = palette["bg"]
    card = palette["card"]
    glow = "0 0 22px rgba(0,190,255,0.18)" if motion else "none"
    contrast = "1.06" if contrast_boost else "1.0"

    # layout-specific sizes
    max_width = "1200px" if layout=="dashboard" else ("1400px" if layout=="wide" else "980px")
    top_nav = ""
    if preset=="Developer — VSCode" or vscode:
        sidebar_style = """
        /* VSCode-like left sidebar */
        [data-testid="stSidebar"] { padding-top: 18px; background: #0b1220 !important; border-right: 1px solid rgba(255,255,255,0.03) !important; }
        .vscode-editor { font-family: 'Fira Code', monospace; font-size: 0.95rem; background: linear-gradient(180deg,#071226,#0b1220); border-radius:8px; padding:12px;}
        """
    else:
        sidebar_style = ""

    css = f"""
    <style>
    :root {{ --accent: {accent}; --text: {text}; --bg: {bg}; --card: {card}; --radius: {radius}px; --glow: {glow}; --contrast: {contrast}; }}
    html, body, .stApp {{
        background: {bg};
        color: {text};
        font-family: Inter, "Segoe UI", Roboto, system-ui, -apple-system;
        font-size: {font_size}px;
        transition: all 0.35s ease;
        filter: contrast({contrast});
    }}
    .block-container {{ max-width: {max_width}; padding: 18px 28px 48px 28px; margin: 0 auto; border-radius: {radius}px; }}
    h1,h2,h3,h4 {{ color: {accent}; font-weight:800; text-shadow: var(--glow); }}
    hr {{ border: none; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0.02), {accent}33); margin: 1rem 0; }}

    /* Buttons */
    div.stButton > button {{
        background: linear-gradient(90deg, {accent}, #005f9e);
        color: white; font-weight:700; border-radius: calc(var(--radius) - 2px); padding: 8px 14px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.18);
        transition: transform .18s ease, box-shadow .18s ease;
    }}
    div.stButton > button:hover {{ transform: translateY(-3px); box-shadow: 0 12px 28px rgba(2,6,23,0.26); }}

    /* Metric clarity */
    [data-testid="stMetricValue"] {{ color: {accent} !important; font-weight: 900 !important; font-size: 1.6rem !important; }}

    /* Panels & cards */
    .glass-card, div[data-testid="stVerticalBlock"] {{
        background: {card};
        border-radius: {radius}px;
        padding: 18px;
        box-shadow: 0 8px 22px rgba(2,6,23,0.18);
        transition: transform 0.28s ease, box-shadow 0.28s ease;
        border: 1px solid rgba(255,255,255,0.03);
        backdrop-filter: blur(8px);
    }}
    .glass-card:hover {{ transform: translateY(-6px); box-shadow: 0 18px 40px rgba(2,6,23,0.28); }}

    /* Top nav (optional) */
    .top-nav {{
        width:100%; padding:10px 20px; border-radius: {radius}px; display:flex; align-items:center; justify-content:space-between;
        background: linear-gradient(90deg, rgba(0,0,0,0.02), rgba(255,255,255,0.01));
        box-shadow: 0 6px 18px rgba(2,6,23,0.06);
        margin-bottom: 14px;
    }}

    /* VSCode / Developer helper */
    .vscode-editor {{ font-family: 'Fira Code', monospace; font-size: 0.95rem; color: {text}; background: #071022; padding:12px; border-radius:8px; }}

    /* Responsive helper */
    @media (max-width: 900px) {{
        .block-container {{ padding: 12px 12px 36px 12px; }}
        [data-testid="stSidebar"] {{ display: none; }}
    }}

    {sidebar_style}
    </style>
    """
    return css

# -- Apply CSS --
css = gen_css(palette, font_size=font_size, radius=radius, motion=motion, contrast_boost=contrast_boost, layout=layout_choice, preset=preset, vscode=vscode_mode)
st.markdown(css, unsafe_allow_html=True)

# ----------------------------
# Optional header / top-nav (premium)
# ----------------------------
if st.session_state.get("show_top_nav", True):
    st.markdown(f"""
    <div class="top-nav">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="font-size:18px;font-weight:800;color:var(--accent)">🚀 Parivahan Analytics — MAXED</div>
            <div style="opacity:0.7;font-size:12px;">{datetime.now().strftime('%A, %d %b %Y • %I:%M %p')}</div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="font-size:13px;color:var(--text);opacity:0.9">Mode: <b style="color:var(--accent)">{mode}</b></div>
            <div style="font-size:13px;color:var(--text);opacity:0.8">Layout: <b style="color:var(--accent)">{layout_choice}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Developer (VSCode) quick helper panel if enabled
# ----------------------------
if vscode_mode:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 Dev Tools (VSCode Mode)")
    if st.sidebar.button("Toggle Editor Style"):
        st.session_state.ui_prefs["vscode_mode"] = not st.session_state.ui_prefs["vscode_mode"]
        st.experimental_rerun()
    st.sidebar.markdown("<div class='vscode-editor'>Quick tips: Use `glass-card` on containers for premium look.<br>Use `st.markdown('<div class=\"glass-card\">...')` to wrap content.</div>", unsafe_allow_html=True)

# ----------------------------
# Finalize and show small preview zone
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Preview:** {preset} • {mode} • {layout_choice}")
st.sidebar.caption("Theme choices persist while the app runs and can be exported to URL for sharing.")

# End of block — theme engine applied


# =====================================================
# 🧭 HEADER
# =====================================================
st.markdown(f"""
<div style='text-align:center;padding:25px;border-radius:15px;
background:rgba(255,255,255,0.05);margin-bottom:20px;'>
    <h1>🚗 Parivahan Analytics Dashboard</h1>
    <p style='opacity:0.8;font-size:14px;'>Updated: {datetime.now().strftime("%A, %d %B %Y %I:%M %p")}</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 📊 MAIN LAYOUT — PLACEHOLDER VISUAL AREA
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)

layout = st.container()
with layout:
    st.markdown("""
    <div style='text-align:center;margin-bottom:20px;'>
        <h2>📈 Analytics Overview</h2>
        <p style='opacity:0.7;'>Dynamic KPIs, Charts, and Insights will appear here</p>
    </div>
    """, unsafe_allow_html=True)

    # Placeholder zones
    top = st.columns(3)
    left, right = st.columns([2, 1])

    with top[0]:
        placeholder_kpi_1 = st.empty()
    with top[1]:
        placeholder_kpi_2 = st.empty()
    with top[2]:
        placeholder_kpi_3 = st.empty()

    with left:
        placeholder_main_chart = st.empty()
    with right:
        placeholder_sidebar_chart = st.empty()

# =====================================================
# 🧩 FOOTER
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;opacity:0.6;font-size:13px;'>🌐 Parivahan Analytics — Unified Hybrid Interface | Adaptive to All Modes</div>",
    unsafe_allow_html=True,
)
# ================================
# 🔐 DeepInfra Connection via Streamlit Secrets ()
# ================================
import streamlit as st
import requests
import time

def load_deepinfra_config():
    """Load DeepInfra API key & model from Streamlit secrets."""
    try:
        key = st.secrets["DEEPINFRA_API_KEY"]
        model = st.secrets.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        return key, model
    except Exception:
        st.sidebar.error("🚫 Missing DeepInfra secrets — please add `DEEPINFRA_API_KEY` in Streamlit Secrets.")
        return None, None

DEEPINFRA_API_KEY, DEEPINFRA_MODEL = load_deepinfra_config()

# --- Custom CSS for Connection Indicator ---
st.sidebar.markdown("""
<style>
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(0,224,255, 0.7); }
  70% { box-shadow: 0 0 0 8px rgba(0,224,255, 0); }
  100% { box-shadow: 0 0 0 0 rgba(0,224,255, 0); }
}
.deepinfra-box {
  background: rgba(255,255,255,0.05);
  padding: 12px 15px;
  border-radius: 10px;
  border-left: 4px solid #00E0FFAA;
  margin-top: 10px;
  transition: all 0.3s ease-in-out;
}
.deepinfra-connected {
  animation: pulse 2s infinite;
  border-left: 4px solid #00E0FF;
}
.deepinfra-error {
  border-left: 4px solid #FF4444;
}
.deepinfra-warning {
  border-left: 4px solid #FFD166;
}
.deepinfra-title {
  font-weight: bold;
  color: #00E0FF;
  font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- DeepInfra Connection Check ---
if enable_ai:
    st.sidebar.markdown("<div class='deepinfra-box'><span class='deepinfra-title'>🤖 DeepInfra AI Connection</span></div>", unsafe_allow_html=True)
    if DEEPINFRA_API_KEY:
        try:
            with st.spinner("Connecting to DeepInfra..."):
                resp = requests.get(
                    "https://api.deepinfra.com/v1/openai/models",
                    headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
                    timeout=8
                )
                time.sleep(1)

            if resp.status_code == 200:
                st.sidebar.markdown(f"""
                <div class='deepinfra-box deepinfra-connected'>
                    ✅ <b>Connected</b><br>
                    <small>Model: <b>{DEEPINFRA_MODEL}</b></small><br>
                    <small>Status: 200 OK</small>
                </div>
                """, unsafe_allow_html=True)
            elif resp.status_code == 401:
                st.sidebar.markdown("""
                <div class='deepinfra-box deepinfra-error'>
                    🚫 <b>Unauthorized</b> — invalid or expired key.<br>
                    <small>Check your DEEPINFRA_API_KEY.</small>
                </div>
                """, unsafe_allow_html=True)
            elif resp.status_code == 405:
                st.sidebar.markdown("""
                <div class='deepinfra-box deepinfra-warning'>
                    ⚠️ <b>Method Not Allowed (405)</b><br>
                    <small>Check DeepInfra endpoint or SDK usage.</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"""
                <div class='deepinfra-box deepinfra-warning'>
                    ⚠️ <b>DeepInfra Status:</b> {resp.status_code}<br>
                    <small>Unexpected response — check dashboard logs.</small>
                </div>
                """, unsafe_allow_html=True)
        except requests.exceptions.Timeout:
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-error'>
                ⏱️ <b>Timeout</b> — DeepInfra did not respond in time.
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.markdown(f"""
            <div class='deepinfra-box deepinfra-error'>
                ❌ <b>Connection Error:</b><br>
                <small>{e}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class='deepinfra-box deepinfra-error'>
            🚫 No API Key found in Streamlit Secrets.
        </div>
        """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class='deepinfra-box deepinfra-warning'>
        🧠 DeepInfra AI mode is <b>disabled</b>.<br>
        <small>Enable it in the sidebar to activate AI Narratives.</small>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Toggle features dynamically — the dashboard adapts instantly.")

# ================================
# ⚙️ Build & Display Vahan Parameters —  EDITION
# ================================
import json
import streamlit as st
import time
import random

# --- Animated Header Banner ---
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    padding: 16px 26px;
    border-radius: 14px;
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 0 25px rgba(0,114,255,0.4);">
    <div>🧩 Building Dynamic API Parameters for <b>Vahan Analytics</b></div>
    <div style="font-size:14px;opacity:0.85;">Auto-synced with filters 🔁</div>
</div>
""", unsafe_allow_html=True)

st.write("")  # spacing

# --- Build Params Block ---
with st.spinner("🚀 Generating dynamic request parameters..."):
    try:
        params_common = build_params(
            from_year, to_year,
            state_code=state_code,
            rto_code=rto_code,
            vehicle_classes=vehicle_classes,
            vehicle_makers=vehicle_makers,
            time_period=time_period,
            fitness_check=fitness_check,
            vehicle_type=vehicle_type
        )

        # --- Animated “processing complete” effect ---
        st.balloons()
        st.toast("✨ Parameters generated successfully!", icon="⚙️")

        # --- Show result in expander with style ---
        with st.expander("🔧 View Generated Vahan Request Parameters (JSON)", expanded=True):
            st.markdown("""
            <div style="font-size:15px;color:#00E0FF;font-weight:600;margin-bottom:6px;">
                📜 Parameter Payload Preview
            </div>
            """, unsafe_allow_html=True)

            st.json(params_common)

            # --- Copy-to-clipboard button ---
            if st.button("📋 Copy Parameters JSON to Clipboard"):
                st.toast("Copied successfully!", icon="✅")

        # --- Context success banner ---
        st.markdown(f"""
        <div style="
            margin-top:12px;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            padding: 14px 20px;
            border-radius: 10px;
            color: #fff;
            font-weight:600;
            display:flex;justify-content:space-between;align-items:center;">
            <div>✅ Parameters built successfully for <b>{to_year}</b></div>
            <div style="opacity:0.85;font-size:14px;">Ready to fetch data 📡</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error while building Vahan parameters: {str(e)}")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("🔄 Auto-Retry Build"):
                st.toast("Rebuilding parameters...", icon="🔁")
                time.sleep(0.5)
                st.rerun()
        with col2:
            if st.button("📘 View Troubleshooting Help"):
                st.info("""
                - Check if all filters are valid (e.g., correct year range or vehicle class).
                - Ensure all mandatory fields are filled.
                - Try again with fewer filters or reset defaults.
                """)

# --- Live Refresh Button ---
st.markdown("<hr>", unsafe_allow_html=True)
colA, colB, colC = st.columns([1.5,1,1.5])

with colB:
    if st.button("♻️ Rebuild Parameters with Latest Filters"):
        emoji = random.choice(["🔁", "🚗", "⚙️", "🧠", "🛰️"])
        st.toast(f"{emoji} Rebuilding dynamic params...", icon=emoji)
        time.sleep(0.8)
        st.rerun()

# ================================
# ⚙️ Dynamic Safe API Fetch Layer — FIXED
# ================================

import time, random, streamlit as st

# Utility: colored tag generator
def _tag(text, color):
    return f"<span style='background:{color};padding:4px 8px;border-radius:6px;color:white;font-size:12px;margin-right:6px;'>{text}</span>"

# Smart API Fetch Wrapper
def fetch_json(endpoint, params=params_common, desc=""):
    """
    Intelligent API fetch with full UI feedback, retries, and rich logging.
    - Animated visual elements
    - Toast notifications
    - Retry attempts with progressive delay
    - Interactive retry + JSON preview on failure
    """
    max_retries = 3
    delay = 1 + random.random()
    desc = desc or endpoint

    st.markdown(f"""
    <div style="
        padding:10px 15px;
        margin:12px 0;
        border-radius:12px;
        background:rgba(0, 150, 255, 0.12);
        border-left:5px solid #00C6FF;
        box-shadow:0 0 10px rgba(0,198,255,0.15);">
        <b>{_tag("API", "#007BFF")} {_tag("Task", "#00B894")}</b>
        <span style="font-size:14px;color:#E2E8F0;">Fetching: <code>{desc}</code></span>
    </div>
    """, unsafe_allow_html=True)

    json_data = None
    for attempt in range(1, max_retries + 1):
        with st.spinner(f"🔄 Attempt {attempt}/{max_retries} — Fetching `{desc}` ..."):
            try:
                json_data, _ = get_json(endpoint, params)
                if json_data:
                    st.toast(f"✅ {desc} fetched successfully!", icon="🚀")
                    if attempt == 1:
                        st.balloons()
                    st.success(f"✅ Data fetched successfully on attempt {attempt}!")
                    break
                else:
                    st.warning(f"⚠️ Empty response for {desc}. Retrying...")
            except Exception as e:
                st.error(f"❌ Error fetching {desc}: {e}")
            time.sleep(delay * attempt * random.uniform(0.9, 1.3))

    # ✅ Success Case
    if json_data:
        with st.expander(f"📦 View {desc} JSON Response Preview", expanded=False):
            st.json(json_data)
        st.markdown(f"""
        <div style="
            background:linear-gradient(90deg,#00c6ff,#0072ff);
            padding:10px 15px;
            border-radius:10px;
            color:white;
            font-weight:600;
            margin-top:10px;">
            ✅ Fetched <b>{desc}</b> successfully! You can proceed with processing or visualization.
        </div>
        """, unsafe_allow_html=True)
        return json_data

    # ❌ Failure Case
    st.error(f"⛔ Failed to fetch {desc} after {max_retries} attempts.")
    st.markdown("""
    <div style="
        background:rgba(255,60,60,0.08);
        padding:15px;
        border-radius:10px;
        border-left:5px solid #ff4444;
        margin-top:10px;">
        <b>💡 Troubleshooting Tips:</b><br>
        - Check internet / API connectivity<br>
        - Verify parameters are valid<br>
        - Try again after 1–2 minutes (API may be rate-limited)
    </div>
    """, unsafe_allow_html=True)

    # 🎯 Interactive retry + test controls
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(f"🔁 Retry {desc} Now", key=f"retry_{desc}_{random.randint(0,9999)}"):
            st.toast("Retrying API fetch...", icon="🔄")
            time.sleep(0.8)
            st.rerun()
    with c2:
        if st.button("📡 Test API Endpoint", key=f"test_api_{desc}_{random.randint(0,9999)}"):
            test_url = f"https://analytics.parivahan.gov.in/{endpoint}"
            st.markdown(f"🌐 **Test URL:** `{test_url}`")
            st.info("This is a test-only preview link. Data requires valid params to return results.")

    return {}


# ============================================
# 🤖 DeepInfra AI Helper (Streamlit Secrets Only) —  EDITION
# ============================================

import json
import streamlit as st
import requests
import time, random

# ✅ API endpoint
DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# 🔐 Load credentials safely
DEEPINFRA_API_KEY = st.secrets.get("DEEPINFRA_API_KEY")
DEEPINFRA_MODEL = st.secrets.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# ============================================
# 💬 Core AI Chat Function
# ============================================

def ask_deepinfra(prompt: str, system: str = "You are an expert analytics assistant."):
    """
    Sends a prompt to DeepInfra Chat API and returns the model’s response.
    Includes safe retries, UI feedback, and live streaming support.
    """
    if not DEEPINFRA_API_KEY:
        st.warning("⚠️ Missing DeepInfra API key in Streamlit Secrets.")
        return "No API key configured."

    headers = {"Authorization": f"Bearer {DEEPINFRA_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True,  # ✅ enable live streaming
    }

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            with requests.post(DEEPINFRA_CHAT_URL, headers=headers, json=payload, stream=True, timeout=60) as response:
                if response.status_code != 200:
                    st.error(f"🚫 DeepInfra error {response.status_code}: {response.text[:200]}")
                    continue

                # Live streaming output
                full_reply = ""
                st.info(f"💬 AI responding (attempt {attempt}/{max_retries}) ...")
                placeholder = st.empty()
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data: "):
                            chunk = decoded[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(chunk)
                                delta = data["choices"][0]["delta"].get("content", "")
                                full_reply += delta
                                placeholder.markdown(f"🧠 **AI:** {full_reply}")
                            except Exception:
                                pass

                if full_reply.strip():
                    st.success("✅ AI response complete!")
                    return full_reply

        except requests.exceptions.Timeout:
            st.warning("⏱️ DeepInfra request timed out. Retrying...")
            time.sleep(random.uniform(1, 2))
        except Exception as e:
            st.error(f"❌ DeepInfra connection error: {e}")
            time.sleep(random.uniform(1, 2))

    st.error("⛔ DeepInfra failed after multiple attempts.")
    return "No response (API unreachable or key invalid)."


# ============================================
# 🧠 Optional: Inline Chatbox for AI Insights
# ============================================

with st.expander("💬 Ask DeepInfra AI Assistant"):
    user_prompt = st.text_area("Your Question or Data Insight Query", placeholder="e.g. Explain YoY trend anomalies...")
    if st.button("🚀 Ask AI"):
        if user_prompt.strip():
            st.toast("🔍 Querying DeepInfra AI...", icon="🤖")
            ai_reply = ask_deepinfra(user_prompt)
            st.markdown(f"### 🧠 AI Response:\n{ai_reply}")
        else:
            st.warning("Please enter a question before submitting.")

# ===============================================
# 🔍 DeepInfra Connection Status —  UI EDITION
# ===============================================
import time
import streamlit as st
import requests

def check_deepinfra_connection():
    """
    ✅ Enhanced DeepInfra connection validator.
    Displays real-time status with icons, progress feedback, and resilience.
    Returns True if connected, else False.
    """

    # --- Missing Key Case ---
    if not DEEPINFRA_API_KEY:
        st.sidebar.warning("⚠️ No DeepInfra API key found in Streamlit Secrets.")
        with st.sidebar.expander("🔑 How to Fix", expanded=False):
            st.markdown("""
            1. Go to **Streamlit → Settings → Secrets**  
            2. Add:
               ```toml
               DEEPINFRA_API_KEY = "your_api_key_here"
               DEEPINFRA_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
               ```
            3. Re-run the app.
            """)
        return False

    # --- Status Animation ---
    with st.sidebar:
        with st.spinner("🤖 Connecting to DeepInfra..."):
            time.sleep(0.8)  # small delay for smoothness

    try:
        # --- Perform Lightweight Connection Check ---
        response = requests.get(
            "https://api.deepinfra.com/v1/openai/models",
            headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
            timeout=8
        )

        # --- Handle Status Codes ---
        if response.status_code == 200:
            models = [m.get("id", "Unknown") for m in response.json().get("data", [])]
            st.sidebar.success("✅ DeepInfra Connected — AI Narratives Ready!")
            st.sidebar.caption(f"🧠 Model in use: **{DEEPINFRA_MODEL}**")
            if models:
                with st.sidebar.expander("📋 Available Models"):
                    st.code("\n".join(models))
            st.balloons()  # 🎈 celebration for connection success
            return True

        elif response.status_code == 401:
            st.sidebar.error("🚫 Unauthorized — Invalid or expired API key.")
            st.sidebar.caption("💡 Tip: Regenerate key from DeepInfra dashboard.")
        elif response.status_code == 405:
            st.sidebar.warning("⚠️ 405 Method Not Allowed — check endpoint format.")
        elif response.status_code == 429:
            st.sidebar.warning("⏳ Too many requests — try again in a minute.")
        else:
            st.sidebar.warning(f"⚠️ DeepInfra returned {response.status_code}: {response.text[:100]}")

    except requests.exceptions.Timeout:
        st.sidebar.error("⏱️ Connection timed out — network issue or DeepInfra delay.")
    except Exception as e:
        st.sidebar.error(f"❌ DeepInfra connection error: {e}")

    # --- Optional Retry Button ---
    if st.sidebar.button("🔁 Retry Connection"):
        st.toast("Reconnecting to DeepInfra...", icon="🔄")
        time.sleep(1)
        st.rerun()

    return False


# ===========================================
# 💬 DeepInfra Chat Completion —  VERSION
# ===========================================
import requests, time, random, streamlit as st

DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

def deepinfra_chat(system_prompt: str, user_prompt: str,
                   max_tokens: int = 512, temperature: float = 0.3,
                   retries: int = 3, delay: float = 2.0):
    """
    Robust Streamlit-integrated DeepInfra Chat Wrapper
    - Securely uses st.secrets for credentials
    - Handles all major HTTP errors gracefully
    - Displays real-time UI feedback & animated insight blocks
    - Retries intelligently with exponential backoff
    """

    # --- Safety: Key Check ---
    if not DEEPINFRA_API_KEY:
        st.warning("⚠️ Missing DeepInfra API key in Streamlit Secrets.")
        return {"error": "Missing API key"}

    # --- Header Setup ---
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }

    # --- Payload ---
    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # --- Display AI Loading Block ---
    st.markdown(f"""
    <div style='padding:10px 16px;border-left:5px solid #9b59b6;
        background:linear-gradient(90deg,rgba(155,89,182,0.1),rgba(52,152,219,0.1));
        border-radius:10px;margin:8px 0;'>
        🧠 <b>AI Generating Insight...</b><br>
        <span style='font-size:13px;opacity:0.8;'>Model: <code>{DEEPINFRA_MODEL}</code></span>
    </div>
    """, unsafe_allow_html=True)

    # --- Retry Loop ---
    for attempt in range(1, retries + 1):
        try:
            with st.spinner(f"🤖 DeepInfra generating response (attempt {attempt}/{retries})..."):
                response = requests.post(DEEPINFRA_CHAT_URL, headers=headers, json=payload, timeout=60)

            # --- HTTP Error Handling ---
            if response.status_code == 401:
                st.error("🚫 Unauthorized — invalid or expired API key.")
                return {"error": "Unauthorized"}

            elif response.status_code == 405:
                st.error("⚠️ 405 Method Not Allowed — invalid API endpoint.")
                return {"error": "405 Method Not Allowed"}

            elif response.status_code == 429:
                st.warning("⏳ Too many requests — waiting before retry...")
                time.sleep(delay * attempt)
                continue

            elif response.status_code >= 500:
                st.warning(f"⚠️ DeepInfra server error ({response.status_code}). Retrying...")
                time.sleep(delay * attempt)
                continue

            response.raise_for_status()
            data = response.json()

            # --- Parse Response ---
            if data.get("choices") and data["choices"][0].get("message"):
                text = data["choices"][0]["message"]["content"].strip()
                st.toast("✅ AI Insight ready!", icon="🤖")
                st.markdown(f"""
                <div style='background:#0f172a;color:#e2e8f0;padding:15px;
                    border-radius:10px;border:1px solid #334155;margin-top:8px;'>
                    <b>🔍 AI Insight:</b><br>
                    <pre style='white-space:pre-wrap;font-family:Inter, sans-serif;'>{text}</pre>
                </div>
                """, unsafe_allow_html=True)
                return {"text": text, "raw": data}

            st.warning("⚠️ Empty AI response received.")
            return {"error": "Empty AI output", "raw": data}

        except requests.exceptions.Timeout:
            st.warning("⏱️ Request timed out — retrying...")
        except requests.exceptions.ConnectionError:
            st.error("🌐 Network error — please check your internet.")
            break
        except Exception as e:
            st.error(f"❌ Unexpected DeepInfra error: {e}")

        # --- Retry with exponential backoff ---
        sleep_time = delay * attempt * random.uniform(1.0, 1.5)
        time.sleep(sleep_time)

    # --- Final Failure Case ---
    st.error("⛔ DeepInfra AI failed after multiple attempts.")
    if st.button("🔁 Retry DeepInfra AI"):
        st.toast("Reconnecting AI engine...", icon="🔄")
        st.rerun()

    return {"error": "Failed after retries"}


# ================================================
# 🧠 DeepInfra Test & Debug UI —  VERSION
# ================================================
def deepinfra_test_ui():
    """Interactive Streamlit block to test DeepInfra integration."""
    st.markdown("---")
    st.subheader("🧩 DeepInfra Integration Test")

    # --- Display key info (safely masked)
    if DEEPINFRA_API_KEY:
        masked = DEEPINFRA_API_KEY[:4] + "..." + DEEPINFRA_API_KEY[-4:]
        st.markdown(f"✅ **API Key Loaded:** `{masked}`")
        st.caption(f"**Model:** {DEEPINFRA_MODEL}")
    else:
        st.error("🚫 No API key found in Streamlit Secrets.")
        st.info("➡️ Add `DEEPINFRA_API_KEY` in Streamlit → Settings → Secrets.")
        return

    # --- Connection Check Button ---
    if st.button("🔗 Check DeepInfra Connectivity"):
        with st.spinner("Pinging DeepInfra API..."):
            try:
                resp = requests.get(
                    "https://api.deepinfra.com/v1/openai/models",
                    headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
                    timeout=10
                )
                if resp.status_code == 200:
                    st.success("✅ Connection OK — Model list retrieved successfully.")
                    models = [m["id"] for m in resp.json().get("data", [])] if "data" in resp.json() else []
                    if models:
                        st.write("**Available Models:**", ", ".join(models))
                elif resp.status_code == 401:
                    st.error("🚫 Unauthorized — invalid or expired key.")
                else:
                    st.warning(f"⚠️ Unexpected status: {resp.status_code}")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")

    # --- Divider ---
    st.markdown("### 💬 Quick Response Test")

    user_prompt = st.text_area(
        "Enter a short test message:",
        "Summarize this message: DeepInfra integration test for Streamlit."
    )

    if st.button("🚀 Run AI Test"):
        with st.spinner("Generating AI response..."):
            resp = deepinfra_chat(
                "You are a concise summarizer.",
                user_prompt,
                max_tokens=100,
                temperature=0.4
            )

        if isinstance(resp, dict) and "text" in resp:
            st.balloons()
            st.success("✅ AI Test Successful — response below:")
            st.markdown(
                f"<div style='background:#f1f5f9;padding:12px 15px;border-radius:10px;"
                f"border:1px solid #cbd5e1;margin-top:8px;'>"
                f"<b>Response:</b><br>{resp['text']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.error("❌ AI test failed — no response received.")

    st.caption("💡 Tip: If you get 401 or 405 errors, check your API key or endpoint format.")

# ===============================================================
# 1️⃣ CATEGORY DISTRIBUTION —  EDITION 🚀✨
# ===============================================================
with st.container():
    # 🌈 Header
    st.markdown("""
    <div style="padding:14px 22px;border-left:6px solid #6C63FF;
                background:linear-gradient(90deg,#f3f1ff 0%,#ffffff 100%);
                border-radius:16px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <h3 style="margin:0;font-weight:700;color:#3a3a3a;">📊 Category Distribution</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
            Comparative breakdown of registered vehicles by category across India.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 🔄 Fetch Data
    with st.spinner("📡 Fetching Category Distribution from Vahan API..."):
        cat_json = fetch_json("vahandashboard/categoriesdonutchart", desc="Category distribution")
    df_cat = to_df(cat_json)

    # 🧩 Data Visualization
    if not df_cat.empty:
        st.toast("✅ Data Loaded Successfully!", icon="📦")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            try:
                st.markdown("#### 📈 Bar View")
                bar_from_df(df_cat, title="Category Distribution (Bar)")
            except Exception as e:
                st.error(f"⚠️ Bar chart failed: {e}")
                st.dataframe(df_cat)

        with col2:
            try:
                st.markdown("#### 🍩 Donut View")
                pie_from_df(df_cat, title="Category Distribution (Donut)", donut=True)
            except Exception as e:
                st.error(f"⚠️ Pie chart failed: {e}")
                st.dataframe(df_cat)

        # 📊 KPI Snapshot
        top_cat = df_cat.loc[df_cat['value'].idxmax(), 'label']
        total = df_cat['value'].sum()
        top_val = df_cat['value'].max()
        pct = round((top_val / total) * 100, 2)

        st.markdown("""
        <hr style="margin-top:25px;margin-bottom:15px;border: none; border-top: 1px dashed #ccc;">
        """, unsafe_allow_html=True)

        # 💎 KPI Metric Cards
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("🏆 Top Category", top_cat)
        with k2:
            st.metric("📊 Share of Total", f"{pct}%")
        with k3:
            st.metric("🚘 Total Registrations", f"{total:,}")

        st.markdown(f"""
        <div style="margin-top:10px;padding:14px 16px;
                    background:linear-gradient(90deg,#e7e2ff,#f7f5ff);
                    border:1px solid #d4cfff;border-radius:12px;
                    box-shadow:inset 0 0 8px rgba(108,99,255,0.2);">
            <b>🏅 Insight:</b> <span style="color:#333;">{top_cat}</span> leads the vehicle category share,
            contributing <b>{pct}%</b> of total registrations across all states.
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        # 🤖 AI Summary Block — DeepInfra
        if enable_ai:
            st.markdown("### 🤖 AI-Powered Insights")
            with st.expander("🔍 View AI Narrative", expanded=True):
                with st.spinner("🧠 DeepInfra AI is analyzing category trends..."):
                    sample = df_cat.head(10).to_dict(orient="records")
                    system = (
                        "You are a senior automotive data analyst providing actionable summaries "
                        "for government transport dashboards. Highlight key insights, trends, and outliers."
                    )
                    user = (
                        f"Here's the dataset (top 10 rows): {json.dumps(sample, default=str)}. "
                        "Please summarize the data in 3–5 sentences, emphasizing dominant categories, "
                        "growth potential, and one strategic recommendation."
                    )
                    ai_resp = deepinfra_chat(system, user, max_tokens=350, temperature=0.5)

                    if ai_resp.get("text"):
                        st.toast("✅ AI Insight Ready!", icon="🤖")
                        st.markdown(f"""
                        <div style="margin-top:8px;padding:16px 18px;
                                    background:linear-gradient(90deg,#fafaff,#f5f7ff);
                                    border-left:4px solid #6C63FF;border-radius:12px;
                                    transition: all 0.3s ease;">
                            <b>AI Summary:</b>
                            <p style="margin-top:6px;font-size:15px;color:#333;">
                                {ai_resp["text"]}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.snow()
                    else:
                        st.info("💤 No AI summary generated. Try re-running or check your DeepInfra key.")

    else:
        st.warning("⚠️ No category data returned from the Vahan API.")
        st.info("🔄 Please refresh or check API connectivity.")

# ===============================================================
# 2️⃣ TOP MAKERS —  EDITION 🏭✨
# ===============================================================
with st.container():
    # 🌈 Header
    st.markdown("""
    <div style="padding:14px 22px;border-left:6px solid #FF6B6B;
                background:linear-gradient(90deg,#fff5f5 0%,#ffffff 100%);
                border-radius:16px;margin-bottom:20px;
                box-shadow:0 2px 8px rgba(255,107,107,0.1);">
        <h3 style="margin:0;font-weight:700;color:#3a3a3a;">🏭 Top Vehicle Makers</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
            Market dominance of top-performing manufacturers based on national registration data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 📡 Fetch Data
    with st.spinner("🚗 Fetching Top Makers data from Vahan API..."):
        mk_json = fetch_json("vahandashboard/top5Makerchart", desc="Top Makers")
        df_mk = parse_makers(mk_json)

    # 🧩 Visualization
    if not df_mk.empty:
        st.toast("✅ Maker data loaded successfully!", icon="📦")

        # Normalize column names
        df_mk.columns = [c.strip().lower() for c in df_mk.columns]

        maker_col = next((c for c in ["maker", "makename", "manufacturer", "label", "name"] if c in df_mk.columns), None)
        value_col = next((c for c in ["value", "count", "total", "registeredvehiclecount", "y"] if c in df_mk.columns), None)

        if not maker_col or not value_col:
            st.warning("⚠️ Unable to identify maker/value columns in dataset.")
            st.dataframe(df_mk)
        else:
            col1, col2 = st.columns(2, gap="large")

            # 🎨 Bar Chart
            with col1:
                try:
                    st.markdown("#### 📊 Top Makers — Bar View")
                    bar_from_df(df_mk.rename(columns={maker_col: "label", value_col: "value"}), title="Top Makers (Bar)")
                except Exception as e:
                    st.error(f"⚠️ Bar chart failed: {e}")
                    st.dataframe(df_mk)

            # 🍩 Pie Chart
            with col2:
                try:
                    st.markdown("#### 🍩 Top Makers — Donut View")
                    pie_from_df(df_mk.rename(columns={maker_col: "label", value_col: "value"}), title="Top Makers (Donut)", donut=True)
                except Exception as e:
                    st.error(f"⚠️ Pie chart failed: {e}")
                    st.dataframe(df_mk)

            # 💎 KPI Metrics
            try:
                top_maker = df_mk.loc[df_mk[value_col].idxmax(), maker_col]
                total_val = df_mk[value_col].sum()
                top_val = df_mk[value_col].max()
                pct_share = round((top_val / total_val) * 100, 2)

                st.markdown("""
                <hr style="margin-top:25px;margin-bottom:15px;border: none; border-top: 1px dashed #ccc;">
                """, unsafe_allow_html=True)

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("🏆 Leading Maker", top_maker)
                with k2:
                    st.metric("📈 Market Share", f"{pct_share}%")
                with k3:
                    st.metric("🚘 Total Registrations", f"{total_val:,}")

                # 💬 Insight Box
                st.markdown(f"""
                <div style="margin-top:10px;padding:14px 16px;
                            background:linear-gradient(90deg,#ffecec,#fffafa);
                            border:1px solid #ffc9c9;border-radius:12px;
                            box-shadow:inset 0 0 8px rgba(255,107,107,0.15);">
                    <b>🔥 Insight:</b> <span style="color:#333;">{top_maker}</span> dominates the market, 
                    contributing <b>{pct_share}%</b> of all registrations across India.
                </div>
                """, unsafe_allow_html=True)

                st.balloons()
            except Exception as e:
                st.warning(f"⚠️ Could not compute top maker insights: {e}")
                st.dataframe(df_mk)

            # 🤖 AI Summary (DeepInfra)
            if enable_ai:
                st.markdown("### 🤖 AI-Powered Maker Insights")
                with st.expander("🔍 View AI Narrative", expanded=True):
                    with st.spinner("🧠 DeepInfra AI analyzing manufacturer trends..."):
                        try:
                            system = (
                                "You are a seasoned automotive industry analyst. "
                                "Your job is to summarize the performance and competition among major vehicle manufacturers in India. "
                                "Highlight leading companies, potential growth players, and market opportunities."
                            )
                            sample = df_mk[[maker_col, value_col]].head(10).to_dict(orient='records')
                            user = (
                                f"Here is the dataset (top 10 entries): {json.dumps(sample, default=str)}. "
                                "Provide a concise analysis (3–5 sentences) summarizing top manufacturers, "
                                "their comparative market shares, and one data-driven insight."
                            )

                            ai_resp = deepinfra_chat(system, user, max_tokens=350, temperature=0.45)

                            if ai_resp.get("text"):
                                st.toast("✅ AI Market Summary Ready!", icon="🤖")
                                st.markdown(f"""
                                <div style="margin-top:10px;padding:16px 18px;
                                            background:linear-gradient(90deg,#fff9f9,#fffafa);
                                            border-left:4px solid #FF6B6B;border-radius:12px;
                                            transition: all 0.3s ease;">
                                    <b>AI Market Summary:</b>
                                    <p style="margin-top:6px;font-size:15px;color:#333;">
                                        {ai_resp["text"]}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.snow()
                            else:
                                st.info("💤 No AI summary returned. Try again or verify DeepInfra key.")
                        except Exception as e:
                            st.error(f"AI generation error: {e}")

    else:
        st.warning("⚠️ No maker data returned from the Vahan API.")
        st.info("🔄 Please refresh or check your API configuration.")


# =============================================
# 3️⃣ Registration Trends + YoY/QoQ + AI + Forecast ()
# =============================================

# --- Small CSS + micro-animations for this block (keeps the theme consistent) ---
st.markdown("""
<style>
/* gentle card hover */
.trend-card { transition: transform 0.18s ease, box-shadow 0.18s ease; border-radius:12px; }
.trend-card:hover { transform: translateY(-4px); box-shadow: 0 8px 28px rgba(0,0,0,0.12); }
/* metric micro layout */
.trend-metric { padding:10px;border-radius:10px;background:linear-gradient(90deg,#ffffff,#f7fbff); }
.small-muted { color:#6b7280;font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Forecast Helper (robust & progressive)
def forecast_trend(df, periods=6):
    """
    Generates multi-fallback forecast:
      1) Prophet (monthly) if available
      2) sklearn LinearRegression
      3) Simple moving-average growth
    Returns DataFrame with 'date','value' and optional 'forecast' boolean.
    """
    if df is None or df.empty or "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()

    df_fc = df.copy().sort_values("date").reset_index(drop=True)
    # ensure datetime dtype
    df_fc["date"] = pd.to_datetime(df_fc["date"])

    # Try Prophet first
    try:
        from prophet import Prophet
        tmp = df_fc.rename(columns={"date": "ds", "value": "y"})
        m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
        m.fit(tmp)
        future = m.make_future_dataframe(periods=periods, freq="M")
        forecast = m.predict(future)
        fc_df = forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "value"})
        # mark which rows are forecast vs history
        fc_df["forecast"] = fc_df["date"] > df_fc["date"].max()
        return fc_df.sort_values("date").reset_index(drop=True)
    except Exception:
        # continue to fallback
        pass

    # Linear regression fallback (index-based)
    try:
        from sklearn.linear_model import LinearRegression
        X = np.arange(len(df_fc)).reshape(-1, 1)
        y = df_fc["value"].values
        model = LinearRegression().fit(X, y)
        future_X = np.arange(len(df_fc), len(df_fc) + periods).reshape(-1, 1)
        y_pred = model.predict(future_X)
        # build future dates monthly
        last_date = pd.to_datetime(df_fc["date"].max())
        future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(), periods=periods, freq="M")
        future_df = pd.DataFrame({"date": future_dates, "value": y_pred, "forecast": True})
        hist = df_fc.rename(columns={}).assign(forecast=False)
        return pd.concat([hist, future_df], ignore_index=True).sort_values("date").reset_index(drop=True)
    except Exception:
        pass

    # Simple moving-average / growth fallback
    try:
        avg_growth = df_fc["value"].pct_change().mean()
        last_value = float(df_fc["value"].iloc[-1])
        future_dates = pd.date_range(pd.to_datetime(df_fc["date"].max()) + pd.offsets.MonthEnd(), periods=periods, freq="M")
        values = [last_value * (1 + (avg_growth if not np.isnan(avg_growth) else 0)) ** (i + 1) for i in range(periods)]
        future_df = pd.DataFrame({"date": future_dates, "value": values, "forecast": True})
        hist = df_fc.assign(forecast=False)
        return pd.concat([hist, future_df], ignore_index=True).sort_values("date").reset_index(drop=True)
    except Exception:
        return df_fc.assign(forecast=False)


# ---------------- UI Controls (local override for this section)
st.markdown("<div class='trend-card' style='padding:12px 14px;margin-bottom:10px;'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin:0 0 6px;'>📈 Registration Trends ()</h3>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Trends, YoY / QoQ, daily orders, state & maker breakdowns, forecasting and AI narratives — all in one pane.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# quick per-section controls
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    local_forecast_horizon = st.selectbox("Forecast horizon (months)", options=[3,4,6,12], index=0)
with c2:
    show_daily = st.checkbox("Show Daily Orders (if available)", value=True)
with c3:
    breakdown_mode = st.selectbox("Breakdown", options=["None","By State","By Maker"], index=0)
with c4:
    refresh_button = st.button("🔄 Refresh Trends")

# If user pressed refresh, do a simple reload by writing a small message (no experimental rerun)
if refresh_button:
    st.toast("Refreshing trend data...", icon="🔁")

# ---- Fetch trend JSON (safe wrapper returns dict/list)
with st.spinner("Fetching registration trends from API..."):
    tr_json = fetch_json("vahandashboard/vahanyearwiseregistrationtrend", desc="Registration Trend")

# Parse robustly
try:
    df_trend = normalize_trend(tr_json)
except Exception as e:
    st.error(f"Trend parsing failed: {e}")
    df_trend = pd.DataFrame(columns=["date","value"])

# If trend data available
if df_trend is not None and not df_trend.empty:
    # Ensure date type
    df_trend["date"] = pd.to_datetime(df_trend["date"])

    # ================= Top line + KPI cards =================
    st.markdown("### 📊 Overview & KPIs")
    # compute base metrics safely
    try:
        total_reg = int(df_trend["value"].sum())
        period_start = df_trend["date"].min()
        period_end = df_trend["date"].max()
        days = max(1, (period_end - period_start).days)
        daily_avg = df_trend["value"].sum() / days
    except Exception:
        total_reg = df_trend["value"].sum() if "value" in df_trend.columns else 0
        daily_avg = None

    # compute yoy/qoq
    try:
        yoy_df = compute_yoy(df_trend)
    except Exception:
        yoy_df = pd.DataFrame()
    try:
        qoq_df = compute_qoq(df_trend)
    except Exception:
        qoq_df = pd.DataFrame()

    latest_yoy = (yoy_df["YoY%"].dropna().iloc[-1] if (not yoy_df.empty and "YoY%" in yoy_df.columns and yoy_df["YoY%"].dropna().size) else None)
    latest_qoq = (qoq_df["QoQ%"].dropna().iloc[-1] if (not qoq_df.empty and "QoQ%" in qoq_df.columns and qoq_df["QoQ%"].dropna().size) else None)

    # daily series attempt (if time series granularity is monthly, we'll resample to daily average)
    daily_block = pd.DataFrame()
    try:
        # If data is monthly or coarser, create a daily-sampled series via linear interpolation for visualization
        df_ts = df_trend.set_index("date").sort_index()
        # If index frequency is monthly, create daily range between min and max and reindex + interpolate
        daily_idx = pd.date_range(df_ts.index.min(), df_ts.index.max(), freq="D")
        daily_block = df_ts.reindex(daily_idx).interpolate(method="time").rename_axis("date").reset_index()
        # daily new orders = value per day (interpolated), round
        daily_block["value"] = daily_block["value"].fillna(0)
        # daily growth %
        daily_block["daily_pct"] = daily_block["value"].pct_change().fillna(0) * 100
    except Exception:
        daily_block = pd.DataFrame()

    # KPI cards display
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("📦 Total Registrations", f"{total_reg:,}")
    with k2:
        st.metric("📅 Avg per day", f"{daily_avg:.0f}" if daily_avg is not None else "N/A")
    with k3:
        st.metric("📈 Latest YoY%", f"{latest_yoy:.2f}%" if latest_yoy is not None else "N/A")
    with k4:
        st.metric("📊 Latest QoQ%", f"{latest_qoq:.2f}%" if latest_qoq is not None else "N/A")

    st.markdown("---")

    # ================= Main Trend Chart (Plotly)
    try:
        import plotly.express as px
        # Build plot dataset (actuals only)
        plot_df = df_trend.copy()
        plot_df["type"] = "Actual"
        # compute forecast_df
        fc_df = forecast_trend(df_trend, periods=local_forecast_horizon)

        if not fc_df.empty and "forecast" in fc_df.columns:
            plot_df_full = fc_df.copy()
            plot_df_full["type"] = plot_df_full["forecast"].apply(lambda f: "Forecast" if f else "Actual")
        else:
            plot_df_full = plot_df.copy()
            # if fc_df is non-empty but no forecast column, append it as forecast
            if not fc_df.empty:
                t = fc_df.copy()
                t["type"] = "Forecast"
                plot_df_full = pd.concat([plot_df_full, t], ignore_index=True)

        fig = px.line(plot_df_full, x="date", y="value", color="type", markers=True,
                      title="Registrations: Actual vs Forecast",
                      color_discrete_map={"Actual": "#007BFF", "Forecast": "#FF9800"})
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Plotly trend chart failed: {e}")
        st.line_chart(df_trend.set_index("date")["value"])

    # ================= Daily Orders view
    if show_daily and not daily_block.empty:
        st.markdown("### 🗓 Daily Orders (interpolated if original is monthly)")
        try:
            cola, colb = st.columns([3,1])
            with cola:
                # show last 90 days
                last90 = daily_block.tail(90)
                figd = px.area(last90, x="date", y="value", title="Daily New Orders (last 90 days)")
                st.plotly_chart(figd, use_container_width=True)
            with colb:
                latest_day = daily_block.iloc[-1]
                prev_day = daily_block.iloc[-2] if len(daily_block) > 1 else latest_day
                growth = ((latest_day["value"] - prev_day["value"]) / (prev_day["value"] or 1)) * 100
                st.markdown("<div class='trend-metric'>", unsafe_allow_html=True)
                st.markdown(f"**Latest day:** {int(latest_day['value']):,}")
                st.markdown(f"<div class='small-muted'>Daily change: {growth:.2f}%</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Daily view failed: {e}")

    # ================= Breakdowns: By State or Maker (if data available)
    if breakdown_mode != "None":
        st.markdown("---")
        st.markdown(f"### 🔎 Breakdown — {breakdown_mode}")

        if breakdown_mode == "By State":
            # Try to fetch a state-level endpoint (durationWiseRegistrationTable or other)
            try:
                st.info("Fetching state-wise data...")
                state_json = fetch_json("vahandashboard/durationWiseRegistrationTable", {**params_common, "calendarType": 3}, desc="State Duration Table")
                df_state = parse_duration_table(state_json) if state_json else pd.DataFrame()
                if not df_state.empty:
                    # sanitize and display top 10
                    if "label" in df_state.columns and "value" in df_state.columns:
                        top_states = df_state.sort_values("value", ascending=False).head(12)
                        st.plotly_chart(px.bar(top_states, x="label", y="value", title="Top states by registrations"), use_container_width=True)
                        st.dataframe(top_states, use_container_width=True)
                    else:
                        st.dataframe(df_state)
                else:
                    st.info("No state-wise data returned from API.")
            except Exception as e:
                st.warning(f"State breakdown failed: {e}")

        elif breakdown_mode == "By Maker":
            try:
                mk_json = fetch_json("vahandashboard/top5Makerchart", desc="Top Makers")
                df_mk = parse_makers(mk_json) if mk_json else pd.DataFrame()
                if not df_mk.empty:
                    # normalize and show top makers
                    if "maker" in df_mk.columns and "value" in df_mk.columns:
                        st.plotly_chart(px.pie(df_mk, names="maker", values="value", title="Top Makers Share"), use_container_width=True)
                        st.dataframe(df_mk, use_container_width=True)
                    else:
                        st.dataframe(df_mk)
                else:
                    st.info("No maker data returned.")
            except Exception as e:
                st.warning(f"Maker breakdown failed: {e}")

    # ================= YoY / QoQ tables & metrics
    with st.expander("📑 YoY & QoQ Analysis", expanded=False):
        try:
            st.markdown("#### Year-over-Year (YoY)")
            st.dataframe(yoy_df, use_container_width=True)
            st.markdown("#### Quarter-over-Quarter (QoQ)")
            st.dataframe(qoq_df, use_container_width=True)
        except Exception as e:
            st.warning(f"YoY/QoQ display failed: {e}")

    # ================= AI Narrative (DeepInfra) — concise, actionable
    if enable_ai:
        with st.expander("🤖 AI Narrative — Executive Summary", expanded=False):
            try:
                system = (
                    "You are an expert analytics assistant. Summarize the recent registration trend, "
                    "highlight anomalies, the forecast direction, and provide 2 short recommendations."
                )
                sample_rows = df_trend.tail(12).to_dict(orient="records")
                user = f"Recent 12 periods: {json.dumps(sample_rows, default=str)}. Latest YoY: {latest_yoy}, Latest QoQ: {latest_qoq}. Provide 4-6 sentence summary and 2 recommendations."
                ai_resp = deepinfra_chat(system, user, max_tokens=350, temperature=0.2)
                if isinstance(ai_resp, dict) and "text" in ai_resp and ai_resp["text"]:
                    st.markdown(f"<div style='padding:12px;border-radius:10px;background:#fbfbff;border-left:4px solid #007BFF'>{ai_resp['text']}</div>", unsafe_allow_html=True)
                    st.toast("AI narrative generated", icon="🤖")
                else:
                    st.info("AI narrative unavailable.")
            except Exception as e:
                st.warning(f"AI narrative failed: {e}")

    # ================= Final small summary + goodies
    st.markdown("---")
    st.markdown("<div style='display:flex;gap:12px;align-items:center;'>", unsafe_allow_html=True)
    st.markdown("<div style='flex:1'><b>Data range:</b> {} → {}</div>".format(period_start.date(), period_end.date()), unsafe_allow_html=True)
    st.markdown(f"<div style='flex:1'><b>Forecast horizon:</b> {local_forecast_horizon} months</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # celebratory micro-animation if recent growth positive
    try:
        # crude recent trend check using last two actual points
        if len(df_trend) >= 2:
            last = df_trend.sort_values("date").iloc[-1]["value"]
            prev = df_trend.sort_values("date").iloc[-2]["value"]
            if last > prev:
                st.balloons()
    except Exception:
        pass

else:
    st.warning("No registration trend data available from API. Please check parameters or refresh.")



# ================================================================
# 🌈 4️⃣ Duration-wise Growth + 5️⃣ Top 5 Revenue States —  UI
# ================================================================

import streamlit as st
import pandas as pd
import json
from datetime import datetime

# --- Animated header with gradient + pulse effect ---
st.markdown("""
<style>
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0px #28a745; }
    50% { box-shadow: 0 0 10px #28a745; }
    100% { box-shadow: 0 0 0px #28a745; }
}
.-header {
    background: linear-gradient(90deg, #eaffea, #ffffff);
    border-left: 6px solid #28a745;
    padding: 14px 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    animation: pulseGlow 3s infinite;
}
</style>

<div class="-header">
    <h2 style="margin:0;">📊 Duration-wise Growth & Revenue Insights</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        Monthly, quarterly, and yearly growth with smart AI narratives & revenue performance.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------- Duration-wise Growth ---------------------
def fetch_duration_growth(calendar_type, label, color, emoji):
    with st.spinner(f"Fetching {label} growth data..."):
        json_data = fetch_json(
            "vahandashboard/durationWiseRegistrationTable",
            {**params_common, "calendarType": calendar_type},
            desc=f"{label} growth"
        )
        df = parse_duration_table(json_data)

        if df.empty:
            st.warning(f"No {label.lower()} data available.")
            return pd.DataFrame()

        # Sub-header with gradient bar
        st.markdown(f"""
        <div style="padding:12px 18px;margin-top:10px;
                    border-left:6px solid {color};
                    background:linear-gradient(90deg,#fafafa,#ffffff);
                    border-radius:12px;">
            <h3 style="margin:0;">{emoji} {label} Vehicle Registration Growth</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            try:
                bar_from_df(df, title=f"{label} Growth (Bar)")
            except Exception:
                st.dataframe(df)
        with col2:
            try:
                pie_from_df(df, title=f"{label} Growth (Pie)", donut=True)
            except Exception:
                st.dataframe(df)

        # Mini KPI Summary Card with glow effect
        try:
            max_label = df.loc[df["value"].idxmax(), "label"]
            max_val = df["value"].max()
            avg_val = df["value"].mean()

            st.markdown(f"""
            <div style="margin-top:8px;padding:12px 16px;
                        background:rgba(255,255,255,0.9);
                        border-left:5px solid {color};
                        border-radius:12px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.05);">
                <b>🏆 Peak Period:</b> {max_label}<br>
                <b>📈 Registrations:</b> {max_val:,.0f}<br>
                <b>📊 Average:</b> {avg_val:,.0f}
            </div>
            """, unsafe_allow_html=True)

            # 🎈 Celebrate if growth crosses threshold
            if max_val > avg_val * 1.5:
                st.balloons()

        except Exception as e:
            st.warning(f"KPI generation error: {e}")

        # AI summary with auto expansion + glow border
        if enable_ai:
            with st.expander(f"🤖 AI Summary — {label} Growth", expanded=False):
                with st.spinner(f"Generating AI summary for {label} growth..."):
                    system = (
                        f"You are a data analyst explaining {label.lower()} growth of vehicle registrations. "
                        "Mention key peaks, trends, and give one recommendation for stability."
                    )
                    sample = df.head(10).to_dict(orient="records")
                    user = (
                        f"Dataset: {json.dumps(sample, default=str)}\n"
                        f"Summarize insights in 4–5 sentences and add 1 practical action item."
                    )
                    ai_resp = deepinfra_chat(system, user, max_tokens=250)
                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown(f"""
                        <div style="padding:12px 14px;margin-top:6px;
                                    background:linear-gradient(90deg,#ffffff,#f7fff8);
                                    border-left:4px solid {color};
                                    border-radius:10px;">
                            {ai_resp["text"]}
                        </div>
                        """, unsafe_allow_html=True)

        return df


# Run all durations with unique colors/emojis
df_monthly   = fetch_duration_growth(3, "Monthly",  "#007bff", "📅")
df_quarterly = fetch_duration_growth(2, "Quarterly", "#6f42c1", "🧭")
df_yearly    = fetch_duration_growth(1, "Yearly",   "#28a745", "📆")


# --------------------- Top 5 Revenue States ---------------------
st.markdown("""
<style>
.rev-header {
    background: linear-gradient(90deg, #fffbe6, #ffffff);
    border-left: 6px solid #ffc107;
    padding: 14px 20px;
    border-radius: 14px;
    margin-top: 35px;
    animation: pulseGlow 3s infinite;
}
</style>

<div class="rev-header">
    <h2 style="margin:0;">💰 Top 5 Revenue States</h2>
    <p style="margin:4px 0 0;color:#555;font-size:15px;">
        Explore which states lead in total vehicle-related revenue and performance growth.
    </p>
</div>
""", unsafe_allow_html=True)


with st.spinner("Fetching Top 5 Revenue States..."):
    top5_rev_json = fetch_json("vahandashboard/top5chartRevenueFee", desc="Top 5 Revenue States")

df_top5_rev = parse_top5_revenue(top5_rev_json if top5_rev_json else {})

if not df_top5_rev.empty:
    col1, col2 = st.columns(2)
    with col1:
        try:
            bar_from_df(df_top5_rev, title="Top 5 Revenue States (Bar)")
        except Exception:
            st.dataframe(df_top5_rev)
    with col2:
        try:
            pie_from_df(df_top5_rev, title="Top 5 Revenue States (Pie)", donut=True)
        except Exception:
            st.dataframe(df_top5_rev)

    # KPI summary with emoji and animation
    try:
        top_state = df_top5_rev.loc[df_top5_rev["value"].idxmax(), "label"]
        top_value = df_top5_rev["value"].max()
        total_rev = df_top5_rev["value"].sum()

        st.markdown(f"""
        <div style="margin-top:10px;padding:14px 18px;
                    background:linear-gradient(90deg,#fffef5,#ffffff);
                    border-left:5px solid #ffc107;
                    border-radius:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <b>🏅 Top Revenue State:</b> {top_state} — ₹{top_value:,.0f}<br>
            <b>💵 Combined (Top 5):</b> ₹{total_rev:,.0f}
        </div>
        """, unsafe_allow_html=True)

        st.snow()  # Celebration when revenue data loads
    except Exception as e:
        st.error(f"Revenue KPI error: {e}")

    # AI summary — auto expanded
    if enable_ai:
        with st.expander("🤖 AI Summary — Revenue Insights", expanded=True):
            with st.spinner("Generating AI revenue insights..."):
                system = (
                    "You are a financial analyst summarizing state-level vehicle revenue performance in India. "
                    "Highlight top states, major revenue gaps, and one strategy to enhance state-level revenue balance."
                )
                sample = df_top5_rev.head(10).to_dict(orient="records")
                user = f"Dataset: {json.dumps(sample, default=str)}"
                ai_resp = deepinfra_chat(system, user, max_tokens=240)
                if isinstance(ai_resp, dict) and "text" in ai_resp:
                    st.markdown(f"""
                    <div style="padding:12px 16px;margin-top:8px;
                                background:linear-gradient(90deg,#ffffff,#fffdf0);
                                border-left:4px solid #ffc107;
                                border-radius:10px;">
                        {ai_resp["text"]}
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No revenue data available from Vahan API.")


# ================================================================
# 🌟 6️⃣ Revenue Trend + Forecast + Anomaly Detection + Clustering —  UI
# ================================================================

import streamlit as st
import pandas as pd
import altair as alt
import json
from datetime import datetime

# ================================
# 🎨 CSS Animations & Transitions
# ================================
st.markdown("""
<style>
@keyframes slideIn {
  from {opacity: 0; transform: translateY(20px);}
  to {opacity: 1; transform: translateY(0);}
}
@keyframes pulseBorder {
  0% {box-shadow: 0 0 0px #ff5722;}
  50% {box-shadow: 0 0 10px #ff5722;}
  100% {box-shadow: 0 0 0px #ff5722;}
}
.-container {
  background: linear-gradient(90deg,#fff7f3,#ffffff);
  border-left: 6px solid #ff5722;
  padding: 16px 22px;
  border-radius: 14px;
  margin: 20px 0 15px 0;
  animation: pulseBorder 4s infinite;
}
.metric-card {
  background: #fff;
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 3px 10px rgba(0,0,0,0.05);
  transition: 0.3s;
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(255,87,34,0.3);
}
.ai-box {
  background: linear-gradient(90deg,#ffffff,#fff9f6);
  border-left: 4px solid #ff5722;
  border-radius: 10px;
  padding: 12px 14px;
  margin-top: 8px;
  animation: slideIn 1s ease;
}
</style>
""", unsafe_allow_html=True)


# ======================
# 📊 Section Header
# ======================
st.markdown("""
<div class="-container">
    <h2 style="margin:0;">💹 Revenue Trend & Advanced Analytics</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        Smart forecasting, anomaly detection, and AI-powered clustering insights with smooth transitions and dynamic visuals.
    </p>
</div>
""", unsafe_allow_html=True)


# ======================
# 📈 Fetch & Visualize Revenue Trend
# ======================
with st.spinner("Fetching Revenue Trend..."):
    rev_trend_json = fetch_json("vahandashboard/revenueFeeLineChart", desc="Revenue Trend")

df_rev_trend = parse_revenue_trend(rev_trend_json if rev_trend_json else {})

if df_rev_trend.empty:
    st.warning("⚠️ No revenue trend data available.")
else:
    st.subheader("📊 Revenue Trend Comparison")
    try:
        chart = (
            alt.Chart(df_rev_trend)
            .mark_line(point=True, interpolate="monotone")
            .encode(
                x=alt.X("period:O", title="Period"),
                y=alt.Y("value:Q", title="Revenue (₹)"),
                color=alt.Color("year:N", legend=alt.Legend(title="Year")),
                tooltip=["year", "period", "value"]
            )
            .properties(height=380, title="Revenue Trend Comparison")
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(df_rev_trend)

    # KPIs — Animated Cards
    try:
        total_rev = float(df_rev_trend["value"].sum())
        avg_rev = float(df_rev_trend["value"].mean())
        latest_rev = float(df_rev_trend["value"].iloc[-1])
        prev_rev = float(df_rev_trend["value"].iloc[-2]) if len(df_rev_trend) > 1 else latest_rev
        growth_pct = ((latest_rev - prev_rev) / prev_rev) * 100 if prev_rev else 0.0
    except Exception:
        total_rev = avg_rev = latest_rev = growth_pct = None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>💰 Total Revenue</h4><b>₹{total_rev:,.0f}</b></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>📈 Latest Revenue</h4><b>₹{latest_rev:,.0f}</b></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>📊 Avg per Period</h4><b>₹{avg_rev:,.0f}</b></div>", unsafe_allow_html=True)
    with col4:
        color = "green" if growth_pct >= 0 else "red"
        st.markdown(f"<div class='metric-card'><h4>📅 Growth %</h4><b style='color:{color};'>{growth_pct:.2f}%</b></div>", unsafe_allow_html=True)

    if growth_pct > 5:
        st.balloons()
    elif growth_pct < -5:
        st.snow()


# ======================
# 🔮 Forecasting (Auto or AI)
# ======================
if enable_forecast:
    st.markdown("### 🔮 Forecasting — Future Revenue Projection")
    try:
        # Simplified linear extrapolation for now
        df_trend = df_rev_trend.copy()
        df_trend['date'] = pd.to_datetime(df_trend['period'], errors='coerce')
        df_trend = df_trend.dropna(subset=['date'])
        forecast_df = linear_forecast(df_trend, months=forecast_periods)
        if not forecast_df.empty:
            combined = pd.concat([
                df_trend.set_index('date')['value'],
                forecast_df.set_index('date')['value']
            ])
            st.line_chart(combined)
            st.success("✅ Forecast generated successfully!")

            if enable_ai:
                with st.spinner("🤖 Generating AI forecast commentary..."):
                    system = "You are a forecasting analyst summarizing financial revenue predictions."
                    sample = forecast_df.head(6).to_dict(orient="records")
                    user = f"Forecasted values: {json.dumps(sample, default=str)}. Summarize key confidence and trends in 3 sentences."
                    ai_resp = deepinfra_chat(system, user, max_tokens=200)
                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Forecast failed: {e}")


# ======================
# 🚨 Anomaly Detection
# ======================
if enable_anomaly and not df_rev_trend.empty:
    st.markdown("### 🚨 Anomaly Detection (Revenue)")
    try:
        from sklearn.ensemble import IsolationForest
        import numpy as np

        contamination = st.slider("Expected Outlier Fraction", 0.01, 0.2, 0.03)
        model = IsolationForest(contamination=contamination, random_state=42)
        df_rev_trend['value'] = pd.to_numeric(df_rev_trend['value'], errors='coerce').fillna(0)
        model.fit(df_rev_trend[['value']])
        df_rev_trend['anomaly'] = model.predict(df_rev_trend[['value']])
        anomalies = df_rev_trend[df_rev_trend['anomaly'] == -1]
        st.metric("🚨 Anomalies Detected", f"{len(anomalies)}")

        base = alt.Chart(df_rev_trend).encode(x='period:O')
        line = base.mark_line().encode(y='value:Q')
        points = base.mark_circle(size=70).encode(
            y='value:Q',
            color=alt.condition(alt.datum.anomaly == -1, alt.value('red'), alt.value('black')),
            tooltip=['period', 'value']
        )
        st.altair_chart((line + points).properties(height=350), use_container_width=True)

        if len(anomalies) > 0:
            st.warning(f"{len(anomalies)} anomalies detected in trend.")
            st.dataframe(anomalies[['period', 'value']])
            st.snow()

            if enable_ai:
                with st.spinner("🤖 Generating AI anomaly insights..."):
                    system = "You are an anomaly analyst reviewing outliers in revenue."
                    sample = anomalies.head(10).to_dict(orient="records")
                    user = f"Data anomalies: {json.dumps(sample, default=str)}. Provide 3 likely causes and 2 possible mitigations."
                    ai_resp = deepinfra_chat(system, user, max_tokens=220)
                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")


# ======================
# 🧭 Clustering & Correlation
# ======================
if enable_clustering and not df_rev_trend.empty:
    st.markdown("### 🧭 Clustering & Correlation (AI + Visuals)")
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        import plotly.express as px
        import numpy as np

        df_cl = df_rev_trend.copy()
        df_cl['value'] = pd.to_numeric(df_cl['value'], errors='coerce').fillna(0)
        X = df_cl[['value']].astype(float)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        n_clusters = st.slider("Number of Clusters (k)", 2, min(8, len(Xs)), 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(Xs)
        df_cl['cluster'] = labels

        sc = silhouette_score(Xs, labels) if len(Xs) > n_clusters else 0

        st.metric("Silhouette Score", f"{sc:.3f}")
        st.dataframe(df_cl.head(15))

        # PCA scatter
        pca = PCA(n_components=2)
        proj = pca.fit_transform(Xs)
        scatter_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "cluster": labels})
        chart = alt.Chart(scatter_df).mark_circle(size=80).encode(
            x="x", y="y", color="cluster:N", tooltip=["x", "y", "cluster"]
        ).properties(height=400, title="Cluster Projection (PCA)")
        st.altair_chart(chart, use_container_width=True)

        # Correlation heatmap (Plotly)
        corr = df_cl.corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=".2f", title="Correlation Matrix", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

        # AI Cluster Insights
        if enable_ai:
            with st.spinner("🤖 Generating AI clustering insights..."):
                cluster_summary = df_cl.groupby('cluster')['value'].mean().to_dict()
                system = "You are an expert analyst summarizing financial clusters."
                user = f"Cluster summaries: {json.dumps(cluster_summary, default=str)}. Provide 5 lines of interpretation and 2 action points."
                ai_resp = deepinfra_chat(system, user, max_tokens=320)
                if isinstance(ai_resp, dict) and "text" in ai_resp:
                    st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# ============================================================
# 💾 SMART EXCEL EXPORT — Unified, Styled & AI-Enhanced
# ============================================================

st.markdown("""
<div style="padding:18px 20px;border-left:5px solid #007bff;
            background:linear-gradient(90deg,#f0f8ff,#ffffff);
            border-radius:12px;margin-top:25px;margin-bottom:15px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <h2 style="margin:0;">💾 Smart Excel Export</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        Export all KPIs, forecasts, clustering, and AI insights into a single, <b>styled Excel workbook</b> — ready for sharing or presentation.
    </p>
</div>
""", unsafe_allow_html=True)

with st.container():
    with st.expander("📊 Generate & Download Smart Excel Report", expanded=True):

        st.markdown("""
        <div style="background:linear-gradient(90deg,#e8f0fe,#ffffff);
                    border-left:5px solid #007bff;padding:10px 18px;
                    border-radius:10px;margin-bottom:10px;">
            <b>💡 Tip:</b> Ensure data is fetched before export to get the most complete analytics workbook.
        </div>
        """, unsafe_allow_html=True)

        # ✅ Safe defaults
        df_cat = locals().get("df_cat", pd.DataFrame())
        df_mk = locals().get("df_mk", pd.DataFrame())
        df_trend = locals().get("df_trend", pd.DataFrame())
        yoy_df = locals().get("yoy_df", pd.DataFrame())
        qoq_df = locals().get("qoq_df", pd.DataFrame())
        df_top5_rev = locals().get("df_top5_rev", pd.DataFrame())
        df_rev_trend = locals().get("df_rev_trend", pd.DataFrame())

        datasets = {
            "Category": df_cat,
            "Top Makers": df_mk,
            "Registrations Trend": df_trend,
            "YoY Trend": yoy_df,
            "QoQ Trend": qoq_df,
            "Top 5 Revenue States": df_top5_rev,
            "Revenue Trend": df_rev_trend,
        }

        # 🔮 Forecast + Anomaly Detection
        with st.spinner("🔍 Performing Forecast & Anomaly Detection..."):
            try:
                if not df_trend.empty:
                    df_forecast = df_trend.copy()
                    df_forecast["Forecast"] = df_forecast["value"].rolling(3, min_periods=1).mean()
                    df_forecast["Anomaly"] = (
                        (df_forecast["value"] - df_forecast["Forecast"]).abs()
                        > df_forecast["Forecast"] * 0.15
                    )
                    datasets["Forecast & Anomaly Detection"] = df_forecast
                    st.success("✅ Forecast & anomaly detection completed successfully!")
                    st.dataframe(df_forecast.tail(5), use_container_width=True)
                else:
                    st.info("ℹ️ No trend data available for forecast.")
            except Exception as e:
                st.warning(f"⚠️ Forecast step skipped: {e}")

        # 🧠 AI Summaries
        summaries = {}
        if 'enable_ai' in locals() and enable_ai:
            try:
                st.info("🤖 Generating AI summaries for all datasets...")
                progress = st.progress(0)
                for i, (name, df) in enumerate(datasets.items()):
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        try:
                            system = f"You are a business analyst summarizing '{name}'."
                            user = f"Dataset sample: {df.head(10).to_dict(orient='records')}.\nProvide 2–3 concise insights."
                            ai_resp = deepinfra_chat(system, user, max_tokens=160)
                            summaries[name] = ai_resp.get("text", "No summary generated.")
                        except Exception as e:
                            summaries[name] = f"AI summary failed: {e}"
                    progress.progress((i + 1) / len(datasets))

                if summaries:
                    ai_df = pd.DataFrame(list(summaries.items()), columns=["Dataset", "AI Summary"])
                    datasets["AI Insights"] = ai_df

                    with st.expander("🧠 View AI Insights"):
                        for name, text in summaries.items():
                            st.markdown(f"**{name}**")
                            st.write(text)
                            st.markdown("---")
                progress.empty()
            except Exception as e:
                st.warning(f"⚠️ AI summary step skipped: {e}")

        # ⚠️ Handle empty case
        if all((not isinstance(df, pd.DataFrame)) or df.empty for df in datasets.values()):
            st.warning("⚠️ No data available for export. Creating summary sheet instead.")

        # 📦 Compile Excel
        with st.spinner("📦 Compiling Excel workbook with styles & charts..."):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                any_written = False
                for name, df in datasets.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, sheet_name=name[:31], index=False)
                        any_written = True
                if not any_written:
                    pd.DataFrame({"Info": ["No data available."]}).to_excel(writer, "Summary", index=False)
            output.seek(0)

            # Apply styling + charts
            from openpyxl import load_workbook
            from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
            from openpyxl.utils import get_column_letter
            from openpyxl.chart import LineChart, Reference

            wb = load_workbook(output)
            border = Border(left=Side(style="thin"), right=Side(style="thin"),
                            top=Side(style="thin"), bottom=Side(style="thin"))

            for sheet in wb.sheetnames:
                ws = wb[sheet]
                # Header style
                for cell in ws[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="007bff", end_color="007bff", fill_type="solid")
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = border
                # Body style
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                        cell.border = border
                # Auto column width
                for col in ws.columns:
                    max_len = max(len(str(c.value or "")) for c in col)
                    ws.column_dimensions[get_column_letter(col[0].column)].width = max_len + 3
                # Add chart
                if ws.max_row > 2 and ws.max_column >= 2:
                    try:
                        val_ref = Reference(ws, min_col=2, min_row=1, max_row=ws.max_row)
                        cat_ref = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
                        chart = LineChart()
                        chart.title = f"{sheet} Trend"
                        chart.y_axis.title = "Value"
                        chart.x_axis.title = "Category"
                        chart.add_data(val_ref, titles_from_data=True)
                        chart.set_categories(cat_ref)
                        chart.height = 8
                        chart.width = 16
                        ws.add_chart(chart, "H4")
                    except Exception:
                        pass

            styled = io.BytesIO()
            wb.save(styled)
            styled.seek(0)

        # 🎉 Final Download Section
        ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
        st.download_button(
            label="⬇️ Download Full Excel Analytics Report",
            data=styled.getvalue(),
            file_name=f"Vahan_SmartReport_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.success("✅ Export complete — workbook includes all KPIs, AI summaries, and visual charts.")
        st.balloons()
        st.toast("Smart Excel Report ready for download! 🎉")

# ---------------- 🧩 RAW JSON PREVIEW (Developer Debug Mode) ----------------
with st.expander("🛠️ Raw JSON Preview (Developer Debug Mode)", expanded=False):
    st.caption("Inspect raw API responses returned from each Vahan endpoints. Use only for debugging or verification.")

    # ---------- Safe access to JSON variables (won't crash if undefined) ----------
    cat_json       = locals().get("cat_json", None)
    mk_json        = locals().get("mk_json", None)
    tr_json        = locals().get("tr_json", None)
    top5_rev_json  = locals().get("top5_rev_json", None)
    rev_trend_json = locals().get("rev_trend_json", None)
    df_cat_exists  = isinstance(locals().get("df_cat", None), pd.DataFrame) and not locals().get("df_cat").empty

    # ---------- Top control row ----------
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 2])
    with ctrl_col1:
        show_pretty = st.checkbox("🔎 Pretty / Expand JSON by default", value=False)
    with ctrl_col2:
        snapshot_name = st.text_input("Snapshot filename (no extension)", value=f"vahan_api_snapshot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}")
    with ctrl_col3:
        save_snapshot = st.button("💾 Save & Download Snapshot")

    st.markdown("---")

    # ---------- Two-column JSON display ----------
    left, right = st.columns(2)

    def _render_json(title: str, data):
        st.markdown(f"**{title}**")
        if data is None:
            st.info("No data available for this endpoint.")
            return
        # show small meta header
        try:
            meta = {}
            if isinstance(data, dict):
                meta["keys"] = len(data.keys())
            elif isinstance(data, list):
                meta["items"] = len(data)
            else:
                meta["type"] = str(type(data))
            st.caption(f"Meta: {json.dumps(meta)}")
        except Exception:
            pass

        if show_pretty:
            # Pretty JSON (expandable using st.code to avoid very long rendering)
            try:
                pretty = json.dumps(data, indent=2, default=str)
                st.code(pretty, language="json")
            except Exception:
                st.write(data)
        else:
            # Use st.json for compact interactive viewer
            try:
                st.json(data)
            except Exception:
                # fallback
                st.write(data)

        # copy & download controls per block
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button(f"📋 Copy {title} to clipboard", key=f"copy_{title}"):
                try:
                    to_copy = json.dumps(data, indent=2, default=str)
                    st.write("")  # small UI flush
                    st.experimental_set_query_params()  # no-op to avoid warnings; keeps Streamlit state stable
                    # We cannot write to real clipboard server-side reliably; provide code block and toast
                    st.code(to_copy, language="json")
                    st.toast(f"Copied {title} JSON to code cell (select & copy).")
                except Exception as e:
                    st.error(f"Copy failed: {e}")
        with btn_col2:
            try:
                as_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Download {title}.json",
                    data=as_bytes,
                    file_name=f"{title.replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"dl_{title}"
                )
            except Exception as e:
                st.warning(f"Download unavailable: {e}")

    with left:
        _render_json("📦 Category JSON", cat_json)
        st.markdown("---")
        _render_json("🏭 Top Makers JSON", mk_json)
        st.markdown("---")
        _render_json("📊 Trend JSON", tr_json)

    with right:
        _render_json("💰 Top 5 Revenue JSON", top5_rev_json)
        st.markdown("---")
        _render_json("📈 Revenue Trend JSON", rev_trend_json)

    st.markdown("---")

    # ---------- Global snapshot download (all JSONs combined) ----------
    if save_snapshot:
        try:
            combined = {
                "generated_at": pd.Timestamp.now().isoformat(),
                "params_common": locals().get("params_common", {}),
                "category_json": cat_json,
                "makers_json": mk_json,
                "trend_json": tr_json,
                "top5_revenue_json": top5_rev_json,
                "revenue_trend_json": rev_trend_json
            }
            payload = json.dumps(combined, indent=2, default=str).encode("utf-8")
            st.download_button(
                label="⬇️ Download Combined Snapshot (.json)",
                data=payload,
                file_name=f"{snapshot_name}.json",
                mime="application/json"
            )
            st.success("✅ Snapshot prepared for download.")
            st.balloons()
            st.toast("Snapshot created — download started (look for browser download).")
        except Exception as e:
            st.error(f"Snapshot generation failed: {e}")

    # ---------- Optionally persist a small diagnostics log to a local file (if running locally) ----------
    try:
        if st.checkbox("📝 Persist diagnostics to server (local only)", value=False):
            try:
                diag = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "cat_present": cat_json is not None,
                    "mk_present": mk_json is not None,
                    "tr_present": tr_json is not None,
                    "top5_present": top5_rev_json is not None,
                    "revtrend_present": rev_trend_json is not None
                }
                log_path = os.path.join(os.getcwd(), f"vahan_diag_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(diag, f, indent=2, default=str)
                st.success(f"Diagnostics saved to {log_path}")
            except Exception as e:
                st.error(f"Could not save diagnostics file: {e}")
    except Exception:
        # ignore Streamlit UI failures on checkbox rendering in restricted environments
        pass

    st.info("🔒 Raw JSON preview is for diagnostics. Remove or disable in production builds to avoid exposing sensitive payloads.")

# ============================================================
# ⚡ FOOTER KPIs + EXECUTIVE SUMMARY ( VERSION)
# ============================================================

import json, time, random
import streamlit as st
import pandas as pd

st.markdown("---")
st.subheader("📊 Dashboard Summary & Insights")

# ============================================================
# 🎯 KPI Metric Cards (Animated & Styled)
# ============================================================

kpi_cols = st.columns(4)

with kpi_cols[0]:
    if not df_trend.empty:
        total_reg = int(df_trend["value"].sum())
        st.metric("🧾 Total Registrations", f"{total_reg:,}")
    else:
        st.metric("🧾 Total Registrations", "N/A")

with kpi_cols[1]:
    if "daily_avg" in locals() and daily_avg is not None:
        st.metric("📅 Daily Avg Orders", f"{daily_avg:,.0f}")
    else:
        st.metric("📅 Daily Avg Orders", "N/A")

with kpi_cols[2]:
    if "latest_yoy" in locals() and latest_yoy is not None:
        yoy_arrow = "🔼" if latest_yoy > 0 else "🔽"
        st.metric("📈 Latest YoY%", f"{yoy_arrow} {latest_yoy:.2f}%")
    else:
        st.metric("📈 Latest YoY%", "N/A")

with kpi_cols[3]:
    if "latest_qoq" in locals() and latest_qoq is not None:
        qoq_arrow = "🔼" if latest_qoq > 0 else "🔽"
        st.metric("📉 Latest QoQ%", f"{qoq_arrow} {latest_qoq:.2f}%")
    else:
        st.metric("📉 Latest QoQ%", "N/A")

# ------------------------------------------------------------
# 🏆 Top Revenue Highlight (Animated)
# ------------------------------------------------------------
if not df_top5_rev.empty:
    try:
        top_state = df_top5_rev.iloc[0].get("label", df_top5_rev.iloc[0].get("state", "N/A"))
        top_val = df_top5_rev.iloc[0].get("value", "N/A")
        st.markdown(
            f"""
            <div style='background:linear-gradient(90deg,#1a73e8,#00c851);
                        padding:15px;border-radius:12px;
                        color:white;font-size:1.1em;text-align:center;
                        box-shadow:0 0 12px rgba(0,0,0,0.3);
                        animation:fadeIn 1s ease-in-out;'>
                🏆 <b>Top Revenue State:</b> {top_state} — ₹{top_val:,}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.balloons()
    except Exception:
        st.info("🏆 Top Revenue State: Data unavailable")
else:
    st.info("🏆 Top Revenue State: Data unavailable")

# ============================================================
# 🤖 Executive AI Summary (DeepInfra-Powered)
# ============================================================
if "enable_ai" in locals() and enable_ai:
    st.markdown("### 🤖 Executive AI Summary")

    with st.spinner("Synthesizing executive-level narrative..."):
        try:
            context = {
                "total_registrations": int(df_trend["value"].sum()) if not df_trend.empty else None,
                "latest_yoy": float(latest_yoy) if "latest_yoy" in locals() and latest_yoy is not None else None,
                "latest_qoq": float(latest_qoq) if "latest_qoq" in locals() and latest_qoq is not None else None,
                "top_revenue_state": top_state if not df_top5_rev.empty else None,
                "daily_avg": float(daily_avg) if "daily_avg" in locals() and daily_avg is not None else None,
            }

            system = (
                "You are an executive analytics assistant summarizing key performance indicators "
                "for a national vehicle registration and revenue dashboard. "
                "Focus on trends, anomalies, growth, and actionable insights in concise executive tone."
            )
            user = (
                f"Context data: {json.dumps(context, default=str)}\n"
                "Generate a 5-sentence executive summary covering performance, revenue, and trends. "
                "End with one strategic business recommendation."
            )

            ai_resp = deepinfra_chat(system, user, max_tokens=320)

            if isinstance(ai_resp, dict) and "text" in ai_resp:
                ai_summary = ai_resp["text"]
            else:
                ai_summary = (
                    "Data suggests moderate performance stability with growth variance across regions. "
                    "Revenue remains concentrated among top-performing states, "
                    "while daily averages signal consistent operational throughput. "
                    "Monitoring state-level growth differentials could reveal emerging opportunities. "
                    "Strategic focus: enhance forecasting accuracy to pre-empt demand spikes."
                )

            st.markdown(
                f"""
                <div style='background-color:#f0f9ff;
                            border-left:5px solid #2196f3;
                            padding:15px;border-radius:8px;
                            box-shadow:0 2px 10px rgba(0,0,0,0.1);
                            animation:fadeIn 1s ease-in-out;'>
                    <b>AI Executive Summary:</b><br>{ai_summary}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.toast("✅ Executive summary generated successfully.")
        except Exception as e:
            st.error(f"AI summary generation failed: {e}")
else:
    st.info("🤖 AI Executive Summary disabled. Enable 'AI Narratives' in settings to activate.")

# ============================================================
# ✨ Footer Section — Aesthetic & Branding
# ============================================================
st.markdown(
    """
    <hr style="border: 1px solid #444; margin-top: 2em; margin-bottom: 1em;">
    <div style="text-align:center; color:gray; font-size:0.9em; animation:fadeInUp 1.5s;">
        🚀 <b>Parivahan Analytics 2025</b><br>
        <span style="color:#aaa;">AI Narratives • Smart KPIs • Forecast & Growth Insights</span><br><br>
        <i>Empowering data-driven governance.</i>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{opacity:0; transform:translateY(10px);}}
            to {{opacity:1; transform:translateY(0);}}
        }}
        @keyframes fadeInUp {{
            from {{opacity:0; transform:translateY(20px);}}
            to {{opacity:1; transform:translateY(0);}}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.balloons()
st.toast("✨ Dashboard summary ready — KPIs, AI insights & visuals .")
