# =====================================================
# 🚀 PARIVAHAN ANALYTICS — MAXED IMPORTS BLOCK
# =====================================================

# =============================
# 🧱 Standard Library
# =============================
import os
import sys
import io
import json
import time
import random
import math
import traceback
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

# =============================
# 📦 Core Third-Party Libraries
# =============================
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# =============================
# 📊 Excel / Export Utilities
# =============================
from openpyxl import load_workbook, Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.chart import LineChart, BarChart, Reference
import xlsxwriter

# =============================
# 🧠 Machine Learning (Optional)
# =============================
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# =============================
# 🔮 Forecasting / Prophet (Optional)
# =============================
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    from prophet.serialize import model_to_json, model_from_json
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# =============================
# 🧰 Visualization / Reporting Helpers
# =============================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================
# 🔐 Load Environment Variables
# =============================
load_dotenv()

# =============================
# 🧩 Local Package Imports (vahan ecosystem)
# =============================
from vahan.api import build_params, get_json
from vahan.parsing import (
    to_df, normalize_trend, parse_duration_table,
    parse_top5_revenue, parse_revenue_trend, parse_makers
)
from vahan.metrics import (
    compute_yoy, compute_qoq, compute_growth,
    compare_periods, summarize_trends
)
from vahan.charts import (
    bar_from_df, pie_from_df, line_from_trend,
    show_metrics, show_tables, trend_comparison_chart
)
# =============================
# ⚠️ Notes
# =============================
# - Keep this imports block intact across updates.
# - Supports: forecasting, comparison, clustering, daily/monthly analysis, AI commentary.
# - All optional ML/forecasting libs are handled gracefully.
# - Streamlit Cloud compatible — no external binaries required


# =====================================================
# 🚀 PARIVAHAN ANALYTICS — HYBRID UI ENGINE (Aesthetic+)
# =====================================================

import streamlit as st
import requests
from datetime import date, datetime
from urllib.parse import urlencode

# =====================================================
# ⚙️ PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🚗 Parivahan Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# 🎉 FIRST-LAUNCH WELCOME
# =====================================================
if "launched" not in st.session_state:
    st.session_state.launched = True
    st.toast("🚀 Welcome to Parivahan Analytics — Hybrid Experience!", icon="🌍")
    st.balloons()

# =====================================================
# 🧭 SIDEBAR — DYNAMIC FILTER PANEL
# =====================================================
today = date.today()
default_from_year = max(2017, today.year - 1)

# Sidebar Base Style
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: #E2E8F0;
    animation: fadeIn 1.2s ease-in;
    box-shadow: 4px 0 15px rgba(0,0,0,0.25);
}
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(-10px);}
  to {opacity: 1; transform: translateY(0);}
}
.sidebar-section {
    padding: 12px 6px;
    margin-bottom: 14px;
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
    margin-bottom: 8px;
    font-size: 15px;
    text-shadow: 0 0 10px rgba(0,224,255,0.4);
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center; padding:14px 0 8px 0;">
    <h2 style="color:#00E0FF; margin-bottom:4px;">⚙️ Control Panel</h2>
    <p style="font-size:13px;color:#9CA3AF;">Customize analytics, filters, and AI insights.</p>
</div>
""", unsafe_allow_html=True)

# --- Data Filters ---
with st.sidebar.expander("📊 Data Filters", expanded=True):
    from_year = st.number_input("📅 From Year", min_value=2012, max_value=today.year, value=default_from_year)
    to_year = st.number_input("📆 To Year", min_value=from_year, max_value=today.year, value=today.year)
    state_code = st.text_input("🏙️ State Code (blank = All-India)", value="")
    rto_code = st.text_input("🏢 RTO Code (0 = aggregate)", value="0")
    vehicle_classes = st.text_input("🚘 Vehicle Classes (e.g., 2W,3W,4W)", value="")
    vehicle_makers = st.text_input("🏭 Vehicle Makers (comma-separated or IDs)", value="")
    vehicle_type = st.text_input("🛻 Vehicle Type (optional)", value="")
    time_period = st.selectbox("⏱️ Time Period", options=[0, 1, 2], index=0)
    fitness_check = st.selectbox("🧾 Fitness Check", options=[0, 1], index=0)

# --- Smart Analytics Toggles ---
with st.sidebar.expander("🧠 Smart Analytics & AI", expanded=True):
    enable_forecast = st.checkbox("📈 Enable Forecasting", value=True)
    enable_anomaly = st.checkbox("⚠️ Enable Anomaly Detection", value=True)
    enable_clustering = st.checkbox("🔍 Enable Clustering", value=True)
    enable_ai = st.checkbox("🤖 Enable DeepInfra AI Narratives", value=False)
    forecast_periods = st.number_input("⏳ Forecast Horizon (months)", min_value=1, max_value=36, value=3)

# =====================================================
# 🎨 UNIVERSAL HYBRID THEME ENGINE (Enhanced Visuals)
# =====================================================
THEMES = {    
    "Light": {"bg": "#F9FAFB", "text": "#111827", "card": "#FFFFFF", "accent": "#2563EB"},
    "Dark": {"bg": "#0B1120", "text": "#E2E8F0", "card": "#1E293B", "accent": "#38BDF8"},
    "Glass": {"bg": "rgba(15,23,42,0.85)", "text": "#E0F2FE", "card": "rgba(255,255,255,0.08)", "accent": "#00E0FF"},
    "Neumorphic": {"bg": "#E5E9F0", "text": "#1E293B", "card": "#F8FAFC", "accent": "#0078FF"},
    "Gradient": {"bg": "linear-gradient(120deg,#0F172A,#1E3A8A)", "text": "#E0F2FE", "card": "rgba(255,255,255,0.05)", "accent": "#38BDF8"},
    "High Contrast": {"bg": "#000000", "text": "#FFFFFF", "card": "#111111", "accent": "#FFDE00"},
    "VSCode": {"bg": "#0E101A", "text": "#D4D4D4", "card": "#1E1E2E", "accent": "#007ACC"},
    "Fluent": {"bg": "linear-gradient(120deg,#0E1624,#1B2838)", "text": "#E6F0FF", "card": "rgba(255,255,255,0.04)", "accent": "#0099FF"},
    "MacOS": {"bg": "linear-gradient(120deg,#FFFFFF,#EEF2FF)", "text": "#111827", "card": "rgba(255,255,255,0.8)", "accent": "#007AFF"}
}

st.sidebar.markdown("## 🎨 Appearance & Layout")
ui_mode = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=1)
font_size = st.sidebar.slider("Font Size", 12, 20, 15)
radius = st.sidebar.slider("Corner Radius", 6, 24, 12)
motion = st.sidebar.toggle("✨ Motion & Glow Effects", value=True)
palette = THEMES[ui_mode]

# =====================================================
# 💅 THEME CSS BUILDER (Aesthetic Enhanced)
# =====================================================
def build_css(palette, font_size, radius, motion):
    accent, text, bg, card = palette["accent"], palette["text"], palette["bg"], palette["card"]
    glow = f"0 0 18px {accent}55" if motion else "none"
    trans = "all 0.35s ease-in-out"

    return f"""
    <style>
    html, body, .stApp {{
        background: {bg};
        color: {text};
        font-size: {font_size}px;
        font-family: 'Inter', 'Segoe UI', 'SF Pro Display', sans-serif;
        transition: {trans};
    }}
    .block-container {{
        max-width: 1300px;
        padding: 1.8rem 2rem 3rem 2rem;
    }}
    h1, h2, h3, h4, h5 {{
        color: {accent};
        text-shadow: {glow};
        font-weight: 800;
        letter-spacing: 0.3px;
    }}
    div.stButton > button {{
        background: {accent};
        color: white;
        border: none;
        border-radius: {radius}px;
        padding: 0.65rem 1.2rem;
        transition: {trans};
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    div.stButton > button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 0 20px {accent}88;
    }}
    .glass-card {{
        background: {card};
        backdrop-filter: blur(12px);
        border-radius: {radius}px;
        padding: 22px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        transition: {trans};
    }}
    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 14px 35px rgba(0,0,0,0.25);
    }}
    [data-testid="stSidebar"] {{
        border-right: 1px solid {accent}33;
        box-shadow: 6px 0 15px rgba(0,0,0,0.1);
    }}
    [data-testid="stMetricValue"] {{
        color: {accent} !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
    }}
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {accent}77, transparent);
        margin: 1.4rem 0;
    }}
    </style>
    """

# Apply dynamic CSS
st.markdown(build_css(palette, font_size, radius, motion), unsafe_allow_html=True)

# =====================================================
# 🚗 PARIVAHAN ANALYTICS — MAXED HYBRID DASHBOARD
# =====================================================

import streamlit as st
from datetime import datetime
import pytz

# =====================================================
# 🕓 LIVE HEADER — TIME + TITLE
# =====================================================
ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist).strftime("%A, %d %B %Y • %I:%M %p")

st.markdown(f"""
<div style='text-align:center;padding:25px;border-radius:20px;
background:linear-gradient(145deg, rgba(0,0,0,0.5), rgba(255,255,255,0.08));
box-shadow:0 4px 25px rgba(0,0,0,0.3);
margin-bottom:35px;backdrop-filter:blur(10px);'>
    <h1 style='font-size:2.3rem;margin-bottom:5px;'>🚗 Parivahan Analytics Dashboard</h1>
    <p style='opacity:0.85;font-size:14px;margin:0;'>Updated: {current_time} (IST)</p>
    <p style='opacity:0.7;font-size:13px;margin-top:6px;'>
        Yearly • Monthly • State • Maker • Daily Comparative Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 💹 MAIN TITLE — COMPARISON MODE
# =====================================================
ui_mode = "Smart Comparison"
st.markdown(
    f"<h2 style='text-align:center;'>🚘 Parivahan Analytics — {ui_mode}</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;opacity:0.7;'>Month-wise • State-wise • Maker-wise • Daily Comparisons</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# =====================================================
# 📈 KPI ZONE — TOP METRICS
# =====================================================
kpi_cols = st.columns(3)
with kpi_cols[0]:
    st.metric("📊 Total Registrations", "1,520,345", "+4.8% MoM")
with kpi_cols[1]:
    st.metric("⚙️ Avg Daily Registrations", "21,567", "Stable")
with kpi_cols[2]:
    st.metric("🏆 Top State", "Maharashtra", "+3.2% Growth")

# =====================================================
# 🗓️ MONTH-WISE COMPARISON
# =====================================================
st.markdown("### 📅 Month-wise Comparison")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;'>
        <h5>📆 Total Registrations by Month</h5>
        <p style='opacity:0.7;'>Line Chart Placeholder</p>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;'>
        <h5>📊 Month-over-Month Comparison</h5>
        <p style='opacity:0.7;'>Bar Chart Placeholder</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 🌍 STATE-WISE COMPARISON
# =====================================================
st.markdown("### 🏙️ State-wise Comparison")
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;'>
        <h5>🗺️ State-wise Distribution</h5>
        <p style='opacity:0.7;'>Map or Bar Chart Placeholder</p>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;'>
        <h5>🏅 Top & Bottom Performing States</h5>
        <p style='opacity:0.7;'>Table / Comparison Placeholder</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 🏭 MAKER-WISE COMPARISON
# =====================================================
st.markdown("### 🏭 Maker-wise Comparison")
col_mk1, col_mk2 = st.columns(2)

with col_mk1:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;'>
        <h5>🏗️ Monthly Maker Performance</h5>
        <p style='opacity:0.7;'>Stacked Bar Chart Placeholder</p>
    </div>
    """, unsafe_allow_html=True)

with col_mk2:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;'>
        <h5>🔝 Top 10 Makers Comparison</h5>
        <p style='opacity:0.7;'>Horizontal Bar Chart Placeholder</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 📆 DAILY TREND COMPARISON
# =====================================================
st.markdown("### 📆 Daily Trend Comparison")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;'>
        <h5>📈 Daily Average Registrations</h5>
        <p style='opacity:0.7;'>Line Chart Placeholder</p>
    </div>
    """, unsafe_allow_html=True)

with col_d2:
    st.markdown("""
    <div style='background:rgba(255,255,255,0.05);padding:20px;border-radius:15px;'>
        <h5>📊 Day-to-Day Change</h5>
        <p style='opacity:0.7;'>Delta / Bar Chart Placeholder</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 🧭 INSIGHTS SECTION
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;margin-top:25px;margin-bottom:20px;'>
    <h3>🧠 AI-Powered Insights (Coming Soon)</h3>
    <p style='opacity:0.7;'>Narratives, Forecasts, and Anomaly Highlights will auto-generate here.</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 🧾 FOOTER — BRAND STRIP
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;opacity:0.65;font-size:13px;margin-top:10px;'>"
    "🌐 Parivahan Analytics • Hybrid Interface Engine • Maxed Edition"
    "</div>",
    unsafe_allow_html=True,
)

# =====================================================
# 🤖 DeepInfra AI — Secure Connection via Streamlit Secrets
# =====================================================
import streamlit as st
import requests
import time

# =====================================================
# 🔧 CONFIG LOADER
# =====================================================
def load_deepinfra_config():
    """Safely load DeepInfra API key and model from Streamlit secrets."""
    try:
        key = st.secrets["DEEPINFRA_API_KEY"]
        model = st.secrets.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        return key, model
    except Exception:
        return None, None

DEEPINFRA_API_KEY, DEEPINFRA_MODEL = load_deepinfra_config()

# =====================================================
# 🎨 CUSTOM SIDEBAR CSS (Status Cards)
# =====================================================
st.sidebar.markdown("""
<style>
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(0,224,255,0.6); }
  70% { box-shadow: 0 0 0 8px rgba(0,224,255,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,224,255,0); }
}
.deepinfra-box {
  background: rgba(255,255,255,0.05);
  padding: 12px 15px;
  border-radius: 12px;
  border-left: 4px solid #00E0FF99;
  margin-top: 10px;
  transition: all 0.3s ease;
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
.small-text {
  opacity: 0.75;
  font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 🧠 CONNECTION CHECK FUNCTION
# =====================================================
def check_deepinfra_connection(api_key: str):
    """Ping DeepInfra API and return response details."""
    try:
        resp = requests.get(
            "https://api.deepinfra.com/v1/openai/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        return resp.status_code
    except requests.exceptions.Timeout:
        return "timeout"
    except Exception:
        return "error"

# =====================================================
# ⚙️ AI MODE TOGGLE (AUTO DETECT OR MANUAL)
# =====================================================
enable_ai = st.session_state.get("enable_ai", True)

st.sidebar.markdown("<div class='deepinfra-box'><span class='deepinfra-title'>🤖 DeepInfra AI Connection</span></div>", unsafe_allow_html=True)

if enable_ai:
    if DEEPINFRA_API_KEY:
        with st.spinner("Connecting to DeepInfra..."):
            status = check_deepinfra_connection(DEEPINFRA_API_KEY)
            time.sleep(1)

        if status == 200:
            st.sidebar.markdown(f"""
            <div class='deepinfra-box deepinfra-connected'>
                ✅ <b>Connected</b><br>
                <small>Model: <b>{DEEPINFRA_MODEL}</b></small><br>
                <small>Status: 200 OK</small>
            </div>
            """, unsafe_allow_html=True)
        elif status == 401:
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-error'>
                🚫 <b>Unauthorized</b> — invalid or expired key.<br>
                <small>Check your DEEPINFRA_API_KEY.</small>
            </div>
            """, unsafe_allow_html=True)
        elif status == 405:
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-warning'>
                ⚠️ <b>Method Not Allowed (405)</b><br>
                <small>Check DeepInfra endpoint or SDK usage.</small>
            </div>
            """, unsafe_allow_html=True)
        elif status == "timeout":
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-error'>
                ⏱️ <b>Timeout</b> — DeepInfra did not respond in time.
            </div>
            """, unsafe_allow_html=True)
        elif status == "error":
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-error'>
                ❌ <b>Connection Error</b><br>
                <small>Unable to reach DeepInfra API.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"""
            <div class='deepinfra-box deepinfra-warning'>
                ⚠️ <b>DeepInfra Status:</b> {status}<br>
                <small>Unexpected response — check dashboard logs.</small>
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
st.sidebar.caption("💡 Tip: You can toggle AI mode dynamically — the dashboard adapts instantly.")

# =====================================================
# ⚙️ Build & Display Vahan Parameters — MAXED HYBRID EDITION
# =====================================================
import json
import streamlit as st
import time
import random

# --- Animated Header Banner ---
st.markdown("""
<div style="
    background: linear-gradient(90deg, #00c6ff, #0072ff, #00e0ff);
    background-size: 300% 300%;
    animation: gradientShift 4s ease infinite;
    padding: 18px 26px;
    border-radius: 16px;
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 0 25px rgba(0, 224, 255, 0.3);">
    <div>🧩 Building Dynamic API Parameters for <b>Vahan Analytics</b></div>
    <div style="font-size:14px;opacity:0.85;">Auto-synced with active filters 🔁</div>
</div>

<style>
@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
</style>
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

        # --- Success Animation ---
        st.balloons()
        st.toast("✨ Parameters generated successfully!", icon="⚙️")

        # --- Stylish Expander with JSON view ---
        with st.expander("🔧 View Generated Vahan Request Parameters (JSON)", expanded=True):
            st.markdown("""
            <div style="font-size:15px;color:#00E0FF;font-weight:600;margin-bottom:6px;">
                📜 Parameter Payload Preview
            </div>
            """, unsafe_allow_html=True)

            st.json(params_common)

            # --- Copy button (centered) ---
            colL, colM, colR = st.columns([1.2, 1, 1.2])
            with colM:
                if st.button("📋 Copy JSON to Clipboard"):
                    st.toast("Copied successfully!", icon="✅")

        # --- Success Banner ---
        st.markdown(f"""
        <div style="
            margin-top:16px;
            background: rgba(0, 224, 255, 0.1);
            border: 1px solid rgba(0,224,255,0.3);
            padding: 14px 20px;
            border-radius: 12px;
            color: #00E0FF;
            font-weight:600;
            backdrop-filter: blur(8px);
            display:flex;justify-content:space-between;align-items:center;">
            <div>✅ Parameters built successfully for <b>{to_year}</b></div>
            <div style="opacity:0.75;font-size:14px;">Ready to fetch data 📡</div>
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
st.markdown("<hr style='opacity:0.4;'>", unsafe_allow_html=True)
colA, colB, colC = st.columns([1.5,1,1.5])

with colB:
    if st.button("♻️ Rebuild Parameters with Latest Filters"):
        emoji = random.choice(["🔁", "🚗", "⚙️", "🧠", "🛰️"])
        st.toast(f"{emoji} Rebuilding dynamic params...", icon=emoji)
        time.sleep(0.8)
        st.rerun()

# =====================================================
# ⚙️ Dynamic Safe API Fetch Layer — MAXED HYBRID EDITION
# =====================================================
import time, random, streamlit as st

# ----------------------------------------
# 🔹 Internal Tag Builder
# ----------------------------------------
def _tag(text, color):
    """Generate a small neon tag label."""
    return f"<span style='background:{color};padding:4px 8px;border-radius:6px;color:white;font-size:12px;margin-right:6px;'>{text}</span>"

# ----------------------------------------
# 🔹 Smart API Fetch Wrapper
# ----------------------------------------
def fetch_json(endpoint, params=params_common, desc=""):
    """
    Intelligent API fetch wrapper with:
      - Live retry system
      - Animated UI feedback
      - JSON preview expander
      - Self-healing retry button
      - Consistent Parivahan theme visuals
    """
    max_retries = 3
    delay = 1 + random.random()
    desc = desc or endpoint

    # --- Visual Task Header ---
    st.markdown(f"""
    <div style="
        padding:10px 16px;
        margin:12px 0;
        border-radius:12px;
        background:rgba(0, 150, 255, 0.12);
        border-left:5px solid #00C6FF;
        box-shadow:0 0 15px rgba(0,198,255,0.15);
        display:flex;align-items:center;gap:8px;">
        <div>{_tag("API", "#007BFF")} {_tag("TASK", "#00B894")}</div>
        <div style="font-size:14px;color:#E2E8F0;">Fetching <code>{desc}</code> ...</div>
    </div>
    """, unsafe_allow_html=True)

    json_data = None

    # --- Retry Loop ---
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

            # Progressive delay
            time.sleep(delay * attempt * random.uniform(0.9, 1.3))

    # ============================================================
    # ✅ SUCCESS CASE
    # ============================================================
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
            margin-top:10px;
            box-shadow:0 0 15px rgba(0,198,255,0.25);">
            ✅ Successfully fetched <b>{desc}</b>! Ready for visualization 🚗
        </div>
        """, unsafe_allow_html=True)
        return json_data

    # ============================================================
    # ❌ FAILURE CASE
    # ============================================================
    st.error(f"⛔ Failed to fetch {desc} after {max_retries} attempts.")
    st.markdown("""
    <div style="
        background:rgba(255,60,60,0.08);
        padding:15px;
        border-radius:10px;
        border-left:5px solid #ff4444;
        margin-top:10px;">
        <b>💡 Troubleshooting Tips:</b><br>
        • Check internet / API connectivity<br>
        • Verify parameters are valid<br>
        • Try again after 1–2 minutes (API may be rate-limited)
    </div>
    """, unsafe_allow_html=True)

    # --- Interactive Retry Controls ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"🔁 Retry {desc}", key=f"retry_{desc}_{random.randint(0,9999)}"):
            st.toast("Retrying API fetch...", icon="🔄")
            time.sleep(0.8)
            st.rerun()

    with col2:
        if st.button("📡 Test API Endpoint", key=f"test_{desc}_{random.randint(0,9999)}"):
            test_url = f"https://analytics.parivahan.gov.in/{endpoint}"
            st.markdown(f"""
            <div style="background:rgba(0,224,255,0.1);
                        padding:8px 10px;border-radius:8px;margin-top:6px;">
                🌐 <b>Test URL:</b> `{test_url}`<br>
                <small style="opacity:0.7;">(Requires valid parameters for data)</small>
            </div>
            """, unsafe_allow_html=True)

    return {}


# ============================================
# 🤖 DeepInfra AI Helper (Streamlit Secrets Only) — ULTRA EDITION
# ============================================

import json
import streamlit as st
import requests
import time, random

# ✅ Constants
DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# 🔐 Securely load credentials from Streamlit secrets
DEEPINFRA_API_KEY = st.secrets.get("DEEPINFRA_API_KEY")
DEEPINFRA_MODEL = st.secrets.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# ============================================
# 💬 AI Chat Function — Stream + Retry Safe
# ============================================

def ask_deepinfra(prompt: str, system: str = "You are an expert analytics assistant helping with Parivahan data insights."):
    """
    Sends a prompt to DeepInfra API and returns the model’s streamed response.
    Features:
    - Streamed, live updates in Streamlit UI
    - Built-in retries with delay
    - Handles missing key / timeout / API failure gracefully
    """

    # 🔒 Validate setup
    if not DEEPINFRA_API_KEY:
        st.warning("⚠️ DeepInfra API key not found in `st.secrets`. Please configure it first.")
        return "Missing API key."

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True,  # ✅ Real-time output
    }

    max_retries = 2
    delay = 1 + random.random()

    # ========================================
    # 🚀 Attempted Streaming Loop
    # ========================================
    for attempt in range(1, max_retries + 1):
        try:
            st.markdown(f"<small>🧠 DeepInfra Attempt {attempt}/{max_retries}...</small>", unsafe_allow_html=True)
            with requests.post(DEEPINFRA_CHAT_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    st.error(f"🚫 DeepInfra error {resp.status_code}: {resp.text[:200]}")
                    continue

                full_reply = ""
                placeholder = st.empty()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        chunk = decoded[6:]
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0]["delta"].get("content", "")
                            if delta:
                                full_reply += delta
                                placeholder.markdown(f"🧠 **AI:** {full_reply}")
                        except Exception:
                            pass

                if full_reply.strip():
                    st.success("✅ AI response complete!")
                    return full_reply

        except requests.exceptions.Timeout:
            st.warning(f"⏱️ Timeout on attempt {attempt}. Retrying...")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ DeepInfra connection error: {e}")
        time.sleep(delay * attempt)

    # ========================================
    # ❌ Final Fallback
    # ========================================
    st.error("⛔ DeepInfra API failed after multiple attempts.")
    st.markdown("""
    <div style="
        background:rgba(255,60,60,0.08);
        padding:15px;border-radius:10px;
        border-left:5px solid #ff4444;">
        <b>💡 Troubleshooting:</b><br>
        • Verify API key in <code>st.secrets</code><br>
        • Check DeepInfra API status<br>
        • Try again later (possible network lag)
    </div>
    """, unsafe_allow_html=True)

    return "No AI response (DeepInfra unreachable)."


# ============================================
# 🧠 Optional: Inline Chatbox for AI Insights
# ============================================

with st.expander("💬 Ask DeepInfra AI Assistant", expanded=False):
    st.markdown("""
    <div style='font-size:14px;opacity:0.8;margin-bottom:10px;'>
        Use this AI assistant to interpret trends, detect anomalies, or generate narrative insights from analytics data.<br>
        <i>Example:</i> <code>"Explain daily order fluctuations in July 2025"</code>
    </div>
    """, unsafe_allow_html=True)

    user_prompt = st.text_area(
        "💭 Your Question or Data Insight Query",
        placeholder="e.g. Explain YoY trend anomalies or summarize top-performing states...",
        key="deepinfra_input",
    )

    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        ask_clicked = st.button("🚀 Ask AI", use_container_width=True)
    with col2:
        clear_clicked = st.button("🧹 Clear Chat", use_container_width=True)

    if ask_clicked:
        if user_prompt.strip():
            st.toast("🔍 Querying DeepInfra AI...", icon="🤖")
            with st.spinner("🧠 Generating AI insights..."):
                ai_reply = ask_deepinfra(user_prompt)
                if ai_reply and ai_reply.strip():
                    st.markdown(f"""
                    <div style='
                        background:rgba(0,255,180,0.08);
                        border-left:5px solid #00FFC6;
                        border-radius:10px;
                        padding:15px;
                        margin-top:10px;
                        font-size:15px;'>
                        <b>🧠 AI Insight:</b><br>{ai_reply}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No meaningful response received from DeepInfra.")
        else:
            st.warning("✍️ Please enter a valid question before submitting.")

    if clear_clicked:
        st.session_state["deepinfra_input"] = ""
        st.experimental_rerun()

# ===============================================
# 🔍 DeepInfra Connection Status —  MAXED UI EDITION
# ===============================================
import time
import streamlit as st
import requests

def check_deepinfra_connection():
    """
    ✅ Robust DeepInfra connection validator with full Streamlit UI integration.
    Shows animated feedback, retry logic, and available models list.
    Returns True if connection is healthy, else False.
    """

    # --- Missing Key Case ---
    if not DEEPINFRA_API_KEY:
        st.sidebar.warning("⚠️ DeepInfra API key missing from Streamlit Secrets.")
        with st.sidebar.expander("🛠️ Setup Instructions"):
            st.markdown("""
            1. Go to **Settings → Secrets**  
            2. Add:
               ```toml
               DEEPINFRA_API_KEY = "your_api_key_here"
               DEEPINFRA_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
               ```
            3. Save & rerun the app.
            """)
        return False

    # --- Status Animation ---
    with st.sidebar:
        with st.spinner("🤖 Connecting securely to DeepInfra..."):
            time.sleep(0.6)  # smoother animation timing

    # --- Test Connection ---
    try:
        response = requests.get(
            "https://api.deepinfra.com/v1/openai/models",
            headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
            timeout=8,
        )

        if response.status_code == 200:
            data = response.json()
            models = [m.get("id", "Unknown") for m in data.get("data", [])]

            st.sidebar.success("✅ DeepInfra Connected — AI Narratives Ready!")
            st.sidebar.caption(f"🧠 Model in use: **{DEEPINFRA_MODEL}**")
            st.sidebar.progress(100)
            st.balloons()

            # Optional model list
            with st.sidebar.expander("📋 Available Models", expanded=False):
                if models:
                    st.code("\n".join(models))
                else:
                    st.caption("No model list returned (check access permissions).")

            return True

        # --- Common Error Handlers ---
        elif response.status_code == 401:
            st.sidebar.error("🚫 Unauthorized — Invalid or expired API key.")
            st.sidebar.caption("💡 Regenerate your key from DeepInfra Dashboard.")
        elif response.status_code == 429:
            st.sidebar.warning("⏳ Too Many Requests — please wait and retry.")
        elif response.status_code == 405:
            st.sidebar.warning("⚠️ Method Not Allowed — check endpoint format.")
        else:
            st.sidebar.warning(f"⚠️ Unexpected {response.status_code}: {response.text[:80]}")

    except requests.exceptions.Timeout:
        st.sidebar.error("⏱️ Connection timed out — check your network or DeepInfra status.")
    except Exception as e:
        st.sidebar.error(f"❌ DeepInfra connection error: {e}")

    # --- Retry Option ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🔁 Retry Connection", use_container_width=True):
        st.toast("Reconnecting to DeepInfra...", icon="🔄")
        time.sleep(1)
        st.rerun()

    st.sidebar.caption("❌ DeepInfra not connected — AI features disabled.")
    return False


# ===========================================
# 💬 DeepInfra Chat Completion — MAXED EDITION
# ===========================================
import requests, time, random, streamlit as st

DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

def deepinfra_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
    retries: int = 3,
    delay: float = 2.0,
):
    """
    🔮 DeepInfra Chat Completion API — Robust Streamlit Wrapper
    ------------------------------------------------------------
    ✅ Secure — uses Streamlit Secrets for API key & model
    ✅ User-friendly — dynamic status blocks, retries, and rich UI
    ✅ Safe — handles timeouts, rate limits, invalid keys gracefully
    ✅ Smart — exponential retry & live progress feedback
    """

    # --- 1️⃣ Validate API Key ---
    if not DEEPINFRA_API_KEY:
        st.warning("⚠️ Missing DeepInfra API key in Streamlit Secrets.")
        with st.expander("🛠️ Setup Instructions", expanded=False):
            st.markdown("""
            Add these in Streamlit → **Settings → Secrets**  
            ```toml
            DEEPINFRA_API_KEY = "your_api_key_here"
            DEEPINFRA_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            ```
            """)
        return {"error": "Missing API key"}

    # --- 2️⃣ API Header & Payload ---
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # --- 3️⃣ Stylish Info Banner ---
    st.markdown(f"""
    <div style="
        padding:12px 16px;
        border-left:5px solid #8b5cf6;
        border-radius:12px;
        margin:10px 0;
        background:linear-gradient(90deg,rgba(139,92,246,0.1),rgba(59,130,246,0.1));
    ">
        <b>🤖 DeepInfra AI:</b> Generating analytical insight...  
        <span style="opacity:0.7;font-size:13px;">Model: <code>{DEEPINFRA_MODEL}</code></span>
    </div>
    """, unsafe_allow_html=True)

    # --- 4️⃣ Retry Loop ---
    for attempt in range(1, retries + 1):
        try:
            with st.spinner(f"🧠 Generating AI insight (Attempt {attempt}/{retries})..."):
                response = requests.post(
                    DEEPINFRA_CHAT_URL, headers=headers, json=payload, timeout=60
                )

            # --- Handle Common HTTP Errors ---
            if response.status_code == 401:
                st.error("🚫 Unauthorized — invalid or expired API key.")
                return {"error": "Unauthorized"}

            elif response.status_code == 405:
                st.error("⚠️ Method Not Allowed — check DeepInfra endpoint.")
                return {"error": "405 Method Not Allowed"}

            elif response.status_code == 429:
                st.warning("⏳ Rate limited — waiting before retry...")
                time.sleep(delay * attempt)
                continue

            elif response.status_code >= 500:
                st.warning(f"⚠️ DeepInfra server error ({response.status_code}). Retrying...")
                time.sleep(delay * attempt)
                continue

            response.raise_for_status()
            data = response.json()

            # --- Parse Valid AI Response ---
            message = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if message:
                st.toast("✅ AI Insight Ready!", icon="🤖")
                st.markdown(f"""
                <div style='background:#0f172a;color:#e2e8f0;padding:15px;
                    border-radius:12px;border:1px solid #334155;margin-top:10px;'>
                    <b>🔍 AI Insight:</b><br>
                    <div style='white-space:pre-wrap;font-family:Inter, sans-serif;
                                font-size:15px;line-height:1.5;margin-top:4px;'>
                        {message}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                return {"text": message, "raw": data}

            st.warning("⚠️ AI returned an empty response.")
            return {"error": "Empty AI output", "raw": data}

        except requests.exceptions.Timeout:
            st.warning("⏱️ Timeout — retrying connection...")
        except requests.exceptions.ConnectionError:
            st.error("🌐 Network error — please check connectivity.")
            break
        except Exception as e:
            st.error(f"❌ Unexpected DeepInfra error: {e}")

        # --- Exponential Backoff Before Retry ---
        sleep_time = delay * attempt * random.uniform(1.0, 1.4)
        time.sleep(sleep_time)

    # --- 5️⃣ Final Failure Case ---
    st.error("⛔ DeepInfra AI failed after multiple attempts.")
    st.caption("💡 Check internet connection, API key validity, or model availability.")
    if st.button("🔁 Retry AI Connection"):
        st.toast("Reconnecting DeepInfra AI...", icon="🔄")
        time.sleep(1)
        st.rerun()

    return {"error": "Failed after retries"}

# ================================================================
# 🧠 DeepInfra Test & Debug UI — MAXED EDITION
# ================================================================
import streamlit as st
import requests, time

def deepinfra_test_ui():
    """Ultimate Streamlit UI for testing DeepInfra API connection + responses."""

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🧩 DeepInfra Integration — MAXED Diagnostics")

    # ============================================================
    # 🔐 API Key + Model Display
    # ============================================================
    if DEEPINFRA_API_KEY:
        masked = DEEPINFRA_API_KEY[:4] + "..." + DEEPINFRA_API_KEY[-4:]
        st.markdown(f"✅ **API Key Loaded:** `{masked}`")
        st.caption(f"🧠 **Model in Use:** `{DEEPINFRA_MODEL}`")
        st.markdown(
            f"<div style='background:#10b981;padding:6px 10px;color:white;"
            f"border-radius:6px;width:fit-content;'>🟢 Connected (Key Found)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.error("🚫 No API key found in Streamlit Secrets.")
        st.info("➡️ Add `DEEPINFRA_API_KEY` in Streamlit → Settings → Secrets.")
        return

    # ============================================================
    # 🔗 Connection Test Section
    # ============================================================
    st.markdown("### 🔗 Check DeepInfra Connectivity")

    if st.button("🚀 Run Connectivity Test"):
        with st.spinner("Checking DeepInfra API connectivity..."):
            start_time = time.time()
            try:
                resp = requests.get(
                    "https://api.deepinfra.com/v1/openai/models",
                    headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
                    timeout=10
                )
                latency = time.time() - start_time

                if resp.status_code == 200:
                    st.success(f"✅ Connection OK (Latency: {latency:.2f}s)")
                    data = resp.json()
                    models = [m["id"] for m in data.get("data", [])] if "data" in data else []
                    if models:
                        with st.expander("📋 View Available Models"):
                            st.code("\n".join(models))
                elif resp.status_code == 401:
                    st.error("🚫 Unauthorized — invalid or expired API key.")
                elif resp.status_code == 429:
                    st.warning("⏳ Too many requests — wait a bit before retrying.")
                else:
                    st.warning(f"⚠️ Unexpected HTTP {resp.status_code}: {resp.text[:150]}")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")

    # ============================================================
    # 💬 AI Response Test Section
    # ============================================================
    st.markdown("### 💬 Quick AI Response Test")

    user_prompt = st.text_area(
        "Enter a short message to test AI:",
        "Summarize this message: DeepInfra integration test for Streamlit.",
        height=100
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        temp = st.slider("🎛️ Temperature", 0.0, 1.0, 0.4, 0.1)
    with col2:
        max_tok = st.slider("🧩 Max Tokens", 50, 512, 150, 50)

    if st.button("🧠 Run DeepInfra AI Test"):
        with st.spinner("Generating AI response..."):
            start_time = time.time()
            resp = deepinfra_chat(
                "You are a concise summarizer.",
                user_prompt,
                max_tokens=max_tok,
                temperature=temp
            )
            latency = time.time() - start_time

        if isinstance(resp, dict) and "text" in resp:
            st.success(f"✅ AI Test Successful (Response Time: {latency:.2f}s)")
            st.balloons()
            st.markdown(
                f"""
                <div style='background:#0f172a;color:#e2e8f0;padding:15px;
                    border-radius:10px;border:1px solid #334155;margin-top:8px;'>
                    <b>🔍 Response:</b><br>
                    <pre style='white-space:pre-wrap;font-family:Inter, sans-serif;'>{resp['text']}</pre>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("❌ AI test failed — no valid response received.")

    st.caption("💡 Tip: If you see 401 or 405 errors, verify your DeepInfra API key and endpoint.")

# ===============================================================
# 2️⃣ COMPARATIVE ANALYTICS — DAILY / MONTHLY / STATEWISE / MAKERWISE 🚀
# ===============================================================
with st.container():
    st.markdown("""
    <div style="padding:14px 22px;border-left:6px solid #0096FF;
                background:linear-gradient(90deg,#f0faff 0%,#ffffff 100%);
                border-radius:16px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <h3 style="margin:0;font-weight:700;color:#2a2a2a;">📅 Comparative Analytics</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
            Dynamic daily, monthly, state-wise, and maker-wise analysis across registered vehicle data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================
    # ⚙️ Filters
    # ==========================================================
    st.markdown("### 🎛️ Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        time_mode = st.selectbox("⏰ Time Granularity", ["Daily", "Monthly"], index=1)
    with c2:
        compare_basis = st.selectbox("📊 Comparison Basis", ["Statewise", "Makerwise"], index=0)
    with c3:
        top_n = st.slider("🏆 Show Top N", 3, 15, 8)

    # ==========================================================
    # 🔄 Fetch Data Dynamically
    # ==========================================================
    with st.spinner(f"📡 Loading {compare_basis} {time_mode.lower()} data from Vahan..."):
        endpoint = ""
        if compare_basis == "Statewise":
            endpoint = "vahandashboard/statewisecomparisonchart"
        elif compare_basis == "Makerwise":
            endpoint = "vahandashboard/makerwisecomparisonchart"

        json_data = fetch_json(endpoint, desc=f"{compare_basis} {time_mode}")
        df_comp = to_df(json_data)

    if not df_comp.empty:
        st.toast("✅ Comparison Data Loaded!", icon="📈")

        # ======================================================
        # 🔍 Data Cleanup & Aggregation
        # ======================================================
        if time_mode == "Monthly":
            df_comp["label"] = df_comp["label"].astype(str)
            df_comp["month"] = df_comp["label"].apply(lambda x: x.strip()[:3])  # short month
            df_pivot = df_comp.groupby(["month"], as_index=False)["value"].sum().sort_values("month")
        else:
            df_pivot = df_comp.copy()

        # ======================================================
        # 📊 Visualization Layout
        # ======================================================
        colL, colR = st.columns([2, 1], gap="large")

        with colL:
            st.markdown("#### 📈 Trend Chart")
            try:
                line_from_df(df_pivot, title=f"{compare_basis} Trend ({time_mode})")
            except Exception as e:
                st.error(f"⚠️ Trend chart failed: {e}")
                st.dataframe(df_pivot)

        with colR:
            st.markdown("#### 🧱 Top Categories")
            try:
                df_top = df_comp.nlargest(top_n, "value")
                bar_from_df(df_top, title=f"Top {top_n} {compare_basis}")
            except Exception as e:
                st.error(f"⚠️ Bar chart failed: {e}")
                st.dataframe(df_comp)

        # ======================================================
        # 🧩 KPI Metrics
        # ======================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        total_reg = df_comp["value"].sum()
        top_label = df_comp.loc[df_comp["value"].idxmax(), "label"]
        top_value = df_comp["value"].max()
        share = round((top_value / total_reg) * 100, 2)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("🏆 Top Performer", top_label)
        with k2:
            st.metric("📊 Share of Total", f"{share}%")
        with k3:
            st.metric("🪪 Total Registrations", f"{total_reg:,}")

        # ======================================================
        # 💬 Insight Section
        # ======================================================
        st.markdown(f"""
        <div style="margin-top:15px;padding:14px 16px;
                    background:linear-gradient(90deg,#e9f6ff,#f8fbff);
                    border:1px solid #b6e0ff;border-radius:12px;
                    box-shadow:inset 0 0 8px rgba(0,150,255,0.15);">
            <b>📍 Insight:</b> <span style="color:#333;">{top_label}</span> currently leads with a
            <b>{share}%</b> share of total registrations across the {compare_basis.lower()} category.
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        # ======================================================
        # 🧠 Optional AI Insight — DeepInfra
        # ======================================================
        if enable_ai:
            st.markdown("### 🤖 AI-Powered Comparative Summary")
            with st.expander("🔍 View AI Insight", expanded=True):
                with st.spinner("🧠 DeepInfra AI analyzing comparative dataset..."):
                    sample = df_comp.head(10).to_dict(orient="records")
                    system_prompt = (
                        "You are an expert automotive analytics assistant. Analyze comparative registration "
                        "data (statewise/makerwise) and highlight key performers, distribution trends, and "
                        "strategic insights relevant to national transport patterns."
                    )
                    user_prompt = (
                        f"Here is the dataset (sample): {json.dumps(sample, default=str)}. "
                        f"Explain trends across {compare_basis} on a {time_mode.lower()} basis. "
                        "Summarize in 4–5 sentences including one recommendation."
                    )
                    ai_out = deepinfra_chat(system_prompt, user_prompt, max_tokens=350, temperature=0.5)

                    if ai_out.get("text"):
                        st.toast("✅ AI Comparative Summary Ready!", icon="🤖")
                        st.markdown(f"""
                        <div style="margin-top:8px;padding:16px 18px;
                                    background:linear-gradient(90deg,#f8faff,#eef6ff);
                                    border-left:4px solid #0096FF;border-radius:12px;">
                            <b>AI Summary:</b>
                            <p style="margin-top:6px;font-size:15px;color:#333;">
                                {ai_out["text"]}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("💤 AI summary unavailable. Try again or check DeepInfra key.")
    else:
        st.warning("⚠️ No comparison data returned from API.")
        st.info("🔄 Try adjusting filters or reloading the dashboard.")

# ===============================================================
# 3️⃣ TOP MAKERS COMPARATIVE ANALYTICS — MAXED HYBRID BLOCK 🧠⚙️
# ===============================================================
with st.container():
    st.markdown("""
    <div style="padding:14px 22px;border-left:6px solid #FF8C00;
                background:linear-gradient(90deg,#fff8ef 0%,#ffffff 100%);
                border-radius:16px;margin-bottom:20px;box-shadow:0 2px 8px rgba(255,140,0,0.1);">
        <h3 style="margin:0;font-weight:700;color:#3a3a3a;">📅 Comparative Maker Analytics</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
            In-depth maker performance comparisons across months, states, and daily registration bases.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ===============================================================
    # 🎛️ CONTROL PANEL
    # ===============================================================
    st.markdown("### 🎚️ Comparison Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        time_basis = st.selectbox("⏰ Time Range", ["Daily", "Monthly"], index=1)
    with f2:
        compare_type = st.selectbox("🏙️ Comparison Mode", ["Statewise", "Makerwise"], index=1)
    with f3:
        top_limit = st.slider("🔝 Top N Makers", 3, 15, 7)

    # ===============================================================
    # 🔄 FETCH & BUILD DATA
    # ===============================================================
    with st.spinner(f"📡 Fetching {compare_type} {time_basis} comparison data..."):
        endpoint = ""
        if compare_type == "Statewise":
            endpoint = "vahandashboard/statewisecomparisonchart"
        elif compare_type == "Makerwise":
            endpoint = "vahandashboard/makerwisecomparisonchart"
        comp_json = fetch_json(endpoint, desc=f"{compare_type} {time_basis}")
        df_comp = to_df(comp_json)

    if not df_comp.empty:
        st.toast("✅ Comparison Data Loaded Successfully!", icon="📦")

        # ========================================
        # 🧩 DATA CLEANUP
        # ========================================
        df_comp.columns = [c.lower().strip() for c in df_comp.columns]
        label_col = next((c for c in ["label", "maker", "state", "name"] if c in df_comp.columns), "label")
        value_col = next((c for c in ["value", "count", "registeredvehiclecount"] if c in df_comp.columns), "value")

        df_comp = df_comp[[label_col, value_col]].dropna()

        # ========================================
        # 🔢 AGGREGATE BASED ON TIME
        # ========================================
        if time_basis == "Monthly":
            df_comp["month"] = pd.to_datetime(df_comp[label_col], errors="coerce").dt.strftime("%b")
            df_month = df_comp.groupby("month", as_index=False)[value_col].sum()
            df_trend = df_month
        else:
            df_trend = df_comp

        # ========================================
        # 🧱 VISUALIZATION ZONE
        # ========================================
        colL, colR = st.columns([2, 1], gap="large")

        with colL:
            st.markdown("#### 📈 Registration Trend")
            try:
                line_from_df(df_trend.rename(columns={value_col: "value", label_col: "label"}),
                             title=f"{compare_type} Registrations Over Time ({time_basis})")
            except Exception as e:
                st.error(f"⚠️ Trend chart failed: {e}")
                st.dataframe(df_trend)

        with colR:
            st.markdown(f"#### 🧱 Top {top_limit} Performers")
            try:
                df_top = df_comp.nlargest(top_limit, value_col)
                bar_from_df(df_top.rename(columns={label_col: "label", value_col: "value"}),
                            title=f"Top {top_limit} {compare_type}")
            except Exception as e:
                st.error(f"⚠️ Bar chart failed: {e}")
                st.dataframe(df_comp)

        # ========================================
        # 📊 KPI SNAPSHOT
        # ========================================
        st.markdown("<hr>", unsafe_allow_html=True)

        top_item = df_comp.loc[df_comp[value_col].idxmax(), label_col]
        total = df_comp[value_col].sum()
        top_val = df_comp[value_col].max()
        share = round((top_val / total) * 100, 2)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("🏆 Leader", top_item)
        with k2:
            st.metric("📈 Share", f"{share}%")
        with k3:
            st.metric("🚗 Total Registrations", f"{total:,}")

        # ========================================
        # 💡 INSIGHT BOX
        # ========================================
        st.markdown(f"""
        <div style="margin-top:10px;padding:14px 16px;
                    background:linear-gradient(90deg,#fff3e0,#fffaf3);
                    border:1px solid #ffdb99;border-radius:12px;
                    box-shadow:inset 0 0 8px rgba(255,140,0,0.15);">
            <b>📍 Insight:</b> <span style="color:#333;">{top_item}</span> leads across {compare_type.lower()} data, 
            contributing <b>{share}%</b> of total {time_basis.lower()} registrations.
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        # ========================================
        # 🧠 AI INSIGHTS — DEEPINFRA
        # ========================================
        if enable_ai:
            st.markdown("### 🤖 AI-Powered Comparative Summary")
            with st.expander("🔍 View AI Insight", expanded=True):
                with st.spinner("🧠 DeepInfra AI analyzing maker trends..."):
                    try:
                        sample_data = df_comp.head(10).to_dict(orient="records")
                        system_msg = (
                            "You are an expert data analyst specializing in automotive trends. "
                            "Analyze the comparative performance of vehicle makers or states "
                            "across daily/monthly registration datasets."
                        )
                        user_msg = (
                            f"Dataset sample: {json.dumps(sample_data, default=str)}. "
                            f"Provide a compact 4-sentence insight summarizing leader(s), distribution trends, "
                            "and one actionable observation."
                        )
                        ai_output = deepinfra_chat(system_msg, user_msg, max_tokens=350, temperature=0.5)

                        if ai_output.get("text"):
                            st.toast("✅ AI Comparative Summary Ready!", icon="🤖")
                            st.markdown(f"""
                            <div style="margin-top:8px;padding:16px 18px;
                                        background:linear-gradient(90deg,#fffdf6,#fffaf0);
                                        border-left:4px solid #FF8C00;border-radius:12px;">
                                <b>AI Summary:</b>
                                <p style="margin-top:6px;font-size:15px;color:#333;">
                                    {ai_output["text"]}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("💤 AI summary unavailable. Try re-running or check API key.")
                    except Exception as e:
                        st.error(f"AI generation error: {e}")

    else:
        st.warning("⚠️ No comparison data returned from API.")
        st.info("🔄 Try refreshing or verifying API connectivity.")


# ===============================================================
# 3️⃣ REGISTRATION TRENDS — COMPARISON MODE (MAXED 🚀)
# ===============================================================

import pandas as pd
import numpy as np
import plotly.express as px

# 🎨 Section Header
st.markdown("""
<div style="padding:14px 22px;border-left:6px solid #007BFF;
            background:linear-gradient(90deg,#f0f8ff 0%,#ffffff 100%);
            border-radius:16px;margin-bottom:20px;
            box-shadow:0 2px 8px rgba(0,123,255,0.1);">
    <h3 style="margin:0;font-weight:700;color:#003366;">📈 Registration Trends — Comparative Analysis</h3>
    <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
        Explore trends across months, states, makers, and daily registrations — all comparisons maxed, no forecasts or growth models.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================
# 🧭 Fetch & Normalize Data
# ======================
with st.spinner("📡 Fetching Registration Trends..."):
    tr_json = fetch_json("vahandashboard/vahanyearwiseregistrationtrend", desc="Registration Trend")
df_trend = normalize_trend(tr_json)

if df_trend is not None and not df_trend.empty:
    df_trend["date"] = pd.to_datetime(df_trend["date"])
    df_trend = df_trend.sort_values("date")

    # Basic KPIs
    total_reg = int(df_trend["value"].sum())
    period_start, period_end = df_trend["date"].min(), df_trend["date"].max()
    days = max(1, (period_end - period_start).days)
    daily_avg = df_trend["value"].sum() / days

    try:
        yoy_df = compute_yoy(df_trend)
    except Exception:
        yoy_df = pd.DataFrame()
    try:
        qoq_df = compute_qoq(df_trend)
    except Exception:
        qoq_df = pd.DataFrame()

    latest_yoy = yoy_df["YoY%"].dropna().iloc[-1] if not yoy_df.empty and "YoY%" in yoy_df else None
    latest_qoq = qoq_df["QoQ%"].dropna().iloc[-1] if not qoq_df.empty and "QoQ%" in qoq_df else None

    # Daily interpolation for smoothness
    daily_df = pd.DataFrame()
    try:
        df_ts = df_trend.set_index("date").sort_index()
        daily_idx = pd.date_range(df_ts.index.min(), df_ts.index.max(), freq="D")
        daily_df = df_ts.reindex(daily_idx).interpolate("time").rename_axis("date").reset_index()
        daily_df["daily_change%"] = daily_df["value"].pct_change().fillna(0) * 100
    except Exception:
        daily_df = pd.DataFrame()

    # ======================
    # 📊 KPI CARDS
    # ======================
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("🚘 Total Registrations", f"{total_reg:,}")
    with k2:
        st.metric("📅 Avg per Day", f"{daily_avg:.0f}")
    with k3:
        st.metric("📈 Latest YoY%", f"{latest_yoy:.2f}%" if latest_yoy is not None else "N/A")
    with k4:
        st.metric("📊 Latest QoQ%", f"{latest_qoq:.2f}%" if latest_qoq is not None else "N/A")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ======================
    # 🧩 Trend Chart (Month-wise)
    # ======================
    st.markdown("### 📆 Month-wise Registration Trend")
    try:
        fig = px.line(df_trend, x="date", y="value", markers=True, title="Monthly Registration Trend", color_discrete_sequence=["#007BFF"])
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Trend chart failed: {e}")

    # ======================
    # 📅 Daily Base Trend
    # ======================
    if not daily_df.empty:
        st.markdown("### 🗓 Daily Registration Base (Interpolated)")
        figd = px.area(daily_df.tail(90), x="date", y="value", title="Recent 90-Day Daily Registration Trend")
        st.plotly_chart(figd, use_container_width=True)

    # ======================
    # 🧭 State-wise Comparison
    # ======================
    st.markdown("### 🏛 State-wise Registrations")
    try:
        state_json = fetch_json("vahandashboard/durationWiseRegistrationTable", desc="State-wise Registrations")
        df_state = parse_duration_table(state_json)
        if not df_state.empty:
            df_state = df_state.sort_values("value", ascending=False)
            top_states = df_state.head(10)
            st.plotly_chart(px.bar(top_states, x="label", y="value", title="Top 10 States by Registrations"), use_container_width=True)
            st.dataframe(top_states, use_container_width=True)
        else:
            st.info("No state-level data available.")
    except Exception as e:
        st.warning(f"State-wise section failed: {e}")

    # ======================
    # 🏭 Maker-wise Comparison
    # ======================
    st.markdown("### 🏭 Maker-wise Registrations")
    try:
        mk_json = fetch_json("vahandashboard/top5Makerchart", desc="Top Makers")
        df_mk = parse_makers(mk_json)
        if not df_mk.empty:
            df_mk.columns = [c.lower() for c in df_mk.columns]
            if "maker" in df_mk.columns and "value" in df_mk.columns:
                st.plotly_chart(px.pie(df_mk, names="maker", values="value", title="Top Makers Market Share"), use_container_width=True)
                st.dataframe(df_mk, use_container_width=True)
        else:
            st.info("No maker-level data available.")
    except Exception as e:
        st.warning(f"Maker-wise section failed: {e}")

    # ======================
    # 📘 YoY / QoQ Comparison Tables
    # ======================
    with st.expander("📑 YoY & QoQ Comparison"):
        if not yoy_df.empty:
            st.markdown("#### Year-over-Year (YoY)")
            st.dataframe(yoy_df, use_container_width=True)
        if not qoq_df.empty:
            st.markdown("#### Quarter-over-Quarter (QoQ)")
            st.dataframe(qoq_df, use_container_width=True)

    # ======================
    # 🧠 AI Insight (DeepInfra)
    # ======================
    if enable_ai:
        st.markdown("### 🤖 AI-Generated Summary")
        with st.spinner("Analyzing comparative insights with DeepInfra AI..."):
            try:
                system = (
                    "You are an automotive data analyst summarizing national vehicle registration trends. "
                    "Highlight month-wise movement, top states, leading makers, and major shifts. "
                    "Keep the tone analytical and executive."
                )
                data_pack = {
                    "recent_trend": df_trend.tail(6).to_dict(orient="records"),
                    "top_states": df_state.head(5).to_dict(orient="records") if 'df_state' in locals() else [],
                    "top_makers": df_mk.head(5).to_dict(orient="records") if 'df_mk' in locals() else []
                }
                user = f"Here is the registration dataset summary: {json.dumps(data_pack, default=str)}. Summarize in 4–6 sentences with one recommendation."
                ai_resp = deepinfra_chat(system, user, max_tokens=350, temperature=0.4)
                if ai_resp.get("text"):
                    st.markdown(f"""
                    <div style="padding:14px 16px;background:linear-gradient(90deg,#eef5ff,#ffffff);
                                border-left:4px solid #007BFF;border-radius:12px;">
                        {ai_resp["text"]}
                    </div>
                    """, unsafe_allow_html=True)
                    st.snow()
            except Exception as e:
                st.warning(f"AI summary failed: {e}")

    # ======================
    # 🧾 Final Info Footer
    # ======================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:14px;color:#555;'>
        <b>Data Coverage:</b> {period_start.date()} → {period_end.date()} &nbsp; | &nbsp;
        <b>Total Records:</b> {len(df_trend):,} &nbsp; | &nbsp;
        <b>Comparisons:</b> Month, State, Maker, Daily
    </div>
    """, unsafe_allow_html=True)

    # celebratory confetti if positive trend
    try:
        if len(df_trend) > 2 and df_trend.iloc[-1]["value"] > df_trend.iloc[-2]["value"]:
            st.balloons()
    except Exception:
        pass

else:
    st.warning("⚠️ No registration trend data available. Please refresh or check API.")
    
# ================================================================
# 🌈 4️⃣ Duration-wise Comparative Analysis — MAXED ⚡
# ================================================================

import streamlit as st
import pandas as pd
import json

# ================== HEADER ==================
st.markdown("""
<style>
@keyframes pulseGreen {
    0% { box-shadow: 0 0 0px #22c55e; }
    50% { box-shadow: 0 0 10px #22c55e; }
    100% { box-shadow: 0 0 0px #22c55e; }
}
.maxed-header {
    background: linear-gradient(90deg, #ecfff0, #ffffff);
    border-left: 6px solid #22c55e;
    padding: 14px 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    animation: pulseGreen 3s infinite;
}
</style>

<div class="maxed-header">
    <h2 style="margin:0;">📊 Duration-wise Comparative Analytics (All Maxed)</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        Comprehensive month-wise, state-wise, and maker-wise comparisons — pure data mode, no revenue, no forecasts, no clustering.
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================
# 🔢 Helper: Fetch & Parse (Common)
# =============================================================
def fetch_and_parse(endpoint, desc):
    with st.spinner(f"Fetching {desc} data..."):
        js = fetch_json(endpoint, desc=desc)
        if js:
            try:
                df = to_df(js)
                return df
            except Exception:
                try:
                    df = pd.DataFrame(js)
                except Exception:
                    df = pd.DataFrame()
                return df
        else:
            return pd.DataFrame()


# =============================================================
# 🗓 1️⃣ Month-wise Registration Comparison
# =============================================================
st.markdown("""
<div style="padding:12px 16px;border-left:5px solid #3b82f6;
            background:linear-gradient(90deg,#f5faff,#ffffff);
            border-radius:12px;margin-bottom:10px;">
    <h3 style="margin:0;">📅 Month-wise Registration Comparison</h3>
</div>
""", unsafe_allow_html=True)

df_month = fetch_and_parse("vahandashboard/vahanyearwiseregistrationtrend", "Month-wise Registrations")

if not df_month.empty:
    df_month.columns = [c.lower() for c in df_month.columns]
    if "date" in df_month.columns:
        df_month["date"] = pd.to_datetime(df_month["date"])
        df_month["month"] = df_month["date"].dt.strftime("%b-%Y")

    if "value" not in df_month.columns:
        value_col = next((c for c in df_month.columns if c in ["count","registeredvehiclecount","total","y"]), None)
        if value_col: df_month["value"] = df_month[value_col]

    st.plotly_chart(
        px.line(df_month, x="month", y="value", markers=True, title="Monthly Registration Trend", line_shape="spline"),
        use_container_width=True
    )

    # KPIs
    try:
        total = df_month["value"].sum()
        avg = df_month["value"].mean()
        peak_m = df_month.loc[df_month["value"].idxmax(), "month"]
        peak_v = df_month["value"].max()
        k1,k2,k3 = st.columns(3)
        k1.metric("📦 Total Registrations", f"{total:,}")
        k2.metric("📈 Avg / Month", f"{avg:,.0f}")
        k3.metric("🏆 Peak Month", f"{peak_m} ({peak_v:,.0f})")
        if peak_v > avg * 1.5:
            st.balloons()
    except Exception:
        st.warning("KPI computation skipped.")

else:
    st.info("No month-wise registration data available.")


# =============================================================
# 🌍 2️⃣ State-wise Comparative Overview
# =============================================================
st.markdown("""
<div style="padding:12px 16px;border-left:5px solid #9333ea;
            background:linear-gradient(90deg,#faf5ff,#ffffff);
            border-radius:12px;margin-bottom:10px;">
    <h3 style="margin:0;">🧭 State-wise Registration Comparison</h3>
</div>
""", unsafe_allow_html=True)

df_state = fetch_and_parse("vahandashboard/durationWiseRegistrationTable", "State-wise Registrations")

if not df_state.empty and "label" in df_state.columns and "value" in df_state.columns:
    df_top = df_state.sort_values("value", ascending=False).head(15)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(df_top, x="label", y="value", title="Top 15 States by Registrations"), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(df_top, names="label", values="value", title="State Share (Top 15)"), use_container_width=True)

    # KPI metrics
    top_state = df_top.iloc[0]["label"]
    top_val = df_top.iloc[0]["value"]
    total_val = df_top["value"].sum()
    share = round((top_val / total_val) * 100, 2)
    s1,s2,s3 = st.columns(3)
    s1.metric("🏅 Top State", top_state)
    s2.metric("📊 Share", f"{share}%")
    s3.metric("🚘 Combined (Top 15)", f"{total_val:,}")

else:
    st.info("No state-wise data found.")


# =============================================================
# 🏭 3️⃣ Maker-wise Comparison (All India)
# =============================================================
st.markdown("""
<div style="padding:12px 16px;border-left:5px solid #f97316;
            background:linear-gradient(90deg,#fff7f0,#ffffff);
            border-radius:12px;margin-bottom:10px;">
    <h3 style="margin:0;">🏭 Maker-wise Registration Comparison</h3>
</div>
""", unsafe_allow_html=True)

df_maker = fetch_and_parse("vahandashboard/top5Makerchart", "Maker-wise Registrations")

if not df_maker.empty and "label" in df_maker.columns and "value" in df_maker.columns:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(df_maker, x="label", y="value", title="Top Makers by Registrations"), use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(df_maker, names="label", values="value", title="Maker Market Share"), use_container_width=True)

    # KPIs
    try:
        lead_maker = df_maker.loc[df_maker["value"].idxmax(), "label"]
        val = df_maker["value"].max()
        total_val = df_maker["value"].sum()
        pct = round((val / total_val) * 100, 2)
        k1,k2,k3 = st.columns(3)
        k1.metric("🏆 Leading Maker", lead_maker)
        k2.metric("📈 Share", f"{pct}%")
        k3.metric("🚗 Total", f"{total_val:,}")
    except Exception:
        st.warning("Maker KPI error.")
else:
    st.info("No maker-wise data returned.")


# =============================================================
# 📆 4️⃣ Daily Base Comparison (Interpolated)
# =============================================================
st.markdown("""
<div style="padding:12px 16px;border-left:5px solid #10b981;
            background:linear-gradient(90deg,#f5fff9,#ffffff);
            border-radius:12px;margin-bottom:10px;">
    <h3 style="margin:0;">📅 Daily Base Comparison (Interpolated)</h3>
</div>
""", unsafe_allow_html=True)

try:
    if not df_month.empty and "date" in df_month.columns:
        df_daily = df_month.set_index("date").sort_index()
        daily_index = pd.date_range(df_daily.index.min(), df_daily.index.max(), freq="D")
        df_daily = df_daily.reindex(daily_index).interpolate(method="time").rename_axis("date").reset_index()
        df_daily["day"] = df_daily["date"].dt.strftime("%d-%b")

        st.plotly_chart(px.area(df_daily.tail(90), x="day", y="value",
                                title="Daily Registration Trend (last 90 days)",
                                markers=False),
                        use_container_width=True)
        st.metric("📆 Days Covered", len(df_daily))
    else:
        st.info("Cannot generate daily view — no monthly trend data.")
except Exception as e:
    st.warning(f"Daily comparison error: {e}")


# =============================================================
# 🤖 AI Narrative — All Maxed Summary
# =============================================================
if enable_ai:
    with st.expander("🤖 AI Comparative Summary", expanded=True):
        try:
            combined_sample = {
                "monthly": df_month.head(10).to_dict(orient='records') if not df_month.empty else [],
                "statewise": df_state.head(10).to_dict(orient='records') if not df_state.empty else [],
                "makerwise": df_maker.head(10).to_dict(orient='records') if not df_maker.empty else [],
            }
            system = (
                "You are an analytics assistant summarizing India's vehicle registration patterns. "
                "Compare month-wise, state-wise, and maker-wise trends. Mention the strongest performer "
                "in each dimension and one actionable observation for each."
            )
            user = f"Here is the combined dataset: {json.dumps(combined_sample, default=str)}"
            ai_resp = deepinfra_chat(system, user, max_tokens=400)
            if isinstance(ai_resp, dict) and "text" in ai_resp:
                st.markdown(f"""
                <div style="padding:12px 16px;margin-top:8px;
                            background:linear-gradient(90deg,#ffffff,#f9fffa);
                            border-left:4px solid #10b981;
                            border-radius:10px;">
                    {ai_resp["text"]}
                </div>
                """, unsafe_allow_html=True)
                st.toast("AI Summary Ready!", icon="🤖")
        except Exception as e:
            st.error(f"AI Summary failed: {e}")


# ================================================================
# 📊 7️⃣ Comparative Analytics — Month • State • Maker • Daily (MAXED)
# ================================================================

import streamlit as st
import pandas as pd
import altair as alt
import json

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
  0% {box-shadow: 0 0 0px #2196f3;}
  50% {box-shadow: 0 0 10px #2196f3;}
  100% {box-shadow: 0 0 0px #2196f3;}
}
.-container {
  background: linear-gradient(90deg,#f0f8ff,#ffffff);
  border-left: 6px solid #2196f3;
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
  box-shadow: 0 5px 15px rgba(33,150,243,0.3);
}
.ai-box {
  background: linear-gradient(90deg,#ffffff,#f2f9ff);
  border-left: 4px solid #2196f3;
  border-radius: 10px;
  padding: 12px 14px;
  margin-top: 8px;
  animation: slideIn 1s ease;
}
</style>
""", unsafe_allow_html=True)


# ======================
# 🧭 Section Header
# ======================
st.markdown("""
<div class="-container">
    <h2 style="margin:0;">📊 Comparative Analytics</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        State-wise, Maker-wise, Month-wise, and Daily base comparison — all filters enabled and AI-powered insights.
    </p>
</div>
""", unsafe_allow_html=True)


# ======================
# ⚙️ Data Fetch (Generic Placeholder)
# ======================
# Replace with your actual Vahan API fetchers
df_month = fetch_json("vahandashboard/monthComparison", desc="Month-wise Data")
df_state = fetch_json("vahandashboard/stateComparison", desc="State-wise Data")
df_maker = fetch_json("vahandashboard/makerComparison", desc="Maker-wise Data")
df_daily = fetch_json("vahandashboard/dailyComparison", desc="Daily Data")

# Convert to DataFrames (safe)
df_month = pd.DataFrame(df_month) if df_month else pd.DataFrame()
df_state = pd.DataFrame(df_state) if df_state else pd.DataFrame()
df_maker = pd.DataFrame(df_maker) if df_maker else pd.DataFrame()
df_daily = pd.DataFrame(df_daily) if df_daily else pd.DataFrame()

# ======================
# 🔀 Tabs for Comparison Modes
# ======================
tabs = st.tabs(["🗓️ Month-wise", "🌍 State-wise", "🏭 Maker-wise", "📅 Daily"])

# ------------------ MONTH-WISE ------------------
with tabs[0]:
    st.subheader("🗓️ Month-wise Comparison")
    if df_month.empty:
        st.warning("No month-wise data available.")
    else:
        try:
            chart = (
                alt.Chart(df_month)
                .mark_line(point=True)
                .encode(
                    x="month:O", y="value:Q",
                    color="category:N",
                    tooltip=["month", "value", "category"]
                )
                .properties(height=380, title="Month-wise Comparison")
            )
            st.altair_chart(chart, use_container_width=True)

            total = df_month["value"].sum()
            avg = df_month["value"].mean()
            latest = df_month["value"].iloc[-1]
            prev = df_month["value"].iloc[-2] if len(df_month) > 1 else latest
            growth = ((latest - prev) / prev) * 100 if prev else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card'><h4>📆 Total</h4><b>{total:,.0f}</b></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><h4>⚖️ Avg</h4><b>{avg:,.0f}</b></div>", unsafe_allow_html=True)
            color = "green" if growth > 0 else "red"
            c3.markdown(f"<div class='metric-card'><h4>📈 Growth %</h4><b style='color:{color};'>{growth:.2f}%</b></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><h4>🧩 Latest Month</h4><b>{latest:,.0f}</b></div>", unsafe_allow_html=True)

            if enable_ai:
                with st.spinner("🤖 Generating AI month-wise insight..."):
                    system = "You are a data analyst comparing month-wise trends."
                    user = f"Data: {df_month.head(10).to_dict(orient='records')} Summarize top 2 insights and key month trends."
                    ai = deepinfra_chat(system, user, max_tokens=200)
                    if isinstance(ai, dict) and 'text' in ai:
                        st.markdown(f"<div class='ai-box'>{ai['text']}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Month-wise comparison failed: {e}")

# ------------------ STATE-WISE ------------------
with tabs[1]:
    st.subheader("🌍 State-wise Comparison")
    if df_state.empty:
        st.warning("No state-wise data available.")
    else:
        try:
            chart = (
                alt.Chart(df_state)
                .mark_bar()
                .encode(
                    x="state:N", y="value:Q",
                    color="state:N",
                    tooltip=["state", "value"]
                )
                .properties(height=400, title="State-wise Comparison")
            )
            st.altair_chart(chart, use_container_width=True)

            top_state = df_state.loc[df_state['value'].idxmax()]
            bottom_state = df_state.loc[df_state['value'].idxmin()]
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='metric-card'><h4>🏆 Top State</h4><b>{top_state['state']} ({top_state['value']:,.0f})</b></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><h4>🔻 Lowest State</h4><b>{bottom_state['state']} ({bottom_state['value']:,.0f})</b></div>", unsafe_allow_html=True)

            if enable_ai:
                with st.spinner("🤖 Generating AI state-wise insight..."):
                    system = "You are a regional performance analyst."
                    user = f"Top and bottom states with values: {json.dumps({'top': top_state.to_dict(), 'bottom': bottom_state.to_dict()})}. Compare performance gaps and suggest 2 recommendations."
                    ai = deepinfra_chat(system, user, max_tokens=220)
                    if isinstance(ai, dict) and 'text' in ai:
                        st.markdown(f"<div class='ai-box'>{ai['text']}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"State-wise comparison failed: {e}")

# ------------------ MAKER-WISE ------------------
with tabs[2]:
    st.subheader("🏭 Maker-wise Comparison")
    if df_maker.empty:
        st.warning("No maker-wise data available.")
    else:
        try:
            chart = (
                alt.Chart(df_maker)
                .mark_bar()
                .encode(
                    x=alt.X("maker:N", sort='-y'),
                    y="value:Q",
                    color="maker:N",
                    tooltip=["maker", "value"]
                )
                .properties(height=380, title="Maker-wise Comparison")
            )
            st.altair_chart(chart, use_container_width=True)

            top_maker = df_maker.loc[df_maker['value'].idxmax()]
            bottom_maker = df_maker.loc[df_maker['value'].idxmin()]
            st.markdown(f"<div class='metric-card'><h4>⭐ Top Maker</h4><b>{top_maker['maker']}</b> — {top_maker['value']:,.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><h4>⚙️ Lowest Maker</h4><b>{bottom_maker['maker']}</b> — {bottom_maker['value']:,.0f}</div>", unsafe_allow_html=True)

            if enable_ai:
                with st.spinner("🤖 Generating AI maker-wise insight..."):
                    system = "You are a market analyst summarizing manufacturer performance."
                    user = f"Maker data: {df_maker.head(10).to_dict(orient='records')}. Provide 3 insights comparing top vs bottom performers."
                    ai = deepinfra_chat(system, user, max_tokens=220)
                    if isinstance(ai, dict) and 'text' in ai:
                        st.markdown(f"<div class='ai-box'>{ai['text']}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Maker-wise comparison failed: {e}")

# ------------------ DAILY ------------------
with tabs[3]:
    st.subheader("📅 Daily Comparison")
    if df_daily.empty:
        st.warning("No daily data available.")
    else:
        try:
            chart = (
                alt.Chart(df_daily)
                .mark_line(point=True)
                .encode(
                    x="date:T", y="value:Q",
                    color="category:N",
                    tooltip=["date", "value", "category"]
                )
                .properties(height=380, title="Daily Comparison")
            )
            st.altair_chart(chart, use_container_width=True)

            avg_day = df_daily["value"].mean()
            max_day = df_daily.loc[df_daily['value'].idxmax()]
            st.markdown(f"<div class='metric-card'><h4>📆 Avg Daily</h4><b>{avg_day:,.0f}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><h4>🌟 Peak Day</h4><b>{max_day['date']}</b> — {max_day['value']:,.0f}</div>", unsafe_allow_html=True)

            if enable_ai:
                with st.spinner("🤖 Generating AI daily insight..."):
                    system = "You are a data analyst summarizing daily fluctuations."
                    user = f"Daily data: {df_daily.head(10).to_dict(orient='records')}. Explain 3 key trends and volatility insights."
                    ai = deepinfra_chat(system, user, max_tokens=200)
                    if isinstance(ai, dict) and 'text' in ai:
                        st.markdown(f"<div class='ai-box'>{ai['text']}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Daily comparison failed: {e}")

# ============================================================
# 💾 SMART EXCEL EXPORT — MAXED COMPARISON ANALYTICS EDITION
# ============================================================

st.markdown("""
<div style="padding:18px 20px;border-left:5px solid #007bff;
            background:linear-gradient(90deg,#f0f8ff,#ffffff);
            border-radius:12px;margin-top:25px;margin-bottom:15px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <h2 style="margin:0;">💾 Smart Excel Export — Maxed Comparison Edition</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        Export all <b>month-wise</b>, <b>state-wise</b>, <b>maker-wise</b> and <b>daily-base</b> analytics 
        into one unified, <b>styled Excel report</b> — fully ready for analysis or presentation.
    </p>
</div>
""", unsafe_allow_html=True)

with st.container():
    with st.expander("📊 Generate & Download Maxed Analytics Report", expanded=True):

        st.markdown("""
        <div style="background:linear-gradient(90deg,#e8f0fe,#ffffff);
                    border-left:5px solid #007bff;padding:10px 18px;
                    border-radius:10px;margin-bottom:10px;">
            <b>💡 Tip:</b> Ensure all comparison data is loaded before export for best results.
        </div>
        """, unsafe_allow_html=True)

        # ✅ Load all active dataframes from session/local scope
        df_cat = locals().get("df_cat", pd.DataFrame())
        df_mk = locals().get("df_mk", pd.DataFrame())
        df_trend = locals().get("df_trend", pd.DataFrame())
        yoy_df = locals().get("yoy_df", pd.DataFrame())
        qoq_df = locals().get("qoq_df", pd.DataFrame())
        daily_df = locals().get("daily_df", pd.DataFrame())
        monthwise_df = locals().get("monthwise_df", pd.DataFrame())
        statewise_df = locals().get("statewise_df", pd.DataFrame())
        makerwise_df = locals().get("makerwise_df", pd.DataFrame())

        datasets = {
            "Category Overview": df_cat,
            "Top Makers": df_mk,
            "Registrations Trend": df_trend,
            "YoY Comparison": yoy_df,
            "MoM Comparison": qoq_df,
            "Daily Base Comparison": daily_df,
            "Month-wise State Comparison": monthwise_df,
            "Maker-wise State Comparison": makerwise_df,
            "State-wise Summary": statewise_df
        }

        # 🧠 AI Summaries (Optional)
        summaries = {}
        if 'enable_ai' in locals() and enable_ai:
            try:
                st.info("🤖 Generating AI summaries for each dataset...")
                progress = st.progress(0)
                for i, (name, df) in enumerate(datasets.items()):
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        try:
                            sys_prompt = f"You are a data analyst. Summarize key patterns in {name} in 2–3 concise lines."
                            user_prompt = f"Here is a sample of the dataset: {df.head(10).to_dict(orient='records')}"
                            ai_resp = deepinfra_chat(sys_prompt, user_prompt, max_tokens=150)
                            summaries[name] = ai_resp.get("text", "No summary generated.")
                        except Exception as e:
                            summaries[name] = f"AI summary failed: {e}"
                    progress.progress((i + 1) / len(datasets))
                progress.empty()

                if summaries:
                    ai_df = pd.DataFrame(list(summaries.items()), columns=["Dataset", "AI Summary"])
                    datasets["AI Insights"] = ai_df

                    with st.expander("🧠 View AI Insights"):
                        for name, text in summaries.items():
                            st.markdown(f"**{name}**")
                            st.write(text)
                            st.markdown("---")
            except Exception as e:
                st.warning(f"⚠️ AI summaries skipped: {e}")

        # ⚠️ Handle empty datasets
        if all((not isinstance(df, pd.DataFrame)) or df.empty for df in datasets.values()):
            st.warning("⚠️ No valid datasets to export. Only a summary sheet will be created.")

        # 📦 Compile Excel Workbook (Styled)
        with st.spinner("📦 Compiling Excel workbook..."):
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

            # Apply formatting and charts
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
                # Body styling
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                        cell.border = border
                # Auto column width
                for col in ws.columns:
                    max_len = max(len(str(c.value or "")) for c in col)
                    ws.column_dimensions[get_column_letter(col[0].column)].width = max_len + 3
                # Add basic chart
                if ws.max_row > 2 and ws.max_column >= 2:
                    try:
                        val_ref = Reference(ws, min_col=2, min_row=1, max_row=ws.max_row)
                        cat_ref = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
                        chart = LineChart()
                        chart.title = f"{sheet} Overview"
                        chart.y_axis.title = "Values"
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

        # 🎉 Download Button
        ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
        st.download_button(
            label="⬇️ Download Full Comparison Excel Report",
            data=styled.getvalue(),
            file_name=f"Vahan_MaxedComparison_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.success("✅ Maxed Comparison Export Ready — Includes all major datasets & AI summaries.")
        st.toast("Smart Excel Report ready for download 🎉")
        st.balloons()

# ============================================================
# 🧩 RAW JSON PREVIEW (Developer Debug Mode) — MAXED VERSION
# ============================================================

with st.expander("🛠️ Raw JSON Preview (Developer Debug Mode)", expanded=False):
    st.caption("Inspect raw API responses returned from each Parivahan/Vahan endpoint. Use only for debugging or data verification.")

    # ---------- Safe access to JSON variables ----------
    cat_json       = locals().get("cat_json", None)
    mk_json        = locals().get("mk_json", None)
    tr_json        = locals().get("tr_json", None)
    month_json     = locals().get("month_json", None)
    state_json     = locals().get("state_json", None)
    maker_json     = locals().get("maker_json", None)
    daily_json     = locals().get("daily_json", None)

    # ---------- Control Bar ----------
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 2])
    with ctrl_col1:
        show_pretty = st.checkbox("🔎 Pretty / Expand JSON by default", value=False)
    with ctrl_col2:
        snapshot_name = st.text_input("Snapshot filename", value=f"vahan_snapshot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}")
    with ctrl_col3:
        save_snapshot = st.button("💾 Save Snapshot")

    st.markdown("---")

    left, right = st.columns(2)

    def _render_json(title: str, data):
        st.markdown(f"**{title}**")
        if data is None:
            st.info("No data available for this endpoint.")
            return
        try:
            meta = {"type": type(data).__name__, "count": len(data) if isinstance(data, (list, dict)) else "n/a"}
            st.caption(f"Meta: {meta}")
        except Exception:
            pass
        if show_pretty:
            try:
                st.code(json.dumps(data, indent=2, default=str), language="json")
            except Exception:
                st.write(data)
        else:
            try:
                st.json(data)
            except Exception:
                st.write(data)

        # Per-block download
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

    # ---------- Render All JSON Blocks ----------
    with left:
        _render_json("📦 Category JSON", cat_json)
        st.markdown("---")
        _render_json("🏭 Maker JSON", maker_json)
        st.markdown("---")
        _render_json("📊 Month-wise JSON", month_json)

    with right:
        _render_json("🌍 State-wise JSON", state_json)
        st.markdown("---")
        _render_json("📅 Daily Base JSON", daily_json)
        st.markdown("---")
        _render_json("📈 Trend JSON", tr_json)

    st.markdown("---")

    if save_snapshot:
        try:
            combined = {
                "generated_at": pd.Timestamp.now().isoformat(),
                "category_json": cat_json,
                "maker_json": maker_json,
                "trend_json": tr_json,
                "monthwise_json": month_json,
                "statewise_json": state_json,
                "daily_json": daily_json,
            }
            payload = json.dumps(combined, indent=2, default=str).encode("utf-8")
            st.download_button(
                label="⬇️ Download Combined Snapshot (.json)",
                data=payload,
                file_name=f"{snapshot_name}.json",
                mime="application/json"
            )
            st.success("✅ Snapshot ready for download.")
        except Exception as e:
            st.error(f"Snapshot generation failed: {e}")

    st.info("🔒 Raw JSON preview is meant for diagnostics. Disable in production builds.")

# ============================================================
# 📊 FOOTER KPIs + EXECUTIVE SUMMARY — MAXED COMPARISON DATA
# ============================================================

st.markdown("---")
st.subheader("📊 Dashboard Summary & Insights — Comparison Analytics")

# ============================================================
# 🎯 KPI Cards (Core Metrics)
# ============================================================

kpi_cols = st.columns(4)

with kpi_cols[0]:
    if "df_trend" in locals() and not df_trend.empty:
        total_reg = int(df_trend["value"].sum())
        st.metric("🧾 Total Registrations", f"{total_reg:,}")
    else:
        st.metric("🧾 Total Registrations", "N/A")

with kpi_cols[1]:
    if "daily_df" in locals() and not daily_df.empty:
        daily_avg = int(daily_df["value"].mean())
        st.metric("📅 Daily Average", f"{daily_avg:,}")
    else:
        st.metric("📅 Daily Average", "N/A")

with kpi_cols[2]:
    if "monthwise_df" in locals() and not monthwise_df.empty:
        latest_month = monthwise_df.iloc[-1]["label"]
        latest_val = int(monthwise_df.iloc[-1]["value"])
        st.metric("🗓️ Latest Month Registrations", f"{latest_val:,}", help=f"Month: {latest_month}")
    else:
        st.metric("🗓️ Latest Month Registrations", "N/A")

with kpi_cols[3]:
    if "statewise_df" in locals() and not statewise_df.empty:
        top_state = statewise_df.loc[statewise_df["value"].idxmax(), "label"]
        top_val = statewise_df["value"].max()
        st.metric("🌍 Top Performing State", f"{top_state} ({int(top_val):,})")
    else:
        st.metric("🌍 Top Performing State", "N/A")

# ============================================================
# 🧭 Maker-wise and State Insights
# ============================================================

if "makerwise_df" in locals() and not makerwise_df.empty:
    top_maker = makerwise_df.loc[makerwise_df["value"].idxmax(), "label"]
    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,#007bff,#00c851);
                    padding:15px;border-radius:10px;color:white;
                    text-align:center;font-size:1.1em;margin-top:15px;'>
            🏭 <b>Top Maker:</b> {top_maker}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 🤖 AI Executive Summary (DeepInfra)
# ============================================================

if "enable_ai" in locals() and enable_ai:
    st.markdown("### 🤖 Executive AI Summary")
    with st.spinner("Synthesizing executive-level summary..."):
        try:
            context = {
                "total_registrations": total_reg if "total_reg" in locals() else None,
                "daily_average": daily_avg if "daily_avg" in locals() else None,
                "top_state": top_state if "top_state" in locals() else None,
                "top_maker": top_maker if "top_maker" in locals() else None,
                "latest_month": latest_month if "latest_month" in locals() else None,
                "latest_month_value": latest_val if "latest_val" in locals() else None,
            }

            system = (
                "You are an AI analytics assistant summarizing performance metrics "
                "for national vehicle registration data (no revenue or forecasting). "
                "Focus on month-wise, state-wise, maker-wise, and daily-base patterns "
                "to create a crisp executive narrative in 4–5 sentences."
            )
            user = (
                f"Context data: {json.dumps(context, default=str)}\n"
                "Summarize trends, highlight top regions and makers, and end with one strategic recommendation."
            )

            ai_resp = deepinfra_chat(system, user, max_tokens=300)
            ai_text = ai_resp.get("text", "Summary not generated.") if isinstance(ai_resp, dict) else str(ai_resp)

            st.markdown(
                f"""
                <div style='background-color:#f0f9ff;border-left:5px solid #2196f3;
                            padding:15px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1);'>
                    <b>AI Executive Summary:</b><br>{ai_text}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.toast("✅ Executive Summary generated successfully.")
        except Exception as e:
            st.error(f"AI summary generation failed: {e}")
else:
    st.info("🤖 AI Executive Summary disabled — enable 'AI Narratives' to activate.")

# ============================================================
# ✨ Footer Branding
# ============================================================

st.markdown(
    """
    <hr style="border: 1px solid #444; margin-top: 2em; margin-bottom: 1em;">
    <div style="text-align:center; color:gray; font-size:0.9em;">
        🚗 <b>Parivahan Analytics — Comparison Suite 2025</b><br>
        <span style="color:#aaa;">Month-wise • State-wise • Maker-wise • Daily-base KPIs</span><br><br>
        <i>Empowering precision in public transport analytics.</i>
    </div>
    """,
    unsafe_allow_html=True,
)

st.balloons()
st.toast("✅ Dashboard Summary Loaded — All Comparison Datasets Synced.")
