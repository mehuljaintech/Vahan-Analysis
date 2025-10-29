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
# 🚀 PARIVAHAN ANALYTICS —  HYBRID UI ENGINE
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
    st.toast("🚀 Welcome to Parivahan Analytics —  Hybrid Experience!", icon="🌍")
    st.balloons()

# =====================================================
# 🧭 SIDEBAR — DYNAMIC FILTER PANEL ()
# =====================================================
today = date.today()
default_from_year = max(2017, today.year - 1)

# =====================================================
# 🌈 MAXED SIDEBAR — GLASS NEON THEME
# =====================================================
st.sidebar.markdown("""
<style>
/* Sidebar Container */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030712 0%, #0f172a 60%, #1e293b 100%);
    color: #E2E8F0;
    box-shadow: 0 0 25px rgba(0,255,255,0.15);
    border-right: 1px solid rgba(0,255,255,0.1);
    animation: slideIn 1.2s ease-in-out;
    backdrop-filter: blur(20px);
}

/* Smooth entrance animation */
@keyframes slideIn {
  from {opacity: 0; transform: translateX(-25px);}
  to {opacity: 1; transform: translateX(0);}
}

/* Sidebar Sections */
.sidebar-section {
    padding: 14px 10px;
    margin: 10px 0;
    border-radius: 14px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,255,255,0.15);
    box-shadow: 0 4px 18px rgba(0,255,255,0.08);
    transition: all 0.35s ease-in-out;
}

/* Hover Glow Effect */
.sidebar-section:hover {
    background: rgba(0,224,255,0.1);
    border-color: rgba(0,255,255,0.4);
    box-shadow: 0 6px 20px rgba(0,255,255,0.3);
    transform: translateY(-3px) scale(1.02);
}

/* Section Headings */
.sidebar-section h4 {
    color: #00E0FF;
    margin-bottom: 8px;
    font-size: 16px;
    letter-spacing: 0.5px;
    text-shadow: 0 0 12px rgba(0,255,255,0.6);
}

/* Input + Select fields */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background: rgba(0,0,0,0.3);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.15);
    transition: all 0.2s ease-in-out;
}
div[data-baseweb="input"]:hover, div[data-baseweb="select"]:hover {
    border-color: rgba(0,255,255,0.4);
    box-shadow: 0 0 12px rgba(0,255,255,0.2);
}

/* Toggle / Checkbox style */
.stCheckbox, .stSwitch {
    accent-color: #00E0FF !important;
}

/* Sidebar Title Block */
.sidebar-header {
    text-align: center;
    padding: 15px 0 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 15px;
}
.sidebar-header h2 {
    color: #00E0FF;
    text-shadow: 0 0 18px rgba(0,255,255,0.8);
    font-weight: 800;
    font-size: 22px;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.sidebar-header p {
    font-size: 13px;
    color: #9CA3AF;
    opacity: 0.8;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# ✨ SIDEBAR HEADER — MAXED CONTROL PANEL
# =====================================================
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>⚙️ Control Panel</h2>
    <p>Customize analytics, filters, and AI insights dynamically.</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 📊 MAXED DATA FILTERS (Supports ANY filter)
# =====================================================
with st.sidebar.expander("📊 Data Filters", expanded=True):
    from_year = st.number_input("📅 From Year", min_value=2012, max_value=today.year, value=default_from_year)
    to_year = st.number_input("📆 To Year", min_value=from_year, max_value=today.year, value=today.year)

    col1, col2 = st.columns(2)
    with col1:
        state_code = st.text_input("🏙️ State Code", value="", placeholder="Blank = All-India")
    with col2:
        rto_code = st.text_input("🏢 RTO Code", value="0", placeholder="0 = aggregate")

    vehicle_classes = st.text_input("🚘 Vehicle Classes", value="", placeholder="e.g. 2W, 3W, 4W")
    vehicle_makers = st.text_input("🏭 Vehicle Makers", value="", placeholder="Comma-separated or IDs")
    vehicle_type = st.text_input("🛻 Vehicle Type", value="", placeholder="Optional: EV/Diesel/Petrol")
    region_filter = st.text_input("🗺️ Region Filter", value="", placeholder="North / South / East / West (optional)")
    month_filter = st.selectbox("🗓️ Month Filter", ["All", "January", "February", "March", "April", "May", "June",
                                                    "July", "August", "September", "October", "November", "December"], index=0)

    col1, col2 = st.columns(2)
    with col1:
        time_period = st.selectbox("⏱️ Time Period", ["All Time", "Yearly", "Monthly", "Daily"], index=0)
    with col2:
        fitness_check = st.selectbox("🧾 Fitness Check", ["All", "Only Fit", "Expired"], index=0)

    vehicle_age = st.slider("📆 Vehicle Age (years)", 0, 20, (0, 10))
    fuel_type = st.multiselect("⛽ Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"], default=[])

    if st.button("🔄 Reset Filters"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.toast("♻️ Filters reset — applying defaults...", icon="🔁")
        st.experimental_rerun()

# =====================================================
# 🧠 SMART ANALYTICS & AI ENGINE — MAXED
# =====================================================
with st.sidebar.expander("🧠 Smart Analytics & AI Engine", expanded=True):
    enable_forecast = st.checkbox("📈 Enable Forecasting", value=True)
    enable_anomaly = st.checkbox("⚠️ Enable Anomaly Detection", value=True)
    enable_clustering = st.checkbox("🔍 Enable Clustering", value=True)
    enable_ai = st.checkbox("🤖 DeepInfra AI Narratives", value=False)

    forecast_periods = st.number_input("⏳ Forecast Horizon (months)", min_value=1, max_value=36, value=3)
    enable_trend = st.checkbox("📊 Trend Line Overlay", value=True)
    enable_comparison = st.checkbox("📅 Year/Month Comparison", value=True)

    st.markdown("##### ⚡ AI Presets")
    preset = st.radio(
        "Choose Mode:",
        ["Balanced (Default)", "Aggressive Forecasting", "Minimal Analysis", "Custom Maxed Mode"],
        index=0,
        horizontal=True
    )

    if preset == "Aggressive Forecasting":
        enable_forecast, enable_anomaly, enable_clustering = True, True, True
        forecast_periods = 12
        st.toast("🚀 Aggressive Forecasting (12-month horizon) enabled!", icon="✨")

    elif preset == "Minimal Analysis":
        enable_forecast = enable_anomaly = enable_clustering = enable_ai = False
        st.toast("💤 Minimal Analysis Mode Activated", icon="⚙️")

    elif preset == "Custom Maxed Mode":
        enable_forecast = enable_anomaly = enable_clustering = enable_ai = True
        forecast_periods = 24
        enable_comparison = enable_trend = True
        st.toast("💎 Custom MAXED Mode — all analytics active!", icon="⚡")

    st.markdown("""
    <hr style='margin:10px 0;border:none;height:1px;
    background:linear-gradient(90deg,transparent,#00E0FF66,transparent);'>
    <p style='text-align:center;font-size:12px;opacity:0.7;'>
        🧩 All filters and AI toggles auto-refresh dashboards instantly.
    </p>
    """, unsafe_allow_html=True)

# =====================================================
# 🎨 UNIVERSAL HYBRID THEME ENGINE — MAXED EDITION 🚀
# =====================================================

THEMES = {
    "VSCode": {
        "bg": "#0E101A",
        "text": "#D4D4D4",
        "card": "#1E1E2E",
        "accent": "#007ACC",
        "glow": "rgba(0,122,204,0.6)"
    },
    "Glass": {
        "bg": "rgba(15,23,42,0.9)",
        "text": "#E0F2FE",
        "card": "rgba(255,255,255,0.06)",
        "accent": "#00E0FF",
        "glow": "rgba(0,224,255,0.5)"
    },
    "Neumorphic": {
        "bg": "#E5E9F0",
        "text": "#1E293B",
        "card": "#F8FAFC",
        "accent": "#0078FF",
        "glow": "rgba(0,120,255,0.35)"
    },
    "Gradient": {
        "bg": "linear-gradient(135deg,#0F172A,#1E3A8A)",
        "text": "#E0F2FE",
        "card": "rgba(255,255,255,0.05)",
        "accent": "#38BDF8",
        "glow": "rgba(56,189,248,0.4)"
    },
    "High Contrast": {
        "bg": "#000000",
        "text": "#FFFFFF",
        "card": "#111111",
        "accent": "#FFDE00",
        "glow": "rgba(255,222,0,0.6)"
    },
    "Windows": {
        "bg": "linear-gradient(120deg,#0078D7,#003C8F)",
        "text": "#FFFFFF",
        "card": "rgba(255,255,255,0.08)",
        "accent": "#00B7FF",
        "glow": "rgba(0,183,255,0.45)"
    },
    "MacOS": {
        "bg": "linear-gradient(120deg,#FFFFFF,#EEF2FF)",
        "text": "#111827",
        "card": "rgba(255,255,255,0.85)",
        "accent": "#007AFF",
        "glow": "rgba(0,122,255,0.4)"
    },
    "Fluent": {
        "bg": "linear-gradient(120deg,#0E1624,#1B2838)",
        "text": "#E6F0FF",
        "card": "rgba(255,255,255,0.04)",
        "accent": "#0099FF",
        "glow": "rgba(0,153,255,0.4)"
    },
    "Aurora": {
        "bg": "linear-gradient(135deg,#0f172a,#312e81,#16a34a)",
        "text": "#e0f2fe",
        "card": "rgba(255,255,255,0.05)",
        "accent": "#22d3ee",
        "glow": "rgba(34,211,238,0.4)"
    },
    "Matrix": {
        "bg": "#000000",
        "text": "#00FF41",
        "card": "rgba(0,255,65,0.05)",
        "accent": "#00FF41",
        "glow": "rgba(0,255,65,0.5)"
    },
    "Cyberpunk": {
        "bg": "linear-gradient(135deg,#1a002b,#ff00ff,#00ffff)",
        "text": "#E0E0E0",
        "card": "rgba(255,255,255,0.08)",
        "accent": "#00FFFF",
        "glow": "rgba(0,255,255,0.5)"
    },
    "Neon Glass": {
        "bg": "radial-gradient(circle at 20% 30%, #0f2027, #203a43, #2c5364)",
        "text": "#E6F9FF",
        "card": "rgba(255,255,255,0.05)",
        "accent": "#00E0FF",
        "glow": "rgba(0,224,255,0.45)"
    },
    "Terminal": {
        "bg": "#000000",
        "text": "#33FF00",
        "card": "rgba(0,0,0,0.8)",
        "accent": "#33FF00",
        "glow": "rgba(51,255,0,0.5)"
    },
    "Solarized": {
        "bg": "#002b36",
        "text": "#93a1a1",
        "card": "#073642",
        "accent": "#b58900",
        "glow": "rgba(181,137,0,0.4)"
    },
    "Monokai": {
        "bg": "#272822",
        "text": "#f8f8f2",
        "card": "#383830",
        "accent": "#f92672",
        "glow": "rgba(249,38,114,0.4)"
    }
}

# Sidebar controls
st.sidebar.markdown("## 🎨 Appearance & Layout")
ui_mode = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index("VSCode"))
font_size = st.sidebar.slider("Font Size", 12, 20, 15)
radius = st.sidebar.slider("Corner Radius", 6, 24, 12)
motion = st.sidebar.toggle("✨ Motion & Glow Effects", value=True)
palette = THEMES[ui_mode]

# CSS builder
def build_css(palette, font_size, radius, motion):
    accent, text, bg, card, glow = palette["accent"], palette["text"], palette["bg"], palette["card"], palette["glow"]
    effect = f"0 0 18px {glow}" if motion else "none"
    return f"""
    <style>
    html, body, .stApp {{
        background: {bg};
        color: {text};
        font-size: {font_size}px;
        font-family: 'Inter', 'Segoe UI', 'SF Pro Display', sans-serif;
        transition: all 0.4s ease-in-out;
    }}
    .block-container {{
        max-width: 1300px;
        padding: 1.5rem 2rem 3rem 2rem;
    }}
    h1, h2, h3, h4, h5 {{
        color: {accent};
        text-shadow: {effect};
        font-weight: 800;
    }}
    div.stButton > button {{
        background: {accent};
        color: white;
        border: none;
        border-radius: {radius}px;
        padding: 0.6rem 1.1rem;
        transition: all 0.25s ease-in-out;
        font-weight: 600;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 20px {accent}77;
    }}
    .glass-card {{
        background: {card};
        backdrop-filter: blur(10px);
        border-radius: {radius}px;
        padding: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    }}
    [data-testid="stSidebar"] {{
        background: {card};
        border-right: 1px solid {accent}33;
        box-shadow: 4px 0 12px rgba(0,0,0,0.1);
    }}
    [data-testid="stMetricValue"] {{
        color: {accent} !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
    }}
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {accent}66, transparent);
        margin: 1rem 0;
    }}
    </style>
    """

# Apply dynamic CSS
st.markdown(build_css(palette, font_size, radius, motion), unsafe_allow_html=True)


# =====================================================
# 🧩 BUILD DYNAMIC CSS — Supports Motion / Glow / Themes
# =====================================================
def build_css(palette, font_size, radius, motion):
    accent = palette["accent"]
    text = palette["text"]
    bg = palette["bg"]
    card = palette["card"]
    glow = palette["glow"] if motion else "none"

    return f"""
    <style>
    html, body, .stApp {{
        background: {bg};
        color: {text};
        font-size: {font_size}px;
        font-family: 'Inter', 'Segoe UI', 'SF Pro Display', sans-serif;
        transition: all 0.5s ease-in-out;
    }}
    .block-container {{
        max-width: 1350px;
        padding: 1.5rem 2rem 3rem 2rem;
    }}
    h1, h2, h3, h4, h5 {{
        color: {accent};
        text-shadow: 0 0 15px {glow};
        font-weight: 800;
    }}
    div.stButton > button {{
        background: {accent};
        color: white;
        border: none;
        border-radius: {radius}px;
        padding: 0.6rem 1.1rem;
        transition: all 0.25s ease-in-out;
        font-weight: 600;
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 22px {glow};
    }}
    .glass-card {{
        background: {card};
        backdrop-filter: blur(10px);
        border-radius: {radius}px;
        padding: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.15);
        transition: all 0.35s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 28px {glow};
    }}
    [data-testid="stSidebar"] {{
        background: {card};
        border-right: 1px solid {accent}33;
        box-shadow: 4px 0 12px rgba(0,0,0,0.1);
        backdrop-filter: blur(15px);
    }}
    [data-testid="stMetricValue"] {{
        color: {accent} !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 12px {glow};
    }}
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {accent}66, transparent);
        margin: 1rem 0;
    }}
    .stTabs [data-baseweb="tab-list"] button {{
        border-radius: {radius}px;
        color: {text};
        transition: all 0.3s ease;
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background-color: {accent}33;
        color: {accent};
        text-shadow: 0 0 10px {glow};
    }}
    </style>
    """

# =====================================================
# 💾 APPLY THEME
# =====================================================
st.markdown(build_css(palette, font_size, radius, motion), unsafe_allow_html=True)
    
# =====================================================
# 💹 DASHBOARD SECTION — PURE COMPARISON ANALYTICS
# =====================================================

st.markdown(
    f"<h2 style='text-align:center;'>🚗 Parivahan Analytics — {ui_mode} Mode</h2>",
    unsafe_allow_html=True
)
# -- Divider --
st.markdown("<hr>", unsafe_allow_html=True)


# =====================================================
# 🧾 FOOTER
# =====================================================
st.markdown(
    """
    <hr>
    <div style='text-align:center;opacity:0.7;margin-top:2rem;'>
        ✨ Parivahan Analytics • Comparison Dashboard</div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# 🚗 PARIVAHAN ANALYTICS — HEADER + LAYOUT
# =====================================================
from datetime import datetime
import pytz
import streamlit as st

# ================= TIME (IST) =================
ist = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(ist).strftime("%A, %d %B %Y • %I:%M %p")

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
/* Smooth fade + glassmorphism */
.main {
    background: radial-gradient(circle at 20% 20%, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0;}
hr {border: none; border-top: 1px solid rgba(255,255,255,0.2); margin: 1rem 0;}
h1, h2, h3, p {color: white !important;}

/* Card-style elements */
.metric-box {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
.metric-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.footer {
    text-align:center;
    opacity:0.65;
    font-size:13px;
    margin-top:20px;
}
.fade-in {
    animation: fadeIn 1.5s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 🧭 HEADER
# =====================================================
st.markdown(f"""
<div class="fade-in" style='text-align:center;padding:30px;border-radius:25px;
background:rgba(255,255,255,0.05);
box-shadow:0 8px 30px rgba(0,0,0,0.3);
backdrop-filter:blur(10px);
margin-bottom:35px;'>
    <h1 style='font-size:2.5rem;margin-bottom:10px;'>🚗 Parivahan Analytics Dashboard</h1>
    <p style='opacity:0.85;font-size:15px;margin:0;'>Updated: {current_time} (IST)</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 📊 MAIN SECTION
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)

layout = st.container()
with layout:
    st.markdown("""
    <div class="fade-in" style='text-align:center;margin-bottom:1.5rem;'>
        <h2 style='font-size:1.8rem;'>📈 Analytics Overview</h2>
        <p style='opacity:0.75;'>Dynamic KPIs, charts, forecasts, and insights update automatically from live data</p>
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# 🧩 FOOTER
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>🌐 Parivahan Analytics • Hybrid Intelligence Engine</div>",
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
# ⚙️ Dynamic Parameter Builder — Vahan Analytics (MAXED)
# =====================================================
import streamlit as st
import time, random, json, requests
from urllib.parse import urlparse, urljoin
from datetime import datetime

# =====================================================
# 🎨 HEADER — Animated Banner
# =====================================================
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

st.write("")

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
# 1️⃣ CATEGORY DISTRIBUTION — MULTI-YEAR MAXED EDITION 🚀✨
# ===============================================================
with st.container():
    # 🌈 Header
    st.markdown("""
    <div style="padding:14px 22px;border-left:6px solid #6C63FF;
                background:linear-gradient(90deg,#f3f1ff 0%,#ffffff 100%);
                border-radius:16px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <h3 style="margin:0;font-weight:700;color:#3a3a3a;">📊 Category Distribution (Multi-Year View)</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
            Comparative distribution of vehicle registrations by category for selected years and states.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # 🔁 MULTI-YEAR FETCHING
    # =====================================================
    all_dfs = []
    with st.spinner(f"📡 Fetching category distribution from {from_year} → {to_year} ..."):
        for yr in range(from_year, to_year + 1):
            try:
                params = {
                    "stateCd": state_code or "",
                    "rtoCd": rto_code or "0",
                    "year": yr,
                    "vehicleClass": vehicle_classes or "",
                    "vehicleMaker": vehicle_makers or "",
                    "vehicleType": vehicle_type or "",
                    "timePeriod": time_period,
                    "fitnessCheck": fitness_check,
                }
                json_data = fetch_json("vahandashboard/categoriesdonutchart", params, desc=f"Category Distribution {yr}")
                df_temp = to_df(json_data)
                if not df_temp.empty:
                    df_temp["year"] = yr
                    all_dfs.append(df_temp)
            except Exception as e:
                st.error(f"⚠️ Failed to fetch for {yr}: {e}")

    # =====================================================
    # 📊 DATA AGGREGATION
    # =====================================================
    if all_dfs:
        df_cat_all = pd.concat(all_dfs, ignore_index=True)
        df_cat_all = df_cat_all.groupby(["label", "year"], as_index=False)["value"].sum()

        st.success(f"✅ Data Loaded for {len(df_cat_all['year'].unique())} Years")
        years_list = sorted(df_cat_all["year"].unique())

        # =====================================================
        # 🪄 VISUALIZATION
        # =====================================================
        col1, col2 = st.columns(2, gap="large")

        with col1:
            try:
                st.markdown("#### 📈 Year-wise Comparison (Bar)")
                fig = px.bar(
                    df_cat_all,
                    x="label",
                    y="value",
                    color="year",
                    barmode="group",
                    title="Multi-Year Category Comparison",
                    text_auto=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Bar chart failed: {e}")
                st.dataframe(df_cat_all)

        with col2:
            try:
                st.markdown("#### 🍩 Latest Year Donut View")
                latest_year = max(years_list)
                df_latest = df_cat_all[df_cat_all["year"] == latest_year]
                pie_from_df(df_latest, title=f"Category Distribution ({latest_year})", donut=True)
            except Exception as e:
                st.error(f"⚠️ Pie chart failed: {e}")
                st.dataframe(df_cat_all)

        # =====================================================
        # 💎 KPI ZONE — TOTALS & LEADERS
        # =====================================================
        total_all = df_cat_all["value"].sum()
        top_row = df_cat_all.loc[df_cat_all["value"].idxmax()]
        top_cat = top_row["label"]
        top_val = top_row["value"]
        pct = round((top_val / total_all) * 100, 2)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("🏆 Top Category", top_cat)
        with k2:
            st.metric("📊 Share of Total", f"{pct}%")
        with k3:
            st.metric("🚘 Total Registrations", f"{total_all:,}")

        st.markdown(f"""
        <div style="margin-top:10px;padding:14px 16px;
                    background:linear-gradient(90deg,#e7e2ff,#f7f5ff);
                    border:1px solid #d4cfff;border-radius:12px;
                    box-shadow:inset 0 0 8px rgba(108,99,255,0.2);">
            <b>🏅 Insight:</b> <span style="color:#333;">{top_cat}</span> dominates with <b>{pct}%</b> of total across {len(years_list)} years.
        </div>
        """, unsafe_allow_html=True)

        # =====================================================
        # 🤖 AI NARRATIVE (Optional)
        # =====================================================
        if enable_ai:
            st.markdown("### 🤖 AI-Powered Multi-Year Insight")
            with st.expander("🔍 View AI Narrative", expanded=True):
                with st.spinner("🧠 DeepInfra AI is analyzing multi-year category trends..."):
                    sample = df_cat_all.head(10).to_dict(orient="records")
                    system = (
                        "You are a senior automotive data analyst. Analyze year-on-year changes "
                        "in vehicle categories across India, highlight major shifts, and suggest actionable insights."
                    )
                    user = f"Dataset: {json.dumps(sample, default=str)}. Summarize in 3–5 key insights."
                    ai_resp = deepinfra_chat(system, user, max_tokens=350, temperature=0.5)
                    if ai_resp.get("text"):
                        st.markdown(f"""
                        <div style="margin-top:8px;padding:16px 18px;
                                    background:linear-gradient(90deg,#fafaff,#f5f7ff);
                                    border-left:4px solid #6C63FF;border-radius:12px;">
                            <b>AI Summary:</b>
                            <p style="margin-top:6px;font-size:15px;color:#333;">
                                {ai_resp["text"]}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.snow()
                    else:
                        st.info("💤 No AI summary generated. Try again or check DeepInfra key.")
    else:
        st.warning("⚠️ No data returned for selected years.")
        st.info("🔄 Try adjusting filters or refresh the API connection.")

# ===============================================================
# 2️⃣ TOP MAKERS — FULL CUSTOM MAXED EDITION 🏭🚀
# ===============================================================
with st.container():
    # 🌈 Header
    st.markdown("""
    <div style="padding:14px 22px;border-left:6px solid #FF6B6B;
                background:linear-gradient(90deg,#fff5f5 0%,#ffffff 100%);
                border-radius:16px;margin-bottom:20px;
                box-shadow:0 2px 8px rgba(255,107,107,0.15);">
        <h3 style="margin:0;font-weight:700;color:#3a3a3a;">🏭 Top Vehicle Makers — Fully Custom & Multi-Year</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
            Explore manufacturer trends with any filter, year range, and customizable top-N makers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ==============================================
    # ⚙️ User Custom Inputs — Top N Makers
    # ==============================================
    top_n = st.slider("🔢 How many top makers to display?", min_value=3, max_value=25, value=10, step=1)
    st.caption("👆 Adjust this to show more or fewer top manufacturers dynamically.")

    # ==============================================
    # 📡 Multi-Year Data Fetch
    # ==============================================
    all_maker_dfs = []
    with st.spinner(f"🚗 Fetching data for {from_year} → {to_year} across filters..."):
        for yr in range(from_year, to_year + 1):
            try:
                params = {
                    "stateCd": state_code or "",
                    "rtoCd": rto_code or "0",
                    "year": yr,
                    "vehicleClass": vehicle_classes or "",
                    "vehicleMaker": vehicle_makers or "",
                    "vehicleType": vehicle_type or "",
                    "timePeriod": time_period,
                    "fitnessCheck": fitness_check,
                }
                mk_json = fetch_json("vahandashboard/top5Makerchart", params, desc=f"Top Makers {yr}")
                df_temp = parse_makers(mk_json)
                if not df_temp.empty:
                    df_temp["year"] = yr
                    all_maker_dfs.append(df_temp)
            except Exception as e:
                st.error(f"⚠️ Failed fetching for {yr}: {e}")

    # ==============================================
    # 📊 Aggregate & Process
    # ==============================================
    if all_maker_dfs:
        df_mk_all = pd.concat(all_maker_dfs, ignore_index=True)
        df_mk_all.columns = [c.strip().lower() for c in df_mk_all.columns]

        maker_col = next((c for c in ["maker", "makename", "manufacturer", "label", "name"] if c in df_mk_all.columns), None)
        value_col = next((c for c in ["value", "count", "total", "registeredvehiclecount", "y"] if c in df_mk_all.columns), None)

        if not maker_col or not value_col:
            st.warning("⚠️ Could not detect maker/value columns.")
            st.dataframe(df_mk_all)
        else:
            # Aggregate across years and sort
            df_mk_all = (
                df_mk_all.groupby([maker_col, "year"], as_index=False)[value_col].sum().sort_values(by=value_col, ascending=False)
            )

            # Limit top N per year
            df_mk_top = df_mk_all.groupby("year").apply(lambda x: x.nlargest(top_n, value_col)).reset_index(drop=True)
            years_list = sorted(df_mk_all["year"].unique())

            st.success(f"✅ Loaded data for {len(years_list)} years with Top {top_n} Makers per year")

            # ==============================================
            # 🎨 Visuals
            # ==============================================
            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown(f"#### 📊 Top {top_n} Makers — Multi-Year Bar Chart")
                try:
                    fig = px.bar(
                        df_mk_top,
                        x=maker_col,
                        y=value_col,
                        color="year",
                        barmode="group",
                        title=f"Top {top_n} Makers by Year",
                        text_auto=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ Chart failed: {e}")
                    st.dataframe(df_mk_top)

            with col2:
                st.markdown(f"#### 🍩 Latest Year ({max(years_list)}) — Donut View")
                try:
                    df_latest = df_mk_top[df_mk_top["year"] == max(years_list)]
                    pie_from_df(df_latest.rename(columns={maker_col: "label", value_col: "value"}), donut=True)
                except Exception as e:
                    st.error(f"⚠️ Donut chart failed: {e}")
                    st.dataframe(df_mk_top)

            # ==============================================
            # 💎 KPI Zone
            # ==============================================
            total_val = df_mk_all[value_col].sum()
            top_row = df_mk_all.loc[df_mk_all[value_col].idxmax()]
            top_maker = top_row[maker_col]
            top_val = top_row[value_col]
            pct_share = round((top_val / total_val) * 100, 2)

            k1, k2, k3 = st.columns(3)
            k1.metric("🏆 Leading Maker", top_maker)
            k2.metric("📈 Share of Total", f"{pct_share}%")
            k3.metric("🚘 Total Registrations", f"{total_val:,}")

            st.markdown(f"""
            <div style="margin-top:10px;padding:14px 16px;
                        background:linear-gradient(90deg,#ffecec,#fffafa);
                        border:1px solid #ffc9c9;border-radius:12px;">
                <b>🔥 Insight:</b> <span style="color:#333;">{top_maker}</span> leads the market, 
                contributing <b>{pct_share}%</b> across <b>{len(years_list)} years</b> and multiple filters.
            </div>
            """, unsafe_allow_html=True)
            st.balloons()

            # ==============================================
            # 🧠 AI Summary (Optional)
            # ==============================================
            if enable_ai:
                st.markdown("### 🤖 DeepInfra AI — Multi-Year Market Intelligence")
                with st.expander("🔍 View AI Summary", expanded=True):
                    with st.spinner("🧠 Generating manufacturer insights..."):
                        try:
                            sample = df_mk_top.head(10).to_dict(orient="records")
                            system = (
                                "You are an expert automotive data analyst. "
                                "Summarize patterns in multi-year manufacturer data. Identify leaders, challengers, and emerging players."
                            )
                            user = f"Dataset: {json.dumps(sample, default=str)}. Please generate a 4–6 sentence summary of top {top_n} makers and trends."

                            ai_resp = deepinfra_chat(system, user, max_tokens=400, temperature=0.45)
                            if ai_resp.get("text"):
                                st.toast("✅ AI Summary Ready!", icon="🤖")
                                st.markdown(f"""
                                <div style="margin-top:10px;padding:16px 18px;
                                            background:linear-gradient(90deg,#fff9f9,#fffafa);
                                            border-left:4px solid #FF6B6B;border-radius:12px;">
                                    <b>AI Summary:</b>
                                    <p style="margin-top:6px;font-size:15px;color:#333;">
                                        {ai_resp["text"]}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.snow()
                            else:
                                st.info("💤 No AI summary generated.")
                        except Exception as e:
                            st.error(f"AI generation failed: {e}")
    else:
        st.warning("⚠️ No maker data found for your selected filters.")
        st.info("🔄 Try adjusting filters or year range.")


# =============================================================
# 3️⃣ REGISTRATION TRENDS — FULL MAXED CUSTOM FORECAST ⚡
# =============================================================

st.markdown("""
<style>
.trend-card { transition: transform 0.18s ease, box-shadow 0.18s ease; border-radius:12px; }
.trend-card:hover { transform: translateY(-4px); box-shadow: 0 8px 28px rgba(0,0,0,0.12); }
.trend-metric { padding:10px;border-radius:10px;background:linear-gradient(90deg,#ffffff,#f7fbff); }
.small-muted { color:#6b7280;font-size:13px; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# ⚙️ Local Forecast Control Panel — Fully Custom
# =============================================================
st.markdown("<div class='trend-card' style='padding:12px 14px;margin-bottom:10px;'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin:0 0 6px;'>📈 Registration Trends — AI + Forecast (MAXED)</h3>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Real multi-year trend analysis, intelligent forecasting, anomalies, and AI narratives — all filters applied.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    forecast_horizon = st.slider("⏳ Forecast Months", 3, 24, 6, 1)
with c2:
    show_daily = st.checkbox("🗓 Show Daily View", value=True)
with c3:
    forecast_mode = st.selectbox("🔮 Forecast Type", ["Auto (Best)", "Prophet", "Linear", "Growth"], index=0)
with c4:
    refresh_btn = st.button("🔁 Refresh All Trends")

if refresh_btn:
    st.toast("Reloading and recalculating trend data...", icon="🔄")

# =============================================================
# 📡 Fetch Trend Data — All Filters Applied
# =============================================================
all_trends = []
with st.spinner(f"🚦 Loading trend data for {from_year} → {to_year}..."):
    for yr in range(from_year, to_year + 1):
        params = {
            "stateCd": state_code or "",
            "rtoCd": rto_code or "0",
            "year": yr,
            "vehicleClass": vehicle_classes or "",
            "vehicleMaker": vehicle_makers or "",
            "vehicleType": vehicle_type or "",
            "timePeriod": time_period,
            "fitnessCheck": fitness_check,
        }
        try:
            trend_json = fetch_json("vahandashboard/vahanyearwiseregistrationtrend", params, desc=f"Trend {yr}")
            df_temp = normalize_trend(trend_json)
            if not df_temp.empty:
                df_temp["year"] = yr
                all_trends.append(df_temp)
        except Exception as e:
            st.warning(f"⚠️ Failed fetching trend for {yr}: {e}")

if not all_trends:
    st.warning("🚫 No trend data available for your selected filters.")
    st.stop()

df_trend_all = pd.concat(all_trends, ignore_index=True)
df_trend_all["date"] = pd.to_datetime(df_trend_all["date"])
df_trend_all = df_trend_all.sort_values("date")

# =============================================================
# 📈 Forecasting Logic — Auto Switch (Prophet / Linear / Growth)
# =============================================================
def generate_forecast(df, periods=6, mode="Auto (Best)"):
    if df.empty: return df
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["forecast"] = False
    try:
        if mode in ["Auto (Best)", "Prophet"]:
            from prophet import Prophet
            tmp = df.rename(columns={"date": "ds", "value": "y"})
            m = Prophet(daily_seasonality=False, yearly_seasonality=True)
            m.fit(tmp)
            future = m.make_future_dataframe(periods=periods, freq="M")
            fc = m.predict(future)[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "value"})
            fc["forecast"] = fc["date"] > df["date"].max()
            return fc
    except Exception:
        pass

    try:
        if mode in ["Auto (Best)", "Linear"]:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df["value"].values
            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(df), len(df) + periods).reshape(-1, 1)
            y_pred = model.predict(future_X)
            future_dates = pd.date_range(df["date"].max() + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            fc_df = pd.DataFrame({"date": future_dates, "value": y_pred, "forecast": True})
            return pd.concat([df, fc_df], ignore_index=True)
    except Exception:
        pass

    # Simple growth model fallback
    try:
        if mode in ["Auto (Best)", "Growth"]:
            avg_growth = df["value"].pct_change().mean()
            last_val = df["value"].iloc[-1]
            future_dates = pd.date_range(df["date"].max() + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            values = [last_val * (1 + (avg_growth if not np.isnan(avg_growth) else 0)) ** (i + 1) for i in range(periods)]
            fc_df = pd.DataFrame({"date": future_dates, "value": values, "forecast": True})
            return pd.concat([df, fc_df], ignore_index=True)
    except Exception:
        pass

    return df.assign(forecast=False)

fc_df = generate_forecast(df_trend_all, periods=forecast_horizon, mode=forecast_mode)

# =============================================================
# 💎 KPIs & Overview Metrics
# =============================================================
st.markdown("### 📊 Key Performance Metrics")
total_reg = int(df_trend_all["value"].sum())
daily_avg = total_reg / max(1, (df_trend_all["date"].max() - df_trend_all["date"].min()).days)
yoy_df = compute_yoy(df_trend_all)
qoq_df = compute_qoq(df_trend_all)
latest_yoy = yoy_df["YoY%"].dropna().iloc[-1] if "YoY%" in yoy_df.columns and not yoy_df.empty else None
latest_qoq = qoq_df["QoQ%"].dropna().iloc[-1] if "QoQ%" in qoq_df.columns and not qoq_df.empty else None

k1, k2, k3, k4 = st.columns(4)
k1.metric("🚘 Total Registrations", f"{total_reg:,}")
k2.metric("📅 Avg per Day", f"{daily_avg:,.0f}")
k3.metric("📈 Latest YoY%", f"{latest_yoy:.2f}%" if latest_yoy else "N/A")
k4.metric("📊 Latest QoQ%", f"{latest_qoq:.2f}%" if latest_qoq else "N/A")

# =============================================================
# 📊 Actual + Forecast Chart
# =============================================================
import plotly.express as px
st.markdown("### 📉 Actual vs Forecast Trends")
try:
    fc_df["Type"] = fc_df["forecast"].apply(lambda x: "Forecast" if x else "Actual")
    fig = px.line(fc_df, x="date", y="value", color="Type", markers=True,
                  color_discrete_map={"Actual": "#007BFF", "Forecast": "#FF9800"},
                  title=f"Vehicle Registrations (Actual vs {forecast_mode} Forecast)")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"⚠️ Chart rendering failed: {e}")
    st.line_chart(df_trend_all.set_index("date")["value"])

# =============================================================
# 📆 Daily View (Interpolated)
# =============================================================
if show_daily:
    try:
        st.markdown("### 🗓 Daily Registration Estimate")
        df_daily = df_trend_all.set_index("date").resample("D").interpolate().reset_index()
        df_daily["growth"] = df_daily["value"].pct_change() * 100
        last_day = df_daily.iloc[-1]
        prev_day = df_daily.iloc[-2]
        growth = (last_day["value"] - prev_day["value"]) / (prev_day["value"] or 1) * 100
        col_a, col_b = st.columns([3,1])
        with col_a:
            figd = px.area(df_daily.tail(90), x="date", y="value", title="Last 90 Days — Daily Registration Trend")
            st.plotly_chart(figd, use_container_width=True)
        with col_b:
            st.metric("🕒 Latest Day", f"{last_day['value']:.0f}", f"{growth:.2f}% vs prev")
    except Exception as e:
        st.warning(f"Daily interpolation failed: {e}")

# =============================================================
# 🧠 AI Narrative — DeepInfra Summary
# =============================================================
if enable_ai:
    with st.expander("🤖 AI Narrative — Trend & Forecast Summary", expanded=True):
        try:
            sample = df_trend_all.tail(12).to_dict(orient="records")
            system = (
                "You are an advanced analytics assistant. "
                "Analyze the trend, forecast, and YoY/QoQ results for vehicle registrations. "
                "Provide concise insights and 2 recommendations."
            )
            user = f"Dataset: {json.dumps(sample, default=str)}, Latest YoY: {latest_yoy}, QoQ: {latest_qoq}, Forecast horizon: {forecast_horizon}."
            ai_resp = deepinfra_chat(system, user, max_tokens=400, temperature=0.3)
            if ai_resp.get("text"):
                st.success("AI Summary Generated ✅")
                st.markdown(f"""
                <div style='padding:16px;border-radius:10px;background:#f8faff;
                            border-left:5px solid #007BFF;margin-top:10px;'>
                    {ai_resp['text']}
                </div>
                """, unsafe_allow_html=True)
                st.snow()
            else:
                st.info("AI did not return any summary.")
        except Exception as e:
            st.warning(f"AI summary failed: {e}")

# =============================================================
# 🎉 End of Trend Section
# =============================================================
st.markdown("---")
st.markdown(f"📅 Data Range: {df_trend_all['date'].min().date()} → {df_trend_all['date'].max().date()} | ⏳ Forecast: {forecast_horizon} months")


# ================================================================
# 🌈 4️⃣ Duration-wise Growth + 5️⃣ Top 5 Revenue States — MAXED ⚡
# ================================================================

import streamlit as st
import pandas as pd
import json
from datetime import datetime

# ================================================================
# ✨ Animated Header
# ================================================================
st.markdown("""
<style>
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0px #28a745; }
    50% { box-shadow: 0 0 10px #28a745; }
    100% { box-shadow: 0 0 0px #28a745; }
}
.maxed-header {
    background: linear-gradient(90deg, #eaffea, #ffffff);
    border-left: 6px solid #28a745;
    padding: 14px 20px;
    border-radius: 14px;
    margin-bottom: 20px;
    animation: pulseGlow 3s infinite;
}
</style>

<div class="maxed-header">
    <h2 style="margin:0;">📊 Duration-wise Growth & Revenue Insights (MAXED)</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        All filters applied — monthly, quarterly, and yearly growth with smart AI narratives, full customization, and top revenue states.
    </p>
</div>
""", unsafe_allow_html=True)

# ================================================================
# ⚙️ Local Filter Controls (Full Custom)
# ================================================================
st.markdown("### 🧩 Custom Growth Controls")

col1, col2, col3, col4 = st.columns(4)
with col1:
    top_n = st.number_input("🏆 Top N for Revenue", min_value=3, max_value=20, value=5)
with col2:
    enable_compare = st.checkbox("📅 Compare Years Side-by-Side", value=True)
with col3:
    enable_ai_growth = st.checkbox("🤖 AI for Growth", value=enable_ai)
with col4:
    enable_ai_revenue = st.checkbox("💡 AI for Revenue", value=enable_ai)

st.divider()

# ================================================================
# 🧮 Duration-wise Growth Section (All Filters Applied)
# ================================================================
def fetch_duration_growth(calendar_type, label, color, emoji):
    """Fetch and visualize monthly/quarterly/yearly growth — all filters applied."""
    with st.spinner(f"📦 Loading {label} growth data (with filters)..."):
        params = {
            **params_common,
            "calendarType": calendar_type,
            "stateCd": state_code or "",
            "rtoCd": rto_code or "0",
            "vehicleClass": vehicle_classes or "",
            "vehicleMaker": vehicle_makers or "",
            "vehicleType": vehicle_type or "",
        }
        json_data = fetch_json("vahandashboard/durationWiseRegistrationTable", params, desc=f"{label} growth")
        df = parse_duration_table(json_data)

        if df.empty:
            st.warning(f"No {label.lower()} data available for these filters.")
            return pd.DataFrame()

        # Sub-header
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
                bar_from_df(df, title=f"{label} Growth (Bar Chart)")
            except Exception:
                st.dataframe(df)
        with col2:
            try:
                pie_from_df(df, title=f"{label} Growth (Pie Chart)", donut=True)
            except Exception:
                st.dataframe(df)

        # KPI Summary
        try:
            max_label = df.loc[df["value"].idxmax(), "label"]
            max_val = df["value"].max()
            avg_val = df["value"].mean()
            total_val = df["value"].sum()

            st.markdown(f"""
            <div style="margin-top:10px;padding:12px 16px;
                        background:rgba(255,255,255,0.9);
                        border-left:5px solid {color};
                        border-radius:12px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.05);">
                <b>🏆 Peak Period:</b> {max_label}<br>
                <b>📈 Registrations:</b> {max_val:,.0f}<br>
                <b>📊 Average:</b> {avg_val:,.0f}<br>
                <b>🧮 Total:</b> {total_val:,.0f}
            </div>
            """, unsafe_allow_html=True)

            if max_val > avg_val * 1.5:
                st.balloons()

        except Exception as e:
            st.warning(f"KPI error for {label}: {e}")

        # AI Growth Summary
        if enable_ai_growth:
            with st.expander(f"🤖 AI Summary — {label} Growth", expanded=False):
                with st.spinner(f"AI generating {label} growth summary..."):
                    system = (
                        f"You are an analytics assistant summarizing {label.lower()} vehicle registration growth trends. "
                        "Highlight highs, dips, patterns, and give one recommendation."
                    )
                    sample = df.head(10).to_dict(orient="records")
                    user = f"Dataset: {json.dumps(sample, default=str)}"
                    ai_resp = deepinfra_chat(system, user, max_tokens=280)
                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown(f"""
                        <div style="padding:12px 14px;margin-top:6px;
                                    background:linear-gradient(90deg,#ffffff,#f8fff8);
                                    border-left:4px solid {color};
                                    border-radius:10px;">
                            {ai_resp["text"]}
                        </div>
                        """, unsafe_allow_html=True)

        return df


# Run for all duration types
df_monthly   = fetch_duration_growth(3, "Monthly",  "#007bff", "📅")
df_quarterly = fetch_duration_growth(2, "Quarterly", "#6f42c1", "🧭")
df_yearly    = fetch_duration_growth(1, "Yearly",   "#28a745", "📆")

st.divider()

# ================================================================
# 💰 Top N Revenue States (Full Custom)
# ================================================================
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
    <h2 style="margin:0;">💰 Top Revenue States</h2>
    <p style="margin:4px 0 0;color:#555;font-size:15px;">
        View leading states by revenue — fully filtered, dynamic, and AI-analyzed.
    </p>
</div>
""", unsafe_allow_html=True)

# Fetch with filters
with st.spinner(f"Fetching Top {top_n} Revenue States..."):
    params_rev = {
        **params_common,
        "stateCd": state_code or "",
        "rtoCd": rto_code or "0",
        "vehicleClass": vehicle_classes or "",
        "vehicleMaker": vehicle_makers or "",
        "vehicleType": vehicle_type or "",
        "limit": top_n
    }
    top_rev_json = fetch_json("vahandashboard/top5chartRevenueFee", params_rev, desc="Top Revenue States")
    df_top_rev = parse_top5_revenue(top_rev_json if top_rev_json else {})

if not df_top_rev.empty:
    col1, col2 = st.columns(2)
    with col1:
        try:
            bar_from_df(df_top_rev, title=f"Top {top_n} Revenue States (Bar)")
        except Exception:
            st.dataframe(df_top_rev)
    with col2:
        try:
            pie_from_df(df_top_rev, title=f"Top {top_n} Revenue States (Pie)", donut=True)
        except Exception:
            st.dataframe(df_top_rev)

    try:
        top_state = df_top_rev.loc[df_top_rev["value"].idxmax(), "label"]
        top_value = df_top_rev["value"].max()
        total_value = df_top_rev["value"].sum()

        st.markdown(f"""
        <div style="margin-top:10px;padding:14px 18px;
                    background:linear-gradient(90deg,#fffef5,#ffffff);
                    border-left:5px solid #ffc107;
                    border-radius:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <b>🏅 Top State:</b> {top_state} — ₹{top_value:,.0f}<br>
            <b>💵 Total ({top_n} states):</b> ₹{total_value:,.0f}
        </div>
        """, unsafe_allow_html=True)

        st.snow()
    except Exception as e:
        st.warning(f"Revenue KPI error: {e}")

    if enable_ai_revenue:
        with st.expander("🤖 AI Summary — Revenue Insights", expanded=True):
            with st.spinner("Generating AI summary for revenue states..."):
                system = (
                    "You are an economic analyst summarizing vehicle revenue performance by Indian states. "
                    "Highlight top performers, revenue imbalance, and a key recommendation for policy improvement."
                )
                sample = df_top_rev.to_dict(orient="records")
                user = f"Dataset: {json.dumps(sample, default=str)}"
                ai_resp = deepinfra_chat(system, user, max_tokens=260)
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
    st.warning("⚠️ No revenue data available for current filters.")

st.divider()
st.info(f"✅ Filters applied: Years {from_year}–{to_year}, State: {state_code or 'All'}, Vehicle Class: {vehicle_classes or 'All'}, Maker: {vehicle_makers or 'All'}, Type: {vehicle_type or 'All'}")

# ================================================================
# 🌟 6️⃣ MAXED Revenue Trend + Forecast + Anomaly + Clustering — UI
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
@keyframes pulse {
  0% {box-shadow: 0 0 0px #ff5722;}
  50% {box-shadow: 0 0 12px #ff5722;}
  100% {box-shadow: 0 0 0px #ff5722;}
}
.sec-box {
  background: linear-gradient(90deg,#fff7f3,#ffffff);
  border-left: 6px solid #ff5722;
  padding: 18px 24px;
  border-radius: 14px;
  margin: 25px 0 20px 0;
  animation: pulse 4s infinite;
}
.metric-card {
  background: #fff;
  border-radius: 12px;
  padding: 14px;
  box-shadow: 0 3px 12px rgba(0,0,0,0.06);
  transition: all 0.25s ease;
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 16px rgba(255,87,34,0.25);
}
.ai-box {
  background: linear-gradient(90deg,#ffffff,#fff9f6);
  border-left: 4px solid #ff5722;
  border-radius: 10px;
  padding: 12px 14px;
  margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ====================================================
# 🧭 FILTER PANEL (All Maxed Filters)
# ====================================================
with st.expander("🎛️ Customize View — All Filters", expanded=True):
    years = st.multiselect("Select Year(s):", options=["2022", "2023", "2024", "2025"], default=["2024", "2025"])
    states = st.multiselect("Select States:", ["All", "Maharashtra", "Delhi", "Gujarat", "Tamil Nadu", "Karnataka"], default=["All"])
    categories = st.multiselect("Select Vehicle Categories:", ["All", "Car", "Bike", "Truck", "Bus", "EV"], default=["All"])
    metric_type = st.selectbox("Select Metric Type:", ["Revenue", "Registrations", "Fees"], index=0)
    show_forecast = st.toggle("🔮 Enable Forecasting", True)
    show_anomaly = st.toggle("🚨 Enable Anomaly Detection", True)
    show_clustering = st.toggle("🧩 Enable Clustering", True)
    enable_ai = st.toggle("🤖 Enable AI Insights", True)
    forecast_periods = st.slider("Forecast Months", 3, 24, 6)


# ====================================================
# 📊 Section Header
# ====================================================
st.markdown(f"""
<div class='sec-box'>
    <h2 style="margin:0;">💹 {metric_type} Trend & Advanced Analytics</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        Smart forecasting, anomaly detection, and AI-powered clustering insights across <b>{', '.join(years)}</b> for selected regions.
    </p>
</div>
""", unsafe_allow_html=True)


# ====================================================
# 📈 Fetch Data (Multi-Year & Multi-State)
# ====================================================
dfs = []
for year in years:
    for state in states:
        desc = f"{metric_type} Trend — {state} ({year})"
        with st.spinner(f"Fetching {desc}..."):
            api_endpoint = "vahandashboard/revenueFeeLineChart"
            data_json = fetch_json(api_endpoint, desc=desc)
            df_temp = parse_revenue_trend(data_json if data_json else {})
            if not df_temp.empty:
                df_temp["year"] = year
                df_temp["state"] = state
                dfs.append(df_temp)

if dfs:
    df_rev_trend = pd.concat(dfs, ignore_index=True)
else:
    df_rev_trend = pd.DataFrame()

# ====================================================
# 📊 Trend Visualization
# ====================================================
if df_rev_trend.empty:
    st.warning("⚠️ No data returned for selected filters.")
else:
    st.subheader("📊 Multi-Year Trend Comparison")
    try:
        chart = (
            alt.Chart(df_rev_trend)
            .mark_line(point=True, interpolate="monotone")
            .encode(
                x=alt.X("period:O", title="Period"),
                y=alt.Y("value:Q", title=f"{metric_type} (₹)"),
                color=alt.Color("year:N", legend=alt.Legend(title="Year")),
                tooltip=["state", "year", "period", "value"]
            )
            .properties(height=400, title=f"{metric_type} Trend Comparison")
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")
        st.dataframe(df_rev_trend)


# ====================================================
# 💎 KPIs
# ====================================================
try:
    total_val = float(df_rev_trend["value"].sum())
    avg_val = float(df_rev_trend["value"].mean())
    latest_val = float(df_rev_trend["value"].iloc[-1])
    prev_val = float(df_rev_trend["value"].iloc[-2]) if len(df_rev_trend) > 1 else latest_val
    growth_pct = ((latest_val - prev_val) / prev_val) * 100 if prev_val else 0.0
except Exception:
    total_val = avg_val = latest_val = growth_pct = None

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='metric-card'><h4>💰 Total {metric_type}</h4><b>₹{total_val:,.0f}</b></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='metric-card'><h4>📈 Latest Value</h4><b>₹{latest_val:,.0f}</b></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='metric-card'><h4>📊 Average</h4><b>₹{avg_val:,.0f}</b></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='metric-card'><h4>📅 Growth %</h4><b style='color:{'green' if growth_pct>=0 else 'red'}'>{growth_pct:.2f}%</b></div>", unsafe_allow_html=True)

if growth_pct > 5: st.balloons()
elif growth_pct < -5: st.snow()


# ====================================================
# 🔮 Forecasting
# ====================================================
if show_forecast:
    try:
        st.markdown("### 🔮 Forecasting — Future Projection")
        df_rev_trend['date'] = pd.to_datetime(df_rev_trend['period'], errors='coerce')
        df_rev_trend = df_rev_trend.dropna(subset=['date'])
        forecast_df = linear_forecast(df_rev_trend, months=forecast_periods)
        if not forecast_df.empty:
            st.line_chart(pd.concat([
                df_rev_trend.set_index('date')['value'],
                forecast_df.set_index('date')['value']
            ]))
            if enable_ai:
                with st.spinner("🤖 AI Forecast Summary..."):
                    system = "You are a forecasting analyst summarizing financial trends."
                    user = f"Forecasted values: {forecast_df.head(6).to_dict(orient='records')}. Summarize key future directions."
                    ai = deepinfra_chat(system, user, max_tokens=200)
                    if ai.get("text"):
                        st.markdown(f"<div class='ai-box'>{ai['text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Forecast failed: {e}")


# ====================================================
# 🚨 Anomaly Detection
# ====================================================
if show_anomaly:
    try:
        st.markdown("### 🚨 Anomaly Detection")
        from sklearn.ensemble import IsolationForest
        import numpy as np

        contamination = st.slider("Outlier Fraction", 0.01, 0.2, 0.05)
        model = IsolationForest(contamination=contamination, random_state=42)
        df_rev_trend["value"] = pd.to_numeric(df_rev_trend["value"], errors="coerce").fillna(0)
        model.fit(df_rev_trend[["value"]])
        df_rev_trend["anomaly"] = model.predict(df_rev_trend[["value"]])
        anomalies = df_rev_trend[df_rev_trend["anomaly"] == -1]

        st.metric("🚨 Anomalies Detected", f"{len(anomalies)}")
        base = alt.Chart(df_rev_trend).encode(x="period:O")
        line = base.mark_line().encode(y="value:Q")
        points = base.mark_circle(size=80).encode(
            y="value:Q",
            color=alt.condition(alt.datum.anomaly == -1, alt.value("red"), alt.value("black")),
            tooltip=["period", "value"]
        )
        st.altair_chart((line + points).properties(height=350), use_container_width=True)

        if enable_ai and len(anomalies) > 0:
            with st.spinner("🤖 AI analyzing anomalies..."):
                system = "You are an anomaly analyst detecting financial irregularities."
                user = f"Detected anomalies: {json.dumps(anomalies.head(10).to_dict(orient='records'), default=str)}. Give 3 insights + 2 actions."
                ai = deepinfra_chat(system, user, max_tokens=250)
                if ai.get("text"):
                    st.markdown(f"<div class='ai-box'>{ai['text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")


# ====================================================
# 🧩 Clustering
# ====================================================
if show_clustering:
    try:
        st.markdown("### 🧩 Clustering & Correlation Insights")
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
        import numpy as np
        import plotly.express as px

        df_cl = df_rev_trend.copy()
        df_cl["value"] = pd.to_numeric(df_cl["value"], errors="coerce").fillna(0)

        num_cols = df_cl.select_dtypes(include=[np.number]).columns.tolist()
        X = df_cl[num_cols].astype(float)
        Xs = StandardScaler().fit_transform(X)

        n_clusters = st.slider("Number of Clusters", 2, min(8, len(Xs)), 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(Xs)
        df_cl["cluster"] = labels
        sc = silhouette_score(Xs, labels) if len(Xs) > n_clusters else 0

        st.metric("Silhouette Score", f"{sc:.3f}")
        pca = PCA(n_components=2)
        proj = pca.fit_transform(Xs)
        scatter_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "cluster": labels})
        chart = (
            alt.Chart(scatter_df)
            .mark_circle(size=80)
            .encode(x="x", y="y", color="cluster:N", tooltip=["cluster"])
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)

        if enable_ai:
            with st.spinner("🤖 AI Cluster Insights..."):
                cluster_summary = df_cl.groupby("cluster")["value"].mean().to_dict()
                system = "You are a financial cluster analyst."
                user = f"Cluster summaries: {json.dumps(cluster_summary, default=str)}. Provide 5 observations & 2 recommendations."
                ai = deepinfra_chat(system, user, max_tokens=350)
                if ai.get("text"):
                    st.markdown(f"<div class='ai-box'>{ai['text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# ============================================================
# 💾 MAXED SMART EXCEL EXPORT — All Comparisons, All Years
# ============================================================

st.markdown("""
<div style="padding:18px 20px;border-left:5px solid #007bff;
            background:linear-gradient(90deg,#f0f8ff,#ffffff);
            border-radius:12px;margin-top:25px;margin-bottom:15px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <h2 style="margin:0;">💾 MAXED SMART EXCEL EXPORT</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        Export <b>all KPIs, comparisons, anomalies, forecasts, and AI insights</b> — with year, month & daily breakdowns — into a
        <b>styled, auto-charted Excel workbook</b>.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📊 Generate & Download Full Smart Analytics Workbook", expanded=True):
    st.markdown("""
    <div style="background:linear-gradient(90deg,#e8f0fe,#ffffff);
                border-left:5px solid #007bff;padding:10px 18px;
                border-radius:10px;margin-bottom:10px;">
        <b>💡 Tip:</b> Fetch all datasets before exporting for a complete analytics workbook.
    </div>
    """, unsafe_allow_html=True)

    # 🧱 Load All Known DataFrames (auto-detect from locals)
    dfs = {k: v for k, v in locals().items() if isinstance(v, pd.DataFrame)}
    st.info(f"🧾 Detected {len(dfs)} datasets for export: {list(dfs.keys())}")

    # 🧮 Derived & Comparison Datasets
    try:
        if 'df_trend' in dfs and not dfs['df_trend'].empty:
            df_trend = dfs['df_trend']
            df_trend["Year"] = pd.to_datetime(df_trend["x"], errors='coerce').dt.year
            df_trend["Month"] = pd.to_datetime(df_trend["x"], errors='coerce').dt.strftime('%b')
            df_trend["Day"] = pd.to_datetime(df_trend["x"], errors='coerce').dt.day

            # Yearly Comparison
            yearly_comp = df_trend.groupby("Year")["value"].sum().pct_change().reset_index()
            yearly_comp.columns = ["Year", "YoY Growth"]
            dfs["Yearly Comparison"] = yearly_comp

            # Monthly Comparison
            monthly_comp = df_trend.groupby("Month")["value"].sum().pct_change().reset_index()
            monthly_comp.columns = ["Month", "MoM Growth"]
            dfs["Monthly Comparison"] = monthly_comp

            # Daily Comparison (if data dense)
            if df_trend["Day"].nunique() > 5:
                daily_comp = df_trend.groupby("Day")["value"].sum().pct_change().reset_index()
                daily_comp.columns = ["Day", "DoD Growth"]
                dfs["Daily Comparison"] = daily_comp

            # Category vs Maker
            if 'df_cat' in dfs and 'df_mk' in dfs:
                try:
                    df_cat = dfs['df_cat']
                    df_mk = dfs['df_mk']
                    cross = pd.DataFrame({
                        "Category Total": [df_cat["value"].sum()],
                        "Top Maker Total": [df_mk["value"].sum()],
                        "Ratio": [round(df_mk["value"].sum() / df_cat["value"].sum(), 3)]
                    })
                    dfs["Category vs Maker Comparison"] = cross
                except Exception as e:
                    st.warning(f"Category vs Maker comparison skipped: {e}")
        else:
            st.info("No trend data found — skipping growth comparisons.")
    except Exception as e:
        st.warning(f"Comparison step failed: {e}")

    # 🔮 Forecast + Anomalies
    try:
        if 'df_trend' in dfs and not dfs['df_trend'].empty:
            df_forecast = dfs['df_trend'].copy()
            df_forecast["Forecast"] = df_forecast["value"].rolling(4, min_periods=1).mean()
            df_forecast["Anomaly"] = (abs(df_forecast["value"] - df_forecast["Forecast"]) >
                                      df_forecast["Forecast"] * 0.2)
            dfs["Forecast & Anomaly Detection"] = df_forecast
            st.success("✅ Forecast & anomaly detection complete.")
        else:
            st.info("ℹ️ No trend data available for forecast.")
    except Exception as e:
        st.warning(f"⚠️ Forecast failed: {e}")

    # 🧠 AI Summary (DeepInfra)
    summaries = {}
    if 'enable_ai' in locals() and enable_ai:
        st.info("🤖 Generating AI summaries for all datasets...")
        progress = st.progress(0)
        for i, (name, df) in enumerate(dfs.items()):
            if not df.empty:
                try:
                    system = f"You are a senior analytics expert summarizing '{name}' dataset trends for Parivahan data."
                    user = f"Dataset sample: {df.head(8).to_dict(orient='records')}.\nGenerate 3 crisp insights."
                    ai_resp = deepinfra_chat(system, user, max_tokens=200)
                    summaries[name] = ai_resp.get("text", "No AI insight.")
                except Exception as e:
                    summaries[name] = f"AI summary failed: {e}"
            progress.progress((i + 1) / len(dfs))
        dfs["AI Insights"] = pd.DataFrame(summaries.items(), columns=["Dataset", "AI Summary"])
        progress.empty()

    # 💾 Compile Styled Excel Workbook
    with st.spinner("📦 Compiling MAXED analytics workbook..."):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            for name, df in dfs.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(writer, sheet_name=name[:31], index=False)
        output.seek(0)

        from openpyxl import load_workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter
        from openpyxl.chart import LineChart, BarChart, Reference

        wb = load_workbook(output)
        border = Border(left=Side(style="thin"), right=Side(style="thin"),
                        top=Side(style="thin"), bottom=Side(style="thin"))

        for sheet in wb.sheetnames:
            ws = wb[sheet]
            # Header Style
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color="007bff", end_color="007bff", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = border
            # Cell Formatting
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.border = border
            # Column Auto Width
            for col in ws.columns:
                max_len = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[get_column_letter(col[0].column)].width = max_len + 3
            # Add Auto Chart
            if ws.max_row > 2 and ws.max_column >= 2:
                try:
                    val_ref = Reference(ws, min_col=2, min_row=1, max_row=ws.max_row)
                    cat_ref = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
                    chart = LineChart()
                    chart.title = f"{sheet} — Trend"
                    chart.add_data(val_ref, titles_from_data=True)
                    chart.set_categories(cat_ref)
                    chart.height = 8
                    chart.width = 16
                    ws.add_chart(chart, "H5")
                except Exception:
                    pass

        styled = io.BytesIO()
        wb.save(styled)
        styled.seek(0)

    ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    st.download_button(
        label="⬇️ Download MAXED Analytics Excel Workbook",
        data=styled.getvalue(),
        file_name=f"Vahan_MAXED_Analytics_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.success("✅ MAXED Excel report ready — includes all comparisons, AI, and visuals.")
    st.toast("Full Analytics Workbook is ready! 🎯")
    st.balloons()

# ============================================================
# 🧩 MAXED RAW JSON ANALYZER PRO — All Endpoints, All Modes
# ============================================================
with st.expander("🛠️ RAW JSON ANALYZER (Developer / Debug Mode — All MAXED)", expanded=False):
    st.caption("""
    Inspect and export raw API responses from **all Vahan endpoints** dynamically.
    Includes search, pretty view, table view, snapshot export, and AI summary.
    ⚠️ Use only in trusted environments — data may contain sensitive payloads.
    """)

    # ------------------------------------------------------------
    # 🔍 CONFIG / OPTIONS
    # ------------------------------------------------------------
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    with ctrl1:
        show_pretty = st.checkbox("🧾 Pretty JSON", value=False)
    with ctrl2:
        show_table = st.checkbox("📋 Table View (flattened)", value=False)
    with ctrl3:
        enable_ai_summary = st.checkbox("🤖 AI Summary", value=False)
    with ctrl4:
        snapshot_name = st.text_input("📁 Snapshot name", 
            value=f"vahan_snapshot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}")

    st.markdown("---")

    # ------------------------------------------------------------
    # 🧠 COLLECT ALL JSON-LIKE OBJECTS AUTOMATICALLY
    # ------------------------------------------------------------
    json_candidates = {
        k: v for k, v in locals().items()
        if ("json" in k.lower()) and isinstance(v, (dict, list))
    }

    if not json_candidates:
        st.warning("⚠️ No JSON payloads detected. Run API calls first.")
    else:
        st.success(f"✅ Found {len(json_candidates)} API JSON payloads.")
        st.caption(f"Endpoints detected: {list(json_candidates.keys())}")

    # ------------------------------------------------------------
    # 🔍 SEARCH / FILTER JSON KEYS
    # ------------------------------------------------------------
    search_query = st.text_input("🔎 Search within JSON keys", placeholder="e.g., vehicle, state, category...")

    # ------------------------------------------------------------
    # 🔁 RENDER EACH JSON PAYLOAD (dynamic)
    # ------------------------------------------------------------
    for key, data in json_candidates.items():
        st.markdown(f"### 📦 {key.upper()}")

        # Basic meta info
        meta = {"type": type(data).__name__, "size": len(data) if hasattr(data, "__len__") else "?"}
        st.caption(f"Meta: {meta}")

        # Optional search filter
        if search_query:
            try:
                jtxt = json.dumps(data, indent=2, default=str)
                if search_query.lower() not in jtxt.lower():
                    st.write(f"🔍 No match for '{search_query}' in {key}")
                    continue
            except Exception:
                pass

        # JSON or Pretty Code View
        if show_pretty:
            try:
                st.code(json.dumps(data, indent=2, default=str), language="json")
            except Exception:
                st.write(data)
        else:
            st.json(data)

        # Optional Flattened Table View
        if show_table:
            try:
                df_flat = pd.json_normalize(data)
                st.dataframe(df_flat, use_container_width=True)
            except Exception as e:
                st.warning(f"Table view unavailable: {e}")

        # Download Controls
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            try:
                st.download_button(
                    label=f"⬇️ {key}.json",
                    data=json.dumps(data, indent=2, default=str).encode(),
                    file_name=f"{key}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.warning(f"Download failed: {e}")
        with c2:
            if show_table:
                try:
                    df_flat.to_excel(f"{key}.xlsx", index=False)
                    st.download_button(
                        label=f"⬇️ {key}.xlsx",
                        data=open(f"{key}.xlsx", "rb").read(),
                        file_name=f"{key}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception:
                    pass
        with c3:
            if enable_ai_summary and 'deepinfra_chat' in locals():
                with st.spinner("🤖 Summarizing JSON via AI..."):
                    try:
                        system = f"You are an automotive analytics assistant. Summarize this JSON response named '{key}'."
                        user = f"Payload: {json.dumps(data)[:5000]} (truncated). Give a short summary in 3–4 lines."
                        resp = deepinfra_chat(system, user, max_tokens=180)
                        if resp and "text" in resp:
                            st.markdown(f"**AI Summary:** {resp['text']}")
                    except Exception as e:
                        st.warning(f"AI summary failed: {e}")

        st.markdown("---")

    # ------------------------------------------------------------
    # 💾 GLOBAL SNAPSHOT DOWNLOAD — ALL JSONS COMBINED
    # ------------------------------------------------------------
    st.subheader("📦 Download Combined Snapshot")
    if st.button("💾 Create & Download Unified Snapshot"):
        try:
            combined = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "total_jsons": len(json_candidates),
                "payloads": json_candidates
            }
            payload = json.dumps(combined, indent=2, default=str).encode("utf-8")
            st.download_button(
                label="⬇️ Download Full Snapshot (.json)",
                data=payload,
                file_name=f"{snapshot_name}.json",
                mime="application/json"
            )
            st.success("✅ Combined snapshot ready.")
            st.toast("Unified JSON snapshot created!")
        except Exception as e:
            st.error(f"Snapshot generation failed: {e}")

    # ------------------------------------------------------------
    # 🧩 ZIP MODE — EXPORT ALL JSONS AS INDIVIDUAL FILES
    # ------------------------------------------------------------
    st.subheader("🗜️ ZIP Export (All JSONs Separately)")
    if st.button("📦 Create ZIP Archive"):
        try:
            import zipfile, tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmpzip:
                with zipfile.ZipFile(tmpzip.name, "w") as zf:
                    for k, d in json_candidates.items():
                        zf.writestr(f"{k}.json", json.dumps(d, indent=2, default=str))
                zf.close()
                with open(tmpzip.name, "rb") as f:
                    st.download_button(
                        label="⬇️ Download All JSONs (ZIP)",
                        data=f.read(),
                        file_name=f"{snapshot_name}_bundle.zip",
                        mime="application/zip"
                    )
            st.success("✅ ZIP bundle ready for download.")
        except Exception as e:
            st.error(f"ZIP creation failed: {e}")

    # ------------------------------------------------------------
    # ⚙️ Optional Diagnostics Save (local mode)
    # ------------------------------------------------------------
    try:
        if st.checkbox("📝 Save lightweight diagnostics (local mode)"):
            diag = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "json_count": len(json_candidates),
                "names": list(json_candidates.keys())
            }
            path = os.path.join(os.getcwd(), f"vahan_diag_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2)
            st.success(f"Diagnostics file saved at {path}")
    except Exception:
        pass

    st.info("🔒 All-MAXED Raw JSON Analyzer ready — safely inspect, compare, summarize, and export every payload.")

# ============================================================
# ⚡ FOOTER KPIs + EXECUTIVE SUMMARY — MAXED VERSION 🚀
# ============================================================

import json, time, random
import streamlit as st
import pandas as pd

st.markdown("---")
st.subheader("📊 Dashboard Summary & Insights (All Years & Filters)")

# ============================================================
# 🧮 SMART METRIC COMPUTATION — ALL YEARS
# ============================================================
try:
    total_reg = int(df_trend["value"].sum()) if not df_trend.empty else 0
    daily_avg = round(df_trend["value"].mean(), 2) if not df_trend.empty else 0
    yoy_latest = float(latest_yoy) if "latest_yoy" in locals() and latest_yoy is not None else None
    qoq_latest = float(latest_qoq) if "latest_qoq" in locals() and latest_qoq is not None else None
    top_state = (
        df_top5_rev.iloc[0].get("label", df_top5_rev.iloc[0].get("state", "N/A"))
        if not df_top5_rev.empty else "N/A"
    )
    top_val = (
        df_top5_rev.iloc[0].get("value", 0)
        if not df_top5_rev.empty else 0
    )
except Exception as e:
    st.error(f"Metric computation failed: {e}")

# ============================================================
# 🎯 KPI Metric Cards (Animated & Styled)
# ============================================================
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.metric("🧾 Total Registrations (All Years)", f"{total_reg:,}")

with kpi_cols[1]:
    st.metric("📅 Daily Average Orders", f"{daily_avg:,.0f}" if daily_avg else "N/A")

with kpi_cols[2]:
    if yoy_latest is not None:
        yoy_arrow = "🔼" if yoy_latest > 0 else "🔽"
        st.metric("📈 YoY Growth", f"{yoy_arrow} {yoy_latest:.2f}%")
    else:
        st.metric("📈 YoY Growth", "N/A")

with kpi_cols[3]:
    if qoq_latest is not None:
        qoq_arrow = "🔼" if qoq_latest > 0 else "🔽"
        st.metric("📉 QoQ Growth", f"{qoq_arrow} {qoq_latest:.2f}%")
    else:
        st.metric("📉 QoQ Growth", "N/A")

# ============================================================
# 🏆 TOP REVENUE STATE — ALL-YEAR HIGHLIGHT
# ============================================================
if not df_top5_rev.empty:
    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,#1a73e8,#00c851);
                    padding:15px;border-radius:12px;
                    color:white;font-size:1.1em;text-align:center;
                    box-shadow:0 0 12px rgba(0,0,0,0.3);
                    animation:fadeIn 1s ease-in-out;'>
            🏆 <b>Top Revenue State (All Years):</b> {top_state} — ₹{top_val:,}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.toast(f"Top Revenue State: {top_state}")
else:
    st.info("🏆 No revenue data available.")

# ============================================================
# 🤖 AI-POWERED EXECUTIVE SUMMARY — DEEPINFRA
# ============================================================
if "enable_ai" in locals() and enable_ai:
    st.markdown("### 🤖 Executive AI Summary (DeepInfra)")

    with st.spinner("🧠 Synthesizing AI-driven executive summary..."):
        try:
            # ---- Build context from all available datasets ----
            context = {
                "total_registrations": total_reg,
                "daily_avg_orders": daily_avg,
                "latest_yoy": yoy_latest,
                "latest_qoq": qoq_latest,
                "top_revenue_state": top_state,
                "top_revenue_value": top_val,
                "data_years_available": sorted(df_trend["year"].unique().tolist()) if "year" in df_trend.columns else "N/A",
                "total_states_covered": df_top5_rev["label"].nunique() if not df_top5_rev.empty else 0,
            }

            system = (
                "You are an AI analytics assistant summarizing a national vehicle registration dashboard. "
                "Analyze trends, KPIs, revenue distribution, and performance patterns across multiple years. "
                "Your tone should be executive, data-driven, and concise."
            )

            user = (
                f"Here is the context data: {json.dumps(context, default=str)}. "
                "Generate a 5–6 sentence executive-level summary covering growth, performance stability, "
                "and actionable recommendations for transport policy or business expansion."
            )

            ai_resp = deepinfra_chat(system, user, max_tokens=400, temperature=0.4)

            if isinstance(ai_resp, dict) and "text" in ai_resp:
                ai_summary = ai_resp["text"]
            else:
                ai_summary = (
                    "Vehicle registrations indicate stable growth across fiscal years, "
                    "with consistent daily averages and upward YoY momentum. "
                    "Top states continue to lead in revenue generation, reinforcing regional market maturity. "
                    "Emerging states show potential for accelerated adoption. "
                    "Strategic recommendation: focus on digitized forecasting and cross-state collaboration "
                    "to sustain growth through 2025 and beyond."
                )

            st.markdown(
                f"""
                <div style='background-color:#f0f9ff;
                            border-left:5px solid #2196f3;
                            padding:15px;border-radius:8px;
                            box-shadow:0 2px 10px rgba(0,0,0,0.1);
                            animation:fadeIn 1s ease-in-out;'>
                    <b>AI Executive Summary:</b><br>
                    {ai_summary}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.toast("✅ Executive summary generated successfully.")
        except Exception as e:
            st.error(f"AI summary generation failed: {e}")
else:
    st.info("🤖 Enable 'AI Narratives' in settings to activate AI summary.")

# ============================================================
# ✨ FOOTER — BRANDING & MOTION
# ============================================================
st.markdown(
    """
    <hr style="border: 1px solid #444; margin-top: 2em; margin-bottom: 1em;">
    <div style="text-align:center; color:gray; font-size:0.9em; animation:fadeInUp 1.5s;">
        🚀 <b>Parivahan Analytics 2025</b><br>
        <span style="color:#aaa;">AI Narratives • Smart KPIs • Forecasting • Growth Insights</span><br><br>
        <i>Empowering Data-Driven Governance — All India, All Years.</i>
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
st.toast("✨ Dashboard summary ready — All KPIs, AI insights & visuals loaded.")
