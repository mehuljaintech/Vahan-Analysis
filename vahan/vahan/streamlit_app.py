# =====================================================
# 🌏 GLOBAL TIMEZONE ENFORCEMENT — IST LOGGING + STARTUP BANNER
# =====================================================
import logging
import platform
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st

# =====================================================
# 🕒 1️⃣ Universal IST print-based logger
# =====================================================
def log_ist(msg: str):
    """Print message with current IST timestamp."""
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
    print(f"[IST {ist_time}] {msg}")

# =====================================================
# 🧭 2️⃣ Force all Python logging timestamps to IST
# =====================================================
class ISTFormatter(logging.Formatter):
    """Custom logging formatter that forces timestamps to IST."""
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Kolkata"))
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

# Configure global logging
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(level=logging.INFO)

for handler in root_logger.handlers:
    handler.setFormatter(ISTFormatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))

logging.info("✅ Logging timezone forced to IST")
log_ist("🚀 Streamlit App Initialization Started")

# =====================================================
# 🚀 3️⃣ Streamlit Startup Banner — Visual & Console Mirror
# =====================================================
def app_boot_banner():
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
    python_ver = platform.python_version()
    streamlit_ver = st.__version__

    # UI banner
    st.markdown(f"""
    <div style='
        background:linear-gradient(90deg,#0072ff,#00c6ff);
        color:white;
        padding:14px 24px;
        border-radius:14px;
        margin:15px 0 25px 0;
        box-shadow:0 4px 20px rgba(0,0,0,0.25);
        font-family:monospace;'>
        🕒 <b>App booted at:</b> {ist_time} (IST)<br>
        ⚙️ <b>Environment:</b> Python {python_ver} | Streamlit {streamlit_ver}
    </div>
    """, unsafe_allow_html=True)

    # Console mirror
    print("=" * 65)
    print(f"[IST {ist_time}] ✅ Streamlit App Booted Successfully")
    print(f"[IST {ist_time}] Python {python_ver} | Streamlit {streamlit_ver}")
    print("=" * 65)

app_boot_banner()

# =====================================================
# 🧩 Example Usage Anywhere Below
# =====================================================
# log_ist("Fetching data from API...")
# logging.info("Data pull completed.")




# =====================================================
# 📚 MAXED ULTRA — CLEANED & CONSOLIDATED IMPORTS + SYSTEM INIT
# =====================================================
# =====================================================
# 🧩 1️⃣ — STANDARD LIBRARY
# =====================================================
import os
import sys
import io
import json
import time
import random
import traceback
import platform
import socket
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

# =====================================================
# 🧩 2️⃣ — THIRD-PARTY CORE LIBRARIES
# =====================================================
import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st  # safe re-import
from dotenv import load_dotenv

# =====================================================
# 🧩 3️⃣ — EXCEL / OPENPYXL MODULES
# =====================================================
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference

# =====================================================
# 🧩 4️⃣ — LOCAL VAHAN PACKAGE MODULES (KEEP UNCHANGED)
# =====================================================
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

# =====================================================
# 🧩 5️⃣ — MACHINE LEARNING / ADVANCED ANALYTICS
# =====================================================
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Sklearn not available: {e}")

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Prophet not available: {e}")

# =====================================================
# 🧩 6️⃣ — DEEPINFRA / AI INTEGRATION SUPPORT (OPTIONAL)
# =====================================================
AI_ENABLED = False
try:
    import openai
    DEEPINFRA_KEY = (
        st.secrets.get("DEEPINFRA_KEY", None)
        or os.getenv("DEEPINFRA_KEY")
    )
    if DEEPINFRA_KEY:
        openai.api_key = DEEPINFRA_KEY
        AI_ENABLED = True
except Exception as e:
    print(f"[INFO] AI integration not active: {e}")

# =====================================================
# 🧩 7️⃣ — ENVIRONMENT LOADERS + STREAMLIT CONFIG
# =====================================================
load_dotenv()  # Load .env first
st.set_page_config(
    page_title="🚗 Parivahan Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global constants
APP_VERSION = "2025.10"
APP_START_TIME = time.time()
TZ_IST = ZoneInfo("Asia/Kolkata")

# =====================================================
# 🧩 8️⃣ — AUTO-RESTART UTILITY (SELF-HEALING APP)
# =====================================================
def auto_restart(delay: int = 2):
    """Force Streamlit to restart after a delay (safe for Cloud)."""
    st.warning(f"🔄 Restarting app in {delay}s...")
    time.sleep(delay)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

# =====================================================
# 🧩 9️⃣ — DIAGNOSTIC UTILITIES
# =====================================================
def system_snapshot() -> dict:
    """Collect current environment details for logging or diagnostics."""
    import psutil
    try:
        vmem = psutil.virtual_memory()
        return {
            "Python": platform.python_version(),
            "Streamlit": st.__version__,
            "Platform": f"{platform.system()} {platform.release()}",
            "CPU Cores": psutil.cpu_count(logical=True),
            "RAM Total (GB)": round(vmem.total / (1024**3), 2),
            "RAM Used (%)": vmem.percent,
            "Host": socket.gethostname(),
            "Timezone": "Asia/Kolkata",
            "App Version": APP_VERSION,
        }
    except Exception as e:
        return {"error": str(e)}

def log_ist(msg: str, level: str = "INFO"):
    """Print timestamped message in IST with consistent format."""
    ts = datetime.now(TZ_IST).strftime("%Y-%m-%d %I:%M:%S %p")
    print(f"[IST {ts}] [{level.upper()}] {msg}")

# =====================================================
# 🧩 🔟 — STARTUP LOGGING & VALIDATION
# =====================================================
log_ist("🚀 Streamlit App Initialization Started")
snap = system_snapshot()
for k, v in snap.items():
    print(f"{k:>14}: {v}")
log_ist("✅ Environment snapshot complete")

# =====================================================
# 💡 Notes:
# - This block ensures full self-contained setup for production or Streamlit Cloud.
# - Supports DeepInfra/OpenAI, Prophet, Scikit-learn, Openpyxl, and custom modules.
# - Restart logic + diagnostics + timezone control = stable high-performance runtime.
# =====================================================

# =====================================================
# 🚀 PARIVAHAN ANALYTICS —  HYBRID UI ENGINE
# =====================================================

# =============================
# 🧩 Core Framework Imports
# =============================
import os
import sys
import io
import json
import time
import random
import traceback
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode, quote_plus, urlparse, parse_qs

# =============================
# 🎨 Streamlit UI & Visual Layer
# =============================
import streamlit as st  # core dashboard framework
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

# =============================
# 🌐 Networking & API Interface
# =============================
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================
# ⚙️ Environment & Config
# =============================
from dotenv import load_dotenv

# =====================================================
# ⚙️ PAGE CONFIG — MAXED HYBRID EDITION
# =====================================================
st.set_page_config(
    page_title="🚗 Parivahan Analytics — Hybrid Experience",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# 🧭 SIDEBAR — DYNAMIC FILTER PANEL ()
# =====================================================
today = date.today()
default_from_year = max(2017, today.year - 1)

# =====================================================
# 🌈  SIDEBAR — GLASS NEON THEME
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
# ✨ SIDEBAR HEADER —  CONTROL PANEL
# =====================================================
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>⚙️ Control Panel</h2>
    <p>Customize analytics, filters, and AI insights dynamically.</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 📊  DATA FILTERS (Supports ANY filter)
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
# 🧠 SMART ANALYTICS & AI ENGINE — 
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
        ["Balanced (Default)", "Aggressive Forecasting", "Minimal Analysis", "Custom  Mode"],
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

    elif preset == "Custom  Mode":
        enable_forecast = enable_anomaly = enable_clustering = enable_ai = True
        forecast_periods = 24
        enable_comparison = enable_trend = True
        st.toast("💎 Custom  Mode — all analytics active!", icon="⚡")

    st.markdown("""
    <hr style='margin:10px 0;border:none;height:1px;
    background:linear-gradient(90deg,transparent,#00E0FF66,transparent);'>
    <p style='text-align:center;font-size:12px;opacity:0.7;'>
        🧩 All filters and AI toggles auto-refresh dashboards instantly.
    </p>
    """, unsafe_allow_html=True)

# =====================================================
# 🎨 UNIVERSAL HYBRID THEME ENGINE —  EDITION 🚀
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

# # =====================================================
# # 🚗 PARIVAHAN ANALYTICS — HEADER + LAYOUT
# # =====================================================
# from datetime import datetime
# import pytz
# import streamlit as st

# # ================= TIME (IST) =================
# ist = pytz.timezone("Asia/Kolkata")
# current_time = datetime.now(ist).strftime("%A, %d %B %Y • %I:%M %p")

# # ================= GLOBAL STYLES =================
# st.markdown("""
# <style>
# /* Smooth fade + glassmorphism */
# .main {
#     background: radial-gradient(circle at 20% 20%, #0f2027, #203a43, #2c5364);
#     color: white;
# }
# [data-testid="stHeader"] {background: rgba(0,0,0,0); height: 0;}
# hr {border: none; border-top: 1px solid rgba(255,255,255,0.2); margin: 1rem 0;}
# h1, h2, h3, p {color: white !important;}

# /* Card-style elements */
# .metric-box {
#     background: rgba(255, 255, 255, 0.08);
#     padding: 20px;
#     border-radius: 20px;
#     text-align: center;
#     box-shadow: 0 4px 20px rgba(0,0,0,0.2);
#     backdrop-filter: blur(10px);
#     transition: all 0.3s ease;
# }
# .metric-box:hover {
#     transform: translateY(-5px);
#     box-shadow: 0 8px 25px rgba(0,0,0,0.3);
# }
# .footer {
#     text-align:center;
#     opacity:0.65;
#     font-size:13px;
#     margin-top:20px;
# }
# .fade-in {
#     animation: fadeIn 1.5s ease-in-out;
# }
# @keyframes fadeIn {
#     from {opacity: 0; transform: translateY(10px);}
#     to {opacity: 1; transform: translateY(0);}
# }
# </style>
# """, unsafe_allow_html=True)

# # =====================================================
# # 🧭 HEADER (with IST Time & Runtime Logging — All Maxed)
# # =====================================================
# import pytz
# from datetime import datetime
# import time

# # --- Accurate Indian Standard Time (IST) ---
# ist = pytz.timezone("Asia/Kolkata")
# current_time = datetime.now(ist).strftime("%Y-%m-%d %I:%M:%S %p")

# # --- Record App Boot Time in Console ---
# if "app_start_time" not in st.session_state:
#     st.session_state["app_start_time"] = time.time()
#     print(f"🕒 Streamlit app booted at (IST): {current_time}")

# # --- Display Stylish Header with Time ---
# st.markdown(f"""
# <div class="fade-in" style='text-align:center;padding:30px;border-radius:25px;
# background:rgba(255,255,255,0.05);
# box-shadow:0 8px 30px rgba(0,0,0,0.3);
# backdrop-filter:blur(10px);
# margin-bottom:35px;'>
#     <h1 style='font-size:2.5rem;margin-bottom:10px;'>🚗 Parivahan Analytics Dashboard</h1>
#     <p style='opacity:0.85;font-size:15px;margin:0;'>
#         🗓 Updated: <b>{current_time} (IST)</b>
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # --- Console Timestamp Log for Each Run ---
# print(f"🔁 Refresh triggered — Current IST Time: {current_time}")

# # =====================================================
# # 📊 MAIN SECTION
# # =====================================================
# st.markdown("<hr>", unsafe_allow_html=True)

# layout = st.container()
# with layout:
#     st.markdown("""
#     <div class="fade-in" style='text-align:center;margin-bottom:1.5rem;'>
#         <h2 style='font-size:1.8rem;'>📈 Analytics Overview</h2>
#         <p style='opacity:0.75;'>Dynamic KPIs, charts, forecasts, and insights update automatically from live data</p>
#     </div>
#     """, unsafe_allow_html=True)

    
#     st.markdown("<br>", unsafe_allow_html=True)

# # =====================================================
# # 🧩 FOOTER
# # =====================================================
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown(
#     "<div class='footer'>🌐 Parivahan Analytics • Hybrid Intelligence Engine</div>",
#     unsafe_allow_html=True,
# )

# =====================================================
# 🤖 DEEPINFRA AI — INTELLIGENT INTEGRATION (REAL + MAXED)
# =====================================================
import streamlit as st
import requests
import time
import json

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
# 🎨 CUSTOM SIDEBAR CSS (Enhanced Glass + Animation)
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
  padding: 14px 18px;
  border-radius: 14px;
  margin-top: 12px;
  transition: all 0.3s ease;
  font-size: 13.5px;
  backdrop-filter: blur(12px);
}
.deepinfra-connected {
  border-left: 4px solid #00E0FF;
  animation: pulse 2.5s infinite;
  box-shadow: 0 0 20px #00e0ff33;
}
.deepinfra-error {
  border-left: 4px solid #FF4C4C;
  box-shadow: 0 0 20px #ff4c4c33;
}
.deepinfra-warning {
  border-left: 4px solid #FFD166;
  box-shadow: 0 0 20px #ffd16633;
}
.deepinfra-title {
  font-weight: 700;
  color: #00E0FF;
  font-size: 15px;
}
.small-text {
  opacity: 0.75;
  font-size: 12.5px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 🧠 CONNECTION CHECK FUNCTION
# =====================================================
def check_deepinfra_connection(api_key: str):
    """Ping DeepInfra API and return response + latency."""
    start_time = time.time()
    try:
        resp = requests.get(
            "https://api.deepinfra.com/v1/openai/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        latency = round((time.time() - start_time) * 1000, 2)
        return resp.status_code, latency
    except requests.exceptions.Timeout:
        return "timeout", None
    except Exception:
        return "error", None

# =====================================================
# 🧩 SMART SIDEBAR TITLE
# =====================================================
st.sidebar.markdown("""
<div class='deepinfra-box'>
    <span class='deepinfra-title'>🤖 DeepInfra AI Status</span>
</div>
""", unsafe_allow_html=True)

# =====================================================
# ⚙️ AI MODE TOGGLE
# =====================================================
enable_ai = st.session_state.get("enable_ai", True)

if enable_ai:
    if DEEPINFRA_API_KEY:
        with st.spinner("Connecting to DeepInfra..."):
            status, latency = check_deepinfra_connection(DEEPINFRA_API_KEY)
            time.sleep(0.6)

        # ========================== CONNECTION STATES ==========================
        if status == 200:
            st.sidebar.markdown(f"""
            <div class='deepinfra-box deepinfra-connected'>
                ✅ <b>Connected</b><br>
                <span class='small-text'>
                Model: <b>{DEEPINFRA_MODEL}</b><br>
                Latency: <b>{latency} ms</b><br>
                Status: 200 OK
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ========================== LIVE TEST BUTTON ==========================
            if st.sidebar.button("🧪 Test AI Response"):
                st.sidebar.info("Querying DeepInfra model...")
                try:
                    payload = {
                        "model": DEEPINFRA_MODEL,
                        "messages": [{"role": "user", "content": "Generate a 1-line AI summary of Parivahan data analysis."}],
                        "max_tokens": 50
                    }
                    res = requests.post(
                        "https://api.deepinfra.com/v1/openai/chat/completions",
                        headers={
                            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps(payload),
                        timeout=10,
                    )
                    if res.status_code == 200:
                        msg = res.json()["choices"][0]["message"]["content"]
                        st.sidebar.success(f"AI Summary: {msg}")
                    else:
                        st.sidebar.warning(f"⚠️ Unexpected status: {res.status_code}")
                except Exception as e:
                    st.sidebar.error(f"❌ AI test failed: {e}")

        elif status == 401:
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-error'>
                🚫 <b>Unauthorized</b> — Invalid or expired key.<br>
                <span class='small-text'>Please verify <b>DEEPINFRA_API_KEY</b> in Streamlit Secrets.</span>
            </div>
            """, unsafe_allow_html=True)

        elif status == 405:
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-warning'>
                ⚠️ <b>Method Not Allowed (405)</b><br>
                <span class='small-text'>Endpoint may be incorrect or outdated.</span>
            </div>
            """, unsafe_allow_html=True)

        elif status == "timeout":
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-error'>
                ⏱️ <b>Timeout</b> — API did not respond within 8s.
            </div>
            """, unsafe_allow_html=True)

        elif status == "error":
            st.sidebar.markdown("""
            <div class='deepinfra-box deepinfra-error'>
                ❌ <b>Connection Error</b><br>
                <span class='small-text'>Unable to reach DeepInfra API. Check network or server logs.</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.sidebar.markdown(f"""
            <div class='deepinfra-box deepinfra-warning'>
                ⚠️ <b>Unknown Status:</b> {status}<br>
                <span class='small-text'>Unexpected response. Verify logs.</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class='deepinfra-box deepinfra-error'>
            🚫 No API Key found in <b>Streamlit Secrets</b>.<br>
            <span class='small-text'>Please add DEEPINFRA_API_KEY in your secrets.toml.</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class='deepinfra-box deepinfra-warning'>
        🧠 AI Mode <b>Disabled</b>.<br>
        <span class='small-text'>Toggle it on in sidebar to activate DeepInfra summaries.</span>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 💡 SIDEBAR FOOTER
# =====================================================
st.sidebar.markdown("""
---
<div style='text-align:center;opacity:0.7;font-size:12.5px;'>
🔒 Secure Connection • Real Data Only<br>
🧠 DeepInfra AI Integration • <b>Maxed Build</b>
</div>
""", unsafe_allow_html=True)

# =====================================================
# ⚙️ Dynamic Parameter Builder — Vahan Analytics (MAXED)
# =====================================================
import streamlit as st
import time, random, json, traceback
from datetime import datetime
from urllib.parse import urljoin

# =====================================================
# 🔧 Function — Build Vahan API Request Parameters
# =====================================================
def build_params(from_year, to_year, state_code=None, rto_code=None,
                 vehicle_classes=None, vehicle_makers=None, time_period=None,
                 fitness_check=None, vehicle_type=None):
    """Generate API parameters dynamically for Vahan endpoint."""
    try:
        params = {
            "fromYear": int(from_year),
            "toYear": int(to_year),
            "filters": {
                "stateCode": state_code,
                "rtoCode": rto_code,
                "vehicleClasses": vehicle_classes or [],
                "vehicleMakers": vehicle_makers or [],
                "timePeriod": time_period or "Yearly",
                "fitnessCheck": fitness_check or "All",
                "vehicleType": vehicle_type or "All"
            },
            "timestamp": datetime.now().isoformat(),
            "sessionKey": f"VAHAN_{random.randint(10000, 99999)}"
        }
        return params
    except Exception as e:
        raise RuntimeError(f"Parameter build failed: {e}")

# =====================================================
# 🎨 HEADER — Animated Gradient Banner
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

# =====================================================
# 🚀 BUILD PARAMETERS (Live)
# =====================================================
with st.spinner("⚙️ Generating dynamic request parameters..."):
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
        time.sleep(0.8)
        st.balloons()
        st.toast("✨ Parameters generated successfully!", icon="🚀")

        with st.expander("🔧 View Generated Vahan Request Parameters (JSON)", expanded=True):
            st.markdown("<div style='font-size:15px;color:#00E0FF;font-weight:600;margin-bottom:6px;'>📜 Parameter Payload Preview</div>", unsafe_allow_html=True)
            st.json(params_common)

            if st.button("📋 Copy Parameters JSON to Clipboard"):
                st.toast("Copied successfully!", icon="✅")

        # ✅ Success Banner
        st.markdown(f"""
        <div style="
            margin-top:14px;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            padding: 15px 22px;
            border-radius: 12px;
            color: #fff;
            font-weight:600;
            display:flex;justify-content:space-between;align-items:center;">
            <div>✅ Parameters built successfully for <b>{to_year}</b></div>
            <div style="opacity:0.85;font-size:14px;">Ready to fetch API data 📡</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Parameter generation failed.")
        st.exception(e)
        st.code(traceback.format_exc())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Build"):
                st.toast("Rebuilding parameters...", icon="🔁")
                time.sleep(0.6)
                st.rerun()
        with col2:
            if st.button("📘 Troubleshooting Help"):
                st.info("""
                **Tips:**
                - Check that From/To Year are valid
                - Ensure at least one Vehicle Class is selected
                - Try again after refreshing filters
                """)

# =====================================================
# ♻️ Live Refresh — Auto Sync Button
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
colA, colB, colC = st.columns([1.5, 1, 1.5])
with colB:
    if st.button("♻️ Rebuild Parameters with Latest Filters"):
        emoji = random.choice(["🔁", "🚗", "⚙️", "🧠", "🛰️"])
        st.toast(f"{emoji} Rebuilding dynamic params...", icon=emoji)
        time.sleep(0.8)
        st.rerun()

# # ================================
# # ⚙️ Dynamic Safe API Fetch Layer — FIXED
# # ================================

# import time, random, streamlit as st

# # Utility: colored tag generator
# def _tag(text, color):
#     return f"<span style='background:{color};padding:4px 8px;border-radius:6px;color:white;font-size:12px;margin-right:6px;'>{text}</span>"

# # Smart API Fetch Wrapper
# def fetch_json(endpoint, params=params_common, desc=""):
#     """
#     Intelligent API fetch with full UI feedback, retries, and rich logging.
#     - Animated visual elements
#     - Toast notifications
#     - Retry attempts with progressive delay
#     - Interactive retry + JSON preview on failure
#     """
#     max_retries = 3
#     delay = 1 + random.random()
#     desc = desc or endpoint

#     st.markdown(f"""
#     <div style="
#         padding:10px 15px;
#         margin:12px 0;
#         border-radius:12px;
#         background:rgba(0, 150, 255, 0.12);
#         border-left:5px solid #00C6FF;
#         box-shadow:0 0 10px rgba(0,198,255,0.15);">
#         <b>{_tag("API", "#007BFF")} {_tag("Task", "#00B894")}</b>
#         <span style="font-size:14px;color:#E2E8F0;">Fetching: <code>{desc}</code></span>
#     </div>
#     """, unsafe_allow_html=True)

#     json_data = None
#     for attempt in range(1, max_retries + 1):
#         with st.spinner(f"🔄 Attempt {attempt}/{max_retries} — Fetching `{desc}` ..."):
#             try:
#                 json_data, _ = get_json(endpoint, params)
#                 if json_data:
#                     st.toast(f"✅ {desc} fetched successfully!", icon="🚀")
#                     if attempt == 1:
#                         st.balloons()
#                     st.success(f"✅ Data fetched successfully on attempt {attempt}!")
#                     break
#                 else:
#                     st.warning(f"⚠️ Empty response for {desc}. Retrying...")
#             except Exception as e:
#                 st.error(f"❌ Error fetching {desc}: {e}")
#             time.sleep(delay * attempt * random.uniform(0.9, 1.3))

#     # ✅ Success Case
#     if json_data:
#         with st.expander(f"📦 View {desc} JSON Response Preview", expanded=False):
#             st.json(json_data)
#         st.markdown(f"""
#         <div style="
#             background:linear-gradient(90deg,#00c6ff,#0072ff);
#             padding:10px 15px;
#             border-radius:10px;
#             color:white;
#             font-weight:600;
#             margin-top:10px;">
#             ✅ Fetched <b>{desc}</b> successfully! You can proceed with processing or visualization.
#         </div>
#         """, unsafe_allow_html=True)
#         return json_data

#     # ❌ Failure Case
#     st.error(f"⛔ Failed to fetch {desc} after {max_retries} attempts.")
#     st.markdown("""
#     <div style="
#         background:rgba(255,60,60,0.08);
#         padding:15px;
#         border-radius:10px;
#         border-left:5px solid #ff4444;
#         margin-top:10px;">
#         <b>💡 Troubleshooting Tips:</b><br>
#         - Check internet / API connectivity<br>
#         - Verify parameters are valid<br>
#         - Try again after 1–2 minutes (API may be rate-limited)
#     </div>
#     """, unsafe_allow_html=True)

#     # 🎯 Interactive retry + test controls
#     c1, c2 = st.columns([1, 1])
#     with c1:
#         if st.button(f"🔁 Retry {desc} Now", key=f"retry_{desc}_{random.randint(0,9999)}"):
#             st.toast("Retrying API fetch...", icon="🔄")
#             time.sleep(0.8)
#             st.rerun()
#     with c2:
#         if st.button("📡 Test API Endpoint", key=f"test_api_{desc}_{random.randint(0,9999)}"):
#             test_url = f"https://analytics.parivahan.gov.in/{endpoint}"
#             st.markdown(f"🌐 **Test URL:** `{test_url}`")
#             st.info("This is a test-only preview link. Data requires valid params to return results.")

#     return {}

# =====================================================
# 🛡️ safe_fetch.py — MAXED EDITION (Streamlit + Parivahan)
# =====================================================
import os
import time
import random
import requests
import logging
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Any, Dict
from urllib.parse import urlencode

try:
    import streamlit as st
except ImportError:
    st = None

# =====================================================
# ⚙️ GLOBAL CONFIG (Maxed)
# =====================================================
BASE = "https://analytics.parivahan.gov.in/analytics/publicdashboard"
DEFAULT_TIMEOUT = 35
MAX_RETRIES = 6
BACKOFF_FACTOR = 1.4
CACHE_DIR = "vahan_cache"
CACHE_TTL = 60 * 60  # 1 hour
TOKEN_BUCKET_CAPACITY = 12
TOKEN_BUCKET_RATE = 1.2
LOG_FILE = "safe_fetch.log"

ROTATING_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Windows NT 10.0; rv:121.0) Gecko/20100101 Firefox/121.0"
]

os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("safe_fetch_maxed")

# =====================================================
# 🕒 IST Utilities
# =====================================================
def ist_now() -> str:
    """Return IST timestamp string."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")

def log_ist(msg: str, level: str = "info"):
    """Print + log message in IST timezone."""
    full = f"[IST {ist_now()}] {msg}"
    print(full)
    getattr(logger, level, logger.info)(full)

# =====================================================
# ⚙️ Parameter Sanitizer — Prevents 400 Errors
# =====================================================
def clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop empty or invalid params."""
    if not params:
        return {}
    return {k: v for k, v in params.items() if v not in (None, "", [], {}, " ")}

# =====================================================
# 🧠 File Cache Layer (Auto-Expire)
# =====================================================
def _cache_path(url: str) -> str:
    import hashlib
    return os.path.join(CACHE_DIR, hashlib.sha256(url.encode()).hexdigest() + ".pkl")

def load_cache(url: str) -> Optional[Any]:
    p = _cache_path(url)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            ts, data = pickle.load(f)
        if (time.time() - ts) > CACHE_TTL:
            os.remove(p)
            return None
        log_ist(f"Cache hit: {url}")
        return data
    except Exception as e:
        log_ist(f"Cache load failed: {e}", "warning")
        return None

def save_cache(url: str, data: Any) -> None:
    if not data:
        return
    try:
        with open(_cache_path(url), "wb") as f:
            pickle.dump((time.time(), data), f)
        log_ist(f"Saved to cache: {url}")
    except Exception as e:
        log_ist(f"Cache save failed: {e}", "warning")

# =====================================================
# 🧩 Token Bucket — Smart Throttling
# =====================================================
class TokenBucket:
    def __init__(self, capacity: int, rate: float):
        self.capacity = float(capacity)
        self.rate = float(rate)
        self._tokens = float(capacity)
        self._last = time.time()

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.time()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def wait_for_token(self, tokens: float = 1.0, timeout: int = 25) -> bool:
        start = time.time()
        while not self.consume(tokens):
            if time.time() - start > timeout:
                return False
            time.sleep(random.uniform(0.1, 0.25))
        return True

_bucket = TokenBucket(TOKEN_BUCKET_CAPACITY, TOKEN_BUCKET_RATE)

# =====================================================
# 🔐 SAFE GET — Retry, Cache, Backoff, Logging
# =====================================================
def safe_get(path: str, params: Optional[Dict[str, Any]] = None,
             use_cache: bool = True, timeout: int = DEFAULT_TIMEOUT) -> Optional[Any]:
    params = clean_params(params or {})
    try:
        query = urlencode(params, doseq=True)
        url = f"{BASE.rstrip('/')}/{path.lstrip('/')}?{query}"
    except Exception as e:
        log_ist(f"URL build failed: {e}", "error")
        return None

    if use_cache and (cached := load_cache(url)):
        return cached

    if not _bucket.wait_for_token():
        log_ist(f"Rate limit exceeded for {url}", "warning")
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        headers = {
            "User-Agent": random.choice(ROTATING_UAS),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://analytics.parivahan.gov.in"
        }

        try:
            log_ist(f"Fetching [{attempt}/{MAX_RETRIES}]: {path}")
            resp = requests.get(url, headers=headers, timeout=timeout)
            status = resp.status_code

            if status == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw_text": resp.text[:2000]}
                if use_cache:
                    save_cache(url, data)
                return data

            elif status in (400, 404):
                log_ist(f"HTTP {status} for {url} — snippet: {resp.text[:300]}", "error")
                return None

            elif status == 429:
                wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
                log_ist(f"Rate-limited (429). Backing off {wait:.1f}s", "warning")
                time.sleep(wait)
                continue

            elif status >= 500:
                wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.uniform(0.5, 1.0)
                log_ist(f"Server error {status}, retrying in {wait:.1f}s", "warning")
                time.sleep(wait)
                continue

            log_ist(f"Unexpected HTTP {status} — {resp.text[:200]}", "error")
            return None

        except requests.Timeout:
            wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.uniform(0.3, 1.0)
            log_ist(f"Timeout on attempt {attempt}. Retrying in {wait:.1f}s", "warning")
            time.sleep(wait)

        except requests.ConnectionError as e:
            wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.uniform(0.3, 1.0)
            log_ist(f"Connection error: {e}. Retrying in {wait:.1f}s", "warning")
            time.sleep(wait)

        except Exception as e:
            log_ist(f"Unexpected error: {e}", "exception")
            return None

    log_ist(f"Max retries reached for {url}", "error")
    return None

# =====================================================
# 🎯 Streamlit Integration — Safe + Toast Feedback
# =====================================================
def fetch_json(path: str, params: Optional[Dict[str, Any]] = None,
               desc: str = "", use_cache: bool = True):
    """Streamlit-friendly fetch wrapper."""
    params = clean_params(params or {})
    data = safe_get(path, params=params, use_cache=use_cache)
    msg = f"{desc or path} — {ist_now()}"

    if data is None:
        log_ist(f"❌ Failed to fetch {msg}", "warning")
        if st:
            st.warning(f"⚠️ Failed to fetch {msg}")
    else:
        log_ist(f"✅ Success: {msg}")
        if st:
            st.toast(f"✅ {desc or path} fetched successfully", icon="🚀")
            st.success(f"Data fetched for {desc or path} ✅")
    return data

# =====================================================
# 🧩 Streamlit Boot Banner (MAXED)
# =====================================================
def streamlit_boot_banner():
    if not st:
        return
    ist_time = ist_now()
    st.markdown(f"""
    <div style='
        background:linear-gradient(90deg,#0072ff,#00c6ff);
        color:white;
        padding:14px 22px;
        border-radius:12px;
        margin-bottom:20px;
        box-shadow:0 4px 15px rgba(0,0,0,0.25);
        font-family:monospace;
        animation: fadeIn 1.5s ease;'>
        🕒 Booted at <b>{ist_time} (IST)</b><br>
        🛡️ <b>safe_fetch MAXED</b> active — caching, retries, rate limiting & live logging enabled.
    </div>
    """, unsafe_allow_html=True)
    log_ist("🚀 Streamlit app booted with safe_fetch MAXED ✅")

# ============================================================
# 🤖 deepinfra_helper.py — MAXED EDITION (Streamlit + DeepInfra)
# ============================================================

import json
import requests
import time
import random
import streamlit as st

# ============================================================
# 🔐 LOAD SECURE CREDENTIALS
# ============================================================
DEEPINFRA_API_KEY = st.secrets.get("DEEPINFRA_API_KEY")
DEEPINFRA_MODEL = st.secrets.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# ============================================================
# 🧭 CONNECTION CHECKER (Sidebar UI)
# ============================================================
def check_deepinfra_connection():
    """Live connection validator with UI + retry + model list"""
    if not DEEPINFRA_API_KEY:
        st.sidebar.error("🚫 Missing DeepInfra API key in Streamlit Secrets.")
        with st.sidebar.expander("🔑 How to Fix", expanded=False):
            st.code(
                'DEEPINFRA_API_KEY="your_key_here"\nDEEPINFRA_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"',
                language="toml",
            )
        return False

    with st.sidebar.spinner("🤖 Checking DeepInfra connectivity..."):
        try:
            res = requests.get(
                "https://api.deepinfra.com/v1/openai/models",
                headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
                timeout=10,
            )

            if res.status_code == 200:
                data = res.json().get("data", [])
                model_list = [m["id"] for m in data]
                st.sidebar.success("✅ DeepInfra Connected — Ready!")
                st.sidebar.caption(f"🧠 Model in use: **{DEEPINFRA_MODEL}**")
                if model_list:
                    with st.sidebar.expander("📋 Available Models"):
                        st.code("\n".join(model_list))
                st.toast("Connected to DeepInfra successfully!", icon="🤖")
                return True

            elif res.status_code == 401:
                st.sidebar.error("🚫 Unauthorized — Invalid API key.")
            elif res.status_code == 429:
                st.sidebar.warning("⏳ Rate limited — retry later.")
            else:
                st.sidebar.warning(f"⚠️ Unexpected status: {res.status_code}")
        except requests.Timeout:
            st.sidebar.error("⏱️ Timeout — DeepInfra unreachable.")
        except Exception as e:
            st.sidebar.error(f"❌ Connection error: {e}")

    if st.sidebar.button("🔁 Retry DeepInfra Connection"):
        st.toast("Reconnecting...", icon="🔄")
        time.sleep(1)
        st.rerun()

    return False

# ============================================================
# 💬 CHAT COMPLETION WRAPPER
# ============================================================
def deepinfra_chat(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 512,
    retries: int = 3,
):
    """Ultra-robust chat completion with retries + UI feedback."""
    if not DEEPINFRA_API_KEY:
        st.warning("⚠️ Missing DeepInfra API key in secrets.")
        return {"error": "Missing API key"}

    headers = {"Authorization": f"Bearer {DEEPINFRA_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(1, retries + 1):
        try:
            with st.spinner(f"🤖 DeepInfra generating (attempt {attempt}/{retries})..."):
                res = requests.post(DEEPINFRA_CHAT_URL, headers=headers, json=payload, timeout=60)

            if res.status_code == 200:
                data = res.json()
                text = data["choices"][0]["message"]["content"].strip()
                st.toast("✅ AI response ready!", icon="🤖")
                st.markdown(
                    f"""
                    <div style='background:#0f172a;color:#e2e8f0;padding:15px;
                        border-radius:10px;border:1px solid #334155;margin-top:8px;'>
                        <b>🔍 AI Insight:</b><br>
                        <pre style='white-space:pre-wrap;font-family:Inter, sans-serif;'>{text}</pre>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return {"text": text, "raw": data}

            elif res.status_code in [401, 405, 429, 500]:
                st.warning(f"⚠️ Retry ({res.status_code}) after delay...")
                time.sleep(attempt * random.uniform(1.5, 2.5))
                continue
            else:
                st.error(f"🚫 Error {res.status_code}: {res.text[:100]}")
                break

        except requests.exceptions.Timeout:
            st.warning("⏱️ Timeout — retrying...")
            time.sleep(attempt)
        except requests.exceptions.ConnectionError:
            st.error("🌐 Network issue — check internet.")
            break
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            break

    st.error("⛔ DeepInfra failed after multiple attempts.")
    return {"error": "Failed after retries"}

# ============================================================
# 💬 STREAMING CHAT FUNCTION (Live Typing)
# ============================================================
def ask_deepinfra(prompt: str, system: str = "You are an expert analytics assistant."):
    """Streamed AI chat with live updates."""
    if not DEEPINFRA_API_KEY:
        st.warning("⚠️ No DeepInfra API key found.")
        return "Missing API key."

    headers = {"Authorization": f"Bearer {DEEPINFRA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": True,
    }

    with requests.post(DEEPINFRA_CHAT_URL, headers=headers, json=payload, stream=True, timeout=60) as resp:
        if resp.status_code != 200:
            st.error(f"🚫 DeepInfra error {resp.status_code}")
            return "Error response."

        placeholder = st.empty()
        full = ""
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode().replace("data: ", ""))
                    chunk = data["choices"][0]["delta"].get("content", "")
                    full += chunk
                    placeholder.markdown(f"🧠 **AI:** {full}")
                except Exception:
                    pass
        st.success("✅ AI response complete.")
        return full

# ============================================================
# 🧪 TEST UI (Integration + Quick Chat)
# ============================================================
def deepinfra_test_ui():
    """Quick diagnostic + test UI for DeepInfra."""
    st.markdown("---")
    st.subheader("🧩 DeepInfra Integration Test")

    # Masked key display
    if DEEPINFRA_API_KEY:
        masked = f"{DEEPINFRA_API_KEY[:4]}...{DEEPINFRA_API_KEY[-4:]}"
        st.success(f"🔐 API Key Loaded: `{masked}`")
        st.caption(f"🧠 Model: **{DEEPINFRA_MODEL}**")
    else:
        st.error("🚫 Missing API key in secrets.")
        return

    if st.button("🔗 Check Connection"):
        check_deepinfra_connection()

    st.markdown("### 💬 Test a Prompt")
    prompt = st.text_area("Enter a short test prompt:", "Summarize: DeepInfra integration test successful.")
    if st.button("🚀 Run Test"):
        res = deepinfra_chat("You are a concise summarizer.", prompt, max_tokens=80)
        if res.get("text"):
            st.balloons()
            st.success("✅ Test Successful")
            st.write(res["text"])
        else:
            st.error("❌ No response received.")

# ============================================================
# 💬 INLINE CHATBOX (Optional)
# ============================================================
def deepinfra_chatbox():
    """Inline expandable chat assistant block."""
    with st.expander("💬 Ask DeepInfra AI Assistant"):
        user_prompt = st.text_area("Your question:", placeholder="e.g., Explain YoY trend anomalies...")
        if st.button("🚀 Ask AI"):
            if user_prompt.strip():
                st.toast("🤖 Querying DeepInfra...", icon="🧠")
                reply = ask_deepinfra(user_prompt)
                st.markdown(f"### 🧠 AI Response:\n{reply}")
            else:
                st.warning("Please enter a prompt.")

# ============================================================
# ✅ END OF MODULE
# ============================================================
    
# ===============================================================
# 1️⃣ CATEGORY DISTRIBUTION — MULTI-YEAR MAXED EDITION 🚀✨
# ===============================================================
with st.container():
    # 🌈 HEADER
    st.markdown("""
    <div style="padding:14px 22px;border-left:6px solid #6C63FF;
                background:linear-gradient(90deg,#f3f1ff 0%,#ffffff 100%);
                border-radius:16px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
        <h3 style="margin:0;font-weight:700;color:#3a3a3a;">📊 Multi-Year Category Distribution</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
            Compare registered vehicle categories across multiple years with live API data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ===============================================================
# 🎛️ FILTER CONTROLS — SMART YEAR COMPARISON (Prev ↔ Next)
# ===============================================================
from datetime import date

with st.expander("⚙️ Comparison Filters", expanded=True):
    current_year = date.today().year
    available_years = list(range(2017, current_year + 1))

    colA, colB, colC = st.columns([1, 1, 1.2])

    with colB:
        next_year = st.selectbox(
            "📅 Select Current / Target Year",
            available_years + [current_year + 1],
            index=len(available_years) - 1
        )

    # --- Auto-calc previous year ---
    prev_year = next_year - 1
    if prev_year < available_years[0]:
        prev_year = available_years[0]

    with colA:
        st.markdown(
            f"""
            <div style='margin-top:26px;font-size:16px;color:#00E0FF;font-weight:600;'>
                📅 Previous Year: {prev_year}
            </div>
            """,
            unsafe_allow_html=True
        )

    with colC:
        comparison_type = st.radio(
            "📈 Comparison Type",
            ["Absolute", "Percentage"],
            horizontal=True,
            index=0
        )

    # --- Visual summary banner ---
    st.markdown(
        f"""
        <div style='margin-top:10px;font-size:14px;opacity:0.85;
                    border-left:4px solid #00E0FF;padding-left:10px;
                    background:linear-gradient(90deg,#f8fdff,#ffffff);
                    border-radius:8px;padding:8px 10px;'>
            🔍 Comparing <b style='color:#0072ff;'>{prev_year}</b> ➜ 
            <b style='color:#00c6ff;'>{next_year}</b> 
            • Mode: <b>{comparison_type}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Store in session for use later ---
    st.session_state["comparison_filter"] = {
        "prev_year": prev_year,
        "next_year": next_year,
        "comparison_type": comparison_type
    }

    # ===============================================================
    # 🔄 FETCH DATA FROM API
    # ===============================================================
    with st.spinner(f"📡 Fetching category data for {prev_year} and {next_year} from Vahan API..."):
        cat_prev = fetch_json(f"vahandashboard/categoriesdonutchart?year={prev_year}", desc=f"Category {prev_year}")
        cat_next = fetch_json(f"vahandashboard/categoriesdonutchart?year={next_year}", desc=f"Category {next_year}")

    df_prev = to_df(cat_prev)
    df_next = to_df(cat_next)

    if not df_prev.empty and not df_next.empty:
        st.toast("✅ Multi-year data loaded successfully!", icon="📦")

        # ===============================================================
        # 🧩 PREPROCESSING
        # ===============================================================
        merged = df_prev.merge(df_next, on="label", suffixes=(f"_{prev_year}", f"_{next_year}"), how="outer").fillna(0)
        merged["Δ_change"] = merged[f"value_{next_year}"] - merged[f"value_{prev_year}"]
        merged["Δ_percent"] = (
            (merged["Δ_change"] / merged[f"value_{prev_year}"].replace(0, 1)) * 100
        ).round(2)

        if comparison_type == "Percentage":
            display_df = merged.sort_values("Δ_percent", ascending=False)
        else:
            display_df = merged.sort_values("Δ_change", ascending=False)

        # ===============================================================
        # 📊 CHARTS — SIDE BY SIDE & COMPARISON VIEW
        # ===============================================================
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f"#### 📘 {prev_year} Category Distribution")
            bar_from_df(df_prev, title=f"{prev_year} — Category Share")

        with col2:
            st.markdown(f"#### 📗 {next_year} Category Distribution")
            bar_from_df(df_next, title=f"{next_year} — Category Share")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🔁 Year-over-Year Comparison (Δ Change)")

        try:
            import plotly.express as px
            fig = px.bar(
                display_df,
                x="label",
                y="Δ_change" if comparison_type == "Absolute" else "Δ_percent",
                color="Δ_change",
                color_continuous_scale="Viridis",
                title=f"YoY Change in Category Distribution ({prev_year} → {next_year})",
                text_auto=".2s",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Comparison chart failed: {e}")
            st.dataframe(display_df)

        # ===============================================================
        # 🏆 KPI METRICS — INSIGHT SNAPSHOT
        # ===============================================================
        top_gain = display_df.iloc[display_df["Δ_change"].idxmax()]
        top_drop = display_df.iloc[display_df["Δ_change"].idxmin()]
        total_prev = df_prev["value"].sum()
        total_next = df_next["value"].sum()
        growth_rate = ((total_next - total_prev) / total_prev * 100) if total_prev else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("📈 Total Growth", f"{growth_rate:.2f}%")
        with k2:
            st.metric("🏆 Top Gainer", top_gain["label"], f"{top_gain['Δ_change']:,}")
        with k3:
            st.metric("📉 Top Drop", top_drop["label"], f"{top_drop['Δ_change']:,}")
        with k4:
            st.metric("🚘 Total Vehicles (New)", f"{total_next:,}")

        # ===============================================================
        # 🤖 AI NARRATIVE — DEEPINFRA INSIGHTS
        # ===============================================================
        if enable_ai:
            st.markdown("### 🤖 AI-Powered Insights")
            with st.expander("🔍 View AI Narrative", expanded=True):
                with st.spinner("🧠 DeepInfra AI is analyzing YoY category shifts..."):
                    sample = display_df.head(10).to_dict(orient="records")
                    system = (
                        "You are an automotive data strategist for national transport analytics. "
                        "Provide a professional comparison between two years of vehicle registration data, "
                        "focusing on which categories grew or declined and why that might be."
                    )
                    user = (
                        f"Dataset (top 10 rows): {json.dumps(sample, default=str)}. "
                        f"Compare {prev_year} vs {next_year} category trends. Summarize in 4–6 sentences."
                    )
                    ai_resp = deepinfra_chat(system, user, max_tokens=400, temperature=0.4)

                    if ai_resp.get("text"):
                        st.toast("✅ AI Summary Generated!", icon="🤖")
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
                        st.info("💤 No AI summary generated. Try again later or verify your DeepInfra key.")

    else:
        st.warning("⚠️ No category data available for selected years.")
        st.info("🔄 Try different years or check API connectivity.")

# ===============================================================
# 2️⃣ TOP MAKERS — MULTI-YEAR MAXED EDITION 🏭🚀
# ===============================================================
with st.container():
    # 🌈 Header
    st.markdown("""
    <div style="padding:14px 22px;border-left:6px solid #FF6B6B;
                background:linear-gradient(90deg,#fff5f5 0%,#ffffff 100%);
                border-radius:16px;margin-bottom:20px;
                box-shadow:0 2px 8px rgba(255,107,107,0.1);">
        <h3 style="margin:0;font-weight:700;color:#3a3a3a;">🏭 Top Vehicle Makers — Multi-Year Comparison</h3>
        <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
            Track market share evolution of top manufacturers across years.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ===============================================================
    # 🎛️ FILTER CONTROLS
    # ===============================================================
    with st.expander("⚙️ Comparison Filters", expanded=True):
        colA, colB, colC = st.columns([1,1,1.2])
        with colA:
            prev_year = st.selectbox("📅 Previous Year", [2018, 2019, 2020, 2021, 2022, 2023, 2024], index=3)
        with colB:
            next_year = st.selectbox("📅 Next / Current Year", [2019, 2020, 2021, 2022, 2023, 2024, 2025], index=5)
        with colC:
            comparison_type = st.radio("📊 Comparison Type", ["Absolute", "Percentage"], horizontal=True)

        st.markdown(
            f"<div style='margin-top:8px;font-size:14px;opacity:0.7;'>Comparing <b>{prev_year}</b> vs <b>{next_year}</b></div>",
            unsafe_allow_html=True
        )

    # ===============================================================
    # 📡 FETCH MAKER DATA
    # ===============================================================
    with st.spinner(f"🚗 Fetching Top Makers for {prev_year} & {next_year} from Vahan API..."):
        mk_prev = fetch_json(f"vahandashboard/top5Makerchart?year={prev_year}", desc=f"Top Makers {prev_year}")
        mk_next = fetch_json(f"vahandashboard/top5Makerchart?year={next_year}", desc=f"Top Makers {next_year}")

    df_prev = parse_makers(mk_prev)
    df_next = parse_makers(mk_next)

    # ===============================================================
    # 🧩 PROCESS DATA
    # ===============================================================
    if not df_prev.empty and not df_next.empty:
        st.toast("✅ Multi-year Maker data loaded successfully!", icon="📦")

        maker_col = next((c for c in ["maker", "makename", "manufacturer", "label", "name"] if c in df_prev.columns), None)
        value_col = next((c for c in ["value", "count", "total", "registeredvehiclecount", "y"] if c in df_prev.columns), None)

        if not maker_col or not value_col:
            st.warning("⚠️ Could not find maker/value columns.")
            st.dataframe(df_prev)
        else:
            df_prev.columns = [c.strip().lower() for c in df_prev.columns]
            df_next.columns = [c.strip().lower() for c in df_next.columns]

            merged = df_prev.merge(df_next, on=maker_col, suffixes=(f"_{prev_year}", f"_{next_year}"), how="outer").fillna(0)
            merged["Δ_change"] = merged[f"{value_col}_{next_year}"] - merged[f"{value_col}_{prev_year}"]
            merged["Δ_percent"] = ((merged["Δ_change"] / merged[f"{value_col}_{prev_year}"].replace(0, 1)) * 100).round(2)

            if comparison_type == "Percentage":
                display_df = merged.sort_values("Δ_percent", ascending=False)
            else:
                display_df = merged.sort_values("Δ_change", ascending=False)

            # ===============================================================
            # 📊 DUAL VIEW — BAR & DONUT
            # ===============================================================
            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown(f"#### 📘 {prev_year} — Top Makers")
                bar_from_df(df_prev.rename(columns={maker_col: "label", value_col: "value"}), title=f"{prev_year} Maker Share")

            with col2:
                st.markdown(f"#### 📗 {next_year} — Top Makers")
                pie_from_df(df_next.rename(columns={maker_col: "label", value_col: "value"}), title=f"{next_year} Maker Share", donut=True)

            # ===============================================================
            # 📈 YoY COMPARISON CHART
            # ===============================================================
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 🔁 Year-over-Year Maker Performance")

            try:
                import plotly.express as px
                fig = px.bar(
                    display_df,
                    x=maker_col,
                    y="Δ_change" if comparison_type == "Absolute" else "Δ_percent",
                    color="Δ_change",
                    color_continuous_scale="Sunset",
                    text_auto=".2s",
                    title=f"YoY Market Shift ({prev_year} → {next_year})",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Comparison chart failed: {e}")
                st.dataframe(display_df)

            # ===============================================================
            # 🏆 KPI SNAPSHOT
            # ===============================================================
            total_prev = df_prev[value_col].sum()
            total_next = df_next[value_col].sum()
            growth_rate = ((total_next - total_prev) / total_prev * 100) if total_prev else 0
            top_gain = display_df.iloc[display_df["Δ_change"].idxmax()]
            top_drop = display_df.iloc[display_df["Δ_change"].idxmin()]

            st.markdown("<hr>", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("📈 Total Market Growth", f"{growth_rate:.2f}%")
            with k2:
                st.metric("🏆 Top Gainer", top_gain[maker_col], f"{top_gain['Δ_change']:,}")
            with k3:
                st.metric("📉 Top Decliner", top_drop[maker_col], f"{top_drop['Δ_change']:,}")
            with k4:
                st.metric("🚗 Total Vehicles (New)", f"{total_next:,}")

            st.markdown(f"""
            <div style="margin-top:10px;padding:14px 16px;
                        background:linear-gradient(90deg,#ffecec,#fffafa);
                        border:1px solid #ffc9c9;border-radius:12px;
                        box-shadow:inset 0 0 8px rgba(255,107,107,0.15);">
                <b>🔥 Insight:</b> <span style="color:#333;">{top_gain[maker_col]}</span> shows the strongest YoY growth,
                gaining <b>{top_gain['Δ_change']:,}</b> registrations, while
                <span style="color:#555;">{top_drop[maker_col]}</span> faced a decline.
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

            # ===============================================================
            # 🤖 AI MARKET INSIGHTS — DeepInfra
            # ===============================================================
            if enable_ai:
                st.markdown("### 🤖 AI-Powered Market Analysis")
                with st.expander("🔍 View AI Summary", expanded=True):
                    with st.spinner("🧠 DeepInfra analyzing manufacturer market shift..."):
                        try:
                            sample = display_df[[maker_col, f"{value_col}_{prev_year}", f"{value_col}_{next_year}", "Δ_change", "Δ_percent"]].head(10).to_dict(orient='records')
                            system = (
                                "You are a senior automotive market analyst. "
                                "Analyze manufacturer performance between two years of registration data, "
                                "highlight market leaders, growth trends, and strategic insights."
                            )
                            user = (
                                f"Dataset (top 10 rows): {json.dumps(sample, default=str)}. "
                                f"Compare {prev_year} and {next_year} market shifts. "
                                "Summarize insights in 4–6 sentences focusing on top performers and trends."
                            )

                            ai_resp = deepinfra_chat(system, user, max_tokens=400, temperature=0.45)

                            if ai_resp.get("text"):
                                st.toast("✅ AI Market Summary Ready!", icon="🤖")
                                st.markdown(f"""
                                <div style="margin-top:10px;padding:16px 18px;
                                            background:linear-gradient(90deg,#fff9f9,#fffafa);
                                            border-left:4px solid #FF6B6B;border-radius:12px;">
                                    <b>AI Market Summary:</b>
                                    <p style="margin-top:6px;font-size:15px;color:#333;">
                                        {ai_resp["text"]}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.snow()
                            else:
                                st.info("💤 No AI summary generated. Try re-running or check DeepInfra key.")
                        except Exception as e:
                            st.error(f"AI generation error: {e}")
    else:
        st.warning("⚠️ No maker data returned from the Vahan API.")
        st.info("🔄 Please refresh or check API configuration.")


# # ===============================================================
# # 🚀 CATEGORY & MAKER DISTRIBUTION — MAXED + COMPARATIVE EDITION
# # ===============================================================
# import io
# import json
# import numpy as np
# import pandas as pd
# import streamlit as st
# from datetime import datetime

# # ---------------------------------------------------------------
# # ⚙️ Core Safe Utilities
# # ---------------------------------------------------------------
# def safe_to_df(obj):
#     try:
#         if isinstance(obj, pd.DataFrame):
#             return obj
#         if isinstance(obj, (list, dict)):
#             return pd.json_normalize(obj)
#         return pd.DataFrame()
#     except Exception as e:
#         st.warning(f"⚠️ Data conversion failed: {e}")
#         return pd.DataFrame()

# def safe_col(df, options):
#     if df is None or df.empty:
#         return None
#     cols = [c.strip().lower() for c in df.columns]
#     for opt in options:
#         if opt.lower() in cols:
#             return df.columns[cols.index(opt.lower())]
#     return None

# def normalize_df(df):
#     if df.empty:
#         return df
#     df.columns = [c.strip().lower() for c in df.columns]
#     label = safe_col(df, ["label", "category", "makename", "manufacturer", "type", "name"])
#     value = safe_col(df, ["value", "count", "total", "registeredvehiclecount", "y"])
#     period = safe_col(df, ["date", "month", "year", "period", "time", "updateddate"])
#     if label and value:
#         df = df.rename(columns={label: "label", value: "value"})
#     if period:
#         df = df.rename(columns={period: "period"})
#         df["period"] = pd.to_datetime(df["period"], errors="coerce")
#     return df

# def ai_summary_block(df, role_desc, user_prompt):
#     try:
#         if not enable_ai:
#             return
#         st.markdown("### 🤖 AI-Powered Insights")
#         with st.expander("🔍 AI Narrative", expanded=True):
#             with st.spinner("🧠 DeepInfra analyzing data..."):
#                 sample = df.head(12).to_dict(orient="records")
#                 system = (
#                     f"You are a senior analytics expert specializing in {role_desc}. "
#                     "Write clear, comparative insights (YoY, MoM, QoQ, daily) in 3–6 sentences."
#                 )
#                 user = user_prompt + f"\nDataset sample: {json.dumps(sample, default=str)}"
#                 ai_resp = deepinfra_chat(system, user, max_tokens=380, temperature=0.45)
#                 text = ai_resp.get("text", "").strip()
#                 if text:
#                     st.toast("✅ AI Insight Ready!", icon="🤖")
#                     st.markdown(f"""
#                     <div style="margin-top:10px;padding:16px 18px;
#                                 background:linear-gradient(90deg,#fafaff,#f5f7ff);
#                                 border-left:4px solid #6C63FF;border-radius:12px;">
#                         <b>AI Summary:</b>
#                         <p style="margin-top:6px;font-size:15px;color:#333;">{text}</p>
#                     </div>
#                     """, unsafe_allow_html=True)
#                     st.snow()
#     except Exception as e:
#         st.error(f"AI generation error: {e}")

# # ---------------------------------------------------------------
# # 📊 Period Comparison Utilities
# # ---------------------------------------------------------------
# def compute_comparisons(df, value_col="value", period_col="period"):
#     """Add daily, monthly, quarterly, yearly comparisons."""
#     try:
#         if period_col not in df.columns:
#             return df
#         df = df.sort_values(period_col)
#         df["Year"] = df[period_col].dt.year
#         df["Month"] = df[period_col].dt.month
#         df["Quarter"] = df[period_col].dt.to_period("Q").astype(str)
#         df["Day"] = df[period_col].dt.date

#         comparisons = {}
#         for freq, label in [("D", "Daily"), ("M", "Monthly"), ("Q", "Quarterly"), ("Y", "Yearly")]:
#             period_df = df.groupby(pd.Grouper(key=period_col, freq=freq))[value_col].sum().reset_index()
#             if len(period_df) > 1:
#                 prev, curr = period_df.iloc[-2][value_col], period_df.iloc[-1][value_col]
#                 change = round(((curr - prev) / prev * 100), 2) if prev else 0
#                 comparisons[label] = {"prev": prev, "curr": curr, "change": change}
#         return comparisons
#     except Exception as e:
#         st.warning(f"Comparison failed: {e}")
#         return {}

# # ===============================================================
# # 1️⃣ CATEGORY DISTRIBUTION
# # ===============================================================
# with st.container():
#     st.markdown("""
#     <div style="padding:14px 22px;border-left:6px solid #6C63FF;
#                 background:linear-gradient(90deg,#f3f1ff 0%,#ffffff 100%);
#                 border-radius:16px;margin-bottom:20px;">
#         <h3 style="margin:0;font-weight:700;color:#3a3a3a;">📊 Category Distribution</h3>
#         <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
#             Comparative breakdown and period-over-period trends of registered vehicles by category.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

#     with st.spinner("📡 Fetching Category Distribution..."):
#         cat_json = fetch_json("vahandashboard/categoriesdonutchart", desc="Category Distribution")
#     df_cat = normalize_df(safe_to_df(cat_json))

#     if df_cat.empty:
#         st.warning("⚠️ No category data returned.")
#         st.stop()

#     st.toast("✅ Category Data Loaded", icon="📦")

#     # ---- Charts ----
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("#### 📈 Bar View")
#         bar_from_df(df_cat, title="Category Distribution (Bar)")
#     with col2:
#         st.markdown("#### 🍩 Donut View")
#         pie_from_df(df_cat, title="Category Distribution (Donut)", donut=True)

#     # ---- KPI Snapshot ----
#     total = df_cat["value"].sum()
#     top_cat = df_cat.loc[df_cat["value"].idxmax(), "label"]
#     pct = round(df_cat["value"].max() / total * 100, 2)
#     k1, k2, k3 = st.columns(3)
#     k1.metric("🏆 Top Category", top_cat)
#     k2.metric("📊 Share of Total", f"{pct}%")
#     k3.metric("🚘 Total Registrations", f"{total:,}")

#     # ---- Period Comparisons ----
#     comps = compute_comparisons(df_cat)
#     if comps:
#         st.markdown("### 📅 Period Comparisons")
#         comp_df = pd.DataFrame([
#             {"Period": p, "Prev": v["prev"], "Curr": v["curr"], "Change (%)": v["change"]}
#             for p, v in comps.items()
#         ])
#         st.dataframe(comp_df, use_container_width=True)
#         for p, v in comps.items():
#             icon = "📈" if v["change"] >= 0 else "📉"
#             st.metric(f"{icon} {p} Change", f"{v['change']}%", f"{v['curr'] - v['prev']:+,.0f}")

#     # ---- Insight ----
#     st.markdown(f"""
#     <div style="margin-top:10px;padding:14px 16px;
#                 background:linear-gradient(90deg,#e7e2ff,#f7f5ff);
#                 border:1px solid #d4cfff;border-radius:12px;">
#         <b>🏅 Insight:</b> <span style="color:#333;">{top_cat}</span> leads with <b>{pct}%</b> share.
#     </div>
#     """, unsafe_allow_html=True)
#     st.balloons()

#     ai_summary_block(
#         df_cat,
#         "vehicle category trend analysis",
#         "Provide YoY, MoM, and QoQ comparative insights for the category distribution and one key takeaway."
#     )

# # ===============================================================
# # 2️⃣ TOP MAKERS
# # ===============================================================
# with st.container():
#     st.markdown("""
#     <div style="padding:14px 22px;border-left:6px solid #FF6B6B;
#                 background:linear-gradient(90deg,#fff5f5 0%,#ffffff 100%);
#                 border-radius:16px;margin-bottom:20px;">
#         <h3 style="margin:0;font-weight:700;color:#3a3a3a;">🏭 Top Vehicle Makers</h3>
#         <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
#             Comparative and temporal trends of leading manufacturers in India.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)

#     with st.spinner("🚗 Fetching Top Makers..."):
#         mk_json = fetch_json("vahandashboard/top5Makerchart", desc="Top Makers")
#     df_mk = normalize_df(safe_to_df(mk_json))

#     if df_mk.empty:
#         st.warning("⚠️ No maker data returned.")
#         st.stop()

#     st.toast("✅ Maker Data Loaded", icon="📦")

#     # ---- Charts ----
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("#### 📊 Bar View")
#         bar_from_df(df_mk, title="Top Makers (Bar)")
#     with col2:
#         st.markdown("#### 🍩 Donut View")
#         pie_from_df(df_mk, title="Top Makers (Donut)", donut=True)

#     # ---- KPI Snapshot ----
#     total = df_mk["value"].sum()
#     top_maker = df_mk.loc[df_mk["value"].idxmax(), "label"]
#     pct = round(df_mk["value"].max() / total * 100, 2)
#     k1, k2, k3 = st.columns(3)
#     k1.metric("🏆 Leading Maker", top_maker)
#     k2.metric("📈 Market Share", f"{pct}%")
#     k3.metric("🚘 Total Registrations", f"{total:,}")

#     # ---- Comparative Analysis ----
#     comps = compute_comparisons(df_mk)
#     if comps:
#         st.markdown("### 📅 Period Comparisons")
#         comp_df = pd.DataFrame([
#             {"Period": p, "Prev": v["prev"], "Curr": v["curr"], "Change (%)": v["change"]}
#             for p, v in comps.items()
#         ])
#         st.dataframe(comp_df, use_container_width=True)
#         for p, v in comps.items():
#             icon = "📈" if v["change"] >= 0 else "📉"
#             st.metric(f"{icon} {p} Change", f"{v['change']}%", f"{v['curr'] - v['prev']:+,.0f}")

#     # ---- Insight ----
#     st.markdown(f"""
#     <div style="margin-top:10px;padding:14px 16px;
#                 background:linear-gradient(90deg,#ffecec,#fffafa);
#                 border:1px solid #ffc9c9;border-radius:12px;">
#         <b>🔥 Insight:</b> <span style="color:#333;">{top_maker}</span> leads with <b>{pct}%</b> market share.
#     </div>
#     """, unsafe_allow_html=True)
#     st.balloons()

#     ai_summary_block(
#         df_mk,
#         "automotive manufacturer competition",
#         "Compare YoY, MoM, and QoQ growth patterns across top manufacturers, and summarize one strategic takeaway."
#     )

# # # =============================================
# # 3️⃣ Registration Trends + YoY/QoQ/MoM/DoD + AI + Forecast (MAXED) 🚀
# # =============================================

# import pandas as pd
# import numpy as np
# import plotly.express as px
# from datetime import timedelta

# # --- Theme CSS (animations & layout)
# st.markdown("""
# <style>
# .trend-card { transition: all 0.2s ease; border-radius:12px; }
# .trend-card:hover { transform: translateY(-4px); box-shadow: 0 8px 28px rgba(0,0,0,0.12); }
# .trend-metric { padding:10px;border-radius:10px;background:linear-gradient(90deg,#ffffff,#f7fbff); }
# .trend-table th, .trend-table td { padding:6px 8px !important; font-size:14px !important; }
# .up { color:#16a34a; font-weight:600; } .down { color:#dc2626; font-weight:600; }
# .small-muted { color:#6b7280;font-size:13px; }
# </style>
# """, unsafe_allow_html=True)


# # =========================
# # 🔢 Forecasting Fallbacks
# # =========================
# def forecast_trend(df, periods=6):
#     """Multi-fallback forecast: Prophet → Linear → Moving Avg"""
#     if df is None or df.empty or "date" not in df.columns or "value" not in df.columns:
#         return pd.DataFrame()
#     df_fc = df.copy().sort_values("date").reset_index(drop=True)
#     df_fc["date"] = pd.to_datetime(df_fc["date"])

#     # Prophet (monthly)
#     try:
#         from prophet import Prophet
#         tmp = df_fc.rename(columns={"date": "ds", "value": "y"})
#         m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
#         m.fit(tmp)
#         future = m.make_future_dataframe(periods=periods, freq="M")
#         fc = m.predict(future)
#         out = fc[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "value"})
#         out["forecast"] = out["date"] > df_fc["date"].max()
#         return out
#     except Exception:
#         pass

#     # Linear fallback
#     try:
#         from sklearn.linear_model import LinearRegression
#         X = np.arange(len(df_fc)).reshape(-1, 1)
#         y = df_fc["value"].values
#         model = LinearRegression().fit(X, y)
#         future_X = np.arange(len(df_fc), len(df_fc) + periods).reshape(-1, 1)
#         y_pred = model.predict(future_X)
#         last_date = df_fc["date"].max()
#         future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(), periods=periods, freq="M")
#         hist = df_fc.assign(forecast=False)
#         future_df = pd.DataFrame({"date": future_dates, "value": y_pred, "forecast": True})
#         return pd.concat([hist, future_df], ignore_index=True)
#     except Exception:
#         pass

#     # Moving average fallback
#     avg_growth = df_fc["value"].pct_change().mean()
#     last_val = df_fc["value"].iloc[-1]
#     dates = pd.date_range(df_fc["date"].max() + pd.offsets.MonthEnd(), periods=periods, freq="M")
#     vals = [last_val * (1 + (avg_growth or 0)) ** (i + 1) for i in range(periods)]
#     future_df = pd.DataFrame({"date": dates, "value": vals, "forecast": True})
#     return pd.concat([df_fc.assign(forecast=False), future_df])


# # =========================
# # 📊 Unified Comparison Computation
# # =========================
# def compute_all_maxed(df, date_col="date", value_col="value"):
#     df = df.copy()
#     df[date_col] = pd.to_datetime(df[date_col])
#     df = df.sort_values(date_col)
#     df.set_index(date_col, inplace=True)

#     daily = df[value_col].resample("D").mean()
#     monthly = df[value_col].resample("M").sum()
#     quarterly = df[value_col].resample("Q").sum()
#     yearly = df[value_col].resample("Y").sum()

#     data = {
#         "DoD%": daily.pct_change().iloc[-1] * 100 if len(daily) > 1 else None,
#         "MoM%": monthly.pct_change().iloc[-1] * 100 if len(monthly) > 1 else None,
#         "QoQ%": quarterly.pct_change().iloc[-1] * 100 if len(quarterly) > 1 else None,
#         "YoY%": yearly.pct_change().iloc[-1] * 100 if len(yearly) > 1 else None,
#     }
#     df_out = pd.DataFrame(list(data.items()), columns=["Metric", "Change%"])
#     df_out["Direction"] = df_out["Change%"].apply(lambda x: "↑" if x and x > 0 else ("↓" if x and x < 0 else "→"))
#     df_out["Color"] = df_out["Change%"].apply(lambda x: "up" if x and x > 0 else ("down" if x and x < 0 else ""))
#     return df_out


# def display_all_maxed(df):
#     if df.empty:
#         st.info("No comparison data available.")
#         return
#     html = "<table class='trend-table'><tr><th>Metric</th><th>Change</th></tr>"
#     for _, row in df.iterrows():
#         val = "N/A" if pd.isna(row["Change%"]) else f"{row['Change%']:.2f}% {row['Direction']}"
#         html += f"<tr><td>{row['Metric']}</td><td class='{row['Color']}'>{val}</td></tr>"
#     html += "</table>"
#     st.markdown(html, unsafe_allow_html=True)


# # =========================
# # 🔄 UI & Data Fetch
# # =========================
# st.markdown("<div class='trend-card' style='padding:12px 14px;margin-bottom:10px;'>", unsafe_allow_html=True)
# st.markdown("<h3 style='margin:0 0 6px;'>📈 Registration Trends — All Comparisons MAXED</h3>", unsafe_allow_html=True)
# st.markdown("<div class='small-muted'>Full daily, monthly, quarterly, yearly analysis, with forecast & AI insights.</div>", unsafe_allow_html=True)
# st.markdown("</div>", unsafe_allow_html=True)

# c1, c2, c3, c4 = st.columns([1,1,1,1])
# with c1:
#     horizon = st.selectbox("Forecast horizon (months)", [3,6,12], index=1)
# with c2:
#     show_daily = st.checkbox("Show Daily Breakdown", value=True)
# with c3:
#     show_ai = st.checkbox("Enable AI Insight", value=enable_ai)
# with c4:
#     refresh = st.button("🔄 Refresh Trends")

# if refresh:
#     st.toast("Refreshing trend data...", icon="🔁")

# # Fetch Data
# with st.spinner("Fetching registration trend data..."):
#     tr_json = fetch_json("vahandashboard/vahanyearwiseregistrationtrend", desc="Registration Trend")
#     df_trend = normalize_trend(tr_json) if tr_json else pd.DataFrame(columns=["date","value"])

# if df_trend is not None and not df_trend.empty:
#     df_trend["date"] = pd.to_datetime(df_trend["date"])
#     df_trend = df_trend.sort_values("date")

#     total = int(df_trend["value"].sum())
#     st.metric("🚘 Total Registrations", f"{total:,}")

#     # ========== All Comparisons ==========
#     st.markdown("### 🧮 Unified Comparison Summary (All MAXED)")
#     comp_df = compute_all_maxed(df_trend)
#     display_all_maxed(comp_df)

#     # ========== Forecast ==========
#     fc_df = forecast_trend(df_trend, periods=horizon)
#     if not fc_df.empty:
#         fc_df["Type"] = np.where(fc_df.get("forecast", False), "Forecast", "Actual")
#         fig = px.line(fc_df, x="date", y="value", color="Type", markers=True,
#                       title="Actual vs Forecasted Registrations")
#         st.plotly_chart(fig, use_container_width=True)

#     # ========== Daily & Monthly Trend Views ==========
#     if show_daily:
#         st.markdown("### 🗓 Daily & Monthly Trend Views")
#         daily = df_trend.set_index("date")["value"].resample("D").interpolate().reset_index()
#         daily["MA7"] = daily["value"].rolling(7).mean()
#         monthly = df_trend.set_index("date")["value"].resample("M").sum().reset_index()
#         col1, col2 = st.columns(2)
#         with col1:
#             st.plotly_chart(px.line(daily, x="date", y=["value","MA7"], title="Daily Trend (7-day Avg)"), use_container_width=True)
#         with col2:
#             st.plotly_chart(px.bar(monthly, x="date", y="value", title="Monthly Totals"), use_container_width=True)

#     # ========== AI Summary ==========
#     if show_ai:
#         st.markdown("### 🤖 AI Narrative — Trend & Forecast Insight")
#         with st.spinner("Generating AI insight..."):
#             system = (
#                 "You are a senior data analyst. Summarize this registration trend across daily, monthly, "
#                 "quarterly, and yearly comparisons. Highlight dominant growth patterns, anomalies, "
#                 "and give one strategic recommendation."
#             )
#             sample = df_trend.tail(12).to_dict(orient="records")
#             user = f"Here is the recent trend data: {json.dumps(sample, default=str)}"
#             ai_resp = deepinfra_chat(system, user, max_tokens=350, temperature=0.3)
#             if ai_resp.get("text"):
#                 st.markdown(f"""
#                 <div style='padding:12px 16px;border-left:4px solid #007BFF;
#                             background:linear-gradient(90deg,#f7f9ff,#ffffff);
#                             border-radius:12px;'>
#                     <b>AI Summary:</b><br>{ai_resp['text']}
#                 </div>""", unsafe_allow_html=True)
#                 st.snow()
#             else:
#                 st.info("AI summary not available.")

#     st.balloons()
# else:
#     st.warning("⚠️ No registration trend data available.")

# # vahan_duration_maxed.py
# # MAXED Duration fetch + UI utilities (drop into your Streamlit app)
# import os
# import time
# import random
# import logging
# import pickle
# import hashlib
# from datetime import datetime
# from zoneinfo import ZoneInfo
# from typing import Optional, Any, Dict

# import streamlit as st
# import pandas as pd
# import altair as alt
# import requests

# # -------------------------
# # Logging & IST helper
# # -------------------------
# LOG = logging.getLogger("vahan_maxed")
# if not LOG.handlers:
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# def ist_now(fmt: str = "%Y-%m-%d %I:%M:%S %p") -> str:
#     return datetime.now(ZoneInfo("Asia/Kolkata")).strftime(fmt)

# def log_ist(msg: str, level: str = "info"):
#     getattr(LOG, level)(f"[IST {ist_now()}] {msg}")

# # -------------------------
# # CONFIG
# # -------------------------
# BASE = os.getenv("VAHAN_API_BASE", "https://analytics.parivahan.gov.in/analytics/publicdashboard")
# DEFAULT_TIMEOUT = int(os.getenv("VAHAN_TIMEOUT", "30"))
# MAX_RETRIES = int(os.getenv("VAHAN_MAX_RETRIES", "5"))
# BACKOFF_FACTOR = float(os.getenv("VAHAN_BACKOFF", "1.2"))
# CACHE_DIR = os.getenv("VAHAN_CACHE_DIR", "vahan_cache")
# CACHE_TTL = int(os.getenv("VAHAN_CACHE_TTL_SEC", str(60 * 60)))  # default 1 hour
# ROTATING_UAS = [
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
#     "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
#     "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0",
# ]
# os.makedirs(CACHE_DIR, exist_ok=True)

# # -------------------------
# # Simple file cache helpers
# # -------------------------
# def _cache_path(url: str) -> str:
#     key = hashlib.sha256(url.encode("utf-8")).hexdigest()
#     return os.path.join(CACHE_DIR, f"{key}.pkl")

# def load_cache(url: str) -> Optional[Any]:
#     p = _cache_path(url)
#     try:
#         if not os.path.exists(p):
#             return None
#         with open(p, "rb") as f:
#             ts, data = pickle.load(f)
#         if time.time() - ts > CACHE_TTL:
#             try: os.remove(p)
#             except Exception: pass
#             return None
#         log_ist(f"Cache hit: {url}")
#         return data
#     except Exception as e:
#         log_ist(f"Cache load failed: {e}", "warning")
#         return None

# def save_cache(url: str, data: Any) -> None:
#     if data is None:
#         return
#     p = _cache_path(url)
#     try:
#         with open(p, "wb") as f:
#             pickle.dump((time.time(), data), f)
#         log_ist(f"Saved to cache: {url}")
#     except Exception as e:
#         log_ist(f"Cache save failed: {e}", "warning")

# # -------------------------
# # Token bucket rate limiter
# # -------------------------
# class TokenBucket:
#     def __init__(self, capacity: float, rate: float):
#         self.capacity = float(capacity)
#         self.rate = float(rate)
#         self._tokens = float(capacity)
#         self._last = time.time()
#     def _refill(self):
#         now = time.time()
#         self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
#         self._last = now
#     def consume(self, tokens: float = 1.0) -> bool:
#         self._refill()
#         if self._tokens >= tokens:
#             self._tokens -= tokens
#             return True
#         return False
#     def wait_for_token(self, tokens: float = 1.0, timeout: int = 15) -> bool:
#         start = time.time()
#         while not self.consume(tokens):
#             if time.time() - start > timeout:
#                 return False
#             time.sleep(max(0.02, 0.1 * random.random()))
#         return True

# # default small rate so we don't slam upstream
# _bucket = TokenBucket(capacity=10.0, rate=1.0)

# # -------------------------
# # Robust fetch (safe_get)
# # -------------------------
# def safe_get(path: str, params: Optional[Dict[str,Any]] = None, use_cache: bool = True, timeout: int = DEFAULT_TIMEOUT) -> Optional[Any]:
#     params = params or {}
#     # build canonical URL for caching/diagnostics
#     try:
#         from urllib.parse import urlencode
#         query = urlencode(params, doseq=True)
#         url = f"{BASE.rstrip('/')}/{path.lstrip('/')}?{query}"
#     except Exception as e:
#         log_ist(f"URL build failed: {e}", "error")
#         return None

#     # cache fast-path
#     if use_cache:
#         cached = load_cache(url)
#         if cached is not None:
#             return cached

#     # token-bucket wait
#     if not _bucket.wait_for_token(timeout=10):
#         log_ist(f"Rate limiter timeout for {url}", "warning")
#         return None

#     attempt = 0
#     while attempt < MAX_RETRIES:
#         attempt += 1
#         ua = random.choice(ROTATING_UAS)
#         headers = {
#             "User-Agent": ua,
#             "Accept": "application/json, text/plain, */*",
#             "Accept-Language": "en-US,en;q=0.9",
#             "Referer": "https://analytics.parivahan.gov.in",
#             "X-Requested-With": "XMLHttpRequest",
#         }
#         try:
#             log_ist(f"Fetching ({attempt}/{MAX_RETRIES}): {url}")
#             resp = requests.get(url, headers=headers, params=None, timeout=timeout)
#             status = getattr(resp, "status_code", None)

#             if status == 200:
#                 try:
#                     data = resp.json()
#                 except Exception:
#                     data = {"raw_text": resp.text[:8000]}
#                 if use_cache and data:
#                     save_cache(url, data)
#                 return data

#             if status == 400:
#                 # Bad request - don't retry
#                 log_ist(f"400 Bad Request for {url}", "error")
#                 log_ist(f"Snippet: {resp.text[:800]}", "error")
#                 return None

#             if status == 404:
#                 log_ist(f"404 Not Found for {url}", "error")
#                 return None

#             if status == 429:
#                 wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.uniform(0.5, 2.0)
#                 log_ist(f"429 Rate limited. Sleeping {wait:.1f}s then retrying.", "warning")
#                 time.sleep(wait)
#                 continue

#             if status and status >= 500:
#                 wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.uniform(0.3, 1.3)
#                 log_ist(f"Server error {status}. Sleeping {wait:.1f}s then retrying.", "warning")
#                 time.sleep(wait)
#                 continue

#             # otherwise treat as error
#             log_ist(f"Unexpected HTTP {status} for {url}. Snippet: {resp.text[:400]}", "error")
#             return None

#         except requests.Timeout:
#             wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
#             log_ist(f"Timeout on attempt {attempt}. Sleeping {wait:.1f}s", "warning")
#             time.sleep(wait)
#             continue
#         except requests.ConnectionError as e:
#             wait = BACKOFF_FACTOR * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
#             log_ist(f"Connection error: {e}. Sleeping {wait:.1f}s", "warning")
#             time.sleep(wait)
#             continue
#         except Exception as e:
#             log_ist(f"Unexpected fetch error: {e}", "error")
#             return None

#     log_ist(f"Max retries reached for {url}", "error")
#     return None

# # convenience wrapper matching your earlier API
# def fetch_json(path: str, params: Optional[Dict[str,Any]] = None, desc: str = "", use_cache: bool = True):
#     res = safe_get(path, params=params or {}, use_cache=use_cache)
#     return res

# # -------------------------
# # Default params_common and helpful randomization
# # -------------------------
# params_common = {
#     "fromYear": int(os.getenv("VAHAN_FROM_YEAR", "2024")),
#     "toYear": int(os.getenv("VAHAN_TO_YEAR", "2025")),
#     "stateCode": "",
#     "rtoCode": 0,
#     "vehicleClasses": "",
#     "vehicleMakers": "",
#     "timePeriod": "All Time",
#     "fitnessCheck": "All",
#     "vehicleType": ""
# }

# def params_randomize(base: Dict[str,Any]) -> Dict[str,Any]:
#     p = base.copy()
#     # sometimes required endpoints expect different param names -> include aliases
#     p.setdefault("state_cd", p.get("stateCode", ""))
#     # safe random toggles
#     if random.random() < 0.08:  # small chance to try ALL
#         p["stateCode"] = "ALL"
#         p["rtoCode"] = "ALL"
#     return p

# # -------------------------
# # Parse duration table: accept many shapes
# # -------------------------
# def parse_duration_table(json_data: Any) -> pd.DataFrame:
#     try:
#         if not json_data:
#             return pd.DataFrame()
#         # If container with data key
#         if isinstance(json_data, dict):
#             # common patterns
#             for k in ("data", "result", "rows", "response"):
#                 if k in json_data and json_data[k]:
#                     json_data = json_data[k]
#                     break
#         # if list of dicts -> normalize
#         if isinstance(json_data, list):
#             df = pd.json_normalize(json_data)
#         elif isinstance(json_data, dict):
#             df = pd.json_normalize([json_data])
#         else:
#             return pd.DataFrame()

#         # pick label and value flexibly
#         col_lower = [c.lower() for c in df.columns]
#         label_col = None
#         value_col = None
#         for possibility in ["label", "category", "name", "period", "month", "year"]:
#             if possibility in col_lower:
#                 label_col = df.columns[col_lower.index(possibility)]
#                 break
#         for possibility in ["value", "count", "total", "registeredvehiclecount", "y"]:
#             if possibility in col_lower:
#                 value_col = df.columns[col_lower.index(possibility)]
#                 break
#         # fallback heuristics
#         if label_col is None and len(df.columns) >= 1:
#             label_col = df.columns[0]
#         if value_col is None and len(df.columns) >= 2:
#             value_col = df.columns[1]

#         if label_col is None or value_col is None:
#             # try numeric columns as value
#             numeric_cols = df.select_dtypes(include="number").columns.tolist()
#             if numeric_cols:
#                 value_col = numeric_cols[0]
#             else:
#                 # no usable columns
#                 return pd.DataFrame()

#         out = df[[label_col, value_col]].copy()
#         out.columns = ["label", "value"]
#         # coerce numeric
#         out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0)
#         # ensure label is str
#         out["label"] = out["label"].astype(str)
#         return out
#     except Exception as e:
#         log_ist(f"parse_duration_table error: {e}", "warning")
#         return pd.DataFrame()

# # -------------------------
# # Charts helpers (Altair + streamlit)
# # -------------------------
# def bar_from_df(df: pd.DataFrame, title: str = "Bar Chart"):
#     if df is None or df.empty:
#         st.info("No data to show.")
#         return
#     df_plot = df.copy()
#     df_plot = df_plot.sort_values("value", ascending=False).head(50)
#     chart = (
#         alt.Chart(df_plot)
#         .mark_bar()
#         .encode(x=alt.X("value:Q", title="Value"), y=alt.Y("label:N", sort='-x', title=None), tooltip=["label", "value"])
#         .properties(height=400, title=title)
#     )
#     st.altair_chart(chart, use_container_width=True)

# def pie_from_df(df: pd.DataFrame, title: str = "Pie Chart", donut: bool = False):
#     if df is None or df.empty:
#         st.info("No data to show.")
#         return
#     df_plot = df.copy()
#     df_plot = df_plot.groupby("label", as_index=False)["value"].sum().sort_values("value", ascending=False).head(12)
#     chart = (
#         alt.Chart(df_plot)
#         .encode(theta=alt.Theta("value:Q"), color=alt.Color("label:N"), tooltip=["label","value"])
#     )
#     if donut:
#         chart = chart.mark_arc(innerRadius=50).properties(height=360, title=title)
#     else:
#         chart = chart.mark_arc().properties(height=360, title=title)
#     st.altair_chart(chart, use_container_width=True)

# # -------------------------
# # AI hook (optional): DeepInfra / echo stub
# # -------------------------
# def deepinfra_chat(system: str, user: str, max_tokens: int = 300, temperature: float = 0.4) -> Dict[str,Any]:
#     key = os.getenv("DEEPINFRA_KEY") or os.getenv("DEEPINFRA_API_KEY")
#     if not key:
#         # no key configured — return a safe placeholder
#         return {"text": ""}
#     # If you have a DeepInfra client, integrate here.
#     # For safety (no network calls from this function), we return placeholder.
#     return {"text": ""}

# # -------------------------
# # Higher-level duration fetcher used by UI
# # -------------------------
# def fetch_duration_growth(calendar_type: int, label: str, color: str, emoji: str) -> pd.DataFrame:
#     """
#     calendar_type is informative; label chooses endpoint logic.
#     calendar_type mapping in your UI remains unchanged.
#     """
#     with st.spinner(f"{emoji} Fetching {label} data..."):
#         # map label -> preferred endpoint (auto-fallback to many)
#         endpoint_candidates = [
#             # most specific first
#             "vahandashboard/durationWiseRegistrationTable",
#             "vahandashboard/monthWiseRegistrationChart",
#             "vahandashboard/quarterlyRegistrationChart",
#             "vahandashboard/dailyRegistrationChart",
#             "vahandashboard/vahanyearwiseregistrationtrend",
#             # legacy/alternate names
#             "vahandashboard/registrationByDuration",
#         ]
#         # Try candidates but stop early on success
#         tried = []
#         params = params_randomize(params_common)
#         # ensure calendar alias if using durationWiseRegistrationTable
#         cal_map = {3: "ME", 2: "QE", 1: "YE", 4: "D"}  # ME=month, QE=quarter, YE=year, D=daily (some endpoints want 'ME' etc)
#         if calendar_type in cal_map:
#             params_try = {**params, "state_cd": "ALL", "reportType":"R", "calendarType": cal_map[calendar_type]}
#         else:
#             params_try = {**params}

#         for cand in endpoint_candidates:
#             tried.append(cand)
#             json_data = fetch_json(cand, params=params_try, desc=f"{label} growth", use_cache=True)
#             df = parse_duration_table(json_data)
#             if not df.empty:
#                 # render UI blocks as in your original code
#                 # Sub-header
#                 st.markdown(f"""
#                 <div style="padding:12px 18px;margin-top:10px;
#                             border-left:6px solid {color};
#                             background:linear-gradient(90deg,#fafafa,#ffffff);
#                             border-radius:12px;">
#                     <h3 style="margin:0;">{emoji} {label} Vehicle Registration Growth</h3>
#                 </div>
#                 """, unsafe_allow_html=True)
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     try:
#                         bar_from_df(df, title=f"{label} Growth (Bar)")
#                     except Exception:
#                         st.dataframe(df)
#                 with col2:
#                     try:
#                         pie_from_df(df, title=f"{label} Growth (Donut)", donut=True)
#                     except Exception:
#                         st.dataframe(df)
#                 # KPI card
#                 try:
#                     total = int(df["value"].sum())
#                     top_label = df.loc[df["value"].idxmax(), "label"]
#                     pct = round(df["value"].max() / total * 100, 2) if total else 0
#                     k1, k2, k3 = st.columns(3)
#                     k1.metric("🏆 Peak", top_label)
#                     k2.metric("📊 Share", f"{pct}%")
#                     k3.metric("🚘 Total", f"{total:,}")
#                 except Exception:
#                     pass

#                 # AI summary
#                 if os.getenv("DEEPINFRA_KEY"):
#                     with st.expander(f"🤖 AI Summary — {label} Growth", expanded=False):
#                         system = f"You are a senior analytics expert. Summarize {label} registration trends."
#                         sample = df.head(8).to_dict(orient="records")
#                         user = f"Dataset sample: {sample}"
#                         ai = deepinfra_chat(system, user, max_tokens=220)
#                         text = ai.get("text", "")
#                         if text:
#                             st.markdown(f"<div style='padding:10px;border-left:4px solid {color};'>{text}</div>", unsafe_allow_html=True)

#                 log_ist(f"{label} fetched successfully at {ist_now()}")
#                 return df

#             # if empty, try next candidate after small jitter
#             time.sleep(random.uniform(0.15, 0.45))

#         # If we reach here nothing returned valid data
#         st.warning(f"⚠️ Failed to fetch {label} — tried endpoints: {', '.join(tried)}")
#         log_ist(f"Failed to fetch {label} at {ist_now()}", "warning")
#         return pd.DataFrame()

# # -------------------------
# # Example usage: UI wiring (copy into your app layout)
# # -------------------------
# def run_durations_ui():
#     st.markdown("## 📊 Duration-wise Growth & Revenue Insights")
#     df_monthly = fetch_duration_growth(3, "Monthly", "#007bff", "📅")
#     df_quarterly = fetch_duration_growth(2, "Quarterly", "#6f42c1", "🧭")
#     df_yearly = fetch_duration_growth(1, "Yearly", "#28a745", "📆")
#     # Optionally merge and compare if any present
#     dfs = []
#     if not df_monthly.empty:
#         m = df_monthly.copy(); m["period"] = "Monthly"; dfs.append(m)
#     if not df_quarterly.empty:
#         q = df_quarterly.copy(); q["period"] = "Quarterly"; dfs.append(q)
#     if not df_yearly.empty:
#         y = df_yearly.copy(); y["period"] = "Yearly"; dfs.append(y)
#     if not dfs:
#         st.info("⚠️ No duration datasets available — skipping unified comparison.")
#         return
#     df_compare = pd.concat(dfs, ignore_index=True)
#     # pivot and show
#     pivot_df = df_compare.pivot_table(index="label", columns="period", values="value", aggfunc="sum").fillna(0)
#     st.subheader("📈 Unified Duration Comparison")
#     st.dataframe(pivot_df.style.format("{:,.0f}"))
#     # small line chart
#     try:
#         chart_df = pivot_df.reset_index().melt(id_vars="label", var_name="period", value_name="value")
#         line = alt.Chart(chart_df).mark_line(point=True).encode(x="label:N", y="value:Q", color="period:N", tooltip=["label","period","value"]).properties(height=360)
#         st.altair_chart(line, use_container_width=True)
#     except Exception as e:
#         log_ist(f"chart error: {e}", "warning")

# # Only run example UI if direct run (keeps file safe to import)
# if __name__ == "__main__":
#     st.set_page_config(layout="wide")
#     log_ist("App booting")
#     # Boot banner
#     st.markdown(f"<div style='padding:8px;border-left:4px solid #007bff'><b>Booted (IST):</b> {ist_now()}</div>", unsafe_allow_html=True)
#     run_durations_ui()

# # --------------------- Top 5 Revenue States ---------------------
# st.markdown("""
# <style>
# .rev-header {
#     background: linear-gradient(90deg, #fffbe6, #ffffff);
#     border-left: 6px solid #ffc107;
#     padding: 14px 20px;
#     border-radius: 14px;
#     margin-top: 35px;
#     animation: pulseGlow 3s infinite;
# }
# </style>

# <div class="rev-header">
#     <h2 style="margin:0;">💰 Top 5 Revenue States</h2>
#     <p style="margin:4px 0 0;color:#555;font-size:15px;">
#         Explore which states lead in total vehicle-related revenue and performance growth.
#     </p>
# </div>
# """, unsafe_allow_html=True)


# with st.spinner("Fetching Top 5 Revenue States..."):
#     top5_rev_json = fetch_json("vahandashboard/top5chartRevenueFee", desc="Top 5 Revenue States")

# df_top5_rev = parse_top5_revenue(top5_rev_json if top5_rev_json else {})

# if not df_top5_rev.empty:
#     col1, col2 = st.columns(2)
#     with col1:
#         try:
#             bar_from_df(df_top5_rev, title="Top 5 Revenue States (Bar)")
#         except Exception:
#             st.dataframe(df_top5_rev)
#     with col2:
#         try:
#             pie_from_df(df_top5_rev, title="Top 5 Revenue States (Pie)", donut=True)
#         except Exception:
#             st.dataframe(df_top5_rev)

#     # KPI summary with emoji and animation
#     try:
#         top_state = df_top5_rev.loc[df_top5_rev["value"].idxmax(), "label"]
#         top_value = df_top5_rev["value"].max()
#         total_rev = df_top5_rev["value"].sum()

#         st.markdown(f"""
#         <div style="margin-top:10px;padding:14px 18px;
#                     background:linear-gradient(90deg,#fffef5,#ffffff);
#                     border-left:5px solid #ffc107;
#                     border-radius:12px;
#                     box-shadow:0 2px 8px rgba(0,0,0,0.05);">
#             <b>🏅 Top Revenue State:</b> {top_state} — ₹{top_value:,.0f}<br>
#             <b>💵 Combined (Top 5):</b> ₹{total_rev:,.0f}
#         </div>
#         """, unsafe_allow_html=True)

#         st.snow()  # Celebration when revenue data loads
#     except Exception as e:
#         st.error(f"Revenue KPI error: {e}")

#     # AI summary — auto expanded
#     if enable_ai:
#         with st.expander("🤖 AI Summary — Revenue Insights", expanded=True):
#             with st.spinner("Generating AI revenue insights..."):
#                 system = (
#                     "You are a financial analyst summarizing state-level vehicle revenue performance in India. "
#                     "Highlight top states, major revenue gaps, and one strategy to enhance state-level revenue balance."
#                 )
#                 sample = df_top5_rev.head(10).to_dict(orient="records")
#                 user = f"Dataset: {json.dumps(sample, default=str)}"
#                 ai_resp = deepinfra_chat(system, user, max_tokens=240)
#                 if isinstance(ai_resp, dict) and "text" in ai_resp:
#                     st.markdown(f"""
#                     <div style="padding:12px 16px;margin-top:8px;
#                                 background:linear-gradient(90deg,#ffffff,#fffdf0);
#                                 border-left:4px solid #ffc107;
#                                 border-radius:10px;">
#                         {ai_resp["text"]}
#                     </div>
#                     """, unsafe_allow_html=True)
# else:
#     st.warning("⚠️ No revenue data available from Vahan API.")

# # ================================================================
# # 🌟 6️⃣ Revenue Trend + Forecast + Anomaly Detection + Clustering —  UI
# # ================================================================

# import streamlit as st
# import pandas as pd
# import altair as alt
# import json
# from datetime import datetime

# # ================================
# # 🎨 CSS Animations & Transitions
# # ================================
# st.markdown("""
# <style>
# @keyframes slideIn {
#   from {opacity: 0; transform: translateY(20px);}
#   to {opacity: 1; transform: translateY(0);}
# }
# @keyframes pulseBorder {
#   0% {box-shadow: 0 0 0px #ff5722;}
#   50% {box-shadow: 0 0 10px #ff5722;}
#   100% {box-shadow: 0 0 0px #ff5722;}
# }
# .-container {
#   background: linear-gradient(90deg,#fff7f3,#ffffff);
#   border-left: 6px solid #ff5722;
#   padding: 16px 22px;
#   border-radius: 14px;
#   margin: 20px 0 15px 0;
#   animation: pulseBorder 4s infinite;
# }
# .metric-card {
#   background: #fff;
#   border-radius: 12px;
#   padding: 12px;
#   box-shadow: 0 3px 10px rgba(0,0,0,0.05);
#   transition: 0.3s;
# }
# .metric-card:hover {
#   transform: translateY(-2px);
#   box-shadow: 0 5px 15px rgba(255,87,34,0.3);
# }
# .ai-box {
#   background: linear-gradient(90deg,#ffffff,#fff9f6);
#   border-left: 4px solid #ff5722;
#   border-radius: 10px;
#   padding: 12px 14px;
#   margin-top: 8px;
#   animation: slideIn 1s ease;
# }
# </style>
# """, unsafe_allow_html=True)


# # ======================
# # 📊 Section Header
# # ======================
# st.markdown("""
# <div class="-container">
#     <h2 style="margin:0;">💹 Revenue Trend & Advanced Analytics</h2>
#     <p style="margin:4px 0 0;color:#444;font-size:15px;">
#         Smart forecasting, anomaly detection, and AI-powered clustering insights with smooth transitions and dynamic visuals.
#     </p>
# </div>
# """, unsafe_allow_html=True)


# # ======================
# # 📈 Fetch & Visualize Revenue Trend
# # ======================
# with st.spinner("Fetching Revenue Trend..."):
#     rev_trend_json = fetch_json("vahandashboard/revenueFeeLineChart", desc="Revenue Trend")

# df_rev_trend = parse_revenue_trend(rev_trend_json if rev_trend_json else {})

# if df_rev_trend.empty:
#     st.warning("⚠️ No revenue trend data available.")
# else:
#     st.subheader("📊 Revenue Trend Comparison")
#     try:
#         chart = (
#             alt.Chart(df_rev_trend)
#             .mark_line(point=True, interpolate="monotone")
#             .encode(
#                 x=alt.X("period:O", title="Period"),
#                 y=alt.Y("value:Q", title="Revenue (₹)"),
#                 color=alt.Color("year:N", legend=alt.Legend(title="Year")),
#                 tooltip=["year", "period", "value"]
#             )
#             .properties(height=380, title="Revenue Trend Comparison")
#         )
#         st.altair_chart(chart, use_container_width=True)
#     except Exception:
#         st.dataframe(df_rev_trend)

#     # KPIs — Animated Cards
#     try:
#         total_rev = float(df_rev_trend["value"].sum())
#         avg_rev = float(df_rev_trend["value"].mean())
#         latest_rev = float(df_rev_trend["value"].iloc[-1])
#         prev_rev = float(df_rev_trend["value"].iloc[-2]) if len(df_rev_trend) > 1 else latest_rev
#         growth_pct = ((latest_rev - prev_rev) / prev_rev) * 100 if prev_rev else 0.0
#     except Exception:
#         total_rev = avg_rev = latest_rev = growth_pct = None

#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.markdown(f"<div class='metric-card'><h4>💰 Total Revenue</h4><b>₹{total_rev:,.0f}</b></div>", unsafe_allow_html=True)
#     with col2:
#         st.markdown(f"<div class='metric-card'><h4>📈 Latest Revenue</h4><b>₹{latest_rev:,.0f}</b></div>", unsafe_allow_html=True)
#     with col3:
#         st.markdown(f"<div class='metric-card'><h4>📊 Avg per Period</h4><b>₹{avg_rev:,.0f}</b></div>", unsafe_allow_html=True)
#     with col4:
#         color = "green" if growth_pct >= 0 else "red"
#         st.markdown(f"<div class='metric-card'><h4>📅 Growth %</h4><b style='color:{color};'>{growth_pct:.2f}%</b></div>", unsafe_allow_html=True)

#     if growth_pct > 5:
#         st.balloons()
#     elif growth_pct < -5:
#         st.snow()


# if enable_forecast:
#     st.markdown("### 🔮 Forecasting — Future Revenue Projection")
#     try:
#         df_trend = df_rev_trend.copy()
#         df_trend['date'] = pd.to_datetime(df_trend['period'], errors='coerce')
#         df_trend = df_trend.dropna(subset=['date'])
#         forecast_df = linear_forecast(df_trend, months=forecast_periods)
#         if not forecast_df.empty:
#             combined = pd.concat([
#                 df_trend.set_index('date')['value'],
#                 forecast_df.set_index('date')['value']
#             ])
#             st.line_chart(combined)
#             st.success("✅ Forecast generated successfully!")

#             if enable_ai:
#                 with st.spinner("🤖 Generating AI forecast commentary..."):
#                     system = "You are a forecasting analyst summarizing financial revenue predictions."
#                     sample = forecast_df.head(6).to_dict(orient="records")
#                     user = f"Forecasted values: {json.dumps(sample, default=str)}. Summarize key confidence and trends in 3 sentences."
#                     ai_resp = deepinfra_chat(system, user, max_tokens=200)
#                     if isinstance(ai_resp, dict) and "text" in ai_resp:
#                         st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)
#     except Exception as e:
#         st.warning(f"Forecast failed: {e}")


# # ======================
# # 🚨 Anomaly Detection
# # ======================
# if enable_anomaly and not df_rev_trend.empty:
#     st.markdown("### 🚨 Anomaly Detection (Revenue)")
#     try:
#         from sklearn.ensemble import IsolationForest
#         import numpy as np

#         contamination = st.slider("Expected Outlier Fraction", 0.01, 0.2, 0.03)
#         model = IsolationForest(contamination=contamination, random_state=42)
#         df_rev_trend['value'] = pd.to_numeric(df_rev_trend['value'], errors='coerce').fillna(0)
#         model.fit(df_rev_trend[['value']])
#         df_rev_trend['anomaly'] = model.predict(df_rev_trend[['value']])
#         anomalies = df_rev_trend[df_rev_trend['anomaly'] == -1]
#         st.metric("🚨 Anomalies Detected", f"{len(anomalies)}")

#         base = alt.Chart(df_rev_trend).encode(x='period:O')
#         line = base.mark_line().encode(y='value:Q')
#         points = base.mark_circle(size=70).encode(
#             y='value:Q',
#             color=alt.condition(alt.datum.anomaly == -1, alt.value('red'), alt.value('black')),
#             tooltip=['period', 'value']
#         )
#         st.altair_chart((line + points).properties(height=350), use_container_width=True)

#         if len(anomalies) > 0:
#             st.warning(f"{len(anomalies)} anomalies detected in trend.")
#             st.dataframe(anomalies[['period', 'value']])
#             st.snow()

#             if enable_ai:
#                 with st.spinner("🤖 Generating AI anomaly insights..."):
#                     system = "You are an anomaly analyst reviewing outliers in revenue."
#                     sample = anomalies.head(10).to_dict(orient="records")
#                     user = f"Data anomalies: {json.dumps(sample, default=str)}. Provide 3 likely causes and 2 possible mitigations."
#                     ai_resp = deepinfra_chat(system, user, max_tokens=220)
#                     if isinstance(ai_resp, dict) and "text" in ai_resp:
#                         st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Anomaly detection failed: {e}")


# # ======================
# # 🧭 Clustering & Correlation (Auto-Adaptive)
# # ======================
# if enable_clustering and not df_rev_trend.empty:
#     st.markdown("### 🧭 Clustering & Correlation (AI + Visuals)")
#     try:
#         from sklearn.preprocessing import StandardScaler
#         from sklearn.cluster import KMeans
#         from sklearn.decomposition import PCA
#         from sklearn.metrics import silhouette_score
#         import plotly.express as px
#         import altair as alt
#         import numpy as np
#         import pandas as pd

#         df_cl = df_rev_trend.copy()
#         df_cl['value'] = pd.to_numeric(df_cl['value'], errors='coerce').fillna(0)

#         # --- Pick all numeric columns for clustering ---
#         num_cols = df_cl.select_dtypes(include=[np.number]).columns.tolist()
#         if not num_cols:
#             st.warning("No numeric columns found for clustering.")
#             st.stop()

#         X = df_cl[num_cols].astype(float)
#         scaler = StandardScaler()
#         Xs = scaler.fit_transform(X)

#         # --- Ensure valid number of clusters ---
#         max_clusters = max(2, min(8, len(Xs)))
#         n_clusters = st.slider("Number of Clusters (k)", 2, max_clusters, 3)
#         if len(Xs) < n_clusters:
#             n_clusters = len(Xs)

#         kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
#         labels = kmeans.fit_predict(Xs)
#         df_cl['cluster'] = labels

#         sc = silhouette_score(Xs, labels) if len(Xs) > n_clusters else 0
#         st.metric("Silhouette Score", f"{sc:.3f}")

#         st.dataframe(df_cl.head(10))

#         # --- PCA or fallback visualization ---
#         if Xs.shape[1] >= 2:
#             pca = PCA(n_components=2)
#             proj = pca.fit_transform(Xs)
#             scatter_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "cluster": labels})
#             chart = (
#                 alt.Chart(scatter_df)
#                 .mark_circle(size=80)
#                 .encode(x="x", y="y", color="cluster:N", tooltip=["x", "y", "cluster"])
#                 .properties(height=400, title="Cluster Projection (PCA)")
#             )
#             st.altair_chart(chart, use_container_width=True)
#         else:
#             # fallback for 1D data
#             scatter_df = pd.DataFrame({"x": Xs.flatten(), "cluster": labels})
#             chart = (
#                 alt.Chart(scatter_df)
#                 .mark_circle(size=80)
#                 .encode(x="x", y="cluster:N", color="cluster:N", tooltip=["x", "cluster"])
#                 .properties(height=400, title="Cluster Visualization (1D Data)")
#             )
#             st.altair_chart(chart, use_container_width=True)

#         # --- Correlation heatmap ---
#         if len(num_cols) > 1:
#             corr = df_cl[num_cols + ['cluster']].corr(numeric_only=True)
#             fig_corr = px.imshow(
#                 corr,
#                 text_auto=".2f",
#                 title="Correlation Matrix",
#                 color_continuous_scale="RdBu_r",
#             )
#             st.plotly_chart(fig_corr, use_container_width=True)
#         else:
#             st.info("ℹ️ Not enough numeric columns for correlation matrix.")

#         # --- AI Cluster Insights ---
#         if enable_ai:
#             with st.spinner("🤖 Generating AI clustering insights..."):
#                 cluster_summary = df_cl.groupby('cluster')['value'].mean().to_dict()
#                 system = "You are an expert analyst summarizing financial clusters."
#                 user = f"Cluster summaries: {json.dumps(cluster_summary, default=str)}. Provide 5 lines of interpretation and 2 action points."
#                 ai_resp = deepinfra_chat(system, user, max_tokens=320)
#                 if isinstance(ai_resp, dict) and "text" in ai_resp:
#                     st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)

#     except Exception as e:
#         st.error(f"Clustering failed: {e}")

# # ================================================================
# # 🌟 6️⃣ Revenue Trend + Forecast + Anomaly Detection + Clustering —  UI (ALL-MAXED)
# # ================================================================

# import streamlit as st
# import pandas as pd
# import altair as alt
# import json
# import numpy as np
# from datetime import datetime, timedelta

# # ================================
# # 🎨 CSS Animations & Transitions (kept)
# # ================================
# st.markdown("""
# <style>
# @keyframes slideIn {
#   from {opacity: 0; transform: translateY(20px);}
#   to {opacity: 1; transform: translateY(0);}
# }
# @keyframes pulseBorder {
#   0% {box-shadow: 0 0 0px #ff5722;}
#   50% {box-shadow: 0 0 10px #ff5722;}
#   100% {box-shadow: 0 0 0px #ff5722;}
# }
# .-container {
#   background: linear-gradient(90deg,#fff7f3,#ffffff);
#   border-left: 6px solid #ff5722;
#   padding: 16px 22px;
#   border-radius: 14px;
#   margin: 20px 0 15px 0;
#   animation: pulseBorder 4s infinite;
# }
# .metric-card {
#   background: #fff;
#   border-radius: 12px;
#   padding: 12px;
#   box-shadow: 0 3px 10px rgba(0,0,0,0.05);
#   transition: 0.3s;
# }
# .metric-card:hover {
#   transform: translateY(-2px);
#   box-shadow: 0 5px 15px rgba(255,87,34,0.3);
# }
# .ai-box {
#   background: linear-gradient(90deg,#ffffff,#fff9f6);
#   border-left: 4px solid #ff5722;
#   border-radius: 10px;
#   padding: 12px 14px;
#   margin-top: 8px;
#   animation: slideIn 1s ease;
# }
# .compare-card { padding:10px;border-radius:10px;background:linear-gradient(90deg,#fff,#fff8f5); border-left:5px solid #ff5722; margin-bottom:8px;}
# .up { color:#16a34a; font-weight:600; } .down { color:#dc2626; font-weight:600; }
# .small-muted { color:#6b7280;font-size:13px; }
# </style>
# """, unsafe_allow_html=True)


# # ======================
# # 📊 Section Header
# # ======================
# st.markdown("""
# <div class="-container">
#     <h2 style="margin:0;">💹 Revenue Trend & Advanced Analytics — ALL-MAXED</h2>
#     <p style="margin:4px 0 0;color:#444;font-size:15px;">
#         Smart forecasting, anomaly detection, cross-period comparisons (DoD/MoM/QoQ/YoY), and AI-powered clustering insights.
#     </p>
# </div>
# """, unsafe_allow_html=True)


# # ======================
# # 📈 Fetch & Visualize Revenue Trend (existing fetch assumed)
# # ======================
# with st.spinner("Fetching Revenue Trend..."):
#     rev_trend_json = fetch_json("vahandashboard/revenueFeeLineChart", desc="Revenue Trend")

# df_rev_trend = parse_revenue_trend(rev_trend_json if rev_trend_json else {})

# if df_rev_trend.empty:
#     st.warning("⚠️ No revenue trend data available.")
# else:
#     # normalize numeric
#     df_rev_trend = df_rev_trend.copy()
#     df_rev_trend['value'] = pd.to_numeric(df_rev_trend['value'], errors='coerce').fillna(0)

#     # try parse a datetime column from 'period' where possible
#     # Many Vahan endpoints return 'period' as YYYY-MM or month name; attempt common parses
#     def try_parse_period_to_date(s):
#         try:
#             # first try ISO-like formats
#             dt = pd.to_datetime(s, errors='coerce', dayfirst=False)
#             if pd.isna(dt):
#                 # try parse 'YYYY-MM' or 'YYYY' + month names
#                 for fmt in ("%Y-%m", "%b %Y", "%B %Y", "%Y/%m", "%Y"):
#                     try:
#                         dt = pd.to_datetime(s, format=fmt, errors='coerce')
#                         if not pd.isna(dt):
#                             break
#                     except Exception:
#                         dt = pd.NaT
#             return dt
#         except Exception:
#             return pd.NaT

#     df_rev_trend['_date_parsed'] = df_rev_trend['period'].astype(str).apply(try_parse_period_to_date)
#     # fallback: if all NaT, construct date from 'year' and 'period' if present
#     if df_rev_trend['_date_parsed'.replace('_','')].size == 0:
#         pass
#     # use parsed date where available, else create an index sequence
#     if df_rev_trend['_date_parsed'.replace('_','')].notna().any():
#         # prefer parsed date
#         df_rev_trend['date'] = df_rev_trend['_date_parsed'.replace('_','')]
#     else:
#         # fallback: use integer index as date-like sequence
#         df_rev_trend['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_rev_trend), freq='M')

#     # if previous block didn't set date due to name mismatch, correct above:
#     if 'date' not in df_rev_trend.columns or df_rev_trend['date'].isna().all():
#         df_rev_trend['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_rev_trend), freq='M')

#     # Basic revenue chart (Altair)
#     st.subheader("📊 Revenue Trend Comparison")
#     try:
#         chart = (
#             alt.Chart(df_rev_trend)
#             .mark_line(point=True, interpolate="monotone")
#             .encode(
#                 x=alt.X("date:T", title="Date"),
#                 y=alt.Y("value:Q", title="Revenue (₹)"),
#                 color=alt.Color("year:N", legend=alt.Legend(title="Year")),
#                 tooltip=["year", "period", "value"]
#             )
#             .properties(height=380, title="Revenue Trend Comparison")
#         )
#         st.altair_chart(chart, use_container_width=True)
#     except Exception:
#         st.dataframe(df_rev_trend)

#     # KPIs — Animated Cards
#     try:
#         total_rev = float(df_rev_trend["value"].sum())
#         avg_rev = float(df_rev_trend["value"].mean())
#         latest_rev = float(df_rev_trend["value"].iloc[-1])
#         prev_rev = float(df_rev_trend["value"].iloc[-2]) if len(df_rev_trend) > 1 else latest_rev
#         growth_pct = ((latest_rev - prev_rev) / prev_rev) * 100 if prev_rev else 0.0
#     except Exception:
#         total_rev = avg_rev = latest_rev = growth_pct = 0.0

#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.markdown(f"<div class='metric-card'><h4>💰 Total Revenue</h4><b>₹{total_rev:,.0f}</b></div>", unsafe_allow_html=True)
#     with col2:
#         st.markdown(f"<div class='metric-card'><h4>📈 Latest Revenue</h4><b>₹{latest_rev:,.0f}</b></div>", unsafe_allow_html=True)
#     with col3:
#         st.markdown(f"<div class='metric-card'><h4>📊 Avg per Period</h4><b>₹{avg_rev:,.0f}</b></div>", unsafe_allow_html=True)
#     with col4:
#         color = "#16a34a" if growth_pct >= 0 else "#dc2626"
#         st.markdown(f"<div class='metric-card'><h4>📅 Growth %</h4><b style='color:{color};'>{growth_pct:.2f}%</b></div>", unsafe_allow_html=True)

#     if growth_pct > 5:
#         st.balloons()
#     elif growth_pct < -5:
#         st.snow()

#     # =========================
#     # 🔁 ALL-MAXED COMPARISONS
#     # =========================
#     def compute_revenue_comparisons(df, date_col='date', value_col='value'):
#         """Return a small summary dict with DoD/MoM/QoQ/YoY latest percent changes and a timeseries table."""
#         out = {}
#         d = df.copy()
#         # ensure datetime index
#         d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
#         d = d.dropna(subset=[date_col]).sort_values(date_col)
#         if d.empty:
#             return out, pd.DataFrame()
#         d = d.set_index(date_col).asfreq('D')
#         d[value_col] = d[value_col].interpolate(method='time').fillna(0)

#         # daily series
#         daily = d[value_col].resample('D').sum()
#         monthly = d[value_col].resample('M').sum()
#         quarterly = d[value_col].resample('Q').sum()
#         yearly = d[value_col].resample('Y').sum()

#         def pct_last(series, periods=1):
#             if len(series) < periods + 1:
#                 return np.nan
#             last = series.iloc[-1]
#             prev = series.iloc[-1 - periods]
#             if prev == 0:
#                 return np.nan
#             return (last - prev) / prev * 100

#         out['DoD%'] = pct_last(daily, 1)
#         out['MoM%'] = pct_last(monthly, 1)
#         out['QoQ%'] = pct_last(quarterly, 1)
#         out['YoY%'] = pct_last(yearly, 1)

#         # build comparison timeseries DataFrame (monthly/quarterly/yearly + daily sample)
#         comp_ts = pd.DataFrame({
#             'date': daily.index,
#             'daily_value': daily.values
#         })
#         comp_ts['monthly'] = monthly.reindex(comp_ts['date'], method='ffill').values
#         comp_ts['quarterly'] = quarterly.reindex(comp_ts['date'], method='ffill').values
#         comp_ts['yearly'] = yearly.reindex(comp_ts['date'], method='ffill').values

#         return out, comp_ts.reset_index(drop=False).rename(columns={'index': 'date'})

#     comps_summary, comps_ts = compute_revenue_comparisons(df_rev_trend, date_col='date', value_col='value')

#     # Display comparison cards
#     st.markdown("### 🧾 Revenue Comparisons — DoD / MoM / QoQ / YoY")
#     try:
#         c1, c2, c3, c4 = st.columns(4)
#         metrics = [
#             ("📅 DoD%", comps_summary.get('DoD%')),
#             ("📆 MoM%", comps_summary.get('MoM%')),
#             ("🗓 QoQ%", comps_summary.get('QoQ%')),
#             ("📈 YoY%", comps_summary.get('YoY%')),
#         ]
#         for col, (label, value) in zip([c1, c2, c3, c4], metrics):
#             if value is None or (isinstance(value, float) and np.isnan(value)):
#                 display = "N/A"
#                 css = ""
#             else:
#                 arrow = "▲" if value > 0 else ("▼" if value < 0 else "→")
#                 css = "up" if value > 0 else "down" if value < 0 else ""
#                 display = f"{arrow} {value:.2f}%"
#             col.markdown(f"<div class='compare-card'><div style='font-weight:600'>{label}</div><div style='font-size:18px' class='{css}'>{display}</div><div class='small-muted'>Latest change</div></div>", unsafe_allow_html=True)
#     except Exception as e:
#         st.warning(f"Comparison cards failed: {e}")

#     # Comparison timeseries chart
#     if not comps_ts.empty:
#         try:
#             st.markdown("### 📊 Comparison Time Series (Daily sample with Monthly/Quarterly/Yearly overlays)")
#             # sample last 180 days for performance
#             plot_ts = comps_ts.tail(180).melt(id_vars='date', value_vars=['daily_value', 'monthly', 'quarterly', 'yearly'], var_name='series', value_name='value')
#             chart = (
#                 alt.Chart(plot_ts)
#                 .mark_line(point=False)
#                 .encode(
#                     x=alt.X('date:T'),
#                     y=alt.Y('value:Q', title='Revenue (₹)'),
#                     color='series:N',
#                     tooltip=['date:T', 'series:N', 'value:Q']
#                 )
#                 .properties(height=360)
#             )
#             st.altair_chart(chart, use_container_width=True)
#         except Exception as e:
#             st.warning(f"Comparison timeseries failed: {e}")

#     # =========================
#     # 🔮 Forecast (linear fallback; quick & robust)
#     # =========================
#     def simple_linear_forecast(df, months=6, date_col='date', value_col='value'):
#         dd = df.copy()
#         dd[date_col] = pd.to_datetime(dd[date_col], errors='coerce')
#         dd = dd.dropna(subset=[date_col]).sort_values(date_col)
#         if dd.empty:
#             return pd.DataFrame()
#         # aggregate monthly
#         monthly = dd.set_index(date_col)[value_col].resample('M').sum().reset_index()
#         monthly = monthly.dropna().reset_index(drop=True)
#         monthly['t'] = np.arange(len(monthly))
#         X = monthly[['t']].values
#         y = monthly['value'].values
#         if len(X) < 2:
#             return pd.DataFrame()
#         # linear fit
#         coeffs = np.polyfit(monthly['t'], y, 1)
#         slope, intercept = coeffs[0], coeffs[1]
#         future_t = np.arange(len(monthly), len(monthly) + months)
#         future_dates = pd.date_range(monthly['date'].max() + pd.offsets.MonthEnd(), periods=months, freq='M')
#         preds = intercept + slope * future_t
#         future_df = pd.DataFrame({'date': future_dates, 'value': preds, 'forecast': True})
#         hist = monthly.rename(columns={'date': 'date', 'value': 'value'}).assign(forecast=False)
#         # reconstruct hist to have 'date' and 'value', keep hist as monthly
#         hist = hist[['date', 'value']].assign(forecast=False)
#         return pd.concat([hist, future_df], ignore_index=True).sort_values('date').reset_index(drop=True)

#     if enable_forecast:
#         st.markdown("### 🔮 Forecast — Revenue Projection (linear fallback)")
#         try:
#             fc_df = simple_linear_forecast(df_rev_trend, months=forecast_periods, date_col='date', value_col='value')
#             if not fc_df.empty:
#                 # display line chart of historical monthly + forecast
#                 # combine monthly historical and forecast
#                 hist_monthly = df_rev_trend.set_index('date')['value'].resample('M').sum().reset_index()
#                 combined = pd.concat([
#                     hist_monthly.assign(type='Historical'),
#                     fc_df[fc_df['forecast'] == True].rename(columns={'value':'value'}).assign(type='Forecast')
#                 ], ignore_index=True).sort_values('date')
#                 fig = alt.Chart(combined).mark_line(point=True).encode(
#                     x='date:T', y='value:Q', color='type:N', tooltip=['date:T','value:Q','type:N']
#                 ).properties(height=360, title='Historical (Monthly) vs Forecast')
#                 st.altair_chart(fig, use_container_width=True)
#                 st.success("✅ Forecast generated (linear fallback).")
#                 if enable_ai:
#                     with st.spinner("🤖 Generating AI forecast commentary..."):
#                         system = "You are a forecasting analyst summarizing financial revenue predictions."
#                         sample = fc_df.tail(forecast_periods).to_dict(orient='records')
#                         user = f"Forecasted values: {json.dumps(sample, default=str)}. Summarize confidence and trend in 3 sentences."
#                         ai_resp = deepinfra_chat(system, user, max_tokens=200)
#                         if isinstance(ai_resp, dict) and ai_resp.get("text"):
#                             st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)
#             else:
#                 st.info("Forecast unavailable (not enough data).")
#         except Exception as e:
#             st.warning(f"Forecast failed: {e}")

#     # ======================
#     # 🚨 Anomaly Detection (already present) — keep & integrate
#     # ======================
#     if enable_anomaly:
#         st.markdown("### 🚨 Anomaly Detection (Revenue)")
#         try:
#             from sklearn.ensemble import IsolationForest
#             contamination = st.slider("Expected Outlier Fraction", 0.01, 0.2, 0.03)
#             model = IsolationForest(contamination=contamination, random_state=42)
#             rev_for_model = df_rev_trend[['value']].copy()
#             model.fit(rev_for_model)
#             df_rev_trend['anomaly'] = model.predict(rev_for_model)
#             anomalies = df_rev_trend[df_rev_trend['anomaly'] == -1]
#             st.metric("🚨 Anomalies Detected", f"{len(anomalies)}")

#             base = alt.Chart(df_rev_trend).encode(x='date:T')
#             line = base.mark_line().encode(y='value:Q')
#             points = base.mark_circle(size=70).encode(
#                 y='value:Q',
#                 color=alt.condition(alt.datum.anomaly == -1, alt.value('red'), alt.value('black')),
#                 tooltip=['period', 'value']
#             )
#             st.altair_chart((line + points).properties(height=350), use_container_width=True)

#             if len(anomalies) > 0:
#                 st.warning(f"{len(anomalies)} anomalies detected in trend.")
#                 st.dataframe(anomalies[['period', 'value']])
#                 st.snow()
#                 if enable_ai:
#                     with st.spinner("🤖 Generating AI anomaly insights..."):
#                         system = "You are an anomaly analyst reviewing outliers in revenue."
#                         sample = anomalies.head(10).to_dict(orient="records")
#                         user = f"Data anomalies: {json.dumps(sample, default=str)}. Provide 3 likely causes and 2 mitigations."
#                         ai_resp = deepinfra_chat(system, user, max_tokens=220)
#                         if isinstance(ai_resp, dict) and ai_resp.get("text"):
#                             st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Anomaly detection failed: {e}")

#     # ======================
#     # 🧭 Clustering & Correlation (Auto-Adaptive) — keep existing
#     # ======================
#     if enable_clustering:
#         st.markdown("### 🧭 Clustering & Correlation (AI + Visuals)")
#         try:
#             from sklearn.preprocessing import StandardScaler
#             from sklearn.cluster import KMeans
#             from sklearn.decomposition import PCA
#             from sklearn.metrics import silhouette_score
#             import plotly.express as px

#             df_cl = df_rev_trend.copy()
#             df_cl['value'] = pd.to_numeric(df_cl['value'], errors='coerce').fillna(0)

#             # numeric columns
#             num_cols = df_cl.select_dtypes(include=[np.number]).columns.tolist()
#             if not num_cols:
#                 st.warning("No numeric columns found for clustering.")
#             else:
#                 X = df_cl[num_cols].astype(float)
#                 scaler = StandardScaler()
#                 Xs = scaler.fit_transform(X)

#                 max_clusters = max(2, min(8, len(Xs)))
#                 n_clusters = st.slider("Number of Clusters (k)", 2, max_clusters, 3)
#                 if len(Xs) < n_clusters:
#                     n_clusters = len(Xs)

#                 kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
#                 labels = kmeans.fit_predict(Xs)
#                 df_cl['cluster'] = labels

#                 sc = silhouette_score(Xs, labels) if len(Xs) > n_clusters else 0
#                 st.metric("Silhouette Score", f"{sc:.3f}")

#                 st.dataframe(df_cl.head(10))

#                 if Xs.shape[1] >= 2:
#                     pca = PCA(n_components=2)
#                     proj = pca.fit_transform(Xs)
#                     scatter_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "cluster": labels})
#                     chart = (
#                         alt.Chart(scatter_df)
#                         .mark_circle(size=80)
#                         .encode(x="x", y="y", color="cluster:N", tooltip=["x", "y", "cluster"])
#                         .properties(height=400, title="Cluster Projection (PCA)")
#                     )
#                     st.altair_chart(chart, use_container_width=True)
#                 else:
#                     scatter_df = pd.DataFrame({"x": Xs.flatten(), "cluster": labels})
#                     chart = (
#                         alt.Chart(scatter_df)
#                         .mark_circle(size=80)
#                         .encode(x="x", y="cluster:N", color="cluster:N", tooltip=["x", "cluster"])
#                         .properties(height=400, title="Cluster Visualization (1D)")
#                     )
#                     st.altair_chart(chart, use_container_width=True)

#                 # correlation heatmap (if many numeric cols)
#                 if len(num_cols) > 1:
#                     corr = df_cl[num_cols + ['cluster']].corr(numeric_only=True)
#                     fig_corr = px.imshow(corr, text_auto=".2f", title="Correlation Matrix", color_continuous_scale="RdBu_r")
#                     st.plotly_chart(fig_corr, use_container_width=True)
#                 else:
#                     st.info("ℹ️ Not enough numeric columns for correlation matrix.")

#                 if enable_ai:
#                     with st.spinner("🤖 Generating AI clustering insights..."):
#                         cluster_summary = df_cl.groupby('cluster')['value'].mean().to_dict()
#                         system = "You are an expert analyst summarizing financial clusters."
#                         user = f"Cluster summaries: {json.dumps(cluster_summary, default=str)}. Provide 5 lines of interpretation and 2 action points."
#                         ai_resp = deepinfra_chat(system, user, max_tokens=320)
#                         if isinstance(ai_resp, dict) and ai_resp.get("text"):
#                             st.markdown(f"<div class='ai-box'>{ai_resp['text']}</div>", unsafe_allow_html=True)

#         except Exception as e:
#             st.error(f"Clustering failed: {e}")

# # end revenue section


# ===============================================================
# 📦 PARIVAHAN — SMART EXCEL EXPORT (MAXED, HARDENED & INTEGRATED)
# ===============================================================
# This block merges:
#   1️⃣ Smart Excel Export (full logic + derived comparisons + AI summaries)
#   2️⃣ Cleaned Imports (for the full Vahan Streamlit app)
# Ready for Streamlit Cloud.

# =============================
# 📚 Cleaned & Consolidated Imports
# =============================
# Standard library
import os
import sys
import io
import json
import time
import random
import traceback
from datetime import datetime, date, timedelta

# Third-party
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv
import requests

# Excel / Openpyxl
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference

# Optional ML & Forecast Libraries
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

# Local vahan modules
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

# Load environment variables early
load_dotenv()

# ===============================================================
# 💾 SMART EXCEL EXPORT — Maxed + Hardened
# ===============================================================

# ---------- Helper: Find DataFrame by common variable names ----------
def find_df_by_names(names):
    for name in names:
        if name in globals() and isinstance(globals()[name], pd.DataFrame):
            return globals()[name]
        if name in locals() and isinstance(locals()[name], pd.DataFrame):
            return locals()[name]
    return None


# ---------- Normalize time-series ----------
def normalize_time_series(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    date_cols = [c for c in df.columns if c.lower() in ("date", "period", "x", "ds")]
    value_cols = [c for c in df.columns if c.lower() in ("value", "y", "count", "total", "registeredvehiclecount")]

    if not date_cols:
        for c in df.columns:
            try:
                pd.to_datetime(df[c].dropna().iloc[:3])
                date_cols = [c]
                break
            except Exception:
                continue

    if not value_cols:
        for c in df.columns:
            if c in date_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                value_cols = [c]
                break

    if not date_cols or not value_cols:
        return pd.DataFrame()

    date_col, value_col = date_cols[0], value_cols[0]
    df = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# ---------- Simple Linear Forecast ----------
def linear_forecast(df, months=6, date_col="date", value_col="value"):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values(date_col).reset_index(drop=True)
    df["t"] = np.arange(len(df))
    X, y = df[["t"]].values, df[value_col].values

    try:
        coef = np.polyfit(df["t"].astype(float), y.astype(float), 1)
        slope, intercept = coef
        future_t = np.arange(len(df), len(df) + months)
        future_vals = intercept + slope * future_t
        last_date = df[date_col].max()
        future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=months, freq="MS")
        fc = pd.DataFrame({date_col: future_dates, value_col: future_vals})
        fc["forecast"] = True
        hist = df[[date_col, value_col]].copy()
        hist["forecast"] = False
        return pd.concat([hist, fc], ignore_index=True)
    except Exception:
        return pd.DataFrame()


# ===============================================================
# 🎨 UI BLOCK
# ===============================================================
st.markdown("""
<div style="padding:18px 20px;border-left:5px solid #007bff;
            background:linear-gradient(90deg,#f0f8ff,#ffffff);
            border-radius:12px;margin-top:25px;margin-bottom:15px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <h2 style="margin:0;">💾 SMART EXCEL EXPORT — Maxed</h2>
    <p style="margin:4px 0 0;color:#444;font-size:15px;">
        Export <b>all detected DataFrames</b> plus derived comparisons, forecasts, anomalies and AI insights
        into a styled Excel workbook (auto-charts included).
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📊 Generate & Download Full Smart Analytics Workbook", expanded=True):
    # Detect existing dataframes in runtime
    dfs = {k: v for k, v in globals().items() if isinstance(v, pd.DataFrame)}
    dfs.update({k: v for k, v in locals().items() if isinstance(v, pd.DataFrame)})

    st.info(f"🧾 Detected {len(dfs)} datasets for export: {list(dfs.keys())}")

    derived = {}
    ts_df = None
    candidates = [
        "df_cat", "df_mk", "df_trend", "yoy_df", "qoq_df", "df_top5_rev",
        "df_rev_trend", "df_trend_all", "df_mk_all", "df_cat_all", "df_forecast"
    ]

    # Find first valid trend dataframe
    for name in candidates:
        if name in dfs and not dfs[name].empty:
            tmp = normalize_time_series(dfs[name])
            if not tmp.empty:
                ts_df = tmp
                break

    # Derived comparisons and forecast
    if ts_df is not None:
        try:
            df_trend = ts_df.copy()
            df_trend["Year"] = df_trend["date"].dt.year
            df_trend["MonthName"] = df_trend["date"].dt.strftime("%b")

            yearly = df_trend.groupby("Year")["value"].sum().reset_index()
            yearly["YoY Growth"] = yearly["value"].pct_change().fillna(0)
            derived["Yearly Comparison"] = yearly

            monthly = df_trend.groupby(["Year", "MonthName"])["value"].sum().reset_index()
            derived["Monthly Breakdown"] = monthly

            if df_trend.shape[0] >= 60:
                daily = df_trend.groupby(df_trend["date"].dt.date)["value"].sum().reset_index()
                daily["DoD Growth"] = daily["value"].pct_change().fillna(0)
                derived["Daily Breakdown"] = daily

            fc = linear_forecast(df_trend, months=6)
            fc["RollingMean"] = fc["value"].rolling(4, min_periods=1).mean()
            fc["Anomaly"] = (fc["value"] - fc["RollingMean"]).abs() > (fc["RollingMean"] * 0.2)
            derived["Forecast & Anomaly"] = fc
            dfs["Normalized Trend"] = df_trend
        except Exception as e:
            st.warning(f"⚠️ Failed to derive comparisons: {e}")
    else:
        st.info("ℹ️ No valid time-series found for comparison.")

    # Category vs Maker cross summary
    try:
        if "df_cat" in dfs and "df_mk" in dfs and not dfs["df_cat"].empty and not dfs["df_mk"].empty:
            c, m = dfs["df_cat"], dfs["df_mk"]
            s = pd.DataFrame({
                "Category Total": [c["value"].sum()],
                "Maker Total": [m["value"].sum()],
                "Maker/Category Ratio": [round(m["value"].sum() / max(1, c["value"].sum()), 4)]
            })
            derived["Category vs Maker Summary"] = s
    except Exception as e:
        st.warning(f"⚠️ Category vs Maker derivation failed: {e}")

    # Merge derived frames
    dfs.update(derived)

    # ---------------- AI Summaries (DeepInfra) ----------------
    summaries = {}
    enable_ai_flag = globals().get("enable_ai", locals().get("enable_ai", False))
    if enable_ai_flag:
        st.info("🤖 Generating AI summaries...")
        prog = st.progress(0)
        total = len(dfs)
        for i, (name, df) in enumerate(dfs.items(), start=1):
            try:
                if df.empty:
                    summaries[name] = "No data."
                elif "deepinfra_chat" in globals():
                    system = f"You are an analytics expert summarizing dataset '{name}'."
                    user = f"Sample data: {df.head(5).to_dict(orient='records')}\nProvide 3 concise insights."
                    ai_resp = deepinfra_chat(system, user, max_tokens=180)
                    summaries[name] = ai_resp.get("text", "No response.") if isinstance(ai_resp, dict) else str(ai_resp)
                else:
                    summaries[name] = "DeepInfra not configured."
            except Exception as e:
                summaries[name] = f"AI failed: {e}"
            prog.progress(i / total)
        prog.empty()
        dfs["AI Summaries"] = pd.DataFrame(list(summaries.items()), columns=["Dataset", "AI Summary"])

    # ---------------- Excel Export ----------------
    exportable = {k: v for k, v in dfs.items() if isinstance(v, pd.DataFrame) and not v.empty}
    if not exportable:
        st.warning("⚠️ No valid dataframes to export.")
    else:
        with st.spinner("📦 Compiling styled Excel workbook..."):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for name, df in exportable.items():
                    safe_name = str(name)[:31]
                    try:
                        df.to_excel(writer, sheet_name=safe_name, index=False)
                    except Exception:
                        df.astype(str).to_excel(writer, sheet_name=safe_name, index=False)
            output.seek(0)

            wb = load_workbook(output)
            thin = Side(style="thin")
            border = Border(left=thin, right=thin, top=thin, bottom=thin)

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                try:
                    for cell in list(ws.rows)[0]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill(start_color="007bff", end_color="007bff", fill_type="solid")
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                        cell.border = border
                except Exception:
                    pass

                for col in ws.columns:
                    col_letter = get_column_letter(col[0].column)
                    max_len = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                    ws.column_dimensions[col_letter].width = min(max_len + 4, 60)
                    for cell in col:
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                        cell.border = border

                try:
                    if ws.max_row > 2 and ws.max_column >= 2:
                        val_ref = Reference(ws, min_col=2, min_row=1, max_row=ws.max_row)
                        cat_ref = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
                        chart = LineChart()
                        chart.title = f"{sheet_name} Trend"
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

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            label="⬇️ Download Analytics Excel Workbook",
            data=styled.getvalue(),
            file_name=f"Parivahan_Analytics_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.success("✅ Excel report ready — includes comparisons, forecasts, anomalies & AI summaries.")
        st.balloons()

# RAW JSON ANALYZER PRO — MAXED EDITION
# Paste this block into your Streamlit app (or import it).
# Features added (MAXED):
# - Automatic endpoint tagging
# - Safe sanitization (remove unserializable/binary blobs)
# - Auto-detect & auto-flatten lists of dicts
# - Search + highlight in pretty JSON (HTML highlight)
# - Deep/Quick AI summary modes via DeepInfra (if available)
# - ZIP contains both .json and .xlsx (flattened) option
# - Safe Mode toggle to avoid local file writes (good for Streamlit Cloud)
# - UI toggles for Export options & Smart Flattening

import streamlit as st
import pandas as pd
import json
import os
import io
import tempfile
import zipfile
from datetime import datetime
from typing import Any, Dict, Tuple

# Helper: safe JSON serializer
def safe_serialize(obj: Any, max_len: int = 10000) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        try:
            return json.dumps(str(obj))
        except Exception:
            return '"<UNSERIALIZABLE>"'

# Helper: sanitize payload (strip binary/unserializable keys recursively)
def sanitize_payload(obj: Any, depth: int = 3) -> Any:
    if depth < 0:
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                json.dumps(v, default=str)
                out[k] = sanitize_payload(v, depth - 1)
            except Exception:
                out[k] = str(type(v).__name__)
        return out
    if isinstance(obj, list):
        out = []
        for v in obj:
            try:
                json.dumps(v, default=str)
                out.append(sanitize_payload(v, depth - 1))
            except Exception:
                out.append(str(type(v).__name__))
        return out
    # primitive
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

# Helper: auto flatten if payload is list of dicts
def try_flatten(payload: Any) -> Tuple[pd.DataFrame, bool]:
    try:
        if isinstance(payload, list) and payload and all(isinstance(x, dict) for x in payload):
            df = pd.json_normalize(payload)
            return df, True
        if isinstance(payload, dict):
            # some endpoints return {"data": [...]} or similar
            for k in payload:
                if isinstance(payload[k], list) and payload[k] and all(isinstance(x, dict) for x in payload[k]):
                    df = pd.json_normalize(payload[k])
                    return df, True
        return pd.DataFrame(), False
    except Exception:
        return pd.DataFrame(), False

# Helper: highlight search matches (very small, safe HTML)
def highlight_json_text(jtext: str, query: str) -> str:
    if not query:
        return jtext
    try:
        q = st.session_state.get("json_search_query", query).strip()
        if not q:
            return jtext
        # simple case-insensitive replace with mark tag
        import html
        esc = html.escape(q)
        return jtext.replace(q, f"<mark style='background:#ffe58f;border-radius:3px;padding:0 2px'>{esc}</mark>")
    except Exception:
        return jtext

# UI: Maxed RAW JSON Analyzer
st.markdown("""
<div style="padding:14px;border-left:6px solid #6366F1;background:#fafafa;border-radius:10px;margin-bottom:12px;">
  <h3 style="margin:0">🧩 RAW JSON ANALYZER PRO — MAXED</h3>
  <p style="margin:4px 0 0;color:#444;">Inspect, sanitize, flatten, export (JSON/XLSX/ZIP) and summarize API payloads safely.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("🛠️ RAW JSON ANALYZER (Developer / Debug Mode — All ) — Maxed", expanded=False):
    # CONFIG
    col_a, col_b, col_c, col_d = st.columns([2,2,2,2])
    with col_a:
        show_pretty = st.checkbox("🧾 Pretty JSON (code block)", value=False)
    with col_b:
        show_table = st.checkbox("📋 Table View (flattened when available)", value=True)
    with col_c:
        enable_ai_summary = st.checkbox("🤖 AI Summary (Quick)", value=False)
    with col_d:
        enable_ai_deep = st.checkbox("🧠 AI Deep Mode (Long)", value=False)

    # Export toggles
    col_e, col_f, col_g = st.columns([1.5,1.5,1])
    with col_e:
        include_xlsx_in_zip = st.checkbox("📑 Include XLSX inside ZIP", value=True)
    with col_f:
        safe_mode = st.checkbox("🔒 Safe Mode (no server writes) — recommended for Cloud", value=True)
    with col_g:
        auto_flatten = st.checkbox("✨ Auto Flatten Lists → Tables", value=True)

    snapshot_name = st.text_input("📁 Snapshot filename (no extension)", value=f"vahan_snapshot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}")
    st.markdown("---")

    # Collect JSON-like objects automatically from locals (similar to your existing approach)
    json_candidates = {k:v for k,v in locals().items() if ("json" in k.lower() or k.lower().endswith("_json")) and isinstance(v, (dict, list))}

    # If none, also look for variables that user commonly uses (df_ prefixed JSONs stored)
    if not json_candidates:
        json_candidates = {k:v for k,v in st.session_state.items() if ("json" in k.lower() or k.lower().endswith("_json")) and isinstance(v, (dict, list))}

    if not json_candidates:
        st.warning("⚠️ No JSON payloads detected. Run API calls first or ensure variables are in the app's scope.")
    else:
        st.success(f"✅ Found {len(json_candidates)} JSON payload(s)")
        st.caption(f"Detected keys: {list(json_candidates.keys())}")

    # Search
    search_query = st.text_input("🔎 Search within JSON (keys/values)", value="", key="json_search_query")

    # Render each payload
    for key, data in json_candidates.items():
        st.markdown(f"### 📦 {key}")

        # Tagging: attempt to infer endpoint name from key
        inferred = key.replace('_', ' ').title()
        st.caption(f"Endpoint tag: {inferred} • Type: {type(data).__name__} • Items: {len(data) if hasattr(data,'__len__') else '?'}")

        # Sanitize copy for UI/exports
        safe_payload = sanitize_payload(data, depth=4)

        # Optional search filter — present if nothing matches
        try:
            txt = json.dumps(safe_payload, indent=2, default=str)
        except Exception:
            txt = safe_serialize(safe_payload)

        if search_query and search_query.strip():
            if search_query.lower() not in txt.lower():
                st.info(f"🔍 No match for '{search_query}' in {key}")
                continue

        # Pretty / Raw view
        if show_pretty:
            # apply highlight
            html_snip = '<pre style="max-height:420px;overflow:auto;background:#0f172a;color:#e6eef8;padding:12px;border-radius:8px;">'
            highlighted = txt
            if search_query:
                highlighted = highlighted.replace(search_query, f"<mark style='background:#ffe58f'>{search_query}</mark>")
            # escape then wrap (we already have json string, safe enough)
            html_snip += highlighted + '</pre>'
            st.components.v1.html(html_snip, height=420)
        else:
            # interactive JSON viewer
            try:
                st.json(safe_payload)
            except Exception:
                st.code(txt, language='json')

        # Auto-flatten suggestion
        df_flat, flattened = (pd.DataFrame(), False)
        if auto_flatten:
            df_flat, flattened = try_flatten(safe_payload)

        if show_table and (flattened or isinstance(safe_payload, (dict, list))):
            if not df_flat.empty:
                st.markdown("**Tabular preview (auto-flattened)**")
                st.dataframe(df_flat.head(250), use_container_width=True)
            else:
                # attempt lightweight normalization
                try:
                    df_try = pd.json_normalize(safe_payload)
                    if not df_try.empty:
                        st.markdown("**Flattened preview**")
                        st.dataframe(df_try.head(200), use_container_width=True)
                        df_flat = df_try
                except Exception:
                    pass

        # Download controls (in-memory) — JSON + XLSX if flattened
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            try:
                as_bytes = json.dumps(safe_payload, indent=2, default=str).encode('utf-8')
                st.download_button(label=f"⬇️ {key}.json", data=as_bytes, file_name=f"{key}.json", mime="application/json")
            except Exception as e:
                st.warning(f"JSON download failed: {e}")
        with col2:
            if not df_flat.empty:
                try:
                    buf = io.BytesIO()
                    df_flat.to_excel(buf, index=False)
                    buf.seek(0)
                    st.download_button(label=f"⬇️ {key}.xlsx", data=buf.getvalue(), file_name=f"{key}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.warning(f"XLSX download failed: {e}")
        with col3:
            # AI summary options
            if enable_ai_summary or enable_ai_deep:
                if 'deepinfra_chat' in globals() or 'deepinfra_chat' in locals():
                    mode = 'Deep' if enable_ai_deep else 'Quick'
                    if st.button(f"🤖 Generate {mode} AI Summary — {key}"):
                        with st.spinner("Summarizing via AI..."):
                            try:
                                system = "You are an automotive analytics assistant. Provide a concise summary."
                                user = f"Payload ({key}) sample: {json.dumps(safe_payload)[:8000]} (truncated). Mode: {mode}"
                                # Deep mode request more tokens
                                max_t = 450 if enable_ai_deep else 180
                                resp = deepinfra_chat(system, user, max_tokens=max_t, temperature=0.25)
                                if isinstance(resp, dict) and 'text' in resp and resp['text']:
                                    st.markdown(f"**AI Summary ({mode}):** {resp['text']}")
                                else:
                                    st.info("💤 AI returned no text. Check your DeepInfra key or retry.")
                            except Exception as e:
                                st.error(f"AI summary failed: {e}")
                else:
                    st.info("⚠️ deepinfra_chat not found in environment — AI disabled.")

        st.markdown('---')

    # GLOBAL SNAPSHOT (JSON + optional XLSX) and ZIP
    st.subheader("📦 Create Global Snapshot / ZIP")
    create_snapshot = st.button("💾 Create Unified Snapshot (JSON & XLSX + ZIP)")

    if create_snapshot:
        try:
            # Build combined structure
            combined = {k: sanitize_payload(v) for k,v in json_candidates.items()}
            combined_meta = {"generated_at": datetime.now().isoformat(), "count": len(combined)}
            combined_package = {"meta": combined_meta, "payloads": combined}
            combined_bytes = json.dumps(combined_package, indent=2, default=str).encode('utf-8')

            # Create ZIP in memory
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as z:
                # add combined json
                z.writestr(f"{snapshot_name}.json", combined_bytes)

                # add per-payload json and flattened xlsx (if available)
                for k,v in json_candidates.items():
                    safe_v = sanitize_payload(v)
                    z.writestr(f"{k}.json", json.dumps(safe_v, indent=2, default=str))
                    # flattened
                    df_f, ok = try_flatten(safe_v)
                    if ok and include_xlsx_in_zip:
                        buf = io.BytesIO()
                        df_f.to_excel(buf, index=False)
                        buf.seek(0)
                        z.writestr(f"{k}.xlsx", buf.read())

            zip_buf.seek(0)

            if safe_mode:
                # Provide download button only — no server filesystem writes
                st.download_button("⬇️ Download Snapshot ZIP", data=zip_buf.getvalue(), file_name=f"{snapshot_name}_bundle.zip", mime="application/zip")
                st.success("✅ Snapshot ZIP prepared (in-memory).")
            else:
                # also write to server and show path
                tmp_path = os.path.join(tempfile.gettempdir(), f"{snapshot_name}_bundle.zip")
                with open(tmp_path, 'wb') as f:
                    f.write(zip_buf.getvalue())
                st.download_button("⬇️ Download Snapshot ZIP", data=open(tmp_path, 'rb').read(), file_name=f"{snapshot_name}_bundle.zip", mime="application/zip")
                st.success(f"✅ Snapshot ZIP written to: {tmp_path}")

        except Exception as e:
            st.error(f"Snapshot creation failed: {e}")

    st.markdown('---')

    # Optional Diagnostics Save (local only)
    try:
        if st.checkbox("📝 Save diagnostics JSON locally (server only)"):
            if safe_mode:
                st.warning("Disabled in Safe Mode. Turn off Safe Mode to enable local saves.")
            else:
                diag = {"timestamp": datetime.now().isoformat(), "found_keys": list(json_candidates.keys())}
                p = os.path.join(os.getcwd(), f"vahan_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(diag, f, indent=2)
                st.success(f"Diagnostics saved to: {p}")
    except Exception:
        pass

    st.info("🔒 RAW JSON ANALYZER PRO — Maxed features available. Disable developer blocks for production.")

# ============================================================
# ⚡ FOOTER KPIs + EXECUTIVE SUMMARY —  ALL-MAXED VERSION 🚀
# ============================================================

import json, time, random
import streamlit as st
import pandas as pd

st.markdown("---")
st.subheader("📊 Dashboard Summary & Insights — All Years, All Filters, All Maxed")

# ============================================================
# 🧮 SMART METRIC COMPUTATION — ALL YEARS + CAGR
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

    # Calculate CAGR across all years if available
    if "year" in df_trend.columns and len(df_trend["year"].unique()) > 1:
        years_sorted = sorted(df_trend["year"].unique())
        start_val = df_trend[df_trend["year"] == years_sorted[0]]["value"].sum()
        end_val = df_trend[df_trend["year"] == years_sorted[-1]]["value"].sum()
        cagr = ((end_val / start_val) ** (1 / (len(years_sorted) - 1)) - 1) * 100 if start_val > 0 else 0
    else:
        cagr = 0

except Exception as e:
    st.error(f"Metric computation failed: {e}")

# ============================================================
# 🎯 KPI Metric Cards (Animated & Styled, All-Maxed)
# ============================================================
kpi_cols = st.columns(5)

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

with kpi_cols[4]:
    st.metric("📊 CAGR (All Years)", f"{cagr:.2f}%" if cagr else "N/A")

# ============================================================
# 🗓️ MULTI-YEAR SUMMARY TABLE (Dynamic)
# ============================================================
if "year" in df_trend.columns:
    year_summary = (
        df_trend.groupby("year")["value"]
        .agg(["sum", "mean"])
        .rename(columns={"sum": "Total Registrations", "mean": "Daily Avg"})
        .reset_index()
    )
    st.markdown("### 📆 Year-wise Summary (Aggregated)")
    st.dataframe(year_summary, use_container_width=True)

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
# 🤖 AI-POWERED EXECUTIVE SUMMARY — DEEPINFRA (All Context)
# ============================================================
if "enable_ai" in locals() and enable_ai:
    st.markdown("### 🤖 Executive AI Summary (DeepInfra MAXED)")

    with st.spinner("🧠 Synthesizing AI-driven executive summary..."):
        try:
            # ---- Build context dynamically ----
            context = {
                "total_registrations": total_reg,
                "daily_avg_orders": daily_avg,
                "latest_yoy": yoy_latest,
                "latest_qoq": qoq_latest,
                "cagr": cagr,
                "top_revenue_state": top_state,
                "top_revenue_value": top_val,
                "data_years_available": sorted(df_trend["year"].unique().tolist()) if "year" in df_trend.columns else "N/A",
                "total_states_covered": df_top5_rev["label"].nunique() if not df_top5_rev.empty else 0,
            }

            # Capture user filter context if available
            selected_states = st.session_state.get("selected_states", [])
            selected_categories = st.session_state.get("selected_categories", [])
            selected_years = st.session_state.get("selected_years", [])

            system = (
                "You are an AI analytics assistant summarizing a national vehicle registration dashboard. "
                "Analyze trends, KPIs, CAGR, and performance across states, categories, and years. "
                "Your tone should be concise, strategic, and executive-level."
            )

            user = (
                f"Context data: {json.dumps(context, default=str)}. "
                f"User-selected filters: States={selected_states}, Categories={selected_categories}, Years={selected_years}. "
                "Generate a 6-sentence executive summary covering key growth, performance highlights, and strategic recommendations."
            )

            ai_resp = deepinfra_chat(system, user, max_tokens=400, temperature=0.4)
            ai_summary = ai_resp.get("text") if isinstance(ai_resp, dict) else str(ai_resp)

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
# ✨ FOOTER — BRANDING, TIMESTAMP & MOTION (All-Maxed)
# ============================================================
st.markdown(
    f"""
    <hr style="border: 1px solid #444; margin-top: 2em; margin-bottom: 1em;">
    <div style="text-align:center; color:gray; font-size:0.9em; animation:fadeInUp 1.5s;">
        🚗 <b>Parivahan Analytics — MAXED 2025</b><br>
        <span style="color:#aaa;">AI Narratives • Smart KPIs • Forecasting • Growth Insights</span><br>
        <i>Empowering Data-Driven Governance — All India, All Years.</i><br><br>
        <small>⏱️ Last refreshed: {time.strftime("%d %b %Y, %I:%M %p")}</small>
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
st.toast("✨ Dashboard summary ready — All KPIs, AI insights, and multi-year data loaded successfully.")


