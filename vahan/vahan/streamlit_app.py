# =====================================================
# 🚗 VAHAN ULTRA MASTER DASHBOARD — ALL MAXED EDITION
# =====================================================
import os, sys, io, json, time, logging, platform, traceback, random
from datetime import date, datetime, timedelta
from functools import wraps
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# 📚 Cleaned & Consolidated Imports
# =============================

# ---------- Standard Library ----------
import io
import json
import time
import random
import traceback
from datetime import date, timedelta

# ---------- Third-Party Libraries ----------
import numpy as np
import pandas as pd
import requests
import altair as alt
from dotenv import load_dotenv

# ============================================================
# 🚀 VAHAN ALL-MAXED MODE — Full Power Imports (No Limits)
# ============================================================

# ---------- Standard Library ----------
import os
import sys
import io
import re
import math
import csv
import json
import time
import random
import string
import socket
import base64
import hashlib
import logging
import zipfile
import traceback
import datetime
import statistics
import itertools
import concurrent.futures
from functools import lru_cache, partial, reduce
from datetime import date, datetime, timedelta
from urllib.parse import urlencode, quote, unquote

# ---------- Third-Party Core ----------
import numpy as np
import pandas as pd
import requests
import altair as alt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

# ---------- Machine Learning / Forecasting ----------
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from prophet import Prophet

# ---------- Excel / OpenPyXL ----------
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import (
    LineChart, BarChart, PieChart, Reference, Series
)

# ---------- Visualization Addons ----------
import plotly.io as pio
pio.templates.default = "plotly_white"
sns.set_theme(style="whitegrid")

# ---------- Utilities ----------
from colorama import Fore, Style, init as colorama_init
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

import os, sys

# ---- Safe Colorama initialization ----
try:
    # On Streamlit Cloud or restricted environments, disable colorama wrapping
    if "STREAMLIT_SERVER" in os.environ or "streamlit" in sys.modules:
        # Avoid recursion by forcing no wrap
        import colorama
        colorama.deinit()  # reset any existing wrapping
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        colorama.init(strip=True, convert=False, autoreset=True)
    else:
        # Normal local mode
        import colorama
        colorama.init(autoreset=True)

except Exception as e:
    # Disable colorama completely if it fails
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("⚠️ Console color initialization disabled:", e)
# ---------- Local VAHAN Package (ALL IMPORTS) ----------
from vahan.api import *
from vahan.parsing import *
from vahan.metrics import *
from vahan.charts import *

# ============================================================
# 🚀 ALL-MAXED GLOBAL INITIALIZATION BLOCK (v2.0)
# ============================================================
import os
import sys
import random
import time
import json
import math
import logging
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# Optional AI + ML + Forecasting Modules
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from prophet import Prophet

# Excel + Visualization + Utils
from openpyxl import load_workbook
from io import BytesIO

# Initialize color output for cross-platform logs
colorama_init(autoreset=True)

# ---------- Warnings + Pandas Config ----------
warnings_filter = __import__("warnings")
warnings_filter.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.4f}".format)
np.set_printoptions(suppress=True)

# ---------- Load Environment ----------
load_dotenv()
TODAY = date.today()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---------- Streamlit Config (MUST COME FIRST) ----------
st.set_page_config(
    page_title="🚗 Vahan Master Ultra — ALL-MAXED Mode",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Print Startup Summary ----------
print(f"\n{Fore.GREEN}✅ ALL-MAXED ENVIRONMENT READY — FULL FEATURE SET ENABLED{Style.RESET_ALL}")
print(f"{Fore.CYAN}📦 Modules Loaded: numpy, pandas, streamlit, plotly, sklearn, prophet, openpyxl, requests, dotenv, logging{Style.RESET_ALL}")
print(f"{Fore.YELLOW}🧠 Mode: Developer | Analyst | Research — unrestricted ALL-MAXED imports{Style.RESET_ALL}")
print(f"{Fore.MAGENTA}🕒 Today: {TODAY} | Seed: {RANDOM_SEED}{Style.RESET_ALL}")
print(f"{Fore.BLUE}📁 Working Dir: {os.getcwd()}{Style.RESET_ALL}\n")

# ---------- Initialize Empty DataFrame ----------
df = pd.DataFrame()

# =====================================================
# ⚡ VAHAN ALL-MAXED ULTRA+ BOOT ENGINE (v4.0)
# =====================================================
import os, sys, time, platform, logging, threading, psutil
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import streamlit as st
from colorama import Fore, Style, init as colorama_init
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =====================================================
# 🌏 COLOR + GLOBAL INIT
# =====================================================
colorama_init(autoreset=True)
APP_NAME = "🚗 Parivahan Analytics"
APP_VERSION = "vMAX-ULTRA"
APP_ENV = "Production"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================
# ⚙️ LOGGING — ULTRA-MAXED (IST + Colors + Rotation + Compatibility)
# =====================================================
import os
import sys
import logging
from datetime import datetime
from colorama import Fore, Style, init as colorama_init
from zoneinfo import ZoneInfo

# Initialize colorama for Windows terminals
colorama_init(autoreset=True)

# Ensure log directory exists
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================
# 🕒 IST LOGGING HELPERS
# =====================================================
def log_ist(msg: str, level: str = "INFO", color: str = Fore.CYAN):
    """Print a timestamped message in IST timezone with color."""
    try:
        ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
        print(f"{color}[IST {ist_time}] [{level.upper()}] {msg}{Style.RESET_ALL}")
    except Exception as e:
        print(f"[IST Logging Fallback] {msg} (Error: {e})")

class ISTFormatter(logging.Formatter):
    """Custom logging formatter that formats times in IST."""
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Kolkata"))
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

# =====================================================
# 🧠 GLOBAL LOGGING CONFIGURATION (Dual Channel)
# =====================================================
def setup_global_logging():
    """Configure root logger with color console + rotating file (daily)."""
    log_file = os.path.join(LOG_DIR, "vahan_ultra.log")
    formatter = ISTFormatter("%(asctime)s | %(levelname)-8s | %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    # --- Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # --- File Handler (daily rotation)
    from logging.handlers import TimedRotatingFileHandler
    fh = TimedRotatingFileHandler(log_file, when="midnight", backupCount=10, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Notify success
    log_ist("✅ Logging initialized with daily rotation", "INFO", Fore.GREEN)

# Initialize logging on import
setup_global_logging()

def _get_ist_now():
    """Return current IST timestamp string safely."""
    try:
        return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %I:%M:%S")

def log(msg: str, level: str = "INFO"):
    """
    Safe, backward-compatible logger.
    Uses internal _get_ist_now() to avoid name collision with user vars.
    """
    try:
        ts = _get_ist_now()
    except Exception:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    level = str(level).upper().strip()
    prefix = f"[IST {ts}] [{level}]"
    color_map = {
        "INFO": Fore.CYAN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "DEBUG": Fore.BLUE,
        "SUCCESS": Fore.GREEN,
    }
    color = color_map.get(level, Fore.WHITE)

    try:
        print(f"{color}{prefix} {msg}{Style.RESET_ALL}")
        logging.log(getattr(logging, level, logging.INFO), msg)
    except Exception as e:
        print(f"[LOGGING FAILSAFE] {msg} (Error: {e})")
        
# =====================================================
# 💻 SYSTEM DIAGNOSTICS
# =====================================================
def system_diagnostics():
    """Collect and print system-level metrics."""
    try:
        cpu_usage = psutil.cpu_percent(interval=0.8)
        mem = psutil.virtual_memory()
        boot_time = datetime.fromtimestamp(psutil.boot_time(), ZoneInfo("Asia/Kolkata"))
        uptime = datetime.now(ZoneInfo("Asia/Kolkata")) - boot_time
        log_ist(f"💻 OS: {platform.system()} {platform.release()} | Python {platform.python_version()}", "INFO", Fore.MAGENTA)
        log_ist(f"🧠 CPU: {cpu_usage:.1f}% | Memory: {mem.percent}% of {round(mem.total / (1024**3), 2)} GB", "INFO", Fore.MAGENTA)
        log_ist(f"⏱️ Uptime: {str(timedelta(seconds=int(uptime.total_seconds())))}", "INFO", Fore.MAGENTA)
    except Exception as e:
        log_ist(f"⚠️ Diagnostics failed: {e}", "ERROR", Fore.RED)

system_diagnostics()

# =====================================================
# 🚀 BOOT BANNER
# =====================================================
def app_boot_banner(app_name=APP_NAME, version=APP_VERSION, env=APP_ENV):
    """Display an animated startup banner in Streamlit."""
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
    python_ver = platform.python_version()
    streamlit_ver = st.__version__
    sys_os = platform.system()
    cpu_count = os.cpu_count()
    cwd = os.getcwd()

    gradient = "linear-gradient(90deg,#0072ff,#00c6ff)"
    shadow = "0 4px 20px rgba(0,0,0,0.25)"
    st.markdown(f"""
    <div style='background:{gradient};color:white;padding:18px 26px;
         border-radius:18px;margin:20px 0 30px 0;box-shadow:{shadow};
         font-family:monospace;line-height:1.6;'>
        <h3>🌍 {app_name} — {version}</h3>
        🕒 <b>Boot Time:</b> {ist_time} (IST)<br>
        ⚙️ <b>Environment:</b> {env} | Python {python_ver} | Streamlit {streamlit_ver}<br>
        💻 OS: {sys_os} | CPU Cores: {cpu_count}<br>
        📁 Working Dir: {cwd}
    </div>
    """, unsafe_allow_html=True)

app_boot_banner()

# =====================================================
# 🎉 FIRST LAUNCH EVENT
# =====================================================
if "launched" not in st.session_state:
    st.session_state.launched = True
    st.toast("🚀 Welcome to VAHAN ULTRA MAX Edition!", icon="🌍")
    st.balloons()
    log_ist("🎉 Streamlit App Launched Successfully", "INFO", Fore.GREEN)

# =====================================================
# 🔁 AUTO FILE WATCHER (DEV MODE)
# =====================================================
class AutoReloadHandler(FileSystemEventHandler):
    """Auto-refresh Streamlit when .py files change (Dev mode only)."""
    def __init__(self, watch_paths):
        self.watch_paths = watch_paths
        self.last_reload = time.time()

    def on_any_event(self, event):
        if not event.src_path.endswith(".py"):
            return
        if time.time() - self.last_reload < 2:  # avoid storm
            return
        self.last_reload = time.time()
        file_name = os.path.basename(event.src_path)
        log_ist(f"♻️ Detected change in {file_name} — triggering rerun", "INFO", Fore.YELLOW)
        try:
            st.toast(f"🔄 Auto-reloading due to {file_name}", icon="⚙️")
            st.rerun()
        except Exception as e:
            log_ist(f"⚠️ Reload failed: {e}", "ERROR", Fore.RED)

def start_auto_reload(watch_dir="."):
    """Start a background observer for live reload in development."""
    handler = AutoReloadHandler([watch_dir])
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=True)
    observer.daemon = True
    observer.start()
    log_ist(f"👀 Auto-reload active — watching {os.path.abspath(watch_dir)}", "INFO", Fore.CYAN)
    return observer

if "watchdog_started" not in st.session_state:
    observer = start_auto_reload(".")
    st.session_state.watchdog_started = True

# =====================================================
# 🧩 HEARTBEAT MONITOR THREAD
# =====================================================
def start_heartbeat(interval=600):
    """Thread that logs periodic heartbeat messages."""
    def _loop():
        while True:
            log_ist("💓 Heartbeat: System running fine.", "INFO", Fore.BLUE)
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    log_ist("💓 Heartbeat thread started", "INFO", Fore.BLUE)

if "heartbeat" not in st.session_state:
    start_heartbeat(interval=900)  # every 15 minutes
    st.session_state.heartbeat = True

# =====================================================
# ✅ FINAL BOOT MESSAGE
# =====================================================
log_ist("✅ VAHAN ALL-MAXED ULTRA+ Boot Complete", "INFO", Fore.GREEN)
st.toast("✅ All Systems Ready — Full Power Mode 🔋", icon="✅")

# =====================================================
# ⚙️ DYNAMIC SIDEBAR — ALL-MAXED ULTRA (v3.0)
# =====================================================
import os
import json
import time
import random
import platform
from datetime import date, datetime
from pathlib import Path
import streamlit as st
import pandas as pd

# ---------- SAFE DEFAULTS ----------
STORAGE_DIR = Path.home() / ".vahan"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
PRESETS_FILE = STORAGE_DIR / "sidebar_presets.json"

if "init_done" not in st.session_state:
    st.session_state["init_done"] = True
    st.session_state.setdefault("session_seed", random.randint(100000, 999999))
    st.session_state.setdefault("user_cookie", f"user_{random.randint(1000,9999)}")
    # sidebar defaults
    st.session_state.setdefault("from_year", None)
    st.session_state.setdefault("to_year", None)
    st.session_state.setdefault("time_period", "Yearly")
    st.session_state.setdefault("fitness_check", False)
    st.session_state.setdefault("enable_forecast", True)
    st.session_state.setdefault("enable_anomaly", True)
    st.session_state.setdefault("enable_cluster", True)
    st.session_state.setdefault("enable_ai", False)
    st.session_state.setdefault("auto_refresh", True)
    st.session_state.setdefault("watchdog_enabled", True)

# Toggle to hide/show entire sidebar UI (keeps state)
SHOW_SIDEBAR_UI = st.session_state.get("show_sidebar", False)

# ---------- small helper utilities ----------
def safe_load_presets():
    try:
        if PRESETS_FILE.exists():
            with PRESETS_FILE.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as e:
        st.warning(f"⚠️ Could not load presets: {e}")
    return {}

def safe_save_presets(presets: dict):
    try:
        with PRESETS_FILE.open("w", encoding="utf-8") as fh:
            json.dump(presets, fh, indent=2)
        return True
    except Exception as e:
        st.error(f"💥 Failed to save presets: {e}")
        return False

def to_df(json_obj, label_keys=("label",), value_key="value"):
    # Compatible with many API shapes (keeps original behavior)
    if isinstance(json_obj, dict) and "labels" in json_obj and "data" in json_obj:
        labels = json_obj["labels"]
        values = json_obj["data"]
        min_len = min(len(labels), len(values))
        return pd.DataFrame({"label": labels[:min_len], "value": values[:min_len]})
    data = json_obj.get("data", json_obj) if isinstance(json_obj, dict) else json_obj
    if isinstance(data, dict):
        data = [data]
    rows = []
    for item in data:
        label = None
        for k in label_keys:
            if k in item:
                label = item[k]
                break
        if label is None and label_keys:
            label = item.get(label_keys[0], None)
        value = item.get(value_key, None)
        if value is None:
            for vk in ("count", "value", "total"):
                if vk in item:
                    value = item[vk]
                    break
        if label is not None and value is not None:
            rows.append({"label": label, "value": value})
    return pd.DataFrame(rows)

# ---------- compact CSS (keeps app neat when sidebar hidden) ----------
HIDE_SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] {display: none !important;}
div[data-testid="collapsedControl"] {display: none !important;}
.stButton>button {min-height: 36px;}
.sidebar-title {font-weight:700; font-size:16px; margin-bottom:6px;}
.compact-input .stNumberInput>div>div>input {height:34px;}
</style>
"""
if not SHOW_SIDEBAR_UI:
    st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

# ---------- Bootstrap / header ----------
today = date.today()
default_from_year = today.year - 1

# ---------- Load presets ----------
_presets = safe_load_presets()
preset_names = sorted(list(_presets.keys()))

# =====================================================
# ⚙️ SIDEBAR CONTROL PANEL — ULTRA MAXED v4.0
# =====================================================
import streamlit as st
import time, platform
from datetime import date

today = date.today()
default_from_year = 2015

# -----------------------------------------------------
# 🧩 SAFE SESSION INITIALIZATION (once only)
# -----------------------------------------------------
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

init_state("show_sidebar", True)
init_state("session_seed", int(time.time()))
init_state("user_cookie", f"user_{int(time.time())}")
init_state("from_year", default_from_year)
init_state("to_year", today.year)
init_state("state_code", "")
init_state("rto_code", "")
init_state("vehicle_classes", "")
init_state("vehicle_makers", "")
init_state("time_period", "Yearly")
init_state("fitness_check", False)
init_state("vehicle_type", "")
init_state("enable_forecast", True)
init_state("enable_anomaly", True)
init_state("enable_cluster", True)
init_state("enable_ai", False)
init_state("base_url", "https://analytics.parivahan.gov.in")
init_state("timeout", 30)
init_state("export_format", "CSV")
init_state("auto_refresh", True)
init_state("enable_beta", False)

# -----------------------------------------------------
# 🚀 SIDEBAR UI (wrapped for safety)
# -----------------------------------------------------
try:
    st.sidebar.markdown("<div class='sidebar-title'>⚙️ Control Panel — ULTRA MAXED</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Top control buttons
    top_col1, top_col2 = st.sidebar.columns([3, 1])
    with top_col1:
        if st.sidebar.button("🔁 Toggle Sidebar UI"):
            st.session_state["show_sidebar"] = not st.session_state.get("show_sidebar", True)
            st.toast("🔀 Sidebar toggled")
            st.rerun()

    with top_col2:
        if st.sidebar.button("🧹 Clear Session"):
            keep = {k: st.session_state[k] for k in ["session_seed", "user_cookie"] if k in st.session_state}
            st.session_state.clear()
            st.session_state.update(keep)
            st.toast("🧹 Session cleared & preserved essentials")
            st.rerun()

    # -------------------------------------------------
    # 📊 DATA FILTERS
    # -------------------------------------------------
    # ---------- Data Filters ----------
    with st.sidebar.expander("📊 Data Filters", expanded=True):
        st.markdown("### Time range")

        # ✅ Always fall back to integers (never None)
        default_from_year = st.session_state.get("from_year", today.year - 1) or (today.year - 3)
        default_to_year = st.session_state.get("to_year", today.year) or today.year

        from_year = st.sidebar.number_input("From Year", min_value=2012, max_value=today.year, value=default_from_year)
        to_year = st.sidebar.number_input("To Year", min_value=from_year, max_value=today.year, value=today.year)
        state_code = st.sidebar.text_input("State Code (blank=All-India)", value="")
        rto_code = st.sidebar.text_input("RTO Code (0=aggregate)", value="0")
        vehicle_classes = st.sidebar.text_input("Vehicle Classes (e.g., 2W,3W,4W if accepted)", value="")
        vehicle_makers = st.sidebar.text_input("Vehicle Makers (comma-separated or IDs)", value="")
        time_period = st.sidebar.selectbox("Time Period", options=[0,1,2], index=0)
        fitness_check = st.sidebar.selectbox("Fitness Check", options=[0,1], index=0)
        vehicle_type = st.sidebar.text_input("Vehicle Type (optional)", value="")


    # -------------------------------------------------
    # 🧠 AI / FORECASTING
    # -------------------------------------------------
    with st.sidebar.expander("🧠 AI / Forecasting", expanded=False):
        enable_forecast = st.checkbox("📈 Forecasting", st.session_state.enable_forecast, key="input_enable_forecast")
        enable_anomaly = st.checkbox("⚠️ Anomaly Detection", st.session_state.enable_anomaly, key="input_enable_anomaly")
        enable_cluster = st.checkbox("🔍 Clustering", st.session_state.enable_cluster, key="input_enable_cluster")
        enable_ai = st.checkbox("🤖 AI Narratives", st.session_state.enable_ai, key="input_enable_ai")

        st.session_state.update({
            "enable_forecast": enable_forecast,
            "enable_anomaly": enable_anomaly,
            "enable_cluster": enable_cluster,
            "enable_ai": enable_ai
        })

        if enable_forecast and enable_anomaly:
            st.caption("📊 Forecasts will include anomaly markers.")
        if enable_cluster:
            st.caption("🧩 Cluster model adapts dynamically.")
        if enable_ai:
            st.caption("💬 AI narratives enabled — generating insights...")

        if st.button("🔒 Lock AI Settings"):
            st.session_state["_ai_locked"] = True
            st.toast("🧠 AI config locked for this session")

    # -------------------------------------------------
    # 🧭 API & Experimental Settings
    # -------------------------------------------------
    with st.sidebar.expander("🧭 API & Reports", expanded=False):
        st.text_input("🌐 API Base URL", st.session_state.base_url, key="input_base_url")
        st.slider("⏳ API Timeout (s)", 5, 120, st.session_state.timeout, key="input_timeout")
        st.selectbox("📤 Export Format", ["CSV", "XLSX", "PDF"],
                     index=["CSV", "XLSX", "PDF"].index(st.session_state.export_format),
                     key="input_export_format")
        st.checkbox("♻️ Auto-refresh on data change", value=st.session_state.auto_refresh, key="input_auto_refresh")

    with st.sidebar.expander("🧪 Experimental", expanded=False):
        enable_beta = st.checkbox("🧪 Enable Experimental Features", st.session_state.enable_beta, key="input_enable_beta")
        st.session_state.enable_beta = enable_beta
        if enable_beta:
            st.info("Experimental mode ON — unstable features enabled")

    # -------------------------------------------------
    # 💾 Preset Management
    # -------------------------------------------------
    with st.sidebar.expander("💾 Presets", expanded=False):
        presets = safe_load_presets()
        if presets:
            sel = st.selectbox("Load preset", ["-- none --"] + list(sorted(presets.keys())), index=0, key="preset_select")
            if sel and sel != "-- none --":
                if st.button("▶️ Apply Selected Preset"):
                    p = presets[sel]
                    for k, v in p.items():
                        st.session_state[k] = v
                    st.toast(f"Applied preset '{sel}'")
                    time.sleep(0.3)
                    st.rerun()

            del_name = st.text_input("Delete preset name", value="", key="preset_delname_input")
            if st.button("🗑️ Delete Preset"):
                dn = del_name.strip()
                if dn and dn in presets:
                    del presets[dn]
                    safe_save_presets(presets)
                    st.toast(f"Deleted preset '{dn}'")
                else:
                    st.error("Preset not found")
        else:
            st.caption("No presets saved yet — use 'Save Preset' above.")

    # -------------------------------------------------
    # 🧾 Footer Info
    # -------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.caption(f"🧩 User: `{st.session_state.user_cookie}` | 🔑 Seed: `{st.session_state.session_seed}`")
    st.sidebar.caption(f"📦 Python: {platform.python_version()} | Streamlit: {st.__version__}")

except Exception as e:
    st.sidebar.error(f"⚠️ Auto-recovered from sidebar error: {e}")
    import traceback
    traceback.print_exc()
    for k in ["preset_name_input", "preset_select", "preset_delname_input"]:
        if k in st.session_state:
            del st.session_state[k]
    time.sleep(0.2)
    st.rerun()

# =====================================================
# 🔁 LIVE AUTO-REFRESH (Reactive Engine — ULTRA MAXED v3.0)
# =====================================================
import streamlit as st
import threading
import time
from datetime import datetime, timedelta

# Ensure proper init
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "auto_refresh_interval" not in st.session_state:
    st.session_state["auto_refresh_interval"] = 120  # seconds
if "auto_refresh_thread" not in st.session_state:
    st.session_state["auto_refresh_thread"] = None
if "auto_refresh_last" not in st.session_state:
    st.session_state["auto_refresh_last"] = datetime.now()
if "auto_refresh_stop" not in st.session_state:
    st.session_state["auto_refresh_stop"] = False


# ---------- REFRESH FUNCTION ----------
def _auto_refresh_loop(interval=120):
    """Runs background refresh loop safely."""
    while not st.session_state.get("auto_refresh_stop", False):
        time.sleep(interval)
        st.session_state["_auto_refresh_trigger"] = datetime.now().isoformat()
        st.session_state["auto_refresh_last"] = datetime.now()
        # Toast works only when visible, so guard with try
        try:
            st.toast("🔁 Auto-refresh triggered!", icon="🕒")
        except Exception:
            pass
        st.rerun()


# ---------- CONTROL PANEL (VISIBLE AT TOP OR FOOTER) ----------
with st.sidebar.expander("🕒 Auto-Refresh Control", expanded=False):
    st.markdown("### 🔄 Live Data Auto-Refresh")
    st.caption("Keep your dashboard continuously synced with the latest data.")

    c1, c2 = st.columns([3, 1])
    with c1:
        interval = st.slider("⏳ Refresh Interval (sec)", 30, 600,
                             st.session_state.get("auto_refresh_interval", 120),
                             step=30, key="auto_refresh_interval")
    with c2:
        st.caption(" ")

    # Start/Stop buttons
    c3, c4 = st.columns(2)
    with c3:
        if st.button("▶️ Start Auto-Refresh"):
            st.session_state["auto_refresh_stop"] = False
            if st.session_state.get("auto_refresh_thread") is None or not st.session_state["auto_refresh_thread"]:
                thread = threading.Thread(
                    target=_auto_refresh_loop,
                    args=(st.session_state["auto_refresh_interval"],),
                    daemon=True
                )
                thread.start()
                st.session_state["auto_refresh_thread"] = True
                st.toast(f"🔁 Auto-refresh started ({interval}s interval).", icon="🟢")
    with c4:
        if st.button("⏹️ Stop Auto-Refresh"):
            st.session_state["auto_refresh_stop"] = True
            st.session_state["auto_refresh_thread"] = None
            st.toast("⏹️ Auto-refresh stopped.", icon="🔴")

    # Show status and time since last refresh
    if not st.session_state.get("auto_refresh_stop", False):
        elapsed = (datetime.now() - st.session_state.get("auto_refresh_last", datetime.now())).seconds
        next_refresh_in = st.session_state.get("auto_refresh_interval", 120) - elapsed
        if next_refresh_in < 0:
            next_refresh_in = 0
        st.markdown(f"**🕒 Next refresh in:** `{next_refresh_in}s`")
        st.progress(max(0, min(1, 1 - elapsed / st.session_state.get("auto_refresh_interval", 120))))
    else:
        st.info("⏸️ Auto-refresh paused.")


# ---------- AUTO START (if enabled) ----------
if st.session_state.get("auto_refresh", True) and not st.session_state.get("auto_refresh_stop", False):
    if "auto_refresh_thread" not in st.session_state or not st.session_state["auto_refresh_thread"]:
        thread = threading.Thread(
            target=_auto_refresh_loop,
            args=(st.session_state["auto_refresh_interval"],),
            daemon=True
        )
        thread.start()
        st.session_state["auto_refresh_thread"] = True
        st.toast(f"🔁 Auto-refresh active every {st.session_state['auto_refresh_interval']}s (ULTRA-MAXED).", icon="🕒")

# =====================================================
# 🚀 VAHAN ANALYTICS API ENGINE — ALL-MAXED PURE v7.0
# =====================================================
import os, re, json, time, random, logging, pickle, requests
from datetime import datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any
from contextlib import nullcontext

# =====================================================
# 🧩 STREAMLIT SAFE IMPORT
# =====================================================
try:
    import streamlit as st
except ImportError:
    class st:
        @staticmethod
        def expander(*_, **__): return nullcontext()
        @staticmethod
        def json(_): pass
        @staticmethod
        def spinner(_): return nullcontext()
        @staticmethod
        def warning(msg): print("[WARNING]", msg)
        @staticmethod
        def error(msg): print("[ERROR]", msg)

# =====================================================
# ⚙️ CONSTANTS
# =====================================================
BASE_URL = "https://analytics.parivahan.gov.in/analytics/publicdashboard"
MAX_RETRIES = 4
DEFAULT_TIMEOUT = 60

# =====================================================
# 🕒 UTILITIES — LOGGING + TIME
# =====================================================
def ist_now() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")

def log(msg: str, level: str = "INFO"):
    prefix = f"[IST {ist_now()}] [{level}]"
    line = f"{prefix} {msg}"
    color = {
        "INFO": "\033[94m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "SUCCESS": "\033[92m",
    }.get(level, "\033[0m")
    print(color + line + "\033[0m")
    try:
        if level == "ERROR":
            st.error(line)
        elif level == "WARNING":
            st.warning(line)
    except Exception:
        pass

# =====================================================
# 🧰 CLEAN STRING
# =====================================================
def clean_str(v: str) -> str:
    if not v: return ""
    return re.sub(r"[^A-Za-z0-9_\- ,./]", "", str(v).strip())

# =====================================================
# 🧩 PARAMETER BUILDER (NO PREFILL)
# =====================================================
def build_params(
    from_year: int,
    to_year: int,
    *,
    state_code: str,
    rto_code: str,
    vehicle_classes: str,
    vehicle_makers: str,
    time_period: str,
    fitness_check: bool,
    vehicle_type: str,
    extra_params: Optional[dict] = None,
) -> dict:
    """Strict builder — no prefilled defaults."""
    current_year = datetime.now().year
    errors = []

    # strict checks
    if from_year is None or to_year is None:
        errors.append("Both from_year and to_year must be provided.")
    if not isinstance(from_year, int) or not isinstance(to_year, int):
        errors.append("Year values must be integers.")
    if from_year > to_year:
        errors.append(f"From year ({from_year}) cannot exceed To year ({to_year}).")
    if from_year < 2000 or to_year > current_year:
        errors.append(f"Year range must be between 2000 and {current_year}.")
    if not isinstance(fitness_check, bool):
        errors.append("Fitness flag must be boolean (True/False).")
    if not all([state_code, rto_code, vehicle_classes, vehicle_makers, time_period, vehicle_type]):
        errors.append("All core parameters must be explicitly provided — no blanks allowed.")

    if errors:
        for e in errors:
            log(f"❌ Parameter Error: {e}", "ERROR")
        raise ValueError(" | ".join(errors))

    time_period = time_period.title().strip()
    if time_period not in ["Yearly", "Quarterly", "Monthly"]:
        log(f"⚠️ Invalid time_period '{time_period}', defaulting to 'Yearly'", "WARNING")
        time_period = "Yearly"

    params = {
        "from_year": from_year,
        "to_year": to_year,
        "state_cd": clean_str(state_code),
        "rto_cd": clean_str(rto_code),
        "vclass": clean_str(vehicle_classes),
        "maker": clean_str(vehicle_makers),
        "time_period": time_period,
        "include_fitness": "Y" if fitness_check else "N",
        "veh_type": clean_str(vehicle_type),
        "_session_seed": datetime.now().strftime("%Y%m%d%H%M%S"),
    }

    if extra_params:
        for k, v in extra_params.items():
            if v not in [None, "", [], {}]:
                params[k] = v

    params["_meta"] = {
        "created": ist_now(),
        "validated": True,
        "safe_hash": abs(hash(json.dumps(params, sort_keys=True))) % 1_000_000,
    }

    log(f"🧩 Params built successfully → hash {params['_meta']['safe_hash']}", "SUCCESS")

    try:
        with st.expander("🔍 VAHAN Parameter Summary", expanded=False):
            st.json(params)
    except Exception:
        pass

    return params


# =====================================================
# 📊 VAHAN INTELLIGENCE DASHBOARD — ALL-MAXED (REAL DATA EDITION)
# =====================================================
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# 🎨 GLOBAL STYLE + HEADER
# =====================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-weight: 700 !important;
}
.metric-card {
    background: linear-gradient(135deg, #004e92, #000428);
    color: white;
    padding: 1.2rem;
    border-radius: 1rem;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    transition: transform 0.2s ease-in-out;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}
.refresh-indicator {
    animation: blink 1.5s infinite alternate;
    color: #00eaff;
}
@keyframes blink {
    from { opacity: 0.4; }
    to { opacity: 1; }
}
hr.glow {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    border-radius: 2px;
    margin: 2rem 0;
}
footer {
    text-align: center;
    opacity: 0.8;
    font-size: 13px;
    padding: 1.5rem 0;
}
footer small {
    font-size: 11px;
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 🚗 HEADER
# =====================================================
st.markdown("""
<div style='text-align:center;margin-bottom:2rem;'>
  <h1>🚗 <b>Vahan Intelligence Dashboard</b></h1>
  <p>AI-Driven Analytics • Forecasts • Trends • All-India Transport Intelligence</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 🔁 REFRESH INDICATOR (Fixed - no conflict with log())
# =====================================================
ist_refresh_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%I:%M:%S %p")
st.markdown(f"""
<div style='text-align:center;margin-bottom:1rem;font-size:14px;'>
  <span class='refresh-indicator'>🔄 Auto-Refreshed at {ist_refresh_time} IST</span>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 🧾 FOOTER
# =====================================================
st.markdown("""
<footer>
  <hr class="glow">
  <div>
    ✨ <b>Vahan Intelligence Engine</b> — AI-Augmented Dashboard<br>
    <small>© 2025 Transport Data Division • Auto-refresh Enabled</small>
  </div>
</footer>
""", unsafe_allow_html=True)

# =====================================================
# 🎉 ONE-TIME LAUNCH TOAST
# =====================================================
if "dashboard_loaded" not in st.session_state:
    st.session_state["dashboard_loaded"] = True
    st.toast("🚀 Real VAHAN Dashboard Loaded (All-Maxed Edition)", icon="🌈")
    st.balloons()

# =====================================================
# 🤖 DEEPINFRA & UNIVERSAL AI — ALL-MAXED CONNECTOR v2.0
# =====================================================
import os
import json
import time
import random
import requests
import traceback
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# -------------------------
# 🔐 Load secrets/env safely
# -------------------------
load_dotenv()  # load .env if present

def get_secret(key: str, default: Optional[str] = "") -> str:
    # priority: st.secrets -> env -> default
    try:
        v = st.secrets.get(key)
        if v:
            return v
    except Exception:
        pass
    return os.getenv(key, default) or default

# Primary keys / config
DEEPINFRA_API_KEY = get_secret("DEEPINFRA_API_KEY", "")
DEEPINFRA_MODEL = get_secret("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
DEEPINFRA_TIMEOUT = int(get_secret("DEEPINFRA_TIMEOUT", "8"))

AI_PROVIDER = get_secret("AI_PROVIDER", "deepinfra").lower()  # 'deepinfra' by default
AI_MODEL = get_secret("AI_MODEL", DEEPINFRA_MODEL if AI_PROVIDER == "deepinfra" else get_secret("OPENAI_MODEL","gpt-4o-mini"))
AI_API_KEY = get_secret("AI_API_KEY", "")  # generic key fallback
AI_TIMEOUT = int(get_secret("AI_TIMEOUT", "20"))

# endpoint mapping (basic)
ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepinfra": "https://api.deepinfra.com/v1/openai/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    # 'gemini' usually uses Google client libs — leaving generic root for reachability checks
    "gemini": "https://generativelanguage.googleapis.com/v1beta1/models",
}

AI_URL = ENDPOINTS.get(AI_PROVIDER, ENDPOINTS["openai"])

# =====================================================
# 🧠 SAFE SIDEBAR TOGGLE (ALL-MAXED v3 — ULTRA EDITION)
# =====================================================
import streamlit as st
from datetime import datetime

# -----------------------------------------------------
# 🎨 Sidebar header
# -----------------------------------------------------
st.sidebar.markdown(
    """
    <div style='padding:8px 10px;
                border-radius:8px;
                background:linear-gradient(90deg,#0b3d91,#1a73e8);
                color:white;
                font-weight:700;
                text-shadow:0 0 6px rgba(255,255,255,0.3);'>
        🤖 DeepInfra / Universal AI Connector
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# 🧩 Initialize State (Safe, Idempotent)
# -----------------------------------------------------
def init_state(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default

init_state("enable_ai", True)
init_state("last_ai_check", None)
init_state("ai_status", "idle")

# -----------------------------------------------------
# 🧠 Unified Toggle Component (Duplicate-Proof)
# -----------------------------------------------------
def safe_checkbox(label, session_key, default=True, help_text=None):
    """Safely create a checkbox synced with session_state (duplicate-safe)."""
    widget_key = f"{session_key}_widget"  # Unique key per widget
    val = st.sidebar.checkbox(label, key=widget_key, value=st.session_state.get(session_key, default), help=help_text)
    if val != st.session_state[session_key]:
        st.session_state[session_key] = val
    return val

# -----------------------------------------------------
# 🚀 AI Enable Toggle
# -----------------------------------------------------
enable_ai = safe_checkbox(
    "Enable AI",
    "enable_ai",
    default=True,
    help_text="Toggle DeepInfra / Universal AI connector",
)

# -----------------------------------------------------
# 📊 Dynamic Status Display (Instant Feedback)
# -----------------------------------------------------
if enable_ai:
    st.session_state.ai_status = "active"
    status_color = "#16c784"
    emoji = "✅"
    status_text = "AI Enabled — DeepInfra connector active."
else:
    st.session_state.ai_status = "disabled"
    status_color = "#f39c12"
    emoji = "⚠️"
    status_text = "AI Disabled — manual mode only."

# -----------------------------------------------------
# 🕒 Last AI Check Timestamp (optional tracking)
# -----------------------------------------------------
if enable_ai:
    st.session_state.last_ai_check = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------------------------------
# 💬 Sidebar Visual Status Badge
# -----------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style='margin-top:8px;
                padding:10px 12px;
                border-radius:10px;
                background:{status_color}20;
                border:1px solid {status_color};
                color:{status_color};
                font-weight:600;
                text-align:center;'>
        {emoji} {status_text}<br>
        <small style='color:#888;'>Last check: {st.session_state.last_ai_check or "—"}</small>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# 🔧 Future Expansion Example
# -----------------------------------------------------
# (add other connectors dynamically)
# st.sidebar.toggle("Enable OpenAI", key="enable_openai")
# st.sidebar.toggle("Enable HuggingFace", key="enable_hf")


# -------------------------
# 🔌 Ping / connectivity checks
# -------------------------
def _now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ping_provider(provider: str = AI_PROVIDER, api_key: str = None, timeout: int = AI_TIMEOUT):
    """Ping provider root (non intrusive). Returns (status_code_or_str, latency_ms_or_None)."""
    url = AI_URL.split("/v1")[0] if "/" in AI_URL else AI_URL
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        start = time.time()
        resp = requests.get(url, headers=headers, timeout=timeout)
        latency = round((time.time() - start) * 1000, 1)
        return resp.status_code, latency
    except requests.exceptions.Timeout:
        return "timeout", None
    except Exception as e:
        return f"error: {str(e)[:120]}", None

# show connectivity card
def render_provider_card():
    if not enable_ai:
        st.sidebar.info("AI Mode: Disabled")
        return

    key_to_show = AI_API_KEY or (DEEPINFRA_API_KEY if AI_PROVIDER == "deepinfra" else "")
    has_key = bool(key_to_show)
    if not has_key:
        st.sidebar.error("❌ No API key found. Add to st.secrets or environment (AI_API_KEY / DEEPINFRA_API_KEY).")
        return

    status, latency = ping_provider(api_key=key_to_show)
    st.session_state.last_ai_check = _now_ts()
    if status == 200:
        st.sidebar.success(f"✅ {AI_PROVIDER.title()} reachable — {latency} ms")
        st.sidebar.caption(f"Model: **{AI_MODEL}** | Last check: {st.session_state.last_ai_check}")
    elif status == "timeout":
        st.sidebar.error("⏱️ Provider timed out")
    elif isinstance(status, int) and status == 401:
        st.sidebar.error("🚫 Unauthorized — invalid API key")
    else:
        st.sidebar.warning(f"⚠️ Status: {status} | Last check: {st.session_state.last_ai_check}")

render_provider_card()
st.sidebar.markdown("---")

# -------------------------
# 🧠 Session chat memory
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# ⚙️ Universal Chat (safe, streaming-friendly)
# -------------------------
def universal_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    stream: bool = True,
    temperature: float = 0.2,
    max_tokens: int = 512,
    retries: int = 3,
    timeout: int = AI_TIMEOUT,
) -> Dict[str, Any]:
    """
    Returns dict with keys: {ok:bool, text:str, error:Optional[str]}
    Streamed incremental updates are shown in-place (st.empty()).
    """
    provider = (provider or AI_PROVIDER).lower()
    model = model or AI_MODEL
    api_key = api_key or (AI_API_KEY if AI_API_KEY else (DEEPINFRA_API_KEY if provider=="deepinfra" else None))
    url = ENDPOINTS.get(provider, ENDPOINTS["openai"])
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build payload for OpenAI-style endpoints (DeepInfra compatible)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": bool(stream),
    }

    # Fallback adjustments for Anthropic style or provider differences could be added here
    attempt = 0
    placeholder = st.empty()
    last_text = ""

    while attempt < retries:
        attempt += 1
        try:
            with requests.post(url, headers=headers, json=payload, stream=stream, timeout=timeout) as r:
                if r.status_code != 200:
                    # Show a helpful short message
                    snippet = r.text[:400].strip().replace("\n"," ")
                    placeholder.warning(f"⚠️ Provider responded {r.status_code}: {snippet}")
                    # For 4xx/5xx, may retry depending on code
                    if 400 <= r.status_code < 500:
                        return {"ok": False, "error": f"HTTP {r.status_code}", "text": ""}
                    # else try again after backoff
                    raise RuntimeError(f"HTTP {r.status_code} - {snippet}")

                if stream:
                    # stream lines (OpenAI/DeepInfra style)
                    for raw in r.iter_lines(decode_unicode=True):
                        if not raw:
                            continue
                        # typical stream: 'data: {...}'
                        line = raw.strip()
                        if line.startswith("data:"):
                            payload_line = line[len("data:"):].strip()
                        else:
                            payload_line = line
                        if payload_line == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload_line)
                            # try common locations for delta/content
                            choices = obj.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta", {})
                                chunk = delta.get("content") or delta.get("text") or ""
                                if chunk:
                                    last_text += chunk
                                    placeholder.markdown(f"**🤖 AI:** {last_text}")
                        except Exception:
                            # not JSON or unknown chunk — append raw
                            last_text += payload_line
                            placeholder.markdown(f"**🤖 AI:** {last_text}")

                    # finish
                    st.session_state.chat_history.append({"user": user_prompt, "ai": last_text, "ts": _now_ts()})
                    st.toast("✅ AI response complete", icon="🤖")
                    return {"ok": True, "text": last_text, "error": None}

                else:
                    # Non-streaming path
                    data = r.json()
                    # try to extract the main text in common schema
                    text = ""
                    if "choices" in data:
                        try:
                            text = data["choices"][0]["message"]["content"]
                        except Exception:
                            text = json.dumps(data)[:1000]
                    else:
                        text = json.dumps(data)[:1000]
                    placeholder.markdown(f"**🤖 AI:** {text}")
                    st.session_state.chat_history.append({"user": user_prompt, "ai": text, "ts": _now_ts()})
                    return {"ok": True, "text": text, "error": None}

        except Exception as e:
            backoff = (2 ** attempt) + random.random()
            placeholder.error(f"⚠️ Attempt {attempt} failed: {str(e)[:120]}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            continue

    # final failure
    placeholder.error("⛔ AI request failed after retries.")
    return {"ok": False, "text": "", "error": "failed after retries"}

# -------------------------
# 🧩 Utility: insight generator (non-blocking UX)
# -------------------------
def universal_insight(df, topic="dataset"):
    """
    Generate a short insight summary about the DataFrame.
    This uses a small sample and runs the chat (may be slow depending on provider).
    """
    if df is None or getattr(df, "empty", True):
        st.warning("No data available for AI insight.")
        return
    sample_md = df.head(8).to_markdown(index=False)
    prompt = f"You're a senior data analyst. Given this sample of {topic} (markdown table):\n\n{sample_md}\n\nProvide a concise summary (3 bullets): key trends, notable anomalies, and 2 short recommendations."
    return universal_chat("You are a data analytics expert.", prompt, stream=True, max_tokens=300)

# -------------------------
# 🧪 Test / UI helpers
# -------------------------
def universal_test_ui():
    st.subheader("🧪 AI Connector Test")
    if enable_ai:
        masked = (AI_API_KEY or DEEPINFRA_API_KEY) or ""
        masked_display = f"{masked[:4]}...{masked[-4:]}" if masked else "—"
        st.info(f"Provider: **{AI_PROVIDER}** | Model: **{AI_MODEL}** | Key: `{masked_display}`")

        if st.button("🔗 Ping Provider"):
            status, latency = ping_provider()
            if latency:
                st.success(f"Reachable — {status} | {latency} ms")
            else:
                st.warning(f"Status: {status}")
    else:
        st.info("AI Mode is currently disabled in sidebar.")

    with st.expander("💬 Quick test prompt"):
        prompt = st.text_area("Prompt", "Summarize the state of registrations given a short table sample.")
        if st.button("🚀 Run test prompt"):
            if prompt.strip():
                universal_chat("You are a helpful assistant.", prompt, stream=True, max_tokens=240)
            else:
                st.warning("Enter a prompt first.")

# Safe helper
def set_query_param(key, value):
    try:
        st.query_params[key] = value  # Streamlit ≥1.40
    except TypeError:
        st.experimental_set_query_params(**{key: value})  # Older versions

def get_query_param(key, default=None):
    try:
        return st.query_params.get(key, default)  # Streamlit ≥1.40
    except TypeError:
        return st.experimental_get_query_params().get(key, [default])[0]

# Use safely
if st.sidebar.button("Open AI Test Panel"):
    set_query_param("_open_ai_test", "1")

if get_query_param("_open_ai_test"):
    universal_test_ui()

# -------------------------
# ✅ End of ALL-MAXED AI connector
# -------------------------

# import plotly.express as px
# import altair as alt
# import matplotlib.pyplot as plt

# # ---------- Fetch & display category + makers ----------------------------------
# st.subheader("📊 Vehicle Categories & Top Makers")

# try:
#     with st.spinner("Fetching Category distribution..."):
#         cat_json, cat_url = get_json("vahandashboard/categoriesdonutchart", params)
#         df_cat = to_df(cat_json)

#     col1, col2 = st.columns([2, 3])

#     # ---- Category distribution ----
#     with col1:
#         st.markdown("### 🚗 Category Distribution")

#         if not df_cat.empty:
#             try:
#                 fig = px.bar(
#                     df_cat,
#                     x="label",
#                     y="value",
#                     color="label",
#                     title="Vehicle Category Distribution",
#                     text_auto=True,
#                 )
#                 fig.update_layout(
#                     xaxis_title="Category",
#                     yaxis_title="Count",
#                     showlegend=False,
#                     template="plotly_white",
#                 )
#                 st.plotly_chart(fig, use_container_width=True)

#             except Exception as e:
#                 st.warning(f"⚠️ Plotly failed ({e}). Falling back to Altair.")
#                 alt_chart = (
#                     alt.Chart(df_cat)
#                     .mark_bar()
#                     .encode(
#                         x="label:N",
#                         y="value:Q",
#                         color="label:N",
#                         tooltip=["label", "value"],
#                     )
#                     .properties(title="Vehicle Category Distribution")
#                 )
#                 st.altair_chart(alt_chart, use_container_width=True)
#         else:
#             st.info("ℹ️ No category data found for the given filters.")

# =========================================================
# 🔥 ALL-MAXED — Category Analytics (multi-frequency, multi-year) — MAXED BLOCK
# Drop-in Streamlit module. Replace or import into your app.
# Created: ALL-MAXED v2 — resilient, instrumented, cached, mock-safe
# =========================================================
import time
import math
import json
import random
import logging
import requests
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger("all_maxed_category")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.DEBUG)

# =====================================================
# 🚀 ALL-MAXED ANALYTICS CORE v12.0
# -----------------------------------------------------
# Fully loaded mock data, chart builders, and analytics UI helpers
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import random, uuid
import plotly.express as px
from datetime import datetime
from typing import Dict, Any
from plotly.colors import qualitative

# -----------------------------------------------------
# 🎯 Master Category Reference
# -----------------------------------------------------
CATEGORIES_MASTER = [
    "Motorcycle", "Car", "Truck", "Bus", "Tractor",
    "E-Rickshaw", "Trailer", "Pickup", "Ambulance", "Taxi"
]

# -----------------------------------------------------
# 💾 Deterministic Mock Data Generator (Multi-Frequency)
# -----------------------------------------------------
def deterministic_mock_categories(year: int, freq: str = "Monthly", seed_base: str = "categories") -> Dict[str, Any]:
    """Generate reproducible, realistic mock data for categories (daily, monthly, yearly)."""
    rnd = random.Random(hash((year, seed_base)) & 0xFFFFFFFF)
    data = []

    if freq == "Yearly":
        for c in CATEGORIES_MASTER:
            val = rnd.randint(50_000, 2_500_000)
            data.append({"label": c, "value": val, "year": year})

    elif freq == "Monthly":
        for month in range(1, 13):
            for c in CATEGORIES_MASTER:
                base = rnd.randint(10_000, 200_000)
                seasonal_boost = 1.2 if month in [3, 9, 12] else 1.0
                value = int(base * seasonal_boost * (0.8 + rnd.random() * 0.5))
                data.append({
                    "label": c,
                    "value": value,
                    "year": year,
                    "month": month,
                    "month_name": datetime(year, month, 1).strftime("%b")
                })

    elif freq == "Daily":
        for month in range(1, 13):
            for day in range(1, 29):
                for c in CATEGORIES_MASTER:
                    base = rnd.randint(200, 15000)
                    val = int(base * (0.8 + rnd.random() * 0.6))
                    data.append({
                        "label": c, "value": val, "year": year, "month": month, "day": day
                    })

    return {
        "data": data,
        "meta": {
            "generatedAt": datetime.utcnow().isoformat(),
            "note": f"deterministic mock for {year} ({freq})",
            "freq": freq
        }
    }

# -----------------------------------------------------
# ⚙️ Formatting helpers
# -----------------------------------------------------
def format_number(n):
    """Return number formatted as K, M, or Cr."""
    if n >= 10_000_000:
        return f"{n/10_000_000:.2f} Cr"
    elif n >= 100_000:
        return f"{n/100_000:.2f} L"
    elif n >= 1_000:
        return f"{n/1_000:.2f} K"
    return f"{n:,}"

# -----------------------------------------------------
# 🎨 Global Chart Style Settings
# -----------------------------------------------------
COLOR_PALETTE = qualitative.Plotly + qualitative.D3 + qualitative.Vivid
DEFAULT_TEMPLATE = "plotly_white"
TITLE_STYLE = dict(size=20, color="#111", family="Segoe UI Semibold")

# -----------------------------------------------------
# 🧩 MAXED CHART HELPERS (Legend, Hover, UI polish)
# -----------------------------------------------------
def _unique_key(prefix="chart"):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def bar_from_df(df: pd.DataFrame, title="Bar Chart", x="label", y="value",
                color=None, barmode="group", height=500, section_id="bar"):
    """Enhanced bar chart with full UX polish."""
    if df is None or df.empty:
        st.warning("⚠️ No data to plot.")
        return

    fig = px.bar(
        df, x=x, y=y, color=color or x, text_auto=".2s",
        title=title, color_discrete_sequence=COLOR_PALETTE,
        barmode=barmode
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} registrations",
        textfont_size=12, textangle=0, cliponaxis=False
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title_font=TITLE_STYLE,
        xaxis_title=x.title(),
        yaxis_title=y.title(),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, title=None, bgcolor="rgba(0,0,0,0)"
        ),
        height=height, bargap=0.2, margin=dict(t=60, b=40, l=40, r=20)
    )
    st.plotly_chart(fig, use_container_width=True, key=_unique_key(section_id))


def pie_from_df(df: pd.DataFrame, title="Pie Chart", donut=True, section_id="pie", height=450):
    """Enhanced donut/pie chart with interactivity + auto legends."""
    if df is None or df.empty:
        st.warning("⚠️ No data to plot.")
        return

    fig = px.pie(
        df, names="label", values="value", hole=0.45 if donut else 0.0,
        title=title, color_discrete_sequence=COLOR_PALETTE
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} registrations<br>%{percent}",
        pull=[0.03] * len(df),
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title_font=TITLE_STYLE,
        legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=0),
        height=height, margin=dict(t=60, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True, key=_unique_key(section_id))


def trend_from_df(df: pd.DataFrame, title="Trend Over Time", section_id="trend", height=500):
    """Advanced line chart supporting animation + multiple years."""
    if df is None or df.empty:
        st.warning("⚠️ No trend data available.")
        return

    if "month_name" in df.columns:
        df["month_order"] = pd.Categorical(
            df["month_name"], 
            categories=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            ordered=True
        )

    fig = px.line(
        df, x="month_order" if "month_order" in df.columns else "year",
        y="value", color="label", markers=True,
        title=title, color_discrete_sequence=COLOR_PALETTE,
        line_shape="spline"
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,.0f} registrations")
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title_font=TITLE_STYLE,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"
        ),
        height=height, margin=dict(t=60, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True, key=_unique_key(section_id))

# -----------------------------------------------------
# 🧠 Auto Dashboard Section — Single Function
# -----------------------------------------------------
def render_category_dashboard(year: int, freq="Monthly"):
    """Render full UI: fetch mock data, show KPI, bar, pie, trend — ALL-MAXED."""
    st.subheader(f"📊 Category Distribution — {year} ({freq})")

    # Generate deterministic data
    mock_json = deterministic_mock_categories(year, freq=freq)
    df = pd.DataFrame(mock_json["data"])

    total = df["value"].sum()
    top = df.sort_values("value", ascending=False).iloc[0]
    st.success(f"🏆 **Top Category:** {top['label']} — {format_number(top['value'])} registrations")
    st.caption(f"Total: {format_number(total)} | Generated: {mock_json['meta']['generatedAt']}")

    # Layout 2-col + trend
    c1, c2 = st.columns([2, 1])
    with c1:
        bar_from_df(df, title=f"{year} {freq} Breakdown (Bar)", color="label", section_id=f"bar_{year}")
    with c2:
        pie_from_df(df, title=f"{year} Share (Donut)", section_id=f"pie_{year}")

    # Optional trend
    if "month_name" in df.columns:
        trend_from_df(df, title=f"{year} Monthly Trend (Animated)", section_id=f"trend_{year}")
    else:
        trend_from_df(df, title=f"{year} Category Trend", section_id=f"trend_{year}")

    return df


# ============================================================
# ⚙️ Synthetic Timeseries Expansion — MAXED ULTRA VERSION
# ------------------------------------------------------------
# Expands per-category annual totals into realistic timeseries
# with seasonality, trend variation, and reproducible randomness
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime

def year_to_timeseries(
    df_year: pd.DataFrame,
    year: int,
    freq: str = "Monthly",
    trend_strength: float = 0.15,
    noise_strength: float = 0.10,
    seasonal_boost: bool = True,
    seed_base: str = "timeseries"
) -> pd.DataFrame:
    """
    Expand per-category totals into an enhanced, realistic synthetic timeseries.
    
    Parameters
    ----------
    df_year : pd.DataFrame
        Must contain columns ["label", "value"].
    year : int
        Target year.
    freq : str
        One of ["Daily", "Monthly", "Quarterly", "Yearly"].
    trend_strength : float
        Amplitude of upward/downward trend variation (0.0 = flat).
    noise_strength : float
        Random noise amplitude applied to each point.
    seasonal_boost : bool
        If True, apply seasonal peaks around Mar, Jun, Sep, Dec.
    seed_base : str
        Used to derive deterministic seed per (year, freq).
        
    Returns
    -------
    pd.DataFrame
        Columns: ds, label, value, year, month, quarter, month_name
    """
    if df_year is None or df_year.empty:
        return pd.DataFrame(columns=["ds","label","value","year","month","quarter","month_name"])

    # --- deterministic seed
    seed = abs(hash((year, freq, seed_base))) % (2**32)
    rng = np.random.default_rng(seed)

    # --- index generation
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    freq = freq.capitalize()
    if freq == "Daily":
        idx = pd.date_range(start=start, end=end, freq="D")
    elif freq == "Monthly":
        idx = pd.date_range(start=start, end=end, freq="M")
    elif freq == "Quarterly":
        idx = pd.date_range(start=start, end=end, freq="Q")
    else:
        idx = pd.date_range(start=start, end=end, freq="Y")

    n = len(idx)
    rows = []

    # --- helper: seasonal curve (sinusoidal)
    def seasonal_factor(i):
        if not seasonal_boost:
            return 1.0
        return 1.0 + 0.25 * np.sin((i / n) * 2 * np.pi * 4)  # 4 seasonal peaks

    for _, r in df_year.iterrows():
        cat = r.get("label", "Unknown")
        total = float(r.get("value", 0.0))
        if total <= 0:
            continue

        base_per = total / max(1, n)

        # random trend (upward or downward)
        trend = np.linspace(1 - trend_strength, 1 + trend_strength, n)
        if rng.random() > 0.5:
            trend = trend[::-1]

        # small random noise each step
        noise = rng.normal(0, noise_strength, n)

        # --- compute values
        vals = base_per * trend * (1 + noise)
        vals = np.maximum(vals, 0)

        for i, ts in enumerate(idx):
            factor = seasonal_factor(i)
            v = vals[i] * factor
            rows.append({
                "ds": ts,
                "label": cat,
                "value": float(v),
                "year": int(year),
                "month": ts.month,
                "quarter": ts.quarter,
                "month_name": ts.strftime("%b")
            })

    df_out = pd.DataFrame(rows)
    df_out["value"] = df_out["value"].round(2)

    # --- safety normalization: ensure yearly totals remain consistent
    grouped = df_out.groupby("label")["value"].sum().to_dict()
    for cat in grouped:
        original_total = float(df_year.loc[df_year["label"] == cat, "value"].iloc[0])
        if grouped[cat] > 0:
            df_out.loc[df_out["label"] == cat, "value"] *= original_total / grouped[cat]

    df_out.reset_index(drop=True, inplace=True)
    return df_out



# -------------------------
# Fetch block: multi-year category fetch (maxed)
# -------------------------

# ============================================================
# 🚘 CATEGORY FETCHER — ALL-MAXED ULTRA
# ------------------------------------------------------------
# Robust, self-healing fetcher + renderer for per-year category
# distributions with complete UX polish, analytics, and fallbacks.
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import random
from datetime import datetime
from colorama import Fore

def fetch_year_category(year: int, params: dict, show_debug: bool = True) -> pd.DataFrame:
    """Fetch category donut for a given year and render local charts, insights, and summaries.

    ✅ Always returns a non-empty DataFrame with ['label','value','year'].
    ✅ Includes deterministic mock fallback and rich Plotly visualizations.
    """

    # --- Prepare request ---
    p = params.copy() if params else {}
    p["year"] = int(year)

    st.markdown(f"## 📊 Vehicle Categories — {year}")

    # --- Fetch safely ---
    try:
        cat_json, cat_url = get_json("vahandashboard/categoriesdonutchart", p)
    except Exception as e:
        logger.exception(Fore.RED + f"❌ get_json failed for year {year}: {e}")
        cat_json, cat_url = deterministic_mock_categories(year), f"mock://categoriesdonutchart/{year}"

    # --- Debug panel ---
    if show_debug:
        with st.expander(f"🧩 Debug JSON — Categories {year}", expanded=False):
            st.write("**URL:**", cat_url)
            st.json(cat_json if isinstance(cat_json, (dict, list)) else str(cat_json))

    # --- Normalize JSON to DataFrame ---
    try:
        df = to_df(cat_json)
    except Exception as e:
        logger.warning(Fore.YELLOW + f"⚠️ to_df failed for {year}: {e}")
        df = to_df(deterministic_mock_categories(year))

    if df is None or df.empty:
        logger.warning(Fore.YELLOW + f"⚠️ Empty df for {year}, generating deterministic mock")
        df = to_df(deterministic_mock_categories(year))

    df = df.copy()
    df["year"] = int(year)

    # --- Data quality & totals ---
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df = df.sort_values("value", ascending=False)
    total_reg = int(df["value"].sum())

    st.caption(f"🔗 **Source:** {cat_url}")
    st.markdown(f"**Total Registrations ({year}):** {total_reg:,}")

    # --- Charts layout ---
    c1, c2 = st.columns([1.8, 1.2])
    with c1:
        try:
            fig_bar = px.bar(
                df,
                x="label",
                y="value",
                color="label",
                text_auto=".2s",
                title=f"🚗 Category Distribution — {year}",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_bar.update_layout(
                template="plotly_white",
                showlegend=False,
                margin=dict(t=60, b=40, l=40, r=40),
                title_font=dict(size=20, family="Segoe UI", color="#222"),
                height=450,
                xaxis_title="Category",
                yaxis_title="Registrations",
                bargap=0.2,
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{year}")
        except Exception as e:
            st.warning(f"⚠️ Bar chart failed: {e}")
            st.dataframe(df)

    with c2:
        try:
            fig_pie = px.pie(
                df,
                names="label",
                values="value",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title=f"Category Share — {year}",
            )
            fig_pie.update_traces(
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>%{value:,} registrations<br>%{percent}",
                pull=[0.05]*len(df),
            )
            fig_pie.update_layout(
                template="plotly_white",
                margin=dict(t=40, b=20, l=20, r=20),
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{year}")
        except Exception as e:
            st.warning(f"⚠️ Pie chart failed: {e}")
            st.dataframe(df)

    # --- Top category insight ---
    try:
        top = df.iloc[0]
        pct = (top["value"] / total_reg) * 100 if total_reg else 0
        st.success(f"🏆 **Top Category:** {top['label']} — {int(top['value']):,} registrations ({pct:.1f}%)")
    except Exception:
        st.warning("⚠️ Could not determine top category")

    # --- Extra insights table ---
    df["share_%"] = (df["value"] / total_reg * 100).round(2)
    st.dataframe(
        df.style.format({"value": "{:,.0f}", "share_%": "{:.2f}%"}).bar(
            subset=["share_%"], color="#4CAF50"
        ),
        use_container_width=True,
        height=320,
    )

    # --- Minor animations / expansion ---
    with st.expander("📈 Trend simulation (synthetic)", expanded=False):
        df_ts = year_to_timeseries(df, year, freq="Monthly")
        fig_line = px.line(
            df_ts,
            x="ds",
            y="value",
            color="label",
            line_group="label",
            title=f"Synthetic Monthly Trend — {year}",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_line.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5),
            height=400,
        )
        st.plotly_chart(fig_line, use_container_width=True, key=f"trend_{year}")

    return df

# =====================================================
# -------------------------
# Main Streamlit UI — All-Maxed Block
# -------------------------
# =====================================================
def all_maxed_category_block(params: Optional[dict] = None):
    import numpy as np 
    import pandas as pd 
    import plotly.express as px 
    import plotly.graph_objects as go 
    import time, math, json 
    from dateutil.relativedelta import relativedelta 
    import streamlit as st
    """Render the maxed category analytics block inside Streamlit.

    Provide `params` to pass to API calls (e.g., region filters). If omitted, defaults are used.
    """
    start_overall = time.time()
    params = params or {}

    st.markdown("## 🚗 ALL-MAXED — Category Analytics (Multi-frequency, Multi-year)")

    # -------------------------
    # Controls — All Maxed (with unique keys)
    # -------------------------
    freq = st.radio(
        "Aggregation Frequency",
        ["Daily", "Monthly", "Quarterly", "Yearly"],
        index=1,
        horizontal=True,
        key="allmaxed_freq_radio"
    )
    
    mode = st.radio(
        "View Mode",
        ["Separate (Small Multiples)", "Combined (Overlay / Stacked)"],
        index=1,
        horizontal=True,
        key="allmaxed_mode_radio"
    )
    
    current_year = datetime.now().year
    
    start_year = st.number_input(
        "From Year",
        min_value=2010,
        max_value=current_year,
        value=current_year-1,
        key="allmaxed_start_year"
    )
    
    end_year = st.number_input(
        "To Year",
        min_value=start_year,
        max_value=current_year,
        value=current_year,
        key="allmaxed_end_year"
    )
    
    years = list(range(int(start_year), int(end_year)+1))
    
    show_heatmap = st.checkbox(
        "Show Heatmap (year × category)",
        value=True,
        key="allmaxed_show_heatmap"
    )
    
    show_radar = st.checkbox(
        "Show Radar (per year)",
        value=True,
        key="allmaxed_show_radar"
    )
    
    do_forecast = st.checkbox(
        "Enable Forecasting",
        value=True,
        key="allmaxed_do_forecast"
    )
    
    do_anomaly = st.checkbox(
        "Enable Anomaly Detection",
        value=False,
        key="allmaxed_do_anomaly"
    )
    
    do_clustering = st.checkbox(
        "Enable Clustering (KMeans)",
        value=False,
        key="allmaxed_do_clustering"
    )
    
    enable_ai = st.checkbox(
        "Enable AI Narrative (requires provider)",
        value=False,
        key="allmaxed_enable_ai"
    )

    st.info(f"🚀 Starting ALL-MAXED category pipeline (debug ON) — years: {years} | freq: {freq} | mode: {mode}")

    # -------------------------
    # Fetch multi-year category data
    # -------------------------
    all_year_dfs = []
    with st.spinner("Fetching category data for selected years..."):
        for y in years:
            try:
                df_y = fetch_year_category(y, params, show_debug=False)
                if df_y is None or df_y.empty:
                    st.warning(f"No category data for {y}")
                    continue
                all_year_dfs.append(df_y)
            except Exception as e:
                logger.exception(f"Error fetching {y}: {e}")
                st.error(f"Error fetching {y}: {e}")

    if not all_year_dfs:
        st.info("No category data loaded for selected range. Displaying deterministic mocks for demonstration.")
        # Generate mocks
        for y in years:
            all_year_dfs.append(to_df(deterministic_mock_categories(y)).assign(year=y))

    df_cat_all = pd.concat(all_year_dfs, ignore_index=True)

    # -------------------------
    # Frequency expansion -> time series (synthetic if needed)
    # -------------------------
    ts_list = []
    for y in sorted(df_cat_all["year"].unique()):
        df_y = df_cat_all[df_cat_all["year"]==y].reset_index(drop=True)
        ts = year_to_timeseries(df_y, int(y), freq=freq)
        ts_list.append(ts)
    df_ts = pd.concat(ts_list, ignore_index=True) if ts_list else pd.DataFrame(columns=["ds","label","value","year"])
    df_ts["ds"] = pd.to_datetime(df_ts["ds"])

    # Resample to requested frequency (group-by label)
    if freq == "Daily":
        resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq="D")])["value"].sum().reset_index()
    elif freq == "Monthly":
        resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq="M")])["value"].sum().reset_index()
    elif freq == "Quarterly":
        resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq="Q")])["value"].sum().reset_index()
    else:
        resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq="Y")])["value"].sum().reset_index()

    resampled["year"] = resampled["ds"].dt.year

    pivot = resampled.pivot_table(index="ds", columns="label", values="value", aggfunc="sum").fillna(0)
    pivot_year = resampled.pivot_table(index="year", columns="label", values="value", aggfunc="sum").fillna(0)

    # =====================================================
    # 📊 VISUALIZATIONS (ALL-MAXED)
    # =====================================================
    st.subheader("📊 Visualizations — Multi-year & Multi-frequency (All-Maxed)")
    
    # --- Safety Checks ---
    if "resampled" not in locals() or resampled is None or resampled.empty:
        st.warning("⚠️ No valid 'resampled' data available for visualization.")
        st.stop()
    
    if "pivot" not in locals() or pivot is None or pivot.empty:
        st.warning("⚠️ No valid 'pivot' data available for visualization.")
        st.stop()
    
    if "mode" not in locals():
        mode = "Combined (Overlay / Stacked)"
    
    # -------------------------
    # Combined View
    # -------------------------
    if mode.startswith("Combined"):
        st.markdown("### 🌈 Stacked & Overlay Trends — Combined View")
    
        # --- Stacked Area Chart ---
        try:
            fig_area = px.area(
                resampled,
                x="ds",
                y="value",
                color="label",
                title="Stacked Registrations by Category Over Time",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_area.update_layout(
                legend_title_text="Category",
                xaxis_title="Date",
                yaxis_title="Registrations",
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig_area, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Stacked area failed: {e}")
    
        # --- Overlay Line Chart ---
        try:
            fig_line = px.line(
                resampled,
                x="ds",
                y="value",
                color="label",
                title="Category Trends (Overlay)",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_line.update_traces(line=dict(width=2))
            fig_line.update_layout(
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Registrations",
                hovermode="x unified",
            )
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Overlay lines failed: {e}")
    
    # -------------------------
    # Separate Mode (Small Multiples)
    # -------------------------
    else:
        st.markdown("### 🧩 Small Multiples — Yearly Category Distribution")
    
        try:
            years_sorted = sorted(resampled["year"].unique())
        except Exception:
            years_sorted = []
    
        sel_small = st.multiselect(
            "Select specific years for small multiples (limit 6)",
            years_sorted,
            default=years_sorted[-min(3, len(years_sorted)):] if years_sorted else [],
        )
    
        if not sel_small:
            st.info("Select at least one year to show small multiples.")
        else:
            for y in sel_small[:6]:
                d = resampled[resampled["year"] == y]
                if d.empty:
                    st.caption(f"⚠️ No data for {y}")
                    continue
                try:
                    fig_bar = px.bar(
                        d,
                        x="label",
                        y="value",
                        color="label",
                        text_auto=True,
                        title=f"Category Distribution — {y}",
                        color_discrete_sequence=px.colors.qualitative.Pastel1,
                    )
                    fig_bar.update_traces(textfont_size=11, textangle=0)
                    fig_bar.update_layout(
                        showlegend=False,
                        template="plotly_white",
                        yaxis_title="Registrations",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Failed to plot {y}: {e}")

    # -------------------------
    # Optional Advanced Visuals
    # -------------------------
    if show_heatmap:
        st.markdown("### 🔥 Category Heatmap (Year × Category)")
        try:
            pivot_heat = pivot_year.copy()
            fig_heat = px.imshow(
                pivot_heat.T,
                labels=dict(x="Year", y="Category", color="Registrations"),
                aspect="auto",
                color_continuous_scale="YlOrRd",
                title="Heatmap of Registrations per Category per Year",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Heatmap failed: {e}")

    if show_radar:
        st.markdown("### 🕸️ Radar Chart — Category Profiles per Year")
        try:
            import plotly.graph_objects as go
            cats = list(pivot_year.columns)
            fig_radar = go.Figure()
            for y in sorted(pivot_year.index)[-min(4, len(pivot_year.index)):]:
                vals = pivot_year.loc[y].values
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=cats,
                    fill='toself',
                    name=str(y),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                template="plotly_white",
                title="Radar Comparison of Category Patterns",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Radar chart failed: {e}")

    # -------------------------
    # 🍩 Donut & Sunburst (All-Maxed)
    # -------------------------
    st.markdown("### 🍩 Donut & Sunburst — Latest Available Period (All-Maxed)")

    if resampled.empty:
        st.warning("⚠️ No resampled data available for donut/sunburst charts.")
    else:
        # Find latest period with valid data
        latest_period = (
            resampled.loc[resampled["value"] > 0, "ds"].max()
            if not resampled.empty
            else None
        )

        if latest_period is None or pd.isna(latest_period):
            st.info("No valid non-zero data found for latest period visualization.")
        else:
            d_latest = (
                resampled[resampled["ds"] == latest_period]
                .groupby("label", as_index=False)["value"]
                .sum()
                .sort_values("value", ascending=False)
            )

            total_latest = d_latest["value"].sum()
            d_latest["Share_%"] = (d_latest["value"] / total_latest * 100).round(2)

            if not d_latest.empty and total_latest > 0:
                # --- Donut Chart ---
                try:
                    fig_donut = px.pie(
                        d_latest,
                        names="label",
                        values="value",
                        hole=0.55,
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title=f"Category Split — {latest_period.strftime('%Y-%m')} (Total: {int(total_latest):,})",
                        hover_data={"Share_%": True, "value": True},
                    )
                    fig_donut.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        pull=[0.05] * len(d_latest),
                    )
                    fig_donut.update_layout(
                        template="plotly_white",
                        showlegend=True,
                        legend_title_text="Category",
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Donut chart failed: {e}")

                # --- Sunburst Chart ---
                try:
                    sb = (
                        df_cat_all.groupby(["year", "label"], as_index=False)["value"]
                        .sum()
                        .sort_values(["year", "value"], ascending=[True, False])
                    )

                    fig_sb = px.sunburst(
                        sb,
                        path=["year", "label"],
                        values="value",
                        color="value",
                        color_continuous_scale="Sunset",
                        title="🌞 Sunburst — Year → Category → Value",
                        hover_data={"value": True},
                    )
                    fig_sb.update_layout(template="plotly_white")
                    st.plotly_chart(fig_sb, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Sunburst chart failed: {e}")

                # --- Data summary below ---
                with st.expander("📋 Latest Period Data Summary"):
                    st.dataframe(
                        d_latest.style.format(
                            {"value": "{:,}", "Share_%": "{:.2f}"}
                        ),
                        use_container_width=True,
                    )
            else:
                st.info("⚠️ Latest period has zero or empty category values.")

    # -------------------------
    # 🔥 HEATMAP — Year × Category (All-Maxed)
    # -------------------------
    if show_heatmap:
        st.markdown("### 🔥 Heatmap — Year × Category (All-Maxed)")

        if pivot_year.empty:
            st.info("⚠️ No category data available for heatmap.")
        else:
            try:
                # Normalize optionally (for visual contrast)
                heat = pivot_year.copy()
                heat_norm = heat.div(heat.max(axis=1), axis=0).fillna(0)

                # Add toggle for normalization
                normalize_opt = st.toggle("Normalize heatmap (relative per year)", value=True)
                heat_used = heat_norm if normalize_opt else heat

                fig_h = go.Figure(
                    data=go.Heatmap(
                        z=heat_used.values,
                        x=heat_used.columns.astype(str),
                        y=heat_used.index.astype(str),
                        colorscale="Viridis",
                        hoverongaps=False,
                        texttemplate="%{z:.1f}" if normalize_opt else None,
                    )
                )
                fig_h.update_layout(
                    title=(
                        "Normalized Registrations by Category per Year"
                        if normalize_opt
                        else "Absolute Registrations by Category per Year"
                    ),
                    xaxis_title="Category",
                    yaxis_title="Year",
                    template="plotly_white",
                    coloraxis_colorbar=dict(title="Registrations" if not normalize_opt else "Share (0–1)"),
                    height=500,
                )
                st.plotly_chart(fig_h, use_container_width=True)

                # Optional: Table summary
                with st.expander("📋 View Heatmap Data Table"):
                    st.dataframe(
                        heat_used.round(2)
                        .style.format("{:,.0f}" if not normalize_opt else "{:.2f}")
                        .background_gradient(cmap="viridis"),
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"⚠️ Heatmap rendering failed: {e}")

    # -------------------------
    # 🌈 RADAR — Snapshot per Year (All-Maxed)
    # -------------------------
    if show_radar:
        st.markdown("### 🌈 Radar — Category Profile Snapshot (All-Maxed)")

        if pivot_year.empty:
            st.info("⚠️ Not enough data for radar visualization.")
        else:
            try:
                yrs_for_radar = sorted(pivot_year.index)[-min(4, len(pivot_year.index)):]
                cats = pivot_year.columns.tolist()

                # Normalize per category for better radar comparison
                radar_df = pivot_year.copy()
                radar_df_norm = radar_df.div(radar_df.max(axis=0), axis=1).fillna(0)

                normalize_radar = st.toggle("Normalize radar per category (0–1)", value=True)
                df_radar_used = radar_df_norm if normalize_radar else radar_df

                fig_r = go.Figure()
                for y in yrs_for_radar:
                    vals = df_radar_used.loc[y].values.tolist()
                    fig_r.add_trace(
                        go.Scatterpolar(
                            r=vals,
                            theta=cats,
                            fill="toself",
                            name=str(y),
                            hovertemplate="<b>%{theta}</b><br>Value: %{r:.2f}<extra></extra>",
                        )
                    )

                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1] if normalize_radar else [0, df_radar_used.values.max()],
                            showline=True,
                            linewidth=1,
                            gridcolor="lightgray",
                        )
                    ),
                    showlegend=True,
                    title="Category Distribution Radar (Last Years)",
                    template="plotly_white",
                    height=600,
                )
                st.plotly_chart(fig_r, use_container_width=True)

                # Add summary data
                with st.expander("📋 Radar Data Used"):
                    st.dataframe(
                        df_radar_used.round(2)
                        .style.format("{:,.0f}" if not normalize_radar else "{:.2f}")
                        .background_gradient(cmap="cool"),
                        use_container_width=True,
                    )

            except Exception as e:
                st.warning(f"⚠️ Radar chart rendering failed: {e}")

    # -------------------------
    # 🔮 FORECASTING — All-Maxed (Linear + Prophet + Auto Insights)
    # -------------------------
    if do_forecast:
        st.markdown("## 🔮 Forecasting (All-Maxed)")

        # ---------------------
        # 1️⃣ Select category & horizon
        # ---------------------
        categories = (
            pivot_year.columns.tolist()
            if not pivot_year.empty
            else df_cat_all["label"].unique().tolist()
        )

        if not categories:
            st.info("⚠️ No categories available for forecasting.")
        else:
            cat_to_forecast = st.selectbox("📊 Choose category to forecast", categories)
            horizon_years = st.slider("Forecast horizon (years)", 1, 10, 3)
            st.caption("Select a category and choose how many future years to forecast.")

            # ---------------------
            # 2️⃣ Prepare time series
            # ---------------------
            if cat_to_forecast in pivot_year.columns:
                series = pivot_year[[cat_to_forecast]].reset_index().rename(
                    columns={cat_to_forecast: "y", "index": "year"}
                )
            else:
                series = pd.DataFrame(columns=["year", "y"])

            if series.empty or series["y"].isna().all():
                st.info("⚠️ Insufficient data for forecasting this category.")
            else:
                series["ds"] = pd.to_datetime(series["year"].astype(str) + "-01-01")
                series = series[["ds", "y"]].dropna()

                # ---------------------
                # 3️⃣ Linear Regression Forecast
                # ---------------------
                st.markdown("### 📈 Linear Regression Forecast")
                try:
                    from sklearn.linear_model import LinearRegression
                    X = np.arange(len(series)).reshape(-1, 1)
                    y = series["y"].values
                    model = LinearRegression().fit(X, y)
                    fut_idx = np.arange(len(series) + horizon_years).reshape(-1, 1)
                    preds = model.predict(fut_idx)

                    fut_dates = pd.date_range(
                        start=series["ds"].iloc[0],
                        periods=len(series) + horizon_years,
                        freq="YS",
                    )
                    df_fore = pd.DataFrame({"ds": fut_dates, "Linear": preds})
                    df_fore["Type"] = ["Historical"] * len(series) + ["Forecast"] * horizon_years

                    fig_l = px.line(df_fore, x="ds", y="Linear", color="Type",
                                    title=f"Linear Trend Forecast — {cat_to_forecast}")
                    fig_l.add_scatter(x=series["ds"], y=series["y"], mode="markers+lines",
                                      name="Observed", line=dict(color="blue"))
                    fig_l.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_l, use_container_width=True)

                    # KPI summary
                    last_val = series["y"].iloc[-1]
                    next_val = preds[len(series)]
                    growth = ((next_val - last_val) / last_val) * 100 if last_val else np.nan
                    st.metric("Next Year Projection", f"{next_val:,.0f}", f"{growth:+.1f}% vs last year")
                except Exception as e:
                    st.warning(f"⚠️ Linear regression forecast failed: {e}")

                # ---------------------
                # 4️⃣ Prophet Forecast (if available)
                # ---------------------
                st.markdown("### 🧙 Prophet Forecast (Advanced, if available)")
                try:
                    from prophet import Prophet

                    m = Prophet(
                        yearly_seasonality=True,
                        seasonality_mode="multiplicative",
                        changepoint_prior_scale=0.05,
                    )
                    m.fit(series)
                    future = m.make_future_dataframe(periods=horizon_years, freq="Y")
                    forecast = m.predict(future)

                    figp = go.Figure()
                    figp.add_trace(go.Scatter(
                        x=series["ds"], y=series["y"],
                        mode="markers+lines", name="Observed", line=dict(color="blue")))
                    figp.add_trace(go.Scatter(
                        x=forecast["ds"], y=forecast["yhat"],
                        mode="lines", name="Forecast (yhat)", line=dict(color="orange", width=3)))
                    figp.add_trace(go.Scatter(
                        x=forecast["ds"], y=forecast["yhat_upper"],
                        mode="lines", name="Upper Bound", line=dict(color="lightgray", dash="dot")))
                    figp.add_trace(go.Scatter(
                        x=forecast["ds"], y=forecast["yhat_lower"],
                        mode="lines", name="Lower Bound", line=dict(color="lightgray", dash="dot")))

                    figp.update_layout(
                        title=f"Prophet Forecast — {cat_to_forecast}",
                        template="plotly_white",
                        height=550,
                        legend=dict(orientation="h", y=-0.2),
                        xaxis_title="Year",
                        yaxis_title="Registrations",
                    )
                    st.plotly_chart(figp, use_container_width=True)

                    # Optional insight
                    last_year = series["ds"].dt.year.max()
                    fut_y = forecast.tail(horizon_years)["yhat"].mean()
                    st.success(f"📊 Prophet projects an **average of {fut_y:,.0f}** registrations/year for the next {horizon_years} years.")
                except ImportError:
                    st.info("🧠 Prophet not installed — only linear forecast shown.")
                except Exception as e:
                    st.warning(f"⚠️ Prophet forecast failed: {e}")

                # ---------------------
                # 5️⃣ Display Forecast Data
                # ---------------------
                with st.expander("📋 View Forecast Data Table"):
                    try:
                        comb = df_fore.copy()
                        if "forecast" in locals():
                            comb = comb.merge(
                                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                                on="ds", how="outer"
                            )
                        st.dataframe(
                            comb.round(2).style.background_gradient(cmap="PuBuGn"),
                            use_container_width=True,
                        )
                    except Exception:
                        st.dataframe(df_fore, use_container_width=True)

    # -------------------------
    # ⚠️ ANOMALY DETECTION — All-Maxed
    # -------------------------
    if do_anomaly:
        st.markdown("## ⚠️ Anomaly Detection (All-Maxed)")
        st.caption("Detects outliers and abnormal spikes/drops per category time series using IsolationForest + backup z-score method.")

        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np

            anomalies = []
            anomaly_records = []

            if resampled.empty:
                st.warning("No resampled data available for anomaly detection.")
            else:
                categories = sorted(resampled["label"].unique())
                prog_bar = st.progress(0.0)

                for i, cat in enumerate(categories):
                    prog_bar.progress((i + 1) / len(categories))
                    ser = (
                        resampled[resampled["label"] == cat]
                        .set_index("ds")["value"]
                        .fillna(0)
                        .sort_index()
                    )

                    if len(ser) < 8 or ser.std() == 0:
                        continue

                    # --- Adaptive contamination ---
                    cont_rate = min(0.05, max(0.01, ser.std() / (ser.mean() + 1e-9) * 0.02))

                    # --- IsolationForest ---
                    try:
                        iso = IsolationForest(
                            contamination=cont_rate,
                            random_state=42,
                            n_estimators=200,
                            bootstrap=True,
                        )
                        X = ser.values.reshape(-1, 1)
                        preds = iso.fit_predict(X)
                        an_idxs = ser.index[preds == -1]
                        ser_an = ser.loc[an_idxs]
                        for dt, val in ser_an.items():
                            anomaly_records.append(
                                {"Category": cat, "Date": dt, "Value": val}
                            )
                    except Exception:
                        # --- Fallback: rolling z-score ---
                        zscores = (ser - ser.rolling(6, min_periods=2).mean()) / ser.rolling(6, min_periods=2).std()
                        ser_an = ser[np.abs(zscores) > 2.8]
                        for dt, val in ser_an.items():
                            anomaly_records.append(
                                {"Category": cat, "Date": dt, "Value": val}
                            )

                prog_bar.empty()

                # -------------------------------
                # 🧾 Summary + Visualization
                # -------------------------------
                if not anomaly_records:
                    st.success("✅ No significant anomalies detected across categories.")
                else:
                    df_an = pd.DataFrame(anomaly_records)
                    df_an["Date"] = pd.to_datetime(df_an["Date"])
                    st.markdown("### 📋 Detected Anomalies Summary")
                    st.dataframe(
                        df_an.sort_values("Date", ascending=False).style.format(
                            {"Value": "{:,.0f}"}
                        ),
                        use_container_width=True,
                        height=300,
                    )

                    # --- Category selector for visualization ---
                    sel_cat = st.selectbox(
                        "🔍 View anomalies for a specific category", sorted(df_an["Category"].unique())
                    )
                    ser = (
                        resampled[resampled["label"] == sel_cat]
                        .set_index("ds")["value"]
                        .fillna(0)
                        .sort_index()
                    )
                    an_dates = df_an[df_an["Category"] == sel_cat]["Date"]

                    fig_a = go.Figure()
                    fig_a.add_trace(
                        go.Scatter(
                            x=ser.index,
                            y=ser.values,
                            mode="lines+markers",
                            name="Value",
                            line=dict(color="steelblue"),
                        )
                    )
                    if not an_dates.empty:
                        fig_a.add_trace(
                            go.Scatter(
                                x=an_dates,
                                y=ser.loc[an_dates],
                                mode="markers",
                                name="Anomaly",
                                marker=dict(color="red", size=10, symbol="x"),
                            )
                        )
                    fig_a.update_layout(
                        title=f"Anomalies in {sel_cat} — Time Series Overlay",
                        template="plotly_white",
                        xaxis_title="Date",
                        yaxis_title="Registrations",
                        height=500,
                    )
                    st.plotly_chart(fig_a, use_container_width=True)

                    # Quick stats
                    st.info(f"📊 {len(df_an)} total anomalies detected across {len(categories)} categories.")
        except Exception as e:
            st.warning(f"⚠️ Anomaly detection failed: {e}")

    # -------------------------
    # 🔍 CLUSTERING (KMeans) — All-Maxed
    # -------------------------
    if do_clustering:
        st.markdown("## 🔍 Clustering (KMeans) — All-Maxed Mode")
        st.caption("Groups years by their category registration mix using KMeans with normalization, PCA view & silhouette score.")

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score

            if pivot_year.empty:
                st.warning("No pivot_year data available for clustering.")
            else:
                # Normalize the data
                X = pivot_year.fillna(0).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Choose K range
                max_k = min(8, max(3, len(pivot_year) - 1))
                k = st.slider("Number of clusters (K)", 2, max_k, min(4, max_k))

                # Fit KMeans
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = km.fit_predict(X_scaled)
                inertia = km.inertia_

                # Compute silhouette (safe)
                sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan

                df_cluster = pd.DataFrame({
                    "Year": pivot_year.index.astype(str),
                    "Cluster": labels
                })
                st.markdown("### 🧾 Cluster Assignment Summary")
                st.dataframe(df_cluster, use_container_width=True, height=300)

                c1, c2, c3 = st.columns(3)
                c1.metric("Clusters", str(k))
                c2.metric("Inertia (↓ better)", f"{inertia:.2f}")
                c3.metric("Silhouette (↑ better)", f"{sil:.3f}" if not np.isnan(sil) else "n/a")

                # --- Cluster Centers ---
                centers = pd.DataFrame(km.cluster_centers_, columns=pivot_year.columns)
                centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=pivot_year.columns)
                st.markdown("### 🧠 Cluster Centers — (approximate category mix)")
                st.dataframe(centers.style.format("{:,.0f}"), use_container_width=True)

                # --- PCA 2D visualization ---
                try:
                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X_scaled)
                    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
                    df_pca["Year"] = pivot_year.index.astype(str)
                    df_pca["Cluster"] = labels.astype(str)

                    fig_pca = px.scatter(
                        df_pca,
                        x="PC1",
                        y="PC2",
                        color="Cluster",
                        symbol="Cluster",
                        hover_data=["Year"],
                        title="📊 PCA Projection — Years clustered by category mix",
                    )
                    fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))
                    fig_pca.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_pca, use_container_width=True)
                except Exception as e:
                    st.warning(f"PCA visualization failed: {e}")

                # --- Radar by cluster (avg category mix) ---
                st.markdown("### 🌐 Radar — Cluster-wise average category mix")
                try:
                    fig_r = go.Figure()
                    for c in sorted(df_cluster["Cluster"].unique()):
                        cluster_mean = pivot_year.iloc[df_cluster["Cluster"] == c].mean()
                        fig_r.add_trace(go.Scatterpolar(
                            r=cluster_mean.values.tolist(),
                            theta=pivot_year.columns.tolist(),
                            fill="toself",
                            name=f"Cluster {c}",
                        ))
                    fig_r.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        title="Cluster-wise average category mix (Radar View)",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_r, use_container_width=True)
                except Exception as e:
                    st.warning(f"Radar view failed: {e}")

                st.success(f"✅ Clustering completed — {k} groups formed, silhouette={sil:.3f}.")

        except Exception as e:
            st.warning(f"Clustering failed: {e}")

    # =====================================================
    # 🤖 AI Narrative (All-Maxed, guarded)
    # =====================================================
    if enable_ai and do_forecast:
        st.markdown("## 🤖 AI Narrative (Summary & Recommendations) — All-Maxed")
        try:
            # Basic safety checks
            if pivot_year is None or pivot_year.empty or df_cat_all is None or df_cat_all.empty:
                st.warning("⚠️ No valid data available for AI narrative.")
            else:
                # --- UI controls for AI block ---
                st.caption("Generate a concise analyst-style summary plus 3 actionable recommendations. "
                           "AI output requires a configured provider (universal_chat or other).")
                allow_send = st.checkbox("I consent to sending an anonymized summary of aggregated metrics to an AI provider", value=False)
                preview_prompt = st.checkbox("Preview the prompt sent to the AI (truncated)", value=False)
                show_raw = st.checkbox("Show raw AI output (if available)", value=False)

                # Prepare compact aggregated payload (limit tokens & remove PII)
                agg = pivot_year.reset_index().copy()
                # Ensure years are strings, and values are ints
                agg["year"] = agg["year"].astype(str)
                for col in agg.columns:
                    if col != "year":
                        agg[col] = agg[col].fillna(0).astype(int)

                # small as list of dicts but truncated to safe length
                small_records = agg.to_dict(orient="records")
                small_preview = json.dumps(small_records)[:3000]  # truncate for preview/prompt

                # Basic computed metrics for deterministic fallback and for context
                total_by_year = agg.set_index("year").sum(axis=1)
                total_first = float(total_by_year.iloc[0]) if len(total_by_year) > 0 else 0.0
                total_last = float(total_by_year.iloc[-1]) if len(total_by_year) > 0 else 0.0
                years_count = max(1, len(total_by_year) - 1)
                cagr_val = ((total_last / total_first) ** (1 / years_count) - 1) * 100 if total_first > 0 and len(total_by_year) > 1 else 0.0

                # top categories overall + latest year growth per category
                top_overall = (
                    df_cat_all.groupby("label")["value"].sum().sort_values(ascending=False)
                )
                top3 = top_overall.head(3)
                latest_year = pivot_year.index.max() if not pivot_year.empty else None
                growth_per_cat = {}
                if latest_year is not None and latest_year - 1 in pivot_year.index:
                    prev = pivot_year.loc[latest_year - 1] if (latest_year - 1) in pivot_year.index else None
                    curr = pivot_year.loc[latest_year]
                    if prev is not None:
                        for cat in pivot_year.columns:
                            prev_v = float(prev.get(cat, 0))
                            curr_v = float(curr.get(cat, 0))
                            growth_per_cat[cat] = ((curr_v - prev_v) / prev_v * 100) if prev_v > 0 else (100.0 if curr_v > 0 else 0.0)

                # Construct system + user prompts with brevity and clear instructions
                system_prompt = (
                    "You are a senior transport data analyst. Provide a concise, factual summary (3-6 bullet points) of trends "
                    "in vehicle category registrations based on the provided aggregated metrics. Then give 3 short, prioritized, "
                    "actionable recommendations for policymakers or transport planners. Use percentages where relevant. Do not invent facts."
                )

                user_context = (
                    f"Aggregated yearly totals (truncated): {small_preview}\n\n"
                    f"Top categories overall: {', '.join(top3.index.tolist())}.\n"                )

                # Preview prompt
                if preview_prompt:
                    st.expander("Prompt preview (truncated)", expanded=False).write({"system": system_prompt, "user": user_context[:4000]})

                ai_text = None
                ai_raw = None

                # Only attempt network/LLM call if user consented
                if allow_send:
                    try:
                        # Try universal_chat if available
                        if "universal_chat" in globals() or "universal_chat" in locals():
                            # safe call: small, low temperature, deterministic
                            ai_resp = universal_chat(system_prompt, user_context, stream=False, temperature=0.0, max_tokens=500, retries=2)
                            # normalize response types
                            if isinstance(ai_resp, dict):
                                ai_raw = ai_resp
                                ai_text = ai_resp.get("text") or ai_resp.get("response") or ai_resp.get("output")
                            elif isinstance(ai_resp, str):
                                ai_text = ai_resp
                                ai_raw = {"text": ai_resp}
                        else:
                            st.info("No AI provider (`universal_chat`) found in this environment. Falling back to deterministic summary.")
                    except Exception as e:
                        st.warning(f"AI provider error or network issue: {e}")
                        ai_text = None

                # If AI didn't produce text, build deterministic humanized fallback summary
                if not ai_text:
                    # Build deterministic summary programmatically
                    bullets = []
                    # 1) Top categories
                    bullets.append(f"Top categories overall: {', '.join(top3.index.tolist())}.")
                    # 2) Trend direction by total
                    if not math.isnan(cagr_val) and abs(cagr_val) > 0.01:
                        dir_word = "increased" if cagr_val > 0 else "declined"
                        bullets.append(f"Total registrations {dir_word} at ~{abs(cagr_val):.2f}% CAGR between {years[0]} and {years[-1]}.")
                    else:
                        bullets.append("Total registrations remained roughly flat across the selected years.")
                    # 3) Notable category moves (top few)
                    notable = []
                    for cat in top3.index:
                        # recent growth if available
                        g = growth_per_cat.get(cat, None)
                        if g is not None:
                            notable.append(f"{cat} {('up' if g>0 else 'down')} {abs(g):.1f}% YoY (latest).")
                    if notable:
                        bullets.append("Notable recent moves: " + "; ".join(notable))
                    # 4) Data quality note
                    bullets.append("Data note: aggregated counts shown; verify monthly cadence for short-term trends.")
                    # Recommendations (auto-generated)
                    recs = [
                        "Monitor and support high-growth categories (e.g., EVs, light-commercial) with targeted policy incentives.",
                        "Improve inspection and road-safety programs focused on top-volume categories to reduce incidents.",
                        "Increase data cadence (move to monthly ingestion if possible) to enable finer forecasting and earlier anomaly detection."
                    ]

                    # Render fallback
                    st.markdown("### 🧠 Quick Narrative (deterministic fallback)")
                    for b in bullets:
                        st.markdown(f"- {b}")
                    st.markdown("**Recommendations:**")
                    for i, r in enumerate(recs, 1):
                        st.markdown(f"{i}. {r}")

                else:
                    # Show AI output and optionally raw JSON
                    st.markdown("### 🧠 AI Summary")
                    st.markdown(ai_text)
                    if show_raw and ai_raw is not None:
                        st.expander("Raw AI response (debug)", expanded=False).write(ai_raw)

                # Cache last ai_text in the current Streamlit session state for reuse
                try:
                    st.session_state["_last_ai_narrative"] = ai_text or "\n".join(bullets)
                except Exception:
                    pass

        except Exception as e:
            st.error(f"💥 AI Narrative generation failed: {e}")
    # =====================================================
    # 🧩 ALL-MAXED FINAL SUMMARY + DEBUG INSIGHTS (FULL SELF-CONTAINED)
    # =====================================================
    st.markdown("## 🧠 Final Summary & Debug Insights — ALL-MAXED")
    
    try:
        summary_start = time.time()
    
        # ----------------------------------------------------
        # 1️⃣ SAFE INPUT + FALLBACKS
        # ----------------------------------------------------
        df_src = df_cat_all.copy() if "df_cat_all" in locals() else pd.DataFrame()
        freq = freq if "freq" in locals() else "Monthly"
        years = years if "years" in locals() and years else [2024, 2025]
        current_year = datetime.now().year
    
        if df_src.empty:
            st.warning("⚠️ No valid ALL-MAXED data found to summarize.")
            st.stop()
    
        # ----------------------------------------------------
        # 2️⃣ BASIC CLEANUP + ADD TIME FIELDS
        # ----------------------------------------------------
        df_src = df_src.copy()
        if "ds" not in df_src.columns and "date" in df_src.columns:
            df_src["ds"] = pd.to_datetime(df_src["date"])
        elif "ds" not in df_src.columns:
            df_src["ds"] = pd.to_datetime(df_src["year"].astype(str) + "-01-01")
    
        df_src["year"] = df_src["ds"].dt.year
        df_src["month"] = df_src["ds"].dt.month
        df_src["label"] = df_src["label"].astype(str)
    
        # ----------------------------------------------------
        # 3️⃣ RESAMPLING BASED ON FREQUENCY
        # ----------------------------------------------------
        resampled = (
            df_src.groupby(["label", "year"])["value"].sum().reset_index()
            if freq == "Yearly"
            else df_src.copy()
        )
    
        pivot = resampled.pivot_table(
            index="ds", columns="label", values="value", aggfunc="sum"
        ).fillna(0)
    
        pivot_year = (
            resampled.pivot_table(
                index="year", columns="label", values="value", aggfunc="sum"
            ).fillna(0)
            if "year" in resampled.columns
            else pd.DataFrame()
        )
    
        # ----------------------------------------------------
        # 4️⃣ KPI METRICS (YoY, CAGR, MoM, Category Shares)
        # ----------------------------------------------------
        st.subheader("💎 Key Metrics & Growth (All-Maxed)")
    
        if pivot_year.empty:
            st.warning("⚠️ No yearly data found for KPI computation.")
            st.stop()
    
        # --- Compute totals and YoY
        year_totals = pivot_year.sum(axis=1).rename("TotalRegistrations").to_frame()
        year_totals["YoY_%"] = year_totals["TotalRegistrations"].pct_change() * 100
        year_totals["TotalRegistrations"] = (
            year_totals["TotalRegistrations"].fillna(0).astype(int)
        )
        year_totals["YoY_%"] = (
            year_totals["YoY_%"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
    
        # --- CAGR
        if len(year_totals) >= 2:
            first = float(year_totals["TotalRegistrations"].iloc[0])
            last = float(year_totals["TotalRegistrations"].iloc[-1])
            years_count = max(1, len(year_totals) - 1)
            cagr = ((last / first) ** (1 / years_count) - 1) * 100 if first > 0 else 0.0
        else:
            cagr = 0.0
    
        # --- MoM if monthly
        if freq == "Monthly":
            resampled["month_period"] = (
                resampled["year"].astype(str) + "-" + resampled["month"].astype(str)
            )
            month_totals = resampled.groupby("month_period")["value"].sum().reset_index()
            month_totals["MoM_%"] = month_totals["value"].pct_change() * 100
            latest_mom = (
                f"{month_totals['MoM_%'].iloc[-1]:.2f}%"
                if len(month_totals) > 1 and not np.isnan(month_totals["MoM_%"].iloc[-1])
                else "n/a"
            )
        else:
            latest_mom = "n/a"
    
        # --- Category shares
        latest_year = int(year_totals.index.max())
        latest_total = int(year_totals.loc[latest_year, "TotalRegistrations"])
        cat_share = (
            (pivot_year.loc[latest_year] / pivot_year.loc[latest_year].sum() * 100)
            .sort_values(ascending=False)
            .round(1)
        )
    
        # ----------------------------------------------------
        # 5️⃣ DISPLAY KPIs
        # ----------------------------------------------------
        c1, = st.columns(1)
        c1.metric("📅 Years Loaded", f"{years[0]} → {years[-1]}", f"{len(years)} yrs")
    
        st.markdown("#### 📘 Category Share (Latest Year)")
        st.dataframe(
            pd.DataFrame(
                {
                    "Category": cat_share.index,
                    "Share_%": cat_share.values,
                    "Volume": pivot_year.loc[latest_year].astype(int).values,
                }
            ).sort_values("Share_%", ascending=False),
            use_container_width=True,
        )
    
        with st.expander("🔍 Yearly Totals & Growth"):
            st.dataframe(
                year_totals.style.format(
                    {"TotalRegistrations": "{:,}", "YoY_%": "{:.2f}"}
                )
            )
    
        # ----------------------------------------------------
        # 6️⃣ DEEP INSIGHTS + TRENDS — FIXED
        # ----------------------------------------------------
        total_all = df_src["value"].sum()
        n_cats = df_src["label"].nunique()
        n_years = df_src["year"].nunique()
        
        # --- Top Category as dict (scalar values)
        top_cat_row = (
            df_src.groupby("label")["value"]
            .sum()
            .reset_index()
            .sort_values("value", ascending=False)
            .iloc[0]
        )
        top_cat = {
            "label": str(top_cat_row["label"]),
            "value": float(top_cat_row["value"])
        }
        top_cat_share = (top_cat["value"] / total_all) * 100 if total_all > 0 else 0
        
        # --- Top Year as dict (scalar values)
        top_year_row = (
            df_src.groupby("year")["value"]
            .sum()
            .reset_index()
            .sort_values("value", ascending=False)
            .iloc[0]
        )
        top_year = {
            "year": int(top_year_row["year"]),
            "value": float(top_year_row["value"])
        }
        
        # --- Display metrics safely
        st.metric("🏆 Absolute Top Category", top_cat["label"], f"{top_cat_share:.2f}% share")
        st.metric("📅 Peak Year", f"{top_year['year']}", f"{top_year['value']:,.0f} registrations")
        
        # --- Plot: Top 10 Categories
        st.write("### 🧾 Top 10 Categories — Overall")
        top_debug = (
            df_src.groupby("label")["value"]
            .sum()
            .reset_index()
            .sort_values("value", ascending=False)
        )
        fig_top10 = px.bar(
            top_debug.head(10),
            x="label",
            y="value",
            text_auto=True,
            color="value",
            color_continuous_scale="Blues",
            title="Top 10 Categories (All Years)",
        )
        fig_top10.update_layout(template="plotly_white", margin=dict(t=50, b=40))
        st.plotly_chart(fig_top10, use_container_width=True)

    
        # ----------------------------------------------------
        # 7️⃣ ADVANCED DEBUG METRICS
        # ----------------------------------------------------
        volatility = (
            df_src.groupby("year")["value"].sum().pct_change().std() * 100
            if len(df_src["year"].unique()) > 2
            else 0
        )
        dominance_ratio = (top_cat["value"] / total_all) * n_cats if total_all > 0 else 0
        direction = "increased" if cagr > 0 else "declined"
    
        summary_time = time.time() - summary_start
        st.markdown("### ⚙️ Debug Performance Metrics")
        st.code(
            f"""
    Years analyzed: {years}
    Categories: {n_cats}
    Rows processed: {len(df_src):,}
    Total registrations: {total_all:,.0f}
    Top category: {top_cat['label']} → {top_cat['value']:,.0f} ({top_cat_share:.2f}%)
    Peak year: {int(top_year['year'])} → {top_year['value']:,.0f}
    Dominance ratio: {dominance_ratio:.2f}
    Runtime: {summary_time:.2f}s
            """,
            language="yaml",
        )
    
        # ----------------------------------------------------
        # 8️⃣ SMART SUMMARY (NO INTERNAL TRY)
        # ----------------------------------------------------
        # Ensure top_cat is a single dict
        # Ensure top_cat is a dict
        # Ensure top_cat is a dict
        if isinstance(top_cat, list):
            top_cat = top_cat[0] if top_cat else {"label": "N/A", "value": 0}
        
        # Safe years and top_year check
        years_valid = years is not None and len(years) > 0
        top_year_valid = top_year is not None and "year" in top_year and "value" in top_year
        
        if top_cat and years_valid and top_year_valid:
            st.success(
                f"From **{years[0]}** to **{years[-1]}**, total registrations {direction} "
                f"**{top_cat.get('label', 'N/A')}** leads with **{top_cat_share:.2f}%** share. "
                f"Peak year: **{top_year['year']}** with **{top_year['value']:,.0f}** registrations. "
            )
            logger.info(f"✅ ALL-MAXED summary completed in {summary_time:.2f}s")
        else:
            st.error("⛔ ALL-MAXED summary failed: Missing or invalid data.")
            logger.warning("⚠️ ALL-MAXED summary skipped due to incomplete data.")
 
    
    # ----------------------------------------------------
    # 9️⃣ CATCH GLOBAL ERRORS
    # ----------------------------------------------------
    except Exception as e:
        logger.exception(f"ALL-MAXED summary failed: {e}")
        st.error(f"⛔ ALL-MAXED summary failed: {e}")
    
# -----------------------------------------------------
# 🧩 Safe Entry Point — Streamlit-only Execution Guard
# -----------------------------------------------------
if __name__ == "__main__":
    import streamlit as st

    st.markdown("# 🚗 ALL-MAXED CATEGORY ANALYTICS")

    try:
        all_maxed_category_block()
    except Exception as e:
        import traceback
        st.error(f"💥 Error while rendering All-Maxed block: {e}")
        st.code(traceback.format_exc(), language="python")


#     # ---- Top Makers ----
#     with col2:
#         st.markdown("### 🏭 Top Vehicle Makers")

#         try:
#             mk_json, mk_url = get_json("vahandashboard/top5Makerchart", params)
#             df_mk = to_df(mk_json, label_keys=("makerName", "manufacturer", "name", "label"))

#             if not df_mk.empty:
#                 fig2 = px.bar(
#                     df_mk,
#                     x="label",
#                     y="value",
#                     color="label",
#                     text_auto=True,
#                     title="Top 5 Makers",
#                 )
#                 fig2.update_layout(
#                     xaxis_title="Maker",
#                     yaxis_title="Count",
#                     showlegend=False,
#                     template="plotly_white",
#                 )
#                 st.plotly_chart(fig2, use_container_width=True)
#             else:
#                 st.info("ℹ️ No maker data available.")

#         except Exception as e:
#             st.error(f"❌ Error fetching or displaying makers: {e}")

# except Exception as e:
#     st.error(f"🚨 Data fetch error: {e}")

# ===============================================================
# 🏭 MAKERS — ALL-MAXED ANALYTICS SUITE (Streamlit)
# ===============================================================
# Mirrors the ALL-MAXED Category Analytics style but focused on Vehicle Makers.
# Assumptions: `get_json`, `to_df`, `params`, `enable_ai`, `deepinfra_chat`,
# helper viz functions (bar_from_df, pie_from_df, stack_from_df) exist in your app environment.

# -----------------------------
# MAKERS ANALYTICS — ALL MAXED (DROP-IN)
# -----------------------------
# import math
# import json
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# import random
# import logging

# st.markdown("## 🏭 ALL-MAXED — Makers Analytics (multi-frequency, multi-year)")

# # =====================================================
# # CONTROLS — ALL ON MAIN PAGE (no sidebar)
# # =====================================================
# section_id = rto_opt.lower() if "rto_opt" in locals() else "main"

# # Frequency & Mode
# freq = st.radio(
#     "Aggregation Frequency",
#     ["Daily", "Monthly", "Quarterly", "Yearly"],
#     index=3,
#     horizontal=True,
#     key=f"freq_{section_id}"
# )

# mode = st.radio(
#     "View Mode",
#     ["Separate (Small Multiples)", "Combined (Overlay / Stacked)"],
#     index=1,
#     horizontal=True,
#     key=f"mode_{section_id}"
# )

# # Year range
# today = datetime.now()
# current_year = today.year
# default_from_year = current_year - 1


# from_year = st.sidebar.number_input(
#     "From Year",
#     min_value=2012,
#     max_value=today.year,
#     value=default_from_year,
#     key=f"from_year_{section_id}"
# )

# to_year = st.sidebar.number_input(
#     "To Year",
#     min_value=from_year,
#     max_value=today.year,
#     value=today.year,
#     key=f"to_year_{section_id}"
# )

# state_code = st.sidebar.text_input(
#     "State Code (blank=All-India)",
#     value="",
#     key=f"state_{section_id}"
# )

# rto_code = st.sidebar.text_input(
#     "RTO Code (0=aggregate)",
#     value="0",
#     key=f"rto_{section_id}"
# )

# vehicle_classes = st.sidebar.text_input(
#     "Vehicle Classes (e.g., 2W,3W,4W if accepted)",
#     value="",
#     key=f"classes_{section_id}"
# )

# vehicle_makers = st.sidebar.text_input(
#     "Vehicle Makers (comma-separated or IDs)",
#     value="",
#     key=f"makers_{section_id}"
# )

# time_period = st.sidebar.selectbox(
#     "Time Period",
#     options=[0, 1, 2],
#     index=0,
#     key=f"period_{section_id}"
# )

# fitness_check = st.sidebar.selectbox(
#     "Fitness Check",
#     options=[True, False],
#     index=0,
#     format_func=lambda x: "Enabled" if x else "Disabled",
#     key=f"fitness_{section_id}"
# )

# vehicle_type = st.sidebar.text_input(
#     "Vehicle Type (optional)",
#     value="",
#     key=f"type_{section_id}"
# )

# # Extra feature toggles
# st.divider()
# col3, col4, col5 = st.columns(3)
# with col3:
#     show_heatmap = st.checkbox("Show Heatmap (year × maker)", True, key=f"heatmap_{section_id}")
#     show_radar = st.checkbox("Show Radar (per year)", True, key=f"radar_{section_id}")
# with col4:
#     do_forecast = st.checkbox("Enable Forecasting", True, key=f"forecast_{section_id}")
#     do_anomaly = st.checkbox("Enable Anomaly Detection", False, key=f"anomaly_{section_id}")
# with col5:
#     do_clustering = st.checkbox("Enable Clustering (KMeans)", False, key=f"cluster_{section_id}")

# params_common = build_params(
#     from_year=from_year,
#     to_year=to_year,
#     state_code=state_code or "ALL",
#     rto_code=rto_code or "0",
#     vehicle_classes=vehicle_classes or "ALL",
#     vehicle_makers=vehicle_makers or "ALL",
#     time_period=freq,
#     fitness_check=fitness_check,
#     vehicle_type=vehicle_type or "ALL"
# )

# years = list(range(int(from_year), int(to_year) + 1))

# st.info(f"🔗 Using parameters: {params_common}")

# # =====================================================
# # 🚗 VAHAN MAKER ANALYTICS — ALL-MAXED VISUAL ENGINE
# # =====================================================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# import uuid
# import math
# from datetime import datetime
# import time

# # Start timer for KPI/summary section
# summary_start = time.time()

# # -----------------------------------------------------
# # ⚡ GLOBAL VISUAL THEME
# # -----------------------------------------------------
# MAXED_COLORS = px.colors.qualitative.Safe + px.colors.qualitative.Plotly
# TITLE_FONT = dict(size=20, color="#111", family="Segoe UI Semibold")
# LABEL_FONT = dict(size=13, color="#333", family="Segoe UI")
# HOVER_TMPL = "<b>%{label}</b><br>%{y:,.0f} registrations<br>Year: %{x}"

# # -----------------------------------------------------
# # 🔹 UNIVERSAL SAFE WRAPPER
# # -----------------------------------------------------
# def _safe_df(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
#     if df is None or df.empty:
#         st.warning("⚠️ Empty dataframe provided to chart renderer.")
#         return pd.DataFrame(columns=required)
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         st.warning(f"⚠️ Missing columns for chart: {missing}")
#         for c in missing:
#             df[c] = np.nan
#     return df


# # -----------------------------------------------------
# # 🟦 MAXED BAR CHART (COMBINED / STACKED / NORMALIZED)
# # -----------------------------------------------------
# def _bar_from_df(
#     df: pd.DataFrame,
#     title: str,
#     combined: bool = False,
#     stacked: bool = False,
#     normalized: bool = False,
#     section_id: str = "",
#     color_field: str = "year",
# ):
#     """Render a polished bar chart with advanced legend, hover, and adaptive layout."""
#     try:
#         df = _safe_df(df, ["label", "value"])
#         unique_key = f"barmaxed_{section_id}_{uuid.uuid4().hex[:6]}"

#         if df.empty:
#             st.info("ℹ️ No data to render.")
#             return

#         if combined and color_field in df.columns:
#             mode = "stack" if stacked else "group"
#             fig = px.bar(
#                 df,
#                 x="label",
#                 y="value",
#                 color=color_field,
#                 barmode=mode,
#                 text_auto=True,
#                 title=title,
#                 color_discrete_sequence=MAXED_COLORS,
#             )
#         else:
#             fig = px.bar(
#                 df,
#                 x="label",
#                 y="value",
#                 color="label",
#                 text_auto=True,
#                 title=title,
#                 color_discrete_sequence=MAXED_COLORS,
#             )

#         # --- Normalization (percentage)
#         if normalized:
#             fig.for_each_trace(
#                 lambda t: t.update(y=t.y / np.sum(t.y) * 100 if np.sum(t.y) > 0 else t.y)
#             )
#             fig.update_yaxes(title_text="Share (%)")

#         # --- Layout polish
#         fig.update_layout(
#             template="plotly_white",
#             title_font=TITLE_FONT,
#             legend_title_text="",
#             legend=dict(
#                 orientation="h",
#                 yanchor="bottom",
#                 y=-0.25,
#                 xanchor="center",
#                 x=0.5,
#                 bgcolor="rgba(250,250,250,0.9)",
#                 bordercolor="rgba(0,0,0,0.1)",
#                 borderwidth=1,
#             ),
#             xaxis_title="Category / Maker",
#             yaxis_title="Registrations",
#             margin=dict(t=60, b=80, l=40, r=30),
#             bargap=0.15,
#             height=480,
#         )

#         # --- Enhanced hover
#         fig.update_traces(
#             hovertemplate="<b>%{x}</b><br>Registrations: %{y:,.0f}<extra></extra>",
#             textfont=LABEL_FONT,
#         )

#         st.plotly_chart(fig, use_container_width=True, key=unique_key)

#     except Exception as e:
#         st.warning(f"⚠️ MAXED Bar chart failed: {e}")


# # -----------------------------------------------------
# # 🟣 MAXED PIE / DONUT CHART (3D FEEL + CENTER LABEL)
# # -----------------------------------------------------
# def _pie_from_df(
#     df: pd.DataFrame,
#     title: str,
#     section_id: str = "",
#     donut: bool = True,
#     legend: bool = True,
# ):
#     """Render a fully-styled donut/pie chart with hover & label polish."""
#     try:
#         df = _safe_df(df, ["label", "value"])
#         unique_key = f"piemaxed_{section_id}_{uuid.uuid4().hex[:6]}"

#         if df.empty or df["value"].sum() <= 0:
#             st.info("ℹ️ No valid data for pie chart.")
#             return

#         fig = px.pie(
#             df,
#             names="label",
#             values="value",
#             hole=0.45 if donut else 0,
#             title=title,
#             color_discrete_sequence=MAXED_COLORS,
#         )

#         fig.update_traces(
#             textinfo="percent+label",
#             hovertemplate="<b>%{label}</b><br>%{value:,.0f} registrations<br>%{percent}",
#             pull=[0.05] * len(df),
#         )

#         # Center label for donut
#         if donut:
#             total = df["value"].sum()
#             fig.add_annotation(
#                 text=f"<b>{total:,.0f}</b><br><span style='font-size:12px;color:#666'>Total</span>",
#                 showarrow=False,
#                 font=dict(size=14),
#                 x=0.5,
#                 y=0.5,
#             )

#         fig.update_layout(
#             template="plotly_white",
#             title_font=TITLE_FONT,
#             margin=dict(t=60, b=40, l=40, r=40),
#             showlegend=legend,
#             height=420,
#             legend_title_text="",
#         )

#         st.plotly_chart(fig, use_container_width=True, key=unique_key)

#     except Exception as e:
#         st.warning(f"⚠️ MAXED Pie chart failed: {e}")


# # -----------------------------------------------------
# # 🟨 MAXED LINE / TREND CHART
# # -----------------------------------------------------
# def _line_from_df(
#     df: pd.DataFrame,
#     x_field: str = "year",
#     y_field: str = "value",
#     color_field: str = "label",
#     title: str = "",
#     section_id: str = "",
#     smooth: bool = False,
# ):
#     """Render a smooth trend line chart with multi-series overlay."""
#     try:
#         df = _safe_df(df, [x_field, y_field, color_field])
#         unique_key = f"linemaxed_{section_id}_{uuid.uuid4().hex[:6]}"

#         if df.empty:
#             st.info("ℹ️ No data to render trend.")
#             return

#         fig = px.line(
#             df,
#             x=x_field,
#             y=y_field,
#             color=color_field,
#             markers=True,
#             title=title,
#             color_discrete_sequence=MAXED_COLORS,
#         )

#         if smooth:
#             # Basic moving average smoothing for visual
#             df_sorted = df.sort_values(x_field)
#             fig.data = []
#             for lbl in df[color_field].unique():
#                 sub = df_sorted[df_sorted[color_field] == lbl]
#                 sub["smoothed"] = sub[y_field].rolling(3, min_periods=1).mean()
#                 fig.add_trace(go.Scatter(
#                     x=sub[x_field], y=sub["smoothed"],
#                     name=lbl, mode="lines+markers"
#                 ))

#         fig.update_layout(
#             template="plotly_white",
#             title_font=TITLE_FONT,
#             xaxis_title=x_field.capitalize(),
#             yaxis_title=y_field.capitalize(),
#             legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
#             margin=dict(t=60, b=60, l=40, r=30),
#             height=460,
#         )

#         st.plotly_chart(fig, use_container_width=True, key=unique_key)

#     except Exception as e:
#         st.warning(f"⚠️ MAXED Line chart failed: {e}")


# # -----------------------------------------------------
# # FETCH FUNCTION (ROBUST + MOCK FALLBACK)
# # -----------------------------------------------------
# # =====================================================
# # 🚀 ALL-MAXED MAKER FETCH & VISUAL MODULE
# # =====================================================

# # -----------------------------------------------------
# # 🔧 FETCH FUNCTION — SAFE + SMART + MOCK-RESILIENT
# # -----------------------------------------------------
# def fetch_maker_year(year: int, params_common: dict):
#     """Fetch top vehicle makers for a given year — fully maxed with safe params + mock fallback."""
#     logger.info(Fore.CYAN + f"🚀 Fetching top makers for {year}...")

#     # --- Safe param cleanup ---
#     safe_params = params_common.copy()
#     safe_params["fromYear"] = year
#     safe_params["toYear"] = year

#     for k in ["fitnessCheck", "stateCode", "rtoCode", "vehicleType"]:
#         if k in safe_params and (
#             safe_params[k] in ["ALL", "0", "", None, False]
#         ):
#             safe_params.pop(k, None)

#     mk_json, mk_url = None, None
#     try:
#         mk_json, mk_url = get_json("vahandashboard/top5Makerchart", safe_params)
#     except Exception as e:
#         logger.error(Fore.RED + f"❌ API fetch failed for {year}: {e}")
#         mk_json, mk_url = None, "MOCK://top5Makerchart"

#     # --- Status caption ---
#     color = "orange" if mk_url and "MOCK" in mk_url else "green"
#     st.markdown(
#         f"🔗 **API ({year}):** <span style='color:{color}'>{mk_url or 'N/A'}</span>",
#         unsafe_allow_html=True,
#     )

#     with st.expander(f"🧩 JSON Debug — {year}", expanded=False):
#         st.json(mk_json)

#     # --- Validation: check for expected fields ---
#     is_valid = False
#     df = pd.DataFrame()

#     if isinstance(mk_json, dict):
#         # ✅ Case 1: Chart.js-style JSON
#         if "datasets" in mk_json and "labels" in mk_json:
#             data_values = mk_json["datasets"][0].get("data", [])
#             labels = mk_json.get("labels", [])
#             if data_values and labels:
#                 df = pd.DataFrame({"label": labels, "value": data_values})
#                 is_valid = True

#         # ✅ Case 2: API returned dict with "data" or "result"
#         elif "data" in mk_json:
#             df = pd.DataFrame(mk_json["data"])
#             is_valid = not df.empty
#         elif "result" in mk_json:
#             df = pd.DataFrame(mk_json["result"])
#             is_valid = not df.empty

#     elif isinstance(mk_json, list) and mk_json:
#         # ✅ Case 3: Direct list of records
#         df = pd.DataFrame(mk_json)
#         is_valid = not df.empty

#     # --- Handle missing or invalid data ---
#     if not is_valid or df.empty:
#         logger.warning(Fore.YELLOW + f"⚠️ Using mock data for {year}")
#         st.warning(f"⚠️ No valid API data for {year}, generating mock values.")
#         random.seed(year)

#         makers = [
#             "Maruti Suzuki", "Tata Motors", "Hyundai", "Mahindra", "Hero MotoCorp",
#             "Bajaj Auto", "TVS Motor", "Honda", "Kia", "Toyota", "Renault",
#             "Ashok Leyland", "MG Motor", "Eicher", "Piaggio", "BYD", "Olectra", "Force Motors"
#         ]
#         random.shuffle(makers)
#         top = makers[:10]
#         base = random.randint(200_000, 1_000_000)
#         growth = 1 + (year - 2020) * 0.06
#         df = pd.DataFrame({
#             "label": top,
#             "value": [int(base * random.uniform(0.5, 1.5) * growth) for _ in top]
#         })
#     else:
#         st.success(f"✅ Valid API data loaded for {year}")

#     # --- Normalize columns ---
#     df.columns = [c.lower() for c in df.columns]
#     df["year"] = year
#     df = df.sort_values("value", ascending=False)

#     # --- Visual output ---
#     if not df.empty:
#         st.info(f"🏆 **{year}** → **{df.iloc[0]['label']}** — {df.iloc[0]['value']:,} registrations")
#         _bar_from_df(df, f"Top Makers ({year})", combined=False)
#         _pie_from_df(df, f"Maker Share ({year})")

#     return df
# # -----------------------------------------------------
# # 🔁 MAIN LOOP — MULTI-YEAR FETCH
# # -----------------------------------------------------
# all_years = []
# with st.spinner("⏳ Fetching maker data for all selected years..."):
#     for y in years:
#         try:
#             dfy = fetch_maker_year(y, params_common)   # ✅ FIXED: pass params_common
#             all_years.append(dfy)
#         except Exception as e:
#             st.error(f"❌ {y} fetch error: {e}")
#             logger.error(Fore.RED + f"Fetch error {y}: {e}")

# # ===============================================================
# # 🚘 MAKER ANALYTICS — FULLY MAXED + SAFE + DEBUG READY
# # ===============================================================
# import pandas as pd, numpy as np, math, time, random
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from colorama import Fore
# from sklearn.linear_model import LinearRegression
# from dateutil.relativedelta import relativedelta

# # ----------------------------------
# # 🧩 Helpers
# # ----------------------------------
# def normalize_freq_rule(freq):
#     return {"Daily": "D", "Monthly": "M", "Quarterly": "Q"}.get(freq, "Y")

# def year_to_timeseries_maker(df_year, year, freq):
#     """Convert maker-year totals into evenly distributed synthetic timeseries."""
#     rule = normalize_freq_rule(freq)
#     idx = pd.date_range(
#         start=f"{year}-01-01",
#         end=f"{year}-12-31",
#         freq=("D" if freq == "Daily" else "M"),
#     )
#     rows = []
#     for _, r in df_year.iterrows():
#         maker = r.get("label", f"Maker_{_}")
#         total = float(r.get("value", 0))
#         per = total / max(1, len(idx))
#         for ts in idx:
#             rows.append({"ds": ts, "label": maker, "value": per, "year": year})
#     return pd.DataFrame(rows)

# # ===============================================================
# # 🧭 FETCH MAKER DATA (PER YEAR)
# # ===============================================================
# def fetch_maker_year(year: int, params_common: dict):
#     """Fetch top vehicle makers for a given year — fully maxed with safe params + mock fallback."""
#     logger.info(Fore.CYAN + f"🚀 Fetching top makers for {year}...")

#     safe_params = params_common.copy()
#     safe_params["fromYear"] = year
#     safe_params["toYear"] = year

#     mk_json, mk_url = None, None
#     try:
#         mk_json, mk_url = get_json("vahandashboard/top5Makerchart", safe_params)
#     except Exception as e:
#         logger.error(Fore.RED + f"❌ API fetch failed for {year}: {e}")
#         mk_json, mk_url = None, "MOCK://top5Makerchart"

#     color = "orange" if mk_url and "MOCK" in mk_url else "green"
#     st.markdown(f"🔗 **API ({year}):** <span style='color:{color}'>{mk_url or 'N/A'}</span>", unsafe_allow_html=True)

#     with st.expander(f"🧩 JSON Debug — {year}", expanded=False):
#         st.json(mk_json)

#     if not mk_json or (isinstance(mk_json, dict) and not mk_json.get("datasets")):
#         st.warning(f"⚠️ No valid API data for {year}, generating mock values.")
#         random.seed(year)
#         makers = [
#             "Maruti Suzuki", "Tata Motors", "Hyundai", "Mahindra", "Hero MotoCorp",
#             "Bajaj Auto", "TVS Motor", "Honda", "Kia", "Toyota", "Renault",
#             "Ashok Leyland", "MG Motor", "Eicher", "Piaggio", "BYD", "Olectra", "Force Motors"
#         ]
#         random.shuffle(makers)
#         mk_json = {
#             "datasets": [{"data": [random.randint(200_000, 1_200_000) for _ in range(5)],
#                           "label": "Vehicle Registered"}],
#             "labels": makers[:5]
#         }

#     # --- Normalize API data
#     if isinstance(mk_json, dict) and "datasets" in mk_json:
#         data = mk_json["datasets"][0]["data"]
#         labels = mk_json["labels"]
#         mk_json = [{"label": l, "value": v} for l, v in zip(labels, data)]

#     df = pd.DataFrame(mk_json)
#     df.columns = [c.lower() for c in df.columns]
#     df["year"] = year
#     df = df.sort_values("value", ascending=False)

#     if not df.empty:
#         st.info(f"🏆 **{year}** → **{df.iloc[0]['label']}** — {df.iloc[0]['value']:,} registrations")
#         _bar_from_df(df, f"Top Makers ({year})", combined=False)
#         _pie_from_df(df, f"Maker Share ({year})")
#     return df

# # ===============================================================
# # ⏳ MULTI-YEAR DATA COLLECTION
# # ===============================================================
# with st.spinner("Fetching maker data for selected years..."):
#     all_year_dfs = []
#     for y in years:
#         try:
#             df_y = fetch_maker_year(y, params_common)
#             if df_y is not None and not df_y.empty:
#                 all_year_dfs.append(df_y)
#             else:
#                 st.warning(f"No data for {y}")
#         except Exception as e:
#             st.error(f"Error fetching {y}: {e}")

# if not all_year_dfs:
#     st.error("🚫 No maker data loaded for selected range.")
#     st.stop()

# df_maker_all = pd.concat(all_year_dfs, ignore_index=True)

# # ===============================================================
# # 📈 TIME SERIES & METRICS
# # ===============================================================
# rule = normalize_freq_rule(freq)
# ts_frames = [year_to_timeseries_maker(df_maker_all[df_maker_all["year"] == y], y, freq)
#              for y in sorted(df_maker_all["year"].unique())]
# df_ts = pd.concat(ts_frames, ignore_index=True)
# df_ts["ds"] = pd.to_datetime(df_ts["ds"])

# resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq=rule)])["value"].sum().reset_index()
# resampled["year"] = resampled["ds"].dt.year
# pivot = resampled.pivot_table(index="ds", columns="label", values="value", aggfunc="sum").fillna(0)
# pivot_year = resampled.pivot_table(index="year", columns="label", values="value", aggfunc="sum").fillna(0)

# # ===============================================================
# # 💎 KPI METRICS — Makers
# # ===============================================================
# st.subheader("💎 Key Metrics & Growth (Makers)")

# if pivot_year.empty:
#     st.warning("⚠️ No yearly data found for KPI computation.")
#     st.stop()

# # --- Compute totals and YoY
# year_totals = pivot_year.sum(axis=1).rename("TotalRegistrations").to_frame()
# year_totals["YoY_%"] = year_totals["TotalRegistrations"].pct_change() * 100
# year_totals["TotalRegistrations"] = year_totals["TotalRegistrations"].fillna(0).astype(int)
# year_totals["YoY_%"] = year_totals["YoY_%"].replace([np.inf, -np.inf], np.nan).fillna(0)

# # --- CAGR
# if len(year_totals) >= 2:
#     first = float(year_totals["TotalRegistrations"].iloc[0])
#     last = float(year_totals["TotalRegistrations"].iloc[-1])
#     years_count = max(1, len(year_totals) - 1)
#     cagr = ((last / first) ** (1 / years_count) - 1) * 100 if first > 0 else 0.0
# else:
#     cagr = 0.0

# # --- MoM (if monthly)
# if freq == "Monthly":
#     resampled["month_period"] = resampled["year"].astype(str) + "-" + resampled["month"].astype(str)
#     month_totals = resampled.groupby("month_period")["value"].sum().reset_index()
#     month_totals["MoM_%"] = month_totals["value"].pct_change() * 100
#     latest_mom = f"{month_totals['MoM_%'].iloc[-1]:.2f}%" if len(month_totals) > 1 and not np.isnan(month_totals["MoM_%"].iloc[-1]) else "n/a"
# else:
#     latest_mom = "n/a"

# # --- Category (Maker) shares
# latest_year = int(year_totals.index.max())
# latest_total = int(year_totals.loc[latest_year, "TotalRegistrations"])
# cat_share = (pivot_year.loc[latest_year] / pivot_year.loc[latest_year].sum() * 100).sort_values(ascending=False).round(1)

# # ----------------------------------------------------
# # DISPLAY KPIs
# # ----------------------------------------------------
# c1, = st.columns(1)
# c1.metric("📅 Years Loaded", f"{years[0]} → {years[-1]}", f"{len(years)} yrs")

# st.markdown("#### 📘 Maker Share (Latest Year)")
# st.dataframe(
#     pd.DataFrame({
#         "Maker": cat_share.index,
#         "Share_%": cat_share.values,
#         "Volume": pivot_year.loc[latest_year].astype(int).values
#     }).sort_values("Share_%", ascending=False),
#     use_container_width=True
# )

# with st.expander("🔍 Yearly Totals & Growth"):
#     st.dataframe(year_totals.style.format({"TotalRegistrations": "{:,}", "YoY_%": "{:.2f}"}))

# # ----------------------------------------------------
# # DEEP INSIGHTS + TRENDS
# # ----------------------------------------------------
# total_all = df_maker_all["value"].sum()
# n_cats = df_maker_all["label"].nunique()
# n_years = df_maker_all["year"].nunique()

# # --- Top Maker
# top_cat_row = df_maker_all.groupby("label")["value"].sum().reset_index().sort_values("value", ascending=False).iloc[0]
# top_cat = {"label": str(top_cat_row["label"]), "value": float(top_cat_row["value"])}
# top_cat_share = (top_cat["value"] / total_all) * 100 if total_all > 0 else 0

# # --- Top Year
# top_year_row = df_maker_all.groupby("year")["value"].sum().reset_index().sort_values("value", ascending=False).iloc[0]
# top_year = {"year": int(top_year_row["year"]), "value": float(top_year_row["value"])}

# st.metric("🏆 Absolute Top Maker", top_cat["label"], f"{top_cat_share:.2f}% share")
# st.metric("📅 Peak Year", f"{top_year['year']}", f"{top_year['value']:,.0f} registrations")

# # --- Plot: Top 10 Makers
# st.write("### 🧾 Top 10 Makers — Overall")
# top_debug = df_maker_all.groupby("label")["value"].sum().reset_index().sort_values("value", ascending=False)
# fig_top10 = px.bar(top_debug.head(10), x="label", y="value", text_auto=True,
#                    color="value", color_continuous_scale="Blues", title="Top 10 Makers (All Years)")
# fig_top10.update_layout(template="plotly_white", margin=dict(t=50, b=40))
# st.plotly_chart(fig_top10, use_container_width=True)

# # ----------------------------------------------------
# # ADVANCED DEBUG METRICS
# # ----------------------------------------------------
# volatility = df_maker_all.groupby("year")["value"].sum().pct_change().std() * 100 \
#     if len(df_maker_all["year"].unique()) > 2 else 0
# dominance_ratio = (top_cat["value"] / total_all) * n_cats if total_all > 0 else 0
# direction = "increased" if cagr > 0 else "declined"

# summary_time = time.time() - summary_start
# st.markdown("### ⚙️ Debug Performance Metrics")
# st.code(f"""
# Years analyzed: {years}
# Makers: {n_cats}
# Rows processed: {len(df_maker_all):,}
# Total registrations: {total_all:,.0f}
# Top maker: {top_cat['label']} → {top_cat['value']:,.0f} ({top_cat_share:.2f}%)
# Peak year: {int(top_year['year'])} → {top_year['value']:,.0f}
# Dominance ratio: {dominance_ratio:.2f}
# Runtime: {summary_time:.2f}s
# """, language="yaml")

# # ----------------------------------------------------
# # 8️⃣ SMART SUMMARY — All-Maxed
# # ----------------------------------------------------
# if isinstance(top_cat, list):
#     top_cat = top_cat[0] if top_cat else {"label": "N/A", "value": 0}

# years_valid = years is not None and len(years) > 0
# top_year_valid = top_year is not None and "year" in top_year and "value" in top_year

# if top_cat and years_valid and top_year_valid:
#     st.success(
#         f"From **{years[0]}** to **{years[-1]}**, total registrations {direction}. "
#         f"**{top_cat.get('label', 'N/A')}** leads with **{top_cat_share:.2f}%** share. "
#         f"Peak year: **{top_year['year']}** with **{top_year['value']:,.0f}** registrations."
#     )
#     logger.info(f"✅ ALL-MAXED summary completed in {summary_time:.2f}s")
# else:
#     st.error("⛔ ALL-MAXED summary failed: Missing or invalid data.")
#     logger.warning("⚠️ ALL-MAXED summary skipped due to incomplete data")

# ----------------------------------------------------
# 9️⃣ MAKERS + STATES SECTIONS (reuse same pattern)
# ----------------------------------------------------
# Repeat the above block for Makers and States by replacing:
# df_src → df_maker_all / df_state_all
# pivot_year → pivot_maker_year / pivot_state_year
# top_cat → top_maker / top_state
# top_cat_share → top_mk_share / top_state_share
# Adjust headings & success messages accordingly

    # ----------------------------------------------------
    # 9️⃣ MAKERS + STATES SECTIONS (reuse same pattern)
    # ----------------------------------------------------
    # repeat the above block for Makers/States as needed


# =========================================================
# 🔥 ALL-MAXED — Maker (multi-frequency, multi-year) — MAXED BLOCK
# Drop-in Streamlit module. Replace or import into your app.
# Created: ALL-MAXED v2 — resilient, instrumented, cached, mock-safe
# =========================================================
# ======================================================
# 🚀 MAXED-OUT DASHBOARD TEMPLATE — STREAMLIT + PLOTLY
# ======================================================

# =========================
# ALL-MAXED Maker Controls + Setup
# =========================

import time
import math
import json
import random
import logging
import requests
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Logging Setup
# -------------------------
logger = logging.getLogger("all_maxed_maker")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

## st.markdown("## 🏭 ALL-MAXED — Makers Analytics (multi-frequency, multi-year)")

# =====================================================
# CONTROLS — ALL ON MAIN PAGE (no sidebar)
# =====================================================

import streamlit as st
from datetime import datetime

# -------------------------------
# SECTION ID
# -------------------------------
section_id = "main"

# -------------------------------
# FREQUENCY & VIEW MODE
# -------------------------------
freq = st.radio(
    "Aggregation Frequency",
    ["Daily", "Monthly", "Quarterly", "Yearly"],
    index=3,  # default Yearly
    horizontal=True,
    key=f"freq_{section_id}"
)

mode = st.radio(
    "View Mode",
    ["Separate (Small Multiples)", "Combined (Overlay / Stacked)"],
    index=1,  # default Combined
    horizontal=True,
    key=f"mode_{section_id}"
)

# -------------------------------
# YEAR RANGE
# -------------------------------
today = datetime.now()
current_year = today.year
default_from_year = current_year - 1

from_year = st.sidebar.number_input(
    "From Year",
    min_value=2012,
    max_value=current_year,
    value=default_from_year,
    key=f"from_year_{section_id}"
)

to_year = st.sidebar.number_input(
    "To Year",
    min_value=from_year,
    max_value=current_year,
    value=current_year,
    key=f"to_year_{section_id}"
)

# -------------------------------
# LOCATION & VEHICLE FILTERS
# -------------------------------
state_code = st.sidebar.text_input(
    "State Code (blank=All-India)",
    value="",
    key=f"state_{section_id}"
)

rto_code = st.sidebar.text_input(
    "RTO Code (0=aggregate)",
    value="0",
    key=f"rto_{section_id}"
)

vehicle_classes = st.sidebar.text_input(
    "Vehicle Classes (e.g., 2W,3W,4W)",
    value="",
    key=f"classes_{section_id}"
)

vehicle_makers = st.sidebar.text_input(
    "Vehicle Makers (comma-separated or IDs)",
    value="",
    key=f"makers_{section_id}"
)

vehicle_type = st.sidebar.text_input(
    "Vehicle Type (optional)",
    value="",
    key=f"type_{section_id}"
)

time_period = st.sidebar.selectbox(
    "Time Period",
    options=[0, 1, 2],
    index=0,
    key=f"period_{section_id}"
)

fitness_check = st.sidebar.selectbox(
    "Fitness Check",
    options=[True, False],
    index=0,
    format_func=lambda x: "Enabled" if x else "Disabled",
    key=f"fitness_{section_id}"
)

# -------------------------------
# EXTRA FEATURE TOGGLES
# -------------------------------
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    show_heatmap = st.checkbox("Show Heatmap (year × maker)", True, key=f"heatmap_{section_id}")
    show_radar = st.checkbox("Show Radar (per year)", True, key=f"radar_{section_id}")

with col2:
    do_forecast = st.checkbox("Enable Forecasting", True, key=f"forecast_{section_id}")
    do_anomaly = st.checkbox("Enable Anomaly Detection", False, key=f"anomaly_{section_id}")

with col3:
    do_clustering = st.checkbox("Enable Clustering (KMeans)", False, key=f"cluster_{section_id}")

# -------------------------------
# DERIVE YEARS
# -------------------------------
years = list(range(int(from_year), int(to_year) + 1))


# =====================================================
# 🚀 ALL-MAXED MAKERS ANALYTICS CORE v1.0
# -----------------------------------------------------
# Fully loaded mock data, chart builders, and analytics UI helpers for Makers
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import random, uuid
import plotly.express as px
from datetime import datetime
from typing import Dict, Any
from plotly.colors import qualitative

# -----------------------------------------------------
# 🎯 Master Maker Reference
# -----------------------------------------------------
MAKERS_MASTER = [
    "Maruti Suzuki", "Tata Motors", "Hyundai", "Mahindra", "Hero MotoCorp",
    "Bajaj Auto", "TVS Motor", "Honda", "Kia", "Toyota", "Renault",
    "Ashok Leyland", "MG Motor", "Eicher", "Piaggio", "BYD", "Olectra", "Force Motors"
]

# -----------------------------------------------------
# 💾 Deterministic Mock Data Generator (Multi-Frequency)
# -----------------------------------------------------
def deterministic_mock_makers(year: int, freq: str = "Monthly", seed_base: str = "makers") -> Dict[str, Any]:
    """Generate reproducible, realistic mock data for makers (daily, monthly, yearly)."""
    rnd = random.Random(hash((year, seed_base)) & 0xFFFFFFFF)
    data = []

    if freq == "Yearly":
        for m in MAKERS_MASTER:
            val = rnd.randint(50_000, 2_500_000)
            data.append({"label": m, "value": val, "year": year})

    elif freq == "Monthly":
        for month in range(1, 13):
            for m in MAKERS_MASTER:
                base = rnd.randint(10_000, 200_000)
                seasonal_boost = 1.2 if month in [3, 9, 12] else 1.0
                value = int(base * seasonal_boost * (0.8 + rnd.random() * 0.5))
                data.append({
                    "label": m,
                    "value": value,
                    "year": year,
                    "month": month,
                    "month_name": datetime(year, month, 1).strftime("%b")
                })

    elif freq == "Daily":
        for month in range(1, 13):
            for day in range(1, 29):
                for m in MAKERS_MASTER:
                    base = rnd.randint(200, 15_000)
                    val = int(base * (0.8 + rnd.random() * 0.6))
                    data.append({
                        "label": m, "value": val, "year": year, "month": month, "day": day
                    })

    return {
        "data": data,
        "meta": {
            "generatedAt": datetime.utcnow().isoformat(),
            "note": f"deterministic mock for {year} ({freq})",
            "freq": freq
        }
    }

# -----------------------------------------------------
# ⚙️ Formatting helpers
# -----------------------------------------------------
def format_number(n):
    """Return number formatted as K, M, or Cr."""
    if n >= 10_000_000:
        return f"{n/10_000_000:.2f} Cr"
    elif n >= 100_000:
        return f"{n/100_000:.2f} L"
    elif n >= 1_000:
        return f"{n/1_000:.2f} K"
    return f"{n:,}"

# -----------------------------------------------------
# 🎨 Global Chart Style Settings
# -----------------------------------------------------
COLOR_PALETTE = qualitative.Plotly + qualitative.D3 + qualitative.Vivid
DEFAULT_TEMPLATE = "plotly_white"
TITLE_STYLE = dict(size=20, color="#111", family="Segoe UI Semibold")

# -----------------------------------------------------
# 🧩 MAXED CHART HELPERS (Legend, Hover, UI polish)
# -----------------------------------------------------
def _unique_key(prefix="chart"):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

# -----------------------------
# BAR CHART — MAKERS
# -----------------------------
def bar_from_makers(df: pd.DataFrame, title="Top Makers", x="label", y="value",
                    color=None, height=500, section_id="bar", combined=False):
    if df is None or df.empty:
        st.warning("⚠️ No data to plot.")
        return

    # Set barmode based on combined toggle
    barmode = "stack" if combined else "group"

    fig = px.bar(
        df, x=x, y=y, color=color or x, text_auto=".2s",
        title=title, color_discrete_sequence=COLOR_PALETTE,
        barmode=barmode
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} registrations",
        textfont_size=12, textangle=0, cliponaxis=False
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title_font=TITLE_STYLE,
        xaxis_title=x.title(),
        yaxis_title=y.title(),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, title=None, bgcolor="rgba(0,0,0,0)"
        ),
        height=height, bargap=0.2, margin=dict(t=60, b=40, l=40, r=20)
    )
    st.plotly_chart(fig, use_container_width=True, key=_unique_key(section_id))


# -----------------------------
# PIE / DONUT CHART — MAKERS
# -----------------------------
def pie_from_makers(df: pd.DataFrame, title="Maker Share", donut=True, section_id="pie", height=450):
    if df is None or df.empty:
        st.warning("⚠️ No data to plot.")
        return

    fig = px.pie(
        df, names="label", values="value", hole=0.45 if donut else 0.0,
        title=title, color_discrete_sequence=COLOR_PALETTE
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} registrations<br>%{percent}",
        pull=[0.03] * len(df),
    )
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title_font=TITLE_STYLE,
        legend=dict(orientation="v", yanchor="top", y=0.95, xanchor="left", x=0),
        height=height, margin=dict(t=60, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True, key=_unique_key(section_id))


# -----------------------------
# TREND / LINE CHART — MAKERS
# -----------------------------
def trend_from_makers(df: pd.DataFrame, title="Trend Over Time", section_id="trend", height=500):
    if df is None or df.empty:
        st.warning("⚠️ No trend data available.")
        return

    # Optional monthly ordering
    if "month_name" in df.columns:
        df["month_order"] = pd.Categorical(
            df["month_name"],
            categories=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            ordered=True
        )
        x_axis = "month_order"
    else:
        x_axis = "year"

    fig = px.line(
        df, x=x_axis, y="value", color="label", markers=True,
        title=title, color_discrete_sequence=COLOR_PALETTE,
        line_shape="spline"
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,.0f} registrations")
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        title_font=TITLE_STYLE,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"
        ),
        height=height, margin=dict(t=60, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True, key=_unique_key(section_id))
# -----------------------------------------------------
# 🧠 Auto Dashboard Section — Single Function (MAKERS)
# -----------------------------------------------------
def render_maker_dashboard(year: int, freq="Monthly"):
    """Render full UI: fetch mock makers data, show KPI, bar, pie, trend — ALL-MAXED."""
    st.subheader(f"📊 Maker Distribution — {year} ({freq})")

    # Generate deterministic data
    mock_json = deterministic_mock_makers(year, freq=freq)
    df = pd.DataFrame(mock_json["data"])

    total = df["value"].sum()
    top = df.sort_values("value", ascending=False).iloc[0]
    st.success(f"🏆 **Top Maker:** {top['label']} — {format_number(top['value'])} registrations")
    st.caption(f"Total: {format_number(total)} | Generated: {mock_json['meta']['generatedAt']}")

    # Layout 2-col + trend
    c1, c2 = st.columns([2, 1])
    with c1:
        bar_from_makers(df, title=f"{year} {freq} Breakdown (Bar)", color="label", section_id=f"bar_{year}")
    with c2:
        pie_from_makers(df, title=f"{year} Share (Donut)", section_id=f"pie_{year}")

    # Optional trend
    if "month_name" in df.columns:
        trend_from_makers(df, title=f"{year} Monthly Trend (Animated)", section_id=f"trend_{year}")
    else:
        trend_from_makers(df, title=f"{year} Maker Trend", section_id=f"trend_{year}")

    return df


# ============================================================
# ⚙️ Synthetic Timeseries Expansion — MAKER ULTRA MAXED
# ------------------------------------------------------------
# Generates per-category realistic timeseries for Maker dashboards
# Includes trend, seasonality, noise, and deterministic reproducibility
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime

def maker_year_to_timeseries(
    df_year: pd.DataFrame,
    year: int,
    freq: str = "Monthly",
    trend_strength: float = 0.15,
    noise_strength: float = 0.10,
    seasonal_boost: bool = True,
    seed_base: str = "maker_timeseries"
) -> pd.DataFrame:
    """
    Expand annual category totals into Maker-ready realistic timeseries.
    
    Returns columns: ds, label, value, year, month, quarter, month_name, maker_key
    """
    if df_year is None or df_year.empty:
        return pd.DataFrame(columns=[
            "ds","label","value","year","month","quarter","month_name","maker_key"
        ])

    # --- deterministic seed for reproducibility
    seed = abs(hash((year, freq, seed_base))) % (2**32)
    rng = np.random.default_rng(seed)

    # --- time index generation
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    freq = freq.capitalize()
    if freq == "Daily":
        idx = pd.date_range(start=start, end=end, freq="D")
    elif freq == "Monthly":
        idx = pd.date_range(start=start, end=end, freq="M")
    elif freq == "Quarterly":
        idx = pd.date_range(start=start, end=end, freq="Q")
    else:
        idx = pd.date_range(start=start, end=end, freq="Y")

    n = len(idx)
    rows = []

    # --- seasonal sinusoidal factor helper
    def seasonal_factor(i):
        if not seasonal_boost:
            return 1.0
        return 1.0 + 0.25 * np.sin((i / n) * 2 * np.pi * 4)  # 4 seasonal peaks

    for _, r in df_year.iterrows():
        cat = r.get("label", "Unknown")
        total = float(r.get("value", 0.0))
        if total <= 0:
            continue

        base_per = total / max(1, n)

        # trend (up/down) + noise
        trend = np.linspace(1 - trend_strength, 1 + trend_strength, n)
        if rng.random() > 0.5:
            trend = trend[::-1]
        noise = rng.normal(0, noise_strength, n)

        vals = base_per * trend * (1 + noise)
        vals = np.maximum(vals, 0)

        for i, ts in enumerate(idx):
            factor = seasonal_factor(i)
            v = vals[i] * factor
            rows.append({
                "ds": ts,
                "label": cat,
                "value": float(v),
                "year": int(year),
                "month": ts.month,
                "quarter": ts.quarter,
                "month_name": ts.strftime("%b"),
                "maker_key": f"{cat}_{ts.strftime('%Y%m%d')}"
            })

    df_out = pd.DataFrame(rows)
    df_out["value"] = df_out["value"].round(2)

    # --- normalize to match original annual totals
    grouped = df_out.groupby("label")["value"].sum().to_dict()
    for cat in grouped:
        original_total = float(df_year.loc[df_year["label"] == cat, "value"].iloc[0])
        if grouped[cat] > 0:
            df_out.loc[df_out["label"] == cat, "value"] *= original_total / grouped[cat]

    df_out.reset_index(drop=True, inplace=True)
    return df_out



# ============================================================
# 🚀 MAKER DASHBOARD FETCHER — ALL-MAXED ULTRA
# ------------------------------------------------------------
# Fetch Top 5 Makers, render bar + pie charts, insights, 
# and synthetic trends with fallback.
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from typing import Dict

# -----------------------------
# Fallback deterministic mock
# -----------------------------

def maker_mock_top5(year: int) -> Dict:
    """Return deterministic mock Top 5 Makers data."""
    return {
        "data": [
            {"label": f"Maker {i}", "value": np.random.randint(500, 2000)}
            for i in range(1, 6)
        ]
    }

#-----------------------------
#MAXED Maker fetch + render
#-----------------------------
# -----------------------------------------------------
# 🔧 FETCH FUNCTION — SAFE + SMART + MOCK-RESILIENT
# -----------------------------------------------------

params_common = build_params(
    from_year=from_year,
    to_year=to_year,
    state_code=state_code or "ALL",
    rto_code=rto_code or "0",
    vehicle_classes=vehicle_classes or "ALL",
    vehicle_makers=vehicle_makers or "ALL",
    time_period=freq,
    fitness_check=fitness_check,
    vehicle_type=vehicle_type or "ALL"
)

years = list(range(int(from_year), int(to_year) + 1))

st.info(f"🔗 Using parameters: {params_common}")

import random
import pandas as pd
import streamlit as st
from colorama import Fore
import logging

logger = logging.getLogger(__name__)

def fetch_maker_top5(year: int, params_common: dict, show_debug: bool = False):
    """
    Fetch top vehicle makers for a given year — fully maxed with safe params + mock fallback.
    
    Parameters:
        year (int): Year to fetch
        params_common (dict): Parameters for the API
        show_debug (bool): If True, show raw JSON in an expander

    Returns:
        pd.DataFrame: DataFrame of top makers with 'label', 'value', 'year'
    """
    logger.info(Fore.CYAN + f"🚀 Fetching top makers for {year}...")

    # --- Safe param cleanup ---
    safe_params = params_common.copy()
    safe_params["fromYear"] = year
    safe_params["toYear"] = year

    for k in ["fitnessCheck", "stateCode", "rtoCode", "vehicleType"]:
        if k in safe_params and safe_params[k] in ["ALL", "0", "", None, False]:
            safe_params.pop(k, None)

    mk_json, mk_url = None, None
    try:
        mk_json, mk_url = get_json("vahandashboard/top5Makerchart", safe_params)
    except Exception as e:
        logger.error(Fore.RED + f"❌ API fetch failed for {year}: {e}")
        mk_json, mk_url = None, "MOCK://top5Makerchart"

    # --- Status caption ---
    color = "orange" if mk_url and "MOCK" in mk_url else "green"
    st.markdown(
        f"🔗 **API ({year}):** <span style='color:{color}'>{mk_url or 'N/A'}</span>",
        unsafe_allow_html=True,
    )

    # --- Optional debug ---
    if show_debug:
        with st.expander(f"🧩 JSON Debug — {year}", expanded=False):
            st.json(mk_json)

    # --- Validate JSON & extract data ---
    is_valid = False
    df = pd.DataFrame()

    if isinstance(mk_json, dict):
        if "datasets" in mk_json and "labels" in mk_json:
            data_values = mk_json["datasets"][0].get("data", [])
            labels = mk_json.get("labels", [])
            if data_values and labels:
                df = pd.DataFrame({"label": labels, "value": data_values})
                is_valid = True
        elif "data" in mk_json:
            df = pd.DataFrame(mk_json["data"])
            is_valid = not df.empty
        elif "result" in mk_json:
            df = pd.DataFrame(mk_json["result"])
            is_valid = not df.empty
    elif isinstance(mk_json, list) and mk_json:
        df = pd.DataFrame(mk_json)
        is_valid = not df.empty

    # --- Handle missing/invalid data with deterministic mock ---
    if not is_valid or df.empty:
        logger.warning(Fore.YELLOW + f"⚠️ Using mock data for {year}")
        st.warning(f"⚠️ No valid API data for {year}, generating mock values.")
        random.seed(year)

        makers = [
            "Maruti Suzuki", "Tata Motors", "Hyundai", "Mahindra", "Hero MotoCorp",
            "Bajaj Auto", "TVS Motor", "Honda", "Kia", "Toyota", "Renault",
            "Ashok Leyland", "MG Motor", "Eicher", "Piaggio", "BYD", "Olectra", "Force Motors"
        ]
        random.shuffle(makers)
        top = makers[:10]
        base = random.randint(200_000, 1_000_000)
        growth = 1 + (year - 2020) * 0.06
        df = pd.DataFrame({
            "label": top,
            "value": [int(base * random.uniform(0.5, 1.5) * growth) for _ in top]
        })
    else:
        st.success(f"✅ Valid API data loaded for {year}")

    # --- Normalize columns ---
    df.columns = [c.lower() for c in df.columns]
    df["year"] = year
    df = df.sort_values("value", ascending=False)

    # --- Visual output ---
    if not df.empty:
        st.info(f"🏆 **{year}** → **{df.iloc[0]['label']}** — {df.iloc[0]['value']:,} registrations")
        bar_from_makers(df, f"Top Makers ({year})", combined=False)
        pie_from_makers(df, f"Maker Share ({year})")

    return df
# -----------------------------------------------------
# 🔁 MAIN LOOP — MULTI-YEAR FETCH
# -----------------------------------------------------
all_years = []
with st.spinner("⏳ Fetching maker data for all selected years..."):
    for y in years:
        try:
            dfy = fetch_maker_top5(y, params_common, show_debug=True)   # ✅ FIXED: pass params_common
            all_years.append(dfy)
        except Exception as e:
            st.error(f"❌ {y} fetch error: {e}")
            logger.error(Fore.RED + f"Fetch error {y}: {e}")

# =====================================================
# -------------------------
# Main Streamlit UI — All-Maxed Maker Block
# -------------------------
# =====================================================

from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def all_maxed_maker_block(params_common: dict = None, freq="Monthly", section_id="maker_section"):
    """
    Render the MAXED multi-year Maker analytics block inside Streamlit.
    Handles multi-year fetching, deterministic fallback, and synthetic time series.
    """

    import streamlit as st
    import pandas as pd
    from datetime import datetime

    # Ensure params_common is a dict
    params_common = params_common or {}

    # Determine year range
    years = list(range(int(from_year), int(to_year)+1))

    # -------------------------
    # Fetch multi-year maker data
    # -------------------------
    all_year_dfs = []
    with st.spinner("Fetching maker data for selected years..."):
        for y in years:
            try:
                df_y = fetch_maker_top5(y, params_common, show_debug=True)
                if df_y is None or df_y.empty:
                    raise ValueError(f"No maker data for {y}")
            except Exception as e:
                st.warning(f"⚠️ {e}. Using deterministic mock for {y}.")
                logger.exception(f"Error fetching maker data for {y}: {e}")
                mock_data = maker_mock_top5(y).get("data", [])
                df_y = pd.DataFrame(mock_data)
                df_y["year"] = y
                if "label" not in df_y.columns:
                    df_y["label"] = df_y.get("name", [f"Maker {i+1}" for i in range(len(df_y))])
                if "value" not in df_y.columns:
                    df_y["value"] = pd.to_numeric(df_y.get("score", 0), errors="coerce").fillna(0)
            all_year_dfs.append(df_y)

    # Concatenate and sort
    df_maker_all = pd.concat(all_year_dfs, ignore_index=True)
    df_maker_all = df_maker_all.sort_values(["year", "value"], ascending=[True, False]).reset_index(drop=True)

    # -------------------------
    # Frequency expansion -> synthetic time series
    # -------------------------
    ts_list = []
    for y in sorted(df_maker_all["year"].unique()):
        df_y = df_maker_all[df_maker_all["year"] == y].reset_index(drop=True)
        ts = year_to_timeseries(df_y.rename(columns={"label":"label", "value":"value"}), int(y), freq=freq)
        ts_list.append(ts)

    df_ts = pd.concat(ts_list, ignore_index=True) if ts_list else pd.DataFrame(columns=["ds","label","value","year"])
    df_ts["ds"] = pd.to_datetime(df_ts["ds"])

    # -------------------------
    # Resample to requested frequency
    # -------------------------
    freq_map = {"Daily":"D", "Monthly":"M", "Quarterly":"Q", "Yearly":"Y"}
    resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq=freq_map.get(freq, "M"))])["value"].sum().reset_index()
    resampled["year"] = resampled["ds"].dt.year

    # -------------------------
    # Pivot tables for heatmap / radar / combined view
    # -------------------------
    pivot = resampled.pivot_table(index="ds", columns="label", values="value", aggfunc="sum").fillna(0)
    pivot_year = resampled.pivot_table(index="year", columns="label", values="value", aggfunc="sum").fillna(0)

    
    # -------------------------
    # Frequency expansion -> synthetic timeseries
    # -------------------------
    ts_list = []
    for y in sorted(df_maker_all["year"].unique()):
        df_y = df_maker_all[df_maker_all["year"]==y].reset_index(drop=True)
        ts = year_to_timeseries(df_y.rename(columns={"label":"label","value":"value"}), int(y), freq=freq)
        ts_list.append(ts)
    
    df_ts = pd.concat(ts_list, ignore_index=True) if ts_list else pd.DataFrame(columns=["ds","label","value","year"])
    df_ts["ds"] = pd.to_datetime(df_ts["ds"])
    
    # -------------------------
    # Resample to requested frequency
    # -------------------------
    if freq == "Daily":
        resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq="D")])["value"].sum().reset_index()
    elif freq == "Monthly":
        resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq="M")])["value"].sum().reset_index()
    elif freq == "Quarterly":
        resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq="Q")])["value"].sum().reset_index()
    else:
        resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq="Y")])["value"].sum().reset_index()
    
    resampled["year"] = resampled["ds"].dt.year
    
    # -------------------------
    # Pivot tables for heatmap / radar / combined view
    # -------------------------
    pivot = resampled.pivot_table(index="ds", columns="label", values="value", aggfunc="sum").fillna(0)
    pivot_year = resampled.pivot_table(index="year", columns="label", values="value", aggfunc="sum").fillna(0)

    # -------------------------
    # Display heatmap
    # -------------------------
    if show_heatmap:
        st.subheader("📊 Year × Maker Heatmap")
        import plotly.express as px
        fig = px.imshow(
            pivot_year.T,
            labels=dict(x="Year", y="Maker", color="Value"),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Display radar (per year)
    # -------------------------
    if show_radar:
        st.subheader("📈 Radar / Polar — Makers per Year")
        import plotly.graph_objects as go
        fig = go.Figure()
        for y in pivot_year.index:
            fig.add_trace(
                go.Scatterpolar(
                    r=pivot_year.loc[y].values,
                    theta=pivot_year.columns.tolist(),
                    fill='toself',
                    name=str(y)
                )
            )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 📊 VISUALIZATIONS — ALL-MAXED MAKER
    # =====================================================
    st.subheader("📊 Maker Visualizations — Multi-year & Multi-frequency (All-Maxed)")
    
    # --- Safety Checks ---
    if "resampled" not in locals() or resampled is None or resampled.empty:
        st.warning("⚠️ No valid 'resampled' maker data available for visualization.")
        st.stop()
    
    if "pivot" not in locals() or pivot is None or pivot.empty:
        st.warning("⚠️ No valid 'pivot' maker data available for visualization.")
        st.stop()
    
    if "mode" not in locals():
        mode = "Combined (Overlay / Stacked)"
    
    # -------------------------
    # Combined / Small Multiples — Maker
    # -------------------------
    if mode.startswith("Combined"):
        st.markdown("### 🌈 Stacked & Overlay Trends — Combined Maker View")
    
        # --- Stacked Area Chart ---
        try:
            fig_area = px.area(
                resampled,
                x="ds",
                y="value",
                color="label",
                title="Stacked Registrations by Maker Over Time",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_area.update_layout(
                legend_title_text="Maker",
                xaxis_title="Date",
                yaxis_title="Registrations",
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig_area, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Stacked area failed: {e}")
    
        # --- Overlay Line Chart ---
        try:
            fig_line = px.line(
                resampled,
                x="ds",
                y="value",
                color="label",
                title="Maker Trends (Overlay)",
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_line.update_traces(line=dict(width=2))
            fig_line.update_layout(
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Registrations",
                hovermode="x unified",
            )
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Overlay lines failed: {e}")
    
    # -------------------------
    # Separate Mode (Small Multiples)
    # -------------------------
    else:
        st.markdown("### 🧩 Small Multiples — Yearly Maker Distribution")
    
        try:
            years_sorted = sorted(resampled["year"].unique())
        except Exception:
            years_sorted = []
    
        sel_small = st.multiselect(
            "Select specific years for small multiples (limit 6)",
            years_sorted,
            default=years_sorted[-min(3, len(years_sorted)):] if years_sorted else [],
        )
    
        if not sel_small:
            st.info("Select at least one year to show small multiples.")
        else:
            for y in sel_small[:6]:
                d = resampled[resampled["year"] == y]
                if d.empty:
                    st.caption(f"⚠️ No data for {y}")
                    continue
                try:
                    fig_bar = px.bar(
                        d,
                        x="label",
                        y="value",
                        color="label",
                        text_auto=True,
                        title=f"Maker Distribution — {y}",
                        color_discrete_sequence=px.colors.qualitative.Pastel1,
                    )
                    fig_bar.update_traces(textfont_size=11, textangle=0)
                    fig_bar.update_layout(
                        showlegend=False,
                        template="plotly_white",
                        yaxis_title="Registrations",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Failed to plot {y}: {e}")
    # -------------------------
    # Optional Advanced Visuals — Maker
    # -------------------------
    if show_heatmap:
        st.markdown("### 🔥 Maker Heatmap (Year × Maker)")
        try:
            pivot_heat = pivot_year.copy()
            fig_heat = px.imshow(
                pivot_heat.T,
                labels=dict(x="Year", y="Maker", color="Registrations"),
                aspect="auto",
                color_continuous_scale="YlOrRd",
                title="Heatmap of Registrations per Maker per Year",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Heatmap failed: {e}")
    
    if show_radar:
        st.markdown("### 🕸️ Radar Chart — Maker Profiles per Year")
        try:
            import plotly.graph_objects as go
            makers = list(pivot_year.columns)
            fig_radar = go.Figure()
            for y in sorted(pivot_year.index)[-min(4, len(pivot_year.index)):]:
                vals = pivot_year.loc[y].values
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=makers,
                    fill='toself',
                    name=str(y),
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                template="plotly_white",
                title="Radar Comparison of Maker Patterns",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Radar chart failed: {e}")
    
    # -------------------------
    # 🍩 Donut & Sunburst (All-Maxed) — Maker
    # -------------------------
    st.markdown("### 🍩 Donut & Sunburst — Latest Available Period (All-Maxed)")
    
    if resampled.empty:
        st.warning("⚠️ No resampled data available for donut/sunburst charts.")
    else:
        latest_period = (
            resampled.loc[resampled["value"] > 0, "ds"].max()
            if not resampled.empty
            else None
        )
    
        if latest_period is None or pd.isna(latest_period):
            st.info("No valid non-zero data found for latest period visualization.")
        else:
            d_latest = (
                resampled[resampled["ds"] == latest_period]
                .groupby("label", as_index=False)["value"]
                .sum()
                .sort_values("value", ascending=False)
            )
    
            total_latest = d_latest["value"].sum()
            d_latest["Share_%"] = (d_latest["value"] / total_latest * 100).round(2)
    
            if not d_latest.empty and total_latest > 0:
                # --- Donut Chart ---
                try:
                    fig_donut = px.pie(
                        d_latest,
                        names="label",
                        values="value",
                        hole=0.55,
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        title=f"Maker Split — {latest_period.strftime('%Y-%m')} (Total: {int(total_latest):,})",
                        hover_data={"Share_%": True, "value": True},
                    )
                    fig_donut.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        pull=[0.05] * len(d_latest),
                    )
                    fig_donut.update_layout(
                        template="plotly_white",
                        showlegend=True,
                        legend_title_text="Maker",
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Donut chart failed: {e}")
    
                # --- Sunburst Chart ---
                try:
                    sb = (
                        df_maker_all.groupby(["year", "label"], as_index=False)["value"]
                        .sum()
                        .sort_values(["year", "value"], ascending=[True, False])
                    )
    
                    fig_sb = px.sunburst(
                        sb,
                        path=["year", "label"],
                        values="value",
                        color="value",
                        color_continuous_scale="Sunset",
                        title="🌞 Sunburst — Year → Maker → Value",
                        hover_data={"value": True},
                    )
                    fig_sb.update_layout(template="plotly_white")
                    st.plotly_chart(fig_sb, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Sunburst chart failed: {e}")
    
                # --- Data summary below ---
                with st.expander("📋 Latest Period Maker Data Summary"):
                    st.dataframe(
                        d_latest.style.format(
                            {"value": "{:,}", "Share_%": "{:.2f}"}
                        ),
                        use_container_width=True,
                    )
            else:
                st.info("⚠️ Latest period has zero or empty Maker values.")
    
    if show_heatmap:
        st.markdown("### 🔥 Heatmap — Year × Maker (All-Maxed)")
    
        if pivot_year.empty:
            st.info("⚠️ No Maker data available for heatmap.")
        else:
            try:
                heat = pivot_year.copy()
                heat_norm = heat.div(heat.max(axis=1), axis=0).fillna(0)
                normalize_opt = st.toggle(
                    "Normalize heatmap (relative per year)",
                    value=True,
                    key=f"normalize_heatmap_{section_id}_pivot"
                )
                heat_used = heat_norm if normalize_opt else heat
    
                fig_h = go.Figure(
                    data=go.Heatmap(
                        z=heat_used.values,
                        x=heat_used.columns.astype(str),
                        y=heat_used.index.astype(str),
                        colorscale="Viridis",
                        hoverongaps=False,
                        texttemplate="%{z:.1f}" if normalize_opt else None,
                    )
                )
                fig_h.update_layout(
                    title=(
                        "Normalized Registrations by Maker per Year"
                        if normalize_opt
                        else "Absolute Registrations by Maker per Year"
                    ),
                    xaxis_title="Maker",
                    yaxis_title="Year",
                    template="plotly_white",
                    coloraxis_colorbar=dict(title="Registrations" if not normalize_opt else "Share (0–1)"),
                    height=500,
                )
                st.plotly_chart(fig_h, use_container_width=True)
    
                with st.expander("📋 View Heatmap Data Table"):
                    st.dataframe(
                        heat_used.round(2)
                        .style.format("{:,.0f}" if not normalize_opt else "{:.2f}")
                        .background_gradient(cmap="viridis"),
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"⚠️ Heatmap rendering failed: {e}")

    
    # -------------------------
    # 🌈 RADAR — Snapshot per Year (All-Maxed)
    # -------------------------
    if show_radar:
        st.markdown("### 🌈 Radar — Maker Profile Snapshot (All-Maxed)")
    
        if pivot_year.empty:
            st.info("⚠️ Not enough Maker data for radar visualization.")
        else:
            try:
                yrs_for_radar = sorted(pivot_year.index)[-min(4, len(pivot_year.index)):]
                makers = pivot_year.columns.tolist()
                radar_df = pivot_year.copy()
                radar_df_norm = radar_df.div(radar_df.max(axis=0), axis=1).fillna(0)
                normalize_radar = st.toggle("Normalize radar per Maker (0–1)", value=True)
                df_radar_used = radar_df_norm if normalize_radar else radar_df
    
                fig_r = go.Figure()
                for y in yrs_for_radar:
                    vals = df_radar_used.loc[y].values.tolist()
                    fig_r.add_trace(
                        go.Scatterpolar(
                            r=vals,
                            theta=makers,
                            fill="toself",
                            name=str(y),
                            hovertemplate="<b>%{theta}</b><br>Value: %{r:.2f}<extra></extra>",
                        )
                    )
    
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1] if normalize_radar else [0, df_radar_used.values.max()],
                            showline=True,
                            linewidth=1,
                            gridcolor="lightgray",
                        )
                    ),
                    showlegend=True,
                    title="Maker Distribution Radar (Last Years)",
                    template="plotly_white",
                    height=600,
                )
                st.plotly_chart(fig_r, use_container_width=True)
    
                with st.expander("📋 Radar Data Used"):
                    st.dataframe(
                        df_radar_used.round(2)
                        .style.format("{:,.0f}" if not normalize_radar else "{:.2f}")
                        .background_gradient(cmap="cool"),
                        use_container_width=True,
                    )
    
            except Exception as e:
                st.warning(f"⚠️ Radar chart rendering failed: {e}")

    # -------------------------
    # 🔮 FORECASTING — All-Maxed (Linear + Prophet + Auto Insights) — Maker
    # -------------------------
    if do_forecast:
        st.markdown("## 🔮 Forecasting — Maker (All-Maxed)")
    
        # ---------------------
        # 1️⃣ Select maker & horizon
        # ---------------------
        makers = pivot_year.columns.tolist() if not pivot_year.empty else df_maker_all["label"].unique().tolist()
    
        if not makers:
            st.info("⚠️ No makers available for forecasting.")
        else:
            maker_to_forecast = st.selectbox(
                "📊 Choose maker to forecast",
                makers,
                key=f"maker_select_{section_id}"
            )
            horizon_years = st.slider(
                "Forecast horizon (years)",
                1, 10, 3,
                key=f"horizon_slider_{section_id}"
            )
            st.caption("Select a maker and choose how many future years to forecast.")
    
            # ---------------------
            # 2️⃣ Prepare time series
            # ---------------------
            if maker_to_forecast in pivot_year.columns:
                series = pivot_year[[maker_to_forecast]].reset_index().rename(
                    columns={maker_to_forecast: "y", "index": "year"}
                )
            else:
                series = pd.DataFrame(columns=["year", "y"])
    
            if series.empty or series["y"].isna().all():
                st.info("⚠️ Insufficient data for forecasting this maker.")
            else:
                series["ds"] = pd.to_datetime(series["year"].astype(str) + "-01-01")
                series = series[["ds", "y"]].dropna()
    
                # ---------------------
                # 3️⃣ Linear Regression Forecast
                # ---------------------
                st.markdown("### 📈 Linear Regression Forecast")
                try:
                    import numpy as np
                    from sklearn.linear_model import LinearRegression
    
                    X = np.arange(len(series)).reshape(-1, 1)
                    y = series["y"].values
                    model = LinearRegression().fit(X, y)
                    fut_idx = np.arange(len(series) + horizon_years).reshape(-1, 1)
                    preds = model.predict(fut_idx)
    
                    fut_dates = pd.date_range(
                        start=series["ds"].iloc[0],
                        periods=len(series) + horizon_years,
                        freq="YS",
                    )
                    df_fore = pd.DataFrame({"ds": fut_dates, "Linear": preds})
                    df_fore["Type"] = ["Historical"] * len(series) + ["Forecast"] * horizon_years
    
                    fig_l = px.line(
                        df_fore, x="ds", y="Linear", color="Type",
                        title=f"Linear Trend Forecast — {maker_to_forecast}"
                    )
                    fig_l.add_scatter(
                        x=series["ds"], y=series["y"], mode="markers+lines",
                        name="Observed", line=dict(color="blue")
                    )
                    fig_l.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_l, use_container_width=True)
    
                    # KPI summary
                    last_val = series["y"].iloc[-1]
                    next_val = preds[len(series)]
                    growth = ((next_val - last_val) / last_val) * 100 if last_val else np.nan
                    st.metric(
                        "Next Year Projection",
                        f"{next_val:,.0f}",
                        f"{growth:+.1f}% vs last year"
                    )
                except Exception as e:
                    st.warning(f"⚠️ Linear regression forecast failed: {e}")
    
                # ---------------------
                # 4️⃣ Prophet Forecast (if available)
                # ---------------------
                st.markdown("### 🧙 Prophet Forecast (Advanced, if available)")
                try:
                    from prophet import Prophet  # Import inside try
                    m = Prophet(yearly_seasonality=True, seasonality_mode="multiplicative", changepoint_prior_scale=0.05)
                    m.fit(series)
                    future = m.make_future_dataframe(periods=horizon_years, freq="Y")
                    forecast = m.predict(future)
                
                    # Plot
                    figp = go.Figure()
                    figp.add_trace(go.Scatter(x=series["ds"], y=series["y"], mode="markers+lines", name="Observed", line=dict(color="blue")))
                    figp.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast (yhat)", line=dict(color="orange", width=3)))
                    figp.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(color="lightgray", dash="dot")))
                    figp.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(color="lightgray", dash="dot")))
                    figp.update_layout(title=f"Prophet Forecast — {maker_to_forecast}", template="plotly_white", height=550, legend=dict(orientation="h", y=-0.2), xaxis_title="Year", yaxis_title="Registrations")
                    st.plotly_chart(figp, use_container_width=True)
                
                    # Optional insight
                    fut_y = forecast.tail(horizon_years)["yhat"].mean()
                    st.success(f"📊 Prophet projects an **average of {fut_y:,.0f}** registrations/year for the next {horizon_years} years.")
                
                except ImportError:
                    st.info("🧠 Prophet not installed — only linear forecast shown.")
                except Exception as e:
                    st.warning(f"⚠️ Prophet forecast failed: {e}")

    
                # ---------------------
                # 5️⃣ Display Forecast Data
                # ---------------------
                with st.expander("📋 View Forecast Data Table"):
                    try:
                        comb = df_fore.copy()
                        if "forecast" in locals():
                            comb = comb.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="outer")
                        st.dataframe(comb.round(2).style.background_gradient(cmap="PuBuGn"), use_container_width=True)
                    except Exception:
                        st.dataframe(df_fore, use_container_width=True)

    # -------------------------
    # ⚠️ ANOMALY DETECTION — All-Maxed (Maker)
    # -------------------------
    if do_anomaly:
        st.markdown("## ⚠️ Anomaly Detection — Maker (All-Maxed)")
        st.caption("Detects outliers and abnormal spikes/drops per maker time series using IsolationForest + backup z-score method.")
    
        try:
            from sklearn.ensemble import IsolationForest
            import numpy as np
    
            anomalies = []
            anomaly_records = []
    
            if resampled.empty:
                st.warning("No resampled maker data available for anomaly detection.")
            else:
                makers = sorted(resampled["label"].unique())
                prog_bar = st.progress(0.0)
    
                for i, mk in enumerate(makers):
                    prog_bar.progress((i + 1) / len(makers))
                    ser = (
                        resampled[resampled["label"] == mk]
                        .set_index("ds")["value"]
                        .fillna(0)
                        .sort_index()
                    )
    
                    if len(ser) < 8 or ser.std() == 0:
                        continue
    
                    # --- Adaptive contamination ---
                    cont_rate = min(0.05, max(0.01, ser.std() / (ser.mean() + 1e-9) * 0.02))
    
                    # --- IsolationForest ---
                    try:
                        iso = IsolationForest(
                            contamination=cont_rate,
                            random_state=42,
                            n_estimators=200,
                            bootstrap=True,
                        )
                        X = ser.values.reshape(-1, 1)
                        preds = iso.fit_predict(X)
                        an_idxs = ser.index[preds == -1]
                        ser_an = ser.loc[an_idxs]
                        for dt, val in ser_an.items():
                            anomaly_records.append(
                                {"Maker": mk, "Date": dt, "Value": val}
                            )
                    except Exception:
                        # --- Fallback: rolling z-score ---
                        zscores = (ser - ser.rolling(6, min_periods=2).mean()) / ser.rolling(6, min_periods=2).std()
                        ser_an = ser[np.abs(zscores) > 2.8]
                        for dt, val in ser_an.items():
                            anomaly_records.append(
                                {"Maker": mk, "Date": dt, "Value": val}
                            )
    
                prog_bar.empty()
    
                # -------------------------------
                # 🧾 Summary + Visualization
                # -------------------------------
                if not anomaly_records:
                    st.success("✅ No significant anomalies detected across makers.")
                else:
                    df_an = pd.DataFrame(anomaly_records)
                    df_an["Date"] = pd.to_datetime(df_an["Date"])
                    st.markdown("### 📋 Detected Anomalies Summary — Makers")
                    st.dataframe(
                        df_an.sort_values("Date", ascending=False).style.format(
                            {"Value": "{:,.0f}"}
                        ),
                        use_container_width=True,
                        height=300,
                    )
    
                    # --- Maker selector for visualization ---
                    sel_mk = st.selectbox(
                        "🔍 View anomalies for a specific maker", sorted(df_an["Maker"].unique())
                    )
                    ser = (
                        resampled[resampled["label"] == sel_mk]
                        .set_index("ds")["value"]
                        .fillna(0)
                        .sort_index()
                    )
                    an_dates = df_an[df_an["Maker"] == sel_mk]["Date"]
    
                    fig_a = go.Figure()
                    fig_a.add_trace(
                        go.Scatter(
                            x=ser.index,
                            y=ser.values,
                            mode="lines+markers",
                            name="Value",
                            line=dict(color="steelblue"),
                        )
                    )
                    if not an_dates.empty:
                        fig_a.add_trace(
                            go.Scatter(
                                x=an_dates,
                                y=ser.loc[an_dates],
                                mode="markers",
                                name="Anomaly",
                                marker=dict(color="red", size=10, symbol="x"),
                            )
                        )
                    fig_a.update_layout(
                        title=f"Anomalies in {sel_mk} — Time Series Overlay",
                        template="plotly_white",
                        xaxis_title="Date",
                        yaxis_title="Registrations",
                        height=500,
                    )
                    st.plotly_chart(fig_a, use_container_width=True)
    
                    # Quick stats
                    st.info(f"📊 {len(df_an)} total anomalies detected across {len(makers)} makers.")
        except Exception as e:
            st.warning(f"⚠️ Anomaly detection failed: {e}")

    # -------------------------
    # 🔍 CLUSTERING (KMeans) — All-Maxed (Maker)
    # -------------------------
    if do_clustering:
        st.markdown("## 🔍 Clustering (KMeans) — All-Maxed Maker Mode")
        st.caption("Groups years by their maker registration mix using KMeans with normalization, PCA view & silhouette score.")
    
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
    
            if pivot_year.empty:
                st.warning("No pivot_year data available for clustering.")
            else:
                # Normalize the data
                X = pivot_year.fillna(0).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
    
                # Choose K range
                max_k = min(8, max(3, len(pivot_year) - 1))
                k = st.slider("Number of clusters (K)", 2, max_k, min(4, max_k))
    
                # Fit KMeans
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = km.fit_predict(X_scaled)
                inertia = km.inertia_
    
                # Compute silhouette (safe)
                sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
    
                df_cluster = pd.DataFrame({
                    "Year": pivot_year.index.astype(str),
                    "Cluster": labels
                })
                st.markdown("### 🧾 Cluster Assignment Summary — Makers")
                st.dataframe(df_cluster, use_container_width=True, height=300)
    
                c1, c2, c3 = st.columns(3)
                c1.metric("Clusters", str(k))
                c2.metric("Inertia (↓ better)", f"{inertia:.2f}")
                c3.metric("Silhouette (↑ better)", f"{sil:.3f}" if not np.isnan(sil) else "n/a")
    
                # --- Cluster Centers ---
                centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=pivot_year.columns)
                st.markdown("### 🧠 Cluster Centers — Approximate Maker Mix")
                st.dataframe(centers.style.format("{:,.0f}"), use_container_width=True)
    
                # --- PCA 2D visualization ---
                try:
                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X_scaled)
                    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
                    df_pca["Year"] = pivot_year.index.astype(str)
                    df_pca["Cluster"] = labels.astype(str)
    
                    fig_pca = px.scatter(
                        df_pca,
                        x="PC1",
                        y="PC2",
                        color="Cluster",
                        symbol="Cluster",
                        hover_data=["Year"],
                        title="📊 PCA Projection — Years clustered by Maker Mix",
                    )
                    fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))
                    fig_pca.update_layout(template="plotly_white", height=500)
                    st.plotly_chart(fig_pca, use_container_width=True)
                except Exception as e:
                    st.warning(f"PCA visualization failed: {e}")
    
                # --- Radar by cluster (avg maker mix) ---
                st.markdown("### 🌐 Radar — Cluster-wise Average Maker Mix")
                try:
                    fig_r = go.Figure()
                    for c in sorted(df_cluster["Cluster"].unique()):
                        cluster_mean = pivot_year.iloc[df_cluster["Cluster"] == c].mean()
                        fig_r.add_trace(go.Scatterpolar(
                            r=cluster_mean.values.tolist(),
                            theta=pivot_year.columns.tolist(),
                            fill="toself",
                            name=f"Cluster {c}",
                        ))
                    fig_r.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        title="Cluster-wise Average Maker Mix (Radar View)",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_r, use_container_width=True)
                except Exception as e:
                    st.warning(f"Radar view failed: {e}")
    
                st.success(f"✅ Clustering completed — {k} groups formed, silhouette={sil:.3f}.")
    
        except Exception as e:
            st.warning(f"Clustering failed: {e}")

    # =====================================================
    # 🤖 AI Narrative (All-Maxed, guarded, Maker)
    # =====================================================
    if enable_ai and do_forecast:
        st.markdown("## 🤖 AI Narrative (Summary & Recommendations) — All-Maxed Maker")
        try:
            # Basic safety checks
            if pivot_year is None or pivot_year.empty or df_cat_all is None or df_cat_all.empty:
                st.warning("⚠️ No valid data available for AI narrative.")
            else:
                # --- UI controls for AI block ---
                st.caption(
                    "Generate a concise analyst-style summary plus 3 actionable recommendations. "
                    "AI output requires a configured provider (universal_chat or other)."
                )
                allow_send = st.checkbox(
                    "I consent to sending an anonymized summary of aggregated metrics to an AI provider",
                    value=False
                )
                preview_prompt = st.checkbox("Preview the prompt sent to the AI (truncated)", value=False)
                show_raw = st.checkbox("Show raw AI output (if available)", value=False)
    
                # Prepare compact aggregated payload (limit tokens & remove PII)
                agg = pivot_year.reset_index().copy()
                # Ensure years are strings, and values are ints
                agg["year"] = agg["year"].astype(str)
                for col in agg.columns:
                    if col != "year":
                        agg[col] = agg[col].fillna(0).astype(int)
    
                # small as list of dicts but truncated to safe length
                small_records = agg.to_dict(orient="records")
                small_preview = json.dumps(small_records)[:3000]  # truncate for preview/prompt
    
                # Basic computed metrics for deterministic fallback and for context
                total_by_year = agg.set_index("year").sum(axis=1)
                total_first = float(total_by_year.iloc[0]) if len(total_by_year) > 0 else 0.0
                total_last = float(total_by_year.iloc[-1]) if len(total_by_year) > 0 else 0.0
                years_count = max(1, len(total_by_year) - 1)
                cagr_val = (
                    ((total_last / total_first) ** (1 / years_count) - 1) * 100
                    if total_first > 0 and len(total_by_year) > 1 else 0.0
                )
    
                # top makers overall + latest year growth per maker
                top_overall = (
                    df_cat_all.groupby("label")["value"].sum().sort_values(ascending=False)
                )
                top3 = top_overall.head(3)
                latest_year = pivot_year.index.max() if not pivot_year.empty else None
                growth_per_maker = {}
                if latest_year is not None and latest_year - 1 in pivot_year.index:
                    prev = pivot_year.loc[latest_year - 1] if (latest_year - 1) in pivot_year.index else None
                    curr = pivot_year.loc[latest_year]
                    if prev is not None:
                        for maker in pivot_year.columns:
                            prev_v = float(prev.get(maker, 0))
                            curr_v = float(curr.get(maker, 0))
                            growth_per_maker[maker] = (
                                ((curr_v - prev_v) / prev_v * 100) if prev_v > 0 else (100.0 if curr_v > 0 else 0.0)
                            )
    
                # Construct system + user prompts with brevity and clear instructions
                system_prompt = (
                    "You are a senior transport data analyst. Provide a concise, factual summary (3-6 bullet points) of trends "
                    "in vehicle maker registrations based on the provided aggregated metrics. Then give 3 short, prioritized, "
                    "actionable recommendations for policymakers or transport planners. Use percentages where relevant. Do not invent facts."
                )
    
                user_context = (
                    f"Aggregated yearly totals (truncated): {small_preview}\n\n"
                    f"Top makers overall: {', '.join(top3.index.tolist())}.\n"
                )
    
                # Preview prompt
                if preview_prompt:
                    st.expander("Prompt preview (truncated)", expanded=False).write({
                        "system": system_prompt,
                        "user": user_context[:4000]
                    })
    
                ai_text = None
                ai_raw = None
    
                # Only attempt network/LLM call if user consented
                if allow_send:
                    try:
                        # Try universal_chat if available
                        if "universal_chat" in globals() or "universal_chat" in locals():
                            ai_resp = universal_chat(system_prompt, user_context, stream=False, temperature=0.0, max_tokens=500, retries=2)
                            if isinstance(ai_resp, dict):
                                ai_raw = ai_resp
                                ai_text = ai_resp.get("text") or ai_resp.get("response") or ai_resp.get("output")
                            elif isinstance(ai_resp, str):
                                ai_text = ai_resp
                                ai_raw = {"text": ai_resp}
                        else:
                            st.info("No AI provider (`universal_chat`) found in this environment. Falling back to deterministic summary.")
                    except Exception as e:
                        st.warning(f"AI provider error or network issue: {e}")
                        ai_text = None
    
                # If AI didn't produce text, build deterministic humanized fallback summary
                if not ai_text:
                    bullets = []
                    # 1) Top makers
                    bullets.append(f"Top makers overall: {', '.join(top3.index.tolist())}.")
                    # 2) Trend direction by total
                    if not math.isnan(cagr_val) and abs(cagr_val) > 0.01:
                        dir_word = "increased" if cagr_val > 0 else "declined"
                        bullets.append(
                            f"Total registrations {dir_word} at ~{abs(cagr_val):.2f}% CAGR between {years[0]} and {years[-1]}."
                        )
                    else:
                        bullets.append("Total registrations remained roughly flat across the selected years.")
                    # 3) Notable maker moves (top few)
                    notable = []
                    for maker in top3.index:
                        g = growth_per_maker.get(maker, None)
                        if g is not None:
                            notable.append(f"{maker} {('up' if g>0 else 'down')} {abs(g):.1f}% YoY (latest).")
                    if notable:
                        bullets.append("Notable recent moves: " + "; ".join(notable))
                    # 4) Data quality note
                    bullets.append("Data note: aggregated counts shown; verify monthly cadence for short-term trends.")
                    # Recommendations (auto-generated)
                    recs = [
                        "Monitor and support high-growth makers (e.g., EV, light-commercial) with targeted policy incentives.",
                        "Improve inspection and road-safety programs focused on top-volume makers to reduce incidents.",
                        "Increase data cadence (move to monthly ingestion if possible) to enable finer forecasting and earlier anomaly detection."
                    ]
    
                    # Render fallback
                    st.markdown("### 🧠 Quick Narrative (deterministic fallback)")
                    for b in bullets:
                        st.markdown(f"- {b}")
                    st.markdown("**Recommendations:**")
                    for i, r in enumerate(recs, 1):
                        st.markdown(f"{i}. {r}")
    
                else:
                    # Show AI output and optionally raw JSON
                    st.markdown("### 🧠 AI Summary")
                    st.markdown(ai_text)
                    if show_raw and ai_raw is not None:
                        st.expander("Raw AI response (debug)", expanded=False).write(ai_raw)
    
                # Cache last ai_text in the current Streamlit session state for reuse
                try:
                    st.session_state["_last_ai_narrative"] = ai_text or "\n".join(bullets)
                except Exception:
                    pass
    
        except Exception as e:
            st.error(f"💥 AI Narrative generation failed: {e}")

    # =====================================================
    # 🧩 ALL-MAXED FINAL SUMMARY + DEBUG INSIGHTS (MAKER MODE)
    # =====================================================
    st.markdown("## 🧠 Final Summary & Debug Insights — ALL-MAXED Maker")
    
    try:
        summary_start = time.time()
    
        # ----------------------------------------------------
        # 1️⃣ SAFE INPUT + FALLBACKS
        # ----------------------------------------------------
        df_src = df_maker_all.copy() if "df_maker_all" in locals() else pd.DataFrame()
        freq = freq if "freq" in locals() else "Monthly"
        years = years if "years" in locals() and years else [2024, 2025]
        current_year = datetime.now().year
    
        if df_src.empty:
            st.warning("⚠️ No valid ALL-MAXED Maker data found to summarize.")
            st.stop()
    
        # ----------------------------------------------------
        # 2️⃣ BASIC CLEANUP + ADD TIME FIELDS
        # ----------------------------------------------------
        df_src = df_src.copy()
        if "ds" not in df_src.columns and "date" in df_src.columns:
            df_src["ds"] = pd.to_datetime(df_src["date"])
        elif "ds" not in df_src.columns:
            df_src["ds"] = pd.to_datetime(df_src["year"].astype(str) + "-01-01")
    
        df_src["year"] = df_src["ds"].dt.year
        df_src["month"] = df_src["ds"].dt.month
        df_src["maker"] = df_src["label"].astype(str)
    
        # ----------------------------------------------------
        # 3️⃣ RESAMPLING BASED ON FREQUENCY
        # ----------------------------------------------------
        resampled = (
            df_src.groupby(["maker", "year"])["value"].sum().reset_index()
            if freq == "Yearly"
            else df_src.copy()
        )
    
        pivot = resampled.pivot_table(
            index="ds", columns="maker", values="value", aggfunc="sum"
        ).fillna(0)
    
        pivot_year = (
            resampled.pivot_table(
                index="year", columns="maker", values="value", aggfunc="sum"
            ).fillna(0)
            if "year" in resampled.columns
            else pd.DataFrame()
        )
    
        # ----------------------------------------------------
        # 4️⃣ KPI METRICS (YoY, CAGR, MoM, Maker Shares)
        # ----------------------------------------------------
        st.subheader("💎 Key Metrics & Growth (All-Maxed Maker)")
    
        if pivot_year.empty:
            st.warning("⚠️ No yearly data found for KPI computation.")
            st.stop()
    
        # --- Compute totals and YoY
        year_totals = pivot_year.sum(axis=1).rename("TotalRegistrations").to_frame()
        year_totals["YoY_%"] = year_totals["TotalRegistrations"].pct_change() * 100
        year_totals["TotalRegistrations"] = year_totals["TotalRegistrations"].fillna(0).astype(int)
        year_totals["YoY_%"] = year_totals["YoY_%"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
        # --- CAGR
        if len(year_totals) >= 2:
            first = float(year_totals["TotalRegistrations"].iloc[0])
            last = float(year_totals["TotalRegistrations"].iloc[-1])
            years_count = max(1, len(year_totals) - 1)
            cagr = ((last / first) ** (1 / years_count) - 1) * 100 if first > 0 else 0.0
        else:
            cagr = 0.0
    
        # --- MoM if monthly
        if freq == "Monthly":
            resampled["month_period"] = resampled["year"].astype(str) + "-" + resampled["month"].astype(str)
            month_totals = resampled.groupby("month_period")["value"].sum().reset_index()
            month_totals["MoM_%"] = month_totals["value"].pct_change() * 100
            latest_mom = (
                f"{month_totals['MoM_%'].iloc[-1]:.2f}%"
                if len(month_totals) > 1 and not np.isnan(month_totals["MoM_%"].iloc[-1])
                else "n/a"
            )
        else:
            latest_mom = "n/a"
    
        # --- Maker shares
        latest_year = int(year_totals.index.max())
        latest_total = int(year_totals.loc[latest_year, "TotalRegistrations"])
        maker_share = (
            (pivot_year.loc[latest_year] / pivot_year.loc[latest_year].sum() * 100)
            .sort_values(ascending=False)
            .round(1)
        )
    
        # ----------------------------------------------------
        # 5️⃣ DISPLAY KPIs
        # ----------------------------------------------------
        c1, = st.columns(1)
        c1.metric("📅 Years Loaded", f"{years[0]} → {years[-1]}", f"{len(years)} yrs")
    
        st.markdown("#### 📘 Maker Share (Latest Year)")
        st.dataframe(
            pd.DataFrame({
                "Maker": maker_share.index,
                "Share_%": maker_share.values,
                "Volume": pivot_year.loc[latest_year].astype(int).values,
            }).sort_values("Share_%", ascending=False),
            use_container_width=True,
        )
    
        with st.expander("🔍 Yearly Totals & Growth"):
            st.dataframe(year_totals.style.format({"TotalRegistrations": "{:,}", "YoY_%": "{:.2f}"}))
    
        # ----------------------------------------------------
        # 6️⃣ DEEP INSIGHTS + TRENDS — FIXED
        # ----------------------------------------------------
        total_all = df_src["value"].sum()
        n_makers = df_src["maker"].nunique()
        n_years = df_src["year"].nunique()
    
        # --- Top Maker
        top_maker_row = (
            df_src.groupby("maker")["value"]
            .sum()
            .reset_index()
            .sort_values("value", ascending=False)
            .iloc[0]
        )
        top_maker = {
            "maker": str(top_maker_row["maker"]),
            "value": float(top_maker_row["value"])
        }
        top_maker_share = (top_maker["value"] / total_all) * 100 if total_all > 0 else 0
    
        # --- Top Year
        top_year_row = (
            df_src.groupby("year")["value"]
            .sum()
            .reset_index()
            .sort_values("value", ascending=False)
            .iloc[0]
        )
        top_year = {
            "year": int(top_year_row["year"]),
            "value": float(top_year_row["value"])
        }
    
        # --- Display metrics safely
        st.metric("🏆 Absolute Top Maker", top_maker["maker"], f"{top_maker_share:.2f}% share")
        st.metric("📅 Peak Year", f"{top_year['year']}", f"{top_year['value']:,.0f} registrations")
    
        # --- Plot: Top 10 Makers
        st.write("### 🧾 Top 10 Makers — Overall")
        top_debug = (
            df_src.groupby("maker")["value"]
            .sum()
            .reset_index()
            .sort_values("value", ascending=False)
        )
        fig_top10 = px.bar(
            top_debug.head(10),
            x="maker",
            y="value",
            text_auto=True,
            color="value",
            color_continuous_scale="Blues",
            title="Top 10 Makers (All Years)",
        )
        fig_top10.update_layout(template="plotly_white", margin=dict(t=50, b=40))
        st.plotly_chart(fig_top10, use_container_width=True)
    
        # ----------------------------------------------------
        # 7️⃣ ADVANCED DEBUG METRICS
        # ----------------------------------------------------
        volatility = (
            df_src.groupby("year")["value"].sum().pct_change().std() * 100
            if len(df_src["year"].unique()) > 2
            else 0
        )
        dominance_ratio = (top_maker["value"] / total_all) * n_makers if total_all > 0 else 0
        direction = "increased" if cagr > 0 else "declined"
    
        summary_time = time.time() - summary_start
        st.markdown("### ⚙️ Debug Performance Metrics")
        st.code(
            f"""
    Years analyzed: {years}
    Makers: {n_makers}
    Rows processed: {len(df_src):,}
    Total registrations: {total_all:,.0f}
    Top maker: {top_maker['maker']} → {top_maker['value']:,.0f} ({top_maker_share:.2f}%)
    Peak year: {int(top_year['year'])} → {top_year['value']:,.0f}
    Dominance ratio: {dominance_ratio:.2f}
    Runtime: {summary_time:.2f}s
            """,
            language="yaml",
        )
    
        # ----------------------------------------------------
        # 8️⃣ SMART SUMMARY
        # ----------------------------------------------------
        if top_maker and years and top_year:
            st.success(
                f"From **{years[0]}** to **{years[-1]}**, total registrations {direction}. "
                f"**{top_maker.get('maker', 'N/A')}** leads with **{top_maker_share:.2f}%** share. "
                f"Peak year: **{top_year['year']}** with **{top_year['value']:,.0f}** registrations."
            )
            logger.info(f"✅ ALL-MAXED Maker summary completed in {summary_time:.2f}s")
        else:
            st.error("⛔ ALL-MAXED Maker summary failed: Missing or invalid data.")
            logger.warning("⚠️ ALL-MAXED Maker summary skipped due to incomplete data.")
    
        # ----------------------------------------------------
        # 9️⃣ CATCH GLOBAL ERRORS
    # ----------------------------------------------------
    except Exception as e:
        logger.exception(f"ALL-MAXED Maker summary failed: {e}")
        st.error(f"⛔ ALL-MAXED Maker summary failed: {e}")
    
# -----------------------------------------------------
# 🧩 Safe Entry Point — Streamlit-only Execution Guard
# -----------------------------------------------------
if __name__ == "__main__":
    import streamlit as st
    
    st.markdown("# 🚗 ALL-MAXED MAKER ANALYTICS")
    
    try:
        all_maxed_maker_block()
    except Exception as e:
        import traceback
        st.error(f"💥 Error while rendering All-Maxed block: {e}")
        st.code(traceback.format_exc(), language="python")

import streamlit as st
import pandas as pd 
import numpy as np 
import random 
from datetime import datetime 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans 

import pandas as pd
import random
import streamlit as st

def safe_get_top5_(params):
    """
    Fetch top 5 revenue states from API.
    If API fails, generates realistic fallback data.
    Returns a DataFrame with columns ['State','Revenue'].
    """
    try:
        top5_json, url = get_json("vahandashboard/top5chartRevenueFee", params)
        df = parse_top5_revenue(top5_json)
        # Ensure required columns exist
        if df.empty or "State" not in df.columns or "Revenue" not in df.columns:
            raise ValueError("API returned invalid data")
        st.success(f"✅ Top 5 revenue fetched from API ({url})")
    except Exception as e:
        st.warning(f"⚠️ API unavailable or failed: {e}\nUsing fallback mock data.")
        states = ["MH", "DL", "KA", "TN", "UP"]
        revenues = [random.randint(500, 2000) for _ in states]
        df = pd.DataFrame({"State": states, "Revenue": revenues})
    # Make sure column names are consistent
    df = df[["State", "Revenue"]]
    return df


st.markdown("## 💰 ALL-MAXED — Top 5 Revenue States Analytics Suite")
df_top5 = safe_get_top5_(params_common1)

# Always rename for plotting consistency
df_top5_plot = df_top5.rename(columns={"Revenue": "Revenue_₹Cr"})

# Bar chart
fig_bar = px.bar(
    df_top5_plot,
    x="State",
    y="Revenue_₹Cr",
    title="Top 5 Revenue States (Bar)",
    text="Revenue_₹Cr",
    labels={"Revenue_₹Cr": "Revenue (₹ Cr)", "State": "State"}
)
fig_bar.update_layout(template="plotly_white")
st.plotly_chart(fig_bar, use_container_width=True)

# Pie chart
fig_pie = px.pie(
    df_top5_plot,
    names="State",
    values="Revenue_₹Cr",
    title="Top 5 Revenue States (Pie)",
    hole=0.4
)
st.plotly_chart(fig_pie, use_container_width=True)

# KPI Summary
total_rev = df_top5_plot["Revenue_₹Cr"].sum()
top_state = df_top5_plot.loc[df_top5_plot["Revenue_₹Cr"].idxmax(), "State"]
top_value = df_top5_plot["Revenue_₹Cr"].max()

st.markdown("### 💎 Key Metrics")
st.write(f"- **Total Revenue:** ₹{total_rev:,} Cr")
st.write(f"- **Top State:** {top_state} with ₹{top_value:,} Cr")

# --------------------------------------------------
# 🔹 Advanced Analytics — Trend Simulation
# --------------------------------------------------
st.markdown("### 🔮 Simulated Multi-Year Trend (Safe + Maxed)")

# Generate mock revenue trend per state (2019-2025)
years = list(range(2019, 2026))
trend_data = []
for state in df_top5["State"]:
    base = df_top5.loc[df_top5["State"]==state, "Revenue"].values[0]
    for y in years:
        val = int(base * (1 + 0.08*(y-2019)) + random.randint(-50,50))
        trend_data.append({"State": state, "Year": y, "Revenue": val})
df_trend = pd.DataFrame(trend_data)

# Multi-year comparison chart
fig_trend = px.line(df_trend, x="Year", y="Revenue", color="State",
                    markers=True, title="Top 5 Revenue States: Multi-Year Trend")
st.plotly_chart(fig_trend, use_container_width=True)

# --------------------------------------------------
# 🔹 Anomaly Detection (IsolationForest)
# --------------------------------------------------
st.markdown("### ⚠️ Anomaly Detection — Revenue Trends")
try:
    df_trend["Anomaly"] = df_trend.groupby("State")["Revenue"].transform(
        lambda x: IsolationForest(contamination=0.1, random_state=42).fit_predict(x.values.reshape(-1,1))
    )
    fig_anom = px.scatter(df_trend, x="Year", y="Revenue", color=df_trend["Anomaly"].map({1:"Normal",-1:"Anomaly"}),
                          facet_col="State", title="Revenue Trend Anomalies by State")
    st.plotly_chart(fig_anom, use_container_width=True)
except Exception as e:
    st.warning(f"Anomaly detection failed: {e}")

# --------------------------------------------------
# 🔹 Clustering (KMeans)
# --------------------------------------------------
st.markdown("### 🔍 Clustering — Revenue Patterns Across States")
try:
    pivot = df_trend.pivot(index="Year", columns="State", values="Revenue").fillna(0)
    k = min(3, len(pivot.columns))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    pivot["Cluster"] = km.fit_predict(pivot.values)
    st.dataframe(pivot)
except Exception as e:
    st.warning(f"Clustering unavailable: {e}")

st.markdown("---")
st.success("✅ ALL-MAXED Top 5 Revenue States Dashboard Ready!")


# ---------- Trend series + resampling & multi-year comparisons ------------------
with st.spinner('Fetching trend series...'):
    tr_json, tr_url = get_json('vahandashboard/vahanyearwiseregistrationtrend', params)
    df_tr = to_df(tr_json)

if not df_tr.empty:
    def parse_label(l):
        for fmt in ('%Y-%m-%d','%Y-%m','%b %Y','%Y'):
            try: 
                return pd.to_datetime(l, format=fmt)
            except: 
                pass
        try:
            return pd.to_datetime(l)
        except:
            return pd.NaT

    df_tr['date'] = df_tr['label'].apply(parse_label)
    df_tr = df_tr.dropna(subset=['date']).sort_values('date')
    df_tr['value'] = pd.to_numeric(df_tr['value'], errors='coerce')
    df_tr = df_tr.set_index('date')

    freq_map = {'Daily': 'D', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
    df_tr = df_tr.resample(freq_map.get(frequency, 'M')).sum()

# ---------------- MULTI-YEAR COMPARISONS ----------------
st.subheader('📈 Multi-year Comparisons')

if df_tr.empty:
    st.warning('No trend data for chosen filters.')
else:
    df_tr['year'] = df_tr.index.year
    years = sorted(df_tr['year'].unique())

    selected_years = st.sidebar.multiselect(
        'Select years to compare',
        years,
        default=years if len(years) <= 2 else years[-2:]
    )

    # Show separate year charts
    st.markdown('### 🔹 Separate Year Charts')
    cols = st.columns(len(selected_years) if selected_years else 1)
    for y, c in zip(selected_years, cols):
        with c:
            s = df_tr[df_tr['year'] == y]['value']
            st.markdown(f"**{y}**")
            st.line_chart(s)

    # Combined comparison chart
    st.markdown('### 🔸 Combined Comparison (Each Year as a Separate Line)')

    df_tr_reset = df_tr.reset_index()

    # Choose x-axis format
    if frequency == 'Yearly':
        df_tr_reset['period_label'] = df_tr_reset['date'].dt.strftime('%Y')
    elif frequency == 'Quarterly':
        df_tr_reset['period_label'] = 'Q' + df_tr_reset['date'].dt.quarter.astype(str)
    elif frequency == 'Monthly':
        df_tr_reset['period_label'] = df_tr_reset['date'].dt.strftime('%b')
    else:  # Daily or others
        df_tr_reset['period_label'] = df_tr_reset['date'].dt.strftime('%d-%b')

    pivot = (
        df_tr_reset.pivot_table(
            index='period_label',
            columns='year',
            values='value',
            aggfunc='sum'
        )
        .fillna(0)
    )

    # Plot combined line chart with Plotly
    fig = px.line(
        pivot,
        x=pivot.index,
        y=pivot.columns,
        markers=True,
        title="Multi-Year Comparison of Registrations",
        labels={"x": "Period", "value": "Registrations"},
    )
    fig.update_layout(template="plotly_white", legend_title_text="Year")
    st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 📈 ALL-MAXED — Time Series Trend Analytics Suite
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta

st.markdown("## 📈 ALL-MAXED — Time Series Trend + Growth Analytics")

import pandas as pd
import numpy as np
from datetime import datetime
import random
import logging
from colorama import Fore

def safe_get_trend(params):
    """
    🔥 ALL-MAXED SAFE TREND FETCHER — No pre dependencies.
    Fetches registration trend from API or generates a realistic fallback (2020–2026).
    Handles missing keys, network failures, and ensures normalized dataframe output.
    Returns:
        (df_trend, tr_url)
    """
    logger = logging.getLogger(__name__)

    # --- 1️⃣ Attempt real API fetch
    try:
        tr_json, tr_url = get_json("vahandashboard/vahanyearwiseregistrationtrend", params)
        logger.info(Fore.CYAN + f"✅ Trend API success → {tr_url}")
    except Exception as e:
        logger.error(Fore.RED + f"❌ Trend fetch failed: {e}")
        tr_json, tr_url = None, "MOCK://vahan/trend"

    # --- 2️⃣ Validate / fallback to mock if API empty or invalid
    if not tr_json or (
        isinstance(tr_json, dict)
        and not any(k in tr_json for k in ["data", "result", "datasets"])
    ):
        logger.warning(Fore.YELLOW + f"⚠️ Using fallback trend data ({tr_url})")

        # Simulate monthly trend 2020–2026 with YoY + seasonality + noise
        np.random.seed(42)
        base_value = 480_000
        months = pd.date_range("2020-01-01", "2026-12-01", freq="MS")

        data = []
        for dt in months:
            growth = 1 + 0.07 * (dt.year - 2020)        # 7% YoY growth
            seasonal = 1 + 0.12 * np.sin((dt.month / 12) * 2 * np.pi)
            noise = np.random.normal(1.0, 0.05)
            val = int(base_value * growth * seasonal * noise)
            data.append({"date": dt.strftime("%Y-%m-%d"), "value": val})

        tr_json = {"data": data}

    # --- 3️⃣ Normalize structure regardless of source
    try:
        df = normalize_trend(tr_json)
    except Exception:
        df = pd.DataFrame(tr_json.get("data", tr_json))
        if "date" not in df.columns:
            df["date"] = pd.date_range("2020-01-01", periods=len(df), freq="MS")
        if "value" not in df.columns:
            df["value"] = np.random.randint(300_000, 900_000, len(df))

    # --- 4️⃣ Final cleanup + metadata
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    logger.info(Fore.GREEN + f"📈 Trend data ready → {len(df)} rows from {df['year'].min()}–{df['year'].max()}")
    return df, tr_url

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging
from colorama import Fore

logger = logging.getLogger(__name__)

# -------------------------------------------------
# 🔥 ALL-MAXED Growth Table Generator + Fallback
# -------------------------------------------------
def safe_get_growth_table(calendarType, params, label="Growth"):
    """
    Unified ALL-MAXED growth table fetcher with mock fallback.
    calendarType: 1=Yearly, 2=Quarterly, 3=Monthly, 4=Daily
    Returns DataFrame with columns ['label','value'] and realistic data.
    """

    try:
        j, url = get_json(
            "vahandashboard/durationWiseRegistrationTable",
            {**params, "calendarType": calendarType}
        )
    except Exception as e:
        logger.warning(Fore.YELLOW + f"⚠️ Growth fetch failed for {label}: {e}")
        j, url = None, f"MOCK://growth/{label.lower()}"

    # If invalid JSON, use mock
    if not j or (isinstance(j, dict) and j.get("error")):
        logger.warning(Fore.YELLOW + f"⚠️ Using mock fallback for {label} ({url})")
        j = {"data": []}

    # Try parse
    try:
        df = parse_duration_table(j)
    except Exception as e:
        logger.warning(Fore.RED + f"💥 Failed to parse duration table: {e}")
        df = pd.DataFrame(columns=["label", "value"])

    # Fallback with realistic mock data if empty
    if df.empty:
        np.random.seed(42)
        now = datetime.now()

        if calendarType == 1:  # Yearly
            years = list(range(2020, 2026))
            values = [random.randint(500000, 2500000) * (1 + 0.1*(y-2020)) for y in years]
            df = pd.DataFrame({"label": [str(y) for y in years], "value": values})

        elif calendarType == 2:  # Quarterly
            quarters = [f"Q{q} {y}" for y in range(2020, 2026) for q in range(1, 5)]
            values = np.random.randint(100000, 700000, len(quarters))
            df = pd.DataFrame({"label": quarters, "value": values})

        elif calendarType == 3:  # Monthly
            months = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
            df = pd.DataFrame({
                "label": [dt.strftime("%b %Y") for dt in months],
                "value": (
                    (np.sin(np.arange(len(months)) / 6) * 0.15 + 1.05)
                    * np.linspace(400000, 950000, len(months))
                    + np.random.randint(-50000, 50000, len(months))
                ).astype(int)
            })

        elif calendarType == 4:  # Daily
            days = pd.date_range(now - timedelta(days=60), now, freq="D")
            df = pd.DataFrame({
                "label": [d.strftime("%d-%b") for d in days],
                "value": np.maximum(
                    0,
                    (np.sin(np.arange(len(days)) / 3) * 0.2 + 1)
                    * np.random.randint(15000, 30000, len(days))
                ).astype(int)
            })

        else:
            df = pd.DataFrame(columns=["label", "value"])

        logger.info(Fore.CYAN + f"✅ Generated mock {label} data ({len(df)} rows)")

    return df, url


# 🔧 Pull params from globals (no pre)
params = globals().get("params", {})

# 🚀 Generate all four
df_quarter = safe_get_growth_table(2, params, "Quarterly")[0]
df_year    = safe_get_growth_table(1, params, "Yearly")[0]
df_month   = safe_get_growth_table(3, params, "Monthly")[0]
df_daily   = safe_get_growth_table(4, params, "Daily")[0]

import streamlit as st
import pandas as pd
import numpy as np
import random, math
from datetime import datetime, timedelta
from colorama import Fore
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# 🧱 Safe JSON + Parse Helpers (Stubs if not loaded)
# --------------------------------------------------
def get_json(endpoint, params):
    """Stub for API fetching, replace with real API if available."""
    raise RuntimeError("API not available (mock mode)")

def parse_revenue_trend(j):
    """Stub for API parser, replace if real format known."""
    if not j or "data" not in j:
        return pd.DataFrame(columns=["period", "value", "year"])
    return pd.DataFrame(j["data"])

# ----------------------------------------------
# 💰 Safe Revenue Trend Fetcher (Mock Supported)
# ----------------------------------------------
def safe_get_revenue_trend(params):
    """Fetches Vahan revenue trend with realistic fallback simulation."""
    try:
        j, url = get_json("vahandashboard/revenueFeeLineChart", params)
    except Exception as e:
        print(Fore.YELLOW + f"⚠️ Revenue trend API failed → {e}")
        j, url = None, "MOCK://revenue"

    try:
        df = parse_revenue_trend(j)
    except Exception as e:
        print(Fore.RED + f"💥 Revenue parsing failed → {e}")
        df = pd.DataFrame(columns=["period", "value", "year"])

    if df.empty:
        np.random.seed(42)
        now_year = datetime.now().year
        years = list(range(2019, now_year + 1))
        periods = [f"FY{y}-{str(y+1)[-2:]}" for y in years]

        base_values = []
        for i, y in enumerate(years):
            growth_factor = 1 + 0.08 * (i - 2)
            covid_penalty = 0.6 if y == 2020 else (0.85 if y == 2021 else 1)
            ev_boost = 1.2 if y >= 2024 else 1
            base_values.append(int(2000 * growth_factor * covid_penalty * ev_boost + random.randint(-200, 200)))

        df = pd.DataFrame({
            "period": np.repeat(periods, 4),
            "year": np.repeat(years, 4),
            "value": np.concatenate([
                np.round(np.linspace(v * 0.18, v * 0.3, 4) + np.random.randint(-20, 20, 4))
                for v in base_values
            ])
        })

        print(Fore.CYAN + f"✅ Generated mock revenue trend ({len(df)} points)")
    return df


# ===============================================================
# 📊 MAIN VISUALIZATION — REVENUE TREND + ANALYTICS
# ===============================================================

params = globals().get("params", {})

with st.spinner("Fetching revenue trend..."):
    df_rev_trend = safe_get_revenue_trend(params)

if not df_rev_trend.empty:
    import altair as alt
    st.subheader("💰 Revenue Trend Comparison")
    chart = alt.Chart(df_rev_trend).mark_line(point=True).encode(
        x=alt.X('period:O', title='Period'),
        y=alt.Y('value:Q', title='Revenue (₹ Cr)'),
        color='year:N'
    ).properties(title="Revenue Trend Comparison (Actual or Mock)")
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning("No revenue data available.")

# ===============================================================
# 🧠 ADVANCED TREND ANALYTICS (Safe + Maxed)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Try optional libraries safely
try:
    import pycountry
except ImportError:
    st.warning("Installing pycountry...")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "pycountry"])
    import pycountry

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
except ImportError:
    st.warning("Installing scikit-learn...")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans

st.markdown("---")
st.subheader("🧠 Advanced Trend Analytics")

# ===============================================================
# 🔹 Load df_trend from API or fallback
# ===============================================================
def load_trend_data():
    try:
        # Example safe API call (replace get_json with your actual function)
        params_common = {"state_cd": "ALL", "veh_catg": "ALL", "fuel_type": "ALL", "maker": "ALL"}
        tr_json, tr_url = get_json("vahandashboard/vahanyearwiseregistrationtrend", params_common)
        df = normalize_trend(tr_json)
        if df.empty:
            raise ValueError("Empty trend data from API")
        st.success(f"✅ Loaded trend data from API ({len(df):,} records)")
        return df
    except Exception as e:
        st.warning(f"⚠️ Trend API unavailable — generating mock data: {e}")
        now = datetime.now()
        dates = pd.date_range(now - timedelta(days=365 * 5), now, freq="M")
        df = pd.DataFrame({
            "date": dates,
            "value": (np.sin(np.arange(len(dates)) / 4) * 0.1 + 1.05)
                    * np.linspace(500000, 950000, len(dates))
                    + np.random.randint(-30000, 30000, len(dates))
        }).astype({"value": int})
        return df

df_trend = load_trend_data()

# ===============================================================
# 1️⃣ KPI Summary
# ===============================================================
st.markdown("### 💎 Key Metrics")
df_trend["date"] = pd.to_datetime(df_trend["date"])
df_trend["year"] = df_trend["date"].dt.year

year_sum = df_trend.groupby("year")["value"].sum()
total_cagr = ((year_sum.iloc[-1] / year_sum.iloc[0]) ** (1 / (len(year_sum) - 1)) - 1) * 100 if len(year_sum) > 1 else np.nan
latest_yoy = ((year_sum.iloc[-1] - year_sum.iloc[-2]) / year_sum.iloc[-2]) * 100 if len(year_sum) > 1 else np.nan

c1, = st.columns(1)
c1.metric("📅 Duration", f"{year_sum.index.min()} → {year_sum.index.max()}", f"{len(year_sum)} years")

# ===============================================================
# 2️⃣ Heatmap — Year × Month
# ===============================================================
st.markdown("### 🔥 Heatmap — Registrations by Month & Year")
df_trend["month_name"] = df_trend["date"].dt.strftime("%b")
heat = df_trend.pivot_table(index="year", columns="month_name", values="value", aggfunc="sum").fillna(0)
fig_h = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Viridis"))
fig_h.update_layout(title="Heatmap: Year × Month", xaxis_title="Month", yaxis_title="Year")
st.plotly_chart(fig_h, use_container_width=True)

# ===============================================================
# 3️⃣ Forecasting — Linear + Prophet
# ===============================================================
st.markdown("### 🔮 Forecasting — Registrations")
from dateutil.relativedelta import relativedelta

series = df_trend.groupby("year")["value"].sum().reset_index()
series.columns = ["ds", "y"]
series["ds"] = pd.to_datetime(series["ds"].astype(str) + "-01-01")
horizon = st.slider(
    "Forecast horizon (years)",
    1,
    10,
    3,
    key="forecast_horizon_slider_unique_001"
)

# Linear Forecast
try:
    X = np.arange(len(series)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, series["y"])
    fut_idx = np.arange(len(series) + horizon).reshape(-1, 1)
    preds = lr.predict(fut_idx)
    fut_dates = pd.to_datetime([(series["ds"].iloc[0] + relativedelta(years=int(i))).strftime("%Y-01-01") for i in range(len(series) + horizon)])
    df_fore = pd.DataFrame({"ds": fut_dates, "Linear Forecast": preds})
    fig_lin = px.line(df_fore, x="ds", y="Linear Forecast", title="📈 Linear Forecast")
    fig_lin.add_scatter(x=series["ds"], y=series["y"], mode="lines+markers", name="Historical")
    st.plotly_chart(fig_lin, use_container_width=True)
except Exception as e:
    st.warning(f"Linear forecast failed: {e}")

# Prophet Forecast
if Prophet:
    try:
        m = Prophet(yearly_seasonality=True)
        m.fit(series)
        future = m.make_future_dataframe(periods=horizon, freq="Y")
        forecast = m.predict(future)
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=series["ds"], y=series["y"], name="Historical"))
        figp.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Prophet Forecast"))
        figp.update_layout(title="🧭 Prophet Forecast (Yearly)")
        st.plotly_chart(figp, use_container_width=True)
    except Exception as e:
        st.info(f"Prophet forecast failed: {e}")
else:
    st.info("Prophet not installed or unavailable.")

# ===============================================================
# 4️⃣ Anomaly Detection
# ===============================================================
st.markdown("### ⚠️ Anomaly Detection (IsolationForest)")
try:
    iso = IsolationForest(contamination=0.03, random_state=42)
    preds = iso.fit_predict(df_trend[["value"]])
    df_trend["anomaly"] = preds
    fig_a = px.scatter(df_trend, x="date", y="value",
                       color=df_trend["anomaly"].map({1: "Normal", -1: "Anomaly"}))
    fig_a.update_layout(title="Anomaly Detection — Time Series", legend_title="Status")
    st.plotly_chart(fig_a, use_container_width=True)
except Exception as e:
    st.warning(f"Anomaly detection failed: {e}")

# ===============================================================
# 5️⃣ Clustering — Monthly Patterns
# ===============================================================
st.markdown("### 🔍 Clustering (KMeans — Monthly Patterns)")
try:
    monthly = df_trend.copy()
    monthly["month"] = monthly["date"].dt.month
    month_pivot = monthly.pivot_table(index="year", columns="month", values="value", aggfunc="sum").fillna(0)
    k = st.slider("Clusters (k)", 2, min(10, len(month_pivot)), 3)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(month_pivot)
    month_pivot["Cluster"] = km.labels_
    st.dataframe(month_pivot)
    fig_c = px.scatter(month_pivot.reset_index(), x="year", y=month_pivot.columns[0],
                       color="Cluster", title="Cluster Visualization")
    st.plotly_chart(fig_c, use_container_width=True)
except Exception as e:
    st.info(f"Clustering unavailable: {e}")

# ===============================================================
# ✅ Final Summary + Data Quality
# ===============================================================
st.markdown("---")
st.subheader("🧾 Final Summary & Debug Info")

try:
    total_records = len(df_trend)
    total_sum = df_trend["value"].sum()
    peak_value = df_trend["value"].max()
    peak_date = df_trend.loc[df_trend["value"].idxmax(), "date"].strftime("%Y-%m-%d")
    st.write(f"- **Total Records:** {total_records}")
    st.write(f"- **Total Registrations:** {total_sum:,.0f}")
    st.write(f"- **Peak:** {peak_value:,.0f} on **{peak_date}**")
except Exception as e:
    st.warning(f"Recap failed: {e}")

try:
    missing_ratio = df_trend["value"].isna().mean()
    if missing_ratio > 0:
        st.warning(f"⚠️ Missing values detected: {missing_ratio*100:.2f}%")
    else:
        st.success("✅ No missing values detected")
except Exception as e:
    st.warning(f"Data quality check failed: {e}")



# ---------- Forecasting & Anomalies -------------------------------------------
# --- Ensure required variables exist ---
# ================================
# ALLL-MAXED Forecasting & Anomaly
# ================================

import pandas as pd
import numpy as np
import streamlit as st
import math

# -------------------------
# Permanently enable ML
# -------------------------
enable_ml = True

# -------------------------
# Ensure historical data exists
# -------------------------
if "df_tr" not in globals() or df_tr is None or df_tr.empty:
    # Create minimal synthetic timeseries
    dates = pd.date_range("2023-01-01", periods=12, freq="M")
    values = np.random.randint(1000, 5000, size=len(dates))
    df_tr = pd.DataFrame({"date": dates, "value": values})

freq_map = {"M": "MS", "Y": "YS"}
frequency = "M"

def lazy(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return None

try:
    from prophet import Prophet
    prophet_mod = Prophet
except Exception:
    prophet_mod = None

# ---------- Forecasting & Anomaly Detection ----------
if enable_ml and not df_tr.empty:
    st.subheader("📊 Forecasting & Anomaly Detection — ALLL-MAXED")

    fc_col1, fc_col2 = st.columns([2,3])
    with fc_col1:
        method = st.selectbox(
            "Forecast method",
            ["Naive seasonality", "SARIMAX", "Prophet", "RandomForest", "XGBoost"],
            key="forecast_method_maxed"
        )
        horizon = st.number_input("Forecast horizon (periods)", 1, 60, 12, key="forecast_horizon_maxed")
    with fc_col2:
        st.info("Auto-shows all visuals & stats. Methods run only if their packages are available.")

    ts = df_tr.set_index("date")["value"].astype(float)
    freq = freq_map.get(frequency, "M")

    # ---- FORECAST ----
    if st.button("Run Forecast (ALLL-MAXED)", key="run_forecast_allmaxed"):
        st.markdown("### 🔮 Forecast Results")
        fc = None
        idx = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq=freq)

        # Forecast logic
        if method == "Naive seasonality":
            last = ts[-12:] if len(ts) >= 12 else ts
            preds = np.tile(last.values, int(np.ceil(horizon / len(last))))[:horizon]
            fc = pd.Series(preds, index=idx)

        elif method == "SARIMAX" and lazy("statsmodels"):
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
            res = model.fit(disp=False)
            fc = pd.Series(res.get_forecast(steps=horizon).predicted_mean, index=idx)

        elif method == "Prophet" and lazy("prophet") and prophet_mod:
            pdf = ts.reset_index().rename(columns={"date": "ds", "value": "y"})
            m = prophet_mod()
            m.fit(pdf)
            future = m.make_future_dataframe(periods=horizon, freq="M")
            fc = m.predict(future).set_index("ds")["yhat"].tail(horizon)

        elif method == "RandomForest" and lazy("sklearn"):
            from sklearn.ensemble import RandomForestRegressor
            df_feat = pd.DataFrame({"y": ts})
            for l in range(1,13):
                df_feat[f"lag_{l}"] = df_feat["y"].shift(l)
            df_feat = df_feat.dropna()
            X = df_feat.drop(columns=["y"]).values
            y_arr = df_feat["y"].values
            model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y_arr)
            last = df_feat.drop(columns=["y"]).iloc[-1].values
            preds, cur = [], last.copy()
            for _ in range(horizon):
                p = model.predict(cur.reshape(1,-1))[0]
                preds.append(p)
                cur = np.roll(cur, 1)
                cur[0] = p
            fc = pd.Series(preds, index=idx)

        elif method == "XGBoost" and lazy("xgboost"):
            import xgboost as xgb
            df_feat = pd.DataFrame({"y": ts})
            for l in range(1,13):
                df_feat[f"lag_{l}"] = df_feat["y"].shift(l)
            df_feat = df_feat.dropna()
            X = df_feat.drop(columns=["y"])
            y_arr = df_feat["y"]
            dtrain = xgb.DMatrix(X, label=y_arr)
            bst = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=200)
            last = X.iloc[-1].values
            preds, cur = [], last.copy()
            for _ in range(horizon):
                dcur = xgb.DMatrix(cur.reshape(1,-1))
                p = bst.predict(dcur)[0]
                preds.append(p)
                cur = np.roll(cur,1)
                cur[0] = p
            fc = pd.Series(preds, index=idx)

        if fc is not None and not fc.empty:
            combined = pd.concat([ts, fc])
            st.line_chart(combined)
            st.metric("Forecast Mean", round(fc.mean(), 2))
            st.metric("Forecast Std", round(fc.std(), 2))
            st.metric("Forecast Start", str(fc.index[0].date()))
            st.metric("Forecast End", str(fc.index[-1].date()))
            with st.expander("📈 Forecast Data Summary"):
                st.dataframe(fc.describe().to_frame("Forecast Summary"))

    # ---- ANOMALY DETECTION ----
    st.markdown("### ⚠️ Anomaly Detection")
    a_method = st.selectbox(
        "Anomaly method",
        ["Z-score", "IQR", "IsolationForest"],
        key="anom_method_allmaxed"
    )

    if st.button("Run Anomaly Detection (ALLL-MAXED)", key="run_anom_allmaxed"):
        if a_method == "Z-score":
            z = (ts - ts.mean()) / ts.std()
            anoms = z.abs() > 3
        elif a_method == "IQR":
            q1, q3 = ts.quantile(0.25), ts.quantile(0.75)
            iqr = q3 - q1
            anoms = (ts < q1 - 1.5*iqr) | (ts > q3 + 1.5*iqr)
        elif a_method == "IsolationForest" and lazy("sklearn"):
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(random_state=0).fit(ts.values.reshape(-1,1))
            preds = iso.predict(ts.values.reshape(-1,1))
            anoms = preds == -1
        else:
            anoms = pd.Series(False, index=ts.index)

        out = ts[anoms]
        st.metric("Anomalies Detected", out.shape[0])
        st.line_chart(ts)
        if not out.empty:
            st.scatter_chart(out)
            with st.expander("🔍 Anomaly Data"):
                st.dataframe(out)

# ---------------------------------------------------

# # ---------- RAG / LLM + vector DB (FAISS or fallback) -------------------------------------
# if enable_rag:
#     st.subheader("🧠 RAG (Retrieval-Augmented Generation) + Vector Index")

#     docs = []

#     # Collect all docs from current dataframes
#     if not df_cat.empty:
#         docs += (df_cat["label"].astype(str) + " :: " + df_cat["value"].astype(str)).tolist()
#     if not df_mk.empty:
#         docs += (df_mk["label"].astype(str) + " :: " + df_mk["value"].astype(str)).tolist()
#     if not df_tr.empty:
#         docs += [f"{idx.strftime('%Y-%m-%d')} :: {int(v)}" for idx, v in df_tr["value"].items()]

#     if not docs:
#         st.info("ℹ️ No documents available for RAG — please fetch data first.")
#     else:
#         st.success(f"✅ Built in-memory corpus with **{len(docs)}** entries")

#         # ---- Embedding step ----
#         with st.spinner("Generating embeddings..."):
#             try:
#                 # Try real embeddings if available
#                 try:
#                     from sentence_transformers import SentenceTransformer
#                     model = SentenceTransformer("all-MiniLM-L6-v2")
#                     emb = model.encode(docs, show_progress_bar=False, convert_to_numpy=True).astype("float32")
#                     st.caption("🧩 Using SentenceTransformer embeddings (MiniLM).")
#                 except ImportError:
#                     rng = np.random.default_rng(42)
#                     emb = np.stack([rng.normal(size=768) for _ in docs]).astype("float32")
#                     st.caption("⚙️ Using random embeddings (demo mode).")

#                 # ---- Build FAISS or fallback ----
#                 try:
#                     import faiss
#                     faiss.normalize_L2(emb)
#                     index = faiss.IndexFlatIP(emb.shape[1])
#                     index.add(emb)
#                     st.success("📚 FAISS index built successfully.")

#                     def query_rag(query, topk=5):
#                         qv = emb[:1] if not query else model.encode([query]).astype("float32")
#                         faiss.normalize_L2(qv)
#                         D, I = index.search(qv, topk)
#                         return [docs[i] for i in I[0] if i < len(docs)]

#                 except Exception as e:
#                     st.warning(f"⚠️ FAISS not available — using naive cosine fallback. ({e})")
#                     import numpy as np
#                     from numpy.linalg import norm

#                     def query_rag(query, topk=5):
#                         if not query:
#                             return docs[:topk]
#                         qv = emb[0] if emb is not None else np.random.normal(size=768)
#                         sims = np.dot(emb, qv) / (norm(emb, axis=1) * norm(qv) + 1e-9)
#                         topk_idx = np.argsort(sims)[::-1][:topk]
#                         return [docs[i] for i in topk_idx]

#             except Exception as e:
#                 st.error(f"❌ Error building embeddings or index: {e}")
#                 query_rag = lambda q, topk=5: docs[:topk]

#         # ---- RAG Query UI ----
#         st.markdown("### 🔍 Ask a Question")
#         q = st.text_input("Enter your question")
#         if st.button("Run RAG Query") and q:
#             with st.spinner("Retrieving relevant information..."):
#                 hits = query_rag(q)
#                 if llm_key:
#                     st.markdown(f"**LLM Answer (mock)** for: `{q}`")
#                     st.info(f"🤖 Would call LLM API here with top-{len(hits)} docs.")
#                 else:
#                     st.markdown("**Retrieved Documents (Top 5)**")
#                     st.write("\n---\n".join(hits))



# # ---------- NLP tools hooks ----------------------------------------------------
# if enable_nlp:
#     st.subheader('NLP tools (nltk / spacy)')
#     st.info('This will lazy-load NLP packages and run simple tokenization / NER demo')
#     if nltk is None or spacy is None:
#         st.warning('nltk or spacy not installed; install them to run NLP demos')
#     else:
#         st.write('Running simple demo...')
#         # simple tokenization demo
#         text = st.text_area('Text for NLP demo', 'The quick brown fox jumps over the lazy dog. New Delhi saw 10000 registrations in Jan 2024.')
#         import nltk
#         nltk.download('punkt', quiet=True)
#         from nltk.tokenize import word_tokenize
#         toks = word_tokenize(text)
#         st.write('Tokens:', toks)
#         import spacy
#         nlp = spacy.load('en_core_web_sm')
#         doc = nlp(text)
#         st.write('Entities:', [(ent.text, ent.label_) for ent in doc.ents])

# # =====================================================
# # 🧠 NLP ANALYZER — ALL-MAXED ULTIMATE FUSION LAB (SAFE)
# # =====================================================
# enable_nlp = st.checkbox("🗣️ Enable NLP Analyzer (ALL-MAXED ULTIMATE SAFE)", False, key="nlp_toggle")

# if enable_nlp:
#     import pandas as pd
#     import numpy as np
#     import io, base64, re
#     import matplotlib.pyplot as plt
#     import plotly.express as px
#     from collections import Counter

#     st.markdown("## 🧠 NLP Analyzer — ALL-MAXED ULTIMATE (SAFE MODE)")

#     # --- Try importing optional NLP libs safely
#     def safe_import(module_name, install_name=None):
#         try:
#             return __import__(module_name)
#         except ModuleNotFoundError:
#             install_name = install_name or module_name
#             st.warning(f"⚠️ `{module_name}` not found — attempting auto-install...")
#             import subprocess, sys
#             try:
#                 subprocess.run(
#                     [sys.executable, "-m", "pip", "install", install_name, "--quiet"],
#                     check=True
#                 )
#                 return __import__(module_name)
#             except Exception as e:
#                 st.error(f"❌ Failed to import {module_name}: {e}")
#                 return None

#     nltk = safe_import("nltk")
#     spacy = safe_import("spacy")
#     seaborn = safe_import("seaborn")
#     from wordcloud import WordCloud
#     from openpyxl import Workbook
#     from openpyxl.utils.dataframe import dataframe_to_rows

#     if nltk:
#         nltk.download("punkt", quiet=True)
#         nltk.download("averaged_perceptron_tagger", quiet=True)
#         nltk.download("vader_lexicon", quiet=True)
#         from nltk.tokenize import word_tokenize
#         from nltk.sentiment import SentimentIntensityAnalyzer
#         sia = SentimentIntensityAnalyzer()
#     else:
#         word_tokenize = lambda x: x.split()
#         sia = None

#     # --- Input Section
#     model_choice = st.selectbox("Choose NLP Engine", ["spaCy", "NLTK", "None (Fallback)"], index=0)
#     text_input = st.text_area("📝 Enter text for full analysis:",
#                               "Maharashtra saw record EV registrations in 2024.")
#     if not text_input.strip():
#         st.stop()

#     # --- Load spaCy if available
#     if spacy:
#         try:
#             nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             from spacy.cli import download
#             download("en_core_web_sm")
#             nlp = spacy.load("en_core_web_sm")
#     else:
#         nlp = None

#     # --- Core NLP Analysis
#     toks = word_tokenize(text_input) if nltk else text_input.split()
#     pos_tags = nltk.pos_tag(toks) if nltk else []
#     sent_score = sia.polarity_scores(text_input) if sia else {"compound": 0, "pos": 0, "neg": 0, "neu": 1}

#     if nlp:
#         doc = nlp(text_input)
#         ents = [(ent.text, ent.label_) for ent in doc.ents]
#         keywords = [tok.text.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
#     else:
#         ents = []
#         keywords = [t.lower() for t in toks if len(t) > 3]

#     top_kw = Counter(keywords).most_common(15)
#     st.metric("Sentiment (compound)", f"{sent_score['compound']:.3f}")
#     st.write("🔑 Keywords:", top_kw)
#     st.write("🧩 Entities:", ents if ents else "— None —")

#     # --- WordCloud
#     wc = WordCloud(width=600, height=300, background_color="white").generate(" ".join(keywords))
#     fig, ax = plt.subplots()
#     ax.imshow(wc, interpolation="bilinear")
#     ax.axis("off")
#     st.pyplot(fig)

#     # --- Summary DataFrame
#     df_summary = pd.DataFrame([{
#         "Tokens": len(toks),
#         "Sentiment": sent_score["compound"],
#         "Pos": sent_score["pos"],
#         "Neg": sent_score["neg"],
#         "Neu": sent_score["neu"],
#         "TopKeywords": ", ".join([k for k, _ in top_kw])
#     }])
#     st.dataframe(df_summary)

#     # --- Charts
#     if seaborn:
#         import seaborn as sns
#         fig, ax = plt.subplots()
#         sns.heatmap(df_summary[["Sentiment", "Pos", "Neg", "Neu"]].T, cmap="RdYlGn", annot=True, ax=ax)
#         st.pyplot(fig)

#     fig_kw = px.bar(pd.DataFrame(top_kw, columns=["Keyword", "Frequency"]),
#                     x="Keyword", y="Frequency", title="Keyword Frequency")
#     st.plotly_chart(fig_kw, use_container_width=True)

#     # --- Export
#     wb = Workbook()
#     ws = wb.active
#     ws.title = "Summary"
#     for r in dataframe_to_rows(df_summary, index=False, header=True):
#         ws.append(r)
#     buf = io.BytesIO()
#     wb.save(buf)
#     buf.seek(0)

#     st.download_button("📥 Download Excel",
#                        data=buf,
#                        file_name="NLP_AllMaxed_Safe.xlsx",
#                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

#     st.download_button("🧠 Download JSON",
#                        data=df_summary.to_json(orient="records", indent=2),
#                        file_name="nlp_allmaxed_safe.json",
#                        mime="application/json")

#     st.success("✅ NLP Analyzer fully loaded — all maxed (safe mode)!")


# # ---------- Exports & comparisons ------------------------------------------------
# st.subheader('Exports & comparisons')
# if not df_tr.empty:
#     st.download_button('Download trend CSV', df_tr.reset_index().to_csv(index=False), 'trend.csv')
# if not df_cat.empty:
#     st.download_button('Download categories CSV', df_cat.to_csv(index=False), 'categories.csv')
# if not df_mk.empty:
#     st.download_button('Download makers CSV', df_mk.to_csv(index=False), 'makers.csv')



# # ---------- Footer & next steps ------------------------------------------------
# st.markdown('---')
# st.caption('Ultra  V2 — build: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'))


# =====================================================
# 🚀 ALL-MAXED Final Dashboard Footer
# =====================================================
try:
    import time
    from datetime import datetime

    # Track runtime if app_start_time exists
    runtime = ""
    if "app_start_time" in globals():
        try:
            runtime_secs = time.time() - app_start_time
            runtime = f" • Runtime: {runtime_secs:,.1f}s"
        except Exception:
            runtime = ""

    # Count loaded DataFrames
    df_count = sum(1 for v in globals().values() if isinstance(v, pd.DataFrame))
    df_text = f" • DataFrames: {df_count}" if df_count else ""

    # Current UTC timestamp
    footer_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Render ALL-MAXED gradient footer
    st.markdown(f"""
    <div style="
        text-align:center;
        padding:16px;
        border-radius:14px;
        background:linear-gradient(90deg,#051937,#004d7a,#008793,#00bf72,#a8eb12);
        color:white;
        margin-top:18px;
        box-shadow:0 0 12px rgba(0,0,0,0.4);
    ">
        <h3 style="margin:0;">🚗 Parivahan Analytics — ALL-MAXED DASHBOARD</h3>
        <div style="opacity:0.95;font-size:14px;">
            Snapshot: <code>{footer_ts}</code>{runtime}{df_text} • Built with ❤️ & ⚙️ Streamlit • ALLLL MAXED ✅
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("💡 Tip: Use the exported Excel for polished reports and the ZIP snapshot for full archival backups.")

    # Optional celebratory balloons
    try:
        st.balloons()
    except Exception:
        pass

except Exception as e:
    st.warning(f"Footer rendering failed: {e}")
