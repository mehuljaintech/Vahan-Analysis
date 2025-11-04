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

# =====================================================
# 🖥️ Universal Console + Color Setup (ALL-MAXED SAFE)
# =====================================================
import sys, platform
from colorama import Fore, Style, init as colorama_init
from rich.console import Console

try:
    # Initialize colorama only on Windows local terminals
    if platform.system() == "Windows" and sys.stdout.isatty():
        colorama_init(autoreset=True)
    else:
        # Stub for non-Windows or non-interactive envs (e.g. Streamlit Cloud)
        class DummyColor:
            RESET = ""
            def __getattr__(self, name): return ""
        Fore = Style = DummyColor()
    
    # ✅ Rich console (auto-handles non-TTY safely)
    console = Console(force_terminal=False, soft_wrap=True)
    console.log("[bold green]✅ Console initialized successfully[/bold green]")

except Exception as e:
    # Fallback — avoid total crash
    print("⚠️ Console init failed:", e)
    class DummyConsole:
        def log(self, *a, **kw): print(*a)
        def print(self, *a, **kw): print(*a)
    console = DummyConsole()

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
    """Render the ALL-MAXED Category Analytics block inside Streamlit.

    Fully enhanced with advanced UI, legends, charts, KPIs, forecasting, and clustering.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    start_overall = time.time()
    params = params or {}

    st.markdown("## 🚗 **ALL-MAXED — Category Analytics (Multi-Frequency, Multi-Year)**")
    st.caption("⚡ Deep insights powered by synthetic time-series, AI commentary, and full visualization suite.")

    # -------------------------
    # ⚙️ Controls
    # -------------------------
    freq = st.radio("Aggregation Frequency", ["Daily", "Monthly", "Quarterly", "Yearly"], index=1, horizontal=True)
    mode = st.radio("View Mode", ["Separate (Small Multiples)", "Combined (Overlay / Stacked)"], index=1, horizontal=True)
    current_year = datetime.now().year
    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input("From Year", 2010, current_year, current_year - 4)
    with c2:
        end_year = st.number_input("To Year", start_year, current_year, current_year)
    years = list(range(int(start_year), int(end_year) + 1))

    st.divider()
    c3, c4, c5, c6 = st.columns(4)
    show_heatmap = c3.toggle("Heatmap", True)
    show_radar = c4.toggle("Radar", True)
    do_forecast = c5.toggle("Forecast", True)
    do_cluster = c6.toggle("Cluster", False)

    st.info(f"🚀 Starting ALL-MAXED Category Analytics — Years: {years} | Freq: {freq} | Mode: {mode}")

    # -------------------------
    # 📡 Data Fetching
    # -------------------------
    all_year_dfs = []
    with st.spinner("Fetching category data for all selected years..."):
        for y in years:
            try:
                df_y = fetch_year_category(y, params, show_debug=False)
                if df_y is not None and not df_y.empty:
                    all_year_dfs.append(df_y)
            except Exception as e:
                logger.exception(e)
                st.warning(f"⚠️ Using mock data for {y}")
                all_year_dfs.append(to_df(deterministic_mock_categories(y)).assign(year=y))

    if not all_year_dfs:
        st.error("No category data available. Using deterministic mocks.")
        all_year_dfs = [to_df(deterministic_mock_categories(y)).assign(year=y) for y in years]

    df_cat_all = pd.concat(all_year_dfs, ignore_index=True)

    # -------------------------
    # ⏱️ Frequency expansion → synthetic time-series
    # -------------------------
    ts_list = [year_to_timeseries(df_cat_all[df_cat_all["year"] == y], y, freq=freq) for y in years]
    df_ts = pd.concat(ts_list, ignore_index=True)
    df_ts["ds"] = pd.to_datetime(df_ts["ds"])

    # Group by
    resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq={"Daily": "D", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}[freq])])["value"].sum().reset_index()
    resampled["year"] = resampled["ds"].dt.year

    pivot_year = resampled.pivot_table(index="year", columns="label", values="value", aggfunc="sum").fillna(0)

    # -------------------------
    # 💎 KPI Metrics
    # -------------------------
    st.subheader("💎 Key Metrics")
    year_totals = pivot_year.sum(axis=1).rename("Total").to_frame()
    year_totals["YoY_%"] = year_totals["Total"].pct_change() * 100
    if len(year_totals) >= 2:
        first, last = year_totals["Total"].iloc[0], year_totals["Total"].iloc[-1]
        years_count = len(year_totals) - 1
        cagr = ((last / first) ** (1 / years_count) - 1) * 100 if first > 0 else np.nan
    else:
        cagr = np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Years", f"{years[0]} → {years[-1]}", f"{len(years)} yrs")
    c2.metric("CAGR", f"{cagr:.2f}%" if not np.isnan(cagr) else "n/a")
    c3.metric("Latest YoY", f"{year_totals['YoY_%'].iloc[-1]:.2f}%" if not np.isnan(year_totals['YoY_%'].iloc[-1]) else "n/a")

# -------------------------
# ALL-MAXED: Unlimited Charts, Stats & Model Metrics (drop-in)
# -------------------------
import io, math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from statistics import mode as stat_mode
from datetime import datetime
from io import BytesIO

st.markdown("## 🔥 ALL-MAXED EXTENDED VISUALS & METRICS")

# --- defensive references to existing data objects ---
_resampled = globals().get("resampled", pd.DataFrame()).copy()
_pivot = globals().get("pivot", pd.DataFrame()).copy()         # wide timeseries (ds x category)
_pivot_year = globals().get("pivot_year", pd.DataFrame()).copy()  # year x category
_df_cat_all = globals().get("df_cat_all", pd.DataFrame()).copy()  # raw per-year totals

if _resampled.empty and _pivot.empty and _pivot_year.empty and _df_cat_all.empty:
    st.info("No data available for extended visuals. Run fetch/pipeline first.")
else:
    # --- Normalize/derive columns ---
    try:
        if "ds" in _resampled.columns:
            _resampled["ds"] = pd.to_datetime(_resampled["ds"])
            _resampled["year"] = _resampled["ds"].dt.year
            _resampled["month"] = _resampled["ds"].dt.month
            _resampled["month_name"] = _resampled["ds"].dt.strftime("%b")
        elif "year" in _resampled.columns:
            _resampled["year"] = _resampled["year"].astype(int)
            _resampled["ds"] = pd.to_datetime(_resampled["year"].astype(str) + "-01-01")
            _resampled["month"] = 1
            _resampled["month_name"] = "Jan"
    except Exception:
        pass

    # --- Controls: which categories to focus on ---
    all_categories = []
    if not _pivot.empty:
        all_categories = list(_pivot.columns)
    elif not _df_cat_all.empty:
        all_categories = sorted(_df_cat_all["label"].unique().tolist())
    elif not _resampled.empty:
        all_categories = sorted(_resampled["label"].unique().tolist())

    top_n_default = 8 if len(all_categories) >= 8 else max(1, len(all_categories))
    top_n = st.number_input("Max categories to show (top-N by total)", min_value=1, max_value=max(1, len(all_categories)), value=top_n_default, step=1)
    top_labels = []
    if not _resampled.empty:
        top_labels = _resampled.groupby("label")["value"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    elif not _pivot_year.empty:
        top_labels = _pivot_year.sum(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    else:
        top_labels = all_categories[:top_n]

    st.caption(f"Showing top {len(top_labels)} categories: {', '.join(top_labels[:6])}{'...' if len(top_labels)>6 else ''}")

    # -------------------------
    # 1) Time series overview: stacked + per-category small multiples + interactive legend
    # -------------------------
    st.subheader("1 — Time Series Overview")

    # Stacked area for top categories only (handles wide pivot or resampled)
    try:
        if not _pivot.empty:
            df_area = _pivot.reset_index()[["ds"] + top_labels] if "ds" in _pivot.reset_index().columns else _pivot[top_labels].reset_index()
            fig_area = px.area(df_area, x="ds", y=top_labels, title="Stacked area — top categories (interactive legend)", labels={"value":"Registrations","ds":"Date"})
            fig_area.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0))
            st.plotly_chart(fig_area, use_container_width=True)
        elif not _resampled.empty:
            df_area = _resampled[_resampled["label"].isin(top_labels)].pivot_table(index="ds", columns="label", values="value", aggfunc="sum").fillna(0).reset_index()
            fig_area = px.area(df_area, x="ds", y=[c for c in df_area.columns if c!="ds"], title="Stacked area — top categories", labels={"value":"Registrations","ds":"Date"})
            fig_area.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0))
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("Not enough time-series data for stacked area.")
    except Exception as e:
        st.warning(f"Stacked area error: {e}")

    # Overlay lines with smoothing option
    try:
        st.markdown("### Overlay lines (per category)")
        smoothing = st.checkbox("Apply rolling smoothing (window months)", value=False)
        smooth_window = st.slider("Smoothing window (points)", 1, 12, 3) if smoothing else 1
        df_lines = _resampled[_resampled["label"].isin(top_labels)].sort_values("ds").copy()
        if smoothing and smooth_window > 1 and not df_lines.empty:
            df_lines["value_smooth"] = df_lines.groupby("label")["value"].transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
            ycol = "value_smooth"
        else:
            ycol = "value"
        fig_lines = px.line(df_lines, x="ds", y=ycol, color="label", title="Overlay lines — category trends", labels={ycol:"Registrations","ds":"Date"})
        fig_lines.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_lines, use_container_width=True)
    except Exception as e:
        st.warning(f"Overlay lines error: {e}")

    # -------------------------
    # 2) Small multiples (grid) — bar or line per category
    # -------------------------
    st.subheader("2 — Small Multiples")
    try:
        kind = st.selectbox("Small multiples: chart type", ["bar","line","area"], index=1)
        chosen_years = st.multiselect("Which years (small multiples) - leave empty = all", sorted(_resampled["year"].unique().tolist() if "year" in _resampled.columns else []), default=sorted(_resampled["year"].unique().tolist()[-3:]) if "year" in _resampled.columns else [])
        display_labels = top_labels
        small_df = _resampled[_resampled["label"].isin(display_labels)].copy()
        if chosen_years:
            small_df = small_df[small_df["year"].isin(chosen_years)]
        # limit amount
        max_plots = st.number_input("Max small multiples to render", 1, 12, 6)
        for i, lab in enumerate(display_labels[:max_plots]):
            sub = small_df[small_df["label"]==lab]
            if sub.empty: continue
            title = f"{lab} — {sub['year'].min() if 'year' in sub.columns else ''} → {sub['year'].max() if 'year' in sub.columns else ''}"
            if kind=="bar":
                f = px.bar(sub, x="ds", y="value", title=title, text_auto=True)
            elif kind=="area":
                f = px.area(sub, x="ds", y="value", title=title)
            else:
                f = px.line(sub, x="ds", y="value", title=title)
            f.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(f, use_container_width=True)
    except Exception as e:
        st.warning(f"Small multiples error: {e}")

    # -------------------------
    # 3) Donut, Sunburst, Treemap (rich legends & hover)
    # -------------------------
    st.subheader("3 — Donut / Sunburst / Treemap (rich hover)")
    try:
        latest_ds = None
        if not _resampled.empty:
            latest_ds = _resampled["ds"].max()
            period_label = latest_ds.strftime("%Y-%m-%d")
            d_latest = _resampled[_resampled["ds"]==latest_ds].groupby("label")["value"].sum().reset_index()
        else:
            if not _df_cat_all.empty:
                d_latest = _df_cat_all.groupby("label")["value"].sum().reset_index()
                period_label = f"{_df_cat_all['year'].max()} (aggregate)"
            else:
                d_latest = pd.DataFrame(columns=["label","value"])
                period_label = "latest"
        if not d_latest.empty:
            fig_p = px.pie(d_latest.sort_values("value", ascending=False).head(50), names="label", values="value", hole=0.5,
                           title=f"Donut — Category split ({period_label})")
            fig_p.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="<b>%{label}</b><br>%{value:,.0f} registrations<br>%{percent}")
            st.plotly_chart(fig_p, use_container_width=True)

            # sunburst (year->category)
            if not _df_cat_all.empty:
                sb = _df_cat_all.groupby(["year","label"])["value"].sum().reset_index()
                fig_s = px.sunburst(sb, path=["year","label"], values="value", title="Sunburst: Year → Category")
                st.plotly_chart(fig_s, use_container_width=True)
            # treemap
            fig_t = px.treemap(d_latest.sort_values("value", ascending=False).head(200), path=["label"], values="value", title="Treemap — latest category sizes")
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("No data to draw donut/sunburst/treemap.")
    except Exception as e:
        st.warning(f"Donut/Sunburst/Treemap error: {e}")

    # -------------------------
    # 4) Monthly x Category heatmap and sortable top-months
    # -------------------------
    st.subheader("4 — Month × Category Heatmap & Rankings")
    try:
        if "month_name" in _resampled.columns:
            mon = _resampled.groupby(["month_name","label"])["value"].sum().reset_index()
            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            mon["month_name"] = pd.Categorical(mon["month_name"], categories=month_order, ordered=True)
            mon_pivot = mon.pivot_table(index="month_name", columns="label", values="value", aggfunc="sum").fillna(0)
            # restrict columns to top_labels
            mon_small = mon_pivot[top_labels] if len(top_labels)>0 else mon_pivot
            fig_heat = go.Figure(data=go.Heatmap(z=mon_small.values, x=mon_small.columns, y=mon_small.index, colorscale="Cividis"))
            fig_heat.update_layout(title="Month vs Category heatmap", xaxis_title="Category", yaxis_title="Month")
            st.plotly_chart(fig_heat, use_container_width=True)

            # show top months per category
            if st.checkbox("Show top months per selected category"):
                sel_cat = st.selectbox("Select category for monthly ranking", top_labels)
                temp = mon[mon["label"]==sel_cat].sort_values("value", ascending=False).reset_index(drop=True)
                st.dataframe(temp.head(12).style.format({"value":"{:,}"}))
        else:
            st.info("Monthly breakdown not available (no 'ds' dates).")
    except Exception as e:
        st.warning(f"Month×Category heatmap error: {e}")

    # -------------------------
    # 5) Distribution & Descriptive Stats (mean/median/mode/std/CV)
    # -------------------------
    st.subheader("5 — Distribution & Stats")
    try:
        stats = _resampled.groupby("label")["value"].agg(["count","sum","mean","median","std"]).reset_index().rename(columns={"sum":"total"})
        # compute mode robustly (use try/except)
        def safe_mode(s):
            try:
                m = s.mode()
                if not m.empty:
                    return float(m.iloc[0])
                return np.nan
            except Exception:
                return np.nan
        stats["mode"] = _resampled.groupby("label")["value"].apply(safe_mode).values
        stats["cv_pct"] = (stats["std"] / stats["mean"]).replace([np.inf, -np.inf], np.nan) * 100
        st.write("Descriptive summary (top categories):")
        st.dataframe(stats.sort_values("total", ascending=False).head(top_n).style.format({"mean":"{:.1f}", "median":"{:.1f}", "std":"{:.1f}", "total":"{:,}", "mode":"{:.1f}", "cv_pct":"{:.1f}%"}))
        # histogram/kde per chosen category
        cat_for_dist = st.selectbox("Distribution: pick a category", top_labels)
        dist_kind = st.selectbox("Plot distribution as", ["histogram","box","violin","kde"], index=0)
        dcat = _resampled[_resampled["label"]==cat_for_dist]
        if not dcat.empty:
            if dist_kind=="histogram":
                fig = px.histogram(dcat, x="value", nbins=40, title=f"Histogram — {cat_for_dist}")
            elif dist_kind=="box":
                fig = px.box(dcat, y="value", title=f"Boxplot — {cat_for_dist}", points="all")
            elif dist_kind=="violin":
                fig = px.violin(dcat, y="value", box=True, points="all", title=f"Violin — {cat_for_dist}")
            else:  # kde - approximate via histogram + smooth line
                fig = px.histogram(dcat, x="value", nbins=60, histnorm='density', title=f"KDE-like histogram — {cat_for_dist}")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Distribution/stats error: {e}")

    # -------------------------
    # 6) Correlation matrix & scatter-matrix
    # -------------------------
    st.subheader("6 — Correlation & Pairwise")
    try:
        if not _pivot.empty:
            corr = _pivot.corr()
            fig_corr = px.imshow(corr, text_auto=False, aspect="auto", title="Correlation matrix (Pearson) - categories")
            st.plotly_chart(fig_corr, use_container_width=True)
            if st.checkbox("Show scatter matrix (scatter_matrix) for top categories"):
                from plotly.express import scatter_matrix
                sm = _resampled[_resampled["label"].isin(top_labels)].pivot_table(index="ds", columns="label", values="value", aggfunc="sum").reset_index().drop(columns=["ds"], errors="ignore")
                if not sm.empty:
                    fig_sm = scatter_matrix(sm[top_labels[:min(6,len(top_labels))]], dimensions=top_labels[:min(6,len(top_labels))], title="Scatter matrix (top categories)")
                    st.plotly_chart(fig_sm, use_container_width=True)
        else:
            st.info("No wide pivot table for correlation/pairwise plots.")
    except Exception as e:
        st.warning(f"Correlation/pairwise error: {e}")

    # -------------------------
    # 7) Top / bottom lists + downloads (CSV/Excel)
    # -------------------------
    st.subheader("7 — Top/Bottom & Exports")
    try:
        totals = _resampled.groupby("label")["value"].sum().reset_index().sort_values("value", ascending=False)
        topk = totals.head(50)
        botk = totals.tail(50)
        st.markdown("**Top categories (by total)**")
        st.dataframe(topk.style.format({"value":"{:,}"}))
        st.markdown("**Bottom categories (by total)**")
        st.dataframe(botk.style.format({"value":"{:,}"}))

        # CSV downloads
        csv_bytes = topk.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download top categories (CSV)", csv_bytes, file_name="top_categories.csv", mime="text/csv")

        # Excel workbook: sheets = totals, resampled sample, pivot_year
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            totals.to_excel(writer, sheet_name="totals", index=False)
            if not _resampled.empty:
                _resampled.head(1000).to_excel(writer, sheet_name="resampled_sample", index=False)
            if not _pivot_year.empty:
                _pivot_year.to_excel(writer, sheet_name="pivot_year", index=True)
            writer.save()
        excel_buffer.seek(0)
        st.download_button("⬇️ Download summary workbook (XLSX)", excel_buffer, file_name="vahan_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.warning(f"Export error: {e}")

# -------------------------
# MODEL EVALUATION: Confusion / Classification / ROC-AUC (if you provide y_true/y_pred/y_score)
# -------------------------
st.markdown("## 🧪 Model Evaluation (optional)")

# look for candidate DF that may have model columns
candidate = None
for nm in ("df_model", "candidate_df", "df_cat_all", "resampled"):
    df_try = globals().get(nm)
    if isinstance(df_try, pd.DataFrame) and not df_try.empty and set(["y_true","y_pred"]).issubset(df_try.columns):
        candidate = df_try
        break

# fallback: scan any global DF for those columns
if candidate is None:
    for k,v in list(globals().items()):
        if isinstance(v, pd.DataFrame) and not v.empty:
            if set(["y_true","y_pred"]).issubset(v.columns):
                candidate = v
                break

if candidate is None:
    st.info("No DataFrame with columns `y_true` and `y_pred` found. To evaluate a model, provide a DataFrame with columns: y_true, y_pred, optional y_score.")
else:
    try:
        from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
        y_true = candidate["y_true"]
        y_pred = candidate["y_pred"]
        y_score = candidate["y_score"] if "y_score" in candidate.columns else None

        st.write("### Confusion Matrix")
        labels = np.unique(np.concatenate([y_true.unique(), y_pred.unique()]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues", text=cm))
        fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.write("### Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose().fillna(0)
        st.dataframe(df_report.style.format({"precision":"{:.2f}","recall":"{:.2f}","f1-score":"{:.2f}","support":"{:.0f}"}))

        if y_score is not None:
            # binary case
            if len(np.unique(y_true)) == 2:
                auc_val = roc_auc_score(y_true, y_score)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc_val:.3f}"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="random"))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig_roc, use_container_width=True)
                st.success(f"ROC AUC: {auc_val:.4f}")
            else:
                st.info("y_score provided but multiclass ROC/AUC not plotted here.")
    except Exception as e:
        st.warning(f"Model evaluation error: {e}")

    # Final note & summary
    st.markdown("---")
    st.success("✅ ALL-MAXED visuals rendered. Use controls above to show/hide charts and export data.")

# -------------------------
# 🔮 All-Maxed Forecasting & Analytics
# -------------------------
if do_forecast:
    st.header("🔮 All-Maxed Forecasting & Predictive Intelligence")

    categories = pivot_year.columns.tolist() if not pivot_year.empty else df_cat_all["label"].unique().tolist()
    if not categories:
        st.info("No categories available for forecasting.")
    else:
        cat = st.selectbox("📊 Choose category", categories)
        horizon_years = st.slider("Forecast horizon (years)", 1, 5, 2)
        horizon_months = horizon_years * 12

        # Prepare series
        df = pivot_year[[cat]].reset_index().rename(columns={cat: "y", "year": "ds"})
        df["ds"] = pd.to_datetime(df["ds"].astype(str) + "-01-01")

        # --- Basic stats
        st.subheader("📈 Data Profile")
        stats = df["y"].describe()
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{stats['mean']:,.0f}")
        col2.metric("Median", f"{df['y'].median():,.0f}")
        col3.metric("Std Dev", f"{stats['std']:,.0f}")

        # ===============================
        # LINEAR REGRESSION
        # ===============================
        from sklearn.linear_model import LinearRegression
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["y"].values
        lr = LinearRegression().fit(X, y)
        future_X = np.arange(len(df) + horizon_years).reshape(-1, 1)
        lr_pred = lr.predict(future_X)
        lr_dates = pd.date_range(df["ds"].iloc[0], periods=len(future_X), freq="YS")
        df_lr = pd.DataFrame({"ds": lr_dates, "Forecast": lr_pred, "Model": "Linear"})

        # ===============================
        # PROPHET (Yearly)
        # ===============================
        try:
            from prophet import Prophet
            m = Prophet(yearly_seasonality=True)
            m.fit(df)
            fut = m.make_future_dataframe(periods=horizon_years, freq="Y")
            fc = m.predict(fut)
            df_prophet_y = fc[["ds", "yhat"]].rename(columns={"yhat": "Forecast"})
            df_prophet_y["Model"] = "Prophet (Yearly)"
        except Exception:
            df_prophet_y = pd.DataFrame()

        # ===============================
        # PROPHET (Monthly)
        # ===============================
        try:
            df_m = df.copy().resample("M", on="ds").mean(numeric_only=True).dropna().reset_index()
            m2 = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m2.fit(df_m.rename(columns={"ds": "ds", "y": "y"}))
            fut_m = m2.make_future_dataframe(periods=horizon_months, freq="M")
            fc_m = m2.predict(fut_m)
            df_prophet_m = fc_m[["ds", "yhat"]].rename(columns={"yhat": "Forecast"})
            df_prophet_m["Model"] = "Prophet (Monthly)"
        except Exception:
            df_prophet_m = pd.DataFrame()

        # ===============================
        # ARIMA
        # ===============================
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(df["y"], order=(1, 1, 1))
            fit = model.fit()
            fc = fit.forecast(steps=horizon_years)
            df_arima = pd.DataFrame({
                "ds": pd.date_range(df["ds"].iloc[-1], periods=horizon_years + 1, freq="Y")[1:],
                "Forecast": fc,
                "Model": "ARIMA"
            })
        except Exception:
            df_arima = pd.DataFrame()

        # Merge
        forecast_all = pd.concat([df_lr, df_prophet_y, df_prophet_m, df_arima], ignore_index=True)

        # ===============================
        # VISUALS
        # ===============================
        fig = px.line(forecast_all, x="ds", y="Forecast", color="Model",
                      title=f"Forecast Comparison — {cat}",
                      template="plotly_white")
        fig.add_scatter(x=df["ds"], y=df["y"], mode="markers+lines",
                        name="Actual", line=dict(width=3, color="black"))
        fig.update_layout(legend_title="Model", xaxis_title="Date",
                          yaxis_title="Registrations", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # METRICS
        # ===============================
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_pred = lr.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        st.write(f"**Linear Model Metrics:** RMSE {rmse:,.0f}, MAE {mae:,.0f}, R² {r2:.3f}, MAPE {mape:.2f}%")

        # ===============================
        # RESIDUALS
        # ===============================
        resid = y - y_pred
        fig_res = px.scatter(x=df["ds"], y=resid, title="Residuals Over Time")
        fig_res.add_hline(y=0, line_dash="dot")
        st.plotly_chart(fig_res, use_container_width=True)

        # ===============================
        # CONFUSION + ROC (directional)
        # ===============================
        try:
            dirs = np.sign(np.diff(y))
            pred_dirs = np.sign(np.diff(y_pred))
            from sklearn.metrics import confusion_matrix, roc_curve, auc
            cm = confusion_matrix(dirs > 0, pred_dirs > 0)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               title="Directional Accuracy Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
            fpr, tpr, _ = roc_curve(dirs > 0, pred_dirs)
            roc_auc = auc(fpr, tpr)
            fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve — AUC {roc_auc:.3f}")
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception:
            st.info("Directional metrics unavailable.")

        # ===============================
        # GROWTH
        # ===============================
        last, nxt = df["y"].iloc[-1], df_lr["Forecast"].iloc[-1]
        growth = ((nxt - last) / last) * 100
        st.metric("Projected Growth", f"{growth:.2f} %", f"Next {horizon_years} years")

        # ===============================
        # CORRELATION HEATMAP (optional)
        # ===============================
        st.subheader("📊 Correlation Between Categories")
        try:
            corr = pivot_year.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig_corr, use_container_width=True)
        except Exception:
            st.info("Correlation heatmap unavailable.")

    # -------------------------
# ⚠️ ALL-MAXED Anomaly Detection (multi-algo, visual, exportable)
# -------------------------
if do_anomaly:
    st.subheader("⚠️ ALL-MAXED Anomaly Detection — Multi-Algorithm")

    # --- UI controls ---
    algs = {
        "IsolationForest": "isolation",
        "LocalOutlierFactor": "lof",
        "Z-score (pointwise)": "zscore",
        "MAD (robust)": "mad",
        "Seasonal Residual (decompose)": "seasonal"
    }
    chosen = st.multiselect("Choose detectors to run", list(algs.keys()), default=list(algs.keys()))
    contamination = st.slider("Contamination / expected anomaly fraction (for model-based)", 0.001, 0.2, 0.03, step=0.001)
    zscore_thresh = st.slider("Z-score threshold (abs)", 1.5, 5.0, 3.0, step=0.1)
    mad_thresh = st.slider("MAD multiplier (robust)", 2.0, 6.0, 3.5, step=0.1)
    use_seasonal = "Seasonal Residual (decompose)" in chosen
    category_choice = st.multiselect("Select categories to analyze (blank = all)", sorted(resampled["label"].unique().tolist()), default=[])
    run_btn = st.button("🔎 Run Anomaly Detection (ALL-MAXED)")

    # --- helpers ---
    from datetime import datetime as _dt
    import math
    import pandas as _pd
    import numpy as _np

    def _mad_based_outlier(points, thresh=3.5):
        """Return boolean mask of outliers using Median Absolute Deviation"""
        if len(points) == 0:
            return _np.array([], dtype=bool)
        med = _np.median(points)
        diff = _np.abs(points - med)
        mad = _np.median(diff)
        if mad == 0:
            # fallback to std
            return _np.abs(points - med) > (thresh * (_np.std(points) + 1e-9))
        mod_z_score = 0.6745 * diff / mad
        return mod_z_score > thresh

    # Try optional imports; graceful fallback
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        IsolationForest = None
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except Exception:
        LocalOutlierFactor = None
    try:
        from scipy.stats import zscore
    except Exception:
        zscore = None
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except Exception:
        seasonal_decompose = None

    # --- prepare categories list ---
    all_categories = sorted(resampled["label"].unique().tolist())
    if category_choice:
        cats_to_run = category_choice
    else:
        cats_to_run = all_categories

    # Guard
    if run_btn:
        st.info(f"Running {len(chosen)} detectors over {len(cats_to_run)} categories (contamination={contamination})...")
        overall_anomalies = []  # list of dicts
        progress_bar = st.progress(0)
        total = len(cats_to_run)
        idx = 0

        for cat in cats_to_run:
            idx += 1
            progress_bar.progress(idx / total)
            # series for category
            ser_df = resampled[resampled["label"] == cat].sort_values("ds").reset_index(drop=True)
            if ser_df.empty or ser_df["value"].isna().all():
                continue

            ser = ser_df["value"].fillna(0).astype(float).values
            dates = pd.to_datetime(ser_df["ds"]).values

            # results collector per timestamp
            results_mask = _np.zeros(len(ser), dtype=int)  # counts how many detectors flagged
            details = { "category": cat, "n_points": len(ser), "detectors": {} }

            # 1) IsolationForest
            if "IsolationForest" in chosen and IsolationForest is not None and len(ser) >= 6:
                try:
                    iso = IsolationForest(contamination=contamination, random_state=RANDOM_SEED)
                    iso_preds = iso.fit_predict(ser.reshape(-1,1))  # -1 outlier, 1 inlier
                    iso_mask = iso_preds == -1
                    results_mask += iso_mask.astype(int)
                    details["detectors"]["IsolationForest"] = int(iso_mask.sum())
                except Exception as e:
                    details["detectors"]["IsolationForest_error"] = str(e)

            # 2) LocalOutlierFactor (unsupervised, non-probabilistic)
            if "LocalOutlierFactor" in chosen and LocalOutlierFactor is not None and len(ser) >= 6:
                try:
                    lof = LocalOutlierFactor(n_neighbors=min(20, max(2, len(ser)-1)), contamination=contamination)
                    lof_preds = lof.fit_predict(ser.reshape(-1,1))  # -1 outlier
                    lof_mask = lof_preds == -1
                    results_mask += lof_mask.astype(int)
                    details["detectors"]["LocalOutlierFactor"] = int(lof_mask.sum())
                except Exception as e:
                    details["detectors"]["LocalOutlierFactor_error"] = str(e)

            # 3) Z-score
            if "Z-score (pointwise)" in chosen and zscore is not None:
                try:
                    zs = zscore(ser, nan_policy="omit")
                    zs_mask = _np.abs(zs) > zscore_thresh
                    results_mask += zs_mask.astype(int)
                    details["detectors"]["Z-score"] = int(zs_mask.sum())
                except Exception as e:
                    details["detectors"]["Zscore_error"] = str(e)

            # 4) MAD
            if "MAD (robust)" in chosen:
                try:
                    mad_mask = _mad_based_outlier(ser, thresh=mad_thresh)
                    results_mask += mad_mask.astype(int)
                    details["detectors"]["MAD"] = int(mad_mask.sum())
                except Exception as e:
                    details["detectors"]["MAD_error"] = str(e)

            # 5) Seasonal residuals (decompose)
            if "Seasonal Residual (decompose)" in chosen and seasonal_decompose is not None:
                try:
                    # build a freq-aware series if dates are regular; fallback to monthly if possible
                    s_idx = pd.to_datetime(ser_df["ds"])
                    tmp = pd.Series(ser, index=s_idx)
                    # choose period based on median spacing
                    if len(tmp) >= 12:
                        # attempt monthly seasonality if monthly index
                        period = 12 if tmp.index.inferred_freq and ("M" in tmp.index.inferred_freq or "MS" in tmp.index.inferred_freq) else max(2, int(len(tmp)/4))
                    else:
                        period = max(2, int(len(tmp)/2))
                    dec = seasonal_decompose(tmp, period=period, model="additive", extrapolate_trend="freq")
                    resid = dec.resid.fillna(0).values
                    # flag large residuals > 3*std or > 3*mad
                    resid_mask = _np.abs(resid) > (3 * (_np.nanstd(resid) + 1e-9))
                    results_mask += resid_mask.astype(int)
                    details["detectors"]["SeasonalResiduals"] = int(resid_mask.sum())
                except Exception as e:
                    details["detectors"]["Seasonal_error"] = str(e)

            # final selection: any detector flagged
            any_mask = results_mask > 0
            n_anom = int(any_mask.sum())
            details["n_anomalies"] = n_anom
            details["anomaly_pct"] = float(n_anom / max(1, len(ser))) * 100.0

            # collect anomaly rows
            for i, flagged in enumerate(any_mask):
                if flagged:
                    overall_anomalies.append({
                        "category": cat,
                        "ds": str(pd.to_datetime(dates[i])),
                        "value": float(ser[i]),
                        "detector_score": int(results_mask[i]),
                        "detector_details": details["detectors"]
                    })

        progress_bar.progress(1.0)
        st.success(f"Anomaly detection finished. Found {len(overall_anomalies)} anomaly points across {len(cats_to_run)} categories.")

        # --- summary table per category ---
        if overall_anomalies:
            df_anom = pd.DataFrame(overall_anomalies)
            # aggregate summary
            summary = df_anom.groupby("category").agg(
                anomalies_count=("ds", "count"),
                avg_detector_score=("detector_score", "mean")
            ).reset_index().sort_values("anomalies_count", ascending=False)

            st.markdown("### 🧾 Anomaly Summary (per category)")
            st.dataframe(summary)

            # show top categories with example anomalies
            top_examples = df_anom.groupby("category").agg(sample_dates=("ds", lambda x: ", ".join(list(x.astype(str)[:5]))),
                                                          sample_values=("value", lambda x: ", ".join([f"{v:,.0f}" for v in list(x[:5])]))
                                                         ).reset_index().sort_values("category")
            st.markdown("### 🔎 Example anomaly timestamps and values")
            st.dataframe(top_examples)

            # Allow choosing one category to inspect visually
            inspect_cat = st.selectbox("Inspect anomalies for category", top_examples["category"].tolist())
            inspect_rows = df_anom[df_anom["category"] == inspect_cat]
            ser_df = resampled[resampled["label"] == inspect_cat].sort_values("ds").reset_index(drop=True)
            ser_df["ds"] = pd.to_datetime(ser_df["ds"])
            ser_df["is_anom"] = ser_df["ds"].astype(str).isin(inspect_rows["ds"].astype(str).tolist())

            # Plot time series with anomalies highlighted
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ser_df["ds"], y=ser_df["value"], mode="lines+markers", name="Value",
                                     hovertemplate="%{x}<br>%{y:,.0f}"))
            # anomalies
            anom_rows = ser_df[ser_df["is_anom"]]
            if not anom_rows.empty:
                fig.add_trace(go.Scatter(x=anom_rows["ds"], y=anom_rows["value"], mode="markers", name="Anomaly",
                                         marker=dict(color="red", size=10, symbol="x"),
                                         hovertemplate="ANOMALY: %{x}<br>%{y:,.0f}"))
            fig.update_layout(title=f"Time series with anomalies — {inspect_cat}",
                              xaxis_title="Date", yaxis_title="Value", template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Residuals panel if seasonal was run
            if seasonal_decompose is not None and "Seasonal Residual (decompose)" in chosen:
                try:
                    tmp = pd.Series(ser_df["value"].values, index=ser_df["ds"])
                    dec = seasonal_decompose(tmp, period=max(2, min(12, int(len(tmp)/2))), model="additive", extrapolate_trend="freq")
                    resid = dec.resid.fillna(0)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines+markers", name="Residuals"))
                    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception:
                    pass

            # Download buttons
            st.download_button("⬇️ Download anomalies CSV", df_anom.to_csv(index=False).encode("utf-8"), "anomalies_allmaxed.csv", "text/csv")
            st.download_button("⬇️ Download anomalies JSON", df_anom.to_json(orient="records", indent=2).encode("utf-8"), "anomalies_allmaxed.json", "application/json")

        else:
            st.info("No anomalies detected with current settings.")

    # --- End run_btn block ---

# -------------------------
# 🔍 ALL-MAXED Clustering
# -------------------------
if do_clustering:
    st.subheader("🔍 ALL-MAXED Clustering — Multi-Algorithm with Visual Analytics")

    try:
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA

        try:
            import umap
            HAS_UMAP = True
        except Exception:
            HAS_UMAP = False

        # --- UI controls ---
        algo_choice = st.selectbox(
            "Choose clustering algorithm",
            ["KMeans", "MiniBatchKMeans", "Agglomerative", "DBSCAN"]
        )
        n_clusters = st.slider("Number of clusters (KMeans/Agglomerative)", 2, min(10, len(pivot_year)), 3)
        scale_method = st.radio("Scaling method", ["StandardScaler", "MinMaxScaler", "None"], horizontal=True)
        show_3d = st.toggle("Show 3D PCA/UMAP projection", value=False)
        use_umap = HAS_UMAP and st.toggle("Use UMAP for projection (if available)", value=False)
        random_state = 42
        run_cluster = st.button("🚀 Run Clustering (ALL-MAXED)")

        if run_cluster:
            st.info(f"Running {algo_choice} clustering on {pivot_year.shape[0]} samples × {pivot_year.shape[1]} features...")

            X = pivot_year.fillna(0).values
            # --- Scaling ---
            if scale_method == "StandardScaler":
                X_scaled = StandardScaler().fit_transform(X)
            elif scale_method == "MinMaxScaler":
                X_scaled = MinMaxScaler().fit_transform(X)
            else:
                X_scaled = X.copy()

            # --- Fit clustering model ---
            if algo_choice == "KMeans":
                model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
                labels = model.fit_predict(X_scaled)
            elif algo_choice == "MiniBatchKMeans":
                model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
                labels = model.fit_predict(X_scaled)
            elif algo_choice == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X_scaled)
            elif algo_choice == "DBSCAN":
                eps = st.slider("DBSCAN eps (radius)", 0.1, 5.0, 1.0, step=0.1)
                min_samples = st.slider("DBSCAN min_samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)

            # --- Prepare DataFrame ---
            df_cluster = pd.DataFrame({
                "year": pivot_year.index,
                "cluster": labels.astype(int)
            })
            df_cluster["cluster_label"] = df_cluster["cluster"].apply(lambda x: f"Cluster {x}" if x >= 0 else "Noise")
            st.success(f"✅ Clustering done — {df_cluster['cluster'].nunique()} clusters found.")

            # --- Cluster summary ---
            cluster_sizes = df_cluster["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
            st.markdown("### 📊 Cluster Size Summary")
            st.dataframe(cluster_sizes)

            # --- Optional silhouette score ---
            if len(set(labels)) > 1 and algo_choice != "DBSCAN":
                try:
                    sil = silhouette_score(X_scaled, labels)
                    st.metric("Silhouette Score", f"{sil:.3f}")
                except Exception:
                    st.caption("Silhouette score unavailable for current configuration.")

            # --- Cluster means per category ---
            df_means = pivot_year.copy()
            df_means["cluster"] = labels
            cluster_means = df_means.groupby("cluster").mean(numeric_only=True)
            st.markdown("### 🧠 Cluster Means (per category variable)")
            st.dataframe(cluster_means.style.background_gradient(cmap="Blues"))

            # --- 2D / 3D projection ---
            reducer = PCA(n_components=3 if show_3d else 2, random_state=random_state)
            proj = reducer.fit_transform(X_scaled)

            if use_umap and HAS_UMAP:
                reducer = umap.UMAP(random_state=random_state, n_neighbors=10, min_dist=0.3)
                proj = reducer.fit_transform(X_scaled)

            df_proj = pd.DataFrame(proj, columns=["x", "y"] + (["z"] if show_3d else []))
            df_proj["cluster"] = labels
            df_proj["year"] = pivot_year.index

            if show_3d:
                fig = px.scatter_3d(
                    df_proj, x="x", y="y", z="z", color=df_proj["cluster"].astype(str),
                    hover_data=["year"], title="3D Cluster Visualization (PCA/UMAP Projection)"
                )
            else:
                fig = px.scatter(
                    df_proj, x="x", y="y", color=df_proj["cluster"].astype(str),
                    hover_data=["year"], title="2D Cluster Visualization (PCA/UMAP Projection)"
                )
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey")))
            fig.update_layout(template="plotly_white", legend_title="Cluster")
            st.plotly_chart(fig, use_container_width=True)

            # --- Radar plot (mean per cluster per top features) ---
            st.markdown("### 🕸️ Radar Chart (cluster feature profiles)")
            try:
                import plotly.express as px
                n_features = min(6, pivot_year.shape[1])
                top_feats = pivot_year.sum().nlargest(n_features).index.tolist()
                radar_df = cluster_means[top_feats].reset_index()
                radar_melt = radar_df.melt(id_vars="cluster", var_name="feature", value_name="value")
                fig_radar = px.line_polar(
                    radar_melt, r="value", theta="feature", color=radar_melt["cluster"].astype(str),
                    line_close=True, markers=True, title="Cluster Profiles (Top Features)"
                )
                fig_radar.update_traces(fill='toself', opacity=0.6)
                st.plotly_chart(fig_radar, use_container_width=True)
            except Exception as e:
                st.warning(f"Radar chart failed: {e}")

    except Exception as e:
        st.warning(f"ALL-MAXED clustering failed: {e}")

# =====================================================
# 🤖 AI Narrative (ALL-MAXED Analytics + Forecast Insight)
# =====================================================
if enable_ai and do_forecast:
    st.subheader("🤖 AI Narrative — Advanced Multi-Year Insight Engine")

    try:
        import numpy as np, json, math
        import pandas as pd
        from io import StringIO
        import plotly.express as px
        from sklearn.preprocessing import MinMaxScaler

        if pivot_year is None or pivot_year.empty:
            st.warning("⚠️ No valid data available for AI narrative.")
        else:
            # --- Data prep ---
            df_flat = pivot_year.reset_index().melt(id_vars="year", var_name="category", value_name="registrations")
            df_flat["registrations"] = pd.to_numeric(df_flat["registrations"], errors="coerce").fillna(0)
            total_reg = df_flat.groupby("year")["registrations"].sum().reset_index()
            small = df_flat.to_dict(orient="records")
            st.caption(f"📊 {len(small)} records ready for AI analysis.")

            # --- Core stats ---
            years = sorted(pivot_year.index)
            cagr = None
            if len(years) >= 2:
                first, last = years[0], years[-1]
                v0 = total_reg.loc[total_reg["year"] == first, "registrations"].values[0]
                v1 = total_reg.loc[total_reg["year"] == last, "registrations"].values[0]
                if v0 > 0:
                    cagr = ((v1 / v0) ** (1 / (len(years) - 1)) - 1) * 100

            top_overall = (
                df_flat.groupby("category")["registrations"].sum().sort_values(ascending=False).head(5)
            )
            st.metric("📈 CAGR (overall)", f"{cagr:.2f}%" if cagr else "N/A")
            st.metric("🏆 Top Category", top_overall.index[0] if len(top_overall) else "N/A")

            # --- Correlation matrix (numeric only) ---
            corr = pivot_year.corr(numeric_only=True)
            if not corr.empty:
                fig_corr = px.imshow(
                    corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    title="🔗 Correlation between Categories (registrations)"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # --- AI Analysis ---
            system_prompt = (
                "You are a senior transport data scientist. "
                "Analyze multi-year vehicle category trends, year-over-year growth, CAGR, and category correlations. "
                "Identify top-growing and declining segments, anomalies, and forecast implications. "
                "Provide a concise, actionable summary and 5 recommendations for policymakers or planners. "
                "Include quantified % changes, insights on volatility, and relevant sustainability context."
            )

            user_prompt = (
                f"Dataset (partial preview): {json.dumps(small)[:4000]}...\n\n"
                f"Years: {years}\n"
                f"Top overall: {top_overall.to_dict()}\n"
                f"CAGR: {cagr:.2f if cagr else 0}%\n"
                f"Forecast Target: {cat_to_forecast if 'cat_to_forecast' in locals() else 'N/A'}\n"
                f"Horizon: {horizon_years if 'horizon_years' in locals() else 'N/A'} years.\n"
                "Generate an advanced summary and 5 bullet-point recommendations with real insight."
            )

            # --- Try universal_chat or fallback ---
            ai_text = None
            try:
                ai_resp = universal_chat(
                    system_prompt, user_prompt,
                    stream=True, temperature=0.3, max_tokens=800, retries=3
                )
                if isinstance(ai_resp, dict):
                    ai_text = ai_resp.get("text") or ai_resp.get("response") or ai_resp.get("output")
                elif isinstance(ai_resp, str):
                    ai_text = ai_resp
            except Exception:
                ai_text = None

            # --- Fallback (no AI available) ---
            if not ai_text:
                st.markdown("### 🧠 Quick Narrative (Fallback Mode)")
                lines = []
                lines.append(f"Top categories overall: **{', '.join(top_overall.index.tolist())}**.")
                if cagr:
                    direction = "increased" if cagr > 0 else "declined"
                    lines.append(f"From {years[0]} to {years[-1]}, total registrations {direction} by **{abs(cagr):.2f}% CAGR**.")
                top_growth = (
                    df_flat.groupby("category")["registrations"].agg(["first","last"])
                    .assign(growth=lambda d: ((d["last"]/d["first"])-1)*100)
                    .sort_values("growth", ascending=False)
                )
                g_up = top_growth.head(3)
                g_down = top_growth.tail(3)
                lines.append(f"Top growth: {', '.join([f'{i} (+{g_up.loc[i,'growth']:.1f}%)' for i in g_up.index])}.")
                lines.append(f"Declining: {', '.join([f'{i} ({g_down.loc[i,'growth']:.1f}%)' for i in g_down.index])}.")
                lines.append("**Recommendations:**")
                lines.append("1️⃣ Prioritize infrastructure for fast-growing EV and commercial segments.")
                lines.append("2️⃣ Introduce scrappage incentives for older declining categories.")
                lines.append("3️⃣ Enhance data cadence to monthly for granular policy feedback.")
                lines.append("4️⃣ Monitor high-volatility categories for demand shocks.")
                lines.append("5️⃣ Integrate forecasting outputs into policy dashboards.")
                st.markdown("\n\n".join(lines))
            else:
                st.markdown("### 🧠 AI Summary & Recommendations")
                st.markdown(ai_text)

            # --- Optional: AI-generated chart summary ---
            st.markdown("### 📊 AI-Assisted Insight Chart (category growth)")
            df_growth = (
                df_flat.groupby("category")["registrations"].agg(["first","last"])
                .assign(GrowthPct=lambda d: ((d["last"]/d["first"])-1)*100)
                .reset_index().sort_values("GrowthPct", ascending=False)
            )
            fig_growth = px.bar(df_growth, x="category", y="GrowthPct",
                                color="GrowthPct", text_auto=".1f",
                                title="Growth % by Category (First vs Last Year)")
            st.plotly_chart(fig_growth, use_container_width=True)

            # --- Radar of volatility ---
            st.markdown("### 🌪️ Volatility Radar (standard deviation per category)")
            vol = df_flat.groupby("category")["registrations"].std().reset_index(name="std_dev")
            fig_vol = px.line_polar(vol, r="std_dev", theta="category",
                                    line_close=True, title="Volatility across Categories")
            fig_vol.update_traces(fill='toself', opacity=0.6)
            st.plotly_chart(fig_vol, use_container_width=True)

    except Exception as e:
        st.error(f"💥 AI Narrative (ALL-MAXED) failed: {e}")

    # =====================================================
    # 🧩 ALL-MAXED FINAL SUMMARY — SEPARATE PER-CATEGORY (NO GLOBAL SUM)
    # =====================================================
    st.markdown("## 🧠 Final Summary & Debug Insights — ALL-MAXED (Per-category, no global sum)")
    
    try:
        summary_start = time.time()
    
        # Basic session info
        n_cats = int(df_cat_all['label'].nunique())
        n_rows = int(len(df_cat_all))
        year_list = sorted(df_cat_all['year'].unique())
    
        st.info(f"🔎 Categories: {n_cats} | Rows: {n_rows:,} | Years: {year_list[0]} → {year_list[-1]}")
    
        # -------------------------
        # Per-category aggregated metrics (no single global total)
        # -------------------------
        grp = df_cat_all.groupby("label")["value"].agg(
            total="sum",
            mean="mean",
            median="median",
            std="std",
            observations="count"
        ).reset_index()
    
        # mode (robust)
        def safe_mode(s):
            m = s.mode()
            return float(m.iloc[0]) if not m.empty else np.nan
    
        modes = df_cat_all.groupby("label")["value"].agg(mode=safe_mode).reset_index()
        grp = grp.merge(modes, on="label", how="left")
    
        # Per-category CAGR (first vs last year) — use pivot_year if present, else compute from df_cat_all
        cat_cagr = []
        years_sorted = sorted(df_cat_all["year"].unique())
        n_periods = max(1, len(years_sorted) - 1)
        pivot_year_local = df_cat_all.groupby(["year", "label"])["value"].sum().unstack(fill_value=0)
    
        for cat in grp["label"].tolist():
            try:
                # use pivot_year_local to find first & last
                first = float(pivot_year_local[cat].iloc[0]) if cat in pivot_year_local.columns and len(pivot_year_local[cat])>0 else 0.0
                last = float(pivot_year_local[cat].iloc[-1]) if cat in pivot_year_local.columns and len(pivot_year_local[cat])>0 else 0.0
                if first > 0:
                    cagr_cat = ((last / first) ** (1 / max(1, n_periods)) - 1) * 100
                else:
                    cagr_cat = np.nan
            except Exception:
                cagr_cat = np.nan
            cat_cagr.append(cagr_cat)
    
        grp["CAGR_%"] = cat_cagr
    
        # Format for display
        disp = grp.copy()
        disp["total"] = disp["total"].map(lambda x: int(x))
        disp["mean"] = disp["mean"].map(lambda x: float(x))
        disp["median"] = disp["median"].map(lambda x: float(x))
        disp["std"] = disp["std"].map(lambda x: float(x) if not pd.isna(x) else x)
        disp["mode"] = disp["mode"].map(lambda x: float(x) if not pd.isna(x) else x)
        disp["CAGR_%"] = disp["CAGR_%"].map(lambda x: f"{x:.2f}%" if not pd.isna(x) else "n/a")
    
        st.write("### 🧾 Per-Category Summary (total, mean, median, std, mode, CAGR, observations)")
        st.dataframe(disp.sort_values("total", ascending=False).reset_index(drop=True).style.format({
            "total":"{:,}",
            "mean":"{:,.1f}",
            "median":"{:,.1f}",
            "std":"{:,.1f}",
            "mode":"{:.0f}",
        }))
    
        # -------------------------
        # Per-year × category table (no grand sum)
        # -------------------------
        st.write("### 📋 Year × Category (wide table, no full-sum row)")
        year_cat = df_cat_all.groupby(["year","label"])["value"].sum().unstack(fill_value=0)
        st.dataframe(year_cat.style.format("{:,.0f}").reset_index().rename_axis(None, axis=1))
    
        # -------------------------
        # Top N categories (by total) — bar & table
        # -------------------------
        top_n = 10
        top_df = grp.sort_values("total", ascending=False).head(top_n)
        st.markdown(f"### 🏆 Top {top_n} Categories (by total) — separate totals")
        fig_top = px.bar(top_df, x="label", y="total", text="total", title=f"Top {top_n} Categories (Separate totals)", labels={"label":"Category","total":"Registrations"})
        fig_top.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_top.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', template="plotly_white")
        st.plotly_chart(fig_top, use_container_width=True)
    
        # -------------------------
        # Distribution visuals (boxplots / violin) using expanded timeseries if available
        # -------------------------
        st.markdown("### 📈 Distributions — per-category (boxplot / violin)")
    
        if "df_ts" in globals() and not df_ts.empty:
            # use df_ts which has ds,label,value,year
            ds_vis = df_ts.copy()
            # if too many categories, limit to top 12 for visibility
            cat_order = top_df["label"].tolist()
            ds_vis = ds_vis[ds_vis["label"].isin(cat_order)]
            fig_box = px.box(ds_vis, x="label", y="value", points="outliers", title="Boxplot: value distribution per category (sampled)", labels={"value":"Registrations","label":"Category"})
            st.plotly_chart(fig_box, use_container_width=True)
    
            fig_violin = px.violin(ds_vis, x="label", y="value", box=True, points="outliers", title="Violin: value distribution per category", labels={"value":"Registrations","label":"Category"})
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("No expanded timeseries (df_ts) available — box/violin plots skipped.")
    
        # -------------------------
        # Monthly heatmap (if we can compute months)
        # -------------------------
        st.markdown("### 🔥 Month × Category Heatmap (if monthly data exists)")
        try:
            if "df_ts" in globals() and not df_ts.empty:
                df_ts_local = df_ts.copy()
                df_ts_local["month"] = df_ts_local["ds"].dt.strftime("%b")
                # pivot month x category
                heat = df_ts_local.groupby(["month","label"])["value"].sum().unstack(fill_value=0)
                # reorder months to calendar if possible
                month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                heat = heat.reindex(index=month_order).fillna(0)
                # short-circuit if empty
                if not heat.empty:
                    fig_hm = px.imshow(heat.fillna(0), labels=dict(x="Category", y="Month", color="Registrations"), x=heat.columns, y=heat.index, title="Month × Category Heatmap")
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.info("Month × Category heatmap not available (no monthly-like data).")
            else:
                st.info("Monthly heatmap skipped — df_ts missing or empty.")
        except Exception as e:
            st.warning(f"Monthly heatmap generation failed: {e}")
    
        # -------------------------
        # Basic descriptive stats (mean, median, mode) across categories
        # -------------------------
        st.markdown("### ⚙️ Descriptive Stats (quick): mean / median / mode across categories")
        desc = grp[["label","total","mean","median","std","mode","observations"]].sort_values("total", ascending=False).reset_index(drop=True)
        st.dataframe(desc.head(50).style.format({"total":"{:,}","mean":"{:,.1f}","median":"{:,.1f}","std":"{:,.1f}","mode":"{:.0f}"}))
    
        # -------------------------
        # Debug: trend for each of top 5 categories separately (no global sum)
        # -------------------------
        st.write("### 🔧 Per-Category Trend (Top 5)")
        top5 = top_df["label"].tolist()[:5]
        for cat in top5:
            dcat = df_cat_all[df_cat_all["label"]==cat].groupby("year")["value"].sum().reset_index()
            if dcat.empty:
                continue
            fig_tr = px.line(dcat, x="year", y="value", markers=True, title=f"Trend — {cat}")
            fig_tr.update_layout(template="plotly_white", yaxis_title="Registrations", xaxis_title="Year")
            st.plotly_chart(fig_tr, use_container_width=True)
    
        # -------------------------
        # Performance / debug info
        # -------------------------
        summary_time = time.time() - summary_start
        st.markdown("### 🧪 Debug Performance & Notes")
        st.code(
            f"""
    Categories: {n_cats}
    Rows processed: {n_rows:,}
    Years: {year_list[0]} → {year_list[-1]}
    Pivot years shape: {pivot_year.shape if 'pivot_year' in locals() else 'n/a'}
    Computation time: {summary_time:.2f} sec
            """,
            language="yaml",
        )

    except Exception as e:
        logger.exception(f"Summary block (separate) failed: {e}")
        st.error(f"⛔ Summary generation failed: {e}")


# If invoked directly, render the block
if __name__ == "__main__":
    all_maxed_category_block()


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
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random
import logging

st.markdown("## 🏭 ALL-MAXED — Makers Analytics (multi-frequency, multi-year)")

# =====================================================
# CONTROLS — ALL ON MAIN PAGE (no sidebar)
# =====================================================
section_id = rto_opt.lower() if "rto_opt" in locals() else "main"

# Frequency & Mode
freq = st.radio(
    "Aggregation Frequency",
    ["Daily", "Monthly", "Quarterly", "Yearly"],
    index=3,
    horizontal=True,
    key=f"freq_{section_id}"
)

mode = st.radio(
    "View Mode",
    ["Separate (Small Multiples)", "Combined (Overlay / Stacked)"],
    index=1,
    horizontal=True,
    key=f"mode_{section_id}"
)

# Year range
today = datetime.now()
current_year = today.year
default_from_year = current_year - 1


from_year = st.sidebar.number_input(
    "From Year",
    min_value=2012,
    max_value=today.year,
    value=default_from_year,
    key=f"from_year_{section_id}"
)

to_year = st.sidebar.number_input(
    "To Year",
    min_value=from_year,
    max_value=today.year,
    value=today.year,
    key=f"to_year_{section_id}"
)

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
    "Vehicle Classes (e.g., 2W,3W,4W if accepted)",
    value="",
    key=f"classes_{section_id}"
)

vehicle_makers = st.sidebar.text_input(
    "Vehicle Makers (comma-separated or IDs)",
    value="",
    key=f"makers_{section_id}"
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

vehicle_type = st.sidebar.text_input(
    "Vehicle Type (optional)",
    value="",
    key=f"type_{section_id}"
)

# Extra feature toggles
st.divider()
col3, col4, col5 = st.columns(3)
with col3:
    show_heatmap = st.checkbox("Show Heatmap (year × maker)", True, key=f"heatmap_{section_id}")
    show_radar = st.checkbox("Show Radar (per year)", True, key=f"radar_{section_id}")
with col4:
    do_forecast = st.checkbox("Enable Forecasting", True, key=f"forecast_{section_id}")
    do_anomaly = st.checkbox("Enable Anomaly Detection", False, key=f"anomaly_{section_id}")
with col5:
    do_clustering = st.checkbox("Enable Clustering (KMeans)", False, key=f"cluster_{section_id}")

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

# =====================================================
# 🚗 VAHAN MAKER ANALYTICS — ALL-MAXED SECTION
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
from colorama import Fore

# -----------------------------------------------------
# ⚡ MAXED CHART HELPERS (UNIQUE KEYS + BETTER UX)
# -----------------------------------------------------
import uuid
import plotly.express as px
import streamlit as st


def _bar_from_df(df: pd.DataFrame, title: str, combined: bool = False, section_id: str = ""):
    """Render a robust bar chart with unique Streamlit key, error safety, and style."""
    try:
        # 🔑 Ensure unique key for Streamlit
        unique_key = f"bar_{section_id}_{uuid.uuid4().hex[:6]}"

        # 🧱 Create Plotly figure
        if combined and "year" in df.columns:
            fig = px.bar(
                df,
                x="label",
                y="value",
                color="year",
                barmode="group",
                text_auto=True,
                title=title,
            )
        else:
            fig = px.bar(
                df,
                x="label",
                y="value",
                color="label",
                text_auto=True,
                title=title,
            )

        # 🧩 Layout polish
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Maker",
            yaxis_title="Registrations",
            showlegend=True,
            title_font=dict(size=18, color="#222", family="Segoe UI"),
            margin=dict(t=50, b=40, l=40, r=20),
            bargap=0.2,
            height=450,
        )

        # 📊 Display
        st.plotly_chart(fig, use_container_width=True, key=unique_key)

    except Exception as e:
        st.warning(f"⚠️ Bar chart render failed: {e}")


def _pie_from_df(df: pd.DataFrame, title: str, section_id: str = ""):
    """Render a pie chart safely with unique Streamlit key, hover effects, and styling."""
    try:
        # 🔑 Unique key
        unique_key = f"pie_{section_id}_{uuid.uuid4().hex[:6]}"

        # 🧱 Create Plotly figure
        fig = px.pie(
            df,
            names="label",
            values="value",
            hole=0.45,
            title=title,
        )

        # 🧩 Layout polish
        fig.update_traces(
            textinfo="percent+label",
            pull=[0.05] * len(df),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} registrations<br>%{percent}",
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(t=60, b=40, l=40, r=40),
            title_font=dict(size=18, color="#222", family="Segoe UI"),
            showlegend=False,
            height=400,
        )

        # 📊 Display
        st.plotly_chart(fig, use_container_width=True, key=unique_key)

    except Exception as e:
        st.warning(f"⚠️ Pie chart render failed: {e}")


# -----------------------------------------------------
# FETCH FUNCTION (ROBUST + MOCK FALLBACK)
# -----------------------------------------------------
# =====================================================
# 🚀 ALL-MAXED MAKER FETCH & VISUAL MODULE
# =====================================================

# -----------------------------------------------------
# 🔧 FETCH FUNCTION — SAFE + SMART + MOCK-RESILIENT
# -----------------------------------------------------
def fetch_maker_year(year: int, params_common: dict):
    """Fetch top vehicle makers for a given year — fully maxed with safe params + mock fallback."""
    logger.info(Fore.CYAN + f"🚀 Fetching top makers for {year}...")

    # --- Safe param cleanup ---
    safe_params = params_common.copy()
    safe_params["fromYear"] = year
    safe_params["toYear"] = year

    for k in ["fitnessCheck", "stateCode", "rtoCode", "vehicleType"]:
        if k in safe_params and (
            safe_params[k] in ["ALL", "0", "", None, False]
        ):
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

    with st.expander(f"🧩 JSON Debug — {year}", expanded=False):
        st.json(mk_json)

    # --- Validation: check for expected fields ---
    is_valid = False
    df = pd.DataFrame()

    if isinstance(mk_json, dict):
        # ✅ Case 1: Chart.js-style JSON
        if "datasets" in mk_json and "labels" in mk_json:
            data_values = mk_json["datasets"][0].get("data", [])
            labels = mk_json.get("labels", [])
            if data_values and labels:
                df = pd.DataFrame({"label": labels, "value": data_values})
                is_valid = True

        # ✅ Case 2: API returned dict with "data" or "result"
        elif "data" in mk_json:
            df = pd.DataFrame(mk_json["data"])
            is_valid = not df.empty
        elif "result" in mk_json:
            df = pd.DataFrame(mk_json["result"])
            is_valid = not df.empty

    elif isinstance(mk_json, list) and mk_json:
        # ✅ Case 3: Direct list of records
        df = pd.DataFrame(mk_json)
        is_valid = not df.empty

    # --- Handle missing or invalid data ---
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
        _bar_from_df(df, f"Top Makers ({year})", combined=False)
        _pie_from_df(df, f"Maker Share ({year})")

    return df
# -----------------------------------------------------
# 🔁 MAIN LOOP — MULTI-YEAR FETCH
# -----------------------------------------------------
all_years = []
with st.spinner("⏳ Fetching maker data for all selected years..."):
    for y in years:
        try:
            dfy = fetch_maker_year(y, params_common)   # ✅ FIXED: pass params_common
            all_years.append(dfy)
        except Exception as e:
            st.error(f"❌ {y} fetch error: {e}")
            logger.error(Fore.RED + f"Fetch error {y}: {e}")

# ===============================================================
# 🚘 MAKER ANALYTICS — FULLY MAXED + SAFE + DEBUG READY
# ===============================================================
import pandas as pd, numpy as np, math, time, random
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta

# ----------------------------------
# 🧩 Helpers
# ----------------------------------
def normalize_freq_rule(freq):
    return {"Daily": "D", "Monthly": "M", "Quarterly": "Q"}.get(freq, "Y")

def year_to_timeseries_maker(df_year, year, freq):
    """Convert maker-year totals into evenly distributed synthetic timeseries."""
    rule = normalize_freq_rule(freq)
    idx = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        freq=("D" if freq == "Daily" else "M"),
    )
    rows = []
    for _, r in df_year.iterrows():
        maker = r.get("label", f"Maker_{_}")
        total = float(r.get("value", 0))
        per = total / max(1, len(idx))
        for ts in idx:
            rows.append({"ds": ts, "label": maker, "value": per, "year": year})
    return pd.DataFrame(rows)

# ===============================================================
# 🧭 FETCH MAKER DATA (PER YEAR)
# ===============================================================
def fetch_maker_year(year: int, params_common: dict):
    """Fetch top vehicle makers for a given year — fully maxed with safe params + mock fallback."""
    logger.info(Fore.CYAN + f"🚀 Fetching top makers for {year}...")

    safe_params = params_common.copy()
    safe_params["fromYear"] = year
    safe_params["toYear"] = year

    mk_json, mk_url = None, None
    try:
        mk_json, mk_url = get_json("vahandashboard/top5Makerchart", safe_params)
    except Exception as e:
        logger.error(Fore.RED + f"❌ API fetch failed for {year}: {e}")
        mk_json, mk_url = None, "MOCK://top5Makerchart"

    color = "orange" if mk_url and "MOCK" in mk_url else "green"
    st.markdown(f"🔗 **API ({year}):** <span style='color:{color}'>{mk_url or 'N/A'}</span>", unsafe_allow_html=True)

    with st.expander(f"🧩 JSON Debug — {year}", expanded=False):
        st.json(mk_json)

    if not mk_json or (isinstance(mk_json, dict) and not mk_json.get("datasets")):
        st.warning(f"⚠️ No valid API data for {year}, generating mock values.")
        random.seed(year)
        makers = [
            "Maruti Suzuki", "Tata Motors", "Hyundai", "Mahindra", "Hero MotoCorp",
            "Bajaj Auto", "TVS Motor", "Honda", "Kia", "Toyota", "Renault",
            "Ashok Leyland", "MG Motor", "Eicher", "Piaggio", "BYD", "Olectra", "Force Motors"
        ]
        random.shuffle(makers)
        mk_json = {
            "datasets": [{"data": [random.randint(200_000, 1_200_000) for _ in range(5)],
                          "label": "Vehicle Registered"}],
            "labels": makers[:5]
        }

    # --- Normalize API data
    if isinstance(mk_json, dict) and "datasets" in mk_json:
        data = mk_json["datasets"][0]["data"]
        labels = mk_json["labels"]
        mk_json = [{"label": l, "value": v} for l, v in zip(labels, data)]

    df = pd.DataFrame(mk_json)
    df.columns = [c.lower() for c in df.columns]
    df["year"] = year
    df = df.sort_values("value", ascending=False)

    if not df.empty:
        st.info(f"🏆 **{year}** → **{df.iloc[0]['label']}** — {df.iloc[0]['value']:,} registrations")
        _bar_from_df(df, f"Top Makers ({year})", combined=False)
        _pie_from_df(df, f"Maker Share ({year})")
    return df

# ===============================================================
# ⏳ MULTI-YEAR DATA COLLECTION
# ===============================================================
with st.spinner("Fetching maker data for selected years..."):
    all_year_dfs = []
    for y in years:
        try:
            df_y = fetch_maker_year(y, params_common)
            if df_y is not None and not df_y.empty:
                all_year_dfs.append(df_y)
            else:
                st.warning(f"No data for {y}")
        except Exception as e:
            st.error(f"Error fetching {y}: {e}")

if not all_year_dfs:
    st.error("🚫 No maker data loaded for selected range.")
    st.stop()

df_maker_all = pd.concat(all_year_dfs, ignore_index=True)

# ===============================================================
# 📈 TIME SERIES & METRICS
# ===============================================================
rule = normalize_freq_rule(freq)
ts_frames = [year_to_timeseries_maker(df_maker_all[df_maker_all["year"] == y], y, freq)
             for y in sorted(df_maker_all["year"].unique())]
df_ts = pd.concat(ts_frames, ignore_index=True)
df_ts["ds"] = pd.to_datetime(df_ts["ds"])

resampled = df_ts.groupby(["label", pd.Grouper(key="ds", freq=rule)])["value"].sum().reset_index()
resampled["year"] = resampled["ds"].dt.year
pivot = resampled.pivot_table(index="ds", columns="label", values="value", aggfunc="sum").fillna(0)
pivot_year = resampled.pivot_table(index="year", columns="label", values="value", aggfunc="sum").fillna(0)

# ===============================================================
# 💎 METRICS
# ===============================================================
st.subheader("💎 Key Metrics & Growth — Makers")
year_totals = pivot_year.sum(axis=1).rename("TotalRegistrations").to_frame()
year_totals["YoY_%"] = year_totals["TotalRegistrations"].pct_change() * 100

if len(year_totals) >= 2:
    first, last = year_totals["TotalRegistrations"].iloc[0], year_totals["TotalRegistrations"].iloc[-1]
    years_count = max(1, len(year_totals) - 1)
    cagr = ((last / first) ** (1 / years_count) - 1) * 100 if first > 0 else np.nan
else:
    cagr = np.nan

c1, c2, c3 = st.columns(3)
c1.metric("📅 Years Loaded", f"{years[0]} → {years[-1]}", f"{len(years)} years")
c2.metric("📈 CAGR (Total)", f"{cagr:.2f}%" if not math.isnan(cagr) else "n/a")
c3.metric("📊 Latest YoY", f"{year_totals['YoY_%'].iloc[-1]:.2f}%" if not np.isnan(year_totals['YoY_%'].iloc[-1]) else "n/a")

# ===============================================================
# 📊 VISUALS (ALL MODES)
# ===============================================================
st.subheader("📊 Visualizations — Makers Multi-year & Multi-frequency")

if mode.startswith("Combined"):
    st.markdown("### Stacked Area — Combined")
    st.plotly_chart(px.area(pivot.reset_index(), x="ds", y=pivot.columns.tolist(),
                            title="Stacked registrations by maker over time",
                            template="plotly_white"), use_container_width=True)
else:
    years_sorted = sorted(resampled["year"].unique())
    sel_small = st.multiselect("Select specific years", years_sorted, default=years_sorted[-3:])
    for y in sel_small:
        d = resampled[resampled["year"] == y]
        fig = px.bar(d, x="label", y="value", color="label", title=f"Maker distribution — {y}", text_auto=True)
        fig.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ===============================================================
# 🧠 DEBUG & SUMMARY
# ===============================================================
st.markdown("## 🧠 Debug + Insight Summary (Makers)")
agg = df_maker_all.groupby("label")["value"].sum().reset_index().sort_values("value", ascending=False)
topmaker = agg.iloc[0]["label"]; topval = agg.iloc[0]["value"]
total = agg["value"].sum(); share = (topval / total) * 100 if total else 0

st.metric("🏆 Top Maker (All Years)", f"{topmaker}", f"{share:.2f}% share")
st.dataframe(agg.style.format({"value": "{:,}"}))
fig_top10 = px.bar(agg.head(10), x="label", y="value", text_auto=".2s",
                   title="Top 10 Makers (All Years Combined)")
st.plotly_chart(fig_top10, use_container_width=True)

# ===============================================================
# ⚙️ DEBUG METRICS
# ===============================================================
st.markdown("### ⚙️ Diagnostics")
st.code(f"""
Years loaded: {years}
Rows in df_maker_all: {len(df_maker_all):,}
Unique makers: {df_maker_all['label'].nunique()}
Min registrations: {df_maker_all['value'].min():,.0f}
Max registrations: {df_maker_all['value'].max():,.0f}
""", language="yaml")


# # ---------- Trend series + resampling & multi-year comparisons ------------------
# with st.spinner('Fetching trend series...'):
#     tr_json, tr_url = get_json('vahandashboard/vahanyearwiseregistrationtrend', params)
#     df_tr = to_df(tr_json)

# if not df_tr.empty:
#     def parse_label(l):
#         for fmt in ('%Y-%m-%d','%Y-%m','%b %Y','%Y'):
#             try: 
#                 return pd.to_datetime(l, format=fmt)
#             except: 
#                 pass
#         try:
#             return pd.to_datetime(l)
#         except:
#             return pd.NaT

#     df_tr['date'] = df_tr['label'].apply(parse_label)
#     df_tr = df_tr.dropna(subset=['date']).sort_values('date')
#     df_tr['value'] = pd.to_numeric(df_tr['value'], errors='coerce')
#     df_tr = df_tr.set_index('date')

#     freq_map = {'Daily': 'D', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
#     df_tr = df_tr.resample(freq_map.get(frequency, 'M')).sum()

# # ---------------- MULTI-YEAR COMPARISONS ----------------
# st.subheader('📈 Multi-year Comparisons')

# if df_tr.empty:
#     st.warning('No trend data for chosen filters.')
# else:
#     df_tr['year'] = df_tr.index.year
#     years = sorted(df_tr['year'].unique())

#     selected_years = st.sidebar.multiselect(
#         'Select years to compare',
#         years,
#         default=years if len(years) <= 2 else years[-2:]
#     )

#     # Show separate year charts
#     st.markdown('### 🔹 Separate Year Charts')
#     cols = st.columns(len(selected_years) if selected_years else 1)
#     for y, c in zip(selected_years, cols):
#         with c:
#             s = df_tr[df_tr['year'] == y]['value']
#             st.markdown(f"**{y}**")
#             st.line_chart(s)

#     # Combined comparison chart
#     st.markdown('### 🔸 Combined Comparison (Each Year as a Separate Line)')

#     df_tr_reset = df_tr.reset_index()

#     # Choose x-axis format
#     if frequency == 'Yearly':
#         df_tr_reset['period_label'] = df_tr_reset['date'].dt.strftime('%Y')
#     elif frequency == 'Quarterly':
#         df_tr_reset['period_label'] = 'Q' + df_tr_reset['date'].dt.quarter.astype(str)
#     elif frequency == 'Monthly':
#         df_tr_reset['period_label'] = df_tr_reset['date'].dt.strftime('%b')
#     else:  # Daily or others
#         df_tr_reset['period_label'] = df_tr_reset['date'].dt.strftime('%d-%b')

#     pivot = (
#         df_tr_reset.pivot_table(
#             index='period_label',
#             columns='year',
#             values='value',
#             aggfunc='sum'
#         )
#         .fillna(0)
#     )

#     # Plot combined line chart with Plotly
#     fig = px.line(
#         pivot,
#         x=pivot.index,
#         y=pivot.columns,
#         markers=True,
#         title="Multi-Year Comparison of Registrations",
#         labels={"x": "Period", "value": "Registrations"},
#     )
#     fig.update_layout(template="plotly_white", legend_title_text="Year")
#     st.plotly_chart(fig, use_container_width=True)

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
@st.cache_data(show_spinner=False)
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

c1, c2, c3 = st.columns(3)
c1.metric("📅 Duration", f"{year_sum.index.min()} → {year_sum.index.max()}", f"{len(year_sum)} years")
c2.metric("📈 CAGR", f"{total_cagr:.2f}%" if not math.isnan(total_cagr) else "n/a")
c3.metric("📊 Latest YoY", f"{latest_yoy:.2f}%" if not math.isnan(latest_yoy) else "n/a")

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
horizon = st.slider("Forecast horizon (years)", 1, 10, 3)

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
    st.write(f"- **CAGR:** {total_cagr:.2f}%")
    st.write(f"- **Latest YoY:** {latest_yoy:.2f}%")
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

# # ---------- RTO/State detailed breakdown ---------------------------------------
# st.subheader('RTO / State breakdown')
# # User can choose to fetch state/rto endpoints if available
# rto_opt = st.selectbox('Show breakdown by', ['State','RTO','None'])
# if rto_opt != 'None':
#     # For demo, attempt to call same categories endpoint with state param
#     target = 'vahandashboard/statewise' if rto_opt=='State' else 'vahandashboard/rtowise'
#     try:
#         br_json, _ = get_json(target, params)
#         df_br = to_df(br_json)
#         st.dataframe(df_br.head(200))
#     except Exception as e:
#         st.warning(f'Breakdown endpoint not available: {e}')

# ============================================================
# 🌍 ALL-MAXED RTO / STATE BREAKDOWN
# Includes top-N, YoY growth, interactive charts & comparisons
# ============================================================

# =========================================================
# 🌐 ALL-MAXED — State / RTO Analytics (multi-year, multi-frequency)
# =========================================================

# =========================================================
# 🚦 ALL-MAXED — RTO / State Analytics (multi-year, multi-frequency)
# Drop-in Streamlit module. Fully instrumented, mock-safe, debug-friendly.
# =========================================================
import time, math, json, random, logging
from typing import Optional, Dict, Any
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Logging setup
# -------------------------
logger = logging.getLogger("all_maxed_rto_state")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.DEBUG)


# -------------------------
# Mock generator
# -------------------------
def deterministic_mock_rto_state(year: int, seed_base="rto_state") -> Dict[str, Any]:
    """Generate deterministic mock for RTO/State analytics."""
    rnd = random.Random(hash((year, seed_base)) & 0xFFFFFFFF)
    states = [
        "Maharashtra","Uttar Pradesh","Tamil Nadu","Gujarat","Karnataka",
        "Rajasthan","Bihar","Haryana","Madhya Pradesh","Telangana",
        "West Bengal","Delhi","Punjab","Kerala","Odisha"
    ]
    data = [{"label": s, "value": rnd.randint(50000, 1200000)} for s in states]
    return {"data": data, "generatedAt": datetime.utcnow().isoformat()}


# -------------------------
# Visualization helpers
# -------------------------
def bar_chart(df, title):
    try:
        fig = px.bar(df, x="label", y="value", text_auto=True, title=title)
        fig.update_layout(template="plotly_white", xaxis_title="State / RTO", yaxis_title="Revenue / Fees")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"bar_chart failed: {e}")
        st.write(df)

def pie_chart(df, title):
    try:
        fig = px.pie(df, names="label", values="value", hole=0.55, title=title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"pie_chart failed: {e}")
        st.write(df)


# -------------------------
# Data fetch with fallback + live debug
# -------------------------
def fetch_rto_state_year(year: int, params: dict, show_debug=True) -> pd.DataFrame:
    """Fetch RTO/state revenue/fee breakdown for a specific year."""
    try:
        j, url = get_json("vahandashboard/top5chartRevenueFee", {**params, "year": year})
    except Exception as e:
        logger.warning(f"Fetch exception {year}: {e}")
        j, url = deterministic_mock_rto_state(year), f"mock://rto_state/{year}"

    if show_debug:
        with st.expander(f"🧩 Debug JSON — RTO/State {year}"):
            st.write("**URL:**", url)
            st.json(j)

    # Normalize
    if j and "data" in j:
        df = pd.DataFrame(j["data"])
    else:
        df = pd.DataFrame(deterministic_mock_rto_state(year)["data"])
    df["year"] = int(year)

    # Render preview charts
    st.markdown(f"### 🗺️ RTO / State — {year}")
    bar_chart(df, f"RTO / State Revenue — {year}")
    pie_chart(df, f"Revenue Distribution — {year}")

    return df


# -------------------------
# Expand yearly totals to timeseries (synthetic)
# -------------------------
def expand_to_timeseries(df_year, year, freq="Monthly"):
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    idx = pd.date_range(start=start, end=end, freq="M" if freq=="Monthly" else "Y")
    rows = []
    for _, r in df_year.iterrows():
        per = r["value"] / len(idx)
        for ts in idx:
            rows.append({"ds": ts, "label": r["label"], "value": per, "year": year})
    return pd.DataFrame(rows)


# =========================================================
# MAIN STREAMLIT UI BLOCK
# =========================================================
# =========================================================
# MAIN STREAMLIT UI BLOCK (Fixed Unique Keys)
# =========================================================
def all_maxed_rto_state_block(params: Optional[dict] = None, section_id: str = "rto_state"):
    params = params or {}
    st.markdown("## 🏛️ ALL-MAXED — RTO / State Revenue & Fee Analytics")

    freq = st.radio(
        "Aggregation Frequency",
        ["Monthly", "Yearly"],
        index=0,
        horizontal=True,
        key=f"freq_{section_id}"
    )

    current_year = datetime.now().year

    start_year = st.number_input(
        "From Year",
        2010,
        current_year,
        current_year - 1,
        key=f"start_year_{section_id}"
    )
    end_year = st.number_input(
        "To Year",
        start_year,
        current_year,
        current_year,
        key=f"end_year_{section_id}"
    )

    years = list(range(int(start_year), int(end_year) + 1))
    st.info(f"Debug ON — years: {years}, freq: {freq}")

    # (rest of your block stays the same)

    # -------------------------
    # Fetch Data
    # -------------------------
    all_years = []
    with st.spinner("Fetching RTO/State data..."):
        for y in years:
            df = fetch_rto_state_year(y, params, show_debug=False)
            all_years.append(df)
    df_all = pd.concat(all_years, ignore_index=True)

    # -------------------------
    # Time-series expansion
    # -------------------------
    ts = pd.concat([expand_to_timeseries(df_all[df_all["year"]==y], y, freq) for y in years])
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts["year"] = ts["ds"].dt.year

    pivot_year = ts.pivot_table(index="year", columns="label", values="value", aggfunc="sum").fillna(0)
    pivot = ts.pivot_table(index="ds", columns="label", values="value", aggfunc="sum").fillna(0)

    # -------------------------
    # KPI Metrics
    # -------------------------
    st.subheader("💎 Key Metrics")
    total = pivot_year.sum(axis=1)
    yoy = total.pct_change()*100
    cagr = ((total.iloc[-1]/total.iloc[0])**(1/(len(total)-1))-1)*100 if len(total)>1 else np.nan
    c1,c2,c3 = st.columns(3)
    c1.metric("Years", f"{years[0]} → {years[-1]}")
    c2.metric("CAGR", f"{cagr:.2f}%" if not math.isnan(cagr) else "n/a")
    c3.metric("YoY Latest", f"{yoy.iloc[-1]:.2f}%" if not yoy.isna().iloc[-1] else "n/a")

    # -------------------------
    # Visualizations
    # -------------------------
    st.subheader("📊 Visualizations")
    fig_area = px.area(pivot.reset_index(), x="ds", y=pivot.columns, title="Combined — RTO/State Revenue Over Time")
    st.plotly_chart(fig_area, use_container_width=True)

    # Heatmap
    st.markdown("### 🔥 Heatmap — Year × State")
    fig_h = go.Figure(data=go.Heatmap(
        z=pivot_year.values, x=pivot_year.columns.astype(str), y=pivot_year.index.astype(str),
        colorscale="Viridis"))
    fig_h.update_layout(title="Revenue heatmap (year × state)")
    st.plotly_chart(fig_h, use_container_width=True)

    # Radar
    st.markdown("### 🌈 Radar — Snapshot (last 3 years)")
    try:
        fig_r = go.Figure()
        for y in pivot_year.index[-3:]:
            fig_r.add_trace(go.Scatterpolar(
                r=pivot_year.loc[y].values, theta=pivot_year.columns, fill="toself", name=str(y)
            ))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_r, use_container_width=True)
    except Exception as e:
        st.warning(f"Radar failed: {e}")

    # -------------------------
    # Forecast (Linear)
    # -------------------------
    st.subheader("🔮 Forecast (Linear)")
    state_sel = st.selectbox("Select state to forecast", pivot_year.columns)
    X = np.arange(len(pivot_year)).reshape(-1,1)
    y = pivot_year[state_sel].values
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X,y)
    fut = np.arange(len(pivot_year)+3).reshape(-1,1)
    preds = lr.predict(fut)
    fut_years = list(range(pivot_year.index[0], pivot_year.index[0]+len(fut)))
    figf = px.line(x=fut_years, y=preds, title=f"Forecast for {state_sel}")
    figf.add_scatter(x=pivot_year.index, y=y, mode="markers+lines", name="Historical")
    st.plotly_chart(figf, use_container_width=True)

    # -------------------------
    # Final summary
    # -------------------------
    st.subheader("🧠 Summary")
    top_state = pivot_year.sum(axis=0).idxmax()
    top_val = pivot_year.sum(axis=0).max()
    st.success(f"Top State Overall: **{top_state}** with total ₹{top_val:,.0f}")
    st.dataframe(pivot_year.style.format("{:,.0f}"))

    logger.info("ALL-MAXED RTO/STATE block complete ✅")


# -------------------------
# Standalone run
# -------------------------
if __name__ == "__main__":
    all_maxed_rto_state_block()


# # ---------- Forecasting & Anomalies -------------------------------------------
# if enable_ml and not df_tr.empty:
#     st.subheader('Forecasting & Anomaly Detection')
#     fc_col1, fc_col2 = st.columns([2,3])
#     with fc_col1:
#         method = st.selectbox('Forecast method', ['Naive seasonality','SARIMAX','Prophet','RandomForest','XGBoost'])
#         horizon = st.number_input('Horizon (periods)', 1, 60, 12)
#     with fc_col2:
#         st.info('Some methods require optional packages (statsmodels, prophet, sklearn, xgboost).')

#     ts = df_tr['value'].astype(float)
#     # naive
#     if st.button('Run forecast'):
#         try:
#             if method=='Naive seasonality':
#                 last = ts[-12:] if len(ts)>=12 else ts
#                 preds = np.tile(last.values, int(np.ceil(horizon/len(last))))[:horizon]
#                 idx = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq=freq_map.get(frequency,'M'))
#                 fc = pd.Series(preds, index=idx)
#                 st.line_chart(pd.concat([ts, fc]))
#             elif method=='SARIMAX':
#                 sm = lazy('statsmodels')
#                 if sm is None:
#                     st.error('statsmodels not installed')
#                 else:
#                     from statsmodels.tsa.statespace.sarimax import SARIMAX
#                     mod = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
#                     res = mod.fit(disp=False)
#                     pred = res.get_forecast(steps=horizon).predicted_mean
#                     idx = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='M')
#                     fc = pd.Series(pred.values, index=idx)
#                     st.line_chart(pd.concat([ts, fc]))
#             elif method=='Prophet':
#                 if prophet_mod is None:
#                     st.error('prophet not installed')
#                 else:
#                     from prophet import Prophet
#                     pdf = ts.reset_index().rename(columns={'date':'ds','value':'y'})
#                     m = Prophet()
#                     m.fit(pdf)
#                     future = m.make_future_dataframe(periods=horizon, freq='M')
#                     fc = m.predict(future).set_index('ds')['yhat'].tail(horizon)
#                     st.line_chart(pd.concat([ts, fc]))
#             elif method=='RandomForest':
#                 skl = lazy('sklearn')
#                 if skl is None:
#                     st.error('scikit-learn not installed')
#                 else:
#                     from sklearn.ensemble import RandomForestRegressor
#                     df_feat = pd.DataFrame({'y':ts})
#                     for l in range(1,13): df_feat[f'lag_{l}'] = df_feat['y'].shift(l)
#                     df_feat = df_feat.dropna()
#                     X = df_feat.drop(columns=['y']).values; y = df_feat['y'].values
#                     model = RandomForestRegressor(n_estimators=100).fit(X,y)
#                     last = df_feat.drop(columns=['y']).iloc[-1].values
#                     preds=[]; cur = last.copy()
#                     for i in range(horizon):
#                         p = model.predict(cur.reshape(1,-1))[0]
#                         preds.append(p); cur = np.roll(cur,1); cur[0]=p
#                     idx = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='M')
#                     fc = pd.Series(preds, index=idx)
#                     st.line_chart(pd.concat([ts, fc]))
#             elif method=='XGBoost':
#                 if xgb is None:
#                     st.error('xgboost not installed')
#                 else:
#                     import xgboost as xgb
#                     df_feat = pd.DataFrame({'y':ts})
#                     for l in range(1,13): df_feat[f'lag_{l}'] = df_feat['y'].shift(l)
#                     df_feat = df_feat.dropna()
#                     X = df_feat.drop(columns=['y']); y = df_feat['y']
#                     dtrain = xgb.DMatrix(X, label=y)
#                     bst = xgb.train({'objective':'reg:squarederror'}, dtrain, num_boost_round=100)
#                     last = X.iloc[-1].values; preds=[]; cur = last.copy()
#                     for i in range(horizon):
#                         dcur = xgb.DMatrix(cur.reshape(1,-1)); p = bst.predict(dcur)[0]; preds.append(p); cur = np.roll(cur,1); cur[0]=p
#                     idx = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='M')
#                     fc = pd.Series(preds, index=idx)
#                     st.line_chart(pd.concat([ts, fc]))
#         except Exception as e:
#             st.error(f'Forecast failed: {e}')

#     # anomaly detection
#     st.markdown('**Anomaly detection**')
#     a_method = st.selectbox('Anomaly method', ['Z-score','IQR','IsolationForest'])
#     if st.button('Run anomaly detection'):
#         try:
#             if a_method=='Z-score':
#                 z = (ts - ts.mean())/ts.std()
#                 anoms = z.abs() > 3
#             elif a_method=='IQR':
#                 q1 = ts.quantile(0.25); q3 = ts.quantile(0.75); iqr=q3-q1
#                 anoms = (ts < q1 - 1.5*iqr) | (ts > q3 + 1.5*iqr)
#             else:
#                 skl = lazy('sklearn')
#                 if skl is None:
#                     st.error('scikit-learn not installed')
#                     anoms = pd.Series(False, index=ts.index)
#                 else:
#                     from sklearn.ensemble import IsolationForest
#                     iso = IsolationForest(random_state=0).fit(ts.values.reshape(-1,1)); preds = iso.predict(ts.values.reshape(-1,1)); anoms = preds==-1
#             out = ts[anoms]
#             st.write('Anomalies found:', out.shape[0])
#             if not out.empty: st.write(out)
#         except Exception as e:
#             st.error(f'Anomaly detection failed: {e}')

#---------------------------------------------------

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


# ================================
# 🧠 RAG / LLM + Multi-Source Vector Intelligence (ALL-MAXED) — FIXED
# ================================

enable_rag = st.checkbox("🧠 Enable RAG + LLM Intelligence (ALL-MAXED)", value=True, key="enable_rag_allmax")

if enable_rag:
    st.markdown(
        """
        <div style="padding:14px 22px;border-left:6px solid #8A2BE2;
                    background:linear-gradient(90deg,#f8f6ff,#ffffff 100%);
                    border-radius:16px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h3 style="margin:0;font-weight:700;color:#3a3a3a;">🧠 RAG + LLM Intelligence (ALL-MAXED)</h3>
            <p style="margin:4px 0 0;color:#555;font-size:14.5px;">
                Unified retrieval + LLM reasoning across all VAHAN datasets — categories, makers, trends, state breakdowns, forecasts & anomalies.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    docs = []

    def add_docs(df, prefix):
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            for _, r in df.iterrows():
                # safe string building
                parts = []
                for k in r.index:
                    v = r[k]
                    if pd.isna(v):
                        continue
                    parts.append(f"{k}: {v}")
                if parts:
                    docs.append(f"{prefix} :: " + " | ".join(parts))

    # try to collect documents from known globals (safe)
    add_docs(globals().get("df_cat"), "Category")
    add_docs(globals().get("df_cat_all"), "CategoryAll")
    add_docs(globals().get("df_mk"), "Maker")
    add_docs(globals().get("df_maker_all"), "MakerAll")
    if "df_tr" in globals() and isinstance(globals().get("df_tr"), (pd.DataFrame, pd.Series)):
        tr_df = globals().get("df_tr")
        if isinstance(tr_df, pd.Series):
            tr_df = tr_df.reset_index().rename(columns={0: "value"})
        add_docs(tr_df, "Trend")
    add_docs(globals().get("df_br"), "State/RTO")

    # optional additional lists
    if "yoy_change" in globals() and isinstance(globals().get("yoy_change"), pd.DataFrame):
        for idx, row in globals().get("yoy_change").iterrows():
            docs.append(f"YoY :: {idx} => {row.to_dict()}")

    if not docs:
        st.info("ℹ️ No documents available for RAG — please fetch or generate data first.")
    else:
        st.success(f"✅ Built knowledge corpus with **{len(docs):,} entries**")

    # ---- Embeddings generation (best-effort; fallback to random)
    emb = None
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")  # smaller & common fallback
        emb = model.encode(docs, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        st.caption("✅ Embeddings generated (SentenceTransformer)")
    except Exception as e:
        st.warning(f"⚠️ Embedding model not available, using random demo embeddings: {e}")
        rng = np.random.default_rng(42)
        emb = rng.normal(size=(len(docs), 384)).astype("float32")
        model = None

    # ---- Build index (FAISS preferred, else Annoy, else fallback)
    index = None
    index_type = "None"
    if emb is not None:
        try:
            import faiss
            faiss.normalize_L2(emb)
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            index_type = "FAISS (Inner Product)"
        except Exception:
            try:
                from annoy import AnnoyIndex
                index = AnnoyIndex(emb.shape[1], 'angular')
                for i, v in enumerate(emb):
                    index.add_item(i, v.tolist())
                index.build(10)
                index_type = "Annoy (Angular)"
            except Exception:
                index = None
                index_type = "Flat (numpy fallback)"
    st.caption(f"📚 Vector index built using **{index_type}**")

    # ---- Query helper
    def query_rag(query, topk=10):
        if not query:
            return []
        if model is None:
            # random fallback selection
            idxs = np.random.choice(len(docs), min(topk, len(docs)), replace=False)
            return [docs[i] for i in idxs.tolist()]

        # produce query embedding and search depending on index
        qv = model.encode([query], convert_to_numpy=True).astype("float32")
        try:
            import faiss
            faiss.normalize_L2(qv)
            D, I = index.search(qv, topk)
            unique_idxs = []
            for i in I[0]:
                if i >= 0 and i < len(docs) and i not in unique_idxs:
                    unique_idxs.append(int(i))
            return [docs[i] for i in unique_idxs]
        except Exception:
            try:
                # Annoy path
                qv0 = qv[0].tolist()
                idxs = index.get_nns_by_vector(qv0, topk)
                return [docs[i] for i in idxs if i < len(docs)]
            except Exception:
                # numpy dot fallback
                from numpy.linalg import norm
                qv0 = qv[0]
                sims = np.dot(emb, qv0) / (norm(emb, axis=1) * norm(qv0) + 1e-9)
                topk_idx = np.argsort(sims)[::-1][:topk]
                return [docs[int(i)] for i in topk_idx]

    # ---- UI for RAG queries
    st.markdown("### 🔍 Ask a Question or Insight Query")
    q = st.text_input("🗨️ What do you want to know about the VAHAN data?", key="rag_query_input")
    topk = st.slider("Top-K results", 3, 20, 8, key="rag_topk")

    # make sure hits always exists
    hits = []

    if st.button("🚀 Run RAG + LLM Query", key="rag_run_btn"):
        if not docs:
            st.warning("No docs to search. Fetch data first.")
        elif not q:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Retrieving and synthesizing insights..."):
                try:
                    hits = query_rag(q, topk=topk)
                    if not hits:
                        st.info("No matches found for the query (or using demo embeddings).")
                    else:
                        st.markdown(f"**Retrieved {len(hits)} relevant documents:**")
                        st.write("\n\n---\n\n".join(hits[:topk]))
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")
                    hits = []

            # Try summarizer if hits present
            if hits:
                try:
                    from transformers import pipeline
                    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                    joined = " ".join(hits[:min(len(hits), topk)])
                    summary = summarizer(joined, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
                    st.markdown("### 🧭 LLM Summary Insight")
                    st.success(summary)
                except Exception as e:
                    st.info(f"🤖 Summarizer unavailable or failed ({e}). You can plug your LLM here.")

    # ========== Debug & intelligence summary (always safe) ==========
    st.markdown("---")
    st.subheader("🧠 RAG Debug + Intelligence Summary")

    try:
        debug_block = {"query": q, "topk": int(topk), "retrieved_count": len(hits)}
        st.json({"retrieval": debug_block})

        # Top entity summary attempt (safe)
        st.markdown("### 🏆 Top Entity Summary Across Known Data")
        top_summary = {}
        candidates = {
            "Category": globals().get("df_cat"),
            "Maker": globals().get("df_mk") or globals().get("df_maker_all") or globals().get("df_makers"),
            "State": globals().get("df_br"),
            "Trend": globals().get("df_tr") if "df_tr" in globals() else None,
        }
        for name, df in candidates.items():
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                # best-effort value column discovery
                val_cols = [c for c in df.columns if any(k in c.lower() for k in ["value", "count", "total", "registered", "registration"])]
                if not val_cols:
                    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
                    val_cols = numcols
                if val_cols:
                    col = val_cols[0]
                    try:
                        idxmax = df[col].idxmax()
                        top_row = df.loc[idxmax]
                        top_summary[name] = {
                            "Top": str(top_row.iloc[0]) if len(top_row) else str(top_row.to_dict()),
                            "Value": float(top_row[col]),
                            "Mean": float(df[col].mean()),
                            "Total": float(df[col].sum()),
                        }
                        st.markdown(f"**{name}** → {top_summary[name]['Top']} (Top Value {top_summary[name]['Value']:,.0f}, Mean {top_summary[name]['Mean']:,.0f})")
                    except Exception:
                        continue
        if not top_summary:
            st.info("No structured numeric columns found to compute top-entity summary.")
        # store summary in session
        st.session_state["rag_debug_summary"] = {"retrieval": debug_block, "top_summary": top_summary, "hits_preview": hits[:topk]}

        # AI post-summary toggle (use checkbox)
        use_ai_summary = st.checkbox("🤖 Generate AI narrative using configured LLM (DeepInfra / OpenAI)", value=False, key="rag_use_ai")
        if use_ai_summary:
            # prefer available secrets; fail gracefully otherwise
            st.info("Attempting to call configured LLM (if API key present)...")
            try:
                # assemble simple payload
                payload_text = json.dumps(st.session_state.get("rag_debug_summary", {}), default=str)[:8000]
                # You can insert your API-call here. For safety we only show info.
                st.info("LLM call placeholder — implement your API call here with the payload.")
            except Exception as e:
                st.warning(f"AI post-summary failed: {e}")

        # allow debug JSON download
        st.download_button(
            "⬇️ Download Debug JSON",
            data=json.dumps(st.session_state.get("rag_debug_summary", {"retrieval": debug_block}), indent=2),
            file_name="rag_debug_summary.json",
            mime="application/json",
            key="rag_debug_download"
        )

    except Exception as e:
        st.error(f"⚠️ Debug Summary failed: {e}")


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

# =====================================================
# 🧠 NLP ANALYZER — ALL-MAXED ULTIMATE FUSION LAB 🔥
# =====================================================
enable_nlp = st.checkbox("🗣️ Enable NLP Analyzer (ALL-MAXED ULTIMATE)", False, key="nlp_toggle")

if enable_nlp:
    import pandas as pd
    import numpy as np
    import nltk, spacy, re, io, base64, seaborn as sns
    from nltk.tokenize import word_tokenize
    from collections import Counter
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import plotly.express as px
    from nltk.sentiment import SentimentIntensityAnalyzer
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.styles import Alignment, Font

    st.markdown("## 🧠 NLP Analyzer — ALL-MAXED ULTIMATE FUSION LAB")

    # --- Lazy Downloads
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()

    # --- Text Input Section
    model_choice = st.selectbox("Choose NLP Engine", ["spaCy (default)", "NLTK", "Transformers (HuggingFace)"], index=0)

    text_mode = st.radio("Input Mode", ["Single Text", "Multiple Texts (compare mode)"], horizontal=True)

    if text_mode == "Single Text":
        texts = [st.text_area("📝 Paste text to analyze:", "Maharashtra saw record EV registrations in 2024.")]
    else:
        raw = st.text_area("Enter multiple texts separated by '---'",
                           "2023: Maharashtra had 8 lakh registrations. --- 2024: Maharashtra had 10 lakh registrations.")
        texts = [x.strip() for x in raw.split('---') if x.strip()]

    # --- Load spaCy model
    def load_spacy():
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    nlp = load_spacy()

    results = []
    entity_map = []

    for idx, t in enumerate(texts):
        if not t.strip():
            continue

        st.markdown(f"### 📄 Text {idx+1}")
        toks = word_tokenize(t)
        pos_tags = nltk.pos_tag(toks)
        sent_score = sia.polarity_scores(t)
        doc = nlp(t)

        ents = [(ent.text, ent.label_) for ent in doc.ents]
        keywords = [tok.text.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
        top_kw = Counter(keywords).most_common(12)

        st.metric("Sentiment (compound)", f"{sent_score['compound']:.3f}")
        st.write("🧩 Entities:", ents if ents else "— None —")
        st.write("🔑 Keywords:", top_kw)
        st.write("🧱 POS (sample):", pos_tags[:10])

        # WordCloud
        wc = WordCloud(width=600, height=300, background_color="white").generate(" ".join(keywords))
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Append data
        results.append({
            "TextID": idx + 1,
            "Tokens": len(toks),
            "Sentiment": sent_score["compound"],
            "Pos": sent_score["pos"],
            "Neg": sent_score["neg"],
            "Neu": sent_score["neu"],
            "TopKeywords": ", ".join([k for k, _ in top_kw])
        })

        for ent_text, ent_type in ents:
            entity_map.append({"TextID": idx + 1, "Entity": ent_text, "Type": ent_type})

    if not results:
        st.warning("No valid text found.")
        st.stop()

    # ===========================
    # 🧾 SUMMARY SECTION
    # ===========================
    df_results = pd.DataFrame(results)
    df_entities = pd.DataFrame(entity_map) if entity_map else pd.DataFrame(columns=["TextID", "Entity", "Type"])

    avg_sent = df_results["Sentiment"].mean()
    st.subheader("🏁 NLP Summary Dashboard")
    st.metric("📊 Average Sentiment", f"{avg_sent:.3f}")
    st.dataframe(df_results, use_container_width=True)

    # --- Sentiment Chart
    fig_sent = px.bar(df_results, x="TextID", y="Sentiment", color="Sentiment",
                      text_auto=True, title="Sentiment per Text", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig_sent, use_container_width=True)

    # --- Keyword frequency
    all_kw = []
    for r in results:
        all_kw.extend(r["TopKeywords"].split(", "))
    kw_freq = Counter(all_kw).most_common(20)
    df_kw = pd.DataFrame(kw_freq, columns=["Keyword", "Frequency"])
    fig_kw = px.bar(df_kw, x="Keyword", y="Frequency", text_auto=True, title="Keyword Frequency")
    st.plotly_chart(fig_kw, use_container_width=True)

    # --- Entity Analysis
    if not df_entities.empty:
        st.subheader("🏷️ Named Entities Overview")
        ent_counts = df_entities["Type"].value_counts().head(10)
        st.bar_chart(ent_counts)
        st.dataframe(df_entities.head(30), use_container_width=True)

    # --- Heatmap for sentiment distribution
    fig, ax = plt.subplots()
    sns.heatmap(df_results[["Sentiment", "Pos", "Neg", "Neu"]].T, cmap="RdYlGn", annot=True, ax=ax)
    st.pyplot(fig)

    # ==========================================================
    # 💾 EXPORT SECTION — Excel (auto width) + JSON
    # ==========================================================
    st.subheader("📦 Export NLP Results")

    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = "Summary"

    for r in dataframe_to_rows(df_results, index=False, header=True):
        ws_sum.append(r)

    for col in ws_sum.columns:
        max_len = max(len(str(c.value)) if c.value else 0 for c in col)
        ws_sum.column_dimensions[col[0].column_letter].width = max_len + 2

    ws_ent = wb.create_sheet("Entities")
    for r in dataframe_to_rows(df_entities, index=False, header=True):
        ws_ent.append(r)
    for col in ws_ent.columns:
        max_len = max(len(str(c.value)) if c.value else 0 for c in col)
        ws_ent.column_dimensions[col[0].column_letter].width = max_len + 2

    ws_kw = wb.create_sheet("Keywords")
    for r in dataframe_to_rows(df_kw, index=False, header=True):
        ws_kw.append(r)
    for col in ws_kw.columns:
        max_len = max(len(str(c.value)) if c.value else 0 for c in col)
        ws_kw.column_dimensions[col[0].column_letter].width = max_len + 2

    # Save Excel in-memory
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode()
    st.download_button(
        label="📥 Download All NLP Results (Excel)",
        data=base64.b64decode(b64),
        file_name="NLP_AllMaxed_Fusion.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # JSON Export
    st.download_button(
        label="🧩 Download Results (JSON)",
        data=df_results.to_json(orient="records", indent=2),
        file_name="nlp_allmaxed_results.json",
        mime="application/json"
    )

    st.success("✅ All NLP results and charts generated — fully ALL-MAXED and export-ready!")


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
# 🚀 Final Dashboard Footer (ALL-MAXED Polished Edition)
# =====================================================
try:
    import time
    from datetime import datetime

    # compute runtime duration if start_ts exists
    runtime = ""
    if "app_start_time" in globals():
        try:
            runtime_secs = time.time() - app_start_time
            runtime = f" • Runtime: {runtime_secs:,.1f}s"
        except Exception:
            runtime = ""
    
    # count how many dataframes we loaded
    df_count = sum(1 for v in globals().values() if isinstance(v, pd.DataFrame))
    df_text = f" • DataFrames: {df_count}" if df_count else ""

    footer_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(f"""
    <div style="text-align:center;padding:16px;border-radius:14px;
                background:linear-gradient(90deg,#051937,#004d7a,#008793,#00bf72,#a8eb12);
                color:white;margin-top:18px;box-shadow:0 0 12px rgba(0,0,0,0.4);">
        <h3 style="margin:0;">🚗 Parivahan Analytics — ALL-MAXED DASHBOARD</h3>
        <div style="opacity:0.95;font-size:14px;">
            Snapshot: <code>{footer_ts}</code>{runtime}{df_text} • Built with ❤️ & ⚙️ Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("💡 Tip: Use the exported Excel for polished reports and the ZIP snapshot for full archival backups.")

    # graceful balloon launch
    try:
        st.balloons()
    except Exception:
        pass

except Exception as e:
    st.warning(f"Footer rendering failed: {e}")
