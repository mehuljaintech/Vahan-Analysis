# =====================================================
# 🌏 GLOBAL INIT — MAXED OUT (IST, CACHING, USER, RETRY)
# =====================================================
import os
import sys
import json
import uuid
import time
import random
import platform
import logging
import requests
import traceback
import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps
from requests.adapters import HTTPAdapter, Retry

# =====================================================
# 🕒 1️⃣ Universal IST print-based logger
# =====================================================
def log_ist(msg: str):
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
    print(f"[IST {ist_time}] {msg}")

# =====================================================
# 🧭 2️⃣ Force all logging timestamps to IST
# =====================================================
class ISTFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Kolkata"))
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(level=logging.INFO)

for handler in root_logger.handlers:
    handler.setFormatter(ISTFormatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))

logging.info("✅ Logging timezone forced to IST")
log_ist("🚀 Streamlit App Initialization Started")

# =====================================================
# 👤 3️⃣ Random User / Session Identity (Anti-limit)
# =====================================================
if "user_id" not in st.session_state:
    st.session_state["user_id"] = f"user_{uuid.uuid4().hex[:8]}_{random.randint(1000,9999)}"
USER_ID = st.session_state["user_id"]

log_ist(f"🎯 Session started for user: {USER_ID}")

# =====================================================
# ♻️ 4️⃣ Ultra-Reliable Caching System
# =====================================================
@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def cached_json_fetch(url, params=None, headers=None):
    """Fetch API data with cache, retries, and IST logging."""
    log_ist(f"🌐 Fetching URL (cached): {url}")
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        resp = session.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        log_ist(f"✅ API Success [{url}] — {len(str(data))} chars")
        return data
    except Exception as e:
        logging.error(f"❌ API Error @ {url}: {e}")
        st.warning(f"API temporarily unavailable: {url}")
        return {}

# =====================================================
# 🧱 5️⃣ Global Error-Safe Wrapper
# =====================================================
def safe_exec(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_msg = f"⚠️ Exception in {fn.__name__}: {e}"
            logging.error(err_msg)
            traceback.print_exc()
            st.error(f"An internal error occurred — {fn.__name__}")
            return None
    return wrapper

# =====================================================
# 🚀 6️⃣ Streamlit Startup Banner (Visual + Console)
# =====================================================
def app_boot_banner():
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
    python_ver = platform.python_version()
    streamlit_ver = st.__version__
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    st.markdown(f"""
    <div style='
        background:linear-gradient(90deg,#0072ff,#00c6ff);
        color:white;
        padding:14px 24px;
        border-radius:14px;
        margin:15px 0 25px 0;
        box-shadow:0 4px 20px rgba(0,0,0,0.25);
        font-family:monospace;'>
        🕒 <b>App Booted:</b> {ist_time} (IST)<br>
        👤 <b>User:</b> {USER_ID}<br>
        ⚙️ <b>Environment:</b> Python {python_ver} | Streamlit {streamlit_ver}<br>
        🧠 <b>System:</b> {os_info}
    </div>
    """, unsafe_allow_html=True)

    print("=" * 70)
    print(f"[IST {ist_time}] ✅ Streamlit App Booted Successfully")
    print(f"[IST {ist_time}] User ID: {USER_ID}")
    print(f"[IST {ist_time}] Python {python_ver} | Streamlit {streamlit_ver}")
    print("=" * 70)

app_boot_banner()

# =====================================================
# 💾 7️⃣ Example Usage
# =====================================================
# data = cached_json_fetch("https://api.example.com/data")
# log_ist("📊 Data fetched successfully.")
# logging.info("Processing completed.")


# app/streamlit_app.py
# =====================================================
# 🌏 MAXED+ UNIVERSAL IMPORTS — AUTO INSTALL + AUTO REQUIREMENTS + CACHING
# =====================================================
import sys, os, subprocess, importlib, platform, json, time, traceback, warnings, random
from datetime import datetime
from zoneinfo import ZoneInfo
warnings.filterwarnings("ignore")

# =====================================================
# ⚙️ AUTO-INSTALL + AUTO-REQUIREMENTS MANAGEMENT
# =====================================================
REQUIREMENTS_FILE = "requirements.txt"
_installed = set()

def ensure_package(pkg_name, import_as=None):
    """Ensure package exists, import it, and auto-update requirements.txt."""
    global _installed
    try:
        mod = importlib.import_module(import_as or pkg_name)
        return mod
    except ModuleNotFoundError:
        print(f"📦 Installing missing package: {pkg_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name, "--quiet"])
        try:
            mod = importlib.import_module(import_as or pkg_name)
            # Auto-update requirements.txt
            _installed.add(pkg_name)
            if os.path.exists(REQUIREMENTS_FILE):
                with open(REQUIREMENTS_FILE, "r+") as f:
                    lines = f.read().splitlines()
                    if pkg_name not in lines:
                        f.write(f"\n{pkg_name}")
            else:
                with open(REQUIREMENTS_FILE, "w") as f:
                    f.write(f"{pkg_name}\n")
            return mod
        except Exception as e:
            print(f"⚠️ Failed to import {pkg_name}: {e}")
            return None

# =====================================================
# 🧠 CORE PYTHON UTILITIES
# =====================================================
import io, math, logging
import numpy as np
import pandas as pd
from functools import wraps, lru_cache

# =====================================================
# 🕒 GLOBAL IST LOGGER
# =====================================================
class ISTFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Kolkata"))
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

logging.basicConfig(level=logging.INFO)
for h in logging.getLogger().handlers:
    h.setFormatter(ISTFormatter("%(asctime)s | %(levelname)s | %(message)s"))

def log_ist(msg): 
    print(f"[IST {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %I:%M:%S %p')}] {msg}")

# =====================================================
# 🌱 ENVIRONMENT MANAGEMENT
# =====================================================
dotenv = ensure_package("python-dotenv")
from dotenv import load_dotenv
load_dotenv()

# =====================================================
# 📊 STREAMLIT + CORE VISUALIZATION
# =====================================================
st = ensure_package("streamlit")
alt = ensure_package("altair")
matplotlib = ensure_package("matplotlib")
plotly = ensure_package("plotly")
import plotly.express as px
import matplotlib.pyplot as plt
sns = ensure_package("seaborn")

# =====================================================
# 📈 DATA & ML STACK
# =====================================================
sklearn = ensure_package("scikit-learn", "sklearn")
statsmodels = ensure_package("statsmodels")

try:
    prophet = ensure_package("prophet")
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# AI & Deep Learning
try:
    torch = ensure_package("torch")
    transformers = ensure_package("transformers")
except Exception:
    torch = transformers = None

# Boosted trees (optional)
for pkg in ["xgboost", "lightgbm", "catboost"]:
    try:
        ensure_package(pkg)
    except Exception:
        pass

# =====================================================
# 📁 FILE HANDLING + EXPORTS
# =====================================================
openpyxl = ensure_package("openpyxl")
xlsxwriter = ensure_package("xlsxwriter")
pdfkit = ensure_package("pdfkit")
reportlab = ensure_package("reportlab")
pypandoc = ensure_package("pypandoc")
python_docx = ensure_package("python-docx")
odfpy = ensure_package("odfpy")

from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# =====================================================
# ☁️ API / NETWORK UTILITIES
# =====================================================
requests = ensure_package("requests")
requests_cache = ensure_package("requests-cache")
retrying = ensure_package("retrying")
aiohttp = ensure_package("aiohttp")
tenacity = ensure_package("tenacity")

# =====================================================
# 🧰 SYSTEM & PERF UTILITIES
# =====================================================
joblib = ensure_package("joblib")
tqdm = ensure_package("tqdm")
psutil = ensure_package("psutil")
diskcache = ensure_package("diskcache")

# =====================================================
# 🧭 CACHE SYSTEM (TTL + DISKCACHE)
# =====================================================
if diskcache:
    from diskcache import Cache
    cache = Cache(".cache")
else:
    cache = None

from functools import wraps
import time

def cached(ttl=3600):
    """Universal cache decorator — supports diskcache (if available) or in-memory fallback."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{fn.__name__}:{str(args)}:{str(kwargs)}"  # ✅ no backslashes

            try:
                if cache:
                    # Check if cached and within TTL
                    if key in cache and (time.time() - cache.created(key) < ttl):
                        return cache[key]
                    val = fn(*args, **kwargs)
                    cache[key] = val
                    return val
                else:
                    # No cache available, just run the function
                    return fn(*args, **kwargs)
            except Exception as e:
                print(f"[cache] Error in {fn.__name__}: {e}")
                return fn(*args, **kwargs)

        return wrapper
    return decorator

# =====================================================
# 🧱 GLOBAL STARTUP LOG
# =====================================================
log_ist("🚀 All MAXED+ Imports Loaded")
print("=" * 80)
print(f"🕒 Booted @ {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %I:%M:%S %p')} IST")
print(f"🧠 Python {platform.python_version()} | Streamlit {st.__version__} | Pandas {pd.__version__}")
print(f"📦 Total Auto-Installed Packages: {len(_installed)} → {list(_installed)}")
print("=" * 80)

# =====================================================
# ✅ VAHAN MODULES (KEEP YOUR ORIGINAL)
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
# 🔧 MAXED ERROR-HARDENED, STATELESS FETCHER — FULL HTTP ERROR HANDLING
# =====================================================
import threading
import collections
import random
import time
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
from typing import Optional, List, Dict

# ----------------------
# Configuration knobs
# ----------------------
MAX_ATTEMPTS = 6                         # absolute max tries across rotations
BASE_BACKOFF = 1.2                       # base multiplier for exponential backoff
JITTER = 0.35                            # jitter fraction
KEY_COOLDOWN_SECONDS = 300               # cooldown for a key after severe failure (5 minutes)
KEY_BLOCK_SECONDS = 3600                 # block a key after repeated critical failures (1 hour)
ENDPOINT_CIRCUIT_BREAK_SECONDS = 120     # circuit-break window for unhealthy endpoints
CIRCUIT_BREAK_THRESHOLD = 3              # failures before tripping circuit
MAX_PROXY_USAGE_RATIO = 0.35             # probability to use proxy if proxies exist
DISABLE_CACHE_FOR_STATLESS = False       # set True to always bypass cache for stateless calls

# ----------------------
# In-memory maps (ephemeral)
# ----------------------
_key_cooldowns: Dict[str, float] = {}            # key -> available_from_timestamp
_key_blocks: Dict[str, float] = {}               # key -> blocked_until_timestamp
_endpoint_failures: Dict[str, collections.deque] = {}  # endpoint -> deque[timestamps]
_endpoint_circuit: Dict[str, float] = {}         # endpoint -> circuit_tripped_until
_lock = threading.Lock()

# Ensure API_CONFIG supports mirrors list (optional)
# Example:
# API_CONFIG["parivahan"]["mirrors"] = ["https://backup1...", "https://backup2..."]
for svc in API_CONFIG.values():
    svc.setdefault("mirrors", [])

# ----------------------
# Helper utilities
# ----------------------
def _now_ts() -> float:
    return time.time()

def _is_key_available(key: str) -> bool:
    """Return True if key is not in cooldown or block."""
    now = _now_ts()
    if key in _key_blocks and _key_blocks[key] > now:
        return False
    if key in _key_cooldowns and _key_cooldowns[key] > now:
        return False
    return True

def _cooldown_key(key: str, seconds: int = KEY_COOLDOWN_SECONDS):
    with _lock:
        _key_cooldowns[key] = _now_ts() + seconds
    log_ist(f"🔒 Key cooldown applied for {seconds}s for key prefix: {str(key)[:6]}")

def _block_key(key: str, seconds: int = KEY_BLOCK_SECONDS):
    with _lock:
        _key_blocks[key] = _now_ts() + seconds
    log_ist(f"⛔ Key blocked for {seconds}s for key prefix: {str(key)[:6]}")

def _register_endpoint_failure(endpoint: str):
    """Register a failure and trip circuit if threshold reached."""
    now = _now_ts()
    dq = _endpoint_failures.setdefault(endpoint, collections.deque(maxlen=20))
    dq.append(now)
    # Count failures in last ENDPOINT_CIRCUIT_BREAK_SECONDS
    cutoff = now - ENDPOINT_CIRCUIT_BREAK_SECONDS
    recent = [t for t in dq if t >= cutoff]
    if len(recent) >= CIRCUIT_BREAK_THRESHOLD:
        _endpoint_circuit[endpoint] = now + ENDPOINT_CIRCUIT_BREAK_SECONDS
        log_ist(f"🛑 Circuit tripped for endpoint {endpoint} until {_endpoint_circuit[endpoint]}")

def _is_endpoint_available(endpoint: str) -> bool:
    until = _endpoint_circuit.get(endpoint, 0)
    return _now_ts() > until

def _choose_mirror(api_name: str) -> Optional[str]:
    cfg = API_CONFIG.get(api_name, {})
    mirrors = cfg.get("mirrors", []) or []
    base = cfg.get("base", "")
    candidates = [base] + mirrors
    random.shuffle(candidates)
    # Return first that is not currently under circuit break (use full candidate as key)
    for c in candidates:
        if _is_endpoint_available(c):
            return c
    return None

def _attempt_backoff(attempt: int, scale: float = BASE_BACKOFF):
    backoff = (scale ** attempt) + (random.random() * JITTER * scale)
    log_ist(f"⏳ Backing off {backoff:.2f}s (attempt {attempt})")
    time.sleep(backoff)

# ----------------------
# Key selection with cooldown awareness
# ----------------------
def _get_available_token(api_name: str) -> Optional[str]:
    """Pick an available token (not in cooldown/block)."""
    cfg = API_CONFIG.get(api_name, {})
    keys = list(cfg.get("keys", []) or [])
    random.shuffle(keys)
    for k in keys:
        if _is_key_available(k):
            return k
    return None

def _mark_key_failure(api_name: str, key: Optional[str], status_code: Optional[int] = None, fatal: bool = False):
    if not key:
        return
    # For certain status codes, escalate
    if status_code in (401,):
        # likely invalid token -> block for longer
        _block_key(key, seconds=KEY_BLOCK_SECONDS)
    elif status_code in (403,):
        # forbidden, cooldown the key and escalate if repeated
        _cooldown_key(key, seconds=KEY_COOLDOWN_SECONDS)
    elif status_code in (429,):
        # rate limit: cooldown key a bit
        _cooldown_key(key, seconds=max(KEY_COOLDOWN_SECONDS // 2, 60))
    else:
        # generic error -> short cooldown
        _cooldown_key(key, seconds=60)

    log_ist(f"⚠️ Marked failure for key prefix {str(key)[:6]} status={status_code} fatal={fatal}")

# ----------------------
# Full MAXED stateless fetcher
# ----------------------
def fetch_api_maxed(api_name: str,
                    endpoint: str,
                    params: dict = None,
                    method: str = "GET",
                    json_body: dict = None,
                    allow_redirects: bool = True,
                    disable_cache: bool = DISABLE_CACHE_FOR_STATLESS,
                    max_attempts: int = MAX_ATTEMPTS) -> dict:
    """
    The MAXED stateless fetcher:
      - rotates mirrors if available
      - respects endpoint circuit breaker
      - rotates keys but avoids keys in cooldown/block
      - special handling for 401/403/429
      - jittered exponential backoff
      - cookie-free, ephemeral sessions
    """
    # Optionally bypass cache completely when requested
    cache_bypass = disable_cache

    cfg = API_CONFIG.get(api_name, {})
    auth_type = cfg.get("auth_type", "bearer")

    attempt = 0
    tried_keys = set()
    last_exc = None

    # pick a mirror or base that is available
    base_choice = _choose_mirror(api_name)
    if not base_choice:
        log_ist(f"❌ No healthy endpoint available for service {api_name} (all circuits tripped).")
        return {}

    url_base = base_choice.rstrip("/")
    url = f"{url_base}/{endpoint.lstrip('/')}"

    while attempt < max_attempts:
        attempt += 1

        # choose token that is not in cooldown / block
        token = _get_available_token(api_name)
        if token is None:
            # no available keys — use a token-less request if allowed, else wait & retry
            log_ist(f"⚠️ No available API keys for {api_name} (attempt {attempt}) — trying without token or waiting shortly.")
            # small wait with jitter before retrying to allow keys to cool
            _attempt_backoff(attempt)
        else:
            tried_keys.add(token)

        # build ephemeral session + headers
        headers = build_spoofed_headers(api_name=api_name)
        if token and auth_type == "bearer":
            headers["Authorization"] = f"Bearer {token}"
        # if param-based, we'll inject later into params copy

        # ephemeral session (cookie-cleared)
        session = requests.Session()
        session.cookies.clear()
        session.trust_env = False
        # mount retry adapter but rely on our algorithm mostly
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))

        # optionally select proxy for this single-request
        proxy = _choose_proxy()
        proxies = {"http": proxy, "https": proxy} if proxy else None
        if proxy:
            log_ist(f"🔀 (Proxy) Using proxy for this call: {proxy}")

        # prepare params copy
        req_params = dict(params or {})
        if auth_type == "param" and token:
            req_params["apikey"] = token

        try:
            log_ist(f"🌐 [{api_name.upper()}] Try#{attempt} -> {url} (token={'yes' if token else 'no'})")
            if method.upper() == "GET":
                resp = session.get(url, headers=headers, params=req_params, timeout=30, allow_redirects=allow_redirects, proxies=proxies)
            else:
                resp = session.request(method.upper(), url, headers=headers, params=req_params, json=json_body or {}, timeout=60, allow_redirects=allow_redirects, proxies=proxies)

            status = resp.status_code

            # Handle common statuses with dedicated logic
            if status == 204:
                log_ist(f"⚪ 204 No Content for {url} — returning empty dict")
                return {}

            if 200 <= status < 300:
                # success
                try:
                    payload = resp.json()
                except ValueError:
                    payload = {"text": resp.text}
                log_ist(f"✅ {api_name} success {status} -> {url}")
                return payload

            # CLIENT ERRORS (400-499)
            if 400 <= status < 500:
                if status == 400:
                    log_ist(f"❗ 400 Bad Request for {url}: {resp.text[:200]}")
                    _register_endpoint_failure(url)
                    _mark_key_failure(api_name, token, status_code=status)
                    # often user params wrong — do not retry infinitely
                    _attempt_backoff(attempt)
                    last_exc = HTTPError(f"400 Bad Request: {resp.text}")
                    # break loop if repeated
                    if attempt >= max_attempts:
                        break
                    continue

                if status == 401:
                    # Unauthorized — token invalid. Block this key longer
                    log_ist(f"🔐 401 Unauthorized — token likely invalid for {api_name}")
                    _mark_key_failure(api_name, token, status_code=401, fatal=True)
                    _register_endpoint_failure(url)
                    # immediately rotate to next key and backoff
                    _attempt_backoff(attempt + 1)
                    continue

                if status == 403:
                    # Forbidden — rotate key, escalate cooldown, try mirror endpoint
                    log_ist(f"⛔ 403 Forbidden — rotating token and possibly mirror (attempt {attempt})")
                    _mark_key_failure(api_name, token, status_code=403)
                    _register_endpoint_failure(url)
                    # try a different mirror immediately if available
                    base_choice = _choose_mirror(api_name)
                    if base_choice and base_choice.rstrip("/") != url_base:
                        url_base = base_choice.rstrip("/")
                        url = f"{url_base}/{endpoint.lstrip('/')}"
                        log_ist(f"🔁 Switching to mirror endpoint: {url_base}")
                    _attempt_backoff(attempt + 1)
                    continue

                if status == 404:
                    log_ist(f"🔍 404 Not Found for {url}. Endpoint may be incorrect.")
                    # no point in repeating many times
                    _register_endpoint_failure(url)
                    last_exc = HTTPError("404 Not Found")
                    break

                if status == 405:
                    log_ist(f"❌ 405 Method Not Allowed for {url}")
                    _register_endpoint_failure(url)
                    last_exc = HTTPError("405 Method Not Allowed")
                    break

                # Other 4xx: log and backoff shorter
                log_ist(f"⚠️ Client error {status} for {url}: {resp.text[:200]}")
                _mark_key_failure(api_name, token, status_code=status)
                _register_endpoint_failure(url)
                _attempt_backoff(attempt)
                continue

            # SERVER ERRORS (500-599)
            if 500 <= status < 600:
                log_ist(f"🔥 Server error {status} from {url} — will retry with backoff")
                _register_endpoint_failure(url)
                # for some server errors, escalate key cooldown lightly
                _mark_key_failure(api_name, token, status_code=status)
                _attempt_backoff(attempt * 2)
                continue

            # Rate-limit specifically
            if status == 429:
                retry_after = None
                try:
                    retry_after = int(resp.headers.get("Retry-After") or 0)
                except Exception:
                    retry_after = None
                wait = retry_after if retry_after and retry_after > 0 else (2 ** attempt) + random.random() * 2
                log_ist(f"🔁 429 Rate limited. Waiting {wait}s then rotating key if needed.")
                _mark_key_failure(api_name, token, status_code=429)
                time.sleep(wait)
                continue

            # Fallback for unknown statuses
            log_ist(f"❗ Unexpected HTTP status {status} for {url}. Body head: {str(resp.text)[:300]}")
            _register_endpoint_failure(url)
            _mark_key_failure(api_name, token, status_code=status)
            _attempt_backoff(attempt)
            continue

        except (Timeout, ConnectionError) as net_exc:
            log_ist(f"⚠️ Network error on attempt {attempt} for {url}: {net_exc}")
            _register_endpoint_failure(url)
            _mark_key_failure(api_name, token, fatal=False)
            _attempt_backoff(attempt)
            last_exc = net_exc
            continue

        except RequestException as req_exc:
            log_ist(f"⚠️ Requests exception on attempt {attempt} for {url}: {req_exc}")
            _register_endpoint_failure(url)
            _mark_key_failure(api_name, token, fatal=False)
            _attempt_backoff(attempt)
            last_exc = req_exc
            continue

        finally:
            try:
                session.close()
            except Exception:
                pass

    # exhausted attempts
    log_ist(f"❌ All attempts exhausted for {api_name} -> {endpoint}. Last error: {last_exc}")
    return {}

# ----------------------
# Small health & debug UI
# ----------------------
def maxed_fetcher_status():
    st.subheader("🛠 MAXED Fetcher Status")
    st.write("Key cooldowns (prefixes):")
    for k, t in list(_key_cooldowns.items())[:10]:
        st.write(f"- {str(k)[:8]} -> until {time.ctime(t)}")
    st.write("Key blocks (prefixes):")
    for k, t in list(_key_blocks.items())[:10]:
        st.write(f"- {str(k)[:8]} -> until {time.ctime(t)}")
    st.write("Endpoint circuits:")
    for ep, until in _endpoint_circuit.items():
        st.write(f"- {ep} -> tripped until {time.ctime(until)}")
    st.success("MAXED fetcher reporting ready.")

# =====================================================
# 🚀 STREAMLIT PAGE CONFIG — MAXED OUT EDITION
# =====================================================
import streamlit as st
import platform
import random
import psutil
import socket
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------- PAGE CONFIG (HARDENED) ----------------
st.set_page_config(
    page_title="🚀 Vahan Registrations — Parivahan Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://parivahan.gov.in",
        "Report a bug": "mailto:support@parivahan.gov.in",
        "About": "Vahan Analytics — Smart AI-Powered Registration Insights Dashboard",
    },
)

# ---------------- THEME OVERRIDE (CSS) ----------------
st.markdown("""
    <style>
    /* Smooth fade-in */
    .stApp {
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    /* Headline glow */
    .glow {
        font-size: 30px;
        font-weight: 700;
        text-align: center;
        color: white;
        text-shadow: 0px 0px 12px rgba(0,255,255,0.9);
        background: linear-gradient(90deg,#0072ff,#00c6ff);
        padding: 14px 18px;
        border-radius: 12px;
        margin-bottom: 18px;
        box-shadow: 0px 2px 14px rgba(0,0,0,0.3);
    }
    .subtext {
        text-align:center;
        color:#ccc;
        font-size:15px;
        margin-top:-8px;
        margin-bottom:25px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- RANDOM USER/DEVICE SPOOF ----------------
def random_user_agent():
    browsers = ["Chrome", "Firefox", "Edge", "Safari", "Brave"]
    platforms = ["Windows NT 10.0; Win64; x64", "Macintosh; Intel Mac OS X 13_0", "X11; Linux x86_64"]
    versions = [str(random.randint(90, 120)) for _ in range(3)]
    browser = random.choice(browsers)
    platform_str = random.choice(platforms)
    version_str = random.choice(versions)
    return f"Mozilla/5.0 ({platform_str}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version_str}.0.{random.randint(1000,9999)} Safari/537.36"

USER_AGENT = random_user_agent()
CLIENT_ID = f"client_{random.randint(100000,999999)}"

# ---------------- APP HEADER ----------------
st.markdown(f"""
<div class='glow'>🚀 Vahan Registrations Dashboard</div>
<div class='subtext'>
Parivahan Analytics — KPIs • AI Narratives (DeepInfra) • Forecasting • Clustering • Anomaly Detection • Smart Exports
</div>
""", unsafe_allow_html=True)

# ---------------- ENVIRONMENT SNAPSHOT ----------------
try:
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
    host = socket.gethostname()
    py_ver = platform.python_version()
    mem = psutil.virtual_memory()
    st.sidebar.markdown(f"""
    ### ⚙️ Environment Snapshot
    - 🕒 **IST:** {ist_time}
    - 💻 **Host:** `{host}`
    - 🐍 **Python:** {py_ver}
    - 🧠 **Memory Used:** {mem.percent}%
    - 🧩 **User-Agent:** `{USER_AGENT[:35]}...`
    """)
except Exception as e:
    st.sidebar.error(f"Environment info unavailable: {e}")

# ---------------- RELOAD SAFETY ----------------
if "page_loaded_once" not in st.session_state:
    st.session_state["page_loaded_once"] = True
    st.toast("✅ App Initialized — Welcome!", icon="🚀")
else:
    st.toast("🔄 Reloaded — All systems nominal", icon="🧠")

# ---------------- PREFETCH + CACHE CONFIG ----------------
@st.cache_data(show_spinner=False, ttl=3600, max_entries=200)
def preload_static_assets():
    """Load heavy static data once per session (fast reloads)."""
    return {"status": "cached", "timestamp": datetime.now().isoformat()}

preload_static_assets()

# ---------------- FINAL LOG LINE ----------------
print(f"[IST {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}] ✅ Streamlit UI Booted — {CLIENT_ID} — UA: {USER_AGENT}")

# =====================================================
# 🧭 SIDEBAR FILTERS — MAXED OUT EDITION
# =====================================================
import streamlit as st
import os
import random
import platform
from datetime import date, datetime
from zoneinfo import ZoneInfo

# ---------------- SECTION HEADER ----------------
st.sidebar.markdown("""
<style>
.sidebar-title {
    font-size:20px;
    font-weight:700;
    background:linear-gradient(90deg,#0072ff,#00c6ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.sidebar-sub {
    color:#aaa;
    font-size:13px;
    margin-top:-8px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-title'>⚙️ Filters & Options</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>Customize data scope, analytics, and AI modes</div>", unsafe_allow_html=True)

# ---------------- BASE VARIABLES ----------------
today = date.today()
default_from_year = max(2017, today.year - 1)

# Cache last used filters across reruns
if "filters" not in st.session_state:
    st.session_state.filters = {}

def get_filter_value(key, default):
    return st.session_state.filters.get(key, default)

def set_filter_value(key, value):
    st.session_state.filters[key] = value

# ---------------- YEAR RANGE ----------------
from_year = st.sidebar.number_input(
    "📅 From Year",
    min_value=2012,
    max_value=today.year,
    value=get_filter_value("from_year", default_from_year),
    help="Start year for registration data (min: 2012)",
)
set_filter_value("from_year", from_year)

to_year = st.sidebar.number_input(
    "📆 To Year",
    min_value=from_year,
    max_value=today.year,
    value=get_filter_value("to_year", today.year),
    help="End year for registration data (cannot be earlier than From Year)",
)
set_filter_value("to_year", to_year)

# ---------------- LOCATION FILTERS ----------------
state_code = st.sidebar.text_input(
    "🏛️ State Code (blank = All India)",
    value=get_filter_value("state_code", ""),
    help="Enter state code (e.g., MH, DL, TN). Leave blank for All-India aggregate.",
)
set_filter_value("state_code", state_code)

rto_code = st.sidebar.text_input(
    "🏢 RTO Code (0 = Aggregate)",
    value=get_filter_value("rto_code", "0"),
    help="Specific RTO code, or 0 for state-level aggregation.",
)
set_filter_value("rto_code", rto_code)

# ---------------- VEHICLE FILTERS ----------------
vehicle_classes = st.sidebar.text_input(
    "🚗 Vehicle Classes (e.g., 2W, 3W, 4W)",
    value=get_filter_value("vehicle_classes", ""),
)
set_filter_value("vehicle_classes", vehicle_classes)

vehicle_makers = st.sidebar.text_input(
    "🏭 Vehicle Makers (comma-separated or IDs)",
    value=get_filter_value("vehicle_makers", ""),
)
set_filter_value("vehicle_makers", vehicle_makers)

vehicle_type = st.sidebar.text_input(
    "🚙 Vehicle Type (optional)",
    value=get_filter_value("vehicle_type", ""),
)
set_filter_value("vehicle_type", vehicle_type)

# ---------------- TIME FILTERS ----------------
time_period_labels = {
    0: "Monthly",
    1: "Quarterly",
    2: "Yearly"
}
time_period = st.sidebar.selectbox(
    "⏱️ Time Period",
    options=list(time_period_labels.keys()),
    index=get_filter_value("time_period", 0),
    format_func=lambda x: time_period_labels[x],
)
set_filter_value("time_period", time_period)

fitness_check = st.sidebar.selectbox(
    "🧾 Fitness Check",
    options=[0, 1],
    index=get_filter_value("fitness_check", 0),
    format_func=lambda x: "No" if x == 0 else "Yes",
)
set_filter_value("fitness_check", fitness_check)

# ---------------- ADVANCED TOGGLES ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Advanced Analytics")

enable_forecast = st.sidebar.checkbox("📈 Enable Forecasting", value=True)
enable_anomaly = st.sidebar.checkbox("🚨 Enable Anomaly Detection", value=True)
enable_clustering = st.sidebar.checkbox("🔍 Enable Clustering", value=True)
enable_ai = st.sidebar.checkbox("🤖 Enable DeepInfra AI Narratives", value=True)
forecast_periods = st.sidebar.number_input("📅 Forecast Horizon (months)", min_value=1, max_value=36, value=3)

# ---------------- DEEPINFRA CONFIG ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 DeepInfra Settings")

# Try multiple key sources (env, secrets, fallback)
DEEPINFRA_API_KEY = (
    os.environ.get("DEEPINFRA_API_KEY")
    or (st.secrets.get("DEEPINFRA_API_KEY") if "DEEPINFRA_API_KEY" in st.secrets else None)
)

DEEPINFRA_MODEL = os.environ.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# Validate key presence
if enable_ai:
    if not DEEPINFRA_API_KEY:
        st.sidebar.error("🚫 DeepInfra API key not found — AI features disabled.")
        enable_ai = False
    else:
        st.sidebar.success(f"✅ DeepInfra connected ({DEEPINFRA_MODEL.split('/')[-1]})")

# ---------------- DEV & DEBUG TOGGLES ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Developer Options")
dev_mode = st.sidebar.toggle("🧪 Developer Mode", value=False)
safe_mode = st.sidebar.toggle("🛡️ Safe Mode (disable risky ops)", value=True)

# ---------------- LIVE SUMMARY ----------------
st.sidebar.markdown("---")
summary_md = f"""
**Active Filters:**
- 📅 {from_year} → {to_year}
- 🌍 State: `{state_code or 'All-India'}`, RTO: `{rto_code}`
- 🚗 Classes: `{vehicle_classes or 'All'}`, Makers: `{vehicle_makers or 'All'}`
- ⏱️ Period: `{time_period_labels[time_period]}`
- 🧾 Fitness: `{'Yes' if fitness_check else 'No'}`
- 🤖 AI: `{'Enabled' if enable_ai else 'Disabled'}`
"""
st.sidebar.info(summary_md)

# ---------------- LOG CONFIRMATION ----------------
print(f"[{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}] ✅ Sidebar filters loaded — Forecast:{enable_forecast} | Anomaly:{enable_anomaly} | AI:{enable_ai}")

# =====================================================
# ⚙️ VAHAN PARAMS + UNIVERSAL SAFE FETCHER — MAXED OUT
# =====================================================
import random
import time
import json
import traceback
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------- Build Params (Base) ----------------
params_common = build_params(
    from_year, to_year,
    state_code=state_code,
    rto_code=rto_code,
    vehicle_classes=vehicle_classes,
    vehicle_makers=vehicle_makers,
    time_period=time_period,
    fitness_check=fitness_check,
    vehicle_type=vehicle_type,
)

# =====================================================
# 🌐 UNIVERSAL SAFE FETCH FUNCTION
# =====================================================

# Rotating headers to spoof browser requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/122.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
]

# Optional token rotation (DeepInfra / Parivahan)
TOKEN_POOL = []
try:
    if "api_keys" in st.secrets:
        for kset in st.secrets["api_keys"].values():
            if isinstance(kset, list):
                TOKEN_POOL += kset
except Exception:
    pass

def random_headers():
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "X-Request-ID": f"{random.randint(100000,999999)}-{int(time.time())}",
        "Referer": "https://parivahan.gov.in/",
    }
    if TOKEN_POOL:
        headers["Authorization"] = f"Bearer {random.choice(TOKEN_POOL)}"
    return headers

# =====================================================
# 🧠 Robust fetcher with exponential backoff + cache
# =====================================================
@st.cache_data(show_spinner=False, ttl=900, max_entries=100)
def fetch_json(endpoint: str, params: dict = params_common, desc: str = "") -> dict:
    """
    Universal safe API fetcher with retries, 403/429 handling,
    randomized spoof headers, and silent fallback.
    """
    base_url = "https://vahanapi.parivahan.gov.in/"
    url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
    max_retries = 5
    backoff_base = 2
    timeout = 20

    for attempt in range(1, max_retries + 1):
        headers = random_headers()
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            code = response.status_code

            # Handle rate limits / forbidden
            if code == 403:
                st.info(f"🚫 Forbidden (403) while fetching {desc}. Retrying with new headers...")
                time.sleep(random.uniform(1.0, 3.0))
                continue
            elif code == 429:
                wait = random.uniform(3.0, 7.0)
                st.warning(f"⚠️ Rate limited (429). Retrying after {wait:.1f}s...")
                time.sleep(wait)
                continue
            elif code >= 500:
                wait = backoff_base ** attempt + random.uniform(0.2, 1.0)
                st.warning(f"🌀 Server error {code}. Retry {attempt}/{max_retries} after {wait:.1f}s...")
                time.sleep(wait)
                continue
            elif code not in (200, 201):
                st.error(f"❌ Unexpected HTTP {code} for {desc}")
                return {}

            try:
                json_data = response.json()
                if not json_data:
                    st.warning(f"⚠️ Empty response for {desc}")
                    return {}
                print(f"[IST {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}] ✅ {desc or endpoint} fetched OK")
                return json_data
            except Exception:
                st.error(f"⚠️ Invalid JSON in {desc} response")
                return {}

        except requests.exceptions.Timeout:
            st.warning(f"⏳ Timeout fetching {desc}. Retry {attempt}/{max_retries}...")
            time.sleep(backoff_base ** attempt)
        except requests.exceptions.ConnectionError:
            st.warning(f"🔌 Connection error — retry {attempt}/{max_retries}...")
            time.sleep(backoff_base ** attempt)
        except Exception as e:
            print(f"⚠️ {desc} — Unexpected error: {e}")
            traceback.print_exc()
            time.sleep(1.5)

    # After retries exhausted
    st.error(f"❗ Failed to fetch {desc or endpoint} after {max_retries} retries.")
    return {}

# =====================================================
# 🌈 Usage Example
# =====================================================
# with st.spinner("Fetching Vahan data..."):
#     trend_data = fetch_json("vahandashboard/registrationtrend", params_common, desc="Registration Trend")
#     if trend_data:
#         st.success("✅ Registration Trend Data Loaded")
#     else:
#         st.error("⚠️ Failed to load registration trend.")

# =====================================================
# 🧠 DeepInfra Universal Chat Helper (Fully Maxed)
# =====================================================
import random
import string
import time
import traceback

DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# Possible user-agents for light spoofing
_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2_1 like Mac OS X)",
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X)",
    "Mozilla/5.0 (Android 14; Mobile)",
]

# Robust DeepInfra client
def deepinfra_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    retries: int = 5,
    model: str = None
):
    """
    Ultra-resilient DeepInfra API client with retry, rotation, and full error capture.
    Compatible with OpenAI-format responses.
    """

    if not DEEPINFRA_API_KEY:
        return {"error": "🔑 DeepInfra API key not configured. Please set DEEPINFRA_API_KEY."}

    if not model:
        model = DEEPINFRA_MODEL

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
        "stream": False
    }

    # Randomized rotating headers to avoid fingerprinting
    def _rotating_headers():
        return {
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": random.choice(_UA_POOL),
            "X-Request-ID": ''.join(random.choices(string.ascii_letters + string.digits, k=12)),
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        }

    for attempt in range(retries):
        headers = _rotating_headers()
        try:
            resp = requests.post(
                DEEPINFRA_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            status = resp.status_code

            # === Handle status codes ===
            if status == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    return {"text": content, "raw": data, "status": 200}
                else:
                    return {"error": "Malformed response", "raw": data, "status": 200}

            elif status in [429, 408, 500, 502, 503, 504]:  # rate limit or transient errors
                wait = (2 ** attempt) + random.random()
                st.info(f"⏳ DeepInfra retrying (attempt {attempt+1}/{retries}) after {wait:.1f}s due to transient error {status}")
                time.sleep(wait)
                continue

            elif status == 401:
                return {"error": "🚫 Unauthorized — check DeepInfra API key", "status": 401}
            elif status == 403:
                return {"error": "❌ Forbidden — key lacks access or is invalid", "status": 403}
            elif status == 404:
                return {"error": "🔍 Model or endpoint not found", "status": 404}
            else:
                return {"error": f"HTTP {status}: {resp.text[:500]}", "status": status}

        except requests.exceptions.Timeout:
            st.warning(f"⚠️ Timeout on DeepInfra (attempt {attempt+1}/{retries}), retrying...")
            time.sleep(2 ** attempt + random.random())
        except requests.exceptions.ConnectionError as ce:
            st.warning(f"🌐 Connection error: {ce}, retrying...")
            time.sleep(2 ** attempt + random.random())
        except Exception as e:
            st.error(f"🔥 Unexpected DeepInfra error: {e}\n{traceback.format_exc()}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt + random.random())
            else:
                return {"error": str(e), "trace": traceback.format_exc(), "status": "exception"}

    return {"error": f"Failed after {retries} retries", "status": "exhausted"}

# =====================================================
# 1️⃣ MAXED Category Distribution + Full Analysis (Real + Predicted)
# =====================================================
import io
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from datetime import date, datetime
from zoneinfo import ZoneInfo
import streamlit as st
import math
import json
import warnings
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# --- Helper: try to coerce dataframe to canonical form ---
def normalize_cat_df(df):
    """
    Try to find sensible date/index and value columns and a category column.
    Returns df with columns: ['category','date','value'] (date is datetime)
    """
    if df is None:
        return pd.DataFrame(columns=["category","date","value"])
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    # potential name matches
    cat_cols = [cols_lower.get(k) for k in ("category","cat","name","label") if k in cols_lower] + \
               [c for c in df.columns if "category" in c.lower() or "cat" in c.lower() or "name" in c.lower()]
    date_cols = [cols_lower.get(k) for k in ("date","day","datetime","ds","period") if k in cols_lower] + \
                [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "period" in c.lower() or "ds" in c.lower()]
    value_cols = [cols_lower.get(k) for k in ("value","count","y","registrations","total","cnt") if k in cols_lower] + \
                 [c for c in df.columns if any(x in c.lower() for x in ["value","count","reg","total","cnt","y"])]
    # pick first sensible
    category = cat_cols[0] if cat_cols else (df.columns[0] if len(df.columns) >= 1 else None)
    datecol = date_cols[0] if date_cols else None
    valuecol = value_cols[0] if value_cols else (df.columns[-1] if len(df.columns) >= 2 else None)

    # If category appears numeric, maybe different shape: treat rows as categories with values
    # Attempt common shapes:
    if datecol is None and valuecol is not None and category is not None and df[category].dtype == object:
        # assume each row is a category with numeric value
        out = pd.DataFrame({
            "category": df[category].astype(str),
            "date": pd.to_datetime([datetime(THIS_YEAR,1,1)]*len(df)),
            "value": pd.to_numeric(df[valuecol], errors="coerce").fillna(0)
        })
        return out

    # If date column exists, parse it
    if datecol:
        try:
            df[datecol] = pd.to_datetime(df[datecol], errors="coerce", dayfirst=False)
        except Exception:
            df[datecol] = pd.to_datetime(df[datecol].astype(str), errors="coerce")
    else:
        # no date column — create a default date (this year)
        df["_generated_date"] = pd.to_datetime(df.get("_generated_date", datetime(THIS_YEAR,1,1)))
        datecol = "_generated_date"

    # Map category/value
    if category is None:
        # if only two columns, assume first is label
        if len(df.columns) >= 2:
            category = df.columns[0]
        else:
            df["category"] = "all"
            category = "category"
    if valuecol is None:
        # try to infer numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        valuecol = numeric_cols[0] if numeric_cols else None
    if valuecol is None:
        # fallback: use row counts
        df["_value"] = 1
        valuecol = "_value"

    out = pd.DataFrame({
        "category": df[category].astype(str),
        "date": pd.to_datetime(df[datecol], errors="coerce"),
        "value": pd.to_numeric(df[valuecol], errors="coerce").fillna(0)
    })
    out = out.dropna(subset=["category"]).reset_index(drop=True)
    return out

# --- Fetch & normalize ---
with st.spinner("Fetching category distribution (maxed)..."):
    raw_cat_json = fetch_json("vahandashboard/categoriesdonutchart", params_common, desc="Category distribution")
    cat_df_raw = to_df(raw_cat_json) if raw_cat_json else pd.DataFrame()
df_cat = normalize_cat_df(cat_df_raw)

if df_cat.empty:
    st.warning("No category data available. Ensure endpoint returned valid data.")
else:
    # ----- Aggregations -----
    # Ensure date present: fill missing dates with Jan 1 THIS_YEAR
    df_cat["date"] = df_cat["date"].fillna(pd.to_datetime(datetime(THIS_YEAR,1,1)))
    # daily/monthly/yearly aggregations
    df_cat["day"] = df_cat["date"].dt.floor("D")
    df_cat["month"] = df_cat["date"].dt.to_period("M").dt.to_timestamp()
    df_cat["year"] = df_cat["date"].dt.year

    # totals per category overall (real)
    total_by_cat = df_cat.groupby("category", as_index=False)["value"].sum().sort_values("value", ascending=False)
    overall_total = total_by_cat["value"].sum()

    # monthly trend (pivot)
    monthly = df_cat.groupby(["month","category"], as_index=False)["value"].sum()
    monthly_pivot = monthly.pivot(index="month", columns="category", values="value").fillna(0).sort_index()

    # yearly totals per category
    yearly = df_cat.groupby(["year","category"], as_index=False)["value"].sum()
    yearly_pivot = yearly.pivot(index="year", columns="category", values="value").fillna(0).sort_index()

    # totals by year (all categories)
    totals_by_year = df_cat.groupby("year", as_index=False)["value"].sum().set_index("year")["value"]

    # ----- KPI metrics: prev / this / predicted next -----
    prev_total = float(totals_by_year.get(PREV_YEAR, 0.0))
    this_total = float(totals_by_year.get(THIS_YEAR, 0.0))
    # We'll create predictions monthly and aggregate to next_year_total
    # Forecast input: monthly series of totals (index=month timestamp)
    monthly_totals = monthly.groupby("month", as_index=True)["value"].sum().sort_index()
    forecast_monthly = None
    predicted_next_total = None

    # Forecasting: Prophet preferred, linear trend fallback
    try:
        if PROPHET_AVAILABLE and len(monthly_totals) >= 12:
            # prophet expects ds & y
            from prophet import Prophet
            df_prop = monthly_totals.reset_index().rename(columns={"month":"ds","value":"y"})
            # ensure monthly frequency
            df_prop = df_prop.sort_values("ds")
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            m.fit(df_prop)
            future_periods = 12  # predict next 12 months (covers next year)
            future = m.make_future_dataframe(periods=future_periods, freq='MS')
            forecast = m.predict(future)
            # extract only future months for next-year (filter ds year == NEXT_YEAR)
            forecast_monthly = forecast[["ds","yhat","yhat_lower","yhat_upper"]].set_index("ds")
            # sum predicted months where ds.year == NEXT_YEAR
            predicted_next_total = float(forecast_monthly[forecast_monthly.index.year == NEXT_YEAR]["yhat"].sum())
        else:
            raise Exception("Prophet not available or insufficient history")
    except Exception as e:
        # Linear regression fallback on monthly index
        try:
            from sklearn.linear_model import LinearRegression
            # create numeric index
            if len(monthly_totals) >= 6:
                X = np.arange(len(monthly_totals)).reshape(-1,1)
                y = monthly_totals.values
                lr = LinearRegression()
                lr.fit(X, y)
                future_idx = np.arange(len(monthly_totals), len(monthly_totals) + 12).reshape(-1,1)
                yhat = lr.predict(future_idx)
                # build forecast_monthly for the next 12 months starting next month after last month in data
                last_month = pd.to_datetime(monthly_totals.index.max())
                next_months = pd.date_range(start=(last_month + pd.offsets.MonthBegin(1)).replace(day=1), periods=12, freq='MS')
                forecast_monthly = pd.DataFrame({
                    "yhat": np.maximum(yhat, 0),
                    "yhat_lower": np.maximum(yhat * 0.85, 0),
                    "yhat_upper": np.maximum(yhat * 1.15, 0),
                }, index=next_months)
                predicted_next_total = float(forecast_monthly[yhat >= 0]["yhat"].sum()) if len(forecast_monthly)>0 else 0.0
            else:
                predicted_next_total = float(np.nan)
                forecast_monthly = None
        except Exception:
            predicted_next_total = float(np.nan)
            forecast_monthly = None

    # If forecasting failed, set predicted to NaN
    if predicted_next_total is None or (isinstance(predicted_next_total, float) and math.isnan(predicted_next_total)):
        predicted_next_total = 0.0

    # ----- KPIS and comparisons -----
    def pct_change(a, b):
        try:
            if a == 0:
                return float("inf") if b != 0 else 0.0
            return ((b - a) / abs(a)) * 100.0
        except Exception:
            return 0.0

    kpi_prev_vs_this = pct_change(prev_total, this_total)
    kpi_this_vs_next = pct_change(this_total, predicted_next_total)

    # ----- UI: Summary KPIs -----
    st.subheader("Category Distribution — Real & Predicted (Maxed)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(label=f"Prev Year ({PREV_YEAR})", value=f"{int(prev_total):,}", delta=f"{kpi_prev_vs_this:.2f}% vs prev")
    k2.metric(label=f"This Year ({THIS_YEAR})", value=f"{int(this_total):,}", delta=f"{kpi_this_vs_next:.2f}% vs this")
    k3.metric(label=f"Predicted Next Year ({NEXT_YEAR})", value=f"{int(predicted_next_total):,}", delta=f"{pct_change(prev_total, predicted_next_total):.2f}% vs prev")
    k4.metric(label="Overall (All-time)", value=f"{int(overall_total):,}", delta=None)

    # ----- Top categories bar & donut (polished) -----
    st.markdown("### Top Categories — Real")
    topn = st.slider("Top N categories to show", min_value=3, max_value=min(50, len(total_by_cat)), value=min(10, len(total_by_cat)))
    top_cats = total_by_cat.head(topn)

    # altair bar
    bar = alt.Chart(top_cats).mark_bar().encode(
        x=alt.X("value:Q", title="Registrations"),
        y=alt.Y("category:N", sort='-x', title="Category"),
        tooltip=[alt.Tooltip("category:N"), alt.Tooltip("value:Q", format=",")]
    ).properties(height=40*min(len(top_cats),20), width=700, title=f"Top {topn} Categories (Total)")
    st.altair_chart(bar, use_container_width=True)

    # donut with plotly
    pie = px.pie(top_cats, names="category", values="value", hole=0.45, title="Top Categories (Donut)")
    pie.update_traces(textinfo='percent+label', hoverinfo='label+value')
    st.plotly_chart(pie, use_container_width=True)

    # ----- Time series with prediction ribbon -----
    st.markdown("### Monthly Trend — Real + Predicted")
    # prepare series for plot: monthly_totals (real) + forecast_monthly (pred)
    ts_real = monthly_totals.rename("real").reset_index()
    ts_real["month"] = pd.to_datetime(ts_real["month"])
    if forecast_monthly is not None and not forecast_monthly.empty:
        fc = forecast_monthly.reset_index().rename(columns={"index":"month"})
        fc["month"] = pd.to_datetime(fc["month"])
        # combine for plotting
        plot_df = pd.concat([
            ts_real.assign(type="real").rename(columns={"value":"y"}),
            fc.assign(type="predicted").rename(columns={"yhat":"y"})
        ], ignore_index=True, sort=False)
    else:
        plot_df = ts_real.assign(type="real").rename(columns={"value":"y"})

    # Plotly line with ribbon
    fig = px.line(plot_df, x="month", y="y", color="type", markers=True, title="Monthly Registrations — Real vs Predicted")
    if forecast_monthly is not None and not forecast_monthly.empty:
        fig.add_traces(px.scatter(fc, x="month", y="yhat").data)
        # add ribbon using filled area between lower & upper
        fig.add_traces([dict(x=fc["month"], y=fc["yhat_upper"], mode='lines', line=dict(width=0), showlegend=False),
                        dict(x=fc["month"], y=fc["yhat_lower"], mode='lines', fill='tonexty', fillcolor='rgba(0,176,246,0.2)', line=dict(width=0), showlegend=False)])
    fig.update_layout(legend_title_text="Series", xaxis_title="Month", yaxis_title="Registrations")
    st.plotly_chart(fig, use_container_width=True)

    # ----- Category-level comparisons (prev vs this vs predicted next) -----
    st.markdown("### Category-level Comparison: Prev vs This vs Predicted Next")
    # compute per-category prev, this, predicted (distribute predicted proportionally by historical share)
    hist_share = total_by_cat.copy()
    hist_share["share"] = hist_share["value"] / hist_share["value"].sum() if hist_share["value"].sum() > 0 else 0
    hist_share = hist_share.set_index("category")
    predicted_per_cat = (hist_share["share"].fillna(0) * predicted_next_total).reset_index().rename(columns={0:"predicted"})
    prev_per_cat = yearly_pivot.loc[PREV_YEAR] if PREV_YEAR in yearly_pivot.index else pd.Series(0, index=yearly_pivot.columns)
    this_per_cat = yearly_pivot.loc[THIS_YEAR] if THIS_YEAR in yearly_pivot.index else pd.Series(0, index=yearly_pivot.columns)

    # Build table
    cats = sorted(set(list(prev_per_cat.index) + list(this_per_cat.index) + list(hist_share.index)))
    rows = []
    for c in cats:
        prev_v = float(prev_per_cat.get(c, 0))
        this_v = float(this_per_cat.get(c, 0))
        pred_v = float(hist_share.loc[c]["share"] * predicted_next_total) if c in hist_share.index else 0.0
        rows.append({
            "category": c, f"{PREV_YEAR}": prev_v, f"{THIS_YEAR}": this_v,
            f"{NEXT_YEAR} (pred)": pred_v,
            "growth_prev_to_this %": pct_change(prev_v, this_v),
            "growth_this_to_next %": pct_change(this_v, pred_v)
        })
    comp_df = pd.DataFrame(rows).sort_values(f"{THIS_YEAR}", ascending=False)
    st.dataframe(comp_df.style.format({f"{PREV_YEAR}":"{:,}", f"{THIS_YEAR}":"{:,}", f"{NEXT_YEAR} (pred)":"{:,}", "growth_prev_to_this %":"{:.2f}%", "growth_this_to_next %":"{:.2f}%"}), height=350)

    # ----- Additional interactive charts: daily/monthly/yearly toggles -----
    st.markdown("### Drilldowns & Extra Views")
    show_daily = st.checkbox("Show daily trend", value=False)
    show_monthly_by_cat = st.checkbox("Show monthly stacked by category", value=True)
    show_yearly_heatmap = st.checkbox("Show yearly heatmap (category x year)", value=False)

    if show_daily:
        daily = df_cat.groupby(["day","category"], as_index=False)["value"].sum()
        daily_pivot = daily.pivot(index="day", columns="category", values="value").fillna(0)
        # small sample line chart (plot top categories)
        top_cats_list = top_cats["category"].tolist()
        daily_long = daily[daily["category"].isin(top_cats_list)]
        fig_daily = px.line(daily_long, x="day", y="value", color="category", title="Daily Registrations — Top Categories")
        fig_daily.update_layout(xaxis_title="Day", yaxis_title="Registrations")
        st.plotly_chart(fig_daily, use_container_width=True)

    if show_monthly_by_cat:
        # stacked area
        monthly_long = monthly.sort_values("month")
        fig_monthly = px.area(monthly_long, x="month", y="value", color="category", title="Monthly Stacked by Category")
        fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Registrations")
        st.plotly_chart(fig_monthly, use_container_width=True)

    if show_yearly_heatmap:
        # heatmap using plotly
        heat_df = yearly.reset_index() if False else yearly.reset_index()
        heat_pivot = yearly_pivot.fillna(0)
        heat_map = px.imshow(heat_pivot.T, labels=dict(x="Year", y="Category", color="Registrations"),
                             x=heat_pivot.index.astype(str).tolist(), y=heat_pivot.columns.tolist(),
                             aspect="auto", title="Yearly Heatmap (Categories x Year)")
        st.plotly_chart(heat_map, use_container_width=True)

    # ----- Exports & Downloads -----
    st.markdown("### Export Data & Charts")
    csv = df_cat.to_csv(index=False).encode("utf-8")
    st.download_button("Download raw category CSV", csv, file_name=f"category_raw_{THIS_YEAR}.csv", mime="text/csv")
    # Excel
    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_cat.to_excel(writer, sheet_name="raw", index=False)
            comp_df.to_excel(writer, sheet_name="comparison", index=False)
            monthly_totals.reset_index().to_excel(writer, sheet_name="monthly_totals", index=False)
            # forecast sheet
            if forecast_monthly is not None:
                forecast_monthly.reset_index().to_excel(writer, sheet_name="forecast_monthly", index=True)
        excel_buffer.seek(0)
        st.download_button("Download analysis Excel", excel_buffer, file_name=f"vahan_category_analysis_{THIS_YEAR}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("Excel export failed (openpyxl/xlsxwriter missing). CSV still available.")

    # ----- AI Summary (optional) -----
    if enable_ai:
        with st.spinner("Generating AI narrative (DeepInfra)..."):
            system = "You are a data analyst assistant. Summarize the real registration counts and predictions, focusing on top categories, growth/decline, and one actionable recommendation."
            sample_rows = comp_df.head(10).to_dict(orient="records")
            user_prompt = f"Real totals: prev_year={int(prev_total):,}, this_year={int(this_total):,}, predicted_next_year={int(predicted_next_total):,}. Top categories sample: {json.dumps(sample_rows, default=str)}. Provide 6 bullet points with: (1) top 3 categories, (2) categories growing fastest, (3) categories falling, (4) next-year risk signals, (5) one recommendation for stakeholders, (6) short 2-sentence executive summary."
            ai_out = deepinfra_chat(system, user_prompt, max_tokens=280)
            if isinstance(ai_out, dict) and "text" in ai_out:
                st.markdown("**AI Narrative (Live)**")
                st.write(ai_out["text"])
            else:
                st.info("AI narrative unavailable or key missing.")

    # ----- Final detailed download of charts & data -----
    st.markdown("---")
    st.success("✅ Category Distribution full analysis complete (real & predicted).")


# =====================================================
# 2️⃣ MAXED Top Makers — Real + Predicted + Full Analysis
# =====================================================
import io
import math
import json
import random
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
from datetime import date, datetime
from zoneinfo import ZoneInfo
import streamlit as st
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# ------------------ Helpers ------------------
def normalize_maker_df(df):
    """
    Normalize parse_makers output into canonical columns:
    returns DataFrame with columns ['maker','date','value'] (date may be same for all rows).
    If df already has time-series rows, parse date; otherwise set to THIS_YEAR start.
    """
    if df is None:
        return pd.DataFrame(columns=["maker","date","value"])
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # detect maker column
    maker_col = None
    for k in ("maker","manufacturer","label","name"):
        if k in cols_lower:
            maker_col = cols_lower[k]; break
    if maker_col is None:
        # fallback: first column
        maker_col = df.columns[0] if len(df.columns) > 0 else None

    # detect value column
    value_col = None
    for k in ("value","count","registrations","total","y"):
        if k in cols_lower:
            value_col = cols_lower[k]; break
    if value_col is None:
        # pick numeric column if present
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_col = num_cols[0] if num_cols else (df.columns[-1] if len(df.columns)>=2 else None)

    # detect date-like column
    date_col = None
    for k in ("date","ds","month","period","time"):
        if k in cols_lower:
            date_col = cols_lower[k]; break
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # If dataset is simple maker -> value (no dates), create a single-row per maker dated THIS_YEAR start
    if date_col is None:
        df = df[[maker_col, value_col]].rename(columns={maker_col: "maker", value_col: "value"})
        df["date"] = pd.to_datetime(datetime(THIS_YEAR,1,1))
    else:
        df = df.rename(columns={maker_col: "maker", value_col: "value", date_col: "date"})
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(datetime(THIS_YEAR,1,1))
    df["maker"] = df["maker"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(pd.to_datetime(datetime(THIS_YEAR,1,1)))
    return df[["maker","date","value"]]

def pct_change(a, b):
    try:
        if a == 0:
            return float("inf") if b != 0 else 0.0
        return ((b - a) / abs(a)) * 100.0
    except Exception:
        return 0.0

def forecast_monthly_series(series, periods=12):
    """
    series: pd.Series indexed by period timestamps (monthly) with numeric values
    returns DataFrame index=forecast_months ts columns ['yhat','yhat_lower','yhat_upper']
    Uses Prophet if available and enough history, else linear regression fallback. Returns None if insufficient data.
    """
    try:
        s = series.dropna().sort_index()
        if len(s) < 6:
            return None
        # prefer Prophet
        if PROPHET_AVAILABLE and len(s) >= 12:
            from prophet import Prophet
            df_prop = s.reset_index().rename(columns={s.index.name or 'index':'ds', 0:'y'})
            # ensure columns named ds,y
            df_prop.columns = ['ds','y']
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=periods, freq='MS')
            fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].set_index("ds")
            return fc.tail(periods)
        else:
            # linear regression on index
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(s)).reshape(-1,1)
            y = s.values
            lr = LinearRegression()
            lr.fit(X,y)
            future_X = np.arange(len(s), len(s)+periods).reshape(-1,1)
            yhat = lr.predict(future_X)
            last_ts = s.index.max()
            next_months = pd.date_range(start=(pd.to_datetime(last_ts) + pd.offsets.MonthBegin(1)).replace(day=1), periods=periods, freq='MS')
            fc = pd.DataFrame({
                "yhat": np.maximum(yhat, 0),
                "yhat_lower": np.maximum(yhat * 0.85, 0),
                "yhat_upper": np.maximum(yhat * 1.15, 0),
            }, index=next_months)
            return fc
    except Exception:
        return None

# ------------------ Fetch & prepare ------------------
with st.spinner("Fetching Top Makers (maxed)..."):
    mk_json = fetch_json("vahandashboard/top5Makerchart", params_common, desc="Top Makers")
    mk_df_raw = parse_makers(mk_json) if mk_json else pd.DataFrame()
df_mk = normalize_maker_df(mk_df_raw)

if df_mk.empty:
    st.warning("No Top Makers data available.")
else:
    # derive time buckets
    df_mk["month"] = df_mk["date"].dt.to_period("M").dt.to_timestamp()
    df_mk["year"] = df_mk["date"].dt.year
    df_mk["day"] = df_mk["date"].dt.floor("D")

    # overall totals and ranking (real)
    totals_by_maker = df_mk.groupby("maker", as_index=False)["value"].sum().sort_values("value", ascending=False)
    overall_total = totals_by_maker["value"].sum()

    # monthly totals (all makers combined)
    monthly_totals = df_mk.groupby("month", as_index=True)["value"].sum().sort_index()

    # yearly totals per maker (pivot)
    yearly = df_mk.groupby(["year","maker"], as_index=False)["value"].sum()
    yearly_pivot = yearly.pivot(index="year", columns="maker", values="value").fillna(0).sort_index()

    # per-maker monthly pivot
    monthly_by_maker = df_mk.groupby(["month","maker"], as_index=False)["value"].sum()
    monthly_pivot = monthly_by_maker.pivot(index="month", columns="maker", values="value").fillna(0).sort_index()

    # KPI totals: prev / this / predicted next (global)
    prev_total = float(monthly_totals[monthly_totals.index.year == PREV_YEAR].sum()) if not monthly_totals.empty else float(yearly_pivot.loc[PREV_YEAR].sum() if PREV_YEAR in yearly_pivot.index else 0.0)
    this_total = float(monthly_totals[monthly_totals.index.year == THIS_YEAR].sum()) if not monthly_totals.empty else float(yearly_pivot.loc[THIS_YEAR].sum() if THIS_YEAR in yearly_pivot.index else 0.0)

    # Forecast overall monthly -> next year
    fc_monthly_overall = forecast_monthly_series(monthly_totals, periods=12)
    predicted_next_total = float(fc_monthly_overall[fc_monthly_overall.index.year == NEXT_YEAR]["yhat"].sum()) if fc_monthly_overall is not None else 0.0

    # Per-maker predicted allocation:
    # If per-maker history is sufficient, forecast each maker individually (preferred).
    maker_forecasts = {}
    for maker in monthly_pivot.columns:
        series = monthly_pivot[maker]
        fc = forecast_monthly_series(series, periods=12)
        if fc is not None:
            maker_forecasts[maker] = fc
    # For makers without individual forecast, distribute overall predicted_next_total proportional to historical share
    hist_share = totals_by_maker.set_index("maker")["value"] / (totals_by_maker["value"].sum() if totals_by_maker["value"].sum()>0 else 1)
    maker_predicted_next = {}
    for maker in totals_by_maker["maker"].tolist():
        if maker in maker_forecasts:
            pred = float(maker_forecasts[maker][maker_forecasts[maker].index.year == NEXT_YEAR]["yhat"].sum()) if not maker_forecasts[maker].empty else 0.0
            maker_predicted_next[maker] = pred
        else:
            maker_predicted_next[maker] = float(hist_share.get(maker, 0.0) * predicted_next_total)

    # totals by year (all categories)
    totals_by_year = df_mk.groupby("year", as_index=True)["value"].sum().sort_index()
    # ensure numeric defaults
    prev_total = float(prev_total or 0.0)
    this_total = float(this_total or 0.0)
    predicted_next_total = float(predicted_next_total or 0.0)

    # KPI comparisons
    kpi_prev_vs_this = pct_change(prev_total, this_total)
    kpi_this_vs_next = pct_change(this_total, predicted_next_total)

    # ------------------ UI: KPIs ------------------
    st.subheader("Top Makers — Real & Predicted (Maxed)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label=f"Prev Year ({PREV_YEAR})", value=f"{int(prev_total):,}", delta=f"{kpi_prev_vs_this:.2f}% vs prev")
    c2.metric(label=f"This Year ({THIS_YEAR})", value=f"{int(this_total):,}", delta=f"{kpi_this_vs_next:.2f}% vs this")
    c3.metric(label=f"Predicted Next Year ({NEXT_YEAR})", value=f"{int(predicted_next_total):,}", delta=f"{pct_change(prev_total, predicted_next_total):.2f}% vs prev")
    c4.metric(label="Overall (All-time)", value=f"{int(overall_total):,}")

    # ------------------ Top makers visuals ------------------
    st.markdown("### Top Makers — Current Ranking")
    top_n = st.slider("Top N makers to display", min_value=3, max_value=min(50, len(totals_by_maker)), value=min(10, len(totals_by_maker)))
    top_makers = totals_by_maker.head(top_n)

    # Bar chart (altair)
    bar = alt.Chart(top_makers).mark_bar().encode(
        x=alt.X("value:Q", title="Registrations"),
        y=alt.Y("maker:N", sort='-x', title="Maker"),
        tooltip=[alt.Tooltip("maker:N"), alt.Tooltip("value:Q", format=",")]
    ).properties(height=40*min(len(top_makers),20), width=700, title=f"Top {top_n} Makers (Total)")
    st.altair_chart(bar, use_container_width=True)

    # Donut (plotly)
    pie = px.pie(top_makers, names="maker", values="value", hole=0.45, title="Top Makers (Donut)")
    pie.update_traces(textinfo='percent+label', hoverinfo='label+value')
    st.plotly_chart(pie, use_container_width=True)

    # ------------------ Monthly trend (overall) ------------------
    st.markdown("### Monthly Registrations — Real vs Predicted (Overall)")
    real_ts = monthly_totals.reset_index().rename(columns={"month":"ds","value":"y"})
    if fc_monthly_overall is not None:
        fc_df = fc_monthly_overall.reset_index().rename(columns={"index":"ds"})
        fc_df["type"] = "predicted"
        real_df = real_ts.assign(type="real")
        plot_df = pd.concat([real_df.rename(columns={"ds":"month","y":"value"}).assign(type="real"),
                             fc_df.rename(columns={"ds":"month","yhat":"value"}).assign(type="predicted")], ignore_index=True, sort=False)
    else:
        plot_df = real_ts.rename(columns={"ds":"month","y":"value"}).assign(type="real")

    fig = px.line(plot_df, x="month", y="value", color="type", markers=True,
                  title="Monthly Registrations — Real vs Predicted (Overall)")
    fig.update_layout(xaxis_title="Month", yaxis_title="Registrations")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ Per-maker comparisons table ------------------
    st.markdown("### Maker-level: Prev vs This vs Predicted Next (detailed)")
    rows = []
    for maker in totals_by_maker["maker"].tolist():
        prev_v = float(yearly_pivot.loc[PREV_YEAR, maker]) if PREV_YEAR in yearly_pivot.index and maker in yearly_pivot.columns else 0.0
        this_v = float(yearly_pivot.loc[THIS_YEAR, maker]) if THIS_YEAR in yearly_pivot.index and maker in yearly_pivot.columns else 0.0
        pred_v = float(maker_predicted_next.get(maker, 0.0))
        rows.append({
            "maker": maker,
            f"{PREV_YEAR}": prev_v,
            f"{THIS_YEAR}": this_v,
            f"{NEXT_YEAR} (pred)": pred_v,
            "growth_prev_to_this %": pct_change(prev_v, this_v),
            "growth_this_to_next %": pct_change(this_v, pred_v),
            "historical_share %": float(hist_share.get(maker, 0.0))*100
        })
    maker_comp_df = pd.DataFrame(rows).sort_values(f"{THIS_YEAR}", ascending=False)
    st.dataframe(maker_comp_df.style.format({f"{PREV_YEAR}":"{:,}", f"{THIS_YEAR}":"{:,}", f"{NEXT_YEAR} (pred)":"{:,}", "growth_prev_to_this %":"{:.2f}%", "growth_this_to_next %":"{:.2f}%", "historical_share %":"{:.2f}%"}), height=400)

    # ------------------ Stacked monthly area by maker ------------------
    st.markdown("### Monthly Stacked by Maker (Top makers)")
    monthly_top = monthly_by_maker[monthly_by_maker["maker"].isin(top_makers["maker"].tolist())]
    fig_area = px.area(monthly_top.sort_values("month"), x="month", y="value", color="maker", title="Monthly Stacked by Maker (Top)")
    fig_area.update_layout(xaxis_title="Month", yaxis_title="Registrations")
    st.plotly_chart(fig_area, use_container_width=True)

    # ------------------ Heatmap: makers x year ------------------
    st.markdown("### Yearly Heatmap (Maker x Year)")
    heat_pivot = yearly_pivot.fillna(0)
    if not heat_pivot.empty:
        heat_map = px.imshow(heat_pivot.T, labels=dict(x="Year", y="Maker", color="Registrations"),
                             x=heat_pivot.index.astype(str).tolist(), y=heat_pivot.columns.tolist(),
                             aspect="auto", title="Yearly Heatmap (Maker x Year)")
        st.plotly_chart(heat_map, use_container_width=True)

    # ------------------ Growth buckets & alerts ------------------
    st.markdown("### Growth / Decline Buckets")
    maker_comp_df["bucket"] = maker_comp_df["growth_this_to_next %"].apply(lambda p: "High growth" if p>25 else ("Moderate growth" if p>5 else ("Stable" if -5<=p<=5 else "Decline")))
    buckets = maker_comp_df.groupby("bucket")["maker"].count().reset_index().rename(columns={"maker":"count"})
    st.table(buckets)

    # ------------------ Exports ------------------
    st.markdown("### Export Data & Charts")
    csv = df_mk.to_csv(index=False).encode("utf-8")
    st.download_button("Download raw makers CSV", csv, file_name=f"top_makers_raw_{THIS_YEAR}.csv", mime="text/csv")
    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_mk.to_excel(writer, sheet_name="raw", index=False)
            maker_comp_df.to_excel(writer, sheet_name="comparison", index=False)
            monthly_pivot.reset_index().to_excel(writer, sheet_name="monthly_by_maker", index=True)
            if fc_monthly_overall is not None:
                fc_monthly_overall.reset_index().to_excel(writer, sheet_name="forecast_overall", index=True)
            # individual maker forecasts (if any)
            for mk, fc in maker_forecasts.items():
                safe_name = str(mk)[:25]
                fc.reset_index().to_excel(writer, sheet_name=f"fc_{safe_name}", index=True)
        excel_buffer.seek(0)
        st.download_button("Download makers analysis Excel", excel_buffer, file_name=f"makers_analysis_{THIS_YEAR}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("Excel export failed. CSV available.")

    # ------------------ AI Narrative (optional) ------------------
    if enable_ai:
        with st.spinner("Generating AI narrative for Makers..."):
            system = "You are a senior market analyst. Provide a concise, data-driven commentary on maker market shares and forecasts."
            sample_rows = maker_comp_df.head(8).to_dict(orient="records")
            user_prompt = (
                f"Prev year total={int(prev_total):,}, this year total={int(this_total):,}, predicted next year total={int(predicted_next_total):,}."
                f"Top makers sample: {json.dumps(sample_rows, default=str)}"
                "Provide: (1) top 3 makers, (2) fastest growers, (3) makers in decline, (4) key risk signals, (5) 2 actionable recommendations, and (6) an executive 2-sentence summary."
            )
            ai_out = deepinfra_chat(system, user_prompt, max_tokens=360)
            if isinstance(ai_out, dict) and "text" in ai_out:
                st.markdown("**AI Narrative (Live)**")
                st.write(ai_out["text"])
            else:
                st.info("AI narrative unavailable or key missing.")

    # ------------------ Finalize ------------------
    st.markdown("---")
    st.success("✅ Top Makers analysis complete (real & predicted).")


# ============================================================
# 🚀 3️⃣ FULLY MAXED — YEARLY / MONTHLY / DAILY TREND + FORECASTS
# ============================================================
with st.spinner("📡 Fetching full registration trend data..."):
    tr_json = fetch_json("vahandashboard/vahanyearwiseregistrationtrend", desc="Registration Trend")

try:
    df_trend = normalize_trend(tr_json)
except Exception as e:
    st.error(f"❌ Trend parsing failed: {e}")
    df_trend = pd.DataFrame(columns=["date", "value"])

if not df_trend.empty:
    # 🧹 Basic cleanup
    df_trend = df_trend.sort_values("date")
    df_trend["year"] = df_trend["date"].dt.year
    df_trend["month"] = df_trend["date"].dt.month_name()
    df_trend["day"] = df_trend["date"].dt.day

    # ===============================
    # 📈 Charts: Real Values
    # ===============================
    st.subheader("📊 Registration Trends (All Views)")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Yearly Overview",
        "🗓️ Monthly Trends",
        "📆 Daily View",
        "🔮 Forecast & Prediction"
    ])

    # ========== Yearly ==========
    with tab1:
        yearly_df = df_trend.groupby("year", as_index=False)["value"].sum()
        yearly_df["YoY%"] = yearly_df["value"].pct_change() * 100
        st.bar_chart(yearly_df.set_index("year")["value"])
        st.dataframe(yearly_df.style.background_gradient("Blues"), use_container_width=True)

    # ========== Monthly ==========
    with tab2:
        monthly_df = df_trend.copy()
        monthly_df["ym"] = df_trend["date"].dt.to_period("M").astype(str)
        monthly_sum = monthly_df.groupby("ym", as_index=False)["value"].sum()
        st.line_chart(monthly_sum.set_index("ym")["value"])
        st.dataframe(monthly_sum.tail(12).style.background_gradient("Greens"), use_container_width=True)

    # ========== Daily ==========
    with tab3:
        st.area_chart(df_trend.set_index("date")["value"])
        st.dataframe(df_trend.tail(30).style.background_gradient("Oranges"), use_container_width=True)

    # ===============================
    # 🔮 Forecasting (Next Year)
    # ===============================
    with tab4:
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np

            df_pred = df_trend.copy()
            df_pred["t"] = np.arange(len(df_pred))
            model = LinearRegression().fit(df_pred[["t"]], df_pred["value"])
            future_t = np.arange(len(df_pred), len(df_pred) + 365)
            future_preds = model.predict(future_t.reshape(-1, 1))

            df_future = pd.DataFrame({
                "date": pd.date_range(df_pred["date"].max() + pd.Timedelta(days=1), periods=365),
                "predicted": future_preds
            })

            # Combine real + predicted
            df_all = pd.concat([
                df_pred.rename(columns={"value": "actual"})[["date", "actual"]],
                df_future
            ], ignore_index=True)

            st.line_chart(df_all.set_index("date"))

            # 📊 Metrics
            total_pred = int(df_future["predicted"].sum())
            avg_pred = int(df_future["predicted"].mean())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Total (Next Year)", f"{total_pred:,}")
            with col2:
                st.metric("Predicted Daily Avg (Next Year)", f"{avg_pred:,}")

            st.dataframe(df_future.head(15).style.background_gradient("Purples"), use_container_width=True)
        except Exception as e:
            st.warning(f"Forecasting failed: {e}")

    # ===============================
    # 🧾 YoY / QoQ / MoM / CAGR
    # ===============================
    st.subheader("📉 Comparative Growth Analysis")
    yoy_df = compute_yoy(df_trend)
    qoq_df = compute_qoq(df_trend)

    latest_yoy = yoy_df["YoY%"].dropna().iloc[-1] if not yoy_df.empty else None
    latest_qoq = qoq_df["QoQ%"].dropna().iloc[-1] if not qoq_df.empty else None

    total_val = int(df_trend["value"].sum())
    avg_daily = df_trend["value"].mean()
    avg_monthly = df_trend.groupby(df_trend["date"].dt.to_period("M"))["value"].sum().mean()

    colK1, colK2, colK3, colK4, colK5 = st.columns(5)
    with colK1:
        st.metric("Total Registrations", f"{total_val:,}")
    with colK2:
        st.metric("Daily Avg", f"{avg_daily:,.0f}")
    with colK3:
        st.metric("Monthly Avg", f"{avg_monthly:,.0f}")
    with colK4:
        st.metric("YoY Growth%", f"{latest_yoy:.2f}%" if latest_yoy else "N/A")
    with colK5:
        st.metric("QoQ Growth%", f"{latest_qoq:.2f}%" if latest_qoq else "N/A")

    st.markdown("---")
    st.dataframe(yoy_df.tail(5).style.background_gradient("coolwarm"), use_container_width=True)
    st.dataframe(qoq_df.tail(5).style.background_gradient("coolwarm"), use_container_width=True)

    # ===============================
    # 🧠 AI-Driven Narrative + Insights
    # ===============================
    if enable_ai:
        with st.spinner("🤖 Generating full AI-powered insight report..."):
            system = (
                "You are an expert automotive analyst. Use YoY, QoQ, MoM, and forecast data "
                "to give an advanced, data-driven narrative including anomalies, growth, "
                "comparisons between previous/current/next years, and 3 strategic recommendations."
            )

            sample_data = df_trend.tail(24).to_dict(orient="records")
            user = (
                f"Dataset sample: {json.dumps(sample_data, default=str)}\n"
                f"YoY: {latest_yoy}, QoQ: {latest_qoq}, DailyAvg: {avg_daily}, Forecast avg: {avg_pred}\n"
                "Generate an advanced but readable report with trend analysis, growth drivers, "
                "potential risks, and predictions for next year."
            )

            ai_resp = deepinfra_chat(system, user, max_tokens=900)
            if isinstance(ai_resp, dict) and "text" in ai_resp:
                st.markdown("### 🧠 AI Trend Report")
                st.info(ai_resp["text"])
            else:
                st.info("⚠️ No AI response received.")

else:
    st.warning("⚠️ No trend data available.")

# =====================================================
# 4️⃣ MAXED — Duration-wise Growth (Monthly / Quarterly / Yearly)
# =====================================================
import io
import math
import json
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
import streamlit as st
from datetime import date, datetime
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# -------------------------
# Helper: normalize whatever parse_duration_table returns
# Expected canonical output: columns ['duration_label','period','value']
# period should be a datetime-like (monthly/quart/annual as timestamp)
# -------------------------
def normalize_duration_df(df, calendar_type):
    """
    calendar_type: 3=Monthly, 2=Quarterly, 1=Yearly (per your API)
    Attempts to return DataFrame with columns:
      - duration_label (str): bucket or label
      - period (datetime): timestamp to aggregate by (for monthly/quart/yr)
      - value (numeric)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["duration_label","period","value"])
    d = df.copy()
    cols_lower = {c.lower(): c for c in d.columns}

    # Find candidate label column
    label_candidates = [cols_lower.get(k) for k in ("duration","label","bucket","name") if k in cols_lower]
    value_candidates = [cols_lower.get(k) for k in ("value","count","registrations","total","y") if k in cols_lower]
    period_candidates = [cols_lower.get(k) for k in ("period","date","month","year","ds") if k in cols_lower]

    label_col = label_candidates[0] if label_candidates else (d.columns[0] if len(d.columns)>0 else None)
    value_col = value_candidates[0] if value_candidates else (d.select_dtypes(include=[np.number]).columns[0] if len(d.select_dtypes(include=[np.number]).columns)>0 else None)
    period_col = period_candidates[0] if period_candidates else None

    if label_col:
        d = d.rename(columns={label_col: "duration_label"})
    else:
        d["duration_label"] = "all"

    if value_col:
        d = d.rename(columns={value_col: "value"})
    else:
        d["value"] = 0

    if period_col:
        d = d.rename(columns={period_col: "period"})
        # coerce to datetime
        try:
            d["period"] = pd.to_datetime(d["period"], errors="coerce")
        except Exception:
            d["period"] = pd.to_datetime(d["period"].astype(str), errors="coerce")
    else:
        # create synthetic period depending on calendar_type
        default_ts = datetime(THIS_YEAR, 1, 1)
        d["period"] = pd.to_datetime(d.get("period", default_ts))

    d["duration_label"] = d["duration_label"].astype(str)
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(0)
    return d[["duration_label","period","value"]]

# -------------------------
# Forecast helper (prophet preferred, linear fallback)
# Input: series indexed by timestamp (monthly/quartly/yearly) returning forecast df
# -------------------------
def forecast_series(s: pd.Series, periods: int, freq: str = "MS"):
    """
    s: pd.Series indexed by pd.Timestamp
    periods: number of future periods to forecast
    freq: frequency string for future (e.g., 'MS' monthly start, 'QS' quarterly start, 'YS' yearly start)
    returns DataFrame index=future timestamps with ['yhat','yhat_lower','yhat_upper'] or None on failure
    """
    try:
        s = s.dropna().sort_index()
        if len(s) < 6:
            return None
        # Prophet
        if PROPHET_AVAILABLE and len(s) >= 12:
            from prophet import Prophet
            dfp = s.reset_index().rename(columns={s.index.name or 'index':'ds', 0:'y'}) if isinstance(s, pd.Series) else s.reset_index().rename(columns={s.index.name or 'index':'ds', s.name:'y'})
            dfp.columns = ['ds','y']
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=periods, freq=freq)
            fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].set_index("ds")
            return fc.tail(periods)
        else:
            # Linear regression fallback
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(s)).reshape(-1,1)
            y = s.values
            lr = LinearRegression()
            lr.fit(X,y)
            future_X = np.arange(len(s), len(s)+periods).reshape(-1,1)
            yhat = lr.predict(future_X)
            last_ts = s.index.max()
            next_periods = pd.date_range(start=(pd.to_datetime(last_ts) + pd.offsets.DateOffset(**({'months':1} if freq.startswith('M') else {'quarters':1} if freq.startswith('Q') else {'years':1}))), periods=periods, freq=freq)
            fc = pd.DataFrame({
                "yhat": np.maximum(yhat, 0),
                "yhat_lower": np.maximum(yhat * 0.85, 0),
                "yhat_upper": np.maximum(yhat * 1.15, 0)
            }, index=next_periods)
            return fc
    except Exception:
        return None

# -------------------------
# Single unified function that fetches and runs analysis for a single calendar type
# -------------------------
def analyze_duration(calendar_type: int, label: str, forecast_periods: int = 12):
    """
    calendar_type: 3 = monthly, 2 = quarterly, 1 = yearly
    label: display label
    forecast_periods: months/quarters/years to forecast (choose 12/4/1 respectively)
    """
    with st.spinner(f"Fetching {label} duration-wise growth..."):
        json_data = fetch_json("vahandashboard/durationWiseRegistrationTable", {**params_common, "calendarType": calendar_type}, desc=f"{label} growth")
        raw_df = parse_duration_table(json_data) if json_data else pd.DataFrame()
    df = normalize_duration_df(raw_df, calendar_type)
    if df.empty:
        st.warning(f"No {label} duration data found.")
        return pd.DataFrame()

    st.subheader(f"{label} Vehicle Registration Growth — Maxed Analysis")

    # period normalization depending on calendar type
    if calendar_type == 3:  # monthly
        df["period_ts"] = df["period"].dt.to_period("M").dt.to_timestamp()
        freq = "MS"
        periods = forecast_periods  # months
    elif calendar_type == 2:  # quarterly
        df["period_ts"] = df["period"].dt.to_period("Q").dt.to_timestamp()
        freq = "QS"
        periods = max(4, min(forecast_periods, 8))
    else:  # yearly
        df["period_ts"] = df["period"].dt.to_period("Y").dt.to_timestamp()
        freq = "YS"
        periods = 1 if forecast_periods < 4 else forecast_periods // 12

    # aggregate to period x duration_label
    pivot = df.groupby(["period_ts","duration_label"], as_index=False)["value"].sum()
    pivot_full = pivot.pivot(index="period_ts", columns="duration_label", values="value").fillna(0).sort_index()

    # totals series (all durations)
    totals_series = pivot_full.sum(axis=1).rename("total")

    # per-duration totals (historical)
    total_by_duration = df.groupby("duration_label", as_index=False)["value"].sum().sort_values("value", ascending=False)

    # historical prev/this totals for this calendar perspective (by year)
    # compute total per-year from df
    df["year"] = df["period"].dt.year
    totals_by_year = df.groupby("year", as_index=True)["value"].sum().sort_index()
    prev_total = float(totals_by_year.get(PREV_YEAR, 0.0))
    this_total = float(totals_by_year.get(THIS_YEAR, 0.0))

    # Forecast totals by forecasting the totals_series
    fc = forecast_series(totals_series, periods=periods, freq=freq)
    predicted_next_total = 0.0
    if fc is not None:
        # Sum predicted values that fall into NEXT_YEAR depending on freq
        predicted_next_total = float(fc[fc.index.year == NEXT_YEAR]["yhat"].sum()) if any(fc.index.year == NEXT_YEAR) else float(fc["yhat"].sum())

    # For per-duration predictions: try forecast per-duration if enough history else allocate proportionally
    duration_forecasts = {}
    for dur in pivot_full.columns:
        series = pivot_full[dur]
        dfc = forecast_series(series, periods=periods, freq=freq)
        duration_forecasts[dur] = dfc

    # Where forecasts missing, allocate using historical shares (last available year)
    last_hist = total_by_duration.set_index("duration_label")["value"]
    total_hist_sum = last_hist.sum() if last_hist.sum() > 0 else 1.0
    hist_share = (last_hist / total_hist_sum).to_dict()

    predicted_per_duration = {}
    for dur in pivot_full.columns:
        if duration_forecasts.get(dur) is not None:
            dff = duration_forecasts[dur]
            if any(dff.index.year == NEXT_YEAR):
                predicted_per_duration[dur] = float(dff[dff.index.year == NEXT_YEAR]["yhat"].sum())
            else:
                predicted_per_duration[dur] = float(dff["yhat"].sum())
        else:
            predicted_per_duration[dur] = float(hist_share.get(dur, 0.0) * predicted_next_total)

    # KPI calculations
    def pct(a,b):
        try:
            if a == 0:
                return float("inf") if b!=0 else 0.0
            return ((b-a)/abs(a))*100.0
        except Exception:
            return 0.0

    kpi_prev_vs_this = pct(prev_total, this_total)
    kpi_this_vs_next = pct(this_total, predicted_next_total)

    # -----------------------
    # UI: KPIs & Top buckets
    # -----------------------
    k1,k2,k3,k4 = st.columns(4)
    k1.metric(f"{label} Prev Year ({PREV_YEAR})", f"{int(prev_total):,}", delta=f"{kpi_prev_vs_this:.2f}%")
    k2.metric(f"{label} This Year ({THIS_YEAR})", f"{int(this_total):,}", delta=f"{kpi_this_vs_next:.2f}%")
    k3.metric(f"{label} Predicted Next ({NEXT_YEAR})", f"{int(predicted_next_total):,}", delta=f"{pct(prev_total,predicted_next_total):.2f}%")
    k4.metric("Historical Total (all-time)", f"{int(total_by_duration['value'].sum()):,}")

    st.markdown("#### Top duration buckets (historical)")
    st.dataframe(total_by_duration.head(20).style.format({"value":"{:,}"}), use_container_width=True)

    # -----------------------
    # Charts
    # -----------------------
    st.markdown("### Trend: totals over time (real + prediction ribbon)")
    # Build plot df
    real_df = totals_series.reset_index().rename(columns={"period_ts":"period","total":"value"})
    real_df["type"] = "actual"
    plot_df = real_df.copy()
    if fc is not None:
        fc_df = fc.reset_index().rename(columns={"index":"period","yhat":"value","yhat_lower":"lower","yhat_upper":"upper"})
        fc_df["type"] = "predicted"
        # plotly with ribbon
        fig = px.line(pd.concat([real_df.assign(type="actual"), fc_df.assign(type="predicted")], sort=False),
                      x="period", y="value", color="type", title=f"{label} — Total registrations: Real vs Predicted")
        # ribbon
        fig.add_traces([dict(x=fc_df["period"], y=fc_df["upper"], mode='lines', showlegend=False, line=dict(width=0)),
                        dict(x=fc_df["period"], y=fc_df["lower"], mode='lines', fill='tonexty', fillcolor='rgba(0,176,246,0.15)', line=dict(width=0), showlegend=False)])
    else:
        fig = px.line(real_df, x="period", y="value", title=f"{label} — Total registrations (historical)")

    fig.update_layout(xaxis_title="Period", yaxis_title="Registrations", legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    # Per-duration stacked area
    st.markdown("### Stacked by duration bucket")
    stacked_df = pivot.reset_index().rename(columns={"period_ts":"period"})
    if not stacked_df.empty:
        fig2 = px.area(stacked_df.sort_values("period"), x="period", y=[c for c in pivot_full.columns], title=f"{label} — Stacked by Duration Bucket")
        fig2.update_layout(xaxis_title="Period", yaxis_title="Registrations")
        st.plotly_chart(fig2, use_container_width=True)

    # Per-duration comparison table: prev / this / predicted next
    st.markdown("### Per-duration comparison: Prev vs This vs Predicted Next")
    rows = []
    # Build per-duration historical year totals
    per_dur_year = df.groupby(["year","duration_label"], as_index=False)["value"].sum().pivot(index="duration_label", columns="year", values="value").fillna(0)
    for dur in pivot_full.columns:
        prev_v = float(per_dur_year.get(PREV_YEAR, {}).get(dur, per_dur_year.loc[dur][PREV_YEAR] if PREV_YEAR in per_dur_year.columns and dur in per_dur_year.index else 0.0)) if dur in per_dur_year.index else 0.0
        this_v = float(per_dur_year.get(THIS_YEAR, {}).get(dur, per_dur_year.loc[dur][THIS_YEAR] if THIS_YEAR in per_dur_year.columns and dur in per_dur_year.index else 0.0)) if dur in per_dur_year.index else 0.0
        pred_v = float(predicted_per_duration.get(dur, 0.0))
        rows.append({
            "duration_label": dur,
            f"{PREV_YEAR}": prev_v,
            f"{THIS_YEAR}": this_v,
            f"{NEXT_YEAR} (pred)": pred_v,
            "growth_prev_to_this %": pct(prev_v, this_v),
            "growth_this_to_next %": pct(this_v, pred_v)
        })
    comp_df = pd.DataFrame(rows).sort_values(f"{THIS_YEAR}", ascending=False)
    st.dataframe(comp_df.style.format({f"{PREV_YEAR}":"{:,}", f"{THIS_YEAR}":"{:,}", f"{NEXT_YEAR} (pred)":"{:,}", "growth_prev_to_this %":"{:.2f}%", "growth_this_to_next %":"{:.2f}%"}), use_container_width=True)

    # Heatmap duration x year
    st.markdown("### Heatmap: Duration bucket x Year")
    if not per_dur_year.empty:
        heat = per_dur_year.fillna(0)
        # reorder rows by total
        heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
        fig_heat = px.imshow(heat, labels=dict(x="Year", y="Duration Bucket", color="Registrations"),
                             x=[str(x) for x in heat.columns.tolist()], y=heat.index.tolist(), aspect="auto",
                             title=f"{label} — Duration buckets across years")
        st.plotly_chart(fig_heat, use_container_width=True)

    # Anomalies: detect large month-over-month or quarter-over-quarter jumps per duration
    st.markdown("### Anomaly & Change Detection")
    anomalies = []
    for dur in pivot_full.columns:
        s = pivot_full[dur].sort_index()
        if len(s) >= 3:
            mom = s.pct_change().fillna(0)
            large = mom[mom.abs() > 0.5]  # >50% change flagged
            for idx,v in large.items():
                anomalies.append({"duration": dur, "period": idx, "mom_change_pct": v*100, "value": float(s.loc[idx])})
    if anomalies:
        anom_df = pd.DataFrame(anomalies).sort_values("period", ascending=False)
        st.table(anom_df.head(20).assign(period=lambda df_: df_["period"].dt.strftime("%Y-%m-%d")))
    else:
        st.info("No large anomalies detected (threshold: 50% MoM/QtQ)")

    # Exports
    st.markdown("### Exports")
    raw_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"Download raw {label} CSV", raw_csv, file_name=f"duration_{label.lower()}_raw.csv", mime="text/csv")
    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="raw", index=False)
            pivot_full.reset_index().to_excel(writer, sheet_name="pivot", index=True)
            comp_df.to_excel(writer, sheet_name="comparison", index=False)
            if fc is not None:
                fc.reset_index().to_excel(writer, sheet_name="forecast_totals", index=True)
            # per-duration forecasts
            for dur, dfc in duration_forecasts.items():
                if dfc is not None:
                    safe = str(dur)[:28]
                    dfc.reset_index().to_excel(writer, sheet_name=f"fc_{safe}", index=True)
        excel_buffer.seek(0)
        st.download_button(f"Download {label} Excel", excel_buffer, file_name=f"duration_{label.lower()}_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("Excel export failed (openpyxl missing). CSV available.")

    # AI summary
    if enable_ai:
        with st.spinner("Generating AI summary for duration growth..."):
            system = "You are a data analyst. Summarize the duration-wise growth results, with emphasis on which duration buckets are growing/falling, anomaly signals, and recommended actions."
            sample = comp_df.head(8).to_dict(orient="records")
            user = f"Summary sample: {json.dumps(sample, default=str)}\nPrev total: {int(prev_total):,}, This total: {int(this_total):,}, Predicted next: {int(predicted_next_total):,}. Provide 6 bullet points: top 3 buckets, fastest growing, biggest decline, anomaly signals, one operational recommendation, short executive summary."
            ai_out = deepinfra_chat(system, user, max_tokens=400)
            if isinstance(ai_out, dict) and "text" in ai_out:
                st.markdown("**AI Duration Narrative**")
                st.write(ai_out["text"])
            else:
                st.info("AI narrative unavailable or key missing.")

    st.markdown("---")
    st.success(f"✅ {label} Duration analysis complete.")
    return df

# -------------------------
# Run for Monthly / Quarterly / Yearly
# -------------------------
df_monthly = analyze_duration(3, "Monthly", forecast_periods=12)
df_quarterly = analyze_duration(2, "Quarterly", forecast_periods=8)
df_yearly = analyze_duration(1, "Yearly", forecast_periods=2)

# =====================================================
# 5️⃣ MAXED — Top 5 Revenue States (Real + Predicted + Full Analysis)
# =====================================================
import io
import math
import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
from datetime import date, datetime
from zoneinfo import ZoneInfo
import streamlit as st
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# -------------------------
# Local forecast helper (Prophet preferred, linear fallback)
# -------------------------
def forecast_series_local(series: pd.Series, periods:int=12, freq:str="MS"):
    """
    series: pd.Series indexed by timestamps (monthly) or yearly timestamps
    returns DataFrame with index=future timestamps and columns ['yhat','yhat_lower','yhat_upper'] or None
    """
    try:
        s = series.dropna().sort_index()
        if len(s) < 6:
            return None
        if PROPHET_AVAILABLE and len(s) >= 12:
            from prophet import Prophet
            dfp = s.reset_index().rename(columns={s.index.name or 'index':'ds', s.name:'y'})
            dfp.columns = ['ds','y']
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=periods, freq=freq)
            fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].set_index("ds")
            return fc.tail(periods)
        else:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(s)).reshape(-1,1)
            y = s.values
            lr = LinearRegression().fit(X,y)
            future_X = np.arange(len(s), len(s)+periods).reshape(-1,1)
            yhat = lr.predict(future_X)
            last_ts = s.index.max()
            # build next timestamps using freq
            next_idx = pd.date_range(start=(pd.to_datetime(last_ts) + pd.offsets.MonthBegin(1)).replace(day=1), periods=periods, freq=freq)
            fc = pd.DataFrame({
                "yhat": np.maximum(yhat, 0),
                "yhat_lower": np.maximum(yhat*0.85, 0),
                "yhat_upper": np.maximum(yhat*1.15, 0),
            }, index=next_idx)
            return fc
    except Exception:
        return None

def pct_change(a,b):
    try:
        if a == 0:
            return float("inf") if b != 0 else 0.0
        return ((b - a)/abs(a))*100.0
    except Exception:
        return 0.0

# ------------------ Fetch & parse ------------------
with st.spinner("Fetching Top 5 Revenue States (maxed)..."):
    top5_rev_json = fetch_json("vahandashboard/top5chartRevenueFee", params_common, desc="Top 5 Revenue States")
    raw_rev = parse_top5_revenue(top5_rev_json if top5_rev_json else {})

# Normalize parse_top5_revenue output: try to produce ['state','date','revenue'] or ['state','value']
def normalize_revenue_df(df):
    if df is None:
        return pd.DataFrame(columns=["state","date","revenue"])
    d = df.copy()
    cols = {c.lower():c for c in d.columns}
    # find state
    state_col = cols.get("state") or cols.get("state_name") or cols.get("label") or (d.columns[0] if len(d.columns)>0 else None)
    # find revenue/value
    rev_col = cols.get("revenue") or cols.get("value") or cols.get("amount") or cols.get("fee") or (d.select_dtypes(include=[np.number]).columns[0] if d.select_dtypes(include=[np.number]).shape[1]>0 else None)
    # find date
    date_col = cols.get("date") or cols.get("period") or cols.get("month") or cols.get("year")
    if state_col:
        d = d.rename(columns={state_col:"state"})
    else:
        d["state"] = "Unknown"
    if rev_col:
        d = d.rename(columns={rev_col:"revenue"})
    else:
        d["revenue"] = 0.0
    if date_col:
        d = d.rename(columns={date_col:"date"})
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    else:
        # if no date, assume THIS_YEAR snapshot
        d["date"] = pd.to_datetime(datetime(THIS_YEAR,1,1))
    d["state"] = d["state"].astype(str)
    d["revenue"] = pd.to_numeric(d["revenue"], errors="coerce").fillna(0.0)
    return d[["state","date","revenue"]]

df_rev = normalize_revenue_df(raw_rev)

if df_rev.empty:
    st.warning("No revenue data available.")
else:
    # Time buckets
    df_rev["month"] = df_rev["date"].dt.to_period("M").dt.to_timestamp()
    df_rev["year"] = df_rev["date"].dt.year
    df_rev["day"] = df_rev["date"].dt.floor("D")

    # Top states by historical total
    totals_by_state = df_rev.groupby("state", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    top_states = totals_by_state.head(10)
    overall_revenue = totals_by_state["revenue"].sum()

    # Monthly totals (all-states)
    monthly_totals = df_rev.groupby("month", as_index=True)["revenue"].sum().sort_index()

    # Yearly per-state pivot
    yearly = df_rev.groupby(["year","state"], as_index=False)["revenue"].sum()
    yearly_pivot = yearly.pivot(index="year", columns="state", values="revenue").fillna(0).sort_index()

    # Prev / This totals (by year)
    prev_total = float(yearly_pivot.loc[PREV_YEAR].sum()) if PREV_YEAR in yearly_pivot.index else float(monthly_totals[monthly_totals.index.year==PREV_YEAR].sum() if not monthly_totals.empty else 0.0)
    this_total = float(yearly_pivot.loc[THIS_YEAR].sum()) if THIS_YEAR in yearly_pivot.index else float(monthly_totals[monthly_totals.index.year==THIS_YEAR].sum() if not monthly_totals.empty else 0.0)

    # Forecast overall monthly -> next year
    fc_overall = forecast_series_local(monthly_totals, periods=12, freq="MS")
    predicted_next_total = float(fc_overall[fc_overall.index.year==NEXT_YEAR]["yhat"].sum()) if fc_overall is not None and any(fc_overall.index.year==NEXT_YEAR) else (float(fc_overall["yhat"].sum()) if fc_overall is not None else 0.0)

    # Per-state monthly pivot and per-state forecasts where data allows
    monthly_by_state = df_rev.groupby(["month","state"], as_index=False)["revenue"].sum()
    monthly_pivot = monthly_by_state.pivot(index="month", columns="state", values="revenue").fillna(0).sort_index()

    state_forecasts = {}
    for st_name in monthly_pivot.columns:
        series = monthly_pivot[st_name]
        fc = forecast_series_local(series, periods=12, freq="MS")
        state_forecasts[st_name] = fc

    # Where per-state forecasts missing, distribute overall predicted next total by historical share
    hist_share = totals_by_state.set_index("state")["revenue"] / (totals_by_state["revenue"].sum() if totals_by_state["revenue"].sum()>0 else 1.0)
    predicted_per_state = {}
    for sname in totals_by_state["state"].tolist():
        if state_forecasts.get(sname) is not None and not state_forecasts[sname].empty:
            dfc = state_forecasts[sname]
            predicted_per_state[sname] = float(dfc[dfc.index.year==NEXT_YEAR]["yhat"].sum()) if any(dfc.index.year==NEXT_YEAR) else float(dfc["yhat"].sum())
        else:
            predicted_per_state[sname] = float(hist_share.get(sname, 0.0) * predicted_next_total)

    # KPIs & comparisons
    kpi_prev_vs_this = pct_change(prev_total, this_total)
    kpi_this_vs_next = pct_change(this_total, predicted_next_total)

    st.subheader("Top 5 Revenue States — Real & Predicted (Maxed)")
    a,b,c,d = st.columns(4)
    a.metric(f"Prev Year Revenue ({PREV_YEAR})", f"₹{int(prev_total):,}", delta=f"{kpi_prev_vs_this:.2f}%")
    b.metric(f"This Year Revenue ({THIS_YEAR})", f"₹{int(this_total):,}", delta=f"{kpi_this_vs_next:.2f}%")
    c.metric(f"Predicted Next Year Revenue ({NEXT_YEAR})", f"₹{int(predicted_next_total):,}", delta=f"{pct_change(prev_total, predicted_next_total):.2f}%")
    d.metric("Historical Total (all-time)", f"₹{int(overall_revenue):,}")

    # Top 5 specific view
    top5 = totals_by_state.head(5)
    st.markdown("### Top 5 Revenue States — Current Ranking")
    bar = alt.Chart(top5.rename(columns={"state":"label","revenue":"value"})).mark_bar().encode(
        x=alt.X("value:Q", title="Revenue (₹)"),
        y=alt.Y("label:N", sort='-x', title="State"),
        tooltip=[alt.Tooltip("label:N"), alt.Tooltip("value:Q", format=",")]
    ).properties(height=40*min(len(top5),10), width=700, title="Top 5 Revenue States (Total)")
    st.altair_chart(bar, use_container_width=True)

    pie = px.pie(top5, names="state", values="revenue", hole=0.45, title="Top 5 Revenue States (Donut)")
    pie.update_traces(textinfo='percent+label', hoverinfo='label+value')
    st.plotly_chart(pie, use_container_width=True)

    # Monthly trend with predicted ribbon
    st.markdown("### Monthly Revenue — Real vs Predicted")
    real_ts = monthly_totals.reset_index().rename(columns={"month":"ds","revenue":"y"}) if not monthly_totals.empty else pd.DataFrame(columns=["ds","y"])
    if fc_overall is not None:
        fc_df = fc_overall.reset_index().rename(columns={"index":"ds"})
        plot_df = pd.concat([real_ts.rename(columns={"ds":"month","y":"value"}).assign(type="actual"), fc_df.rename(columns={"ds":"month","yhat":"value"}).assign(type="predicted")], ignore_index=True, sort=False)
        fig = px.line(plot_df, x="month", y="value", color="type", markers=True, title="Monthly Revenue — Real vs Predicted")
        # ribbon
        fig.add_traces([dict(x=fc_df["ds"], y=fc_df["yhat_upper"], mode='lines', line=dict(width=0), showlegend=False),
                        dict(x=fc_df["ds"], y=fc_df["yhat_lower"], mode='lines', fill='tonexty', fillcolor='rgba(0,176,246,0.15)', line=dict(width=0), showlegend=False)])
    else:
        fig = px.line(real_ts.rename(columns={"ds":"month","y":"value"}), x="month", y="value", title="Monthly Revenue (Historical)")
    fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (₹)", legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    # Per-state comparison table (prev / this / predicted)
    st.markdown("### Per-State: Prev vs This vs Predicted Next")
    rows = []
    states_all = totals_by_state["state"].tolist()
    per_state_year = df_rev.groupby(["year","state"], as_index=False)["revenue"].sum().pivot(index="state", columns="year", values="revenue").fillna(0)
    for sname in states_all:
        prev_v = float(per_state_year.loc[sname][PREV_YEAR]) if PREV_YEAR in per_state_year.columns and sname in per_state_year.index else 0.0
        this_v = float(per_state_year.loc[sname][THIS_YEAR]) if THIS_YEAR in per_state_year.columns and sname in per_state_year.index else 0.0
        pred_v = float(predicted_per_state.get(sname, 0.0))
        rows.append({
            "state": sname,
            f"{PREV_YEAR} (₹)": prev_v,
            f"{THIS_YEAR} (₹)": this_v,
            f"{NEXT_YEAR} (pred ₹)": pred_v,
            "growth_prev_to_this %": pct_change(prev_v, this_v),
            "growth_this_to_next %": pct_change(this_v, pred_v),
            "historical_share %": float(hist_share.get(sname, 0.0))*100
        })
    comp_df = pd.DataFrame(rows).sort_values(f"{THIS_YEAR} (₹)", ascending=False)
    st.dataframe(comp_df.style.format({f"{PREV_YEAR} (₹)":"₹{:,}", f"{THIS_YEAR} (₹)":"₹{:,}", f"{NEXT_YEAR} (pred ₹)":"₹{:,}", "growth_prev_to_this %":"{:.2f}%", "growth_this_to_next %":"{:.2f}%","historical_share %":"{:.2f}%"}), height=420)

    # Heatmap: state x year revenue
    st.markdown("### Yearly Heatmap — State x Year Revenue")
    heat = yearly_pivot.fillna(0)
    if not heat.empty:
        heat = heat[sorted(heat.columns, key=lambda s: -heat.loc[:,s].sum())]  # sort cols by total desc
        fig_heat = px.imshow(heat.T, labels=dict(x="Year", y="State", color="Revenue (₹)"),
                             x=heat.index.astype(str).tolist(), y=heat.columns.tolist(), aspect="auto", title="State x Year Revenue Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

    # Anomalies by state (large YoY swings)
    st.markdown("### Revenue Anomalies & Alerts")
    anomalies = []
    for sname in monthly_pivot.columns:
        s = monthly_pivot[sname].sort_index()
        if len(s) >= 6:
            mom = s.pct_change().fillna(0)
            large = mom[mom.abs() > 0.5]
            for idx,v in large.items():
                anomalies.append({"state": sname, "period": idx, "mom_change_pct": v*100, "revenue": float(s.loc[idx])})
    if anomalies:
        anom_df = pd.DataFrame(anomalies).sort_values("period", ascending=False)
        st.table(anom_df.head(20).assign(period=lambda df_: df_["period"].dt.strftime("%Y-%m-%d")))
    else:
        st.info("No large month-over-month revenue anomalies detected (threshold: 50%).")

    # Exports
    st.markdown("### Export Data & Charts")
    csv = df_rev.to_csv(index=False).encode("utf-8")
    st.download_button("Download raw revenue CSV", csv, file_name=f"top5_revenue_raw_{THIS_YEAR}.csv", mime="text/csv")
    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_rev.to_excel(writer, sheet_name="raw", index=False)
            comp_df.to_excel(writer, sheet_name="per_state_comparison", index=False)
            monthly_pivot.reset_index().to_excel(writer, sheet_name="monthly_by_state", index=True)
            if fc_overall is not None:
                fc_overall.reset_index().to_excel(writer, sheet_name="forecast_overall", index=True)
            for sname, fc in state_forecasts.items():
                if fc is not None:
                    safe = str(sname)[:28]
                    fc.reset_index().to_excel(writer, sheet_name=f"fc_{safe}", index=True)
        excel_buffer.seek(0)
        st.download_button("Download revenue analysis Excel", excel_buffer, file_name=f"revenue_analysis_{THIS_YEAR}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("Excel export failed; CSV available.")

    # AI summary (DeepInfra)
    if enable_ai:
        with st.spinner("Generating AI revenue narrative..."):
            system = "You are a financial analyst. Summarize the revenue patterns, top states, growth/decline, predicted next-year revenue and strategic recommendations."
            sample = comp_df.head(8).to_dict(orient="records")
            user = f"Prev total={int(prev_total):,}, this total={int(this_total):,}, predicted_next={int(predicted_next_total):,}. Sample per-state rows: {json.dumps(sample, default=str)}. Provide 6 bullets: top states, fastest growth, decline signals, next-year forecast risk, 2 recommendations, 2-sentence exec summary."
            ai_out = deepinfra_chat(system, user, max_tokens=360)
            if isinstance(ai_out, dict) and "text" in ai_out:
                st.markdown("**AI Revenue Narrative (Live)**")
                st.write(ai_out["text"])
            else:
                st.info("AI narrative unavailable or key missing.")

    st.markdown("---")
    st.success("✅ Top 5 Revenue States analysis complete (real & predicted).")

# =====================================================
# 6️⃣ MAXED — Revenue Trend (Real + Predicted + Full Analysis)
# =====================================================
import io
import math
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import statsmodels.api as sm
from datetime import date, datetime
from zoneinfo import ZoneInfo
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# ----------------- Fetch & normalize -----------------
with st.spinner("Fetching Revenue Trend (maxed)..."):
    rev_trend_json = fetch_json("vahandashboard/revenueFeeLineChart", params_common, desc="Revenue Trend")
    df_rev_trend = parse_revenue_trend(rev_trend_json if rev_trend_json else {})

# safe-normalize: expect columns like ['period','value','year','month']
def normalize_rev_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["period","value"])
    d = df.copy()
    # try to find period & value
    cols = {c.lower(): c for c in d.columns}
    period_col = cols.get("period") or cols.get("date") or cols.get("month") or cols.get("ds")
    value_col = cols.get("value") or cols.get("revenue") or cols.get("amount") or (d.select_dtypes(include=[np.number]).columns[0] if d.select_dtypes(include=[np.number]).shape[1]>0 else None)
    if period_col:
        d = d.rename(columns={period_col: "period"})
        d["period"] = pd.to_datetime(d["period"], errors="coerce")
    else:
        d["period"] = pd.to_datetime(d.get("period", pd.Timestamp(datetime(THIS_YEAR,1,1))))
    if value_col:
        d = d.rename(columns={value_col: "value"})
    else:
        d["value"] = 0.0
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(0.0)
    d = d.sort_values("period").reset_index(drop=True)
    return d[["period","value"]]

df_rev = normalize_rev_df(df_rev_trend)

if df_rev.empty:
    st.warning("No revenue trend data available.")
else:
    # ---------------- basic aggregates ----------------
    df_rev["year"] = df_rev["period"].dt.year
    df_rev["month"] = df_rev["period"].dt.to_period("M").dt.to_timestamp()
    df_rev["day"] = df_rev["period"].dt.floor("D")

    totals_by_year = df_rev.groupby("year", as_index=True)["value"].sum().sort_index()
    prev_total = float(totals_by_year.get(PREV_YEAR, 0.0))
    this_total = float(totals_by_year.get(THIS_YEAR, 0.0))

    # monthly totals series for forecasting
    monthly_totals = df_rev.groupby("month", as_index=True)["value"].sum().sort_index()

    # Forecast monthly (next 12 months) using Prophet preferred, LR fallback
    def forecast_monthly(series, periods=12):
        try:
            s = series.dropna().sort_index()
            if len(s) < 6:
                return None
            if PROPHET_AVAILABLE and len(s) >= 12:
                from prophet import Prophet
                dfp = s.reset_index().rename(columns={s.index.name or 'index':'ds', s.name:'y'})
                dfp.columns = ['ds','y']
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(dfp)
                future = m.make_future_dataframe(periods=periods, freq='MS')
                fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].set_index("ds")
                return fc.tail(periods)
            else:
                # LR fallback
                from sklearn.linear_model import LinearRegression
                X = np.arange(len(s)).reshape(-1,1)
                y = s.values
                lr = LinearRegression().fit(X,y)
                fut_X = np.arange(len(s), len(s)+periods).reshape(-1,1)
                yhat = lr.predict(fut_X)
                last = s.index.max()
                next_idx = pd.date_range(start=(pd.to_datetime(last) + pd.offsets.MonthBegin(1)).replace(day=1), periods=periods, freq='MS')
                fc = pd.DataFrame({"yhat": np.maximum(yhat,0), "yhat_lower": np.maximum(yhat*0.85,0), "yhat_upper": np.maximum(yhat*1.15,0)}, index=next_idx)
                return fc
        except Exception:
            return None

    fc_monthly = forecast_monthly(monthly_totals, periods=12)
    predicted_next_total = float(fc_monthly[fc_monthly.index.year==NEXT_YEAR]["yhat"].sum()) if fc_monthly is not None and any(fc_monthly.index.year==NEXT_YEAR) else (float(fc_monthly["yhat"].sum()) if fc_monthly is not None else 0.0)

    # ---------------- KPIs & comparisons ----------------
    def pct(a,b):
        try:
            if a == 0:
                return float("inf") if b!=0 else 0.0
            return ((b-a)/abs(a))*100.0
        except Exception:
            return 0.0

    kpi_prev_vs_this = pct(prev_total, this_total)
    kpi_this_vs_next = pct(this_total, predicted_next_total)

    st.subheader("Revenue Trend — Real & Predicted (Maxed)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"Prev Year ({PREV_YEAR}) Revenue", f"₹{int(prev_total):,}", delta=f"{kpi_prev_vs_this:.2f}%")
    c2.metric(f"This Year ({THIS_YEAR}) Revenue", f"₹{int(this_total):,}", delta=f"{kpi_this_vs_next:.2f}%")
    c3.metric(f"Predicted Next Year ({NEXT_YEAR}) Revenue", f"₹{int(predicted_next_total):,}", delta=f"{pct(prev_total,predicted_next_total):.2f}%")
    c4.metric("Total Historical Revenue", f"₹{int(df_rev['value'].sum()):,}")

    st.markdown("### Interactive Charts — choose view & smoothing")
    view = st.selectbox("Choose timeseries view", options=["Monthly totals","Daily series","Cumulative"], index=0)
    smooth = st.slider("Smoothing window (months) for moving average", min_value=1, max_value=12, value=3)

    # prepare plotting DF
    monthly_df = monthly_totals.reset_index().rename(columns={"month":"period","value":"value"})
    if fc_monthly is not None:
        fc_df = fc_monthly.reset_index().rename(columns={"index":"period"})
        fc_df = fc_df.rename(columns={"yhat":"value"})
        plot_df = pd.concat([monthly_df.assign(type="actual"), fc_df.assign(type="predicted")], ignore_index=True, sort=False)
    else:
        plot_df = monthly_df.assign(type="actual")

    # Plotly time series with ribbons and markers
    fig = go.Figure()
    # actual line
    fig.add_trace(go.Scatter(x=monthly_df["period"], y=monthly_df["value"], mode="lines+markers", name="Actual", line=dict(width=2)))
    # moving average
    ma = monthly_df["value"].rolling(window=smooth, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=monthly_df["period"], y=ma, mode="lines", name=f"{smooth}-mo MA", line=dict(dash="dash")))

    # predicted
    if fc_monthly is not None:
        fig.add_trace(go.Scatter(x=fc_df["period"], y=fc_df["value"], mode="lines+markers", name="Predicted", line=dict(color="orange", width=2)))
        # ribbon
        fig.add_trace(go.Scatter(x=list(fc_df["period"]) + list(fc_df["period"][::-1]),
                                 y=list(fc_monthly["yhat_upper"]) + list(fc_monthly["yhat_lower"][::-1]),
                                 fill='toself', fillcolor='rgba(255,165,0,0.12)', line=dict(color='rgba(255,165,0,0)'), hoverinfo="skip", showlegend=True, name="Prediction CI"))

    fig.update_layout(title="Revenue: Actual vs Predicted (Monthly)", xaxis_title="Month", yaxis_title="Revenue (₹)", legend_title="Series", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # additional quick charts
    st.markdown("### Additional Views")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Seasonality / Decomposition")
        try:
            # time series decomposition if enough points
            if len(monthly_totals) >= 24:
                res = sm.tsa.seasonal_decompose(monthly_totals, model="additive", period=12, two_sided=True, extrapolate_trend='freq')
                fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=("Observed","Trend","Seasonal","Residual"))
                fig_decomp.add_trace(go.Scatter(x=monthly_totals.index, y=res.observed, name="Observed"), row=1, col=1)
                fig_decomp.add_trace(go.Scatter(x=monthly_totals.index, y=res.trend, name="Trend"), row=2, col=1)
                fig_decomp.add_trace(go.Scatter(x=monthly_totals.index, y=res.seasonal, name="Seasonal"), row=3, col=1)
                fig_decomp.add_trace(go.Scatter(x=monthly_totals.index, y=res.resid, name="Residual"), row=4, col=1)
                fig_decomp.update_layout(height=800)
                st.plotly_chart(fig_decomp, use_container_width=True)
            else:
                st.info("Not enough history for seasonal decomposition (need >=24 months).")
        except Exception:
            st.info("Seasonal decomposition unavailable.")
    with col_b:
        st.markdown("#### Cumulative & Waterfall")
        cum = monthly_df.copy()
        cum["cum"] = cum["value"].cumsum()
        fig_cum = px.area(cum, x="period", y="cum", title="Cumulative Revenue")
        st.plotly_chart(fig_cum, use_container_width=True)

    # ---------------- statistical metrics ----------------
    st.markdown("### Statistical Metrics & Anomalies")
    roi_12m = monthly_totals.tail(12).sum() if len(monthly_totals)>=1 else 0
    mom = monthly_totals.pct_change().dropna()
    anomalies = mom[ mom.abs() > 0.5 ]  # >50% month-over-month change flagged
    metrics = {
        "Total historical revenue": int(df_rev["value"].sum()),
        f"Revenue last 12 months": int(roi_12m),
        "Mean monthly": int(monthly_totals.mean()) if len(monthly_totals)>0 else 0,
        "Std monthly": float(monthly_totals.std()) if len(monthly_totals)>0 else 0.0,
        "Latest MoM %": float(mom.iloc[-1]*100) if len(mom)>0 else None,
    }
    st.json(metrics)

    if not anomalies.empty:
        st.warning("Anomalies detected (MoM > 50%):")
        anomalies_df = anomalies.reset_index().rename(columns={0:"MoM"})
        anomalies_df["MoM%"] = anomalies_df.iloc[:,1]*100
        st.dataframe(anomalies_df.style.format({"MoM%":"{:.2f}%"}), use_container_width=True)
    else:
        st.info("No major MoM anomalies detected.")

    # ---------------- per-period comparisons ----------------
    st.markdown("### Prev vs This vs Predicted Next Year — Breakdown")
    # prev & this per-month totals aggregated into yearly totals already computed
    # create comparison dataframe
    prev_vs_this = []
    # If fc_monthly exists, sum into next year too
    for yr in sorted(set([PREV_YEAR, THIS_YEAR, NEXT_YEAR])):
        if yr == NEXT_YEAR and fc_monthly is not None:
            val = float(fc_monthly[fc_monthly.index.year==NEXT_YEAR]["yhat"].sum()) if any(fc_monthly.index.year==NEXT_YEAR) else float(fc_monthly["yhat"].sum())
        else:
            val = float(totals_by_year.get(yr, 0.0)) if yr in totals_by_year.index else float(monthly_totals[monthly_totals.index.year==yr].sum() if not monthly_totals.empty else 0.0)
        prev_vs_this.append({"year": yr, "value": val})
    comp_df = pd.DataFrame(prev_vs_this)
    comp_df["YoY%"] = comp_df["value"].pct_change()*100
    st.dataframe(comp_df.style.format({"value":"₹{:,}", "YoY%":"{:.2f}%"}), use_container_width=True)

    # ---------------- Exports ----------------
    st.markdown("### Export / Download")
    csv = df_rev.to_csv(index=False).encode("utf-8")
    st.download_button("Download revenue trend CSV", csv, file_name=f"revenue_trend_{THIS_YEAR}.csv", mime="text/csv")
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_rev.to_excel(writer, sheet_name="raw", index=False)
            monthly_totals.reset_index().to_excel(writer, sheet_name="monthly", index=False)
            if fc_monthly is not None:
                fc_monthly.reset_index().to_excel(writer, sheet_name="forecast", index=True)
            comp_df.to_excel(writer, sheet_name="year_comparison", index=False)
        buffer.seek(0)
        st.download_button("Download revenue trend Excel", buffer, file_name=f"revenue_trend_analysis_{THIS_YEAR}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("Excel export failed; CSV available.")

    # ---------------- AI narrative ----------------
    if enable_ai:
        with st.spinner("Generating AI revenue trend narrative..."):
            system = "You are a senior financial analyst. Produce a short, data-driven narrative focusing on trend, forecast, anomalies, and 3 actionable recommendations."
            sample = monthly_totals.tail(12).reset_index().rename(columns={"month":"period","value":"revenue"}).to_dict(orient="records")
            user = f"Sample last 12 months: {json.dumps(sample, default=str)}\nLatest YoY%: {kpi_prev_vs_this:.2f}, Predicted next year total: ₹{int(predicted_next_total):,}\nProvide: (1) 3 key observations, (2) top 2 risks, (3) 3 actions for ops/strategy."
            ai_out = deepinfra_chat(system, user, max_tokens=360)
            if isinstance(ai_out, dict) and "text" in ai_out:
                st.markdown("**AI Revenue Trend Narrative**")
                st.write(ai_out["text"])
            else:
                st.info("AI narrative unavailable or key missing.")

    st.markdown("---")
    st.success("✅ Revenue Trend analysis complete (real + predicted).")

# =====================================================
# 7️⃣ MAXED — Forecasting (Prev Year / This Year / Next Year — Real + Predicted + Full Analysis)
# =====================================================
import io, math, json, numpy as np, pandas as pd, streamlit as st, plotly.express as px, plotly.graph_objects as go, warnings
from datetime import date
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR, THIS_YEAR, NEXT_YEAR = TODAY.year - 1, TODAY.year, TODAY.year + 1

def auto_agg_timescale(df):
    """Automatically infer daily/monthly/yearly frequency."""
    if "date" not in df.columns:
        return df
    df = df.copy().sort_values("date")
    diff_days = (df["date"].diff().median() or pd.Timedelta("1D")).days
    if diff_days <= 1:
        df["period"] = df["date"].dt.to_period("D").dt.to_timestamp()
    elif diff_days < 25:
        df["period"] = df["date"].dt.to_period("M").dt.to_timestamp()
    else:
        df["period"] = df["date"].dt.to_period("Y").dt.to_timestamp()
    return df

def prophet_forecast(df, months=12):
    from prophet import Prophet
    dfp = df.rename(columns={"date": "ds", "value": "y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=months, freq="MS")
    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fc.columns = ["date", "value", "lower", "upper"]
    fc["type"] = "Predicted"
    return fc

def arima_forecast(df, months=12):
    from statsmodels.tsa.arima.model import ARIMA
    series = df.set_index("date")["value"]
    model = ARIMA(series, order=(1,1,1))
    fit = model.fit()
    fc = fit.get_forecast(steps=months)
    pred_df = pd.DataFrame({
        "date": pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq="MS"),
        "value": fc.predicted_mean,
        "lower": fc.conf_int()["lower value"],
        "upper": fc.conf_int()["upper value"],
        "type": "Predicted"
    })
    return pred_df

def linear_forecast(df, months=12):
    df = df.copy().sort_values("date")
    df["x"] = np.arange(len(df))
    X, y = df[["x"]].values, df["value"].values
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X, y)
    future_x = np.arange(len(df), len(df)+months).reshape(-1, 1)
    preds = lr.predict(future_x)
    future_dates = pd.date_range(df["date"].max() + pd.offsets.MonthBegin(1), periods=months, freq="MS")
    fc = pd.DataFrame({
        "date": future_dates,
        "value": preds,
        "lower": preds * 0.9,
        "upper": preds * 1.1,
        "type": "Predicted"
    })
    return fc

if not df_trend.empty:
    st.subheader("🔮 Advanced Forecasting — Real + Predicted + Comparative Insights")

    df_trend = auto_agg_timescale(df_trend)
    df_trend["type"] = "Actual"
    df_trend = df_trend.sort_values("date")

    # ---------------- Try Prophet → ARIMA → Linear ----------------
    forecast_df = pd.DataFrame()
    if PROPHET_AVAILABLE:
        try:
            forecast_df = prophet_forecast(df_trend, months=forecast_periods)
            model_used = "Prophet"
        except Exception as e:
            st.warning(f"Prophet failed → fallback to ARIMA: {e}")
            try:
                forecast_df = arima_forecast(df_trend, months=forecast_periods)
                model_used = "ARIMA"
            except Exception:
                forecast_df = linear_forecast(df_trend, months=forecast_periods)
                model_used = "Linear Regression"
    else:
        forecast_df = linear_forecast(df_trend, months=forecast_periods)
        model_used = "Linear Regression"

    full_df = pd.concat([df_trend, forecast_df], ignore_index=True)
    full_df = full_df.sort_values("date")

    # ---------------- Yearly totals ----------------
    full_df["year"] = full_df["date"].dt.year
    year_totals = full_df.groupby(["year", "type"], as_index=False)["value"].sum()

    prev_actual = year_totals.query("year == @PREV_YEAR and type == 'Actual'")["value"].sum()
    this_actual = year_totals.query("year == @THIS_YEAR and type == 'Actual'")["value"].sum()
    next_pred = year_totals.query("year == @NEXT_YEAR and type == 'Predicted'")["value"].sum()

    def pct(a,b): return ((b-a)/abs(a))*100 if a else 0

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Prev Year ({PREV_YEAR})", f"₹{int(prev_actual):,}", delta=None)
    c2.metric(f"This Year ({THIS_YEAR})", f"₹{int(this_actual):,}", delta=f"{pct(prev_actual,this_actual):.2f}% vs Prev")
    c3.metric(f"Predicted Next ({NEXT_YEAR})", f"₹{int(next_pred):,}", delta=f"{pct(this_actual,next_pred):.2f}% vs This")

    st.markdown(f"**Forecast Model Used:** {model_used}")

    # ---------------- Plotly Chart ----------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_trend["date"], y=df_trend["value"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["value"], name="Predicted", mode="lines+markers", line=dict(color="orange")))
    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
        y=list(forecast_df["upper"]) + list(forecast_df["lower"][::-1]),
        fill='toself', fillcolor='rgba(255,165,0,0.1)', line=dict(color='rgba(255,165,0,0)'),
        hoverinfo="skip", showlegend=True, name="Confidence Band"
    ))
    fig.update_layout(
        title="Forecast: Actual vs Predicted",
        xaxis_title="Date", yaxis_title="Value", hovermode="x unified",
        template="plotly_white", legend_title="Series"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Additional Insights ----------------
    st.markdown("### 🔍 Deep Insights & Metrics")
    df_trend["MA(3)"] = df_trend["value"].rolling(3).mean()
    df_trend["YoY%"] = df_trend["value"].pct_change(12) * 100
    st.dataframe(df_trend.tail(12).style.format({"value":"₹{:,}", "YoY%":"{:.2f}%"}), use_container_width=True)

    heat = df_trend.copy()
    heat["month"] = heat["date"].dt.month
    heat["year"] = heat["date"].dt.year
    pivot = heat.pivot_table(index="year", columns="month", values="value", aggfunc="sum")
    fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis", title="Heatmap: Revenue by Month-Year")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Cumulative line
    cum_df = full_df.copy()
    cum_df["cumulative"] = cum_df["value"].cumsum()
    fig_cum = px.area(cum_df, x="date", y="cumulative", color="type", title="Cumulative Revenue (Actual + Forecast)")
    st.plotly_chart(fig_cum, use_container_width=True)

    # ---------------- Statistical Metrics ----------------
    st.markdown("### 📈 Statistical Summary")
    desc = df_trend["value"].describe().to_frame().T
    desc["latest_MoM%"] = df_trend["value"].pct_change().iloc[-1] * 100
    desc["latest_YoY%"] = df_trend["YoY%"].iloc[-1]
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    # ---------------- Export ----------------
    csv = full_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast Data (CSV)", csv, file_name=f"forecast_full_{THIS_YEAR}.csv")

    # ---------------- AI Forecast Narrative ----------------
    if enable_ai:
        with st.spinner("Generating AI Forecast Summary..."):
            sample = forecast_df.head(12).to_dict(orient="records")
            system = "You are a senior data scientist providing executive-level forecast analysis."
            user = f"""Here are forecasted values: {json.dumps(sample, default=str)}.
            Actual {THIS_YEAR} total = {this_actual:.0f}, Predicted {NEXT_YEAR} total = {next_pred:.0f}.
            Provide a concise 5-line summary including:
            1. Main trend
            2. Growth/fall
            3. Model confidence
            4. Top 2 risks
            5. 2 strategic actions."""
            ai_out = deepinfra_chat(system, user, max_tokens=400)
            if isinstance(ai_out, dict) and "text" in ai_out:
                st.markdown("**AI Forecast Summary**")
                st.write(ai_out["text"])
            else:
                st.info("AI forecast summary unavailable.")

# =====================================================
# 🔥 MAXED Anomaly Detection & Explainability (multi-method)
# =====================================================
import numpy as np
import pandas as pd
import io
import json
import math
import traceback
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date
import warnings
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

if enable_anomaly:
    st.subheader("🚨 MAXED Anomaly Detection & Explainability")
    if df_trend is None or df_trend.empty:
        st.info("No trend data available for anomaly detection.")
    elif 'value' not in df_trend.columns:
        st.info("No 'value' column detected for anomaly detection.")
    else:
        # --- UI controls ---
        st.markdown("**Detection Controls**")
        colA, colB, colC = st.columns([1.2,1,1])
        with colA:
            method = st.selectbox("Primary detection method", options=[
                "IsolationForest (robust)", "LocalOutlierFactor", "Z-score (rolling)",
                "STL Residuals (seasonal)", "Prophet Residuals"
            ], index=0)
            contamination = st.slider("Contamination (expected outlier fraction)", min_value=0.001, max_value=0.2, value=0.02, step=0.001)
            flag_threshold = st.slider("Z-score / residual threshold (%)", min_value=2.0, max_value=8.0, value=3.5, step=0.1)
        with colB:
            detect_scope = st.selectbox("Scope", options=["Global (time-series)","Per-category","Per-state","Per-maker"], index=0)
            date_granularity = st.selectbox("Granularity for detection", options=["Daily","Monthly","Yearly"], index=1)
        with colC:
            run_multi = st.checkbox("Run multiple detectors & ensemble", value=True)
            preview_n = st.number_input("Preview top N anomalies", min_value=3, max_value=200, value=20)

        # --- Prepare timeseries according to granularity ---
        ts = df_trend.copy()
        ts = ts.sort_values("date").reset_index(drop=True)
        if date_granularity == "Daily":
            ts["period"] = ts["date"].dt.floor("D")
        elif date_granularity == "Monthly":
            ts["period"] = ts["date"].dt.to_period("M").dt.to_timestamp()
        else:
            ts["period"] = ts["date"].dt.to_period("Y").dt.to_timestamp()

        # aggregate if needed
        agg_ts = ts.groupby("period", as_index=False)["value"].sum().rename(columns={"period":"date","value":"value"})
        agg_ts = agg_ts.sort_values("date").reset_index(drop=True)

        # --- Helper detectors ---
        detectors_results = {}

        # 1) IsolationForest
        def run_isolationforest(series, contamination=0.02):
            try:
                if not SKLEARN_AVAILABLE:
                    return None, "sklearn_missing"
                from sklearn.ensemble import IsolationForest
                vals = series.values.reshape(-1,1)
                iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                iso.fit(vals)
                scores = iso.decision_function(vals) * -1.0  # higher -> more anomalous
                preds = iso.predict(vals)  # -1 anomaly, 1 normal
                df = pd.DataFrame({"date": series.index, "value": series.values, "score": scores, "anomaly": preds})
                df["anomaly_flag"] = df["anomaly"] == -1
                return df, "ok"
            except Exception as e:
                return None, str(e)

        # 2) LocalOutlierFactor
        def run_lof(series, n_neighbors=20):
            try:
                if not SKLEARN_AVAILABLE:
                    return None, "sklearn_missing"
                from sklearn.neighbors import LocalOutlierFactor
                vals = series.values.reshape(-1,1)
                lof = LocalOutlierFactor(n_neighbors=min(20, max(5, len(series)//5)), contamination=contamination, novelty=False)
                preds = lof.fit_predict(vals)
                scores = -lof.negative_outlier_factor_
                df = pd.DataFrame({"date": series.index, "value": series.values, "score": scores, "anomaly": preds})
                df["anomaly_flag"] = df["anomaly"] == -1
                return df, "ok"
            except Exception as e:
                return None, str(e)

        # 3) Rolling Z-score
        def run_zscore(series, window=12, threshold=3.5):
            s = series.copy()
            rolling_mean = s.rolling(window=window, min_periods=1, center=False).mean()
            rolling_std = s.rolling(window=window, min_periods=1, center=False).std().replace(0, np.nan).fillna(0.0)
            z = (s - rolling_mean) / (rolling_std.replace(0, np.nan).fillna(1.0))
            df = pd.DataFrame({"date": s.index, "value": s.values, "zscore": z})
            df["anomaly_flag"] = df["zscore"].abs() > threshold
            df["score"] = df["zscore"].abs()
            return df, "ok"

        # 4) STL residuals
        def run_stl_resid(series, period=12, threshold_pct=3.5):
            try:
                import statsmodels.api as sm
                s = series.copy()
                if len(s) < period*2:  # not enough history for STL
                    return None, "insufficient_history"
                res = sm.tsa.seasonal_decompose(s, model="additive", period=period, extrapolate_trend='freq')
                resid = res.resid.fillna(0)
                resid_z = (resid - resid.mean()) / (resid.std() or 1.0)
                df = pd.DataFrame({"date": resid.index, "value": series.values, "resid": resid.values, "score": resid_z.abs()})
                df["anomaly_flag"] = df["score"] > threshold_pct
                return df, "ok"
            except Exception as e:
                return None, str(e)

        # 5) Prophet residuals
        def run_prophet_resid(df_full, threshold_pct=3.5):
            if not PROPHET_AVAILABLE:
                return None, "prophet_missing"
            try:
                from prophet import Prophet
                dfp = df_full.reset_index().rename(columns={"date":"ds","value":"y"}) if "date" in df_full.index.names else df_full.rename(columns={"date":"ds","value":"y"})
                if dfp.shape[0] < 12:
                    return None, "insufficient_history"
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(dfp[["ds","y"]])
                fut = m.make_future_dataframe(periods=0, freq='MS')
                pred = m.predict(dfp[["ds"]])
                resid = dfp["y"].values - pred["yhat"].values
                resid_z = (resid - resid.mean()) / (resid.std() or 1.0)
                df = pd.DataFrame({"date": pd.to_datetime(dfp["ds"]), "value": dfp["y"].values, "resid": resid, "score": np.abs(resid_z)})
                df["anomaly_flag"] = df["score"] > threshold_pct
                return df, "ok"
            except Exception as e:
                return None, str(e)

        # --- Run primary detector (and ensemble if selected) ---
        # Use aggregated series indexed by date
        series = agg_ts.set_index("date")["value"]
        primary_res = None
        status_msg = ""

        if method == "IsolationForest":
            primary_res, status_msg = run_isolationforest(series, contamination=contamination)
        elif method == "LocalOutlierFactor":
            primary_res, status_msg = run_lof(series)
        elif method == "Z-score (rolling)":
            primary_res, status_msg = run_zscore(series, window= max(3, min(36, len(series)//4)), threshold=flag_threshold)
        elif method == "STL Residuals (seasonal)":
            primary_res, status_msg = run_stl_resid(series, period=12, threshold_pct=flag_threshold)
        elif method == "Prophet Residuals":
            primary_res, status_msg = run_prophet_resid(agg_ts, threshold_pct=flag_threshold)
        else:
            primary_res, status_msg = run_isolationforest(series, contamination=contamination)

        detectors_results[method] = {"df": primary_res, "status": status_msg}

        # Ensemble: run additional detectors and build a consensus score
        ensemble_df = None
        if run_multi:
            extras = ["IsolationForest","LocalOutlierFactor","Z-score (rolling)","STL Residuals (seasonal)","Prophet Residuals"]
            for ex in extras:
                if ex == method:  # already ran
                    continue
                try:
                    if ex == "IsolationForest":
                        res, stat = run_isolationforest(series, contamination=contamination)
                    elif ex == "LocalOutlierFactor":
                        res, stat = run_lof(series)
                    elif ex == "Z-score (rolling)":
                        res, stat = run_zscore(series, window=max(3, min(36, len(series)//4)), threshold=flag_threshold)
                    elif ex == "STL Residuals (seasonal)":
                        res, stat = run_stl_resid(series, period=12, threshold_pct=flag_threshold)
                    elif ex == "Prophet Residuals":
                        res, stat = run_prophet_resid(agg_ts, threshold_pct=flag_threshold)
                    else:
                        res, stat = None, "skipped"
                except Exception as e:
                    res, stat = None, str(e)
                detectors_results[ex] = {"df": res, "status": stat}

            # build ensemble table
            # start from index series.index -> for each detector create binary flag column, and score column (normalized)
            ensemble = pd.DataFrame(index=series.index)
            for name, info in detectors_results.items():
                ddf = info.get("df")
                if ddf is None or not isinstance(ddf, pd.DataFrame):
                    continue
                # align by date
                tmp = ddf.set_index("date")
                # fill missing dates with non-anomalous defaults
                ensemble[f"{name}_flag"] = tmp.reindex(ensemble.index)["anomaly_flag"].fillna(False).astype(int)
                # normalize score to 0-1
                sc = tmp.reindex(ensemble.index)["score"].fillna(0.0)
                if sc.max() > sc.min():
                    scn = (sc - sc.min()) / (sc.max() - sc.min())
                else:
                    scn = sc
                ensemble[f"{name}_score"] = scn.fillna(0.0)
            if ensemble.shape[1] > 0:
                # consensus columns end with _flag
                flag_cols = [c for c in ensemble.columns if c.endswith("_flag")]
                score_cols = [c for c in ensemble.columns if c.endswith("_score")]
                ensemble["consensus_score"] = ensemble[score_cols].mean(axis=1) if score_cols else 0.0
                ensemble["consensus_votes"] = ensemble[flag_cols].sum(axis=1) if flag_cols else 0
                # mark as anomaly where consensus_score or votes exceed thresholds
                ensemble["ensemble_anomaly"] = (ensemble["consensus_votes"] >= max(1, math.ceil(len(flag_cols)/2))) | (ensemble["consensus_score"] > 0.6)
                ensemble_df = ensemble.reset_index().rename(columns={"index":"date"})
            else:
                ensemble_df = None

        # --- Choose final anomalies to display ---
        if ensemble_df is not None:
            # prefer ensemble if available
            final_anoms = ensemble_df[ensemble_df["ensemble_anomaly"]].copy()
            final_anoms["score"] = ensemble_df.loc[final_anoms.index, "consensus_score"]
            source_note = "Ensemble"
        else:
            if isinstance(primary_res, pd.DataFrame):
                final_anoms = primary_res[primary_res["anomaly_flag"]].copy()
                final_anoms["score"] = final_anoms.get("score", final_anoms.get("zscore", np.abs(final_anoms.get("resid", 0))))
                source_note = method
            else:
                final_anoms = pd.DataFrame(columns=["date","value","score"])
                source_note = "None"

        # sort & preview
        if not final_anoms.empty:
            final_anoms = final_anoms.sort_values("score", ascending=False).head(preview_n)
            st.markdown(f"#### Detected anomalies — method: **{source_note}** — total flagged: **{len(final_anoms):,}**")
            # human-friendly table
            display_df = final_anoms.reset_index(drop=True).copy()
            if "date" in display_df.columns:
                display_df["date"] = pd.to_datetime(display_df["date"])
            st.dataframe(display_df.style.format({"value":"{:,}", "score":"{:.4f}"}), height=300)
        else:
            st.info("No anomalies detected by the selected method(s).")

        # --- Visualization: show anomalies on time series ---
        st.markdown("### Visual: Anomalies on Time Series")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg_ts["date"], y=agg_ts["value"], mode="lines+markers", name="Value", line=dict(width=2)))
        if ensemble_df is not None and not ensemble_df.empty:
            idxs = ensemble_df[ensemble_df["ensemble_anomaly"]]["index"] if "index" in ensemble_df.columns else ensemble_df[ensemble_df["ensemble_anomaly"]]["date"]
            anom_dates = pd.to_datetime(ensemble_df[ensemble_df["ensemble_anomaly"]]["date"])
            anom_vals = agg_ts.set_index("date").reindex(anom_dates)["value"].values
            fig.add_trace(go.Scatter(x=anom_dates, y=anom_vals, mode="markers", name="Anomaly (ensemble)", marker=dict(color="red", size=10, symbol="x")))
        elif not final_anoms.empty:
            anom_dates = pd.to_datetime(final_anoms["date"])
            anom_vals = final_anoms["value"].values
            fig.add_trace(go.Scatter(x=anom_dates, y=anom_vals, mode="markers", name=f"Anomaly ({source_note})", marker=dict(color="red", size=10, symbol="x")))
        fig.update_layout(title="Time Series with Anomalies", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

        # --- Per-period / per-category anomalies (if scope selected) ---
        if detect_scope != "Global (time-series)":
            st.markdown("### Scope-based Anomaly Detection (per-group)")
            # determine grouping column (if available)
            group_col = None
            if detect_scope == "Per-category" and "category" in df_trend.columns:
                group_col = "category"
            elif detect_scope == "Per-state" and "state" in df_trend.columns:
                group_col = "state"
            elif detect_scope == "Per-maker" and "maker" in df_trend.columns:
                group_col = "maker"
            if group_col is None:
                st.info(f"No column available for scope {detect_scope}. Available columns: {', '.join(df_trend.columns)}")
            else:
                # aggregate per group & period
                grp = df_trend.copy()
                if date_granularity == "Monthly":
                    grp["period"] = grp["date"].dt.to_period("M").dt.to_timestamp()
                elif date_granularity == "Daily":
                    grp["period"] = grp["date"].dt.floor("D")
                else:
                    grp["period"] = grp["date"].dt.to_period("Y").dt.to_timestamp()
                agg = grp.groupby([group_col,"period"], as_index=False)["value"].sum()
                # run zscore per group (fast) and flag anomalies
                anomalies_group = []
                for name, g in agg.groupby(group_col):
                    s = g.set_index("period")["value"].sort_index()
                    if len(s) < 6:
                        continue
                    z = (s - s.mean()) / (s.std() or 1.0)
                    flagged = z[ z.abs() > flag_threshold ]
                    for idx, val in flagged.items():
                        anomalies_group.append({"group": name, "period": idx, "value": float(s.loc[idx]), "zscore": float(z.loc[idx])})
                if anomalies_group:
                    anom_gdf = pd.DataFrame(anomalies_group).sort_values(["group","period"], ascending=[True,False])
                    st.dataframe(anom_gdf.head(200).assign(period=lambda d: d["period"].dt.strftime("%Y-%m-%d")))
                else:
                    st.info("No group-level anomalies detected with current thresholds.")

        # --- Exports & artifacts ---
        st.markdown("### Export anomalies & diagnostics")
        if not final_anoms.empty:
            csv_buf = final_anoms.to_csv(index=False).encode("utf-8")
            st.download_button("Download anomalies CSV", csv_buf, file_name=f"anomalies_{THIS_YEAR}.csv", mime="text/csv")
        # diagnostics: per-detector statuses
        st.markdown("Detector statuses:")
        for name, info in detectors_results.items():
            st.write(f"- **{name}**: {info.get('status')}")

        # --- AI explainability for anomalies ---
        if enable_ai and (('final_anoms' in locals() and not final_anoms.empty) or ('anom_df' in locals() and not locals().get('anom_df', pd.DataFrame()).empty)):
            with st.spinner("Generating AI anomaly insights..."):
                try:
                    sample = (final_anoms.head(10) if not final_anoms.empty else pd.DataFrame()).to_dict(orient="records")
                    if not sample:
                        sample = (locals().get('anom_df', pd.DataFrame()).head(10).to_dict(orient="records") if 'anom_df' in locals() else [])
                    system = "You are an analytics investigator. Review the anomaly rows and provide likely causes, prioritized remediation steps, and whether each anomaly looks like data issue vs real-world event."
                    user = f"Anomalies sample: {json.dumps(sample, default=str)}\nProvide 1-sentence cause hypotheses for each (max 3), 3 prioritized remediation steps, and a short recommendation for business owners."
                    ai_resp = deepinfra_chat(system, user, max_tokens=420)
                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown("**AI Anomaly Insights & Remediation**")
                        st.write(ai_resp["text"])
                    else:
                        st.info("AI response unavailable for anomaly insights.")
                except Exception as e:
                    st.error(f"AI anomaly explainability failed: {e}")

        st.markdown("---")
        st.success("✅ MAXED anomaly detection completed.")
else:
    st.info("Anomaly detection disabled.")

# ---------------- ⚡ MAXED Clustering, Prediction & Correlation Suite ----------------
if enable_clustering:
    st.markdown("## 🧠 Advanced Clustering, Prediction & Correlation Analysis")
    st.markdown("Full-year trend comparison, predictive clustering, and correlation insights — all AI-augmented.")

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    # 🔹 Step 1: Build Base DataFrame (from trend or revenue)
    base_df = None
    if not df_top5_rev.empty:
        base_df = df_top5_rev.copy()
    elif not df_trend.empty:
        base_df = df_trend.copy()
    else:
        st.info("No source data found for clustering or prediction.")
        base_df = pd.DataFrame()

    if not base_df.empty:
        # Normalize column naming
        if "date" in base_df.columns:
            base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
            base_df["year"] = base_df["date"].dt.year
            base_df["month"] = base_df["date"].dt.month
        base_df["value"] = pd.to_numeric(base_df.get("value", base_df.iloc[:, -1]), errors="coerce")

        # 🔹 Step 2: Compute year-wise aggregates for prev, this, next year
        yearwise = base_df.groupby("year")["value"].sum().reset_index()
        this_year = yearwise["year"].max()
        prev_year = this_year - 1
        next_year = this_year + 1

        st.markdown("### 📅 Yearly Comparison & Growth Metrics")
        try:
            prev_val = yearwise.loc[yearwise["year"] == prev_year, "value"].sum()
            curr_val = yearwise.loc[yearwise["year"] == this_year, "value"].sum()
            growth = ((curr_val - prev_val) / prev_val * 100) if prev_val else 0

            st.metric("📈 Current Year Total", f"{curr_val:,.0f}")
            st.metric("📊 Previous Year Total", f"{prev_val:,.0f}")
            st.metric("🚀 YoY Growth", f"{growth:.2f}%")

            # Prediction for next year using regression
            model = LinearRegression()
            X = yearwise[["year"]]
            y = yearwise["value"]
            model.fit(X, y)
            pred_next = model.predict([[next_year]])[0]

            st.metric("🔮 Next Year Forecast", f"{pred_next:,.0f}")

            # Show chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yearwise["year"], y=yearwise["value"],
                                     mode='lines+markers', name="Actual"))
            fig.add_trace(go.Scatter(x=[next_year], y=[pred_next],
                                     mode='markers', name="Predicted", marker=dict(color="orange", size=12)))
            fig.update_layout(title="📊 Yearly Actual vs Predicted Values", height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Yearly metrics failed: {e}")

        # 🔹 Step 3: Daily / Monthly rolling analysis
        st.markdown("### 📅 Rolling & Seasonal Patterns")
        try:
            if "date" in base_df.columns:
                df_roll = base_df.set_index("date").resample("M").sum()
                df_roll["rolling_mean_3"] = df_roll["value"].rolling(3).mean()
                df_roll["rolling_std_3"] = df_roll["value"].rolling(3).std()

                st.line_chart(df_roll[["value", "rolling_mean_3"]])

                # Plot seasonal comparison
                fig2 = px.box(base_df, x="month", y="value", color="year", title="Monthly Distribution per Year")
                st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Rolling analysis failed: {e}")

        # 🔹 Step 4: Clustering (on numeric scaled features)
        numeric_cols = base_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            X = base_df[numeric_cols].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            n_clusters = st.slider("🔢 Number of Clusters (k)", 2, min(10, len(X_scaled)), 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            base_df["cluster"] = clusters

            # Silhouette
            if len(X_scaled) > n_clusters + 1:
                score = silhouette_score(X_scaled, clusters)
                st.metric("🎯 Silhouette Score", f"{score:.3f}")

            # PCA projection
            pca = PCA(n_components=min(3, X_scaled.shape[1]))
            proj = pca.fit_transform(X_scaled)
            proj_df = pd.DataFrame({
                "PC1": proj[:, 0],
                "PC2": proj[:, 1],
                "PC3": proj[:, 2] if proj.shape[1] > 2 else 0,
                "Cluster": clusters
            })
            fig3d = px.scatter_3d(proj_df, x="PC1", y="PC2", z="PC3",
                                  color=proj_df["Cluster"].astype(str),
                                  title="3D Cluster Projection (PCA)",
                                  opacity=0.8)
            st.plotly_chart(fig3d, use_container_width=True)

            # 🔹 Correlation Matrix
            corr = base_df.select_dtypes(include=[np.number]).corr()
            st.markdown("### 🔗 Correlation Heatmap")
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

            # 🔹 Cluster stats
            st.markdown("### 🧩 Cluster Centroids & Distribution")
            st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols))
            st.bar_chart(base_df["cluster"].value_counts())

            # 🔹 Feature importance with Random Forest
            st.markdown("### 🌳 Feature Importance (Random Forest)")
            try:
                rf = RandomForestRegressor(random_state=42)
                rf.fit(X_scaled, base_df["value"])
                importances = pd.DataFrame({
                    "Feature": numeric_cols,
                    "Importance": rf.feature_importances_
                }).sort_values("Importance", ascending=False)
                fig_imp = px.bar(importances, x="Feature", y="Importance",
                                 title="Feature Importance in Value Prediction")
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.warning(f"Feature importance failed: {e}")

        # 🔹 Step 5: AI insights
        if enable_ai:
            with st.spinner("Generating AI insights on clusters, trends, and predictions..."):
                system = "You are a senior data scientist. Summarize all cluster, correlation, and prediction results clearly."
                summary_data = {
                    "yearwise": yearwise.to_dict(orient="records"),
                    "next_year_pred": float(pred_next),
                    "silhouette_score": float(score) if 'score' in locals() else None,
                    "corr_top": corr.unstack().nlargest(5).to_dict()
                }
                user = f"Data summary:\n{json.dumps(summary_data, indent=2)}\nProvide 7 key insights and 3 next-step actions."
                ai_resp = deepinfra_chat(system, user, max_tokens=400)
                if "text" in ai_resp:
                    st.markdown("**🤖 AI Summary Insights**")
                    st.write(ai_resp["text"])

    else:
        st.warning("⚠️ No sufficient data for maxed clustering or predictive analysis.")

# ---------------- 🚀 SMART EXPORTS (AI + ML + Power BI + Tableau + HTML + Tkinter) ----------------

st.markdown("## 💾 Smart Multi-Format Export Suite (AI + ML + BI Ready)")
st.markdown(
    "Seamlessly export complete analytics with **predictions, KPIs, correlations, AI narratives, and multi-format outputs** "
    "for Excel, HTML, Power BI, Tableau, JSON, and even Tkinter preview — fully maxed-out."
)

with st.expander("📤 Generate & Download Reports", expanded=True):
    import base64
    import json
    import io
    from openpyxl import Workbook, load_workbook
    from openpyxl.chart import LineChart, Reference
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    # ------------------------------------------------------
    # 1️⃣ Aggregate all core datasets
    # ------------------------------------------------------
    datasets = {
        "Category": df_cat,
        "Top Makers": df_mk,
        "Registrations Trend": df_trend,
        "YoY Trend": yoy_df,
        "QoQ Trend": qoq_df,
        "Top 5 Revenue States": df_top5_rev,
        "Revenue Trend": df_rev_trend,
    }

    # ------------------------------------------------------
    # 2️⃣ Predictive Forecasts (Prev/This/Next Year)
    # ------------------------------------------------------
    try:
        if not df_trend.empty:
            df_trend["date"] = pd.to_datetime(df_trend["date"], errors="coerce")
            df_trend["year"] = df_trend["date"].dt.year
            df_trend["month"] = df_trend["date"].dt.month
            df_yearly = df_trend.groupby("year")["value"].sum().reset_index()

            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            X = df_yearly[["year"]]
            y = df_yearly["value"]
            model.fit(X, y)
            next_year = df_yearly["year"].max() + 1
            pred_next = model.predict([[next_year]])[0]

            df_yearly["Type"] = "Actual"
            df_yearly = pd.concat(
                [df_yearly, pd.DataFrame({"year": [next_year], "value": [pred_next], "Type": ["Predicted"]})]
            )

            datasets["Yearly Prediction"] = df_yearly

            # Rolling & anomalies
            df_forecast = df_trend.copy()
            df_forecast["Forecast_3M"] = df_forecast["value"].rolling(3, min_periods=1).mean()
            df_forecast["Anomaly"] = (
                (df_forecast["value"] - df_forecast["Forecast_3M"]).abs()
                > df_forecast["Forecast_3M"] * 0.15
            )
            datasets["Forecast & Anomaly Detection"] = df_forecast
    except Exception as e:
        st.warning(f"Forecasting step skipped: {e}")

    # ------------------------------------------------------
    # 3️⃣ AI Summaries (per dataset + overall KPI Summary)
    # ------------------------------------------------------
    summaries = {}
    if enable_ai:
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    system = f"You are a senior data analyst creating a report summary for '{name}'."
                    user = f"Analyze this data and summarize trends, KPIs, growth, and anomalies:\n{df.head(10).to_dict(orient='records')}"
                    ai_resp = deepinfra_chat(system, user, max_tokens=250)
                    summaries[name] = ai_resp.get("text", "No summary.")
                except Exception:
                    summaries[name] = "AI summary failed."
        ai_df = pd.DataFrame(list(summaries.items()), columns=["Dataset", "AI Summary"])
        datasets["AI Insights"] = ai_df

    # Overall dashboard summary
    try:
        if enable_ai:
            combined_summary = deepinfra_chat(
                "You are a BI expert summarizing a full analytics report.",
                f"Here are dataset summaries:\n{json.dumps(summaries, indent=2)}\nProvide a one-page executive summary with KPIs and recommendations.",
                max_tokens=400,
            )
            one_page_summary = combined_summary.get("text", "Summary unavailable.")
        else:
            one_page_summary = "AI not enabled — summary skipped."
    except Exception as e:
        one_page_summary = f"Summary generation failed: {e}"

    # ------------------------------------------------------
    # 4️⃣ Excel Export (multi-sheet, styled, charts)
    # ------------------------------------------------------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)

        # One-page AI summary
        summary_df = pd.DataFrame({"Executive Summary": [one_page_summary]})
        summary_df.to_excel(writer, sheet_name="OnePageSummary", index=False)

    output.seek(0)
    wb = load_workbook(output)
    thin = Border(left=Side(style="thin"), right=Side(style="thin"),
                  top=Side(style="thin"), bottom=Side(style="thin"))

    # Style & chart embedding
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        # Headers
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="002060", end_color="002060", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin
        # Body
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin
        # Autofit
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col) + 3
            ws.column_dimensions[get_column_letter(col[0].column)].width = width
        # Charts
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
                ws.add_chart(chart, "H5")
            except Exception:
                pass

    styled = io.BytesIO()
    wb.save(styled)
    styled.seek(0)

    ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")

    st.download_button(
        "⬇️ Download Full Excel Report",
        data=styled.getvalue(),
        file_name=f"Vahan_Report_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ------------------------------------------------------
    # 5️⃣ HTML Dashboard Export
    # ------------------------------------------------------
    html_path = io.StringIO()
    try:
        html_summary = f"""
        <html>
        <head><title>Vahan Analytics Dashboard</title></head>
        <body style='font-family:Segoe UI;background:#f9f9f9;padding:20px'>
        <h1>🚀 Vahan Analytics Summary</h1>
        <p><b>Date:</b> {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
        <h2>📊 Executive Summary</h2>
        <p>{one_page_summary}</p>
        <h2>🧾 Dataset Insights</h2>
        {"".join([f"<h3>{k}</h3><p>{v}</p>" for k,v in summaries.items()])}
        <hr><p>Generated automatically by Vahan Analytics Suite.</p>
        </body></html>
        """
        html_path.write(html_summary)
        html_bytes = html_path.getvalue().encode("utf-8")
        st.download_button(
            "🌐 Export as HTML Dashboard",
            data=html_bytes,
            file_name=f"Vahan_Report_{ts}.html",
            mime="text/html",
        )
    except Exception as e:
        st.warning(f"HTML export failed: {e}")

    # ------------------------------------------------------
    # 6️⃣ Power BI, Tableau, JSON, Tkinter-ready exports
    # ------------------------------------------------------
    try:
        # CSV (Power BI / Tableau)
        csv_bytes = io.BytesIO()
        df_concat = pd.concat(
            [df.assign(Source=name) for name, df in datasets.items() if isinstance(df, pd.DataFrame) and not df.empty],
            ignore_index=True
        )
        df_concat.to_csv(csv_bytes, index=False)
        st.download_button(
            "📊 Export CSV for Power BI / Tableau",
            data=csv_bytes.getvalue(),
            file_name=f"Vahan_Data_{ts}.csv",
            mime="text/csv",
        )

        # JSON (for Tkinter / API use)
        json_bytes = json.dumps(datasets, default=str).encode("utf-8")
        st.download_button(
            "🧩 Export JSON (for API / Tkinter use)",
            data=json_bytes,
            file_name=f"Vahan_Data_{ts}.json",
            mime="application/json",
        )
    except Exception as e:
        st.warning(f"Power BI/Tableau export failed: {e}")

    st.success("✅ All formats exported: Excel, HTML, CSV, JSON + full AI summaries & predictions!")

# =====================================================
# 🛠️ MAXED Raw JSON Preview + Multi-format Exports
# =====================================================
import io, os, json, zipfile, textwrap, warnings
import pandas as pd, numpy as np
import plotly.express as px, plotly.io as pio
from datetime import date, datetime
import streamlit as st
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# -------------------------
# Quick safe-normalizers & forecast helpers
# -------------------------
def safe_to_df(obj):
    try:
        return to_df(obj) if obj is not None else pd.DataFrame()
    except Exception:
        try:
            return pd.json_normalize(obj)
        except Exception:
            return pd.DataFrame()

def forecast_monthly(series: pd.Series, periods=12):
    """
    Try prophet -> ARIMA -> Linear fallback.
    series: pd.Series indexed by timestamps (monthly)
    returns DataFrame indexed by future timestamps with columns yhat,yhat_lower,yhat_upper, or None
    """
    s = series.dropna().sort_index()
    if len(s) < 3:
        return None
    # Prophet
    if PROPHET_AVAILABLE and len(s) >= 12:
        try:
            from prophet import Prophet
            dfp = s.reset_index().rename(columns={s.index.name or 'index':'ds', s.name:'y'})
            dfp.columns = ['ds','y']
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=periods, freq='MS')
            fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].set_index("ds")
            return fc.tail(periods)
        except Exception:
            pass
    # ARIMA
    try:
        from statsmodels.tsa.arima.model import ARIMA
        series_indexed = s.copy()
        model = ARIMA(series_indexed, order=(1,1,1))
        res = model.fit()
        fc = res.get_forecast(steps=periods)
        idx = pd.date_range(start=series_indexed.index.max() + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
        pred_df = pd.DataFrame({
            "yhat": fc.predicted_mean,
            "yhat_lower": fc.conf_int().iloc[:,0],
            "yhat_upper": fc.conf_int().iloc[:,1],
        }, index=idx)
        return pred_df
    except Exception:
        pass
    # Linear fallback
    try:
        from sklearn.linear_model import LinearRegression
        X = np.arange(len(s)).reshape(-1,1)
        y = s.values
        lr = LinearRegression().fit(X,y)
        fut_X = np.arange(len(s), len(s)+periods).reshape(-1,1)
        preds = lr.predict(fut_X)
        idx = pd.date_range(start=s.index.max() + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
        dfp = pd.DataFrame({"yhat": preds, "yhat_lower": preds*0.9, "yhat_upper": preds*1.1}, index=idx)
        return dfp
    except Exception:
        return None

# -------------------------
# Prepare dataframes from raw JSONs (use existing variables)
# -------------------------
cat_df_preview = safe_to_df(cat_json)
mk_df_preview  = safe_to_df(mk_json)
trend_df_preview = safe_to_df(tr_json)
top5_rev_preview = safe_to_df(top5_rev_json)
rev_trend_preview = safe_to_df(rev_trend_json)

# Also use already-parsed frames if exist to ensure canonical shapes
df_cat = df_cat if 'df_cat' in globals() else cat_df_preview
df_mk  = df_mk  if 'df_mk' in globals() else mk_df_preview
df_trend = df_trend if 'df_trend' in globals() else trend_df_preview
df_top5_rev = df_top5_rev if 'df_top5_rev' in globals() else top5_rev_preview
df_rev_trend = df_rev_trend if 'df_rev_trend' in globals() else rev_trend_preview

# -------------------------
# UI: Raw JSON preview (debug) - collapsible sections
# -------------------------
with st.expander("🛠️ Raw JSON Preview (debug) — category / makers / trend / top5 / rev-trend", expanded=False):
    st.markdown("**Category JSON**")
    st.json(cat_json or {})
    st.markdown("**Top Makers JSON**")
    st.json(mk_json or {})
    st.markdown("**Registration Trend JSON**")
    st.json(tr_json or {})
    st.markdown("**Top 5 Revenue JSON**")
    st.json(top5_rev_json or {})
    st.markdown("**Revenue Trend JSON**")
    st.json(rev_trend_json or {})

# -------------------------
# Build One-Page Executive KPI Summary (real + predicted)
# -------------------------
def compute_year_totals(df, date_col='date', value_col='value'):
    if df is None or df.empty or date_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=float)
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    s = d.groupby(d[date_col].dt.year)[value_col].sum().sort_index()
    return s

totals_series = compute_year_totals(df_trend, date_col='date', value_col='value')
prev_total = float(totals_series.get(PREV_YEAR, 0.0))
this_total = float(totals_series.get(THIS_YEAR, 0.0))

# Predict next year by forecasting monthly totals then summing months that fall in NEXT_YEAR
monthly_totals = None
predicted_next_total = 0.0
if 'date' in df_trend.columns and 'value' in df_trend.columns and not df_trend.empty:
    df_monthly = df_trend.copy()
    df_monthly['date'] = pd.to_datetime(df_monthly['date'], errors='coerce')
    df_monthly = df_monthly.set_index('date').resample('M')['value'].sum().sort_index()
    monthly_totals = df_monthly
    fc = forecast_monthly(df_monthly, periods=12)
    if fc is not None:
        predicted_next_total = float(fc[fc.index.year == NEXT_YEAR]['yhat'].sum()) if any(fc.index.year==NEXT_YEAR) else float(fc['yhat'].sum())
    else:
        predicted_next_total = 0.0

# KPI cards
st.markdown("## 📋 Executive KPI Summary — Real & Predicted (No synthetic data)")
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Prev Year ({PREV_YEAR})", f"{int(prev_total):,}")
k2.metric(f"This Year ({THIS_YEAR})", f"{int(this_total):,}", delta=f"{((this_total - prev_total)/abs(prev_total)*100) if prev_total else 0:.2f}%")
k3.metric(f"Predicted Next Year ({NEXT_YEAR})", f"{int(predicted_next_total):,}", delta=f"{((predicted_next_total - this_total)/abs(this_total)*100) if this_total else 0:.2f}%")
k4.metric("Overall historical (trend)", f"{int(df_trend['value'].sum()) if not df_trend.empty else 0:,}")

# Quick chart: monthly actual + predicted ribbon
if monthly_totals is not None:
    st.markdown("### Monthly: Actual (history) + Predicted (next 12 months)")
    fc = forecast_monthly(monthly_totals, periods=12)
    plot_df = monthly_totals.reset_index().rename(columns={'index':'month','value':'actual'})
    if fc is not None:
        fc_reset = fc.reset_index().rename(columns={'index':'month'})
        # combine for plot
        plot_comb = pd.concat([plot_df.assign(type='actual').rename(columns={'actual':'value'}), 
                               fc_reset.assign(type='predicted').rename(columns={'yhat':'value'})], ignore_index=True)
        fig = px.line(plot_comb, x='month', y='value', color='type', title='Monthly Real vs Predicted')
        # add ribbon if available
        if 'yhat_upper' in fc.columns and 'yhat_lower' in fc.columns:
            fig.add_traces([dict(x=fc_reset['month'], y=fc_reset['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False),
                            dict(x=fc_reset['month'], y=fc_reset['yhat_lower'], mode='lines', fill='tonexty', fillcolor='rgba(0,176,246,0.15)', line=dict(width=0), showlegend=False)])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(plot_df.set_index('month')['actual'])

# -------------------------
# Export builder UI
# -------------------------
st.markdown("## ⤓ Multi-format Exports — choose formats and generate")
export_excel = st.checkbox("Export Excel (styled multi-sheet)", value=True)
export_html = st.checkbox("Export Single-file HTML dashboard (self-contained)", value=True)
export_bundle = st.checkbox("Export HTML+CSS+JS bundle (zipped)", value=False)
export_csv = st.checkbox("Export CSV (Power BI / Tableau)", value=True)
export_json = st.checkbox("Export JSON", value=True)
export_tk = st.checkbox("Export Tkinter preview app (Python script)", value=False)
export_zip_all = st.checkbox("Export ALL artifacts as ZIP", value=True)

forecast_horizon = st.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=12)
model_choice = st.selectbox("Forecast model preference", options=["auto (Prophet→ARIMA→Linear)","Prophet","ARIMA","Linear"], index=0)

# When user clicks button — create exports immediately
if st.button("🔁 Generate Exports Now"):
    artifacts = {}  # name -> bytes

    # ---------------- Excel (pandas + openpyxl)
    if export_excel:
        try:
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                # write datasets (use canonical dataframes)
                write_order = [
                    ("Executive KPI", pd.DataFrame([{
                        "metric": "prev_year_total",
                        "year": PREV_YEAR,
                        "value": prev_total
                    }, {
                        "metric": "this_year_total",
                        "year": THIS_YEAR,
                        "value": this_total
                    }, {
                        "metric": "predicted_next_year_total",
                        "year": NEXT_YEAR,
                        "value": predicted_next_total
                    }]).rename(columns={"metric":"label"})),
                    ("Category", df_cat),
                    ("Top Makers", df_mk),
                    ("Registrations Trend", df_trend),
                    ("YoY Trend", yoy_df if 'yoy_df' in globals() else pd.DataFrame()),
                    ("QoQ Trend", qoq_df if 'qoq_df' in globals() else pd.DataFrame()),
                    ("Top5 Revenue", df_top5_rev),
                    ("Revenue Trend", df_rev_trend),
                ]
                # add forecast sheet for monthly fc if available
                if monthly_totals is not None:
                    fc = forecast_monthly(monthly_totals, periods=forecast_horizon)
                    if fc is not None:
                        fc_out = fc.reset_index().rename(columns={'index':'date','yhat':'value'})
                        write_order.append(("Monthly Forecast", fc_out))
                for name, df in write_order:
                    try:
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            df.to_excel(writer, sheet_name=name[:31], index=False)
                    except Exception:
                        # skip sheet if failure
                        pass
                # AI summaries sheet if available
                if enable_ai:
                    try:
                        ai_summ = []
                        for nm, df in write_order:
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                sample = df.head(6).to_dict(orient='records')
                                sys_prompt = f"You are an analyst. Summarize dataset '{nm}' quickly."
                                user_prompt = f"Sample: {sample}"
                                ai_resp = deepinfra_chat(sys_prompt, user_prompt, max_tokens=180)
                                ai_text = ai_resp.get("text", "") if isinstance(ai_resp, dict) else str(ai_resp)
                                ai_summ.append({"sheet": nm, "summary": ai_text})
                        pd.DataFrame(ai_summ).to_excel(writer, sheet_name="AI_Summaries", index=False)
                    except Exception:
                        pass
            out.seek(0)
            artifacts["report.xlsx"] = out.getvalue()
            st.success("Excel report built.")
        except Exception as e:
            st.error(f"Excel build failed: {e}")

    # ---------------- Single-file HTML dashboard (self-contained)
    if export_html:
        try:
            # Build a small HTML report with embedded Plotly charts (to_html) and summaries
            parts = []
            parts.append("<!doctype html><html><head><meta charset='utf-8'><title>Vahan Analytics Report</title>")
            parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
            # Simple CSS
            parts.append("<style>body{font-family:Segoe UI,Arial;background:#f7f9fb;color:#111;padding:20px} .kpi{display:flex;gap:20px} .kpi .card{background:#fff;padding:12px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.06)}</style></head><body>")
            parts.append(f"<h1>Vahan Analytics — Report</h1><p>Generated: {datetime.now().isoformat()}</p>")
            # KPIs
            parts.append("<div class='kpi'>")
            for label, val in [(f"Prev Year ({PREV_YEAR})", prev_total), (f"This Year ({THIS_YEAR})", this_total), (f"Pred Next ({NEXT_YEAR})", predicted_next_total)]:
                parts.append(f"<div class='card'><h3>{label}</h3><p style='font-size:20px;font-weight:700'>{int(val):,}</p></div>")
            parts.append("</div><hr>")
            # Add small Plotly chart HTML
            if monthly_totals is not None:
                fc = forecast_monthly(monthly_totals, periods=forecast_horizon)
                plot = None
                if fc is not None:
                    df_plot = monthly_totals.reset_index().rename(columns={'index':'month','value':'actual'})
                    fc_r = fc.reset_index().rename(columns={'index':'month'})
                    dfp = pd.concat([df_plot.assign(type='actual').rename(columns={'actual':'value'}), fc_r.assign(type='predicted').rename(columns={'yhat':'value'})], ignore_index=True)
                    fig = px.line(dfp, x='month', y='value', color='type', title='Monthly Real vs Predicted')
                    plot = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                else:
                    fig = px.line(monthly_totals.reset_index().rename(columns={'index':'month','value':'value'}), x='month', y='value', title='Monthly Actuals')
                    plot = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                parts.append("<h2>Monthly Trend</h2>")
                parts.append(plot)
            parts.append("<hr><h2>AI Executive Summary</h2><pre>{}</pre>".format(one_page_summary if 'one_page_summary' in globals() else "No summary"))
            parts.append("</body></html>")
            html_blob = "\n".join(parts).encode("utf-8")
            artifacts["report.html"] = html_blob
            st.success("HTML dashboard generated.")
        except Exception as e:
            st.error(f"HTML build failed: {e}")

    # ---------------- CSV for Power BI / Tableau (flat)
    if export_csv:
        try:
            flat = []
            for name, df in {
                "Category": df_cat, "Top Makers": df_mk, "Trend": df_trend,
                "Top5Revenue": df_top5_rev, "RevenueTrend": df_rev_trend
            }.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    d = df.copy()
                    d["__source"] = name
                    flat.append(d)
            if flat:
                df_flat = pd.concat(flat, ignore_index=True, sort=False)
                csv_bytes = df_flat.to_csv(index=False).encode("utf-8")
                artifacts["vahan_powerbi_tableau.csv"] = csv_bytes
                st.success("CSV for BI exported.")
        except Exception as e:
            st.error(f"CSV export failed: {e}")

    # ---------------- JSON export
    if export_json:
        try:
            # Convert pandas frames to JSON-serializable
            json_bundle = {}
            for name, df in {
                "Category": df_cat, "Top Makers": df_mk, "Trend": df_trend,
                "Top5Revenue": df_top5_rev, "RevenueTrend": df_rev_trend,
                "ForecastMonthly": (fc.reset_index().rename(columns={'index':'date','yhat':'value'}) if 'fc' in locals() and fc is not None else pd.DataFrame())
            }.items():
                try:
                    json_bundle[name] = json.loads(df.to_json(orient='records', date_format='iso'))
                except Exception:
                    json_bundle[name] = []
            jbytes = json.dumps(json_bundle, indent=2, default=str).encode("utf-8")
            artifacts["vahan_data.json"] = jbytes
            st.success("JSON export ready.")
        except Exception as e:
            st.error(f"JSON export failed: {e}")

    # ---------------- Tkinter preview script (simple) - optional
    if export_tk:
        try:
            tk_script = textwrap.dedent(f"""
            # Auto-generated Tkinter preview for Vahan JSON
            import json, sys, tkinter as tk, webbrowser, tempfile, os
            from tkinter import ttk
            data_path = 'vahan_data.json'  # replace with your path
            with open(data_path,'r',encoding='utf-8') as f:
                data = json.load(f)
            root = tk.Tk()
            root.title('Vahan Data Preview')
            root.geometry('1000x700')
            txt = tk.Text(root)
            txt.pack(fill='both',expand=True)
            txt.insert('1.0', json.dumps(data, indent=2))
            root.mainloop()
            """).strip()
            artifacts["tk_preview.py"] = tk_script.encode("utf-8")
            st.success("Tkinter preview script generated.")
        except Exception as e:
            st.error(f"Tkinter script creation failed: {e}")

    # ---------------- Build ZIP if requested (or bundle option)
    if export_zip_all or export_bundle:
        try:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, blob in artifacts.items():
                    zf.writestr(name, blob)
            zip_buf.seek(0)
            artifacts["vahan_report_bundle.zip"] = zip_buf.getvalue()
            st.download_button("📦 Download bundle ZIP", zip_buf.getvalue(), file_name=f"vahan_bundle_{TODAY}.zip", mime="application/zip")
        except Exception as e:
            st.error(f"Bundle ZIP creation failed: {e}")

    # ---------------- Individual downloads for artifacts created
    for fname, blob in artifacts.items():
        try:
            mime = "application/octet-stream"
            if fname.endswith(".xlsx"):
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif fname.endswith(".html"):
                mime = "text/html"
            elif fname.endswith(".csv"):
                mime = "text/csv"
            elif fname.endswith(".json"):
                mime = "application/json"
            elif fname.endswith(".py"):
                mime = "text/x-python"
            st.download_button(f"⬇️ Download {fname}", data=blob, file_name=fname, mime=mime)
        except Exception as e:
            st.warning(f"Failed to create download for {fname}: {e}")

    st.success("All requested exports generated. Check downloads above.")

# -------------------------
# End of MAXED exports block
# -------------------------

# ================================================================
# 🏁 MAXED FOOTER — COMPLETE ANALYTICS & EXECUTIVE SUMMARY
# ================================================================
st.markdown("---")
st.subheader("📊 Full Dashboard Summary & Predictive Analytics")

# --- KPI Metrics ---
if not df_trend.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Registrations (All Years)", int(df_trend['value'].sum()))
    with col2:
        st.metric("Current Year", df_trend['year'].max())
    with col3:
        st.metric("Previous Year", df_trend['year'].min())
    with col4:
        st.metric("Next Year (Predicted)", df_trend['year'].max() + 1)

    # --- Multi-Year Aggregations ---
    df_trend['year'] = pd.to_numeric(df_trend['year'], errors='coerce')
    df_trend['month'] = pd.to_datetime(df_trend['date']).dt.month if 'date' in df_trend.columns else None
    df_trend['quarter'] = pd.to_datetime(df_trend['date']).dt.quarter if 'date' in df_trend.columns else None

    year_summary = df_trend.groupby('year')['value'].sum().reset_index(name='total')
    month_summary = df_trend.groupby(['year', 'month'])['value'].sum().reset_index(name='total')
    quarter_summary = df_trend.groupby(['year', 'quarter'])['value'].sum().reset_index(name='total')

    st.markdown("### 📅 Multi-Period Trends")
    st.write("**Yearly Summary**", year_summary)
    st.write("**Quarterly Summary**", quarter_summary)
    st.write("**Monthly Summary**", month_summary)

    # --- Predictive Forecast (ARIMA-style simple regression fallback) ---
    with st.spinner("Predicting Next Year..."):
        try:
            from sklearn.linear_model import LinearRegression
            X = year_summary[['year']]
            y = year_summary['total']
            model = LinearRegression().fit(X, y)
            next_year = year_summary['year'].max() + 1
            pred_next = model.predict([[next_year]])[0]
            growth_pct = ((pred_next - y.iloc[-1]) / y.iloc[-1]) * 100
            st.metric("Predicted Next Year Registrations", f"{int(pred_next):,}", f"{growth_pct:.2f}% ↑" if growth_pct > 0 else f"{growth_pct:.2f}% ↓")
        except Exception as e:
            st.warning(f"Prediction failed: {e}")

    # --- Visuals ---
    st.markdown("### 📈 Comparative Charts")
    chart_year = alt.Chart(year_summary).mark_line(point=True).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('total:Q', title='Registrations'),
        tooltip=['year', 'total']
    ).properties(title="Yearly Trend Comparison").interactive()

    st.altair_chart(chart_year, use_container_width=True)

    if not quarter_summary.empty:
        chart_quarter = alt.Chart(quarter_summary).mark_bar().encode(
            x=alt.X('quarter:O', title='Quarter'),
            y=alt.Y('total:Q', title='Registrations'),
            color='year:N',
            tooltip=['year', 'quarter', 'total']
        ).properties(title="Quarterly Comparison").interactive()
        st.altair_chart(chart_quarter, use_container_width=True)

    if not month_summary.empty:
        chart_month = alt.Chart(month_summary).mark_area(opacity=0.5).encode(
            x=alt.X('month:O', title='Month'),
            y=alt.Y('total:Q', title='Registrations'),
            color='year:N',
            tooltip=['year', 'month', 'total']
        ).properties(title="Monthly Comparison").interactive()
        st.altair_chart(chart_month, use_container_width=True)

# --- Extended KPI metrics ---
if not df_top5_rev.empty:
    with st.expander("💰 Revenue Insights"):
        top_state = df_top5_rev.iloc[0].get('label', 'N/A')
        st.metric("Top Revenue State", top_state)
        st.bar_chart(df_top5_rev.set_index('label')['value'])

if latest_yoy is not None:
    st.metric("Latest YoY%", f"{latest_yoy:.2f}%")
if latest_qoq is not None:
    st.metric("Latest QoQ%", f"{latest_qoq:.2f}%")

# --- AI Executive Summary + Forecast ---
if enable_ai:
    st.markdown("### 🤖 AI Executive Summary")
    with st.spinner("Generating full predictive AI analysis..."):
        try:
            system = """You are a senior data scientist. 
            Provide an analytical executive summary using the following context: 
            Include trend direction, anomalies, YoY/QoQ changes, top states, and one forecast prediction for next year. 
            Output: concise 6-sentence executive summary + 3 strategic recommendations."""
            
            ctx = {
                "total_registrations": int(df_trend['value'].sum()) if not df_trend.empty else None,
                "year_summary": year_summary.to_dict(orient='records'),
                "latest_yoy": float(latest_yoy) if latest_yoy is not None else None,
                "latest_qoq": float(latest_qoq) if latest_qoq is not None else None,
                "top_revenue_state": df_top5_rev.iloc[0]['label'] if not df_top5_rev.empty else None,
            }

            user = f"Context JSON: {json.dumps(ctx, default=str)}"
            ai_resp = deepinfra_chat(system, user, max_tokens=500)
            if "text" in ai_resp:
                st.markdown(ai_resp["text"])
        except Exception as e:
            st.error(f"AI analysis failed: {e}")

# --- Export Options ---
st.markdown("### 📤 Export Data & Reports")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    csv = df_trend.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Export CSV", csv, "registrations.csv", "text/csv")
with col2:
    st.download_button("⬇️ Export Excel", df_trend.to_excel("registrations.xlsx", index=False, engine="openpyxl"))
with col3:
    st.download_button("⬇️ Export JSON", json.dumps(df_trend.to_dict(orient="records")), "registrations.json")
with col4:
    st.download_button("⬇️ Export HTML", df_trend.to_html(), "registrations.html")
with col5:
    st.download_button("⬇️ Export Power BI / Tableau Compatible CSV", csv, "registrations_powerbi.csv")

# --- Final Visual ---
st.success("✅ All metrics, analytics, and AI summaries loaded successfully.")
