from __future__ import annotations

# =====================================================
# 🚀 VAHAN API MODULE — MAXED EDITION
# =====================================================
import os
import io
import json
import time
import uuid
import math
import random
import logging
import traceback
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps, lru_cache
from requests.adapters import HTTPAdapter, Retry
import streamlit as st

# =====================================================
# 🕒 Universal IST Logger
# =====================================================
def log_ist(msg: str):
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")
    print(f"[IST {ist_time}] {msg}")

# # =====================================================
# # ⚙️ API CONFIG
# # =====================================================
# BASE_URL = "https://vahan.parivahan.gov.in/vahan/vahandashboard"
# HEADERS = {
#     "User-Agent": f"VahanClient/{uuid.uuid4().hex[:8]}",
#     "Accept": "application/json, text/plain, */*",
#     "Referer": "https://vahan.parivahan.gov.in/vahan4dashboard/",
# }
# TIMEOUT = 25
# MAX_RETRIES = 3

# # =====================================================
# # 🧱 Robust Session with Retry
# # =====================================================
# session = requests.Session()
# retries = Retry(
#     total=MAX_RETRIES,
#     backoff_factor=1.2,
#     status_forcelist=[429, 500, 502, 503, 504],
# )
# session.mount("https://", HTTPAdapter(max_retries=retries))

# # =====================================================
# # ♻️ Streamlit Caching Layer (1 hr TTL)
# # =====================================================
# @st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
# def cached_request(endpoint: str, params=None):
#     url = f"{BASE_URL}/{endpoint.strip('/')}"
#     log_ist(f"🌐 API CALL (cached): {url}")

#     try:
#         resp = session.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
#         resp.raise_for_status()
#         data = resp.json()

#         if not data:
#             raise ValueError("Empty response")

#         log_ist(f"✅ SUCCESS: {endpoint} [{len(str(data))} chars]")
#         return data

#     except Exception as e:
#         logging.error(f"❌ API Error ({endpoint}): {e}")
#         log_ist(f"⚠️ Retrying via fallback for {endpoint}")
#         return playwright_fallback(endpoint, params=params)

# # =====================================================
# # 🕸️ Playwright Fallback (Headless Fetch)
# # =====================================================
# def playwright_fallback(endpoint, params=None):
#     """Fallback if direct API fails — uses Playwright to render."""
#     try:
#         from playwright.sync_api import sync_playwright
#     except ImportError:
#         log_ist("⚠️ Playwright not installed — skipping fallback")
#         return {}

#     try:
#         with sync_playwright() as p:
#             browser = p.firefox.launch(headless=True)
#             context = browser.new_context()
#             page = context.new_page()
#             url = f"{BASE_URL}/{endpoint.strip('/')}"
#             page.goto(url, wait_until="networkidle", timeout=40000)
#             time.sleep(2)
#             content = page.content()
#             browser.close()

#             if "{" not in content:
#                 raise ValueError("Non-JSON HTML response")

#             json_str = content.split("{", 1)[1].rsplit("}", 1)[0]
#             json_data = json.loads("{" + json_str + "}")
#             log_ist(f"🧩 Playwright fallback succeeded for {endpoint}")
#             return json_data

#     except Exception as e:
#         logging.error(f"💀 Playwright fallback failed: {e}")
#         return {}

# # =====================================================
# # 🧠 Safe Wrapper (Error-Proof Execution)
# # =====================================================
# def safe_exec(fn):
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         try:
#             return fn(*args, **kwargs)
#         except Exception as e:
#             logging.error(f"⚠️ Exception in {fn.__name__}: {e}")
#             traceback.print_exc()
#             st.error(f"An internal error occurred in {fn.__name__}")
#             return None
#     return wrapper

# # =====================================================
# # 📦 Utility: to_df() — JSON → DataFrame
# # =====================================================
# @safe_exec
# def to_df(data):
#     if not data:
#         return pd.DataFrame()
#     if isinstance(data, dict):
#         for key in ["result", "data", "records"]:
#             if key in data and isinstance(data[key], list):
#                 return pd.DataFrame(data[key])
#         return pd.DataFrame([data])
#     elif isinstance(data, list):
#         return pd.DataFrame(data)
#     return pd.DataFrame()

# # =====================================================
# # 🔥 Unified Fetch Function
# # =====================================================
# @safe_exec
# def fetch_vahan(endpoint, params=None):
#     """Primary function to fetch data with retry + cache + fallback."""
#     cache_key = f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}"
#     log_ist(f"📡 Fetching endpoint: {endpoint} | params: {params}")

#     data = cached_request(endpoint, params)
#     if not data:
#         st.warning(f"⚠️ No data returned for {endpoint}")
#     return data

# # =====================================================
# # 🧩 Example API Endpoints
# # =====================================================
# @safe_exec
# def fetch_registration_trend(state_code=None):
#     params = {"stateCode": state_code} if state_code else None
#     return fetch_vahan("registrationtrend", params=params)

# @safe_exec
# def fetch_category_distribution(state_code=None):
#     params = {"stateCode": state_code} if state_code else None
#     return fetch_vahan("categorywise", params=params)

# @safe_exec
# def fetch_top_makers(year=None):
#     params = {"year": year or datetime.now().year}
#     return fetch_vahan("topmakers", params=params)

# # =====================================================
# # 🧾 Example Usage (Test Mode)
# # =====================================================
# if __name__ == "__main__":
#     log_ist("🧪 Running self-test for MAXED API")
#     res = fetch_registration_trend()
#     df = to_df(res)
#     print(df.head())
#     log_ist("✅ Test completed")

import os
import sys
import time
import uuid
import json
import math
import random
import platform
import traceback
import logging
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps
from typing import Any, Callable, Optional, Dict

# Streamlit & core
import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# === Environment / dotenv (optional) ===
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# === Detect optional libs ===
try:
    import requests
except Exception:
    requests = None

try:
    import aiohttp
except Exception:
    aiohttp = None

try:
    import tenacity
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:
    tenacity = None

try:
    from diskcache import Cache
    _DISKCACHE_AVAILABLE = True
    diskcache = Cache(".cache")
except Exception:
    _DISKCACHE_AVAILABLE = False
    diskcache = None

try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except Exception:
    sync_playwright = None
    _PLAYWRIGHT_AVAILABLE = False

# ML libs
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import torch, transformers  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# === Timezone / IST helpers ===
IST_ZONE = ZoneInfo("Asia/Kolkata")

def now_ist(fmt: str = "%Y-%m-%d %I:%M:%S %p") -> str:
    return datetime.now(IST_ZONE).strftime(fmt)

# === Logging forced to IST ===
class ISTFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, IST_ZONE)
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

# configure root logger (idempotent)
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    logging.basicConfig(level=logging.INFO)
for h in _root_logger.handlers:
    h.setFormatter(ISTFormatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))

def log_ist(msg: str, level: str = "info") -> None:
    """Console + logger friendly IST-stamped message (safe for cloud)."""
    text = f"[IST {now_ist()}] {msg}"
    if level.lower() == "debug":
        logging.debug(text)
    elif level.lower() == "warning":
        logging.warning(text)
    elif level.lower() == "error":
        logging.error(text)
    else:
        logging.info(text)
    # Also print for immediate console visibility
    print(text)

log_ist("✅ MAXED INIT booting...")

# === Session / user identity (Streamlit-safe) ===
if "maxed_user_id" not in st.session_state:
    st.session_state["maxed_user_id"] = f"user_{uuid.uuid4().hex[:8]}_{random.randint(1000,9999)}"
MAXED_USER_ID = st.session_state["maxed_user_id"]
log_ist(f"Session identity: {MAXED_USER_ID}")

# === Global config ===
DEFAULT_TIMEOUT = int(os.getenv("MAXED_TIMEOUT", "30"))
DEFAULT_CACHE_TTL = int(os.getenv("MAXED_CACHE_TTL_SEC", str(60 * 60)))  # 1 hour
DEFAULT_MAX_RETRIES = int(os.getenv("MAXED_MAX_RETRIES", "4"))

# === Hybrid caching decorator (diskcache preferred, fallback to st.cache_data) ===
def cached(ttl: int = DEFAULT_CACHE_TTL, key_fn: Optional[Callable[..., str]] = None):
    """
    Decorator that uses diskcache (if available) else streamlit.cache_data.
    key_fn receives (args, kwargs) and must return string key if provided.
    """
    def decorator(fn):
        if _DISKCACHE_AVAILABLE and diskcache is not None:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    if key_fn:
                        key = key_fn(*args, **kwargs)
                    else:
                        # stable key: function name + serialized args
                        key = f"{fn.__name__}:{json.dumps({'args': args, 'kwargs': kwargs}, default=str, sort_keys=True)}"
                    # diskcache expiration handling
                    if key in diskcache:
                        created = diskcache.created(key)
                        if time.time() - created < ttl:
                            log_ist(f"💾 diskcache HIT: {fn.__name__}")
                            return diskcache[key]
                    val = fn(*args, **kwargs)
                    diskcache.set(key, val)
                    log_ist(f"💾 diskcache SET: {fn.__name__}")
                    return val
                except Exception as e:
                    log_ist(f"cache wrapper error (diskcache): {e}", "warning")
                    return fn(*args, **kwargs)
            return wrapper
        else:
            # fallback to Streamlit cache_data (coarse-grained)
            st_cache = st.cache_data(ttl=ttl, show_spinner=False)
            return st_cache(fn)
    return decorator

def clear_all_caches() -> None:
    "Clear diskcache and streamlit caches where available."
    if _DISKCACHE_AVAILABLE and diskcache is not None:
        try:
            diskcache.clear()
            log_ist("🧹 diskcache cleared")
        except Exception as e:
            log_ist(f"diskcache clear failed: {e}", "warning")
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        log_ist("🧹 streamlit caches cleared")
    except Exception:
        pass

# === Auto-reload watcher (optional; no-op on cloud if watchdog missing) ===
def start_autoreload(watch_dir: str = ".", debounce_seconds: float = 2.5):
    """
    Starts a background watchdog to auto-clear caches and attempt st.rerun() on local dev.
    Safe: does nothing if watchdog not installed or if running on Streamlit Cloud where OS permissions might not allow it.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except Exception:
        log_ist("watchdog not available — autoreload disabled", "debug")
        return None

    class _Handler(FileSystemEventHandler):
        def __init__(self):
            self._last = 0.0
        def on_any_event(self, event):
            now = time.time()
            if now - self._last < debounce_seconds:
                return
            self._last = now
            try:
                log_ist("🔁 Code change detected — clearing caches and attempting rerun")
                clear_all_caches()
                time.sleep(0.2)
                try:
                    st.experimental_rerun()
                except Exception:
                    # older streamlit versions / cloud may not support rerun call here
                    log_ist("st.experimental_rerun() not permitted in this environment", "warning")
            except Exception as e:
                log_ist(f"Auto-reload handler error: {e}", "warning")

    try:
        observer = Observer()
        handler = _Handler()
        observer.schedule(handler, path=watch_dir, recursive=True)
        observer.daemon = True
        observer.start()
        log_ist(f"Auto-reload watcher started on {watch_dir}")
        return observer
    except Exception as e:
        log_ist(f"Could not start file watcher: {e}", "warning")
        return None

# Call start_autoreload only when in local dev (env var opt-in)
if os.getenv("MAXED_ENABLE_AUTORELOAD", "false").lower() in ("1", "true", "yes"):
    _watcher = start_autoreload(os.getenv("MAXED_AUTORELOAD_DIR", "."))

# === HTTP clients & resilient fetchers ===
# requests.Session with Retry adapter
def create_requests_session(max_retries: int = DEFAULT_MAX_RETRIES, backoff: float = 1.5):
    sess = requests.Session() if requests else None
    if sess:
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry as URLLibRetry
            r = URLLibRetry(
                total=max_retries,
                read=max_retries,
                connect=max_retries,
                backoff_factor=backoff,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET","POST","PUT","DELETE","HEAD","OPTIONS"])
            )
            sess.mount("https://", HTTPAdapter(max_retries=r))
            sess.mount("http://", HTTPAdapter(max_retries=r))
        except Exception:
            pass
    return sess

_requests_session = create_requests_session()

# Tenacity-based decorator if available
def tenacity_retry_decorator(max_attempts: int = DEFAULT_MAX_RETRIES):
    if tenacity is None:
        def _noop(fn):
            return fn
        return _noop

    def _decor(fn):
        return tenacity.retry(
            stop=tenacity.stop_after_attempt(max_attempts),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True
        )(fn)
    return _decor

# Generic safe fetch (sync) with optional Playwright fallback
@tenacity_retry_decorator()
def safe_get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    """
    Robust synchronous GET:
     - uses requests with session & retries
     - returns parsed JSON where possible, or raw text as fallback
     - if 403/blocked and PLAYWRIGHT available, tries headless browser fallback
    """
    if requests is None:
        raise RuntimeError("requests not installed in this environment")

    hdrs = headers.copy() if headers else {}
    hdrs.setdefault("User-Agent", os.getenv("MAXED_USER_AGENT", f"maxed-client/{uuid.uuid4().hex[:6]}"))
    try:
        log_ist(f"HTTP GET {url} params={params}")
        resp = _requests_session.get(url, params=params, headers=hdrs, timeout=timeout)
        status = getattr(resp, "status_code", None)
        if status == 403 and _PLAYWRIGHT_AVAILABLE:
            log_ist("403 detected; attempting Playwright fallback", "warning")
            return playwright_get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # Attempt JSON decode
        try:
            return resp.json()
        except Exception:
            return {"raw_text": resp.text[:2000]}
    except Exception as e:
        log_ist(f"safe_get error: {e}", "warning")
        # If Playwright available, try fallback
        if _PLAYWRIGHT_AVAILABLE:
            try:
                return playwright_get(url, params=params, headers=headers, timeout=timeout)
            except Exception as e2:
                log_ist(f"playwright fallback failed: {e2}", "error")
        raise

def playwright_get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    """
    Uses Playwright to open URL and extract JSON payload from body innerText when APIs are blocked.
    Returns parsed JSON when possible or raw text snippet.
    """
    if not _PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not available")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=headers.get("User-Agent") if headers and "User-Agent" in headers else None)
            page = context.new_page()
            log_ist(f"Playwright navigating to: {url}")
            page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            body = page.evaluate("() => document.body.innerText")
            browser.close()
            try:
                return json.loads(body)
            except Exception:
                return {"raw_text_snippet": body[:2000]}
    except Exception as e:
        log_ist(f"playwright_get error: {e}", "error")
        raise

# Async fetch helper using aiohttp (if installed)
async def async_get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    if aiohttp is None:
        raise RuntimeError("aiohttp not installed")
    hdrs = headers.copy() if headers else {}
    hdrs.setdefault("User-Agent", os.getenv("MAXED_USER_AGENT", f"maxed-async/{uuid.uuid4().hex[:6]}"))
    try:
        async with aiohttp.ClientSession(headers=hdrs) as sess:
            async with sess.get(url, params=params, timeout=timeout) as resp:
                text = await resp.text()
                try:
                    return await resp.json()
                except Exception:
                    return {"raw_text": text[:2000]}
    except Exception as e:
        log_ist(f"async_get error: {e}", "warning")
        raise

# === Utility helpers ===
def safe_exec(fn: Callable):
    """Decorator to catch exceptions, render friendly Streamlit error, and return None."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            trace = traceback.format_exc()
            log_ist(f"Exception in {fn.__name__}: {e}\n{trace}", "error")
            try:
                st.error(f"Internal error in `{fn.__name__}` — check logs.")
            except Exception:
                pass
            return None
    return wrapper

def export_df(df: pd.DataFrame, prefix: str = "export") -> Dict[str, str]:
    """
    Export DataFrame to multiple formats and return paths (or string blobs).
    In Cloud, returns in-memory bytes as strings for download buttons in Streamlit.
    """
    if df is None:
        return {}
    ts = datetime.now(IST_ZONE).strftime("%Y%m%d_%H%M%S")
    name_base = f"{prefix}_{ts}"
    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    # Excel (in-memory)
    try:
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        xlsx_bytes = buf.getvalue()
    except Exception:
        xlsx_bytes = None
    # JSON
    json_str = df.to_json(orient="records", indent=2, date_format="iso")
    return {"csv_name": f"{name_base}.csv", "csv_bytes": csv_bytes,
            "xlsx_name": (f"{name_base}.xlsx" if xlsx_bytes else None), "xlsx_bytes": xlsx_bytes,
            "json_name": f"{name_base}.json", "json_str": json_str}

# === App boot banner for Streamlit UI ===
def app_boot_banner(show_ui: bool = True):
    python_ver = platform.python_version()
    st_ver = getattr(st, "__version__", "unknown")
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    banner_html = f"""
    <div style='background:linear-gradient(90deg,#0f172a,#0ea5e9);padding:12px;border-radius:10px;color:white'>
      <strong>PARIVAHAN ANALYTICS — MAXED</strong> &nbsp; • &nbsp; Booted @ {now_ist()} IST<br/>
      <small>user: {MAXED_USER_ID} • python {python_ver} • streamlit {st_ver} • {os_info}</small>
    </div>
    """
    if show_ui:
        try:
            st.markdown(banner_html, unsafe_allow_html=True)
        except Exception:
            pass
    log_ist("App boot banner displayed")

# === Example small usage helpers ===
@cached(ttl=3600)
def example_cached_fetch(url: str, params: Optional[Dict[str, Any]] = None):
    """Simple cached GET wrapper that returns JSON or text snippet."""
    return safe_get(url, params=params)

@safe_exec
def health_check():
    """Return environment & feature checks."""
    info = {
        "time_ist": now_ist(),
        "user_id": MAXED_USER_ID,
        "python": platform.python_version(),
        "streamlit": getattr(st, "__version__", "unknown"),
        "diskcache": _DISKCACHE_AVAILABLE,
        "playwright": _PLAYWRIGHT_AVAILABLE,
        "aiohttp": aiohttp is not None,
        "prophet": PROPHET_AVAILABLE,
        "torch": TORCH_AVAILABLE,
    }
    return info

# Boot banner (UI + console)
try:
    app_boot_banner(show_ui=True)
except Exception:
    log_ist("Failed to render app banner in UI", "warning")

log_ist("MAXED INIT completed — ready.")

# === Exports (for convenience import) ===
__all__ = [
    "now_ist", "log_ist", "MAXED_USER_ID", "DEFAULT_TIMEOUT", "DEFAULT_CACHE_TTL",
    "cached", "clear_all_caches", "start_autoreload", "safe_get",
    "async_get", "playwright_get", "safe_exec", "export_df", "example_cached_fetch",
    "health_check", "app_boot_banner"
]

# =====================================================
# 🚀 VAHAN MODULES — MAXED IMPORT SYSTEM
# =====================================================
import importlib, traceback

def _safe_import(module_name, fallback=None):
    """Safely import a module and handle missing/failed imports gracefully."""
    try:
        mod = importlib.import_module(module_name)
        print(f"✅ Loaded {module_name}")
        return mod
    except Exception as e:
        print(f"⚠️ Failed to import {module_name}: {e}")
        traceback.print_exc()
        return fallback

# =====================================================
# ✅ CORE VAHAN MODULES (MAXED)
# =====================================================
vahan_api     = _safe_import("vahan.api")
vahan_parsing = _safe_import("vahan.parsing")
vahan_metrics = _safe_import("vahan.metrics")
vahan_charts  = _safe_import("vahan.charts")

# -----------------------------------------------------
# 📦 SAFE FUNCTION EXTRACTION WITH FALLBACKS
# -----------------------------------------------------
def _extract(mod, name, default=lambda *a, **kw: None):
    return getattr(mod, name, default) if mod else default

# 🔌 API + PARAM HELPERS
build_params  = _extract(vahan_api, "build_params")
get_json      = _extract(vahan_api, "get_json")

# 🧩 PARSING UTILITIES
to_df              = _extract(vahan_parsing, "to_df")
normalize_trend    = _extract(vahan_parsing, "normalize_trend")
parse_duration_table = _extract(vahan_parsing, "parse_duration_table")
parse_top5_revenue = _extract(vahan_parsing, "parse_top5_revenue")
parse_revenue_trend = _extract(vahan_parsing, "parse_revenue_trend")
parse_makers       = _extract(vahan_parsing, "parse_makers")

# 📊 METRIC CALCULATIONS
compute_yoy = _extract(vahan_metrics, "compute_yoy")
compute_qoq = _extract(vahan_metrics, "compute_qoq")

# 🎨 VISUALIZATION COMPONENTS
bar_from_df   = _extract(vahan_charts, "bar_from_df")
pie_from_df   = _extract(vahan_charts, "pie_from_df")
line_from_trend = _extract(vahan_charts, "line_from_trend")
show_metrics  = _extract(vahan_charts, "show_metrics")
show_tables   = _extract(vahan_charts, "show_tables")

# =====================================================
# 🔁 AUTO-RELOAD (OPTIONAL)
# =====================================================
def reload_vahan():
    """Reload all Vahan modules without restarting Streamlit."""
    for mod in [vahan_api, vahan_parsing, vahan_metrics, vahan_charts]:
        try:
            if mod:
                importlib.reload(mod)
                print(f"🔁 Reloaded {mod.__name__}")
        except Exception as e:
            print(f"⚠️ Failed to reload {mod}: {e}")
            traceback.print_exc()

# =====================================================
# 🧠 MAXED DIAGNOSTICS
# =====================================================
print("\n" + "=" * 80)
print("🚀 VAHAN MODULES — MAXED import system active")
print(f"📦 API: {'✅' if vahan_api else '❌'} | Parsing: {'✅' if vahan_parsing else '❌'} | "
      f"Metrics: {'✅' if vahan_metrics else '❌'} | Charts: {'✅' if vahan_charts else '❌'}")
print("=" * 80 + "\n")

# =====================================================
# 🔧 MAXED+ STATELESS FETCHER — FULL HTTP HARDENING
# =====================================================
import threading
import collections
import random
import time
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
from typing import Optional, Dict

# =====================================================
# 🌐 MAXED API CONFIG SAFEGUARD
# =====================================================
try:
    API_CONFIG
except NameError:
    API_CONFIG = {}

# ensure mirrors field exists
for svc in API_CONFIG.values():
    if isinstance(svc, dict):
        svc.setdefault("mirrors", [])

# =====================================================
# ⚙️ CONFIGURATION KNOBS
# =====================================================
MAX_ATTEMPTS = 6
BASE_BACKOFF = 1.4
JITTER = 0.35
KEY_COOLDOWN_SECONDS = 300        # cooldown after mild fail
KEY_BLOCK_SECONDS = 3600          # block after repeated fail
ENDPOINT_CIRCUIT_BREAK_SECONDS = 120
CIRCUIT_BREAK_THRESHOLD = 3
MAX_PROXY_USAGE_RATIO = 0.35
DISABLE_CACHE_FOR_STATELESS = False

# =====================================================
# 🧩 HELPERS
# =====================================================
def _now() -> float: return time.time()

def _is_key_ok(k: str) -> bool:
    t = _now()
    return not (
        (k in _key_blocks and _key_blocks[k] > t)
        or (k in _key_cooldowns and _key_cooldowns[k] > t)
    )

def _cooldown_key(k: str, sec=KEY_COOLDOWN_SECONDS):
    with _lock: _key_cooldowns[k] = _now() + sec
    log_ist(f"🔒 cooldown {sec}s for key {k[:6]}")

def _block_key(k: str, sec=KEY_BLOCK_SECONDS):
    with _lock: _key_blocks[k] = _now() + sec
    log_ist(f"⛔ blocked {sec}s for key {k[:6]}")

def _fail_endpoint(ep: str):
    now = _now()
    dq = _endpoint_failures.setdefault(ep, collections.deque(maxlen=20))
    dq.append(now)
    recent = [t for t in dq if t > now - ENDPOINT_CIRCUIT_BREAK_SECONDS]
    if len(recent) >= CIRCUIT_BREAK_THRESHOLD:
        _endpoint_circuit[ep] = now + ENDPOINT_CIRCUIT_BREAK_SECONDS
        log_ist(f"🛑 circuit trip for {ep}")

def _ep_ok(ep: str) -> bool:
    return _now() > _endpoint_circuit.get(ep, 0)

def _mirror(api: str) -> Optional[str]:
    cfg = API_CONFIG.get(api, {})
    cand = [cfg.get("base", "")] + list(cfg.get("mirrors", []) or [])
    random.shuffle(cand)
    for c in cand:
        if _ep_ok(c): return c
    return None

def _backoff(a: int, s: float = BASE_BACKOFF):
    t = (s ** a) + (random.random() * JITTER * s)
    log_ist(f"⏳ backoff {t:.1f}s (try {a})")
    time.sleep(t)

def _pick_token(api: str) -> Optional[str]:
    ks = list(API_CONFIG.get(api, {}).get("keys", []) or [])
    random.shuffle(ks)
    for k in ks:
        if _is_key_ok(k):
            return k
    return None

def _mark_fail(api: str, k: Optional[str], code: Optional[int] = None):
    if not k: return
    if code == 401: _block_key(k)
    elif code == 403: _cooldown_key(k, KEY_COOLDOWN_SECONDS)
    elif code == 429: _cooldown_key(k, KEY_COOLDOWN_SECONDS // 2)
    else: _cooldown_key(k, 60)
    log_ist(f"⚠️ fail key {k[:6]} code={code}")

# =====================================================
# 🌐 MAIN FETCHER
# =====================================================
def fetch_api_maxed(api_name: str,
                    endpoint: str,
                    params: dict = None,
                    method: str = "GET",
                    json_body: dict = None,
                    allow_redirects: bool = True,
                    disable_cache: bool = DISABLE_CACHE_FOR_STATELESS,
                    max_attempts: int = MAX_ATTEMPTS) -> dict:
    """MAXED+ Stateless HTTP fetcher with key-rotation, mirrors, and circuit breaker."""
    cfg = API_CONFIG.get(api_name, {})
    auth_type = cfg.get("auth_type", "bearer")

    base = _mirror(api_name)
    if not base:
        log_ist(f"❌ all endpoints tripped for {api_name}")
        return {}

    url = f"{base.rstrip('/')}/{endpoint.lstrip('/')}"
    attempt, last_exc = 0, None

    while attempt < max_attempts:
        attempt += 1
        token = _pick_token(api_name)
        if not token:
            log_ist(f"⚠️ no usable token for {api_name} try#{attempt}")
            _backoff(attempt)
            continue

        headers = build_spoofed_headers(api_name)
        if auth_type == "bearer": headers["Authorization"] = f"Bearer {token}"

        session = requests.Session()
        session.cookies.clear()
        session.trust_env = False
        proxy = _choose_proxy()
        proxies = {"http": proxy, "https": proxy} if proxy else None
        if proxy and random.random() < MAX_PROXY_USAGE_RATIO:
            log_ist(f"🔀 proxy: {proxy}")

        req_params = dict(params or {})
        if auth_type == "param": req_params["apikey"] = token

        try:
            log_ist(f"🌐 [{api_name.upper()}] try#{attempt} -> {url}")
            if method.upper() == "GET":
                r = session.get(url, headers=headers, params=req_params,
                                timeout=30, allow_redirects=allow_redirects, proxies=proxies)
            else:
                r = session.request(method.upper(), url, headers=headers,
                                    params=req_params, json=json_body or {},
                                    timeout=60, allow_redirects=allow_redirects, proxies=proxies)

            c = r.status_code
            if c == 204: return {}
            if 200 <= c < 300:
                try:
                    return r.json()
                except ValueError:
                    return {"text": r.text}

            if c in (400, 401, 403, 404, 405, 429) or 500 <= c < 600:
                _fail_endpoint(url)
                _mark_fail(api_name, token, c)
                if c == 403:
                    m = _mirror(api_name)
                    if m and m.rstrip("/") != base.rstrip("/"):
                        base, url = m, f"{m.rstrip('/')}/{endpoint.lstrip('/')}"
                        log_ist(f"🔁 switched mirror {base}")
                _backoff(attempt)
                continue

            log_ist(f"❗unexpected {c} {url}")
            _fail_endpoint(url)
            _mark_fail(api_name, token, c)
            _backoff(attempt)

        except (Timeout, ConnectionError, RequestException) as e:
            last_exc = e
            _fail_endpoint(url)
            _mark_fail(api_name, token)
            _backoff(attempt)
        finally:
            session.close()

    log_ist(f"❌ exhausted {api_name}/{endpoint} after {max_attempts} tries ({last_exc})")
    return {}

# =====================================================
# 🩺 STREAMLIT DEBUG PANEL
# =====================================================
def maxed_fetcher_status():
    st.subheader("🛠 MAXED+ Fetcher Status")
    st.write("### 🔑 Key Cooldowns")
    for k, t in list(_key_cooldowns.items())[:10]:
        st.write(f"{k[:8]} → until {time.ctime(t)}")
    st.write("### ⛔ Key Blocks")
    for k, t in list(_key_blocks.items())[:10]:
        st.write(f"{k[:8]} → until {time.ctime(t)}")
    st.write("### 🧩 Endpoint Circuits")
    for e, t in _endpoint_circuit.items():
        st.write(f"{e} → tripped until {time.ctime(t)}")
    st.success("✅ MAXED+ fetcher operational")

# =====================================================
# 🚀 VAHAN STREAMLIT BOOT — MAXED++ EDITION
# =====================================================
import os
import platform
import random
import psutil
import socket
from datetime import datetime, date
from zoneinfo import ZoneInfo
import streamlit as st

APP_VERSION = "2025.10-MAXED++"
BUILD_ID = random.randint(10000, 99999)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🚀 Vahan Registrations — MAXED++ Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://parivahan.gov.in",
        "About": "Vahan Analytics — Smart AI-Powered Registration Dashboard (MAXED++ Build)",
    },
)

# ---------------- THEME (CSS) ----------------
st.markdown("""
<style>
.stApp {animation: fadeIn 1s ease-in;}
@keyframes fadeIn {from {opacity:0;} to {opacity:1;}}
.glow {
    font-size:30px;font-weight:700;text-align:center;color:white;
    text-shadow:0px 0px 12px rgba(0,255,255,0.9);
    background:linear-gradient(90deg,#0072ff,#00c6ff);
    padding:14px 18px;border-radius:12px;margin-bottom:18px;
    box-shadow:0px 2px 14px rgba(0,0,0,0.3);
}
.subtext {
    text-align:center;color:#ccc;font-size:15px;
    margin-top:-8px;margin-bottom:25px;
}
.sidebar-title {
    font-size:20px;font-weight:700;
    background:linear-gradient(90deg,#0072ff,#00c6ff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.sidebar-sub {color:#aaa;font-size:13px;margin-top:-8px;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

# ---------------- RANDOM USER/DEVICE SPOOF ----------------
def random_user_agent():
    browsers = ["Chrome", "Firefox", "Edge", "Safari", "Brave"]
    platforms = [
        "Windows NT 10.0; Win64; x64",
        "Macintosh; Intel Mac OS X 13_0",
        "X11; Linux x86_64"
    ]
    browser = random.choice(browsers)
    version = random.randint(90, 130)
    return f"Mozilla/5.0 ({random.choice(platforms)}) AppleWebKit/537.36 (KHTML, like Gecko) {browser}/{version}.0.{random.randint(1000,9999)} Safari/537.36"

USER_AGENT = random_user_agent()
CLIENT_ID = f"client_{random.randint(100000,999999)}"

# ---------------- HEADER ----------------
st.markdown(f"""
<div class='glow'>🚀 Vahan Registrations Dashboard — MAXED++</div>
<div class='subtext'>AI-Driven Insights • Forecasting • Clustering • Anomaly Detection • Smart Exports</div>
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
    - 🏗️ **Build:** `{APP_VERSION}` ({BUILD_ID})
    """)
except Exception as e:
    st.sidebar.error(f"⚠️ Env info unavailable: {e}")

# ---------------- RELOAD SAFETY ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = random.randint(100000, 999999)
    st.toast("✅ App Initialized — MAXED session active", icon="🚀")
else:
    st.toast("🔄 Reloaded — session alive", icon="♻️")

# ---------------- PREFETCH + CACHE ----------------
@st.cache_data(show_spinner=False, ttl=3600, max_entries=300)
def preload_static_assets():
    return {"status": "cached", "timestamp": datetime.now().isoformat()}

preload_static_assets()

print(f"[{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}] ✅ UI Booted — {CLIENT_ID} ({APP_VERSION}) — UA: {USER_AGENT}")

# =====================================================
# 🧭 SIDEBAR FILTERS — MAXED++ EDITION
# =====================================================
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Filters & Options</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>Customize analytics, AI, and forecasting modes</div>", unsafe_allow_html=True)

today = date.today()
default_from_year = max(2017, today.year - 1)
filters = st.session_state.setdefault("filters", {})

def getf(k, default): return filters.get(k, default)
def setf(k, v): filters[k] = v

# --- Year Range
from_year = st.sidebar.number_input("📅 From Year", 2012, today.year, getf("from_year", default_from_year))
setf("from_year", from_year)
to_year = st.sidebar.number_input("📆 To Year", from_year, today.year, getf("to_year", today.year))
setf("to_year", to_year)

# --- Location
state_code = st.sidebar.text_input("🏛️ State Code", getf("state_code", ""))
rto_code = st.sidebar.text_input("🏢 RTO Code", getf("rto_code", "0"))
setf("state_code", state_code); setf("rto_code", rto_code)

# --- Vehicle Filters
vehicle_classes = st.sidebar.text_input("🚗 Vehicle Classes", getf("vehicle_classes", ""))
vehicle_makers = st.sidebar.text_input("🏭 Vehicle Makers", getf("vehicle_makers", ""))
vehicle_type = st.sidebar.text_input("🚙 Vehicle Type", getf("vehicle_type", ""))
for k, v in [("vehicle_classes", vehicle_classes), ("vehicle_makers", vehicle_makers), ("vehicle_type", vehicle_type)]:
    setf(k, v)

# --- Time Period
period_opts = {0: "Monthly", 1: "Quarterly", 2: "Yearly"}
time_period = st.sidebar.selectbox("⏱️ Time Period", list(period_opts.keys()), index=getf("time_period", 0), format_func=lambda x: period_opts[x])
setf("time_period", time_period)

# --- Fitness
fitness_check = st.sidebar.selectbox("🧾 Fitness Check", [0, 1], index=getf("fitness_check", 0), format_func=lambda x: "No" if x == 0 else "Yes")
setf("fitness_check", fitness_check)

# --- Analytics Toggles
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Advanced Analytics")
enable_forecast = st.sidebar.checkbox("📈 Forecasting", value=True)
enable_anomaly = st.sidebar.checkbox("🚨 Anomaly Detection", value=True)
enable_clustering = st.sidebar.checkbox("🔍 Clustering", value=True)
enable_ai = st.sidebar.checkbox("🤖 DeepInfra AI Narratives", value=True)
forecast_periods = st.sidebar.number_input("📅 Forecast Horizon (months)", 1, 36, 3)

# --- DeepInfra Config
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 DeepInfra Settings")
DEEPINFRA_API_KEY = (
    os.environ.get("DEEPINFRA_API_KEY")
    or (st.secrets.get("DEEPINFRA_API_KEY") if "DEEPINFRA_API_KEY" in st.secrets else None)
)
DEEPINFRA_MODEL = os.environ.get("DEEPINFRA_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

if enable_ai:
    if not DEEPINFRA_API_KEY:
        st.sidebar.error("🚫 No DeepInfra key — AI disabled.")
        enable_ai = False
    else:
        st.sidebar.success(f"✅ Connected to {DEEPINFRA_MODEL.split('/')[-1]}")

# --- Developer / Debug
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Developer Options")
dev_mode = st.sidebar.toggle("🧪 Developer Mode", value=False)
safe_mode = st.sidebar.toggle("🛡️ Safe Mode", value=True)

# --- Summary
summary_md = f"""
**Active Filters:**
- 📅 {from_year} → {to_year}
- 🌍 State: `{state_code or 'All-India'}`, RTO: `{rto_code}`
- 🚗 Classes: `{vehicle_classes or 'All'}`, Makers: `{vehicle_makers or 'All'}`
- ⏱️ Period: `{period_opts[time_period]}`
- 🧾 Fitness: `{'Yes' if fitness_check else 'No'}`
- 🤖 AI: `{'Enabled' if enable_ai else 'Disabled'}`
"""
st.sidebar.info(summary_md)

print(f"[{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}] ✅ Sidebar ready — Forecast:{enable_forecast} | Anomaly:{enable_anomaly} | AI:{enable_ai}")

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
import requests
from functools import lru_cache

from vahan.api import build_params, get_json  # ✅ your core API helpers

# =====================================================
# 🧩 BUILD BASE PARAMS (MAXED)
# =====================================================
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

# Attach session + trace metadata
params_common["client_id"] = st.session_state.get("session_id", random.randint(1000, 9999))
params_common["request_ts"] = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()

# =====================================================
# 🧠 SAFE FETCH FUNCTION — MAXED
# =====================================================
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def safe_fetch(endpoint: str = VAHAN_ENDPOINT, params: dict = None, retries: int = MAX_RETRIES):
    """Universal fetcher with retry, caching, error resilience, and rich logging."""
    if not params:
        params = {}

    headers = {
        "User-Agent": st.session_state.get("user_agent", "Mozilla/5.0"),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://vahan.parivahan.gov.in/",
        "Cache-Control": "no-cache",
        "X-Session-ID": str(st.session_state.get("session_id", random.randint(1000, 9999))),
    }

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            resp = requests.get(endpoint, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, dict) and data.get("data"):
                        st.toast(f"✅ Data fetched on attempt {attempt}", icon="📦")
                        return data
                    else:
                        st.warning(f"⚠️ Empty response on attempt {attempt}")
                except Exception:
                    st.error("❌ JSON parse failed")
            elif resp.status_code in (403, 429):
                wait = RETRY_DELAY * attempt * 2
                st.warning(f"⏳ Rate limited (HTTP {resp.status_code}), retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                st.error(f"❌ HTTP {resp.status_code}: {resp.text[:100]}")
        except requests.RequestException as e:
            wait = RETRY_DELAY * attempt
            st.warning(f"🌐 Attempt {attempt}/{retries} failed — {e.__class__.__name__}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
        except Exception as e:
            st.error(f"💥 Unexpected error: {e}")
            traceback.print_exc()
            break

    st.error("🚫 Failed to fetch data after multiple retries.")
    return {"error": True, "data": []}

# =====================================================
# 📦 FETCH WRAPPER — PRETTY LAYER
# =====================================================
def get_vahan_data(tag: str = "Trend", params: dict = None):
    """Layered call with logging and timing."""
    start = time.time()
    st.info(f"🚀 Fetching {tag} data from Parivahan ...", icon="🛰️")
    result = safe_fetch(VAHAN_ENDPOINT, params or params_common)
    duration = time.time() - start

    if result.get("error"):
        st.error(f"❌ Fetch for {tag} failed in {duration:.2f}s.")
    else:
        st.success(f"✅ {tag} data fetched in {duration:.2f}s ({len(result.get('data', []))} records).")

    print(f"[{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%H:%M:%S')}] {tag} fetch complete — {duration:.2f}s")
    return result

# =====================================================
# 🧾 EXAMPLE USAGE
# =====================================================
# trend_json = get_vahan_data("Registration Trend")
# df_trend = to_df(trend_json)
# st.dataframe(df_trend.head())

# =====================================================
# 🌐 UNIVERSAL SAFE FETCH FUNCTION — MAXED EDITION
# =====================================================
import random
import time
import json
import traceback
import requests
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps

# =====================================================
# 🧩 HEADER + TOKEN ROTATION
# =====================================================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/122.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
]

TOKEN_POOL = []
try:
    if "api_keys" in st.secrets:
        for kset in st.secrets["api_keys"].values():
            if isinstance(kset, list):
                TOKEN_POOL += kset
except Exception:
    pass

def random_headers():
    """Generate spoofed headers for each fetch."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://parivahan.gov.in/",
        "X-Request-ID": f"{random.randint(100000,999999)}-{int(time.time())}",
        "X-Client-ID": f"vahan-maxed-{random.randint(1000,9999)}",
    }
    if TOKEN_POOL:
        headers["Authorization"] = f"Bearer {random.choice(TOKEN_POOL)}"
    return headers

# =====================================================
# ⚙️ UNIVERSAL SAFE FETCHER (CACHED + RETRY)
# =====================================================
def with_cache(ttl=3600):
    """Decorator for Streamlit cache with TTL and safe fallback."""
    def decorator(func):
        @st.cache_data(ttl=ttl, show_spinner=False)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@with_cache(ttl=3600)
def universal_fetch(
    url: str,
    params: dict = None,
    method: str = "GET",
    retries: int = 5,
    backoff: float = 2.0,
    timeout: int = 20,
    json_body: dict = None,
    verbose: bool = True,
):
    """MAXED safe network fetcher with retry, rotation, and caching."""
    if not params:
        params = {}

    headers = random_headers()
    session_id = f"SID-{random.randint(1000,9999)}"
    attempt = 0
    start = time.time()

    if verbose:
        st.toast(f"🌐 Fetching {url.split('/')[-1]} …", icon="🛰️")

    while attempt < retries:
        attempt += 1
        try:
            if method.upper() == "POST":
                resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            else:
                resp = requests.get(url, headers=headers, params=params, timeout=timeout)

            code = resp.status_code
            if code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, (dict, list)) and data:
                        if verbose:
                            st.toast(f"✅ Success on attempt {attempt}", icon="📦")
                        duration = time.time() - start
                        print(f"[{session_id}] ✅ {url} ({code}) in {duration:.2f}s")
                        return data
                    else:
                        st.warning(f"⚠️ Empty JSON response (attempt {attempt})")
                except json.JSONDecodeError:
                    st.warning(f"⚠️ Invalid JSON — retrying ({attempt}/{retries})")
            elif code in (403, 429):
                wait = backoff * attempt
                st.warning(f"⏳ Rate limited ({code}) — retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                st.error(f"❌ HTTP {code} — {resp.text[:120]}")
        except requests.RequestException as e:
            wait = backoff * attempt
            st.warning(f"🌐 Network error ({e.__class__.__name__}) — retry {attempt}/{retries} in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            st.error(f"💥 Unexpected error: {e}")
            traceback.print_exc()
            break

    st.error("🚫 All fetch attempts failed.")
    print(f"[{session_id}] ❌ Failed after {retries} retries")
    return {"error": True, "data": []}

# # =====================================================
# # 🧠 FETCH WRAPPER FOR VAHAN — MAXED
# # =====================================================
# def get_vahan_json(tag: str = "RegistrationTrend", params: dict = None):
#     """Unified call for Parivahan endpoints with logging + retry safety."""
#     endpoint = f"https://vahan.parivahan.gov.in/vahandashboard/{tag.lower()}"
#     st.info(f"🚀 Fetching `{tag}` from Parivahan...", icon="🛰️")

#     data = universal_fetch(endpoint, params=params or {}, retries=5, backoff=2.5)
#     if data.get("error"):
#         st.error(f"❌ {tag} fetch failed.")
#     else:
#         st.success(f"✅ {tag} data fetched ({len(data.get('data', [])) if isinstance(data, dict) else 'OK'})")

#     return data

# =====================================================
# 📊 DIAGNOSTIC LOG
# =====================================================
print("=" * 90)
print("🌐 UNIVERSAL SAFE FETCHER — MAXED Edition Active")
print(f"🕒 {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %I:%M:%S %p')} | UA Pool: {len(USER_AGENTS)} | Token Pool: {len(TOKEN_POOL)}")
print("=" * 90)

# =====================================================
# 🌐 UNIVERSAL MAXED FETCHER (Vahan API)
# =====================================================
import time
import random
import json
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import streamlit as st

# =====================================================
# 🧩 HEADER + TOKEN ROTATION
# =====================================================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/122.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
]

TOKEN_POOL = []
try:
    if "api_keys" in st.secrets:
        for kset in st.secrets["api_keys"].values():
            if isinstance(kset, list):
                TOKEN_POOL += kset
except Exception:
    pass

def random_headers():
    """Generate randomized spoof headers with optional token rotation."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://parivahan.gov.in/",
        "X-Request-ID": f"{random.randint(100000,999999)}-{int(time.time())}",
    }
    if TOKEN_POOL:
        headers["Authorization"] = f"Bearer {random.choice(TOKEN_POOL)}"
    return headers


# =====================================================
# 🧠 MAXED Robust Fetcher with Cache, Backoff & Fallback
# =====================================================
@st.cache_data(show_spinner=False, ttl=900, max_entries=100)
def fetch_json(
    endpoint: str,
    params: dict = None,
    desc: str = "",
    base_url: str = "https://vahanapi.parivahan.gov.in/",
    fallback_url: str = "https://vahan.parivahan.gov.in/vahandashboard/",
    max_retries: int = 5,
    timeout: int = 20,
) -> dict:
    """
    🌐 MAXED universal safe API fetcher:
    - randomized spoof headers + optional token
    - retries with exponential backoff
    - auto fallback to secondary API endpoint
    - TTL cache via Streamlit
    - detailed diagnostics & toasts
    """
    if params is None:
        params = {}

    backoff_base = 2
    endpoint = endpoint.lstrip("/")
    urls_to_try = [base_url.rstrip("/") + "/" + endpoint]
    if fallback_url:
        urls_to_try.append(fallback_url.rstrip("/") + "/" + endpoint)

    for url in urls_to_try:
        for attempt in range(1, max_retries + 1):
            headers = random_headers()
            try:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
                code = response.status_code

                # 🎯 Success
                if code in (200, 201):
                    try:
                        json_data = response.json()
                        if json_data:
                            msg = f"✅ {desc or endpoint} fetched OK"
                            print(f"[{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%H:%M:%S')}] {msg}")
                            st.toast(msg, icon="📦")
                            return json_data
                        else:
                            st.warning(f"⚠️ Empty response for {desc}")
                            return {}
                    except json.JSONDecodeError:
                        st.warning(f"⚠️ Invalid JSON for {desc}")
                        continue

                # 🚫 Forbidden / Rate Limited
                elif code == 403:
                    st.info(f"🚫 Forbidden (403). Rotating headers & retrying…")
                    time.sleep(random.uniform(1, 3))
                elif code == 429:
                    wait = random.uniform(3, 7)
                    st.warning(f"⚠️ Rate limited (429). Retrying after {wait:.1f}s...")
                    time.sleep(wait)

                # 🌀 Server Error
                elif code >= 500:
                    wait = backoff_base ** attempt + random.uniform(0.3, 1.0)
                    st.warning(f"🌀 Server error {code}. Retry {attempt}/{max_retries} after {wait:.1f}s...")
                    time.sleep(wait)

                else:
                    st.error(f"❌ Unexpected HTTP {code} for {desc or endpoint}")
                    break

            except requests.exceptions.Timeout:
                wait = backoff_base ** attempt
                st.warning(f"⏳ Timeout fetching {desc}. Retry {attempt}/{max_retries} after {wait:.1f}s...")
                time.sleep(wait)
            except requests.exceptions.ConnectionError:
                wait = backoff_base ** attempt
                st.warning(f"🔌 Connection error — retry {attempt}/{max_retries} after {wait:.1f}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"⚠️ {desc} — Unexpected error: {e}")
                traceback.print_exc()
                time.sleep(1.5)

        # 🔁 Try fallback URL if first base fails
        st.warning(f"🔁 Switching to fallback endpoint for {desc or endpoint}")

    st.error(f"❗ Failed to fetch {desc or endpoint} after {max_retries} retries.")
    return {}


# =====================================================
# 📊 DIAGNOSTIC LOG
# =====================================================
print("=" * 80)
print("🌐 Vahan MAXED Fetcher Active")
print(f"🕒 {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %I:%M:%S %p')} | UA Pool: {len(USER_AGENTS)} | Tokens: {len(TOKEN_POOL)}")
print("=" * 80)

# =====================================================
# 🔮 DUAL-YEAR + NEXT-YEAR PREDICTION SUITE — MAXED
# =====================================================
import pandas as pd
import numpy as np
import math
import io
import json
import logging
import altair as alt
import plotly.express as px
from datetime import date, datetime
from zoneinfo import ZoneInfo
import streamlit as st
import time
import traceback

# optional ML libs
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

logger = logging.getLogger("vahan_dual_year_maxed")

def ist_now():
    return datetime.now(ZoneInfo("Asia/Kolkata"))

def ist_now_str():
    return ist_now().strftime("%Y-%m-%d %I:%M:%S %p")

# =====================================================
# 🧩 NORMALIZER — accepts any trend-like JSON
# =====================================================
def normalize_trend_to_df(trend_json):
    """Convert various Vahan trend JSONs into a normalized DataFrame with date/year/value."""
    if not trend_json:
        return pd.DataFrame(columns=["date", "value"])
    try:
        if isinstance(trend_json, dict) and "labels" in trend_json and "data" in trend_json:
            df = pd.DataFrame({
                "label": trend_json.get("labels", []),
                "value": pd.to_numeric(trend_json.get("data", []), errors="coerce")
            })
        elif isinstance(trend_json, dict):
            df = pd.DataFrame(list(trend_json.items()), columns=["label", "value"])
        elif isinstance(trend_json, list):
            df = pd.json_normalize(trend_json)
            if "label" not in df.columns:
                df.rename(columns={df.columns[0]: "label"}, inplace=True)
            if "value" not in df.columns:
                df.rename(columns={df.columns[-1]: "value"}, inplace=True)
        else:
            return pd.DataFrame(columns=["date", "value"])

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["label"], errors="coerce")
        df["year"] = df["date"].dt.year
        df = df.dropna(subset=["value"]).sort_values(["date", "year"], na_position="last").reset_index(drop=True)
        return df[["date", "year", "value"]].drop_duplicates()
    except Exception as e:
        logger.error(f"normalize_trend_to_df failed: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=["date", "value"])

# =====================================================
# ⚙️ PARAM HELPERS
# =====================================================
def build_dual_year_params(from_year, to_year, extra_params=None):
    extra = extra_params or {}
    prev, curr = params_common.copy(), params_common.copy()
    for k in ("fromYear", "toYear"):
        if k in prev:
            prev[k] = from_year
            curr[k] = to_year
    prev.update(extra)
    curr.update(extra)
    return prev, curr

# =====================================================
# 🔄 FETCH + NORMALIZE — Dual-Year Mode
# =====================================================
@st.cache_data(show_spinner=False, ttl=600)
def fetch_dual_year(endpoint, desc="", from_year=None, to_year=None, extra_params=None):
    now = ist_now_str()
    today = date.today()
    to_year = to_year or today.year
    from_year = from_year or (to_year - 1)

    st.info(f"📡 Fetching {desc or endpoint}: {from_year} & {to_year} ({now})")
    params_prev, params_this = build_dual_year_params(from_year, to_year, extra_params)

    json_prev = fetch_json(endpoint, params=params_prev, desc=f"{desc} ({from_year})")
    json_this = fetch_json(endpoint, params=params_this, desc=f"{desc} ({to_year})")

    df_prev = normalize_trend_to_df(json_prev)
    df_this = normalize_trend_to_df(json_this)
    return df_prev, df_this

# =====================================================
# 🔮 FORECAST ENGINE — Linear / Prophet Hybrid
# =====================================================
def forecast_next_year(df, value_col="value", year_col="year", periods=1):
    """Predict next-year value(s) using Prophet (if available) or LinearRegression fallback."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[year_col, value_col, "type"])

    df = df.copy().dropna(subset=[year_col, value_col])
    df[year_col] = df[year_col].astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # --- Prophet path ---
    if PROPHET_AVAILABLE and len(df) >= 3:
        try:
            prophet_df = df.rename(columns={year_col: "ds", value_col: "y"})
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
            model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=periods, freq="Y")
            forecast = model.predict(future)
            out = forecast[["ds", "yhat"]].rename(columns={"ds": year_col, "yhat": value_col})
            out[year_col] = out[year_col].dt.year
            out["type"] = "Predicted"
            df["type"] = "Actual"
            return pd.concat([df, out.tail(periods)], ignore_index=True).drop_duplicates(subset=[year_col])
        except Exception as e:
            logger.warning(f"Prophet forecast failed: {e}")

    # --- Linear Regression fallback ---
    if SKLEARN_AVAILABLE and len(df) >= 2:
        try:
            X = df[[year_col]].values
            y = df[value_col].values
            model = LinearRegression().fit(X, y)
            last = df[year_col].max()
            preds = [{"year": last + i, "value": float(model.predict([[last + i]])[0]), "type": "Predicted"} for i in range(1, periods+1)]
            df["type"] = "Actual"
            return pd.concat([df, pd.DataFrame(preds)], ignore_index=True)
        except Exception as e:
            logger.warning(f"Linear regression failed: {e}")

    # --- fallback simple slope ---
    if len(df) >= 2:
        y1, y2 = df.iloc[-2][year_col], df.iloc[-1][year_col]
        v1, v2 = df.iloc[-2][value_col], df.iloc[-1][value_col]
        slope = (v2 - v1) / (y2 - y1 or 1)
        last = y2
        preds = [{"year": last + i, "value": v2 + slope * i, "type": "Predicted"} for i in range(1, periods+1)]
        df["type"] = "Actual"
        return pd.concat([df, pd.DataFrame(preds)], ignore_index=True)

    # --- fallback constant ---
    df["type"] = "Actual"
    preds = [{year_col: int(df[year_col].max()) + 1, value_col: float(df[value_col].iloc[-1]), "type": "Predicted"}]
    return pd.concat([df, pd.DataFrame(preds)], ignore_index=True)

# =====================================================
# 📈 COMPARISON BUILDER
# =====================================================
def build_prev_this_next(df_prev, df_this, periods=1):
    """Combine prev & this year, add forecasted next."""
    def to_yearly(df):
        if "date" in df.columns:
            df["year"] = pd.to_datetime(df["date"]).dt.year
        df = df.groupby("year", as_index=False)["value"].sum()
        return df

    y_prev = to_yearly(df_prev)
    y_this = to_yearly(df_this)

    combined = pd.concat([y_prev, y_this]).drop_duplicates(subset=["year"]).sort_values("year")
    forecasted = forecast_next_year(combined, periods=periods)
    return forecasted

# =====================================================
# 🎯 DISPLAY SUITE
# =====================================================
def show_comparison_cards(df, fmt="{:,.0f}"):
    if df is None or df.empty:
        st.info("No summary available")
        return
    df = df.sort_values("year")
    cols = st.columns(len(df))
    prev_val = None
    for i, (_, row) in enumerate(df.iterrows()):
        growth = ""
        if prev_val and not pd.isna(prev_val):
            growth_val = ((row["value"] - prev_val) / prev_val) * 100 if prev_val else 0
            growth = f"{growth_val:+.2f}%"
        prev_val = row["value"]
        cols[i].metric(f"{row['year']} ({row['type']})", fmt.format(row["value"]), growth)

def show_forecast_chart(df):
    if df is None or df.empty:
        st.info("No data for chart")
        return
    df = df.sort_values("year")
    base = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("value:Q", title="Registrations"),
        color="type:N",
        tooltip=["year", "value", "type"]
    )
    st.altair_chart(base.properties(height=400), use_container_width=True)
    st.plotly_chart(px.bar(df, x="year", y="value", color="type", title="Actual vs Predicted"), use_container_width=True)

# =====================================================
# 📦 EXCEL EXPORTER
# =====================================================
def export_dual_year_excel(dfs_dict, filename="vahan_dual_year_maxed"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, df in dfs_dict.items():
            try:
                df.to_excel(writer, sheet_name=str(sheet)[:31], index=False)
            except Exception:
                pd.DataFrame({"raw": [json.dumps(df, default=str)]}).to_excel(writer, sheet_name=str(sheet)[:31], index=False)
    buffer.seek(0)
    return buffer.getvalue()

# =====================================================
# 🚀 MASTER ORCHESTRATOR
# =====================================================
def dual_year_analysis_and_ui(endpoint, desc, from_year=None, to_year=None, extra_params=None, predict_periods=1):
    """Full Dual-Year + Forecast pipeline with Streamlit UI."""
    st.header(f"📊 {desc}: {from_year or (date.today().year-1)} ↔ {to_year or date.today().year} ↔ 🔮 Predicted")
    df_prev, df_this = fetch_dual_year(endpoint, desc, from_year, to_year, extra_params)
    df_all = build_prev_this_next(df_prev, df_this, periods=predict_periods)

    show_comparison_cards(df_all)
    show_forecast_chart(df_all)

    with st.expander("📂 Raw Data"):
        st.dataframe(df_prev, use_container_width=True)
        st.dataframe(df_this, use_container_width=True)

    excel_bytes = export_dual_year_excel({
        f"{desc}_prev": df_prev,
        f"{desc}_this": df_this,
        f"{desc}_combined": df_all
    }, filename=desc.replace(" ", "_"))

    ts = ist_now().strftime("%Y%m%d_%H%M%S")
    st.download_button("⬇️ Download Excel Report", excel_bytes,
                       file_name=f"{desc.replace(' ','_')}_{ts}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    return {"prev": df_prev, "this": df_this, "combined": df_all}


# =====================================================
# 🚀 VAHAN ANALYTICS — API CORE (ALL MAXED EDITION)
# =====================================================
import os
import time
import json
import pickle
import random
import logging
import requests
import streamlit as st
from urllib.parse import urlencode
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional

# =====================================================
# 🎨 STREAMLIT HEADER
# =====================================================
st.markdown("""
<div style="
    background:linear-gradient(90deg,#0072ff,#00c6ff);
    padding:16px 26px;
    border-radius:14px;
    color:#fff;
    font-size:18px;
    font-weight:700;
    display:flex;justify-content:space-between;align-items:center;
    box-shadow:0 0 25px rgba(0,114,255,0.4);">
    <div>🧩 <b>VAHAN ANALYTICS — All-Maxed API Engine</b></div>
    <div style="font-size:14px;opacity:0.9;">Auto-synced | Cached | Throttled | Resilient</div>
</div>
""", unsafe_allow_html=True)
st.write("")

# =====================================================
# 🧠 PARAMETER BUILDER (DYNAMIC)
# =====================================================
def build_params(
    from_year: int,
    to_year: int,
    state_code: Optional[str] = None,
    rto_code: Optional[str] = None,
    vehicle_classes: Optional[list] = None,
    vehicle_makers: Optional[list] = None,
    time_period: Optional[str] = None,
    fitness_check: Optional[str] = None,
    vehicle_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Create parameter dictionary safely and dynamically."""
    params = {
        "fromYear": from_year,
        "toYear": to_year,
        "stateCd": state_code,
        "rtoCd": rto_code,
        "vehicleClass": vehicle_classes,
        "vehicleMaker": vehicle_makers,
        "timePeriod": time_period,
        "fitnessCheck": fitness_check,
        "vehicleType": vehicle_type,
    }
    return {k: v for k, v in params.items() if v not in (None, "", [], {}, " ")}


with st.spinner("🚀 Building request parameters..."):
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
        st.balloons()
        st.toast("✨ Parameters built successfully!", icon="⚙️")
        with st.expander("🔧 View Generated Parameters (JSON)", expanded=False):
            st.json(params_common)
    except Exception as e:
        st.error(f"❌ Error building parameters: {e}")
        if st.button("🔁 Retry"):
            st.toast("Rebuilding...", icon="🔄")
            time.sleep(0.5)
            st.rerun()

# =====================================================
# ⚙️ ULTRA-SAFE FETCHER CONFIG
# =====================================================
BASE = "https://analytics.parivahan.gov.in/analytics/publicdashboard"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.4
CACHE_DIR = "vahan_cache"
CACHE_TTL = 3600  # 1 hour
ROTATING_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0"
]
os.makedirs(CACHE_DIR, exist_ok=True)

logger = logging.getLogger("vahan_safe_fetch")
logger.setLevel(logging.INFO)

# =====================================================
# 🧩 UTILS
# =====================================================
def ist_now() -> str:
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")

def log_ist(msg: str):
    m = f"[IST {ist_now()}] {msg}"
    print(m)
    logger.info(m)

def clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if v not in (None, "", [], {}, " ")}

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
        log_ist(f"Cache hit ✅ {url}")
        return data
    except Exception:
        return None

def save_cache(url: str, data: Any):
    try:
        with open(_cache_path(url), "wb") as f:
            pickle.dump((time.time(), data), f)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")

# =====================================================
# 🧱 TOKEN BUCKET (THROTTLING)
# =====================================================
class TokenBucket:
    def __init__(self, cap: int, rate: float):
        self.capacity = cap
        self.rate = rate
        self.tokens = cap
        self.last = time.time()
    def wait(self):
        while True:
            now = time.time()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            time.sleep(0.25)

_bucket = TokenBucket(10, 1.0)

# =====================================================
# 🧩 SAFE FETCH
# =====================================================
def safe_get(path: str, params: Optional[Dict[str, Any]] = None, use_cache=True):
    params = clean_params(params or {})
    query = urlencode(params, doseq=True)
    url = f"{BASE.rstrip('/')}/{path.lstrip('/')}?{query}"

    if use_cache:
        cached = load_cache(url)
        if cached is not None:
            return cached

    _bucket.wait()
    for attempt in range(1, MAX_RETRIES + 1):
        headers = {
            "User-Agent": random.choice(ROTATING_UAS),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://analytics.parivahan.gov.in"
        }
        try:
            log_ist(f"Fetching ({attempt}/{MAX_RETRIES}): {path}")
            resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw": resp.text}
                save_cache(url, data)
                return data
            elif resp.status_code in (429, 500, 502, 503):
                wait = BACKOFF_FACTOR ** attempt + random.uniform(0.5, 2.0)
                log_ist(f"Retrying after {wait:.1f}s — {resp.status_code}")
                time.sleep(wait)
            else:
                log_ist(f"Unexpected status {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            log_ist(f"⚠️ Attempt {attempt} failed: {e}")
            time.sleep(BACKOFF_FACTOR * attempt)
    return None

# =====================================================
# 🧩 FETCH WRAPPER (STREAMLIT-FRIENDLY)
# =====================================================
def fetch_json(path: str, params: Optional[Dict[str, Any]] = None, desc: str = "", use_cache=True):
    data = safe_get(path, params=params, use_cache=use_cache)
    if data:
        msg = f"✅ {desc or path} fetched successfully ({ist_now()})"
        log_ist(msg)
        st.success(msg)
    else:
        msg = f"❌ Failed to fetch {desc or path} ({ist_now()})"
        log_ist(msg)
        st.warning(msg)
    return data

# =====================================================
# 🧩 FINAL BOOT MESSAGE
# =====================================================
st.markdown(f"""
<div style='
    background:linear-gradient(90deg,#0072ff,#00c6ff);
    color:white;
    padding:10px 20px;
    border-radius:10px;
    margin-top:15px;
    box-shadow:0 4px 15px rgba(0,0,0,0.2);
    font-family:monospace;'>
    🕒 Booted at <b>{ist_now()} (IST)</b><br>
    🧠 safe_fetch active — caching, retries, throttling, and dynamic params enabled.
</div>
""", unsafe_allow_html=True)

log_ist("✅ All-Maxed Vahan API initialized.")

# =====================================================
# 🧠 DeepInfra Universal Chat Helper (ALL MAXED EDITION)
# =====================================================
import os
import random
import string
import time
import json
import pickle
import traceback
import requests
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# ⚙️ GLOBAL CONFIG
# =====================================================
DEEPINFRA_CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
DEEPINFRA_MODEL = os.getenv("DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
CACHE_TTL = 1800  # 30 minutes
CACHE_DIR = "deepinfra_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2_1 like Mac OS X)",
    "Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X)",
    "Mozilla/5.0 (Android 14; Mobile)"
]

# =====================================================
# ⏰ TIME HELPERS
# =====================================================
def ist_now():
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")

# =====================================================
# 💾 CACHING HELPERS
# =====================================================
def _cache_key(prompt: str, model: str) -> str:
    import hashlib
    return os.path.join(CACHE_DIR, hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest() + ".pkl")

def load_cache(prompt: str, model: str):
    p = _cache_key(prompt, model)
    if not os.path.exists(p):
        return None
    try:
        ts, data = pickle.load(open(p, "rb"))
        if time.time() - ts > CACHE_TTL:
            os.remove(p)
            return None
        return data
    except Exception:
        return None

def save_cache(prompt: str, model: str, data: dict):
    try:
        pickle.dump((time.time(), data), open(_cache_key(prompt, model), "wb"))
    except Exception:
        pass

# =====================================================
# 🧠 UNIVERSAL CHAT FUNCTION
# =====================================================
def deepinfra_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
    retries: int = 5,
    model: str = None,
    use_cache: bool = True
):
    """
    Fully Maxed DeepInfra API client:
    - Retries, exponential backoff, jitter
    - Rotating User-Agent + Request ID spoof
    - Caching layer
    - Streamlit integration
    """

    if not DEEPINFRA_API_KEY:
        return {"error": "🔑 Missing DeepInfra API key. Please configure DEEPINFRA_API_KEY."}

    model = model or DEEPINFRA_MODEL
    cache_key = f"{system_prompt[:60]}::{user_prompt[:80]}"

    if use_cache:
        cached = load_cache(cache_key, model)
        if cached:
            st.info("💾 Loaded cached DeepInfra response.")
            return cached

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
        "stream": False,
    }

    def _rotating_headers():
        return {
            "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": random.choice(_UA_POOL),
            "X-Request-ID": "".join(random.choices(string.ascii_letters + string.digits, k=12)),
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        }

    # =================================================
    # 🔁 RETRY LOOP
    # =================================================
    for attempt in range(1, retries + 1):
        headers = _rotating_headers()
        start = time.time()
        try:
            resp = requests.post(
                DEEPINFRA_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            status = resp.status_code
            latency = round(time.time() - start, 2)

            if status == 200:
                try:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    final = {
                        "text": content,
                        "raw": data,
                        "status": 200,
                        "model": model,
                        "latency": latency,
                        "time": ist_now(),
                    }
                    if use_cache:
                        save_cache(cache_key, model, final)
                    st.success(f"✅ DeepInfra ({model}) responded in {latency}s")
                    return final
                except Exception as e:
                    st.error(f"⚠️ Parsing error: {e}")
                    return {"error": "Malformed DeepInfra response", "status": "parse_error"}

            elif status in [429, 500, 502, 503, 504]:
                wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                st.warning(f"⏳ Retry {attempt}/{retries} after {wait:.1f}s (HTTP {status})")
                time.sleep(wait)
                continue

            elif status == 401:
                return {"error": "🚫 Unauthorized — invalid API key", "status": 401}
            elif status == 403:
                return {"error": "❌ Forbidden — access denied", "status": 403}
            elif status == 404:
                return {"error": "🔍 Model not found or unavailable", "status": 404}
            else:
                return {"error": f"Unexpected HTTP {status}", "details": resp.text[:300]}

        except requests.exceptions.Timeout:
            st.warning(f"⚠️ Timeout (attempt {attempt}/{retries}), retrying...")
            time.sleep(2 ** attempt + random.random())
        except requests.exceptions.ConnectionError as ce:
            st.warning(f"🌐 Connection error: {ce}, retrying...")
            time.sleep(2 ** attempt + random.random())
        except Exception as e:
            st.error(f"🔥 Unexpected error: {e}\n{traceback.format_exc()}")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return {"error": str(e), "trace": traceback.format_exc()}

    return {"error": f"❌ Failed after {retries} retries", "status": "exhausted"}

# =====================================================
# 🧩 BOOT BANNER
# =====================================================
st.markdown(f"""
<div style='
    background:linear-gradient(90deg,#8338ec,#3a86ff);
    color:white;
    padding:10px 20px;
    border-radius:10px;
    margin:10px 0;
    box-shadow:0 4px 15px rgba(0,0,0,0.25);
    font-family:monospace;'>
    🔮 DeepInfra Chat Helper (All Maxed) — initialized at <b>{ist_now()}</b><br>
    🔁 Caching + Retry + Adaptive Headers + Live Logging
</div>
""", unsafe_allow_html=True)

print(f"[{ist_now()}] ✅ DeepInfra All-Maxed Client ready.")

# ------------------------------------------------------------------


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
from vahan.api import fetch_json, deepinfra_chat, params_common
from vahan.parsing import parse_makers
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# =====================================================
# 🧩 Helpers
# =====================================================
@st.cache_data(show_spinner=False, ttl=900)
def normalize_maker_df(df):
    """
    Normalize parse_makers output into canonical columns:
    ['maker', 'date', 'value']
    """
    THIS_YEAR = datetime.now().year
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["maker", "date", "value"])

    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    maker_col = next((cols_lower[k] for k in ("maker", "manufacturer", "label", "name") if k in cols_lower), df.columns[0])
    value_col = next((cols_lower[k] for k in ("value", "count", "registrations", "total") if k in cols_lower), df.select_dtypes(include=[np.number]).columns[0])
    date_col = next((cols_lower[k] for k in ("date", "ds", "month", "period", "time") if k in cols_lower), None)

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.rename(columns={maker_col: "maker", value_col: "value"})
    if date_col: df = df.rename(columns={date_col: "date"})
    else: df["date"] = pd.to_datetime(datetime(THIS_YEAR, 1, 1))

    df["maker"] = df["maker"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").fillna(pd.to_datetime(datetime(THIS_YEAR, 1, 1)))

    return df[["maker", "date", "value"]]

def pct_change(a, b):
    try:
        return ((b - a) / abs(a)) * 100.0 if a != 0 else (float("inf") if b != 0 else 0.0)
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False, ttl=1200)
def forecast_monthly_series(series, periods=12):
    try:
        s = series.dropna().sort_index()
        if len(s) < 6:
            return None

        # Prophet preferred if available
        try:
            from prophet import Prophet
            df_prop = s.reset_index().rename(columns={s.index.name or 'index': 'ds', 0: 'y'})
            df_prop.columns = ['ds', 'y']
            m = Prophet(yearly_seasonality=True)
            m.fit(df_prop)
            future = m.make_future_dataframe(periods=periods, freq='MS')
            fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
            return fc.tail(periods)
        except Exception:
            # fallback: linear regression
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(s)).reshape(-1, 1)
            y = s.values
            lr = LinearRegression().fit(X, y)
            future_X = np.arange(len(s), len(s) + periods).reshape(-1, 1)
            yhat = lr.predict(future_X)
            next_months = pd.date_range(start=s.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
            return pd.DataFrame({
                "yhat": np.maximum(yhat, 0),
                "yhat_lower": np.maximum(yhat * 0.85, 0),
                "yhat_upper": np.maximum(yhat * 1.15, 0)
            }, index=next_months)
    except Exception:
        return None

# =====================================================
# 🧠 Load Data — Full Safety Wrapper
# =====================================================
with st.spinner("🔄 Fetching Top Makers (MAXED Mode)..."):
    mk_json = fetch_json("vahandashboard/top5Makerchart", params_common, desc="Top Makers")
    mk_df_raw = parse_makers(mk_json) if mk_json else pd.DataFrame()

df_mk = normalize_maker_df(mk_df_raw)
if df_mk.empty:
    st.warning("⚠️ No data available for Top Makers.")
    st.stop()

# =====================================================
# 📊 Computations & Forecasts
# =====================================================
df_mk["month"] = df_mk["date"].dt.to_period("M").dt.to_timestamp()
df_mk["year"] = df_mk["date"].dt.year
totals_by_maker = df_mk.groupby("maker", as_index=False)["value"].sum().sort_values("value", ascending=False)
monthly_totals = df_mk.groupby("month", as_index=True)["value"].sum()
yearly_pivot = df_mk.groupby(["year", "maker"], as_index=False)["value"].sum().pivot(index="year", columns="maker", values="value").fillna(0)
monthly_pivot = df_mk.groupby(["month", "maker"], as_index=False)["value"].sum().pivot(index="month", columns="maker", values="value").fillna(0)

prev_total = float(monthly_totals[monthly_totals.index.year == PREV_YEAR].sum())
this_total = float(monthly_totals[monthly_totals.index.year == THIS_YEAR].sum())
fc_monthly_overall = forecast_monthly_series(monthly_totals, periods=12)
predicted_next_total = float(fc_monthly_overall[fc_monthly_overall.index.year == NEXT_YEAR]["yhat"].sum()) if fc_monthly_overall is not None else 0.0

# Per-maker forecasts
maker_forecasts, hist_share = {}, totals_by_maker.set_index("maker")["value"] / max(totals_by_maker["value"].sum(), 1)
for mk in monthly_pivot.columns:
    fc = forecast_monthly_series(monthly_pivot[mk], periods=12)
    if fc is not None: maker_forecasts[mk] = fc

maker_pred_next = {
    mk: (float(maker_forecasts[mk][maker_forecasts[mk].index.year == NEXT_YEAR]["yhat"].sum())
         if mk in maker_forecasts else float(hist_share.get(mk, 0.0) * predicted_next_total))
    for mk in totals_by_maker["maker"]
}

# =====================================================
# 📈 Dashboard (MAXED)
# =====================================================
st.header("🏭 Top Makers — Real + Predicted (MAXED)")
tabs = st.tabs(["📊 Overview", "📈 Trends", "📋 Detailed Table", "🤖 AI Narrative"])

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{PREV_YEAR}", f"{int(prev_total):,}", delta=None)
    col2.metric(f"{THIS_YEAR}", f"{int(this_total):,}", delta=f"{pct_change(prev_total, this_total):.2f}% vs prev")
    col3.metric(f"{NEXT_YEAR} (pred)", f"{int(predicted_next_total):,}", delta=f"{pct_change(this_total, predicted_next_total):.2f}% vs this")
    col4.metric("All-Time Total", f"{int(df_mk['value'].sum()):,}")

    st.altair_chart(
        alt.Chart(totals_by_maker.head(10)).mark_bar().encode(
            x=alt.X("value:Q", title="Registrations"),
            y=alt.Y("maker:N", sort='-x'),
            tooltip=["maker", "value"]
        ).properties(title="Top 10 Makers (Total)"),
        use_container_width=True
    )

with tabs[1]:
    real = monthly_totals.reset_index().rename(columns={"month": "ds", "value": "y"}).assign(type="real")
    if fc_monthly_overall is not None:
        fc_df = fc_monthly_overall.reset_index().rename(columns={"index": "ds", "yhat": "y"}).assign(type="predicted")
        combined = pd.concat([real, fc_df])
    else:
        combined = real

    fig = px.line(combined, x="ds", y="y", color="type", title="Monthly Registrations — Real vs Predicted")
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    rows = []
    for mk in totals_by_maker["maker"]:
        prev_v = yearly_pivot.loc[PREV_YEAR, mk] if PREV_YEAR in yearly_pivot.index else 0
        this_v = yearly_pivot.loc[THIS_YEAR, mk] if THIS_YEAR in yearly_pivot.index else 0
        pred_v = maker_pred_next.get(mk, 0)
        rows.append({
            "maker": mk,
            f"{PREV_YEAR}": prev_v,
            f"{THIS_YEAR}": this_v,
            f"{NEXT_YEAR} (pred)": pred_v,
            "growth_this_to_next %": pct_change(this_v, pred_v)
        })
    table = pd.DataFrame(rows).sort_values(f"{THIS_YEAR}", ascending=False)
    st.dataframe(table.style.format("{:,.0f}"))

    # Export
    with st.expander("📤 Export Data"):
        csv = df_mk.to_csv(index=False).encode("utf-8")
        st.download_button("Download Raw CSV", csv, file_name=f"makers_raw_{THIS_YEAR}.csv", mime="text/csv")

with tabs[3]:
    if st.toggle("🧠 Generate AI Narrative"):
        with st.spinner("Generating AI insights..."):
            user_prompt = f"Analyze top makers for {THIS_YEAR} and predict {NEXT_YEAR}. Data: {table.head(8).to_dict(orient='records')}"
            ai = deepinfra_chat("You are a senior automotive analyst.", user_prompt, max_tokens=400)
            st.write(ai.get("text", "⚠️ No AI output available."))

st.success("✅ Top Makers MAXED analysis complete.")

# ============================================================
# 🚀 3️⃣ FULLY MAXED — YEARLY / MONTHLY / DAILY TREND + FORECASTS
# ============================================================
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
from datetime import datetime, date
from zoneinfo import ZoneInfo
import streamlit as st
from vahan.api import fetch_json, deepinfra_chat, params_common
from vahan.parsing import normalize_trend
from vahan.metrics import compute_yoy, compute_qoq

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# =====================================================
# 🔧 Helpers + Forecasting
# =====================================================
@st.cache_data(show_spinner=False, ttl=900)
def forecast_series(series, periods=12):
    """
    Hybrid Prophet → LinearRegression fallback
    """
    s = series.dropna().sort_index()
    if len(s) < 6:
        return None
    try:
        from prophet import Prophet
        df_prop = s.reset_index().rename(columns={s.index.name or 'index': 'ds', 0: 'y'})
        m = Prophet(yearly_seasonality=True)
        m.fit(df_prop)
        future = m.make_future_dataframe(periods=periods, freq='MS')
        fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
        return fc.tail(periods)
    except Exception:
        # fallback: linear regression
        from sklearn.linear_model import LinearRegression
        X = np.arange(len(s)).reshape(-1, 1)
        y = s.values
        lr = LinearRegression().fit(X, y)
        future_X = np.arange(len(s), len(s) + periods).reshape(-1, 1)
        yhat = lr.predict(future_X)
        next_months = pd.date_range(start=s.index[-1] + pd.offsets.MonthBegin(1),
                                    periods=periods, freq='MS')
        return pd.DataFrame({
            "yhat": np.maximum(yhat, 0),
            "yhat_lower": np.maximum(yhat * 0.85, 0),
            "yhat_upper": np.maximum(yhat * 1.15, 0)
        }, index=next_months)

def pct(a, b):
    try: return ((b - a) / abs(a)) * 100 if a != 0 else 0
    except: return 0

# =====================================================
# 🛰️ Fetch Data
# =====================================================
with st.spinner("📡 Fetching registration trend data (MAXED Mode)..."):
    tr_json = fetch_json("vahandashboard/vahanyearwiseregistrationtrend",
                         params_common, desc="Registration Trend")

try:
    df_trend = normalize_trend(tr_json)
except Exception as e:
    st.error(f"❌ Trend parsing failed: {e}")
    df_trend = pd.DataFrame(columns=["date", "value"])

if df_trend.empty:
    st.warning("⚠️ No trend data available.")
    st.stop()

# =====================================================
# 🧹 Cleanup
# =====================================================
df_trend = df_trend.sort_values("date")
df_trend["year"] = df_trend["date"].dt.year
df_trend["month"] = df_trend["date"].dt.month_name()
df_trend["day"] = df_trend["date"].dt.day

# =====================================================
# 📈 Tabs: Yearly / Monthly / Daily / Forecast
# =====================================================
st.header("📊 Registration Trends — MAXED Edition")
tabY, tabM, tabD, tabF, tabA = st.tabs([
    "📅 Yearly Overview",
    "🗓️ Monthly Trends",
    "📆 Daily View",
    "🔮 Forecast & Prediction",
    "🤖 AI Insights"
])

# ---------------- YEARLY ----------------
with tabY:
    yearly_df = df_trend.groupby("year", as_index=False)["value"].sum()
    yearly_df["YoY%"] = yearly_df["value"].pct_change() * 100
    st.altair_chart(
        alt.Chart(yearly_df).mark_bar().encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("value:Q", title="Registrations"),
            tooltip=["year", "value", "YoY%"]
        ).properties(title="Yearly Registration Totals"),
        use_container_width=True
    )
    st.dataframe(yearly_df.style.format({"value": ",", "YoY%": "{:.2f}"}).background_gradient("Blues"))

# ---------------- MONTHLY ----------------
with tabM:
    df_trend["ym"] = df_trend["date"].dt.to_period("M").astype(str)
    monthly_df = df_trend.groupby("ym", as_index=False)["value"].sum()
    st.line_chart(monthly_df.set_index("ym")["value"])
    st.dataframe(monthly_df.tail(12).style.background_gradient("Greens"))

# ---------------- DAILY ----------------
with tabD:
    st.area_chart(df_trend.set_index("date")["value"])
    st.dataframe(df_trend.tail(30).style.background_gradient("Oranges"))

# ---------------- FORECAST ----------------
with tabF:
    monthly_series = df_trend.groupby(df_trend["date"].dt.to_period("M"))["value"].sum()
    fc = forecast_series(monthly_series, periods=12)

    if fc is not None:
        real = monthly_series.reset_index().rename(columns={"date": "ds", "value": "y"})
        fc_reset = fc.reset_index().rename(columns={"index": "ds"})
        combined = pd.concat([
            real.assign(type="Actual", y=real["y"]),
            fc_reset.assign(type="Predicted", y=fc_reset["yhat"])
        ])

        fig = px.line(combined, x="ds", y="y", color="type",
                      title="Real vs Forecast — Monthly Registrations")
        st.plotly_chart(fig, use_container_width=True)

        total_pred = int(fc["yhat"].sum())
        avg_pred = int(fc["yhat"].mean())
        col1, col2 = st.columns(2)
        col1.metric("Predicted Total (Next 12 M)", f"{total_pred:,}")
        col2.metric("Predicted Monthly Avg", f"{avg_pred:,}")
        st.dataframe(fc.head(12).style.background_gradient("Purples"))
    else:
        st.warning("⚠️ Forecast unavailable (insufficient data).")

# =====================================================
# 📊 Comparative Growth Metrics
# =====================================================
st.markdown("---")
st.subheader("📉 Growth Metrics (YoY / QoQ / MoM / CAGR)")
yoy_df = compute_yoy(df_trend)
qoq_df = compute_qoq(df_trend)

latest_yoy = yoy_df["YoY%"].dropna().iloc[-1] if not yoy_df.empty else None
latest_qoq = qoq_df["QoQ%"].dropna().iloc[-1] if not qoq_df.empty else None

total_val = int(df_trend["value"].sum())
avg_daily = df_trend["value"].mean()
avg_monthly = df_trend.groupby(df_trend["date"].dt.to_period("M"))["value"].sum().mean()

colK1, colK2, colK3, colK4, colK5 = st.columns(5)
colK1.metric("Total Registrations", f"{total_val:,}")
colK2.metric("Daily Avg", f"{avg_daily:,.0f}")
colK3.metric("Monthly Avg", f"{avg_monthly:,.0f}")
colK4.metric("YoY Growth %", f"{latest_yoy:.2f}%" if latest_yoy else "N/A")
colK5.metric("QoQ Growth %", f"{latest_qoq:.2f}%" if latest_qoq else "N/A")

st.dataframe(yoy_df.tail(5).style.background_gradient("coolwarm"))
st.dataframe(qoq_df.tail(5).style.background_gradient("coolwarm"))

# =====================================================
# 🧠 AI-Driven Narrative
# =====================================================
with tabA:
    if st.toggle("🤖 Generate AI Insight Report"):
        with st.spinner("Generating AI insights via DeepInfra ..."):
            system_prompt = (
                "You are a senior automotive market analyst. "
                "Use trend, YoY, QoQ, and forecast data to produce an advanced, "
                "data-driven report including insights, anomalies, growth patterns, "
                "and 3 strategic recommendations."
            )
            sample = df_trend.tail(24).to_dict(orient="records")
            user_prompt = (
                f"SampleData: {json.dumps(sample, default=str)}\n"
                f"YoY: {latest_yoy}, QoQ: {latest_qoq}, AvgDaily: {avg_daily:,.0f}, ForecastAvg: {avg_pred if 'avg_pred' in locals() else 0}\n"
                "Generate an executive summary."
            )
            ai_resp = deepinfra_chat(system_prompt, user_prompt, max_tokens=900)
            if isinstance(ai_resp, dict) and "text" in ai_resp:
                st.markdown("### 🧠 AI Trend Report")
                st.info(ai_resp["text"])
            else:
                st.info("⚠️ No AI response received.")

# =====================================================
# 📤 Export Data
# =====================================================
st.markdown("---")
with st.expander("📤 Export Data"):
    csv_bytes = df_trend.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv_bytes,
                       file_name=f"vahan_trend_{THIS_YEAR}.csv",
                       mime="text/csv")
    json_bytes = json.dumps(df_trend.to_dict(orient="records"),
                            indent=2, default=str).encode("utf-8")
    st.download_button("⬇️ Download JSON", json_bytes,
                       file_name=f"vahan_trend_{THIS_YEAR}.json",
                       mime="application/json")

st.success("✅ Trend + Forecast MAXED analysis complete.")

# =====================================================
# 4️⃣ MAXED — Duration-wise Growth (Monthly / Quarterly / Yearly)
# =====================================================
import io
import math
import json
import time
import random
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date, datetime
from zoneinfo import ZoneInfo
import warnings

warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# =====================================================
# ✅ Safe Fetcher with Retry & Auto-Logging (MAXED)
# =====================================================
@st.cache_data(show_spinner=False, ttl=300)
def fetch_json(endpoint, params, desc="data", retries=3, delay=2):
    """Universal JSON fetcher with retry + status reporting."""
    try:
        for i in range(retries):
            try:
                data = get_json(endpoint, params)
                if data:
                    return data
            except Exception as e:
                st.warning(f"⚠️ Fetch attempt {i+1}/{retries} for {desc} failed — {e}")
                time.sleep(delay + random.random())
        st.error(f"🚫 Failed to fetch {desc} after {retries} retries.")
        return {}
    except Exception as e:
        st.error(f"🔥 Fatal fetch error for {desc}: {e}")
        return {}

# =====================================================
# 🧮 Normalize Duration DataFrame
# =====================================================
def normalize_duration_df(df, calendar_type):
    if df is None or df.empty:
        return pd.DataFrame(columns=["duration_label", "period", "value"])
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}

    label_col = next((cols[k] for k in ["duration", "label", "bucket", "name"] if k in cols), None)
    val_col = next((cols[k] for k in ["value", "count", "registrations", "total"] if k in cols), None)
    period_col = next((cols[k] for k in ["period", "date", "month", "year", "ds"] if k in cols), None)

    d["duration_label"] = d[label_col] if label_col else "all"
    d["value"] = pd.to_numeric(d[val_col], errors="coerce").fillna(0) if val_col else 0
    if period_col:
        d["period"] = pd.to_datetime(d[period_col], errors="coerce")
    else:
        d["period"] = pd.date_range(start=datetime(THIS_YEAR, 1, 1), periods=len(d), freq="M")

    return d[["duration_label", "period", "value"]]

# =====================================================
# 🔮 Forecast Helper (Prophet → Linear fallback)
# =====================================================
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def forecast_series(s: pd.Series, periods: int, freq: str):
    try:
        s = s.dropna().sort_index()
        if len(s) < 4:
            return None
        if PROPHET_AVAILABLE and len(s) >= 12:
            dfp = s.reset_index()
            dfp.columns = ["ds", "y"]
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=periods, freq=freq)
            fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds")
            return fc.tail(periods)
        else:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(s)).reshape(-1, 1)
            y = s.values
            lr = LinearRegression().fit(X, y)
            future_X = np.arange(len(s), len(s)+periods).reshape(-1, 1)
            yhat = lr.predict(future_X)
            freq_map = {"MS": "M", "QS": "Q", "YS": "Y"}
            next_idx = pd.date_range(s.index[-1] + pd.offsets.DateOffset(1),
                                     periods=periods, freq=freq_map.get(freq, "M"))
            return pd.DataFrame({
                "yhat": np.maximum(yhat, 0),
                "yhat_lower": np.maximum(yhat*0.85, 0),
                "yhat_upper": np.maximum(yhat*1.15, 0)
            }, index=next_idx)
    except Exception:
        return None

# =====================================================
# 📊 Duration Analysis Core
# =====================================================
def analyze_duration(calendar_type: int, label: str, forecast_periods: int):
    with st.spinner(f"Fetching {label} duration-wise data..."):
        json_data = fetch_json("vahandashboard/durationWiseRegistrationTable",
                               {**params_common, "calendarType": calendar_type},
                               desc=f"{label} growth")
        df_raw = parse_duration_table(json_data) if json_data else pd.DataFrame()

    df = normalize_duration_df(df_raw, calendar_type)
    if df.empty:
        st.warning(f"No {label} duration data found.")
        return df

    st.subheader(f"📈 {label} Registration Growth (MAXED)")

    # Frequency mapping
    freq_map = {3: "MS", 2: "QS", 1: "YS"}
    freq = freq_map.get(calendar_type, "MS")
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["year"] = df["period"].dt.year

    # Aggregate
    pivot = df.groupby(["period", "duration_label"], as_index=False)["value"].sum()
    pivot_full = pivot.pivot(index="period", columns="duration_label", values="value").fillna(0)

    totals = pivot_full.sum(axis=1)
    fc = forecast_series(totals, forecast_periods, freq)

    # KPI calculations
    totals_by_year = df.groupby("year")["value"].sum()
    prev_total = totals_by_year.get(PREV_YEAR, 0)
    this_total = totals_by_year.get(THIS_YEAR, 0)
    next_total = float(fc["yhat"].sum()) if fc is not None else this_total

    def pct(a,b): return ((b-a)/a*100) if a else 0

    k1,k2,k3,k4 = st.columns(4)
    k1.metric(f"Prev Year ({PREV_YEAR})", f"{int(prev_total):,}", delta=f"{pct(prev_total,this_total):.2f}%")
    k2.metric(f"This Year ({THIS_YEAR})", f"{int(this_total):,}", delta=f"{pct(this_total,next_total):.2f}%")
    k3.metric(f"Predicted Next ({NEXT_YEAR})", f"{int(next_total):,}", delta=f"{pct(prev_total,next_total):.2f}%")
    k4.metric("Total Duration Buckets", len(df["duration_label"].unique()))

    # Charts
    fig = px.line(totals, title=f"{label} — Real vs Predicted")
    if fc is not None:
        fc_df = fc.reset_index()
        fc_df["type"] = "Predicted"
        fig.add_scatter(x=fc_df["ds"], y=fc_df["yhat"], mode="lines", name="Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # Duration bucket chart
    st.markdown(f"### {label} — Stacked by Duration Bucket")
    fig2 = px.area(pivot_full, x=pivot_full.index, y=pivot_full.columns, title=f"{label} — Stacked Trend")
    st.plotly_chart(fig2, use_container_width=True)

    # Table comparison
    per_dur_year = df.groupby(["year", "duration_label"])["value"].sum().unstack().fillna(0)
    comp = per_dur_year.T
    comp["Growth %"] = pct(comp.get(PREV_YEAR, 0), comp.get(THIS_YEAR, 0))
    st.dataframe(comp.style.format("{:,.0f}"), use_container_width=True)

    # Export
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), f"duration_{label.lower()}.csv")
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="raw")
            pivot_full.to_excel(writer, sheet_name="pivot")
        buf.seek(0)
        st.download_button("⬇️ Download Excel", buf, f"duration_{label.lower()}.xlsx")
    except Exception as e:
        st.warning(f"Excel export unavailable: {e}")

    # AI Summary
    if enable_ai:
        with st.spinner("🧠 Generating AI summary..."):
            sample = comp.head(5).to_dict()
            prompt = f"Summarize {label} registration trends. Key stats: prev={int(prev_total)}, this={int(this_total)}, next_pred={int(next_total)}. Data={json.dumps(sample,default=str)}"
            ai_out = deepinfra_chat("You are a Vahan analytics assistant.", prompt, max_tokens=350)
            if ai_out and "text" in ai_out:
                st.markdown("### 🤖 AI Summary")
                st.write(ai_out["text"])

    st.success(f"✅ {label} Duration Analysis Complete.")
    return df

# =====================================================
# 🚀 Run all duration types
# =====================================================
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

# =====================================================
# 🧠 Local Forecast Helper (Prophet preferred, Linear fallback)
# =====================================================
def forecast_series_local(series: pd.Series, periods:int=12, freq:str="MS"):
    """
    Forecast a numeric time series using Prophet if available, else LinearRegression.
    """
    try:
        s = series.dropna().sort_index()
        if len(s) < 6:
            return None

        if PROPHET_AVAILABLE and len(s) >= 12:
            from prophet import Prophet
            dfp = s.reset_index().rename(columns={s.index.name or 'index':'ds', s.name:'y'})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=periods, freq=freq)
            fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].set_index("ds")
            return fc.tail(periods)
        else:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(s)).reshape(-1,1)
            y = s.values
            lr = LinearRegression().fit(X, y)
            future_X = np.arange(len(s), len(s)+periods).reshape(-1,1)
            yhat = lr.predict(future_X)
            next_idx = pd.date_range(start=(pd.to_datetime(s.index.max()) + pd.offsets.MonthBegin(1)).replace(day=1), periods=periods, freq=freq)
            fc = pd.DataFrame({
                "yhat": np.maximum(yhat, 0),
                "yhat_lower": np.maximum(yhat*0.85, 0),
                "yhat_upper": np.maximum(yhat*1.15, 0),
            }, index=next_idx)
            return fc
    except Exception:
        return None


def pct_change(a, b):
    try:
        if a == 0:
            return float("inf") if b != 0 else 0.0
        return ((b - a) / abs(a)) * 100.0
    except Exception:
        return 0.0


# =====================================================
# 🌐 Fetch & Normalize Data
# =====================================================
with st.spinner("📡 Fetching Top 5 Revenue States (MAXED)..."):
    top5_rev_json = fetch_json("vahandashboard/top5chartRevenueFee", params_common, desc="Top 5 Revenue States")
    raw_rev = parse_top5_revenue(top5_rev_json if top5_rev_json else {})


def normalize_revenue_df(df):
    """Normalize revenue dataframe into ['state','date','revenue'] format."""
    if df is None:
        return pd.DataFrame(columns=["state","date","revenue"])

    d = df.copy()
    cols = {c.lower():c for c in d.columns}

    state_col = cols.get("state") or cols.get("state_name") or cols.get("label") or (d.columns[0] if len(d.columns)>0 else None)
    rev_col = cols.get("revenue") or cols.get("value") or cols.get("amount") or cols.get("fee") or (d.select_dtypes(include=[np.number]).columns[0] if d.select_dtypes(include=[np.number]).shape[1]>0 else None)
    date_col = cols.get("date") or cols.get("period") or cols.get("month") or cols.get("year")

    if state_col: d = d.rename(columns={state_col:"state"})
    else: d["state"] = "Unknown"

    if rev_col: d = d.rename(columns={rev_col:"revenue"})
    else: d["revenue"] = 0.0

    if date_col:
        d = d.rename(columns={date_col:"date"})
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    else:
        d["date"] = pd.to_datetime(datetime(THIS_YEAR,1,1))

    d["state"] = d["state"].astype(str)
    d["revenue"] = pd.to_numeric(d["revenue"], errors="coerce").fillna(0.0)
    return d[["state","date","revenue"]]


df_rev = normalize_revenue_df(raw_rev)

if df_rev.empty:
    st.warning("⚠️ No revenue data available.")
else:
    # =====================================================
    # 🧹 Data Processing
    # =====================================================
    df_rev["month"] = df_rev["date"].dt.to_period("M").dt.to_timestamp()
    df_rev["year"] = df_rev["date"].dt.year

    totals_by_state = df_rev.groupby("state", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    top_states = totals_by_state.head(10)
    overall_revenue = totals_by_state["revenue"].sum()

    monthly_totals = df_rev.groupby("month", as_index=True)["revenue"].sum().sort_index()
    yearly = df_rev.groupby(["year","state"], as_index=False)["revenue"].sum()
    yearly_pivot = yearly.pivot(index="year", columns="state", values="revenue").fillna(0).sort_index()

    prev_total = float(yearly_pivot.loc[PREV_YEAR].sum()) if PREV_YEAR in yearly_pivot.index else 0.0
    this_total = float(yearly_pivot.loc[THIS_YEAR].sum()) if THIS_YEAR in yearly_pivot.index else 0.0

    fc_overall = forecast_series_local(monthly_totals, periods=12, freq="MS")
    predicted_next_total = float(fc_overall["yhat"].sum()) if fc_overall is not None else 0.0

    # Per-state forecasts
    monthly_by_state = df_rev.groupby(["month","state"], as_index=False)["revenue"].sum()
    monthly_pivot = monthly_by_state.pivot(index="month", columns="state", values="revenue").fillna(0).sort_index()

    state_forecasts = {}
    for st_name in monthly_pivot.columns:
        fc = forecast_series_local(monthly_pivot[st_name], periods=12, freq="MS")
        state_forecasts[st_name] = fc

    hist_share = totals_by_state.set_index("state")["revenue"] / (totals_by_state["revenue"].sum() or 1.0)
    predicted_per_state = {
        s: (float(fc["yhat"].sum()) if fc is not None else float(hist_share.get(s, 0.0)*predicted_next_total))
        for s, fc in state_forecasts.items()
    }

    # =====================================================
    # 📊 KPIs
    # =====================================================
    kpi_prev_vs_this = pct_change(prev_total, this_total)
    kpi_this_vs_next = pct_change(this_total, predicted_next_total)

    st.subheader("💰 Top 5 Revenue States — Real + Predicted (MAXED)")
    a,b,c,d = st.columns(4)
    a.metric(f"Prev Year ({PREV_YEAR})", f"₹{int(prev_total):,}", delta=f"{kpi_prev_vs_this:.2f}%")
    b.metric(f"This Year ({THIS_YEAR})", f"₹{int(this_total):,}", delta=f"{kpi_this_vs_next:.2f}%")
    c.metric(f"Predicted Next ({NEXT_YEAR})", f"₹{int(predicted_next_total):,}", delta=f"{pct_change(prev_total, predicted_next_total):.2f}%")
    d.metric("All-Time Total", f"₹{int(overall_revenue):,}")

    # =====================================================
    # 🧭 Visuals
    # =====================================================
    st.markdown("### 🥇 Top 5 Revenue States")
    top5 = totals_by_state.head(5)
    bar = alt.Chart(top5).mark_bar().encode(
        x=alt.X("revenue:Q", title="Revenue (₹)"),
        y=alt.Y("state:N", sort='-x', title="State"),
        tooltip=["state:N", alt.Tooltip("revenue:Q", format=",")]
    ).properties(height=40*min(len(top5),10), width=700)
    st.altair_chart(bar, use_container_width=True)

    pie = px.pie(top5, names="state", values="revenue", hole=0.45, title="Revenue Share (Top 5)")
    pie.update_traces(textinfo='percent+label', hoverinfo='label+value')
    st.plotly_chart(pie, use_container_width=True)

    # =====================================================
    # 📈 Monthly Trend (Real + Predicted)
    # =====================================================
    st.markdown("### 📆 Monthly Revenue — Real vs Predicted")
    real_ts = monthly_totals.reset_index().rename(columns={"month":"ds","revenue":"y"})
    if fc_overall is not None:
        fc_df = fc_overall.reset_index().rename(columns={"index":"ds"})
        plot_df = pd.concat([
            real_ts.rename(columns={"ds":"month","y":"value"}).assign(type="actual"),
            fc_df.rename(columns={"ds":"month","yhat":"value"}).assign(type="predicted")
        ], ignore_index=True)
        fig = px.line(plot_df, x="month", y="value", color="type", markers=True, title="Monthly Revenue — Real vs Predicted")
        fig.add_traces([
            dict(x=fc_df["ds"], y=fc_df["yhat_upper"], mode='lines', line=dict(width=0), showlegend=False),
            dict(x=fc_df["ds"], y=fc_df["yhat_lower"], mode='lines', fill='tonexty', fillcolor='rgba(0,176,246,0.15)', line=dict(width=0), showlegend=False)
        ])
    else:
        fig = px.line(real_ts, x="ds", y="y", title="Monthly Revenue (Historical)")
    fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 🧾 Per-State Comparison Table
    # =====================================================
    st.markdown("### 🧾 Per-State — Prev vs This vs Predicted")
    per_state_year = df_rev.groupby(["year","state"], as_index=False)["revenue"].sum().pivot(index="state", columns="year", values="revenue").fillna(0)
    rows = []
    for s in totals_by_state["state"]:
        prev_v = float(per_state_year.loc[s][PREV_YEAR]) if PREV_YEAR in per_state_year.columns else 0.0
        this_v = float(per_state_year.loc[s][THIS_YEAR]) if THIS_YEAR in per_state_year.columns else 0.0
        pred_v = float(predicted_per_state.get(s, 0.0))
        rows.append({
            "state": s,
            f"{PREV_YEAR} (₹)": prev_v,
            f"{THIS_YEAR} (₹)": this_v,
            f"{NEXT_YEAR} (pred ₹)": pred_v,
            "growth_prev→this %": pct_change(prev_v, this_v),
            "growth_this→next %": pct_change(this_v, pred_v),
            "historical_share %": float(hist_share.get(s,0))*100
        })
    comp_df = pd.DataFrame(rows).sort_values(f"{THIS_YEAR} (₹)", ascending=False)
    st.dataframe(comp_df.style.format({
        f"{PREV_YEAR} (₹)":"₹{:,}",
        f"{THIS_YEAR} (₹)":"₹{:,}",
        f"{NEXT_YEAR} (pred ₹)":"₹{:,}",
        "growth_prev→this %":"{:.2f}%",
        "growth_this→next %":"{:.2f}%",
        "historical_share %":"{:.2f}%"
    }), use_container_width=True, height=420)

    # =====================================================
    # 🌡️ Heatmap
    # =====================================================
    st.markdown("### 🌡️ Yearly Revenue Heatmap")
    heat = yearly_pivot.fillna(0)
    if not heat.empty:
        heat = heat[sorted(heat.columns, key=lambda s: -heat[s].sum())]
        fig_heat = px.imshow(heat.T, labels=dict(x="Year", y="State", color="Revenue (₹)"),
                             x=heat.index.astype(str).tolist(), y=heat.columns.tolist(), aspect="auto",
                             title="State × Year Revenue Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

    # =====================================================
    # 🚨 Anomalies
    # =====================================================
    st.markdown("### 🚨 Revenue Anomalies (MoM > 50%)")
    anomalies = []
    for s in monthly_pivot.columns:
        mom = monthly_pivot[s].pct_change().fillna(0)
        large = mom[mom.abs() > 0.5]
        for idx,v in large.items():
            anomalies.append({"state": s, "period": idx, "MoM change %": v*100, "revenue": float(monthly_pivot.loc[idx,s])})
    if anomalies:
        st.dataframe(pd.DataFrame(anomalies).sort_values("period", ascending=False).head(20))
    else:
        st.info("✅ No major MoM anomalies detected.")

    # =====================================================
    # 💾 Export
    # =====================================================
    st.markdown("### 💾 Export Data")
    csv = df_rev.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name=f"revenue_raw_{THIS_YEAR}.csv", mime="text/csv")

    try:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_rev.to_excel(writer, sheet_name="raw", index=False)
            comp_df.to_excel(writer, sheet_name="per_state", index=False)
            monthly_pivot.reset_index().to_excel(writer, sheet_name="monthly_by_state", index=False)
            if fc_overall is not None:
                fc_overall.reset_index().to_excel(writer, sheet_name="forecast_overall", index=False)
            for s, fc in state_forecasts.items():
                if fc is not None:
                    fc.reset_index().to_excel(writer, sheet_name=f"fc_{str(s)[:25]}", index=False)
        excel_buffer.seek(0)
        st.download_button("⬇️ Download Excel", excel_buffer,
                           file_name=f"revenue_analysis_{THIS_YEAR}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("Excel export failed — CSV available instead.")

    # =====================================================
    # 🧠 AI Summary (DeepInfra)
    # =====================================================
    if enable_ai:
        with st.spinner("🤖 Generating AI narrative..."):
            system = "You are a financial analyst. Analyze the revenue trends and predicted growth for Indian states, highlight top performers, laggards, and next-year expectations."
            sample = comp_df.head(7).to_dict(orient="records")
            user = f"Prev={int(prev_total):,}, This={int(this_total):,}, PredNext={int(predicted_next_total):,}. Sample data={json.dumps(sample, default=str)}. Give 6 concise bullet points + 2-line summary."
            ai_out = deepinfra_chat(system, user, max_tokens=400)
            if isinstance(ai_out, dict) and "text" in ai_out:
                st.markdown("### 🧠 AI Revenue Narrative")
                st.info(ai_out["text"])
            else:
                st.info("⚠️ No AI output or key missing.")

    st.success("✅ MAXED Revenue Analysis Complete.")

# ============================================================
# 6️⃣ FULLY MAXED — Revenue Trend (Real + Predicted + Full Analysis)
# ============================================================
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

# ============================================================
# 🧩 GLOBALS
# ============================================================
TODAY = date.today()
PREV_YEAR = TODAY.year - 1
THIS_YEAR = TODAY.year
NEXT_YEAR = THIS_YEAR + 1

# ============================================================
# 📡 FETCH & NORMALIZE DATA
# ============================================================
with st.spinner("📡 Fetching Revenue Trend (fully maxed)..."):
    rev_trend_json = fetch_json(
        "vahandashboard/revenueFeeLineChart",
        params=params_common,
        desc="Revenue Trend"
    )

def normalize_rev_df(df):
    """Safe normalization to ensure uniform schema."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["period", "value"])
    d = df.copy()
    cols = {c.lower(): c for c in d.columns}
    period_col = cols.get("period") or cols.get("date") or cols.get("month")
    value_col = cols.get("value") or cols.get("revenue") or cols.get("amount")

    if not period_col:
        period_col = d.columns[0]
    if not value_col:
        value_col = d.select_dtypes(include=[np.number]).columns[0]

    d = d.rename(columns={period_col: "period", value_col: "value"})
    d["period"] = pd.to_datetime(d["period"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(0)
    return d.dropna(subset=["period"]).sort_values("period")

df_rev = normalize_rev_df(parse_revenue_trend(rev_trend_json or {}))

if df_rev.empty:
    st.warning("⚠️ No revenue data available.")
    st.stop()

# ============================================================
# 📆 TIME COMPONENTS
# ============================================================
df_rev["year"] = df_rev["period"].dt.year
df_rev["month"] = df_rev["period"].dt.to_period("M").dt.to_timestamp()
df_rev["day"] = df_rev["period"].dt.floor("D")

# ============================================================
# 📊 AGGREGATIONS
# ============================================================
totals_by_year = df_rev.groupby("year")["value"].sum()
monthly_totals = df_rev.groupby("month")["value"].sum().sort_index()

prev_total = float(totals_by_year.get(PREV_YEAR, 0))
this_total = float(totals_by_year.get(THIS_YEAR, 0))

# ============================================================
# 🔮 FORECASTING
# ============================================================
def forecast_monthly(series, periods=12):
    try:
        s = series.dropna().sort_index()
        if len(s) < 6:
            return None
        if "prophet" in globals() or "prophet" in locals():
            from prophet import Prophet
            dfp = s.reset_index().rename(columns={"month": "ds", "value": "y"})
            dfp.columns = ["ds", "y"]
            m = Prophet(yearly_seasonality=True)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=periods, freq="MS")
            fc = m.predict(future)
            fc = fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]]
            return fc.tail(periods)
        else:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(s)).reshape(-1, 1)
            y = s.values
            lr = LinearRegression().fit(X, y)
            fut_X = np.arange(len(s), len(s) + periods).reshape(-1, 1)
            yhat = lr.predict(fut_X)
            next_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
            return pd.DataFrame({
                "yhat": yhat,
                "yhat_lower": yhat * 0.9,
                "yhat_upper": yhat * 1.1
            }, index=next_idx)
    except Exception:
        return None

fc_monthly = forecast_monthly(monthly_totals, periods=12)
predicted_next_total = (
    float(fc_monthly[fc_monthly.index.year == NEXT_YEAR]["yhat"].sum())
    if fc_monthly is not None else 0.0
)

# ============================================================
# 🧾 KPIs
# ============================================================
def pct(a, b):
    return ((b - a) / abs(a)) * 100 if a else 0.0

kpi_prev_vs_this = pct(prev_total, this_total)
kpi_this_vs_next = pct(this_total, predicted_next_total)

st.subheader("💰 Revenue Trend — Real + Forecasted (MAXED)")
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"{PREV_YEAR} Revenue", f"₹{int(prev_total):,}")
col2.metric(f"{THIS_YEAR} Revenue", f"₹{int(this_total):,}", delta=f"{kpi_prev_vs_this:.2f}% vs Prev")
col3.metric(f"{NEXT_YEAR} Forecast", f"₹{int(predicted_next_total):,}", delta=f"{kpi_this_vs_next:.2f}% vs Current")
col4.metric("Total", f"₹{int(df_rev['value'].sum()):,}")

# ============================================================
# 📈 VISUALIZATIONS
# ============================================================
st.markdown("### Interactive Revenue Trends")

smooth = st.slider("Smoothing (months)", 1, 12, 3)
ma = monthly_totals.rolling(smooth).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=monthly_totals.index, y=monthly_totals.values, name="Actual", mode="lines+markers"))
fig.add_trace(go.Scatter(x=monthly_totals.index, y=ma, name=f"{smooth}-mo MA", line=dict(dash="dot")))

if fc_monthly is not None:
    fig.add_trace(go.Scatter(x=fc_monthly.index, y=fc_monthly["yhat"], name="Forecast", line=dict(color="orange", width=3)))
    fig.add_trace(go.Scatter(
        x=list(fc_monthly.index) + list(fc_monthly.index[::-1]),
        y=list(fc_monthly["yhat_upper"]) + list(fc_monthly["yhat_lower"][::-1]),
        fill="toself", fillcolor="rgba(255,165,0,0.2)",
        line=dict(color="rgba(255,165,0,0)"), name="Confidence Band"
    ))

fig.update_layout(title="Revenue: Actual vs Forecast", xaxis_title="Month", yaxis_title="Revenue (₹)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🔍 SEASONALITY + ANOMALIES
# ============================================================
if len(monthly_totals) >= 24:
    try:
        res = sm.tsa.seasonal_decompose(monthly_totals, model="additive", period=12)
        st.markdown("### 🔍 Seasonal Decomposition")
        st.line_chart(res.trend.rename("Trend"))
        st.line_chart(res.seasonal.rename("Seasonal"))
    except Exception:
        st.info("Decomposition unavailable.")

mom = monthly_totals.pct_change()
anomalies = mom[mom.abs() > 0.5]
if not anomalies.empty:
    st.warning("⚠️ Anomalies detected (>50% MoM change)")
    st.dataframe(anomalies.rename("MoM%").mul(100).round(2))

# ============================================================
# 📤 EXPORTS
# ============================================================
st.markdown("### 📤 Export Data")
csv = df_rev.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv, file_name=f"revenue_trend_{THIS_YEAR}.csv")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_rev.to_excel(writer, sheet_name="Raw", index=False)
    monthly_totals.reset_index().to_excel(writer, sheet_name="Monthly", index=False)
    if fc_monthly is not None:
        fc_monthly.to_excel(writer, sheet_name="Forecast")
buffer.seek(0)
st.download_button("⬇️ Download Excel", buffer, file_name=f"revenue_trend_{THIS_YEAR}.xlsx")

# ============================================================
# 🤖 AI NARRATIVE (DeepInfra)
# ============================================================
if enable_ai:
    with st.spinner("🤖 Generating AI Revenue Summary..."):
        sample = monthly_totals.tail(12).reset_index().rename(columns={"month": "period", "value": "revenue"}).to_dict(orient="records")
        system_prompt = (
            "You are a senior financial data analyst. Summarize the following revenue trend data, "
            "highlighting YoY, MoM, anomalies, and forecast with 3 strategic recommendations."
        )
        user_prompt = f"Recent data: {json.dumps(sample, default=str)}\nYoY change: {kpi_prev_vs_this:.2f}% | Predicted next year ₹{int(predicted_next_total):,}"
        ai_out = deepinfra_chat(system_prompt, user_prompt, max_tokens=400)
        if ai_out and "text" in ai_out:
            st.markdown("### 🧠 AI-Generated Insight")
            st.info(ai_out["text"])
        else:
            st.info("⚠️ No AI output received.")

st.success("✅ Revenue Trend (Fully Maxed) completed.")

# =====================================================
# 7️⃣ MAXED — Forecasting (Prev / This / Next Year — Real + Predicted + Full Analysis)
# =====================================================
import io, math, json, numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go, warnings
from datetime import date
warnings.filterwarnings("ignore")

TODAY = date.today()
PREV_YEAR, THIS_YEAR, NEXT_YEAR = TODAY.year - 1, TODAY.year, TODAY.year + 1

# =====================================================
# 🧠 Helper Functions
# =====================================================
def pct(a, b):
    return ((b - a) / abs(a)) * 100 if a else 0

def auto_agg_timescale(df):
    """Automatically infer daily / monthly / yearly frequency."""
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
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
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
        "lower": fc.conf_int().iloc[:, 0],
        "upper": fc.conf_int().iloc[:, 1],
        "type": "Predicted"
    })
    return pred_df

def linear_forecast(df, months=12):
    from sklearn.linear_model import LinearRegression
    df = df.copy().sort_values("date")
    df["x"] = np.arange(len(df))
    X, y = df[["x"]].values, df["value"].values
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

# =====================================================
# 📊 Main Logic
# =====================================================
if "df_trend" in globals() and not df_trend.empty:
    st.subheader("🔮 Forecasting — Real + Predicted + Full MAXED Insights")

    df_trend = auto_agg_timescale(df_trend)
    df_trend["type"] = "Actual"
    df_trend = df_trend.sort_values("date")

    # Model selection order: Prophet → ARIMA → Linear
    forecast_periods = 12
    forecast_df, model_used = pd.DataFrame(), "N/A"
    try:
        forecast_df = prophet_forecast(df_trend, forecast_periods)
        model_used = "Prophet"
    except Exception as e1:
        try:
            st.warning(f"Prophet failed → trying ARIMA ({e1})")
            forecast_df = arima_forecast(df_trend, forecast_periods)
            model_used = "ARIMA"
        except Exception as e2:
            st.warning(f"ARIMA failed → using Linear Regression ({e2})")
            forecast_df = linear_forecast(df_trend, forecast_periods)
            model_used = "Linear Regression"

    # Merge + clean
    full_df = pd.concat([df_trend, forecast_df], ignore_index=True).sort_values("date")
    full_df["year"] = full_df["date"].dt.year

    # =====================================================
    # 📈 KPI Metrics
    # =====================================================
    year_totals = full_df.groupby(["year", "type"], as_index=False)["value"].sum()
    prev_actual = year_totals.query("year == @PREV_YEAR and type == 'Actual'")["value"].sum()
    this_actual = year_totals.query("year == @THIS_YEAR and type == 'Actual'")["value"].sum()
    next_pred = year_totals.query("year == @NEXT_YEAR and type == 'Predicted'")["value"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Prev Year ({PREV_YEAR})", f"₹{int(prev_actual):,}")
    c2.metric(f"This Year ({THIS_YEAR})", f"₹{int(this_actual):,}", delta=f"{pct(prev_actual,this_actual):.2f}% vs Prev")
    c3.metric(f"Predicted ({NEXT_YEAR})", f"₹{int(next_pred):,}", delta=f"{pct(this_actual,next_pred):.2f}% vs This")

    st.markdown(f"**Model Used:** `{model_used}`")

    # =====================================================
    # 📉 Forecast Visualization
    # =====================================================
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_trend["date"], y=df_trend["value"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["value"], name="Predicted", mode="lines+markers", line=dict(color="orange")))
    fig.add_trace(go.Scatter(
        x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]),
        y=list(forecast_df["upper"]) + list(forecast_df["lower"][::-1]),
        fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,165,0,0)'),
        hoverinfo="skip", showlegend=True, name="Confidence Band"
    ))
    fig.update_layout(title="Forecast: Actual vs Predicted", xaxis_title="Date", yaxis_title="Value (₹)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 🔍 Deep Insights
    # =====================================================
    st.markdown("### 🔍 Deep Insights & Metrics")
    df_trend["MA(3)"] = df_trend["value"].rolling(3).mean()
    df_trend["YoY%"] = df_trend["value"].pct_change(12) * 100
    st.dataframe(df_trend.tail(12).style.format({"value":"₹{:,}", "YoY%":"{:.2f}%"}), use_container_width=True)

    # Heatmap
    heat = df_trend.copy()
    heat["month"], heat["year"] = heat["date"].dt.month, heat["date"].dt.year
    pivot = heat.pivot_table(index="year", columns="month", values="value", aggfunc="sum")
    fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis", title="Heatmap: Revenue by Month-Year")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Cumulative
    cum_df = full_df.copy()
    cum_df["cumulative"] = cum_df["value"].cumsum()
    fig_cum = px.area(cum_df, x="date", y="cumulative", color="type", title="Cumulative Revenue (Actual + Forecast)")
    st.plotly_chart(fig_cum, use_container_width=True)

    # =====================================================
    # 📈 Statistical Summary
    # =====================================================
    desc = df_trend["value"].describe().to_frame().T
    desc["latest_MoM%"] = df_trend["value"].pct_change().iloc[-1] * 100
    desc["latest_YoY%"] = df_trend["YoY%"].iloc[-1]
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    # =====================================================
    # 📤 Export Options
    # =====================================================
    csv = full_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Forecast Data (CSV)", csv, file_name=f"forecast_full_{THIS_YEAR}.csv")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_trend.to_excel(writer, sheet_name="Actual", index=False)
        forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
        full_df.to_excel(writer, sheet_name="Combined", index=False)
    output.seek(0)
    st.download_button("⬇️ Download Forecast Data (Excel)", output, file_name=f"forecast_full_{THIS_YEAR}.xlsx")

    # =====================================================
    # 🤖 AI Narrative
    # =====================================================
    if "enable_ai" in globals() and enable_ai:
        with st.spinner("🤖 Generating AI Forecast Summary..."):
            sample = forecast_df.head(12).to_dict(orient="records")
            system = (
                "You are a senior data scientist providing an executive summary of forecast data. "
                "Be concise, business-oriented, and mention risks and opportunities."
            )
            user = f"""Forecast data: {json.dumps(sample, default=str)}.
            Actual {THIS_YEAR} = {this_actual:.0f}, Predicted {NEXT_YEAR} = {next_pred:.0f}.
            Write a 5-line summary including:
            1. Key trend
            2. Growth/Fall %
            3. Confidence level
            4. 2 main risks
            5. 2 strategic recommendations."""
            ai_out = deepinfra_chat(system, user, max_tokens=400)
            if isinstance(ai_out, dict) and "text" in ai_out:
                st.markdown("### 🧠 AI Forecast Summary")
                st.info(ai_out["text"])
            else:
                st.info("⚠️ No AI summary returned.")

    st.success("✅ MAXED Forecasting Completed Successfully.")
else:
    st.warning("⚠️ Trend data not found or empty.")

# ============================================================
# 🚀 MAXED++ Anomaly Detection, Explainability & Auto-Insights
# ============================================================
import numpy as np
import pandas as pd
import io, json, math, traceback, time, warnings
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime

warnings.filterwarnings("ignore")
TODAY = date.today()
THIS_YEAR = TODAY.year
PREV_YEAR = THIS_YEAR - 1
NEXT_YEAR = THIS_YEAR + 1

# ------------------------------------------------------------
# 🚦 Run only if user enabled anomaly detection
# ------------------------------------------------------------
if enable_anomaly:
    st.header("🚨 MAXED++ Anomaly Detection & Explainability")
    if df_trend is None or df_trend.empty:
        st.info("No trend data available for anomaly detection.")
    else:
        # ----------------------------------------
        # ⚙️ Controls
        # ----------------------------------------
        colA, colB, colC = st.columns([1.3, 1.2, 1])
        with colA:
            method = st.selectbox("Detection Method", [
                "IsolationForest", "LocalOutlierFactor",
                "Rolling Z-score", "STL Residuals", "Prophet Residuals"
            ], index=0)
            contamination = st.slider("Contamination (expected % outliers)", 0.001, 0.2, 0.02, step=0.001)
            threshold = st.slider("Z/residual threshold", 2.0, 8.0, 3.5, 0.1)
        with colB:
            gran = st.selectbox("Granularity", ["Daily", "Monthly", "Yearly"], index=1)
            ensemble = st.checkbox("Run ensemble detectors", True)
        with colC:
            detect_scope = st.selectbox("Detection scope", [
                "Global (all data)", "Per-category", "Per-state", "Per-maker"
            ], index=0)
            topN = st.number_input("Preview top N anomalies", 5, 100, 20)

        st.divider()

        # ----------------------------------------
        # 🧮 Aggregate as per granularity
        # ----------------------------------------
        df = df_trend.copy()
        if "date" not in df.columns or "value" not in df.columns:
            st.error("Missing required 'date' or 'value' columns.")
        else:
            df = df.sort_values("date").reset_index(drop=True)
            if gran == "Daily":
                df["period"] = df["date"].dt.floor("D")
            elif gran == "Monthly":
                df["period"] = df["date"].dt.to_period("M").dt.to_timestamp()
            else:
                df["period"] = df["date"].dt.to_period("Y").dt.to_timestamp()
            ts = df.groupby("period", as_index=False)["value"].sum()

            # ----------------------------------------
            # 🧠 Detectors
            # ----------------------------------------
            def safe_run(name, func, *args, **kwargs):
                try:
                    return func(*args, **kwargs), "ok"
                except Exception as e:
                    return None, str(e)

            def isolationforest(series):
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
                vals = series.values.reshape(-1, 1)
                model.fit(vals)
                scores = -model.decision_function(vals)
                preds = model.predict(vals)
                df = pd.DataFrame({
                    "date": series.index, "value": series.values,
                    "score": scores, "flag": preds == -1
                })
                return df

            def lof(series):
                from sklearn.neighbors import LocalOutlierFactor
                vals = series.values.reshape(-1, 1)
                model = LocalOutlierFactor(contamination=contamination)
                preds = model.fit_predict(vals)
                scores = -model.negative_outlier_factor_
                df = pd.DataFrame({
                    "date": series.index, "value": series.values,
                    "score": scores, "flag": preds == -1
                })
                return df

            def zscore(series):
                roll_mean = series.rolling(12, min_periods=3).mean()
                roll_std = series.rolling(12, min_periods=3).std().replace(0, np.nan).fillna(1)
                z = (series - roll_mean) / roll_std
                df = pd.DataFrame({
                    "date": series.index, "value": series.values,
                    "score": z.abs(), "flag": z.abs() > threshold
                })
                return df

            def stl(series):
                import statsmodels.api as sm
                res = sm.tsa.seasonal_decompose(series, model="additive", period=12, extrapolate_trend='freq')
                resid = res.resid.fillna(0)
                z = (resid - resid.mean()) / (resid.std() or 1.0)
                df = pd.DataFrame({
                    "date": series.index, "value": series.values,
                    "score": np.abs(z), "flag": np.abs(z) > threshold
                })
                return df

            def prophet_resid(df_full):
                from prophet import Prophet
                dfp = df_full.rename(columns={"date": "ds", "value": "y"})
                m = Prophet(yearly_seasonality=True)
                m.fit(dfp)
                pred = m.predict(dfp[["ds"]])
                resid = dfp["y"] - pred["yhat"]
                z = (resid - resid.mean()) / (resid.std() or 1)
                df = pd.DataFrame({
                    "date": dfp["ds"], "value": dfp["y"],
                    "score": np.abs(z), "flag": np.abs(z) > threshold
                })
                return df

            series = ts.set_index("period")["value"]

            results = {}
            fns = {
                "IsolationForest": isolationforest,
                "LocalOutlierFactor": lof,
                "Rolling Z-score": zscore,
                "STL Residuals": stl,
                "Prophet Residuals": prophet_resid
            }

            res_main, status = safe_run(method, fns[method], series if method != "Prophet Residuals" else ts)
            results[method] = {"df": res_main, "status": status}

            # ----------------------------------------
            # 🧩 Ensemble (optional)
            # ----------------------------------------
            if ensemble:
                for mname, fn in fns.items():
                    if mname == method:
                        continue
                    r, s = safe_run(mname, fn, series if mname != "Prophet Residuals" else ts)
                    results[mname] = {"df": r, "status": s}

            # ----------------------------------------
            # ⚖️ Ensemble aggregation
            # ----------------------------------------
            ens = pd.DataFrame(index=series.index)
            for n, info in results.items():
                dfres = info["df"]
                if isinstance(dfres, pd.DataFrame):
                    tmp = dfres.set_index("date")
                    ens[f"{n}_flag"] = tmp["flag"].astype(int)
                    ens[f"{n}_score"] = (tmp["score"] - tmp["score"].min()) / (tmp["score"].max() - tmp["score"].min() + 1e-6)

            ens["vote"] = ens.filter(like="_flag").sum(axis=1)
            ens["score_mean"] = ens.filter(like="_score").mean(axis=1)
            ens["final_flag"] = (ens["vote"] >= math.ceil(len([c for c in ens.columns if c.endswith('_flag')])/2)) | (ens["score_mean"] > 0.7)

            # ----------------------------------------
            # 📊 Visualization
            # ----------------------------------------
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts["period"], y=ts["value"], mode="lines", name="Value"))
            anoms = ens[ens["final_flag"]].reset_index()
            if not anoms.empty:
                fig.add_trace(go.Scatter(
                    x=anoms["period"], y=ts.set_index("period").reindex(anoms["period"])["value"],
                    mode="markers", name="Anomalies", marker=dict(color="red", size=10, symbol="x")
                ))
            fig.update_layout(title="Detected Anomalies", xaxis_title="Date", yaxis_title="Registrations")
            st.plotly_chart(fig, use_container_width=True)

            # ----------------------------------------
            # 🧾 Table
            # ----------------------------------------
            if not anoms.empty:
                preview = ts.set_index("period").reindex(anoms["period"]).reset_index()
                preview["score"] = anoms["score_mean"].values
                preview = preview.sort_values("score", ascending=False).head(topN)
                st.dataframe(preview.style.background_gradient("Reds"), use_container_width=True)
            else:
                st.info("No anomalies detected.")

            # ----------------------------------------
            # 🧠 AI-based Narrative
            # ----------------------------------------
            if enable_ai and not anoms.empty:
                with st.spinner("🤖 Generating AI-based anomaly explanations..."):
                    sample = preview.to_dict(orient="records")
                    system = (
                        "You are a senior data analyst for vehicle registrations. "
                        "Explain the likely causes behind each anomaly, "
                        "identify whether they are due to data errors, policy changes, or seasonality, "
                        "and summarize the key trends."
                    )
                    user = f"Here are the anomalies: {json.dumps(sample, default=str)}"
                    ai_resp = deepinfra_chat(system, user, max_tokens=700)
                    if isinstance(ai_resp, dict) and "text" in ai_resp:
                        st.markdown("### 🧩 AI Anomaly Summary")
                        st.info(ai_resp["text"])

            # ----------------------------------------
            # 💾 Export
            # ----------------------------------------
            buf = ens.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download anomaly data", buf, "maxed_anomalies.csv", "text/csv")

            # ----------------------------------------
            # 🧭 Diagnostics
            # ----------------------------------------
            st.markdown("#### Detector status:")
            for name, info in results.items():
                st.write(f"- **{name}** → {info['status']}")
else:
    st.info("Anomaly detection disabled.")

# ==========================================================
# ⚡ MAXED — Clustering, Prediction & Correlation Suite (Full AI + Auto Models)
# ==========================================================
if enable_clustering:
    st.markdown("## 🧠 MAXED Clustering, Prediction & Correlation Intelligence")
    st.markdown("Automated cluster tuning, correlation mining, and hybrid model predictions with AI insights.")

    import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import warnings
    warnings.filterwarnings("ignore")

    # ---------------- Base data setup ----------------
    base_df = df_top5_rev.copy() if not df_top5_rev.empty else df_trend.copy() if not df_trend.empty else pd.DataFrame()
    if base_df.empty:
        st.warning("⚠️ No base data available for clustering or prediction.")
    else:
        # Normalize and clean
        if "date" in base_df.columns:
            base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
            base_df["year"] = base_df["date"].dt.year
            base_df["month"] = base_df["date"].dt.month
        base_df["value"] = pd.to_numeric(base_df.get("value", base_df.iloc[:, -1]), errors="coerce").fillna(0)

        # ===================================================
        # 1️⃣ Yearly comparison & multi-model forecasting
        # ===================================================
        st.markdown("### 📆 Yearly Comparison & Forecasting (Linear + RF + Prophet)")
        yearwise = base_df.groupby("year", as_index=False)["value"].sum()
        if not yearwise.empty:
            prev_year, this_year, next_year = yearwise["year"].max()-1, yearwise["year"].max(), yearwise["year"].max()+1
            prev_val = yearwise.loc[yearwise["year"]==prev_year, "value"].sum()
            curr_val = yearwise.loc[yearwise["year"]==this_year, "value"].sum()
            growth = ((curr_val-prev_val)/prev_val*100) if prev_val else 0
            st.metric("📊 Current Year", f"{curr_val:,.0f}")
            st.metric("📈 Prev Year", f"{prev_val:,.0f}")
            st.metric("🚀 YoY Growth", f"{growth:.2f}%")

            # Models
            X = yearwise[["year"]]
            y = yearwise["value"]

            preds = {}
            # Linear
            lr = LinearRegression().fit(X, y)
            preds["Linear"] = lr.predict([[next_year]])[0]

            # Random Forest
            rf = RandomForestRegressor(random_state=42).fit(X, y)
            preds["RandomForest"] = rf.predict([[next_year]])[0]

            # Prophet (optional)
            try:
                from prophet import Prophet
                dfp = yearwise.rename(columns={"year":"ds","value":"y"})
                dfp["ds"] = pd.to_datetime(dfp["ds"], format="%Y")
                m = Prophet(yearly_seasonality=True).fit(dfp)
                fc = m.make_future_dataframe(periods=1, freq="Y")
                fcst = m.predict(fc).iloc[-1]["yhat"]
                preds["Prophet"] = fcst
            except Exception:
                preds["Prophet"] = None

            pred_avg = np.nanmean([v for v in preds.values() if v])
            st.metric("🔮 Hybrid Predicted Next Year", f"{pred_avg:,.0f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yearwise["year"], y=yearwise["value"], name="Actual", mode="lines+markers"))
            fig.add_trace(go.Scatter(x=[next_year], y=[pred_avg], name="Predicted", mode="markers", marker=dict(size=12, color="orange")))
            fig.update_layout(title="Yearly Actual vs Forecast", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # ===================================================
        # 2️⃣ Rolling trends & seasonality
        # ===================================================
        if "date" in base_df.columns:
            st.markdown("### 📅 Rolling & Seasonal Patterns")
            df_roll = base_df.set_index("date").resample("M").sum()
            df_roll["rolling_mean_3"] = df_roll["value"].rolling(3).mean()
            st.line_chart(df_roll[["value","rolling_mean_3"]])
            fig_box = px.box(base_df, x="month", y="value", color="year", title="Monthly Distribution by Year")
            st.plotly_chart(fig_box, use_container_width=True)

        # ===================================================
        # 3️⃣ Auto Clustering & PCA Projection
        # ===================================================
        st.markdown("### 🧩 Auto Clustering & PCA Insights")
        num_cols = base_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            X = base_df[num_cols].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            # Auto-optimal cluster (2–8)
            sil_scores = {}
            for k in range(2, min(8, len(X_scaled))):
                km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
                sil_scores[k] = silhouette_score(X_scaled, km.labels_)
            best_k = max(sil_scores, key=sil_scores.get)
            st.metric("🎯 Optimal Clusters (k)", best_k)
            st.metric("📊 Best Silhouette", f"{sil_scores[best_k]:.3f}")

            kmeans = KMeans(n_clusters=best_k, random_state=42).fit(X_scaled)
            base_df["cluster"] = kmeans.labels_

            # PCA 3D visualization
            pca = PCA(n_components=3)
            proj = pca.fit_transform(X_scaled)
            proj_df = pd.DataFrame({
                "PC1": proj[:,0], "PC2": proj[:,1], "PC3": proj[:,2],
                "Cluster": base_df["cluster"].astype(str)
            })
            fig3d = px.scatter_3d(proj_df, x="PC1", y="PC2", z="PC3",
                                  color="Cluster", title="3D Cluster Projection (PCA)", opacity=0.8)
            st.plotly_chart(fig3d, use_container_width=True)

        # ===================================================
        # 4️⃣ Correlation Heatmap + Key Pair Finder
        # ===================================================
        st.markdown("### 🔗 Correlation & Top Relationships")
        corr = base_df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

        # Top correlated pairs
        corr_pairs = corr.unstack().dropna()
        corr_pairs = corr_pairs[(corr_pairs != 1)]
        top_corrs = corr_pairs.abs().sort_values(ascending=False).head(5)
        st.write("**Top 5 correlated features:**")
        st.dataframe(top_corrs.reset_index().rename(columns={"level_0":"Feature A","level_1":"Feature B",0:"|r|"}) )

        # ===================================================
        # 5️⃣ Feature Importance (RF)
        # ===================================================
        st.markdown("### 🌳 Feature Importance in Predicting Value")
        try:
            Xf = base_df[num_cols].drop(columns=["value"], errors="ignore")
            yf = base_df["value"]
            rf_imp = RandomForestRegressor(random_state=42).fit(Xf, yf)
            imp_df = pd.DataFrame({"Feature": Xf.columns, "Importance": rf_imp.feature_importances_}).sort_values("Importance", ascending=False)
            fig_imp = px.bar(imp_df, x="Feature", y="Importance", title="Random Forest Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Feature importance failed: {e}")

        # ===================================================
        # 6️⃣ Export Data
        # ===================================================
        csv = base_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Clustering Dataset (CSV)", csv, file_name=f"maxed_cluster_{this_year}.csv")

        # ===================================================
        # 7️⃣ AI Insights
        # ===================================================
        if enable_ai:
            with st.spinner("🤖 Generating AI-driven insights..."):
                system = "You are a senior data scientist. Summarize findings on clusters, correlations, and forecasts."
                summary_data = {
                    "best_k": int(best_k),
                    "silhouette": round(sil_scores[best_k],3),
                    "next_year_pred": float(pred_avg),
                    "top_corr": top_corrs.head(3).to_dict(),
                    "growth_pct": float(growth)
                }
                user = f"Data Summary:\n{json.dumps(summary_data, indent=2)}\nProvide:\n1️⃣ 5 insights on trends, clusters, correlations\n2️⃣ 3 business actions.\nKeep it concise & analytical."
                ai_out = deepinfra_chat(system, user, max_tokens=500)
                if isinstance(ai_out, dict) and "text" in ai_out:
                    st.markdown("### 🤖 AI Summary")
                    st.write(ai_out["text"])
                else:
                    st.info("AI summary unavailable.")

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
