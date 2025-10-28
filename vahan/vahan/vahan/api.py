# vahan/api.py
"""
MAXED vahan/api.py
- Full error handling (HTTP 4xx/5xx, timeouts, connection errors, JSON errors)
- Exponential backoff retries
- Random browser-like headers & UA rotation
- Optional Playwright headless fallback for 403 / bot-blocked endpoints
- Smart conditional file cache (TTL), DOES NOT cache error/403/empty results
- Bulk/concurrent fetching helpers
- Utilities: clear_cache(), set_manual_cookie(), set_proxy(), enable_playwright_fallback()
- Returns (data, url) for all fetch functions
"""

from urllib.parse import urlencode
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import requests
import json
import logging
import random
import time
import os
import hashlib
import pickle
from tqdm import tqdm
from colorama import Fore, Style, init as color_init
from typing import Optional, Tuple, Any, Dict, List

# Optional Playwright import (safe)
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# ---------------- CONFIG ----------------
BASE = os.getenv("VAHAN_API_BASE", "https://analytics.parivahan.gov.in/analytics/publicdashboard")
DEFAULT_TIMEOUT = float(os.getenv("VAHAN_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("VAHAN_MAX_RETRIES", "5"))
BACKOFF_FACTOR = float(os.getenv("VAHAN_BACKOFF", "1.5"))
MAX_WORKERS = int(os.getenv("VAHAN_MAX_WORKERS", "8"))
CACHE_DIR = os.getenv("VAHAN_CACHE_DIR", "vahan_cache")
CACHE_TTL_SECONDS = int(os.getenv("VAHAN_CACHE_TTL_SEC", str(60 * 60)))  # default 1 hour
RANDOM_UAS = [
    # Modern browser UAs; add more if desired
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0"
]

# ---------------- LOGGING ----------------
color_init(autoreset=True)
logging.basicConfig(level=logging.INFO, format=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} | {Fore.YELLOW}%(levelname)s{Style.RESET_ALL} | %(message)s")
logger = logging.getLogger("vahan_api")

# ---------------- SESSION & PROXY ----------------
session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES))
session.mount("http://", requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES))
_GLOBAL_PROXY: Optional[Dict[str, str]] = None
_MANUAL_COOKIE: Optional[Dict[str, str]] = None
_PLAYWRIGHT_FALLBACK_ENABLED = PLAYWRIGHT_AVAILABLE  # by default True if available

# ---------------- CACHE UTILITIES ----------------
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.pkl")

def load_from_cache(url: str) -> Optional[Any]:
    path = _cache_key(url)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            ts, data = pickle.load(f)
        if (datetime.utcnow() - ts).total_seconds() > CACHE_TTL_SECONDS:
            try:
                os.remove(path)
            except Exception:
                pass
            return None
        logger.info(Fore.CYAN + f"💾 Cache hit: {url}")
        return data
    except Exception as e:
        logger.warning(Fore.YELLOW + f"Cache load failed: {e}")
        return None

def save_to_cache(url: str, data: Any) -> None:
    # Only save valid, non-empty responses
    if data is None:
        return
    # For dict/list: must be non-empty
    if isinstance(data, (dict, list)) and len(data) == 0:
        return
    path = _cache_key(url)
    try:
        with open(path, "wb") as f:
            pickle.dump((datetime.utcnow(), data), f)
        logger.info(Fore.CYAN + f"💾 Saved to cache: {url}")
    except Exception as e:
        logger.warning(Fore.YELLOW + f"Cache save failed: {e}")

def clear_cache() -> None:
    try:
        for fname in os.listdir(CACHE_DIR):
            fpath = os.path.join(CACHE_DIR, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        logger.info(Fore.GREEN + "🧹 Cache cleared")
    except Exception as e:
        logger.error(Fore.RED + f"Cache clear failed: {e}")

# ---------------- CONFIG HELPERS ----------------
def set_proxy(proxy_dict: Dict[str, str]) -> None:
    """Set global proxy, e.g. {'http': 'http://...', 'https': 'http://...'}"""
    global _GLOBAL_PROXY, session
    _GLOBAL_PROXY = proxy_dict
    session.proxies.update(proxy_dict)
    logger.info(Fore.CYAN + f"Proxy set: {proxy_dict}")

def set_manual_cookie(cookie_dict: Dict[str, str]) -> None:
    """Set manual cookies (e.g., {'JSESSIONID': 'abc...'}) to reuse a browser session"""
    global _MANUAL_COOKIE
    _MANUAL_COOKIE = cookie_dict
    session.cookies.update(cookie_dict)
    logger.info(Fore.CYAN + "Manual cookies set for session.")

def enable_playwright_fallback(enable: bool) -> None:
    global _PLAYWRIGHT_FALLBACK_ENABLED
    if enable and not PLAYWRIGHT_AVAILABLE:
        logger.warning(Fore.YELLOW + "Playwright not installed; cannot enable fallback.")
    _PLAYWRIGHT_FALLBACK_ENABLED = bool(enable)
    logger.info(Fore.CYAN + f"Playwright fallback enabled: {_PLAYWRIGHT_FALLBACK_ENABLED}")

# ---------------- HEADER ROTATION ----------------
def _random_headers() -> Dict[str, str]:
    ua = random.choice(RANDOM_UAS)
    headers = {
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://analytics.parivahan.gov.in/analytics/vahanpublicreport?lang=en",
        "Origin": "https://analytics.parivahan.gov.in",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    }
    return headers

# ---------------- PLAYWRIGHT FALLBACK ----------------
def _playwright_fetch(url: str, timeout: int = 60) -> Optional[Any]:
    """Use Playwright headless browser to fetch endpoint content. Try to parse JSON from body text."""
    if not PLAYWRIGHT_AVAILABLE:
        logger.error(Fore.RED + "Playwright not available.")
        return None
    try:
        logger.info(Fore.YELLOW + f"🎭 Playwright fetching: {url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent=random.choice(RANDOM_UAS))
            page = context.new_page()
            # if manual cookies are provided, set them
            if _MANUAL_COOKIE:
                cookies = [{"name": k, "value": v, "domain": "analytics.parivahan.gov.in"} for k, v in _MANUAL_COOKIE.items()]
                context.add_cookies(cookies)
            page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            # The publicdashboard endpoints usually return JSON body — get innerText
            body = page.evaluate("() => document.body.innerText")
            browser.close()
        # Try to parse JSON
        try:
            return json.loads(body)
        except Exception:
            logger.warning(Fore.YELLOW + "Playwright fetched content but JSON parse failed — returning raw text snippet.")
            return {"raw_text_snippet": body[:2000]}
    except PlaywrightTimeoutError as e:
        logger.warning(Fore.YELLOW + f"Playwright timeout: {e}")
        return None
    except Exception as e:
        logger.error(Fore.RED + f"Playwright fetch error: {e}")
        return None

# ---------------- SAFE GET (retry/backoff/all-errors) ----------------
def _safe_get(url: str, timeout: int = int(DEFAULT_TIMEOUT)) -> Optional[Any]:
    """Core resilient fetcher. Will:
       - try requests w/ randomized headers,
       - handle connection/timeouts/HTTP errors,
       - fallback to Playwright for 403 when enabled,
       - return parsed JSON or special dicts (raw_text, html snippet)
    """
    # first quick validation
    if not url or not url.startswith("http"):
        logger.error(Fore.RED + f"Invalid URL: {url}")
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        headers = _random_headers()
        try:
            logger.info(Fore.GREEN + f"Fetching ({attempt}/{MAX_RETRIES}): {url}")
            resp = session.get(url, headers=headers, timeout=timeout, proxies=_GLOBAL_PROXY or {})
            status = getattr(resp, "status_code", None)
            # If 403 and playwright enabled, try playwright fallback (but do not cache 403)
            if status == 403:
                logger.warning(Fore.YELLOW + f"HTTP 403 for {url}")
                if _PLAYWRIGHT_FALLBACK_ENABLED:
                    data = _playwright_fetch(url, timeout=timeout)
                    return data
                else:
                    return None
            # handle 2xx
            resp.raise_for_status()
            # try JSON
            try:
                return resp.json()
            except json.JSONDecodeError:
                # return raw text if JSON fails
                logger.warning(Fore.YELLOW + "JSON decode failed; returning raw text.")
                return {"raw_text": resp.text[:2000]}
        except requests.Timeout as e:
            logger.warning(Fore.YELLOW + f"Timeout ({attempt}): {e}")
        except requests.ConnectionError as e:
            logger.warning(Fore.YELLOW + f"Connection error ({attempt}): {e}")
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else "Unknown"
            logger.warning(Fore.RED + f"HTTP error {code}: {e}")
            # For client errors (4xx except 429), do not retry except maybe 429
            if 400 <= code < 500 and code != 429:
                # return None (do not cache)
                logger.error(Fore.RED + f"Client error {code}: not retrying further.")
                return None
            # for 5xx or 429, do exponential backoff and retry
        except Exception as e:
            logger.exception(Fore.RED + f"Unexpected error during fetch: {e}")
            return None

        # backoff before next retry
        sleep_for = BACKOFF_FACTOR * (2 ** (attempt - 1)) + random.random()
        logger.info(Fore.CYAN + f"Retrying after {sleep_for:.1f}s...")
        time.sleep(sleep_for)

    logger.error(Fore.RED + f"Max retries reached for {url}")
    return None

# ---------------- PUBLIC get_json wrapper (with smart cache) ----------------
def get_json(path: str, params: Dict[str, Any], timeout: int = int(DEFAULT_TIMEOUT), use_cache: bool = True) -> Tuple[Optional[Any], str]:
    """
    Fetch JSON from endpoint path under BASE with params.
    Returns (data, url). Uses smart cache (only stores valid responses).
    """
    try:
        query = urlencode(params, doseq=True)
        url = f"{BASE.rstrip('/')}/{path.lstrip('/')}?{query}"
    except Exception as e:
        logger.error(Fore.RED + f"Failed to build URL: {e}")
        return None, ""

    # Load from cache when enabled
    if use_cache:
        cached = load_from_cache(url)
        if cached is not None:
            return cached, url

    # Fetch fresh
    data = _safe_get(url, timeout=timeout)

    # Save to cache only when data is valid and not an error
    if use_cache and data is not None:
        # avoid caching special error dicts
        if not (isinstance(data, dict) and (data == {} or "raw_text" in data and len(data.get("raw_text", "")) == 0)):
            save_to_cache(url, data)

    return data, url

# ---------------- BULK / CONCURRENT FETCH ----------------
def fetch_bulk(endpoints: List[str], common_params: Dict[str, Any], show_progress: bool = True, concurrent: bool = True) -> Dict[str, Any]:
    """
    endpoints: list of path strings (e.g., "vahandashboard/categoriesdonutchart")
    common_params: dict of params to be merged per request
    """
    results = {}
    iterator = tqdm(endpoints, desc="Fetching Bulk", disable=not show_progress, colour="green")
    def fetch_one(ep: str):
        try:
            data, url = get_json(ep, common_params)
            if data is not None:
                logger.info(Fore.GREEN + f"✅ Success: {ep}")
            else:
                logger.warning(Fore.YELLOW + f"⚠️ Failed/Empty: {ep}")
            return ep, data
        except Exception as e:
            logger.exception(Fore.RED + f"Bulk fetch error for {ep}: {e}")
            return ep, None

    if concurrent:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(endpoints)))) as ex:
            futures = {ex.submit(fetch_one, ep): ep for ep in endpoints}
            for fut in as_completed(futures):
                ep, data = fut.result()
                results[ep] = data
    else:
        for ep in iterator:
            ep, data = fetch_one(ep)
            results[ep] = data

    return results

# ---------------- YEAR RANGE FETCH ----------------
def fetch_range(path: str, year_start: int, year_end: int, **kwargs) -> List[Any]:
    """Fetch data year-by-year; return combined list/entries safely."""
    combined = []
    for y in range(year_start, year_end + 1):
        params = build_params(y, y, **{k: kwargs.get(k) for k in ['state_code','rto_code','vehicle_classes','vehicle_makers'] if k in kwargs} ) if 'build_params' in globals() else kwargs
        # If build_params isn't present in this module (it might be in your app), allow kwargs passthrough:
        # try to ensure 'fromYear' and 'toYear' are set
        if isinstance(params, dict) and 'fromYear' not in params:
            params['fromYear'] = y
            params['toYear'] = y
        data, _ = get_json(path, params)
        if isinstance(data, list):
            combined.extend(data)
        elif data:
            combined.append(data)
    return combined

# ---------------- SMALL UTILITIES ----------------
def preview_json(path: str, params: Dict[str, Any], n: int = 5) -> Any:
    data, _ = get_json(path, params)
    if isinstance(data, list):
        return data[:n]
    if isinstance(data, dict):
        # return first n keys
        return dict(list(data.items())[:n])
    return data

def is_data_available(data: Any) -> bool:
    if data is None:
        return False
    if isinstance(data, (list, dict)):
        return bool(data)
    return True

def print_json(data: Any) -> None:
    try:
        print(Fore.MAGENTA + json.dumps(data, indent=2, default=str))
    except Exception:
        print(data)

# ---------------- OPTIONAL: wrapper helpers for Streamlit UI ---------------
# These are convenience functions you can call from your streamlit_app.py
def clear_cache_and_reload():
    """Clear cache and return a message; useful to call from Streamlit button handler."""
    clear_cache()
    return "Cache cleared."

def fetch_with_retry(path: str, params: dict, timeout: int = int(DEFAULT_TIMEOUT), use_cache: bool = True):
    """Compatibility wrapper; returns same (data, url)"""
    return get_json(path, params, timeout=timeout, use_cache=use_cache)
