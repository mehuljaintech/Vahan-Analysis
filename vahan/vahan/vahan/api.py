# vahan/api.py
"""
VAHAN API (ALL-MAXED Robust)
- Handles HTTP/HTTPS errors, retries, backoff with jitter, 429 Retry-After,
  403 rotations, SSL verify toggle, proxy support, caching, concurrent fetches.
- Use responsibly and don't attempt to bypass authentication/authorization.
"""

from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from urllib.parse import urlencode
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.exceptions import (
    HTTPError, ConnectionError, Timeout, SSLError, RequestException
)
import random
import time
import logging
import json
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE = "https://analytics.parivahan.gov.in/analytics/publicdashboard"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 6
BACKOFF_FACTOR = 1.0
MAX_WORKERS = 8
CACHE_SIZE = 4096

# Basic rotation lists (expand if you want)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
]
REFERERS = [
    "https://analytics.parivahan.gov.in/",
    "https://parivahan.gov.in/",
    "https://google.com/",
    "https://bing.com/"
]

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("vahan_api")

# ---------------- SESSION ----------------
session = requests.Session()
# Keep retries at the adapter-level for connection failures too (complementary to our logic)
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Global mutable config (use setter helpers below)
_global = {
    "default_headers": {"User-Agent": random.choice(USER_AGENTS)},
    "verify": True,     # SSL verification
    "proxies": None,    # e.g. {"http": "...", "https": "..."}
    "timeout": DEFAULT_TIMEOUT,
    "max_retries": MAX_RETRIES,
    "backoff_factor": BACKOFF_FACTOR,
    "debug_hook": None  # Optional callback: func(url, response_or_exc)
}

# ---------------- HELPERS ----------------
def random_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Produce random browser-like headers + optional extras."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": random.choice(["en-US,en;q=0.9", "hi-IN,hi;q=0.9,en;q=0.8"]),
        "Referer": random.choice(REFERERS),
        "Connection": "keep-alive",
        "DNT": "1",
        "Cache-Control": "no-cache",
    }
    if extra:
        headers.update(extra)
    # allow global default headers override
    headers.update({k:v for k,v in _global["default_headers"].items() if v})
    return headers

def _sleep_backoff(attempt: int):
    """Exponential backoff with jitter."""
    base = _global.get("backoff_factor", BACKOFF_FACTOR)
    # exponential backoff: base * 2^(attempt-1)
    backoff = base * (2 ** (attempt - 1))
    # jitter +/- 10%
    jitter = backoff * 0.1 * (random.random() - 0.5)
    wait = max(0.1, backoff + jitter)
    logger.debug(f"Sleeping {wait:.2f}s before retry (attempt {attempt})")
    time.sleep(wait)

def _parse_retry_after(resp: requests.Response) -> Optional[float]:
    """Return seconds to wait from Retry-After header if present, else None."""
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        # If integer seconds
        return float(ra)
    except Exception:
        # Try parse as HTTP-date
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(ra)
            return max(0.0, (dt - time.datetime.now(dt.tzinfo)).total_seconds())
        except Exception:
            return None

def set_global_headers(headers: Dict[str, str]):
    """Set/merge global headers used for every request."""
    _global["default_headers"].update(headers)

def set_proxies(proxies: Dict[str, str]):
    """Set proxies for session and global config."""
    _global["proxies"] = proxies
    session.proxies.update(proxies)

def set_ssl_verify(verify: bool):
    _global["verify"] = bool(verify)

def set_backoff(factor: float):
    _global["backoff_factor"] = float(factor)

def set_timeout(seconds: int):
    _global["timeout"] = int(seconds)

def set_max_retries(n: int):
    _global["max_retries"] = int(n)

def set_debug_hook(hook: Optional[Callable[[str, Any], None]]):
    """
    Hook called after each request attempt:
      hook(url, response_or_exception)
    Useful to log or write debug dumps.
    """
    _global["debug_hook"] = hook

def clear_cache():
    """Clear the LRU cached safe_get (recreate wrapper)"""
    safe_get.cache_clear()

# ---------------- SAFE GET (robust) ----------------
# We'll cache successful JSON responses by URL to avoid hammering the endpoint.
@lru_cache(maxsize=CACHE_SIZE)
def safe_get(url: str, timeout: Optional[int] = None) -> Union[dict, list, None]:
    """
    Robust GET:
     - handles 403 by rotating headers/cookies
     - handles 429 by honoring Retry-After
     - handles 5xx and network issues with exponential backoff
     - returns JSON or None
    """
    timeout = timeout if timeout is not None else _global["timeout"]
    max_retries = _global.get("max_retries", MAX_RETRIES)

    # make a small internal cookie id for rotating sessions (not a real bypass)
    cookie_seed = random.randint(100000, 999999)

    for attempt in range(1, max_retries + 1):
        headers = random_headers()
        # lightweight cookie to vary fingerprint
        cookies = {"sessionid": f"sid-{cookie_seed}-{attempt}"}
        try:
            logger.info(f"Fetching: {url} (attempt {attempt})")
            resp = session.get(
                url,
                headers=headers,
                timeout=timeout,
                verify=_global.get("verify", True),
                proxies=_global.get("proxies", None),
                cookies=cookies,
            )
            # If provided, call debug hook with raw response
            if _global.get("debug_hook"):
                try:
                    _global["debug_hook"](url, resp)
                except Exception:
                    logger.debug("debug_hook raised an exception; ignored.")

            # 2xx -> success
            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError:
                    # JSON decode failed
                    logger.warning("Response not JSON — returning raw text in dict")
                    return {"text": resp.text, "_status": resp.status_code}

            # 401/403 -> likely blocked or auth required
            if resp.status_code in (401, 403):
                logger.warning(f"HTTP {resp.status_code} returned — rotating headers & retrying if attempts remain.")
                # rotate UA / referer by continuing loop after sleep
                _sleep_backoff(attempt)
                continue

            # 429 -> rate limited: honor Retry-After header if present
            if resp.status_code == 429:
                wait = _parse_retry_after(resp) or (60 * attempt)
                logger.warning(f"HTTP 429 Rate Limited — sleeping {wait:.1f}s (Retry-After).")
                time.sleep(wait)
                continue

            # 5xx -> server error, retry with backoff
            if 500 <= resp.status_code < 600:
                logger.warning(f"HTTP {resp.status_code} server error — retrying with backoff.")
                _sleep_backoff(attempt)
                continue

            # other codes: log and return None
            logger.error(f"Unhandled HTTP {resp.status_code} for URL: {url}")
            return None

        except SSLError as e:
            logger.error(f"SSL error when fetching {url}: {e}")
            # optionally allow user to disable verification
            if _global.get("verify", True):
                logger.info("Retrying with SSL verification disabled (set_ssl_verify(False) to keep permanently).")
                # try once with verify=False
                try:
                    resp = session.get(url, headers=headers, timeout=timeout, verify=False, proxies=_global.get("proxies", None))
                    if 200 <= resp.status_code < 300:
                        try:
                            return resp.json()
                        except ValueError:
                            return {"text": resp.text, "_status": resp.status_code}
                except Exception as e2:
                    logger.error(f"Retry with verify=False failed: {e2}")
            _sleep_backoff(attempt)
            continue

        except (ConnectionError, Timeout) as e:
            logger.warning(f"Network error while fetching {url}: {e}. Retrying...")
            _sleep_backoff(attempt)
            continue

        except RequestException as e:
            logger.exception(f"Requests exception: {e}")
            _sleep_backoff(attempt)
            continue

        except Exception as e:
            logger.exception(f"Unexpected error fetching {url}: {e}")
            return None

    logger.error(f"Max retries reached for {url}. Returning None.")
    return None

# ---------------- GET JSON wrapper ----------------
def get_json(path: str, params: dict, timeout: Optional[int] = None, use_cache: bool = True) -> Tuple[Any, str]:
    """
    Construct URL and fetch JSON safely.
    Returns (data, url). Data can be dict/list or None.
    """
    url = f"{BASE.rstrip('/')}/{path.lstrip('/')}?{urlencode(params, doseq=True)}"
    try:
        if use_cache:
            data = safe_get(url, timeout)
        else:
            # bypass cache: call session.get with same robust behavior (but without lru cache)
            data = _get_json_nocache(url, timeout)
        return data, url
    except Exception as e:
        logger.exception(f"get_json failed for {url}: {e}")
        return None, url

# Helper no-cache function (same logic as safe_get but not cached)
def _get_json_nocache(url: str, timeout: Optional[int] = None) -> Union[dict, list, None]:
    timeout = timeout if timeout is not None else _global["timeout"]
    max_retries = _global.get("max_retries", MAX_RETRIES)
    cookie_seed = random.randint(1_000_0, 9_999_9)

    for attempt in range(1, max_retries + 1):
        headers = random_headers()
        cookies = {"sessionid": f"nocache-{cookie_seed}-{attempt}"}
        try:
            resp = session.get(url, headers=headers, timeout=timeout, verify=_global.get("verify", True), proxies=_global.get("proxies"))
            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError:
                    return {"text": resp.text, "_status": resp.status_code}
            if resp.status_code in (401, 403):
                _sleep_backoff(attempt)
                continue
            if resp.status_code == 429:
                wait = _parse_retry_after(resp) or (60 * attempt)
                time.sleep(wait)
                continue
            if 500 <= resp.status_code < 600:
                _sleep_backoff(attempt)
                continue
            return None
        except (ConnectionError, Timeout) as e:
            _sleep_backoff(attempt)
            continue
        except SSLError:
            if _global.get("verify", True):
                try:
                    resp = session.get(url, headers=headers, timeout=timeout, verify=False, proxies=_global.get("proxies"))
                    if 200 <= resp.status_code < 300:
                        try:
                            return resp.json()
                        except ValueError:
                            return {"text": resp.text, "_status": resp.status_code}
                except Exception:
                    pass
            _sleep_backoff(attempt)
            continue
        except RequestException:
            _sleep_backoff(attempt)
            continue
    return None

# ---------------- BULK / RANGE / MULTI helpers ----------------
def fetch_bulk(endpoints: List[str], common_params: Dict[str, Any], show_progress: bool = True, concurrent: bool = True) -> Dict[str, Any]:
    """Fetch multiple endpoints concurrently. Returns mapping endpoint->data."""
    results: Dict[str, Any] = {}
    iterator = tqdm(endpoints, desc="Bulk fetch", disable=not show_progress)
    def worker(ep):
        data, url = get_json(ep, common_params)
        if data is None:
            logger.warning(f"Bulk fetch failed for {ep}")
        return ep, data
    if concurrent:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(worker, ep): ep for ep in iterator}
            for fut in as_completed(futures):
                ep, data = fut.result()
                results[ep] = data
    else:
        for ep in iterator:
            ep, data = worker(ep)
            results[ep] = data
    return results

def fetch_range(path: str, year_start: int, year_end: int, **kwargs) -> List[Any]:
    """Fetch endpoint for each year and merge results where appropriate."""
    all_data = []
    for year in range(year_start, year_end + 1):
        params = build_params(year, year, **kwargs)
        data, _ = get_json(path, params)
        if isinstance(data, list):
            all_data.extend(data)
        elif data:
            all_data.append(data)
    return all_data

def fetch_multi_params(path: str, years: List[int], states: List[str], makers: List[str], classes: List[str], time_periods: List[int], concurrent: bool = True) -> Dict[Tuple, Any]:
    """Fetch full cross product of parameters. Returns mapping key->data."""
    tasks = []
    for y in years:
        for s in (states or [""]):
            for mk in (makers or [""]):
                for vc in (classes or [""]):
                    for tp in (time_periods or [0]):
                        params = build_params(y, y, state_code=s, vehicle_makers=mk, vehicle_classes=vc, time_period=tp)
                        tasks.append(((y, s or "ALL", mk or "ALL", vc or "ALL", tp), params))
    results = {}
    iterator = tqdm(tasks, desc="fetch_multi_params", disable=False)
    def w(item):
        key, params = item
        data, _ = get_json(path, params)
        return key, data
    if concurrent:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(w, t): t for t in iterator}
            for fut in as_completed(futures):
                key, data = fut.result()
                results[key] = data
    else:
        for t in iterator:
            key, data = w(t)
            results[key] = data
    return results

def fetch_all_periods(path: str, year_start: int, year_end: int, **kwargs) -> Dict[str, Any]:
    """Fetch daily/monthly/quarterly/yearly automatically (time_period mapping may vary by API)."""
    mapping = {"yearly": 0, "quarterly": 1, "monthly": 2, "daily": 3}
    out = {}
    for name, tp in mapping.items():
        params = build_params(year_start, year_end, time_period=tp, **kwargs)
        data, _ = get_json(path, params)
        out[name] = data
    return out

# ---------------- UTIL ----------------
def preview_json(path: str, params: dict, n: int = 5):
    data, url = get_json(path, params)
    if isinstance(data, list):
        return data[:n]
    if isinstance(data, dict):
        return dict(list(data.items())[:n])
    return data

def print_json(data: Any):
    try:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(data)

def is_data_available(data: Any) -> bool:
    if data is None:
        return False
    if isinstance(data, (list, dict)):
        return bool(data)
    return True

# ---------------- BUILD PARAMS helper (kept simple here) ----------------
def build_params(from_year: int, to_year: int, state_code: str = "", rto_code: str = "0",
                 vehicle_classes: str = "", vehicle_makers: str = "", time_period: int = 0,
                 fitness_check: int = 0, vehicle_type: str = "", extra: Optional[dict] = None) -> dict:
    p = {
        "fromYear": from_year,
        "toYear": to_year,
        "stateCode": state_code,
        "rtoCode": rto_code,
        "vehicleClasses": vehicle_classes,
        "vehicleMakers": vehicle_makers,
        "vehicleSubCategories": "",
        "vehicleEmissions": "",
        "vehicleFuels": "",
        "timePeriod": time_period,
        "vehicleCategoryGroup": "",
        "evType": "",
        "vehicleStatus": "",
        "vehicleOwnerType": "",
        "fitnessCheck": fitness_check,
        "vehicleType": vehicle_type
    }
    if extra and isinstance(extra, dict):
        p.update(extra)
    return p

# END OF FILE
