# vahan/api.py
import requests
from urllib.parse import urlencode
from functools import lru_cache
from tqdm import tqdm
from colorama import Fore, Style, init
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- CONFIG ----------------
BASE = "https://analytics.parivahan.gov.in/analytics/publicdashboard"
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.0
MAX_WORKERS = 10  # concurrency for bulk fetch

# ---------------- LOGGING ----------------
init(autoreset=True)
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} | {Fore.YELLOW}%(levelname)s{Style.RESET_ALL} | %(message)s"
)
logger = logging.getLogger("vahan_api")

# ---------------- SESSION ----------------
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ---------------- HELPERS ----------------
def build_params(
    from_year, to_year, state_code="", rto_code="0",
    vehicle_classes="", vehicle_makers="", time_period=0,
    fitness_check=0, vehicle_type="", extra=None
):
    """Build full query parameters for Vahan API (supports extra dynamic fields)"""
    params = {
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
        params.update(extra)
    elif extra:
        logger.warning(Fore.RED + "Extra parameters should be a dict. Ignored.")
    return params

@lru_cache(maxsize=1024)
def safe_get(url: str, timeout=DEFAULT_TIMEOUT):
    """Maxed GET request with retries, backoff, and full error handling"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(Fore.GREEN + f"Fetching URL: {url} (Attempt {attempt})")
            r = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response else "Unknown"
            logger.warning(Fore.YELLOW + f"HTTP {code}: {e}")
            if code in [401, 403]:
                logger.error(Fore.RED + "Unauthorized or forbidden. Returning None.")
                return None
            if code in [429, 500, 502, 503, 504]:
                wait = BACKOFF_FACTOR * attempt
                logger.info(Fore.CYAN + f"Retrying after {wait}s...")
                time.sleep(wait)
            else:
                return None
        except requests.RequestException as e:
            wait = BACKOFF_FACTOR * attempt
            logger.warning(Fore.YELLOW + f"Request exception: {e}. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            logger.exception(Fore.RED + f"Unexpected error: {e}")
            return None
    logger.error(Fore.RED + f"Max retries reached for URL: {url}")
    return None

def get_json(path: str, params: dict, timeout=DEFAULT_TIMEOUT, use_cache=True):
    """Fetch JSON from Vahan API safely"""
    url = f"{BASE}/{path}?{urlencode(params, doseq=True)}"
    if use_cache:
        return safe_get(url, timeout), url
    try:
        r = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json(), url
    except Exception as e:
        logger.exception(Fore.RED + f"Failed to fetch: {e}")
        return None, url

def fetch_bulk(endpoints: list, common_params: dict, show_progress=True, concurrent=True):
    """Fetch multiple endpoints at once, optionally concurrently (MAXED)"""
    results = {}
    iterator = tqdm(endpoints, desc="Fetching Bulk API", colour="green") if show_progress else endpoints

    def fetch(ep):
        data, _ = get_json(ep, common_params)
        if data:
            logger.info(Fore.BLUE + f"Success: {ep}")
        else:
            logger.warning(Fore.YELLOW + f"Failed: {ep}")
        return ep, data

    if concurrent:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_ep = {executor.submit(fetch, ep): ep for ep in iterator}
            for future in as_completed(future_to_ep):
                ep, data = future.result()
                results[ep] = data
    else:
        for ep in iterator:
            ep, data = fetch(ep)
            results[ep] = data
    return results

def fetch_range(path, year_start, year_end, state_code="", rto_code="0", vehicle_classes="", vehicle_makers=""):
    """Fetch data year by year and combine results safely"""
    all_data = []
    for year in range(year_start, year_end + 1):
        params = build_params(year, year, state_code, rto_code, vehicle_classes, vehicle_makers)
        data, _ = get_json(path, params)
        if isinstance(data, list):
            all_data.extend(data)
        elif data:
            all_data.append(data)
    return all_data

def preview_json(path: str, params: dict, n=5):
    """Preview first n items safely"""
    data, _ = get_json(path, params)
    if isinstance(data, list):
        return data[:n]
    if isinstance(data, dict):
        return dict(list(data.items())[:n])
    return data

def print_json(data):
    """Pretty-print JSON in colored format"""
    import json
    print(Fore.MAGENTA + json.dumps(data, indent=2))

def is_data_available(data):
    """Check if data exists and is non-empty"""
    if data is None:
        return False
    if isinstance(data, (list, dict)):
        return bool(data)
    return True
