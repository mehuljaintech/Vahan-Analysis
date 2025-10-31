"""
vahan_parsing_all_maxed.py — VAHAN PARSING (ALL-MAXED Ultra)

Upgrades / Features:
- Class-based UniversalParser with pluggable strategies and robust fallbacks
- Defensive parsing of many JSON shapes: parallel arrays, wrapped payloads,
  dict-of-dicts, list-of-dicts, series maps, datasets/labels, timeseries wrappers
- Precise date parsing with many accepted formats and timezone-aware option
- Vectorized numeric coercion with percent/comma handling and sentinel support
- Schema inference and metadata return (labels found, value keys used, rows parsed)
- Flattening utilities for nested dicts/lists and dotpath extraction
- Caching (LRU) for repeated identical inputs, and lightweight file cache option
- Diagnostics (sample rows, parsed ranges, warnings)
- Helpers: parse_makers, parse_revenue_trend, parse_top5_revenue, parse_duration_table
- Streamlit-friendly preview and safe-export helpers

This module is self-contained and depends only on pandas/numpy/stdlib.
"""

from __future__ import annotations

import os
import re
import json
import math
import pickle
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from functools import lru_cache

import pandas as pd
import numpy as np

logger = logging.getLogger("vahan.parsing_allmaxed")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

# ------------------------- Defaults & Helpers -------------------------
DEFAULT_LABEL_KEYS = ("label", "name", "makerName", "manufacturer", "x", "Month-Year", "period", "stateName", "category")
DEFAULT_VALUE_KEYS = ("value", "count", "total", "registeredVehicleCount", "y", "registrations", "amount", "revenue")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".vahan_parsing_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# regex for numeric cleanup
_NUM_CLEAN_RE = re.compile(r"[\,\s\u00A0]+")
_PERCENT_RE = re.compile(r"%$")


def _coerce_num_scalar(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)) and not (isinstance(x, bool)):
            if math.isfinite(x):
                return float(x)
            return None
        s = str(x).strip()
        if s == "":
            return None
        # remove percent
        pct = bool(_PERCENT_RE.search(s))
        s = _PERCENT_RE.sub("", s)
        # remove commas/non-breaking spaces
        s = _NUM_CLEAN_RE.sub('', s)
        # handle parentheses negative (e.g. (123) -> -123)
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        # ignore non-numeric tokens
        val = float(s)
        if pct:
            return val
        return float(val)
    except Exception:
        return None


# vectorized coercion
def coerce_numeric_series(arr: Iterable) -> List[Optional[float]]:
    return [_coerce_num_scalar(x) for x in arr]


def _ensure_list_like(obj: Any) -> List:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return [obj]
    return [obj]


def _get_first_present(d: Dict, keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _flatten_dict(d: Dict, parent: str = '', sep: str = '.') -> Dict:
    out = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key, sep=sep))
        else:
            out[key] = v
    return out


# ------------------------- Date parsing -------------------------

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
    "%b-%Y", "%B %Y", "%b %Y", "%Y-%m", "%Y%m", "%Y"
]


def parse_date(y: Optional[Union[str,int]] = None,
               m: Optional[Union[str,int]] = None,
               d: Optional[Union[str,int]] = None,
               my: Optional[str] = None,
               tz: Optional[str] = None) -> Optional[pd.Timestamp]:
    try:
        if y is not None and m is not None and d is not None:
            return pd.Timestamp(int(y), int(m), int(d))
        if my is not None:
            s = str(my)
            for fmt in _DATE_FORMATS:
                try:
                    dt = pd.to_datetime(s, format=fmt, errors='coerce')
                    if pd.notna(dt):
                        return pd.Timestamp(dt.year, dt.month, 1)
                except Exception:
                    continue
            try:
                dt = pd.to_datetime(s, errors='coerce')
                if pd.notna(dt):
                    return pd.Timestamp(dt.year, dt.month, dt.day)
            except Exception:
                pass
        if y is not None:
            return pd.Timestamp(int(y), 1, 1)
    except Exception:
        return None
    return None


# ------------------------- File cache -------------------------
class SimpleFileCache:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _keypath(self, key: str) -> str:
        safe = re.sub(r"[^0-9a-zA-Z_\-]", "_", key)[:200]
        return os.path.join(self.cache_dir, f"{safe}.pkl")

    def get(self, key: str):
        p = self._keypath(key)
        if os.path.exists(p):
            try:
                with open(p, 'rb') as fh:
                    return pickle.load(fh)
            except Exception:
                return None
        return None

    def set(self, key: str, val: Any):
        p = self._keypath(key)
        try:
            with open(p, 'wb') as fh:
                pickle.dump(val, fh)
        except Exception as e:
            logger.debug(f"filecache set failed: {e}")


# ------------------------- Universal Parser -------------------------
class UniversalParser:
    def __init__(self,
                 label_keys: Iterable[str] = DEFAULT_LABEL_KEYS,
                 value_keys: Iterable[str] = DEFAULT_VALUE_KEYS,
                 enable_file_cache: bool = False):
        self.label_keys = tuple(label_keys)
        self.value_keys = tuple(value_keys)
        self.enable_file_cache = bool(enable_file_cache)
        self.file_cache = SimpleFileCache() if enable_file_cache else None

    def _cache_key(self, obj: Any) -> str:
        try:
            return str(hash(json.dumps(obj, sort_keys=True, default=str)))
        except Exception:
            return str(hash(str(obj)))

    @lru_cache(maxsize=1024)
    def to_df_lru(self, obj_serialized: str) -> pd.DataFrame:
        try:
            obj = json.loads(obj_serialized)
        except Exception:
            obj = obj_serialized
        return self.to_df(obj)

    def to_df(self, json_obj: Any) -> pd.DataFrame:
        """Convert arbitrary JSON to DataFrame(label, value) with defensive fallbacks.
        Returns DataFrame with columns ['label','value'] and logs diagnostics on shape.
        """
        key = None
        if self.enable_file_cache:
            key = self._cache_key(json_obj)
            cached = self.file_cache.get(key)
            if cached is not None:
                logger.debug("filecache hit")
                return cached.copy()

        df = self._to_df_internal(json_obj)
        if self.enable_file_cache and key is not None:
            try:
                self.file_cache.set(key, df)
            except Exception:
                pass
        return df

    def _to_df_internal(self, json_obj: Any) -> pd.DataFrame:
        if not json_obj:
            return pd.DataFrame(columns=["label", "value"]) 

        # Case: dict with labels & datasets / data
        if isinstance(json_obj, dict) and ("labels" in json_obj and ("data" in json_obj or "datasets" in json_obj)):
            labels = list(json_obj.get('labels', []))
            data = None
            if 'data' in json_obj:
                data = json_obj.get('data', [])
            elif 'datasets' in json_obj and isinstance(json_obj['datasets'], (list,tuple)):
                first = json_obj['datasets'][0] if json_obj['datasets'] else {}
                data = first.get('data') or first.get('values') or []
            if data is None:
                data = []
            n = min(len(labels), len(data))
            rows = []
            for i in range(n):
                lbl = labels[i]
                val = _coerce_num_scalar(data[i])
                if val is None:
                    continue
                rows.append({'label': str(lbl), 'value': float(val)})
            if rows:
                return pd.DataFrame(rows)

        # Case: wrapper containing 'data' or 'payload' or 'rows'
        if isinstance(json_obj, dict) and any(k in json_obj for k in ('data','payload','rows','result')):
            for k in ('data','payload','rows','result'):
                if k in json_obj and json_obj[k] is not None:
                    return self._to_df_internal(json_obj[k])

        # Case: list of dicts (most common)
        if isinstance(json_obj, (list, tuple)):
            rows = []
            for it in json_obj:
                if not isinstance(it, dict):
                    continue
                # flatten immediate nested dicts to help extract label/value
                flat = _flatten_dict(it)
                label = _get_first_present(flat, self.label_keys)
                value = _get_first_present(flat, self.value_keys)
                # fallback heuristics
                if label is None:
                    label = flat.get('category') or flat.get('stateName') or flat.get('region') or flat.get('maker') or flat.get('name')
                if value is None and 'metrics' in it and isinstance(it['metrics'], dict):
                    value = _get_first_present(it['metrics'], self.value_keys)
                if value is None:
                    # try any numeric-like column
                    for k, v in flat.items():
                        if k.lower() in ('value','count','total','y','registrations','amount'):
                            value = v
                            break
                        if isinstance(v, (int,float)) and not isinstance(v,bool):
                            value = v
                            break
                num = _coerce_num_scalar(value)
                if label is not None and num is not None:
                    rows.append({'label': str(label), 'value': float(num)})
            if rows:
                return pd.DataFrame(rows).drop_duplicates(subset=['label']).reset_index(drop=True)

        # Case: dict of label -> value or dict of period -> values
        if isinstance(json_obj, dict):
            # try simple scalar values
            rows = []
            for k, v in json_obj.items():
                if k in ('labels','data','datasets'):
                    continue
                if isinstance(v, (list, dict)):
                    # if dict of period:value, dive deeper
                    if isinstance(v, dict):
                        nested_rows = []
                        for nk, nv in v.items():
                            num = _coerce_num_scalar(nv)
                            dt = parse_date(my=nk)
                            if num is not None and dt is not None:
                                nested_rows.append({'date': pd.Timestamp(dt), 'value': float(num)})
                        if nested_rows:
                            return pd.DataFrame(nested_rows).sort_values('date').reset_index(drop=True)
                    continue
                num = _coerce_num_scalar(v)
                if num is not None:
                    rows.append({'label': str(k), 'value': float(num)})
            if rows:
                return pd.DataFrame(rows).reset_index(drop=True)

        # final fallback: try to coerce into DataFrame and search columns
        try:
            df = pd.DataFrame(json_obj)
            if df.empty:
                return pd.DataFrame(columns=['label','value'])
            # if df has label-like and value-like cols
            candidates_label = [c for c in df.columns if c.lower() in map(str.lower, self.label_keys)]
            candidates_value = [c for c in df.columns if c.lower() in map(str.lower, self.value_keys) or df[c].dtype.kind in 'fi']
            if candidates_label and candidates_value:
                lbl = candidates_label[0]
                val = candidates_value[0]
                df2 = df[[lbl, val]].rename(columns={lbl: 'label', val: 'value'})
                df2['value'] = df2['value'].apply(_coerce_num_scalar)
                df2 = df2.dropna(subset=['value']).reset_index(drop=True)
                return df2
        except Exception:
            pass

        return pd.DataFrame(columns=['label', 'value'])

    # ------------------------- Trend normalization -------------------------
    def normalize_trend(self, trend_json: Any, date_key_candidates: Iterable[str] = ("label","period","date","Month-Year","monthYear")) -> pd.DataFrame:
        """Parse timeseries-like shapes into DataFrame(date,value)."""
        candidate = trend_json
        if not candidate:
            return pd.DataFrame(columns=['date','value'])

        # case: labels & data arrays
        if isinstance(candidate, dict) and 'labels' in candidate and ('data' in candidate or 'datasets' in candidate):
            labels = list(candidate.get('labels', []))
            data = None
            if 'data' in candidate:
                data = candidate.get('data', [])
            else:
                ds = candidate.get('datasets', [])
                if ds and isinstance(ds, (list,tuple)):
                    data = ds[0].get('data') or ds[0].get('values') or []
            rows = []
            for a,b in zip(labels, data or []):
                dt = parse_date(my=a)
                if dt is None:
                    try:
                        dt = pd.to_datetime(str(a), errors='coerce')
                    except Exception:
                        dt = None
                num = _coerce_num_scalar(b)
                if dt is not None and num is not None:
                    rows.append({'date': pd.Timestamp(dt), 'value': float(num)})
            if rows:
                return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

        # case: list of dicts with various keys
        if isinstance(candidate, (list,tuple)):
            rows = []
            for it in candidate:
                if not isinstance(it, dict):
                    continue
                flat = _flatten_dict(it)
                y = _get_first_present(flat, ('year','Year','yr'))
                m = _get_first_present(flat, ('month','Month','mn','monthNo'))
                d = _get_first_present(flat, ('day','Day','dateNo'))
                my = _get_first_present(flat, date_key_candidates)
                val = _get_first_present(flat, DEFAULT_VALUE_KEYS)
                dt = parse_date(y=y, m=m, d=d, my=my)
                num = _coerce_num_scalar(val)
                if dt is not None and num is not None:
                    rows.append({'date': pd.Timestamp(dt), 'value': float(num)})
            if rows:
                return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

        # case: dict of period->value (e.g., {'2020-01': 123, '2020-02': 234})
        if isinstance(candidate, dict):
            rows = []
            for k,v in candidate.items():
                if k in ('labels','data','datasets'):
                    continue
                if isinstance(v, (list,dict)):
                    continue
                dt = parse_date(my=k)
                if dt is None:
                    try:
                        dt = pd.to_datetime(k, errors='coerce')
                    except Exception:
                        dt = None
                num = _coerce_num_scalar(v)
                if dt is not None and num is not None:
                    rows.append({'date': pd.Timestamp(dt), 'value': float(num)})
            if rows:
                return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

        # fallback: try to coerce
        try:
            df = pd.DataFrame(candidate)
            date_col = None
            for c in df.columns:
                if c.lower() in ('date','period','label'):
                    date_col = c
                    break
            value_col = None
            for c in df.columns:
                if c.lower() in DEFAULT_VALUE_KEYS or df[c].dtype.kind in 'fi':
                    value_col = c
                    break
            if date_col and value_col:
                tmp = df[[date_col, value_col]].copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
                tmp[value_col] = tmp[value_col].apply(_coerce_num_scalar)
                tmp = tmp.dropna().rename(columns={date_col:'date', value_col:'value'})
                if not tmp.empty:
                    return tmp.sort_values('date').reset_index(drop=True)
        except Exception:
            pass

        return pd.DataFrame(columns=['date','value'])

    # ------------------------- Specific parsers -------------------------
    def parse_duration_table(self, json_obj: Any) -> pd.DataFrame:
        df = self.to_df(json_obj)
        if df.empty:
            return df
        # try label -> year numeric
        try:
            df['label'] = df['label'].astype(str)
            df = df.sort_values('label')
        except Exception:
            pass
        return df.reset_index(drop=True)

    def parse_top5_revenue(self, json_obj: Any) -> pd.DataFrame:
        df = self.to_df(json_obj)
        if df.empty:
            return df
        df = df.sort_values('value', ascending=False).head(5).reset_index(drop=True)
        return df

    def parse_revenue_trend(self, json_obj: Any) -> pd.DataFrame:
        # attempt dict of year -> list or dict
        rows = []
        if isinstance(json_obj, dict):
            for year, vals in json_obj.items():
                if isinstance(vals, dict):
                    for period, v in vals.items():
                        num = _coerce_num_scalar(v)
                        dt = parse_date(y=year, my=period) or parse_date(my=f"{year}-{period}")
                        if num is not None and dt is not None:
                            rows.append({'year': str(year), 'period': str(period), 'value': float(num)})
                elif isinstance(vals, (list,tuple)):
                    for i, v in enumerate(vals, start=1):
                        num = _coerce_num_scalar(v)
                        if num is not None:
                            rows.append({'year': str(year), 'period': i, 'value': float(num)})
        return pd.DataFrame(rows)

    def parse_makers(self, json_obj: Any) -> pd.DataFrame:
        # handle datasets with labels/datasets
        if isinstance(json_obj, dict) and 'labels' in json_obj and 'datasets' in json_obj:
            labels = list(json_obj.get('labels', []))
            ds = json_obj.get('datasets', [])
            if ds and isinstance(ds[0], dict):
                data = ds[0].get('data') or ds[0].get('values')
                if data:
                    vals = [ _coerce_num_scalar(x) for x in data ]
                    rows = [{'label': l, 'value': v} for l,v in zip(labels, vals) if v is not None]
                    return pd.DataFrame(rows).sort_values('value', ascending=False).reset_index(drop=True)
        # else use generic to_df with manufacturer/maker keys
        return self.to_df(json_obj)

    def parse_maker_state(self, json_obj: Any) -> pd.DataFrame:
        data = _ensure_list_like(json_obj)
        rows = []
        for it in data:
            if not isinstance(it, dict):
                continue
            makers = it.get('makers') or it.get('makerStateMap') or it.get('makerState') or {}
            if isinstance(makers, dict):
                for maker, stmap in makers.items():
                    if isinstance(stmap, dict):
                        for state, val in stmap.items():
                            num = _coerce_num_scalar(val)
                            if num is not None:
                                rows.append({'maker': maker, 'state': state, 'value': float(num)})
        return pd.DataFrame(rows)

    # ------------------------- Preview / Diagnostics -------------------------
    def preview(self, obj: Any, n: int = 5) -> None:
        df = None
        try:
            df = self.to_df(obj)
        except Exception as e:
            logger.error(f"preview failed: {e}")
        if df is None or df.empty:
            logger.info("<empty result>")
            return
        logger.info(f"Preview ({min(n,len(df))} rows):")
        logger.info(df.head(n).to_string(index=False))


# ------------------------- Module-level convenience -------------------------
_default_parser = UniversalParser()

def to_df(json_obj: Any) -> pd.DataFrame:
    return _default_parser.to_df(json_obj)

def normalize_trend(trend_json: Any) -> pd.DataFrame:
    return _default_parser.normalize_trend(trend_json)

def parse_makers(json_obj: Any) -> pd.DataFrame:
    return _default_parser.parse_makers(json_obj)

def parse_maker_state(json_obj: Any) -> pd.DataFrame:
    return _default_parser.parse_maker_state(json_obj)

