# vahan/parsing.py
"""
Robust parser utilities for Vahan analytics JSON -> pandas.DataFrame

Features:
- Flexible to_df() with customizable label/value key lists
- normalize_trend() handles many shapes: parallel arrays, list-of-dicts,
  dict-of-year:value, Month-Year labels, nested 'data' fields
- parse_duration_table(), parse_top5_revenue(), parse_revenue_trend()
- parse_makers() and parse_maker_state() that extract maker/manufacturer info
- Defensive numeric coercion and consistent output column names
"""

from typing import Iterable, Tuple, Any, Dict
import pandas as pd
import numpy as np
import math
import logging

logger = logging.getLogger("vahan.parsing")

# ---------------- Helpers ----------------
def _coerce_num(x):
    if x is None:
        return None
    try:
        # remove commas, percent signs, and coerce
        s = str(x).replace(",", "").replace("%", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def _ensure_list_like(obj):
    if isinstance(obj, dict):
        # special-case: dict that is actually mapping of many items (not a single record)
        # we don't convert here; caller will handle
        return obj
    if obj is None:
        return []
    return obj if isinstance(obj, (list, tuple)) else [obj]

def _get_first_present(d: dict, keys: Iterable[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

# ---------------- to_df ----------------
def to_df(json_obj: Any,
          label_keys: Tuple[str, ...] = ("label", "name", "makerName", "manufacturer", "x"),
          value_keys: Tuple[str, ...] = ("value", "count", "total", "registeredVehicleCount", "y")) -> pd.DataFrame:
    """
    Universal JSON -> DataFrame with columns ['label','value'].

    label_keys: keys to try for label extraction (in order)
    value_keys: keys to try for numeric value extraction (in order)

    Handles:
      - {"labels": [...], "data": [...]} parallel arrays
      - {"data": [...] } wrapper
      - list[dict] with many possible key names
      - dict of label:value pairs (e.g., {"2020": 100, "2021":150})
    """
    if not json_obj:
        return pd.DataFrame(columns=["label", "value"])

    # 1) Parallel arrays common pattern
    if isinstance(json_obj, dict) and "labels" in json_obj and "data" in json_obj:
        labels = json_obj.get("labels") or []
        values = json_obj.get("data") or []
        n = min(len(labels), len(values))
        out = pd.DataFrame({"label": labels[:n], "value": [ _coerce_num(v) for v in values[:n] ]})
        out = out.dropna(subset=["value"]).reset_index(drop=True)
        return out

    # 2) Wrapped in {"data": ...}
    if isinstance(json_obj, dict) and "data" in json_obj:
        return to_df(json_obj["data"], label_keys=label_keys, value_keys=value_keys)

    # 3) List of dicts
    if isinstance(json_obj, (list, tuple)):
        rows = []
        for item in json_obj:
            if not isinstance(item, dict):
                continue
            label = _get_first_present(item, label_keys)
            value = _get_first_present(item, value_keys)
            # some APIs put label under nested structures (e.g., item['maker']['name'])
            if label is None:
                # try some common nested heuristics
                if "maker" in item and isinstance(item["maker"], dict):
                    label = _get_first_present(item["maker"], label_keys)
                if label is None and "manufacturer" in item and isinstance(item["manufacturer"], dict):
                    label = _get_first_present(item["manufacturer"], label_keys)
            # fallback to 'key' or 'label' if present anywhere as string
            if label is None and "label" in item:
                label = item.get("label")
            if value is None:
                # sometimes value is buried in nested object: e.g., item['metrics']['value']
                if "metrics" in item and isinstance(item["metrics"], dict):
                    value = _get_first_present(item["metrics"], value_keys)
            if label is None and value is not None:
                # last resort: use index-ish label
                label = item.get("category") or item.get("stateName") or item.get("state") or item.get("region")
            if label is not None and value is not None:
                num = _coerce_num(value)
                if num is not None and (not (isinstance(num, float) and math.isnan(num))):
                    rows.append({"label": label, "value": num})
        if rows:
            return pd.DataFrame(rows).drop_duplicates(subset=["label"]).reset_index(drop=True)
        # if rows empty, fall through to other formats

    # 4) dict of label:value pairs -> convert to rows
    if isinstance(json_obj, dict):
        # exclude keys that are metadata or 'data' handled earlier
        candidate_rows = []
        for k, v in json_obj.items():
            if k in ("labels", "data"):
                continue
            # if value is list/dict, skip (handled elsewhere)
            if isinstance(v, (list, dict)):
                continue
            num = _coerce_num(v)
            if num is not None:
                candidate_rows.append({"label": k, "value": num})
        if candidate_rows:
            return pd.DataFrame(candidate_rows).sort_values("label").reset_index(drop=True)

    # nothing matched -> return empty DF
    return pd.DataFrame(columns=["label", "value"])


# ---------------- Trend normalization ----------------
def parse_date(y=None, m=None, my=None):
    """
    Convert components into pd.Timestamp at period start.
    Accepts:
      y -> 2020 or '2020'
      m -> month string or number
      my -> 'Jan-2020', '2020-01', 'Jan 2020', '202001' etc.
    """
    try:
        if y and m:
            # try month_to_int
            mm = None
            try:
                mm = int(m)
            except Exception:
                try:
                    mm = pd.to_datetime(str(m), format="%b").month
                except Exception:
                    try:
                        mm = pd.to_datetime(str(m), format="%B").month
                    except Exception:
                        mm = None
            if mm:
                return pd.Timestamp(int(y), int(mm), 1)
        if my:
            # try many formats
            for fmt in ("%b-%Y", "%B-%Y", "%Y-%m", "%Y/%m", "%b %Y", "%B %Y", "%Y%m", "%Y"):
                try:
                    dt = pd.to_datetime(str(my), format=fmt, errors="coerce")
                    if pd.notna(dt):
                        return pd.Timestamp(dt.year, dt.month, 1)
                except Exception:
                    continue
            # last resort: generic parse
            dt = pd.to_datetime(str(my), errors="coerce")
            if pd.notna(dt):
                return pd.Timestamp(dt.year, dt.month, 1)
        if y:
            return pd.Timestamp(int(y), 1, 1)
    except Exception:
        return None
    return None


def normalize_trend(trend_json: Any) -> pd.DataFrame:
    """
    Normalize trend JSON into DataFrame with columns ['date','value'].
    Accepts:
      - {'labels': [...], 'data': [...]} parallel arrays (labels can be Month-Year)
      - list of dicts with keys like {year, month, Month-Year, count, value}
      - dict mapping period->value e.g., {"2020-01": 100, "2020-02": 120}
    """
    if not trend_json:
        return pd.DataFrame(columns=["date", "value"])

    # Parallel arrays
    if isinstance(trend_json, dict) and "labels" in trend_json and "data" in trend_json:
        labels = trend_json.get("labels") or []
        values = trend_json.get("data") or []
        n = min(len(labels), len(values))
        rows = []
        for i in range(n):
            lbl = labels[i]
            val = _coerce_num(values[i])
            dt = parse_date(my=lbl) or pd.to_datetime(str(lbl), errors="coerce")
            if val is not None and (not (isinstance(val, float) and math.isnan(val))):
                if pd.isna(dt):
                    # attempt parse with year-only
                    try:
                        dt = pd.Timestamp(int(lbl), 1, 1)
                    except Exception:
                        dt = pd.NaT
                rows.append({"date": pd.to_datetime(dt), "value": val})
        if rows:
            df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            return df

    # If wrapped
    if isinstance(trend_json, dict) and "data" in trend_json:
        return normalize_trend(trend_json["data"])

    # List of dicts
    data = trend_json if isinstance(trend_json, (list, tuple)) else ([trend_json] if isinstance(trend_json, dict) else None)
    rows = []
    if data:
        for it in data:
            if not isinstance(it, dict):
                continue
            # many possible key names
            y = _get_first_present(it, ("year", "Year", "yr"))
            m = _get_first_present(it, ("month", "Month", "mn", "monthNo"))
            my = _get_first_present(it, ("Month-Year", "monthYear", "label", "period", "monthYearLabel", "x"))
            cnt = _get_first_present(it, ("count", "value", "registrations", "total", "y"))
            dt = parse_date(y=y, m=m, my=my)
            val = _coerce_num(cnt)
            if dt is not None and val is not None:
                rows.append({"date": pd.to_datetime(dt), "value": val})
        if rows:
            df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            return df

    # Dict of period->value (fallback)
    if isinstance(trend_json, dict):
        rows2 = []
        for k, v in trend_json.items():
            # skip metadata keys
            if k in ("labels", "data"):
                continue
            # skip complex child objects
            if isinstance(v, (list, dict)):
                continue
            val = _coerce_num(v)
            if val is None:
                continue
            dt = parse_date(my=k) or pd.to_datetime(str(k), errors="coerce")
            if pd.isna(dt):
                # sometimes key is numeric year '2020'
                try:
                    dt = pd.Timestamp(int(k), 1, 1)
                except Exception:
                    dt = pd.NaT
            if not pd.isna(dt):
                rows2.append({"date": pd.to_datetime(dt), "value": val})
        if rows2:
            df = pd.DataFrame(rows2).sort_values("date").reset_index(drop=True)
            return df

    # nothing parseable
    return pd.DataFrame(columns=["date", "value"])


# ---------------- Duration table parser ----------------
def parse_duration_table(json_obj: Any) -> pd.DataFrame:
    """
    Parses registration duration table responses.
    Expected fields: yearAsString, year, registeredVehicleCount, value
    """
    if not json_obj:
        return pd.DataFrame(columns=["label", "value"])
    data = json_obj.get("data", json_obj) if isinstance(json_obj, dict) else json_obj
    data = data if isinstance(data, (list, tuple)) else [data]
    rows = []
    for it in data:
        if not isinstance(it, dict):
            continue
        label = it.get("yearAsString") or it.get("year") or it.get("label") or it.get("period")
        val = _get_first_present(it, ("registeredVehicleCount", "count", "value", "total"))
        num = _coerce_num(val)
        if label is None:
            continue
        if num is not None:
            rows.append({"label": str(label), "value": num})
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------- Top-5 revenue parser ----------------
def parse_top5_revenue(json_obj: Any) -> pd.DataFrame:
    """
    Many endpoints return {labels: [...], data: [...]} for top-5 charts.
    This will pull top entries and ensure numeric 'value'.
    """
    if not json_obj:
        return pd.DataFrame(columns=["label", "value"])
    # parallel arrays
    if isinstance(json_obj, dict) and "labels" in json_obj and "data" in json_obj:
        labels = json_obj.get("labels") or []
        values = json_obj.get("data") or []
        n = min(len(labels), len(values))
        rows = []
        for i in range(n):
            val = _coerce_num(values[i])
            if val is not None:
                rows.append({"label": labels[i], "value": val})
        if rows:
            df = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
            return df.head(5)
    # fallback to to_df
    df = to_df(json_obj, label_keys=("label", "stateName", "region", "category"), value_keys=("value", "revenue", "amount"))
    if not df.empty:
        return df.sort_values("value", ascending=False).head(5).reset_index(drop=True)
    return pd.DataFrame(columns=["label", "value"])


# ---------------- Revenue trend parser ----------------
def parse_revenue_trend(json_obj: Any) -> pd.DataFrame:
    """
    Expected format: { '2021': [v1, v2, ...], '2022': [ ... ] }
    Returns DataFrame with columns year, period, value
    """
    if not json_obj:
        return pd.DataFrame(columns=["year", "period", "value"])
    rows = []
    # If a mapping of year->list
    if isinstance(json_obj, dict):
        for year, vals in json_obj.items():
            if isinstance(vals, (list, tuple)):
                for idx, v in enumerate(vals, start=1):
                    num = _coerce_num(v)
                    if num is not None:
                        rows.append({"year": str(year), "period": idx, "value": num})
            else:
                # if value a scalar, try coerce
                num = _coerce_num(vals)
                if num is not None:
                    rows.append({"year": str(year), "period": 1, "value": num})
    # fallback
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["year", "period", "value"])


# ---------------- Maker / state parsers ----------------
def parse_maker_state(json_obj: Any) -> pd.DataFrame:
    """
    Parses nested makers->states map:
    e.g. [{ "makers": { "MakerA": {"KA": 100, "MH": 50}, ... } }, ...]
    Returns rows: maker, state, value
    """
    if not json_obj:
        return pd.DataFrame(columns=["maker", "state", "value"])

    # If list wrapper
    data = json_obj if isinstance(json_obj, (list, tuple)) else [json_obj] if isinstance(json_obj, dict) else []
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        makers = item.get("makers") or item.get("makersMap") or item.get("makerStateMap") or {}
        if isinstance(makers, dict):
            for maker, stmap in makers.items():
                if isinstance(stmap, dict):
                    for state, v in stmap.items():
                        num = _coerce_num(v)
                        if num is not None:
                            rows.append({"maker": maker, "state": state, "value": num})
    if rows:
        return pd.DataFrame(rows).reset_index(drop=True)
    return pd.DataFrame(columns=["maker", "state", "value"])


def parse_makers(json_obj):
    """Parses the Top Makers JSON into a clean DataFrame with label and value columns."""
    try:
        # Ensure JSON has the correct structure
        if not json_obj or "labels" not in json_obj or "datasets" not in json_obj:
            return pd.DataFrame(columns=["label", "value"])

        labels = json_obj.get("labels", [])
        datasets = json_obj.get("datasets", [])
        if not datasets:
            return pd.DataFrame(columns=["label", "value"])

        data = datasets[0].get("data", [])
        if not labels or not data:
            return pd.DataFrame(columns=["label", "value"])

        df = pd.DataFrame({
            "label": labels,
            "value": data
        })

        # Sort descending by value
        df = df.sort_values("value", ascending=False).reset_index(drop=True)
        return df

    except Exception as e:
        print(f"Maker parsing failed: {e}")
        return pd.DataFrame(columns=["label", "value"])

# ---------------- Small helper for testing / preview --------------
def preview_df(df: pd.DataFrame, n=5):
    if df is None:
        print("<None>")
        return
    try:
        print(df.head(n))
    except Exception:
        print(df)

# End of file
